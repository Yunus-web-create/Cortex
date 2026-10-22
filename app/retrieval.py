"""Hybrid retrieval: BM25 (lexical) + dense vectors, fused with RRF, diversified with MMR."""
from __future__ import annotations

import logging
import math
import re
import threading
import time
from collections import Counter

import numpy as np

from . import config, db, embeddings

STOPWORDS = set("""a an and are as at be been being but by can could did do does doing for from had has have
having he her here hers him his how i if in into is it its itself just me more most my no nor not of off on
once only or other our out own same she should so some such than that the their them then there these
they this those through to too under until up very was we were what when where which while who whom why will
with would you your yours also may might must shall upon within without across per via etc""".split())

_lock = threading.Lock()
_cache: dict = {"version": -1}
_version = 0

_rerank_lock = threading.Lock()
_reranker = None
_rerank_state = {"status": "cold", "error": "", "model": config.RERANK_MODEL}


def _load_reranker():
    """Cross-encoder reranker (ms-marco MiniLM, ~80 MB). Optional: retrieval
    still works — just less precise — if it can't load."""
    global _reranker
    if not config.RERANK_ENABLED:
        return None
    if _reranker is not None:
        return _reranker
    with _rerank_lock:
        if _reranker is not None or _rerank_state["status"] == "unavailable":
            return _reranker
        _rerank_state["status"] = "loading"
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            _reranker = TextCrossEncoder(model_name=config.RERANK_MODEL,
                                         threads=config.RERANK_THREADS)
            _rerank_state["status"] = "ready"
            logging.getLogger("cortex.retrieval").info("Reranker ready: %s", config.RERANK_MODEL)
        except Exception as exc:
            _rerank_state["status"] = "unavailable"
            _rerank_state["error"] = str(exc)[:300]
            _reranker = None
    return _reranker


def reranker_status() -> dict:
    return dict(_rerank_state)


def warm_reranker() -> None:
    threading.Thread(target=_load_reranker, daemon=True).start()


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def invalidate() -> None:
    global _version
    with _lock:
        _version += 1


def _stem(tok: str) -> str:
    if len(tok) > 5 and tok.endswith("ing"):
        return tok[:-3]
    if len(tok) > 5 and tok.endswith("ied"):
        return tok[:-3] + "y"
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 4 and tok.endswith("es") and not tok.endswith("ses"):
        return tok[:-2]
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def tokenize(text: str) -> list[str]:
    """Tokens include both the raw word and its stem, so odd stems can never
    hide an exact match (BM25 just sees a slightly larger vocabulary)."""
    toks = re.findall(r"[a-z0-9][a-z0-9'\-]*", text.lower())
    out: list[str] = []
    for t in toks:
        t = t.strip("'-")
        if len(t) < 2 or t in STOPWORDS:
            continue
        stem = _stem(t)
        out.append(stem)
        if stem != t:
            out.append(t)
    return out


class BM25:
    def __init__(self, docs_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(docs_tokens)
        self.doc_len = np.array([len(d) for d in docs_tokens], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if self.n else 0.0
        self.tfs = [Counter(d) for d in docs_tokens]
        df = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        self.idf = {t: math.log(1 + (self.n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        out = np.zeros(self.n, dtype=np.float32)
        if not self.n:
            return out
        for t in set(query_tokens):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, tf in enumerate(self.tfs):
                f = tf.get(t)
                if not f:
                    continue
                dl = self.doc_len[i] or 1.0
                out[i] += idf * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return out


def _build() -> dict:
    global _cache
    with _lock:
        if _cache.get("version") == _version:
            return _cache
        chunks = db.all_chunks()
        ids, mat = db.all_embeddings()
        tokens = [tokenize(c["text"]) for c in chunks]
        bm25 = BM25(tokens)
        id_to_row = {cid: i for i, cid in enumerate(ids)}
        # map chunk position -> embedding row (-1 if missing)
        emb_row = np.array([id_to_row.get(c["id"], -1) for c in chunks], dtype=np.int64)
        _cache = {
            "version": _version,
            "chunks": chunks,
            "bm25": bm25,
            "mat": mat,
            "emb_row": emb_row,
            "built_at": time.time(),
        }
        return _cache


def warm() -> None:
    try:
        _build()
    except Exception:
        pass


def _mmr(cand_idx: list[int], rel: np.ndarray, vecs: np.ndarray | None,
         k: int, lam: float = config.MMR_LAMBDA) -> list[int]:
    if not cand_idx:
        return []
    if vecs is None or len(cand_idx) <= k:
        return cand_idx[:k]
    selected: list[int] = []
    pool = list(cand_idx)
    sim_max = np.full(len(pool), -1.0, dtype=np.float32)
    while pool and len(selected) < k:
        best, best_score = None, -1e9
        for j, idx in enumerate(pool):
            redundancy = 0.0 if not selected else float(sim_max[j])
            score = lam * float(rel[idx]) - (1 - lam) * redundancy
            if score > best_score:
                best, best_score = j, score
        chosen = pool.pop(best)
        selected.append(chosen)
        if pool and vecs is not None:
            sims = vecs[pool] @ vecs[chosen]
            sim_max[: len(pool)] = np.maximum(sim_max[: len(pool)], sims)
    return selected


def search(query: str, top_k: int = config.TOP_K,
           use_semantic: bool = True,
           doc_ids: list[str] | None = None) -> tuple[list[dict], dict]:
    """Return (sources, diagnostics). doc_ids restricts retrieval to chosen documents."""
    t0 = time.perf_counter()
    cache = _build()
    chunks: list[dict] = cache["chunks"]
    if not chunks:
        return [], {"candidates": 0, "mode": "empty", "ms": 0}

    q_tokens = tokenize(query)
    n = len(chunks)
    pool = min(config.CANDIDATES_PER_RETRIEVER, n)
    scope = None
    if doc_ids:
        allowed = set(doc_ids)
        scope = np.array([c["doc_id"] in allowed for c in chunks], dtype=bool)
        if not scope.any():
            scope = None

    # ---- document/heading prior: chunks whose title shares the query's terms
    #      get a small boost (helps "compare the plans" reach the pricing doc)
    q_set = set(q_tokens)
    prior = np.zeros(n, dtype=np.float32)
    if q_set:
        prior_cache = cache.get("prior")
        if prior_cache is None:
            prior_cache = []
            for c in chunks:
                title_tokens = set(tokenize(c["doc_name"] + " " + (c["heading"] or "")))
                prior_cache.append(title_tokens)
            cache["prior"] = prior_cache
        inv = 1.0 / max(1, len(q_set))
        for i, title_tokens in enumerate(prior_cache):
            hit = len(q_set & title_tokens)
            prior[i] = hit * inv

    # ---- lexical
    bm = cache["bm25"].scores(q_tokens)
    if scope is not None:
        bm = np.where(scope, bm, 0.0)
    lex_score = bm.copy()
    if bm.max() > 0:
        lex_score = bm / bm.max() + config.PRIOR_WEIGHT * prior
    if scope is not None:
        lex_score = np.where(scope, lex_score, 0.0)
    bm_rank = np.argsort(-lex_score)[:pool]
    lexical_rank = [int(i) for i in bm_rank if lex_score[i] > 0]

    # ---- dense
    qv = embeddings.embed_query(query) if use_semantic else None
    dense = None
    dense_rank: list[int] = []
    mat: np.ndarray | None = cache["mat"]
    if qv is not None and mat is not None and len(mat):
        dense = mat @ qv
        emb_row = cache["emb_row"]
        # scores per chunk (chunks with embeddings only)
        dense_by_chunk = np.full(n, -1.0, dtype=np.float32)
        mask = emb_row >= 0
        dense_by_chunk[mask] = dense[emb_row[mask]]
        dense_ranked = dense_by_chunk + config.PRIOR_WEIGHT_DENSE * prior
        if scope is not None:
            dense_ranked = np.where(scope, dense_ranked, -1.0)
        order = np.argsort(-dense_ranked)[:pool]
        dense_rank = [int(i) for i in order if dense_ranked[i] > 0]

    # ---- reciprocal rank fusion
    fused: dict[int, float] = {}
    for rank, idx in enumerate(lexical_rank):
        fused[idx] = fused.get(idx, 0.0) + 1.0 / (config.RRF_K + rank + 1)
    for rank, idx in enumerate(dense_rank):
        fused[idx] = fused.get(idx, 0.0) + 1.0 / (config.RRF_K + rank + 1)
    if not fused:  # nothing matched lexically or semantically
        fused = {int(i): 0.0 for i in range(min(pool, n))}

    ordered = sorted(fused.keys(), key=lambda i: -fused[i])[: config.FUSED_POOL]

    # ---- cross-encoder rerank: joint(query, passage) scoring is far more
    #      precise than cosine, and gives a well-calibrated confidence signal
    rerank_scores: dict[int, float] = {}
    t_rerank = time.perf_counter()
    reranker = _load_reranker() if ordered else None
    if reranker is not None and ordered:
        pool = ordered[: config.RERANK_POOL]
        try:
            pairs = [chunks[i]["text"][: config.RERANK_MAX_CHARS] for i in pool]
            raw = [float(x) for x in reranker.rerank(query, pairs)]
            rerank_scores = {i: _sigmoid(r) for i, r in zip(pool, raw)}
            ordered = sorted(pool, key=lambda i: -rerank_scores[i])
        except Exception as exc:  # pragma: no cover - model/runtime dependent
            logging.getLogger("cortex.retrieval").warning("rerank failed: %s", exc)
            rerank_scores = {}
    rerank_ms = round((time.perf_counter() - t_rerank) * 1000, 1)

    # ---- build vectors for MMR
    vecs = None
    if dense is not None and mat is not None and len(mat):
        rows = [cache["emb_row"][i] for i in ordered if cache["emb_row"][i] >= 0]
        if len(rows) == len(ordered):
            vecs = mat[rows]

    if vecs is not None and ordered:
        if rerank_scores:
            rel = np.array([rerank_scores.get(i, 0.0) for i in ordered], dtype=np.float32)
            lam = 0.88          # trust the cross-encoder; only break near-ties
        else:
            rel = np.array([
                float(dense[cache["emb_row"][i]]) if cache["emb_row"][i] >= 0 else 0.0
                for i in ordered
            ], dtype=np.float32)
            lam = config.MMR_LAMBDA
        picked_pos = _mmr(list(range(len(ordered))), rel, vecs, top_k, lam)
    else:
        picked_pos = list(range(min(top_k, len(ordered))))

    top_lex = float(bm.max()) if n else 0.0
    top_dense = float(dense.max()) if dense is not None and len(dense) else 0.0

    sources: list[dict] = []
    for pos in picked_pos:
        i = ordered[pos]
        c = chunks[i]
        if rerank_scores:
            score = rerank_scores.get(i, 0.0)
            method = "reranked"
        elif dense is not None:
            di = cache["emb_row"][i]
            score = float(dense[di]) if di >= 0 else 0.0
            method = "hybrid" if bm[i] > 0 else "semantic"
        else:
            score = float(bm[i] / top_lex) if top_lex > 0 else 0.0
            method = "keyword"
        heading = c["heading"] or ""
        body = c["text"]
        if heading and body.startswith(heading):
            body = body[len(heading):].lstrip("\n ")
        parts = [p.strip() for p in heading.split("›") if p.strip()]
        short = " › ".join(parts[-2:]) if parts else ""
        sources.append({
            "chunk_id": c["id"],
            "doc_id": c["doc_id"],
            "doc_name": c["doc_name"],
            "ext": c["ext"],
            "page": c["page"],
            "heading": heading,
            "heading_short": short,
            "text": body,
            "score": round(score, 4),
            "bm25": round(float(bm[i]), 3),
            "method": method,
        })

    if sources:
        confidence = max(s["score"] for s in sources)
    else:
        confidence = 0.0

    diag = {
        "candidates": n,
        "lexical_hits": len(lexical_rank),
        "semantic_hits": len(dense_rank),
        "mode": ("hybrid+rerank" if rerank_scores else
                 ("hybrid" if dense is not None else "keyword")),
        "confidence": round(float(confidence), 4),
        "rerank_ms": rerank_ms,
        "reranked": bool(rerank_scores),
        "ms": round((time.perf_counter() - t0) * 1000, 1),
    }
    return sources, diag
