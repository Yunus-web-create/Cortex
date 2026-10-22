"""Answer-quality layer: adaptive query expansion, grounding verification,
conversation titles, follow-up questions and history compression."""
from __future__ import annotations

import json
import math
import re
from collections import Counter

from . import config, gateway, router
from .retrieval import tokenize

async def _fast_model() -> str:
    """Cheapest chat model the active provider actually offers."""
    try:
        models = await gateway.models()
    except Exception:
        models = []
    return router.helper_model(models, "fast") or config.FAST_MODEL


async def _ask(system: str, user: str, max_tokens: int) -> str:
    """One-shot helper call that works with any provider, not just Anthropic."""
    if not gateway.configured():
        return ""
    if gateway.status().get("blocked"):
        return ""          # the account can't spend yet; don't burn a request per helper
    if gateway.provider_name() == "free":
        return ""          # one request at a time: helpers would double every turn
    model = await _fast_model()
    if not model:
        return ""
    return await gateway.complete(
        model,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=max_tokens, temperature=0.2)


# --------------------------------------------------------------- expansion
EXPAND_PROMPT = """You expand a search query for a document knowledge base.
Write {n} alternative search queries for the same information need. Each must use DIFFERENT vocabulary:
synonyms, the formal policy term, and the likely document wording. Do not answer the question.
Output one query per line, no numbering, no quotes, under 20 words each."""


async def expand_queries(question: str, n: int = 2) -> list[str]:
    """Cheap model writes vocabulary variants to improve recall on hard questions."""
    if n <= 0:
        return []
    try:
        text = await _ask(EXPAND_PROMPT.format(n=n), question[:600], 150)
        out = []
        for line in text.splitlines():
            q = re.sub(r"^[\-\d\.\)\s]+", "", line).strip().strip('"')
            if 6 < len(q) < 160 and q.lower() != question.lower():
                out.append(q)
        return out[:n]
    except Exception:
        return []


# ------------------------------------------------------- claim spot-check
_VERSION = re.compile(r"\b\d{1,2}\.\d{1,2}(?:\.\d{1,2})?\b")


def unsupported_numbers(answer: str, sources: list[dict], question: str = "") -> list[str]:
    """Version-like numbers in the answer that appear in none of its sources.

    A cheap hallucination tripwire: no model call, no false claims of certainty —
    it only ever produces a warning the reader can dismiss.
    """
    if not sources or not answer:
        return []
    text = " ".join(str(s.get("text") or "") for s in sources)
    text += " " + " ".join(str(s.get("url") or "") for s in sources)
    q = question or ""
    missing = []
    for tok in _VERSION.findall(answer):
        if tok in missing or tok in q or tok in text:
            continue
        missing.append(tok)
    return missing[:3]


# --------------------------------------------------------------- grounding
_NUM = re.compile(r"\b\d[\d,.]*\b")
_CITE = re.compile(r"\[(\d{1,2})\]")


def _content_tokens(text: str) -> set[str]:
    return {t for t in tokenize(text) if len(t) > 2}


def verify_grounding(answer: str, sources: list[dict]) -> dict:
    """Estimate how many cited claims are actually supported by the cited passage.

    A claim counts as supported when enough of its content words appear in the
    cited source, or when its numbers appear verbatim (numbers are the easiest
    thing for a model to get wrong).
    """
    if not answer or not sources:
        return {"claims": 0, "supported": 0, "score": None}

    # split on paragraph/sentence boundaries, but not between a sentence and
    # its trailing citation ("managed. [1]" must stay together)
    parts = re.split(r"(?<=[.!?])\s+(?!\[)|\n+", answer)
    sentences: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # merge orphaned citations back into the sentence they belong to
        if re.fullmatch(r"(\[\d{1,2}\]\s*)+", part) and sentences:
            sentences[-1] += " " + part
        else:
            sentences.append(part)

    claims = 0
    supported = 0
    details = []

    for sent in sentences:
        cites = {int(m) for m in _CITE.findall(sent)}
        cites = {c for c in cites if 1 <= c <= len(sources)}
        if not cites:
            continue
        claims += 1
        probe = re.sub(r"^[-*•\d.)\s]+", "", _CITE.sub(" ", sent)).strip()
        probe_tokens = _content_tokens(probe)
        numbers = set(_NUM.findall(probe))
        best = 0.0
        for c in cites:
            src_tokens = _content_tokens(sources[c - 1].get("text", ""))
            if not probe_tokens or not src_tokens:
                continue
            overlap = len(probe_tokens & src_tokens) / len(probe_tokens)
            src_numbers = set(_NUM.findall(sources[c - 1].get("text", "")))
            number_hit = 1.0 if numbers and numbers <= src_numbers else 0.0
            best = max(best, max(overlap, 0.65 * number_hit + 0.35 * overlap))
        ok = best >= config.GROUNDING_MIN_OVERLAP
        supported += int(ok)
        details.append({"claim": probe[:120], "overlap": round(best, 3), "supported": ok})

    return {
        "claims": claims,
        "supported": supported,
        "score": round(supported / claims, 3) if claims else None,
        "details": details[:12],
    }


# ----------------------------------------------------------------- titles
TITLE_PROMPT = """Write a title of at most 6 words for a chat that starts with the user message below.
Use the user's own topic words. No quotes, no trailing punctuation, no "Chat about"."""


async def make_title(question: str) -> str | None:
    try:
        text = (await _ask(TITLE_PROMPT, question[:300], 24)).strip()
        text = text.strip('"\'`.* \n')
        if 3 <= len(text) <= 60:
            return text[:60]
    except Exception:
        pass
    return None


# -------------------------------------------------------------- follow-ups
FOLLOWUP_PROMPT = """Given the question and answer below, write 3 short follow-up questions the user is
likely to ask next. They must be answerable from the same document set, be specific, and not repeat the
original question. One per line, no numbering, no quotes, max 12 words each."""


async def followups(question: str, answer: str, sources: list[dict]) -> list[str]:
    if not answer:
        return []
    if gateway.provider_name() == "openrouter" and gateway.free_only():
        return []      # a $0 OpenRouter key gets 50 calls a day; spend them on answers
    try:
        context = "\n".join(f"- {s['doc_name']}: {s['text'][:200]}" for s in sources[:3])
        text = await _ask(FOLLOWUP_PROMPT,
                          f"Question: {question}\n\nAnswer: {answer[:900]}\n\nContext:\n{context}", 160)
        out = [re.sub(r"^[\-\d\.\)\s]+", "", ln).strip().strip('"')
               for ln in text.splitlines() if ln.strip()]
        return [q for q in out if 8 < len(q) < 140][:3]
    except Exception:
        return []


# ------------------------------------------------------- history compression
SUMMARY_PROMPT = """Summarise this conversation for use as context. Keep names, numbers, decisions and
open questions. Maximum 90 words, plain prose, no bullet points."""


async def compress_history(history: list[dict]) -> tuple[str, list[dict]]:
    """Fold older turns into a summary so long chats stay inside the context budget."""
    if len(history) <= config.HISTORY_TURNS * 2:
        return "", history
    if not gateway.configured():
        return "", history[-config.HISTORY_TURNS:]
    older = history[:-config.HISTORY_TURNS]
    recent = history[-config.HISTORY_TURNS:]
    transcript = "\n".join(f"{m['role']}: {(m.get('content') or '')[:400]}" for m in older)
    try:
        summary = (await _ask(SUMMARY_PROMPT, transcript[:6000], 200)).strip()
        return summary, recent
    except Exception:
        return "", recent
