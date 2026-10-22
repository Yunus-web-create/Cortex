"""Claude integration: query rewriting, grounded answer streaming, demo fallback."""
from __future__ import annotations

import asyncio
import re
from typing import AsyncIterator

from . import config

# USD per 1M tokens (input, output) — edit if pricing changes
PRICING = {
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
}


def has_key() -> bool:
    return bool(config.anthropic_key())


def _client():
    import anthropic
    return anthropic.AsyncAnthropic(api_key=config.anthropic_key())


def serialise_block(block) -> dict:
    """Convert an SDK content block into a plain dict the API accepts back."""
    btype = getattr(block, "type", "")
    if btype == "text":
        return {"type": "text", "text": getattr(block, "text", "")}
    if btype == "tool_use":
        return {"type": "tool_use", "id": getattr(block, "id", ""),
                "name": getattr(block, "name", ""), "input": getattr(block, "input", {}) or {}}
    if btype == "thinking":
        return {"type": "thinking", "thinking": getattr(block, "thinking", ""),
                "signature": getattr(block, "signature", "")}
    if hasattr(block, "model_dump"):
        try:
            return block.model_dump(exclude_none=True)
        except Exception:
            pass
    return {"type": btype or "text", "text": str(block)}


def estimate_cost(model: str, in_tokens: int, out_tokens: int) -> float:
    if model.endswith(":free"):
        return 0.0                     # OpenRouter's zero-price tier
    pin, pout = PRICING.get(model, (3.0, 15.0))
    return round((in_tokens * pin + out_tokens * pout) / 1_000_000, 6)


# ------------------------------------------------------------------ prompts
ANSWER_SYSTEM = """You are Cortex, a knowledge assistant for {org}. You answer questions using ONLY the numbered sources inside <sources>.

Answer rules:
- Ground every factual claim in the sources and cite it with bracketed numbers placed right after the claim, e.g. [1] or [2][3]. Cite the source that actually contains the fact.
- Prefer exact specifics from the sources: names, numbers, dates, versions, prices. Synthesise across sources when that answers the question better.
- If the sources do not contain the answer, say so plainly in one short sentence and name what kind of document would contain it. Never guess and never invent details.
- If sources conflict, present both and flag the discrepancy with the citations.
- Format for fast reading: a direct answer in the first line, then short paragraphs or bullets. Bold key facts. No filler openers, no restating the question.
- Refer to documents by their titles when it helps ("According to the Employee Handbook, ...").
{extra}
"""

STRICT_EXTRA = ("- STRICT MODE: if the answer is not in the sources, do not use outside knowledge, "
                "even if you know it. Only small talk or questions about your capabilities may be "
                "answered directly.")
LOOSE_EXTRA = ("- Helpful mode: for questions clearly outside the knowledge base you may answer from "
               "general knowledge, but begin that part with '(general knowledge)' and never attach "
               "citations to it.")


def build_system(org: str, sources: list[dict], strict: bool) -> str:
    extra = STRICT_EXTRA if strict else LOOSE_EXTRA
    blocks = []
    used = 0
    for i, s in enumerate(sources, start=1):
        loc = f"page {s['page']}" if s.get("page") else ""
        head = f' · section "{s["heading"]}"' if s.get("heading") else ""
        body = s["text"].strip()
        if used + len(body) > config.MAX_CONTEXT_CHARS:
            body = body[: max(0, config.MAX_CONTEXT_CHARS - used)]
        if not body:
            break
        used += len(body)
        blocks.append(f'[{i}] "{s["doc_name"]}" · {loc}{head}\n{body}')
    joined = "\n\n".join(blocks)
    return (ANSWER_SYSTEM.format(org=org, extra=extra)
            + "\n<sources>\n" + joined + "\n</sources>")


def build_messages(history: list[dict], question: str) -> list[dict]:
    msgs = []
    for m in history[-config.HISTORY_TURNS:]:
        role = "assistant" if m["role"] == "assistant" else "user"
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if len(content) > 1500:
            content = content[:1500] + " …"
        if msgs and msgs[-1]["role"] == role:
            msgs[-1]["content"] += "\n\n" + content
        else:
            msgs.append({"role": role, "content": content})
    msgs.append({"role": "user", "content": question.strip()[:4000]})
    if msgs[0]["role"] != "user":
        msgs = msgs[1:]
    return msgs


REWRITE_PROMPT = """Rewrite the user's latest message into ONE standalone search query that will retrieve
the right passages from a document knowledge base. Resolve pronouns and implied references using the
conversation. Keep the user's own vocabulary (synonyms hurt recall). Output ONLY the query — no quotes,
no explanation, under 25 words. If the message is already standalone, return it unchanged."""


async def rewrite_query(question: str, history: list[dict]) -> str:
    """Turn a follow-up into a standalone retrieval query (cheap model)."""
    if not has_key() or not history:
        return question
    recent = build_messages(history, question)[-5:]
    if len(recent) < 2:
        return question
    try:
        client = _client()
        resp = await client.messages.create(
            model=config.FAST_MODEL,
            max_tokens=80,
            system=REWRITE_PROMPT,
            messages=recent[:-1] + [{
                "role": "user",
                "content": f"Conversation above.\nLatest message: {question}\n\nStandalone search query:",
            }],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
        text = text.strip('"\'` \n')
        if 3 <= len(text) <= 300:
            return text
    except Exception:
        pass
    return question


# --------------------------------------------------------------- generation
async def stream_answer(question: str, sources: list[dict], history: list[dict],
                        model: str, strict: bool = True,
                        org: str = "this knowledge base") -> AsyncIterator[dict]:
    """Yield {'type': 'delta'|'usage'|'error', ...} events."""
    if not has_key():
        async for ev in demo_answer(question, sources):
            yield ev
        return

    system = build_system(org, sources, strict)
    messages = build_messages(history, question)
    try:
        client = _client()
        async with client.messages.stream(
            model=model,
            max_tokens=config.MAX_TOKENS,
            system=system,
            messages=messages,
        ) as stream:
            async for text in stream.text_stream:
                if text:
                    yield {"type": "delta", "text": text}
            final = await stream.get_final_message()
            usage = getattr(final, "usage", None)
            in_tok = int(getattr(usage, "input_tokens", 0) or 0)
            out_tok = int(getattr(usage, "output_tokens", 0) or 0)
            yield {
                "type": "usage",
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "cost_usd": estimate_cost(model, in_tok, out_tok),
                "stop_reason": getattr(final, "stop_reason", None),
            }
    except Exception as exc:  # pragma: no cover - network dependent
        import anthropic
        if isinstance(exc, anthropic.AuthenticationError):
            msg = "Your Anthropic API key was rejected. Open Settings and check the key."
        elif isinstance(exc, anthropic.RateLimitError):
            msg = "Anthropic rate limit reached. Wait a moment and try again."
        elif isinstance(exc, anthropic.APIConnectionError):
            msg = "Could not reach the Anthropic API (network issue)."
        elif isinstance(exc, anthropic.BadRequestError):
            msg = f"Anthropic rejected the request: {getattr(exc, 'message', str(exc))[:200]}"
        else:
            msg = f"Model error: {type(exc).__name__}: {str(exc)[:200]}"
        yield {"type": "error", "message": msg}


# ------------------------------------------------------------ demo fallback
def _sentence_split(text: str) -> list[str]:
    out: list[str] = []
    for para in re.split(r"\n\s*\n", text or ""):
        para = re.sub(r"\s+", " ", para).strip()
        if not para:
            continue
        out.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", para) if s.strip())
    return out


def _clean_sentence(sent: str) -> str:
    sent = re.sub(r"\*\*|__|[*_`]", "", sent)     # markdown bold/italics/code
    sent = re.sub(r"^[\-•|\s]+", "", sent)
    sent = re.sub(r"\s+", " ", sent).strip()
    if len(sent) > 300:
        sent = sent[:297].rsplit(" ", 1)[0] + "…"
    return sent


def _is_table_like(sent: str) -> bool:
    if "| ---" in sent or sent.count("|") >= 2:
        return True
    return sent.count("; ") >= 3 and sent.count(":") >= 3


def _key(sent: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", sent.lower())[:70].strip()


def _long_split(sent: str) -> list[str]:
    """CSV rows arrive as one giant 'sentence' — split those into usable pieces."""
    if len(sent) <= 280:
        return [sent]
    parts = [p.strip() for p in re.split(r"(?<=;)\s+(?=Plan:)|\s*;\s*", sent) if p.strip()]
    return parts or [sent]


def extractive_answer(question: str, sources: list[dict], max_sentences: int = 4) -> str:
    """IDF-weighted extractive fallback used when no API key is configured."""
    import math
    from collections import Counter
    from .retrieval import tokenize

    q = set(tokenize(question))
    if not q:
        return "Ask a question about the documents in your knowledge base."

    cands: list[tuple[int, str, set[str]]] = []
    for idx, src in enumerate(sources[:3], start=1):
        heading_words = set(tokenize(src.get("heading", "")))
        body = src.get("text") or ""
        first_line, sep, rest = body.partition("\n\n")
        if src.get("heading") and first_line.strip() and first_line.strip() in src["heading"]:
            body = rest if sep else first_line
        for sent in _sentence_split(body):
            for piece in _long_split(sent):
                if len(piece) < 30 or _is_table_like(piece):
                    continue
                cands.append((idx, piece, heading_words))
    if not cands:
        return ("I found related material but couldn't pull a clean answer from it. "
                "Add a model provider key for a synthesised answer.")

    GENERIC = re.compile(r"\b(overview|summar\w*|key polic\w*|main points|compare|"
                         r"what does .{0,60}cover|explain)\b", re.I)
    if GENERIC.search(question):
        # prefer passages whose document/heading actually is about the question
        ranked = []
        for idx, src in enumerate(sources, start=1):
            title_tokens = set(tokenize((src.get("heading") or "") + " " + src.get("doc_name", "")))
            ranked.append((len(q & title_tokens), float(src.get("score") or 0), idx, src))
        ranked.sort(key=lambda x: (-x[0], -x[1]))
        if any(r[0] > 0 for r in ranked):
            ranked = [r for r in ranked if r[0] > 0]
        picks = []
        for _, _, idx, src in ranked[:3]:
            body = src.get("text") or ""
            first, sep, rest = body.partition("\n\n")
            if src.get("heading") and first.strip() and first.strip() in src["heading"]:
                body = rest if sep else first
            lead = [x for x in _sentence_split(body) if len(x) > 45 and not _is_table_like(x)]
            if lead:
                picks.append((0, idx, _clean_sentence(lead[0])))
        if len(picks) >= 2:
            bullets = "\n".join(f"- {sent} [{idx}]" for _, idx, sent in picks)
            return "Here's an overview from your documents:\n\n" + bullets

    toks = [set(tokenize(c[1])) for c in cands]
    df: Counter = Counter()
    for t in toks:
        df.update(t)
    n = len(cands)
    idf = lambda t: math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
    norm = max(1.0, len(q) ** 0.5)

    scored: list[tuple[float, int, str]] = []
    seen: set[str] = set()
    for (idx, piece, hw), ts in zip(cands, toks):
        key = _key(piece)
        if key in seen:
            continue
        shared = q & ts
        if not shared:
            continue
        seen.add(key)
        score = sum(idf(t) for t in shared) / norm
        score += 0.15 * sum(idf(t) for t in (q & hw)) / norm
        scored.append((score, idx, _clean_sentence(piece)))

    # best two sentences per source, then drop weak matches relative to the best
    per_source: dict[int, list[tuple[float, int, str]]] = {}
    for item in scored:
        per_source.setdefault(item[1], []).append(item)
    kept: list[tuple[float, int, str]] = []
    for idx, items in per_source.items():
        items.sort(key=lambda x: -x[0])
        kept.extend(items[:2])
    kept.sort(key=lambda x: -x[0])
    if kept:
        best = kept[0][0]
        kept = [x for x in kept if x[0] >= best * 0.55]
    picks = kept[:max_sentences]
    picks.sort(key=lambda x: (x[1], -x[0]))
    if len(picks) < 2 and sources:
        top = sources[0]
        body = top.get("text") or ""
        first, sep, rest = body.partition("\n\n")
        if top.get("heading") and first.strip() and first.strip() in top["heading"]:
            body = rest if sep else first
        existing = {_key(p[2]) for p in picks}
        for sent in _sentence_split(body):
            if len(sent) > 45 and not _is_table_like(sent) and _key(sent) not in existing:
                picks.append((0.0, 1, _clean_sentence(sent)))
                break
        picks.sort(key=lambda x: (x[1], -x[0]))
    if not picks:
        return ("I found related material but couldn't pull a clean answer from it. "
                "Add a model provider key for a synthesised answer.")

    bullets = "\n".join(f"- {sent} [{idx}]" for _, idx, sent in picks)
    web = sum(1 for x in sources if x.get("url"))
    lead = ("Here's what I found online:" if web and web >= len(sources) - 1
            else "Here's what your documents say:" if not web
            else "Here's what I found:")
    return (lead + "\n\n" + bullets)


async def demo_answer(question: str, sources: list[dict]) -> AsyncIterator[dict]:
    text = extractive_answer(question, sources)
    words = re.findall(r"\S+\s*", text)
    buf = ""
    for i, w in enumerate(words):
        buf += w
        if len(buf) > 12 or i == len(words) - 1:
            yield {"type": "delta", "text": buf}
            buf = ""
            await asyncio.sleep(0.012)
    yield {"type": "usage", "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "demo": True}


# ------------------------------------------------------------- suggestions
SUGGEST_PROMPT = """You generate starter questions for a document knowledge base.
Given the document titles and section headings below, write exactly 4 short, specific questions a user
could ask and have answered by these documents. Vary the topics. One per line, no numbering, no quotes,
max 12 words each. Output only the questions."""


async def suggest_questions(docs: list[dict], headings: list[str]) -> list[str]:
    """Four starter questions, from whichever provider is active (falls back offline)."""
    if not docs:
        return []
    fallback = _fallback_questions(docs, headings)
    try:
        from . import gateway, router
        if not gateway.configured():
            return fallback
        catalogue = await gateway.models()
        model = (router.candidates("fast", catalogue, limit=1) or [""])[0]
        if not model:
            return fallback
        listing = "\n".join(f"- {d['name']}" for d in docs[:12])
        text = await gateway.complete(
            model,
            [{"role": "system", "content": SUGGEST_PROMPT},
             {"role": "user", "content": f"Documents:\n{listing}\n\nHeadings:\n" +
              "\n".join(f"- {h}" for h in headings[:60])}],
            max_tokens=220, temperature=0.4, timeout=30)
        qs = [re.sub(r"^[\-\d\.\)\s]+", "", ln).strip() for ln in (text or "").splitlines()
              if ln.strip()]
        qs = [q for q in qs if 12 < len(q) < 140][:4]
        return qs or fallback
    except Exception:
        return fallback


BAD_SECTIONS = {"purpose and scope", "about this handbook", "about this document",
                "contents", "introduction", "overview", "getting help", "getting started",
                "what aurora flow is", "core concepts"}


def _clean_doc_title(name: str) -> str:
    stem = re.sub(r"\.[a-z0-9]{2,5}$", "", name, flags=re.I)
    stem = re.sub(r"[_\-]+", " ", stem)
    stem = re.sub(r"\b(19|20)\d{2}\b|\bv?\d+(\.\d+)+\b", "", stem)
    stem = re.sub(r"\s{2,}", " ", stem).strip(" -–—")
    return stem or name


def _fallback_questions(docs: list[dict], headings: list[str]) -> list[str]:
    """Deterministic, natural-sounding starter questions when no API key is present."""
    titles = [_clean_doc_title(d["name"]) for d in docs]
    lowered = [t.lower() for t in titles]
    qs: list[str] = []

    def add(q: str) -> None:
        if q and q not in qs:
            qs.append(q)

    if any("polic" in t or "handbook" in t for t in lowered):
        add("What are the key policies I should know about?")
    if any(d["ext"] == ".csv" for d in docs):
        add("Compare the plans, their prices and limits.")
    if any("support" in t or "playbook" in t for t in lowered):
        add("How quickly does support respond to a critical issue?")

    sections: list[str] = []
    for h in headings:
        if "›" in h or "—" in h:
            continue
        clean = h.strip().rstrip(".:")
        if not (4 <= len(clean) <= 46) or clean.lower() in BAD_SECTIONS:
            continue
        if any(clean.lower() in x.lower() or x.lower() in clean.lower() for x in sections):
            continue
        sections.append(clean)
        if len(sections) >= 4:
            break

    for sec in sections:
        if len(qs) >= 4:
            break
        add(f"What does the knowledge base say about {sec.lower()}?")
    for d, t in zip(docs, titles):
        if len(qs) >= 4:
            break
        if d["ext"] == ".csv":
            continue
        add(f"Summarise the {t}.")
    return qs[:4]
