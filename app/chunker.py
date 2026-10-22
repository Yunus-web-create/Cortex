"""Structure-aware chunking: keeps headings with their content, sensible overlap."""
from __future__ import annotations

import re

from . import config

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")
# plain-text heading: short, capitalised, unpunctuated, followed by a new block
PLAIN_HEADING_RE = re.compile(r"^(?:\d+(?:\.\d+)*[.)]?\s+)?[A-Z][^.!?,;:]{1,58}$")


def _extract_heading(lines: list[str]) -> tuple[int, str, list[str]] | None:
    """Detect a leading heading line (markdown or plain text)."""
    if not lines:
        return None
    first = lines[0].strip()
    m = HEADING_RE.match(first)
    if m:
        return len(m.group(1)), m.group(2).strip(), lines[1:]
    if len(lines) == 1:
        # standalone block that reads like a title/heading (common for PDFs)
        if (PLAIN_HEADING_RE.match(first) and len(first.split()) <= 8
                and not first.endswith((".", ":", ";", ","))):
            title = re.sub(r"^\d+(?:\.\d+)*[.)]?\s+", "", first).strip()
            if title:
                return 2, title, []
        return None
    if len(lines) > 1:
        nxt = lines[1].strip()
        if (PLAIN_HEADING_RE.match(first)
                and len(first.split()) <= 8
                and not first.endswith((".", ":", ";", ","))
                and nxt and (nxt[0].isupper() or nxt[0].isdigit())):
            title = re.sub(r"^\d+(?:\.\d+)*[.)]?\s+", "", first).strip()
            if title:
                return 2, title, lines[1:]
    return None


def _units_from_blocks(blocks: list[dict]) -> list[dict]:
    """Turn parser blocks into atomic units, tracking the active heading path."""
    units: list[dict] = []
    headings: dict[int, str] = {}

    for block in blocks:
        page = int(block.get("page", 1) or 1)
        text = (block.get("text") or "").strip()
        if not text:
            continue

        lines = text.split("\n")
        found = _extract_heading(lines)
        if found:
            level, title, rest = found
            headings[level] = title
            for deeper in [lv for lv in headings if lv > level]:
                headings.pop(deeper, None)
            text = "\n".join(rest).strip()
            if not text:
                continue

        path = " › ".join(headings[lv] for lv in sorted(headings))
        if len(text) > config.CHUNK_SIZE:
            sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])", text)
            buf = ""
            for sent in sentences:
                if buf and len(buf) + len(sent) + 1 > config.CHUNK_SIZE:
                    units.append({"text": buf.strip(), "page": page, "heading": path})
                    buf = sent
                else:
                    buf = f"{buf} {sent}".strip()
            if buf:
                units.append({"text": buf.strip(), "page": page, "heading": path})
        else:
            units.append({"text": text, "page": page, "heading": path})

    return [u for u in units if u["text"].strip()]


def chunk_blocks(blocks: list[dict]) -> list[dict]:
    units = _units_from_blocks(blocks)
    if not units:
        return []

    chunks: list[dict] = []
    cur: list[dict] = []
    cur_len = 0

    def flush() -> None:
        nonlocal cur, cur_len
        if not cur:
            return
        body = "\n\n".join(u["text"] for u in cur).strip()
        if not body:
            cur, cur_len = [], 0
            return
        heading = cur[0]["heading"]
        prefix = f"{heading}\n\n" if heading else ""
        chunks.append({
            "ordinal": len(chunks),
            "page": cur[0]["page"],
            "heading": heading,
            "text": (prefix + body).strip(),
        })
        cur, cur_len = [], 0

    for unit in units:
        if cur and unit["heading"] != cur[0]["heading"]:
            flush()
        ulen = len(unit["text"]) + 2
        if cur and cur_len + ulen > config.CHUNK_SIZE:
            flush()
            # sentence-level overlap from the previous chunk for context continuity
            prev = chunks[-1]["text"] if chunks else ""
            if prev and config.CHUNK_OVERLAP > 0:
                tail = prev[-config.CHUNK_OVERLAP:]
                cut = tail.find(". ")
                tail = tail[cut + 2:] if 0 <= cut < len(tail) - 40 else tail
                tail = tail.strip()
                if len(tail) > 40:
                    cur = [{"text": tail, "page": unit["page"], "heading": unit["heading"]}]
                    cur_len = len(tail) + 2
        cur.append(unit)
        cur_len += ulen
        if cur_len >= config.CHUNK_SIZE:
            flush()

    flush()

    # merge tiny tail chunks into their predecessor
    merged: list[dict] = []
    for c in chunks:
        if merged and len(c["text"]) < 120:
            merged[-1]["text"] += "\n\n" + c["text"]
        else:
            merged.append(c)
    for i, c in enumerate(merged):
        c["ordinal"] = i
    return merged
