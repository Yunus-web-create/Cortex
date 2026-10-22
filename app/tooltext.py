"""Recover tool calls that a model writes as *text* instead of the tool channel.

Some open-weight models — Nemotron on OpenRouter is the one that bit us — are
served without a reliable `tool_calls` channel. Instead of returning structured
calls they print the call into the answer:

    <tool_call>
    <function=fetch_url>
    <parameter=url>
    http://example.org
    </parameter>
    </function>
    </tool_call>

Left alone, that markup is streamed to the user as if it were prose, and the tool
never runs. `ToolTextFilter` sits in the delta stream and does three things:

  * holds back text that might be the start of a call (a tag can be split across
    chunks, so `<tool_ca` must not be shown before `ll>` arrives),
  * converts a complete block into a `tool_call` event,
  * drops the markup from the visible answer.

It is deliberately conservative: if a block never closes, or looks like ordinary
prose, the text is released unchanged rather than swallowed.
"""
from __future__ import annotations

import json
import re

OPEN = "<tool_call>"
CLOSE = "</tool_call>"
# a call may also arrive without the outer tag ("<function=name>")
FUNC_OPEN = "<function="
FUNC_RE = re.compile(r"<function\s*=\s*[\"']?([A-Za-z0-9_.\-/:]+)[\"']?\s*>")
PARAM_RE = re.compile(r"<parameter\s*=\s*[\"']?([A-Za-z0-9_.\-]+)[\"']?\s*>(.*?)</parameter>", re.S)
JSON_CALL = re.compile(r"\{\s*\"(?:name|tool)\"\s*:\s*\"([A-Za-z0-9_.\-]+)\"\s*,\s*"
                       r"\"(?:arguments|parameters|args)\"\s*:\s*(\{.*?\})\s*\}", re.S)

_PARAM_TAIL = re.compile(r"</parameter\s*>\s*(?:</function\s*>)?\s*$", re.I)


def _coerce(value: str):
    """<parameter=max_results>5</parameter> should arrive as an int, not '5'."""
    raw = value.strip()
    if raw and raw[0] in "{[":
        try:
            return json.loads(raw)
        except Exception:
            pass
    if re.fullmatch(r"-?\d+", raw):
        try:
            return int(raw)
        except Exception:
            return raw
    if re.fullmatch(r"-?\d*\.\d+", raw):
        try:
            return float(raw)
        except Exception:
            return raw
    if raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    if raw.lower() in ("null", "none"):
        return None
    return raw


def parse_block(block: str) -> dict | None:
    """Turn one <tool_call>…</tool_call> body into {'name':…, 'arguments':{…}}."""
    text = (block or "").strip()
    if not text:
        return None
    fn = FUNC_RE.search(text)
    if fn:
        name = fn.group(1).strip()
        body = text[fn.end():]
        # drop the closing tag so a stray "</function>" never becomes an argument
        body = re.sub(r"</function\s*>\s*$", "", body.strip(), flags=re.I).strip()
        args: dict = {}
        if body.startswith("{"):
            try:
                loaded = json.loads(body)
                if isinstance(loaded, dict):
                    args = loaded
            except Exception:
                args = {}
        if not args:
            for key, value in PARAM_RE.findall(body):
                args[key] = _coerce(value)
        return {"name": name, "arguments": args} if name else None
    found = JSON_CALL.search(text)
    if found:
        try:
            args = json.loads(found.group(2))
        except Exception:
            args = {}
        return {"name": found.group(1), "arguments": args if isinstance(args, dict) else {}}
    return None


def extract(text: str) -> tuple[str, list[dict]]:
    """Strip every complete call block out of a finished string."""
    calls: list[dict] = []
    out = text or ""

    def _take(match: re.Match) -> str:
        call = parse_block(match.group(1))
        if call:
            calls.append(call)
            return ""
        return match.group(0)          # not a call after all — leave it alone

    out = re.sub(r"<tool_call>\s*(.*?)\s*</tool_call>", lambda m: _take(m), out, flags=re.S)
    # an unclosed block at the end of the reply, which is common when the model
    # runs out of max_tokens mid-call
    out = re.sub(r"<tool_call>\s*(<function\s*=.*)$", lambda m: _take(m), out, flags=re.S)
    return out, calls


def _partial_tag(buf: str) -> int:
    """Length of a prefix of `buf` that could be the start of a tag we care about.

    Returning the index from which the caller must not emit text: `<tool_ca` at
    the end of one chunk may become `<tool_call>` in the next.
    """
    best = len(buf)
    for marker in (OPEN, CLOSE, FUNC_OPEN):
        low = buf.lower()
        for size in range(min(len(marker) - 1, len(low)), 0, -1):
            if low.endswith(marker[:size].lower()):
                best = min(best, len(buf) - size)
                break
    return best


class ToolTextFilter:
    """Streaming guard. Feed every delta, emit what is safe to show."""

    def __init__(self, names: set[str] | None = None):
        self._buf = ""
        self._inside = False       # we are buffering a call block
        self._raw_open = OPEN      # the opener exactly as the model wrote it
        self.calls: list[dict] = []
        # Only a *wrapped* <tool_call> is unambiguous. A bare "<function=x>" can
        # be a model describing the format, so it only counts when the name is a
        # tool this turn actually offered.
        self.names = set(names) if names else None

    # -- internals ---------------------------------------------------------
    def _drain_inside(self, final: bool) -> list[tuple[str, object]]:
        out: list[tuple[str, object]] = []
        while True:
            idx = self._buf.find(CLOSE)
            if idx != -1:
                block = self._buf[:idx]
                self._buf = self._buf[idx + len(CLOSE):]
                call = parse_block(block)
                if call:
                    self.calls.append(call)
                    out.append(("call", call))
                else:                   # not a call after all: keep the words
                    out.append(("text", f"{self._raw_open}{block}{self._buf[:0]}{CLOSE}"))
                self._inside = False
                # anything after the close tag is normal text again
                out.extend(self._drain_outside(final))
                return out
            if not final:
                return out
            # never closed. If it clearly is a call, honour it; else release it.
            call = parse_block(self._buf) if FUNC_OPEN.lower() in self._buf.lower() else None
            if call:
                self.calls.append(call)
                out.append(("call", call))
            else:
                out.append(("text", f"{self._raw_open}{self._buf}"))
            self._buf = ""
            self._inside = False
            return out

    def _drain_outside(self, final: bool) -> list[tuple[str, object]]:
        out: list[tuple[str, object]] = []
        while True:
            low = self._buf.lower()
            at = low.find(OPEN.lower())
            fn_at = low.find(FUNC_OPEN.lower())
            if fn_at != -1 and (at == -1 or fn_at < at):
                # "<function=…>" with no wrapper. Treat the rest as one block.
                head = self._buf[:fn_at]
                block = self._buf[fn_at:]
                call = parse_block(block) if PARAM_RE.search(block) else None
                if call and self.names is not None and call["name"] not in self.names:
                    call = None        # a description of the format, not a request
                if call and (final or _PARAM_TAIL.search(block)):
                    self._buf = ""
                    self._inside = False
                    if head:
                        out.append(("text", head))
                    self.calls.append(call)
                    out.append(("call", call))
                    return out
                if final:
                    self._buf = ""
                    if block:
                        out.append(("text", head + block))
                    return out
                if head:
                    self._buf = self._buf[fn_at:]
                    out.append(("text", head))
                return out
            if at == -1:
                if final:
                    out.append(("text", self._buf))
                    self._buf = ""
                    return out
                keep = _partial_tag(self._buf)
                if keep > 0:
                    out.append(("text", self._buf[:keep]))
                    self._buf = self._buf[keep:]
                return out
            head = self._buf[:at]
            self._raw_open = self._buf[at:at + len(OPEN)]
            self._buf = self._buf[at + len(OPEN):]
            self._inside = True
            if head:
                out.append(("text", head))
            out.extend(self._drain_inside(final))
            return out

    # -- public ------------------------------------------------------------
    def feed(self, chunk: str) -> list[tuple[str, object]]:
        if not chunk:
            return []
        self._buf += chunk
        return self._drain_inside(False) if self._inside else self._drain_outside(False)

    def flush(self) -> list[tuple[str, object]]:
        return self._drain_inside(True) if self._inside else self._drain_outside(True)
