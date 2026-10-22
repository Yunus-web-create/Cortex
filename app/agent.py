"""The agent: route a question to the best model, then let it use tools.

Flow:
    route  -> pick the best available model for this kind of task
    stream -> the model answers; if it asks for tools, run them and continue
    finish -> usage, media artefacts, and any citation bookkeeping

Tools cover the workspace (write/read/run code), generation (image, speech,
video), research (web search + page reading) and the small deterministic ones
(calculator, clock, weather, Wikipedia, URL fetch).
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import AsyncIterator

from . import config, gateway, llm, router, tools, web
from .tooltext import ToolTextFilter

log = logging.getLogger("cortex.agent")

MAX_ROUNDS = int(getattr(config, "TOOL_MAX_ROUNDS", 4))
# How long to wait for the first byte of an answer before trying a different model.
# A free endpoint can sit in a shared queue indefinitely; this is the difference
# between "thinking" and "broken".
PATIENCE = {"reasoning": 45.0, "code": 25.0, "creative": 30.0, "vision": 30.0}
PATIENCE_DEFAULT = float(getattr(config, "FIRST_TOKEN_TIMEOUT", 20.0))
# Total wall-clock budget for a turn, tool rounds included. After this the model is
# asked to answer with what it already has instead of starting another round.
TURN_BUDGET = float(getattr(config, "TURN_BUDGET", 75.0))
_CITE_FIX = re.compile(r"[\u3010\uff3b](\d{1,2})[\u3011\uff3d]")
_LOCAL_IMG = re.compile(r"!\[[^\]]*\]\((?!https?:)[^)]*\)")
_FILE_LINK = re.compile(r"\[[^\]]*\.(?:mp4|mov|webm|png|jpe?g|webp|mp3|wav|ogg)\]", re.I)
# a markdown link pointing at a local media path: [image](abc123/file.png)
_LOCAL_LINK = re.compile(r"\[((?:[^\[\]]|\[[^\]]*\])*)\]\((?!https?:|#|mailto:)[^)]*\)")
_RAW_MEDIA = re.compile(r"<\s*/?\s*(?:video|audio|source|img)\b[^>]*>", re.I)
# some models narrate the file instead of linking it: {audio file: speech-x.mp3}
_MEDIA_BRACE = re.compile(
    r"[\{\[(<]\s*(?:the\s+)?(?:audio|video|image|speech|sound|media|file)\s*"
    r"(?:file|clip|track)?\s*[:=]\s*[^\}\])>]{0,120}[\}\])>]", re.I)
# and sometimes add a stage direction under it
_STAGE = re.compile(r"^\s*[*_]{1,2}\(?\s*(?:listen|watch|view|play|see)\s+below\s*\)?[*_]{1,2}\s*$", re.I | re.M)

BASE_SYSTEM = """You are Cortex, an assistant for {org}.

You are not limited to uploaded documents. You can search the live internet and
you have a code workspace where you can write and run real programs. Reach for
those when the question needs them, then answer plainly.

Voice
- Write like a knowledgeable person talking to a colleague: direct, warm, calm.
  Not a press release, not a lecture. Contractions are fine.
- Answer first. The opening sentence should answer the question; context comes
  after. Never open with "Great question", "Certainly", "I'd be happy to", or a
  restatement of what was asked.
- Match the size of the answer to the question. A simple question gets a few
  sentences, not headings and bullet lists. Go long only when the subject
  genuinely has parts, and then use structure to make it scannable.
- Prefer prose. Use a list only when the content is really a list (steps,
  options, sources). Use a heading only when the answer has three or more
  sections, and never nest more than one level.
- Bold sparingly — a few key terms, not every other phrase. No emoji unless the
  person used them first.
- Do not narrate your process, in any tense. "I'll check the current conditions",
  "I searched", "let me look that up" and "I'll use the tool" are all the same
  mistake: the tool line above the answer already says what ran. Open with the
  substance. If the first thing you know is that it is 32 °C in Rustenburg, say
  that, not that you are about to find out.
- Finish when the answer is finished. No summary paragraph, no "let me know if
  you need anything else", no question-for-the-sake-of-it.
- Keep the first line tight. Roughly: answer, then support, then stop. Offer a next step only
  when it is genuinely useful.

Accuracy
- Cite a source with its numbered bracket, e.g. [1] or [2][3], only when the
  claim comes from a numbered passage, and place it right after the claim. Use
  plain ASCII brackets — never full-width 【1】.
- Statements from your own knowledge, or from a tool whose output has no
  numbered source, get no bracket; name where it came from in words instead
  ("per open-meteo", "from the search results") only if it matters.
- Your training data may be older than the date below. For anything that
  changes — prices, versions, releases, who holds a role, "latest", "today" —
  trust the sources over your memory.
- If you are unsure, say what you are unsure about and what would settle it.
  Never invent a fact, a URL, a file, a library or a tool result.

Code
- Write complete, runnable files with the workspace tools, then run them.
- Report what actually happened, including errors. Never claim a test passed
  unless you saw it pass.

Formatting: markdown. Fenced code blocks with a language tag. Tables only for
comparing things side by side.
{extra}"""

TOOL_GUIDANCE = """
Tool notes:
- Tool results are already rendered for the user as cards and media above your
  reply. Never paste a tool's raw JSON or repeat its output verbatim - summarise
  what it means in one or two lines.
- deep_research searches the web and reads the top pages; prefer it over guessing
  about current events. fetch_url reads one specific URL you already have.
- write_file/read_file/list_files/run_code share one workspace per conversation.
  Build in small steps and run what you write.
- generate_image, text_to_speech and generate_video create real media that is
  shown to the user in the chat automatically. Use them when the user asks for
  media, and never embed the result as markdown - just say it is ready.
- Don't search for things that don't change: definitions, how something works,
  well-known history, arithmetic, explanations. Answer from what you know and
  keep the turn fast.
- Search when the answer depends on something current or local - news, prices,
  schedules, versions, releases, who holds a role, "today", "latest", "near me",
  or a fact you are genuinely unsure of.
- Never call a tool to "look something up" that you already answered above.
- One or two searches is normally enough. If two searches and a page fetch still
  have not settled it, answer with what you have and say what is uncertain -
  do not keep searching round after round.
- Ask for tools through the tool interface only. Never write tool syntax, XML
  tags or JSON calls into your answer text.
"""

STRICT_EXTRA = ("- STRICT MODE: answer only from the numbered sources. If they don't cover it, say so "
                "briefly and stop. Tools may be used to check a fact, not to introduce new topics.")
LOOSE_EXTRA = ("- When the sources don't cover the question, answer from your own knowledge. No need to "
               "label it unless the distinction is the point.")

MEDIA_TOOLS = {"generate_image", "text_to_speech", "generate_video"}

# ── answer hygiene ─────────────────────────────────────────────────────────
# Small models sometimes paste a tool's JSON straight into the reply. It is
# unreadable and it contradicts the tool card above it, so it gets scrubbed.
_JSON_KEYS = ('"stdout"', '"stderr"', '"temp_c"', '"results"', '"sources"', '"path"',
              '"exit"', '"expression"', '"current"', '"place"', '"extract"', '"bytes"',
              '"next_days"', '"kind"', '"url"', '"command"')
_FENCE = re.compile(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```")
_INLINE = re.compile(r"\{[^{}]{140,6000}\}", re.S)


def _is_tool_json(blob: str) -> bool:
    return sum(k in blob for k in _JSON_KEYS) >= 2


def _summarise(name: str, result: dict) -> str:
    """Plain-language fallback for when the model's own text was just JSON."""
    if not name:
        return ""
    if result.get("error"):
        return f"That tool failed: **{result['error']}**"
    if name == "weather":
        c = result.get("current") or {}
        days = result.get("next_days") or []
        out = (f"**{c.get('temp_c', '?')} °C in {result.get('place') or 'the requested place'}** "
               f"- {c.get('conditions', '')}, feels like {c.get('feels_like_c', '?')} °C, "
               f"humidity {c.get('humidity_pct', '?')}%, wind {c.get('wind_kmh', '?')} km/h.")
        if days:
            d = days[0]
            out += f" Today **{d.get('max_c')} °C** max / **{d.get('min_c')} °C** min."
        return out + "\n\n*Live from Open-Meteo.*"
    if name == "calculator":
        return f"`{result.get('expression')}` = **{result.get('result')}**"
    if name == "current_datetime":
        return f"It is **{result.get('readable')}** ({result.get('timezone')})."
    if name == "run_code":
        body = (result.get("stdout") or result.get("stderr") or "(no output)").strip()
        return (f"Ran `{result.get('command', 'the file')}` - exit code "
                f"**{result.get('exit', 0)}**.\n\n```\n{body[:600]}\n```")
    if name == "write_file":
        return f"Wrote **{result.get('path', 'the file')}** into the workspace - it is in the panel and the zip."
    if name in MEDIA_TOOLS:
        return "That is done - the file is shown above."
    if name == "wikipedia":
        return f"**{result.get('title')}**\n\n{(result.get('extract') or '')[:800]}"
    if name in ("web_search", "deep_research"):
        rows = result.get("results") or result.get("sources") or []
        if rows:
            return "**What I found**\n\n" + "\n".join(
                f"- [{x.get('title', '')}]({x.get('url', '')})" for x in rows[:4])
    return ""


def _scrub(text: str) -> str:
    out = _LOCAL_IMG.sub("", text)      # the media card already shows the file
    out = _FILE_LINK.sub("", out)
    def _label(m):
        label = m.group(1).strip().strip("[]").strip()
        if not label or label.lower() in ("image", "video", "audio", "file", "media"):
            return ""
        if re.search(r"\.(?:mp4|mov|webm|png|jpe?g|webp|mp3|wav|ogg|pdf|py|js)$", label, re.I):
            return ""
        return label
    out = _LOCAL_LINK.sub(_label, out)
    out = _RAW_MEDIA.sub("", out)          # the player below already plays it
    out = _MEDIA_BRACE.sub("", out)        # "{audio file: speech-...mp3}"
    out = _STAGE.sub("", out)              # "*(listen below)*"
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"[ \t]+([.,!?;:])", r"\1", out)   # "Here you go . Enjoy"
    out = _FENCE.sub(lambda m: "" if _is_tool_json(m.group(1)) else m.group(0), out)
    out = _INLINE.sub(lambda m: "" if _is_tool_json(m.group(0)) else m.group(0), out)
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out



def _is_billing(ev: dict) -> bool:
    """True when the provider refused because the account has no payment method."""
    msg = str(ev.get("message", "")).lower()
    return "payment method" in msg or "customer_verification_required" in msg


COMPACT_SYSTEM = """You are Cortex, a concise assistant for {org}.
Answer in the first line, then one or two short paragraphs. Use the tool result
below if one is present - never invent different numbers. If the sources do not
cover it, say so plainly. Markdown, no filler."""


def system_prompt(org: str, sources: list[dict], strict: bool, with_tools: bool,
                  compact: bool = False) -> str:
    if compact:
        prompt = COMPACT_SYSTEM.format(org=org or "the user")
    else:
        prompt = BASE_SYSTEM.format(org=org or "the user",
                                    extra=STRICT_EXTRA if strict else LOOSE_EXTRA)
    now = datetime.now(timezone.utc).astimezone()
    prompt += (f"\n\nCurrent date and time: {now.strftime('%A, %d %B %Y, %H:%M %Z')} "
               f"({now.isoformat(timespec='minutes')}).")
    if with_tools and not compact:
        prompt += "\n" + TOOL_GUIDANCE
    if sources:
        blocks = []
        for i, s in enumerate(sources, start=1):
            if s.get("kind") == "web":
                head = f'[{i}] WEB · {s.get("doc_name", "page")} · {s.get("url", "")}'
            else:
                loc = f"page {s['page']}" if s.get("page") else ""
                head = f'[{i}] {s.get("doc_name", "document")}{(" · " + loc) if loc else ""}'
                if s.get("heading"):
                    head += f' · "{s["heading"]}"'
            blocks.append(f"<source id=\"{i}\">\n{head}\n\n{s.get('text', '').strip()}\n</source>")
        prompt += "\n\n<sources>\n" + "\n\n".join(blocks) + "\n</sources>"
    return prompt


def build_messages(history: list[dict], question: str) -> list[dict]:
    msgs = []
    for m in history:
        role = m.get("role")
        content = m.get("content") or ""
        if role in ("user", "assistant") and content:
            msgs.append({"role": role, "content": content})
    msgs.append({"role": "user", "content": question})
    return msgs


def tool_schemas(specs: list[dict]) -> list[dict]:
    return [{"type": "function", "function": {
        "name": s["name"], "description": s["description"], "parameters": s["input_schema"]}}
        for s in specs]


def _digest(name: str, result: dict) -> dict:
    """Small, UI-friendly view of a tool result (the model gets the full one)."""
    if "error" in result:
        return {"error": str(result["error"])[:300]}
    if name in MEDIA_TOOLS:
        return {"kind": result.get("kind"), "url": result.get("url"),
                "name": result.get("name"), "mime": result.get("mime"),
                "model": result.get("model")}
    if name in ("write_file", "read_file", "list_files"):
        keep = {k: v for k, v in result.items() if k in ("path", "bytes", "lines", "count", "files")}
        if name == "read_file":
            keep["preview"] = (result.get("content") or "")[:400]
        return keep
    if name == "run_code":
        return {"command": result.get("command"), "exit": result.get("exit"),
                "stdout": (result.get("stdout") or "")[:1200],
                "stderr": (result.get("stderr") or "")[:600], "ms": result.get("ms"),
                "timed_out": result.get("timed_out")}
    if name == "deep_research":
        return {"query": result.get("query"), "provider": result.get("provider"),
                "count": len(result.get("sources") or []),
                "top": [s.get("title", "") for s in (result.get("sources") or [])[:4]],
                "urls": [s.get("url", "") for s in (result.get("sources") or [])[:4]]}
    if name == "web_search":
        rows = result.get("results") or []
        return {"count": len(rows), "top": [r.get("title", "")[:90] for r in rows[:4]],
                "urls": [r.get("url", "") for r in rows[:4]]}
    if name == "weather":
        cur = result.get("current") or {}
        return {"place": result.get("place", ""), "temp_c": cur.get("temp_c"),
                "conditions": cur.get("conditions", "")}
    if name == "calculator":
        return {"expression": result.get("expression"), "result": result.get("result")}
    if name == "current_datetime":
        return {"readable": result.get("readable"), "timezone": result.get("timezone")}
    if name == "wikipedia":
        return {"title": result.get("title"), "extract": (result.get("extract") or "")[:240]}
    if name == "fetch_url":
        return {"url": result.get("url"), "status": result.get("status"),
                "chars": len(result.get("content") or "")}
    return {k: str(v)[:120] for k, v in list(result.items())[:4]}


async def _finish(text: str, last_tool, calls: int, model: str,
                  usage_in: int, usage_out: int) -> AsyncIterator[dict]:
    """Repair the reply if the model echoed raw tool JSON, then close the turn."""
    cleaned = _scrub(_CITE_FIX.sub(r"[\1]", text))
    if not cleaned and last_tool:
        cleaned = _summarise(*last_tool)      # the model ran a tool and never spoke
    if cleaned != text.strip() and cleaned:
        yield {"type": "text_replace", "text": cleaned}
    if not cleaned.strip():
        yield {"type": "delta", "text": (
            "I ran the tools but the model returned no closing text. The raw output is in the "
            "tool cards above." if calls else
            "The model returned an empty answer. Try rephrasing, or pick another model.")}
    free = gateway.free_tier()          # keyless, local, or a no-card free tier (Google AI Studio)
    yield {"type": "usage", "input_tokens": usage_in, "output_tokens": usage_out,
           "model": model, "tool_calls": calls, "tier": "free" if free else "paid",
           "cost_usd": 0.0 if free else llm.estimate_cost(model, usage_in, usage_out)}


async def stream(question: str, sources: list[dict], history: list[dict], *, conv_id: str,
                 model: str | None = None, strict: bool = False, org: str = "",
                 specs: list[dict] | None = None, knowledge: str = "web+docs",
                 has_image: bool = False, _reentered: bool = False) -> AsyncIterator[dict]:
    """Answer, failing over across every provider the install can reach.

    One provider rate-limited, out of credit or simply down should never end the
    turn: the pool in gateway.pool() is tried in order and the reply continues on
    the next one, with a notice saying where it landed.
    """
    chain = gateway.pool()
    if len(chain) <= 1:
        async for ev in _stream_impl(question, sources, history, conv_id=conv_id, model=model,
                                     strict=strict, org=org, specs=specs, knowledge=knowledge,
                                     has_image=has_image, _reentered=_reentered):
            yield ev
        return

    dropped_pin = False
    for i, p in enumerate(chain):
        token = gateway.push_provider(p)
        failed = None
        # A pinned model id belongs to one provider's catalogue. Carrying it into the
        # next provider produced "model … is not found" 404s; after the first provider
        # the turn re-routes itself instead.
        attempt_model = model if (i == 0 or not model) else None
        if i > 0 and model and not dropped_pin:
            dropped_pin = True
            yield {"type": "notice", "message": (
                f"{gateway.provider_label(p)} does not serve {model.split('/')[-1]} — "
                "letting it pick its own best model for this turn.")}
        try:
            for attempt in range(2):        # one quick retry before giving up on a provider
                produced = False
                touched = False
                failed = None
                async for ev in _stream_impl(question, sources, history, conv_id=conv_id,
                                             model=attempt_model, strict=strict, org=org, specs=specs,
                                             knowledge=knowledge, has_image=has_image,
                                             _reentered=_reentered):
                    kind = ev.get("type")
                    if kind == "delta" and str(ev.get("text", "")).strip():
                        produced = True
                    if kind in ("tool_start", "tool_result"):
                        touched = True
                    if kind == "error":
                        failed = ev
                        if produced or i + 1 >= len(chain) or not _failover_ok(ev):
                            yield ev
                            return
                        break
                    yield ev
                if failed is None:
                    return                  # the turn finished normally
                if attempt == 1 or touched:
                    break
                await asyncio.sleep(1.5)    # a short 503 often clears instantly
        finally:
            gateway.pop_provider(token)
        quota = "quota" in str(failed.get("message", "")).lower()
        gateway.mark_down(p, 3600 if _is_billing(failed) else 60 if quota else 600)
        nxt = chain[i + 1]
        reason = ("is at its free-tier limit for a moment" if quota
                  else "needs a payment method" if _is_billing(failed) else "is unavailable")
        yield {"type": "notice", "message": (
            f"{gateway.provider_label(p)} {reason} - carrying on with "
            f"{gateway.provider_label(nxt)}.")}
    yield {"type": "error", "message": (
        "Every free route is busy right now. Nothing was lost - send the message again in a minute, "
        "or install the local model (scripts/install_local_model.sh) for an offline fallback.")}


def _failover_ok(ev: dict) -> bool:
    """Is this failure about the provider rather than about the question?"""
    msg = str(ev.get("message", "")).lower()
    if not msg or "no model provider" in msg:
        return True
    if ev.get("retry"):
        return True
    marks = ("429", "402", "401", "403", "500", "502", "503", "504", "payment method",
             "timeout", "too long", "network", "connection", "overloaded", "capacity")
    return any(m in msg for m in marks)


async def _stream_impl(question: str, sources: list[dict], history: list[dict], *, conv_id: str,
                       model: str | None = None, strict: bool = False, org: str = "",
                       specs: list[dict] | None = None, knowledge: str = "web+docs",
                       has_image: bool = False, _reentered: bool = False) -> AsyncIterator[dict]:
    """One provider: route, stream, run tools, finish."""
    specs = tools.available_specs() if specs is None else specs
    decision = await router.resolve(question, override=model)

    if not decision.get("model"):
        # no model for this task class: say so plainly, then answer in text
        try:
            cat = await gateway.models()
        except Exception:
            cat = []
        missing = decision.get("task", "")
        fallback = (router.candidates("general", cat, limit=1) or [""])[0]
        if fallback:
            # image / speech / video have free local generators behind the tools,
            # so a text model plans the turn and the tool makes the artefact
            decision = {**decision, "model": fallback,
                        "why": decision.get("why", "") or f"text model plans it; the {missing} tool renders locally"}

    yield {"type": "route", "model": decision["model"], "task": decision["task"],
           "why": decision.get("why", ""), "requested": decision.get("requested", "auto"),
           "provider": gateway.provider_label(gateway.active_provider()),
           "free": gateway.free_tier(), "free_tier": gateway.free_tier()}

    if not gateway.configured():
        async for ev in demo_answer(question, sources, specs, history):
            yield ev
        return

    chosen = decision["model"]
    fallbacks = [m for m in decision.get("fallbacks") or [] if m != chosen]
    compact = gateway.provider_name() == "local"      # small models drown in a long prompt
    system = system_prompt(org, sources, strict, bool(specs), compact=compact)
    past = history[-4:] if compact else history
    messages = [{"role": "system", "content": system}] + build_messages(past, question)
    schemas = tool_schemas(specs)
    ctx = {"conv_id": conv_id}

    usage_in = usage_out = 0
    calls = 0
    last_tool: tuple[str, dict] = ("", {})
    text_seen = ""

    if gateway.provider_name() == "local" and specs:
        # Small on-device models are unreliable at choosing tools. Run the
        # deterministic plan first, hand them the result, and let them phrase it.
        try:
            plan = plan_tool(question, {s["name"] for s in specs})
        except Exception:
            plan = None
        if plan:
            name, args = plan
            yield {"type": "tool_start", "id": "plan_1", "name": name,
                   "label": tools.label_for(name), "arguments": args}
            result = await tools.execute(name, args, ctx)
            last_tool = (name, result)
            calls += 1
            yield {"type": "tool_result", "id": "plan_1", "name": name,
                   "label": tools.label_for(name), "ok": "error" not in result,
                   "result": _digest(name, result)}
            if name in MEDIA_TOOLS and result.get("url"):
                yield {"type": "media", "kind": result.get("kind", "file"),
                       "url": result["url"], "name": result.get("name", ""),
                       "mime": result.get("mime", ""), "model": result.get("model", "")}
            messages.append({"role": "system", "content": (
                f"A {name} call already ran for this question. Result:\n"
                f"{tools.summarise_for_model(name, result)[:1800]}\n"
                "Answer using that result; do not invent different numbers.")})

    started_at = time.monotonic()
    patience = PATIENCE.get(decision.get("task", ""), PATIENCE_DEFAULT) if not model else PATIENCE_DEFAULT
    for round_no in range(MAX_ROUNDS + 1):
        spent = time.monotonic() - started_at
        out_of_time = spent > TURN_BUDGET
        last_round = round_no == MAX_ROUNDS or out_of_time
        if out_of_time and round_no < MAX_ROUNDS:
            yield {"type": "notice", "message": (
                f"Answering with what the tools already returned — the free route was taking "
                f"{spent:.0f}s.")}
        round_text = ""
        pending: list[dict] = []
        used_model = chosen
        try:
            async for ev in _one_round(used_model, messages, schemas, last_round, patience):
                if ev["type"] == "delta":
                    # cleaned once at the end: a markdown link split across two
                    # stream chunks would otherwise survive both halves
                    round_text += ev["text"]
                    text_seen += ev["text"]
                    yield ev
                elif ev["type"] == "tool_call":
                    pending.append(ev)
                elif ev["type"] == "usage":
                    usage_in += ev.get("input_tokens", 0)
                    usage_out += ev.get("output_tokens", 0)
                elif ev["type"] == "error":
                    if ev.get("park") and float(ev["park"]) > 120:
                        # the whole provider is out (a daily cap, not a busy minute):
                        # switching models inside it would just burn another call
                        yield ev
                        return
                    if ev.get("retry") and fallbacks and not text_seen:
                        nxt = fallbacks.pop(0).split("/")[-1]
                        busy = "429" in str(ev.get("message", "")) or "503" in str(ev.get("message", ""))
                        yield {"type": "notice", "message": (
                            f"That model is at its free-tier limit — trying {nxt}." if busy
                            else f"{ev['message']} — retrying on {nxt}")}
                        used_model = nxt
                        break
                    yield ev
                    return
            else:
                if pending:
                    def _call(c: dict) -> dict:
                        out = {"id": c["id"], "type": "function",
                               "function": {"name": c["name"],
                                            "arguments": json.dumps(c["arguments"], ensure_ascii=False)}}
                        if c.get("extra_content"):
                            out["extra_content"] = c["extra_content"]
                        return out

                    messages.append({
                        "role": "assistant",
                        "content": round_text or None,
                        "tool_calls": [_call(c) for c in pending],
                    })
                    for call in pending:
                        calls += 1
                        name = re.sub(r"<[^>]*>.*$", "", str(call.get("name") or "")).strip()
                        call["name"] = name
                        yield {"type": "tool_start", "id": call["id"], "name": name,
                               "label": tools.label_for(name), "arguments": call["arguments"]}
                        result = await tools.execute(name, call["arguments"], ctx)
                        last_tool = (name, result)
                        ok = "error" not in result
                        yield {"type": "tool_result", "id": call["id"], "name": name,
                               "label": tools.label_for(name), "ok": ok,
                               "result": _digest(name, result)}
                        if name in MEDIA_TOOLS and ok and result.get("url"):
                            yield {"type": "media", "kind": result.get("kind", "file"),
                                   "url": result["url"], "name": result.get("name", ""),
                                   "mime": result.get("mime", ""), "model": result.get("model", "")}
                        messages.append({
                            "role": "tool", "tool_call_id": call["id"],
                            "content": tools.summarise_for_model(name, result),
                        })
                    continue
                async for ev in _finish(text_seen, last_tool, calls, used_model,
                                        usage_in, usage_out):
                    yield ev
                return
        except Exception as exc:  # pragma: no cover - defensive
            log.exception("agent loop failed")
            yield {"type": "error", "message": f"{type(exc).__name__}: {exc}"[:300]}
            return

    free = gateway.free_tier()
    yield {"type": "usage", "input_tokens": usage_in, "output_tokens": usage_out,
           "model": chosen, "tool_calls": calls, "tier": "free" if free else "paid",
           "cost_usd": 0.0 if free else llm.estimate_cost(chosen, usage_in, usage_out)}


async def _one_round(model: str, messages: list[dict], schemas: list[dict],
                     last_round: bool, patience: float | None = None) -> AsyncIterator[dict]:
    if last_round:
        yield {"type": "delta", "text": ""}
    force_off = last_round or not schemas
    # Some models (Nemotron on a free route is the usual offender) print the tool
    # call as text instead of using the tool channel. Recover it here so the call
    # actually runs and the markup never reaches the screen.
    filt = ToolTextFilter({(s.get("function") or {}).get("name") for s in schemas} - {None})
    made = 0

    def _emit(call: dict, n: int) -> dict:
        return {"type": "tool_call", "id": f"txt_{n}", "name": call["name"],
                "arguments": call.get("arguments") or {}}

    async for ev in gateway.chat_stream(model, messages, tools=None if force_off else schemas,
                                        max_tokens=config.MAX_TOKENS, temperature=None,
                                        first_token=patience):
        if ev["type"] == "delta":
            for kind, value in filt.feed(ev.get("text") or ""):
                if kind == "text":
                    if value:
                        yield {"type": "delta", "text": value}
                elif not force_off:
                    made += 1
                    yield _emit(value, made)
            continue
        if ev["type"] == "tool_call" and force_off:
            continue
        yield ev
    for kind, value in filt.flush():
        if kind == "text":
            if value:
                yield {"type": "delta", "text": value}
        elif not force_off:
            made += 1
            yield _emit(value, made)


# ------------------------------------------------------------------ demo mode
DEMO_CUES = {
    "wikipedia": r"\bwikipedia\b|\bencyclop(?:a)?edia\b",
    "web_search": r"\bsearch (?:the )?(?:web|internet|online)\b|\blook (?:it )?up online\b|"
                  r"\bgoogle\b|\blatest news\b|\bcurrent news\b",
}


def plan_tool(question: str, enabled: set[str]) -> tuple[str, dict] | None:
    """Decide whether a deterministic tool answers this outright (used by demo mode
    and by the retriever to avoid attaching irrelevant documents)."""
    q = question.strip()
    if not q:
        return None
    if "fetch_url" in enabled:
        m = re.search(r'https?://[^\s<>"\']+', q)
        if m and len(m.group(0)) > 12:
            return "fetch_url", {"url": m.group(0).rstrip(".,;:)"), "max_chars": 1500}
    if "weather" in enabled and re.search(r"\bweather\b|\btemperature\b|\bforecast\b|\brain(?:ing)?\b|\bhot\b|\bcold\b", q, re.I):
        place = _place_from(q)
        if place:
            return "weather", {"place": place}
    if "calculator" in enabled:
        expr = _expression_from(q)
        if expr:
            return "calculator", {"expression": expr}
    if "current_datetime" in enabled and re.search(
            r"\bwhat time is it\b|\bwhat(?:'s| is)? the (?:time|date)\b|\bwhat'?s? today'?s date\b|"
            r"\btoday'?s date\b|\bwhat day is it\b|\bcurrent (?:time|date)\b|\bdate today\b|"
            r"\bhow (?:late|early) is it\b|\bday of the week\b|\bwhat date is it\b", q, re.I):
        tz = _timezone_from(q)
        return "current_datetime", ({"timezone_name": tz} if tz else {})
    for tool_name, pattern in (("wikipedia", DEMO_CUES["wikipedia"]), ("web_search", DEMO_CUES["web_search"])):
        if tool_name in enabled and re.search(pattern, q, re.I):
            term = _search_term(q)
            if term:
                return tool_name, ({"query": term, "max_results": 4} if tool_name == "web_search"
                                   else {"query": term})
    return None


_TIMEZONES = {
    "johannesburg": "Africa/Johannesburg", "cape town": "Africa/Johannesburg", "south africa": "Africa/Johannesburg",
    "london": "Europe/London", "paris": "Europe/Paris", "berlin": "Europe/Berlin", "new york": "America/New_York",
    "san francisco": "America/Los_Angeles", "tokyo": "Asia/Tokyo", "sydney": "Australia/Sydney", "dubai": "Asia/Dubai",
}


def _place_from(q: str) -> str:
    m = re.search(r"\b(?:in|at|for|near)\s+([A-Z][\w'’\-]+(?:[\s,]+[A-Z][\w'’\-]+){0,2})", q)
    if m:
        return m.group(1).split(",")[0].strip()
    m = re.search(r"\b(?:weather|forecast|temperature)\s+(?:like\s+)?(?:in|for|at)\s+(.+)$", q, re.I)
    return (m.group(1).strip(" ?.!") if m else "")


def _timezone_from(q: str) -> str:
    low = q.lower()
    for city, tz in _TIMEZONES.items():
        if city in low:
            return tz
    return ""


def _expression_from(q: str) -> str:
    m = re.search(r"(-?\d[\d\s.,]*(?:%|\s*(?:\+|-|\*|/|\^|x|times|divided by|plus|minus|mod)\s*)\s*-?\d[\d\s.,]*)", q, re.I)
    if m:
        expr = m.group(1).strip().rstrip(".,;:?!")
        if re.search(r"[+\-*/^%x]|times|divided|plus|minus|mod", expr, re.I):
            return expr
    m = re.search(r"what\s+is\s+(\d+(?:\.\d+)?)\s*(?:%|percent)\s+of\s+(\d[\d,.]*)", q, re.I)
    if m:
        return f"{m.group(1)}/100*{m.group(2).replace(',', '')}"
    return ""


def _search_term(q: str) -> str:
    t = re.sub(r"^(?:please\s+)?(?:can you\s+)?(?:search (?:the )?(?:web|internet|online) for|search for|look up|"
               r"look it up online|google|tell me about|who (?:is|was)|what (?:is|was))\s+", "", q.strip(), flags=re.I)
    t = re.sub(r"\b(?:on|in)\s+wikipedia\b|\bwikipedia\b|\bon the (?:web|internet)\b", "", t, flags=re.I)
    return t.strip(" ?.!\"'").strip()


def _clean_number(x) -> str:
    try:
        f = float(x)
        return f"{f:,.0f}" if abs(f - round(f)) < 1e-9 and abs(f) < 1e6 else f"{f:,.1f}"
    except (TypeError, ValueError):
        return str(x)


def _demo_render(tool: str, res: dict, question: str = "") -> str:
    if res.get("error"):
        return (f"I tried the **{tool}** tool and it failed: {res['error']}\n\n"
                f"That is a live API error, not a knowledge-base refusal - check the tool panel in Settings.")
    if tool == "weather":
        c = res.get("current") or {}
        days = res.get("next_days") or []
        if re.search(r"\bday after tomorrow\b", question or "", re.I) and len(days) >= 3:
            d = days[2]
            return (f"**In two days** in {res.get('place', '')}: {_clean_number(d.get('max_c'))} °C max / "
                    f"{_clean_number(d.get('min_c'))} °C min"
                    + (f", {d.get('rain_chance_pct')}% chance of rain" if d.get("rain_chance_pct") is not None else "")
                    + ".\n\n*Live from Open-Meteo.*")
        if re.search(r"\btomorrow\b", question or "", re.I) and len(days) >= 2:
            d = days[1]
            return (f"**Tomorrow in {res.get('place', '')}: {_clean_number(d.get('max_c'))} °C** max / "
                    f"{_clean_number(d.get('min_c'))} °C min"
                    + (f", {d.get('rain_chance_pct')}% chance of rain" if d.get("rain_chance_pct") is not None else "")
                    + (f", {d.get('conditions')}" if d.get("conditions") else "")
                    + ".\n\n*Live from Open-Meteo.*")
        line = (f"**{_clean_number(c.get('temp_c'))} °C in {res.get('place', '')}** right now - "
                f"{c.get('conditions', 'n/a')}, feels like {_clean_number(c.get('feels_like_c'))} °C, "
                f"humidity {c.get('humidity_pct', '?')}%, wind {_clean_number(c.get('wind_kmh'))} km/h.")
        if days:
            d = days[0]
            line += (f"\n\nToday: **{_clean_number(d.get('max_c'))} °C** max / "
                     f"{_clean_number(d.get('min_c'))} °C min"
                     + (f", {d.get('rain_chance_pct')}% chance of rain" if d.get("rain_chance_pct") is not None else "")
                     + ".")
        return line + "\n\n*Live from Open-Meteo.*"
    if tool == "calculator":
        return f"`{res.get('expression')}` = **{_clean_number(res.get('result'))}**"
    if tool == "current_datetime":
        return f"It is **{res.get('readable')}** ({res.get('timezone')})."
    if tool == "wikipedia":
        out = f"**{res.get('title')}**\n\n{res.get('extract', '')}"
        if res.get("url"):
            out += f"\n\n[Read more on Wikipedia]({res['url']})"
        return out
    if tool in ("web_search", "deep_research"):
        rows = res.get("results") or res.get("sources") or []
        if not rows:
            return "The search returned no results."
        return "**Top results**\n\n" + "\n".join(
            f"- [{r.get('title', '')}]({r.get('url', '')})" for r in rows[:4])
    return "```json\n" + json.dumps(res, ensure_ascii=False)[:900] + "\n```"


SMALL_TALK = re.compile(r"^\s*(hi|hey|hello|yo|sup|thanks|thank you|cheers|ok|okay|"
                        r"good (morning|afternoon|evening|night)|how are you)\b", re.I)


def _is_small_talk(question: str) -> bool:
    q = (question or "").strip()
    return bool(SMALL_TALK.match(q)) or len(q.split()) <= 2


async def demo_answer(question: str, sources: list[dict], specs: list[dict],
                      history: list[dict] | None = None) -> AsyncIterator[dict]:
    enabled = {s["name"] for s in specs}
    plan = plan_tool(question, enabled)
    if plan is None:
        # a follow-up ("and tomorrow?", "what about Cape Town?") inherits the
        # subject of the previous user turn
        prev = next((m.get("content") for m in reversed(history or [])
                     if m.get("role") == "user" and m.get("content")), "")
        if prev and len(question.split()) <= 8:
            plan = plan_tool(f"{prev} {question}", enabled)
    if plan is None and sources and not _is_small_talk(question):
        async for ev in llm.demo_answer(question, sources):
            yield ev
        return
    if plan is None and _is_small_talk(question):
        text = ("Demo mode: I can't write a reply without a model key - but document search, web "
                "research and the live tools all work. Ask me something factual, or add a key in "
                "Settings \u2192 Model provider.")
        for chunk in re.findall(r"\S+\s*", text):
            yield {"type": "delta", "text": chunk}
        yield {"type": "usage", "input_tokens": 0, "output_tokens": 0,
               "tool_calls": 0, "cost_usd": 0.0, "demo": True}
        return
    if plan is None:
        if router.classify(question) in ("image", "speech", "video"):
            text = ("Generating images, audio and video needs a model provider. Add a key in "
                    "Settings \u2192 Model provider and ask again - documents, live tools and "
                    "the workspace keep working without one.")
        else:
            text = ("I couldn't reach a model provider, so I can only answer from your documents "
                    "or with the simple local tools. Add an API key in Settings to unlock the full assistant.")
        for chunk in re.findall(r"\S+\s*", text):
            yield {"type": "delta", "text": chunk}
        yield {"type": "usage", "input_tokens": 0, "output_tokens": 0}
        return
    name, args = plan
    call_id = "demo-" + re.sub(r"\W", "", name)[:8]
    yield {"type": "tool_start", "id": call_id, "name": name, "label": tools.label_for(name), "arguments": args}
    res = await tools.execute(name, args, {})
    yield {"type": "tool_result", "id": call_id, "name": name, "label": tools.label_for(name),
           "ok": "error" not in res, "result": _digest(name, res)}
    text = _demo_render(name, res, question)
    for chunk in re.findall(r"\S+\s*", text):
        yield {"type": "delta", "text": chunk}
        await asyncio.sleep(0.006)
    yield {"type": "usage", "input_tokens": 0, "output_tokens": 0, "tool_calls": 1, "cost_usd": 0.0}
