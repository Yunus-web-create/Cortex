"""A tiny OpenAI-compatible provider for tests.

Speaks just enough of the protocol for the real gateway client: /models,
streaming chat completions with tool calls, images, speech and video.

Run standalone:  python tests/mock_provider.py 8787
"""
from __future__ import annotations

import asyncio
import base64
import json
import re
import sys

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MODELS = [
    {"id": "anthropic/claude-sonnet-5.5", "name": "Claude Sonnet 5.5", "type": "language", "context_window": 1000000, "released": 1790553600, "tags": ["reasoning", "tool-use", "vision"]},
    {"id": "openai/gpt-5.3-codex", "name": "GPT-5.3 Codex", "type": "language", "context_window": 400000, "released": 1790500000, "tags": ["reasoning", "tool-use"]},
    {"id": "openai/gpt-6.1-sol", "name": "GPT-6.1 Sol", "type": "language", "context_window": 1050000, "released": 1791000000, "tags": ["reasoning"]},
    {"id": "google/gemini-4-argon", "name": "Gemini 4 Argon", "type": "language", "context_window": 1000000, "released": 1790900000, "tags": ["vision", "reasoning"]},
    {"id": "google/gemini-3.5-flash-lite", "name": "Gemini 3.5 Flash Lite", "type": "language", "context_window": 1000000, "released": 1790700000, "tags": ["tool-use"]},
    {"id": "openai/gpt-image-2.5-sunburst", "name": "GPT Image 2.5", "type": "image", "released": 1790800000, "tags": []},
    {"id": "openai/tts-1-hd", "name": "TTS 1 HD", "type": "speech", "released": 1700000000, "tags": []},
    {"id": "google/veo-3.1-generate-001", "name": "Veo 3.1", "type": "video", "released": 1790000000, "tags": []},
]

# 1x1 transparent png
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")

STATE = {"calls": [], "video_polls": 0}


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": MODELS}


def _last_user(msgs):
    for m in reversed(msgs):
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, list):
                return "".join(x.get("text", "") for x in c if isinstance(x, dict))
            return c or ""
    return ""


def _tool_payloads(msgs):
    """Tool messages carry only a tool_call_id; recover the names from the assistant turn."""
    names = {}
    for m in msgs:
        for c in (m.get("tool_calls") or []):
            names[c.get("id")] = (c.get("function") or {}).get("name", "")
    out = []
    for m in msgs:
        if m.get("role") == "tool":
            name = m.get("name") or names.get(m.get("tool_call_id"), "")
            try:
                out.append((name, json.loads(m.get("content") or "{}")))
            except Exception:
                out.append((name, {"raw": str(m.get("content"))[:600]}))
    return out


def _plan(user_text, tools, done=frozenset()):
    q = user_text or ""
    if "weather" in tools and "weather" not in done and re_match(q, r"\bweather\b|\bforecast\b|\btemperature\b"):
        m = re.search(r"\b(?:in|at|for|near)\s+([A-Z][\w'’\-]+(?:\s+[A-Z][\w'’\-]+){0,2})", q)
        return ("weather", {"place": m.group(1) if m else "Johannesburg"})
    if "calculator" in tools and "calculator" not in done:
        m = re.search(r"(-?\d[\d\s.,]*(?:%|\s*(?:\+|-|\*|/|\^)\s*)\s*-?\d[\d\s.,]*)", q)
        if m:
            return ("calculator", {"expression": m.group(1).strip()})
        m = re.search(r"what\s+is\s+(\d+(?:\.\d+)?)\s*(?:%|percent)\s+of\s+(\d[\d,.]*)", q, re.I)
        if m:
            return ("calculator", {"expression": f"{m.group(1)}/100*{m.group(2).replace(',', '')}"})
    if "current_datetime" in tools and "current_datetime" not in done and re_match(q, r"\bwhat time\b|\bwhat(?:'s| is) the (?:time|date)\b|\btoday'?s date\b|\bwhat day is it\b"):
        return ("current_datetime", {})
    if "generate_image" in tools and "generate_image" not in done and re_match(q, r"\b(image|picture|photo|draw|illustrat|logo|render)\b"):
        return ("generate_image", {"prompt": q[:300], "size": "1024x1024"})
    if "write_file" in tools and "write_file" not in done and re_match(q, r"\b(script|program|app|game|file|code|function|class)\b") and re_match(q, r"\b(write|build|make|create|implement|generate)\b"):
        lang = "py" if re_match(q, r"python") else ("js" if re_match(q, r"javascript|node|\bjs\b") else "py")
        if lang == "py":
            body = ("def main():\n    print(\"hello from the workspace\")\n\n\n"
                    "if __name__ == \"__main__\":\n    main()\n")
            return ("write_file", {"path": "hello.py", "content": body})
        return ("write_file", {"path": "hello.js", "content": "console.log('hello from the workspace');\n"})
    if "run_code" in tools and "run_code" not in done and re_match(q, r"\b(run|execute|test)\b"):
        fname = "hello.js" if re_match(q, r"\b(javascript|node|js)\b") else "hello.py"
        cmd = f"node {fname}" if fname.endswith(".js") else f"python3 {fname}"
        return ("run_code", {"command": cmd})
    if "text_to_speech" in tools and "text_to_speech" not in done and re_match(q, r"\b(voice|audio|speak|read (?:this|it) aloud|narrat)\b"):
        return ("text_to_speech", {"text": "Cortex is online and ready."})
    if "wikipedia" in tools and "wikipedia" not in done and re_match(q, r"\bwikipedia\b"):
        term = re.sub(r".*\bwikipedia\b(?:\s+for)?\s*", "", q, flags=re.I).strip(" ?.!")
        return ("wikipedia", {"query": term or q})
    if "deep_research" in tools and "deep_research" not in done and re_match(q, r"\b(latest|news|current|today'?s?|2026|price|release)\b"):
        return ("deep_research", {"question": q[:300]})
    if "web_search" in tools and "web_search" not in done and re_match(q, r"\b(search|look up|google)\b"):
        return ("web_search", {"query": q[:200], "max_results": 4})
    return None


def _compose(q, payloads):
    if not payloads:
        return ("**Short answer.** This is the demo provider speaking, so the words are canned - but "
                "everything else you see (routing, tools, sources, workspace, media) is the real app.\n\n"
                "Ask for the weather, a calculation, an image, or a small script to watch the tools fire.")
    name, r = payloads[-1]
    if name == "weather" and not r.get("error"):
        c = r.get("current") or {}
        days = r.get("next_days") or []
        out = (f"**{c.get('temp_c', '?')} °C in {(r.get('place') or '')}** right now - {c.get('conditions', '')}, "
               f"feels like {c.get('feels_like_c', '?')} °C, humidity {c.get('humidity_pct', '?')}%, "
               f"wind {c.get('wind_kmh', '?')} km/h.")
        if days:
            d = days[0]
            out += (f"\n\nToday runs **{d.get('max_c')} °C** max / **{d.get('min_c')} °C** min"
                    + (f" with a {d.get('rain_chance_pct')}% chance of rain" if d.get("rain_chance_pct") is not None else "")
                    + ".")
        return out + "\n\n*Live from Open-Meteo.*"
    if name == "calculator":
        return f"`{r.get('expression')}` = **{r.get('result')}**"
    if name == "current_datetime":
        return f"It is **{r.get('readable')}** ({r.get('timezone')})."
    if name == "write_file":
        return (f"Wrote **{r.get('path', 'the file')}** ({r.get('bytes', 0)} bytes, "
                f"{r.get('lines', 0)} lines) into the workspace.")
    if name == "run_code":
        out = (r.get("stdout") or "").strip()
        err = (r.get("stderr") or "").strip()
        rc = r.get("exit", 0)
        body = out or err or "(no output)"
        tail = " - it ran clean." if rc == 0 else " - it failed, and the error is above."
        return (f"Ran `{r.get('command', 'the file')}` - exit code **{rc}**{tail}\n\n"
                f"```\n{body[:600]}\n```\n\n"
                "The file sits in the workspace panel, and the whole folder downloads as a zip.")
    if name == "generate_image":
        return f"Here is the image for *{q[:80]}*. Ask for a follow-up and I will keep the same style."
    if name == "wikipedia":
        out = f"**{r.get('title')}**\n\n{(r.get('extract') or '')[:800]}"
        return out + (f"\n\n[Source]({r['url']})" if r.get("url") else "")
    if name in ("web_search", "deep_research"):
        rows = r.get("results") or r.get("sources") or []
        if not rows:
            return "The search came back empty, so I am not going to invent an answer."
        return "**What I found**\n\n" + "\n".join(f"- [{x.get('title','')}]({x.get('url','')})" for x in rows[:4])
    if name == "text_to_speech":
        return "Audio is generated - press play above."
    if r.get("error"):
        return f"That tool failed: **{r['error']}**"
    return "Done."


@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    STATE["calls"].append({"model": body.get("model"), "tools": len(body.get("tools") or []),
                           "messages": body.get("messages")})
    msgs = body.get("messages") or []
    stream = bool(body.get("stream"))
    tools_offered = {t["function"]["name"] for t in (body.get("tools") or [])}
    user_text = _last_user(msgs)
    payloads = _tool_payloads(msgs)
    done = {n for n, _ in payloads}
    plan = _plan(user_text, tools_offered, done) if tools_offered and len(payloads) < 3 else None

    if plan:
        name, args = plan
        if not stream:
            return {"id": "c1", "object": "chat.completion", "model": body.get("model"),
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": None,
                                "tool_calls": [{"id": f"call_{name}", "type": "function",
                                                "function": {"name": name, "arguments": json.dumps(args)}}]},
                                 "finish_reason": "tool_calls"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5}}

        async def gen_tool():
            head = {"choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "id": f"call_{name}", "type": "function",
                 "function": {"name": name, "arguments": json.dumps(args)}}]}, "finish_reason": None}]}
            yield f"data: {json.dumps(head)}\n\n"
            tail = {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
            yield f"data: {json.dumps(tail)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen_tool(), media_type="text/event-stream")

    answer = _compose(user_text, payloads) if user_text else "Hello."

    if not stream:
        return {"id": "c1", "object": "chat.completion", "model": body.get("model"),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": answer},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 8}}

    async def gen2():
        for word in re.findall(r"\S+\s*", answer):
            chunk = {"choices": [{"index": 0, "delta": {"content": word}, "finish_reason": None}]}
            yield f"data: {json.dumps(chunk)}\n\n"
            await asyncio.sleep(0.004)
        final = {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": 12, "completion_tokens": len(answer.split())}}
        yield f"data: {json.dumps(final)}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(gen2(), media_type="text/event-stream")


def re_match(text: str, pattern: str) -> bool:
    return bool(re.search(pattern, text or "", re.I))


@app.post("/v1/images/generations")
async def images(req: Request):
    body = await req.json()
    STATE["calls"].append({"kind": "image", "model": body.get("model"), "prompt": body.get("prompt")})
    return {"created": 0, "data": [{"b64_json": base64.b64encode(PNG).decode()}]}


@app.post("/v1/audio/speech")
async def speech(req: Request):
    body = await req.json()
    STATE["calls"].append({"kind": "speech", "model": body.get("model"), "input": body.get("input")})
    return Response(content=b"ID3\x04\x00\x00\x00\x00\x00\x00" + b"\x00" * 64, media_type="audio/mpeg")


@app.post("/v1/video/generations")
async def video(req: Request):
    body = await req.json()
    STATE["calls"].append({"kind": "video", "model": body.get("model"), "prompt": body.get("prompt")})
    return {"id": "job_1", "status": "queued"}


@app.get("/v1/video/generations/{job}")
async def video_poll(job: str):
    STATE["video_polls"] += 1
    return {"id": job, "status": "succeeded", "data": [{"b64_json": base64.b64encode(b"fake-mp4").decode()}]}


@app.get("/_state")
async def state():
    return STATE


@app.post("/_reset")
async def reset():
    STATE["calls"].clear()
    STATE["video_polls"] = 0
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[1]) if len(sys.argv) > 1 else 8787,
                log_level="warning")
