"""Google AI Studio (OpenAI-compatible) test: free-tier honesty, model parking,
tool-call signatures, and the media fallback when the provider is chat-only.

Runs offline. The HTTP layer is a mock transport, so this proves the plumbing
rather than the network.

Usage:  python tests/gemini_test.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TMP = Path(tempfile.mkdtemp(prefix="cortex_gemini_"))
os.environ["CORTEX_DATA_DIR"] = str(TMP)
os.environ["CORTEX_DB"] = str(TMP / "test.db")
os.environ["CORTEX_FREE_MODEL"] = "0"
os.environ["CORTEX_LOCAL_MODEL"] = "0"
for _var in ("AI_GATEWAY_API_KEY", "VERCEL_AI_GATEWAY_KEY", "VERCEL_TOKEN", "OPENROUTER_API_KEY",
             "OPENAI_API_KEY", "GROQ_API_KEY", "DEEPSEEK_API_KEY", "TOGETHER_API_KEY",
             "CORTEX_API_KEY", "CORTEX_BASE_URL"):
    os.environ[_var] = ""
os.environ["GEMINI_API_KEY"] = "test-key"
os.environ["CORTEX_PROVIDER"] = "google"

import httpx  # noqa: E402

from app import agent, db, gateway, router  # noqa: E402

db.init()
PASS, FAIL = 0, 0
GREEN, RED, OFF = "\033[32m", "\033[31m", "\033[0m"


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  {GREEN}✓{OFF} {name}")
    else:
        FAIL += 1
        print(f"  {RED}✗{OFF} {name} {extra}")


def section(title):
    print(f"\n\033[1m{title}{OFF}")


def run(coro):
    return asyncio.run(coro)


CATALOG = [
    {"id": "gemini-3.6-flash", "name": "3.6 Flash", "vendor": "google", "type": "language"},
    {"id": "gemini-3.5-flash", "name": "3.5 Flash", "vendor": "google", "type": "language"},
    {"id": "gemini-flash-latest", "name": "Flash latest", "vendor": "google", "type": "language"},
    {"id": "veo-3.1-generate-001", "name": "Veo", "vendor": "google", "type": "video"},
    {"id": "gemini-2.5-flash-image", "name": "Image", "vendor": "google", "type": "image"},
]


def fake_client(handler):
    """Point the gateway at a mock transport instead of the network."""
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    gateway._client = lambda p, timeout: client  # type: ignore[assignment]
    return client


def sse(*chunks: dict) -> bytes:
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    return body.encode()


def chunk(delta: dict, finish: str | None = None) -> dict:
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


# ------------------------------------------------------------------ provider
section("The Google key is wired in and understood as a no-card free tier")

provider = gateway.active_provider()
check("active provider is google", getattr(provider, "name", "") == "google", str(provider))
check("base url is the OpenAI-compatible endpoint",
      "generativelanguage.googleapis.com/v1beta/openai" in (getattr(provider, "base_url", "") or ""))
check("bearer auth, not a ?key= query",
      provider.auth_headers().get("Authorization") == "Bearer test-key",
      str(provider.auth_headers()))
check("google is advertised as chat-only",
      set(provider.caps or ()) == {"chat"}, str(provider.caps))
check("counted as free tier (no card)", gateway.free_tier() is True)
check("not 'keyless free' — media routes still fall back",
      gateway.is_free() is False)
check("status says free tier", gateway.status().get("tier") == "free")


# -------------------------------------------------------------------- errors
section("A quota 429 is not mistaken for a billing block")

def resp(code, message):
    return httpx.Response(code, json={"error": {"code": code, "message": message}})


quota = run(gateway._error(resp(429, "You exceeded your current quota, please check your plan and "
                                     "billing details. See ai.google.dev/gemini-api/docs/rate-limits")))
check("429 quota → kind='quota'", quota.kind == "quota", str(quota.kind))
check("429 quota is retryable and not billing", quota.retry is True and "payment method" not in str(quota))
check("the word 'billing details' alone does not trip the billing hint",
      "card" not in str(quota).lower())

card = run(gateway._error(resp(403, "Your account has no payment method on file.")))
check("a real card requirement is still caught",
      card.kind == "billing" and "payment method" in str(card).lower())


# -------------------------------------------------------------- tool sig
section("Gemini 3 thought signatures survive the round trip")

seen = {}


def handler(request: httpx.Request) -> httpx.Response:
    seen["body"] = json.loads(request.content.decode())
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse(
        chunk({"role": "assistant", "tool_calls": [{
            "index": 0, "id": "call_1",
            "extra_content": {"google": {"thought_signature": "SIG-123"}},
            "function": {"name": "calculator", "arguments": "{\"expression\": \"2+2\"}"}}]}),
        chunk({}, "tool_calls"),
        {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 3}},
    ))


async def collect_tool_call():
    client = fake_client(handler)
    try:
        return [ev async for ev in gateway.chat_stream("gemini-3.6-flash",
                                                       [{"role": "user", "content": "2+2?"}],
                                                       tools=[{"type": "function", "function": {
                                                           "name": "calculator",
                                                           "parameters": {"type": "object", "properties": {}}}}])]
    finally:
        await client.aclose()


events = run(collect_tool_call())
call = next((e for e in events if e["type"] == "tool_call"), {})
check("tool call is reassembled", call.get("name") == "calculator", str(call))
check("thought signature is kept on the call",
      (call.get("extra_content") or {}).get("google", {}).get("thought_signature") == "SIG-123",
      str(call.get("extra_content")))
check("reasoning is kept shallow on this provider", seen["body"].get("reasoning_effort") == "low",
      str(seen["body"].get("reasoning_effort")))

# the agent has to echo it back in the assistant message
src = (ROOT / "app" / "agent.py").read_text()
check("agent echoes extra_content back to the API", 'out["extra_content"] = c["extra_content"]' in src)


# ------------------------------------------------------------ model parking
section("Models that 429 are parked, and routing steps around them")


def rate_limited(request: httpx.Request) -> httpx.Response:
    return httpx.Response(429, json={"error": {"code": 429, "message": "Quota exceeded"}})


async def hit_limit():
    client = fake_client(rate_limited)
    try:
        return [ev async for ev in gateway.chat_stream("gemini-3.6-flash",
                                                       [{"role": "user", "content": "hi"}])]
    finally:
        await client.aclose()


run(hit_limit())
parked = gateway.model_cooldowns()
check("a 429 parks the model", "gemini-3.6-flash" in parked, str(parked))
check("router skips the parked model",
      "gemini-3.6-flash" not in router.candidates("general", CATALOG, limit=3),
      str(router.candidates("general", CATALOG, limit=3)))
check("the next model in line is used instead",
      router.candidates("general", CATALOG, limit=1) == ["gemini-3.5-flash"],
      str(router.candidates("general", CATALOG, limit=1)))
check("the picker hides parked models", "gemini-3.6-flash" in parked)

gateway.clear_model_down("gemini-3.6-flash")
check("a successful call clears the park", "gemini-3.6-flash" not in gateway.model_cooldowns())


# ------------------------------------------------------------------ routing
section("Chat-only provider: media tasks go to the free renderer, not a dead endpoint")

for task, word in (("video", "video"), ("image", "image"), ("speech", "voice")):
    picked = router.pick(task, CATALOG)
    check(f"{task}: does not pick the {task} model as the chat model",
          picked["model"] in ("gemini-3.6-flash", "gemini-3.5-flash", "gemini-flash-latest"),
          str(picked))
    check(f"{task}: says the free renderer does the work",
          word in picked["why"] and "free" in picked["why"], picked["why"])

pinned = run(router.resolve("draw a cat", override="gemini-2.5-flash-image", catalog=CATALOG))
check("a pinned image model still drives the turn with a chat model",
      pinned["model"] in ("gemini-3.6-flash", "gemini-3.5-flash", "gemini-flash-latest"),
      str(pinned))


# -------------------------------------------------------------------- scrub
section("Model chatter about the file never leaks into the answer")

clean = agent._scrub(
    "Here is the audio.  \n*(listen below)*\n\n{audio file: speech-20261022.mp3}\n"
    "<audio controls src=\"/media/x.mp3\"></audio>"
)
check("brace narration removed", "audio file" not in clean, repr(clean))
check("stage direction removed", "listen below" not in clean, repr(clean))
check("raw tag removed", "<audio" not in clean, repr(clean))
check("the sentence survives", clean.startswith("Here is the audio."), repr(clean))
check("spacing tidied", "  " not in clean, repr(clean))


print(f"\n\033[1m{PASS} passed, {FAIL} failed\033[0m\n")
sys.exit(1 if FAIL else 0)
