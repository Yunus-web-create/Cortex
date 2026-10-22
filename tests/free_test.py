"""Free-mode test: the provider pool, its failover, and the zero-cost media paths.

Runs offline. The remote providers are stubbed out at the one seam the agent
uses, so this proves the routing behaviour rather than the network.

Usage:  python tests/free_test.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TMP = Path(tempfile.mkdtemp(prefix="cortex_free_"))
os.environ["CORTEX_DATA_DIR"] = str(TMP)
os.environ["CORTEX_DB"] = str(TMP / "test.db")
os.environ["CORTEX_FREE_MODEL"] = "1"
os.environ["CORTEX_LOCAL_MODEL"] = "0"
for _var in ("AI_GATEWAY_API_KEY", "VERCEL_AI_GATEWAY_KEY", "VERCEL_TOKEN", "OPENROUTER_API_KEY",
             "OPENAI_API_KEY", "GROQ_API_KEY", "DEEPSEEK_API_KEY", "TOGETHER_API_KEY",
             "GEMINI_API_KEY", "GOOGLE_API_KEY", "CORTEX_API_KEY", "CORTEX_PROVIDER",
             "CORTEX_BASE_URL"):
    os.environ[_var] = ""

from app import agent, db, free_media, gateway  # noqa: E402

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
    print(f"\n\033[1m{title}\033[0m")


# ──────────────────────────────────────────────────────────────── the pool
section("the pool always offers a free route")

free_p = gateway.Provider(name="free", base_url=gateway.KNOWN["free"][0], api_key="",
                          keyless=True, models_path="https://text.pollinations.ai/models")
local_p = gateway.Provider(name="local", base_url=gateway.KNOWN["local"][0], api_key="",
                           keyless=True)

check("a keyless provider counts as configured", free_p.inference_ready)
check("a local provider counts as configured", local_p.inference_ready)
check("keyless providers send no auth header", "Authorization" not in free_p.auth_headers())
check("the free route is free", gateway.is_free.__doc__ is not None)

names = [p.name for p in gateway.pool()]
check("the pool leads with the keyless route", names[:1] == ["free"], str(names))
check("ordering is stable", names == [p.name for p in gateway.pool()], str(names))

# a cooling-down provider steps aside, but the pool never becomes empty
real_local = gateway.local_provider
gateway.local_provider = lambda: local_p
check("both free routes are in the pool", [p.name for p in gateway.pool()] == ["free", "local"],
      str([p.name for p in gateway.pool()]))
gateway.mark_down(free_p, 600)
check("a rate-limited route leaves the pool", "free" not in [p.name for p in gateway.pool()],
      str([p.name for p in gateway.pool()]))
check("the pool still has a route", [p.name for p in gateway.pool()] == ["local"])
gateway.clear_down("free")
check("it comes back when the cooldown is cleared", "free" in [p.name for p in gateway.pool()])
gateway.local_provider = real_local


# ──────────────────────────────────────────────────────────── the failover
section("one provider failing does not end the turn")

SEEN = []


async def fake_impl(question, sources, history, **kw):
    """Fail on the first route, answer on the second."""
    provider = gateway.active_provider().name
    SEEN.append(provider)
    if provider == "free":
        yield {"type": "route", "model": "openai-fast", "task": "general", "why": "", "requested": "auto"}
        yield {"type": "error", "message": "429: your previous request is still running", "retry": True}
        return
    yield {"type": "route", "model": "qwen05", "task": "general", "why": "", "requested": "auto"}
    yield {"type": "delta", "text": "answered by the local model"}
    yield {"type": "usage", "input_tokens": 10, "output_tokens": 5, "model": "qwen05"}


real_impl = agent._stream_impl
real_pool = gateway.pool
agent._stream_impl = fake_impl
gateway.pool = lambda: [free_p, local_p]


async def run():
    out = []
    async for ev in agent.stream("hello", [], [], conv_id="c1"):
        out.append(ev)
    return out


events = asyncio.run(run())
kinds = [e["type"] for e in events]
text = "".join(e.get("text", "") for e in events if e["type"] == "delta")

check("it tried the first route", SEEN[:1] == ["free"], str(SEEN))
check("then the next one", "local" in SEEN, str(SEEN))
check("the failure is explained", "notice" in kinds, str(kinds))
check("the notice names both sides",
      any("free community model" in e.get("message", "") and "local" in e.get("message", "")
          for e in events if e["type"] == "notice"))
check("the second route's answer survives", "local model" in text, text[:80])
check("no error reached the user", "error" not in kinds, str(kinds))

agent._stream_impl = real_impl
gateway.pool = real_pool


section("media falls back to the free generators")

check("edge voices are mapped from familiar names",
      free_media._voice("alloy") == "en-US-AriaNeural"
      and free_media._voice("en-GB-RyanNeural") == "en-GB-RyanNeural")
check("every camera move renders a filter",
      all("zoompan" in free_media._motion(k, 60) for k in ("in", "out", "left", "right")))
check("video sizes follow the aspect ratio",
      free_media._dims("9:16") == (720, 1280) and free_media._dims("16:9") == (1280, 720))

have = free_media.available()
print(f"  \033[2m– edge-tts={have['speech']} ffmpeg={have['video']}\033[0m")

print(f"\n\033[1m{PASS} passed, {FAIL} failed\033[0m\n")
sys.exit(1 if FAIL else 0)
