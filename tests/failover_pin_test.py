"""Failover must not carry a pinned model id into a provider that never had it.

Seen live: pinning an OpenRouter model, OpenRouter hitting its free-tier limit, and
the turn rolling to Google — which received the OpenRouter id and answered
"model … is not found". After the first provider the turn re-routes itself.

Usage:  python tests/failover_pin_test.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
TMP = Path(tempfile.mkdtemp(prefix="cortex_failover_"))
os.environ.update(CORTEX_DATA_DIR=str(TMP), CORTEX_DB=str(TMP / "t.db"),
                  CORTEX_FREE_MODEL="0", CORTEX_LOCAL_MODEL="0", CORTEX_RERANK="0")
for _v in ("AI_GATEWAY_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY",
           "OPENAI_API_KEY", "CORTEX_API_KEY", "CORTEX_BASE_URL", "CORTEX_PROVIDER"):
    os.environ[_v] = ""

from app import agent, db, gateway  # noqa: E402

db.init()
PASS = FAIL = 0


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name} {extra}")


class Fake:
    """Two providers; the first one is rate-limited, the second answers."""

    def __init__(self):
        self.models_asked = []

    async def chat_stream(self, model, messages, tools=None, max_tokens=None, temperature=None,
                          first_token=None):
        self.models_asked.append(model)
        if model and "openrouter-only" in model:
            yield {"type": "error", "message": "429: free-models-per-day", "retry": True, "park": 80000.0}
            return
        if model is None or "second" in str(model):
            yield {"type": "delta", "text": "answered by the second provider"}
            yield {"type": "usage", "input_tokens": 5, "output_tokens": 5}
            return
        yield {"type": "error", "message": "429: busy", "retry": True}
        yield {"type": "usage", "input_tokens": 0, "output_tokens": 0}


fake = Fake()
gateway.chat_stream = fake.chat_stream
gateway.pool = lambda: [gateway.Provider(name="first", base_url="https://a.example/v1", api_key="k"),
                        gateway.Provider(name="second", base_url="https://b.example/v1", api_key="k")]
gateway.push_provider = lambda p: None
gateway.pop_provider = lambda t: None
gateway.mark_down = lambda p, seconds=900: None
gateway.provider_label = lambda p: {"first": "Provider One", "second": "Provider Two"}.get(
    getattr(p, "name", ""), "a provider")
gateway.active_provider = lambda: gateway.Provider(name="first", base_url="https://a.example/v1",
                                                   api_key="k")
gateway.free_tier = lambda: True
gateway.configured = lambda: True
agent.router.resolve = (lambda q, override=None: _async({
    "model": override or "second/best-model", "task": "general", "why": "pin" if override else "auto",
    "fallbacks": [], "requested": override or "auto"}))


async def _async(value):
    return value


async def collect():
    return [e async for e in agent.stream("hello", [], [], conv_id="c1",
                                          model="openrouter-only/model-x",
                                          specs=[{"name": "calculator", "description": "d",
                                                  "input_schema": {"type": "object", "properties": {}}}])]


events = asyncio.run(collect())
text = "".join(e.get("text", "") for e in events if e["type"] == "delta")
notices = [e["message"] for e in events if e["type"] == "notice"]

print("\npinned model, first provider fails")
check("the first provider was asked for the pinned model",
      fake.models_asked and "openrouter-only" in (fake.models_asked[0] or ""), str(fake.models_asked))
# the first provider is retried once, so the pinned id legitimately appears twice;
# what matters is that the provider that actually answered was asked for its own model
check("the provider that answered was asked for its own model, not the foreign id",
      fake.models_asked[-1] == "second/best-model" and "openrouter-only" not in fake.models_asked[-1],
      str(fake.models_asked))
check("the user was told the pin was dropped",
      any("does not serve" in n for n in notices), str(notices))
check("the turn still completed", "answered by the second provider" in text, text)
check("no 404-style error reached the user",
      not any(e["type"] == "error" for e in events), str([e for e in events if e["type"] == "error"]))

print(f"\n{PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
