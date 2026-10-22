"""OpenRouter test: a $0 account routes only to zero-price models.

The key used here has no credit, which is exactly the "no cost" case: OpenRouter
must never be asked for a paid model. Runs offline — the catalogue and the credit
balance are seeded, no HTTP.

Usage:  python tests/openrouter_test.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TMP = Path(tempfile.mkdtemp(prefix="cortex_openrouter_"))
os.environ["CORTEX_DATA_DIR"] = str(TMP)
os.environ["CORTEX_DB"] = str(TMP / "test.db")
os.environ["CORTEX_FREE_MODEL"] = "0"
os.environ["CORTEX_LOCAL_MODEL"] = "0"
for _var in ("AI_GATEWAY_API_KEY", "VERCEL_AI_GATEWAY_KEY", "VERCEL_TOKEN", "OPENAI_API_KEY",
             "GROQ_API_KEY", "DEEPSEEK_API_KEY", "TOGETHER_API_KEY", "GEMINI_API_KEY",
             "GOOGLE_API_KEY", "CORTEX_API_KEY", "CORTEX_BASE_URL", "OLLAMA_API_KEY"):
    os.environ[_var] = ""
os.environ["OPENROUTER_API_KEY"] = "sk-or-v1-test"
os.environ["CORTEX_PROVIDER"] = "openrouter"

from app import db, gateway, llm, router  # noqa: E402

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


def set_credit(spendable: bool):
    db.set_kv("openrouter_credit", {"at": time.time(), "spendable": spendable})


PAID = {"id": "anthropic/claude-sonnet-5.5", "type": "language", "free": False}
FREE_A = {"id": "nvidia/nemotron-3-ultra-550b-a55b:free", "type": "language", "free": True}
FREE_B = {"id": "minimax/minimax-m3.1-flash:free", "type": "language", "free": True}
FREE_C = {"id": "cohere/north-mini-code:free", "type": "language", "free": True}
CAT = [PAID, FREE_A, FREE_B, FREE_C]


section("The key is wired in as a provider")

provider = gateway.active_provider()
check("active provider is openrouter", getattr(provider, "name", "") == "openrouter", str(provider))
check("base url is the OpenRouter API",
      (provider.base_url or "").startswith("https://openrouter.ai/api/v1"), provider.base_url or "")
check("bearer auth", provider.auth_headers().get("Authorization") == "Bearer sk-or-v1-test")
check("serves chat + media routes", {"chat", "image", "speech", "video"} <= set(provider.caps))


section("A $0 balance means free models only")

set_credit(False)
check("free_only() is true", gateway.free_only() is True)
check("the tier is 'free'", gateway.free_tier() is True)
check("status says free", gateway.status().get("tier") == "free")
kept = [m["id"] for m in router._clean(CAT)]
check("only :free models survive", kept == [FREE_A["id"], FREE_B["id"], FREE_C["id"]], str(kept))
check("paid models are never candidates",
      "anthropic/claude-sonnet-5.5" not in router.candidates("general", CAT, limit=5))
check("routing prefers the strongest free model",
      router.candidates("general", CAT, limit=1) == [FREE_A["id"]],
      str(router.candidates("general", CAT, limit=1)))
check("code uses the code-specialised free model",
      router.candidates("code", CAT, limit=1) == [FREE_C["id"]],
      str(router.candidates("code", CAT, limit=1)))

pick = router.pick("general", CAT)
check("the router explains itself", pick["why"] == "best all-rounder", str(pick))
check("free models cost $0 in the footer", llm.estimate_cost(FREE_A["id"], 50_000, 2_000) == 0.0)
check("a paid id still gets a real estimate", llm.estimate_cost("anthropic/claude-sonnet-5.5", 1000, 100) > 0)


section("With credit, paid models come back")

set_credit(True)
check("free_only() is false", gateway.free_only() is False)
check("the tier is 'paid'", gateway.free_tier() is False)
kept = [m["id"] for m in router._clean(CAT)]
check("paid models are usable again", PAYID_OK := "anthropic/claude-sonnet-5.5" in kept, str(kept))
check(":free duplicates are dropped when paying", all(not m.endswith(":free") for m in kept), str(kept))


section("Free vs paid is read from the catalogue, not guessed")

check("pricing 0/0 counts as free",
      gateway._is_free_priced({"pricing": {"prompt": "0", "completion": "0"}}, "x/y") is True)
check("nonzero pricing is not free",
      gateway._is_free_priced({"pricing": {"prompt": "0.000003", "completion": "0.000015"}}, "x/y") is False)
check("a :free suffix counts even without pricing",
      gateway._is_free_priced({}, "nvidia/nemotron-3-ultra-550b-a55b:free") is True)
check("an unknown model is not assumed free", gateway._is_free_priced({}, "someone/model") is False)


print(f"\n\033[1m{PASS} passed, {FAIL} failed\033[0m\n")
sys.exit(1 if FAIL else 0)
