"""Provider ordering: a $0 OpenRouter key must not lead a turn.

OpenRouter's free tier is 50 requests a day; Google's free tier is far larger and
answers faster. When neither is chosen explicitly, the app leads with Google and
keeps OpenRouter in the pool as a fallback. An explicit CORTEX_PROVIDER always
wins, and a funded OpenRouter account leads again once it is proven to have credit.

Usage:  python tests/provider_order_test.py
"""
import os, sys, tempfile
TMP = tempfile.mkdtemp()
os.environ.update(CORTEX_DATA_DIR=TMP, CORTEX_DB=TMP + "/t.db", CORTEX_PROVIDER="",
                  GEMINI_API_KEY="gem-test", OPENROUTER_API_KEY="sk-or-v1-test",
                  CORTEX_FREE_MODEL="0", CORTEX_LOCAL_MODEL="0")
for v in ("AI_GATEWAY_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY", "DEEPSEEK_API_KEY",
          "TOGETHER_API_KEY", "CORTEX_API_KEY", "CORTEX_BASE_URL"):
    os.environ[v] = ""
sys.path.insert(0, "/home/user/cortex")
from app import db, gateway
db.init()

P, F = 0, 0
def check(name, cond, extra=""):
    global P, F
    if cond: P += 1; print("  ok  ", name)
    else: F += 1; print("  FAIL", name, extra)

# no cached credit yet -> unproven -> openrouter cannot spend -> google leads
db.set_kv("openrouter_credit", None)
check("with no proven credit, google leads", gateway.active_provider().name == "google",
      gateway.active_provider().name)
check("openrouter is still in the pool", "openrouter" in [p.name for p in gateway.pool()],
      str([p.name for p in gateway.pool()]))

# credit confirmed -> openrouter may lead again
import time
db.set_kv("openrouter_credit", {"at": time.time(), "spendable": True})
os.environ.pop("CORTEX_PROVIDER", None)
check("with real credit, openrouter leads again", gateway.active_provider().name == "openrouter",
      gateway.active_provider().name)

# an explicit choice always wins
db.set_kv("openrouter_credit", None)
os.environ["CORTEX_PROVIDER"] = "openrouter"
check("an explicit CORTEX_PROVIDER is never overridden",
      gateway.active_provider().name == "openrouter", gateway.active_provider().name)
print(f"\n{P} passed, {F} failed")
sys.exit(1 if F else 0)
