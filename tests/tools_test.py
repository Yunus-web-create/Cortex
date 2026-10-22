"""Cortex tool-layer test — external API calling, end to end against the live app.

Network-dependent checks (weather, Wikipedia) are skipped automatically when
there is no outbound internet, so the suite still passes offline.

Usage:  python tests/tools_test.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TMP = Path(tempfile.mkdtemp(prefix="cortex_tools_"))
os.environ["CORTEX_DATA_DIR"] = str(TMP)
os.environ["CORTEX_FREE_MODEL"] = "0"
os.environ["CORTEX_LOCAL_MODEL"] = "0"
# hermetic: a developer's real .env must not leak into these assertions
for _var in ("AI_GATEWAY_API_KEY", "VERCEL_AI_GATEWAY_KEY", "VERCEL_TOKEN", "OPENROUTER_API_KEY",
             "OPENAI_API_KEY", "GROQ_API_KEY", "DEEPSEEK_API_KEY", "TOGETHER_API_KEY",
             "GEMINI_API_KEY", "GOOGLE_API_KEY", "CORTEX_API_KEY", "CORTEX_PROVIDER",
             "CORTEX_BASE_URL"):
    os.environ[_var] = ""

os.environ["CORTEX_DB"] = str(TMP / "test.db")

from fastapi.testclient import TestClient  # noqa: E402

from app import db, tools  # noqa: E402
from app.main import app  # noqa: E402

client = TestClient(app)
PASS, FAIL, SKIP = 0, 0, 0
GREEN, RED, YELLOW, DIM, OFF = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  {GREEN}✓{OFF} {name}")
    else:
        FAIL += 1
        print(f"  {RED}✗{OFF} {name} {extra}")


def skip(name, why):
    global SKIP
    SKIP += 1
    print(f"  {YELLOW}–{OFF} {name} {DIM}({why}){OFF}")


def sse(resp):
    events = []
    for line in resp.iter_lines():
        if line.startswith("event:"):
            events.append([line[6:].strip(), None])
        elif line.startswith("data:") and events:
            events[-1][1] = json.loads(line[5:].strip())
    return events


def frame(events):
    return [(n, d) for n, d in events if n]


def ask(question, **extra):
    conv = client.post("/api/conversations", json={}).json()
    payload = {"conversation_id": conv["id"], "message": question, **extra}
    return frame(sse(client.post("/api/chat", json=payload)))


def load_demo():
    files = sorted((ROOT / "demo_docs").glob("*.md"))
    if not files:
        return
    for f in files:
        client.post("/api/documents",
                    files=[("files", (f.name, f.read_bytes(), "text/markdown"))])


print("\n\033[1mtool registry\033[0m")
cat = client.get("/api/tools").json()
names = [t["name"] for t in cat["tools"]]
check("every built-in tool registered", len(names) == 14, str(names))
check("workspace, media and research tools all present",
      {"write_file", "run_code", "generate_image", "text_to_speech",
       "generate_video", "web_search", "weather"}.issubset(set(names)), str(names))
check("catalog carries labels + descriptions",
      all(t["label"] and t["description"] for t in cat["tools"]))
check("every tool starts enabled", all(t["enabled"] for t in cat["tools"]))
check("usable list excludes key-less web search",
      "web_search" not in cat["available"] if not (os.getenv("TAVILY_API_KEY") or os.getenv("BRAVE_API_KEY"))
      else "web_search" in cat["available"])
check("needs hint explains the missing key",
      any("TAVILY" in (t.get("needs") or "") for t in cat["tools"] if t["name"] == "web_search"))
check("no secret values leak through the API", "value" not in json.dumps(cat))

print("\n\033[1mlive tool calls (no model in the loop)\033[0m")
r = client.post("/api/tools/calculator/test", json={"arguments": {"expression": "18/100*4750"}}).json()
check("calculator returns an exact result", r["ok"] and r["result"]["result"] == 855.0, str(r)[:120])

r = client.post("/api/tools/calculator/test", json={"arguments": {"expression": "__import__('os').system('id')"}}).json()
check("calculator refuses code injection", not r["ok"], str(r)[:120])

r = client.post("/api/tools/fetch_url/test", json={"arguments": {"url": "http://localhost:8000/api/health"}}).json()
check("SSRF guard blocks localhost", not r["ok"] and "not allowed" in r["result"]["error"])

r = client.post("/api/tools/fetch_url/test", json={"arguments": {"url": "http://169.254.169.254/latest/meta-data/"}}).json()
check("SSRF guard blocks cloud metadata IP", not r["ok"])

r = client.post("/api/tools/fetch_url/test", json={"arguments": {"url": "file:///etc/passwd"}}).json()
check("non-http schemes rejected", not r["ok"] and "only http" in r["result"]["error"])

r = client.post("/api/tools/does_not_exist/test", json={"arguments": {}})
check("unknown tool returns 404", r.status_code == 404)

try:
    r = client.post("/api/tools/weather/test", json={"arguments": {"place": "Johannesburg"}}).json()
    ok_wx = r["ok"] and isinstance(r["result"].get("current", {}).get("temp_c"), (int, float))
except Exception:
    ok_wx = False
if ok_wx:
    check("weather reaches Open-Meteo and returns a temperature", True)
else:
    skip("weather reaches Open-Meteo", "no outbound network")

try:
    r = client.post("/api/tools/wikipedia/test", json={"arguments": {"query": "Retrieval-augmented generation"}}).json()
    ok_wp = r["ok"] and len(r["result"].get("extract", "")) > 100
except Exception:
    ok_wp = False
if ok_wp:
    check("wikipedia returns an article extract", True)
else:
    skip("wikipedia returns an article extract", "no outbound network")

print("\n\033[1mthe model sees tools and can call them\033[0m")
load_demo()
events = ask("What is 18% of 4750?")
kinds = [n for n, _ in events]
check("stream announces the available tools", "tools" in kinds)
check("tool_start event carries name + arguments",
      any(n == "tool_start" and d.get("name") == "calculator" for n, d in events))
check("tool_result event carries the real result",
      any(n == "tool_result" and d.get("result", {}).get("result") == 855 for n, d in events))
answer = "".join(d.get("text", "") for n, d in events if n == "delta")
check("answer states the computed number", "855" in answer, answer[:120])
check("usage reports the tool call", any(n == "usage" and d.get("tool_calls") == 1 for n, d in events))

events = ask("What's the weather in Johannesburg?")
check("weather is chosen for a weather question",
      any(n == "tool_start" and d.get("name") == "weather" for n, d in events))
answer = "".join(d.get("text", "") for n, d in events if n == "delta")
check("weather answer quotes a real temperature", "°C" in answer, answer[:120])

events = ask("What is the parental leave policy?")
check("document question calls no tool", not any(n == "tool_start" for n, _ in events))
check("document answer still cites sources",
      any(n == "delta" and "[" in d.get("text", "") for n, d in events))

events = ask("What is the parental leave policy?", tools=False)
check("tools can be switched off per request", not any(n == "tools" for n, _ in events))

print("\n\033[1mconfiguration\033[0m")
cfg = client.put("/api/tools", json={"enabled": ["calculator", "current_datetime"]}).json()
check("disable persists", sorted(cfg["config"]["enabled"]) == ["calculator", "current_datetime"])
check("disabled tools leave the available list",
      cfg["available"] == ["calculator", "current_datetime"], str(cfg["available"]))
events = ask("What's the weather in Johannesburg?")
check("a disabled tool is never called", not any(n == "tool_start" and d.get("name") == "weather" for n, d in events))
r = client.post("/api/tools/weather/test", json={"arguments": {"place": "Cape Town"}})
check("testing a disabled tool is refused", r.status_code == 400)

client.put("/api/tools", json={"enabled": ["calculator", "current_datetime", "web_search",
                                           "weather", "wikipedia", "fetch_url"]})
client.put("/api/tools", json={"http_allowlist": ["api.open-meteo.com"]})
r = client.post("/api/tools/fetch_url/test",
                json={"arguments": {"url": "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"}}).json()
check("allowlist blocks hosts outside it", not r["ok"] and "allowlist" in r["result"]["error"].lower())
r = client.post("/api/tools/fetch_url/test",
                json={"arguments": {"url": "https://api.open-meteo.com/v1/forecast?latitude=-26.2&longitude=28.0&current=temperature_2m",
                                    "max_chars": 200}}).json()
check("allowlist permits the listed host", r["ok"], str(r)[:140])

client.put("/api/tools", json={"http_allowlist": [],
                               "secrets": {"api.example.com": {"header": "Authorization", "value": "Bearer sekrit"}}})
cfg = client.get("/api/tools").json()
wire = json.dumps(cfg)
check("stored credentials never come back over the wire", "sekrit" not in wire)
check("credential host is reported without the value",
      cfg["config"]["secrets"].get("api.example.com", {}).get("header") == "Authorization" and
      "value" not in cfg["config"]["secrets"].get("api.example.com", {}))
client.put("/api/tools", json={"secrets": {}})

restored = client.put("/api/tools", json={"enabled": ["calculator", "current_datetime", "web_search",
                                                       "weather", "wikipedia", "fetch_url"]}).json()
check("cleanup restores every tool", len(restored["config"]["enabled"]) == 6)

print(f"\n\033[1m{PASS} passed, {FAIL} failed" + (f", {SKIP} skipped" if SKIP else "") + f"{OFF}\n")
sys.exit(1 if FAIL else 0)
