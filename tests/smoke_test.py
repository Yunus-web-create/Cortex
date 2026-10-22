"""Cortex smoke test — runs the whole pipeline with a mocked Claude client.

Usage:  python tests/smoke_test.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TMP = Path(tempfile.mkdtemp(prefix="cortex_test_"))
os.environ["CORTEX_DATA_DIR"] = str(TMP)
os.environ["CORTEX_FREE_MODEL"] = "0"
os.environ["CORTEX_LOCAL_MODEL"] = "0"
os.environ["CORTEX_DB"] = str(TMP / "test.db")

# hermetic: a developer's real .env must not turn demo-mode assertions on its head
for _var in ("AI_GATEWAY_API_KEY", "VERCEL_AI_GATEWAY_KEY", "VERCEL_TOKEN", "OPENROUTER_API_KEY",
             "OPENAI_API_KEY", "GROQ_API_KEY", "DEEPSEEK_API_KEY", "TOGETHER_API_KEY",
             "GEMINI_API_KEY", "GOOGLE_API_KEY", "CORTEX_API_KEY", "TAVILY_API_KEY",
             "BRAVE_API_KEY", "CORTEX_PROVIDER", "CORTEX_BASE_URL"):
    os.environ.setdefault(_var, "")
    os.environ[_var] = ""

from fastapi.testclient import TestClient  # noqa: E402

from app import config, gateway  # noqa: E402
from app.main import app  # noqa: E402

client = TestClient(app)
PASS, FAIL = 0, 0


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  \033[32m✓\033[0m {name}")
    else:
        FAIL += 1
        print(f"  \033[31m✗\033[0m {name} {extra}")


def sse(client_resp):
    events = []
    for line in client_resp.iter_lines():
        if line.startswith("event:"):
            events.append([line[6:].strip(), None])
        elif line.startswith("data:") and events:
            events[-1][1] = json.loads(line[5:].strip())
    return events


# ── fake provider ────────────────────────────────────────────────────────────
captured: dict = {"calls": []}

ANSWER = ("According to [1], employees get **25 days** of annual leave "
          "and up to 5 days can be carried over [1].")


def _reply(system: str, user: str) -> str:
    """Behave like a well-tuned model for each auxiliary prompt."""
    s = system.lower()
    if "expand a search query" in s:
        return "annual leave entitlement days off\ncarried over unused vacation days"
    if "write a title" in s:
        return "Annual leave entitlement"
    if "follow-up questions" in s:
        return "How do I request leave?\nDoes sick leave carry over?\nWhen does leave accrue?"
    if "summarise this conversation" in s:
        return "The user asked about leave. Cortex cited 25 days and 5 carry-over days."
    if "standalone search query" in s:
        return "annual leave days carry over"
    return "query"


async def fake_chat_stream(model, messages, tools=None, max_tokens=None, temperature=None,
                          first_token=None):
    """Stand in for the one seam every provider shares."""
    captured["model"] = model
    captured["tools"] = tools
    captured["system"] = next((m.get("content") or "" for m in messages
                               if m.get("role") == "system"), "")
    captured["messages"] = messages
    import re
    for word in re.findall(r"\S+\s*", ANSWER):
        yield {"type": "delta", "text": word}
    yield {"type": "usage", "input_tokens": 1500, "output_tokens": 48}


CATALOGUE = [
    {"id": "openai/gpt-5.3-codex", "name": "GPT-5.3 Codex", "type": "language",
     "context": 400000, "released": 1790500000, "tags": ["reasoning", "tool-use", "code"]},
    {"id": "anthropic/claude-sonnet-5.5", "name": "Claude Sonnet 5.5", "type": "language",
     "context": 1000000, "released": 1790553600, "tags": ["reasoning", "tool-use", "vision"]},
]


async def fake_models(force: bool = False):
    return CATALOGUE


def fake_status():
    return {"configured": True, "provider": "fake", "base_url": "", "key_kind": "api-key"}


async def fake_complete(model, messages, *, max_tokens=400, temperature=0.2, timeout=60):
    system = next((m.get("content") or "" for m in messages if m.get("role") == "system"), "")
    user = next((m.get("content") or "" for m in reversed(messages) if m.get("role") == "user"), "")
    captured["calls"].append(system[:60])
    return _reply(system, user)


def run():
    print("\n\033[1mCortex smoke test\033[0m\n")

    # 1. health
    h = client.get("/api/health").json()
    check("health endpoint responds", h["status"] == "ok")
    check("no API key → demo mode", h["llm"]["configured"] is False)

    # 2. ingest the demo pack
    r = client.post("/api/demo/load").json()
    check("demo pack ingests", len(r["added"]) == 5, r.get("errors"))
    check("pdf is part of the pack", any(d["ext"] == ".pdf" for d in r["added"]),
          [d["name"] for d in r["added"]])
    total_chunks = sum(d["n_chunks"] for d in r["added"])
    check("documents produce chunks", total_chunks > 25, total_chunks)

    # 3. ingested file types
    from docx import Document as Docx
    d = Docx()
    d.add_heading("Test Policy", level=1)
    d.add_paragraph("The maximum reimbursement for a taxi is R400 per trip.")
    docx_path = TMP / "policy.docx"
    d.save(docx_path)
    with open(docx_path, "rb") as fh:
        up = client.post("/api/documents", files=[("files", ("policy.docx", fh,
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"))])
    check("docx upload works", up.status_code == 200 and len(up.json()["added"]) == 1,
          up.text[:200])

    txt = ("Zebra Protocol: all backups run at 02:00 UTC. Retention is 45 days for standard "
           "tiers and 400 days for regulated customers.\n")
    up2 = client.post("/api/documents", files=[("files", ("notes.txt", txt.encode(), "text/plain"))])
    check("txt upload works", up2.status_code == 200 and len(up2.json()["added"]) == 1)

    bad = client.post("/api/documents", files=[("files", ("virus.exe", b"MZ", "application/octet-stream"))])
    check("unsupported type is rejected", bad.status_code == 200 and len(bad.json()["errors"]) == 1)

    # 4. retrieval
    s = client.get("/api/search", params={"q": "How many annual leave days do employees get?", "k": 5}).json()
    check("hybrid retrieval returns sources", len(s["sources"]) >= 3, len(s["sources"]))
    check("top hit is the handbook", "Handbook" in s["sources"][0]["doc_name"], s["sources"][0]["doc_name"])
    check("semantic + lexical both fire", s["diagnostics"]["semantic_hits"] > 0 and s["diagnostics"]["lexical_hits"] > 0,
          s["diagnostics"])
    check("heading path is cleaned for display", "›" in s["sources"][0]["heading_short"], s["sources"][0].get("heading_short"))

    s2 = client.get("/api/search", params={"q": "zebra backup retention", "k": 3}).json()
    check("new document is searchable", any("notes.txt" in x["doc_name"] for x in s2["sources"]),
          [x["doc_name"] for x in s2["sources"]])
    s3 = client.get("/api/search", params={"q": "taxi reimbursement limit", "k": 3}).json()
    check("docx content is searchable", any("policy.docx" in x["doc_name"] for x in s3["sources"]))
    s_pdf = client.get("/api/search", params={"q": "which sub-processors are used and where?", "k": 3}).json()
    top = s_pdf["sources"][0] if s_pdf["sources"] else {}
    check("pdf text is extracted and searchable", "Data_Retention" in top.get("doc_name", ""),
          top.get("doc_name"))
    check("pdf page numbers are tracked", top.get("page") == 2, top.get("page"))

    # 5. demo chat (no key)
    conv = None
    with client.stream("POST", "/api/chat", json={"message": "How many annual leave days?"}) as resp:
        ev = sse(resp)
    kinds = [e[0] for e in ev]
    check("chat streams start/sources/delta/done",
          all(k in kinds for k in ("start", "sources", "delta", "done")), kinds)
    sources_ev = next(e[1] for e in ev if e[0] == "sources")
    done_ev = next(e[1] for e in ev if e[0] == "done")
    conv = done_ev["conversation_id"]
    text = "".join(e[1]["text"] for e in ev if e[0] == "delta")
    check("demo answer cites passages", "[1]" in text and len(sources_ev["sources"]) > 0)
    check("assistant message persisted", done_ev.get("saved") is not False)

    msgs = client.get(f"/api/conversations/{conv}").json()["messages"]
    check("conversation stored 2 messages", len(msgs) == 2, len(msgs))
    check("assistant message keeps sources", len(msgs[1]["sources"]) > 0)

    # 6. mocked model path (any provider: we fake the gateway seam)
    gateway.chat_stream = fake_chat_stream
    gateway.complete = fake_complete
    gateway.models = fake_models
    gateway.configured = lambda: True
    gateway.status = fake_status
    os.environ["CORTEX_RUNTIME_KEY"] = "sk-ant-test"
    client.put("/api/settings", json={"strict": True})
    h = client.get("/api/health").json()
    check("key in UI switches out of demo mode", h["llm"]["configured"] is True,
          h.get("llm"))

    with client.stream("POST", "/api/chat",
                       json={"message": "How many leave days do I get?", "conversation_id": conv}) as resp:
        ev = sse(resp)
    text = "".join(e[1]["text"] for e in ev if e[0] == "delta")
    usage = next((e[1] for e in ev if e[0] == "usage"), {})
    done = next(e[1] for e in ev if e[0] == "done")
    check("model answer streams through", "25 days" in text, text[:80])
    check("system prompt embeds numbered sources",
          "<sources>" in captured.get("system", "") and "[1]" in captured.get("system", ""))
    check("strict-mode instruction present", "STRICT MODE" in captured.get("system", ""))
    check("history is passed as chat turns", len(captured.get("messages", [])) >= 2,
          len(captured.get("messages", [])))
    check("usage + cost reported", usage.get("input_tokens") == 1500 and usage.get("cost_usd", 0) > 0, usage)
    check("meta saved on message", done.get("meta", {}).get("input_tokens") == 1500,
          done.get("meta"))
    check("route meta names the model that answered",
          bool(done.get("meta", {}).get("model")), done.get("meta", {}).get("model"))

    # 7. regenerate replaces the last answer instead of duplicating the question
    before = len(client.get(f"/api/conversations/{conv}").json()["messages"])
    with client.stream("POST", "/api/chat", json={"conversation_id": conv, "regenerate": True}) as resp:
        ev = sse(resp)
    after_msgs = client.get(f"/api/conversations/{conv}").json()["messages"]
    check("regenerate keeps message count stable", len(after_msgs) == before, (before, len(after_msgs)))
    check("no duplicate user turns after regenerate",
          sum(1 for m in after_msgs if m["role"] == "user") == 2,
          [m["role"] for m in after_msgs])

    # 8. settings round-trip
    st = client.put("/api/settings", json={"org": "Aurora Labs", "model": "claude-haiku-4-5",
                                           "strict": False, "min_relevance": 0.3}).json()
    check("settings persist", st["org"] == "Aurora Labs" and st["model"] == "claude-haiku-4-5"
          and st["strict"] is False)
    with client.stream("POST", "/api/chat", json={"message": "hello", "conversation_id": conv}) as resp:
        sse(resp)
    check("helpful (non-strict) prompt applied",
          "answer from your own knowledge" in captured.get("system", ""))

    # 9. suggestions + export
    sug = client.get("/api/suggestions").json()
    check("suggestions generated", isinstance(sug.get("questions"), list))
    exp = client.get(f"/api/conversations/{conv}/export")
    check("conversation exports to markdown", exp.status_code == 200 and b"# " in exp.content)

    # 10. delete
    docs = client.get("/api/documents").json()["documents"]
    target = [d for d in docs if d["name"] == "notes.txt"][0]
    client.delete(f"/api/documents/{target['id']}")
    left = client.get("/api/documents").json()["documents"]
    check("document deletion works", all(d["id"] != target["id"] for d in left))
    s4 = client.get("/api/search", params={"q": "zebra backup retention", "k": 3}).json()
    check("deleted content is no longer retrievable",
          not any("notes.txt" in x["doc_name"] for x in s4["sources"]))

    # ── 11. quality layer ────────────────────────────────────────────────
    conv2 = None
    with client.stream("POST", "/api/chat",
                       json={"message": "How many leave days do I get?"}) as resp:
        ev = sse(resp)
    kinds = [e[0] for e in ev]
    conv2 = next(e[1] for e in ev if e[0] == "done")["conversation_id"]
    grounding = next((e[1] for e in ev if e[0] == "grounding"), {})
    check("grounding report emitted", grounding.get("claims", 0) > 0, grounding)
    check("cited claim verifies against its source", grounding.get("score", 0) >= 0.5, grounding)
    titles = [e[1] for e in ev if e[0] == "title"]
    check("auto title generated", bool(titles) and "leave" in titles[0]["title"].lower(), titles)
    fu = next((e[1] for e in ev if e[0] == "followups"), {})
    check("follow-up questions offered", len(fu.get("questions", [])) == 3, fu)

    conv_meta = client.get(f"/api/conversations/{conv2}").json()
    check("title persisted on the conversation", "leave" in conv_meta["title"].lower(), conv_meta["title"])
    last_msg = conv_meta["messages"][-1]
    check("grounding stored with the answer", last_msg["meta"].get("grounding", {}).get("claims", 0) > 0)

    # feedback
    mid = last_msg["id"]
    fb = client.post(f"/api/messages/{mid}/feedback", json={"rating": 1})
    check("feedback accepted", fb.status_code == 200 and fb.json()["rating"] == 1)
    bad_fb = client.post(f"/api/messages/{mid}/feedback", json={"rating": 7})
    check("invalid feedback rejected", bad_fb.status_code == 400)
    q = client.get("/api/quality").json()
    check("quality report aggregates feedback", q["rated"] >= 1 and q["positive"] >= 1, q)

    # document scoping
    docs_all = client.get("/api/documents").json()["documents"]
    handbook = next(d for d in docs_all if "Handbook" in d["name"])
    scoped = client.get("/api/search", params={
        "q": "What is the run timeout on the Scale plan?", "k": 5, "docs": handbook["id"]}).json()
    check("doc scope restricts results", all(s["doc_name"] == handbook["name"] for s in scoped["sources"]),
          [s["doc_name"] for s in scoped["sources"]])
    unscoped = client.get("/api/search", params={"q": "What is the run timeout on the Scale plan?", "k": 5}).json()
    check("scoped search still returns hits", len(scoped["sources"]) > 0)
    check("unscoped search reaches other docs",
          any(s["doc_name"] != handbook["name"] for s in unscoped["sources"]),
          [s["doc_name"] for s in unscoped["sources"]])

    # adaptive expansion fires on an unanswerable question
    captured["calls"].clear()
    with client.stream("POST", "/api/chat",
                       json={"message": "What is the CEO's salary and share options?"}) as resp:
        ev = sse(resp)
    check("expansion prompt used on low-confidence question",
          any("expand a search query" in c.lower() for c in captured["calls"]), captured["calls"])

    print(f"\n\033[1m{PASS} passed, {FAIL} failed\033[0m\n")
    return 1 if FAIL else 0


if __name__ == "__main__":
    try:
        code = run()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    sys.exit(code)
