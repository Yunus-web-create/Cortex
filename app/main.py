"""Cortex — HTTP API + static UI host."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from html import escape as html_escape
from urllib.parse import quote
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse, RedirectResponse,
                               Response, StreamingResponse)
from fastapi.staticfiles import StaticFiles

from . import agent, config, db, embeddings, free_media, gateway, ingest, llm, media, quality, retrieval, router, tools, web, workspace

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("cortex")

app = FastAPI(title="Cortex", version="1.0.0", docs_url="/api/docs")
db.init()
embeddings.warm_up()
retrieval.warm_reranker()
DEFAULTS = {
    "org": "your organisation",
    "model": "auto",
    "strict": False,
    "min_relevance": config.MIN_RELEVANCE,
    "show_sources": True,
    "knowledge": "web+docs",
    "image_model": "",
    "speech_model": "",
    "video_model": "",
}
_generation_slots = asyncio.Semaphore(4)

# ------------------------------------------------------------- access control
ACCESS_PASSWORD = os.getenv("CORTEX_PASSWORD", "").strip()
COOKIE = "cortex_access"
COOKIE_MAX_AGE = 60 * 60 * 24 * 90          # 90 days: an installed app should not nag
PUBLIC_PATHS = {"/api/health", "/login", "/logout", "/manifest.webmanifest", "/sw.js",
                "/icons", "/favicon.ico"}
PUBLIC_PREFIXES = ("/icons/",)


def _cookie_token() -> str:
    """Same scheme as scripts/public_gate.py, so a cookie minted by either works."""
    return hmac.new(b"cortex-gate-v1", ACCESS_PASSWORD.encode(), hashlib.sha256).hexdigest()


def _cookie_ok(request: Request) -> bool:
    value = request.cookies.get(COOKIE, "")
    return bool(value) and hmac.compare_digest(value, _cookie_token())


def _basic_ok(request: Request) -> bool:
    header = request.headers.get("authorization", "")
    if not header.lower().startswith("basic "):
        return False
    try:
        user, _, pw = base64.b64decode(header[6:]).decode().partition(":")
        # accept "anyuser:password" or "password:" (password as username)
        return secrets.compare_digest(pw or user, ACCESS_PASSWORD)
    except Exception:
        return False


def _is_https(request: Request) -> bool:
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    return proto.split(",")[0].strip().lower() == "https"


def _safe_next(value: str) -> str:
    value = (value or "/").strip()
    if not value.startswith("/") or value.startswith("//"):
        return "/"
    return value[:500]


def _is_public(path: str) -> bool:
    return path in PUBLIC_PATHS or path.startswith(PUBLIC_PREFIXES)


@app.middleware("http")
async def basic_auth(request: Request, call_next):
    if not ACCESS_PASSWORD or _is_public(request.url.path):
        return await call_next(request)
    if _cookie_ok(request) or _basic_ok(request):
        return await call_next(request)
    # a browser (or an installed app) gets a real login screen, not a native dialog
    accepts = request.headers.get("accept", "")
    if request.method == "GET" and "text/html" in accepts:
        target = request.url.path
        if request.url.query:
            target += "?" + request.url.query
        return RedirectResponse(f"/login?next={quote(target, safe='')}", status_code=303)
    return JSONResponse({"detail": "Authentication required"}, status_code=401,
                        headers={"WWW-Authenticate": 'Basic realm="Cortex"'})


LOGIN_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Sign in · Cortex</title><link rel="icon" href="/icons/favicon-64.png">
<style>
:root{color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;min-height:100dvh;display:grid;place-items:center;padding:24px;
  background:radial-gradient(1200px 600px at 50% -10%,#1d2b4a 0%,#0b0f18 55%);
  color:#e8ecf6;font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
.card{width:100%;max-width:360px;background:#111826cc;border:1px solid #263149;border-radius:20px;
  padding:28px 24px;box-shadow:0 20px 60px #0008;backdrop-filter:blur(8px)}
.mark{width:56px;height:56px;border-radius:16px;display:grid;place-items:center;margin-bottom:16px;
  background:linear-gradient(145deg,#3b82f6,#8b5cf6);font:700 26px/1 ui-sans-serif}
h1{margin:0 0 4px;font-size:20px;letter-spacing:-.01em}
p{margin:0 0 20px;color:#8fa0bd;font-size:13.5px}
label{display:block;font-size:12px;letter-spacing:.04em;text-transform:uppercase;color:#8fa0bd;margin-bottom:6px}
input{width:100%;padding:12px 14px;border-radius:12px;border:1px solid #2b3852;background:#0b1220;
  color:#e8ecf6;font-size:16px;outline:none}
input:focus{border-color:#4f7cf7;box-shadow:0 0 0 3px #4f7cf733}
button{width:100%;margin-top:14px;padding:12px;border:0;border-radius:12px;cursor:pointer;
  background:linear-gradient(145deg,#3b82f6,#6d5cf6);color:#fff;font-size:15px;font-weight:600}
button:active{transform:translateY(1px)}
.err{margin:0 0 14px;padding:10px 12px;border-radius:10px;background:#3b1720;border:1px solid #7f2233;
  color:#ffb4c0;font-size:13px}
.foot{margin-top:18px;color:#5f6f8c;font-size:11.5px;text-align:center}
@media (prefers-reduced-motion:no-preference){.card{animation:in .35s ease-out}}
@keyframes in{from{opacity:0;transform:translateY(8px)}}
</style></head><body>
<form class="card" method="post" action="/login">
  <div class="mark">C</div>
  <h1>Cortex</h1>
  <p>One assistant, every model. Enter your access password.</p>
  __ERROR__
  <label for="password">Password</label>
  <input id="password" name="password" type="password" autocomplete="current-password"
         autofocus required placeholder="••••••••">
  <input type="hidden" name="next" value="__NEXT__">
  <button type="submit">Open Cortex</button>
  <div class="foot">Stays signed in on this device for 90 days.</div>
</form></body></html>"""


def _login_page(next_path: str, error: bool = False) -> str:
    html = LOGIN_HTML.replace("__NEXT__", html_escape(next_path))
    msg = ('<p class="err">That password did not match. Try again.</p>' if error else "")
    return html.replace("__ERROR__", msg)


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, next: str = "/"):
    target = _safe_next(next)
    if not ACCESS_PASSWORD or _cookie_ok(request):
        return RedirectResponse(target, status_code=303)
    return HTMLResponse(_login_page(target))


@app.post("/login")
async def login_submit(request: Request):
    form = await request.form()
    target = _safe_next(str(form.get("next", "/")))
    given = str(form.get("password", ""))
    if given and secrets.compare_digest(given, ACCESS_PASSWORD):
        response = RedirectResponse(target, status_code=303)
        response.set_cookie(COOKIE, _cookie_token(), max_age=COOKIE_MAX_AGE, httponly=True,
                            samesite="lax", secure=_is_https(request), path="/")
        log.info("login ok from %s", request.client.host if request.client else "?")
        return response
    log.info("login failed from %s", request.client.host if request.client else "?")
    return HTMLResponse(_login_page(target, error=True), status_code=401)


@app.get("/logout")
async def logout():
    response = RedirectResponse("/login", status_code=303)
    response.delete_cookie(COOKIE, path="/")
    return response


@app.middleware("http")
async def no_cache_api(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


# ------------------------------------------------------------------- helpers
def settings() -> dict:
    s = dict(DEFAULTS)
    s.update(db.get_kv("settings", {}) or {})
    if not s.get("model"):
        s["model"] = "auto"
    if s.get("knowledge") not in ("web+docs", "docs", "web"):
        s["knowledge"] = "web+docs"
    return s


def save_settings(patch: dict) -> dict:
    s = settings()
    for k in ("org", "model", "strict", "min_relevance", "show_sources", "knowledge",
              "image_model", "speech_model", "video_model"):
        if k in patch and patch[k] is not None:
            s[k] = patch[k]
    s["min_relevance"] = max(0.0, min(1.0, float(s["min_relevance"])))
    s["org"] = str(s["org"])[:80] or "your organisation"
    db.set_kv("settings", s)
    return s


def sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _demo_files() -> list[Path]:
    if not config.DEMO_DIR.exists():
        return []
    return sorted([p for p in config.DEMO_DIR.iterdir()
                   if p.is_file() and p.suffix.lower() in config.ALLOWED_EXTENSIONS])


# -------------------------------------------------------------------- routes
@app.get("/")
async def index():
    return FileResponse(config.STATIC_DIR / "index.html")


@app.get("/manifest.webmanifest")
async def manifest():
    return FileResponse(config.STATIC_DIR / "manifest.webmanifest",
                        media_type="application/manifest+json")


@app.get("/sw.js")
async def service_worker():
    # served from the root so its scope covers the whole app
    return FileResponse(config.STATIC_DIR / "sw.js", media_type="application/javascript",
                        headers={"Service-Worker-Allowed": "/", "Cache-Control": "no-cache"})


@app.get("/icons/{name}")
async def icon(name: str):
    path = config.STATIC_DIR / "icons" / name
    if not path.is_file() or path.suffix.lower() != ".png":
        raise HTTPException(404, "No such icon")
    return FileResponse(path, media_type="image/png",
                        headers={"Cache-Control": "public, max-age=604800"})


@app.get("/api/health")
async def health():
    s = settings()
    return {
        "status": "ok",
        "version": "1.1.0",
        "llm": {**gateway.status(), "model": s["model"]},
        "embeddings": embeddings.status(),
        "reranker": retrieval.reranker_status(),
        "stats": db.doc_stats(),
        "settings": s,
        "web": {"provider": web.provider_name(), "detail": web.describe(),
                "keyless": web.provider_name() == "keyless"},
        "demo_docs": len(_demo_files()),
        "tools": [t["name"] for t in tools.available_specs()],
        "media": {"free_speech": free_media.available()["speech"],
                  "free_video": free_media.available()["video"]},
        "workspace": str(Path(config.DATA_DIR) / "workspaces"),
        "auth": bool(ACCESS_PASSWORD),
    }


@app.get("/api/settings")
async def get_settings():
    return settings()


@app.put("/api/settings")
async def put_settings(patch: dict):
    s = save_settings(patch)
    db.set_kv("suggest_cache", None)
    return s


@app.post("/api/key")
async def set_key(payload: dict):
    """Store a provider key and prove it can actually run inference."""
    key = (payload.get("api_key") or "").strip()
    provider = (payload.get("provider") or "gateway").strip().lower()
    if not key:
        raise HTTPException(400, "api_key is required")
    if provider not in gateway.KNOWN and provider != "custom":
        raise HTTPException(400, f"Unknown provider '{provider}'")
    base_url = (payload.get("base_url") or "").strip()
    if provider == "custom":
        if not base_url.startswith("http"):
            raise HTTPException(400, "Custom providers need a base URL like https://host/v1")
        os.environ["CORTEX_BASE_URL"] = base_url
        os.environ["CORTEX_PROVIDER"] = "custom"
    elif base_url:
        os.environ["CORTEX_BASE_URL"] = base_url
    gateway.save_key(provider, key)
    if provider == "gateway" and key.startswith("vcp_"):
        os.environ["VERCEL_AI_GATEWAY_KEY"] = key
    db.set_kv("model_catalog", None)
    ready = gateway.configured()
    if not ready:
        kind = "That Vercel token can list models but not run them" if key.startswith("vcp_") else "That key cannot run inference"
        raise HTTPException(401, f"{kind}. Create an AI Gateway API key (vck_…) in Vercel → AI Gateway → API Keys, or use an OpenRouter / OpenAI key.")
    probe = await gateway.complete(
        router.helper_model(await gateway.models()) or "", 
        [{"role": "user", "content": "Reply with OK"}], max_tokens=5)
    if not probe:
        gateway.save_key(provider, "")
        raise HTTPException(401, "The provider rejected this key on a test request.")
    db.set_kv("suggest_cache", None)
    return {"ok": True, "provider": gateway.status(), "probe": probe.strip()[:20]}


@app.delete("/api/key")
async def clear_key():
    for name in gateway.KNOWN:
        gateway.save_key(name, "")
    os.environ.pop("CORTEX_RUNTIME_KEY", None)
    return {"ok": True}


@app.get("/api/models")
async def list_models(force: bool = False):
    cat = await gateway.models(force=force)
    parked = gateway.model_cooldowns()
    if parked:
        # a model this key just got a 429/503 from stays out of the picker
        cat = [m for m in cat if m["id"].split("/")[-1] not in parked]
    return {"models": cat, "options": router.option_list(cat), "provider": gateway.status(),
            "parked": sorted(parked),
            "routes": {task: router.route_preview(task, cat) for task in router.PREFERENCE},
            "count": len(cat),
            "by_type": {t: sum(1 for m in cat if m.get("type") == t)
                        for t in ("language", "image", "video", "speech", "transcription")}}


@app.get("/api/models/capabilities")
async def model_capabilities():
    cat = await gateway.models()
    return {task: router.route_preview(task, cat) for task in router.PREFERENCE}


@app.get("/api/documents")
async def documents():
    return {"documents": db.list_documents(), "stats": db.doc_stats()}


@app.post("/api/documents")
async def upload_documents(files: list[UploadFile] = File(...)):
    results, errors = [], []
    for f in files:
        raw = await f.read()
        name = Path(f.filename or "untitled").name
        ext = Path(name).suffix.lower()
        if ext not in config.ALLOWED_EXTENSIONS:
            errors.append({"name": name, "error": f"Unsupported type '{ext}'"})
            continue
        if len(raw) > config.MAX_UPLOAD_MB * 1024 * 1024:
            errors.append({"name": name, "error": f"Larger than {config.MAX_UPLOAD_MB} MB"})
            continue
        if not raw:
            errors.append({"name": name, "error": "Empty file"})
            continue
        try:
            doc = await run_in_threadpool(ingest.ingest_bytes, raw, name)
            results.append(doc)
        except Exception as exc:
            log.exception("Ingest failed for %s", name)
            errors.append({"name": name, "error": str(exc)[:300]})
    if results:
        db.set_kv("suggest_cache", None)
    return {"added": results, "errors": errors, "stats": db.doc_stats()}


@app.delete("/api/documents/{doc_id}")
async def remove_document(doc_id: str):
    ok = db.delete_document(doc_id)
    retrieval.invalidate()
    db.set_kv("suggest_cache", None)
    if not ok:
        raise HTTPException(404, "Document not found")
    return {"ok": True, "stats": db.doc_stats()}


@app.get("/api/documents/{doc_id}/chunks")
async def doc_chunks(doc_id: str, limit: int = Query(6, ge=1, le=50)):
    rows = [c for c in db.all_chunks() if c["doc_id"] == doc_id][:limit]
    return {"chunks": rows}


@app.post("/api/demo/load")
async def load_demo():
    files = _demo_files()
    if not files:
        raise HTTPException(400, "No demo documents bundled.")
    added, errors = [], []
    for p in files:
        try:
            added.append(await run_in_threadpool(ingest.ingest_file, p))
        except Exception as exc:
            errors.append({"name": p.name, "error": str(exc)[:200]})
    db.set_kv("suggest_cache", None)
    return {"added": added, "errors": errors, "stats": db.doc_stats()}


@app.post("/api/reindex")
async def reindex():
    out = await run_in_threadpool(ingest.reindex_embeddings)
    return {**out, "embeddings": embeddings.status()}


@app.get("/api/search")
async def search_only(q: str, k: int = Query(8, ge=1, le=20), docs: str | None = None):
    if not q.strip():
        raise HTTPException(400, "q is required")
    doc_ids = [d for d in (docs or "").split(",") if d] or None
    sources, diag = await run_in_threadpool(retrieval.search, q, k, True, doc_ids)
    return {"query": q, "sources": sources, "diagnostics": diag}


# ------------------------------------------------------------ conversations
@app.get("/api/conversations")
async def conversations():
    return {"conversations": db.list_conversations()}


@app.post("/api/conversations")
async def new_conversation():
    return db.create_conversation()


@app.get("/api/conversations/{conv_id}")
async def conversation(conv_id: str):
    conv = db.get_conversation(conv_id)
    if not conv:
        raise HTTPException(404, "Conversation not found")
    return {**conv, "messages": db.get_messages(conv_id)}


@app.delete("/api/conversations/{conv_id}")
async def remove_conversation(conv_id: str):
    db.delete_conversation(conv_id)
    return {"ok": True}


@app.get("/api/conversations/{conv_id}/export")
async def export_conversation(conv_id: str):
    conv = db.get_conversation(conv_id)
    if not conv:
        raise HTTPException(404, "Conversation not found")
    msgs = db.get_messages(conv_id)
    lines = [f"# {conv['title']}", f"_Exported from Cortex · {time.strftime('%Y-%m-%d %H:%M')}_", ""]
    for m in msgs:
        who = "You" if m["role"] == "user" else "Cortex"
        lines.append(f"### {who}")
        lines.append(m["content"].strip())
        if m.get("sources"):
            cites = {}
            for s in m["sources"]:
                key = s["doc_name"]
                cites.setdefault(key, []).append(f"p.{s['page']}" if s.get("page") else "")
            lines.append("")
            lines.append("> Sources: " + "; ".join(
                f"{k} ({', '.join(dict.fromkeys([x for x in v if x]))})" for k, v in cites.items()))
        lines.append("")
    body = "\n".join(lines)
    return Response(body, media_type="text/markdown",
                    headers={"Content-Disposition": f'attachment; filename="cortex-{conv_id}.md"'})


# --------------------------------------------------------------------- chat
async def _retrieve(question: str, history: list[dict],
                    doc_ids: list[str] | None = None) -> tuple[str, list[dict], dict]:
    query = question
    if history:
        try:
            query = await asyncio.wait_for(llm.rewrite_query(question, history), timeout=6)
        except Exception:
            query = question
    sources, diag = await run_in_threadpool(retrieval.search, query, config.TOP_K, True, doc_ids)

    # adaptive: when the first pass is unsure, widen with vocabulary variants
    s = settings()
    min_rel = float(s.get("min_relevance", config.MIN_RELEVANCE))
    uncertain = not sources or sources[0]["score"] < config.ADAPTIVE_CONFIDENCE
    # expanding queries only helps when there is a corpus to search; with no
    # documents it costs a model round-trip and returns nothing, which used to put
    # ten seconds in front of every question asked before any upload
    has_corpus = int(diag.get("candidates") or 0) > 0
    if config.ADAPTIVE and uncertain and has_corpus and gateway.configured():
        try:
            variants = await asyncio.wait_for(
                quality.expand_queries(query, config.MAX_EXPANSIONS), timeout=3.5)
        except Exception:
            variants = []
        if variants:
            merged: dict[str, dict] = {str(x["chunk_id"]): x for x in sources}
            for variant in variants:
                extra, _ = await run_in_threadpool(
                    retrieval.search, variant, config.TOP_K, True, doc_ids)
                for src in extra:
                    key = str(src["chunk_id"])
                    if key not in merged or src["score"] > merged[key]["score"]:
                        merged[key] = src
            sources = sorted(merged.values(), key=lambda x: -x["score"])[: config.TOP_K]
            diag["expanded_with"] = variants
            diag["mode"] = diag.get("mode", "") + "+expand"

    if sources and sources[0]["score"] < min_rel:
        sources = sources[:2]
        diag["low_confidence"] = True
    diag["query_used"] = query
    return query, sources, diag


# Questions that are answered by understanding, not by today's facts. These skip
# the pre-fetch so the reply starts in a second or two; if the model decides it
# actually needs the web it can still call a search tool mid-turn.
STABLE_KNOWLEDGE = re.compile(
    r"^\s*(?:what (?:is|are|does|do|was|were)|how (?:does|do|did|is|are|come)|"
    r"why (?:does|do|did|is|are)|explain|define|describe|compare|"
    r"difference between|tell me about)\b", re.I)

# Anything the model should write itself. Searching for "rain on a tin roof"
# returns other people's poems, which is worse than useless in the prompt.
CREATIVE = re.compile(
    r"\b(write|compose|draft|make) (?:me )?(?:a|an|some)? ?(poem|haiku|limerick|story|"
    r"essay|song|lyrics|joke|tagline|slogan|caption|headline|letter|email)\b|"
    r"\b(poem|haiku|limerick|lyrics|short story|screenplay)\b", re.I)

LIVE_CUES = re.compile(
    r"\b(latest|today|tonight|this (?:week|month|year)|current|currently|now|news|headlines?|"
    r"price|prices|pricing|release[sd]?|version|changelog|who is|who won|weather|forecast|"
    r"stock|market|score|schedule|20[2-9]\d)\b", re.I)


def _wants_web(question: str, doc_sources: list[dict], mode: str) -> bool:
    """Decide whether this message needs the live internet."""
    if mode == "docs":
        return False
    if mode == "web":
        return True
    task = router.classify(question)
    if task in ("image", "video", "speech"):
        return False                      # generation needs a model, not a search
    try:
        from .agent import plan_tool
        plan = plan_tool(question, _enabled_tool_names())
    except Exception:
        plan = None
    if plan and plan[0] != "web_search":
        return False                      # a tool answers it directly; searching only adds noise
    factual = bool(re.match(r"^\s*(who|what|when|where|which|whose|how (?:much|many|old|fast|big|"
                            r"tall|long)|is|are|does|do|did|can)\b", question, re.I))
    if len(question.split()) <= 6 and not factual and not LIVE_CUES.search(question):
        return False                      # small talk: "say hello in five words"
    if LIVE_CUES.search(question) and (factual or task not in ("code", "creative")):
        return not _docs_answer_it(doc_sources)
    if CREATIVE.search(question) and not LIVE_CUES.search(question):
        return False                      # a poem does not need the web
    if STABLE_KNOWLEDGE.search(question) and not LIVE_CUES.search(question):
        # "what does a barometer measure" does not need today's news
        return False
    if task in ("code", "creative"):
        return False                      # tools and the model itself are enough
    if not doc_sources:
        return True
    return not _docs_answer_it(doc_sources)


def _docs_answer_it(doc_sources: list[dict]) -> bool:
    """True when the uploads already answer the question well enough to skip the web."""
    if not doc_sources:
        return False
    return float(doc_sources[0].get("score") or 0) >= 0.62


def _trim_docs(question: str, doc_sources: list[dict]) -> list[dict]:
    """Drop document hits that the question is clearly not about.

    Three ways a document set is irrelevant here:
      - generation/engineering tasks (code, image, video, speech) with no strong hit;
      - questions that a live tool answers outright (weather, arithmetic, clock);
      - everything else scoring below a floor.
    """
    if not doc_sources:
        return doc_sources
    top = float(doc_sources[0].get("score") or 0)
    task = router.classify(question)
    if task in ("code", "image", "video", "speech") and top < 0.45:
        return []
    if re.search(r"\b(?:document|documents|pdf|handbook|policy|uploaded|file|files|"
                 r"contract|report|spreadsheet)\b", question, re.I):
        return doc_sources                      # they asked about their own material
    try:
        from .agent import plan_tool
        plan = plan_tool(question, _enabled_tool_names())
    except Exception:
        plan = None
    if plan and plan[0] in ("weather", "calculator", "current_datetime"):
        return []                               # a live tool answers this, not a document
    if plan and top < 0.62:
        return []
    return doc_sources


def _enabled_tool_names() -> set[str]:
    try:
        return {t["name"] for t in tools.catalog() if t.get("enabled")}
    except Exception:
        return set(tools.SPECS)


@app.post("/api/chat")
async def chat(payload: dict):
    regenerate = bool(payload.get("regenerate"))
    question = (payload.get("message") or "").strip()
    conv_id = payload.get("conversation_id")
    conv = db.get_conversation(conv_id) if conv_id else None

    if regenerate:
        if conv is None:
            raise HTTPException(400, "conversation_id is required to regenerate")
        msgs = db.get_messages(conv["id"])
        user_idxs = [i for i, m in enumerate(msgs) if m["role"] == "user"]
        if not user_idxs:
            raise HTTPException(400, "Nothing to regenerate in this conversation")
        cut = user_idxs[-1]
        question = msgs[cut]["content"].strip()
        with db.db() as conn:
            conn.execute("DELETE FROM messages WHERE conv_id = ? AND id > ?",
                         (conv["id"], msgs[cut]["id"]))
        history = msgs[:cut]
    else:
        if not question:
            raise HTTPException(400, "message is required")
        if len(question) > 8000:
            raise HTTPException(400, "Message too long (8000 chars max)")
        if conv is None:
            conv = db.create_conversation(title=question[:70] or "New chat")
        history = db.get_messages(conv["id"])
        if not history:
            db.update_conversation_title(conv["id"], question[:70])
        db.add_message(conv["id"], "user", question)
        db.touch_conversation(conv["id"])

    s = settings()
    model_choice = str(payload.get("model") or s.get("model") or "auto")
    strict = payload.get("strict") if isinstance(payload.get("strict"), bool) else bool(s.get("strict", False))
    knowledge = payload.get("knowledge") or s.get("knowledge", "web+docs")
    if knowledge not in ("web+docs", "docs", "web"):
        knowledge = "web+docs"
    media_items: list[dict] = []

    doc_ids = payload.get("doc_ids")
    if not isinstance(doc_ids, list) or not doc_ids:
        doc_ids = None

    async def event_stream():
        yield sse("start", {"conversation_id": conv["id"], "model": model_choice,
                            "demo": not gateway.configured()})
        parts: list[str] = []
        usage: dict = {}
        meta: dict = {}
        sources: list[dict] = []
        try:
            t0 = time.perf_counter()
            summary, history_used = await quality.compress_history(history)
            if summary:
                history_used = [{"role": "user",
                                 "content": f"[Earlier conversation summary] {summary}"}] + history_used

            if agent._is_small_talk(question):
                # "hello" needs no corpus: retrieval only attaches noise
                query, doc_sources, diag = question, [], {"ms": 0, "candidates": 0}
            else:
                query, doc_sources, diag = await _retrieve(question, history_used, doc_ids)
            diag["retrieval_ms"] = diag.get("ms", 0)
            doc_sources = _trim_docs(question, doc_sources)

            web_bundle = {"results": [], "pages": [], "provider": "", "used": False}
            if _wants_web(question, doc_sources, knowledge):
                try:
                    web_bundle = await asyncio.wait_for(
                        web.research(question, depth=3, n=6), timeout=25)
                    web_bundle["used"] = True
                except Exception:
                    web_bundle = {"results": [], "pages": [], "provider": "", "used": True}
            web_sources = web.to_sources(web_bundle["results"], web_bundle["pages"], limit=4)
            if web_sources:
                # a zero-scoring document is noise next to real web hits
                doc_sources = [d for d in doc_sources if float(d.get("score") or 0) >= 0.05]
            sources = doc_sources + web_sources
            diag["web"] = {"used": web_bundle["used"], "provider": web_bundle["provider"],
                           "results": len(web_bundle["results"]), "read": len(web_bundle["pages"])}
            yield sse("sources", {"sources": sources, "diagnostics": diag, "query_used": query})

            specs = [] if payload.get("tools") is False else tools.available_specs()

            if not sources and not gateway.configured():
                # No key, no documents. Tools can still answer plenty of questions
                # (weather, maths, the clock, Wikipedia), so only refuse when they can't.
                plan = agent.plan_tool(question, {x["name"] for x in specs}) if specs else None
                if plan is None:
                    if router.classify(question) in ("image", "speech", "video"):
                        msg = ("Generating images, audio or video needs a model provider. Add a "
                               "key in Settings \u2192 Model provider and ask again. Documents, "
                               "live tools and the code workspace all work without one.")
                    else:
                        msg = ("I don't have a model provider or any sources yet. Add an API key "
                               "in Settings, upload documents, or ask something a built-in tool "
                               "can answer on its own - the weather, a calculation, the date.")
                    yield sse("delta", {"text": msg})
                    db.add_message(conv["id"], "assistant", msg, [], {"empty_kb": True})
                    yield sse("done", {"conversation_id": conv["id"], "saved": True})
                    return

            if specs:
                yield sse("tools", {"available": [x["name"] for x in specs]})

            async with _generation_slots:
                async for ev in agent.stream(
                        question, sources, history_used, conv_id=conv["id"],
                        model=model_choice, strict=strict, org=s["org"],
                        specs=specs, knowledge=knowledge):
                    kind = ev["type"]
                    if kind == "delta":
                        parts.append(ev["text"])
                        yield sse("delta", {"text": ev["text"]})
                    elif kind == "route":
                        meta["model"] = ev["model"]
                        meta["task"] = ev["task"]
                        meta["why"] = ev["why"]
                        meta["requested"] = ev["requested"]
                        yield sse("route", {k: v for k, v in ev.items() if k != "type"})
                    elif kind in ("tool_start", "tool_result"):
                        yield sse(kind, {k: v for k, v in ev.items() if k != "type"})
                    elif kind == "media":
                        media_items.append(ev)
                        yield sse("media", {k: v for k, v in ev.items() if k != "type"})
                    elif kind == "text_replace":
                        parts[:] = [ev["text"]]        # the cleaned reply replaces the stream
                        yield sse("text_replace", {"text": ev["text"]})
                    elif kind == "notice":
                        yield sse("notice", {"message": ev["message"]})
                    elif kind == "usage":
                        usage = ev
                        yield sse("usage", ev)
                    elif kind == "error":
                        yield sse("error", {"message": ev["message"]})
                        break

            answer = "".join(parts).strip()
            # the answer is finished here; anything timed after this point (grounding
            # checks, follow-up generation) is bookkeeping the user never waited for,
            # so it must not be counted in the latency shown under the answer.
            answered_at = time.perf_counter()
            if answer:
                grounding = (quality.verify_grounding(answer, doc_sources)
                             if doc_sources else {"score": None, "claims": 0, "supported": 0, "details": []})
                yield sse("grounding", grounding)
                stray = quality.unsupported_numbers(answer, doc_sources + web_sources, question)
                if stray:
                    yield sse("warning", {
                        "kind": "unsupported",
                        "message": (f"This answer mentions {', '.join(stray)}, which does not appear in "
                                    "the sources it cites. Verify that figure before relying on it."),
                    })
                gscore = grounding.get("score")
                if gscore is not None and gscore < 0.5 and (grounding.get("claims") or 0) >= 2:
                    yield sse("warning", {
                        "kind": "grounding",
                        "message": ("Some sentences don't line up closely with the passages they cite. "
                                    "Check the sources before relying on this answer."),
                    })
                if gateway.configured():
                    # a slow follow-up model must not hold the finished answer open
                    try:
                        fu = await asyncio.wait_for(
                            quality.followups(question, answer, sources), timeout=4.0)
                    except Exception:
                        fu = None
                    if fu:
                        yield sse("followups", {"questions": fu})
                meta.update({
                    "model": meta.get("model") or usage.get("model") or model_choice,
                    "demo": not gateway.configured(),
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                    "cost_usd": usage.get("cost_usd", 0),
                    "tier": usage.get("tier") or ("free" if gateway.free_tier() else "paid"),
                    "latency_ms": round((answered_at - t0) * 1000),
                    "total_ms": round((time.perf_counter() - t0) * 1000),
                    "retrieval_ms": diag.get("retrieval_ms", 0),
                    "citations": len(sources),
                    "gateway": gateway.provider_name(),
                    "grounding": grounding,
                    "media": [m.get("url") for m in media_items],
                })
                mid = db.add_message(conv["id"], "assistant", answer, sources, meta)
                db.touch_conversation(conv["id"])
                existing = [m for m in db.get_messages(conv["id"]) if m["role"] == "user"]
                if len(existing) == 1 and gateway.configured():
                    # a nice title is cosmetic; never let a slow model hold the turn
                    # open for it (this used to add up to a minute to short chats)
                    try:
                        nice = await asyncio.wait_for(quality.make_title(question), timeout=4.0)
                    except Exception:
                        nice = None
                    if nice:
                        db.update_conversation_title(conv["id"], nice)
                        yield sse("title", {"title": nice})
                yield sse("done", {"conversation_id": conv["id"], "message_id": mid, "meta": meta})
            else:
                yield sse("done", {"conversation_id": conv["id"], "saved": False})
        except Exception as exc:  # pragma: no cover
            log.exception("chat failed")
            yield sse("error", {"message": f"Server error: {type(exc).__name__}"})
            yield sse("done", {"conversation_id": conv["id"], "saved": False})

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                                      "Connection": "keep-alive"})


@app.post("/api/messages/{message_id}/feedback")
async def feedback(message_id: int, payload: dict):
    rating = payload.get("rating")
    if rating not in (1, -1, 0):
        raise HTTPException(400, "rating must be 1, -1 or 0")
    with db.db() as conn:
        row = conn.execute("SELECT meta FROM messages WHERE id = ?", (message_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "Message not found")
        try:
            meta = json.loads(row["meta"] or "{}")
        except Exception:
            meta = {}
        meta["feedback"] = rating
        conn.execute("UPDATE messages SET meta = ? WHERE id = ?", (json.dumps(meta), message_id))
    return {"ok": True, "rating": rating}


@app.get("/api/quality")
async def quality_report():
    """Aggregate answer quality signals for the workspace."""
    with db.db() as conn:
        rows = conn.execute(
            "SELECT meta FROM messages WHERE role = 'assistant' AND meta != '{}'").fetchall()
    total = len(rows)
    grounded = low = rated = positive = 0
    cost = 0.0
    for r in rows:
        try:
            meta = json.loads(r["meta"] or "{}")
        except Exception:
            continue
        g = meta.get("grounding") or {}
        if g.get("score") is not None:
            if g["score"] >= 0.8:
                grounded += 1
            elif g["score"] < 0.5:
                low += 1
        cost += float(meta.get("cost_usd") or 0)
        if meta.get("feedback"):
            rated += 1
            positive += int(meta["feedback"] > 0)
    return {
        "answers": total,
        "well_grounded": grounded,
        "low_grounding": low,
        "rated": rated,
        "positive": positive,
        "cost_usd": round(cost, 4),
    }


# -------------------------------------------------------------- suggestions
@app.get("/api/tools")
async def list_tools():
    cfg = tools.public_config()
    return {
        "tools": tools.catalog(),
        "config": cfg,
        "available": [x["name"] for x in tools.available_specs()],
        "search_key": bool(tools.search_key()[0]),
    }


@app.put("/api/tools")
async def update_tools(patch: dict):
    tools.save_config(patch)
    cfg = tools.public_config()
    return {"config": cfg, "tools": tools.catalog(),
            "available": [x["name"] for x in tools.available_specs()]}


@app.post("/api/tools/{name}/test")
async def test_tool(name: str, payload: dict | None = None):
    if name not in tools.SPECS:
        raise HTTPException(404, f"Unknown tool '{name}'")
    args = (payload or {}).get("arguments") or {}
    cfg = tools.get_config()
    if name not in cfg.get("enabled", []):
        raise HTTPException(400, f"Tool '{name}' is disabled — enable it first.")
    t0 = time.perf_counter()
    result = await tools.execute(name, args)
    return {"tool": name, "ok": "error" not in result, "result": result,
            "ms": round((time.perf_counter() - t0) * 1000)}


# ---------------------------------------------------------------- workspace
@app.get("/api/workspace/{conv_id}/files")
async def workspace_files(conv_id: str):
    return {"files": workspace.listing(conv_id), "tree": workspace.tree_text(conv_id)}


@app.get("/api/workspace/{conv_id}/file")
async def workspace_file(conv_id: str, path: str = Query(...)):
    try:
        return {"path": path, "content": workspace.read(conv_id, path)}
    except workspace.WorkspaceError as exc:
        raise HTTPException(404, str(exc))


@app.post("/api/workspace/{conv_id}/run")
async def workspace_run(conv_id: str, payload: dict):
    command = (payload.get("command") or "").strip()
    if not command:
        raise HTTPException(400, "command is required")
    try:
        return await workspace.run(conv_id, command)
    except workspace.WorkspaceError as exc:
        raise HTTPException(400, str(exc))


@app.get("/api/workspace/{conv_id}/download")
async def workspace_download(conv_id: str):
    if not workspace.listing(conv_id):
        raise HTTPException(404, "workspace is empty")
    path = workspace.bundle(conv_id)
    return FileResponse(path, filename=f"cortex-{workspace.slug(conv_id)}.zip",
                        media_type="application/zip")


@app.get("/api/media/{conv_id}/list")
async def media_list(conv_id: str):
    folder = media.ROOT / workspace.slug(conv_id)
    if not folder.exists():
        return {"items": []}
    items = []
    for f in sorted(folder.glob("*"), key=lambda x: -x.stat().st_mtime):
        items.append({"name": f.name, "url": f"/media/{folder.name}/{f.name}",
                      "bytes": f.stat().st_size, "kind": media_kind(f.suffix)})
    return {"items": items[:60]}


def media_kind(suffix: str) -> str:
    return {".png": "image", ".jpg": "image", ".jpeg": "image", ".webp": "image",
            ".mp3": "audio", ".wav": "audio", ".m4a": "audio", ".ogg": "audio",
            ".mp4": "video", ".webm": "video"}.get(suffix.lower(), "file")


@app.get("/api/suggestions")
async def suggestions(force: bool = False):
    docs = db.list_documents()
    if not docs:
        return {"questions": []}
    cache = db.get_kv("suggest_cache")
    fingerprint = f"{len(docs)}-{sum(d['n_chunks'] for d in docs)}"
    if cache and not force and cache.get("fp") == fingerprint and \
            time.time() - cache.get("at", 0) < 3600:
        return {"questions": cache["questions"], "cached": True}
    headings = []
    seen = set()
    for c in db.all_chunks():
        h = (c.get("heading") or "").strip()
        if h and h not in seen:
            seen.add(h)
            headings.append(h)
        if len(headings) >= 60:
            break
    qs = await llm.suggest_questions(docs, headings)
    # surface the tool layer without burying the document questions
    live_names = {x["name"] for x in tools.available_specs()}
    tips = [(1, "What's the weather in Johannesburg?", "weather"),
            (3, "What is 18% of 4 750?", "calculator")]
    for pos, tip, needs in tips:
        if needs in live_names and tip not in qs:
            qs.insert(min(pos, len(qs)), tip)
    qs = qs[:5]
    db.set_kv("suggest_cache", {"fp": fingerprint, "at": time.time(), "questions": qs})
    return {"questions": qs}


app.mount("/static", StaticFiles(directory=str(config.STATIC_DIR)), name="static")
app.mount("/media", StaticFiles(directory=str(media.ROOT)), name="media")
