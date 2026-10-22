"""External API tools the assistant can call.

Design rules:
  * A tool is only offered to the model when it's enabled *and* actually usable
    (e.g. web search needs a key), so the model never promises what it can't do.
  * Credentials stay server-side. They are injected into outbound requests and
    are never placed in the prompt or returned to the model.
  * Every call is bounded by a timeout, a response-size cap and an allowlist.
"""
from __future__ import annotations

import asyncio
import html
import ipaddress
import json
import math
import os
import re
import socket
from datetime import datetime, timezone
from typing import Any, Callable
from urllib.parse import urlparse
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import urllib.parse

import httpx

from . import db, gateway, media, web, workspace

TIMEOUT = float(os.getenv("CORTEX_TOOL_TIMEOUT", "12"))
MAX_RESPONSE_CHARS = int(os.getenv("CORTEX_TOOL_MAX_CHARS", "6000"))
USER_AGENT = "Cortex/1.0 (+knowledge assistant)"

# ---------------------------------------------------------------- registry
# Each spec is the JSON schema handed to Claude. `available(cfg)` decides
# whether the tool can run at all; `run(cfg, **args)` executes it.
SPECS: dict[str, dict] = {}


def _spec(name: str, description: str, parameters: dict, *, label: str,
          needs: str = "", available: Callable | None = None,
          run: Callable | None = None, needs_ctx: bool = False,
          timeout: float | None = None):
    SPECS[name] = {
        "schema": {"name": name, "description": description, "input_schema": parameters},
        "label": label,
        "needs": needs,
        "available": available,
        "run": run,
        "needs_ctx": needs_ctx,
        "timeout": timeout or (TIMEOUT + 3),
    }


# ── config ──────────────────────────────────────────────────────────────────
_DEFAULT: list[str] = []

DEFAULT_CONFIG = {
    "enabled": ["calculator", "current_datetime", "web_search", "weather",
                "wikipedia", "fetch_url"],
    "http_allowlist": [],          # empty = any public https host
    "secrets": {},                 # host -> {"header": "...", "value": "..."}
    "search_provider": "auto",     # auto | tavily | brave
}


def get_config() -> dict:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    stored = db.get_kv("tools", {}) or {}
    for k, v in stored.items():
        if k in cfg and v is not None:
            cfg[k] = v
    cfg["disabled"] = list(stored.get("disabled") or [])
    # a tool added in a new release should not stay off just because this
    # install saved its config before the tool existed
    if isinstance(cfg.get("enabled"), list):
        disabled = set(stored.get("disabled") or [])
        for name in _DEFAULT:
            if name not in cfg["enabled"] and name not in disabled:
                cfg["enabled"].append(name)
    cfg["_keys"] = {
        "tavily": bool(os.getenv("TAVILY_API_KEY", "").strip()),
        "brave": bool(os.getenv("BRAVE_API_KEY", "").strip()),
    }
    return cfg


def public_config() -> dict:
    """Config safe to hand to a browser: secrets become headers + a set flag."""
    cfg = json.loads(json.dumps(get_config()))
    cfg.pop("_keys", None)
    cfg.pop("disabled", None)
    cfg["secrets"] = {h: {"header": c.get("header", ""), "set": True}
                      for h, c in (cfg.get("secrets") or {}).items()}
    return cfg


def save_config(patch: dict) -> dict:
    previous = get_config().get("secrets") or {}
    cfg = json.loads(json.dumps(get_config()))
    cfg.pop("_keys", None)
    cfg.pop("disabled", None)
    for key in ("enabled", "http_allowlist", "secrets", "search_provider"):
        if key in patch and patch[key] is not None:
            cfg[key] = patch[key]
    cfg["enabled"] = [t for t in cfg["enabled"] if t in SPECS]
    # remember what was switched off, so a new release never silently re-enables it
    cfg["disabled"] = [t for t in SPECS if t not in cfg["enabled"]]
    cfg["http_allowlist"] = [h.strip().lower() for h in cfg["http_allowlist"] if h.strip()][:50]
    secrets = {}
    for host, cred in (cfg.get("secrets") or {}).items():
        if not isinstance(cred, dict) or not cred.get("header"):
            continue
        key = str(host).strip().lower()
        value = str(cred.get("value") or "")
        if not value and key in previous:          # blank means "keep what is stored"
            value = previous[key].get("value", "")
        if value:
            secrets[key] = {"header": str(cred["header"])[:64], "value": value[:1000]}
    cfg["secrets"] = secrets
    db.set_kv("tools", cfg)
    return public_config()


def search_key() -> tuple[str, str]:
    cfg = get_config()
    provider = cfg.get("search_provider", "auto")
    tavily = os.getenv("TAVILY_API_KEY", "").strip()
    brave = os.getenv("BRAVE_API_KEY", "").strip()
    if provider == "tavily" and tavily:
        return "tavily", tavily
    if provider == "brave" and brave:
        return "brave", brave
    if tavily:
        return "tavily", tavily
    if brave:
        return "brave", brave
    return "", ""


def _has_search(_cfg=None) -> bool:
    return bool(search_key()[0])


# ── helpers ────────────────────────────────────────────────────────────────
def _strip_html(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style|nav|footer|header|aside)[^>]*>.*?</\1>", " ", raw)
    text = re.sub(r"(?s)<[^>]+>", " ", raw)
    text = html.unescape(text)
    return re.sub(r"[ \t\r\f\v]+", " ", re.sub(r"\n{3,}", "\n\n", text)).strip()


def _clip(text: str, limit: int = MAX_RESPONSE_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n… [truncated, {len(text) - limit} more characters]"


def _blocked_host(host: str) -> str | None:
    """Reject SSRF targets: localhost, link-local and private ranges."""
    host = (host or "").lower().strip("[]")
    if not host or host in {"localhost", "localhost.localdomain"}:
        return "local host is not allowed"
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return f"could not resolve '{host}'"
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            continue
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return "that address is private or reserved"
    return None


def _host_allowed(url: str, cfg: dict) -> tuple[bool, str]:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False, "only http(s) URLs are allowed"
    host = (parsed.hostname or "").lower()
    block = _blocked_host(host)
    if block:
        return False, block
    allow = cfg.get("http_allowlist") or []
    if allow and not any(host == a or host.endswith("." + a) for a in allow):
        return False, (f"'{host}' is not in the allowlist "
                       f"({', '.join(allow[:4])}{'…' if len(allow) > 4 else ''})")
    return True, ""


def _host_secret(url: str, cfg: dict) -> dict:
    host = (urlparse(url).hostname or "").lower()
    for pattern, cred in (cfg.get("secrets") or {}).items():
        if host == pattern or host.endswith("." + pattern):
            return {cred["header"]: cred["value"]}
    return {}


async def _get(url: str, cfg: dict, *, params: dict | None = None,
               headers: dict | None = None, timeout: float = TIMEOUT) -> httpx.Response:
    hdrs = {"User-Agent": USER_AGENT, "Accept": "application/json, text/html;q=0.9, */*;q=0.8"}
    hdrs.update(_host_secret(url, cfg))
    if headers:
        hdrs.update(headers)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, max_redirects=4) as client:
        return await client.get(url, params=params, headers=hdrs)


# ── tool implementations ───────────────────────────────────────────────────
_ALLOWED_MATH = {
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
    "sqrt": math.sqrt, "pow": pow, "floor": math.floor, "ceil": math.ceil,
    "log": math.log, "exp": math.exp, "pi": math.pi, "e": math.e,
}


def _safe_eval(node, depth=0):
    import ast

    if depth > 25:
        raise ValueError("expression too deep")
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body, depth + 1)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp):
        ops = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
               ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b,
               ast.FloorDiv: lambda a, b: a // b, ast.Mod: lambda a, b: a % b,
               ast.Pow: lambda a, b: a ** b}
        fn = ops.get(type(node.op))
        if not fn:
            raise ValueError("unsupported operator")
        return fn(_safe_eval(node.left, depth + 1), _safe_eval(node.right, depth + 1))
    if isinstance(node, ast.UnaryOp):
        val = _safe_eval(node.operand, depth + 1)
        return -val if isinstance(node.op, ast.USub) else +val
    if isinstance(node, ast.Name) and node.id in _ALLOWED_MATH:
        return _ALLOWED_MATH[node.id]
    if isinstance(node, ast.Call):
        fn = _safe_eval(node.func, depth + 1)
        if not callable(fn) or fn not in _ALLOWED_MATH.values():
            raise ValueError("function not allowed")
        return fn(*[_safe_eval(a, depth + 1) for a in node.args])
    raise ValueError("unsupported expression")


async def run_calculator(cfg, expression: str = "", **_):
    import ast
    expr = (expression or "").strip().replace("×", "*").replace("÷", "/").replace("^", "**")
    if not expr or len(expr) > 300:
        return {"error": "provide a short arithmetic expression"}
    if not re.fullmatch(r"[0-9+\-*/().,%\s_a-zA-Z]*", expr):
        return {"error": "expression contains unsupported characters"}
    try:
        value = _safe_eval(ast.parse(expr, mode="eval"))
    except Exception as exc:
        return {"error": f"could not evaluate: {exc}"}
    return {"expression": expr, "result": value}


async def run_datetime(cfg, timezone_name: str = "", **_):
    tz = (timezone_name or "UTC").strip()
    try:
        zone = ZoneInfo(tz)
    except (ZoneInfoNotFoundError, ValueError):
        return {"error": f"unknown timezone '{tz}'", "hint": "use IANA names like Africa/Johannesburg"}
    now = datetime.now(timezone.utc).astimezone(zone)
    return {"timezone": tz, "iso": now.isoformat(timespec="seconds"),
            "readable": now.strftime("%A, %d %B %Y at %H:%M %Z")}


async def run_web_search(cfg, query: str = "", max_results: int = 5, **_):
    provider, key = search_key()
    if not provider:
        hits = await web.search(query, max_results)
        if not hits:
            return {"error": "search returned nothing (no key configured, free sources empty)"}
        return {"provider": web.provider_name(), "query": query,
                "results": [{"title": h.get("title", ""), "url": h.get("url", ""),
                             "snippet": h.get("snippet", "")} for h in hits]}
    n = max(1, min(int(max_results or 5), 8))
    try:
        if provider == "tavily":
            resp = await _get("https://api.tavily.com/search", cfg, params={
                "api_key": key, "query": query, "max_results": n,
                "search_depth": "basic", "include_answer": "true"})
            data = resp.json()
            results = [{"title": r.get("title", ""), "url": r.get("url", ""),
                        "snippet": (r.get("content") or "")[:420]}
                       for r in (data.get("results") or [])][:n]
            out = {"provider": "tavily", "query": query, "results": results}
            if data.get("answer"):
                out["summary"] = data["answer"]
            return out
        resp = await _get("https://api.search.brave.com/res/v1/web/search", cfg,
                          params={"q": query, "count": n},
                          headers={"X-Subscription-Token": key, "Accept": "application/json"})
        data = resp.json()
        results = [{"title": r.get("title", ""), "url": r.get("url", ""),
                    "snippet": _strip_html(r.get("description") or "")[:420]}
                   for r in ((data.get("web") or {}).get("results") or [])][:n]
        return {"provider": "brave", "query": query, "results": results}
    except Exception as exc:
        return {"error": f"search failed: {type(exc).__name__}: {exc}"[:300]}


async def run_wikipedia(cfg, query: str = "", **_):
    q = (query or "").strip()
    if not q:
        return {"error": "provide a search term"}
    try:
        resp = await _get("https://en.wikipedia.org/w/api.php", cfg, params={
            "action": "query", "list": "search", "srsearch": q, "format": "json",
            "srlimit": 3})
        hits = [(r.get("title") or "") for r in
                ((resp.json().get("query") or {}).get("search") or [])]
        if not hits:
            return {"query": q, "results": [], "note": "no Wikipedia articles matched"}
        slug = urllib.parse.quote(hits[0].replace(" ", "_"), safe="")
        summary = await _get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}", cfg)
        data = summary.json()
        return {"query": q, "title": data.get("title", hits[0]),
                "extract": _clip(_strip_html(data.get("extract") or ""), 1800),
                "url": (data.get("content_urls", {}).get("desktop", {}) or {}).get("page", ""),
                "other_matches": hits[1:]}
    except Exception as exc:
        return {"error": f"wikipedia lookup failed: {type(exc).__name__}"}


WEATHER_CODES = {0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
                 45: "fog", 48: "depositing rime fog", 51: "light drizzle", 53: "drizzle",
                 55: "dense drizzle", 61: "slight rain", 63: "rain", 65: "heavy rain",
                 71: "slight snow", 73: "snow", 75: "heavy snow", 80: "rain showers",
                 81: "moderate rain showers", 82: "violent rain showers",
                 95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with heavy hail"}


async def run_weather(cfg, place: str = "", latitude: float | None = None,
                      longitude: float | None = None, **_):
    try:
        label = place
        if latitude is None or longitude is None:
            if not place:
                return {"error": "provide a place name or latitude/longitude"}
            geo = await _get("https://geocoding-api.open-meteo.com/v1/search", cfg,
                             params={"name": place, "count": 1, "language": "en", "format": "json"})
            hits = geo.json().get("results") or []
            if not hits:
                return {"error": f"could not find a place called '{place}'"}
            top = hits[0]
            latitude, longitude = top["latitude"], top["longitude"]
            label = ", ".join(x for x in [top.get("name"), top.get("admin1"), top.get("country")] if x)
        fc = await _get("https://api.open-meteo.com/v1/forecast", cfg, params={
            "latitude": latitude, "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "forecast_days": 3, "timezone": "auto"})
        data = fc.json()
        cur = data.get("current") or {}
        daily = data.get("daily") or {}
        days = []
        for i, day in enumerate(daily.get("time", [])[:3]):
            days.append({
                "date": day,
                "max_c": daily.get("temperature_2m_max", [None])[i],
                "min_c": daily.get("temperature_2m_min", [None])[i],
                "rain_chance_pct": daily.get("precipitation_probability_max", [None])[i],
            })
        return {
            "place": label or f"{latitude},{longitude}",
            "current": {
                "temp_c": cur.get("temperature_2m"),
                "feels_like_c": cur.get("apparent_temperature"),
                "humidity_pct": cur.get("relative_humidity_2m"),
                "wind_kmh": cur.get("wind_speed_10m"),
                "conditions": WEATHER_CODES.get(cur.get("weather_code"), "unknown"),
            },
            "next_days": days,
            "source": "open-meteo.com",
        }
    except Exception as exc:
        return {"error": f"weather lookup failed: {type(exc).__name__}"}


async def run_fetch_url(cfg, url: str = "", max_chars: int = 4000, **_):
    ok, why = _host_allowed(url or "", cfg)
    if not ok:
        return {"error": why, "url": url}
    try:
        resp = await _get(url, cfg)
        ctype = resp.headers.get("content-type", "")
        if "json" in ctype:
            try:
                body = json.dumps(resp.json(), indent=2, ensure_ascii=False)
            except Exception:
                body = resp.text
        else:
            body = _strip_html(resp.text)
        return {"url": str(resp.url), "status": resp.status_code,
                "content_type": ctype.split(";")[0], "content": _clip(body, int(max_chars or 4000))}
    except Exception as exc:
        host = urlparse(url).hostname or url
        return {"error": (f"could not reach {host} ({type(exc).__name__}: {str(exc)[:120]}). "
                          "The site may be down or the address misspelled - use the search "
                          "snippets instead of trying again.")[:300], "url": url}


# ── specs ──────────────────────────────────────────────────────────────────
_spec("calculator", "Evaluate arithmetic precisely. Use for any maths, unit maths, "
      "percentages or comparisons of numbers instead of doing mental arithmetic.",
      {"type": "object", "properties": {
          "expression": {"type": "string",
                         "description": "Arithmetic expression, e.g. '(250000-200000)/200000*100'"}},
       "required": ["expression"]},
      label="Calculator", run=run_calculator)

_spec("current_datetime", "Get the current date and time in a timezone. Use when the question "
      "refers to 'today', 'now', deadlines, or elapsed time.",
      {"type": "object", "properties": {
          "timezone_name": {"type": "string",
                            "description": "IANA timezone, e.g. Africa/Johannesburg. Defaults to UTC."}}},
      label="Date & time", run=run_datetime)

_spec("web_search", "Search the public web for current information: prices, news, releases, "
      "people, or anything not in the uploaded documents. Returns titles, URLs and snippets.",
      {"type": "object", "properties": {
          "query": {"type": "string", "description": "Search query"},
          "max_results": {"type": "integer", "minimum": 1, "maximum": 8}},
       "required": ["query"]},
      label="Web search", needs="TAVILY_API_KEY or BRAVE_API_KEY",
      available=_has_search, run=run_web_search)

_spec("weather", "Current weather and a 3-day forecast for a place. Use for weather questions.",
      {"type": "object", "properties": {
          "place": {"type": "string", "description": "City or place name"},
          "latitude": {"type": "number"}, "longitude": {"type": "number"}}},
      label="Weather", run=run_weather)

_spec("wikipedia", "Look up a topic on Wikipedia for background facts and definitions.",
      {"type": "object", "properties": {
          "query": {"type": "string", "description": "Topic or article title"}},
       "required": ["query"]},
      label="Wikipedia", run=run_wikipedia)

_spec("fetch_url", "Fetch a URL and return its readable text or JSON. Use this to call an external "
      "API endpoint directly when the user names one, or to read a specific page.",
      {"type": "object", "properties": {
          "url": {"type": "string", "description": "Public http(s) URL"},
          "max_chars": {"type": "integer", "minimum": 200, "maximum": 8000}},
       "required": ["url"]},
      label="Fetch URL", run=run_fetch_url)


# ── public interface ───────────────────────────────────────────────────────
def available_specs() -> list[dict]:
    """Schemas for tools that are enabled *and* currently usable."""
    cfg = get_config()
    out = []
    for name in cfg.get("enabled", []):
        spec = SPECS.get(name)
        if not spec:
            continue
        if spec.get("available") and not spec["available"](cfg):
            continue
        out.append(spec["schema"])
    return out


def catalog() -> list[dict]:
    """Everything the UI needs to render the tools panel."""
    cfg = get_config()
    rows = []
    for name, spec in SPECS.items():
        usable = True if not spec.get("available") else spec["available"](cfg)
        rows.append({
            "name": name,
            "label": spec["label"],
            "description": spec["schema"]["description"],
            "enabled": name in cfg.get("enabled", []),
            "usable": usable,
            "needs": spec.get("needs", ""),
        })
    return sorted(rows, key=lambda r: r["label"])


async def execute(name: str, args: dict, ctx: dict | None = None) -> dict:
    """Run a tool safely. Never raises — errors become data for the model."""
    cfg = get_config()
    spec = SPECS.get(name)
    if not spec or name not in cfg.get("enabled", []):
        return {"error": f"tool '{name}' is not enabled"}
    if spec.get("available") and not spec["available"](cfg):
        return {"error": f"tool '{name}' is not configured ({spec.get('needs')})"}
    if not isinstance(args, dict):
        args = {}
    kwargs = dict(args)
    if spec.get("needs_ctx"):
        kwargs["ctx"] = ctx or {}
    try:
        result = await asyncio.wait_for(spec["run"](cfg, **kwargs), timeout=spec.get("timeout", TIMEOUT + 3))
    except asyncio.TimeoutError:
        return {"error": f"tool '{name}' timed out after {TIMEOUT:.0f}s"}
    except TypeError as exc:
        return {"error": f"bad arguments for '{name}': {exc}"}
    except Exception as exc:  # pragma: no cover - defensive
        return {"error": f"tool '{name}' failed: {type(exc).__name__}: {exc}"[:300]}
    return result if isinstance(result, dict) else {"result": result}


def summarise_for_model(name: str, result: dict) -> str:
    """Compact JSON handed back to the model as the tool result.

    Kept deliberately small: every tool round re-sends the whole conversation, so
    a 5k-char search blob costs its tokens again on each subsequent round.
    """
    try:
        text = json.dumps(result, ensure_ascii=False, default=str)
    except Exception:
        text = str(result)
    limit = 3500 if name in ("web_search", "deep_research") else 5000
    return _clip(text, limit)


def label_for(name: str) -> str:
    return SPECS.get(name, {}).get("label", name)


# ── workspace: write, read, list, run ───────────────────────────────────────
def _conv(ctx: dict | None) -> str:
    conv = (ctx or {}).get("conv_id") or ""
    if not conv:
        raise ValueError("no conversation context for this tool")
    return conv


async def run_write_file(cfg, path: str = "", content: str = "", ctx=None, **_):
    try:
        info = workspace.write(_conv(ctx), path, content)
    except workspace.WorkspaceError as exc:
        return {"error": str(exc)}
    return {"written": True, **info,
            "note": "File saved in the conversation workspace. Run it, or download the project as a zip."}


async def run_read_file(cfg, path: str = "", ctx=None, **_):
    try:
        text = workspace.read(_conv(ctx), path)
    except workspace.WorkspaceError as exc:
        return {"error": str(exc)}
    return {"path": path, "content": text, "chars": len(text)}


async def run_list_files(cfg, ctx=None, **_):
    files = workspace.listing(_conv(ctx))
    return {"files": files, "count": len(files),
            "tree": workspace.tree_text(_conv(ctx))}


async def run_code(cfg, command: str = "", ctx=None, **_):
    try:
        return await workspace.run(_conv(ctx), command)
    except workspace.WorkspaceError as exc:
        return {"error": str(exc)}


# ── generation: image, speech, video ────────────────────────────────────────
async def run_generate_image(cfg, prompt: str = "", size: str = "1024x1024", ctx=None, **_):
    try:
        art = await media.image(_conv(ctx), prompt, size=size)
    except media.MediaError as exc:
        return {"error": str(exc)}
    except gateway.ProviderError as exc:
        return {"error": str(exc)}
    return {**art, "note": "Image attached to the reply."}


async def run_text_to_speech(cfg, text: str = "", voice: str = "alloy", ctx=None, **_):
    try:
        art = await media.speech(_conv(ctx), text, voice=voice)
    except (media.MediaError, gateway.ProviderError) as exc:
        return {"error": str(exc)}
    return {**art, "note": "Audio attached to the reply."}


async def run_generate_video(cfg, prompt: str = "", seconds: int = 5, narration: str = "",
                             ctx=None, **_):
    try:
        art = await media.video(_conv(ctx), prompt, seconds=max(2, min(int(seconds or 5), 12)),
                                narration=narration or "")
    except (media.MediaError, gateway.ProviderError) as exc:
        return {"error": str(exc)}
    return {**art, "note": "Video attached to the reply."}


# ── research: read the web, not just search it ──────────────────────────────
async def run_deep_research(cfg, question: str = "", depth: int = 3, **_):
    # a tool the model called explicitly can afford to read more pages than the
    # automatic pre-fetch, but it still runs under one budget
    data = await web.research(question, depth=max(1, min(int(depth or 3), 4)), n=6,
                              budget=12.0)
    sources = web.to_sources(data["results"], data["pages"])
    return {
        "query": data["query"],
        "provider": data["provider"],
        "sources": [{"title": s["doc_name"], "url": s["url"], "text": s["text"][:1200]}
                    for s in sources],
        "note": "Live web results. Cite them by title and URL.",
    }


def _gateway_ok(cfg) -> bool:
    return gateway.configured()


_spec("write_file", "Create or overwrite a file in the conversation workspace. Use for code, "
      "configs, docs, data - anything that should exist on disk.",
      {"type": "object", "properties": {
          "path": {"type": "string", "description": "Relative path, e.g. app.py or src/main.js"},
          "content": {"type": "string", "description": "Full file content"}},
       "required": ["path", "content"]},
      label="Write file", needs_ctx=True, run=run_write_file)

_spec("read_file", "Read a file from the conversation workspace.",
      {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
      label="Read file", needs_ctx=True, run=run_read_file)

_spec("list_files", "List every file in the conversation workspace with sizes.",
      {"type": "object", "properties": {}}, label="List files", needs_ctx=True, run=run_list_files)

_spec("run_code", "Run a program in the workspace and get stdout/stderr/exit code. "
      "Allowed: python3, node, npm, npx, deno, bun, tsc.",
      {"type": "object", "properties": {
          "command": {"type": "string", "description": "e.g. python3 main.py or npm test"}},
       "required": ["command"]},
      label="Run code", needs_ctx=True, timeout=90, run=run_code)

_spec("generate_image", "Create an image from a text prompt and show it in the chat. Free: renders "
      "locally/keylessly with no provider needed.",
      {"type": "object", "properties": {
          "prompt": {"type": "string", "description": "Detailed visual description"},
          "size": {"type": "string", "description": "e.g. 1024x1024, 1536x1024"}},
       "required": ["prompt"]},
      label="Generate image", needs_ctx=True, needs="a model provider",
      available=_gateway_ok, timeout=240, run=run_generate_image)

_spec("text_to_speech", "Turn text into spoken audio and attach it to the reply. Free: uses "
      "on-device Edge voices when the provider has no TTS model.",
      {"type": "object", "properties": {
          "text": {"type": "string"}, "voice": {"type": "string"}},
       "required": ["text"]},
      label="Speak", needs_ctx=True, timeout=180, run=run_text_to_speech)

_spec("generate_video", "Create a short video from a text prompt (may take a minute). Free: "
      "renders stills and narration locally with ffmpeg when the provider has no video model.",
      {"type": "object", "properties": {
          "prompt": {"type": "string"},
          "seconds": {"type": "integer", "minimum": 2, "maximum": 12},
          "narration": {"type": "string", "description": "Optional voice-over script"}},
       "required": ["prompt"]},
      label="Generate video", needs_ctx=True, timeout=600, run=run_generate_video)

_spec("deep_research", "Search the web and read the top pages. Use for current facts, comparisons, "
      "news, prices, or anything not in the knowledge base.",
      {"type": "object", "properties": {
          "question": {"type": "string"}, "depth": {"type": "integer", "minimum": 1, "maximum": 4}}},
      label="Deep research", run=run_deep_research)

_DEFAULT = ["calculator", "current_datetime", "web_search", "weather", "wikipedia", "fetch_url",
            "deep_research", "write_file", "read_file", "list_files", "run_code",
            "generate_image", "text_to_speech", "generate_video"]
DEFAULT_CONFIG["enabled"] = _DEFAULT  # every tool on, unless the user turns it off
