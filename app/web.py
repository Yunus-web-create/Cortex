"""Internet knowledge: search, fetch, extract, cite.

No key required. Tavily and Brave are used when their keys are present; the
free path chains DuckDuckGo and Wikipedia so a fresh install can still answer
questions about the world.
"""
from __future__ import annotations

import asyncio
import hashlib
import html
import json
import os
import re
import time
from urllib.parse import parse_qs, quote, unquote, urlparse

import httpx

from . import db

UA = "Mozilla/5.0 (compatible; CortexBot/1.1; +https://example.com/bot)"
TIMEOUT = float(os.getenv("CORTEX_WEB_TIMEOUT", "15"))
MAX_PAGE_CHARS = int(os.getenv("CORTEX_WEB_MAX_CHARS", "7000"))
CACHE_TTL = int(os.getenv("CORTEX_WEB_CACHE_TTL", "3600"))

BLOCK = re.compile(
    r"<(script|style|noscript|svg|iframe|form|nav|footer|header|aside|button)\b.*?</\1>", re.S | re.I)


def _keys() -> tuple[str, str]:
    return (os.getenv("TAVILY_API_KEY", "").strip(), os.getenv("BRAVE_API_KEY", "").strip())


def provider_name() -> str:
    if os.getenv("CORTEX_SEARX_URL", "").strip():
        return "searx"
    tavily, brave = _keys()
    if tavily:
        return "tavily"
    if brave:
        return "brave"
    return "keyless"


def describe() -> str:
    """Human label for the search backend in use."""
    name = provider_name()
    return {
        "tavily": "Tavily (full web)",
        "brave": "Brave (full web)",
        "searx": "SearXNG (full web)",
        "keyless": "keyless: Wikipedia, Hacker News, Stack Overflow, GitHub",
    }.get(name, name)


def _cache_key(kind: str, q: str) -> str:
    return "web_" + hashlib.sha1(f"{kind}:{q}".encode()).hexdigest()[:16]


def _cached(kind: str, q: str):
    hit = db.get_kv(_cache_key(kind, q), None)
    if hit and time.time() - hit.get("at", 0) < CACHE_TTL:
        return hit.get("value")
    return None


def _store(kind: str, q: str, value):
    db.set_kv(_cache_key(kind, q), {"at": time.time(), "value": value})


async def _get(url: str, *, params=None, headers=None, timeout=TIMEOUT) -> httpx.Response:
    h = {"User-Agent": UA, "Accept-Language": "en;q=0.9"}
    if headers:
        h.update(headers)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, max_redirects=4) as c:
        return await c.get(url, params=params, headers=h)


# ------------------------------------------------------------------ search
async def _post(url: str, payload: dict, *, headers: dict | None = None,
                timeout: float = TIMEOUT) -> httpx.Response:
    h = {"User-Agent": UA, "Content-Type": "application/json", "Accept": "application/json"}
    if headers:
        h.update(headers)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, max_redirects=4) as c:
        return await c.post(url, json=payload, headers=h)


async def _tavily(q: str, n: int) -> list[dict]:
    """Tavily's search API is POST-only; a GET returns 401 even with a valid key."""
    key, _ = _keys()
    if not key:
        return []
    r = await _post("https://api.tavily.com/search", {
        "api_key": key, "query": q, "max_results": n,
        "search_depth": "advanced", "include_answer": True,
        "include_raw_content": False,
    }, timeout=max(TIMEOUT, 25))
    if r.status_code >= 400:
        raise RuntimeError(f"tavily {r.status_code}")
    data = r.json()
    out = [{"title": x.get("title", ""), "url": x.get("url", ""),
            "snippet": (x.get("content") or "")[:600], "source": "web"}
           for x in (data.get("results") or [])][:n]
    if data.get("answer"):
        out.insert(0, {"title": "Search summary", "url": "", "snippet": data["answer"][:800],
                       "is_summary": True, "source": "web"})
    return out


async def _brave(q: str, n: int) -> list[dict]:
    _, key = _keys()
    r = await _get("https://api.search.brave.com/res/v1/web/search",
                   params={"q": q, "count": n},
                   headers={"X-Subscription-Token": key, "Accept": "application/json"})
    data = r.json()
    return [{"title": html.unescape(x.get("title", "")), "url": x.get("url", ""),
             "snippet": html.unescape(re.sub(r"<[^>]+>", "", x.get("description") or ""))[:600]}
            for x in ((data.get("web") or {}).get("results") or [])][:n]


async def _duckduckgo(q: str, n: int) -> list[dict]:
    r = await _get("https://html.duckduckgo.com/html/", params={"q": q})
    body = r.text
    out = []
    for block in re.findall(r'<a[^>]+class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', body, re.S)[:n * 2]:
        href, title = block
        if "uddg=" in href:
            href = unquote(parse_qs(urlparse(href).query).get("uddg", [""])[0])
        if not href.startswith("http"):
            continue
        out.append({"title": re.sub(r"<[^>]+>", "", html.unescape(title)).strip(),
                    "url": href, "snippet": ""})
        if len(out) >= n:
            break
    return out


async def _searx(q: str, n: int) -> list[dict]:
    """Any SearXNG instance (self-hosted or public with JSON enabled)."""
    base = os.getenv("CORTEX_SEARX_URL", "").strip().rstrip("/")
    if not base:
        return []
    r = await _get(f"{base}/search", params={"q": q, "format": "json", "language": "en"})
    data = r.json()
    return [{"title": x.get("title", ""), "url": x.get("url", ""),
             "snippet": (x.get("content") or "")[:600]} for x in (data.get("results") or [])][:n]


async def _ddg(q: str, n: int) -> list[dict]:
    """DuckDuckGo's instant-answer API: no key, no scraping, no spam."""
    r = await _get("https://api.duckduckgo.com/",
                   params={"q": q, "format": "json", "no_html": 1, "skip_disambig": 1})
    data = r.json()
    out = []
    if data.get("AbstractText") and data.get("AbstractURL"):
        out.append({"title": data.get("Heading") or q, "url": data["AbstractURL"],
                    "snippet": data["AbstractText"][:600]})
    for topic in (data.get("RelatedTopics") or []):
        if len(out) >= n:
            break
        if topic.get("FirstURL") and topic.get("Text"):
            out.append({"title": topic["Text"].split(" - ")[0][:90],
                        "url": topic["FirstURL"], "snippet": topic["Text"][:400]})
        for sub in topic.get("Topics") or []:
            if len(out) >= n:
                break
            if sub.get("FirstURL") and sub.get("Text"):
                out.append({"title": sub["Text"].split(" - ")[0][:90],
                            "url": sub["FirstURL"], "snippet": sub["Text"][:400]})
    return out[:n]


async def _wikipedia(q: str, n: int) -> list[dict]:
    r = await _get("https://en.wikipedia.org/w/api.php",
                   params={"action": "query", "list": "search", "srsearch": q, "format": "json", "srlimit": n})
    hits = ((r.json().get("query") or {}).get("search") or [])
    return [{"title": x.get("title", ""),
             "url": f"https://en.wikipedia.org/wiki/{quote((x.get('title') or '').replace(' ', '_'))}",
             "snippet": re.sub(r"<[^>]+>", "", x.get("snippet") or "")[:600]} for x in hits][:n]


CODE_HINT = re.compile(r"\b(python|javascript|typescript|node|react|api|library|package|"
                       r"sdk|cli|docker|kubernetes|sql|rust|golang|install|error|bug)\b", re.I)


def _recent(q: str) -> bool:
    return bool(re.search(r"\b(latest|newest|today|this (?:week|month|year)|202[4-9]|"
                          r"news|current|release[sd]?|update[sd]?)\b", q, re.I))


async def _hn(q: str, n: int) -> list[dict]:
    """Hacker News (Algolia). No key. Excellent for releases, tools and tech news."""
    params = {"query": q, "tags": "story", "hitsPerPage": n}
    if _recent(q):
        params["numericFilters"] = f"created_at_i>{int(time.time()) - 18 * 30 * 86400}"
    r = await _get("https://hn.algolia.com/api/v1/search", params=params)
    out = []
    for h in (r.json().get("hits") or []):
        url = h.get("url") or f"https://news.ycombinator.com/item?id={h.get('objectID')}"
        title = h.get("title") or h.get("story_title") or ""
        if title:
            out.append({"title": title, "url": url,
                        "snippet": (h.get("story_text") or "")[:300] or f"Hacker News · {h.get('points', 0)} points",
                        "source": "hn"})
    return out[:n]


async def _stackoverflow(q: str, n: int) -> list[dict]:
    """Stack Exchange API. No key for light use. The right source for how-do-I questions."""
    r = await _get("https://api.stackexchange.com/2.3/search/advanced",
                   params={"order": "desc", "sort": "relevance", "q": q,
                           "site": "stackoverflow", "pagesize": n, "filter": "default"})
    out = []
    for h in (r.json().get("items") or []):
        out.append({"title": html.unescape(h.get("title", "")), "url": h.get("link", ""),
                    "snippet": re.sub(r"<[^>]+>", "", h.get("title", ""))[:300],
                    "source": "stackoverflow"})
    return out[:n]


async def _github(q: str, n: int) -> list[dict]:
    """GitHub repo search (unauthenticated, low rate limit, cached for an hour)."""
    r = await _get("https://api.github.com/search/repositories",
                   params={"q": q, "sort": "stars", "order": "desc", "per_page": n},
                   headers={"Accept": "application/vnd.github+json"})
    out = []
    for h in (r.json().get("items") or []):
        if (h.get("stargazers_count") or 0) < 50:
            continue
        desc = (h.get("description") or "")[:240]
        out.append({"title": f"{h.get('full_name', '')} · {h.get('stargazers_count', 0)}★",
                    "url": h.get("html_url", ""), "snippet": desc, "source": "github"})
    return out[:n]


FEDERATED = (_hn, _stackoverflow)


async def _federated(q: str, n: int) -> list[dict]:
    """Query the keyless public APIs in parallel and interleave the results."""
    fns = list(FEDERATED) + ([_github] if CODE_HINT.search(q) else [])
    buckets = await asyncio.gather(*(fn(q, n) for fn in fns), return_exceptions=True)
    merged: list[dict] = []
    for i in range(n):
        for bucket in buckets:
            if isinstance(bucket, Exception) or not isinstance(bucket, list):
                continue
            if i < len(bucket):
                merged.append(bucket[i])
    return merged


async def search(q: str, n: int = 6) -> list[dict]:
    q = (q or "").strip()
    if not q:
        return []
    cached = _cached("search", q)
    if cached:
        return cached
    order = []
    name = provider_name()
    if name == "tavily":
        order = [_tavily, _ddg, _wikipedia]
    elif name == "brave":
        order = [_brave, _ddg, _wikipedia]
    else:
        order = [x for x in (_searx, _ddg, _wikipedia) if x is not _searx or os.getenv('CORTEX_SEARX_URL')]
    primary: list[dict] = []
    for fn in order:
        try:
            primary = await fn(q, n)
            if primary:
                primary = [r for r in primary if r.get("url") or r.get("is_summary")]
                break
        except Exception:
            continue
    full_web = provider_name() in ("tavily", "brave", "searx")
    extra = []
    if primary and not full_web:
        # the keyless chain is narrow, so top it up from the public APIs;
        # a real search backend already covers the web and doesn't need them
        try:
            extra = await _federated(q, max(3, n // 2))
        except Exception:
            extra = []
    # interleave sources so one provider can never fill the whole result set
    def bucket_of(item):
        if item.get("source"):
            return item["source"]
        url = item.get("url") or ""
        if "wikipedia.org" in url:
            return "wikipedia"
        return "web"

    groups: dict[str, list[dict]] = {}
    for item in primary + extra:
        groups.setdefault(bucket_of(item), []).append(item)
    seen, merged = set(), []
    order = list(groups)
    terms = [t for t in re.findall(r"[a-z][a-z'-]{2,}", q.lower()) if t not in STOP]
    while any(groups.values()):
        for name in order:
            bucket = groups.get(name) or []
            while bucket:
                item = bucket.pop(0)
                key = (item.get("url") or "").split("#")[0] or item.get("title")
                if not key or key in seen:
                    continue
                seen.add(key)
                merged.append(item)
                break
    if terms:
        def score(item):
            text = f"{item.get('title', '')} {item.get('snippet', '')}".lower()
            hits = sum(1 for t in terms if t in text)
            if _recent(q) and item.get("source") in ("hn", "stackoverflow", "github"):
                hits += 0.5          # fresher, community-voted sources win ties
            return hits
        scored = [(score(i), i) for i in merged]
        keep = [i for sc, i in scored if sc >= 1] or [i for _, i in scored]
        merged = sorted(keep, key=lambda i: -score(i))
    if merged:
        _store("search", q, merged[: max(n, 8)])
        return merged[: max(n, 8)]
    return []


# ------------------------------------------------------------------- fetch
def extract_text(raw: str, limit: int = MAX_PAGE_CHARS) -> str:
    body = BLOCK.sub(" ", raw)
    body = re.sub(r"<!--.*?-->", " ", body, flags=re.S)
    main = re.search(r"<(article|main)\b[^>]*>(.*?)</\1>", body, re.S | re.I)
    if main and len(main.group(2)) > 800:
        body = main.group(2)
    body = re.sub(r"<(h[1-3])\b[^>]*>(.*?)</\1>", r"\n\n## \2\n", body, flags=re.S | re.I)
    body = re.sub(r"</(p|div|li|tr|section)>", "\n", body, flags=re.I)
    body = re.sub(r"<li\b[^>]*>", "- ", body, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", body)
    text = html.unescape(text)
    text = re.sub(r"[ \t\u00a0]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    lines = [ln.strip() for ln in text.splitlines()]
    junk = re.compile(r"^(home\s*[»›/]|published\b|updated\b|written by\b|by [A-Z][\w .'-]{2,40}$|"
                      r"share\b|advertisement\b|subscribe\b|sign in\b|skip to content\b|"
                      r"cookie\b|all rights reserved|copyright ©)", re.I)
    kept = []
    for ln in lines:
        if len(ln) <= 2 and not ln.startswith("##"):
            continue
        if junk.match(ln):
            continue
        if ln.count("»") >= 1 and len(ln) < 80:
            continue
        kept.append(ln)
    return "\n".join(kept)[:limit]


async def fetch(url: str, limit: int = MAX_PAGE_CHARS) -> dict:
    cached = _cached("page", url)
    if cached:
        return cached
    try:
        r = await _get(url)
        ctype = (r.headers.get("content-type") or "").lower()
        if "json" in ctype:
            try:
                pretty = json.dumps(r.json(), indent=1, ensure_ascii=False)
            except Exception:
                pretty = r.text
            text = pretty[:limit]
        elif "html" in ctype or r.text.lstrip().startswith("<"):
            text = extract_text(r.text, limit)
        else:
            text = r.text[:limit]
        title = ""
        m = re.search(r"<title[^>]*>(.*?)</title>", r.text[:20000], re.S | re.I)
        if m:
            title = html.unescape(m.group(1)).strip()[:160]
        out = {"url": str(r.url), "status": r.status_code, "title": title,
               "text": text, "chars": len(text), "type": ctype.split(";")[0]}
        _store("page", url, out)
        return out
    except Exception as exc:
        return {"url": url, "status": 0, "title": "", "text": "", "chars": 0,
                "error": f"{type(exc).__name__}: {exc}"[:200]}


# --------------------------------------------------------------- research
STOP = {"what", "who", "when", "where", "why", "how", "is", "are", "the", "a", "an", "of", "in",
        "to", "for", "and", "or", "do", "does", "did", "can", "i", "you", "me", "my", "it", "on"}


def _query_from(question: str) -> str:
    words = [w for w in re.findall(r"[\w'-]+", question.lower()) if w not in STOP]
    return " ".join(words[:12]) or question[:120]


def relevance(hits: list[dict], question: str) -> list[dict]:
    """Keep results that actually share vocabulary with the question."""
    terms = {w for w in re.findall(r"[a-z][a-z'-]{2,}", question.lower()) if w not in STOP}
    if not terms:
        return hits
    kept = []
    for h in hits:
        text = f"{h.get('title', '')} {h.get('snippet', '')}".lower()
        overlap = sum(1 for t in terms if t in text)
        if overlap or sum(1 for t in terms if len(t) > 5) == 0:
            kept.append(h)
    return kept or []


def fresh_query(q: str) -> str:
    """"latest X" without a year surfaces last year's comparison post; add the year."""
    if _recent(q) and not re.search(r"\b20\d\d\b", q):
        return f"{q} {time.localtime().tm_year}"
    return q


async def research(question: str, *, depth: int = 3, n: int = 6,
                   budget: float = 7.0) -> dict:
    """Search, then read the top pages **concurrently**, under a time budget.

    Reading pages one after another made every factual turn wait 15-25 seconds
    before the model saw anything, which is the difference between a chat window
    and a queue. They are independent requests, so they run together and whatever
    has not arrived by `budget` is abandoned: the search snippets alone are
    usually enough to answer, and the model can still open a page itself.
    """
    q = fresh_query(_query_from(question))
    hits = relevance(await search(q, n), question)
    if not hits:
        hits = relevance(await search(question[:120], n), question)
    if not hits:
        return {"query": q, "results": [], "pages": [], "provider": provider_name()}

    async def read(hit: dict):
        url = hit.get("url") or ""
        if not url:
            return None
        try:
            page = await asyncio.wait_for(fetch(url), timeout=budget)
        except Exception:
            page = {}
        text = (page or {}).get("text") or ""
        if len(text) < 400:
            # paywall, JS shell or PDF: keep the search snippet so the source still
            # has substance instead of arriving as an empty page
            snippet = hit.get("snippet") or ""
            if len(snippet) <= 120:
                return None
            page = {**(page or {}), "text": snippet, "from_snippet": True}
        page["snippet"] = hit.get("snippet", "")
        return page

    candidates = hits[: max(depth * 3, depth)]
    tasks = [asyncio.create_task(read(h)) for h in candidates]
    done, pending = await asyncio.wait(tasks, timeout=budget + 1.0)
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)   # quiet teardown
    pages = []
    for task in tasks:
        if task not in done or task.cancelled() or task.exception():
            continue
        page = task.result()
        if page:
            pages.append(page)
        if len(pages) >= depth:
            break
    return {"query": q, "results": hits, "pages": pages, "provider": provider_name()}


def to_sources(results: list[dict], pages: list[dict], limit: int = 4) -> list[dict]:
    """Shape web material like document passages so citations reuse one path."""
    out = []
    by_url = {p["url"]: p for p in pages}
    for hit in results[:limit]:
        url = hit.get("url") or ""
        if not url and hit.get("is_summary") and (hit.get("snippet") or "").strip():
            # the search engine's own synthesis of its top results: no URL, but
            # often the cleanest statement of the fact the question is after
            out.append({
                "doc_id": f"web:summary:{hashlib.sha1((hit.get('snippet') or '').encode()).hexdigest()[:8]}",
                "doc_name": "Search summary", "page": None, "heading": "search engine",
                "text": hit["snippet"][:1200], "score": 0.95, "url": "", "kind": "web",
            })
            continue
        if not url:
            continue
        page = by_url.get(url)
        snippet = ((page or {}).get("snippet") or hit.get("snippet") or "").strip()
        body = (page or {}).get("text") or ""
        # the search snippet is clean prose; the page adds depth behind it
        text = (snippet + "\n\n" + body) if len(snippet) > 120 and body else (body or snippet)
        text = re.sub(r"^\s*(From Wikipedia, the free encyclopedia|From Wikipedia)\s*", "", text).strip()
        if not text:
            continue
        out.append({
            "doc_id": f"web:{hashlib.sha1(url.encode()).hexdigest()[:10]}",
            "doc_name": (page or {}).get("title") or hit.get("title") or urlparse(url).netloc,
            "page": None,
            "heading": urlparse(url).netloc,
            "text": text[:4000],
            "score": 0.9 if page else 0.75,
            "url": url,
            "kind": "web",
        })
    return out
