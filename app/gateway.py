"""Provider layer: one OpenAI-compatible client, many vendors.

Any endpoint that speaks the OpenAI Chat Completions dialect works here —
Vercel AI Gateway (default), OpenRouter, OpenAI, Groq, DeepSeek, Together,
Fireworks, or a local Ollama. The rest of the app never talks to a vendor SDK.
"""
from __future__ import annotations

import asyncio
import base64
import contextvars
import json
import re
import os
import time
from dataclasses import dataclass, field
from typing import AsyncIterator
from urllib.parse import quote

import httpx

from . import db

USER_AGENT = "Cortex/1.1 (+https://github.com/)"


class ProviderError(RuntimeError):
    def __init__(self, message: str, status: int = 0, retry: bool = False, kind: str = ""):
        super().__init__(message)
        self.status = status
        self.retry = retry
        self.kind = kind            # e.g. "billing" - the key works, the account can't spend


@dataclass
class Provider:
    name: str
    base_url: str
    api_key: str
    models_path: str = "/models"
    chat_path: str = "/chat/completions"
    image_path: str = "/images/generations"
    speech_path: str = "/audio/speech"
    video_path: str = "/video/generations"
    headers: dict = field(default_factory=dict)
    keyless: bool = False
    # what this provider's API can actually do; the media layer checks before calling
    caps: frozenset = frozenset({"chat", "image", "speech", "video"})

    def auth_headers(self) -> dict:
        """Vercel accepts a personal token under x-vercel-token; gateways want Bearer."""
        h = {"User-Agent": USER_AGENT, **self.headers}
        if not self.api_key:
            return h                      # keyless community endpoint: no auth header at all
        if self.api_key.startswith("vcp_"):
            h["x-vercel-token"] = self.api_key      # catalogue access, no inference
        else:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    @property
    def inference_ready(self) -> bool:
        if self.keyless:
            return True
        return bool(self.api_key) and not self.api_key.startswith("vcp_")


# name -> (base_url, key env vars)
KNOWN = {
    "gateway": ("https://ai-gateway.vercel.sh/v1", ("AI_GATEWAY_API_KEY", "VERCEL_AI_GATEWAY_KEY", "VERCEL_TOKEN")),
    "openrouter": ("https://openrouter.ai/api/v1", ("OPENROUTER_API_KEY",)),
    "openai": ("https://api.openai.com/v1", ("OPENAI_API_KEY",)),
    "groq": ("https://api.groq.com/openai/v1", ("GROQ_API_KEY",)),
    "deepseek": ("https://api.deepseek.com/v1", ("DEEPSEEK_API_KEY",)),
    "together": ("https://api.together.xyz/v1", ("TOGETHER_API_KEY",)),
    "google": ("https://generativelanguage.googleapis.com/v1beta/openai", ("GEMINI_API_KEY", "GOOGLE_API_KEY")),
    "ollama": ("http://localhost:11434/v1", ("OLLAMA_API_KEY",)),
    # llama.cpp's own server, started by scripts/install_local_model.sh
    "local": (os.getenv("CORTEX_LOCAL_URL", "http://127.0.0.1:8080/v1"), ()),
    # no key, no card: a community inference endpoint used only as a last resort
    "free": ("https://text.pollinations.ai/openai", ("POLLINATIONS_API_KEY",)),
}


def _runtime_keys() -> dict:
    return db.get_kv("provider_keys", {}) or {}


def save_key(provider: str, key: str) -> None:
    keys = _runtime_keys()
    if key:
        keys[provider] = key.strip()
    else:
        keys.pop(provider, None)
    db.set_kv("provider_keys", keys)


def _key_for(name: str) -> str:
    runtime = _runtime_keys()
    if runtime.get(name):
        return runtime[name]
    if name not in KNOWN:
        return os.getenv("CORTEX_API_KEY", "").strip()
    if name == "gateway":
        return (os.getenv("AI_GATEWAY_API_KEY") or os.getenv("VERCEL_AI_GATEWAY_KEY")
                or "").strip()
    for env in KNOWN[name][1]:
        if os.getenv(env):
            return os.getenv(env, "").strip()
    return ""


_OVERRIDE: contextvars.ContextVar = contextvars.ContextVar("cortex_provider", default=None)
_LOCAL_AT = 0.0
_LOCAL_UP = False


def local_up() -> bool:
    """Is a local llama.cpp / Ollama server listening? Cached for 30 s."""
    global _LOCAL_AT, _LOCAL_UP
    if os.getenv("CORTEX_LOCAL_MODEL", "1").strip() == "0":
        return False
    now = time.time()
    if now - _LOCAL_AT < 30:
        return _LOCAL_UP
    _LOCAL_AT = now
    base = KNOWN["local"][0]
    try:
        host = base.split("//", 1)[1].split("/")[0]
        name, _, port = host.partition(":")
        import socket
        with socket.create_connection((name, int(port or 80)), timeout=0.35):
            _LOCAL_UP = True
    except Exception:
        _LOCAL_UP = False
    return _LOCAL_UP


def local_provider() -> Provider | None:
    if not local_up():
        return None
    return Provider(name="local", base_url=KNOWN["local"][0].rstrip("/"), api_key="",
                    keyless=True, caps=frozenset({"chat"}))


def push_provider(p: Provider):
    """Pin every gateway call in this task to one provider (used for failover)."""
    return _OVERRIDE.set(p)


def pop_provider(token) -> None:
    try:
        _OVERRIDE.reset(token)
    except ValueError:
        _OVERRIDE.set(None)


def _cooldowns() -> dict:
    now = time.time()
    out = {k: v for k, v in (db.get_kv("provider_cooldown", {}) or {}).items() if v > now}
    return out


def model_cooldowns() -> dict:
    """Models this key cannot use right now, with the unix time they free up."""
    now = time.time()
    raw = db.get_kv("model_cooldown", {}) or {}
    return {k: v for k, v in raw.items() if float(v or 0) > now}


def mark_model_down(model: str, seconds: int) -> None:
    """Park one model: 503 means busy for a while, a quota 429 means for longer."""
    down = model_cooldowns()
    down[model.split("/")[-1]] = time.time() + seconds
    db.set_kv("model_cooldown", down)


def clear_model_down(model: str) -> None:
    down = model_cooldowns()
    key = model.split("/")[-1]
    if key in down:
        down.pop(key, None)
        db.set_kv("model_cooldown", down)


def mark_down(p: Provider, seconds: int = 900) -> None:
    """Skip this provider for a while: it is rate-limited, broke, or can't spend."""
    down = _cooldowns()
    down[p.name] = time.time() + seconds
    db.set_kv("provider_cooldown", down)


def clear_down(name: str) -> None:
    down = _cooldowns()
    if name in down:
        down.pop(name, None)
        db.set_kv("provider_cooldown", down)


LABELS = {
    "gateway": "Vercel AI Gateway", "openrouter": "OpenRouter", "openai": "OpenAI",
    "groq": "Groq", "deepseek": "DeepSeek", "together": "Together", "google": "Google AI Studio",
    "ollama": "Ollama (local)", "local": "local llama.cpp", "free": "free community model",
    "custom": "custom endpoint",
}


def provider_label(p: Provider | None) -> str:
    return LABELS.get(getattr(p, "name", ""), getattr(p, "name", "provider") or "provider")


FREE_FIRST = ("google", "groq", "openrouter", "ollama", "free")


def pool() -> list[Provider]:
    """Every provider this install can actually use, best first.

    The configured provider leads; then card-free tiers; then paid keys the user
    happens to have; then the keyless community endpoint. Providers cooling down
    after a 429/402 stay out unless they are the only option left.
    """
    primary = active_provider()
    out: list[Provider] = []
    seen: set[str] = set()

    def add(p: Provider | None) -> None:
        if p and p.inference_ready and p.name not in seen:
            seen.add(p.name)
            out.append(p)

    add(primary)
    # even a forced provider fails over: a 429 should not end the turn
    if True:
        for name in FREE_FIRST:
            if name == "free":
                if os.getenv("CORTEX_FREE_MODEL", "1").strip() != "0":
                    add(Provider(name="free", base_url=KNOWN["free"][0], api_key="", keyless=True,
                                 models_path="https://text.pollinations.ai/models",
                                 caps=frozenset({"chat", "image"})))
                add(local_provider())        # always-works, unlimited, runs on this machine
                continue
            if name in seen:
                continue
            key = _key_for(name)
            if name == "ollama" and not key:
                continue                     # no server check here; opting in is enough
            if key:
                caps = (frozenset({"chat"}) if name == "google"
                        else frozenset({"chat", "image", "speech", "video"}))
                add(Provider(name=name, base_url=KNOWN[name][0].rstrip("/"), api_key=key,
                             caps=caps))
        block = (db.get_kv("provider_block", {}) or {}).get("kind")
        for name in ("openai", "deepseek", "together", "gateway"):
            if name in seen:
                continue
            if name == "gateway" and block == "billing":
                continue
            key = _key_for(name)
            if key and not key.startswith("vcp_"):
                add(Provider(name=name, base_url=KNOWN[name][0].rstrip("/"), api_key=key))
    down = _cooldowns()
    live = [p for p in out if p.name not in down]
    return live or out[:1]


def active_provider() -> Provider | None:
    """First provider with a key, honouring CORTEX_PROVIDER and any pin."""
    pinned = _OVERRIDE.get()
    if pinned is not None:
        return pinned
    forced = os.getenv("CORTEX_PROVIDER", "").strip().lower()
    order = [forced] if forced in KNOWN else []
    if forced == "custom" or ("custom" in _runtime_keys() and os.getenv("CORTEX_BASE_URL")):
        base = os.getenv("CORTEX_BASE_URL", "").strip()
        key = _key_for("custom")
        if base and key:
            return Provider(name="custom", base_url=base.rstrip("/"), api_key=key)
    order += [p for p in ("gateway", "openrouter", "openai", "groq", "deepseek", "together",
                          "google", "ollama") if p not in order]
    block = (db.get_kv("provider_block", {}) or {}).get("kind")
    for name in order:
        key = _key_for(name)
        base = os.getenv("CORTEX_BASE_URL", "").strip() or KNOWN[name][0]
        if name in ("ollama", "local") and not key and os.getenv("CORTEX_PROVIDER") == name:
            key = name
        if not key:
            continue
        if block == "billing" and name == "gateway":
            continue          # the key works but the account cannot spend; don't keep trying it
        caps = (frozenset({"chat"}) if name == "google"
                else frozenset({"chat", "image", "speech", "video"}))
        return Provider(name=name, base_url=base.rstrip("/"), api_key=key, caps=caps)
    base = os.getenv("CORTEX_BASE_URL", "").strip()
    key = os.getenv("CORTEX_API_KEY", "").strip()
    if base and key:
        return Provider(name="custom", base_url=base.rstrip("/"), api_key=key)
    if os.getenv("CORTEX_FREE_MODEL", "1").strip() != "0":
        return Provider(name="free", base_url=KNOWN["free"][0], api_key="", keyless=True,
                        models_path="https://text.pollinations.ai/models",
                        caps=frozenset({"chat", "image"}))
    return None


def configured() -> bool:
    p = active_provider()
    return bool(p and p.inference_ready)


def provider_name() -> str:
    p = active_provider()
    return p.name if p else "none"


def can(kind: str) -> bool:
    """Can the active provider generate this kind of media itself?"""
    p = active_provider()
    return bool(p and p.inference_ready and kind in (p.caps or ()))


def is_free() -> bool:
    """True when the active provider costs nothing (community keyless or local)."""
    p = active_provider()
    return bool(p and (p.keyless or p.name == "local"))


# Providers whose *default* tier needs no card, so a turn on them bills nothing.
FREE_TIER = {"free", "local", "google", "groq", "ollama"}


def free_tier() -> bool:
    """True when turns on this provider cost nothing (no card, no credit)."""
    p = active_provider()
    if not p:
        return False
    if p.name == "openrouter":
        return not spendable()      # $0 balance → the router only picks :free models
    return p.name in FREE_TIER


def status() -> dict:
    p = active_provider()
    block = db.get_kv("provider_block", {}) or {}
    try:
        chain = [{"name": q.name, "label": provider_label(q), "model_host": q.base_url} for q in pool()]
    except Exception:
        chain = []
    if block and (block.get("provider") != (p.name if p else None)
                  or time.time() - float(block.get("at") or 0) > 3600):
        block = {}                      # stale, or left over from another provider
    return {
        "pool": chain,
        "cooling_down": list(_cooldowns()),
        "configured": bool(p and p.inference_ready),
        "provider": p.name if p else None,
        "base_url": p.base_url if p else None,
        "key_kind": ("vercel-token (catalogue only)" if p and p.api_key.startswith("vcp_")
                     else "api-key" if p else None),
        "blocked": block.get("kind") or None,
        "free": is_free(),
        "free_tier": free_tier(),
        "tier": "free" if free_tier() else "paid",
        "codes": sorted(p.caps) if p else [],
        "local": local_up(),
    }


def _client(p: Provider, timeout: float) -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=timeout, headers=p.auth_headers())


BILLING_HINT = (
    "Your AI Gateway key is valid, but Vercel will not run inference until a payment method is "
    "on file. Add a card at vercel.com -> AI -> Billing (the free credits still apply), then send "
    "the message again. Model listing, routing and the UI all keep working meanwhile."
)


async def _error(resp: httpx.Response) -> ProviderError:
    try:
        body = resp.json()
        if isinstance(body, list):              # Google wraps errors in a list
            body = body[0] if body else {}
        err = body.get("error") if isinstance(body, dict) else None
        msg = err.get("message") if isinstance(err, dict) else (err or body)
        msg = str(msg or "").strip() or json.dumps(body)[:300]
    except Exception:
        msg = (resp.text or "")[:300]
    low = str(msg).lower()
    # "billing" on its own is too broad: Gemini's ordinary quota 429 says "check
    # your plan and billing details", which is not a billing block.
    hard_billing = ("credit card", "customer_verification_required", "payment method",
                    "requires a billing account", "billing must be enabled",
                    "billing is not enabled", "insufficient credit", "no credits")
    if resp.status_code in (402, 403) and any(k in low for k in hard_billing) \
            or any(k in low for k in hard_billing):
        return ProviderError(BILLING_HINT, resp.status_code, False, kind="billing")
    quota = resp.status_code == 429 or "quota" in low or "rate limit" in low
    retry = resp.status_code in (429, 500, 502, 503, 504)
    kind = "quota" if quota else ""
    short = re.sub(r"\s+", " ", msg).split(" For more information")[0]
    short = short.split(" head to:")[0].strip()[:180]
    hint = " (free-tier quota — it resets within a minute)" if quota and resp.status_code == 429 else ""
    return ProviderError(f"{resp.status_code}: {short}{hint}", resp.status_code, retry, kind=kind)


# --------------------------------------------------------------- catalogue
CATALOG_TTL = 6 * 3600


def _is_free_priced(m: dict, mid: str) -> bool:
    """A model that costs nothing per token, by pricing table or by id suffix."""
    if mid.endswith(":free"):
        return True
    pricing = m.get("pricing") or {}
    if isinstance(pricing, dict) and ("prompt" in pricing or "completion" in pricing):
        try:
            return float(pricing.get("prompt") or 0) == 0 and float(pricing.get("completion") or 0) == 0
        except (TypeError, ValueError):
            return False
    return False


def spendable() -> bool:
    """Can this account actually pay for a paid model? Cached for an hour."""
    p = active_provider()
    if not p or p.name != "openrouter":
        return True
    cached = db.get_kv("openrouter_credit", {}) or {}
    if time.time() - float(cached.get("at") or 0) < 3600:
        return bool(cached.get("spendable"))
    return False        # unproven balance: assume free models only, never surprise a 402


async def refresh_credit() -> bool:
    """Ask OpenRouter how much credit the key has; free-only keys get $0."""
    p = active_provider()
    if not p or p.name != "openrouter":
        return True
    spendable_now = True
    try:
        async with _client(p, 20) as c:
            r = await c.get("https://openrouter.ai/api/v1/credits")
            if r.status_code < 400:
                d = (r.json() or {}).get("data") or {}
                left = float(d.get("total_credits") or 0) - float(d.get("total_usage") or 0)
                spendable_now = left > 0.01
    except Exception:
        pass
    db.set_kv("openrouter_credit", {"at": time.time(), "spendable": spendable_now})
    return spendable_now


def free_only() -> bool:
    """True when the active provider can only use zero-price models."""
    p = active_provider()
    if not p:
        return False
    if p.keyless or p.name in ("local", "ollama"):
        return True
    if p.name == "openrouter":
        return not spendable()
    return False


async def models(force: bool = False) -> list[dict]:
    """Normalised model catalogue, cached in SQLite."""
    p = active_provider()
    if not p:
        return []
    store = db.get_kv("model_catalog_v2", {}) or {}
    by_base = store.get("by_base") if isinstance(store, dict) else {}
    by_base = by_base or {}
    cached = by_base.get(p.base_url) or {}
    if (not force and cached.get("models")
            and time.time() - cached.get("at", 0) < CATALOG_TTL):
        return cached.get("models", [])
    if p.name == "openrouter":
        await refresh_credit()
    try:
        async with _client(p, 30) as c:
            url = p.models_path if p.models_path.startswith("http") else p.base_url + p.models_path
            r = await c.get(url)
            if r.status_code >= 400:
                raise await _error(r)
            payload = r.json() or {}
            raw = payload.get("data") if isinstance(payload, dict) else payload
            raw = raw or []
    except Exception:
        return cached.get("models", [])

    out = []
    for m in raw:
        mid = m.get("id") or m.get("name") or ""
        if not mid or m.get("type") in ("evaluation", "embedding", "reranking"):
            continue
        if p.name == "google":
            mid = mid.split("/")[-1]            # the API accepts either form; show the short one
            low = mid.lower()
            if "embedding" in low or low.endswith("-live") or "native-audio" in low \
                    or "live-preview" in low or "translate" in low:
                continue                        # not chat-completions models
            kind = ("image" if ("-image" in low or "nano-banana" in low or low.startswith("imagen"))
                    else "speech" if "tts" in low
                    else "transcription" if "transcribe" in low
                    else "video" if low.startswith("veo")
                    else "language")
            name = (m.get("display_name") or m.get("name") or mid)
            if name.startswith("models/"):
                name = name.split("/")[-1]
            m = {**m, "name": name, "owned_by": "google", "type": kind}
        if p.name == "free":
            m = {**m, "name": m.get("description") or m.get("name") or mid,
                 "owned_by": "pollinations", "type": "language"}
        if p.name in ("local", "ollama"):
            # llama-server reports the file path; show the model, not the disk
            stem = os.path.basename(str(mid)).replace(".gguf", "")
            mid = stem
            m = {**m, "name": stem.replace("-", " ").replace("_", " "),
                 "owned_by": "local", "type": "language"}
        out.append({
            "id": mid,
            "name": m.get("name") or mid.split("/")[-1],
            "vendor": m.get("owned_by") or mid.split("/")[0],
            "type": m.get("type") or "language",
            "context": m.get("context_window") or m.get("context_length") or 0,
            "max_tokens": m.get("max_tokens") or 0,
            "released": m.get("released") or m.get("created") or 0,
            "tags": m.get("tags") or [],
            "description": (m.get("description") or "")[:220],
            "modalities": m.get("modalities") or {},
            "free": _is_free_priced(m, mid),
        })
    by_base[p.base_url] = {"at": time.time(), "models": out}
    if len(by_base) > 8:
        by_base = dict(sorted(by_base.items(), key=lambda kv: kv[1].get("at", 0))[-8:])
    db.set_kv("model_catalog_v2", {"by_base": by_base})
    return out


# ------------------------------------------------------------------- chat
_FREE_LOCK: asyncio.Semaphore | None = None


_RPM: dict[str, list[float]] = {}


def google_pace(model: str) -> float:
    """How long to wait before the next call so the free tier doesn't 429 us.

    Gemini's free tier is a per-model requests-per-minute budget. A single turn
    makes several calls, so pacing them is what keeps a burst from collapsing
    into a failover to the slow community model.
    """
    try:
        limit = int(os.getenv("CORTEX_GOOGLE_RPM", "9"))
    except ValueError:
        limit = 9
    now = time.time()
    hits = [t for t in _RPM.get(model, []) if now - t < 60]
    wait = 0.0 if len(hits) < limit else max(0.0, 60 - (now - hits[0]) + 0.3)
    _RPM[model] = hits + [now + wait]      # book the slot we are about to use
    return wait


def _free_lock() -> asyncio.Semaphore:
    """One in-flight request for the keyless endpoint (it 429s otherwise)."""
    global _FREE_LOCK
    if _FREE_LOCK is None:
        _FREE_LOCK = asyncio.Semaphore(1)
    return _FREE_LOCK


async def chat_stream(model: str, messages: list[dict], *, tools: list[dict] | None = None,
                      temperature: float | None = None, max_tokens: int = 2000,
                      reasoning: str | None = None, timeout: float = 180) -> AsyncIterator[dict]:
    """Yield {'type': 'delta'|'tool_call'|'usage'|'error'} events.

    tool_call events arrive whole (id, name, arguments dict) once the stream
    finishes assembling the deltas.
    """
    p = active_provider()
    if not p:
        yield {"type": "error", "message": "No model provider configured — add a key in Settings."}
        return
    if not p.inference_ready:
        yield {"type": "error", "message": (
            "That Vercel token can list models but cannot run inference. Create an AI Gateway "
            "API key (starts with vck_) at vercel.com → AI Gateway → API Keys and paste it in Settings.")}
        return

    body: dict = {"model": model, "messages": messages, "stream": True,
                  "max_tokens": max_tokens, "stream_options": {"include_usage": True}}
    if temperature is not None:
        body["temperature"] = temperature
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    if reasoning:
        body["reasoning"] = {"effort": reasoning}
    if p.name == "free":
        body["reasoning_effort"] = "low"      # the free 20B model is chatty; keep it quick
    if p.name == "google":
        body["reasoning_effort"] = "low"      # thinking tokens would eat a short answer

    if p.name == "google":
        wait = google_pace(model)
        if wait > 12:
            yield {"type": "notice", "message": (
                "The free tier allows a handful of calls a minute — handing this turn to the "
                "next free route rather than making you wait.")}
            yield {"type": "error", "retry": True,
                   "message": "Free-tier pacing: the per-minute budget is used up."}
            return
        if wait > 0.5:
            yield {"type": "notice", "message": (
                f"Pacing for the free tier — {wait:.0f}s so Google keeps answering.")}
            await asyncio.sleep(wait)

    calls: dict[int, dict] = {}
    usage: dict = {}

    async def _consume(r) -> AsyncIterator[dict]:
        nonlocal usage
        async for line in r.aiter_lines():
            if not line or not line.startswith("data:"):
                continue
            chunk = line[5:].strip()
            if chunk == "[DONE]":
                break
            try:
                obj = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices") or []:
                delta = ch.get("delta") or {}
                if delta.get("content"):
                    clear_model_down(model)
                    block = db.get_kv("provider_block", {}) or {}
                    if block.get("kind") and block.get("provider") == p.name:
                        db.set_kv("provider_block", None)   # this provider works again
                    yield {"type": "delta", "text": delta["content"]}
                for img in delta.get("images") or []:
                    url = ((img.get("image_url") or {}).get("url") or img.get("url") or "")
                    if url:
                        yield {"type": "image", "url": url}
                for tc in delta.get("tool_calls") or []:
                    idx = tc.get("index", 0)
                    slot = calls.setdefault(idx, {"id": "", "name": "", "arguments": "",
                                                  "extra": None})
                    # Gemini 3 wants its opaque thought_signature echoed back with the
                    # tool call, or the follow-up turn is rejected with a 400.
                    if tc.get("extra_content"):
                        slot["extra"] = tc["extra_content"]
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] = fn["name"]
                    if fn.get("arguments"):
                        slot["arguments"] += fn["arguments"]
                if ch.get("finish_reason") == "error":
                    yield {"type": "error", "message": "The model stopped with an error."}

    # The keyless community endpoint serves one request per IP at a time, so a
    # second tab used to fail with 429. Serialise and retry instead of erroring.
    # Google's free tier also throws short 503 "high demand" spikes that clear on
    # a second attempt, so give it one quick retry before parking the model.
    attempts = 6 if p.name == "free" else 3 if p.name == "google" else 1
    for attempt in range(attempts):
        lock = _free_lock() if p.name == "free" else None
        if lock is not None:
            await lock.acquire()
        retry = False
        try:
            async with _client(p, timeout) as c:
                async with c.stream("POST", p.base_url + p.chat_path, json=body) as r:
                    # a quota 429 will not clear in two seconds — park the model and
                    # let the caller try the next one instead of burning requests
                    if r.status_code in (500, 502, 503, 504) and attempt < attempts - 1:
                        await r.aread()
                        retry = True
                    elif r.status_code >= 400:
                        await r.aread()
                        raise await _error(r)
                    else:
                        async for ev in _consume(r):
                            yield ev
        except ProviderError as exc:
            if exc.status == 429:
                mark_model_down(model, 3600)      # quota resets hourly on most free tiers
            elif exc.status in (500, 502, 503, 504) and attempt >= attempts - 1:
                mark_model_down(model, 600)       # genuinely overloaded
            if getattr(exc, "kind", "") == "billing":
                db.set_kv("provider_block", {"kind": "billing", "provider": p.name, "at": time.time()})
            hint = ""
            if exc.status in (401, 403) and getattr(exc, "kind", "") != "billing":
                hint = " Check the key in Settings → Model access."
            yield {"type": "error", "message": str(exc) + hint, "retry": exc.retry}
            return
        except httpx.TimeoutException:
            yield {"type": "error", "message": "The model took too long to respond. Try again or pick a faster model.", "retry": True}
            return
        except httpx.HTTPError as exc:
            yield {"type": "error", "message": f"Network error talking to the provider: {exc}", "retry": True}
            return
        finally:
            if lock is not None:
                lock.release()
        if not retry:
            break
        if attempt == 0 and p.name == "free":
            yield {"type": "notice", "message": "The free model takes one request at a time — queued behind it."}
        await asyncio.sleep((4 + attempt * 3) if p.name == "free" else 1.2 + attempt)

    for slot in calls.values():
        if not slot.get("name"):
            continue
        try:
            args = json.loads(slot["arguments"] or "{}")
        except json.JSONDecodeError:
            args = {"_raw": slot["arguments"]}
        ev = {"type": "tool_call", "id": slot["id"] or f"call_{len(slot)}",
              "name": slot["name"], "arguments": args}
        if slot.get("extra"):
            ev["extra_content"] = slot["extra"]
        yield ev

    if usage:
        yield {"type": "usage", "input_tokens": int(usage.get("prompt_tokens") or 0),
               "output_tokens": int(usage.get("completion_tokens") or 0)}


async def complete(model: str, messages: list[dict], *, max_tokens: int = 400,
                   temperature: float = 0.2, timeout: float = 60) -> str:
    """Non-streaming helper for titles, routing, summaries."""
    p = active_provider()
    if not p or not p.inference_ready:
        return ""
    if p.name == "free":
        # the keyless endpoint takes one request at a time; a background helper
        # would double the latency of every turn (and sometimes 402)
        return ""
    try:
        async with _client(p, timeout) as c:
            r = await c.post(p.base_url + p.chat_path, json={
                "model": model, "messages": messages, "max_tokens": max_tokens,
                "temperature": temperature, "stream": False})
            if r.status_code >= 400:
                raise await _error(r)
            data = r.json()
            return ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    except Exception:
        return ""


# ---------------------------------------------------------------- imaging
async def generate_image(model: str, prompt: str, *, size: str = "1024x1024",
                         count: int = 1, timeout: float = 180) -> list[dict]:
    """Return [{'b64': str} | {'url': str}] from an image-only model."""
    p = active_provider()
    if not p or not p.inference_ready or "image" not in (p.caps or ()):
        raise ProviderError("No image-capable provider configured")
    if p.name == "free":
        return await _free_image(prompt, size=size, timeout=timeout)
    body = {"model": model, "prompt": prompt, "n": count, "size": size}
    async with _client(p, timeout) as c:
        r = await c.post(p.base_url + p.image_path, json=body)
        if r.status_code >= 400:
            raise await _error(r)
        data = r.json()
    out = []
    for item in data.get("data") or []:
        if item.get("b64_json"):
            out.append({"b64": item["b64_json"]})
        elif item.get("url"):
            out.append({"url": item["url"]})
    if not out:
        raise ProviderError("Provider returned no image data")
    return out


async def _free_image(prompt: str, *, size: str = "1024x1024", timeout: float = 180) -> list[dict]:
    """Keyless image generation (Pollinations). Returns base64 so it is stored locally."""
    width, height = 1024, 1024
    try:
        w, h = (size or "1024x1024").lower().split("x")
        width, height = int(w), int(h)
    except Exception:
        pass
    try:
        from . import free_media            # shared queue + retry for the free renderer
        data = await free_media.fetch_image(prompt, width=width, height=height)
    except Exception as exc:
        raise ProviderError(f"Free image service unavailable: {exc}")
    return [{"b64": base64.b64encode(data).decode()}]


async def generate_image_chat(model: str, prompt: str, *, timeout: float = 180) -> list[dict]:
    """Multimodal LLMs return images in the chat response instead."""
    p = active_provider()
    if not p or not p.inference_ready or "image" not in (p.caps or ()):
        raise ProviderError("No image-capable provider configured")
    async with _client(p, timeout) as c:
        r = await c.post(p.base_url + p.chat_path, json={
            "model": model, "messages": [{"role": "user", "content": prompt}], "stream": False})
        if r.status_code >= 400:
            raise await _error(r)
        msg = ((r.json().get("choices") or [{}])[0].get("message") or {})
    out = []
    for img in msg.get("images") or []:
        url = ((img.get("image_url") or {}).get("url") or img.get("url") or "")
        if url.startswith("data:"):
            out.append({"b64": url.split(",", 1)[-1]})
        elif url:
            out.append({"url": url})
    return out


async def speech(model: str, text: str, *, voice: str = "alloy",
                 fmt: str = "mp3", timeout: float = 120) -> bytes:
    p = active_provider()
    if not p or not p.inference_ready or "speech" not in (p.caps or ()) or p.name == "free":
        raise ProviderError("This provider has no speech endpoint.")
    async with _client(p, timeout) as c:
        r = await c.post(p.base_url + p.speech_path, json={
            "model": model, "input": text, "voice": voice, "response_format": fmt})
        if r.status_code >= 400:
            raise await _error(r)
        return r.content


async def video(model: str, prompt: str, *, seconds: int = 5, aspect: str = "16:9",
                timeout: float = 900, poll_every: float = 5.0) -> dict:
    """Submit a video job and poll until it finishes. Returns {'b64'|'url'}."""
    p = active_provider()
    if not p or not p.inference_ready or "video" not in (p.caps or ()) or p.name == "free":
        raise ProviderError("This provider has no video endpoint.")
    async with _client(p, timeout=60) as c:
        r = await c.post(p.base_url + p.video_path, json={
            "model": model, "prompt": prompt, "seconds": seconds, "aspect_ratio": aspect})
        if r.status_code >= 400:
            raise await _error(r)
        job = r.json()

    job_id = job.get("id") or job.get("generation_id") or job.get("request_id")
    if job.get("status") in ("succeeded", "completed") or job.get("data"):
        return _video_payload(job)
    if not job_id:
        return _video_payload(job)

    deadline = time.time() + timeout
    async with _client(p, 60) as c:
        while time.time() < deadline:
            await asyncio.sleep(poll_every)
            r = await c.get(f"{p.base_url}{p.video_path}/{job_id}")
            if r.status_code >= 400:
                raise await _error(r)
            data = r.json()
            state = (data.get("status") or "").lower()
            if state in ("failed", "error", "cancelled"):
                raise ProviderError(data.get("error") or "Video generation failed")
            if state in ("succeeded", "completed", "ready") or data.get("data") or data.get("url"):
                return _video_payload(data)
    raise ProviderError("Video generation timed out")


def _video_payload(data: dict) -> dict:
    items = data.get("data") or data.get("videos") or []
    if isinstance(items, dict):
        items = [items]
    for item in items:
        if isinstance(item, str) and item.startswith("http"):
            return {"url": item}
        if isinstance(item, dict):
            if item.get("b64_json"):
                return {"b64": item["b64_json"]}
            if item.get("url"):
                return {"url": item["url"]}
    if data.get("url"):
        return {"url": data["url"]}
    if data.get("base64"):
        return {"b64": data["base64"]}
    raise ProviderError("Provider returned no video")
