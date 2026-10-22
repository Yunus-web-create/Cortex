"""Generated media: images, speech and video saved inside the app.

Every artefact lands in DATA_DIR/media/<conversation>/ so it survives restarts
and is served back through /media/... instead of a vendor URL that expires.
"""
from __future__ import annotations

import base64
import hashlib
import mimetypes
import os
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx

from . import config, free_media, gateway

ROOT = Path(config.DATA_DIR) / "media"
ROOT.mkdir(parents=True, exist_ok=True)
MAX_DOWNLOAD = int(os.getenv("CORTEX_MEDIA_MAX_BYTES", str(40 * 1024 * 1024)))


class MediaError(RuntimeError):
    pass


def _dir(conv_id: str) -> Path:
    d = ROOT / re.sub(r"[^A-Za-z0-9_-]", "", conv_id or "scratch")[:40]
    d.mkdir(parents=True, exist_ok=True)
    return d


def _name(prefix: str, ext: str) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{stamp}-{hashlib.sha1(str(time.time()).encode()).hexdigest()[:5]}.{ext.lstrip('.')}"


async def _save_bytes(conv_id: str, data: bytes, prefix: str, ext: str) -> dict:
    if len(data) > MAX_DOWNLOAD:
        raise MediaError("generated file is too large to store")
    path = _dir(conv_id) / _name(prefix, ext)
    path.write_bytes(data)
    return _public(conv_id, path)


async def _save_url(conv_id: str, url: str, prefix: str, ext: str) -> dict:
    if url.startswith("data:"):
        header, _, payload = url.partition(",")
        data = base64.b64decode(payload)
        guessed = "png" if "png" in header else "jpg" if "jpeg" in header else ext
        return await _save_bytes(conv_id, data, prefix, guessed)
    async with httpx.AsyncClient(timeout=120, follow_redirects=True) as c:
        r = await c.get(url)
        if r.status_code >= 400:
            raise MediaError(f"could not download generated file ({r.status_code})")
        data = r.content
    suffix = Path(urlparse(url).path).suffix.lstrip(".") or ext
    if len(data) > MAX_DOWNLOAD:
        raise MediaError("generated file is too large to store")
    path = _dir(conv_id) / _name(prefix, suffix)
    path.write_bytes(data)
    return _public(conv_id, path)


def _public(conv_id: str, path: Path) -> dict:
    rel = path.relative_to(ROOT).as_posix()
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return {
        "url": f"/media/{rel}",
        "path": rel,
        "bytes": path.stat().st_size,
        "mime": mime,
        "name": path.name,
    }


async def image(conv_id: str, prompt: str, model: str = "", *, size: str = "1024x1024") -> dict:
    """Provider image model when one works, the keyless renderer otherwise."""
    models = await gateway.models()
    if not model:
        from . import router
        model = (router.candidates("image", models, limit=1) or [""])[0]

    if model and not gateway.is_free() and gateway.can("image"):
        for attempt in ("generate_image", "generate_image_chat"):
            try:
                items = (await gateway.generate_image(model, prompt, size=size)
                         if attempt == "generate_image"
                         else await gateway.generate_image_chat(model, prompt))
            except gateway.ProviderError as exc:
                if getattr(exc, "kind", "") == "billing":
                    p = gateway.active_provider()
                    if p:
                        gateway.mark_down(p, 3600)
                continue
            if items:
                item = items[0]
                saved = (await _save_url(conv_id, item["url"], "image", "png")
                         if item.get("url")
                         else await _save_bytes(conv_id, base64.b64decode(item["b64"]), "image", "png"))
                saved.update({"kind": "image", "model": model, "prompt": prompt})
                return saved

    width, height = 1024, 1024
    try:
        w, h = size.lower().split("x")
        width, height = int(w), int(h)
    except Exception:
        pass
    try:
        data = await free_media.fetch_image(prompt, width=width, height=height)
    except Exception as exc:
        raise MediaError(f"image failed: {exc}")
    saved = await _save_bytes(conv_id, data, "image", "png")
    saved.update({"kind": "image", "model": "flux (free renderer)", "prompt": prompt, "free": True})
    return saved


async def speech(conv_id: str, text: str, model: str = "", *, voice: str = "alloy",
                 fmt: str = "mp3") -> dict:
    models = await gateway.models()
    if not model:
        from . import router
        model = (router.candidates("speech", models, limit=1) or [""])[0]
    if model and not gateway.is_free() and gateway.can("speech"):
        try:
            data = await gateway.speech(model, text[:4000], voice=voice, fmt=fmt)
            saved = await _save_bytes(conv_id, data, "speech", fmt)
            saved.update({"kind": "audio", "model": model, "text": text[:400]})
            return saved
        except Exception:
            pass                       # fall through to the free voices
    try:
        data = await free_media.tts(text[:4000], voice=voice)
    except Exception as exc:
        raise MediaError(f"speech failed: {exc}")
    saved = await _save_bytes(conv_id, data, "speech", "mp3")
    saved.update({"kind": "audio", "model": f"edge-tts ({free_media._voice(voice)})",
                  "text": text[:400], "free": True})
    return saved


async def video(conv_id: str, prompt: str, model: str = "", *, seconds: int = 5,
                aspect: str = "16:9", narration: str = "", voice: str = "") -> dict:
    models = await gateway.models()
    if not model:
        from . import router
        model = (router.candidates("video", models, limit=1) or [""])[0]
    if model and not gateway.is_free() and gateway.can("video"):
        try:
            payload = await gateway.video(model, prompt, seconds=seconds, aspect=aspect)
            saved = (await _save_url(conv_id, payload["url"], "video", "mp4") if payload.get("url")
                     else await _save_bytes(conv_id, base64.b64decode(payload["b64"]), "video", "mp4"))
            saved.update({"kind": "video", "model": model, "prompt": prompt})
            return saved
        except Exception:
            pass                       # fall through to the local renderer
    try:
        data, mode = await free_media.compose_video(
            prompt, seconds=seconds, aspect=aspect, narration=narration, voice=voice)
    except Exception as exc:
        raise MediaError(f"video failed: {exc}")
    saved = await _save_bytes(conv_id, data, "video", "mp4")
    saved.update({"kind": "video", "model": f"local renderer ({mode})", "prompt": prompt,
                  "free": True})
    return saved
