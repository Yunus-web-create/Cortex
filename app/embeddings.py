"""Local semantic embeddings via fastembed (ONNX). No API key required.

Degrades gracefully to keyword-only retrieval if the model can't load
(e.g. offline first run), so the app always works.
"""
from __future__ import annotations

import logging
import threading
from typing import Sequence

import numpy as np

from . import config

log = logging.getLogger("cortex.embeddings")

_lock = threading.Lock()
_load_lock = threading.Lock()
_model = None
_state = {"tried": False, "status": "cold", "error": "", "dim": 0, "model": config.EMBED_MODEL}


def _load():
    """Thread-safe lazy load. Concurrent callers block until the model is ready
    instead of silently falling back to keyword-only retrieval mid-load."""
    global _model
    if _model is not None:
        return _model
    with _load_lock:
        if _model is not None:
            return _model
        if _state["status"] == "unavailable":
            return None
        _state["status"] = "loading"
        _state["tried"] = True
        try:
            from fastembed import TextEmbedding
            _model = TextEmbedding(model_name=config.EMBED_MODEL)
            _state["status"] = "ready"
            _state["error"] = ""
            log.info("Embedding model ready: %s", config.EMBED_MODEL)
        except Exception as exc:  # pragma: no cover - environment dependent
            _state["status"] = "unavailable"
            _state["error"] = str(exc)[:300]
            log.warning("Embeddings unavailable, keyword-only mode: %s", exc)
            _model = None
    return _model


def embed_passages(texts: Sequence[str]) -> np.ndarray | None:
    if not texts:
        return None
    with _lock:
        model = _load()
        if model is None:
            return None
        try:
            vecs = np.array(list(model.passage_embed(list(texts))), dtype=np.float32)
            if _state["dim"] == 0 and vecs.size:
                _state["dim"] = int(vecs.shape[1])
            return _normalize(vecs)
        except Exception as exc:
            log.warning("passage_embed failed: %s", exc)
            return None


def embed_query(text: str) -> np.ndarray | None:
    with _lock:
        model = _load()
        if model is None:
            return None
        try:
            vec = np.array(next(iter(model.query_embed([text]))), dtype=np.float32)
            return _normalize(vec.reshape(1, -1))[0]
        except Exception as exc:
            log.warning("query_embed failed: %s", exc)
            return None


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype(np.float32)


def status() -> dict:
    return {
        "status": _state["status"],
        "model": _state["model"],
        "dim": _state["dim"],
        "error": _state["error"],
        "semantic": _state["status"] == "ready",
    }


def warm_up() -> None:
    """Load in the background at startup (first run downloads ~90 MB)."""
    t = threading.Thread(target=_load, daemon=True)
    t.start()
