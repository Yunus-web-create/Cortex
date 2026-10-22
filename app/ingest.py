"""Ingestion pipeline: file -> text blocks -> chunks -> embeddings -> SQLite."""
from __future__ import annotations

import hashlib
import logging
import time
import uuid
from pathlib import Path

from . import chunker, db, embeddings, parsers, retrieval

log = logging.getLogger("cortex.ingest")


def _existing_by_name(name: str) -> dict | None:
    for d in db.list_documents():
        if d["name"].lower() == name.lower():
            return d
    return None


def ingest_file(path: Path, display_name: str | None = None,
                replace_existing: bool = True) -> dict:
    t0 = time.perf_counter()
    name = display_name or path.name
    blocks, pages = parsers.parse_file(path)
    chunks = chunker.chunk_blocks(blocks)
    if not chunks:
        raise ValueError("Document produced no usable chunks (empty or unreadable).")

    old = _existing_by_name(name) if replace_existing else None
    if old:
        db.delete_document(old["id"])

    doc_id = uuid.uuid4().hex[:12]
    total_chars = sum(len(c["text"]) for c in chunks)
    db.add_document(doc_id, name, path.suffix.lower(), path.stat().st_size,
                    pages, len(chunks), total_chars)

    vectors = embeddings.embed_passages([c["text"] for c in chunks])
    db.add_chunks(doc_id, chunks, vectors)
    retrieval.invalidate()

    doc = {
        "id": doc_id,
        "name": name,
        "ext": path.suffix.lower(),
        "size": path.stat().st_size,
        "pages": pages,
        "n_chunks": len(chunks),
        "n_chars": total_chars,
        "embedded": vectors is not None,
        "replaced": bool(old),
        "ms": round((time.perf_counter() - t0) * 1000),
    }
    log.info("Ingested %s: %s chunks in %sms", name, len(chunks), doc["ms"])
    return doc


def ingest_bytes(data: bytes, filename: str, replace_existing: bool = True) -> dict:
    safe = Path(filename).name
    ext = Path(safe).suffix.lower()
    tmp_dir = Path("/tmp/cortex_uploads")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp = tmp_dir / f"{hashlib.md5(data).hexdigest()[:16]}{ext}"
    tmp.write_bytes(data)
    try:
        return ingest_file(tmp, display_name=safe, replace_existing=replace_existing)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


def reindex_embeddings() -> dict:
    """Re-embed every chunk (use after enabling the embedding model)."""
    chunks = db.all_chunks()
    if not chunks:
        return {"chunks": 0, "embedded": 0}
    vectors = embeddings.embed_passages([c["text"] for c in chunks])
    if vectors is None:
        return {"chunks": len(chunks), "embedded": 0,
                "error": embeddings.status().get("error", "embedding model unavailable")}
    import numpy as np
    # rebuild all embeddings: delete then insert per chunk
    import sqlite3
    with db.db() as conn:
        conn.execute("DELETE FROM embeddings")
        conn.executemany(
            "INSERT INTO embeddings(chunk_id, dim, vec) VALUES(?,?,?)",
            [(c["id"], int(v.shape[0]), np.asarray(v, dtype=np.float32).tobytes())
             for c, v in zip(chunks, vectors)],
        )
    retrieval.invalidate()
    return {"chunks": len(chunks), "embedded": int(len(vectors))}
