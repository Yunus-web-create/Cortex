"""SQLite persistence: documents, chunks, embeddings, conversations, messages."""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from typing import Any, Iterable, Iterator

import numpy as np

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    ext         TEXT NOT NULL,
    size        INTEGER NOT NULL DEFAULT 0,
    pages       INTEGER NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL,
    status      TEXT NOT NULL DEFAULT 'ready',
    n_chunks    INTEGER NOT NULL DEFAULT 0,
    n_chars     INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id      TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal     INTEGER NOT NULL,
    page        INTEGER NOT NULL DEFAULT 1,
    heading     TEXT NOT NULL DEFAULT '',
    text        TEXT NOT NULL,
    n_chars     INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id    INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    dim         INTEGER NOT NULL,
    vec         BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL DEFAULT 'New chat',
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    conv_id     TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    sources     TEXT NOT NULL DEFAULT '[]',
    meta        TEXT NOT NULL DEFAULT '{}',
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conv_id);
CREATE TABLE IF NOT EXISTS kv (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL
);
"""


@contextmanager
def db() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(config.DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init() -> None:
    with db() as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript(SCHEMA)


# ---------------------------------------------------------------- settings kv
def get_kv(key: str, default: Any = None) -> Any:
    with db() as conn:
        row = conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
    if row is None:
        return default
    try:
        return json.loads(row["value"])
    except Exception:
        return default


def set_kv(key: str, value: Any) -> None:
    with db() as conn:
        conn.execute(
            "INSERT INTO kv(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, json.dumps(value)),
        )


# ------------------------------------------------------------------ documents
def add_document(doc_id: str, name: str, ext: str, size: int, pages: int,
                 n_chunks: int, n_chars: int) -> None:
    with db() as conn:
        conn.execute(
            "INSERT INTO documents(id, name, ext, size, pages, created_at, status, n_chunks, n_chars) "
            "VALUES(?,?,?,?,?,?, 'ready', ?, ?)",
            (doc_id, name, ext, size, pages, time.time(), n_chunks, n_chars),
        )


def add_chunks(doc_id: str, chunks: Iterable[dict],
               vectors: np.ndarray | None) -> int:
    with db() as conn:
        ids = []
        for c in chunks:
            cur = conn.execute(
                "INSERT INTO chunks(doc_id, ordinal, page, heading, text, n_chars) "
                "VALUES(?,?,?,?,?,?)",
                (doc_id, c["ordinal"], c.get("page", 1), c.get("heading", ""),
                 c["text"], len(c["text"])),
            )
            ids.append(cur.lastrowid)
        if vectors is not None and len(vectors) == len(ids) and len(ids):
            conn.executemany(
                "INSERT OR REPLACE INTO embeddings(chunk_id, dim, vec) VALUES(?,?,?)",
                [(cid, int(v.shape[0]), v.astype(np.float32).tobytes())
                 for cid, v in zip(ids, vectors)],
            )
    return len(ids)


def list_documents() -> list[dict]:
    with db() as conn:
        rows = conn.execute(
            "SELECT id, name, ext, size, pages, created_at, status, n_chunks, n_chars "
            "FROM documents ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def delete_document(doc_id: str) -> bool:
    with db() as conn:
        cur = conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        return cur.rowcount > 0


def doc_stats() -> dict:
    with db() as conn:
        d = conn.execute("SELECT COUNT(*) c, COALESCE(SUM(n_chars),0) ch FROM documents").fetchone()
        k = conn.execute("SELECT COUNT(*) c FROM chunks").fetchone()
        e = conn.execute("SELECT COUNT(*) c FROM embeddings").fetchone()
    return {"documents": d["c"], "chunks": k["c"], "embedded": e["c"], "chars": d["ch"]}


def all_chunks() -> list[dict]:
    with db() as conn:
        rows = conn.execute(
            "SELECT c.id, c.doc_id, c.ordinal, c.page, c.heading, c.text, d.name AS doc_name, d.ext "
            "FROM chunks c JOIN documents d ON d.id = c.doc_id "
            "ORDER BY c.doc_id, c.ordinal"
        ).fetchall()
    return [dict(r) for r in rows]


def all_embeddings() -> tuple[list[int], np.ndarray | None]:
    with db() as conn:
        rows = conn.execute("SELECT chunk_id, dim, vec FROM embeddings").fetchall()
    if not rows:
        return [], None
    ids = [r["chunk_id"] for r in rows]
    dim = rows[0]["dim"]
    mat = np.zeros((len(rows), dim), dtype=np.float32)
    for i, r in enumerate(rows):
        mat[i] = np.frombuffer(r["vec"], dtype=np.float32)
    return ids, mat


# -------------------------------------------------------------- conversations
def create_conversation(conv_id: str | None = None, title: str = "New chat") -> dict:
    cid = conv_id or uuid.uuid4().hex[:12]
    now = time.time()
    with db() as conn:
        conn.execute(
            "INSERT INTO conversations(id, title, created_at, updated_at) VALUES(?,?,?,?)",
            (cid, title, now, now),
        )
    return {"id": cid, "title": title, "created_at": now, "updated_at": now}


def list_conversations() -> list[dict]:
    with db() as conn:
        rows = conn.execute(
            "SELECT c.id, c.title, c.created_at, c.updated_at, "
            "(SELECT COUNT(*) FROM messages m WHERE m.conv_id = c.id) AS n_messages "
            "FROM conversations c ORDER BY c.updated_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_conversation(conv_id: str) -> dict | None:
    with db() as conn:
        row = conn.execute("SELECT * FROM conversations WHERE id = ?", (conv_id,)).fetchone()
    return dict(row) if row else None


def touch_conversation(conv_id: str, title: str | None = None) -> None:
    with db() as conn:
        if title:
            conn.execute("UPDATE conversations SET updated_at = ?, title = ? WHERE id = ?",
                         (time.time(), title, conv_id))
        else:
            conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?",
                         (time.time(), conv_id))


def delete_conversation(conv_id: str) -> None:
    with db() as conn:
        conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))


def add_message(conv_id: str, role: str, content: str,
                sources: list | None = None, meta: dict | None = None) -> int:
    with db() as conn:
        cur = conn.execute(
            "INSERT INTO messages(conv_id, role, content, sources, meta, created_at) "
            "VALUES(?,?,?,?,?,?)",
            (conv_id, role, content, json.dumps(sources or []),
             json.dumps(meta or {}), time.time()),
        )
    return int(cur.lastrowid)


def get_messages(conv_id: str, limit: int | None = None) -> list[dict]:
    sql = "SELECT * FROM messages WHERE conv_id = ? ORDER BY id ASC"
    with db() as conn:
        rows = conn.execute(sql, (conv_id,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["sources"] = json.loads(d["sources"])
            d["meta"] = json.loads(d["meta"] or "{}")
        except Exception:
            d["sources"], d["meta"] = [], {}
        out.append(d)
    return out[-limit:] if limit else out


def update_conversation_title(conv_id: str, title: str) -> None:
    with db() as conn:
        conn.execute("UPDATE conversations SET title = ? WHERE id = ?", (title[:80], conv_id))
