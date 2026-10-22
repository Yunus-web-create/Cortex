"""Per-conversation code workspace: files a model can write, run and ship.

Jailed to DATA_DIR/workspaces/<conversation>. Commands run with a scrubbed
environment (no API keys), a timeout, and an allowlist of interpreters.
"""
from __future__ import annotations

import asyncio
import os
import re
import shutil
import zipfile
from pathlib import Path

from . import config

ROOT = Path(config.DATA_DIR) / "workspaces"
ROOT.mkdir(parents=True, exist_ok=True)

MAX_BYTES = int(os.getenv("CORTEX_WS_MAX_BYTES", str(2 * 1024 * 1024)))
MAX_FILES = int(os.getenv("CORTEX_WS_MAX_FILES", "200"))
RUN_TIMEOUT = float(os.getenv("CORTEX_WS_TIMEOUT", "60"))
OUTPUT_LIMIT = int(os.getenv("CORTEX_WS_OUTPUT", "20000"))

ALLOWED_BINS = {"python", "python3", "node", "npm", "npx", "pytest", "deno", "bun", "tsc"}

SAFE_NAME = re.compile(r"^[A-Za-z0-9._\-/ ]+$")


class WorkspaceError(RuntimeError):
    pass


def slug(conv_id: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_-]", "", conv_id or "")[:40]
    return clean or "scratch"


def root(conv_id: str) -> Path:
    d = ROOT / slug(conv_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def resolve(conv_id: str, rel: str) -> Path:
    rel = (rel or "").strip().replace("\\", "/")
    if rel.startswith("/") or rel.startswith("~") or ".." in Path(rel).parts:
        raise WorkspaceError("path must stay inside the workspace")
    if not SAFE_NAME.match(rel):
        raise WorkspaceError("filename may only contain letters, numbers, dot, dash, underscore, slash or space")
    target = (root(conv_id) / rel).resolve()
    base = root(conv_id).resolve()
    if base not in target.parents and target != base:
        raise WorkspaceError("path escapes the workspace")
    return target


def write(conv_id: str, rel: str, content: str) -> dict:
    path = resolve(conv_id, rel)
    data = content.encode("utf-8") if isinstance(content, str) else content
    if len(data) > MAX_BYTES:
        raise WorkspaceError(f"file is larger than {MAX_BYTES // 1024} KB")
    files = [p for p in root(conv_id).rglob("*") if p.is_file()]
    if len(files) >= MAX_FILES and path not in files:
        raise WorkspaceError(f"workspace already holds {MAX_FILES} files")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"path": rel, "bytes": len(data), "lines": content.count("\n") + 1 if isinstance(content, str) else 0}


def read(conv_id: str, rel: str, limit: int = 60000) -> str:
    path = resolve(conv_id, rel)
    if not path.is_file():
        raise WorkspaceError(f"no such file: {rel}")
    return path.read_text("utf-8", errors="replace")[:limit]


def delete(conv_id: str, rel: str) -> None:
    path = resolve(conv_id, rel)
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def listing(conv_id: str) -> list[dict]:
    base = root(conv_id)
    out = []
    for p in sorted(base.rglob("*")):
        if p.is_file() and not any(part.startswith(".") and part in (".git", "__pycache__")
                                   for part in p.parts):
            rel = p.relative_to(base).as_posix()
            out.append({"path": rel, "bytes": p.stat().st_size,
                        "lines": _count_lines(p), "kind": _kind(p)})
    return out[:MAX_FILES]


def _count_lines(p: Path) -> int:
    try:
        if p.stat().st_size > 400_000:
            return 0
        return p.read_text("utf-8", errors="ignore").count("\n") + 1
    except Exception:
        return 0


def _kind(p: Path) -> str:
    ext = p.suffix.lower()
    return {".py": "python", ".js": "javascript", ".ts": "typescript", ".tsx": "react",
            ".jsx": "react", ".html": "html", ".css": "css", ".json": "json",
            ".md": "markdown", ".sh": "shell", ".csv": "data", ".txt": "text",
            ".png": "image", ".jpg": "image", ".mp3": "audio", ".mp4": "video"}.get(ext, "file")


async def run(conv_id: str, command: str, *, timeout: float | None = None) -> dict:
    """Run a shell-free command (argv split) inside the workspace."""
    argv = _parse(command)
    if not argv:
        raise WorkspaceError("empty command")
    if argv[0] not in ALLOWED_BINS:
        raise WorkspaceError(
            f"'{argv[0]}' is not allowed. Use one of: {', '.join(sorted(ALLOWED_BINS))}")
    env = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": str(root(conv_id)),
        "LANG": "C.UTF-8",
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "NODE_PATH": str(root(conv_id) / "node_modules"),
    }
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv, cwd=str(root(conv_id)), env=env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    except FileNotFoundError:
        return {"command": command, "exit": 127, "stdout": "", "stderr": f"{argv[0]} is not installed",
                "ms": 0}
    started = asyncio.get_event_loop().time()
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout or RUN_TIMEOUT)
        code = proc.returncode
    except asyncio.TimeoutError:
        proc.kill()
        out, err = await proc.communicate()
        code = -9
    elapsed = int((asyncio.get_event_loop().time() - started) * 1000)
    return {
        "command": command,
        "exit": code,
        "stdout": out.decode("utf-8", "replace")[:OUTPUT_LIMIT],
        "stderr": err.decode("utf-8", "replace")[:OUTPUT_LIMIT],
        "ms": elapsed,
        "timed_out": code == -9,
    }


def _parse(command: str) -> list[str]:
    """Split on whitespace but honour quotes; no shell expansion, no pipes."""
    import shlex
    try:
        return shlex.split(command)
    except ValueError as exc:
        raise WorkspaceError(f"could not parse command: {exc}")


def bundle(conv_id: str) -> Path:
    """Zip the workspace for download."""
    base = root(conv_id)
    target = Path(config.DATA_DIR) / "exports" / f"{slug(conv_id)}.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
        for p in base.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(base).as_posix())
    return target


def tree_text(conv_id: str) -> str:
    files = listing(conv_id)
    if not files:
        return "(workspace is empty)"
    return "\n".join(f"{f['path']}  ({f['bytes']} B)" for f in files)
