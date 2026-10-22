"""Central configuration for Cortex."""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DEMO_DIR = BASE_DIR / "demo_docs"
DATA_DIR = Path(os.getenv("CORTEX_DATA_DIR", BASE_DIR / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = Path(os.getenv("CORTEX_DB", DATA_DIR / "cortex.db"))


def _load_dotenv() -> None:
    """Tiny .env loader (no dependency needed)."""
    for candidate in (BASE_DIR / ".env", Path.cwd() / ".env"):
        if not candidate.exists():
            continue
        try:
            for raw in candidate.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
        except Exception:
            pass


_load_dotenv()

# ---- LLM -------------------------------------------------------------------
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
DEFAULT_MODEL = os.getenv("CORTEX_MODEL", "claude-sonnet-5").strip()
FAST_MODEL = os.getenv("CORTEX_FAST_MODEL", "claude-haiku-4-5").strip()
MAX_TOKENS = int(os.getenv("CORTEX_MAX_TOKENS", "1600"))

AVAILABLE_MODELS = [
    {"id": "claude-sonnet-5", "label": "Claude Sonnet 5", "hint": "Best balance of speed & quality"},
    {"id": "claude-haiku-4-5", "label": "Claude Haiku 4.5", "hint": "Fastest & cheapest"},
    {"id": "claude-opus-5", "label": "Claude Opus 5", "hint": "Highest quality, slower"},
    {"id": "claude-sonnet-4-6", "label": "Claude Sonnet 4.6", "hint": "Previous generation"},
]

# ---- Retrieval -------------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CORTEX_CHUNK_SIZE", "850"))       # characters
CHUNK_OVERLAP = int(os.getenv("CORTEX_CHUNK_OVERLAP", "160"))  # characters
EMBED_MODEL = os.getenv("CORTEX_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
CANDIDATES_PER_RETRIEVER = int(os.getenv("CORTEX_CANDIDATES", "40"))
FUSED_POOL = int(os.getenv("CORTEX_FUSED_POOL", "14"))
TOP_K = int(os.getenv("CORTEX_TOP_K", "8"))
MAX_CONTEXT_CHARS = int(os.getenv("CORTEX_MAX_CONTEXT_CHARS", "15000"))
RRF_K = 60
MMR_LAMBDA = 0.72
# ---- Reranking -------------------------------------------------------------
RERANK_ENABLED = os.getenv("CORTEX_RERANK", "1") not in ("0", "false", "False", "off")
RERANK_MODEL = os.getenv("CORTEX_RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
RERANK_POOL = int(os.getenv("CORTEX_RERANK_POOL", "10"))
RERANK_MAX_CHARS = int(os.getenv("CORTEX_RERANK_MAX_CHARS", "600"))
RERANK_THREADS = int(os.getenv("CORTEX_RERANK_THREADS", str(min(4, os.cpu_count() or 4))))
PRIOR_WEIGHT = float(os.getenv("CORTEX_PRIOR_WEIGHT", "0.35"))        # lexical ranking
PRIOR_WEIGHT_DENSE = float(os.getenv("CORTEX_PRIOR_WEIGHT_DENSE", "0.06"))  # dense ranking
MIN_RELEVANCE = float(os.getenv("CORTEX_MIN_RELEVANCE", "0.18"))

# ---- Adaptive retrieval ----------------------------------------------------
ADAPTIVE = os.getenv("CORTEX_ADAPTIVE", "1") not in ("0", "false", "off")
ADAPTIVE_CONFIDENCE = float(os.getenv("CORTEX_ADAPTIVE_CONFIDENCE", "0.55"))
MAX_EXPANSIONS = int(os.getenv("CORTEX_MAX_EXPANSIONS", "2"))
GROUNDING_MIN_OVERLAP = float(os.getenv("CORTEX_GROUNDING_OVERLAP", "0.42"))

# ---- Tools -----------------------------------------------------------------
TOOL_MAX_ROUNDS = int(os.getenv("CORTEX_TOOL_ROUNDS", "3"))

# ---- App -------------------------------------------------------------------
MAX_UPLOAD_MB = int(os.getenv("CORTEX_MAX_UPLOAD_MB", "25"))
ALLOWED_EXTENSIONS = {
    ".pdf", ".docx", ".txt", ".md", ".markdown", ".csv",
    ".html", ".htm", ".json", ".log",
}
HISTORY_TURNS = 6  # messages of conversational memory sent to the model


def anthropic_key() -> str:
    """Runtime key: env var or the key entered in the UI (in-memory)."""
    return os.getenv("CORTEX_RUNTIME_KEY", "").strip() or ANTHROPIC_API_KEY
