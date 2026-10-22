#!/usr/bin/env bash
# Install a local language model so Cortex keeps answering with no key, no
# network and no bill — the "unlimited" fallback.
#
#   ./scripts/install_local_model.sh            # pick a model for this machine
#   ./scripts/install_local_model.sh --small    # force the 0.5B model
#   ./scripts/install_local_model.sh --large    # force the 3B model (needs ~4 GB RAM)
#   ./scripts/install_local_model.sh --run      # install, then start the server
#
# Everything lands in ~/.local/share/cortex/local (~25 MB of binaries + a GGUF).
set -euo pipefail

DEST="${CORTEX_LOCAL_DIR:-$HOME/.local/share/cortex/local}"
PORT="${CORTEX_LOCAL_PORT:-8080}"
SIZE="auto"

for arg in "$@"; do
  case "$arg" in
    --small) SIZE="small" ;;
    --large) SIZE="large" ;;
    --run) RUN=1 ;;
    --port) shift; PORT="$1" ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
  esac
done

say() { printf '\033[1m▸\033[0m %s\n' "$*"; }
die() { printf '\033[31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

command -v curl >/dev/null || die "curl is required"
mkdir -p "$DEST"

# ── 1. pick a model this machine can actually run ────────────────────────────
if [ "$SIZE" = "auto" ]; then
  if [ -r /proc/meminfo ]; then
    MEM_MB=$(awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo)
  else
    MEM_MB=$(( $(sysctl -n hw.memsize 2>/dev/null || echo 4294967296) / 1048576 ))
  fi
  if   [ "$MEM_MB" -lt 3800 ]; then SIZE="small"
  elif [ "$MEM_MB" -lt 9000 ]; then SIZE="mid"
  else                              SIZE="large"; fi
  say "detected ${MEM_MB} MB of RAM → ${SIZE} model"
fi

case "$SIZE" in
  small) HF="Qwen/Qwen2.5-0.5B-Instruct-GGUF"; FILE="qwen2.5-0.5b-instruct-q4_k_m.gguf"; NEED=1 ;;
  mid)   HF="Qwen/Qwen2.5-1.5B-Instruct-GGUF"; FILE="qwen2.5-1.5b-instruct-q4_k_m.gguf"; NEED=2 ;;
  large) HF="Qwen/Qwen2.5-3B-Instruct-GGUF";   FILE="qwen2.5-3b-instruct-q4_k_m.gguf";   NEED=4 ;;
  *) die "unknown size: $SIZE" ;;
esac

# ── 2. llama.cpp binaries (no compiler, no root) ─────────────────────────────
BIN="$DEST/llama-server"
if [ ! -x "$BIN" ]; then
  case "$(uname -s)" in
    Linux)  OS="ubuntu" ;;
    Darwin) OS="macos" ;;
    *) die "unsupported OS $(uname -s) — on Windows use Ollama and set CORTEX_PROVIDER=ollama" ;;
  esac
  case "$(uname -m)" in
    x86_64|amd64) ARCH="x64" ;;
    aarch64|arm64) ARCH="arm64" ;;
    *) die "unsupported architecture $(uname -m)" ;;
  esac

  say "looking up the latest llama.cpp release"
  API="https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=20"
  URL=$(curl -fsSL "$API" | python3 -c '
import json, sys
want = sys.argv[1]
for rel in json.load(sys.stdin):
    for a in rel.get("assets", []):
        n = a["name"]
        if want in n and "bin" in n and "cudart" not in n and not any(x in n for x in ("cuda", "rocm", "vulkan", "sycl", "openvino")):
            print(a["browser_download_url"]); raise SystemExit
' "$OS-$ARCH") || true
  [ -n "${URL:-}" ] || die "could not find a prebuilt llama.cpp build for $OS-$ARCH"

  say "downloading $(basename "$URL")"
  TMP=$(mktemp -d)
  curl -fL --progress-bar "$URL" -o "$TMP/llama.tar.gz"
  mkdir -p "$TMP/x"
  tar xzf "$TMP/llama.tar.gz" -C "$TMP/x"
  SERVER=$(find "$TMP/x" -name llama-server -type f | head -1)
  [ -n "$SERVER" ] || die "llama-server was not in the archive"
  cp -a "$(dirname "$SERVER")/." "$DEST/"
  rm -rf "$TMP"
  chmod +x "$DEST/llama-server"
  say "llama.cpp installed in $DEST"
else
  say "llama.cpp already installed"
fi

# ── 3. the model file ────────────────────────────────────────────────────────
if [ ! -f "$DEST/$FILE" ]; then
  say "downloading $FILE (about $( [ "$NEED" = 1 ] && echo 0.5 || echo "$NEED" ) GB of RAM to run)"
  curl -fL --progress-bar "https://huggingface.co/$HF/resolve/main/$FILE?download=true" \
       -o "$DEST/$FILE.part"
  mv "$DEST/$FILE.part" "$DEST/$FILE"
else
  say "$FILE already present"
fi

cat > "$DEST/start.sh" <<EOF
#!/usr/bin/env bash
# Start the local model Cortex falls back to. Ctrl-C to stop.
exec "$DEST/llama-server" -m "$DEST/$FILE" --port $PORT --host 127.0.0.1 -c 4096 -t \$(nproc 2>/dev/null || echo 4)
EOF
chmod +x "$DEST/start.sh"

say "installed. Start it with:  $DEST/start.sh"
say "Cortex detects it automatically on the next message (http://127.0.0.1:$PORT/v1)."
say "Stop it any time — remote free routes take over again."

if [ "${RUN:-0}" = "1" ]; then
  say "starting the server"
  exec "$DEST/start.sh"
fi
