#!/usr/bin/env bash
# Keep a public tunnel up, and record its current address in .run/url.txt.
#
#   ./scripts/watch_tunnel.sh
#
# cloudflared quick tunnels drop without warning (the process exits and the
# hostname stops resolving). This wrapper restarts it and republishes the new
# address, so `./scripts/share.sh` always shows the truth.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p .run

PORT=${GATE_PORT:-8010}
BIN=${CLOUDFLARED:-cloudflared}
if ! command -v "$BIN" >/dev/null 2>&1; then
  BIN=$(command -v .run/bin/cloudflared || true)
fi
if [ -z "${BIN:-}" ] || [ ! -x "$BIN" ]; then
  echo "cloudflared not found — run ./scripts/share.sh once to download it." >&2
  exit 1
fi

trap 'kill 0' INT TERM
while true; do
  echo "[tunnel] starting $(date -u +%FT%TZ)"
  "$BIN" tunnel --url "http://127.0.0.1:${PORT}" --no-autoupdate --protocol http2 \
    --logfile .run/tunnel.log &
  PID=$!
  URL=""
  for _ in $(seq 1 45); do
    URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' .run/tunnel.log 2>/dev/null | tail -1)
    [ -n "$URL" ] && break
    sleep 1
  done
  if [ -n "$URL" ]; then
    printf '%s\n' "$URL" > .run/url.txt
    echo "[tunnel] $URL"
  else
    echo "[tunnel] no address yet — see .run/tunnel.log"
  fi
  wait "$PID"
  echo "[tunnel] exited, restarting in 3s"
  sleep 3
done
