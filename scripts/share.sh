#!/usr/bin/env bash
# Put Cortex on a public https URL for your laptop and phone.
#
#   ./scripts/share.sh              # start (or report) the public link
#   ./scripts/share.sh --stop       # close it again
#
# Two moving parts, both optional and both local:
#   scripts/public_gate.py  : password + reverse proxy in front of the app (port 8010)
#   cloudflared             : the tunnel that gives that port a public https name
#
# The password is generated once and kept in .gate (chmod 600). Delete that file to
# pick a new one. Nothing here is required for the app to work locally.
set -euo pipefail
cd "$(dirname "$0")/.."

APP_PORT=${APP_PORT:-8000}
GATE_PORT=${GATE_PORT:-8010}
RUN_DIR=${CORTEX_RUN_DIR:-.run}
mkdir -p "$RUN_DIR"

stop_all() {
  for f in "$RUN_DIR"/gate.pid "$RUN_DIR"/tunnel.pid; do
    if [ -f "$f" ]; then kill "$(cat "$f")" 2>/dev/null || true; rm -f "$f"; fi
  done
  echo "public link closed"
}

if [ "${1:-}" = "--stop" ]; then stop_all; exit 0; fi

python3 - <<'PY'
import secrets, string
from pathlib import Path
p = Path(".gate")
if not p.exists():
    words = ["harbor", "copper", "lantern", "meadow", "quartz", "sable", "tundra",
             "violet", "willow", "zephyr", "ember", "basalt", "cobalt", "juniper"]
    pw = "-".join(secrets.choice(words) for _ in range(3)) + "-" + "".join(
        secrets.choice(string.digits) for _ in range(2))
    p.write_text(pw + "\n")
    p.chmod(0o600)
    print("generated a new access password in .gate")
PY
PASSWORD=$(cat .gate)

if ! curl -s -o /dev/null -m 3 "http://127.0.0.1:${APP_PORT}/api/health"; then
  echo "Cortex is not answering on :${APP_PORT} — start it first (python run.py)." >&2
  exit 1
fi

if ! curl -s -o /dev/null -m 3 "http://127.0.0.1:${GATE_PORT}/"; then
  CORTEX_GATE_PASSWORD="$PASSWORD" nohup python3 scripts/public_gate.py --port "$GATE_PORT" \
    > "$RUN_DIR/gate.log" 2>&1 &
  echo $! > "$RUN_DIR/gate.pid"
  sleep 2
fi

if [ -f "$RUN_DIR/tunnel.pid" ] && kill -0 "$(cat "$RUN_DIR/tunnel.pid")" 2>/dev/null; then
  URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$RUN_DIR/tunnel.log" | tail -1 || true)
else
  BIN=${CLOUDFLARED:-cloudflared}
  if ! command -v "$BIN" >/dev/null 2>&1; then
    mkdir -p "$RUN_DIR/bin"
    BIN="$RUN_DIR/bin/cloudflared"
    if [ ! -x "$BIN" ]; then
      echo "downloading cloudflared…"
      curl -sL -o "$BIN" https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
      chmod +x "$BIN"
    fi
  fi
  nohup "$BIN" tunnel --url "http://127.0.0.1:${GATE_PORT}" --no-autoupdate \
    > "$RUN_DIR/tunnel.log" 2>&1 &
  echo $! > "$RUN_DIR/tunnel.pid"
  for _ in $(seq 1 30); do
    URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$RUN_DIR/tunnel.log" | tail -1 || true)
    [ -n "$URL" ] && break
    sleep 1
  done
fi

if [ -z "${URL:-}" ]; then
  echo "the tunnel did not come up — see $RUN_DIR/tunnel.log" >&2
  exit 1
fi

echo
echo "  Cortex is live at:  $URL"
echo "  Username:           cortex  (anything works)"
echo "  Password:           $PASSWORD"
echo
python3 - "$URL" <<'PY' || true
import sys
try:
    import qrcode
except ImportError:
    sys.exit(0)
qr = qrcode.QRCode(box_size=10, border=2, error_correction=qrcode.constants.ERROR_CORRECT_M)
qr.add_data(sys.argv[1]); qr.make(fit=True)
qr.make_image(fill_color="#111827", back_color="white").save("access-qr.png")
print("  QR code:            access-qr.png (scan it with your phone)")
PY
echo
echo "  Keep this running while you use it; './scripts/share.sh --stop' closes it."
