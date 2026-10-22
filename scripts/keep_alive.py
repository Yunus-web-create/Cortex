"""Keep a public URL for Cortex alive, and always know which one is live.

Quick tunnels drop silently: the process keeps running, the edge returns 1033/530,
and the address in someone's bookmarks is dead. This watchdog probes the live URL
every 45 s and, after two consecutive failures, restarts the tunnel and republishes
the new address to .run/url.txt and ../../cortex-link.txt.

    python scripts/keep_alive.py            # foreground (use start_process / nohup)
"""
from __future__ import annotations

import subprocess
import sys
import time
import os
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / ".run"
LOG = RUN / "watchdog.log"
LINK = ROOT.parent / "cortex-link.txt"
BIN = RUN / "bin" / "cloudflared"
PORT = 8010
CHECK_EVERY = 45


def log(message: str) -> None:
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {message}"
    print(line, flush=True)
    with LOG.open("a") as fh:
        fh.write(line + "\n")


def current_url() -> str:
    try:
        return (RUN / "url.txt").read_text().strip()
    except OSError:
        return ""


def publish(url: str) -> None:
    (RUN / "url.txt").write_text(url + "\n")
    LINK.write_text(
        "Cortex public address\n"
        f"{url}\n"
        "password: see the .gate file in the cortex folder\n"
        f"updated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    )
    log(f"published {url}")


def alive(url: str) -> bool:
    if not url:
        return False
    try:
        r = httpx.get(url + "/api/health", timeout=12, follow_redirects=False)
        return r.status_code in (200, 401)      # 401 = gate is up, password not sent
    except Exception:
        return False


def stop_tunnel() -> None:
    pid_file = RUN / "cloudflared.pid"
    if pid_file.exists():
        try:
            subprocess.run(["kill", pid_file.read_text().strip()], check=False)
        except Exception:
            pass
        pid_file.unlink(missing_ok=True)
    subprocess.run(["pkill", "-f", "cloudflared tunnel --url"], check=False)
    time.sleep(2)


def pid_alive() -> bool:
    """Is the tunnel process we recorded still running?"""
    try:
        pid = int((RUN / "cloudflared.pid").read_text().strip())
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def start_tunnel() -> subprocess.Popen:
    logfile = RUN / "tunnel.log"
    logfile.write_text("")                       # a fresh file, so the URL is unambiguous
    proc = subprocess.Popen(
        [str(BIN), "tunnel", "--url", f"http://127.0.0.1:{PORT}", "--no-autoupdate",
         "--protocol", "http2", "--logfile", str(logfile)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    (RUN / "cloudflared.pid").write_text(str(proc.pid))
    return proc


def wait_for_url(timeout: float = 60) -> str:
    deadline = time.time() + timeout
    logfile = RUN / "tunnel.log"
    while time.time() < deadline:
        try:
            text = logfile.read_text(errors="ignore")
        except OSError:
            text = ""
        for token in reversed(text.replace('"', " ").split()):
            if token.startswith("https://") and ".trycloudflare.com" in token:
                return token.strip()
        time.sleep(1)
    return ""


def main() -> int:
    RUN.mkdir(exist_ok=True)
    if not BIN.exists():
        log(f"missing {BIN} — run scripts/share.sh once to download cloudflared")
        return 1
    failures = 0
    while True:
        url = current_url()
        if not pid_alive() or not alive(url):
            failures += 1
            if failures >= 2 or not url:
                log(f"tunnel unhealthy (url={url or 'none'}) — restarting")
                stop_tunnel()
                start_tunnel()
                new_url = wait_for_url()
                if new_url:
                    publish(new_url)
                    failures = 0
                else:
                    log("no address yet; retrying")
                    failures = 0
        else:
            failures = 0
        time.sleep(CHECK_EVERY)


if __name__ == "__main__":
    sys.exit(main())
