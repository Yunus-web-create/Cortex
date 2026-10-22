"""Latency guard test — proves a stalled free route cannot hang a turn.

A tiny OpenAI-compatible server runs in-process with three behaviours:

  /stall   accepts the request, sends a role delta, then nothing (what a queued
           free endpoint looks like)
  /ok      streams a normal answer
  /quota   returns 429 with X-RateLimit-Reset an hour ahead (a daily cap)

and the checks assert that

  * a stalled stream is abandoned at the patience budget, not at the HTTP timeout,
  * the error is retryable and names the model,
  * that model is parked so the router stops choosing it,
  * a 429 carries the reset window back so the provider can be parked for hours,
  * a healthy stream is untouched (no false positives).

Usage:  python tests/speed_test.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TMP = Path(tempfile.mkdtemp(prefix="cortex_speed_"))
os.environ["CORTEX_DATA_DIR"] = str(TMP)
os.environ["CORTEX_DB"] = str(TMP / "test.db")
os.environ["CORTEX_RERANK"] = "0"
os.environ["CORTEX_FREE_MODEL"] = "0"
os.environ["CORTEX_LOCAL_MODEL"] = "0"

from app import db, gateway  # noqa: E402

db.init()

RESET_AT = int(time.time()) + 3600
STALL_SECONDS = 6.0          # how long the fake queue sits there
PATIENCE = 1.5               # what the test asks chat_stream to wait

PASS = FAIL = 0
GREEN, RED, OFF = "\033[32m", "\033[31m", "\033[0m"


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  {GREEN}✓{OFF} {name}")
    else:
        FAIL += 1
        print(f"  {RED}✗{OFF} {name} {extra}")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("content-length") or 0)
        if length:
            self.rfile.read(length)
        if self.path.startswith("/stall"):
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("transfer-encoding", "chunked")
            self.end_headers()
            self._chunk(b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n')
            time.sleep(STALL_SECONDS)          # the queue
            try:
                self._chunk(b"data: [DONE]\n\n")
                self._chunk(b"", final=True)
            except Exception:
                pass
            return
        if self.path.startswith("/quota"):
            body = json.dumps({"error": {"message": "Rate limit exceeded: free-models-per-day",
                                         "code": 429}}).encode()
            self.send_response(429)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.send_header("x-ratelimit-reset", str(RESET_AT))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("transfer-encoding", "chunked")
        self.end_headers()
        for piece in ("Hello", " there", "."):
            payload = {"choices": [{"delta": {"content": piece}}]}
            self._chunk(f"data: {json.dumps(payload)}\n\n".encode())
            time.sleep(0.05)
        self._chunk(b'data: {"usage":{"prompt_tokens":11,"completion_tokens":3}}\n\n')
        self._chunk(b"data: [DONE]\n\n")
        self._chunk(b"", final=True)

    def _chunk(self, payload: bytes, final: bool = False):
        if final:
            self.wfile.write(b"0\r\n\r\n")
        else:
            self.wfile.write(f"{len(payload):x}\r\n".encode() + payload + b"\r\n")
        self.wfile.flush()


def serve() -> tuple[ThreadingHTTPServer, int]:
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def provider(port: int, path: str) -> gateway.Provider:
    p = gateway.Provider(name="test", base_url=f"http://127.0.0.1:{port}", api_key="k",
                         chat_path=path)
    return p


async def collect(p: gateway.Provider, patience: float | None):
    token = gateway.push_provider(p)
    out = []
    try:
        async for ev in gateway.chat_stream("test/slow-model", [{"role": "user", "content": "hi"}],
                                            max_tokens=50, first_token=patience):
            out.append(ev)
    finally:
        gateway.pop_provider(token)
    return out


def main() -> int:
    srv, port = serve()
    print(f"fake provider on 127.0.0.1:{port}\n")

    print("\033[1mstalled stream\033[0m")
    t0 = time.monotonic()
    events = asyncio.run(collect(provider(port, "/stall"), PATIENCE))
    elapsed = time.monotonic() - t0
    errs = [e for e in events if e["type"] == "error"]
    check("gave up instead of hanging", elapsed < STALL_SECONDS - 1, f"{elapsed:.1f}s")
    check("gave up close to the patience budget", PATIENCE * 0.8 <= elapsed <= PATIENCE + 2.5,
          f"{elapsed:.1f}s for {PATIENCE}s budget")
    check("reported an error", bool(errs), str(events))
    check("the error is retryable so the turn fails over", bool(errs) and errs[0].get("retry") is True)
    check("the error names the model", bool(errs) and "slow-model" in errs[0]["message"],
          errs[0]["message"] if errs else "")
    check("no content leaked from the stalled stream",
          not any(e["type"] == "delta" and e.get("text") for e in events))
    check("the slow model is parked for the router",
          "slow-model" in gateway.model_cooldowns(), str(gateway.model_cooldowns()))

    print("\n\033[1mdaily quota 429\033[0m")
    events = asyncio.run(collect(provider(port, "/quota"), 5))
    errs = [e for e in events if e["type"] == "error"]
    check("quota failure is reported", bool(errs), str(events))
    park = float(errs[0].get("park") or 0) if errs else 0
    check("the reset window is carried back to the caller", 3500 < park < 3700, f"{park}")
    check("the message says how long", errs and "resets in" in errs[0]["message"],
          errs[0]["message"] if errs else "")
    check("a long park is distinguishable from a busy minute", park > 120)

    print("\n\033[1mhealthy stream\033[0m")
    t0 = time.monotonic()
    events = asyncio.run(collect(provider(port, "/ok"), PATIENCE))
    text = "".join(e.get("text", "") for e in events if e["type"] == "delta")
    usage = [e for e in events if e["type"] == "usage"]
    check("answer arrives intact", text == "Hello there.", repr(text))
    check("no false-positive timeout", not any(e["type"] == "error" for e in events), str(events))
    check("usage is reported", bool(usage) and usage[-1]["input_tokens"] == 11)
    check("finished quickly", time.monotonic() - t0 < 3, f"{time.monotonic() - t0:.1f}s")

    print("\n\033[1mpre-answer work is bounded\033[0m")
    # Reading six web pages one at a time used to put 15-25 s in front of every
    # factual question. They are independent, so they must run together under a
    # budget, and a page that never arrives must not hold the turn.
    from app import web

    async def slow_fetch(url, limit=0):
        await asyncio.sleep(30)
        return {"url": url, "text": "x" * 900}

    real_search, real_fetch = web.search, web.fetch
    web.search = lambda q, n=6: asyncio.sleep(0, result=[
        {"url": f"https://example.com/{i}", "title": "anything at all",
         "snippet": "anything at all matters here " * 8}
        for i in range(6)])
    web.fetch = slow_fetch
    try:
        t0 = time.monotonic()
        bundle = asyncio.run(web.research("anything", depth=3, n=6, budget=1.0))
        took = time.monotonic() - t0
    finally:
        web.search, web.fetch = real_search, real_fetch
    check("a stalled page read cannot hold research open", took < 4.0, f"{took:.1f}s")
    check("research still returns the search hits", len(bundle.get("results") or []) == 6,
          str(len(bundle.get("results") or [])))

    # With no documents there is nothing to expand: the extra model round-trip
    # was pure delay, so it must not run.
    from app import main as app_main, quality
    called = {"n": 0}

    async def boom(*a, **k):
        called["n"] += 1
        return ["should not happen"]

    real_expand = quality.expand_queries
    quality.expand_queries = boom
    try:
        asyncio.run(app_main._retrieve("how do barometers work", [{"role": "user", "content": "hi"}]))
    finally:
        quality.expand_queries = real_expand
    check("no query expansion when there is no corpus", called["n"] == 0, f"called {called['n']}x")

    print("\n\033[1mturn budget\033[0m")
    from app import agent
    check("patience is per task, not one global number",
          agent.PATIENCE.get("reasoning", 0) > agent.PATIENCE.get("code", 0) > 0,
          str(agent.PATIENCE))
    check("a turn has a wall-clock budget", agent.TURN_BUDGET > 0, str(agent.TURN_BUDGET))
    check("budget is shorter than the patience of a single stalled round times three",
          agent.TURN_BUDGET < 3 * agent.PATIENCE["reasoning"])

    srv.shutdown()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
