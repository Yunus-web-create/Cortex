"""Password gate + reverse proxy for exposing Cortex beyond the sandbox.

The Arena preview on :8000 stays open (it is already protected by the platform's
traffic token, and the in-app iframe cannot answer a Basic-auth prompt). This
process listens on 127.0.0.1:8010, demands a password, and streams everything —
including server-sent events — to and from the app on :8000. Point a tunnel at
this port and the public URL is protected.

    CORTEX_GATE_PASSWORD=... python scripts/public_gate.py --port 8010

Auth: HTTP Basic, any username, password = CORTEX_GATE_PASSWORD. Browsers cache
it for the session, so phones and laptops ask once. `?k=<token>` with the same
password also works for tools that cannot send headers.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import os
import sys
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

UPSTREAM = os.getenv("CORTEX_GATE_UPSTREAM", "http://127.0.0.1:8000")
PASSWORD = os.getenv("CORTEX_GATE_PASSWORD", "").strip()

COOKIE = "cortex_access"


def _cookie_token() -> str:
    """A stable, non-reversible token so a phone can stay signed in."""
    return hmac.new(b"cortex-gate-v1", PASSWORD.encode(), hashlib.sha256).hexdigest()


HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
    "transfer-encoding", "upgrade", "host", "content-length", "authorization", "cookie",
}
DROP_FROM_UPSTREAM = {"content-length", "content-encoding", "transfer-encoding", "connection"}

app = FastAPI(title="Cortex access gate", docs_url=None, redoc_url=None)
client = httpx.AsyncClient(base_url=UPSTREAM, timeout=httpx.Timeout(None, connect=10.0))


def _cookie_ok(request: Request) -> bool:
    value = request.cookies.get(COOKIE, "")
    return bool(value) and hmac.compare_digest(value, _cookie_token())


def _authorised(request: Request) -> bool:
    if not PASSWORD:
        return True
    if _cookie_ok(request):
        return True
    token = request.query_params.get("k")
    if token and hmac.compare_digest(token, PASSWORD):
        return True
    header = request.headers.get("authorization", "")
    if header.lower().startswith("basic "):
        try:
            _, _, pw = base64.b64decode(header[6:]).decode().partition(":")
            # accept "anything:password" or "password:" so both conventions work
            if hmac.compare_digest(pw or _, PASSWORD):
                return True
        except Exception:
            pass
    return False


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"])
async def proxy(path: str, request: Request):
    if not _authorised(request):
        return JSONResponse(
            {"detail": "Cortex: password required"},
            status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="Cortex"'},
        )

    headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP}
    body = await request.body()
    target = "/" + path
    if request.url.query:
        query = request.url.query
        # keep the auth token out of the upstream request
        query = "&".join(p for p in query.split("&") if not p.startswith("k=")) if query else ""
        target += ("?" + query) if query else ""

    upstream_request = client.build_request(request.method, target, headers=headers, content=body)
    try:
        upstream = await client.send(upstream_request, stream=True)
    except httpx.HTTPError as exc:
        return JSONResponse({"detail": f"Cortex is not reachable: {exc}"}, status_code=502)

    out_headers = {k: v for k, v in upstream.headers.items() if k.lower() not in DROP_FROM_UPSTREAM}
    out_headers.setdefault("X-Accel-Buffering", "no")      # never buffer an SSE stream

    async def stream():
        try:
            async for chunk in upstream.aiter_bytes():
                yield chunk
        finally:
            await upstream.aclose()

    response = StreamingResponse(stream(), status_code=upstream.status_code,
                                 headers=out_headers, media_type=upstream.headers.get("content-type"))
    if not _cookie_ok(request):
        # an installed app launches in its own context; a cookie means it does not
        # ask for the password on every launch
        response.set_cookie(COOKIE, _cookie_token(), max_age=60 * 60 * 24 * 90,
                            httponly=True, samesite="lax",
                            secure=request.url.scheme == "https")
    return response


def main() -> int:
    parser = argparse.ArgumentParser(description="Password gate for Cortex")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    if not PASSWORD:
        print("CORTEX_GATE_PASSWORD is not set — refusing to expose the app without a password.",
              file=sys.stderr)
        return 2
    import uvicorn
    print(f"gate on http://{args.host}:{args.port} -> {UPSTREAM} (password required)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
