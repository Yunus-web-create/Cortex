#!/usr/bin/env python3
"""Start Cortex.  Usage: python run.py [--port 8000]"""
from __future__ import annotations

import argparse
import os

import uvicorn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Cortex knowledge assistant")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    parser.add_argument("--reload", action="store_true", help="auto-reload on code changes (dev)")
    args = parser.parse_args()

    print(f"\n  Cortex → http://localhost:{args.port}\n")
    uvicorn.run("app.main:app", host=args.host, port=args.port,
                reload=args.reload, log_level="info")
