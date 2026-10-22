#!/usr/bin/env python3
"""Build a single-file, offline demo of Cortex.

The workspace file viewer renders HTML in a sandboxed iframe with no network
access, so this variant embeds the corpus, a BM25 retriever and the UI in one
file. It demonstrates the retrieval + citation experience with zero moving parts.

    python tests/build_standalone.py     ->  ../Cortex-demo.html
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = (ROOT / "tests" / "_template_raw.html").read_text(encoding="utf-8")
CORPUS = json.loads((ROOT / "tests" / "corpus.json").read_text(encoding="utf-8"))
OUT = ROOT.parent / "Cortex-demo.html"

SUGGESTED = [
    "How many annual leave days do I get and can I carry any over?",
    "What is the run timeout on the Scale plan?",
    "Which sub-processors process our data?",
    "Can I get a cash refund for a partial month?",
]

html = (TEMPLATE
        .replace("__DATA__", json.dumps(CORPUS, ensure_ascii=False, separators=(",", ":")))
        .replace("__SUGGESTED__", json.dumps(SUGGESTED, ensure_ascii=False))
        .replace("__CHUNKS__", str(len(CORPUS))))

for placeholder in ("__DATA__", "__SUGGESTED__", "__CHUNKS__"):
    assert placeholder not in html, f"unsubstituted placeholder: {placeholder}"

OUT.write_text(html, encoding="utf-8")
print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB, {len(CORPUS)} passages embedded)")
