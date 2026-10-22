"""Retrieval evaluation harness.

Runs a golden question set against the demo corpus and reports recall, MRR and
answer-fact coverage. Used to prove retrieval changes actually help.

    python tests/eval_retrieval.py            # current settings
    python tests/eval_retrieval.py --json     # machine-readable output
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TMP = Path(tempfile.mkdtemp(prefix="cortex_eval_"))
os.environ["CORTEX_DATA_DIR"] = str(TMP)
os.environ["CORTEX_DB"] = str(TMP / "eval.db")

from app import config, db, ingest, retrieval  # noqa: E402

# (question, expected doc substring, facts that must appear in the retrieved text)
GOLDEN = [
    ("How many annual leave days do employees get?", "Handbook", ["25 days", "carried over"]),
    ("How much vacation time do I get per year?", "Handbook", ["25 days"]),
    ("Can I carry unused leave into next year?", "Handbook", ["5 unused days", "31 March"]),
    ("How many sick days do we get?", "Handbook", ["30 days of paid sick leave"]),
    ("How long is parental leave for the primary caregiver?", "Handbook", ["20 weeks"]),
    ("Do I need a doctor's note when I'm sick?", "Handbook", ["medical certificate", "two consecutive days"]),
    ("How many days a week can I work from home?", "Handbook", ["3 days per week", "anchor days"]),
    ("What is the home office allowance for new employees?", "Handbook", ["R12,000"]),
    ("How much internet costs are reimbursed?", "Handbook", ["R800"]),
    ("When must expenses be submitted?", "Handbook", ["30 days", "Expenses"]),
    ("What is the daily meal limit during travel?", "Handbook", ["R650 per day"]),
    ("What is the learning and development budget?", "Handbook", ["R25,000"]),
    ("When do promotion cycles happen?", "Handbook", ["May and November"]),
    ("What is the minimum password length?", "Handbook", ["14 characters"]),
    ("How quickly must a data breach be reported internally?", "Handbook", ["within 1 hour"]),
    ("What is the notice period for managers?", "Handbook", ["60 days"]),
    ("What is the payroll cut-off date?", "Handbook", ["15th"]),

    ("What is the run timeout on the Scale plan?", "Product_Guide", ["60"]),
    ("How much does the Growth plan cost?", "Pricing", ["199"]),
    ("What does the Scale plan cost per month?", "Pricing", ["649"]),
    ("How much is charged per run over the included quota?", "Product_Guide", ["0.004"]),
    ("How many times are failed webhook deliveries retried?", "Playbook", ["3 times", "10 minutes"]),
    ("How long before an approval escalates automatically?", "Product_Guide", ["48 hours"]),
    ("What is the first response time for an S1 incident?", "Playbook", ["15 minutes"]),
    ("Can I get a cash refund for a partial month?", "Playbook", ["does not provide cash refunds", "14-day money-back"]),
    ("Where is customer data hosted?", "Product_Guide", ["Frankfurt", "Virginia"]),
    ("What is the API rate limit?", "Product_Guide", ["600 requests per minute"]),
    ("How does the Anomaly Digest decide something is unusual?", "Product_Guide", ["3 standard deviations"]),
    ("How many Copilot generations are included per month?", "Product_Guide", ["500 generations"]),
    ("How long are customer records retained after closure?", "Data_Retention", ["7 years"]),
    ("Which sub-processors process our data?", "Data_Retention", ["Amazon Web Services", "Cloudflare", "Postmark"]),
    ("How long are database backups kept?", "Data_Retention", ["90 days"]),
    ("Who owns the data retention policy?", "Data_Retention", ["Chief Information Security Officer"]),
]

UNANSWERABLE = [
    "What is the CEO's salary?",
    "How many employees work in the Berlin office?",
    "What is the parental leave policy in Japan?",
    "When is the next company offsite?",
    "What is the stock option strike price?",
]


def ingest_corpus() -> int:
    db.init()
    n = 0
    for f in sorted((ROOT / "demo_docs").iterdir()):
        if f.is_file():
            d = ingest.ingest_file(f)
            n += d["n_chunks"]
    return n


def warm() -> None:
    """Load models and prime caches so latency numbers reflect steady state."""
    retrieval.search("warm up query", 3)
    retrieval.search("second warm up query about leave", 3)


def evaluate(k: int = 8) -> dict:
    hits_at = {1: 0, 3: 0, k: 0}
    mrr = 0.0
    fact_cov = []
    latencies = []
    misses = []

    for question, doc, facts in GOLDEN:
        t0 = time.perf_counter()
        sources, diag = retrieval.search(question, k)
        latencies.append((time.perf_counter() - t0) * 1000)

        ranked_docs = [s["doc_name"] for s in sources]
        first = next((i for i, d in enumerate(ranked_docs) if doc in d), None)
        if first is not None:
            mrr += 1.0 / (first + 1)
            for n in hits_at:
                if first < n:
                    hits_at[n] += 1
        else:
            misses.append((question, doc, ranked_docs[:2]))

        blob = " ".join(s["text"] for s in sources[:3]).lower()
        found = sum(1 for f in facts if f.lower() in blob)
        fact_cov.append(found / len(facts))

    n = len(GOLDEN)
    # unanswerable: a confident top score is a false positive
    false_confident = []
    for q in UNANSWERABLE:
        sources, _ = retrieval.search(q, 3)
        if sources and sources[0]["score"] >= 0.55:
            false_confident.append((q, round(sources[0]["score"], 3), sources[0]["doc_name"]))

    return {
        "questions": n,
        "recall@1": round(hits_at[1] / n, 3),
        "recall@3": round(hits_at[3] / n, 3),
        "recall@k": round(hits_at[k] / n, 3),
        "mrr": round(mrr / n, 3),
        "fact_coverage@3": round(sum(fact_cov) / n, 3),
        "mean_latency_ms": round(sum(latencies) / len(latencies), 1),
        "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95) - 1], 1),
        "unanswerable_false_confident": len(false_confident),
        "misses": misses,
        "false_confident": false_confident,
        "config": {
            "rerank": config.RERANK_ENABLED,
            "top_k": k,
            "candidates": config.CANDIDATES_PER_RETRIEVER,
            "fused_pool": config.FUSED_POOL,
        },
    }


def main() -> int:
    chunks = ingest_corpus()
    print(f"\n\033[1mCortex retrieval evaluation\033[0m — {len(GOLDEN)} questions, {chunks} chunks")
    print(f"reranker: {'on (' + config.RERANK_MODEL + ')' if config.RERANK_ENABLED else 'off'}\n")

    warm()
    r = evaluate(k=8)
    rows = [
        ("Recall@1", r["recall@1"]),
        ("Recall@3", r["recall@3"]),
        ("Recall@8", r["recall@k"]),
        ("MRR", r["mrr"]),
        ("Fact coverage@3", r["fact_coverage@3"]),
        ("Mean latency", f"{r['mean_latency_ms']} ms"),
        ("p95 latency", f"{r['p95_latency_ms']} ms"),
        ("False-confident (unanswerable)", f"{r['unanswerable_false_confident']}/{len(UNANSWERABLE)}"),
    ]
    for name, value in rows:
        bar = ""
        if isinstance(value, float) and 0 <= value <= 1:
            filled = int(value * 24)
            bar = "  " + "█" * filled + "░" * (24 - filled)
        print(f"  {name:32} {value}{bar}")

    if r["misses"]:
        print("\n  misses:")
        for q, doc, got in r["misses"]:
            print(f"   · {q[:62]:64} wanted {doc:16} got {', '.join(d.split('_')[0] for d in got)}")
    if r["false_confident"]:
        print("\n  false-confident on unanswerable:")
        for q, score, doc in r["false_confident"]:
            print(f"   · {q[:58]:60} {score} {doc}")

    if "--json" in sys.argv:
        print("\n" + json.dumps(r, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
