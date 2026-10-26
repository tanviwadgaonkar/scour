"""Evaluate BM25 (and the TF-IDF baseline) on a benchmark.

    python -m scour.eval.run --data data/cranfield
    python -m scour.eval.run --data data/cranfield --k1 1.2 --b 0.75

Prints a Markdown table you can paste straight into the README.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

from scour.bm25 import BM25
from scour.eval.metrics import average_precision, ndcg_at_k, precision_at_k, recall_at_k
from scour.index import InvertedIndex
from scour.tfidf import TFIDF


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with open(path, encoding="utf-8") as fh:
        next(fh)  # header
        for line in fh:
            qid, doc_id, gain = line.rstrip("\n").split("\t")
            qrels[qid][doc_id] = int(gain)
    return qrels


def evaluate(ranker, queries: list[dict], qrels: dict[str, dict[str, int]], k: int = 10) -> dict[str, float]:
    p, r, ap, nd, latencies = [], [], [], [], []
    for q in queries:
        rels = qrels.get(q["id"])
        if not rels:
            continue
        start = time.perf_counter()
        ranked = [doc_id for doc_id, _ in ranker.search(q["text"], k=max(k, 100))]
        latencies.append((time.perf_counter() - start) * 1000)
        p.append(precision_at_k(ranked, rels, k))
        r.append(recall_at_k(ranked, rels, k))
        ap.append(average_precision(ranked, rels))
        nd.append(ndcg_at_k(ranked, rels, k))
    return {
        f"P@{k}": statistics.mean(p),
        f"R@{k}": statistics.mean(r),
        "MAP": statistics.mean(ap),
        f"nDCG@{k}": statistics.mean(nd),
        "p50 ms": statistics.median(latencies),
        "p95 ms": sorted(latencies)[int(0.95 * (len(latencies) - 1))],
        "queries": len(p),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/cranfield")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--neural", action="store_true", help="also evaluate dense, hybrid and reranked retrieval (needs the [neural] extra)")
    parser.add_argument("--rerank-top", type=int, default=50)
    parser.add_argument("--text-only", action="store_true",
                        help="index document bodies only (no titles) for BM25/TF-IDF")
    args = parser.parse_args(argv)

    data = Path(args.data)
    docs = load_jsonl(data / "docs.jsonl")
    queries = load_jsonl(data / "queries.jsonl")
    qrels = load_qrels(data / "qrels.tsv")

    start = time.perf_counter()
    triples = [(d["id"], d.get("title", ""), d["text"]) for d in docs]
    index = InvertedIndex.build(triples, index_titles=not args.text_only)
    build_s = time.perf_counter() - start
    print(f"index ({'text only' if args.text_only else 'title + text'}): {index.num_docs} docs, {index.num_terms} terms, avg length {index.avg_doc_length:.1f}, built in {build_s:.2f}s\n")

    bm25 = BM25(index, args.k1, args.b)
    rows = {
        "TF-IDF (baseline)": evaluate(TFIDF(index), queries, qrels, args.k),
        f"BM25 (k1={args.k1}, b={args.b})": evaluate(bm25, queries, qrels, args.k),
    }
    if args.neural:
        from scour.dense import DenseIndex, SentenceTransformerEmbedder
        from scour.hybrid import HybridRanker
        from scour.rerank import CrossEncoderScorer, RerankedRanker

        embedder = SentenceTransformerEmbedder()
        start = time.perf_counter()
        dense = DenseIndex.cached(data / "dense_index", triples, embedder, embedder.model_name)
        print(f"dense index ready in {time.perf_counter() - start:.1f}s (cached under {data / 'dense_index'}.npy)\n")
        texts = {d["id"]: d["text"] for d in docs}
        hybrid = HybridRanker([bm25, dense])
        rows["Dense (all-MiniLM-L6-v2)"] = evaluate(dense, queries, qrels, args.k)
        rows["Hybrid (BM25 + dense, RRF)"] = evaluate(hybrid, queries, qrels, args.k)
        rows[f"Hybrid + cross-encoder rerank (top {args.rerank_top})"] = evaluate(
            RerankedRanker(hybrid, CrossEncoderScorer(), texts, top_n=args.rerank_top), queries, qrels, args.k)
    cols = [c for c in next(iter(rows.values())) if c != "queries"]
    print("| Ranker | " + " | ".join(cols) + " |")
    print("|---|" + "---|" * len(cols))
    for name, m in rows.items():
        print(f"| {name} | " + " | ".join(f"{m[c]:.3f}" if "ms" not in c else f"{m[c]:.2f}" for c in cols) + " |")
    print(f"\n{rows[next(iter(rows))]['queries']} queries with judgements")


if __name__ == "__main__":
    main()
