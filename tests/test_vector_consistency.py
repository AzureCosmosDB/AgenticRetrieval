"""
Vector-search consistency check.

Runs the same fixed query against Cosmos DB multiple times via
CombinedRetriever and reports whether the returned document IDs,
scores, and ordering are identical across runs.

Usage:
    python test_vector_consistency.py --config config.yaml
    python test_vector_consistency.py --config config.yaml --runs 5 --query "your question"
"""

import argparse
import asyncio
import sys
from pathlib import Path

# ── load config BEFORE anything that reads CONFIG ──────────────────────────
from agentic_retriever import load_config

parser = argparse.ArgumentParser(description="Vector-search consistency checker")
parser.add_argument("--config", type=Path, required=True, help="Path to config YAML")
parser.add_argument("--runs", type=int, default=3, help="Number of repeated runs (default: 3)")
parser.add_argument(
    "--query",
    type=str,
    default="your question here",
    help="Fixed question to use for every run",
)
parser.add_argument(
    "--vector-only",
    action="store_true",
    default=False,
    help="Disable fulltext retrieval (set fulltext_k=0) to isolate vector search",
)
parser.add_argument(
    "--source",
    type=str,
    default=None,
    help="ID of the cosmos source to use (from config cosmos.sources[].id). Default: use all sources.",
)
args = parser.parse_args()
load_config(args.config)

# Now safe to import modules that read CONFIG at import time
from agentic_retriever import CONFIG  # noqa: E402
from utils.cosmos_retriever import CombinedRetriever, _get_source_config  # noqa: E402


def _chunk_signature(chunks):
    """Return a list of (chunk_id, similarity) tuples preserving order."""
    return [(c.chunk_id, round(c.similarity, 8) if c.similarity is not None else None) for c in chunks]


def _print_run(run_idx, chunks):
    print(f"\n{'─' * 60}")
    print(f"Run {run_idx + 1}: {len(chunks)} chunks returned")
    print(f"{'─' * 60}")
    for i, c in enumerate(chunks):
        sim = f"{c.similarity:.8f}" if c.similarity is not None else "N/A"
        source = c.metadata.get("_data_source", "?")
        print(f"  [{i + 1:>3}] id={c.chunk_id:<30s}  sim={sim}  src={source}")


def _print_matrix(all_chunks, num_runs):
    """Print a matrix of doc IDs (rows) × runs (columns) with vector distances."""
    # Build a {doc_id: {run_idx: distance}} mapping and track insertion order
    # similarity = 1 - VectorDistance, so distance = 1 - similarity
    from collections import OrderedDict

    doc_runs: dict[str, dict[int, float | None]] = OrderedDict()
    for run_idx, chunks in enumerate(all_chunks):
        for c in chunks:
            if c.chunk_id not in doc_runs:
                doc_runs[c.chunk_id] = {}
            if c.similarity is not None:
                doc_runs[c.chunk_id][run_idx] = round(1.0 - c.similarity, 8)
            else:
                doc_runs[c.chunk_id][run_idx] = None

    # Determine which docs appear in ALL runs vs only some
    all_run_idxs = set(range(num_runs))
    inconsistent_ids = {
        did for did, runs in doc_runs.items() if set(runs.keys()) != all_run_idxs
    }

    # Column widths
    id_width = min(max((len(d) for d in doc_runs), default=20), 50)
    col_w = 12  # width per run column
    run_headers = [f"Run {i + 1}" for i in range(num_runs)]

    print(f"\n{'═' * 60}")
    print("RUN × DOC ID MATRIX  (values = cosine distance)")
    print(f"{'═' * 60}")
    print(f"  Total unique doc IDs : {len(doc_runs)}")
    print(f"  Consistent (all runs): {len(doc_runs) - len(inconsistent_ids)}")
    print(f"  Inconsistent         : {len(inconsistent_ids)}")

    # Header row
    header = f"  {'Doc ID':<{id_width}s}" + "".join(
        f" {h:>{col_w}s}" for h in run_headers
    )
    print(f"\n{header}")
    print(f"  {'─' * id_width}" + ("─" * (col_w + 1)) * num_runs)

    # Sort: inconsistent docs first (flagged), then consistent ones
    sorted_ids = sorted(doc_runs.keys(), key=lambda d: (d not in inconsistent_ids, d))

    for did in sorted_ids:
        runs = doc_runs[did]
        flag = " *" if did in inconsistent_ids else "  "
        truncated = did[:id_width].ljust(id_width)
        cells = []
        for r in range(num_runs):
            if r in runs:
                dist = runs[r]
                cells.append(f"{dist:.6f}" if dist is not None else "    N/A ")
            else:
                cells.append("      ·     ")
        row = f"{flag}{truncated}" + "".join(f" {c:>{col_w}s}" for c in cells)
        print(row)

    if inconsistent_ids:
        print(f"\n  * = doc ID not present in every run ({len(inconsistent_ids)} total)")


async def single_run(run_idx: int, query: str, fulltext_k_override, source_id: str | None):
    """Create a fresh retriever, run the query, close, and return chunks."""
    sources = _get_source_config(CONFIG)
    if source_id is not None:
        sources = [s for s in sources if s["id"] == source_id]
        if not sources:
            available = [s["id"] for s in _get_source_config(CONFIG)]
            raise ValueError(f"Source {source_id!r} not found. Available: {available}")
    retriever = CombinedRetriever(
        retrieval_sources=sources,
        fulltext_k_override=fulltext_k_override,
        k_diverse=0,  # disable diversity re-ranking for a pure vector test
        eta=0.0,
        rescale_power=0.0,
    )
    await retriever.initialize()
    try:
        chunks = await retriever.retrieve(query)
        return chunks
    finally:
        await retriever.close()


async def main():
    query = args.query
    num_runs = max(2, args.runs)
    fulltext_k_override = 0 if args.vector_only else None

    print(f"Query  : {query!r}")
    print(f"Runs   : {num_runs}")
    print(f"Config : {args.config}")
    print(f"Source : {args.source or '(all)'}")
    print(f"Vector-only: {args.vector_only}")

    all_signatures = []
    all_chunks = []

    for i in range(num_runs):
        chunks = await single_run(i, query, fulltext_k_override, args.source)
        _print_run(i, chunks)
        all_signatures.append(_chunk_signature(chunks))
        all_chunks.append(chunks)

    # ── Run × Doc ID matrix ──────────────────────────────────────────
    _print_matrix(all_chunks, num_runs)

    # ── Consistency analysis ───────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print("CONSISTENCY REPORT")
    print(f"{'═' * 60}")

    baseline = all_signatures[0]

    # 1. ID-set consistency (same documents, regardless of order)
    baseline_ids = set(sig[0] for sig in baseline)
    id_set_consistent = True
    for i, sig in enumerate(all_signatures[1:], start=2):
        run_ids = set(s[0] for s in sig)
        if run_ids != baseline_ids:
            id_set_consistent = False
            missing = baseline_ids - run_ids
            extra = run_ids - baseline_ids
            print(f"  Run {i} ID mismatch vs Run 1:")
            if missing:
                print(f"    Missing: {missing}")
            if extra:
                print(f"    Extra  : {extra}")

    # 2. Ordering consistency (same IDs in same order)
    baseline_order = [sig[0] for sig in baseline]
    order_consistent = all(
        [s[0] for s in sig] == baseline_order for sig in all_signatures[1:]
    )

    # 3. Score consistency (exact same similarity scores)
    score_consistent = all(sig == baseline for sig in all_signatures[1:])

    # 4. Score drift (max absolute difference across runs)
    max_drift = 0.0
    for sig in all_signatures[1:]:
        baseline_score_map = {s[0]: s[1] for s in baseline}
        for cid, score in sig:
            if score is not None and cid in baseline_score_map and baseline_score_map[cid] is not None:
                max_drift = max(max_drift, abs(score - baseline_score_map[cid]))

    status = lambda ok: "✅ PASS" if ok else "❌ FAIL"  # noqa: E731
    print(f"\n  Document set consistent : {status(id_set_consistent)}")
    print(f"  Ordering consistent     : {status(order_consistent)}")
    print(f"  Scores exactly equal    : {status(score_consistent)}")
    print(f"  Max score drift         : {max_drift:.10f}")

    all_pass = id_set_consistent and order_consistent and score_consistent
    print(f"\n  Overall                 : {status(all_pass)}")

    if not all_pass:
        print("\n⚠ Results varied across runs. This may be due to:")
        print("  - diskANN approximate index non-determinism")
        print("  - Cosmos DB cross-partition query ordering")
        print("  - Index build in progress")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
