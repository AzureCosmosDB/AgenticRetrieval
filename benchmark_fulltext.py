"""Read-only paired Cosmos full-text benchmark; no embeddings or answer generation."""

import argparse
import asyncio
import hashlib
import json
import logging
import statistics
import time
from dataclasses import asdict
from pathlib import Path

import httpx
import yaml
from azure.cosmos.aio import CosmosClient
from azure.identity import AzureCliCredential, get_bearer_token_provider
from azure.identity.aio import AzureCliCredential as AsyncAzureCliCredential

from utils.candidates import candidate_identity, fetch_candidate_results, resolve_partition_key_path
from utils.fulltext import fulltext_search, fulltext_search_options
from utils.ranker import rerank_documents


def candidate_keys(rows: dict[str, list[dict]]) -> set[tuple[str, tuple[str, str]]]:
    return {(source_id, candidate_identity(row)) for source_id, items in rows.items() for row in items}


def recall(found: set, reference: set) -> float | None:
    return len(found & reference) / len(reference) if reference else None


def summarize(records: list[dict]) -> dict:
    summary = {}
    for name in dict.fromkeys(v for r in records for v in r["variants"]):
        runs = [r["variants"][name] for r in records]
        summary[name] = {
            "searches": len(runs),
            "query_count": sum(r["query_count"] for r in runs),
            "request_charge": sum(r["request_charge"] for r in runs),
            "elapsed_seconds": sum(r["elapsed_seconds"] for r in runs),
            "median_seconds": statistics.median(r["elapsed_seconds"] for r in runs),
            "p95_seconds": sorted(r["elapsed_seconds"] for r in runs)[
                max(0, int(len(runs) * 0.95 + 0.999) - 1)
            ],
            "mean_baseline_overlap": statistics.mean(r["baseline_overlap"] for r in runs),
        }
        for metric in ("ranker_reference_recall", "labeled_identity_recall"):
            values = [r[metric] for r in runs if r.get(metric) is not None]
            summary[name][f"mean_{metric}"] = statistics.mean(values) if values else None
    return summary


async def run_variant(database, sources: list[dict], question: str) -> dict:
    metrics = []
    started = time.perf_counter()

    async def search(source):
        retrieval = source["retrieval"]
        rows = await fulltext_search(
            database.get_container_client(source["container_name"]),
            retrieval["fulltext_fields"], question, retrieval["fulltext_search_k"],
            partition_key_path=source["partition_key_path"], metrics_fn=metrics.append,
            **fulltext_search_options(retrieval),
        )
        return source["id"], rows

    rows = dict(await asyncio.gather(*(search(source) for source in sources)))
    if any(m.request_charge is None or not m.succeeded for m in metrics):
        raise RuntimeError("Benchmark requires successful queries with measured request charges")
    return {
        "elapsed_seconds": time.perf_counter() - started,
        "request_charge": sum(m.request_charge for m in metrics),
        "query_count": len(metrics),
        "queries": [asdict(m) for m in metrics],
        "candidates": rows,
    }


async def ranker_reference(database, sources, variants, question, http_client, ranker_cfg, token):
    union = {}
    source_by_id = {s["id"]: s for s in sources}
    for variant in variants.values():
        for source_id, rows in variant["candidates"].items():
            for row in rows:
                union.setdefault((source_id, candidate_identity(row)), row)
    keys = list(union)
    refs = [
        (database.get_container_client(source_by_id[sid]["container_name"]), union[(sid, key)])
        for sid, key in keys
    ]
    documents = await fetch_candidate_results(refs)
    if any(doc is None for doc in documents):
        raise RuntimeError("A benchmark candidate could not be fetched; relevance comparison is incomplete")
    excluded = {"_rid", "_self", "_etag", "_attachments", "_ts", "_score", "e"}
    excluded.update(s["embedding_field"] for s in sources)
    texts = ["\n".join(f"{k}: {v}" for k, v in doc.items() if k not in excluded and v) for doc in documents]
    k = min(sum(s["retrieval"]["fulltext_search_k"] for s in sources), len(texts))
    if k == 0:
        return set()
    url = (
        f"https://{ranker_cfg['account_name']}.{ranker_cfg['region']}."
        f"{ranker_cfg['url_suffix']}"
    )
    indices = await rerank_documents(
        http_client, url, token, question, texts, k,
        batch_size=ranker_cfg.get("batch_size", 32), max_retries=ranker_cfg.get("max_retries", 5),
    )
    if indices is None or len(set(indices)) != k:
        raise RuntimeError("Semantic ranker did not produce a complete reference ranking")
    return {keys[i] for i in indices}


async def run(args):
    baseline = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    candidate = yaml.safe_load(args.candidate_config.read_text(encoding="utf-8"))
    cosmos = baseline["cosmos"]
    if any(cosmos.get(k) != candidate["cosmos"].get(k) for k in ("uri", "database_name")):
        raise ValueError("Both configurations must target the same account and database")
    before, after = cosmos["sources"], candidate["cosmos"]["sources"]
    if [(s["id"], s["container_name"]) for s in before] != [(s["id"], s["container_name"]) for s in after]:
        raise ValueError("Both configurations must have the same ordered sources and containers")
    variants = {
        "baseline": before,
        "match_only": [
            {**s, "retrieval": {**s["retrieval"], "fulltext_match_filter": True}} for s in before
        ],
        "candidate": after,
        "candidate_consolidated": [
            {**s, "retrieval": {**s["retrieval"], "fulltext_query_mode": "consolidated"}}
            for s in after
        ],
    }
    questions_path = args.questions or Path(baseline["paths"]["questions_path"])
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    if args.limit:
        questions = questions[:args.limit]
    records = []
    report = {
        "complete": False,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "candidate_config_sha256": hashlib.sha256(args.candidate_config.read_bytes()).hexdigest(),
        "repeats": args.repeats,
        "methodology": (
            "Paired full-text-only candidate searches. Variant order rotates per question/repeat. "
            "RU sums SDK response-hook charges including internal ranking operations; elapsed time "
            "includes concurrent sources but excludes point reads and semantic relevance evaluation. "
            "Baseline overlap is relative candidate recall, NOT true relevance recall. "
            "Ranker-reference recall uses the top-k semantic ranking of the pooled candidates, "
            "NOT human judgments or clinical validation. Labeled identity recall uses exact indexed "
            "UPC lookups as ground truth, only for the added identity probes. No final answers generated."
        ),
        "variants": {
            name: {
                s["id"]: {**fulltext_search_options(s["retrieval"]),
                          "fields": s["retrieval"]["fulltext_fields"],
                          "k": s["retrieval"]["fulltext_search_k"]}
                for s in sources
            } for name, sources in variants.items()
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        report["summary"] = summarize(records) if records else {}
        report["summary_by_kind"] = {
            kind: summarize([r for r in records if r["kind"] == kind])
            for kind in dict.fromkeys(r["kind"] for r in records)
        }
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    credential = AsyncAzureCliCredential(
        tenant_id=cosmos.get("tenant_id"), process_timeout=60,
    ) if cosmos.get("use_rbac_auth") else None
    ranker_credential = AzureCliCredential(
        tenant_id=candidate.get("ranker", {}).get("tenant_id") or cosmos.get("tenant_id"),
        process_timeout=60,
    ) if args.ranker else None
    try:
        async with CosmosClient(cosmos["uri"], credential=credential or cosmos["key"]) as client:
            database = client.get_database_client(cosmos["database_name"])
            for old, new in zip(before, after):
                path = await resolve_partition_key_path(
                    database.get_container_client(old["container_name"]),
                    old.get("partition_key_path"), old["id"],
                )
                old["partition_key_path"] = new["partition_key_path"] = path
            # The variants share source dictionaries with before/after except where
            # their retrieval settings were changed.
            for sources in variants.values():
                for source, resolved in zip(sources, before):
                    source["partition_key_path"] = resolved["partition_key_path"]
            identity_truth = {}
            for index, upc in enumerate(args.identity_upc):
                qid = f"identity-{index + 1}"
                truth = set()
                for source in before:
                    if source["partition_key_path"] != "/upc":
                        raise ValueError("--identity-upc requires sources partitioned by /upc")
                    container = database.get_container_client(source["container_name"])
                    rows = [r async for r in container.query_items(
                        query='SELECT c.id, c.upc AS _partition_key FROM c WHERE c.upc = @upc',
                        parameters=[{"name": "@upc", "value": upc}], partition_key=upc,
                    )]
                    truth.update((source["id"], candidate_identity(r)) for r in rows)
                if not truth:
                    raise ValueError(f"Identity probe UPC {upc!r} does not exist")
                identity_truth[qid] = truth
                questions.append({"question_id": qid, "question_text": upc})
            questions.append({"question_id": "zero-hit", "question_text": "zzzxqvnonexistentproducttoken"})
            async with httpx.AsyncClient(timeout=120) as http_client:
                token = (
                    get_bearer_token_provider(
                        ranker_credential, candidate["ranker"]["token_scope"],
                    ) if ranker_credential else None
                )
                for repeat in range(args.repeats):
                    for index, question in enumerate(questions):
                        qid, text = question["question_id"], question["question_text"]
                        names = list(variants)
                        offset = (index + repeat) % len(names)
                        names = names[offset:] + names[:offset]
                        runs = {}
                        for name in names:
                            runs[name] = await run_variant(database, variants[name], text)
                        base_keys = candidate_keys(runs["baseline"]["candidates"])
                        reference = (
                            await ranker_reference(
                                database, after, runs, text, http_client, candidate["ranker"], token,
                            ) if args.ranker and qid not in identity_truth and qid != "zero-hit" else None
                        )
                        for result in runs.values():
                            found = candidate_keys(result["candidates"])
                            result["baseline_overlap"] = recall(found, base_keys) if base_keys else 1.0
                            result["ranker_reference_recall"] = recall(found, reference) if reference is not None else None
                            result["labeled_identity_recall"] = recall(found, identity_truth[qid]) if qid in identity_truth else None
                        records.append({
                            "question_id": qid, "question_text": text, "repeat": repeat + 1,
                            "kind": "identity" if qid in identity_truth else "negative" if qid == "zero-hit" else "question",
                            "ranker_reference": sorted(reference) if reference is not None else None,
                            "identity_truth": sorted(identity_truth[qid]) if qid in identity_truth else None,
                            "variant_order": names, "variants": runs,
                        })
                        save()
                        print(json.dumps({
                            "question_id": qid, "repeat": repeat + 1,
                            "variants": {name: {k: r[k] for k in (
                                "elapsed_seconds", "request_charge", "baseline_overlap",
                                "ranker_reference_recall", "labeled_identity_recall",
                            )} for name, r in runs.items()},
                        }), flush=True)
        report["complete"] = True
        save()
    finally:
        if credential:
            await credential.close()
        if ranker_credential:
            ranker_credential.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Baseline configuration")
    parser.add_argument("--candidate-config", required=True, type=Path)
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ranker", action="store_true", help="Evaluate pooled candidate relevance with configured ranker")
    parser.add_argument("--identity-upc", action="append", default=[], help="Add an exact-identity recall probe")
    args = parser.parse_args()
    if args.repeats < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--repeats and --limit must be positive")
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
