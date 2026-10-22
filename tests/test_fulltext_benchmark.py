import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import benchmark_fulltext as benchmark


def test_relative_recall_is_partition_and_source_aware():
    baseline = benchmark.candidate_keys({
        "one": [{"id": "a", "_partition_key": "001"}, {"id": "a", "_partition_key": "002"}],
        "two": [{"id": "a", "_partition_key": "001"}],
    })
    found = benchmark.candidate_keys({"one": [{"id": "a", "_partition_key": "001"}]})
    assert benchmark.recall(found, baseline) == pytest.approx(1 / 3)
    assert benchmark.recall(found, set()) is None


def test_summary_aggregates_queries_ru_and_latency_without_missing_relevance_defaults():
    records = [{"variants": {"candidate": {
        "query_count": 3, "request_charge": 4.5, "elapsed_seconds": seconds,
        "baseline_overlap": 0.5, "ranker_reference_recall": None,
        "labeled_identity_recall": 1.0,
    }}} for seconds in (1.0, 3.0)]
    summary = benchmark.summarize(records)["candidate"]
    assert summary["query_count"] == 6
    assert summary["request_charge"] == 9
    assert summary["elapsed_seconds"] == 4
    assert summary["median_seconds"] == 2
    assert summary["p95_seconds"] == 3
    assert summary["mean_ranker_reference_recall"] is None
    assert summary["mean_labeled_identity_recall"] == 1


def test_ranker_reference_deduplicates_candidates_preserves_keys_and_excludes_embeddings(monkeypatch):
    rows = {"source": [
        {"id": "a", "_partition_key": "001"}, {"id": "b", "_partition_key": "002"},
    ]}
    fetch = AsyncMock(return_value=[
        {"id": "a", "title": "cream", "embedding": [1, 2]},
        {"id": "b", "title": "serum", "embedding": [3, 4]},
    ])
    rerank = AsyncMock(return_value=[1])
    monkeypatch.setattr(benchmark, "fetch_candidate_results", fetch)
    monkeypatch.setattr(benchmark, "rerank_documents", rerank)
    result = asyncio.run(benchmark.ranker_reference(
        SimpleNamespace(get_container_client=lambda name: name),
        [{"id": "source", "container_name": "container", "embedding_field": "embedding",
          "retrieval": {"fulltext_search_k": 1}}],
        {"baseline": {"candidates": rows}, "candidate": {"candidates": rows}},
        "serum", object(), {"account_name": "ranker", "region": "region", "url_suffix": "suffix"},
        "token",
    ))
    assert result == {("source", benchmark.candidate_identity(rows["source"][1]))}
    assert len(fetch.await_args.args[0]) == 2
    assert rerank.await_args.args[4] == ["id: a\ntitle: cream", "id: b\ntitle: serum"]
