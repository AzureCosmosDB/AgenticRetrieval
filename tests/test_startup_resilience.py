import asyncio

import pytest
from azure.cosmos import exceptions

from utils.candidates import (
    ConfigurationError,
    fetch_candidate_documents,
    fetch_candidate_results,
    resolve_partition_key_path,
)
from utils.cosmos_retriever import CombinedRetriever
from utils.ranker import RankerUnavailableError, verify_ranker_token


class PolicyContainer:
    id = "products"

    def __init__(self, properties=None, error=None):
        self.properties = properties
        self.error = error
        self.read_calls = 0

    async def read(self):
        self.read_calls += 1
        if self.error:
            raise self.error
        return self.properties


def resolve(container, configured):
    return asyncio.run(resolve_partition_key_path(container, configured, "products"))


def test_container_partition_key_wins_over_config():
    container = PolicyContainer({"partitionKey": {"paths": ["/upc"]}})
    assert resolve(container, "/stale") == "/upc"
    assert resolve(container, None) == "/upc"


def test_config_is_used_only_when_container_cannot_be_read():
    assert resolve(PolicyContainer(error=RuntimeError("denied")), "/pk") == "/pk"
    assert resolve(PolicyContainer({}), "/pk") == "/pk"


@pytest.mark.parametrize("configured", [None, "", "pk", "/"])
def test_unreadable_container_and_bad_config_fails_clearly(configured):
    with pytest.raises(ConfigurationError, match="partition_key_path"):
        resolve(PolicyContainer(error=RuntimeError("denied")), configured)


@pytest.mark.parametrize("paths", [[], ["/a", "/b"]])
def test_partitionless_and_hierarchical_containers_are_rejected(paths):
    with pytest.raises(ConfigurationError, match="products"):
        resolve(PolicyContainer({"partitionKey": {"paths": paths}}), "/pk")


class FlakyContainer:
    id = "products"

    def __init__(self, failures):
        self.failures = failures

    async def read_item(self, *, item, partition_key):
        status = self.failures.get(item)
        if status == 404:
            raise exceptions.CosmosResourceNotFoundError(status_code=404, message="missing")
        if status:
            raise exceptions.CosmosHttpResponseError(status_code=status, message="failed")
        return {"id": item}


def candidates(*ids):
    return [{"id": i, "_partition_key": "p"} for i in ids]


def test_404_and_429_reads_are_skipped_in_place():
    container = FlakyContainer({"b": 404, "c": 429})
    refs = [(container, c) for c in candidates("a", "b", "c", "d")]
    results = asyncio.run(fetch_candidate_results(refs))
    assert [r and r["id"] for r in results] == ["a", None, None, "d"]
    assert [d["id"] for d in asyncio.run(fetch_candidate_documents(refs))] == ["a", "d"]


def test_other_read_errors_still_propagate():
    container = FlakyContainer({"b": 500})
    with pytest.raises(exceptions.CosmosHttpResponseError):
        asyncio.run(fetch_candidate_results([(container, c) for c in candidates("a", "b")]))


def retriever_for(container):
    retriever = object.__new__(CombinedRetriever)
    retriever._ctrproxy = container
    retriever._indexed_fields_cache = {}
    return retriever


def test_bad_index_path_is_skipped_and_policy_is_read_once(caplog):
    container = PolicyContainer({"indexingPolicy": {"fullTextIndexes": [
        {"path": "/title/?"}, {"path": "/bad-name/?"}, {"path": "/claims/[]"}, {"path": "/*"},
    ]}})
    retriever = retriever_for(container)
    expected = ["title", "claims[]"]
    assert asyncio.run(retriever._indexed_fulltext_fields()) == expected
    assert asyncio.run(retriever._indexed_fulltext_fields()) == expected
    assert container.read_calls == 1
    assert sum("Skipping full-text index path" in r.message for r in caplog.records) == 2


def test_failed_policy_read_is_not_cached():
    container = PolicyContainer(error=RuntimeError("boom"))
    retriever = retriever_for(container)
    assert asyncio.run(retriever._indexed_fulltext_fields()) == []
    container.error = None
    container.properties = {"indexingPolicy": {"fullTextIndexes": [{"path": "/title/?"}]}}
    assert asyncio.run(retriever._indexed_fulltext_fields()) == ["title"]


def test_ranker_token_failure_respects_fail_on_error():
    def broken():
        raise RuntimeError("az login required")

    with pytest.raises(RankerUnavailableError, match="az login required"):
        asyncio.run(verify_ranker_token(broken, True))
    assert "unavailable" in asyncio.run(verify_ranker_token(broken, False))
    assert asyncio.run(verify_ranker_token(lambda: "token", True)) is None
    assert asyncio.run(verify_ranker_token("static-token", True)) is None
