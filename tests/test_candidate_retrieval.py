import ast
import asyncio
import copy
import json
import logging
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from utils.candidates import (
    ConfigurationError,
    candidate_identity,
    candidate_projection,
    container_identity,
    fetch_candidate_documents,
    fetch_candidate_results,
    resolve_partition_key_path,
    validate_partition_key_path,
)
from utils.fulltext import fulltext_search, fulltext_search_options
from utils.ranker import RankerUnavailableError

ROOT = Path(__file__).resolve().parents[1]


class Container:
    id = "products"

    def __init__(self):
        self.queries = []
        self.reads = []

    def query_items(self, **kwargs):
        self.queries.append(kwargs)
        vector = "VectorDistance" in kwargs["query"]

        async def rows():
            for document_id in (["a", "b"] if vector else ["b", "c"]):
                row = {"id": document_id, "_partition_key": f"00{document_id}"}
                if vector:
                    row["_score"] = 0.2
                yield row

        return rows()

    async def read_item(self, *, item, partition_key):
        self.reads.append((item, partition_key))
        return {
            "id": item, "upc": partition_key, "title": f"Full product {item}",
            "claims": ["fragrance free"], "e": [0.1] * 1536,
        }


@pytest.mark.parametrize(
    ("path", "expression"),
    [("/upc", 'c["upc"]'), ("/id", 'c["id"]'), ("/tenant/key", 'c["tenant"]["key"]')],
)
def test_projection_contains_only_id_and_partition_key(path, expression):
    assert candidate_projection(path) == f"c.id, {expression} AS _partition_key"


@pytest.mark.parametrize("path", ["", "upc", "/", "/items/[]", "/upc/*", "/upc/"])
def test_invalid_partition_paths_raise(path):
    with pytest.raises(ValueError, match="Invalid partition key path"):
        candidate_projection(path)


def test_candidate_merge_preserves_partition_container_order_and_score():
    first, second = Container(), Container()
    a = {"id": "same", "_partition_key": "001", "_score": 0.25}
    b = {"id": "same", "_partition_key": "002"}
    docs = asyncio.run(fetch_candidate_documents([
        (first, a), (first, b), (first, a), (second, a),
    ]))
    assert first.reads == [("same", "001"), ("same", "002")]
    assert second.reads == [("same", "001")]
    assert [doc["upc"] for doc in docs] == ["001", "002", "001"]
    assert docs[0]["_score"] == 0.25
    assert "_score" not in docs[1]
    assert len(docs[0]["e"]) == 1536
    assert "_partition_key" not in docs[0]
    assert len(json.dumps(a)) < 100
    assert len(json.dumps(docs[0])) > 5000


@pytest.mark.parametrize(
    ("endpoint", "link", "same_container"),
    [
        ("https://ACCOUNT.documents.azure.com:443/", "dbs/db/colls/products", True),
        ("https://account.documents.azure.com", "dbs/other/colls/products", False),
        ("https://other.documents.azure.com", "dbs/db/colls/products", False),
        ("https://account.documents.azure.com", "dbs/db/colls/other", False),
        ("https://account.documents.azure.com:8081", "dbs/db/colls/products", False),
    ],
)
def test_separate_proxies_merge_only_for_same_physical_container(endpoint, link, same_container):
    first, second = Container(), Container()
    first.client_connection = SimpleNamespace(url_connection="https://account.documents.azure.com/")
    first.container_link = "dbs/db/colls/products"
    second.client_connection = SimpleNamespace(url_connection=endpoint)
    second.container_link = link
    candidate = {"id": "same", "_partition_key": "001", "_score": 0.25}
    duplicate = {**candidate, "_score": 0.75}
    docs = asyncio.run(fetch_candidate_documents([(first, candidate), (second, duplicate)]))
    assert len(docs) == (1 if same_container else 2)
    assert first.reads == [("same", "001")]
    assert second.reads == ([] if same_container else [("same", "001")])
    assert docs[0]["_score"] == 0.25
    assert (container_identity(first) == container_identity(second)) is same_container


def test_undefined_and_null_partitions_remain_distinct(monkeypatch):
    undefined = object()
    monkeypatch.setitem(
        sys.modules, "azure.cosmos.partition_key",
        SimpleNamespace(NonePartitionKeyValue=undefined),
    )
    container = Container()
    asyncio.run(fetch_candidate_documents([
        (container, {"id": "one"}), (container, {"id": "one", "_partition_key": None}),
    ]))
    assert container.reads == [("one", undefined), ("one", None)]


def test_point_reads_are_bounded():
    active = maximum = 0

    class BoundedContainer(Container):
        async def read_item(self, **kwargs):
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            await asyncio.sleep(0)
            active -= 1
            return await super().read_item(**kwargs)

    container = BoundedContainer()
    docs = asyncio.run(fetch_candidate_documents([
        (container, {"id": str(i), "_partition_key": str(i)}) for i in range(50)
    ]))
    assert len(docs) == 50
    assert maximum == 16
    assert [doc["id"] for doc in docs] == [str(i) for i in range(50)]


def test_point_read_failures_propagate():
    container = Container()
    container.read_item = AsyncMock(side_effect=RuntimeError("point read failed"))
    with pytest.raises(RuntimeError, match="point read failed"):
        asyncio.run(fetch_candidate_documents([
            (container, {"id": "one", "_partition_key": "001"}),
        ]))


def test_invalid_candidate_id_is_explicit_error():
    with pytest.raises(ValueError, match="valid document ID"):
        candidate_identity({"_partition_key": "001"})


def test_fulltext_fuses_lightweight_candidates_before_reads():
    container = Container()
    result = asyncio.run(fulltext_search(
        container, ["title", "claims[]"], "fragrance", 2, partition_key_path="/upc",
    ))
    assert result == [
        {"id": "b", "_partition_key": "00b"}, {"id": "c", "_partition_key": "00c"},
    ]
    assert len(container.queries) == 2
    for query in container.queries:
        assert 'SELECT TOP 2 c.id, c["upc"] AS _partition_key FROM c' in query["query"]
        assert "SELECT TOP 2 *" not in query["query"]
    assert "ORDER BY RANK FullTextScore" in container.queries[0]["query"]
    assert "WHERE EXISTS(" in container.queries[1]["query"]
    assert container.reads == []


def test_fulltext_does_not_merge_same_id_from_different_partitions():
    class PartitionedContainer(Container):
        def query_items(self, **kwargs):
            async def rows():
                yield {"id": "same", "_partition_key": "001"}
                yield {"id": "same", "_partition_key": "002"}
            return rows()

    result = asyncio.run(fulltext_search(
        PartitionedContainer(), ["title", "claims[]"], "fragrance", 2,
        partition_key_path="/upc",
    ))
    assert [doc["_partition_key"] for doc in result] == ["001", "002"]


def load_functions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(
        node, (ast.FunctionDef, ast.AsyncFunctionDef)
    ) and node.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


@pytest.mark.parametrize("source_alias", [False, True])
@pytest.mark.parametrize("optimized", [False, True])
def test_tool_use_fetches_only_merged_candidates_before_reranking(source_alias, optimized):
    container = Container()
    containers = {"products": container}
    ranker = AsyncMock(side_effect=lambda query, texts, top_k: texts[:top_k])
    namespace = {
        "asyncio": asyncio, "json": json, "re": re, "_ck": Mock(return_value=0.0),
        "_SAFE_FIELD_RE": re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$"),
        "candidate_projection": candidate_projection,
        "fetch_candidate_documents": fetch_candidate_documents,
        "fulltext_search": fulltext_search, "tool_use_rerank": ranker,
        "fulltext_search_options": fulltext_search_options,
        "tool_use_embed": AsyncMock(return_value=[0.1]),
        "_tool_use_source_cfg": {"products": {"search_k": 2, "fulltext_search_k": 2}},
        "_tool_use_source_embed": {"products": "e"},
        "_tool_use_source_ft": {"products": ["title", "claims[]"]},
        "_tool_use_source_partition": {"products": "/upc"},
        "_tool_use_rerank_mul": 1, "_tool_use_use_hyde": False,
        "_tool_use_all_embed": {"e"},
        "_tool_use_context_fields": [],
    }
    if optimized:
        namespace["_tool_use_source_cfg"]["products"].update({
            "fulltext_query_mode": "consolidated", "fulltext_match_filter": True,
        })
    if source_alias:
        alias = Container()
        for proxy in (container, alias):
            proxy.client_connection = SimpleNamespace(url_connection="https://example.documents.azure.com/")
            proxy.container_link = "dbs/db/colls/products"
        containers["alias"] = alias
        for name in ("cfg", "embed", "ft", "partition"):
            mapping = namespace[f"_tool_use_source_{name}"]
            mapping["alias"] = mapping["products"]
    load_functions(ROOT / "dynamic_retriever.py", {
        "tool_use_vec_search", "_tool_use_fulltext_search", "tool_use_do_search", "tool_use_fmt",
    }, namespace)
    result = json.loads(asyncio.run(namespace["tool_use_do_search"](
        "fragrance", containers,
    )))
    assert container.reads == [("a", "00a"), ("b", "00b"), ("c", "00c")]
    assert [doc["docid"] for doc in result] == ["a", "b", "c"]
    assert all("Full product" in doc["text"] for doc in result)
    assert all("e: [" not in doc["text"] for doc in result)
    assert ranker.await_args.args[1] == [doc["text"] for doc in result]
    assert all("SELECT TOP @k c," not in q["query"] for q in container.queries)
    assert '_score' in container.queries[0]["query"]
    if optimized:
        assert "WHERE (FullTextContainsAny(c.title" in container.queries[1]["query"]
    if source_alias:
        assert alias.reads == []


@pytest.mark.parametrize("shared_container", [False, True, "separate-proxy"])
@pytest.mark.parametrize("ranker_fail_on_error", [None, False, True])
@pytest.mark.parametrize("optimized", [False, True])
def test_decomposed_fetch_preserves_ft_precedence_and_vector_similarity(
    shared_container, ranker_fail_on_error, optimized,
):
    path = ROOT / "utils" / "cosmos_retriever.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    original = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "CombinedRetriever")
    original.body = [
        node for node in original.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in {
            "_is_safe_field_path", "_vector_search", "_fulltext_search", "_format_doc", "retrieve",
        }
    ]
    namespace = {
        "Any": Any, "asyncio": asyncio, "copy": copy, "logging": logging, "re": re,
        "fulltext_search": fulltext_search, "candidate_identity": candidate_identity,
        "fulltext_search_options": fulltext_search_options,
        "candidate_projection": candidate_projection, "fetch_candidate_results": fetch_candidate_results,
        "container_identity": container_identity,
        "RetrievedChunk": lambda **kwargs: SimpleNamespace(**kwargs),
        "_is_identifier_like_field": lambda name: name in {"id", "upc"},
        "_timing_enabled": lambda: False, "_ck": Mock(return_value=time.perf_counter()),
        "_format_activity_id_note": lambda _: "", "_multi_activity_reason": lambda _: "",
        "RankerUnavailableError": RankerUnavailableError,
        "rerank_documents": AsyncMock(return_value=None), "CONFIG": {},
        "_log_line": Mock(),
    }
    exec(compile(ast.Module(body=[original], type_ignores=[]), str(path), "exec"), namespace)
    retriever = namespace["CombinedRetriever"]()
    container = Container()
    retriever._llm = SimpleNamespace(embed=AsyncMock(return_value=[0.1]))
    retriever._sources = [{
        "id": "products", "fulltext_k": 2, "vector_k": 2, "embedding_field": "e",
        "partition_key_path": "/upc", "fulltext_fields": ["title", "claims[]"],
    }]
    if optimized:
        retriever._sources[0].update({
            "fulltext_query_mode": "consolidated", "fulltext_match_filter": True,
        })
    retriever._containers = {"products": container}
    if shared_container:
        retriever._sources.append({**retriever._sources[0], "id": "alias"})
        retriever._containers["alias"] = container
        if shared_container == "separate-proxy":
            alias = Container()
            for proxy in (container, alias):
                proxy.client_connection = SimpleNamespace(url_connection="https://example.documents.azure.com/")
                proxy.container_link = "dbs/db/colls/products"
            retriever._containers["alias"] = alias
    retriever._retrieve_cache = SimpleNamespace(get=Mock(return_value=None), set=Mock())
    retriever._indexed_fulltext_fields = AsyncMock(return_value=["title", "claims[]", "not_selected"])
    retriever._source_context_fields = {}
    retriever._expected_vector_dim = 0
    retriever.k_diverse = retriever.k_ranker = 0
    if ranker_fail_on_error is not None:
        retriever.k_ranker = 1
        retriever._use_ranker = True
        retriever._ranker_account = "test"
        retriever._ranker_region = "test"
        retriever._ranker_access_token = "test-token"
        retriever._ranker_http_client = object()
        retriever._ranker_batch_size = 32
        retriever._ranker_fail_on_error = ranker_fail_on_error
    if ranker_fail_on_error:
        with pytest.raises(RankerUnavailableError, match="fail_on_error"):
            asyncio.run(retriever.retrieve("fragrance"))
        retriever._retrieve_cache.set.assert_not_called()
        namespace["rerank_documents"].assert_awaited_once()
        return
    chunks = asyncio.run(retriever.retrieve("fragrance"))
    if ranker_fail_on_error is False:
        namespace["rerank_documents"].assert_awaited_once()
    assert [chunk.chunk_id for chunk in chunks] == ["b", "c", "a"]
    assert container.reads == [("b", "00b"), ("c", "00c"), ("a", "00a")]
    assert [chunk.similarity for chunk in chunks] == [None, None, 0.8]
    assert [chunk.metadata["_data_source"] for chunk in chunks] == [
        "products_fulltext", "products_fulltext", "products_vector",
    ]
    assert all(len(chunk.metadata["embedding"]) == 1536 for chunk in chunks)
    assert all("Full product" in chunk.text for chunk in chunks)
    assert all("SELECT TOP @k c," not in query["query"] for query in container.queries)
    assert all("not_selected" not in query["query"] for query in container.queries)
    if optimized:
        assert any("WHERE (FullTextContainsAny(c.title" in q["query"] for q in container.queries)
    if shared_container == "separate-proxy":
        assert alias.reads == []
