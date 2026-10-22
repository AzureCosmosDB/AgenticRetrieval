import asyncio
import re

import pytest

from utils.fulltext import (
    fulltext_search,
    fulltext_search_options,
    fulltext_search_single_field,
)


class Container:
    def __init__(self):
        self.queries = []

    def query_items(self, **kwargs):
        self.queries.append(kwargs)

        async def rows():
            kwargs["response_hook"]({"x-ms-request-charge": "2.5"}, {})
            yield {"id": "one", "_partition_key": "001"}
            kwargs["response_hook"]({"x-ms-request-charge": "3.0"}, {})
            yield {"id": "two", "_partition_key": "002"}

        return rows()


@pytest.mark.parametrize(
    "config",
    [
        {"fulltext_query_mode": "invalid"},
        {"fulltext_query_mode": None},
        {"fulltext_match_filter": "false"},
        {"fulltext_match_filter": 1},
    ],
)
def test_invalid_options_fail_explicitly(config):
    with pytest.raises(ValueError, match="retrieval.fulltext_"):
        fulltext_search_options(config)


def test_legacy_defaults_are_preserved():
    assert fulltext_search_options({}) == {
        "fulltext_query_mode": "per_field", "fulltext_match_filter": False,
    }


def test_filtered_scalar_uses_same_terms_for_any_match_and_legacy_chunk_ranking():
    container = Container()
    asyncio.run(fulltext_search_single_field(
        container, "details.title", "the fragrance free sensitive skin gentle cream", 3,
        fulltext_match_filter=True,
    ))
    sql = container.queries[0]["query"]
    assert 'WHERE FullTextContainsAny(c.details.title, "fragrance", "free", "sensitive", "skin", "gentle", "cream")' in sql
    assert 'ORDER BY RANK RRF(FullTextScore(c.details.title, "fragrance", "free", "sensitive", "skin", "gentle"), FullTextScore(c.details.title, "cream"))' in sql


@pytest.mark.parametrize("match_filter", [False, True])
def test_consolidation_uses_one_scalar_query_and_retains_each_array_query(match_filter):
    container = Container()
    rows = asyncio.run(fulltext_search(
        container, ["title", "brand", "claims[]", "details.actives[]", "title"],
        "fragrance free", 3, partition_key_path="/upc",
        fulltext_query_mode="consolidated", fulltext_match_filter=match_filter,
    ))
    assert len(rows) == 2
    assert len(container.queries) == 3
    scalar = container.queries[0]["query"]
    assert scalar.startswith('SELECT TOP 3 c.id, c["upc"] AS _partition_key FROM c ')
    assert 'ORDER BY RANK RRF(FullTextScore(c.title, "fragrance", "free"), FullTextScore(c.brand, "fragrance", "free"))' in scalar
    if match_filter:
        assert 'WHERE (FullTextContainsAny(c.title, "fragrance", "free") OR FullTextContainsAny(c.brand, "fragrance", "free"))' in scalar
    else:
        assert "WHERE" not in scalar
    for array_query in container.queries[1:]:
        assert "WHERE EXISTS(" in array_query["query"]
        assert "FullTextScore" not in array_query["query"]
        assert "ORDER BY" not in array_query["query"]


def test_consolidated_single_scalar_uses_all_terms_without_rrf():
    container = Container()
    asyncio.run(fulltext_search(
        container, ["title"], "alpha bravo charlie delta echo foxtrot", 3,
        fulltext_query_mode="consolidated",
    ))
    assert container.queries[0]["query"] == (
        'SELECT TOP 3 * FROM c ORDER BY RANK '
        'FullTextScore(c.title, "alpha", "bravo", "charlie", "delta", "echo", "foxtrot")'
    )


def test_consolidated_scores_never_exceed_service_term_bound():
    container = Container()
    query = " ".join(f"term{i}" for i in range(61))
    asyncio.run(fulltext_search(
        container, ["title", "brand"], query, 3, fulltext_query_mode="consolidated",
    ))
    sql = container.queries[0]["query"]
    assert sql.count("FullTextScore(") == 6
    assert 'FullTextScore(c.title, "term60")' in sql
    assert 'FullTextScore(c.brand, "term60")' in sql
    for args in re.findall(r"FullTextScore\([^,]+, ([^)]+)\)", sql):
        assert len(args.split(", ")) <= 30


@pytest.mark.parametrize("fields", [["claims[]"], ["claims[]", "actives[]"], []])
def test_array_only_and_empty_sources_do_not_issue_scalar_queries(fields):
    container = Container()
    asyncio.run(fulltext_search(
        container, fields, "skin", 3, fulltext_query_mode="consolidated",
    ))
    assert len(container.queries) == len(fields)
    assert all("FullTextScore" not in q["query"] for q in container.queries)


@pytest.mark.parametrize("query,top_k", [("the and", 3), ("skin", 0), ("skin", -1)])
def test_empty_terms_and_disabled_search_do_not_query(query, top_k):
    container = Container()
    assert asyncio.run(fulltext_search(
        container, ["title", "claims[]"], query, top_k, fulltext_query_mode="consolidated",
    )) == []
    assert not container.queries


def test_invalid_field_is_rejected_before_any_query_is_started():
    container = Container()
    with pytest.raises(ValueError, match="Invalid full-text field path"):
        asyncio.run(fulltext_search(
            container, ["title", "claims[].text"], "skin", 3,
            fulltext_query_mode="consolidated",
        ))
    assert not container.queries


def test_metrics_sum_response_charges_per_query_not_shared_container_headers():
    container = Container()
    metrics = []
    asyncio.run(fulltext_search(
        container, ["title", "brand", "claims[]"], "skin", 3,
        fulltext_query_mode="consolidated", metrics_fn=metrics.append,
    ))
    assert len(metrics) == 2
    assert {m.fields for m in metrics} == {("title", "brand"), ("claims[]",)}
    assert all(m.request_charge == 5.5 and m.response_count == 2 for m in metrics)
    assert all(m.succeeded and m.result_count == 2 and m.elapsed_seconds >= 0 for m in metrics)


def test_missing_ru_is_unknown_not_zero():
    class NoHeaders(Container):
        def query_items(self, **kwargs):
            async def rows():
                yield {"id": "one"}
            return rows()

    metrics = []
    asyncio.run(fulltext_search(NoHeaders(), ["title"], "skin", 3, metrics_fn=metrics.append))
    assert metrics[0].request_charge is None


def test_partially_missing_ru_is_not_reported_as_a_complete_charge():
    class PartialHeaders(Container):
        def query_items(self, **kwargs):
            async def rows():
                kwargs["response_hook"]({"x-ms-request-charge": "2"}, {})
                yield {"id": "one"}
                kwargs["response_hook"]({}, {})
            return rows()

    metrics = []
    asyncio.run(fulltext_search(PartialHeaders(), ["title"], "skin", 3, metrics_fn=metrics.append))
    assert metrics[0].response_count == 2
    assert metrics[0].request_charge is None


def test_query_errors_are_reported_and_propagated(caplog):
    class Broken(Container):
        def query_items(self, **kwargs):
            async def rows():
                kwargs["response_hook"]({"x-ms-request-charge": "1"}, {})
                raise RuntimeError("bad query")
                yield
            return rows()

    metrics = []
    with pytest.raises(RuntimeError, match="bad query"):
        asyncio.run(fulltext_search(
            Broken(), ["title"], "skin", 3, metrics_fn=metrics.append,
        ))
    assert "Full-text query failed" in caplog.text
    assert not metrics[0].succeeded
    assert metrics[0].request_charge == 1


def test_cancellation_propagates():
    class Cancelled(Container):
        def query_items(self, **kwargs):
            async def rows():
                raise asyncio.CancelledError()
                yield
            return rows()

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(fulltext_search(Cancelled(), ["title"], "skin", 3))


def test_failed_query_cancels_and_drains_sibling_queries():
    cancelled = []

    class FailureAndWait(Container):
        def query_items(self, **kwargs):
            async def rows():
                if "c.title" in kwargs["query"]:
                    await asyncio.sleep(0)
                    raise RuntimeError("bad query")
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.append(True)
                yield
            return rows()

    with pytest.raises(RuntimeError, match="bad query"):
        asyncio.run(fulltext_search(FailureAndWait(), ["title", "brand"], "skin", 3))
    assert cancelled == [True]
