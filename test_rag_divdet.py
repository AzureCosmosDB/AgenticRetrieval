"""
Unit tests for utility functions in rag_divdet.py.

All tests are pure-logic with no external service dependencies.
The conftest.py at repository root redirects config.yaml -> config.yaml.example.
"""

import rag_divdet
from rag_divdet import (
    LRUCache,
    _format_activity_id_note,
    _is_timing,
    _multi_activity_reason,
)


# ---------------------------------------------------------------------------
# Module public-API smoke tests
# ---------------------------------------------------------------------------


class TestRagDivdetPublicApi:
    """Verify that rag_divdet still exports all expected symbols after the
    cosmos_retriever refactoring."""

    def test_llm_client_exported(self):
        assert hasattr(rag_divdet, "LLMClient")

    def test_lru_cache_exported(self):
        assert hasattr(rag_divdet, "LRUCache")

    def test_data_classes_exported(self):
        for name in ("Question", "RetrievedChunk", "SubQuestionResult", "RoundResult"):
            assert hasattr(rag_divdet, name), f"{name} not found in rag_divdet"

    def test_combined_retriever_re_exported(self):
        """CombinedRetriever is imported into rag_divdet from cosmos_retriever."""
        assert hasattr(rag_divdet, "CombinedRetriever")

    def test_retrieval_sources_re_exported(self):
        assert hasattr(rag_divdet, "RETRIEVAL_SOURCES")
        assert isinstance(rag_divdet.RETRIEVAL_SOURCES, list)

    def test_config_loaded(self):
        assert isinstance(rag_divdet.CONFIG, dict)
        assert "cosmos" in rag_divdet.CONFIG
        assert "llm" in rag_divdet.CONFIG


# ---------------------------------------------------------------------------
# _is_timing
# ---------------------------------------------------------------------------


class TestIsTiming:
    def test_false_by_default(self):
        # _TIMING starts as False; _is_timing() must reflect that.
        original = rag_divdet._TIMING
        try:
            rag_divdet._TIMING = False
            assert _is_timing() is False
        finally:
            rag_divdet._TIMING = original

    def test_true_when_flag_set(self):
        original = rag_divdet._TIMING
        try:
            rag_divdet._TIMING = True
            assert _is_timing() is True
        finally:
            rag_divdet._TIMING = original


# ---------------------------------------------------------------------------
# LRUCache
# ---------------------------------------------------------------------------


class TestLRUCache:
    def test_get_returns_none_for_missing_key(self):
        cache = LRUCache(10)
        assert cache.get("missing") is None

    def test_set_and_get(self):
        cache = LRUCache(10)
        cache.set("k", 42)
        assert cache.get("k") == 42

    def test_set_overwrites_existing(self):
        cache = LRUCache(10)
        cache.set("k", 1)
        cache.set("k", 2)
        assert cache.get("k") == 2

    def test_evicts_least_recently_used(self):
        cache = LRUCache(2)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)  # should evict "a"
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3

    def test_get_refreshes_lru_order(self):
        cache = LRUCache(2)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.get("a")       # "a" is now most-recently used
        cache.set("c", 3)    # should evict "b", not "a"
        assert cache.get("a") == 1
        assert cache.get("b") is None
        assert cache.get("c") == 3

    def test_max_size_one(self):
        cache = LRUCache(1)
        cache.set("a", 1)
        cache.set("b", 2)
        assert cache.get("a") is None
        assert cache.get("b") == 2

    def test_max_size_clamped_to_one_for_zero_input(self):
        cache = LRUCache(0)
        assert cache.max_size == 1

    def test_stores_arbitrary_values(self):
        cache = LRUCache(5)
        cache.set("list", [1, 2, 3])
        cache.set("dict", {"x": 1})
        assert cache.get("list") == [1, 2, 3]
        assert cache.get("dict") == {"x": 1}


# ---------------------------------------------------------------------------
# _format_activity_id_note
# ---------------------------------------------------------------------------


class TestFormatActivityIdNote:
    def test_empty_list_returns_empty_string(self):
        assert _format_activity_id_note([]) == ""

    def test_list_of_empty_strings_returns_empty(self):
        assert _format_activity_id_note(["", "", ""]) == ""

    def test_single_id(self):
        result = _format_activity_id_note(["abc-123"])
        assert result == " [ActivityId=abc-123]"

    def test_multiple_unique_ids(self):
        ids = ["id1", "id2", "id3"]
        result = _format_activity_id_note(ids)
        assert result.startswith(" [ActivityIds=")
        for id_ in ids:
            assert id_ in result

    def test_duplicate_ids_deduplicated(self):
        result = _format_activity_id_note(["same", "same", "same"])
        assert result == " [ActivityId=same]"

    def test_more_than_three_ids_shows_overflow(self):
        ids = ["a", "b", "c", "d", "e"]
        result = _format_activity_id_note(ids)
        assert "+2 more" in result

    def test_exactly_three_ids_no_overflow(self):
        ids = ["a", "b", "c"]
        result = _format_activity_id_note(ids)
        assert "more" not in result


# ---------------------------------------------------------------------------
# _multi_activity_reason
# ---------------------------------------------------------------------------


class TestMultiActivityReason:
    def _meta(self, activity_id="", partition_range_id="", physical_partition_id="",
               has_continuation="", retry_after_ms=""):
        return {
            "activity_id": activity_id,
            "partition_range_id": partition_range_id,
            "physical_partition_id": physical_partition_id,
            "has_continuation": has_continuation,
            "retry_after_ms": retry_after_ms,
        }

    def test_empty_list_returns_empty(self):
        assert _multi_activity_reason([]) == ""

    def test_single_activity_id_returns_empty(self):
        result = _multi_activity_reason([self._meta("id1"), self._meta("id1")])
        assert result == ""

    def test_multiple_activity_ids_returns_reason(self):
        metas = [self._meta("id1"), self._meta("id2")]
        result = _multi_activity_reason(metas)
        assert result.startswith(" [Reason:")

    def test_partition_range_fanout_detected(self):
        metas = [
            self._meta("id1", partition_range_id="range-0"),
            self._meta("id2", partition_range_id="range-1"),
        ]
        result = _multi_activity_reason(metas)
        assert "partition key ranges" in result

    def test_physical_partition_fanout_detected(self):
        metas = [
            self._meta("id1", physical_partition_id="pp-0"),
            self._meta("id2", physical_partition_id="pp-1"),
        ]
        result = _multi_activity_reason(metas)
        assert "physical partitions" in result

    def test_continuation_detected(self):
        metas = [
            self._meta("id1"),
            self._meta("id2", has_continuation="1"),
        ]
        result = _multi_activity_reason(metas)
        assert "continuation" in result

    def test_retry_after_detected(self):
        metas = [
            self._meta("id1"),
            self._meta("id2", retry_after_ms="500"),
        ]
        result = _multi_activity_reason(metas)
        assert "retry-after" in result

    def test_fallback_reason_when_no_specific_reason(self):
        metas = [self._meta("id1"), self._meta("id2")]
        result = _multi_activity_reason(metas)
        assert "multiple backend executions" in result
