"""
Unit tests for cosmos_retriever.py.

All tests are pure-logic (no live Cosmos DB or LLM connections required).
The conftest.py at repository root redirects config.yaml -> config.yaml.example
so module-level imports succeed without real credentials.
"""

import asyncio
import copy
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

# rag_divdet must be imported first to correctly resolve the circular import
# between rag_divdet and cosmos_retriever (rag_divdet is the entry-point module
# that drives the mutual initialization).
import rag_divdet  # noqa: F401
import cosmos_retriever
from cosmos_retriever import (
    CombinedRetriever,
    _as_list_of_strings,
    _build_retrieval_sources,
    _legacy_retrieval_sources,
    greedy_log_det_select,
)
from rag_divdet import RetrievedChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_source(
    id_="s1",
    container_name="c1",
    vector_k=5,
    fulltext_k=3,
    fulltext_fields=None,
    partition_key_path="/pk",
):
    return {
        "id": id_,
        "container_name": container_name,
        "vector_k": vector_k,
        "fulltext_k": fulltext_k,
        "fulltext_fields": fulltext_fields or ["title"],
        "partition_key_path": partition_key_path,
    }


# ---------------------------------------------------------------------------
# Import / public-API smoke tests
# ---------------------------------------------------------------------------


class TestModulePublicApi:
    """Verify that cosmos_retriever exposes the expected public symbols."""

    def test_combined_retriever_exported(self):
        assert hasattr(cosmos_retriever, "CombinedRetriever")

    def test_retrieval_sources_exported(self):
        assert hasattr(cosmos_retriever, "RETRIEVAL_SOURCES")
        assert isinstance(cosmos_retriever.RETRIEVAL_SOURCES, list)

    def test_greedy_log_det_select_exported(self):
        assert callable(cosmos_retriever.greedy_log_det_select)

    def test_stopwords_exported(self):
        assert isinstance(cosmos_retriever.STOPWORDS, set)
        assert len(cosmos_retriever.STOPWORDS) > 0

    def test_cosmos_constants_exported(self):
        assert hasattr(cosmos_retriever, "COSMOS_ENDPOINT")
        assert hasattr(cosmos_retriever, "COSMOS_KEY")
        assert hasattr(cosmos_retriever, "DATABASE_NAME")


# ---------------------------------------------------------------------------
# _as_list_of_strings
# ---------------------------------------------------------------------------


class TestAsListOfStrings:
    def test_returns_list_of_strings(self):
        assert _as_list_of_strings(["a", "b", "c"]) == ["a", "b", "c"]

    def test_strips_whitespace(self):
        assert _as_list_of_strings(["  a  ", " b"]) == ["a", "b"]

    def test_removes_empty_after_strip(self):
        assert _as_list_of_strings(["  ", "", "ok"]) == ["ok"]

    def test_non_list_returns_empty(self):
        assert _as_list_of_strings("not a list") == []
        assert _as_list_of_strings(None) == []
        assert _as_list_of_strings(42) == []
        assert _as_list_of_strings({}) == []

    def test_empty_list(self):
        assert _as_list_of_strings([]) == []

    def test_coerces_non_string_items(self):
        result = _as_list_of_strings([1, 2.5, True])
        assert result == ["1", "2.5", "True"]


# ---------------------------------------------------------------------------
# _legacy_retrieval_sources
# ---------------------------------------------------------------------------


class TestLegacyRetrievalSources:
    _CONFIG = {
        "retrieval": {"k_structured": 10, "k_unstructured": 20, "k_fulltext": 5},
        "cosmos": {
            "structured_container": "struct_ctr",
            "structured_partition_key_path": "/pk",
            "unstructured_container": "unstruct_ctr",
            "unstructured_partition_key_path": "/pk",
        },
    }

    def test_returns_two_sources(self):
        sources = _legacy_retrieval_sources(self._CONFIG)
        assert len(sources) == 2

    def test_structured_source(self):
        sources = _legacy_retrieval_sources(self._CONFIG)
        s = next(s for s in sources if s["id"] == "structured")
        assert s["container_name"] == "struct_ctr"
        assert s["vector_k"] == 10
        assert s["fulltext_k"] == 5
        assert "designation" in s["fulltext_fields"]

    def test_unstructured_source(self):
        sources = _legacy_retrieval_sources(self._CONFIG)
        u = next(s for s in sources if s["id"] == "unstructured")
        assert u["container_name"] == "unstruct_ctr"
        assert u["vector_k"] == 20
        assert u["fulltext_k"] == 0
        assert u["fulltext_fields"] == []

    def test_zero_k_values_default(self):
        cfg = {"retrieval": {}, "cosmos": {}}
        sources = _legacy_retrieval_sources(cfg)
        for s in sources:
            assert s["vector_k"] == 0
            assert s["fulltext_k"] == 0


# ---------------------------------------------------------------------------
# _build_retrieval_sources
# ---------------------------------------------------------------------------


class TestBuildRetrievalSources:
    def _config_with_sources(self, sources_list):
        return {"cosmos": {"sources": sources_list}, "retrieval": {}}

    def test_falls_back_to_legacy_when_no_sources_key(self):
        cfg = {
            "cosmos": {
                "structured_container": "sc",
                "unstructured_container": "uc",
            },
            "retrieval": {"k_structured": 1, "k_unstructured": 2, "k_fulltext": 0},
        }
        result = _build_retrieval_sources(cfg)
        ids = [s["id"] for s in result]
        assert "structured" in ids
        assert "unstructured" in ids

    def test_falls_back_to_legacy_when_sources_is_empty_list(self):
        result = _build_retrieval_sources(self._config_with_sources([]))
        ids = [s["id"] for s in result]
        assert "structured" in ids

    def test_uses_configured_sources_when_present(self):
        sources = [
            {
                "id": "my_source",
                "container_name": "my_container",
                "partition_key_path": "/pk",
                "retrieval": {"vector_k": 10, "fulltext_k": 5, "fulltext_fields": ["body"]},
            }
        ]
        result = _build_retrieval_sources(self._config_with_sources(sources))
        assert len(result) == 1
        s = result[0]
        assert s["id"] == "my_source"
        assert s["container_name"] == "my_container"
        assert s["vector_k"] == 10
        assert s["fulltext_k"] == 5
        assert s["fulltext_fields"] == ["body"]

    def test_auto_generates_id_when_missing(self):
        sources = [{"container_name": "c1", "retrieval": {}}]
        result = _build_retrieval_sources(self._config_with_sources(sources))
        assert result[0]["id"] == "source_1"

    def test_multiple_sources_preserved(self):
        sources = [
            {"id": "a", "container_name": "ca", "retrieval": {"vector_k": 3}},
            {"id": "b", "container_name": "cb", "retrieval": {"fulltext_k": 7}},
        ]
        result = _build_retrieval_sources(self._config_with_sources(sources))
        assert len(result) == 2
        assert result[0]["id"] == "a"
        assert result[1]["id"] == "b"


# ---------------------------------------------------------------------------
# greedy_log_det_select
# ---------------------------------------------------------------------------


class TestGreedyLogDetSelect:
    def _orthonormal_vectors(self, n, dim):
        """Return n orthonormal vectors of dimension dim (n <= dim)."""
        mat = np.eye(dim, dtype=np.float32)
        return mat[:n]

    def test_returns_k_indices(self):
        vecs = self._orthonormal_vectors(5, 5)
        qvec = np.array([1.0, 0, 0, 0, 0], dtype=np.float32)
        result = greedy_log_det_select(vecs, qvec, k=3)
        assert len(result) == 3

    def test_indices_are_unique(self):
        vecs = self._orthonormal_vectors(5, 5)
        qvec = np.zeros(5, dtype=np.float32)
        qvec[0] = 1.0
        result = greedy_log_det_select(vecs, qvec, k=3)
        assert len(set(result)) == len(result)

    def test_indices_in_valid_range(self):
        n = 6
        vecs = self._orthonormal_vectors(n, n)
        qvec = np.ones(n, dtype=np.float32) / np.sqrt(n)
        result = greedy_log_det_select(vecs, qvec, k=4)
        assert all(0 <= i < n for i in result)

    def test_k_greater_or_equal_n_returns_all(self):
        vecs = self._orthonormal_vectors(3, 3)
        qvec = np.array([1.0, 0, 0], dtype=np.float32)
        result = greedy_log_det_select(vecs, qvec, k=5)
        assert sorted(result) == [0, 1, 2]

    def test_k_equals_n_returns_all(self):
        vecs = self._orthonormal_vectors(4, 4)
        qvec = np.ones(4, dtype=np.float32) / 2
        result = greedy_log_det_select(vecs, qvec, k=4)
        assert sorted(result) == [0, 1, 2, 3]

    def test_eta_regularisation_does_not_crash(self):
        vecs = self._orthonormal_vectors(4, 4)
        qvec = np.array([1.0, 0, 0, 0], dtype=np.float32)
        result = greedy_log_det_select(vecs, qvec, k=2, eta=0.1)
        assert len(result) == 2

    def test_rescale_power_does_not_crash(self):
        vecs = self._orthonormal_vectors(4, 4)
        qvec = np.array([1.0, 0, 0, 0], dtype=np.float32)
        result = greedy_log_det_select(vecs, qvec, k=2, rescale_power=2.0)
        assert len(result) == 2

    def test_prefers_diverse_over_similar_vectors(self):
        """Two identical vectors and one orthogonal: diverse selection should
        prefer the orthogonal one over the duplicate."""
        v0 = np.array([1.0, 0.0], dtype=np.float32)
        v1 = np.array([1.0, 0.0], dtype=np.float32)  # same as v0
        v2 = np.array([0.0, 1.0], dtype=np.float32)  # orthogonal
        vecs = np.stack([v0, v1, v2])
        qvec = np.array([1.0, 0.0], dtype=np.float32)
        result = greedy_log_det_select(vecs, qvec, k=2)
        # The two chosen should include v2 (index 2) for maximum diversity
        assert 2 in result


# ---------------------------------------------------------------------------
# CombinedRetriever – construction and pure-sync methods
# ---------------------------------------------------------------------------


class TestCombinedRetrieverInit:
    def test_empty_sources_gives_zero_counts(self):
        r = CombinedRetriever([])
        assert r.source_count == 0
        assert r.total_fulltext_k == 0
        assert r.total_vector_k == 0

    def test_source_count_excludes_missing_container(self):
        # Source with no container_name should be silently dropped
        r = CombinedRetriever([{"id": "bad", "container_name": ""}])
        assert r.source_count == 0

    def test_source_count_with_valid_sources(self):
        sources = [_make_source("s1"), _make_source("s2")]
        r = CombinedRetriever(sources)
        assert r.source_count == 2

    def test_total_vector_k(self):
        sources = [_make_source(vector_k=10), _make_source(id_="s2", container_name="c2", vector_k=5)]
        r = CombinedRetriever(sources)
        assert r.total_vector_k == 15

    def test_total_fulltext_k(self):
        sources = [_make_source(fulltext_k=3), _make_source(id_="s2", container_name="c2", fulltext_k=7)]
        r = CombinedRetriever(sources)
        assert r.total_fulltext_k == 10

    def test_fulltext_k_override(self):
        sources = [_make_source(fulltext_k=3)]
        r = CombinedRetriever(sources, fulltext_k_override=99)
        assert r.total_fulltext_k == 99

    def test_k_diverse_stored(self):
        r = CombinedRetriever([], k_diverse=42)
        assert r.k_diverse == 42

    def test_eta_stored(self):
        r = CombinedRetriever([], eta=0.5)
        assert r.eta == 0.5

    def test_rescale_power_stored(self):
        r = CombinedRetriever([], rescale_power=3.0)
        assert r.rescale_power == 3.0


class TestIsSafeFieldPath:
    def test_simple_name(self):
        assert CombinedRetriever._is_safe_field_path("title") is True

    def test_dotted_path(self):
        assert CombinedRetriever._is_safe_field_path("meta.source") is True

    def test_underscore_allowed(self):
        assert CombinedRetriever._is_safe_field_path("field_name") is True

    def test_leading_digit_rejected(self):
        assert CombinedRetriever._is_safe_field_path("1field") is False

    def test_space_rejected(self):
        assert CombinedRetriever._is_safe_field_path("field name") is False

    def test_hyphen_rejected(self):
        assert CombinedRetriever._is_safe_field_path("field-name") is False

    def test_sql_injection_attempt_rejected(self):
        assert CombinedRetriever._is_safe_field_path("field; DROP TABLE c") is False

    def test_empty_string_rejected(self):
        assert CombinedRetriever._is_safe_field_path("") is False

    def test_multiple_dots(self):
        assert CombinedRetriever._is_safe_field_path("a.b.c") is True


class TestNormalizeSources:
    def _retriever(self):
        return CombinedRetriever([])

    def test_drops_sources_without_container_name(self):
        r = self._retriever()
        result = r._normalize_sources([{"id": "x", "container_name": ""}], None)
        assert result == []

    def test_preserves_valid_source(self):
        src = _make_source("s1", "c1", vector_k=5, fulltext_k=2, fulltext_fields=["body"])
        r = self._retriever()
        result = r._normalize_sources([src], None)
        assert len(result) == 1
        assert result[0]["id"] == "s1"
        assert result[0]["container_name"] == "c1"
        assert result[0]["vector_k"] == 5
        assert result[0]["fulltext_k"] == 2

    def test_fulltext_k_override_applied(self):
        src = _make_source(fulltext_k=3)
        r = self._retriever()
        result = r._normalize_sources([src], fulltext_k_override=10)
        assert result[0]["fulltext_k"] == 10

    def test_unsafe_fulltext_field_filtered_out(self):
        src = _make_source(fulltext_fields=["safe_field", "bad field!", "another.valid"])
        r = self._retriever()
        result = r._normalize_sources([src], None)
        fields = result[0]["fulltext_fields"]
        assert "safe_field" in fields
        assert "another.valid" in fields
        assert "bad field!" not in fields

    def test_negative_k_clamped_to_zero(self):
        src = _make_source(vector_k=-5, fulltext_k=-1)
        r = self._retriever()
        result = r._normalize_sources([src], None)
        assert result[0]["vector_k"] == 0
        assert result[0]["fulltext_k"] == 0

    def test_auto_id_when_missing(self):
        src = {"container_name": "c1"}
        r = self._retriever()
        result = r._normalize_sources([src], None)
        assert result[0]["id"] == "source_1"


class TestFormatDoc:
    def _retriever(self):
        return CombinedRetriever([])

    def test_basic_fields_included(self):
        doc = {"id": "doc1", "title": "Hello", "content": "World"}
        r = self._retriever()
        chunk = r._format_doc(doc, "my_source")
        assert "Hello" in chunk.text
        assert "World" in chunk.text

    def test_system_fields_excluded(self):
        doc = {
            "id": "doc1",
            "title": "Keep",
            "_rid": "EXCLUDE",
            "_self": "EXCLUDE",
            "_etag": "EXCLUDE",
            "_attachments": "EXCLUDE",
            "_ts": 12345,
        }
        r = self._retriever()
        chunk = r._format_doc(doc, "src")
        assert "EXCLUDE" not in chunk.text

    def test_embedding_excluded_from_text(self):
        doc = {"id": "doc1", "title": "Hi", "e": [0.1, 0.2], "_score": 0.9}
        r = self._retriever()
        chunk = r._format_doc(doc, "src")
        assert "[0.1" not in chunk.text

    def test_chunk_id_set(self):
        doc = {"id": "my-id-123", "title": "T"}
        chunk = self._retriever()._format_doc(doc, "src")
        assert chunk.chunk_id == "my-id-123"

    def test_similarity_computed_from_score(self):
        doc = {"id": "x", "title": "T", "_score": 0.3}
        chunk = self._retriever()._format_doc(doc, "src")
        assert chunk.similarity == pytest.approx(0.7)

    def test_similarity_none_when_no_score(self):
        doc = {"id": "x", "title": "T"}
        chunk = self._retriever()._format_doc(doc, "src")
        assert chunk.similarity is None

    def test_data_source_in_metadata(self):
        doc = {"id": "x", "title": "T"}
        chunk = self._retriever()._format_doc(doc, "test_source")
        assert chunk.metadata["_data_source"] == "test_source"

    def test_embedding_stored_in_metadata(self):
        emb = [0.1, 0.2, 0.3]
        doc = {"id": "x", "title": "T", "e": emb}
        chunk = self._retriever()._format_doc(doc, "src")
        assert chunk.metadata["embedding"] == emb

    def test_empty_values_not_included(self):
        doc = {"id": "x", "title": "Present", "summary": ""}
        chunk = self._retriever()._format_doc(doc, "src")
        assert "summary" not in chunk.text.lower()


# ---------------------------------------------------------------------------
# CombinedRetriever retrieve – caching (no Cosmos connection needed)
# ---------------------------------------------------------------------------


class TestCombinedRetrieverCache:
    """Test retrieve() cache hit/miss behaviour using a mock LLM."""

    def _make_retriever(self):
        r = CombinedRetriever([])  # no sources → no DB calls
        # Manually inject a mock LLM so the retriever is "initialized"
        mock_llm = MagicMock()
        mock_llm.embed = AsyncMock(return_value=[0.1] * 8)
        r._llm = mock_llm
        return r

    def test_retrieve_raises_when_not_initialized(self):
        r = CombinedRetriever([])
        with pytest.raises(RuntimeError, match="not initialized"):
            asyncio.run(r.retrieve("test"))

    def test_retrieve_returns_empty_list_with_no_sources(self):
        r = self._make_retriever()
        result = asyncio.run(r.retrieve("hello"))
        assert result == []

    def test_retrieve_caches_result(self):
        r = self._make_retriever()
        asyncio.run(r.retrieve("hello"))
        # Second call should hit the cache (LLM.embed should only be called
        # for sources with vector_k > 0, and we have no sources, so not called
        # at all — but the cache avoids any repeated work)
        cached = r._retrieve_cache.get("hello")
        assert isinstance(cached, list)

    def test_retrieve_returns_deep_copy_not_same_object(self):
        r = self._make_retriever()
        first = asyncio.run(r.retrieve("hello"))
        second = asyncio.run(r.retrieve("hello"))
        assert first is not second
