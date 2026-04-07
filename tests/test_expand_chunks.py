"""Tests for CombinedRetriever._expand_chunks."""

import pytest

from agentic_retriever import RetrievedChunk
from utils.cosmos_retriever import CombinedRetriever, RETRIEVAL_SOURCES


def _build_retriever(chunking_sources: list[dict] | None = None) -> CombinedRetriever:
    """Build a retriever with custom chunking config for testing."""
    if chunking_sources is None:
        chunking_sources = RETRIEVAL_SOURCES
    retriever = CombinedRetriever(
        retrieval_sources=chunking_sources,
        k_diverse=0,
        k_ranker=0,
    )
    return retriever


def _make_source_config(
    source_id: str = "src_1",
    container_name: str = "test-container",
    chunking_enabled: bool = True,
    chunk_field: str = "field",
    embedding_field: str = "embedding",
) -> dict:
    return {
        "id": source_id,
        "container_name": container_name,
        "partition_key_path": "/pk",
        "embedding_field": embedding_field,
        "vector_k": 10,
        "fulltext_k": 5,
        "fulltext_fields": ["field"],
        "chunking_enabled": chunking_enabled,
        "chunk_field": chunk_field,
    }


def _make_chunk(
    chunk_id: str = "doc1",
    source_id: str = "src_1",
    raw_doc: dict | None = None,
    embedding: list[float] | None = None,
) -> RetrievedChunk:
    if raw_doc is None:
        raw_doc = {
            "id": chunk_id,
            "pk": "partition1",
            "field": "Widget A",
            "description": "A useful widget",
            "features": ["fast", "durable"],
        }
    return RetrievedChunk(
        chunk_id=chunk_id,
        text="original text",
        similarity=0.9,
        metadata={
            "_data_source": f"{source_id}_vector",
            "_source_id": source_id,
            "embedding": embedding,
            "_raw_doc": raw_doc,
        },
    )


class TestExpandChunksBasic:

    def test_scalar_field_expanded(self):
        """A scalar field should produce one expanded chunk."""
        retriever = _build_retriever([_make_source_config()])
        raw_doc = {
            "id": "d1", "pk": "p1",
            "field": "Widget A",
            "description": "A useful widget",
        }
        chunks = [_make_chunk(chunk_id="d1", raw_doc=raw_doc)]
        expanded = retriever._expand_chunks(chunks)

        assert len(expanded) == 1
        assert "Description: A useful widget" in expanded[0].text
        assert "Field: Widget A" in expanded[0].text

    def test_list_field_expanded_per_element(self):
        """A list field should produce one chunk per element."""
        retriever = _build_retriever([_make_source_config()])
        raw_doc = {
            "id": "d1", "pk": "p1",
            "field": "Widget A",
            "features": ["fast", "durable", "lightweight"],
        }
        chunks = [_make_chunk(chunk_id="d1", raw_doc=raw_doc)]
        expanded = retriever._expand_chunks(chunks)

        assert len(expanded) == 3
        texts = [c.text for c in expanded]
        assert any("fast" in t for t in texts)
        assert any("durable" in t for t in texts)
        assert any("lightweight" in t for t in texts)

    def test_mixed_fields(self):
        """A doc with a scalar + list field should produce 1 + len(list) chunks."""
        retriever = _build_retriever([_make_source_config()])
        raw_doc = {
            "id": "d1", "pk": "p1",
            "field": "Widget A",
            "description": "A useful widget",
            "features": ["fast", "durable"],
        }
        chunks = [_make_chunk(chunk_id="d1", raw_doc=raw_doc)]
        expanded = retriever._expand_chunks(chunks)

        assert len(expanded) == 3  # 1 scalar + 2 list items

    def test_chunk_field_prefix_present(self):
        """Each expanded chunk should be prefixed with the chunk_field value."""
        retriever = _build_retriever([_make_source_config()])
        raw_doc = {
            "id": "d1", "pk": "p1",
            "field": "Widget A",
            "description": "A useful widget",
        }
        chunks = [_make_chunk(chunk_id="d1", raw_doc=raw_doc)]
        expanded = retriever._expand_chunks(chunks)

        assert len(expanded) == 1
        assert expanded[0].text.startswith("Field: Widget A\n")


class TestExpandChunksExclusions:

    def test_chunk_field_not_expanded(self):
        """The chunk_field itself should not produce its own expanded chunk."""
        retriever = _build_retriever([_make_source_config(chunk_field="field")])
        raw_doc = {
            "id": "d1", "pk": "p1",
            "field": "Widget A",
            "description": "Some desc",
        }
        chunks = [_make_chunk(chunk_id="d1", raw_doc=raw_doc)]
        expanded = retriever._expand_chunks(chunks)

        # Only "description" should produce a chunk, not "field"
        assert len(expanded) == 1
        assert "Description:" in expanded[0].text
        # "field" should only appear as prefix, not as its own field line
        lines = expanded[0].text.split("\n")
        field_lines = [l for l in lines if l.startswith("Field:")]
        assert len(field_lines) == 1  # Only the prefix

    def test_embedding_field_excluded(self):
        """The embedding field should never be expanded into chunks."""
        retriever = _build_retriever([_make_source_config(embedding_field="e")])
        raw_doc = {
            "id": "d1", "pk": "p1",
            "field": "Widget A",
            "description": "Some desc",
            "e": [0.1, 0.2, 0.3],
        }
        chunks = [_make_chunk(chunk_id="d1", source_id="src_1", raw_doc=raw_doc)]
        expanded = retriever._expand_chunks(chunks)

        texts = " ".join(c.text for c in expanded)
        assert "0.1" not in texts
        assert "0.2" not in texts

    def test_builtin_fields_excluded(self):
        """Fields like _rid, _self, _etag, _ts, _attachments, _score should be excluded."""
        retriever = _build_retriever([_make_source_config()])
        raw_doc = {
            "id": "d1", "pk": "p1",
            "field": "Widget A",
            "description": "Some desc",
            "_rid": "abc",
            "_self": "dbs/...",
            "_etag": "\"000\"",
            "_ts": 1234567890,
            "_attachments": "attachments/",
            "_score": 0.5,
        }
        chunks = [_make_chunk(chunk_id="d1", raw_doc=raw_doc)]
        expanded = retriever._expand_chunks(chunks)

        texts = " ".join(c.text for c in expanded)
        assert "_rid" not in texts.lower()
        assert "_self" not in texts.lower()
        assert "_etag" not in texts.lower()

    def test_empty_values_skipped(self):
        """Fields with empty/falsy values should not produce chunks."""
        retriever = _build_retriever([_make_source_config()])
        raw_doc = {
            "id": "d1", "pk": "p1",
            "field": "Widget A",
            "description": "",
            "notes": None,
            "tags": [],
        }
        chunks = [_make_chunk(chunk_id="d1", raw_doc=raw_doc)]
        expanded = retriever._expand_chunks(chunks)

        # No expandable fields → falls back to original chunk
        assert len(expanded) == 1
        assert expanded[0].chunk_id == "d1"


class TestExpandChunksMetadata:

    def test_embedding_not_in_expanded_metadata(self):
        """Expanded chunks should not carry embedding in metadata."""
        retriever = _build_retriever([_make_source_config()])
        raw_doc = {
            "id": "d1", "pk": "p1",
            "field": "Widget A",
            "description": "Some desc",
        }
        chunks = [_make_chunk(chunk_id="d1", raw_doc=raw_doc, embedding=[0.1, 0.2])]
        expanded = retriever._expand_chunks(chunks)

        for c in expanded:
            assert "embedding" not in c.metadata

    def test_raw_doc_not_in_expanded_metadata(self):
        """Expanded chunks should not carry _raw_doc in metadata."""
        retriever = _build_retriever([_make_source_config()])
        raw_doc = {
            "id": "d1", "pk": "p1",
            "field": "Widget A",
            "description": "Some desc",
        }
        chunks = [_make_chunk(chunk_id="d1", raw_doc=raw_doc)]
        expanded = retriever._expand_chunks(chunks)

        for c in expanded:
            assert "_raw_doc" not in c.metadata

    def test_source_id_preserved(self):
        """Expanded chunks should preserve _source_id."""
        retriever = _build_retriever([_make_source_config()])
        raw_doc = {
            "id": "d1", "pk": "p1",
            "field": "Widget A",
            "description": "Some desc",
        }
        chunks = [_make_chunk(chunk_id="d1", source_id="src_1", raw_doc=raw_doc)]
        expanded = retriever._expand_chunks(chunks)

        for c in expanded:
            assert c.metadata.get("_source_id") == "src_1"


class TestExpandChunksDisabled:

    def test_chunking_disabled_passthrough(self):
        """Chunks from sources with chunking disabled should pass through unchanged."""
        retriever = _build_retriever([_make_source_config(chunking_enabled=False)])
        chunks = [_make_chunk(chunk_id="d1")]
        expanded = retriever._expand_chunks(chunks)

        assert len(expanded) == 1
        assert expanded[0].text == "original text"

    def test_no_chunking_sources_passthrough(self):
        """If no sources have chunking enabled, all chunks pass through."""
        retriever = _build_retriever([
            _make_source_config(source_id="s1", chunking_enabled=False),
            _make_source_config(source_id="s2", chunking_enabled=False),
        ])
        chunks = [
            _make_chunk(chunk_id="d1", source_id="s1"),
            _make_chunk(chunk_id="d2", source_id="s2"),
        ]
        expanded = retriever._expand_chunks(chunks)

        assert len(expanded) == 2
        assert all(c.text == "original text" for c in expanded)

    def test_mixed_sources_only_enabled_expanded(self):
        """Only chunks from chunking-enabled sources should be expanded."""
        retriever = _build_retriever([
            _make_source_config(source_id="s1", chunking_enabled=True, chunk_field="field"),
            _make_source_config(source_id="s2", container_name="other", chunking_enabled=False),
        ])
        raw_doc = {
            "id": "d1", "pk": "p1",
            "field": "Widget A",
            "description": "desc",
            "features": ["a", "b"],
        }
        chunks = [
            _make_chunk(chunk_id="d1", source_id="s1", raw_doc=raw_doc),
            _make_chunk(chunk_id="d2", source_id="s2"),
        ]
        expanded = retriever._expand_chunks(chunks)

        # s1: 1 scalar + 2 list items = 3; s2: passthrough = 1
        assert len(expanded) == 4
        s2_chunks = [c for c in expanded if c.metadata.get("_source_id") == "s2"]
        assert len(s2_chunks) == 1
        assert s2_chunks[0].text == "original text"


class TestExpandChunksEdgeCases:

    def test_no_raw_doc_passthrough(self):
        """Chunk without _raw_doc in metadata should pass through."""
        retriever = _build_retriever([_make_source_config()])
        chunk = RetrievedChunk(
            chunk_id="d1",
            text="original text",
            similarity=0.9,
            metadata={"_source_id": "src_1", "_data_source": "src_1_vector"},
        )
        expanded = retriever._expand_chunks([chunk])

        assert len(expanded) == 1
        assert expanded[0].text == "original text"

    def test_dict_field_converted_to_string(self):
        """Dict values should be stringified."""
        retriever = _build_retriever([_make_source_config()])
        raw_doc = {
            "id": "d1", "pk": "p1",
            "field": "Widget A",
            "specs": {"weight": "10kg", "color": "blue"},
        }
        chunks = [_make_chunk(chunk_id="d1", raw_doc=raw_doc)]
        expanded = retriever._expand_chunks(chunks)

        assert len(expanded) == 1
        assert "weight" in expanded[0].text
        assert "Specs:" in expanded[0].text

    def test_empty_list_items_skipped(self):
        """Empty strings in a list field should not produce chunks."""
        retriever = _build_retriever([_make_source_config()])
        raw_doc = {
            "id": "d1", "pk": "p1",
            "field": "Widget A",
            "features": ["fast", "", "  ", "durable"],
        }
        chunks = [_make_chunk(chunk_id="d1", raw_doc=raw_doc)]
        expanded = retriever._expand_chunks(chunks)

        assert len(expanded) == 2  # "fast" and "durable" only

    def test_chunk_ids_unique(self):
        """All expanded chunk IDs should be unique."""
        retriever = _build_retriever([_make_source_config()])
        raw_doc = {
            "id": "d1", "pk": "p1",
            "field": "Widget A",
            "description": "desc",
            "features": ["a", "b", "c"],
        }
        chunks = [_make_chunk(chunk_id="d1", raw_doc=raw_doc)]
        expanded = retriever._expand_chunks(chunks)

        ids = [c.chunk_id for c in expanded]
        assert len(ids) == len(set(ids))

    def test_similarity_preserved(self):
        """Expanded chunks should preserve the original similarity score."""
        retriever = _build_retriever([_make_source_config()])
        raw_doc = {
            "id": "d1", "pk": "p1",
            "field": "Widget A",
            "description": "desc",
            "features": ["a", "b"],
        }
        chunks = [_make_chunk(chunk_id="d1", raw_doc=raw_doc)]
        chunks[0].similarity = 0.75
        expanded = retriever._expand_chunks(chunks)

        for c in expanded:
            assert c.similarity == 0.75
