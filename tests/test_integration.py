"""
Integration tests for AgenticRetrieval.

These tests require a valid config.yaml (generated from config.yaml.example
via CI secrets) with live Azure Cosmos DB and Azure OpenAI endpoints.
All tests are skipped automatically when config.yaml is absent.
"""
import asyncio
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
TEST_QUESTIONS_PATH = Path(__file__).parent / "test_questions.json"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro):
    """Run a coroutine in the module-level event loop."""
    return asyncio.get_event_loop().run_until_complete(coro)


def _load_test_questions():
    with open(TEST_QUESTIONS_PATH, encoding="utf-8") as fh:
        return json.load(fh)


# ===========================================================================
# 1. Embedding endpoint
# ===========================================================================


class TestEmbeddingEndpoint:
    """Verify that the Azure OpenAI embedding endpoint is reachable and returns
    non-empty, non-null vectors."""

    def test_embed_returns_nonempty_list(self, llm_client):
        embedding = _run(llm_client.embed("What is the temperature on Mars?"))
        assert embedding is not None, "embed() returned None"
        assert isinstance(embedding, list), "embed() should return a list"
        assert len(embedding) > 0, "embed() returned an empty list"

    def test_embed_values_are_floats(self, llm_client):
        embedding = _run(llm_client.embed("surface temperature"))
        assert all(isinstance(v, float) for v in embedding), (
            "All embedding values should be floats"
        )

    def test_embed_dimension_matches_config(self, llm_client, config):
        embed_cfg = config.get("embedding") or {}
        llm_cfg = config.get("llm", {})
        expected_dim = int((embed_cfg or llm_cfg).get("embed_dimensions") or 0)
        if expected_dim <= 0:
            pytest.skip("embed_dimensions not set in config – skipping dimension check")
        embedding = _run(llm_client.embed("Mars atmosphere"))
        assert len(embedding) == expected_dim, (
            f"Expected {expected_dim}-dimensional embedding, got {len(embedding)}"
        )


# ===========================================================================
# 2. LLM completion endpoint
# ===========================================================================


class TestLLMEndpoint:
    """Verify that the Azure OpenAI LLM completion endpoint is functional."""

    def test_complete_returns_nonempty_string(self, llm_client):
        answer = _run(
            llm_client.complete(
                "Say exactly: HELLO",
                retries=2,
                label="LLM test-ping",
            )
        )
        assert answer is not None, "complete() returned None"
        assert isinstance(answer, str), "complete() should return a string"
        assert len(answer.strip()) > 0, "complete() returned an empty string"

    def test_complete_returns_coherent_response(self, llm_client):
        prompt = (
            "Answer in one sentence: What planet is fourth from the Sun?"
        )
        answer = _run(
            llm_client.complete(prompt, retries=2, label="LLM test-coherence")
        )
        assert "mars" in answer.lower() or "fourth" in answer.lower(), (
            f"Expected a response mentioning Mars or 'fourth', got: {answer!r}"
        )


# ===========================================================================
# 3. Cosmos DB – vector search
# ===========================================================================


class TestVectorSearch:
    """Verify that vector search against each configured Cosmos DB source
    returns non-null, non-empty results."""

    def test_vector_search_returns_results(self, retriever):
        chunks = _run(retriever.retrieve("temperature on Mars"))
        vector_chunks = [
            c for c in chunks if "_vector" in (c.metadata.get("_data_source") or "")
        ]
        assert len(vector_chunks) > 0, (
            "Vector search returned no results for query 'temperature on Mars'. "
            "Expected at least one chunk from a vector-indexed source."
        )

    def test_vector_search_chunks_have_text(self, retriever):
        chunks = _run(retriever.retrieve("Mars atmosphere composition"))
        vector_chunks = [
            c for c in chunks if "_vector" in (c.metadata.get("_data_source") or "")
        ]
        if not vector_chunks:
            pytest.skip("No vector chunks returned – skipping text content check")
        for chunk in vector_chunks:
            assert chunk.text and chunk.text.strip(), (
                f"Vector chunk {chunk.chunk_id!r} has empty text"
            )

    def test_vector_search_per_source(self, retriever, config):
        """Each source with vector_k > 0 must return at least one result."""
        sources = config.get("cosmos", {}).get("sources", [])
        for source in sources:
            retrieval = source.get("retrieval", {})
            vector_k = int(retrieval.get("vector_k") or 0)
            if vector_k <= 0:
                continue
            source_id = source.get("id", "")
            chunks = _run(retriever.retrieve("Mars day length"))
            source_chunks = [
                c
                for c in chunks
                if (c.metadata.get("_data_source") or "").startswith(source_id)
            ]
            assert len(source_chunks) > 0, (
                f"Source '{source_id}' with vector_k={vector_k} returned no chunks"
            )


# ===========================================================================
# 4. Cosmos DB – full-text search
# ===========================================================================


class TestFullTextSearch:
    """Verify that full-text search against each configured Cosmos DB source
    returns non-null, non-empty results when fulltext_k > 0."""

    def test_fulltext_search_returns_results(self, retriever, config):
        sources = config.get("cosmos", {}).get("sources", [])
        has_fulltext = any(
            int((s.get("retrieval") or {}).get("fulltext_k") or 0) > 0
            and (s.get("retrieval") or {}).get("fulltext_fields")
            for s in sources
        )
        if not has_fulltext:
            pytest.skip("No sources with fulltext_k > 0 configured – skipping")

        chunks = _run(retriever.retrieve("Mars temperature"))
        fulltext_chunks = [
            c for c in chunks if "_fulltext" in (c.metadata.get("_data_source") or "")
        ]
        assert len(fulltext_chunks) > 0, (
            "Full-text search returned no results for query 'Mars temperature'."
        )

    def test_fulltext_search_chunks_have_text(self, retriever, config):
        sources = config.get("cosmos", {}).get("sources", [])
        has_fulltext = any(
            int((s.get("retrieval") or {}).get("fulltext_k") or 0) > 0
            and (s.get("retrieval") or {}).get("fulltext_fields")
            for s in sources
        )
        if not has_fulltext:
            pytest.skip("No sources with fulltext_k > 0 configured – skipping")

        chunks = _run(retriever.retrieve("Mars polar ice"))
        fulltext_chunks = [
            c for c in chunks if "_fulltext" in (c.metadata.get("_data_source") or "")
        ]
        if not fulltext_chunks:
            pytest.skip("No full-text chunks returned – skipping text content check")
        for chunk in fulltext_chunks:
            assert chunk.text and chunk.text.strip(), (
                f"Full-text chunk {chunk.chunk_id!r} has empty text"
            )

    def test_fulltext_search_per_source(self, retriever, config):
        """Each source with fulltext_k > 0 must return at least one result."""
        sources = config.get("cosmos", {}).get("sources", [])
        for source in sources:
            retrieval = source.get("retrieval", {})
            fulltext_k = int(retrieval.get("fulltext_k") or 0)
            fulltext_fields = retrieval.get("fulltext_fields") or []
            if fulltext_k <= 0 or not fulltext_fields:
                continue
            source_id = source.get("id", "")
            chunks = _run(retriever.retrieve("Mars sol day"))
            source_chunks = [
                c
                for c in chunks
                if (c.metadata.get("_data_source") or "").startswith(source_id)
            ]
            assert len(source_chunks) > 0, (
                f"Source '{source_id}' with fulltext_k={fulltext_k} returned no chunks"
            )


# ===========================================================================
# 5. End-to-end RAG pipeline
# ===========================================================================


class TestRAGPipeline:
    """Run the full DecomposedRAGPipeline on the test questions and verify
    that reasoning traces and final answers are produced."""

    @pytest.fixture(scope="class")
    def pipeline_and_retriever(self, config, retriever, llm_client):
        import rag_divdet as rd

        pipeline = rd.DecomposedRAGPipeline(
            retriever=retriever,
            llm=llm_client,
            max_sub_q=2,
            num_rounds=1,
            subq_fanout_cap=2,
            subq_max_concurrency=1,
        )
        return pipeline, retriever

    def test_pipeline_produces_final_answer(self, pipeline_and_retriever):
        pipeline, _ = pipeline_and_retriever
        questions = _load_test_questions()
        q = questions[0]
        result = _run(pipeline.run(q["question_text"]))

        assert result is not None, "pipeline.run() returned None"
        assert "final_answer" in result, "Result missing 'final_answer'"
        final = result["final_answer"]
        assert isinstance(final, str) and len(final.strip()) > 0, (
            "final_answer is empty or not a string"
        )

    def test_pipeline_produces_reasoning_trace(self, pipeline_and_retriever):
        pipeline, _ = pipeline_and_retriever
        questions = _load_test_questions()
        q = questions[1]
        result = _run(pipeline.run(q["question_text"]))

        assert "initial_answer" in result, "Result missing 'initial_answer'"
        assert isinstance(result["initial_answer"], str)
        assert len(result["initial_answer"].strip()) > 0, (
            "initial_answer (preliminary answer) is empty"
        )
        assert "rounds" in result, "Result missing 'rounds'"
        assert isinstance(result["rounds"], list), "'rounds' should be a list"

    def test_pipeline_produces_initial_chunks(self, pipeline_and_retriever):
        pipeline, _ = pipeline_and_retriever
        questions = _load_test_questions()
        q = questions[2]
        result = _run(pipeline.run(q["question_text"]))

        assert "initial_chunks" in result, "Result missing 'initial_chunks'"
        chunks = result["initial_chunks"]
        assert isinstance(chunks, list) and len(chunks) > 0, (
            "initial_chunks is empty – retrieval may have failed"
        )

    def test_all_test_questions_produce_answers(self, pipeline_and_retriever):
        """Run every question in test_questions.json and assert non-empty answers."""
        pipeline, _ = pipeline_and_retriever
        questions = _load_test_questions()
        for q in questions:
            result = _run(pipeline.run(q["question_text"]))
            assert result.get("final_answer", "").strip(), (
                f"Question {q['question_id']!r} produced an empty final_answer"
            )
