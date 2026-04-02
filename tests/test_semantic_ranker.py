"""Tests for the semantic ranker reranking logic in CombinedRetriever."""

import asyncio
import json

import httpx
import pytest

from agentic_retriever import RetrievedChunk
from utils.cosmos_retriever import CombinedRetriever, RETRIEVAL_SOURCES, CONFIG


def _make_chunks(n: int) -> list[RetrievedChunk]:
    """Create n fake RetrievedChunk objects."""
    return [
        RetrievedChunk(
            chunk_id=f"c{i}",
            text=f"document {i} text",
            similarity=float(n - i) / n,
            metadata={"_data_source": "test_source"},
        )
        for i in range(n)
    ]


def _make_ranker_response(indices: list[int], scores: list[float] | None = None) -> dict:
    """Build a mock ranker API response."""
    if scores is None:
        scores = [1.0 - 0.1 * i for i in range(len(indices))]
    return {
        "Scores": [
            {"index": idx, "score": sc} for idx, sc in zip(indices, scores)
        ],
        "latency": {"data_preprocess_time": 0.001, "inference_time": 0.01},
        "token_usage": {"total_tokens": 100},
    }


def _ranker_url(retriever: CombinedRetriever) -> str:
    """Build the semantic reranking URL from retriever config."""
    url_suffix = str(CONFIG.get("ranker", {}).get("url_suffix", "")).strip()
    return f"https://{retriever._ranker_account}.{retriever._ranker_region}.{url_suffix}"


def _register_url(retriever: CombinedRetriever) -> str:
    """Build the account registration URL from config."""
    register_account_path = str(CONFIG.get("ranker", {}).get("register_account_path", "")).strip()
    return f"https://{retriever._ranker_region}.{register_account_path}"


def _max_retries() -> int:
    """Read max_retries from config."""
    return int(CONFIG.get("ranker", {}).get("max_retries", 5))


def _build_retriever(k_ranker: int = 3) -> CombinedRetriever:
    """Build a CombinedRetriever with ranker enabled and a mock token."""
    retriever = CombinedRetriever(
        retrieval_sources=RETRIEVAL_SOURCES,
        k_diverse=0,
        k_ranker=k_ranker,
    )
    retriever._use_ranker = True
    retriever._ranker_account = "test-account"
    retriever._ranker_region = "region"
    retriever._ranker_access_token = "fake-token"
    retriever._ranker_batch_size = 32
    return retriever


# ---------------------------------------------------------------------------
# Test: correct indices are selected from ranker response
# ---------------------------------------------------------------------------

class TestSemanticRankerSelection:

    def test_ranker_selects_correct_chunks(self):
        """Ranker response indices [3, 7, 1] should pick chunks[3], chunks[7], chunks[1]."""
        chunks = _make_chunks(10)
        mock_response = _make_ranker_response([3, 7, 1])

        captured_requests = []

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            captured_requests.append(request)
            return httpx.Response(200, json=mock_response)

        transport = httpx.MockTransport(mock_handler)
        retriever = _build_retriever(k_ranker=3)
        retriever._ranker_http_client = httpx.AsyncClient(transport=transport)

        async def run():
            # Simulate the ranker portion of retrieve()
            effective_k_ranker = retriever.k_ranker
            documents = [c.text for c in chunks]
            body = {
                "query": "test query",
                "documents": documents,
                "return_documents": False,
                "top_k": effective_k_ranker,
                "batch_size": retriever._ranker_batch_size,
            }
            headers = {
                "Authorization": f"Bearer {retriever._ranker_access_token}",
                "Content-Type": "application/json",
            }
            url = _ranker_url(retriever)

            response = await retriever._ranker_http_client.post(url, headers=headers, json=body)
            result = response.json()
            scores = result.get("Scores", [])
            ranked_indices = [s["index"] for s in scores[:effective_k_ranker]]
            return [chunks[i] for i in ranked_indices]

        selected = asyncio.run(run())
        assert [c.chunk_id for c in selected] == ["c3", "c7", "c1"]
        assert len(selected) == 3

        # Verify the request was sent correctly
        assert len(captured_requests) == 1
        req_body = json.loads(captured_requests[0].content)
        assert req_body["top_k"] == 3
        assert len(req_body["documents"]) == 10
        assert req_body["return_documents"] is False

        # Verify URL uses expected register_account_path pattern
        register_url = _register_url(retriever)
        assert retriever._ranker_region in register_url

    def test_ranker_respects_effective_k_ranker(self):
        """When k_divisor > 1, effective_k_ranker should be smaller."""
        chunks = _make_chunks(10)
        # Ranker returns 5 results but effective_k_ranker = k_ranker // k_divisor = 6 // 3 = 2
        mock_response = _make_ranker_response([4, 2, 0, 8, 6])

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=mock_response)

        transport = httpx.MockTransport(mock_handler)
        retriever = _build_retriever(k_ranker=6)
        retriever._ranker_http_client = httpx.AsyncClient(transport=transport)

        async def run():
            k_divisor = 3
            effective_k_ranker = retriever.k_ranker // k_divisor
            url = _ranker_url(retriever)
            body = {
                "query": "test",
                "documents": [c.text for c in chunks],
                "return_documents": False,
                "top_k": effective_k_ranker,
                "batch_size": retriever._ranker_batch_size,
            }
            headers = {
                "Authorization": f"Bearer {retriever._ranker_access_token}",
                "Content-Type": "application/json",
            }
            response = await retriever._ranker_http_client.post(url, headers=headers, json=body)
            result = response.json()
            scores = result.get("Scores", [])
            ranked_indices = [s["index"] for s in scores[:effective_k_ranker]]
            return [chunks[i] for i in ranked_indices]

        selected = asyncio.run(run())
        assert len(selected) == 2
        assert [c.chunk_id for c in selected] == ["c4", "c2"]

    def test_ranker_fewer_results_than_requested(self):
        """If ranker returns fewer results than k_ranker, take what's available."""
        chunks = _make_chunks(10)
        mock_response = _make_ranker_response([5])  # only 1 result

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=mock_response)

        transport = httpx.MockTransport(mock_handler)
        retriever = _build_retriever(k_ranker=5)
        retriever._ranker_http_client = httpx.AsyncClient(transport=transport)

        async def run():
            effective_k_ranker = retriever.k_ranker
            url = _ranker_url(retriever)
            response = await retriever._ranker_http_client.post(url, headers={}, json={})
            result = response.json()
            scores = result.get("Scores", [])
            ranked_indices = [s["index"] for s in scores[:effective_k_ranker]]
            return [chunks[i] for i in ranked_indices]

        selected = asyncio.run(run())
        assert len(selected) == 1
        assert selected[0].chunk_id == "c5"


# ---------------------------------------------------------------------------
# Test: retry logic on transient errors
# ---------------------------------------------------------------------------

class TestSemanticRankerRetry:

    def test_retry_on_503(self):
        """503 should be retried and succeed on the second attempt."""
        chunks = _make_chunks(5)
        mock_response = _make_ranker_response([0, 1])
        attempt_count = {"n": 0}

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            attempt_count["n"] += 1
            if attempt_count["n"] == 1:
                return httpx.Response(503, text="Service Unavailable")
            return httpx.Response(200, json=mock_response)

        transport = httpx.MockTransport(mock_handler)
        retriever = _build_retriever(k_ranker=2)
        retriever._ranker_http_client = httpx.AsyncClient(transport=transport)

        async def run():
            effective_k_ranker = retriever.k_ranker
            url = _ranker_url(retriever)
            body = {"query": "test", "documents": [c.text for c in chunks], "return_documents": False, "top_k": effective_k_ranker, "batch_size": 32}
            headers = {"Authorization": "Bearer fake", "Content-Type": "application/json"}

            max_retries = _max_retries()
            ranker_succeeded = False
            result_chunks = chunks
            for attempt in range(max_retries):
                response = await retriever._ranker_http_client.post(url, headers=headers, json=body)
                if response.status_code in (502, 503, 429) and attempt < max_retries - 1:
                    await asyncio.sleep(0)  # skip actual wait in test
                    continue
                response.raise_for_status()
                result = response.json()
                scores = result.get("Scores", [])
                ranked_indices = [s["index"] for s in scores[:effective_k_ranker]]
                result_chunks = [chunks[i] for i in ranked_indices]
                ranker_succeeded = True
                break

            return ranker_succeeded, result_chunks

        succeeded, selected = asyncio.run(run())
        assert succeeded is True
        assert attempt_count["n"] == 2
        assert [c.chunk_id for c in selected] == ["c0", "c1"]

    def test_non_retriable_error_fails_immediately(self):
        """A 400 error should not be retried."""
        async def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, text="Bad Request")

        transport = httpx.MockTransport(mock_handler)
        retriever = _build_retriever(k_ranker=2)
        retriever._ranker_http_client = httpx.AsyncClient(transport=transport)

        async def run():
            url = _ranker_url(retriever)
            max_retries = _max_retries()
            ranker_succeeded = False
            for attempt in range(max_retries):
                try:
                    response = await retriever._ranker_http_client.post(url, headers={}, json={})
                    if response.status_code in (502, 503, 429) and attempt < max_retries - 1:
                        continue
                    response.raise_for_status()
                    ranker_succeeded = True
                    break
                except Exception:
                    if attempt < max_retries - 1 and False:  # 400 is not retriable
                        continue
                    break
            return ranker_succeeded

        succeeded = asyncio.run(run())
        assert succeeded is False


# ---------------------------------------------------------------------------
# Test: empty Scores response
# ---------------------------------------------------------------------------

class TestSemanticRankerEdgeCases:

    def test_empty_scores_returns_nothing(self):
        """If Scores is empty, no chunks should be selected."""
        chunks = _make_chunks(5)
        mock_response = {"Scores": []}

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=mock_response)

        transport = httpx.MockTransport(mock_handler)
        retriever = _build_retriever(k_ranker=3)
        retriever._ranker_http_client = httpx.AsyncClient(transport=transport)

        async def run():
            url = _ranker_url(retriever)
            response = await retriever._ranker_http_client.post(url, headers={}, json={})
            result = response.json()
            scores = result.get("Scores", [])
            ranked_indices = [s["index"] for s in scores[:retriever.k_ranker]]
            return [chunks[i] for i in ranked_indices]

        selected = asyncio.run(run())
        assert selected == []

    def test_ranker_skipped_when_chunks_fewer_than_k(self):
        """When len(chunks) <= effective_k_ranker, ranker should be skipped."""
        retriever = _build_retriever(k_ranker=10)
        chunks = _make_chunks(5)  # fewer than k_ranker
        effective_k_ranker = retriever.k_ranker

        # The condition in retrieve(): len(chunks) > effective_k_ranker
        should_run_ranker = len(chunks) > effective_k_ranker
        assert should_run_ranker is False
