import asyncio
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from azure.core.credentials import AccessToken
from azure.identity import get_bearer_token_provider

from utils.ranker import rerank_documents


def test_provider_refreshes_expired_token_between_ranker_calls():
    credential = Mock(spec=["get_token"])
    credential.get_token.side_effect = [
        AccessToken("expired-token", 0),
        AccessToken("fresh-token", 4102444800),
    ]
    provider = get_bearer_token_provider(credential, "https://example.invalid/.default")
    captured = []

    def handler(request):
        captured.append(request.headers["Authorization"])
        return httpx.Response(200, json={"Scores": [{"index": 0}]})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            for _ in range(2):
                assert await rerank_documents(
                    client, "https://example.invalid/rank", provider, "query", ["doc"], 1
                ) == [0]

    asyncio.run(run())
    assert captured == ["Bearer expired-token", "Bearer fresh-token"]
    assert credential.get_token.call_count == 2


def test_ranker_checks_provider_on_retry(monkeypatch):
    provider = Mock(side_effect=["first-token", "second-token"])
    captured = []

    def handler(request):
        captured.append(request.headers["Authorization"])
        if len(captured) == 1:
            return httpx.Response(503)
        return httpx.Response(200, json={"Scores": [{"index": 0}]})

    monkeypatch.setattr("utils.ranker.asyncio.sleep", AsyncMock())

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await rerank_documents(
                client, "https://example.invalid/rank", provider, "query", ["doc"], 1
            )

    assert asyncio.run(run()) == [0]
    assert captured == ["Bearer first-token", "Bearer second-token"]


@pytest.mark.parametrize("status", [401, 403])
def test_ranker_does_not_repeat_unauthorized_requests(status, capsys):
    handler = Mock(return_value=httpx.Response(status))

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await rerank_documents(
                client, "https://example.invalid/rank", "fixed-token", "query", ["doc"], 1
            )

    assert asyncio.run(run()) is None
    assert handler.call_count == 1
    assert f"HTTP {status}" in capsys.readouterr().out


def test_ranker_still_accepts_static_tokens():
    def handler(request):
        assert request.headers["Authorization"] == "Bearer fixed-token"
        return httpx.Response(200, json={"Scores": [{"index": 0}]})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await rerank_documents(
                client, "https://example.invalid/rank", "fixed-token", "query", ["doc"], 1
            )

    assert asyncio.run(run()) == [0]
