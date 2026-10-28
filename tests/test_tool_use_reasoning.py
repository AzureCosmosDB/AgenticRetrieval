import asyncio
import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import openai
import pytest

import dynamic_retriever as dr


MESSAGE = (
    "Function tools with reasoning_effort are not supported for gpt-6-luna in "
    "/v1/chat/completions. To use function tools, use /v1/responses or set "
    "reasoning_effort to 'none'."
)
TOOLS = [{"type": "function", "function": {"name": "search", "parameters": {}}}]


def bad_request(message=MESSAGE, *, body=None):
    return openai.BadRequestError(
        message,
        response=httpx.Response(400, request=httpx.Request("POST", "https://example.invalid")),
        body={"message": message} if body is None else body,
    )


@pytest.fixture
def client(monkeypatch):
    create = AsyncMock()
    monkeypatch.setattr(dr, "_tool_use_llm", SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=create),
    )))
    monkeypatch.setattr(dr, "_tool_use_llm_cfg", {
        "llm_endpoint": "https://example.invalid", "llm_model": "deployment-alias",
    })
    monkeypatch.setattr(dr, "_tool_use_reasoning_disabled", set())
    return create


def request(model="deployment-alias", **kwargs):
    return dr.tool_use_chat_completion(
        model=model, messages=[{"role": "user", "content": "Question"}], **kwargs,
    )


def test_compatible_models_keep_original_parameters(client):
    expected = object()
    client.return_value = expected

    assert asyncio.run(request(tools=TOOLS, temperature=1.0)) is expected

    client.assert_awaited_once()
    assert "reasoning_effort" not in client.await_args.kwargs
    assert client.await_args.kwargs["tools"] == TOOLS
    assert client.await_args.kwargs["temperature"] == 1.0
    assert dr._tool_use_reasoning_disabled == set()


@pytest.mark.parametrize("nested_body", [False, True])
def test_exact_error_retries_and_persists_without_mutating_config(client, capsys, nested_body):
    expected = object()
    body = {"error": {"message": MESSAGE}} if nested_body else {"message": MESSAGE}
    client.side_effect = [bad_request(body=body), expected, expected]
    original_config = copy.deepcopy(dr._tool_use_llm_cfg)

    async def run():
        assert await request(tools=TOOLS) is expected
        assert await request() is expected

    asyncio.run(run())

    calls = client.await_args_list
    assert "reasoning_effort" not in calls[0].kwargs
    assert calls[1].kwargs["reasoning_effort"] == "none"
    assert calls[2].kwargs["reasoning_effort"] == "none"
    assert dr._tool_use_llm_cfg == original_config
    warning = capsys.readouterr().out
    assert warning.count("reasoning_effort='none'") == 1
    assert "/responses" in warning
    assert "reasoning" in warning.lower()


def test_override_is_scoped_to_endpoint_and_deployment(client, monkeypatch):
    client.side_effect = [bad_request(), object(), object(), object()]

    async def run():
        await request(tools=TOOLS)
        await request(model="another-model", tools=TOOLS)
        monkeypatch.setattr(dr, "_tool_use_llm_cfg", {"llm_endpoint": "https://other.invalid"})
        await request(tools=TOOLS)

    asyncio.run(run())

    assert client.await_args_list[1].kwargs["reasoning_effort"] == "none"
    assert "reasoning_effort" not in client.await_args_list[2].kwargs
    assert "reasoning_effort" not in client.await_args_list[3].kwargs


@pytest.mark.parametrize("message", [
    "Unsupported parameter: reasoning_effort",
    "Unsupported value: reasoning_effort does not support none",
    "Function tools are not supported for this model",
    "Function tools with reasoning_effort are not supported for this model",
    "Context length exceeded",
])
def test_other_bad_requests_propagate_without_disabling_reasoning(client, message):
    error = bad_request(message)
    client.side_effect = error

    with pytest.raises(openai.BadRequestError) as caught:
        asyncio.run(request(tools=TOOLS))

    assert caught.value is error
    client.assert_awaited_once()
    assert dr._tool_use_reasoning_disabled == set()


@pytest.mark.parametrize("kwargs", [{}, {"tools": []}, {"tools": TOOLS, "reasoning_effort": "none"}])
def test_no_retry_without_tools_or_when_already_disabled(client, kwargs):
    error = bad_request()
    client.side_effect = error

    with pytest.raises(openai.BadRequestError) as caught:
        asyncio.run(request(**kwargs))

    assert caught.value is error
    client.assert_awaited_once()


def test_failed_retry_propagates_and_does_not_loop(client):
    retry_error = bad_request("Unsupported value: none")
    client.side_effect = [bad_request(), retry_error]

    with pytest.raises(openai.BadRequestError) as caught:
        asyncio.run(request(tools=TOOLS))

    assert caught.value is retry_error
    assert client.await_count == 2


@pytest.mark.parametrize("status", [429, 500])
def test_non_bad_request_errors_do_not_trigger_fallback(client, status):
    error = openai.APIStatusError(
        MESSAGE,
        response=httpx.Response(status, request=httpx.Request("POST", "https://example.invalid")),
        body={"message": MESSAGE},
    )
    client.side_effect = error

    with pytest.raises(openai.APIStatusError) as caught:
        asyncio.run(request(tools=TOOLS))

    assert caught.value is error
    client.assert_awaited_once()
    assert dr._tool_use_reasoning_disabled == set()


def test_real_sdk_serializes_runtime_override_without_network(client, monkeypatch):
    payloads = []

    def handle(http_request):
        payloads.append(json.loads(http_request.content))
        if len(payloads) == 1:
            return httpx.Response(400, json={"error": {
                "message": MESSAGE, "type": "invalid_request_error", "code": "unsupported_value",
            }})
        return httpx.Response(200, json={
            "id": "test", "object": "chat.completion", "created": 0,
            "model": "deployment-alias",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "Answer"}}],
        })

    async def run():
        async with openai.AsyncAzureOpenAI(
            azure_endpoint="https://example.invalid", api_version="2024-10-21",
            api_key="test", max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
        ) as sdk:
            monkeypatch.setattr(dr, "_tool_use_llm", sdk)
            assert (await request(tools=TOOLS)).choices[0].message.content == "Answer"
            assert (await request()).choices[0].message.content == "Answer"

    asyncio.run(run())

    assert len(payloads) == 3
    assert "reasoning_effort" not in payloads[0]
    assert payloads[1] == {**payloads[0], "reasoning_effort": "none"}
    assert payloads[2]["reasoning_effort"] == "none"


def test_inflight_requests_retry_but_only_warn_once(client, capsys):
    started = 0
    ready = asyncio.Event()

    async def create(**kwargs):
        nonlocal started
        if kwargs.get("reasoning_effort") != "none":
            started += 1
            if started == 2:
                ready.set()
            await ready.wait()
            raise bad_request()
        return "ok"

    client.side_effect = create

    async def run():
        return await asyncio.gather(request(tools=TOOLS), request(tools=TOOLS))

    assert asyncio.run(run()) == ["ok", "ok"]
    assert client.await_count == 4
    assert capsys.readouterr().out.count("reasoning_effort='none'") == 1


def test_initialization_resets_runtime_override(monkeypatch):
    monkeypatch.setattr(dr, "_tool_use_reasoning_disabled", {("endpoint", "model")})
    monkeypatch.setattr(dr, "CONFIG", {
        "llm": {"llm_endpoint": "https://example.invalid", "api_version": "test",
                "max_retries": 1, "use_rbac_auth": False, "llm_api_key": "test"},
        "embedding": {"embed_endpoint": "https://embed.invalid", "api_version": "test"},
        "cosmos": {"sources": []},
        "ranker": {"use_ranker": False},
    })
    monkeypatch.setattr(dr, "AsyncAzureOpenAI", Mock())
    monkeypatch.setattr(dr.openai, "AsyncOpenAI", Mock())
    for name in (
        "_tool_use_llm", "_tool_use_embed_client", "_tool_use_llm_cfg", "_tool_use_embed_cfg",
        "_tool_use_cosmos_cfg", "_tool_use_source_cfg", "_tool_use_source_embed",
        "_tool_use_source_ft", "_tool_use_source_partition", "_tool_use_all_embed",
        "_tool_use_context_fields", "_tool_use_max_retries", "_tool_use_rerank_mul",
        "_tool_use_prune_k", "_tool_use_context_limit", "_tool_use_use_hyde",
        "_tool_use_use_ranker", "_tool_use_max_search_rounds", "_tool_use_max_agent_turns",
        "_tool_use_query_template",
    ):
        monkeypatch.setattr(dr, name, getattr(dr, name))

    dr.init_tool_use_clients()

    assert dr._tool_use_reasoning_disabled == set()
