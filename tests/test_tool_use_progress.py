import asyncio
import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import openai
import pytest

import dynamic_retriever as dr


def response(*calls, content=None):
    tools = [
        SimpleNamespace(id=f"call_{i}", function=SimpleNamespace(
            name=name, arguments=args if isinstance(args, str) else json.dumps(args),
        ))
        for i, (name, args) in enumerate(calls)
    ]
    message = SimpleNamespace(content=content, tool_calls=tools or None)
    message.model_dump = lambda **kwargs: {
        "role": "assistant", "content": content,
        **({"tool_calls": [
            {"id": call.id, "type": "function", "function": {
                "name": call.function.name, "arguments": call.function.arguments,
            }} for call in tools
        ]} if tools else {}),
    }
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


@pytest.fixture
def agent(monkeypatch):
    requests = []
    scripted = []
    monkeypatch.setattr(dr, "_tool_use_query_template", "Original question: {question}. initial_search first.")
    monkeypatch.setattr(dr, "_tool_use_context_fields", [])
    monkeypatch.setattr(dr, "_tool_use_max_search_rounds", 2)
    monkeypatch.setattr(dr, "_tool_use_max_agent_turns", 10)
    monkeypatch.setattr(dr, "_tool_use_max_retries", 2)
    monkeypatch.setattr(dr, "_tool_use_context_limit", 4000)
    monkeypatch.setattr(dr, "_tool_use_prune_k", 20)
    monkeypatch.setattr(dr, "_tool_use_llm_cfg", {
        "llm_model": "test", "temperature": 0, "max_completion_tokens": 256,
    })

    async def create(**kwargs):
        requests.append(copy.deepcopy(kwargs))
        assert scripted, "Unexpected extra LLM request"
        item = scripted.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def search(query, containers):
        return json.dumps([{"docid": query, "text": f"Verified source finding for {query}; upc: 001."}])

    async def prune(docids, containers, cache):
        return "\n".join(cache[docid] for docid in docids if docid in cache)

    monkeypatch.setattr(dr, "_tool_use_llm", SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=create),
    )))
    search_mock = AsyncMock(side_effect=search)
    monkeypatch.setattr(dr, "tool_use_do_search", search_mock)
    monkeypatch.setattr(dr, "tool_use_do_prune", AsyncMock(side_effect=prune))
    monkeypatch.setattr(dr.asyncio, "sleep", AsyncMock())

    def run(*items):
        scripted.extend(items)
        result = asyncio.run(dr.process_question({
            "question_id": "q1", "question_text": "original",
        }, {}))
        assert not scripted
        return result

    return SimpleNamespace(run=run, requests=requests, search=search_mock)


@pytest.mark.parametrize("budget", [0, 1, 2])
def test_search_round_budget_forces_final_answer_and_counts_parallel_queries_once(agent, monkeypatch, budget):
    monkeypatch.setattr(dr, "_tool_use_max_search_rounds", budget)
    items = [response(("initial_search", {}))]
    for turn in range(budget):
        items.append(response(
            ("search", {"query": f"query {turn} a"}),
            ("search", {"query": f"query {turn} b"}),
        ))
    items.append(response(("final_answer", {"answer": "Supported facts. Unmet: retailer availability."})))
    result = agent.run(*items)
    assert result["status"] == "completed"
    assert result["termination_reason"] == "research_budget_exhausted"
    assert result["research_progress"]["search_rounds"] == budget
    assert result["tool_calls"]["initial_search"] == 1
    assert result["tool_calls"]["search"] == budget * 2
    assert agent.search.await_count == 1 + budget * 2
    final = agent.requests[-1]
    assert [tool["function"]["name"] for tool in final["tools"]] == ["final_answer"]
    assert final["tool_choice"]["function"]["name"] == "final_answer"
    assert "explicitly state unmet requirements" in final["messages"][-1]["content"]
    assert result["answer"]


@pytest.mark.parametrize("duplicate_batch", [False, True])
def test_initial_search_is_executed_once_even_if_model_repeats_it(agent, monkeypatch, duplicate_batch):
    monkeypatch.setattr(dr, "_tool_use_max_agent_turns", 3)
    first = response(("initial_search", {}), ("initial_search", {})) if duplicate_batch else response(("initial_search", {}))
    result = agent.run(
        first, response(("initial_search", {})), response(("initial_search", {})),
        response(("final_answer", {"answer": "Available evidence; unmet requirements remain."})),
    )
    assert agent.search.await_count == 1
    assert result["tool_calls"]["initial_search"] == 1
    assert result["research_progress"]["attempted_queries"] == ["original"]
    assert result["termination_reason"] == "agent_turn_limit"
    assert result["status"] == "partial"
    assert all(
        tool["function"]["name"] != "initial_search"
        for tool in agent.requests[1]["tools"]
    )


@pytest.mark.parametrize("pruning", ["explicit", "mixed", "automatic", "api_overflow"])
def test_pruning_preserves_queries_findings_notes_and_gaps(agent, monkeypatch, pruning):
    items = [
        response(("initial_search", {}), content="Finding: a source item is available."),
        response(("find_information_gaps", {})),
        response(content="Unresolved: retailer stock and vegan certification."),
        response(("search", {"query": "targeted"}), content="Finding: ingredients are documented."),
    ]
    if pruning == "explicit":
        items.append(response(("prune", {"docids": ["targeted"]})))
    elif pruning == "mixed":
        items.append(response(
            ("prune", {"docids": ["targeted"]}),
            ("search", {"query": "must not execute"}),
        ))
    elif pruning == "automatic":
        original_count = dr.count_tokens

        def count(messages):
            if any(m.get("role") == "tool" and '"docid"' in m.get("content", "") for m in messages):
                return 100000
            return original_count(messages)

        monkeypatch.setattr(dr, "count_tokens", count)
    else:
        items.append(openai.BadRequestError(
            "context_length_exceeded",
            response=httpx.Response(400, request=httpx.Request("POST", "https://example.invalid")),
            body={"code": "context_length_exceeded"},
        ))
    items.extend([
        response(("search", {"query": "remaining gap"})),
        response(("final_answer", {"answer": "Evidence-supported answer; stock remains unverified."})),
    ])
    result = agent.run(*items)
    state = result["research_progress"]
    assert state["initial_search_completed"] is True
    assert state["attempted_queries"] == ["original", "targeted", "remaining gap"]
    assert state["findings"]["original"].startswith("Verified source finding")
    assert state["findings"]["targeted"].startswith("Verified source finding")
    assert "ingredients are documented" in " ".join(state["notes"])
    assert "retailer stock" in state["unresolved_gaps"][0]
    compacted = [
        m["content"] for request in agent.requests for m in request["messages"]
        if m.get("role") == "user" and "Research progress survived" in m.get("content", "")
    ]
    assert compacted
    assert any(
        all(text in content for text in ("targeted", "Verified source finding", "ingredients", "retailer stock"))
        for content in compacted
    )
    assert result["tool_calls"]["initial_search"] == 1


@pytest.mark.parametrize("arguments", ['{"answer":""}', '{"answer":"  "}', '{"answer":null}', '[]', '{'])
def test_invalid_normal_final_answer_is_not_returned_as_success(agent, arguments):
    result = agent.run(
        response(("initial_search", {})),
        response(("final_answer", arguments)),
        response(("final_answer", {"answer": "Valid supported answer."})),
    )
    assert result["status"] == "completed"
    assert result["answer"] == "Valid supported answer."
    assert result["tool_calls"]["final_answer"] == 2


@pytest.mark.parametrize("failure", ["empty", "malformed", "missing_call", "http"])
def test_failed_forced_finalization_is_explicit_and_bounded(agent, monkeypatch, failure):
    monkeypatch.setattr(dr, "_tool_use_max_search_rounds", 0)
    if failure == "empty":
        invalid = response(("final_answer", {"answer": ""}))
    elif failure == "malformed":
        invalid = response(("final_answer", "[]"))
    elif failure == "missing_call":
        invalid = response(content="No tool call")
    else:
        invalid = openai.APIStatusError(
            "Unavailable", body=None,
            response=httpx.Response(503, request=httpx.Request("POST", "https://example.invalid")),
        )
    result = agent.run(response(("initial_search", {})), invalid, invalid)
    assert result["status"] == "failed"
    assert result["termination_reason"] == "finalization_failed"
    assert result["error"]
    assert result["answer"] == ""
    assert result["research_progress"]["initial_search_completed"] is True
    assert len(agent.requests) == 3


def test_empty_model_answer_triggers_final_synthesis(agent):
    result = agent.run(
        response(("initial_search", {})), response(content=""),
        response(("final_answer", {"answer": "Only supported facts; other requirements unverified."})),
    )
    assert result["answer"]
    assert result["termination_reason"] == "empty_model_answer"


def test_forced_synthesis_includes_configured_traceability_fields(agent, monkeypatch):
    monkeypatch.setattr(dr, "_tool_use_context_fields", ["upc", "product_id"])
    monkeypatch.setattr(dr, "_tool_use_max_search_rounds", 0)
    agent.run(
        response(("initial_search", {})),
        response(("final_answer", {"answer": "Item (upc: 001; product_id: 7). Unmet: stock."})),
    )
    assert "upc, product_id" in agent.requests[-1]["messages"][-1]["content"]


@pytest.mark.parametrize(
    "pipeline",
    [
        {"max_search_rounds": -1}, {"max_search_rounds": True},
        {"max_search_rounds": "4"}, {"max_search_rounds": None},
        {"max_agent_turns": 0}, {"max_agent_turns": 1.5},
    ],
)
def test_invalid_research_budgets_fail_before_client_initialization(monkeypatch, pipeline):
    monkeypatch.setattr(dr, "CONFIG", {"pipeline": pipeline})
    with pytest.raises(dr.ConfigurationError, match="pipeline."):
        dr.init_tool_use_clients()


def test_research_budget_defaults_and_zero_followup_rounds():
    assert dr._tool_use_research_limits({}) == (4, 50)
    assert dr._tool_use_research_limits({"max_search_rounds": 0, "max_agent_turns": 1}) == (0, 1)


def test_compaction_preserves_complete_ledger_when_context_excerpts_must_be_omitted(
    agent, monkeypatch, capsys,
):
    monkeypatch.setattr(dr, "_tool_use_context_limit", 700)
    monkeypatch.setattr(dr, "_tool_use_max_search_rounds", 0)
    agent.search.side_effect = None
    agent.search.return_value = json.dumps([
        {"docid": str(i), "text": f"Source {i}. " + "fact " * 1000} for i in range(6)
    ])
    result = agent.run(
        response(("initial_search", {})),
        response(("final_answer", {"answer": "Available facts; remaining requirements unverified."})),
    )
    assert len(result["research_progress"]["findings"]) == 6
    assert "source excerpts omitted" in capsys.readouterr().out
    assert dr.count_tokens(agent.requests[-1]["messages"]) < 700


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("arguments", ["[]", '{"docids":null}', '{"docids":[1]}'])
def test_invalid_prune_arguments_are_reported_without_losing_progress(agent, mixed, arguments):
    calls = [("prune", arguments)]
    if mixed:
        calls.append(("search", {"query": "must not execute"}))
    result = agent.run(
        response(("initial_search", {})), response(*calls),
        response(("final_answer", {"answer": "Supported answer."})),
    )
    assert result["tool_calls"]["initial_search"] == 1
    assert result["tool_calls"]["prune"] == 0
    assert result["research_progress"]["findings"]["original"]
    assert agent.search.await_count == 1
