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


def test_research_snapshot_omits_findings_without_changing_internal_findings():
    progress = dr._ToolUseResearchProgress(findings={
        "doc-one": "First document excerpt.", "doc-two": "Second document excerpt.",
    })
    saved = progress.snapshot()
    assert "findings" not in json.loads(json.dumps(saved))
    compacted = progress.snapshot(include_document_text=True)
    assert compacted["findings"] == {
        "doc-one": "First document excerpt.", "doc-two": "Second document excerpt.",
    }
    compacted["findings"].pop("doc-one")
    assert progress.findings["doc-one"] == "First document excerpt."


@pytest.fixture
def agent(monkeypatch):
    requests = []
    scripted = []
    validation_requests = []
    audits = []
    monkeypatch.setattr(dr, "_tool_use_reasoning_disabled", set())
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
        choice = kwargs.get("tool_choice")
        if isinstance(choice, dict) and choice.get("function", {}).get("name") == "audit_answer":
            validation_requests.append(copy.deepcopy(kwargs))
            audit = audits.pop(0) if audits else {"substantive": True, "issues": [], "items": []}
            if isinstance(audit, Exception):
                raise audit
            if isinstance(audit, SimpleNamespace):
                return audit
            return response(("audit_answer", audit))
        requests.append(copy.deepcopy(kwargs))
        assert scripted, "Unexpected extra LLM request"
        item = scripted.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def search(query, containers):
        return json.dumps([{
            "docid": query,
            "text": f"Verified source finding for {query}.\nupc: 001\nproduct_id: 7",
        }])

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
        assert not audits
        return result

    return SimpleNamespace(
        run=run, requests=requests, search=search_mock,
        audits=audits, validation_requests=validation_requests,
    )


@pytest.mark.parametrize("stage", ["agent", "forced", "audit", "gap"])
def test_reasoning_tool_error_recovers_on_every_tool_use_request_path(agent, monkeypatch, stage):
    monkeypatch.setattr(dr, "_tool_use_max_retries", 1)
    error = openai.BadRequestError(
        "Function tools with reasoning_effort are not supported for gpt-6-luna in "
        "/v1/chat/completions. To use function tools, use /v1/responses or set "
        "reasoning_effort to 'none'.",
        response=httpx.Response(400, request=httpx.Request("POST", "https://example.invalid")),
        body=None,
    )
    initial = response(("initial_search", {}))
    final = response(("final_answer", {"answer": "Supported facts; stock is unverified."}))
    items = [initial, final]
    if stage == "agent":
        items.insert(0, error)
    elif stage == "forced":
        monkeypatch.setattr(dr, "_tool_use_max_search_rounds", 0)
        items.insert(1, error)
    elif stage == "audit":
        agent.audits.extend([error, {"substantive": True, "issues": [], "items": []}])
    else:
        items = [
            error, initial, response(("find_information_gaps", {})),
            response(content="Retailer stock remains unverified."), final,
        ]

    result = agent.run(*items)

    assert result["status"] == "completed"
    assert result["tool_calls"]["initial_search"] == 1
    requests = agent.validation_requests if stage == "audit" else agent.requests
    assert any(request.get("reasoning_effort") == "none" for request in requests)
    assert agent.validation_requests[-1]["reasoning_effort"] == "none"
    if stage == "gap":
        assert next(
            request for request in agent.requests if "tools" not in request
        )["reasoning_effort"] == "none"


def test_reasoning_override_survives_later_questions_in_same_execution(agent):
    error = openai.BadRequestError(
        "Function tools with reasoning_effort are not supported for gpt-6-luna in "
        "/v1/chat/completions. To use function tools, use /v1/responses or set "
        "reasoning_effort to 'none'.",
        response=httpx.Response(400, request=httpx.Request("POST", "https://example.invalid")),
        body=None,
    )
    initial = response(("initial_search", {}))
    final = response(("final_answer", {"answer": "Available evidence; stock is unverified."}))
    config = copy.deepcopy(dr._tool_use_llm_cfg)

    assert agent.run(error, initial, final)["status"] == "completed"
    previous_requests = len(agent.requests)
    assert agent.run(initial, final)["status"] == "completed"

    assert all(
        request["reasoning_effort"] == "none" for request in agent.requests[previous_requests:]
    )
    assert dr._tool_use_llm_cfg == config


@pytest.mark.parametrize("path", ["model", "tool", "forced"])
def test_revision_only_responses_require_validated_synthesis(agent, monkeypatch, path):
    revision = (
        "Research completed, but the final answer needs a revision to include "
        "the exact UPC and product ID for every product mentioned."
    )
    if path == "forced":
        monkeypatch.setattr(dr, "_tool_use_max_search_rounds", 0)
    agent.audits.extend([
        {"substantive": False, "issues": ["Revision instruction, not an answer."], "items": []},
        {"substantive": True, "issues": [], "items": []},
    ])
    invalid = response(content=revision) if path == "model" else response(
        ("final_answer", {"answer": revision}),
    )
    result = agent.run(
        response(("initial_search", {})), invalid,
        response(("final_answer", {"answer": "Available evidence; retailer stock remains unverified."})),
    )
    assert result["status"] == "completed"
    assert result["answer"] != revision
    assert result["answer_validation"]["status"] == "passed"
    assert result["answer_validation"]["attempts"] == 2
    assert result["answer_validation"]["rejections"] == 1
    assert result["tool_calls"]["initial_search"] == 1


@pytest.mark.parametrize("path", ["model", "tool", "forced"])
def test_incomplete_inline_identifiers_are_repaired_on_every_exit_path(agent, monkeypatch, path):
    monkeypatch.setattr(dr, "_tool_use_context_fields", ["upc", "product_id"])
    if path == "forced":
        monkeypatch.setattr(dr, "_tool_use_max_search_rounds", 0)
    incomplete = "Item (upc: 001). Stock is unverified."
    complete = "Item (upc: 001; product_id: 7). Stock is unverified."
    agent.audits.extend([
        {"substantive": True, "issues": [], "items": [{
            "quote": "Item (upc: 001)", "context_fields": {"upc": "001"},
        }]},
        {"substantive": True, "issues": [], "items": [{
            "quote": "Item (upc: 001; product_id: 7)",
            "context_fields": {"upc": "001", "product_id": "7"},
        }]},
    ])
    invalid = response(content=incomplete) if path == "model" else response(
        ("final_answer", {"answer": incomplete}),
    )
    result = agent.run(
        response(("initial_search", {})), invalid,
        response(("final_answer", {"answer": complete})),
    )
    assert result["answer"] == complete
    assert result["answer_validation"]["rejections"] == 1
    assert result["answer_validation"]["items"][0]["source_docids"] == ["original"]
    assert agent.search.await_count == 1
    final_tool = next(
        t for t in agent.requests[1]["tools"] if t["function"]["name"] == "final_answer"
    )
    assert "upc, product_id" in final_tool["function"]["description"]
    assert "revision request" in final_tool["function"]["description"]


def test_repeated_revision_only_responses_fail_explicitly_without_more_search(agent):
    agent.audits.extend([
        {"substantive": False, "issues": ["Formatting revision only."], "items": []}
        for _ in range(3)
    ])
    invalid = response(("final_answer", {"answer": "Please revise the final answer."}))
    result = agent.run(response(("initial_search", {})), invalid, invalid, invalid)
    assert result["status"] == "failed"
    assert result["termination_reason"] == "finalization_failed"
    assert result["answer"] == ""
    assert result["error"]
    assert result["answer_validation"]["status"] == "failed"
    assert result["answer_validation"]["attempts"] == 3
    assert agent.search.await_count == 1
    for request in agent.requests[2:]:
        pending = set()
        for message in request["messages"]:
            pending.update(call["id"] for call in message.get("tool_calls", []))
            if message["role"] == "tool":
                pending.discard(message["tool_call_id"])
        assert not pending


def test_validation_service_failure_cannot_become_success(agent):
    failure = openai.APIConnectionError(
        request=httpx.Request("POST", "https://example.invalid"),
    )
    agent.audits.extend([failure, failure, failure])
    candidate = response(("final_answer", {"answer": "Supported answer."}))
    result = agent.run(response(("initial_search", {})), candidate, candidate, candidate)
    assert result["status"] == "failed"
    assert result["answer_validation"]["status"] == "failed"
    assert "Connection" in result["error"]


def test_missing_audit_tool_call_requires_repair(agent):
    agent.audits.append(response(content="The answer looks fine."))
    candidate = response(("final_answer", {"answer": "Supported facts; stock remains unverified."}))
    result = agent.run(response(("initial_search", {})), candidate, candidate)
    assert result["answer_validation"]["status"] == "passed"
    assert result["answer_validation"]["rejections"] == 1


def test_identifier_evidence_survives_pruning(agent, monkeypatch):
    monkeypatch.setattr(dr, "_tool_use_context_fields", ["upc", "product_id"])
    answer = "Item (upc: 001; product_id: 7). Stock remains unverified."
    agent.audits.append({
        "substantive": True, "issues": [], "items": [{
            "quote": "Item (upc: 001; product_id: 7)",
            "context_fields": {"upc": "001", "product_id": "7"},
        }],
    })
    result = agent.run(
        response(("initial_search", {})),
        response(("prune", {"docids": []})),
        response(("final_answer", {"answer": answer})),
    )
    assert result["answer"] == answer
    assert result["answer_validation"]["items"][0]["source_docids"] == ["original"]


@pytest.mark.parametrize("case", [
    "missing_boolean", "invalid_issues", "missing_items", "invented_quote",
    "missing_field", "invented_value", "split_record_values", "leading_zero_loss",
    "not_inline", "omitted_items", "numeric_value",
])
def test_answer_audit_rejects_invalid_or_ungrounded_identity_coverage(case):
    answer = "Item (upc: 001; product_id: 7)."
    item = {
        "quote": "Item (upc: 001; product_id: 7)",
        "context_fields": {"upc": "001", "product_id": "7"},
    }
    audit = {"substantive": True, "issues": [], "items": [item]}
    documents = {"one": "upc: 001\nproduct_id: 7"}
    if case == "missing_boolean":
        del audit["substantive"]
    elif case == "invalid_issues":
        audit["issues"] = "none"
    elif case == "missing_items":
        del audit["items"]
    elif case == "invented_quote":
        item["quote"] = "Another Item (upc: 001; product_id: 7)"
    elif case == "missing_field":
        del item["context_fields"]["product_id"]
    elif case == "invented_value":
        documents = {"one": "upc: 001\nproduct_id: 8"}
    elif case == "split_record_values":
        documents = {"one": "upc: 001\nproduct_id: 8", "two": "upc: 002\nproduct_id: 7"}
    elif case == "leading_zero_loss":
        answer = item["quote"] = "Item (upc: 1; product_id: 7)"
        item["context_fields"]["upc"] = "1"
    elif case == "not_inline":
        item["quote"] = "Item"
    elif case == "omitted_items":
        audit["items"] = []
    else:
        item["context_fields"]["product_id"] = 7
    with pytest.raises(ValueError):
        dr._validate_tool_use_answer_audit(answer, audit, ["upc", "product_id"], documents)


def test_answer_audit_supports_non_product_fields_markdown_and_zero_values(monkeypatch):
    monkeypatch.setattr(dr, "_tool_use_context_fields", ["asset_ref", "region_code"])
    text = dr.tool_use_fmt({"id": "one", "asset_ref": "US-001", "region_code": 0})
    answer = "**Asset** (**asset_ref:** `US-001`; **region_code:** 0)."
    items = dr._validate_tool_use_answer_audit(
        answer, {"substantive": True, "issues": [], "items": [{
            "quote": answer, "context_fields": {"asset_ref": "US-001", "region_code": "0"},
        }]}, ["asset_ref", "region_code"], {"one": text},
    )
    assert items[0]["source_docids"] == ["one"]


@pytest.mark.parametrize("value", ["A__001", "A*001", "A`001"])
def test_answer_audit_preserves_literal_markdown_characters_in_identifiers(value):
    answer = f"Item (asset_ref: {value})."
    items = dr._validate_tool_use_answer_audit(
        answer, {"substantive": True, "issues": [], "items": [{
            "quote": answer, "context_fields": {"asset_ref": value},
        }]}, ["asset_ref"], {"one": f"asset_ref: {value}"},
    )
    assert items[0]["context_fields"]["asset_ref"] == value


def test_evidence_limited_answer_can_pass_without_inventing_an_identified_item():
    assert dr._validate_tool_use_answer_audit(
        "No verified match was found; retailer availability and allergen status remain unverified.",
        {"substantive": True, "issues": [], "items": []}, ["upc", "product_id"], {},
    ) == []


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
    assert "findings" not in state
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


@pytest.mark.parametrize("word_count,omitted", [(100, 0), (1800, 1)])
def test_compaction_uses_complete_findings_within_token_budget(agent, monkeypatch, word_count, omitted):
    monkeypatch.setattr(dr, "_tool_use_context_fields", ["upc", "product_id"])
    text = dr.tool_use_fmt({
        "description": "evidence " * word_count + "Important fact beyond character 512.",
        "upc": "001234567890",
        "product_id": 0,
    })
    agent.search.side_effect = None
    agent.search.return_value = json.dumps([{"docid": "original", "text": text}])

    result = agent.run(
        response(("initial_search", {})),
        response(("prune", {"docids": ["original"]})),
        response(("final_answer", {"answer": "Evidence available; retailer stock is unverified."})),
    )

    messages = agent.requests[-1]["messages"]
    resume = next(
        message["content"] for message in messages
        if "Research progress survived" in (message.get("content") or "")
    )
    summary = json.loads(resume.split("\n", 1)[1])
    assert summary["findings_omitted_from_context"] == omitted
    assert summary["findings"] == ({} if omitted else {"original": text})
    assert any(message.get("content") == text for message in messages)
    assert dr.count_tokens(messages[:3]) <= dr._tool_use_context_limit * 0.75
    assert result["status"] == "completed"
    assert "findings" not in result["research_progress"]


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
    agent.audits.append({
        "substantive": True, "issues": [],
        "items": [{
            "quote": "Item (upc: 001; product_id: 7)",
            "context_fields": {"upc": "001", "product_id": "7"},
        }],
    })
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


def test_compaction_preserves_complete_ledger_when_context_findings_must_be_omitted(
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
    assert "findings" not in result["research_progress"]
    assert "source documents omitted" in capsys.readouterr().out
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
    assert "findings" not in result["research_progress"]
    assert agent.search.await_count == 1
