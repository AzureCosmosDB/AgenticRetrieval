import ast
import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from utils.candidates import fetch_candidate_documents, resolve_partition_key_path


def load_sample():
    path = Path(__file__).resolve().parents[1] / "samples" / "QA_CLI" / "qa_cli.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {"ToolUseAnswerer", "VectorSearchAnswerer", "CompareAnswerer", "_build_cosmos_and_containers"}
    nodes = [
        node for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.AsyncFunctionDef)) and node.name in names
    ]
    namespace = {
        "asyncio": asyncio, "time": time, "json": json,
        "fetch_candidate_documents": fetch_candidate_documents,
        "resolve_partition_key_path": resolve_partition_key_path,
        "_VECTOR_SYSTEM_PROMPT": "Answer from context only.",
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("strategy", ["vector", "compare"])
def test_sample_materializes_partition_aware_vector_candidates(strategy):
    sample = load_sample()
    container = SimpleNamespace(read_item=AsyncMock(
        side_effect=lambda item, partition_key: {
            "id": item, "pk": partition_key, "title": f"Full product {partition_key}", "e": [0.1],
        },
    ))
    candidates = [
        {"id": "same", "_partition_key": "001", "_score": 0.1},
        {"id": "same", "_partition_key": "002", "_score": 0.2},
    ]
    async def search(proxy, embedding, top_k, field, *, partition_key_path):
        assert proxy is container
        assert partition_key_path == "/identity/upc"
        assert (embedding, top_k, field) == ([0.1], 2, "e")
        return candidates

    create = AsyncMock(return_value=SimpleNamespace(choices=[
        SimpleNamespace(message=SimpleNamespace(content="Grounded answer")),
    ]))
    dr = SimpleNamespace(
        tool_use_embed=AsyncMock(return_value=[0.1]), tool_use_vec_search=search,
        tool_use_fmt=lambda doc: doc["title"],
        _tool_use_source_cfg={"products": {"search_k": 2}},
        _tool_use_source_embed={"products": "e"},
        _tool_use_source_partition={"products": "/identity/upc"},
        _tool_use_llm=SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=create),
        )),
        _tool_use_llm_cfg={"llm_model": "test", "max_completion_tokens": 100},
    )
    vector = sample["VectorSearchAnswerer"](dr)
    vector._containers = {"products": container}
    if strategy == "vector":
        answer, _ = asyncio.run(vector.answer("question"))
        assert answer == "Grounded answer"
    else:
        compare = object.__new__(sample["CompareAnswerer"])
        compare._tooluse = SimpleNamespace(answer=AsyncMock(return_value=("tool answer", [])))
        compare._decomposed = SimpleNamespace(answer=AsyncMock(return_value=("decomposed answer", [])))
        compare._vector = vector
        results = asyncio.run(compare.answer_all("question"))
        assert len(results) == 3
        assert results[-1][2][0] == "Grounded answer"
    context = create.await_args.kwargs["messages"][1]["content"]
    assert "Full product 001" in context and "Full product 002" in context
    assert [call.kwargs["partition_key"] for call in container.read_item.await_args_list] == ["001", "002"]
    assert vector.last_research["num_docs"] == 2
    assert all("e" not in doc for doc in vector.last_research["documents"])


@pytest.mark.parametrize("invalid", [False, True])
def test_sample_setup_resolves_actual_partition_paths_and_cleans_up(monkeypatch, invalid):
    import azure.cosmos.aio

    sample = load_sample()
    container = SimpleNamespace(read=AsyncMock(return_value={
        "partitionKey": {"paths": [] if invalid else ["/identity/upc"]},
    }))
    database = Mock()
    database.get_container_client.return_value = container
    cosmos = Mock()
    cosmos.get_database_client.return_value = database
    cosmos.close = AsyncMock()
    monkeypatch.setattr(azure.cosmos.aio, "CosmosClient", Mock(return_value=cosmos))
    dr = SimpleNamespace(
        _tool_use_source_partition={},
        _tool_use_cosmos_cfg={
            "uri": "https://example.invalid", "key": "test-key", "database_name": "db",
            "sources": [{"id": "products", "container_name": "products", "partition_key_path": "/stale"}],
        },
    )
    if invalid:
        with pytest.raises(ValueError, match="no partition key"):
            asyncio.run(sample["_build_cosmos_and_containers"](dr))
        cosmos.close.assert_awaited_once()
    else:
        _, credential, containers = asyncio.run(sample["_build_cosmos_and_containers"](dr))
        assert credential is None
        assert containers == {"products": container}
        assert dr._tool_use_source_partition == {"products": "/identity/upc"}
        cosmos.close.assert_not_awaited()


@pytest.mark.parametrize("status", ["failed", "partial"])
def test_sample_surfaces_tool_use_finalization_status(status):
    sample = load_sample()
    result = {
        "answer": "" if status == "failed" else "Evidence-supported partial answer",
        "status": status, "error": "Final synthesis failed",
    }
    dr = SimpleNamespace(
        process_question=AsyncMock(return_value=result), ToolUseRunError=RuntimeError,
    )
    answerer = sample["ToolUseAnswerer"](dr)
    if status == "failed":
        with pytest.raises(RuntimeError, match="Final synthesis failed"):
            asyncio.run(answerer.answer("question"))
    else:
        answer, footer = asyncio.run(answerer.answer("question"))
        assert answer
        assert any("Partial answer" in text for _, text in footer)
    assert answerer.last_research is result
