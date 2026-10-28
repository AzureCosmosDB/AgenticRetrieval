import ast
import argparse
import asyncio
import contextvars
import datetime
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from dynamic_retriever import _nonempty_question
from utils.ranker import RankerUnavailableError


@pytest.mark.parametrize("interrupted", [False, True, "failed_result"])
def test_completed_answers_persist_before_next_question(tmp_path, interrupted):
    source = Path(__file__).resolve().parents[1] / "dynamic_retriever.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    node = next(
        node for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "run_tool_use_mode"
    )
    questions = [
        {"question_id": "q1", "question_text": "First question"},
        {"question_id": "q2", "question_text": "Second question"},
    ]
    question_path = tmp_path / "questions.json"
    question_path.write_text(json.dumps(questions), encoding="utf-8")
    cosmos = Mock()
    cosmos.close = AsyncMock()
    db = Mock()
    cosmos.get_database_client.return_value = db
    credential = AsyncMock()
    llm = AsyncMock()
    embed = AsyncMock()
    ranker = Mock()
    ranker.aclose = AsyncMock()
    completed = {"question_id": "q1", "answer": "Full first answer", "elapsed_seconds": 1.2}

    async def process(question, containers):
        if question["question_id"] == "q1":
            return completed
        files = list((tmp_path / "standard").glob("results_*.json"))
        assert len(files) == 1
        assert json.loads(files[0].read_text(encoding="utf-8")) == [completed]
        if interrupted is True:
            raise RuntimeError("Second question failed")
        if interrupted == "failed_result":
            return {
                "question_id": "q2", "answer": "", "status": "failed",
                "error": "Final synthesis was empty",
            }
        return {"question_id": "q2", "answer": "Full second answer", "elapsed_seconds": 2.3}

    namespace = {
        "Path": Path, "json": json, "time": time,
        "init_tool_use_clients": Mock(),
        "_tool_use_cosmos_cfg": {
            "sources": [], "use_rbac_auth": True, "uri": "https://example.invalid",
            "database_name": "test",
        },
        "_log_line": Mock(), "_ck": Mock(), "tqdm": lambda items: items,
        "AsyncAzureCliCredential": Mock(return_value=credential),
        "CosmosClient": Mock(return_value=cosmos), "process_question": process,
        "_tool_use_llm": llm, "_tool_use_embed_client": embed,
        "_tool_use_use_ranker": True, "_tool_use_r_http": ranker,
        "_tool_use_r_tok": "token", "_tool_use_r_fail": False,
        "_tool_use_source_partition": {}, "resolve_partition_key_path": AsyncMock(),
        "verify_ranker_token": AsyncMock(return_value=None),
        "ToolUseRunError": RuntimeError,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    args = SimpleNamespace(
        questions_path=question_path, output_root=tmp_path, max_questions=None
    )
    if interrupted:
        message = "Second question failed" if interrupted is True else "1 question\\(s\\) failed"
        with pytest.raises(RuntimeError, match=message):
            asyncio.run(namespace["run_tool_use_mode"](args))
    else:
        asyncio.run(namespace["run_tool_use_mode"](args))

    files = list((tmp_path / "standard").glob("results_*.json"))
    saved = json.loads(files[0].read_text(encoding="utf-8"))
    assert len(saved) == (1 if interrupted is True else 2)
    if interrupted == "failed_result":
        assert saved[-1]["status"] == "failed"
        assert saved[-1]["error"] == "Final synthesis was empty"
    assert saved[0] == completed
    assert not list((tmp_path / "standard").glob("*.tmp"))
    for resource in (cosmos, credential, llm, embed):
        resource.close.assert_awaited_once()
    ranker.aclose.assert_awaited_once()


@pytest.mark.parametrize("fatal", [False, True])
@pytest.mark.parametrize("separate_subq_calls", [False, True])
def test_decomposed_fatal_ranker_error_cancels_work_and_preserves_completed_traces(
    monkeypatch, tmp_path, fatal, separate_subq_calls,
):
    source = Path(__file__).resolve().parents[1] / "dynamic_retriever.py"
    node = next(
        node for node in ast.parse(source.read_text(encoding="utf-8")).body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_run_decomposed_main"
    )
    questions = [
        SimpleNamespace(question_id=f"q{i}", question_text=text, group="questions", ground_truth=None)
        for i, text in enumerate(("first", "ranker", "pending"), start=1)
    ]
    retriever = SimpleNamespace(
        total_fulltext_k=2, total_vector_k=2, source_count=1,
        initialize=AsyncMock(), close=AsyncMock(),
    )
    llm = SimpleNamespace(
        close=AsyncMock(), total_prompt_chars=0, total_llm_calls=0,
    )
    trace = tmp_path / "k4_ft2_vec2" / "intermediate" / "questions" / "q1.json"
    cancelled = []

    async def run(question):
        if question == "ranker" and fatal:
            for _ in range(1000):
                if trace.exists():
                    break
                await asyncio.sleep(0.001)
            assert trace.exists(), "First answer must persist before the fatal ranker error"
            raise RankerUnavailableError("Ranker failed after startup")
        if question == "pending" and fatal:
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.append(question)
        return {"final_answer": f"Answer to {question}"}

    pipeline = SimpleNamespace(run=AsyncMock(side_effect=run), run_efficient=AsyncMock(side_effect=run))
    monkeypatch.setitem(sys.modules, "utils.cosmos_retriever", SimpleNamespace(
        CombinedRetriever=Mock(return_value=retriever), RETRIEVAL_SOURCES=[],
    ))
    args = ["dynamic_retriever.py", "--config", str(tmp_path / "config.yaml")]
    if separate_subq_calls:
        args.append("--separate-subq-calls")
    monkeypatch.setattr(sys, "argv", args)
    namespace = {
        "Path": Path, "Question": SimpleNamespace,
        "argparse": argparse, "asyncio": asyncio, "json": json, "time": time,
        "datetime": datetime, "RankerUnavailableError": RankerUnavailableError,
        "_CURRENT_QUESTION_ID": contextvars.ContextVar("test_question", default=None),
        "_log_line": Mock(), "_ck": Mock(return_value=0), "tqdm": Mock(return_value=MagicMock()),
        "LLMClient": Mock(return_value=llm),
        "DecomposedRAGPipeline": Mock(return_value=pipeline),
        "load_questions": lambda path: {"questions": questions},
        "_nonempty_question": _nonempty_question,
        "CONFIG": {
            "retrieval": {"k_diverse": 0, "eta": 0, "rescale_power": 0},
            "execution": {"max_questions": None, "max_workers": 3},
            "paths": {"questions_path": str(tmp_path / "questions.json"), "output_root": str(tmp_path)},
            "llm": {"llm_model": "test"},
        },
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    if fatal:
        with pytest.raises(RankerUnavailableError, match="after startup"):
            asyncio.run(namespace["_run_decomposed_main"](tmp_path / "config.yaml"))
        assert cancelled == ["pending"]
        assert json.loads(trace.read_text(encoding="utf-8"))["final_answer"] == "Answer to first"
        assert not list((tmp_path / "k4_ft2_vec2").glob("questions_*.json"))
    else:
        asyncio.run(namespace["_run_decomposed_main"](tmp_path / "config.yaml"))
        output = next((tmp_path / "k4_ft2_vec2").glob("questions_*.json"))
        assert len(json.loads(output.read_text(encoding="utf-8"))) == 3
    retriever.close.assert_awaited_once()
    llm.close.assert_awaited_once()


@pytest.mark.parametrize("efficient", [False, True])
def test_fatal_ranker_error_cancels_sibling_subquestions(efficient):
    source = Path(__file__).resolve().parents[1] / "dynamic_retriever.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    pipeline = next(
        node for node in tree.body if isinstance(node, ast.ClassDef)
        and node.name == "DecomposedRAGPipeline"
    )
    name = "run_efficient" if efficient else "_answer_subquestions_bounded"
    method = next(node for node in pipeline.body if isinstance(node, ast.AsyncFunctionDef) and node.name == name)
    namespace = {
        "asyncio": asyncio, "RankerUnavailableError": RankerUnavailableError,
        "SubQuestionResult": SimpleNamespace, "RetrievedChunk": SimpleNamespace,
        "_ck": Mock(return_value=0), "EFFICIENT_PRELIMINARY_PROMPT": "{question}",
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    cancelled = []

    async def execute():
        started = asyncio.Event()

        async def answer_subquestion(question, **kwargs):
            if question == "original":
                return []
            if question == "ranker":
                await started.wait()
                raise RankerUnavailableError("Fatal ranker failure")
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.append(question)

        instance = SimpleNamespace(
            subq_max_concurrency=2, num_rounds=1,
            _answer_subquestion=answer_subquestion,
            retriever=SimpleNamespace(retrieve=answer_subquestion),
            llm=SimpleNamespace(complete=AsyncMock(return_value="Preliminary")),
            _format_context=lambda chunks: "",
            _get_subquestions=AsyncMock(return_value=["ranker", "pending"]),
        )
        argument = "original" if efficient else ["ranker", "pending"]
        with pytest.raises(RankerUnavailableError, match="Fatal ranker"):
            await namespace[name](instance, argument)
        assert cancelled == ["pending"]

    asyncio.run(execute())
