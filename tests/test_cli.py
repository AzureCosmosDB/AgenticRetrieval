import ast
import asyncio
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import dynamic_retriever as dr


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cli(monkeypatch, tmp_path):
    question_path = tmp_path / "questions.json"
    question_path.write_text(json.dumps([
        {"question_id": "first", "question_text": "first"},
        {"question_id": "second", "question_text": "second"},
    ]), encoding="utf-8")
    config = {
        "pipeline": {},
        "paths": {"questions_path": str(question_path), "output_root": str(tmp_path)},
        "execution": {"max_questions": None, "max_workers": 2},
        "retrieval": {"k_diverse": 0, "eta": 0, "rescale_power": 0},
        "llm": {"llm_model": "test"},
    }
    monkeypatch.setattr(dr, "CONFIG", config)
    monkeypatch.setattr(dr, "load_config", Mock(side_effect=lambda path: print("Loaded config")))
    monkeypatch.setattr(dr, "_TIMING", False)
    monkeypatch.setattr(dr, "init_tool_use_clients", Mock())
    monkeypatch.setattr(dr, "_tool_use_cosmos_cfg", {
        "sources": [], "uri": "https://example.invalid", "key": "test",
        "database_name": "test", "use_rbac_auth": False,
    })
    monkeypatch.setattr(dr, "_tool_use_use_ranker", False)
    monkeypatch.setattr(dr, "_tool_use_llm", None)
    monkeypatch.setattr(dr, "_tool_use_embed_client", None)
    cosmos = Mock()
    cosmos.close = AsyncMock()
    monkeypatch.setattr(dr, "CosmosClient", Mock(return_value=cosmos))
    retriever = SimpleNamespace(
        total_fulltext_k=0, total_vector_k=1, source_count=1,
        initialize=AsyncMock(), close=AsyncMock(),
    )
    llm = SimpleNamespace(close=AsyncMock(), total_prompt_chars=0, total_llm_calls=0)
    monkeypatch.setattr(dr, "LLMClient", Mock(return_value=llm))
    monkeypatch.setitem(sys.modules, "utils.cosmos_retriever", SimpleNamespace(
        CombinedRetriever=Mock(return_value=retriever), RETRIEVAL_SOURCES=[],
    ))
    seen = []
    completed = []
    failure = None

    async def answer(text):
        seen.append(text)
        print(f"Retrieving: {text}")
        if failure is not None:
            raise failure
        if text == "first":
            await asyncio.sleep(0.01)
        completed.append(text)
        return f"Answer: {text} (upc: 001234)"

    async def tool_process(question, containers):
        try:
            text = await answer(question["question_text"])
        except ValueError as error:
            return {
                "question_id": question["question_id"], "answer": "",
                "status": "failed", "error": str(error),
            }
        return {"question_id": question["question_id"], "answer": text, "status": "completed"}

    monkeypatch.setattr(dr, "process_question", AsyncMock(side_effect=tool_process))

    async def decomposed_process(text):
        return {"final_answer": await answer(text)}

    pipeline = SimpleNamespace(
        run=AsyncMock(side_effect=decomposed_process),
        run_efficient=AsyncMock(side_effect=decomposed_process),
    )
    monkeypatch.setattr(dr, "DecomposedRAGPipeline", Mock(return_value=pipeline))

    def arguments(mode, *extra):
        monkeypatch.setattr(sys, "argv", [
            str(ROOT / "dynamic_retriever.py"), "--config", str(tmp_path / "config.yaml"),
            "--mode", mode, *extra,
        ])

    def fail(error):
        nonlocal failure
        failure = error

    return SimpleNamespace(
        config=config, arguments=arguments, seen=seen, completed=completed, fail=fail,
        cosmos=cosmos, retriever=retriever, llm=llm, pipeline=pipeline,
    )


@pytest.mark.parametrize("mode", ["tool-use", "decomposed"])
@pytest.mark.parametrize("separate", [False, True])
def test_single_question_overrides_file_and_limit_and_separates_output(
    cli, capsys, tmp_path, mode, separate,
):
    del cli.config["paths"]["questions_path"]
    extra = ["--separate-subq-calls"] if mode == "decomposed" and separate else []
    cli.arguments(
        mode, "--question", "  Serum?  ", "--questions-path", str(tmp_path / "missing.json"),
        "--max-questions", "0", "--answer-only", *extra,
    )
    disabled = logging.root.manager.disable

    asyncio.run(dr.main_async())

    captured = capsys.readouterr()
    assert captured.out == "Answer: Serum? (upc: 001234)\n"
    assert "Loaded config" in captured.err
    assert "Retrieving: Serum?" in captured.err
    assert "Saved" in captured.err or "Done!" in captured.err
    assert "\r" not in captured.err
    assert logging.root.manager.disable == disabled
    assert cli.seen == ["Serum?"]
    if mode == "tool-use":
        saved = next((tmp_path / "standard").glob("results_*.json"))
        cli.cosmos.close.assert_awaited_once()
    else:
        saved = next((tmp_path / "k1_ft0_vec1").glob("cli_*.json"))
        cli.retriever.close.assert_awaited_once()
        cli.llm.close.assert_awaited_once()
        assert json.loads(saved.read_text(encoding="utf-8"))[0]["ground_truth"] is None
    assert json.loads(saved.read_text(encoding="utf-8"))[0]["question_id"] == "cli"


@pytest.mark.parametrize("mode", ["tool-use", "decomposed"])
def test_answer_only_batch_keeps_input_order(cli, capsys, mode):
    cli.arguments(mode, "--answer-only")

    asyncio.run(dr.main_async())

    captured = capsys.readouterr()
    assert captured.out == (
        "Answer: first (upc: 001234)\n\nAnswer: second (upc: 001234)\n"
    )
    if mode == "decomposed":
        assert cli.completed == ["second", "first"]


@pytest.mark.parametrize("mode", ["tool-use", "decomposed"])
def test_single_question_without_answer_only_keeps_normal_diagnostics(cli, capsys, mode):
    cli.arguments(mode, "--question", "Serum?")

    asyncio.run(dr.main_async())

    assert cli.seen == ["Serum?"]
    assert "Loaded config" in capsys.readouterr().out


@pytest.mark.parametrize("mode", ["tool-use", "decomposed"])
def test_answer_only_failures_are_visible_and_restore_stdout(cli, capsys, tmp_path, mode):
    cli.arguments(mode, "--question", "Serum?", "--answer-only")
    cli.fail(ValueError("Backend rejected request"))
    stdout = sys.stdout
    error = dr.ToolUseRunError if mode == "tool-use" else dr.DecomposedRunError

    with pytest.raises(error, match="question\\(s\\) failed"):
        asyncio.run(dr.main_async())

    assert sys.stdout is stdout
    captured = capsys.readouterr()
    assert captured.out == ""
    if mode == "decomposed":
        assert "Backend rejected request" in captured.err
        cli.retriever.close.assert_awaited_once()
        cli.llm.close.assert_awaited_once()
    else:
        saved = next((tmp_path / "standard").glob("results_*.json"))
        assert json.loads(saved.read_text(encoding="utf-8"))[0]["error"] == "Backend rejected request"
        cli.cosmos.close.assert_awaited_once()


@pytest.mark.parametrize("mode", ["tool-use", "decomposed"])
def test_blank_question_fails_before_configuration_or_clients(cli, capsys, mode):
    cli.arguments(mode, "--question", " \t ", "--answer-only")

    with pytest.raises(SystemExit) as error:
        asyncio.run(dr.main_async())

    assert error.value.code == 2
    assert "must not be empty" in capsys.readouterr().err
    dr.load_config.assert_not_called()
    dr.init_tool_use_clients.assert_not_called()
    dr.LLMClient.assert_not_called()


@pytest.mark.parametrize("mode", ["tool-use", "decomposed"])
def test_missing_questions_input_has_actionable_error(cli, capsys, mode):
    del cli.config["paths"]["questions_path"]
    cli.arguments(mode, "--answer-only")

    with pytest.raises(SystemExit) as error:
        asyncio.run(dr.main_async())

    assert error.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "--questions-path is required unless --question is supplied" in captured.err
    dr.init_tool_use_clients.assert_not_called()
    dr.LLMClient.assert_not_called()


@pytest.mark.parametrize("mode", ["tool-use", "decomposed"])
def test_help_remains_visible_in_answer_only_mode(cli, capsys, mode):
    cli.arguments(mode, "--answer-only", "--help")

    with pytest.raises(SystemExit) as error:
        asyncio.run(dr.main_async())

    assert error.value.code == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "--question " in captured.err
    assert "--answer-only" in captured.err
    assert cli.seen == []


@pytest.mark.parametrize("mode", ["tool-use", "decomposed"])
def test_unknown_arguments_remain_visible_in_answer_only_mode(cli, capsys, mode):
    cli.arguments(mode, "--answer-only", "--unknown-option")

    with pytest.raises(SystemExit) as error:
        asyncio.run(dr.main_async())

    assert error.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "unrecognized arguments: --unknown-option" in captured.err


def test_answer_only_rejects_empty_decomposed_answer(cli, capsys):
    cli.arguments("decomposed", "--question", "Serum?", "--answer-only")
    cli.pipeline.run_efficient.side_effect = None
    cli.pipeline.run_efficient.return_value = {"final_answer": ""}

    with pytest.raises(dr.DecomposedRunError, match="nonempty string"):
        asyncio.run(dr.main_async())

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "nonempty string" in captured.err
    cli.retriever.close.assert_awaited_once()
    cli.llm.close.assert_awaited_once()


@pytest.mark.parametrize("mode", ["tool-use", "decomposed"])
def test_answer_only_preserves_unicode(cli, capsys, mode):
    question = "S\u00e9rum?"
    cli.arguments(mode, "--question", question, "--answer-only")

    asyncio.run(dr.main_async())

    assert capsys.readouterr().out == f"Answer: {question} (upc: 001234)\n"


def run_script_entry(tmp_path):
    source = ROOT / "dynamic_retriever.py"
    entry = next(
        node for node in ast.parse(source.read_text(encoding="utf-8")).body
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__"
    )
    namespace = dict(vars(dr), __name__="__main__", __file__=str(tmp_path / source.name))
    exec(compile(ast.Module(body=[entry], type_ignores=[]), str(source), "exec"), namespace)


@pytest.mark.parametrize("mode", ["tool-use", "decomposed"])
def test_timing_entry_point_keeps_stdout_clean_and_writes_logs(cli, capsys, tmp_path, mode):
    cli.arguments(mode, "--question", "Serum?", "--answer-only", "--timing")

    run_script_entry(tmp_path)

    captured = capsys.readouterr()
    assert captured.out == "Answer: Serum? (upc: 001234)\n"
    assert "wrote log:" in captured.err
    assert "updated latest:" in captured.err
    log = (tmp_path / "out" / "timing_5q_latest.log").read_text(encoding="utf-8")
    assert "Loaded config" in log
    assert "Retrieving: Serum?" in log


@pytest.mark.parametrize("mode", ["tool-use", "decomposed"])
@pytest.mark.parametrize("timing", [False, True])
def test_script_failure_has_nonzero_exit_and_stderr_diagnostic(cli, capsys, tmp_path, mode, timing):
    extra = ["--timing"] if timing else []
    cli.arguments(mode, "--question", "Serum?", "--answer-only", *extra)
    cli.fail(ValueError("Backend rejected request"))

    with pytest.raises(SystemExit) as error:
        run_script_entry(tmp_path)

    assert error.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "ERROR:" in captured.err
    assert "question(s) failed" in captured.err


@pytest.mark.parametrize("mode", ["tool-use", "decomposed"])
def test_script_rejects_empty_question_with_nonzero_exit(tmp_path, mode):
    result = subprocess.run(
        [
            sys.executable, str(ROOT / "dynamic_retriever.py"), "--config",
            str(tmp_path / "missing.yaml"), "--mode", mode,
            "--question", " ", "--answer-only",
        ],
        capture_output=True, text=True, encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"}, timeout=30,
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert "must not be empty" in result.stderr
