#!/usr/bin/env python
"""QA_CLI — an interactive terminal app for comparing retrieval strategies.

Ask a question, get an answer. QA_CLI loads a local ``config.yaml`` and lets you
choose HOW the answer is produced:

* **Tool-use** — the agentic LLM-driven function-calling loop
  (``dynamic_retriever.process_question``).
* **Agentic Retrieval** — the decomposed multi-round RAG pipeline
  (``dynamic_retriever.DecomposedRAGPipeline``): decompose the question into
  sub-questions, retrieve, and synthesize over several rounds.
* **Vector search** — a single-shot baseline: embed the question, run one Cosmos
  DB vector (KNN) search per source via ``dynamic_retriever.tool_use_vec_search``,
  then ask the LLM to answer from those documents. No HyDE, diversity, reranking,
  or agentic looping — a clean comparison point.
* **Compare** — runs all three strategies for each question and shows them side
  by side so you can judge the difference.

Usage::

    python qa_cli.py                          # interactive strategy menu
    python qa_cli.py --strategy tool-use      # agentic function-calling loop
    python qa_cli.py --strategy decomposed    # Agentic Retrieval (multi-round)
    python qa_cli.py --strategy vector        # plain vector search
    python qa_cli.py --strategy compare       # run all three, side by side
    python qa_cli.py --config my.yaml
    python qa_cli.py --strategy vector --k 15 # override vector top-k

Type your question at the prompt; enter ``quit`` (or an empty line / Ctrl+C) to
exit.

This file lives entirely under ``samples/`` and does not modify any project
code — it only imports and calls the existing modules.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from rich.columns import Columns
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

# --- Make the project root importable -------------------------------------
# samples/QA_CLI/qa_cli.py  ->  repo root is two levels up.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.candidates import fetch_candidate_documents, resolve_partition_key_path

console = Console()

# Toggled by the --verbose flag; controls extra step output and full tracebacks.
VERBOSE = False

# Directory (next to this script) where --verbose dumps raw research results.
RESULTS_DIR = Path(__file__).resolve().parent / "results"

STRATEGIES = ("tool-use", "decomposed", "vector", "compare")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QA_CLI: compare tool-use, Agentic Retrieval (decomposed), and "
        "plain vector search, one question at a time."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config.yaml",
        help="Path to the local config.yaml (default: ./config.yaml next to this script).",
    )
    parser.add_argument(
        "--strategy",
        choices=list(STRATEGIES),
        default=None,
        help="Answer strategy: 'tool-use' (agentic function-calling loop), "
        "'decomposed' (Agentic Retrieval multi-round RAG), 'vector' (single-shot "
        "Cosmos vector search), or 'compare' (run all three). Default: ask interactively.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Override the per-source top-k for the VECTOR strategy "
        "(default: each source's retrieval.search_k from config).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show intermediate steps and full error tracebacks. Also writes the "
        "raw research results of each method (all three for 'compare') as JSON "
        "into the results/ subfolder next to this script.",
    )
    return parser.parse_args()


def _select_strategy() -> str:
    """Interactive startup menu used when --strategy is not supplied."""
    body = Text()
    body.append("How should QA_CLI answer your questions?\n\n", style="bold")
    body.append("  1) ", style="bold cyan")
    body.append("🤖 Agentic Retrieval (tool-use)", style="bold")
    body.append("    — LLM-driven function-calling loop that decides what to search\n", style="dim")
    body.append("  2) ", style="bold cyan")
    body.append("🧩 Agentic Retrieval (decomposed)", style="bold")
    body.append(" — splits the question into sub-questions over multiple rounds\n", style="dim")
    body.append("  3) ", style="bold cyan")
    body.append("🔎 Vector search", style="bold")
    body.append("     — single-shot vector search\n", style="dim")
    body.append("  4) ", style="bold cyan")
    body.append("⚖️  Compare", style="bold")
    body.append("          — run all three, side by side", style="dim")
    console.print(Panel(body, border_style="cyan", title="🧭 Choose a strategy", title_align="left"))
    choice = Prompt.ask(
        "[bold cyan]Select[/bold cyan]",
        choices=["1", "2", "3", "4"],
        default="1",
    )
    return {"1": "tool-use", "2": "decomposed", "3": "vector", "4": "compare"}[choice]


# =============================================================================
# UI HELPERS
# =============================================================================

def _strategy_label(strategy: str) -> str:
    return {
        "tool-use": "🤖 Agentic Retrieval (tool-use)",
        "decomposed": "🧩 Agentic Retrieval (decomposed)",
        "vector": "🔎 Vector search",
        "compare": "⚖️  Compare",
    }[strategy]


def _print_banner(strategy: str, config_path: Path) -> None:
    body = Text()
    body.append("🔎 ", style="bold cyan")
    body.append("QA_CLI", style="bold white")
    body.append("  ·  Tool-use · Agentic Retrieval · Vector Search\n\n", style="dim")
    body.append("🧭 strategy: ", style="bold")
    body.append(f"{_strategy_label(strategy)}\n", style="cyan")
    body.append("📄 config  : ", style="bold")
    body.append(f"{config_path}\n\n", style="cyan")
    body.append("Type a question and press Enter. ", style="dim")
    body.append("Use ", style="dim")
    body.append("quit", style="bold yellow")
    body.append(" or Ctrl+C to exit.", style="dim")
    console.print(Panel(body, border_style="cyan", title="✨ Welcome", title_align="left"))


def _footer_text(footer_items: list[tuple[str, str]]) -> Text:
    line = Text()
    for i, (emoji, value) in enumerate(footer_items):
        if i:
            line.append("   ")
        line.append(f"{emoji} {value}", style="dim")
    return line


def _answer_panel(answer: str, footer_items: list[tuple[str, str]], title: str, border: str) -> Panel:
    content = Markdown(answer.strip() or "_(empty answer)_")
    return Panel(content, border_style=border, title=title, title_align="left")


def _print_answer(answer: str, footer_items: list[tuple[str, str]]) -> None:
    console.print(_answer_panel(answer, footer_items, "💬 Answer", "green"))
    if footer_items:
        console.print(_footer_text(footer_items))
    console.print()


def _print_comparison(
    results: list[tuple[str, str, tuple[str, list[tuple[str, str]]]]],
) -> None:
    """Render N strategy results as side-by-side panels.

    ``results`` is a list of ``(title, border_style, (answer, footer))`` tuples.
    """
    panels = [
        _answer_panel(answer, footer, title, border)
        for title, border, (answer, footer) in results
    ]
    console.print(Columns(panels, equal=True, expand=True))
    foot = Text()
    for i, (title, border, (_answer, footer)) in enumerate(results):
        if i:
            foot.append("      ")
        foot.append(f"{title.split(' ', 1)[-1]}: ", style=f"bold {border}")
        foot.append_text(_footer_text(footer))
    console.print(foot)
    console.print()


def _print_error(message: str, exc: BaseException | None = None) -> None:
    body = Text(message, style="red")
    if VERBOSE and exc is not None:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        body.append("\n\n")
        body.append(tb.rstrip(), style="white")
    console.print(Panel(body, border_style="red", title="❌ Error", title_align="left"))
    console.print()


def _vlog(message: str) -> None:
    """Print an intermediate step line only when --verbose is enabled."""
    if VERBOSE:
        console.print(f"[dim]   · {message}[/dim]")


def _save_research(method: str, question: str, payload: dict) -> Path:
    """Write one method's raw research result to results/<timestamp>_<method>.json."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    stamp = now.strftime("%Y%m%d_%H%M%S_%f")
    safe_method = method.replace("/", "-")
    path = RESULTS_DIR / f"{stamp}_{safe_method}.json"
    record = {
        "method": method,
        "question": question,
        "timestamp": now.isoformat(timespec="seconds"),
        "result": payload,
    }
    path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return path


# =============================================================================
# SHARED COSMOS SETUP
# =============================================================================

async def _build_cosmos_and_containers(dr):
    """Open a Cosmos client + per-source containers from the tool-use config.

    Returns ``(cosmos_client, credential_or_None, {source_id: container})``.
    ``dr.init_tool_use_clients()`` must have run first so ``_tool_use_cosmos_cfg``
    is populated.
    """
    from azure.cosmos.aio import CosmosClient
    from azure.identity.aio import AzureCliCredential as AsyncAzureCliCredential

    cosmos_cfg = dr._tool_use_cosmos_cfg
    sources = cosmos_cfg["sources"]
    use_rbac_auth = cosmos_cfg.get("use_rbac_auth", False)
    credential = AsyncAzureCliCredential() if use_rbac_auth else None
    cosmos = CosmosClient(
        cosmos_cfg["uri"], credential=credential or cosmos_cfg["key"]
    )
    db = cosmos.get_database_client(cosmos_cfg["database_name"])
    containers = {
        s["id"]: db.get_container_client(s["container_name"]) for s in sources
    }
    try:
        for source in sources:
            dr._tool_use_source_partition[source["id"]] = await resolve_partition_key_path(
                containers[source["id"]], source.get("partition_key_path"), source["id"],
            )
    except BaseException:
        try:
            await cosmos.close()
        finally:
            if credential is not None:
                await credential.close()
        raise
    return cosmos, credential, containers


async def _close_tool_use_clients(dr) -> None:
    """Close the module-level tool-use clients created by init_tool_use_clients()."""
    if getattr(dr, "_tool_use_llm", None) is not None:
        await dr._tool_use_llm.close()
    if getattr(dr, "_tool_use_embed_client", None) is not None:
        await dr._tool_use_embed_client.close()
    if getattr(dr, "_tool_use_use_ranker", False) and getattr(dr, "_tool_use_r_http", None) is not None:
        await dr._tool_use_r_http.aclose()


# =============================================================================
# ANSWERERS
# =============================================================================

class ToolUseAnswerer:
    """Wraps the tool-use (agentic) loop for single questions."""

    def __init__(self, dr_module) -> None:
        self._dr = dr_module
        self._cosmos = None
        self._credential = None
        self._containers: dict = {}
        self.last_research: dict | None = None

    async def setup(self) -> None:
        dr = self._dr
        dr.init_tool_use_clients()
        self._cosmos, self._credential, self._containers = (
            await _build_cosmos_and_containers(dr)
        )

    async def answer(self, question: str) -> tuple[str, list[tuple[str, str]]]:
        dr = self._dr
        q_obj = {"question_id": "qa_cli", "question_text": question}
        result = await dr.process_question(q_obj, self._containers)
        self.last_research = result
        if result.get("status") == "failed":
            raise dr.ToolUseRunError(result["error"])
        answer = result.get("answer", "") or ""
        footer: list[tuple[str, str]] = []
        if result.get("status") == "partial":
            footer.append(("⚠", "Partial answer: agent-turn limit reached"))
        if result.get("elapsed_seconds") is not None:
            footer.append(("⏱", f"{result['elapsed_seconds']}s"))
        if result.get("rounds") is not None:
            footer.append(("🔁", f"{result['rounds']} rounds"))
        tool_calls = result.get("tool_calls") or {}
        searches = tool_calls.get("initial_search", 0) + tool_calls.get("search", 0)
        if searches:
            footer.append(("🔎", f"{searches} searches"))
        return answer, footer

    def research_items(self) -> list[tuple[str, dict | None]]:
        return [("tool-use", self.last_research)]

    async def close(self) -> None:
        dr = self._dr
        if self._cosmos is not None:
            await self._cosmos.close()
        if self._credential is not None:
            await self._credential.close()
        await _close_tool_use_clients(dr)


class DecomposedAnswerer:
    """Wraps the decomposed multi-round RAG pipeline for single questions."""

    def __init__(self, dr_module, config: dict) -> None:
        self._dr = dr_module
        self._config = config
        self._retriever = None
        self._llm = None
        self._pipeline = None
        self.last_research: dict | None = None

    async def setup(self) -> None:
        dr = self._dr
        # Safe to import only AFTER load_config() (cosmos_retriever reads CONFIG
        # at import time).
        from utils.cosmos_retriever import CombinedRetriever, RETRIEVAL_SOURCES

        cfg = self._config
        retrieval_cfg = cfg.get("retrieval") or {}
        ranker_cfg = cfg.get("ranker") or {}
        pipeline_cfg = cfg.get("pipeline") or {}

        self._retriever = CombinedRetriever(
            retrieval_sources=RETRIEVAL_SOURCES,
            fulltext_k_override=None,
            k_diverse=int(retrieval_cfg.get("k_diverse", 0) or 0),
            k_ranker=int(ranker_cfg.get("k_ranker", 0) or 0),
            eta=float(retrieval_cfg.get("eta", 0.0) or 0.0),
            rescale_power=float(retrieval_cfg.get("rescale_power", 0.0) or 0.0),
        )
        await self._retriever.initialize()
        self._llm = dr.LLMClient()
        self._pipeline = dr.DecomposedRAGPipeline(
            self._retriever,
            self._llm,
            int(pipeline_cfg.get("max_sub_questions", 5) or 5),
            int(pipeline_cfg.get("rounds", 2) or 2),
            int(pipeline_cfg.get("subq_fanout_cap", 3) or 3),
            int(pipeline_cfg.get("subq_max_concurrency", 2) or 2),
        )

    async def answer(self, question: str) -> tuple[str, list[tuple[str, str]]]:
        result = await self._pipeline.run_efficient(question)
        self.last_research = result
        answer = result.get("final_answer", "") or ""
        footer: list[tuple[str, str]] = []
        rounds = result.get("rounds") or []
        footer.append(("🧩", f"{len(rounds)} rounds"))
        sub_count = sum(len(r.get("sub_questions") or []) for r in rounds)
        if sub_count:
            footer.append(("❔", f"{sub_count} sub-questions"))
        return answer, footer

    def research_items(self) -> list[tuple[str, dict | None]]:
        return [("decomposed", self.last_research)]

    async def close(self) -> None:
        if self._retriever is not None:
            await self._retriever.close()
        if self._llm is not None:
            await self._llm.close()


# Self-contained prompt: answer strictly from the retrieved context. Kept inline
# so the vector baseline does NOT borrow any agentic/decomposition prompting.
_VECTOR_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using ONLY the "
    "context documents provided. If the answer is not contained in the context, "
    "say you don't know rather than guessing. Be concise and accurate."
)


class VectorSearchAnswerer:
    """Single-shot plain vector (KNN) search baseline over Cosmos DB.

    Pipeline: embed the question -> one ``tool_use_vec_search`` per source ->
    merge + dedupe -> one LLM call answering from the retrieved context. No HyDE,
    diversity selection, reranking, or agentic looping.
    """

    def __init__(self, dr_module, top_k_override: int | None = None) -> None:
        self._dr = dr_module
        self._top_k_override = top_k_override
        self._cosmos = None
        self._credential = None
        self._containers: dict = {}
        self.last_research: dict | None = None

    async def setup(self) -> None:
        dr = self._dr
        dr.init_tool_use_clients()
        self._cosmos, self._credential, self._containers = (
            await _build_cosmos_and_containers(dr)
        )

    async def answer(self, question: str) -> tuple[str, list[tuple[str, str]]]:
        dr = self._dr
        start = time.perf_counter()

        emb = await dr.tool_use_embed(question)

        # One KNN search per source, then merge preserving per-source order.
        source_cfg = dr._tool_use_source_cfg
        source_embed = dr._tool_use_source_embed
        references = []
        for sid, container in self._containers.items():
            top_k = self._top_k_override or int(source_cfg.get(sid, {}).get("search_k", 10) or 10)
            embed_field = source_embed.get(sid, "embedding")
            candidates = await dr.tool_use_vec_search(
                container, emb, top_k, embed_field,
                partition_key_path=dr._tool_use_source_partition[sid],
            )
            references.extend((container, candidate) for candidate in candidates)

        merged = await fetch_candidate_documents(references)
        context = "\n\n".join(dr.tool_use_fmt(doc) for doc in merged)
        if not context.strip():
            answer = "I don't know — the vector search returned no documents."
            elapsed = round(time.perf_counter() - start, 2)
            self.last_research = {"answer": answer, "num_docs": 0, "documents": []}
            return answer, [("⏱", f"{elapsed}s"), ("📄", "0 docs")]

        user_content = (
            f"Context documents:\n{context}\n\n"
            f"Question: {question}\n\nAnswer:"
        )
        r = await dr.tool_use_chat_completion(
            model=dr._tool_use_llm_cfg["llm_model"],
            messages=[
                {"role": "system", "content": _VECTOR_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=dr._tool_use_llm_cfg.get("temperature", 0.0),
            max_completion_tokens=dr._tool_use_llm_cfg["max_completion_tokens"],
        )
        answer = (r.choices[0].message.content or "").strip()
        elapsed = round(time.perf_counter() - start, 2)
        # Strip the (large) embedding vectors before recording the retrieved docs.
        embed_fields = set(source_embed.values()) | {"embedding"}
        clean_docs = [
            {k: v for k, v in doc.items() if k not in embed_fields} for doc in merged
        ]
        self.last_research = {
            "answer": answer,
            "elapsed_seconds": elapsed,
            "num_docs": len(merged),
            "documents": clean_docs,
        }
        footer = [("⏱", f"{elapsed}s"), ("📄", f"{len(merged)} docs")]
        return answer, footer

    def research_items(self) -> list[tuple[str, dict | None]]:
        return [("vector", self.last_research)]

    async def close(self) -> None:
        dr = self._dr
        if self._cosmos is not None:
            await self._cosmos.close()
        if self._credential is not None:
            await self._credential.close()
        await _close_tool_use_clients(dr)


class CompareAnswerer:
    """Runs tool-use, Agentic Retrieval (decomposed), and vector for each question.

    The tool-use and vector answerers share one set of module-level tool-use
    clients (``init_tool_use_clients`` is idempotent for this sample); the vector
    answerer reuses those clients with its own Cosmos containers. The decomposed
    answerer owns its own retriever + LLM client. Cleanup is coordinated so the
    shared tool-use clients are closed exactly once (by the tool-use answerer).
    """

    # (title, border_style) for each side-by-side panel.
    _PANELS = (
        ("🤖 Agentic Retrieval (tool-use)", "blue"),
        ("🧩 Agentic Retrieval (decomposed)", "cyan"),
        ("🔎 Vector search", "magenta"),
    )

    def __init__(self, dr_module, config: dict, top_k_override: int | None) -> None:
        self._dr = dr_module
        self._tooluse = ToolUseAnswerer(dr_module)
        self._decomposed = DecomposedAnswerer(dr_module, config)
        self._vector = VectorSearchAnswerer(dr_module, top_k_override)

    async def setup(self) -> None:
        # Tool-use first: it initializes the shared module-level clients.
        await self._tooluse.setup()
        await self._decomposed.setup()
        # Vector answerer reuses the already-initialized tool-use clients and
        # only needs its own Cosmos containers.
        dr = self._dr
        dr.init_tool_use_clients()
        (
            self._vector._cosmos,
            self._vector._credential,
            self._vector._containers,
        ) = await _build_cosmos_and_containers(dr)

    async def answer_all(
        self, question: str
    ) -> list[tuple[str, str, tuple[str, list[tuple[str, str]]]]]:
        # Run sequentially to keep output (and any verbose logs) readable.
        tooluse_result = await self._tooluse.answer(question)
        decomposed_result = await self._decomposed.answer(question)
        vector_result = await self._vector.answer(question)
        results = (tooluse_result, decomposed_result, vector_result)
        return [
            (title, border, result)
            for (title, border), result in zip(self._PANELS, results)
        ]

    def research_items(self) -> list[tuple[str, dict | None]]:
        return (
            self._tooluse.research_items()
            + self._decomposed.research_items()
            + self._vector.research_items()
        )

    async def close(self) -> None:
        # Close the vector answerer's Cosmos resources WITHOUT closing the shared
        # tool-use clients (the tool-use answerer closes those).
        if self._vector._cosmos is not None:
            await self._vector._cosmos.close()
        if self._vector._credential is not None:
            await self._vector._credential.close()
        await self._decomposed.close()
        await self._tooluse.close()


# =============================================================================
# MAIN LOOP
# =============================================================================

async def _run(args: argparse.Namespace) -> int:
    config_path: Path = args.config
    if not config_path.exists():
        _print_error(
            f"Config file not found: {config_path}\n"
            "Copy config.yaml.example to config.yaml and fill in your values."
        )
        return 1

    # IMPORTANT: load_config() must run BEFORE importing utils.cosmos_retriever,
    # which reads CONFIG at import time.
    import dynamic_retriever as dr

    _vlog(f"Loading config from {config_path} ...")
    dr.load_config(config_path)
    config = dr.CONFIG
    _vlog("Config loaded.")

    strategy = args.strategy or _select_strategy()

    _print_banner(strategy, config_path)

    if strategy == "tool-use":
        answerer = ToolUseAnswerer(dr)
    elif strategy == "decomposed":
        answerer = DecomposedAnswerer(dr, config)
    elif strategy == "vector":
        answerer = VectorSearchAnswerer(dr, args.k)
    else:  # compare
        answerer = CompareAnswerer(dr, config, args.k)

    _vlog(f"Initializing {strategy!r} strategy ...")
    with console.status("🛠️  Connecting to Cosmos DB and models...", spinner="dots"):
        try:
            await answerer.setup()
        except Exception as exc:  # noqa: BLE001 - surface any setup failure to the user
            _print_error(f"Failed to initialize ({type(exc).__name__}): {exc}", exc)
            await _safe_close(answerer)
            return 1
    _vlog("Strategy ready.")

    exit_code = 0
    try:
        while True:
            try:
                question = Prompt.ask("[bold cyan]❓ Your question[/bold cyan]")
            except (EOFError, KeyboardInterrupt):
                console.print()
                break

            question = question.strip()
            if not question or question.lower() in {"quit", "exit", ":q"}:
                break

            try:
                _vlog(f"Dispatching question to {strategy!r} strategy ...")
                start = time.perf_counter()
                with console.status("🔎 Retrieving & reasoning...", spinner="dots"):
                    if strategy == "compare":
                        compare_results = await answerer.answer_all(question)
                    else:
                        answer, footer = await answerer.answer(question)
                _vlog(f"Answer produced in {time.perf_counter() - start:.2f}s.")
                if strategy == "compare":
                    _print_comparison(compare_results)
                else:
                    _print_answer(answer, footer)
                if VERBOSE:
                    for method, payload in answerer.research_items():
                        if payload is None:
                            continue
                        try:
                            saved = _save_research(method, question, payload)
                            console.print(
                                f"[dim]   · saved {method} research results → {saved}[/dim]"
                            )
                        except Exception as exc:  # noqa: BLE001 - saving is best-effort
                            console.print(
                                f"[dim]   · could not save {method} results: "
                                f"{type(exc).__name__}: {exc}[/dim]"
                            )
            except KeyboardInterrupt:
                console.print("\n[dim]Cancelled.[/dim]\n")
            except Exception as exc:  # noqa: BLE001 - keep the REPL alive on errors
                _print_error(f"{type(exc).__name__}: {exc}", exc)
    finally:
        await _safe_close(answerer)

    console.print("[dim]👋 Goodbye![/dim]")
    return exit_code


async def _safe_close(answerer) -> None:
    try:
        await answerer.close()
    except Exception as exc:  # noqa: BLE001 - cleanup must not raise
        console.print(f"[dim]Cleanup warning: {type(exc).__name__}: {exc}[/dim]")


def main() -> None:
    args = _parse_args()
    global VERBOSE
    VERBOSE = bool(args.verbose)
    try:
        exit_code = asyncio.run(_run(args))
    except KeyboardInterrupt:
        exit_code = 130
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
