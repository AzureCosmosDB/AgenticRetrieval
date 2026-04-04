#!/usr/bin/env python
"""RAG pipeline rewritten using the OpenAI Agents SDK.

This replaces the hand-built DecomposedRAGPipeline with agents that have
tools for retrieval, gap analysis, and synthesis.

Usage:
    python agentic_retriever_sdk.py --config config.yaml --cosmos-az-login --max-questions 1
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from openai import AsyncAzureOpenAI
from azure.identity import AzureCliCredential, get_bearer_token_provider
from tqdm import tqdm

from agents import (
    Agent,
    Runner,
    RunConfig,
    ModelSettings,
    OpenAIProvider,
)

# We reuse the existing retriever and config infrastructure
import agentic_retriever
from agentic_retriever import (
    CONFIG,
    load_config,
    RetrievedChunk,
    _log_line,
)
from prompts_sdk import (
    PRELIMINARY_INSTRUCTIONS,
    EFFICIENT_PRELIMINARY_INSTRUCTIONS,
    GAP_ANALYZER_INSTRUCTIONS,
    SUBQUESTION_INSTRUCTIONS,
    SYNTHESIZER_INSTRUCTIONS,
    FINAL_SYNTHESIS_INSTRUCTIONS,
    EFFICIENT_REGENERATOR_INSTRUCTIONS,
    EFFICIENT_SYNTHESIS_INSTRUCTIONS,
)

# =============================================================================
# CONTEXT: shared state passed through all agent tool calls
# =============================================================================

@dataclass
class RAGContext:
    """Shared context for the agent run — holds the retriever and accumulated state."""
    retriever: Any  # CombinedRetriever
    question: str
    retrieved_chunks: list[RetrievedChunk]
    sub_qa_pairs: list[dict]  # [{"question": ..., "answer": ...}]
    round_num: int


# =============================================================================
# AGENTS
# =============================================================================

def build_agents(model_name: str, temperature: float) -> dict[str, Agent]:
    """Build the agent hierarchy for the RAG pipeline."""

    settings = ModelSettings(temperature=temperature)

    # Agent 1: Preliminary Answerer
    preliminary_agent = Agent[RAGContext](
        name="PreliminaryAnswerer",
        instructions=PRELIMINARY_INSTRUCTIONS,
        model=model_name,
        model_settings=settings,
    )

    # Agent 2: Gap Analyzer
    gap_analyzer = Agent[RAGContext](
        name="GapAnalyzer",
        instructions=GAP_ANALYZER_INSTRUCTIONS,
        model=model_name,
        model_settings=settings,
    )

    # Agent 3: Sub-question Answerer
    sub_question_agent = Agent[RAGContext](
        name="SubQuestionResearcher",
        instructions=SUBQUESTION_INSTRUCTIONS,
        model=model_name,
        model_settings=settings,
    )

    # Agent 4: Synthesizer (regeneration in standard pipeline)
    synthesizer = Agent[RAGContext](
        name="Synthesizer",
        instructions=SYNTHESIZER_INSTRUCTIONS,
        model=model_name,
        model_settings=settings,
    )

    # Agent 5: Final Synthesizer (final synthesis in standard pipeline)
    final_synthesizer = Agent[RAGContext](
        name="FinalSynthesizer",
        instructions=FINAL_SYNTHESIS_INSTRUCTIONS,
        model=model_name,
        model_settings=settings,
    )

    # Agent 6: Efficient Preliminary Answerer
    efficient_preliminary = Agent[RAGContext](
        name="EfficientPreliminaryAnswerer",
        instructions=EFFICIENT_PRELIMINARY_INSTRUCTIONS,
        model=model_name,
        model_settings=settings,
    )

    # Agent 7: Efficient Regenerator
    efficient_regenerator = Agent[RAGContext](
        name="EfficientRegenerator",
        instructions=EFFICIENT_REGENERATOR_INSTRUCTIONS,
        model=model_name,
        model_settings=settings,
    )

    # Agent 8: Efficient Synthesizer
    efficient_synthesizer = Agent[RAGContext](
        name="EfficientSynthesizer",
        instructions=EFFICIENT_SYNTHESIS_INSTRUCTIONS,
        model=model_name,
        model_settings=settings,
    )

    return {
        "preliminary": preliminary_agent,
        "gap_analyzer": gap_analyzer,
        "sub_question": sub_question_agent,
        "synthesizer": synthesizer,
        "final_synthesizer": final_synthesizer,
        "efficient_preliminary": efficient_preliminary,
        "efficient_regenerator": efficient_regenerator,
        "efficient_synthesizer": efficient_synthesizer,
    }


# =============================================================================
# PIPELINE: orchestrate agents
# =============================================================================

async def run_rag_pipeline(
    agents: dict[str, Agent],
    provider: OpenAIProvider,
    retriever: Any,
    question: str,
    num_rounds: int = 2,
    max_sub_questions: int = 3,
) -> dict:
    """Run the full decomposed RAG pipeline using agents."""

    run_config = RunConfig(model_provider=provider, tracing_disabled=True)
    context = RAGContext(
        retriever=retriever,
        question=question,
        retrieved_chunks=[],
        sub_qa_pairs=[],
        round_num=0,
    )

    total_input_tokens = 0
    total_output_tokens = 0
    total_requests = 0

    def _track_usage(run_result) -> None:
        nonlocal total_input_tokens, total_output_tokens, total_requests
        usage = run_result.context_wrapper.usage
        total_input_tokens += usage.input_tokens
        total_output_tokens += usage.output_tokens
        total_requests += usage.requests

    # Step 1: Retrieve documents in code, then pass to agent (1 LLM call)
    _log_line(f"Step 1: Preliminary answer for: {question[:80]}", kind="info")
    chunks = await retriever.retrieve(question)
    context.retrieved_chunks = chunks
    formatted_context = "\n\n".join(f"[{i+1}] {c.text}" for i, c in enumerate(chunks))
    result = await Runner.run(
        agents["preliminary"],
        input=f"Question: {question}\n\nContext Documents:\n{formatted_context}",
        context=context,
        run_config=run_config,
    )
    _track_usage(result)
    preliminary_answer = result.final_output
    current_answer = preliminary_answer

    initial_chunks = [
        {"id": c.chunk_id, "src": c.metadata.get("_data_source"), "content": c.text}
        for c in context.retrieved_chunks
    ]

    rounds_data = []

    for rnd in range(1, num_rounds + 1):
        context.round_num = rnd
        _log_line(f"Step 2.{rnd}: Gap analysis (round {rnd})", kind="info")

        # Step 2: Identify gaps
        gap_result = await Runner.run(
            agents["gap_analyzer"],
            input=f"Original question: {question}\n\nPreliminary answer: {current_answer}\n\nIdentify up to {max_sub_questions} sub-questions to fill information gaps.",
            context=context,
            run_config=run_config,
        )
        _track_usage(gap_result)
        # Parse sub-questions from the gap analyzer output
        import re
        match = re.search(r'\[.*\]', gap_result.final_output, re.DOTALL)
        sub_questions = []
        if match:
            try:
                parsed = json.loads(match.group())
                if isinstance(parsed, list):
                    sub_questions = [s.strip() for s in parsed if isinstance(s, str) and s.strip()][:max_sub_questions]
            except (json.JSONDecodeError, ValueError):
                pass

        if not sub_questions:
            _log_line(f"  No gaps found, skipping round {rnd}", kind="info")
            break

        _log_line(f"  Found {len(sub_questions)} sub-questions", kind="info")

        # Step 3: Research sub-questions (parallel, 1 LLM call each)
        async def _research_subq(sq: str) -> None:
            _log_line(f"  Researching: {sq[:80]}", kind="info")
            sq_chunks = await retriever.retrieve(sq, k_divisor=max(1, len(sub_questions)))
            sq_context = "\n\n".join(f"[{i+1}] {c.text}" for i, c in enumerate(sq_chunks))
            sq_result = await Runner.run(
                agents["sub_question"],
                input=f"Sub-question: {sq}\n\nContext Documents:\n{sq_context}",
                context=context,
                run_config=run_config,
            )
            _track_usage(sq_result)
            context.sub_qa_pairs.append({"question": sq, "answer": sq_result.final_output})

        await asyncio.gather(*(_research_subq(sq) for sq in sub_questions))

        # Step 4: Regenerate answer with new information
        if rnd < num_rounds:
            sub_qa_text = "\n\n".join(
                f"Q: {pair['question']}\nA: {pair['answer']}"
                for pair in context.sub_qa_pairs
            )
            regen_result = await Runner.run(
                agents["synthesizer"],
                input=f"Original Question: {question}\n\nPrevious Preliminary Answer:\n{current_answer}\n\nAdditional Information from Sub-questions:\n{sub_qa_text}",
                context=context,
                run_config=run_config,
            )
            _track_usage(regen_result)
            current_answer = regen_result.final_output

        rounds_data.append({
            "round": rnd,
            "sub_questions": [{"q": sq} for sq in sub_questions],
            "sub_qa_count": len(context.sub_qa_pairs),
        })

    # Step 5: Final synthesis
    _log_line("Step 3: Final synthesis", kind="info")
    sub_qa_text = "\n\n".join(
        f"Q{i+1}: {pair['question']}\nA{i+1}: {pair['answer']}"
        for i, pair in enumerate(context.sub_qa_pairs)
    ) or "None"

    final_result = await Runner.run(
        agents["final_synthesizer"],
        input=f"Original Question: {question}\n\nPreliminary Answer (from initial retrieval):\n{preliminary_answer}\n\nSub-questions and their answers:\n{sub_qa_text}",
        context=context,
        run_config=run_config,
    )
    _track_usage(final_result)

    return {
        "initial_chunks": initial_chunks,
        "initial_answer": preliminary_answer,
        "rounds": rounds_data,
        "final_answer": final_result.final_output,
        "usage": {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
            "requests": total_requests,
        },
    }


async def run_rag_pipeline_efficient(
    agents: dict[str, Agent],
    provider: OpenAIProvider,
    retriever: Any,
    question: str,
    num_rounds: int = 2,
    max_sub_questions: int = 3,
) -> dict:
    """Efficient RAG pipeline: each round retrieves k/#subquestions per sub-question,
    combines results, and uses a single LLM call to regenerate the answer."""

    import re
    run_config = RunConfig(model_provider=provider, tracing_disabled=True)
    context = RAGContext(
        retriever=retriever,
        question=question,
        retrieved_chunks=[],
        sub_qa_pairs=[],
        round_num=0,
    )

    total_input_tokens = 0
    total_output_tokens = 0
    total_requests = 0

    def _track_usage(run_result) -> None:
        nonlocal total_input_tokens, total_output_tokens, total_requests
        usage = run_result.context_wrapper.usage
        total_input_tokens += usage.input_tokens
        total_output_tokens += usage.output_tokens
        total_requests += usage.requests

    # Step 1: Initial retrieval + preliminary answer
    _log_line(f"Step 1: Preliminary answer for: {question[:80]}", kind="info")
    chunks = await retriever.retrieve(question)
    context.retrieved_chunks = chunks
    formatted_context = "\n\n".join(f"[{i+1}] {c.text}" for i, c in enumerate(chunks))
    result = await Runner.run(
        agents["efficient_preliminary"],
        input=f"Question: {question}\n\nContext Documents:\n{formatted_context}",
        context=context,
        run_config=run_config,
    )
    _track_usage(result)
    preliminary_answer = result.final_output
    current_answer = preliminary_answer

    initial_chunks = [
        {"id": c.chunk_id, "src": c.metadata.get("_data_source"), "content": c.text}
        for c in context.retrieved_chunks
    ]

    rounds_data = []

    for rnd in range(1, num_rounds + 1):
        context.round_num = rnd
        _log_line(f"Step 2.{rnd}: Gap analysis (round {rnd})", kind="info")

        # Generate sub-questions
        gap_result = await Runner.run(
            agents["gap_analyzer"],
            input=f"Original question: {question}\n\nPreliminary answer: {current_answer}\n\nIdentify up to {max_sub_questions} sub-questions to fill information gaps.",
            context=context,
            run_config=run_config,
        )
        _track_usage(gap_result)

        match = re.search(r'\[.*\]', gap_result.final_output, re.DOTALL)
        sub_questions = []
        if match:
            try:
                parsed = json.loads(match.group())
                if isinstance(parsed, list):
                    sub_questions = [s.strip() for s in parsed if isinstance(s, str) and s.strip()][:max_sub_questions]
            except (json.JSONDecodeError, ValueError):
                pass

        if not sub_questions:
            _log_line(f"  No gaps found, skipping round {rnd}", kind="info")
            break

        num_sub_qs = len(sub_questions)
        _log_line(f"  Found {num_sub_qs} sub-questions, retrieving k/{num_sub_qs} each", kind="info")

        # Retrieve k/N per sub-question in parallel, combine & deduplicate
        async def _retrieve_for_sq(sq: str) -> tuple[str, list[RetrievedChunk]]:
            sq_chunks = await retriever.retrieve(sq, k_divisor=num_sub_qs)
            return sq, sq_chunks

        subq_results = await asyncio.gather(*(_retrieve_for_sq(sq) for sq in sub_questions))

        combined_chunks: list[RetrievedChunk] = []
        seen_ids: set[tuple] = set()
        per_subq_info = []
        for sq, sq_chunks in subq_results:
            per_subq_info.append({"sub_question": sq, "chunks_retrieved": len(sq_chunks)})
            for c in sq_chunks:
                key = (c.chunk_id, c.metadata.get("_data_source"))
                if key not in seen_ids:
                    seen_ids.add(key)
                    combined_chunks.append(c)

        _log_line(f"  Combined {len(combined_chunks)} unique chunks from {sum(len(ch) for _, ch in subq_results)} total", kind="info")

        # Single LLM call to regenerate answer using combined context
        combined_context = "\n\n".join(f"[{i+1}] {c.text}" for i, c in enumerate(combined_chunks))
        regen_result = await Runner.run(
            agents["efficient_regenerator"],
            input=f"Original Question: {question}\n\nPrevious Answer:\n{current_answer}\n\nNew Context Documents (retrieved for identified information gaps):\n{combined_context}",
            context=context,
            run_config=run_config,
        )
        _track_usage(regen_result)
        current_answer = regen_result.final_output

        rounds_data.append({
            "round": rnd,
            "sub_questions": per_subq_info,
            "combined_chunks_count": len(combined_chunks),
            "regenerated_answer": current_answer,
        })

    # Final synthesis
    _log_line("Step 3: Final synthesis", kind="info")
    round_answers = "\n\n".join(
        f"Round {rd['round']} Answer:\n{rd['regenerated_answer']}"
        for rd in rounds_data
    ) or "None"

    final_result = await Runner.run(
        agents["efficient_synthesizer"],
        input=f"Original Question: {question}\n\nPreliminary Answer (from initial retrieval):\n{preliminary_answer}\n\nAnswers from successive retrieval rounds (each round retrieved additional context to fill gaps):\n{round_answers}",
        context=context,
        run_config=run_config,
    )
    _track_usage(final_result)

    return {
        "initial_chunks": initial_chunks,
        "initial_answer": preliminary_answer,
        "rounds": rounds_data,
        "final_answer": final_result.final_output,
        "usage": {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
            "requests": total_requests,
        },
    }


# =============================================================================
# MAIN
# =============================================================================

def load_questions(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    return [{"question_id": q["question_id"], "question_text": q["question_text"],
             "group": path.stem, "ground_truth": q.get("answer")} for q in data]


async def main():
    parser = argparse.ArgumentParser(description="RAG pipeline using OpenAI Agents SDK")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--max-sub-questions", type=int, default=3)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--cosmos-az-login", action="store_true")
    parser.add_argument("--efficient", action="store_true", help="Use efficient pipeline: retrieves k/#subquestions per sub-question, combines, and regenerates")
    parser.add_argument("--questions-path", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()

    # Load config
    load_config(args.config)

    from utils.cosmos_retriever import CombinedRetriever, RETRIEVAL_SOURCES

    # Build Azure OpenAI client for the Agents SDK
    llm_cfg = CONFIG["llm"]
    if llm_cfg["use_rbac_auth"]:
        token_provider = get_bearer_token_provider(
            AzureCliCredential(), llm_cfg["token_scope"]
        )
        azure_client = AsyncAzureOpenAI(
            api_version=llm_cfg["api_version"],
            azure_endpoint=llm_cfg["llm_endpoint"],
            azure_ad_token_provider=token_provider,
        )
    else:
        azure_client = AsyncAzureOpenAI(
            api_version=llm_cfg["api_version"],
            azure_endpoint=llm_cfg["llm_endpoint"],
            api_key=llm_cfg["llm_api_key"],
        )

    provider = OpenAIProvider(openai_client=azure_client, use_responses=False)
    model_name = llm_cfg["llm_model"]
    temperature = llm_cfg["temperature"]

    # Build retriever
    retriever = CombinedRetriever(
        retrieval_sources=RETRIEVAL_SOURCES,
        k_diverse=CONFIG["retrieval"]["k_diverse"],
        k_ranker=CONFIG.get("ranker", {}).get("k_ranker", 0),
        eta=CONFIG["retrieval"]["eta"],
        rescale_power=CONFIG["retrieval"]["rescale_power"],
        cosmos_az_login=args.cosmos_az_login,
    )
    await retriever.initialize()

    # Build agents
    agents = build_agents(model_name, temperature)

    # Load questions
    questions_path = args.questions_path or Path(CONFIG["paths"]["questions_path"])
    questions = load_questions(questions_path)
    if args.max_questions:
        questions = questions[:args.max_questions]

    _log_line(f"Processing {len(questions)} questions with Agents SDK{' (efficient)' if args.efficient else ''}", kind="info")

    # Process questions
    output_root = args.output_root or Path(CONFIG["paths"]["output_root"])
    output_path = output_root / "agents_sdk"
    output_path.mkdir(parents=True, exist_ok=True)

    results = []
    max_workers = args.max_workers or CONFIG["execution"].get("max_workers")
    semaphore = asyncio.Semaphore(int(max_workers))
    _log_line(f"Parallel workers: {max_workers}", kind="info")

    async def process_question(q: dict) -> dict | None:
        async with semaphore:
            try:
                pipeline_fn = run_rag_pipeline_efficient if args.efficient else run_rag_pipeline
                result = await pipeline_fn(
                    agents, provider, retriever,
                    question=q["question_text"],
                    num_rounds=args.rounds,
                    max_sub_questions=args.max_sub_questions,
                )
                result["question_id"] = q["question_id"]
                result["question_text"] = q["question_text"]
                result["ground_truth"] = q.get("ground_truth")

                # Save intermediate
                intermediate_dir = output_path / "intermediate" / q["group"]
                intermediate_dir.mkdir(parents=True, exist_ok=True)
                (intermediate_dir / f"{q['question_id']}.json").write_text(
                    json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                return result
            except Exception as e:
                _log_line(f"Error on {q['question_id']}: {e}", kind="error")
                return None

    tasks = [asyncio.create_task(process_question(q)) for q in questions]
    with tqdm(total=len(questions)) as pbar:
        for task in asyncio.as_completed(tasks):
            result = await task
            if result is not None:
                results.append(result)
            pbar.update(1)

    # Save final answers
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    answers = [{
        "question_id": r["question_id"],
        "question_text": r["question_text"],
        "answer": r["final_answer"],
        "ground_truth": r.get("ground_truth"),
        "llm_model": model_name,
    } for r in results]

    answers_file = output_path / f"answers_{timestamp}.json"
    answers_file.write_text(json.dumps(answers, indent=2, ensure_ascii=False), encoding="utf-8")
    _log_line(f"Done! Answers file: {answers_file}", kind="success")

    # Print token usage summary
    total_input = sum(r.get("usage", {}).get("input_tokens", 0) for r in results)
    total_output = sum(r.get("usage", {}).get("output_tokens", 0) for r in results)
    total_reqs = sum(r.get("usage", {}).get("requests", 0) for r in results)
    total_tok = total_input + total_output
    if total_reqs > 0:
        avg_input = total_input / total_reqs
        avg_output = total_output / total_reqs
        _log_line(
            f"Token usage: {total_tok:,} total ({total_input:,} input + {total_output:,} output) "
            f"across {total_reqs} LLM calls (avg {avg_input:,.0f} input + {avg_output:,.0f} output per call)",
            kind="info",
        )

    await retriever.close()
    await azure_client.close()


if __name__ == "__main__":
    sys.modules.setdefault("agentic_retriever", sys.modules[__name__] if "agentic_retriever" not in sys.modules else sys.modules["agentic_retriever"])
    # Ensure agentic_retriever module is loaded for cosmos_retriever imports
    asyncio.run(main())
