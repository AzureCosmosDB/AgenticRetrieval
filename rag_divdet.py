#!/usr/bin/env python
"""Self-contained Decomposed RAG with fulltext + vector search (concise version)."""

import argparse
import asyncio
import copy
import datetime
import json
import os
import re
import shutil
import sys
import time
import threading
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import yaml

import openai
from azure.cosmos.aio import CosmosClient
from azure.identity.aio import AzureCliCredential as AsyncAzureCliCredential, DefaultAzureCredential
from azure.identity import AzureCliCredential, get_bearer_token_provider
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
from tqdm import tqdm
import numpy as np

from greedy_log_det import greedy_log_det_select

load_dotenv()

# =============================================================================
# TIMING INSTRUMENTATION (enabled via --timing flag)
# =============================================================================

_TIMING: bool = False
_t0: float = 0.0
_print_lock = threading.Lock()


def _ck(label: str, ref: float | None = None) -> float:
    """Print a timing checkpoint; returns current perf_counter value."""
    now = time.perf_counter()
    if _TIMING:
        elapsed = now - (ref if ref is not None else _t0)
        total = now - _t0
        with _print_lock:
            print(f"  [TIMING] {label}: +{elapsed:.3f}s  (total {total:.3f}s)")
    return now


def _format_activity_id_note(activity_ids: list[str]) -> str:
    unique_ids = list(dict.fromkeys(aid for aid in activity_ids if aid))
    if not unique_ids:
        return ""
    if len(unique_ids) == 1:
        return f" [ActivityId={unique_ids[0]}]"
    shown = ", ".join(unique_ids[:3])
    remaining = len(unique_ids) - 3
    suffix = f", +{remaining} more" if remaining > 0 else ""
    return f" [ActivityIds={shown}{suffix}]"


def _multi_activity_reason(response_meta: list[dict[str, str]]) -> str:
    if not response_meta:
        return ""

    activity_ids = [m.get("activity_id", "") for m in response_meta if m.get("activity_id")]
    unique_ids = list(dict.fromkeys(activity_ids))
    if len(unique_ids) <= 1:
        return ""

    partition_ranges = {m.get("partition_range_id", "") for m in response_meta if m.get("partition_range_id")}
    physical_partitions = {m.get("physical_partition_id", "") for m in response_meta if m.get("physical_partition_id")}
    continuation_count = sum(1 for m in response_meta if m.get("has_continuation") == "1")
    retry_hint_count = sum(1 for m in response_meta if m.get("retry_after_ms"))

    reasons: list[str] = []
    if len(partition_ranges) > 1:
        reasons.append(f"fan-out across {len(partition_ranges)} partition key ranges")
    elif len(physical_partitions) > 1:
        reasons.append(f"fan-out across {len(physical_partitions)} physical partitions")

    if continuation_count > 0:
        reasons.append(f"pagination/continuation on {continuation_count} response(s)")

    if retry_hint_count > 0:
        reasons.append(f"retry-after present on {retry_hint_count} response(s)")

    if not reasons:
        reasons.append("multiple backend executions (possible retries or internal query pipeline calls)")

    return f" [Reason: {'; '.join(reasons)}]"


class LRUCache:
    def __init__(self, max_size: int):
        self.max_size = max(1, int(max_size))
        self._data: OrderedDict[str, Any] = OrderedDict()

    def get(self, key: str) -> Any | None:
        value = self._data.get(key)
        if value is None:
            return None
        self._data.move_to_end(key)
        return value

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value
        self._data.move_to_end(key)
        while len(self._data) > self.max_size:
            self._data.popitem(last=False)


# Load config from yaml
_CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(_CONFIG_PATH) as _f:
    CONFIG = yaml.safe_load(_f)

# =============================================================================
# CONFIGURATION & DATA CLASSES
# =============================================================================

@dataclass
class Question:
    question_id: str
    question_text: str
    group: str | None = None
    ground_truth: str | None = None

@dataclass
class RetrievedChunk:
    chunk_id: int | str
    text: str
    similarity: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class SubQuestionResult:
    sub_question: str
    retrieved_chunks: list[dict]
    answer: str

@dataclass
class RoundResult:
    round_num: int
    preliminary_answer_before: str
    sub_question_results: list[SubQuestionResult]
    regenerated_answer: str | None

# =============================================================================
# PROMPTS
# =============================================================================

PRELIMINARY_PROMPT = """You are a helpful assistant that answers questions STRICTLY based on the provided context.

IMPORTANT RULES:
1. ONLY use information explicitly stated in the context below
2. DO NOT make assumptions or infer information not directly stated
3. DO NOT use any external knowledge
4. If the context does not contain enough information to fully answer the question, clearly state what information IS available and what information IS MISSING
5. Be precise and cite specific details from the context
6. Try to cover as many aspects, obtained numeric values and specific details in the answer as possible

Context Documents:
{context}

Question: {question}

Provide your answer in the following format:

ANSWER FROM CONTEXT:
[Your answer based strictly on the provided context]

INFORMATION GAPS:
[List any aspects of the question that cannot be answered from the provided context, or write "None" if the context fully answers the question]

Response:"""

SUBQUESTION_PROMPT = """You are a helpful assistant that answers questions STRICTLY based on the provided context.

IMPORTANT RULES:
1. ONLY use information explicitly stated in the context below
2. DO NOT make assumptions or infer information not directly stated
3. DO NOT use any external knowledge
4. If the context does not contain enough information to fully answer the question, clearly state what information IS available and what information IS MISSING
5. Be precise and cite specific details from the context
6. Provide a COMPREHENSIVE and DETAILED answer - include all relevant information from the context
7. Extract and include specific values, numbers, specifications, and technical details when available
8. Try to cover as many aspects, obtained numeric values and specific details in the answer as possible

Context Documents:
{context}

Question: {question}

Provide a VERBOSE and COMPREHENSIVE answer that includes ALL relevant information from the context.
Include specific details, values, and technical specifications where available.

ANSWER FROM CONTEXT:
[Your detailed answer based strictly on the provided context - be thorough and include all relevant details]

INFORMATION GAPS:
[List any aspects of the question that cannot be answered from the provided context, or write "None" if the context fully answers the question]

Response:"""

REGENERATE_PROMPT = """You are a helpful assistant that synthesizes information to provide a comprehensive answer.

You have:
1. An original question
2. A previous preliminary answer (which may have gaps)
3. Additional information from follow-up sub-questions and their answers

Your task is to generate an UPDATED and MORE COMPLETE answer by incorporating the new information from the sub-questions.

IMPORTANT RULES:
1. ONLY use information from the previous answer and the sub-question answers provided
2. DO NOT make assumptions or add external knowledge
3. Integrate the new information smoothly into a coherent answer
4. If gaps still remain, clearly identify them
5. Try to cover as many aspects, obtained numeric values and specific details in the answer as possible

Original Question: {question}

Previous Preliminary Answer:
{previous_answer}

Additional Information from Sub-questions:
{sub_qa_context}

Provide your updated answer in the following format:

ANSWER FROM CONTEXT:
[Your updated answer incorporating all available information]

INFORMATION GAPS:
[List any aspects of the question that still cannot be answered, or write "None" if the question is now fully answered]

Response:"""

GAP_DECOMPOSE_PROMPT = """You are a helpful assistant that identifies what additional information is needed to fully answer a question.

Given:
1. An original question
2. A preliminary answer based on initial context (which may be incomplete)
3. The information gaps identified in that answer

Generate sub-questions that will help fill these gaps and provide a complete answer.

Guidelines:
- Focus on the INFORMATION GAPS identified in the preliminary answer
- Each sub-question should be SIMPLE and cover only ONE specific aspect
- Each sub-question should target a single missing piece of information
- Do NOT combine multiple aspects into one sub-question
- Sub-questions should be self-contained and answerable independently
- Keep sub-questions short and focused
- Maximum {max_sub_questions} sub-questions

Original Question: {question}

Preliminary Answer:
{preliminary_answer}

Return your response as a JSON array of strings. Return [] if no additional information is needed.
Example: ["What is the maximum temperature?", "What is the minimum temperature?"]

Sub-questions to fill gaps:"""

SYNTHESIS_PROMPT = """You are a helpful assistant that synthesizes information to answer questions comprehensively.

Original Question: {original_question}

Preliminary Answer (from initial retrieval):
{preliminary_answer}

Sub-questions and their answers:
{sub_qa_pairs}

Based on the above information, provide a comprehensive answer to the original question.
Synthesize the information coherently, avoid repetition, and ensure the answer directly addresses the original question.

Prioritize information from the sub-question answers that fill gaps in the preliminary answer.

At the end, add a summary that directly answers the question in a concise way. Try to cover as many aspects, obtained numeric values and specific details in the answer as possible

Format your response as:
[Your comprehensive answer here]

SUMMARY: [Direct answer to the question]

Final Answer:"""

# =============================================================================
# LLM CLIENT
# =============================================================================

class LLMClient:
    def __init__(self):
        llm_cfg = CONFIG["llm"]
        # Embedding config: 'embedding' section overrides 'llm' section for backward compatibility
        embed_cfg = {**llm_cfg, **CONFIG.get("embedding", {})}
        self._use_rbac_auth = bool(llm_cfg["use_rbac_auth"])
        _shared_key = llm_cfg.get("azure_openai_key", "")
        self._llm_api_key = str(llm_cfg.get("llm_api_key") or _shared_key or "").strip()
        self._embed_api_key = str(embed_cfg.get("embed_api_key") or _shared_key or "").strip()
        # Keep for backward compatibility
        self._api_key = _shared_key
        self._token_provider = None
        if self._use_rbac_auth:
            token_scope = llm_cfg.get("token_scope")
            if not token_scope or not str(token_scope).strip():
                token_scope = "https://cognitiveservices.azure.com/.default"
            self._token_provider = get_bearer_token_provider(AzureCliCredential(), token_scope)
        self._llm_client = None
        self._embed_client = None
        self._embed_http_client = None
        self._local_http_client = None
        self._cfg = llm_cfg
        self._embed_cfg = embed_cfg
        # Local fallback config: 'local_llm' section overrides 'llm' section for backward compatibility
        local_cfg = {**llm_cfg, **CONFIG.get("local_llm", {})}
        self._embed_dimensions = int(embed_cfg.get("embed_dimensions") or 0)
        self._use_local_fallback_for_subtasks = bool(local_cfg.get("use_local_fallback_for_subtasks", False))
        self._local_fallback_endpoint = str(local_cfg.get("local_fallback_endpoint", "http://localhost:11434/api/generate") or "").strip()
        self._local_fallback_model = str(local_cfg.get("local_fallback_model", "") or "").strip()
        self._premium_semaphore = asyncio.Semaphore(max(1, int(llm_cfg.get("premium_max_concurrency", 4))))
        self._local_semaphore = asyncio.Semaphore(max(1, int(local_cfg.get("local_max_concurrency", 8))))
        self._response_cache = LRUCache(int(llm_cfg.get("prompt_cache_size", 2048)))
        self._embed_cache = LRUCache(int(embed_cfg.get("embed_cache_size", 4096)))
        self._local_fallback_failure_threshold = 3
        self._local_fallback_cooldown_seconds = 120
        self._local_fallback_failures = 0
        self._local_fallback_disabled_until = 0.0
        self._default_chars_per_token = 4.0
        self._chars_per_token_estimate = self._default_chars_per_token
        self._min_completion_tokens = 256
        self._max_context_tokens_hint: int | None = None
        self._max_output_tokens_hint: int | None = None
        self._introspection_done = False
        self._introspection_lock = asyncio.Lock()

    def _normalize_embedding(self, embedding: list[float]) -> list[float]:
        if self._embed_dimensions <= 0:
            return [float(x) for x in embedding]
        values = [float(x) for x in embedding]
        if len(values) > self._embed_dimensions:
            return values[:self._embed_dimensions]
        if len(values) < self._embed_dimensions:
            return values + [0.0] * (self._embed_dimensions - len(values))
        return values
    
    @property
    def llm_client(self) -> AsyncAzureOpenAI:
        if not self._llm_client:
            client_kwargs = {
                "api_version": self._cfg["api_version"],
                "azure_endpoint": self._cfg["llm_endpoint"],
            }
            if self._use_rbac_auth:
                client_kwargs["azure_ad_token_provider"] = self._token_provider
            else:
                if not self._llm_api_key:
                    raise ValueError("llm.llm_api_key (or llm.azure_openai_key) must be set when llm.use_rbac_auth is false")
                client_kwargs["api_key"] = self._llm_api_key
            self._llm_client = AsyncAzureOpenAI(**client_kwargs)
        return self._llm_client
    
    @property
    def embed_client(self) -> AsyncAzureOpenAI:
        if not self._embed_client:
            client_kwargs = {
                "api_version": self._embed_cfg["api_version"],
                "azure_endpoint": self._embed_cfg["embed_endpoint"],
            }
            if self._use_rbac_auth:
                client_kwargs["azure_ad_token_provider"] = self._token_provider
            else:
                if not self._embed_api_key:
                    raise ValueError("embedding.embed_api_key (or llm.azure_openai_key) must be set when use_rbac_auth is false")
                client_kwargs["api_key"] = self._embed_api_key
            self._embed_client = AsyncAzureOpenAI(**client_kwargs)
        return self._embed_client
    
    def _should_use_local_fallback(self, label: str) -> bool:
        if not self._use_local_fallback_for_subtasks:
            return False
        if self._local_fallback_disabled_until > time.time():
            return False
        return label.startswith("LLM gap-decompose") or label.startswith("LLM sub-Q answer")

    def _truncate_prompt(self, prompt: str, max_chars: int) -> str:
        if len(prompt) <= max_chars:
            return prompt
        head = int(max_chars * 0.6)
        tail = max_chars - head
        return (
            prompt[:head]
            + "\n\n[... context truncated to satisfy model request constraints ...]\n\n"
            + prompt[-tail:]
        )

    @staticmethod
    def _extract_first_int(value: Any) -> int | None:
        if isinstance(value, int):
            return value if value > 0 else None
        if isinstance(value, str):
            match = re.search(r"\d+", value)
            if match:
                number = int(match.group(0))
                return number if number > 0 else None
        return None

    def _update_limit_hints(self, context_tokens: int | None = None, output_tokens: int | None = None) -> None:
        if context_tokens:
            self._max_context_tokens_hint = context_tokens if self._max_context_tokens_hint is None else min(self._max_context_tokens_hint, context_tokens)
        if output_tokens:
            self._max_output_tokens_hint = output_tokens if self._max_output_tokens_hint is None else min(self._max_output_tokens_hint, output_tokens)

    async def _introspect_llm_capabilities(self) -> None:
        if self._introspection_done:
            return
        async with self._introspection_lock:
            if self._introspection_done:
                return
            try:
                model_name = self._cfg["llm_model"]
                model_obj = None
                try:
                    model_obj = await self.llm_client.models.retrieve(model_name)
                except Exception:
                    pass
                if model_obj is None:
                    try:
                        models = await self.llm_client.models.list()
                        for candidate in models.data:
                            if getattr(candidate, "id", None) == model_name:
                                model_obj = candidate
                                break
                    except Exception:
                        pass

                if model_obj is not None:
                    as_dict = model_obj.model_dump() if hasattr(model_obj, "model_dump") else dict(model_obj)
                    context_keys = [
                        "context_length",
                        "max_context_tokens",
                        "input_token_limit",
                        "max_input_tokens",
                        "token_limit",
                    ]
                    output_keys = [
                        "output_token_limit",
                        "max_output_tokens",
                        "max_completion_tokens",
                    ]
                    context_tokens = next((self._extract_first_int(as_dict.get(k)) for k in context_keys if self._extract_first_int(as_dict.get(k))), None)
                    output_tokens = next((self._extract_first_int(as_dict.get(k)) for k in output_keys if self._extract_first_int(as_dict.get(k))), None)
                    self._update_limit_hints(context_tokens=context_tokens, output_tokens=output_tokens)
            finally:
                self._introspection_done = True

    def _update_hints_from_headers(self, headers: dict[str, str]) -> None:
        context_header_keys = [
            "x-model-context-length",
            "x-max-context-tokens",
            "x-max-input-tokens",
            "x-azure-openai-model-context-length",
        ]
        output_header_keys = [
            "x-max-output-tokens",
            "x-max-completion-tokens",
            "x-azure-openai-max-output-tokens",
        ]
        context_tokens = next((self._extract_first_int(headers.get(k)) for k in context_header_keys if self._extract_first_int(headers.get(k))), None)
        output_tokens = next((self._extract_first_int(headers.get(k)) for k in output_header_keys if self._extract_first_int(headers.get(k))), None)
        self._update_limit_hints(context_tokens=context_tokens, output_tokens=output_tokens)

    def _update_hints_from_badrequest(self, error_text: str) -> None:
        txt = error_text.lower()
        context_patterns = [
            r"maximum context length is\s*(\d+)",
            r"max(?:imum)?\s+context\s+length\s*(?:is|:)\s*(\d+)",
            r"max(?:imum)?\s+input\s+tokens?\s*(?:is|:)\s*(\d+)",
        ]
        output_patterns = [
            r"max(?:imum)?\s+output\s+tokens?\s*(?:is|:)\s*(\d+)",
            r"max(?:imum)?\s+completion\s+tokens?\s*(?:is|:)\s*(\d+)",
        ]
        context_tokens = None
        output_tokens = None
        for pattern in context_patterns:
            match = re.search(pattern, txt)
            if match:
                context_tokens = int(match.group(1))
                break
        for pattern in output_patterns:
            match = re.search(pattern, txt)
            if match:
                output_tokens = int(match.group(1))
                break

        requested_match = re.search(r"requested\s*(\d+)\s*tokens?\s*\((\d+)\s*in the messages,\s*(\d+)\s*in the completion", txt)
        if requested_match:
            msg_tokens = int(requested_match.group(2))
            completion_tokens = int(requested_match.group(3))
            if context_tokens is None:
                context_tokens = msg_tokens + completion_tokens - 1
            if output_tokens is None and completion_tokens > 0:
                output_tokens = completion_tokens - 1

        self._update_limit_hints(context_tokens=context_tokens, output_tokens=output_tokens)

    def _effective_max_completion_tokens(self, requested_tokens: int) -> int:
        limit = requested_tokens
        if self._max_output_tokens_hint is not None:
            limit = min(limit, self._max_output_tokens_hint)
        return max(self._min_completion_tokens, int(limit))

    def _effective_prompt_char_limit(self, max_completion_tokens: int) -> int | None:
        if self._max_context_tokens_hint is None:
            return None
        reserve_tokens = max(self._min_completion_tokens, max_completion_tokens) + 256
        available_prompt_tokens = self._max_context_tokens_hint - reserve_tokens
        if available_prompt_tokens <= 0:
            available_prompt_tokens = self._max_context_tokens_hint // 2
        return max(2000, int(available_prompt_tokens * self._chars_per_token_estimate))

    def _safe_fallback_response(self, label: str) -> str:
        if label.startswith("LLM gap-decompose"):
            return "[]"
        if label.startswith("LLM sub-Q answer"):
            return "Insufficient context to answer this sub-question reliably."
        if label.startswith("LLM regenerate"):
            return "Unable to regenerate answer due request constraints."
        if label.startswith("LLM synthesis"):
            return "Unable to synthesize final answer due request constraints."
        return "Unable to generate response due request constraints."

    async def _complete_premium_once(self, prompt: str, label: str, max_completion_tokens: int) -> str:
        await self._introspect_llm_capabilities()
        max_completion_tokens = self._effective_max_completion_tokens(max_completion_tokens)
        prompt_char_limit = self._effective_prompt_char_limit(max_completion_tokens)
        if prompt_char_limit is not None:
            prompt = self._truncate_prompt(prompt, prompt_char_limit)

        t = _ck(f"{label} – start")
        async with self._premium_semaphore:
            raw_response = await self.llm_client.chat.completions.with_raw_response.create(
                messages=[{"role": "user", "content": prompt}],
                model=self._cfg["llm_model"],
                temperature=self._cfg["temperature"],
                max_completion_tokens=max_completion_tokens,
            )
        result = raw_response.parse()
        _ck(f"{label} – done", t)

        headers = {str(k).lower(): str(v) for k, v in raw_response.headers.items()}
        self._update_hints_from_headers(headers)

        usage = getattr(result, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage is not None else None
        if isinstance(prompt_tokens, int) and prompt_tokens > 0:
            observed = len(prompt) / float(prompt_tokens)
            observed = min(8.0, max(2.0, observed))
            self._chars_per_token_estimate = (self._chars_per_token_estimate * 0.8) + (observed * 0.2)

        return result.choices[0].message.content or ""

    async def _complete_premium(self, prompt: str, retries: int, label: str) -> str:
        if retries is None:
            retries = int(self._cfg["max_retries"])
        max_completion_tokens = int(self._cfg["max_completion_tokens"])
        for attempt in range(retries):
            try:
                return await self._complete_premium_once(prompt, label, max_completion_tokens)
            except openai.RateLimitError:
                wait = min(5.0 * (2 ** attempt), 5.0 * (2 ** 8))
                print(f"Rate limited, retry in {wait}s ({attempt + 1}/{retries})")
                await asyncio.sleep(wait)
            except openai.BadRequestError as e:
                error_text = str(e)
                self._update_hints_from_badrequest(error_text)
                err_text = str(e).lower()
                prior_completion_tokens = max_completion_tokens
                max_completion_tokens = self._effective_max_completion_tokens(max_completion_tokens)
                prompt_limit = self._effective_prompt_char_limit(max_completion_tokens)
                changed = False
                if max_completion_tokens < prior_completion_tokens:
                    changed = True
                if prompt_limit is not None and len(prompt) > prompt_limit:
                    prompt = self._truncate_prompt(prompt, prompt_limit)
                    changed = True
                if not changed and ("maximum context" in err_text or "max tokens" in err_text or "token" in err_text):
                    prompt = self._truncate_prompt(prompt, max(2000, int(len(prompt) * 0.7)))
                    max_completion_tokens = max(self._min_completion_tokens, int(max_completion_tokens * 0.75))
                    changed = True
                if changed:
                    print(
                        f"BadRequestError on {label}; retrying with adaptive limits "
                        f"(max_completion_tokens={max_completion_tokens})"
                    )
                    continue
                raise
            except (openai.APIStatusError, openai.APIConnectionError, openai.APITimeoutError) as e:
                wait = min(5.0 * (2 ** attempt), 5.0 * (2 ** 8))
                print(f"LLMAPI error ({type(e).__name__}), retry in {wait}s ({attempt + 1}/{retries})")
                await asyncio.sleep(wait)
        raise Exception("Max retries exceeded")

    async def _complete_local(self, prompt: str, label: str) -> str:
        if not self._local_fallback_endpoint or not self._local_fallback_model:
            raise RuntimeError("Local fallback endpoint/model is not configured")
        if self._local_http_client is None:
            self._local_http_client = httpx.AsyncClient(timeout=120)
        t = _ck(f"{label} (local) – start")
        async with self._local_semaphore:
            response = await self._local_http_client.post(
                self._local_fallback_endpoint,
                json={
                    "model": self._local_fallback_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": self._cfg["temperature"]},
                },
                headers={"Content-Type": "application/json"},
            )
        response.raise_for_status()
        payload = response.json()
        text = payload.get("response")
        if not isinstance(text, str):
            raise RuntimeError(f"Unexpected local fallback payload: {payload}")
        _ck(f"{label} (local) – done", t)
        return text

    async def complete(self, prompt: str, retries: int | None = None, label: str = "LLM complete") -> str:
        if retries is None:
            retries = int(self._cfg["max_retries"])

        use_local = self._should_use_local_fallback(label)
        route_key = "local" if use_local else "premium"
        cache_key = f"{route_key}|{label}|{prompt}"
        cached = self._response_cache.get(cache_key)
        if isinstance(cached, str):
            return cached

        if use_local:
            try:
                local_response = await self._complete_local(prompt, label)
                self._local_fallback_failures = 0
                self._response_cache.set(cache_key, local_response)
                return local_response
            except Exception as e:
                self._local_fallback_failures += 1
                if self._local_fallback_failures >= self._local_fallback_failure_threshold:
                    self._local_fallback_disabled_until = time.time() + self._local_fallback_cooldown_seconds
                    print(
                        f"Local fallback temporarily disabled for {self._local_fallback_cooldown_seconds}s "
                        f"after {self._local_fallback_failures} failures"
                    )
                print(f"Local fallback error ({type(e).__name__}); falling back to premium model")

        try:
            premium_response = await self._complete_premium(prompt, retries, label)
        except openai.BadRequestError as e:
            print(f"BadRequestError on {label}; using safe fallback response")
            premium_response = self._safe_fallback_response(label)
        except Exception:
            if label.startswith("LLM gap-decompose") or label.startswith("LLM sub-Q answer"):
                print(f"{label} failed after retries; using safe fallback response")
                premium_response = self._safe_fallback_response(label)
            else:
                raise
        self._response_cache.set(cache_key, premium_response)
        return premium_response
    
    async def embed(self, text: str) -> list[float]:
        cached = self._embed_cache.get(text)
        if isinstance(cached, list):
            return list(cached)
        embed_endpoint = str(self._embed_cfg.get("embed_endpoint", "")).strip()
        if embed_endpoint.endswith("/api/embeddings"):
            try:
                if self._embed_http_client is None:
                    self._embed_http_client = httpx.AsyncClient(timeout=60)
                response = await self._embed_http_client.post(
                    embed_endpoint,
                    json={"model": self._embed_cfg["embed_model"], "prompt": text},
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
                body = response.text
                parsed = json.loads(body)
                embedding = parsed.get("embedding")
                if not isinstance(embedding, list):
                    raise ValueError("Invalid Ollama embedding response: missing 'embedding' list")
                normalized = self._normalize_embedding(embedding)
                self._embed_cache.set(text, normalized)
                return list(normalized)
            except httpx.HTTPStatusError as e:
                detail = e.response.text if e.response is not None else ""
                status = e.response.status_code if e.response is not None else "unknown"
                raise RuntimeError(f"Embedding endpoint HTTP {status}: {detail[:300]}") from e
            except httpx.RequestError as e:
                raise RuntimeError(
                    f"Cannot connect to embedding service at {embed_endpoint}. "
                    f"Please ensure Ollama (or your configured embedding service) is running and accessible. "
                    f"Error: {e}"
                ) from e
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSON response from embedding endpoint {embed_endpoint}: {e}") from e

        t = _ck("embed – start")
        result = await self.embed_client.embeddings.create(input=[text], model=self._embed_cfg["embed_model"])
        _ck("embed – done", t)
        normalized = self._normalize_embedding(result.data[0].embedding)
        self._embed_cache.set(text, normalized)
        return list(normalized)

    async def close(self):
        if self._llm_client is not None:
            await self._llm_client.close()
            self._llm_client = None
        if self._embed_client is not None:
            await self._embed_client.close()
            self._embed_client = None
        if self._embed_http_client is not None:
            await self._embed_http_client.aclose()
            self._embed_http_client = None
        if self._local_http_client is not None:
            await self._local_http_client.aclose()
            self._local_http_client = None

# =============================================================================
# COSMOS DB RETRIEVER
# =============================================================================

COSMOS_ENDPOINT = os.getenv("COSMOS_ENDPOINT", CONFIG["cosmos"]["uri"])
COSMOS_KEY = os.getenv("COSMOS_KEY", CONFIG["cosmos"]["key"])
DATABASE_NAME = os.getenv("COSMOS_DATABASE_NAME", CONFIG["cosmos"]["database_name"])


def _as_list_of_strings(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _get_source_config(config: dict[str, Any]) -> list[dict[str, Any]]:
    cosmos_cfg = config.get("cosmos", {})
    configured_sources = cosmos_cfg.get("sources")
    if not isinstance(configured_sources, list):
        raise ValueError(
            "Invalid config: cosmos.sources must be a list with at least one source entry "
            "(container_name, retrieval settings, etc.)."
        )
    if not configured_sources:
        raise ValueError(
            "Invalid config: cosmos.sources is empty. Add at least one source entry under cosmos.sources."
        )

    normalized_sources: list[dict[str, Any]] = []
    for idx, source in enumerate(configured_sources, start=1):
        source = source or {}
        retrieval_cfg = source.get("retrieval") or {}
        source_id = str(source.get("id") or f"source_{idx}").strip()
        normalized_sources.append(
            {
                "id": source_id,
                "container_name": source.get("container_name"),
                "partition_key_path": source.get("partition_key_path"),
                "embedding_field": str(source.get("embedding_field") or "e").strip(),
                "vector_k": int(retrieval_cfg.get("vector_k", 0) or 0),
                "fulltext_k": int(retrieval_cfg.get("fulltext_k", 0) or 0),
                "fulltext_fields": _as_list_of_strings(retrieval_cfg.get("fulltext_fields")),
            }
        )
    return normalized_sources


RETRIEVAL_SOURCES = _get_source_config(CONFIG)

# Comprehensive BM25 stopwords list
STOPWORDS = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "a's", "able", "about", "above", "according", "accordingly", "across", "actually", "after", "afterwards", "again", "against", "ain't", "all", "allow", "allows", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "an", "and", "another", "any", "anybody", "anyhow", "anyone", "anything", "anyway", "anyways", "anywhere", "apart", "appear", "appreciate", "appropriate", "are", "aren't", "around", "as", "aside", "ask", "asking", "associated", "at", "available", "away", "awfully", "b", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "behind", "being", "believe", "below", "beside", "besides", "best", "better", "between", "beyond", "both", "brief", "but", "by", "c", "c'mon", "c's", "came", "can", "can't", "cannot", "cant", "cause", "causes", "certain", "certainly", "changes", "clearly", "co", "com", "come", "comes", "concerning", "consequently", "consider", "considering", "contain", "containing", "contains", "corresponding", "could", "couldn't", "course", "currently", "d", "definitely", "described", "despite", "did", "didn't", "different", "do", "does", "doesn't", "doing", "don", "don't", "done", "down", "downwards", "during", "e", "each", "edu", "eg", "eight", "either", "else", "elsewhere", "enough", "entirely", "especially", "et", "etc", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "f", "far", "few", "fifth", "first", "five", "followed", "following", "follows", "for", "former", "formerly", "forth", "four", "from", "further", "furthermore", "g", "get", "gets", "getting", "given", "gives", "go", "goes", "going", "gone", "got", "gotten", "greetings", "h", "had", "hadn't", "happens", "hardly", "has", "hasn't", "have", "haven't", "having", "he", "he's", "hello", "help", "hence", "her", "here", "here's", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "hi", "him", "himself", "his", "hither", "hopefully", "how", "howbeit", "however", "i", "i'd", "i'll", "i'm", "i've", "ie", "if", "ignored", "immediate", "in", "inasmuch", "inc", "indeed", "indicate", "indicated", "indicates", "inner", "insofar", "instead", "into", "inward", "is", "isn't", "it", "it'd", "it'll", "it's", "its", "itself", "j", "just", "k", "keep", "keeps", "kept", "know", "known", "knows", "l", "last", "lately", "later", "latter", "latterly", "least", "less", "lest", "let", "let's", "like", "liked", "likely", "little", "ll", "look", "looking", "looks", "ltd", "m", "mainly", "make", "many", "may", "maybe", "me", "mean", "meanwhile", "merely", "might", "more", "moreover", "most", "mostly", "mr", "mrs", "ms", "much", "must", "my", "myself", "n", "name", "namely", "nd", "near", "nearly", "necessary", "need", "needs", "neither", "never", "nevertheless", "new", "next", "nine", "no", "nobody", "non", "none", "noone", "nor", "normally", "not", "nothing", "novel", "now", "nowhere", "o", "obviously", "of", "off", "often", "oh", "ok", "okay", "old", "on", "once", "one", "ones", "only", "onto", "or", "other", "others", "otherwise", "ought", "our", "ours", "ourselves", "out", "outside", "over", "overall", "own", "p", "particular", "particularly", "per", "perhaps", "placed", "please", "plus", "possible", "presumably", "probably", "provides", "q", "que", "quite", "qv", "r", "rather", "rd", "re", "really", "reasonably", "regarding", "regardless", "regards", "relatively", "respectively", "right", "s", "said", "same", "saw", "say", "saying", "says", "second", "secondly", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sensible", "sent", "serious", "seriously", "seven", "several", "shall", "she", "should", "shouldn't", "since", "six", "so", "some", "somebody", "somehow", "someone", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "specified", "specify", "specifying", "still", "sub", "such", "sup", "sure", "t", "t's", "take", "taken", "tell", "tends", "th", "than", "thank", "thanks", "thanx", "that", "that's", "thats", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "there's", "thereafter", "thereby", "therefore", "therein", "theres", "thereupon", "these", "they", "they'd", "they'll", "they're", "they've", "think", "third", "this", "thorough", "thoroughly", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "took", "toward", "towards", "tried", "tries", "truly", "try", "trying", "twice", "two", "u", "un", "under", "unfortunately", "unless", "unlikely", "until", "unto", "up", "upon", "us", "use", "used", "useful", "uses", "using", "usually", "v", "value", "various", "ve", "very", "via", "viz", "vs", "w", "want", "wants", "was", "wasn't", "way", "we", "we'd", "we'll", "we're", "we've", "welcome", "well", "went", "were", "weren't", "what", "what's", "whatever", "when", "whence", "whenever", "where", "where's", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "who's", "whoever", "whole", "whom", "whose", "why", "will", "willing", "wish", "with", "within", "without", "won't", "wonder", "would", "wouldn't", "x", "y", "yes", "yet", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves", "z", "zero"}


class CombinedRetriever:
    def __init__(
        self,
        retrieval_sources: list[dict[str, Any]],
        fulltext_k_override: int | None = None,
        k_diverse: int = 0,
        eta: float = 0.0,
        rescale_power: float = 0.0,
        cosmos_az_login: bool = False,
    ):
        self.k_diverse = k_diverse
        self.eta = eta
        self.rescale_power = rescale_power
        self._cosmos_az_login = cosmos_az_login
        self._cosmos = None
        self._db = None
        self._containers: dict[str, Any] = {}
        self._llm = None
        self._expected_vector_dim = int((CONFIG.get("embedding") or CONFIG.get("llm", {})).get("embed_dimensions") or 0)
        self._credential = None
        self._retrieve_cache = LRUCache(int(CONFIG.get("retrieval", {}).get("cache_size", 2000)))
        self._sources = self._normalize_sources(retrieval_sources, fulltext_k_override)

    @property
    def total_fulltext_k(self) -> int:
        return sum(int(source.get("fulltext_k", 0) or 0) for source in self._sources)

    @property
    def total_vector_k(self) -> int:
        return sum(int(source.get("vector_k", 0) or 0) for source in self._sources)

    @property
    def source_count(self) -> int:
        return len(self._sources)

    @staticmethod
    def _is_safe_field_path(path: str) -> bool:
        return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*", path))

    def _normalize_sources(
        self,
        retrieval_sources: list[dict[str, Any]],
        fulltext_k_override: int | None,
    ) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for idx, source in enumerate(retrieval_sources, start=1):
            source = source or {}
            source_id = str(source.get("id") or f"source_{idx}").strip()
            container_name = str(source.get("container_name") or "").strip()
            if not container_name:
                continue
            vector_k = int(source.get("vector_k", 0) or 0)
            fulltext_k = int(source.get("fulltext_k", 0) or 0)
            if fulltext_k_override is not None:
                fulltext_k = int(fulltext_k_override)
            embedding_field = str(source.get("embedding_field") or "e").strip()
            if not self._is_safe_field_path(embedding_field):
                embedding_field = "e"
            fulltext_fields = [
                field for field in _as_list_of_strings(source.get("fulltext_fields")) if self._is_safe_field_path(field)
            ]
            normalized.append(
                {
                    "id": source_id,
                    "container_name": container_name,
                    "partition_key_path": str(source.get("partition_key_path") or "").strip(),
                    "embedding_field": embedding_field,
                    "vector_k": max(0, vector_k),
                    "fulltext_k": max(0, fulltext_k),
                    "fulltext_fields": fulltext_fields,
                }
            )
        return normalized

    async def initialize(self):
        use_rbac_auth = CONFIG.get("cosmos", {}).get("use_rbac_auth", False)
        if self._cosmos_az_login:
            credential = AsyncAzureCliCredential()
            self._credential = credential
            print("✓ Using 'az login' (AzureCliCredential) authentication for Cosmos DB")
            self._cosmos = CosmosClient(COSMOS_ENDPOINT, credential=credential)
        elif use_rbac_auth:
            credential = DefaultAzureCredential()
            self._credential = credential
            print("✓ Using Entra ID RBAC authentication for Cosmos DB")
            self._cosmos = CosmosClient(COSMOS_ENDPOINT, credential=credential)
        else:
            if not COSMOS_KEY:
                raise ValueError("Cosmos DB key not configured. Set cosmos.key in config.yaml.")
            print("✓ Using key-based authentication for Cosmos DB")
            self._cosmos = CosmosClient(COSMOS_ENDPOINT, credential=COSMOS_KEY, connection_mode="Direct")

        self._db = self._cosmos.get_database_client(DATABASE_NAME)
        for source in self._sources:
            self._containers[source["id"]] = self._db.get_container_client(source["container_name"])
        self._llm = LLMClient()

    async def _fulltext_search(self, container, fields: list[str], query: str, top_k: int) -> list[dict]:
        if top_k <= 0 or not fields:
            return []
        terms = [t for t in re.findall(r"\w+", query) if t.lower() not in STOPWORDS and len(t) > 1]
        if not terms:
            return []

        chunks = [terms[i:i + 5] for i in range(0, len(terms), 5)]
        score_exprs: list[str] = []
        for field in fields:
            field_expr = f"c.{field}"
            for term_chunk in chunks:
                args = ", ".join(f'"{term}"' for term in term_chunk)
                score_exprs.append(f"FullTextScore({field_expr}, {args})")
        if not score_exprs:
            return []

        if len(score_exprs) == 1:
            order = f"ORDER BY RANK {score_exprs[0]}"
        else:
            order = f"ORDER BY RANK RRF({', '.join(score_exprs)})"

        try:
            sql = f"SELECT TOP {top_k} * FROM c {order}"
            if _TIMING:
                with _print_lock:
                    print(f"  [QUERY] fulltext SQL ({container.id}): {sql}")

            t = _ck(f"fulltext query (top {top_k}, {container.id}) – start")
            query_iterator = container.query_items(query=sql, parameters=[])
            items = []
            async for item in query_iterator:
                items.append(item)

            _ck(f"fulltext query – done ({len(items)} results, {container.id})", t)
            return items
        except Exception as e:
            print(f"Fulltext error ({container.id}): {e}")
            return []

    async def _vector_search(
        self,
        container,
        embedding_field: str,
        query_emb: list[float],
        top_k: int,
        query_text: str = "",
    ) -> list[dict]:
        if top_k <= 0:
            return []
        vector_field = str(embedding_field or "e").strip()
        if not self._is_safe_field_path(vector_field):
            raise ValueError(f"Unsafe embedding field path: {vector_field!r}")
        adjusted_emb = [float(x) for x in query_emb]
        if self._expected_vector_dim > 0:
            if len(adjusted_emb) > self._expected_vector_dim:
                adjusted_emb = adjusted_emb[:self._expected_vector_dim]
            elif len(adjusted_emb) < self._expected_vector_dim:
                adjusted_emb = adjusted_emb + [0.0] * (self._expected_vector_dim - len(adjusted_emb))
        sql = (
            f"SELECT TOP @k c, VectorDistance(c.{vector_field}, @emb) AS score "
            f"FROM c ORDER BY VectorDistance(c.{vector_field}, @emb)"
        )
        if _TIMING:
            with _print_lock:
                text_preview = f", text={query_text!r}" if query_text else ""
                print(
                    f"  [QUERY] vector SQL ({container.id}): {sql}  "
                    f"[@k={top_k}, @emb=<{len(adjusted_emb)}-dim vector>, @field={vector_field!r}{text_preview}]"
                )

        t = _ck(f"vector query (top {top_k}, {container.id}) – start")

        activity_ids: list[str] = []
        response_meta: list[dict[str, str]] = []

        def _capture_activity_id(headers: dict[str, str], _: dict[str, Any]) -> None:
            normalized_headers = {str(k).lower(): str(v) for k, v in headers.items()}
            activity_id = normalized_headers.get("x-ms-activity-id")
            if activity_id:
                activity_ids.append(activity_id)
            response_meta.append({
                "activity_id": activity_id or "",
                "partition_range_id": normalized_headers.get("x-ms-documentdb-partitionkeyrangeid", ""),
                "physical_partition_id": normalized_headers.get("x-ms-cosmos-physical-partition-id", ""),
                "has_continuation": "1" if normalized_headers.get("x-ms-continuation") else "",
                "retry_after_ms": normalized_headers.get("x-ms-retry-after-ms", ""),
            })

        query_iterator = container.query_items(
            query=sql,
            parameters=[{"name": "@k", "value": top_k}, {"name": "@emb", "value": adjusted_emb}],
            response_hook=_capture_activity_id,
        )
        results = []
        async for item in query_iterator:
            results.append(item)

        activity_id_note = _format_activity_id_note(activity_ids)
        reason_note = _multi_activity_reason(response_meta)
        _ck(f"vector query – done ({len(results)} results, {container.id}){activity_id_note}{reason_note}", t)
        docs = []
        t_reads = time.perf_counter()
        for r in results:
            doc = r.get("c") if isinstance(r.get("c"), dict) else None
            if doc is None:
                doc = {k: v for k, v in r.items() if k != "score"}
            if not isinstance(doc, dict):
                continue
            score = r.get("score")
            doc["_score"] = score if score is not None else 0
            docs.append(doc)
        _ck(f"vector materialize x{len(docs)} ({container.id}) – done", t_reads)
        return docs
    
    def _format_doc(self, doc: dict, source: str, embedding_field: str = "e") -> RetrievedChunk:
        emb_field = str(embedding_field or "e").strip()
        embedding = doc.get(emb_field) if isinstance(doc.get(emb_field), list) else doc.get('embedding')
        exclude = {'_rid', '_self', '_etag', '_attachments', '_ts', 'embedding', '_score', emb_field}
        parts = []
        for k, v in doc.items():
            if k not in exclude and v:
                parts.append(f"{k.replace('_', ' ').title()}: {v if not isinstance(v, (list, dict)) else str(v)}")
        return RetrievedChunk(
            chunk_id=doc.get('id', ''),
            text="\n".join(parts),
            similarity=(1 - doc.get('_score', 0)) if '_score' in doc else None,
            metadata={'_data_source': source, 'embedding': embedding}
        )
    
    async def retrieve(self, query: str) -> list[RetrievedChunk]:
        if self._llm is None:
            raise RuntimeError("Retriever is not initialized")
        t_retrieve = _ck(f"retrieve – start (q: {query[:60]!r})")

        cached = self._retrieve_cache.get(query)
        if isinstance(cached, list):
            _ck("retrieve – cache hit", t_retrieve)
            return copy.deepcopy(cached)

        chunks: list[RetrievedChunk] = []
        seen: set[tuple[str, Any]] = set()

        fulltext_tasks: list[tuple[dict[str, Any], asyncio.Task]] = []
        for source in self._sources:
            container = self._containers.get(source["id"])
            if container is None:
                continue
            top_k = int(source.get("fulltext_k", 0) or 0)
            fields = source.get("fulltext_fields") or []
            if top_k <= 0 or not fields:
                continue
            t_fulltext = _ck(f"  retrieve: fulltext/{source['id']} – start (parallel)")
            task = asyncio.create_task(self._fulltext_search(container, fields, query, top_k))
            fulltext_tasks.append(({"source": source, "timer": t_fulltext}, task))

        emb: list[float] | None = None
        vector_sources = [source for source in self._sources if int(source.get("vector_k", 0) or 0) > 0]
        if vector_sources:
            t_emb = _ck("  retrieve: embed query – start")
            emb = await self._llm.embed(query)
            _ck("  retrieve: embed query – done", t_emb)

        vector_tasks: list[tuple[dict[str, Any], asyncio.Task]] = []
        if emb is not None:
            for source in vector_sources:
                container = self._containers.get(source["id"])
                if container is None:
                    continue
                t_vector = _ck(f"  retrieve: vector/{source['id']} – start (parallel)")
                task = asyncio.create_task(
                    self._vector_search(
                        container,
                        str(source.get("embedding_field") or "e"),
                        emb,
                        int(source.get("vector_k", 0) or 0),
                        query,
                    )
                )
                vector_tasks.append(({"source": source, "timer": t_vector}, task))

        for info, task in fulltext_tasks:
            source = info["source"]
            docs = await task
            _ck(f"  retrieve: fulltext/{source['id']} – done ({len(docs)} results)", info["timer"])
            for doc in docs:
                dedupe_key = (source["id"], doc.get("id"))
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                chunks.append(self._format_doc(doc, f"{source['id']}_fulltext", str(source.get("embedding_field") or "e")))

        for info, task in vector_tasks:
            source = info["source"]
            docs = await task
            _ck(f"  retrieve: vector/{source['id']} – done ({len(docs)} results)", info["timer"])
            for doc in docs:
                dedupe_key = (source["id"], doc.get("id"))
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                chunks.append(self._format_doc(doc, f"{source['id']}_vector", str(source.get("embedding_field") or "e")))
        
        # Diversity selection via greedy log-det maximization
        if self.k_diverse > 0 and len(chunks) > self.k_diverse:
            t = _ck("  retrieve: diversity embed missing – start")
            missing_chunks = [c for c in chunks if c.metadata.get('embedding') is None]
            n_missing = len(missing_chunks)
            if missing_chunks:
                missing_embeddings = await asyncio.gather(*(self._llm.embed(c.text) for c in missing_chunks))
                for chunk, embedding in zip(missing_chunks, missing_embeddings):
                    chunk.metadata['embedding'] = embedding
            _ck(f"  retrieve: diversity embed missing – done ({n_missing} embeds)", t)
            t = _ck("  retrieve: greedy log-det – start")
            vectors = np.array([c.metadata['embedding'] for c in chunks], dtype=np.float32)
            if emb is None:
                emb = await self._llm.embed(query)
            query_vec = np.array(emb, dtype=np.float32)
            selected = greedy_log_det_select(vectors, query_vec, self.k_diverse, self.eta, self.rescale_power)
            if len(selected) < self.k_diverse:
                warnings.warn(
                    f"greedy_log_det_select returned {len(selected)}/{self.k_diverse}: "
                    "vectors are nearly linearly dependent",
                    RuntimeWarning,
                    stacklevel=2,
                )
            chunks = [chunks[i] for i in selected]
            _ck(f"  retrieve: greedy log-det – done (selected {len(chunks)} of {self.k_diverse} requested)", t)

        self._retrieve_cache.set(query, copy.deepcopy(chunks))
        
        _ck(f"retrieve – TOTAL ({len(chunks)} chunks returned)", t_retrieve)
        return chunks

    async def close(self):
        if self._llm is not None:
            await self._llm.close()
            self._llm = None
        if self._cosmos is not None:
            await self._cosmos.close()
            self._cosmos = None
        if self._credential is not None:
            await self._credential.close()
            self._credential = None

# =============================================================================
# DECOMPOSED RAG PIPELINE
# =============================================================================

class DecomposedRAGPipeline:
    def __init__(
        self,
        retriever: CombinedRetriever,
        llm: LLMClient,
        max_sub_q: int = 5,
        num_rounds: int = 2,
        subq_fanout_cap: int | None = None,
        subq_max_concurrency: int = 2,
    ):
        self.retriever = retriever
        self.llm = llm
        self.max_sub_q = max_sub_q
        self.num_rounds = num_rounds
        default_fanout = min(max_sub_q, 3)
        self.subq_fanout_cap = max(1, subq_fanout_cap or default_fanout)
        self.subq_max_concurrency = max(1, subq_max_concurrency)
    
    def _format_context(self, chunks: list[RetrievedChunk]) -> str:
        return "\n\n".join(f"[{i+1}] {c.text}" for i, c in enumerate(chunks))
    
    async def _get_subquestions(self, question: str, answer: str) -> list[str]:
        resp = await self.llm.complete(GAP_DECOMPOSE_PROMPT.format(
            question=question, preliminary_answer=answer, max_sub_questions=self.max_sub_q
        ), label="LLM gap-decompose")
        try:
            match = re.search(r'\[.*\]', resp, re.DOTALL)
            if match:
                subs = json.loads(match.group())
                if isinstance(subs, list):
                    filtered: list[str] = []
                    seen: set[str] = set()
                    max_fanout = min(self.max_sub_q, self.subq_fanout_cap)
                    for s in subs:
                        if not isinstance(s, str):
                            continue
                        normalized = s.strip()
                        if not normalized or normalized in seen:
                            continue
                        seen.add(normalized)
                        filtered.append(normalized)
                        if len(filtered) >= max_fanout:
                            break
                    return filtered
        except:
            pass
        return []

    async def _answer_subquestions_bounded(self, sub_qs: list[str]) -> list[SubQuestionResult]:
        semaphore = asyncio.Semaphore(self.subq_max_concurrency)

        async def _run_one(sub_q: str) -> SubQuestionResult:
            async with semaphore:
                return await self._answer_subquestion(sub_q)

        return await asyncio.gather(*(_run_one(sub_q) for sub_q in sub_qs))
    
    async def _answer_subquestion(self, sub_q: str) -> SubQuestionResult:
        chunks = await self.retriever.retrieve(sub_q)
        context = self._format_context(chunks)
        answer = await self.llm.complete(SUBQUESTION_PROMPT.format(context=context, question=sub_q), label=f"LLM sub-Q answer")
        return SubQuestionResult(
            sub_question=sub_q,
            retrieved_chunks=[{"chunk_id": c.chunk_id, "content": c.text, "metadata": {k: v for k, v in c.metadata.items() if k != 'embedding'}} for c in chunks],
            answer=answer
        )
    
    async def run(self, question: str) -> dict:
        t_run = _ck(f"pipeline.run – start")
        # Initial retrieval
        t = _ck("pipeline: initial retrieve – start")
        initial_chunks = await self.retriever.retrieve(question)
        _ck(f"pipeline: initial retrieve – done ({len(initial_chunks)} chunks)", t)
        initial_context = self._format_context(initial_chunks)
        preliminary = await self.llm.complete(PRELIMINARY_PROMPT.format(context=initial_context, question=question),
                                        label="LLM preliminary")
        _ck("pipeline: preliminary answer – done", t_run)
        
        rounds, all_subs = [], []
        current = preliminary
        
        for rnd in range(1, self.num_rounds + 1):
            t_rnd = _ck(f"pipeline: round {rnd} – start")
            sub_qs = await self._get_subquestions(question, current)
            _ck(f"pipeline: round {rnd} gap-decompose – done ({len(sub_qs)} sub-Qs)", t_rnd)
            if not sub_qs:
                break
            
            # Process sub-questions with bounded concurrency to reduce LLM retry pressure
            t = _ck(f"pipeline: round {rnd} sub-Q bounded ({len(sub_qs)}, cap={self.subq_max_concurrency}) – start")
            sub_results = await self._answer_subquestions_bounded(sub_qs)
            _ck(f"pipeline: round {rnd} sub-Q bounded – done", t)
            all_subs.extend(sub_results)
            
            if rnd < self.num_rounds:
                # Regenerate
                sub_ctx = "\n\n".join(f"Q: {s.sub_question}\nA: {s.answer}" for s in all_subs)
                regen = await self.llm.complete(REGENERATE_PROMPT.format(
                    question=question, previous_answer=current, sub_qa_context=sub_ctx
                ), label=f"LLM regenerate rnd {rnd}")
                rounds.append(RoundResult(rnd, current, sub_results, regen))
                current = regen
                _ck(f"pipeline: round {rnd} regenerate – done", t_rnd)
            else:
                rounds.append(RoundResult(rnd, current, sub_results, None))
            _ck(f"pipeline: round {rnd} – TOTAL", t_rnd)
        
        # Synthesize
        t = _ck("pipeline: synthesis – start")
        sub_pairs = "\n\n".join(f"Q{i+1}: {s.sub_question}\nA{i+1}: {s.answer}" for i, s in enumerate(all_subs))
        final = await self.llm.complete(SYNTHESIS_PROMPT.format(
            original_question=question, preliminary_answer=current, sub_qa_pairs=sub_pairs or "None"
        ), label="LLM synthesis")
        _ck("pipeline: synthesis – done", t)
        
        _ck("pipeline.run – TOTAL", t_run)
        return {
            "initial_chunks": [{"id": c.chunk_id, "src": c.metadata.get('_data_source'), "content": c.text} for c in initial_chunks],
            "initial_answer": preliminary,
            "rounds": [{
                "round": r.round_num,
                "sub_questions": [{"q": s.sub_question, "a": s.answer, "chunks": s.retrieved_chunks} for s in r.sub_question_results],
                "regenerated": r.regenerated_answer
            } for r in rounds],
            "final_answer": final
        }

# =============================================================================
# MAIN
# =============================================================================

def load_questions(path: Path) -> dict[str, list[Question]]:
    questions: dict[str, list[Question]] = {}
    for f in path.glob("*.json"):
        if f.stem.startswith("_") or f.stem.endswith("_test_query"):
            continue
        data = json.loads(f.read_text(encoding="utf-8-sig"))
        questions[f.stem] = [Question(q["question_id"], q["question_text"], f.stem, q.get("answer")) for q in data]
    return questions

async def main_async():
    parser = argparse.ArgumentParser()
    pipeline_cfg = CONFIG.get("pipeline", {})
    parser.add_argument("--config", type=Path, default=None, help="Override config yaml path")
    parser.add_argument(
        "--k-fulltext",
        type=int,
        default=None,
        help="Optional override for fulltext_k across all configured sources",
    )
    parser.add_argument("--k-diverse", type=int, default=CONFIG["retrieval"]["k_diverse"], help="Diverse chunks to select via log-det (0=disabled)")
    parser.add_argument("--eta", type=float, default=CONFIG["retrieval"]["eta"], help="Gram matrix regularization")
    parser.add_argument("--rescale-power", type=float, default=CONFIG["retrieval"]["rescale_power"], help="Query-similarity rescale power")
    parser.add_argument("--max-sub-questions", type=int, default=pipeline_cfg.get("max_sub_questions", 5))
    parser.add_argument("--subq-fanout-cap", type=int, default=pipeline_cfg.get("subq_fanout_cap", 3))
    parser.add_argument("--subq-max-concurrency", type=int, default=pipeline_cfg.get("subq_max_concurrency", 2))
    parser.add_argument("--rounds", type=int, default=pipeline_cfg.get("rounds", 2))
    parser.add_argument("--max-questions", type=int, default=CONFIG["execution"]["max_questions"])
    parser.add_argument("--max-workers", type=int, default=CONFIG["execution"]["max_workers"])
    parser.add_argument("--questions-path", type=Path, default=Path(CONFIG["paths"]["questions_path"]))
    parser.add_argument("--output-root", type=Path, default=Path(CONFIG["paths"]["output_root"]))
    parser.add_argument("--timing", action="store_true", help="Print timing checkpoints for each major operation")
    parser.add_argument("--cosmos-az-login", action="store_true", help="Use 'az login' (AzureCliCredential) to authenticate to Cosmos DB")
    args = parser.parse_args()

    global _TIMING, _t0
    _TIMING = args.timing
    _t0 = time.perf_counter()

    retriever = CombinedRetriever(
        retrieval_sources=RETRIEVAL_SOURCES,
        fulltext_k_override=args.k_fulltext,
        k_diverse=args.k_diverse,
        eta=args.eta,
        rescale_power=args.rescale_power,
        cosmos_az_login=args.cosmos_az_login,
    )
    total_fulltext_k = retriever.total_fulltext_k
    total_vector_k = retriever.total_vector_k
    total_k = total_fulltext_k + total_vector_k
    print(
        f"Decomposed RAG: sources={retriever.source_count}, "
        f"fulltext_total={total_fulltext_k}, vector_total={total_vector_k}, diverse={args.k_diverse}"
    )
    if _TIMING:
        print("[TIMING enabled] All checkpoints printed as +<step_elapsed>s (total <from_start>s)")

    t = _ck("retriever.initialize – start")
    await retriever.initialize()
    _ck("retriever.initialize – done", t)
    llm = LLMClient()
    pipeline = DecomposedRAGPipeline(
        retriever,
        llm,
        args.max_sub_questions,
        args.rounds,
        args.subq_fanout_cap,
        args.subq_max_concurrency,
    )
    
    questions_by_file = load_questions(args.questions_path)
    # Flatten to a list for processing while keeping per-file association via q.group
    all_questions: list[Question] = [q for qs in questions_by_file.values() for q in qs]
    if args.max_questions:
        all_questions = all_questions[:args.max_questions]
    print(f"Processing {len(all_questions)} questions")
    
    div_suffix = f"_div{args.k_diverse}" if args.k_diverse > 0 else ""
    output_path = args.output_root / f"k{total_k}_ft{total_fulltext_k}_vec{total_vector_k}{div_suffix}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = []

    async def process(q: Question):
        result = await pipeline.run(q.question_text)
        result["question_id"] = q.question_id
        result["question_text"] = q.question_text
        result["group"] = q.group
        result["ground_truth"] = q.ground_truth
        group_name = q.group or "default"
        group_dir = output_path / "intermediate" / group_name
        await asyncio.to_thread(group_dir.mkdir, parents=True, exist_ok=True)
        result_file = group_dir / f"{q.question_id}.json"
        await asyncio.to_thread(result_file.write_text, json.dumps(result, indent=2))
        return result

    semaphore = asyncio.Semaphore(max(1, args.max_workers))

    async def bounded_process(q: Question):
        async with semaphore:
            return await process(q)

    tasks = [asyncio.create_task(bounded_process(q)) for q in all_questions]
    with tqdm(total=len(all_questions)) as pbar:
        for task in asyncio.as_completed(tasks):
            try:
                results.append(await task)
            except Exception as e:
                print(f"Error: {e}")
            finally:
                pbar.update(1)
    
    # Save final results - one answer file per input questions file
    llm_model = CONFIG["llm"]["llm_model"]
    embed_model = (CONFIG.get("embedding") or CONFIG.get("llm", {})).get("embed_model", "")
    results_by_file: dict[str, list] = {stem: [] for stem in questions_by_file}
    for r in results:
        source_stem = r.get("group", "default")
        if source_stem not in results_by_file:
            results_by_file[source_stem] = []
        results_by_file[source_stem].append({
            "question_id": r["question_id"],
            "question_text": r["question_text"],
            "answer": r["final_answer"],
            "ground_truth": r.get("ground_truth"),
            "llm_model": llm_model,
            "embed_model": embed_model,
        })
    # Single timestamp shared across all output files so they are identifiable as one run
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    for source_stem, answers in results_by_file.items():
        answers_filename = f"{source_stem}_{timestamp}.json"
        await asyncio.to_thread((output_path / answers_filename).write_text, json.dumps(answers, indent=2))
    print(f"Done! Results: {output_path}")

    await retriever.close()
    await llm.close()

if __name__ == "__main__":
    if "--timing" in sys.argv:
        class _TeeStream:
            def __init__(self, *streams):
                self._streams = streams

            def write(self, data):
                for stream in self._streams:
                    stream.write(data)
                return len(data)

            def flush(self):
                for stream in self._streams:
                    stream.flush()

            def isatty(self):
                return any(getattr(stream, "isatty", lambda: False)() for stream in self._streams)

            def __getattr__(self, name):
                return getattr(self._streams[0], name)

        out_dir = Path(__file__).resolve().parent / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        run_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_log_path = out_dir / f"timing_5q_rerun_{run_stamp}.log"
        latest_log_path = out_dir / "timing_5q_latest.log"

        original_stdout = sys.stdout
        original_stderr = sys.stderr
        with run_log_path.open("w", encoding="utf-8", errors="replace") as log_file:
            try:
                sys.stdout = _TeeStream(original_stdout, log_file)
                sys.stderr = _TeeStream(original_stderr, log_file)
                asyncio.run(main_async())
            finally:
                try:
                    sys.stdout.flush()
                    sys.stderr.flush()
                finally:
                    sys.stdout = original_stdout
                    sys.stderr = original_stderr

        shutil.copyfile(run_log_path, latest_log_path)
        print(f"[TIMING] wrote log: {run_log_path}")
        print(f"[TIMING] updated latest: {latest_log_path}")
    else:
        asyncio.run(main_async())
