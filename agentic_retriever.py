#!/usr/bin/env python
"""Self-contained Decomposed RAG with fulltext + vector search (concise version)."""

from __future__ import annotations

import argparse
import asyncio
import copy
import contextvars
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
from typing import Any, TYPE_CHECKING

import httpx
import yaml

import openai
from azure.cosmos.aio import CosmosClient
from azure.identity import DefaultAzureCredential as SyncDefaultAzureCredential, AzureCliCredential, get_bearer_token_provider
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
_TIMING_MARK = "¤"
_CURRENT_QUESTION_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar("current_question_id", default=None)

_ANSI_RESET = "\x1b[0m"
_ANSI_COLORS = {
    "timing": "\x1b[90m",
    "query": "\x1b[2;33m",
    "success": "\x1b[92m",
    "warn": "\x1b[93m",
    "error": "\x1b[91m",
    "info": "\x1b[94m",
}


def _stdout_supports_color() -> bool:
    if os.getenv("NO_COLOR"):
        return False
    if os.getenv("FORCE_COLOR"):
        return True
    stream = sys.stdout
    streams = getattr(stream, "_streams", None)
    if streams:
        return any(bool(getattr(s, "isatty", lambda: False)()) for s in streams)
    return bool(getattr(stream, "isatty", lambda: False)())


def _colorize(text: str, kind: str) -> str:
    if not _stdout_supports_color():
        return text
    if kind == "timing":
        base = _ANSI_COLORS["timing"]
        white = "\x1b[97m"
        pattern = re.compile(r"(?P<num>[+-]?[0-9]*\.?[0-9]+s)")

        def repl(match: re.Match[str]) -> str:
            return f"{white}{match.group('num')}{base}"

        return f"{base}{pattern.sub(repl, text)}{_ANSI_RESET}"
    if kind == "query":
        base = _ANSI_COLORS["query"]
        highlight = "\x1b[22;93m"
        pattern = re.compile(r"(?P<prefix>text=)(?P<quote>['\"])(?P<value>.*?)(?P=quote)")

        def repl(match: re.Match[str]) -> str:
            prefix = match.group("prefix")
            quote = match.group("quote")
            value = match.group("value")
            return f"{prefix}{quote}{highlight}{value}{base}{quote}"

        return f"{base}{pattern.sub(repl, text)}{_ANSI_RESET}"
    color = _ANSI_COLORS.get(kind)
    if not color:
        return text
    return f"{color}{text}{_ANSI_RESET}"


def _log_line(text: str, kind: str = "info", use_lock: bool = False) -> None:
    if kind == "query":
        text = _query_text_with_question_prefix(text)
    styled = _colorize(text, kind)
    if use_lock:
        with _print_lock:
            print(styled)
    else:
        print(styled)


def _get_current_question_id() -> str | None:
    main_mod = sys.modules.get("__main__")
    runtime_var = getattr(main_mod, "_CURRENT_QUESTION_ID", None)
    if runtime_var is not None and hasattr(runtime_var, "get"):
        try:
            value = runtime_var.get()
        except Exception:
            value = None
        if value:
            return value
    return _CURRENT_QUESTION_ID.get()


def _timing_text_with_question_prefix(text: str) -> str:
    question_id = _get_current_question_id()
    if not question_id:
        return text
    marker = f"{_TIMING_MARK} "
    if marker in text:
        return text.replace(marker, f"{marker}{question_id}: ", 1)
    return f"{question_id}: {text}"


def _query_text_with_question_prefix(text: str) -> str:
    question_id = _get_current_question_id()
    if not question_id:
        return text
    return f"    {question_id}: {text.lstrip()}"


def _ck(label: str, ref: float | None = None) -> float:
    """Print a timing checkpoint; returns current perf_counter value."""
    now = time.perf_counter()
    if _TIMING:
        elapsed = now - (ref if ref is not None else _t0)
        total = now - _t0
        _log_line(
            _timing_text_with_question_prefix(f"  {_TIMING_MARK} {label}: +{elapsed:.3f}s  (total {total:.3f}s)"),
            kind="timing",
            use_lock=True,
        )
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


class InvalidLLMResponseError(Exception):
    pass


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


# =============================================================================
# PROMPTS
# =============================================================================

from prompts import (
    PRELIMINARY_PROMPT,
    EFFICIENT_PRELIMINARY_PROMPT,
    SUBQUESTION_PROMPT,
    REGENERATE_PROMPT,
    GAP_DECOMPOSE_PROMPT,
    SYNTHESIS_PROMPT,
    EFFICIENT_REGENERATE_PROMPT,
    EFFICIENT_SYNTHESIS_PROMPT,
)

# Config loaded at runtime via load_config(); do not import-time read.
CONFIG: dict = {}


def load_config(path: Path) -> None:
    """Load (or reload) the YAML configuration into the module-level CONFIG dict."""
    with open(path) as f:
        CONFIG.clear()
        CONFIG.update(yaml.safe_load(f))

    global PRELIMINARY_PROMPT, EFFICIENT_PRELIMINARY_PROMPT, SUBQUESTION_PROMPT
    global REGENERATE_PROMPT, GAP_DECOMPOSE_PROMPT, SYNTHESIS_PROMPT
    global EFFICIENT_REGENERATE_PROMPT, EFFICIENT_SYNTHESIS_PROMPT

    preliminary_prefix = (str(CONFIG.get("pipeline", {}).get("preliminary_prefix")) or "").strip()
    preliminary_prefix = preliminary_prefix + "\n\n" if preliminary_prefix else ""
    subquery_prefix = (str(CONFIG.get("pipeline", {}).get("subquery_prefix")) or "").strip()
    subquery_prefix = subquery_prefix + "\n\n" if subquery_prefix else ""

    PRELIMINARY_PROMPT = preliminary_prefix + PRELIMINARY_PROMPT
    EFFICIENT_PRELIMINARY_PROMPT = preliminary_prefix + EFFICIENT_PRELIMINARY_PROMPT
    EFFICIENT_REGENERATE_PROMPT = subquery_prefix + EFFICIENT_REGENERATE_PROMPT
    SUBQUESTION_PROMPT = subquery_prefix + SUBQUESTION_PROMPT

    dataset_description = (str(CONFIG.get("pipeline", {}).get("dataset_description")) or "").strip()
    dataset_description = dataset_description + "\n\n" if dataset_description else ""

    SYNTHESIS_PROMPT = dataset_description + SYNTHESIS_PROMPT
    EFFICIENT_REGENERATE_PROMPT = dataset_description + EFFICIENT_REGENERATE_PROMPT
    EFFICIENT_SYNTHESIS_PROMPT = dataset_description + EFFICIENT_SYNTHESIS_PROMPT
    GAP_DECOMPOSE_PROMPT = dataset_description + GAP_DECOMPOSE_PROMPT
    REGENERATE_PROMPT = dataset_description + REGENERATE_PROMPT
    SUBQUESTION_PROMPT = dataset_description + SUBQUESTION_PROMPT
    PRELIMINARY_PROMPT = dataset_description + PRELIMINARY_PROMPT

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
# LLM CLIENT
# =============================================================================

class LLMClient:
    def __init__(self, azure_az_login: bool = False):
        llm_cfg = CONFIG["llm"]
        # Embedding config: 'embedding' section overrides 'llm' section for backward compatibility
        embed_cfg = {**llm_cfg, **CONFIG.get("embedding", {})}
        self._use_rbac_auth = bool(llm_cfg["use_rbac_auth"])
        self._use_embed_rbac_auth = bool(embed_cfg.get("use_rbac_auth", False))
        token_scope = llm_cfg.get("token_scope")
        self._token_scope = str(token_scope).strip()
        _shared_key = llm_cfg.get("azure_openai_key", "")
        self._llm_api_key = str(llm_cfg.get("llm_api_key") or _shared_key or "").strip()
        self._embed_api_key = str(embed_cfg.get("embed_api_key") or _shared_key or "").strip()
        # Keep for backward compatibility
        self._api_key = _shared_key
        self._token_provider = None
        if self._use_rbac_auth or self._use_embed_rbac_auth:
            self._token_provider = get_bearer_token_provider(AzureCliCredential(), self._token_scope)
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
        self._min_completion_tokens = 64
        self._max_context_tokens_hint: int | None = None
        self._max_output_tokens_hint: int | None = None
        self._introspection_done = False
        self._introspection_lock = asyncio.Lock()
        self.total_prompt_chars = 0
        self.total_prompt_tokens = 0
        self.total_llm_calls = 0

    @staticmethod
    def _is_key_auth_disabled_error(error: Exception) -> bool:
        status_code = getattr(error, "status_code", None)
        if status_code != 403:
            return False
        txt = str(error).lower()
        return (
            "authenticationtypedisabled" in txt
            or "key based authentication is disabled" in txt
            or "authentication type is disabled" in txt
        )

    def _switch_to_rbac_auth(self) -> None:
        self._use_rbac_auth = True
        credential = AzureCliCredential() if self._azure_az_login else SyncDefaultAzureCredential()
        self._token_provider = get_bearer_token_provider(credential, self._token_scope)
        self._llm_client = None
        self._embed_client = None

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
            if self._use_embed_rbac_auth:
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

    def _is_premium_configured(self) -> bool:
        endpoint = str(self._cfg.get("llm_endpoint", "") or "").strip()
        model = str(self._cfg.get("llm_model", "") or "").strip()
        return bool(endpoint and model)

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

    def _effective_max_completion_tokens_for_prompt(self, prompt: str, requested_tokens: int) -> int:
        limit = self._effective_max_completion_tokens(requested_tokens)
        if self._max_context_tokens_hint is None:
            return limit

        est_prompt_tokens = max(1, int(len(prompt) / max(1.0, self._chars_per_token_estimate)))
        safety_reserve = 32
        available_for_output = self._max_context_tokens_hint - est_prompt_tokens - safety_reserve

        if available_for_output <= 0:
            raise InvalidLLMResponseError(
                "Prompt uses full model context window; no room left for completion tokens"
            )

        return max(1, min(limit, int(available_for_output)))

    def _estimate_prompt_tokens(self, prompt: str) -> int:
        return max(1, int(len(prompt) / max(1.0, self._chars_per_token_estimate)))

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

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                    continue
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    parts.append(text)
            return "".join(parts)
        return ""

    def _validated_completion_text(self, result: Any, label: str) -> str:
        choices = getattr(result, "choices", None) or []
        if not choices:
            raise InvalidLLMResponseError(f"{label}: no choices in completion result")

        choice = choices[0]
        finish_reason = str(getattr(choice, "finish_reason", "") or "").strip().lower()
        message = getattr(choice, "message", None)
        if message is None:
            raise InvalidLLMResponseError(f"{label}: missing message in completion choice")

        refusal = getattr(message, "refusal", None)
        tool_calls = getattr(message, "tool_calls", None)
        raw_content = getattr(message, "content", None)
        text = self._content_to_text(raw_content)

        issues: list[str] = []
        if finish_reason in {"tool_calls", "function_call", "content_filter"}:
            issues.append(f"finish_reason={finish_reason}")
        if isinstance(refusal, str):
            if refusal.strip():
                issues.append("message.refusal present")
        elif refusal:
            issues.append("message.refusal present")
        if tool_calls:
            issues.append("tool_calls present")
        if not text or not text.strip():
            issues.append("empty content")

        if issues:
            refusal_present = bool(refusal.strip()) if isinstance(refusal, str) else bool(refusal)
            tool_calls_present = bool(tool_calls)
            content_type = type(raw_content).__name__ if raw_content is not None else "None"
            content_len = len(text) if isinstance(text, str) else 0
            diag = (
                f"finish_reason={finish_reason or 'None'}, "
                f"refusal={refusal_present}, "
                f"tool_calls={tool_calls_present}, "
                f"content_type={content_type}, "
                f"content_len={content_len}"
            )
            raise InvalidLLMResponseError(f"{label}: invalid completion ({'; '.join(issues)}) [{diag}]")

        return text.strip()

    async def _complete_premium_once(self, prompt: str, label: str, max_completion_tokens: int) -> str:
        await self._introspect_llm_capabilities()
        estimated_prompt_tokens = self._estimate_prompt_tokens(prompt)
        max_completion_tokens = self._effective_max_completion_tokens_for_prompt(prompt, max_completion_tokens)
        if _TIMING:
            _log_line(
                _timing_text_with_question_prefix(
                    "  ¤ "
                    f"{label} token budget: prompt_est={estimated_prompt_tokens}, "
                    f"completion={max_completion_tokens}, "
                    f"context_hint={self._max_context_tokens_hint}, "
                    f"output_hint={self._max_output_tokens_hint}"
                ),
                kind="timing",
                use_lock=True,
            )

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
            self.total_llm_calls += 1
            self.total_prompt_tokens += prompt_tokens
            observed = len(prompt) / float(prompt_tokens)
            observed = min(8.0, max(2.0, observed))
            self._chars_per_token_estimate = (self._chars_per_token_estimate * 0.8) + (observed * 0.2)

        return self._validated_completion_text(result, label)

    async def _complete_premium(self, prompt: str, retries: int, label: str) -> str:
        if retries is None:
            retries = int(self._cfg["max_retries"])
        max_completion_tokens = int(self._cfg["max_completion_tokens"])
        for attempt in range(retries):
            try:
                return await self._complete_premium_once(prompt, label, max_completion_tokens)
            except InvalidLLMResponseError as e:
                _log_line(f"Invalid LLM response on {label}: {e} ({attempt + 1}/{retries})", kind="warn")
                if attempt + 1 >= retries:
                    raise
                wait = min(2.0 * (2 ** attempt), 30.0)
                _log_line(f"Retrying {label} in {wait}s after invalid response", kind="warn")
                await asyncio.sleep(wait)
            except openai.RateLimitError:
                wait = min(5.0 * (2 ** attempt), 5.0 * (2 ** 8))
                _log_line(f"Rate limited, retry in {wait}s ({attempt + 1}/{retries})", kind="warn")
                await asyncio.sleep(wait)
            except openai.BadRequestError as e:
                error_text = str(e)
                self._update_hints_from_badrequest(error_text)
                err_text = error_text.lower()
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
                    _log_line(
                        f"BadRequestError on {label}; retrying with adaptive limits "
                        f"(max_completion_tokens={max_completion_tokens})",
                        kind="warn",
                    )
                    continue
                raise
            except (openai.APIStatusError, openai.APIConnectionError, openai.APITimeoutError) as e:
                if self._is_key_auth_disabled_error(e) and not self._use_rbac_auth:
                    try:
                        self._switch_to_rbac_auth()
                        _log_line("Azure OpenAI key auth is disabled for this resource; switched to Entra ID RBAC", kind="success")
                        continue
                    except Exception as switch_err:
                        _log_line(f"Failed switching to Entra ID RBAC ({type(switch_err).__name__}); continuing retries", kind="error")
                wait = min(5.0 * (2 ** attempt), 5.0 * (2 ** 8))
                _log_line(f"LLMAPI error ({type(e).__name__}), retry in {wait}s ({attempt + 1}/{retries})", kind="error")
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
        if not text.strip():
            raise RuntimeError("Local fallback returned empty content")
        _ck(f"{label} (local) – done", t)
        return text

    async def complete(self, prompt: str, retries: int | None = None, label: str = "LLM complete") -> str:
        if retries is None:
            retries = int(self._cfg["max_retries"])
        self.total_prompt_chars += len(prompt)

        use_local_fallback = self._should_use_local_fallback(label)
        premium_configured = self._is_premium_configured()
        route_key = "premium" if premium_configured else "local"
        cache_key = f"{route_key}|{label}|{prompt}"
        cached = self._response_cache.get(cache_key)
        if isinstance(cached, str):
            return cached

        premium_error: Exception | None = None

        if premium_configured:
            try:
                premium_response = await self._complete_premium(prompt, retries, label)
                self._response_cache.set(cache_key, premium_response)
                return premium_response
            except Exception as e:
                premium_error = e

        if use_local_fallback and (not premium_configured or premium_error is not None):
            try:
                local_response = await self._complete_local(prompt, label)
                self._local_fallback_failures = 0
                self._response_cache.set(cache_key, local_response)
                return local_response
            except Exception as e:
                self._local_fallback_failures += 1
                if self._local_fallback_failures >= self._local_fallback_failure_threshold:
                    self._local_fallback_disabled_until = time.time() + self._local_fallback_cooldown_seconds
                    _log_line(
                        f"Local fallback temporarily disabled for {self._local_fallback_cooldown_seconds}s "
                        f"after {self._local_fallback_failures} failures",
                        kind="warn",
                    )
                _log_line(f"Local fallback error ({type(e).__name__}); local fallback unavailable", kind="error")

        if premium_error is not None:
            if isinstance(premium_error, InvalidLLMResponseError):
                _log_line(f"Invalid LLM response on {label}; using safe fallback response", kind="warn")
                premium_response = self._safe_fallback_response(label)
            elif isinstance(premium_error, openai.BadRequestError):
                _log_line(f"BadRequestError on {label}; using safe fallback response", kind="warn")
                premium_response = self._safe_fallback_response(label)
            elif label.startswith("LLM gap-decompose") or label.startswith("LLM sub-Q answer"):
                _log_line(f"{label} failed after retries; using safe fallback response", kind="warn")
                premium_response = self._safe_fallback_response(label)
            else:
                raise premium_error
        else:
            premium_response = self._safe_fallback_response(label)

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
# COSMOS DB RETRIEVER (moved to utils/cosmos_retriever.py)
# =============================================================================

if TYPE_CHECKING:
    from utils.cosmos_retriever import CombinedRetriever

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

    async def run_efficient(self, question: str) -> dict:
        """Pipeline variant activated by ``--efficient``.

        Works like :meth:`run` but each round divides the retrieval budget
        across the generated sub-questions: every sub-question retrieves
        ``k / #subquestions`` texts.  The retrieved texts from all sub-questions
        are combined (de-duplicated), and a single LLM call uses that combined
        context to produce an updated answer **to the original question** along
        with remaining information gaps.  Those gaps seed the next round.
        """
        t_run = _ck("pipeline.run_efficient – start")

        # --- Step 1: initial retrieval + preliminary answer (identical to run) ---
        t = _ck("pipeline: initial retrieve – start")
        initial_chunks = await self.retriever.retrieve(question)
        _ck(f"pipeline: initial retrieve – done ({len(initial_chunks)} chunks)", t)
        initial_context = self._format_context(initial_chunks)
        preliminary = await self.llm.complete(
            EFFICIENT_PRELIMINARY_PROMPT.format(context=initial_context, question=question),
            label="LLM preliminary",
        )
        _ck("pipeline: preliminary answer – done", t_run)

        rounds_data: list[dict] = []
        current = preliminary

        for rnd in range(1, self.num_rounds + 1):
            t_rnd = _ck(f"pipeline: efficient round {rnd} – start")

            # --- Generate sub-questions from gaps in the current answer ---
            sub_qs = await self._get_subquestions(question, current)
            _ck(f"pipeline: round {rnd} gap-decompose – done ({len(sub_qs)} sub-Qs)", t_rnd)
            if not sub_qs:
                break

            num_sub_qs = len(sub_qs)

            # --- Retrieve k / #subquestions per sub-question (parallel, bounded) ---
            semaphore = asyncio.Semaphore(self.subq_max_concurrency)

            async def _retrieve_for_subq(sub_q: str) -> tuple[str, list[RetrievedChunk]]:
                async with semaphore:
                    chunks = await self.retriever.retrieve(sub_q, k_divisor=num_sub_qs)
                    return sub_q, chunks

            t = _ck(f"pipeline: round {rnd} efficient retrieve ({num_sub_qs} sub-Qs) – start")
            subq_results = await asyncio.gather(*(_retrieve_for_subq(sq) for sq in sub_qs))
            _ck(f"pipeline: round {rnd} efficient retrieve – done", t)

            # Combine & de-duplicate retrieved chunks across all sub-questions
            combined_chunks: list[RetrievedChunk] = []
            seen_ids: set[tuple[str | int, str | None]] = set()
            per_subq_info: list[dict] = []

            for sub_q, chunks in subq_results:
                subq_chunk_records: list[dict] = []
                for c in chunks:
                    key = (c.chunk_id, c.metadata.get("_data_source"))
                    subq_chunk_records.append(
                        {"chunk_id": c.chunk_id, "content": c.text,
                         "metadata": {k: v for k, v in c.metadata.items() if k != "embedding"}}
                    )
                    if key not in seen_ids:
                        seen_ids.add(key)
                        combined_chunks.append(c)
                per_subq_info.append({"sub_question": sub_q, "chunks_retrieved": len(chunks), "chunks": subq_chunk_records})

            _ck(
                f"pipeline: round {rnd} combined {len(combined_chunks)} unique chunks "
                f"(from {sum(len(ch) for _, ch in subq_results)} total)",
                t_rnd,
            )

            # --- Use combined context to regenerate answer to the ORIGINAL question ---
            combined_context = self._format_context(combined_chunks)
            t = _ck(f"pipeline: round {rnd} efficient regenerate – start")
            regen = await self.llm.complete(
                EFFICIENT_REGENERATE_PROMPT.format(
                    question=question,
                    previous_answer=current,
                    context=combined_context,
                ),
                label=f"LLM efficient regen rnd {rnd}",
            )
            _ck(f"pipeline: round {rnd} efficient regenerate – done", t)

            rounds_data.append({
                "round": rnd,
                "sub_questions": per_subq_info,
                "combined_chunks_count": len(combined_chunks),
                "regenerated_answer": regen,
            })
            current = regen
            _ck(f"pipeline: efficient round {rnd} – TOTAL", t_rnd)

        # --- Final synthesis (same pattern as run's SYNTHESIS_PROMPT) ---
        t = _ck("pipeline: efficient synthesis – start")
        round_answer_parts: list[str] = []
        for rd in rounds_data:
            rnd_num = rd.get("round", "?")
            regen_answer = rd.get("regenerated_answer", "")
            round_answer_parts.append(f"Round {rnd_num} Answer:\n{regen_answer}")
        round_answers_text = "\n\n".join(round_answer_parts) or "None"
        final = await self.llm.complete(
            EFFICIENT_SYNTHESIS_PROMPT.format(
                original_question=question,
                preliminary_answer=preliminary,
                round_answers=round_answers_text,
            ),
            label="LLM efficient synthesis",
        )
        _ck("pipeline: efficient synthesis – done", t)

        _ck("pipeline.run_efficient – TOTAL", t_run)
        return {
            "initial_chunks": [
                {"id": c.chunk_id, "src": c.metadata.get("_data_source"), "content": c.text}
                for c in initial_chunks
            ],
            "initial_answer": preliminary,
            "rounds": rounds_data,
            "final_answer": final,
        }

# =============================================================================
# MAIN
# =============================================================================

def load_questions(path: Path) -> dict[str, list[Question]]:
    questions: dict[str, list[Question]] = {}
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    questions[path.stem] = [Question(q["question_id"], q["question_text"], path.stem, q.get("answer")) for q in data]
    return questions

async def main_async():
    # --- Phase 1: parse --config and load configuration -----------------------
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, required=True, help="Path to config yaml file")
    pre_args, _ = pre_parser.parse_known_args()
    load_config(pre_args.config)

    from utils.cosmos_retriever import CombinedRetriever, RETRIEVAL_SOURCES

    # --- Phase 2: full argument parsing (defaults drawn from loaded config) ---
    parser = argparse.ArgumentParser()
    pipeline_cfg = CONFIG.get("pipeline", {})
    parser.add_argument("--config", type=Path, required=True, help="Path to config yaml file")
    parser.add_argument(
        "--k-fulltext",
        type=int,
        default=None,
        help="Optional override for fulltext_k across all configured sources",
    )
    parser.add_argument("--k-diverse", type=int, default=CONFIG["retrieval"]["k_diverse"], help="Diverse chunks to select via log-det (0=disabled)")
    parser.add_argument("--k-ranker", type=int, default=CONFIG.get("ranker", {}).get("k_ranker", 0), help="Rerank k_diverse chunks down to k_ranker via semantic ranker (0=disabled)")
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
    parser.add_argument("--azure-az-login", action="store_true", help="Use 'az login' (AzureCliCredential) to authenticate to Azure OpenAI LLM")
    parser.add_argument("--separate-subq-calls", action="store_true", help="Use separate LLM calls per sub-question instead of the default efficient pipeline (`--efficient` has been removed)")
    args = parser.parse_args()

    global _TIMING, _t0
    _TIMING = args.timing
    _t0 = time.perf_counter()

    retriever = CombinedRetriever(
        retrieval_sources=RETRIEVAL_SOURCES,
        fulltext_k_override=args.k_fulltext,
        k_diverse=args.k_diverse,
        k_ranker=args.k_ranker,
        eta=args.eta,
        rescale_power=args.rescale_power,
        cosmos_az_login=args.cosmos_az_login,
    )
    total_fulltext_k = retriever.total_fulltext_k
    total_vector_k = retriever.total_vector_k
    total_k = total_fulltext_k + total_vector_k
    _log_line(
        f"Decomposed RAG: sources={retriever.source_count}, "
        f"fulltext_total={total_fulltext_k}, vector_total={total_vector_k}, diverse={args.k_diverse}, ranker={args.k_ranker}"
        ,
        kind="info"
    )
    if _TIMING:
        _log_line("¤ enabled: checkpoints printed as +<step_elapsed>s (total <from_start>s)", kind="timing")

    t = _ck("retriever.initialize – start")
    await retriever.initialize()
    _ck("retriever.initialize – done", t)
    llm = LLMClient(azure_az_login=args.azure_az_login)
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
    _log_line(f"Processing {len(all_questions)} questions", kind="info")
    
    div_suffix = f"_div{args.k_diverse}" if args.k_diverse > 0 else ""
    ranker_suffix = f"_rank{args.k_ranker}" if args.k_ranker > 0 else ""
    output_path = args.output_root / f"k{total_k}_ft{total_fulltext_k}_vec{total_vector_k}{div_suffix}{ranker_suffix}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = []

    async def process(q: Question):
        token = _CURRENT_QUESTION_ID.set(q.question_id)
        try:
            if args.separate_subq_calls:
                result = await pipeline.run(q.question_text)
            else:
                result = await pipeline.run_efficient(q.question_text)
        finally:
            _CURRENT_QUESTION_ID.reset(token)
        result["question_id"] = q.question_id
        result["question_text"] = q.question_text
        result["group"] = q.group
        result["ground_truth"] = q.ground_truth
        group_name = q.group or "default"
        group_dir = output_path / "intermediate" / group_name
        await asyncio.to_thread(group_dir.mkdir, parents=True, exist_ok=True)
        result_file = group_dir / f"{q.question_id}.json"
        await asyncio.to_thread(result_file.write_text, json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
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
                _log_line(f"Error: {e}", kind="error")
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
    written_answer_files: list[Path] = []
    for source_stem, answers in results_by_file.items():
        answers_filename = f"{source_stem}_{timestamp}.json"
        answers_file_path = output_path / answers_filename
        await asyncio.to_thread(answers_file_path.write_text, json.dumps(answers, indent=2, ensure_ascii=False), encoding="utf-8")
        written_answer_files.append(answers_file_path)
    if len(written_answer_files) == 1:
        _log_line(f"Done! Answers file: {written_answer_files[0]}", kind="success")
    else:
        _log_line(f"Done! Answers files ({len(written_answer_files)}):", kind="success")
        for file_path in written_answer_files:
            _log_line(f"  - {file_path}", kind="success")

    _log_line(f"Total symbols passed to LLM: {llm.total_prompt_chars:,}", kind="info")
    if llm.total_llm_calls > 0:
        avg_tokens = llm.total_prompt_tokens / llm.total_llm_calls
        _log_line(f"Total premium prompt tokens: {llm.total_prompt_tokens:,} across {llm.total_llm_calls} LLM calls (avg {avg_tokens:,.0f} tokens/call)", kind="info")

    await retriever.close()
    await llm.close()

if __name__ == "__main__":
    # Ensure 'import agentic_retriever' in other modules (e.g. utils.cosmos_retriever)
    # resolves to this same module instance, not a duplicate with empty CONFIG.
    sys.modules.setdefault("agentic_retriever", sys.modules[__name__])

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
        _log_line(f"¤ wrote log: {run_log_path}", kind="timing")
        _log_line(f"¤ updated latest: {latest_log_path}", kind="timing")
    else:
        asyncio.run(main_async())
