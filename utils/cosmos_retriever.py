"""Cosmos DB retriever – extracted from dynamic_retriever.py for readability."""

import asyncio
import copy
import logging
import os
import re
import sys
import warnings
from collections.abc import Callable
from typing import Any

import httpx
import numpy as np
from azure.cosmos.aio import CosmosClient
from azure.identity.aio import AzureCliCredential as AsyncAzureCliCredential, DefaultAzureCredential

import dynamic_retriever as _rag
from dynamic_retriever import (
    _log_line,
    _format_activity_id_note,
    _multi_activity_reason,
    CONFIG,
    LLMClient,
    LRUCache,
    RetrievedChunk,
)
from greedy_log_det import greedy_log_det_select
from utils.fulltext import fulltext_field_from_index_path, is_fulltext_field_path
from utils.candidates import (
    candidate_identity,
    candidate_projection,
    container_identity,
    fetch_candidate_results,
    resolve_partition_key_path,
)


def _runtime_module() -> Any:
    main_mod = sys.modules.get("__main__")
    if main_mod and hasattr(main_mod, "_ck"):
        return main_mod
    return _rag


def _timing_enabled() -> bool:
    return bool(getattr(_runtime_module(), "_TIMING", False))


def _ck(label: str, ref: float | None = None) -> float:
    ck_fn = getattr(_runtime_module(), "_ck")
    return ck_fn(label, ref)

# =============================================================================
# COSMOS DB RETRIEVER
# =============================================================================

COSMOS_ENDPOINT = os.getenv("COSMOS_ENDPOINT", CONFIG["cosmos"]["uri"])
COSMOS_KEY = os.getenv("COSMOS_KEY", CONFIG["cosmos"].get("key", ""))
DATABASE_NAME = os.getenv("COSMOS_DATABASE_NAME", CONFIG["cosmos"]["database_name"])


def _as_list_of_strings(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _is_identifier_like_field(field_name: str) -> bool:
    name = str(field_name or "").strip().lower()
    if not name:
        return False
    if name in {"id", "document_id", "doc_id", "upc", "sku", "ean", "gtin", "uuid"}:
        return True
    return name.endswith("_id")


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
        # Accept canonical names (search_k / fulltext_search_k) and legacy
        # aliases (vector_k / fulltext_k). dynamic_retriever.load_config()
        # rewrites legacy keys with a DeprecationWarning before this code runs,
        # so the legacy fallbacks here are belt-and-suspenders.
        vector_k = int(
            retrieval_cfg.get("search_k", retrieval_cfg.get("vector_k", 0)) or 0
        )
        fulltext_k = int(
            retrieval_cfg.get("fulltext_search_k", retrieval_cfg.get("fulltext_k", 0)) or 0
        )
        normalized_sources.append(
            {
                "id": source_id,
                "container_name": source.get("container_name"),
                "partition_key_path": source.get("partition_key_path"),
                "embedding_field": str(source.get("embedding_field") or "e").strip(),
                "vector_k": vector_k,
                "fulltext_k": fulltext_k,
                "fulltext_fields": _as_list_of_strings(retrieval_cfg.get("fulltext_fields")),
            }
        )
    return normalized_sources


RETRIEVAL_SOURCES = _get_source_config(CONFIG)

# Re-export from utils.fulltext for backward compatibility
from utils.fulltext import fulltext_search  # noqa: E402
from utils.ranker import RankerUnavailableError, rerank_documents, verify_ranker_token  # noqa: E402


class CombinedRetriever:
    def __init__(
        self,
        retrieval_sources: list[dict[str, Any]],
        fulltext_k_override: int | None = None,
        k_diverse: int = 0,
        k_ranker: int = 0,
        eta: float = 0.0,
        rescale_power: float = 0.0,
        cosmos_az_login: bool = False,
    ):
        self.k_diverse = k_diverse
        self.k_ranker = k_ranker
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
        # Store per-source context_fields for use in formatting
        self._source_context_fields = {}
        self._sources = self._normalize_sources(retrieval_sources, fulltext_k_override)
        # Extract context_fields for each source (by id)
        cosmos_cfg = CONFIG.get("cosmos", {})
        config_sources = cosmos_cfg.get("sources", [])
        for src in config_sources:
            sid = str(src.get("id") or "").strip()
            fields = src.get("context_fields", [])
            if sid and isinstance(fields, list):
                self._source_context_fields[sid] = [str(f) for f in fields if str(f).strip()]
        self._ranker_http_client: httpx.AsyncClient | None = None
        ranker_cfg = CONFIG.get("ranker", {})
        self._use_ranker = bool(ranker_cfg.get("use_ranker", False))
        self._ranker_region = str(ranker_cfg.get("region", "")).strip()
        self._ranker_account = str(ranker_cfg.get("account_name", "")).strip()
        self._ranker_batch_size = int(ranker_cfg.get("batch_size", 32))
        self._ranker_fail_on_error = bool(ranker_cfg.get("fail_on_error", False))
        self._indexed_fields_cache: dict[str, list[str]] = {}
        self._ranker_access_token: str | Callable[[], str] | None = None
        if self._use_ranker:
            from azure.identity import AzureCliCredential as SyncAzureCliCredential, get_bearer_token_provider
            tenant_id = str(ranker_cfg.get("tenant_id", "")).strip()
            token_scope = str(ranker_cfg.get("token_scope", "")).strip()
            if not token_scope:
                raise ValueError("ranker.token_scope must be set when ranker.use_ranker is enabled")
            credential = SyncAzureCliCredential(tenant_id=tenant_id, process_timeout=60)
            self._ranker_access_token = get_bearer_token_provider(credential, token_scope)

    @property
    def total_fulltext_k(self) -> int:
        return sum(int(source.get("fulltext_k", 0) or 0) for source in self._sources)

    @property
    def total_vector_k(self) -> int:
        return sum(int(source.get("vector_k", 0) or 0) for source in self._sources)

    @property
    def source_count(self) -> int:
        return len(self._sources)

    @property
    def configured_context_fields(self) -> list[str]:
        fields: set[str] = set()
        for source_fields in self._source_context_fields.values():
            for field in source_fields:
                cleaned = str(field).strip()
                if cleaned:
                    fields.add(cleaned)
        return sorted(fields)

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
            configured_fields = _as_list_of_strings(source.get("fulltext_fields"))
            fulltext_fields = [field for field in configured_fields if is_fulltext_field_path(field)]
            for field in configured_fields:
                if field not in fulltext_fields:
                    logging.warning(
                        f"Source {source_id!r}: skipping unsupported full-text field {field!r} "
                        "(use dotted identifiers, with an optional trailing '[]' for string arrays)"
                    )
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
        tenant_id = str(CONFIG.get("cosmos", {}).get("tenant_id") or "").strip()
        if self._cosmos_az_login or tenant_id:
            credential = AsyncAzureCliCredential(tenant_id=tenant_id) if tenant_id else AsyncAzureCliCredential()
            self._credential = credential
            if tenant_id:
                _log_line(
                    f"✓ Using tenant-scoped AzureCliCredential authentication for Cosmos DB ({tenant_id})",
                    kind="success",
                )
            else:
                _log_line("✓ Using 'az login' (AzureCliCredential) authentication for Cosmos DB", kind="success")
            self._cosmos = CosmosClient(COSMOS_ENDPOINT, credential=credential)
        elif use_rbac_auth:
            credential = DefaultAzureCredential()
            self._credential = credential
            _log_line("✓ Using Entra ID RBAC authentication for Cosmos DB", kind="success")
            self._cosmos = CosmosClient(COSMOS_ENDPOINT, credential=credential)
        else:
            if not COSMOS_KEY:
                raise ValueError("Cosmos DB key not configured. Set cosmos.key in config.yaml.")
            _log_line("✓ Using key-based authentication for Cosmos DB", kind="success")
            self._cosmos = CosmosClient(COSMOS_ENDPOINT, credential=COSMOS_KEY, connection_mode="Direct")

        self._db = self._cosmos.get_database_client(DATABASE_NAME)
        for source in self._sources:
            self._containers[source["id"]] = self._db.get_container_client(source["container_name"])
        self._llm = LLMClient()
        try:
            await self._validate_startup()
        except BaseException:
            await self.close()
            raise

    async def _validate_startup(self) -> None:
        """Fail fast on setup problems that would otherwise surface mid-run."""
        for source in self._sources:
            container = self._containers[source["id"]]
            if int(source.get("vector_k", 0) or 0) > 0 or int(source.get("fulltext_k", 0) or 0) > 0:
                source["partition_key_path"] = await resolve_partition_key_path(
                    container, source.get("partition_key_path"), source["id"],
                )
            if int(source.get("fulltext_k", 0) or 0) > 0:
                self._ctrproxy = container
                await self._indexed_fulltext_fields()
        if self._use_ranker:
            problem = await verify_ranker_token(self._ranker_access_token, self._ranker_fail_on_error)
            if problem:
                _log_line(problem, kind="error")

    async def _fulltext_search(
        self, container, fields: list[str], query: str, top_k: int, *,
        partition_key_path: str,
    ) -> list[dict]:
        if top_k <= 0 or not fields:
            return []
        t = _ck(f"fulltext (top {top_k}, {container.id}, {len(fields)} fields) – start")

        def _log_fulltext_sql(sql: str, k: int, text: str) -> None:
            if _timing_enabled():
                _log_line(
                    f"  fulltext SQL ({container.id}): {sql}  "
                    f"[top_k={k}, text={text!r}]",
                    kind="query",
                    use_lock=True,
                )

        items = await fulltext_search(
            container, fields, query, top_k, log_fn=_log_fulltext_sql,
            partition_key_path=partition_key_path,
        )
        _ck(f"fulltext – done ({len(items)} results, {container.id})", t)
        return items

    async def _vector_search(
        self,
        container,
        embedding_field: str,
        query_emb: list[float],
        top_k: int,
        query_text: str = "",
        *,
        partition_key_path: str,
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
            f"SELECT TOP @k {candidate_projection(partition_key_path)}, "
            f"VectorDistance(c.{vector_field}, @emb) AS _score "
            f"FROM c ORDER BY VectorDistance(c.{vector_field}, @emb)"
        )
        if _timing_enabled():
            text_preview = f", text={query_text!r}" if query_text else ""
            _log_line(
                f"  vector SQL ({container.id}): {sql}  "
                f"[@k={top_k}, @emb=<{len(adjusted_emb)}-dim vector>, @field={vector_field!r}{text_preview}]",
                kind="query",
                use_lock=True,
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
        return results
    
    def _format_doc(self, doc: dict, source: str, embedding_field: str = "e") -> RetrievedChunk:
        emb_field = str(embedding_field or "e").strip()
        embedding = doc.get(emb_field) if isinstance(doc.get(emb_field), list) else doc.get('embedding')
        exclude = {'_rid', '_self', '_etag', '_attachments', '_ts', 'embedding', '_score', emb_field}
        # Always include user-specified context_fields (if present in doc)
        context_source = str(source or "")
        for suffix in ("_fulltext", "_vector"):
            if context_source.endswith(suffix):
                context_source = context_source[: -len(suffix)]
                break
        context_fields = self._source_context_fields.get(context_source, [])
        context_parts = []
        for field in context_fields:
            if field in doc and doc[field] is not None:
                context_parts.append(f"{field}: {doc[field]}")
        # Add the rest of the fields as before (excluding context_fields to avoid duplication)
        parts = []
        for k, v in doc.items():
            if k in exclude or not v or k in context_fields:
                continue
            # Hide non-configured identifier-like fields to avoid leaking extra IDs
            # (for example upc/document_id) when user explicitly configured only one ID field.
            if _is_identifier_like_field(k):
                continue
            parts.append(f"{k.replace('_', ' ').title()}: {v if not isinstance(v, (list, dict)) else str(v)}")
        all_parts = context_parts + parts
        return RetrievedChunk(
            chunk_id=doc.get('id', ''),
            text="\n".join(all_parts),
            similarity=(1 - doc.get('_score', 0)) if '_score' in doc else None,
            metadata={'_data_source': source, 'embedding': embedding}
        )
    
    async def retrieve(self, query: str, k_divisor: int = 1) -> list[RetrievedChunk]:
        """Retrieve chunks for *query*.

        Parameters
        ----------
        query : str
            The search query.
        k_divisor : int, optional
            When > 1 every configured ``fulltext_k`` and ``vector_k`` is divided
            by this value.  Used by the ``--efficient``
            pipeline so each sub-question retrieves ``k / #subquestions`` texts.
        """
        if self._llm is None:
            raise RuntimeError("Retriever is not initialized")
        k_divisor = max(1, int(k_divisor))
        t_retrieve = _ck(f"retrieve – start (q: {query[:60]!r}, k_divisor={k_divisor})")

        cache_key = f"{query}|||divisor={k_divisor}"
        cached = self._retrieve_cache.get(cache_key)
        if isinstance(cached, list):
            _ck("retrieve – cache hit", t_retrieve)
            return copy.deepcopy(cached)

        chunks: list[RetrievedChunk] = []
        seen: set[tuple[tuple[str, str, str] | tuple[str, int], tuple[str, str]]] = set()
        references = []
        origins = []

        fulltext_tasks: list[tuple[dict[str, Any], asyncio.Task]] = []
        for source in self._sources:
            container = self._containers.get(source["id"])
            if container is None:
                continue
            top_k = int(source.get("fulltext_k", 0) or 0) // k_divisor
            if top_k <= 0:
                continue

            # Try configured full-text search fields
            fulltext_search_fields = []

            # First attempt: discover actual full-text indexed fields from container indexing policy.
            # Fallback to config-based values if this fails or yields nothing.
            self._ctrproxy = container
            indexed_fields = await self._indexed_fulltext_fields()
            if indexed_fields:
                # Keep existing loop contract where each item is a field-combination list.
                fulltext_search_fields = [indexed_fields]
                logging.info(
                    f"_fallback_text_search: Loaded {len(indexed_fields)} indexed full-text fields from container policy"
                )

            # Fallback: load from config (e.g., .env or config service)
            if not fulltext_search_fields:
                configured = source.get("fulltext_fields") or []
                if configured:
                    fulltext_search_fields = [configured]

            if not fulltext_search_fields:
                continue

            for fields in fulltext_search_fields:
                if not fields:
                    continue
                t_fulltext = _ck(f"  retrieve: fulltext/{source['id']} – start (parallel)")
                task = asyncio.create_task(self._fulltext_search(
                    container, fields, query, top_k,
                    partition_key_path=source["partition_key_path"],
                ))
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
                vec_k = int(source.get("vector_k", 0) or 0) // k_divisor
                t_vector = _ck(f"  retrieve: vector/{source['id']} – start (parallel)")
                task = asyncio.create_task(
                    self._vector_search(
                        container,
                        str(source.get("embedding_field") or "e"),
                        emb,
                        vec_k,
                        query,
                        partition_key_path=source["partition_key_path"],
                    )
                )
                vector_tasks.append(({"source": source, "timer": t_vector}, task))

        for info, task in fulltext_tasks:
            source = info["source"]
            docs = await task
            _ck(f"  retrieve: fulltext/{source['id']} – done ({len(docs)} results)", info["timer"])
            for doc in docs:
                dedupe_key = (container_identity(self._containers[source["id"]]), candidate_identity(doc))
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                references.append((self._containers[source["id"]], doc))
                origins.append((source, "fulltext"))

        for info, task in vector_tasks:
            source = info["source"]
            docs = await task
            _ck(f"  retrieve: vector/{source['id']} – done ({len(docs)} results)", info["timer"])
            for doc in docs:
                dedupe_key = (container_identity(self._containers[source["id"]]), candidate_identity(doc))
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                references.append((self._containers[source["id"]], doc))
                origins.append((source, "vector"))

        t_fetch = _ck("fetch candidates – start")
        documents = await fetch_candidate_results(references)
        _ck(f"fetch candidates – done ({sum(d is not None for d in documents)} documents)", t_fetch)
        for doc, (source, kind) in zip(documents, origins):
            if doc is None:
                continue
            chunks.append(self._format_doc(
                doc, f"{source['id']}_{kind}", str(source.get("embedding_field") or "e"),
            ))
        
        # Diversity selection via greedy log-det maximization
        if self.k_diverse > 0 and len(chunks) > self.k_diverse:
            t = _ck("  retrieve: diversity embed missing – start")
            missing_chunks = [c for c in chunks if c.metadata.get('embedding') is None]
            n_missing = len(missing_chunks)
            if missing_chunks:
                _log_line(f"  {n_missing} chunks missing embeddings, computing now...", kind="warn")
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

        # Semantic ranker reranking
        effective_k_ranker = self.k_ranker // k_divisor
        if effective_k_ranker > 0 and self._use_ranker and len(chunks) > effective_k_ranker and self._ranker_account and self._ranker_access_token:
            t = _ck("  retrieve: semantic ranker – start")
            if self._ranker_http_client is None:
                self._ranker_http_client = httpx.AsyncClient(timeout=120)
            documents = [c.text for c in chunks]
            url_suffix = "dbinference.azure.com:443/inference/semanticReranking"
            url = f"https://{self._ranker_account}.{self._ranker_region}.{url_suffix}"
            max_retries = int(CONFIG.get("ranker", {}).get("max_retries", 5))
            ranked_indices = await rerank_documents(
                self._ranker_http_client, url, self._ranker_access_token,
                query, documents, effective_k_ranker, self._ranker_batch_size, max_retries,
            )
            if ranked_indices is None and self._ranker_fail_on_error:
                raise RankerUnavailableError(
                    "Semantic ranker failed and ranker.fail_on_error is set; see the "
                    "[rerank_documents] messages above for the cause."
                )
            if ranked_indices is not None:
                chunks = [chunks[i] for i in ranked_indices]
                _ck(f"  retrieve: semantic ranker – done (selected {len(chunks)} of {effective_k_ranker} requested)", t)
            else:
                _log_line("Semantic ranker failed, keeping previous chunks", kind="error")
                _ck("  retrieve: semantic ranker – failed, keeping diversity-selected chunks", t)

        self._retrieve_cache.set(cache_key, copy.deepcopy(chunks))
        
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
        if self._ranker_http_client is not None:
            await self._ranker_http_client.aclose()
            self._ranker_http_client = None

    async def _indexed_fulltext_fields(self):
        """
        Return full-text indexed field names from the container indexing policy.

        Unsupported index paths are skipped with one warning each. The result is
        cached per container after a successful policy read.
        """
        container_id = getattr(self._ctrproxy, "id", None)
        if container_id in self._indexed_fields_cache:
            return list(self._indexed_fields_cache[container_id])
        indexed_fields = []
        try:
            props = await self._ctrproxy.read()
            indexing_policy = props.get("indexingPolicy", {})
            full_text_indexes = indexing_policy.get("fullTextIndexes", [])
        except Exception as index_error:
            logging.warning(
                f"fulltext_search: Could not read fullTextIndexes from container policy: {str(index_error)[:200]}"
            )
            return indexed_fields

        for index in full_text_indexes:
            path = index.get("path", "")
            if not path:
                continue
            try:
                normalized = fulltext_field_from_index_path(path)
            except ValueError as path_error:
                logging.warning(
                    f"fulltext_search: Skipping full-text index path in container "
                    f"{container_id!r}: {path_error}"
                )
                continue
            if normalized and normalized not in indexed_fields:
                indexed_fields.append(normalized)

        if container_id is not None:
            self._indexed_fields_cache[container_id] = list(indexed_fields)
        return indexed_fields