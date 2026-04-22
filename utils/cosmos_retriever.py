"""Cosmos DB retriever – extracted from agentic_retriever.py for readability."""

import asyncio
import copy
import os
import re
import sys
import time
import warnings
from typing import Any

import httpx
import numpy as np
from azure.cosmos.aio import CosmosClient
from azure.identity.aio import AzureCliCredential as AsyncAzureCliCredential, DefaultAzureCredential

import agentic_retriever as _rag
from agentic_retriever import (
    _log_line,
    _format_activity_id_note,
    _multi_activity_reason,
    CONFIG,
    LLMClient,
    LRUCache,
    RetrievedChunk,
)
from greedy_log_det import greedy_log_det_select


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

# Re-export from utils.fulltext for backward compatibility
from utils.fulltext import fulltext_search  # noqa: E402
from utils.ranker import rerank_documents  # noqa: E402


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
        self._sources = self._normalize_sources(retrieval_sources, fulltext_k_override)
        self._ranker_http_client: httpx.AsyncClient | None = None
        ranker_cfg = CONFIG.get("ranker", {})
        self._use_ranker = bool(ranker_cfg.get("use_ranker", False))
        self._ranker_region = str(ranker_cfg.get("region", "")).strip()
        self._ranker_account = str(ranker_cfg.get("account_name", "")).strip()
        self._ranker_batch_size = int(ranker_cfg.get("batch_size", 32))
        self._ranker_access_token: str | None = None
        if self._use_ranker:
            read_token_from_path = bool(ranker_cfg.get("read_token_from_path", True))
            if read_token_from_path:
                token_path = str(ranker_cfg.get("access_token_path", "access_token.txt")).strip()
                if token_path and os.path.isfile(token_path):
                    with open(token_path, "r") as f:
                        self._ranker_access_token = f.read().strip()
            else:
                from azure.identity import AzureCliCredential as SyncAzureCliCredential
                tenant_id = str(ranker_cfg.get("tenant_id", "")).strip()
                token_scope = str(ranker_cfg.get("token_scope", "")).strip()
                if not token_scope:
                    raise ValueError("ranker.token_scope must be a non-empty string when read_token_from_path is false")
                credential = SyncAzureCliCredential(tenant_id=tenant_id) if tenant_id else SyncAzureCliCredential()
                token_obj = credential.get_token(token_scope)
                self._ranker_access_token = token_obj.token

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

        # Register ranker account (idempotent – safe to call every time)
        if self._use_ranker and self._ranker_account and self._ranker_access_token:
            register_account_path = str(CONFIG.get("ranker", {}).get("register_account_path", "")).strip()
            if self._ranker_region and register_account_path:
                if self._ranker_http_client is None:
                    self._ranker_http_client = httpx.AsyncClient(timeout=120)
                register_url = f"https://{self._ranker_region}.{register_account_path}"
                register_payload = {
                    "AccountName": self._ranker_account,
                    "Region": self._ranker_region,
                }
                register_headers = {
                    "Authorization": f"Bearer {self._ranker_access_token}",
                    "Content-Type": "application/json",
                }
                try:
                    resp = await self._ranker_http_client.post(register_url, headers=register_headers, json=register_payload)
                    if resp.status_code == 200:
                        _log_line(f"✓ Ranker account '{self._ranker_account}' registered", kind="success")
                    else:
                        _log_line(f"Ranker account registration returned {resp.status_code}: {resp.text[:200]}", kind="warn")
                except Exception as e:
                    _log_line(f"Ranker account registration failed: {e}", kind="warn")

    async def _fulltext_search(self, container, fields: list[str], query: str, top_k: int) -> list[dict]:
        if top_k <= 0 or not fields:
            return []
        t = _ck(f"fulltext (top {top_k}, {container.id}, {len(fields)} fields) – start")
        items = await fulltext_search(container, fields, query, top_k)
        _ck(f"fulltext – done ({len(items)} results, {container.id})", t)
        return items

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
        seen: set[tuple[str, Any]] = set()

        fulltext_tasks: list[tuple[dict[str, Any], asyncio.Task]] = []
        for source in self._sources:
            container = self._containers.get(source["id"])
            if container is None:
                continue
            top_k = int(source.get("fulltext_k", 0) or 0) // k_divisor
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
                vec_k = int(source.get("vector_k", 0) or 0) // k_divisor
                t_vector = _ck(f"  retrieve: vector/{source['id']} – start (parallel)")
                task = asyncio.create_task(
                    self._vector_search(
                        container,
                        str(source.get("embedding_field") or "e"),
                        emb,
                        vec_k,
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
            url_suffix = str(CONFIG.get("ranker", {}).get("url_suffix", "")).strip()
            url = f"https://{self._ranker_account}.{self._ranker_region}.{url_suffix}"
            max_retries = int(CONFIG.get("ranker", {}).get("max_retries", 5))
            ranked_indices = await rerank_documents(
                self._ranker_http_client, url, self._ranker_access_token,
                query, documents, effective_k_ranker, self._ranker_batch_size, max_retries,
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
