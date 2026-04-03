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

# Comprehensive BM25 stopwords list
STOPWORDS = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "a's", "able", "about", "above", "according", "accordingly", "across", "actually", "after", "afterwards", "again", "against", "ain't", "all", "allow", "allows", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "an", "and", "another", "any", "anybody", "anyhow", "anyone", "anything", "anyway", "anyways", "anywhere", "apart", "appear", "appreciate", "appropriate", "are", "aren't", "around", "as", "aside", "ask", "asking", "associated", "at", "available", "away", "awfully", "b", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "behind", "being", "believe", "below", "beside", "besides", "best", "better", "between", "beyond", "both", "brief", "but", "by", "c", "c'mon", "c's", "came", "can", "can't", "cannot", "cant", "cause", "causes", "certain", "certainly", "changes", "clearly", "co", "com", "come", "comes", "concerning", "consequently", "consider", "considering", "contain", "containing", "contains", "corresponding", "could", "couldn't", "course", "currently", "d", "definitely", "described", "despite", "did", "didn't", "different", "do", "does", "doesn't", "doing", "don", "don't", "done", "down", "downwards", "during", "e", "each", "edu", "eg", "eight", "either", "else", "elsewhere", "enough", "entirely", "especially", "et", "etc", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "f", "far", "few", "fifth", "first", "five", "followed", "following", "follows", "for", "former", "formerly", "forth", "four", "from", "further", "furthermore", "g", "get", "gets", "getting", "given", "gives", "go", "goes", "going", "gone", "got", "gotten", "greetings", "h", "had", "hadn't", "happens", "hardly", "has", "hasn't", "have", "haven't", "having", "he", "he's", "hello", "help", "hence", "her", "here", "here's", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "hi", "him", "himself", "his", "hither", "hopefully", "how", "howbeit", "however", "i", "i'd", "i'll", "i'm", "i've", "ie", "if", "ignored", "immediate", "in", "inasmuch", "inc", "indeed", "indicate", "indicated", "indicates", "inner", "insofar", "instead", "into", "inward", "is", "isn't", "it", "it'd", "it'll", "it's", "its", "itself", "j", "just", "k", "keep", "keeps", "kept", "know", "known", "knows", "l", "last", "lately", "later", "latter", "latterly", "least", "less", "lest", "let", "let's", "like", "liked", "likely", "little", "ll", "look", "looking", "looks", "ltd", "m", "mainly", "make", "many", "may", "maybe", "me", "mean", "meanwhile", "merely", "might", "more", "moreover", "most", "mostly", "mr", "mrs", "ms", "much", "must", "my", "myself", "n", "name", "namely", "nd", "near", "nearly", "necessary", "need", "needs", "neither", "never", "nevertheless", "new", "next", "nine", "no", "nobody", "non", "none", "noone", "nor", "normally", "not", "nothing", "novel", "now", "nowhere", "o", "obviously", "of", "off", "often", "oh", "ok", "okay", "old", "on", "once", "one", "ones", "only", "onto", "or", "other", "others", "otherwise", "ought", "our", "ours", "ourselves", "out", "outside", "over", "overall", "own", "p", "particular", "particularly", "per", "perhaps", "placed", "please", "plus", "possible", "presumably", "probably", "provides", "q", "que", "quite", "qv", "r", "rather", "rd", "re", "really", "reasonably", "regarding", "regardless", "regards", "relatively", "respectively", "right", "s", "said", "same", "saw", "say", "saying", "says", "second", "secondly", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sensible", "sent", "serious", "seriously", "seven", "several", "shall", "she", "should", "shouldn't", "since", "six", "so", "some", "somebody", "somehow", "someone", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "specified", "specify", "specifying", "still", "sub", "such", "sup", "sure", "t", "t's", "take", "taken", "tell", "tends", "th", "than", "thank", "thanks", "thanx", "that", "that's", "thats", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "there's", "thereafter", "thereby", "therefore", "therein", "theres", "thereupon", "these", "they", "they'd", "they'll", "they're", "they've", "think", "third", "this", "thorough", "thoroughly", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "took", "toward", "towards", "tried", "tries", "truly", "try", "trying", "twice", "two", "u", "un", "under", "unfortunately", "unless", "unlikely", "until", "unto", "up", "upon", "us", "use", "used", "useful", "uses", "using", "usually", "v", "value", "various", "ve", "very", "via", "viz", "vs", "w", "want", "wants", "was", "wasn't", "way", "we", "we'd", "we'll", "we're", "we've", "welcome", "well", "went", "were", "weren't", "what", "what's", "whatever", "when", "whence", "whenever", "where", "where's", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "who's", "whoever", "whole", "whom", "whose", "why", "will", "willing", "wish", "with", "within", "without", "won't", "wonder", "would", "wouldn't", "x", "y", "yes", "yet", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves", "z", "zero"}


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
        if self._cosmos_az_login:
            credential = AsyncAzureCliCredential()
            self._credential = credential
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

    async def _fulltext_search_single_field(self, container, field: str, query: str, top_k: int) -> list[dict]:
        """Run a fulltext search on a single field, returning ranked results."""
        if top_k <= 0:
            return []
        terms = [t for t in re.findall(r"\w+", query) if t.lower() not in STOPWORDS and len(t) > 1]
        if not terms:
            return []

        chunks = [terms[i:i + 5] for i in range(0, len(terms), 5)]
        field_expr = f"c.{field}"
        score_exprs = []
        for term_chunk in chunks:
            args = ", ".join(f'"{term}"' for term in term_chunk)
            score_exprs.append(f"FullTextScore({field_expr}, {args})")
        if not score_exprs:
            return []

        if len(score_exprs) == 1:
            order = f"ORDER BY RANK {score_exprs[0]}"
        else:
            order = f"ORDER BY RANK RRF({', '.join(score_exprs)})"

        sql = f"SELECT TOP {top_k} * FROM c {order}"
        try:
            if _timing_enabled():
                _log_line(
                    f"  fulltext SQL ({container.id}/{field}): {sql}  [text={query!r}]",
                    kind="query",
                    use_lock=True,
                )

            t = _ck(f"fulltext query (top {top_k}, {container.id}/{field}) – start")
            query_iterator = container.query_items(query=sql, parameters=[])
            items = []
            async for item in query_iterator:
                items.append(item)

            _ck(f"fulltext query – done ({len(items)} results, {container.id}/{field})", t)
            return items
        except Exception as e:
            _log_line(f"Fulltext error ({container.id}/{field}): {e}", kind="error")
            return []

    async def _fulltext_search(self, container, fields: list[str], query: str, top_k: int) -> list[dict]:
        if top_k <= 0 or not fields:
            return []
        terms = [t for t in re.findall(r"\w+", query) if t.lower() not in STOPWORDS and len(t) > 1]
        if not terms:
            return []

        # Single field: run directly (no need for client-side RRF)
        if len(fields) == 1:
            return await self._fulltext_search_single_field(container, fields[0], query, top_k)

        # Multiple fields: run parallel per-field queries and merge via client-side RRF
        t = _ck(f"fulltext parallel-RRF (top {top_k}, {container.id}, {len(fields)} fields) – start")
        per_field_tasks = [
            asyncio.create_task(self._fulltext_search_single_field(container, field, query, top_k))
            for field in fields
        ]
        per_field_results = await asyncio.gather(*per_field_tasks)

        # Client-side RRF merge: score each doc by 1/(k+rank) across fields, pick top_k
        rrf_k = 60  # standard RRF constant
        doc_scores: dict[str, float] = {}
        doc_map: dict[str, dict] = {}
        for field_items in per_field_results:
            for rank, item in enumerate(field_items):
                doc_id = item.get("id", "")
                if not doc_id:
                    continue
                doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
                if doc_id not in doc_map:
                    doc_map[doc_id] = item

        sorted_ids = sorted(doc_scores, key=lambda did: doc_scores[did], reverse=True)[:top_k]
        merged = [doc_map[did] for did in sorted_ids]
        _ck(f"fulltext parallel-RRF – done ({len(merged)} merged from {sum(len(r) for r in per_field_results)} total, {container.id})", t)
        return merged

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
            body = {
                "query": query,
                "documents": documents,
                "return_documents": False,
                "top_k": effective_k_ranker,
                "batch_size": self._ranker_batch_size,
            }
            headers = {
                "Authorization": f"Bearer {self._ranker_access_token}",
                "Content-Type": "application/json",
            }
            url_suffix = str(CONFIG.get("ranker", {}).get("url_suffix", "")).strip()
            url = f"https://{self._ranker_account}.{self._ranker_region}.{url_suffix}"
            max_retries = int(CONFIG.get("ranker", {}).get("max_retries", 5))
            ranker_succeeded = False
            for attempt in range(max_retries):
                try:
                    response = await self._ranker_http_client.post(url, headers=headers, json=body)
                    if response.status_code in (502, 503, 429) and attempt < max_retries - 1:
                        wait = 2 ** attempt
                        _log_line(f"Semantic ranker returned {response.status_code}, retrying in {wait}s (attempt {attempt + 1}/{max_retries})", kind="warn")
                        await asyncio.sleep(wait)
                        continue
                    response.raise_for_status()
                    result = response.json()
                    scores = result.get("Scores", [])
                    # Select top k_ranker by reranker score, preserving original chunk objects
                    ranked_indices = [s["index"] for s in scores[:effective_k_ranker]]
                    chunks = [chunks[i] for i in ranked_indices]
                    _ck(f"  retrieve: semantic ranker – done (selected {len(chunks)} of {effective_k_ranker} requested)", t)
                    ranker_succeeded = True
                    break
                except Exception as e:
                    if attempt < max_retries - 1 and ("503" in str(e) or "502" in str(e) or "429" in str(e)):
                        wait = 2 ** attempt
                        _log_line(f"Semantic ranker error (attempt {attempt + 1}/{max_retries}): {e}, retrying in {wait}s", kind="warn")
                        await asyncio.sleep(wait)
                        continue
                    _log_line(f"Semantic ranker error: {e}", kind="error")
                    break
            if not ranker_succeeded:
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
