"""
Cosmos DB Document Upload Script with OpenAI Embeddings

This script:
1. Automatically ensures missing Cosmos DB containers for enabled upload targets when
    management settings are available; otherwise skips missing targets.
2. Determines upload targets from `cosmos.sources` in config, using each source's
    `documents_root` (folder or JSONL file path).
3. Scans all files in each enabled folder (content is assumed JSON) or reads JSONL files,
    parses documents, and generates embeddings from the configured embedding endpoint/model.
4. Uploads documents to the matching container and stores vectors in the configured embedding field.
"""

import os
import argparse
import asyncio
import json
import time
import re
import hashlib
from typing import List, Dict, Any, Optional
import httpx
from azure.cosmos.aio import CosmosClient
from azure.cosmos import exceptions
from azure.core.exceptions import HttpResponseError
from azure.identity import DefaultAzureCredential as SyncDefaultAzureCredential
from azure.identity.aio import DefaultAzureCredential as AsyncDefaultAzureCredential
from azure.mgmt.cosmosdb import CosmosDBManagementClient
from azure.mgmt.cosmosdb.models import (
    SqlDatabaseCreateUpdateParameters,
    SqlDatabaseResource,
    SqlContainerCreateUpdateParameters,
    CreateUpdateOptions,
    AutoscaleSettings,
    SqlContainerResource,
    ContainerPartitionKey,
    IndexingPolicy,
    IncludedPath,
    ExcludedPath,
    VectorEmbeddingPolicy,
    VectorEmbedding,
    VectorIndex,
    VectorIndexType,
    FullTextPolicy,
    FullTextPath,
    FullTextIndexPath,
)
from openai import AsyncAzureOpenAI
from tqdm import tqdm
import yaml
from pathlib import Path

LOCAL_EMBED_WORKERS = 8
UPLOAD_WORKERS = 8

# ---------------------------------------------------------------------------
# Module-level config placeholders (set by load_config)
# ---------------------------------------------------------------------------
CONFIG: dict[str, Any] = {}
COSMOS_ENDPOINT: str = ""
COSMOS_KEY: str = ""
DATABASE_NAME: str = ""
COSMOS_ACCOUNT_NAME: str = ""
COSMOS_RESOURCE_GROUP: str = ""
AZURE_SUBSCRIPTION_ID: str = ""
EMBED_ENDPOINT: str = ""
EMBED_MODEL: str = ""
EMBEDDING_DIMENSIONS: int = 1024
EMBED_API_VERSION: str = "2024-05-01-preview"
EMBED_API_KEY: str = ""
EMBEDDING_BATCH_SIZE: int = 20
VECTOR_EMBEDDING_POLICY: dict[str, Any] | None = None
SOURCE_CONFIGS: list[dict[str, Any]] = []
THROUGHPUT_MODE: str = "autoscale"   # "autoscale" or "manual"
THROUGHPUT_VALUE: int = 1000         # max RU/s (autoscale) or fixed RU/s (manual)


def load_config(config_path: Path) -> None:
    """Load configuration from YAML and set module globals."""
    global CONFIG, COSMOS_ENDPOINT, COSMOS_KEY, DATABASE_NAME
    global COSMOS_ACCOUNT_NAME, COSMOS_RESOURCE_GROUP, AZURE_SUBSCRIPTION_ID
    global EMBED_ENDPOINT, EMBED_MODEL, EMBEDDING_DIMENSIONS, EMBED_API_VERSION, EMBED_API_KEY
    global EMBEDDING_BATCH_SIZE, VECTOR_EMBEDDING_POLICY, SOURCE_CONFIGS
    global THROUGHPUT_MODE, THROUGHPUT_VALUE

    with open(config_path) as f:
        CONFIG = yaml.safe_load(f)

    _cosmos_cfg = CONFIG.get("cosmos", {})
    COSMOS_ENDPOINT = str(_cosmos_cfg.get("uri", "")).strip()
    COSMOS_KEY = str(_cosmos_cfg.get("key", "")).strip()
    DATABASE_NAME = str(_cosmos_cfg.get("database_name", "")).strip()
    COSMOS_ACCOUNT_NAME = str(_cosmos_cfg.get("cosmos_account_name", "")).strip()
    COSMOS_RESOURCE_GROUP = str(_cosmos_cfg.get("cosmos_resource_group", "")).strip()
    AZURE_SUBSCRIPTION_ID = str(_cosmos_cfg.get("azure_subscription_id", "")).strip()

    _embed_cfg = CONFIG.get("embedding", {})
    EMBED_ENDPOINT = str(_embed_cfg.get("embed_endpoint", "")).strip().strip('"')
    EMBED_MODEL = str(_embed_cfg.get("embed_model", "")).strip()
    EMBEDDING_DIMENSIONS = int(_embed_cfg.get("embed_dimensions", 1024))
    EMBED_API_VERSION = str(_embed_cfg.get("api_version", "2024-05-01-preview"))
    EMBED_API_KEY = str(_embed_cfg.get("embed_api_key", "") or "").strip()

    EMBEDDING_BATCH_SIZE = int(_cosmos_cfg.get("embedding_batch_size", 20))

    _throughput_mode_raw = str(_cosmos_cfg.get("throughput_mode", "autoscale")).strip().lower()
    if _throughput_mode_raw not in ("autoscale", "manual"):
        raise ValueError(
            f"Invalid cosmos.throughput_mode '{_throughput_mode_raw}'. Must be 'autoscale' or 'manual'."
        )
    THROUGHPUT_MODE = _throughput_mode_raw

    _throughput_raw = _cosmos_cfg.get("throughput_value", 1000)
    try:
        THROUGHPUT_VALUE = int(_throughput_raw)
    except (TypeError, ValueError):
        raise ValueError(
            f"Invalid cosmos.throughput_value '{_throughput_raw}'. Must be a positive integer."
        )

    if THROUGHPUT_VALUE <= 0:
        raise ValueError(
            f"Invalid cosmos.throughput_value '{THROUGHPUT_VALUE}'. Must be a positive integer."
        )
    _vep_raw = _cosmos_cfg.get("vector_embedding_policy_json")
    if _vep_raw:
        VECTOR_EMBEDDING_POLICY = json.loads(_vep_raw) if isinstance(_vep_raw, str) else _vep_raw
    else:
        VECTOR_EMBEDDING_POLICY = None

    SOURCE_CONFIGS = _sources_upload_config_from_yaml(CONFIG)


def _has_value(value: Any) -> bool:
    return bool(value and str(value).strip())


def _as_list_of_strings(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _field_to_cosmos_json_path(field_name: str) -> str:
    normalized = str(field_name or "e").strip()
    segments = [segment for segment in normalized.split(".") if segment]
    if not segments:
        return "/e"
    return "/" + "/".join(segments)


def _safe_load_json_policy(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        return json.loads(raw)
    return None


def _build_indexing_policy_from_fields(text_fields: list[str], embedding_field: str = "e") -> dict[str, Any]:
    """Auto-generate indexing policy from embedding_text_fields."""
    return {
        "indexingMode": "consistent",
        "automatic": True,
        "includedPaths": [{"path": "/*"}],
        "excludedPaths": [
            {"path": "/\"_etag\"/?"},
            {"path": f"/{embedding_field}/*"},
        ],
        "fullTextIndexes": [{"path": f"/{f}"} for f in text_fields],
        "vectorIndexes": [
            {
                "path": f"/{embedding_field}",
                "type": "diskANN",
                "quantizationByteSize": 192,
                "indexingSearchListSize": 100,
            }
        ],
    }


def _build_full_text_policy_from_fields(text_fields: list[str]) -> dict[str, Any]:
    """Auto-generate full-text policy from embedding_text_fields."""
    return {
        "defaultLanguage": "en-US",
        "fullTextPaths": [{"path": f"/{f}", "language": "en-US"} for f in text_fields],
    }


def _sources_upload_config_from_yaml(config: dict[str, Any]) -> list[dict[str, Any]]:
    cosmos_cfg = config.get("cosmos", {})
    configured_sources = cosmos_cfg.get("sources")
    if not isinstance(configured_sources, list):
        raise ValueError(
            "Invalid config: cosmos.sources must be a list with at least one source entry "
            "(container_name, documents_root, indexing/full-text policy, etc.)."
        )
    if not configured_sources:
        raise ValueError(
            "Invalid config: cosmos.sources is empty. Add at least one source entry under cosmos.sources."
        )

    normalized_sources: list[dict[str, Any]] = []
    for idx, source in enumerate(configured_sources, start=1):
        source = source or {}
        source_id = str(source.get("id") or f"source_{idx}").strip()
        embedding_field = str(source.get("embedding_field") or "e").strip()
        embedding_text_fields = _as_list_of_strings(source.get("embedding_text_fields"))

        indexing_policy = _safe_load_json_policy(source.get("indexing_policy_json"))
        full_text_policy = _safe_load_json_policy(source.get("full_text_policy_json"))

        # Auto-generate policies from embedding_text_fields when not explicitly provided
        if not indexing_policy and embedding_text_fields:
            indexing_policy = _build_indexing_policy_from_fields(embedding_text_fields, embedding_field)
        if not full_text_policy and embedding_text_fields:
            full_text_policy = _build_full_text_policy_from_fields(embedding_text_fields)

        normalized_sources.append(
            {
                "id": source_id,
                "container_name": source.get("container_name"),
                "partition_key_path": source.get("partition_key_path"),
                "documents_root": source.get("documents_root"),
                "embedding_field": embedding_field,
                "embedding_text_fields": embedding_text_fields,
                "indexing_policy": indexing_policy,
                "full_text_policy": full_text_policy,
            }
        )
    return normalized_sources


def _normalize_embedding(embedding: List[float]) -> List[float]:
    values = [float(x) for x in embedding]
    if len(values) > EMBEDDING_DIMENSIONS:
        return values[:EMBEDDING_DIMENSIONS]
    if len(values) < EMBEDDING_DIMENSIONS:
        return values + [0.0] * (EMBEDDING_DIMENSIONS - len(values))
    return values


def get_embedding_client() -> AsyncAzureOpenAI | None:
    """Initialize a single embedding client from llm.embed_endpoint/embed_model settings."""
    endpoint = EMBED_ENDPOINT.rstrip("/")
    if endpoint.endswith("/api/embeddings"):
        return None

    azure_endpoint = endpoint
    if "/openai/deployments/" in azure_endpoint:
        azure_endpoint = azure_endpoint.split("/openai/deployments/")[0]

    if not EMBED_API_KEY:
        raise ValueError("embedding.embed_api_key must be set for Azure OpenAI embedding endpoint")

    return AsyncAzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=EMBED_API_KEY,
        api_version=EMBED_API_VERSION,
    )


def get_cosmos_client(use_rbac_auth: bool, credential=None) -> CosmosClient:
    """Initialize and return Cosmos DB client for selected auth mode."""
    if not COSMOS_ENDPOINT:
        raise ValueError("Cosmos DB endpoint not configured. Set COSMOS_ENDPOINT.")
    
    if use_rbac_auth:
        # Use Entra ID RBAC authentication (DefaultAzureCredential)
        if credential is None:
            credential = AsyncDefaultAzureCredential()
        print("✓ Using Entra ID RBAC authentication for Cosmos DB")
        return CosmosClient(COSMOS_ENDPOINT, credential=credential)
    else:
        # Use key-based authentication
        if not COSMOS_KEY:
            raise ValueError("Cosmos DB key not configured. Set cosmos.key in config.yaml.")
        print("✓ Using key-based authentication for Cosmos DB")
        return CosmosClient(COSMOS_ENDPOINT, credential=COSMOS_KEY)


def extract_account_name_from_endpoint(endpoint: str) -> str:
    """Extract the Cosmos DB account name from the endpoint URL."""
    # https://myaccount.documents.azure.com:443/ -> account name is myaccount
    match = re.match(r'https://([^.]+)\.documents\.azure\.com', endpoint)
    if match:
        return match.group(1)
    raise ValueError(f"Could not extract account name from endpoint: {endpoint}")


def enable_vector_search_capability(credential, subscription_id: str, resource_group: str, account_name: str):
    """
    Enable Vector Search capability on the Cosmos DB account if not already enabled.
    This is required before creating containers with vector indexing.
    """
    from azure.mgmt.cosmosdb.models import DatabaseAccountUpdateParameters, Capability
    
    mgmt_client = CosmosDBManagementClient(credential, subscription_id)
    
    # Get current account
    account = mgmt_client.database_accounts.get(resource_group, account_name)
    
    # Check if VectorSearch is already enabled
    current_capabilities = account.capabilities or []
    capability_names = [c.name for c in current_capabilities]
    
    capabilities_to_add = []
    
    if "EnableNoSQLVectorSearch" not in capability_names:
        print("  Enabling NoSQL Vector Search capability...")
        capabilities_to_add.append(Capability(name="EnableNoSQLVectorSearch"))
    
    if "EnableNoSQLFullTextSearch" not in capability_names:
        print("  Enabling NoSQL Full Text Search capability...")
        capabilities_to_add.append(Capability(name="EnableNoSQLFullTextSearch"))
    
    if capabilities_to_add:
        # Add new capabilities to existing ones
        all_capabilities = list(current_capabilities) + capabilities_to_add
        
        update_params = DatabaseAccountUpdateParameters(
            capabilities=all_capabilities
        )
        
        poller = mgmt_client.database_accounts.begin_update(
            resource_group_name=resource_group,
            account_name=account_name,
            update_parameters=update_params
        )
        print("  Waiting for capabilities to be enabled (this may take a few minutes)...")
        poller.result()
        print("  ✓ Vector Search and Full Text Search capabilities enabled")
    else:
        print("  ✓ Vector Search and Full Text Search capabilities already enabled")


def create_database_and_container_via_management(credential, source_specs: List[dict[str, Any]]):
    """
    Create database and configured containers using Azure Resource Manager (control plane).
    Existing containers are never updated here; only missing containers are created.
    This uses Azure RBAC permissions instead of Cosmos DB data plane RBAC.
    """
    # Get configuration
    subscription_id = AZURE_SUBSCRIPTION_ID
    resource_group = COSMOS_RESOURCE_GROUP
    account_name = COSMOS_ACCOUNT_NAME or extract_account_name_from_endpoint(COSMOS_ENDPOINT)
    
    if not subscription_id or not resource_group:
        raise ValueError(
            "Azure subscription ID and resource group are required for management operations. "
            "Set cosmos.azure_subscription_id and cosmos.cosmos_resource_group in config.yaml."
        )
    
    print(f"  Account: {account_name}")
    print(f"  Resource Group: {resource_group}")
    print(f"  Subscription: {subscription_id}")
    
    # Enable Vector Search and Full Text Search capabilities if needed
    print("\n  Checking account capabilities...")
    enable_vector_search_capability(credential, subscription_id, resource_group, account_name)
    
    # Create management client
    mgmt_client = CosmosDBManagementClient(credential, subscription_id)
    
    # Create database
    print(f"\n  Creating database '{DATABASE_NAME}'...")
    try:
        db_params = SqlDatabaseCreateUpdateParameters(
            resource=SqlDatabaseResource(id=DATABASE_NAME)
        )
        poller = mgmt_client.sql_resources.begin_create_update_sql_database(
            resource_group_name=resource_group,
            account_name=account_name,
            database_name=DATABASE_NAME,
            create_update_sql_database_parameters=db_params
        )
        poller.result()  # Wait for completion
        print(f"  ✓ Database '{DATABASE_NAME}' ready")
    except Exception as e:
        if "Conflict" in str(e) or "already exists" in str(e).lower():
            print(f"  ✓ Database '{DATABASE_NAME}' already exists")
        else:
            print(f"  Error creating database: {e}")
            raise
    
    def container_exists(container_name: str) -> bool:
        try:
            mgmt_client.sql_resources.get_sql_container(
                resource_group_name=resource_group,
                account_name=account_name,
                database_name=DATABASE_NAME,
                container_name=container_name,
            )
            return True
        except HttpResponseError as e:
            if getattr(e, "status_code", None) == 404:
                return False
            raise

    container_specs = [
        (
            source.get("id", "unknown"),
            source.get("container_name"),
            source.get("partition_key_path"),
            source.get("embedding_field") or "e",
            source.get("indexing_policy"),
            source.get("full_text_policy"),
        )
        for source in source_specs
    ]
    processed_names = set()
    max_retries = 5
    retry_delay = 30  # seconds

    for source_id, container_name, partition_key_path, embedding_field, index_policy_cfg, fts_policy_cfg in container_specs:
        if not container_name or not str(container_name).strip():
            print(f"  ⚠ Skipping {source_id}: container name is not configured")
            continue
        if container_name in processed_names:
            continue
        processed_names.add(container_name)

        if container_exists(container_name):
            print(f"  ✓ Container '{container_name}' already exists - skipping")
            continue

        if not partition_key_path or not str(partition_key_path).strip():
            print(f"  ⚠ Skipping {source_id}: partition key path is not configured")
            continue

        if not index_policy_cfg:
            print(f"  ⚠ Skipping {source_id}: indexing policy is not configured")
            continue
        if not fts_policy_cfg:
            print(f"  ⚠ Skipping {source_id}: full-text policy is not configured")
            continue

        vector_path = _field_to_cosmos_json_path(str(embedding_field or "e"))

        indexing_policy = IndexingPolicy(
            indexing_mode=index_policy_cfg["indexingMode"],
            automatic=index_policy_cfg["automatic"],
            included_paths=[IncludedPath(path=p["path"]) for p in index_policy_cfg["includedPaths"]],
            excluded_paths=[ExcludedPath(path=p["path"]) for p in index_policy_cfg["excludedPaths"]],
            vector_indexes=[
                VectorIndex(path=vector_path, type=VectorIndexType(v["type"]))
                for v in index_policy_cfg["vectorIndexes"]
            ],
            full_text_indexes=[
                FullTextIndexPath(path=f["path"])
                for f in index_policy_cfg["fullTextIndexes"]
            ]
        )

        if VECTOR_EMBEDDING_POLICY:
            vector_embedding_policy = VectorEmbeddingPolicy(
                vector_embeddings=[
                    VectorEmbedding(
                        path=vector_path,
                        data_type=v["dataType"],
                        dimensions=v["dimensions"],
                        distance_function=v["distanceFunction"]
                    )
                    for v in VECTOR_EMBEDDING_POLICY["vectorEmbeddings"]
                ]
            )
        else:
            vector_embedding_policy = VectorEmbeddingPolicy(
                vector_embeddings=[
                    VectorEmbedding(
                        path=vector_path,
                        data_type="float32",
                        dimensions=EMBEDDING_DIMENSIONS,
                        distance_function="cosine",
                    )
                ]
            )

        full_text_policy = FullTextPolicy(
            default_language=fts_policy_cfg["defaultLanguage"],
            full_text_paths=[
                FullTextPath(path=p["path"], language=p["language"])
                for p in fts_policy_cfg["fullTextPaths"]
            ]
        )

        print(f"\n  Creating container '{container_name}'...")
        container_resource = SqlContainerResource(
            id=container_name,
            partition_key=ContainerPartitionKey(
                paths=[partition_key_path],
                kind="Hash"
            ),
            indexing_policy=indexing_policy,
            vector_embedding_policy=vector_embedding_policy,
            full_text_policy=full_text_policy
        )
        container_params = SqlContainerCreateUpdateParameters(resource=container_resource)
        tried_dedicated_autoscale = False

        for attempt in range(max_retries):
            try:
                # Safety guard: never run create/update against an existing container.
                if container_exists(container_name):
                    print(f"  ✓ Container '{container_name}' already exists - using as-is (no settings update)")
                    break

                poller = mgmt_client.sql_resources.begin_create_update_sql_container(
                    resource_group_name=resource_group,
                    account_name=account_name,
                    database_name=DATABASE_NAME,
                    container_name=container_name,
                    create_update_sql_container_parameters=container_params
                )
                poller.result()
                print(f"  ✓ Container '{container_name}' created")
                break
            except Exception as e:
                error_str = str(e)
                if (
                    "Vector Indexing is not supported for shared throughput offer" in error_str
                    and not tried_dedicated_autoscale
                ):
                    if THROUGHPUT_MODE == "autoscale":
                        throughput_opts = CreateUpdateOptions(
                            autoscale_settings=AutoscaleSettings(max_throughput=THROUGHPUT_VALUE)
                        )
                    else:
                        throughput_opts = CreateUpdateOptions(throughput=THROUGHPUT_VALUE)
                    print(
                        f"  ⚠ Shared throughput doesn't support vector indexing for this container; "
                        f"retrying with dedicated {THROUGHPUT_MODE} throughput ({THROUGHPUT_VALUE} RU)..."
                    )
                    container_params = SqlContainerCreateUpdateParameters(
                        resource=container_resource,
                        options=throughput_opts
                    )
                    tried_dedicated_autoscale = True
                    continue
                if "Conflict" in error_str or "already exists" in error_str.lower():
                    print(f"  ✓ Container '{container_name}' already exists - using as-is (no settings update)")
                    break
                if "capability has not been enabled" in error_str.lower() and attempt < max_retries - 1:
                    print(f"  ⏳ Waiting for capabilities to propagate (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(retry_delay)
                    continue
                print(f"  Error creating container '{container_name}': {e}")
                raise

def generate_embedding_text(doc: Dict[str, Any], text_fields: list[str]) -> str:
    """Build embedding text from configured fields; fallback to full JSON when empty."""
    text_parts: list[str] = []

    for field_name in text_fields:
        value = doc.get(field_name)
        if value is None or value == "":
            continue
        label = field_name.replace("_", " ").title()
        if isinstance(value, list):
            text_parts.append(f"{label}: {', '.join(str(v) for v in value)}")
        elif isinstance(value, dict):
            text_parts.append(f"{label}: {json.dumps(value, ensure_ascii=False)}")
        else:
            text_parts.append(f"{label}: {value}")

    if not text_parts:
        text_parts.append(json.dumps(doc, ensure_ascii=False))

    return "\n".join(text_parts)


async def generate_embeddings_batch(embed_client: AsyncAzureOpenAI | None, texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a batch of texts from a single configured endpoint/model."""
    if EMBED_ENDPOINT.rstrip("/").endswith("/api/embeddings"):
        batch_endpoint = EMBED_ENDPOINT.rstrip("/")
        if batch_endpoint.endswith("/api/embeddings"):
            batch_endpoint = batch_endpoint[: -len("/api/embeddings")] + "/api/embed"

        # Fast path for Ollama /api/embed with batched input
        try:
            async with httpx.AsyncClient(timeout=180) as client:
                response = await client.post(
                    batch_endpoint,
                    json={"model": EMBED_MODEL, "input": texts},
                )
                response.raise_for_status()
                payload = response.json()
            embeddings_payload = payload.get("embeddings")
            if isinstance(embeddings_payload, list) and len(embeddings_payload) == len(texts):
                return [_normalize_embedding(embedding) for embedding in embeddings_payload]
        except Exception:
            # Fallback to parallel single-item /api/embeddings requests
            pass

        semaphore = asyncio.Semaphore(min(max(1, LOCAL_EMBED_WORKERS), len(texts)))

        async def embed_single(text: str) -> List[float]:
            async with semaphore:
                async with httpx.AsyncClient(timeout=180) as client:
                    response = await client.post(
                        EMBED_ENDPOINT,
                        json={"model": EMBED_MODEL, "prompt": text},
                    )
                    response.raise_for_status()
                    payload = response.json()
                if "embedding" not in payload:
                    raise ValueError(f"Unexpected embedding response payload: {payload}")
                return _normalize_embedding(payload["embedding"])

        return await asyncio.gather(*(embed_single(text) for text in texts))

    if embed_client is None:
        raise ValueError("Embedding client is not initialized for Azure OpenAI endpoint")

    response = await embed_client.embeddings.create(
        input=texts,
        model=EMBED_MODEL,
        dimensions=EMBEDDING_DIMENSIONS,
    )
    return [_normalize_embedding(item.embedding) for item in response.data]


def find_all_input_files(root_path: str) -> List[str]:
    """Find all files in the directory tree (content is expected to be JSON)."""
    input_files = []
    for root, dirs, files in os.walk(root_path):
        # Skip hidden directories and the script itself
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for file in files:
            if not file.startswith('.'):
                input_files.append(os.path.join(root, file))
    
    return input_files


async def load_json_document(file_path: str) -> Optional[Dict[str, Any]]:
    """Load a JSON document from file."""
    try:
        def _load() -> Dict[str, Any]:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return await asyncio.to_thread(_load)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file {file_path}: {e}")
        return None
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None


def replace_document_id(doc: Dict[str, Any], relative_path: str) -> Dict[str, Any]:
    """Replace any existing id with a deterministic generated id based on relative path only."""
    digest = hashlib.sha1(relative_path.encode("utf-8")).hexdigest()
    generated_id = f"doc-{digest}"
    doc["id"] = generated_id
    return doc


def _load_jsonl_documents(jsonl_path: str) -> list[dict[str, Any]]:
    """Load documents from a JSONL file, adding _source_line and deterministic id."""
    documents: list[dict[str, Any]] = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            payload = line.strip()
            if not payload:
                continue
            try:
                record = json.loads(payload)
            except json.JSONDecodeError as exc:
                print(f"Error parsing JSONL line {line_num} in {jsonl_path}: {exc}")
                continue
            if not isinstance(record, dict):
                continue
            record['_source_line'] = line_num
            source_key = f"{os.path.basename(jsonl_path)}:{line_num}"
            record = replace_document_id(record, source_key)
            documents.append(record)
    return documents


async def upload_document(container, doc: Dict[str, Any]) -> bool:
    """Upload a single document to Cosmos DB."""
    try:
        await container.upsert_item(doc)
        return True
    except exceptions.CosmosHttpResponseError as e:
        doc_id = doc["id"] if "id" in doc else "unknown"
        print(f"Error uploading document {doc_id}: {e}")
        return False


async def upload_documents_batch(container, docs: List[Dict[str, Any]]) -> tuple[int, int]:
    """Upload a batch of docs in parallel. Returns (success_count, failed_count)."""
    if not docs:
        return 0, 0

    semaphore = asyncio.Semaphore(min(max(1, UPLOAD_WORKERS), len(docs)))

    async def _upload(item: Dict[str, Any]) -> bool:
        async with semaphore:
            return await upload_document(container, item)

    results = await asyncio.gather(*(_upload(item) for item in docs))
    success_count = sum(1 for result in results if result)
    return success_count, len(results) - success_count


async def main_async():
    """Main function to orchestrate the upload process."""
    parser = argparse.ArgumentParser(description="Upload JSON documents with embeddings to Cosmos DB")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config YAML file (default: config.yaml in script directory)",
    )
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else Path(__file__).parent / "config.yaml"
    load_config(config_path)

    print("=" * 60)
    print("Cosmos DB Document Upload with OpenAI Embeddings")
    print("=" * 60)
    
    # Validate configuration
    print("\n📋 Checking configuration...")
    
    if not COSMOS_ENDPOINT:
        print("❌ Error: Cosmos DB endpoint not configured.")
        print("   Please set cosmos.uri in config.yaml.")
        print("   Example: cosmos.uri: https://your-account.documents.azure.com:443/")
        return
    
    use_rbac_auth = CONFIG.get("cosmos", {}).get("use_rbac_auth", False)
    if not use_rbac_auth and not COSMOS_KEY:
        print("❌ Error: Cosmos DB key not configured.")
        print("   Please set cosmos.key in config.yaml or set cosmos.use_rbac_auth to true for RBAC auth.")
        return
    
    if not EMBED_ENDPOINT:
        print("❌ Error: Embedding endpoint not configured.")
        print("   Please set llm.embed_endpoint in config.yaml.")
        return

    if not EMBED_MODEL:
        print("❌ Error: Embedding model not configured.")
        print("   Please set embedding.embed_model in config.yaml.")
        return

    if not EMBED_ENDPOINT.rstrip("/").endswith("/api/embeddings") and not EMBED_API_KEY:
        print("❌ Error: Azure OpenAI API key not configured.")
        print("   Please set embedding.embed_api_key in config.yaml.")
        return

    print(f"✓ Embedding endpoint: {EMBED_ENDPOINT}")
    print(f"✓ Embedding model: {EMBED_MODEL}")
    print(f"✓ Embedding dimensions: {EMBEDDING_DIMENSIONS}")

    upload_targets = []
    for source in SOURCE_CONFIGS:
        source_name = source.get("id", "unknown")
        container_name = source.get("container_name")
        partition_key_path = source.get("partition_key_path")
        documents_root = source.get("documents_root")
        if not _has_value(container_name):
            print(f"⚠ Skipping {source_name} upload: container_name is empty or not set.")
            continue
        if not _has_value(partition_key_path):
            print(f"⚠ Skipping {source_name} upload: partition_key_path is empty or not set.")
            continue
        if not _has_value(documents_root):
            print(f"⚠ Skipping {source_name} upload: documents_root is empty or not set.")
            continue
        upload_targets.append(
            {
                "name": source_name,
                "documents_root": documents_root,
                "container_name": container_name,
                "partition_key_path": partition_key_path,
                "embedding_field": str(source.get("embedding_field") or "e").strip(),
                "embedding_text_fields": source.get("embedding_text_fields") or [],
                "indexing_policy": source.get("indexing_policy"),
                "full_text_policy": source.get("full_text_policy"),
            }
        )

    if not upload_targets:
        print("No upload targets configured. Nothing to upload.")
        return

    print("\n🔌 Initializing clients...")
    use_rbac_auth = CONFIG.get("cosmos", {}).get("use_rbac_auth", False)
    data_plane_credential = AsyncDefaultAzureCredential() if use_rbac_auth else None
    cosmos_client = get_cosmos_client(use_rbac_auth=use_rbac_auth, credential=data_plane_credential)

    # Initialize embedding client (single endpoint/model)
    embed_client = get_embedding_client()
    print("✓ Embedding client initialized")

    database = cosmos_client.get_database_client(DATABASE_NAME)
    print(f"✓ Connected to database '{DATABASE_NAME}'")

    async def container_exists_data_plane_async(container_name: str) -> bool:
        try:
            await database.get_container_client(container_name).read()
            return True
        except exceptions.CosmosResourceNotFoundError:
            return False

    missing_target_specs = [
        target
        for target in upload_targets
        if not await container_exists_data_plane_async(target["container_name"])
    ]

    if missing_target_specs:
        if AZURE_SUBSCRIPTION_ID and COSMOS_RESOURCE_GROUP:
            try:
                credential = SyncDefaultAzureCredential()
                print("\n📦 Missing containers detected; attempting management-plane create...")
                create_database_and_container_via_management(credential, missing_target_specs)
            except Exception as e:
                print(f"⚠ Container auto-create failed: {e}")
        else:
            print("\n⚠ Missing containers detected, but management settings are incomplete.")
            print("   Set cosmos.azure_subscription_id and cosmos.cosmos_resource_group to auto-create missing containers.")

    upload_targets_ready = []
    for target in upload_targets:
        if await container_exists_data_plane_async(target["container_name"]):
            upload_targets_ready.append(target)
        else:
            print(f"⚠ Skipping {target['name']} upload: container '{target['container_name']}' does not exist.")

    if not upload_targets_ready:
        print("No existing target containers available for upload. Nothing to do.")
        if embed_client is not None:
            await embed_client.close()
        await cosmos_client.close()
        if data_plane_credential is not None:
            await data_plane_credential.close()
        return

    batch_size = EMBEDDING_BATCH_SIZE
    successful_uploads = 0
    failed_uploads = 0
    total_files_seen = 0
    uploaded_containers = []
    total_parse_seconds = 0.0
    total_embed_seconds = 0.0
    total_upload_seconds = 0.0

    for target in upload_targets_ready:
        target_name = target["name"]
        documents_root = target["documents_root"]
        container_name = target["container_name"]
        embedding_field = str(target.get("embedding_field") or "e").strip()
        text_fields = target["embedding_text_fields"]

        # Load documents from source
        is_jsonl = documents_root.endswith('.jsonl') and os.path.isfile(documents_root)

        if is_jsonl:
            container = database.get_container_client(container_name)
            uploaded_containers.append(container_name)
            print(f"✓ Connected to container '{container_name}' for {target_name} upload")
            print(f"\n🔍 Loading JSONL ({target_name}): {documents_root}")
            parse_start = time.perf_counter()
            all_docs = await asyncio.to_thread(_load_jsonl_documents, documents_root)
            total_parse_seconds += (time.perf_counter() - parse_start)
            total_files_seen += len(all_docs)
            print(f"✓ Loaded {len(all_docs)} documents from JSONL")
        elif os.path.isdir(documents_root):
            container = database.get_container_client(container_name)
            uploaded_containers.append(container_name)
            print(f"✓ Connected to container '{container_name}' for {target_name} upload")
            print(f"\n🔍 Scanning for files in ({target_name}): {documents_root}")
            input_files = await asyncio.to_thread(find_all_input_files, documents_root)
            total_files_seen += len(input_files)
            print(f"✓ Found {len(input_files)} files")
            if not input_files:
                print(f"⚠ No files found for {target_name}. Skipping.")
                continue
            all_docs = []
            for file_path in tqdm(input_files, desc=f"Loading {target_name} files"):
                parse_start = time.perf_counter()
                doc = await load_json_document(file_path)
                if doc is None or not isinstance(doc, dict):
                    failed_uploads += 1
                    total_parse_seconds += (time.perf_counter() - parse_start)
                    continue
                relative_path = os.path.relpath(file_path, documents_root)
                doc['_source_file'] = relative_path
                doc = replace_document_id(doc, relative_path)
                all_docs.append(doc)
                total_parse_seconds += (time.perf_counter() - parse_start)
        else:
            print(f"⚠ Skipping {target_name} upload: path does not exist: {documents_root}")
            continue

        if not all_docs:
            print(f"⚠ No documents loaded for {target_name}. Skipping.")
            continue

        print(f"\n📄 Processing {target_name} documents (batch size: {batch_size})...")

        for i in range(0, len(all_docs), batch_size):
            batch_docs = all_docs[i:i + batch_size]
            batch_texts = [generate_embedding_text(doc, text_fields) for doc in batch_docs]
            try:
                embed_start = time.perf_counter()
                embeddings = await generate_embeddings_batch(embed_client, batch_texts)
                total_embed_seconds += (time.perf_counter() - embed_start)
                for item_doc, embedding in zip(batch_docs, embeddings):
                    item_doc[embedding_field] = embedding

                upload_start = time.perf_counter()
                success_count, failed_count = await upload_documents_batch(container, batch_docs)
                total_upload_seconds += (time.perf_counter() - upload_start)
                successful_uploads += success_count
                failed_uploads += failed_count
            except Exception as e:
                print(f"\nError processing {target_name} batch: {e}")
                failed_uploads += len(batch_docs)

    # Summary
    print("\n" + "=" * 60)
    print("📊 Upload Summary")
    print("=" * 60)
    print(f"✓ Successfully uploaded: {successful_uploads}")
    print(f"✗ Failed: {failed_uploads}")
    print(f"📁 Total files scanned: {total_files_seen}")
    print(f"🗄️  Database: {DATABASE_NAME}")
    print(f"📦 Containers touched: {', '.join(uploaded_containers) if uploaded_containers else 'none'}")
    print(f"⏱ Parse/build text time: {total_parse_seconds:.1f}s")
    print(f"⏱ Embedding time: {total_embed_seconds:.1f}s")
    print(f"⏱ Upload time: {total_upload_seconds:.1f}s")
    if successful_uploads > 0:
        print(f"⏱ Avg embed+upload per uploaded doc: {(total_embed_seconds + total_upload_seconds) / successful_uploads:.3f}s")
    print("=" * 60)

    if embed_client is not None:
        await embed_client.close()
    await cosmos_client.close()
    if data_plane_credential is not None:
        await data_plane_credential.close()


if __name__ == "__main__":
    asyncio.run(main_async())
