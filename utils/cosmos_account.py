"""Cosmos DB account/database/container management-plane operations.

Extracted from cosmos_db_upload.py so the upload script's own module stays
focused on the data-plane document pipeline. These functions take their
Azure identifiers and policy settings as explicit parameters rather than
reading cosmos_db_upload's module-level config globals directly, so they
don't depend on cosmos_db_upload.load_config() having already run against
this exact module instance.
"""

import re
import time
from typing import Any, List

from azure.core.exceptions import HttpResponseError
from azure.mgmt.cosmosdb import CosmosDBManagementClient
from azure.mgmt.cosmosdb.models import (
    AutoscaleSettings,
    Capability,
    ContainerPartitionKey,
    CreateUpdateOptions,
    DatabaseAccountUpdateParameters,
    ExcludedPath,
    FullTextIndexPath,
    FullTextPath,
    FullTextPolicy,
    IncludedPath,
    IndexingPolicy,
    SqlContainerCreateUpdateParameters,
    SqlContainerResource,
    SqlDatabaseCreateUpdateParameters,
    SqlDatabaseResource,
    VectorEmbedding,
    VectorEmbeddingPolicy,
    VectorIndex,
    VectorIndexType,
)


def extract_account_name_from_endpoint(endpoint: str) -> str:
    """Extract the Cosmos DB account name from the endpoint URL."""
    # https://myaccount.documents.azure.com:443/ -> account name is myaccount
    match = re.match(r'https://([^.]+)\.documents\.azure\.com', endpoint)
    if match:
        return match.group(1)
    raise ValueError(f"Could not extract account name from endpoint: {endpoint}")


def ensure_cosmos_account_exists(credential, subscription_id: str, resource_group: str, account_name: str) -> None:
    """Verify the target Cosmos DB account exists before any other management
    or data-plane operation runs against it.

    This deliberately does not create a missing account: account-level
    choices (region, consistency policy, capacity mode, network rules)
    aren't part of this project's config schema, and guessing them would
    risk silently provisioning an account with defaults the operator never
    chose. Raise a clear, actionable error instead.
    """
    mgmt_client = CosmosDBManagementClient(credential, subscription_id)
    try:
        mgmt_client.database_accounts.get(resource_group, account_name)
    except HttpResponseError as e:
        if getattr(e, "status_code", None) == 404:
            raise ValueError(
                f"Cosmos DB account '{account_name}' was not found in resource group "
                f"'{resource_group}' (subscription '{subscription_id}'). Create the account "
                "first (e.g. `az cosmosdb create --name <account> --resource-group <group> "
                "--kind GlobalDocumentDB`), or check cosmos.cosmos_account_name / "
                "cosmos.cosmos_resource_group / cosmos.azure_subscription_id in your config."
            ) from e
        raise


def enable_vector_search_capability(credential, subscription_id: str, resource_group: str, account_name: str):
    """
    Enable Vector Search capability on the Cosmos DB account if not already enabled.
    This is required before creating containers with vector indexing.
    """
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


def _field_to_cosmos_json_path(field_name: str) -> str:
    normalized = str(field_name or "e").strip()
    segments = [segment for segment in normalized.split(".") if segment]
    if not segments:
        return "/e"
    return "/" + "/".join(segments)


def _build_dedicated_throughput_options(throughput_mode: str, throughput_value: int) -> CreateUpdateOptions:
    """Create dedicated throughput options for new vector-enabled containers."""
    if throughput_mode == "autoscale":
        return CreateUpdateOptions(
            autoscale_settings=AutoscaleSettings(max_throughput=throughput_value)
        )
    return CreateUpdateOptions(throughput=throughput_value)


def create_database_and_container_via_management(
    credential,
    source_specs: List[dict[str, Any]],
    *,
    subscription_id: str,
    resource_group: str,
    account_name: str,
    database_name: str,
    vector_embedding_policy: dict[str, Any] | None,
    embedding_dimensions: int,
    throughput_mode: str,
    throughput_value: int,
) -> None:
    """
    Create database and configured containers using Azure Resource Manager (control plane).
    Existing containers are never updated here; only missing containers are created.
    This uses Azure RBAC permissions instead of Cosmos DB data plane RBAC.
    """
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
    print(f"\n  Creating database '{database_name}'...")
    try:
        db_params = SqlDatabaseCreateUpdateParameters(
            resource=SqlDatabaseResource(id=database_name)
        )
        poller = mgmt_client.sql_resources.begin_create_update_sql_database(
            resource_group_name=resource_group,
            account_name=account_name,
            database_name=database_name,
            create_update_sql_database_parameters=db_params
        )
        poller.result()  # Wait for completion
        print(f"  ✓ Database '{database_name}' ready")
    except Exception as e:
        if "Conflict" in str(e) or "already exists" in str(e).lower():
            print(f"  ✓ Database '{database_name}' already exists")
        else:
            print(f"  Error creating database: {e}")
            raise

    def container_exists(container_name: str) -> bool:
        try:
            mgmt_client.sql_resources.get_sql_container(
                resource_group_name=resource_group,
                account_name=account_name,
                database_name=database_name,
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

        if vector_embedding_policy:
            embedding_policy = VectorEmbeddingPolicy(
                vector_embeddings=[
                    VectorEmbedding(
                        path=vector_path,
                        data_type=v["dataType"],
                        dimensions=v["dimensions"],
                        distance_function=v["distanceFunction"]
                    )
                    for v in vector_embedding_policy["vectorEmbeddings"]
                ]
            )
        else:
            embedding_policy = VectorEmbeddingPolicy(
                vector_embeddings=[
                    VectorEmbedding(
                        path=vector_path,
                        data_type="float32",
                        dimensions=embedding_dimensions,
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

        print(f"\n  Creating container '{container_name}' with dedicated {throughput_mode} throughput...")
        container_resource = SqlContainerResource(
            id=container_name,
            partition_key=ContainerPartitionKey(
                paths=[partition_key_path],
                kind="Hash"
            ),
            indexing_policy=indexing_policy,
            vector_embedding_policy=embedding_policy,
            full_text_policy=full_text_policy
        )
        container_params = SqlContainerCreateUpdateParameters(
            resource=container_resource,
            options=_build_dedicated_throughput_options(throughput_mode, throughput_value)
        )

        for attempt in range(max_retries):
            try:
                # Safety guard: never run create/update against an existing container.
                if container_exists(container_name):
                    print(f"  ✓ Container '{container_name}' already exists - using as-is (no settings update)")
                    break

                poller = mgmt_client.sql_resources.begin_create_update_sql_container(
                    resource_group_name=resource_group,
                    account_name=account_name,
                    database_name=database_name,
                    container_name=container_name,
                    create_update_sql_container_parameters=container_params
                )
                poller.result()
                print(f"  ✓ Container '{container_name}' created")
                break
            except Exception as e:
                error_str = str(e)
                if "Conflict" in error_str or "already exists" in error_str.lower():
                    print(f"  ✓ Container '{container_name}' already exists - using as-is (no settings update)")
                    break
                if "capability has not been enabled" in error_str.lower() and attempt < max_retries - 1:
                    print(f"  ⏳ Waiting for capabilities to propagate (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(retry_delay)
                    continue
                print(f"  Error creating container '{container_name}': {e}")
                raise
