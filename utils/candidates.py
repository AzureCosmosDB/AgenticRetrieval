"""Lightweight Cosmos candidates and bounded, post-merge point reads."""

import asyncio
import json
import logging
from typing import Any
from urllib.parse import urlsplit

from azure.cosmos import exceptions

# Point reads that fail with these statuses (document deleted since the query,
# or throttling that outlasted the SDK's retries) skip the candidate instead of
# failing the whole search.
_SKIPPABLE_READ_STATUSES = (404, 429)


class ConfigurationError(ValueError):
    """Invalid configuration (or container setup) that makes retrieval impossible."""


def validate_partition_key_path(partition_key_path: Any, source_id: str) -> str:
    """Return a usable partition key path or raise a ConfigurationError that says how to fix it."""
    path = partition_key_path.strip() if isinstance(partition_key_path, str) else ""
    segments = path.split("/")[1:]
    if not path.startswith("/") or not segments or any(
        not segment or segment in {"[]", "*", "?"} for segment in segments
    ):
        raise ConfigurationError(
            f"cosmos.sources[{source_id!r}].partition_key_path is missing or invalid "
            f"({partition_key_path!r}). Set it to the container's partition key path "
            "(for example '/pk'), the same value cosmos_db_upload.py uses to create the "
            "container. Containers without a partition key are not supported."
        )
    return path


async def resolve_partition_key_path(container: Any, configured: Any, source_id: str) -> str:
    """Return the container's real partition key path, falling back to the configured one.

    The container definition is authoritative. The configured value is used only
    when the definition cannot be read, and must then be valid. Containers without
    a partition key or with hierarchical keys are rejected.
    """
    try:
        properties = await container.read()
        partition_key = properties.get("partitionKey")
    except Exception as error:
        logging.warning(
            "Could not read the partition key of container %r (%s); using configured "
            "cosmos.sources[%r].partition_key_path.",
            getattr(container, "id", "?"), str(error)[:200], source_id,
        )
        return validate_partition_key_path(configured, source_id)
    if partition_key is None:
        return validate_partition_key_path(configured, source_id)

    paths = partition_key.get("paths") or []
    if len(paths) != 1:
        problem = (
            "has no partition key (legacy partitionless containers are not supported)"
            if not paths else f"uses a hierarchical partition key {paths} (not supported)"
        )
        raise ConfigurationError(
            f"Container {getattr(container, 'id', '?')!r} for source {source_id!r} {problem}."
        )
    actual = validate_partition_key_path(paths[0], source_id)
    if isinstance(configured, str) and configured.strip() and configured.strip() != actual:
        logging.warning(
            "cosmos.sources[%r].partition_key_path is %r but container %r uses %r; using the "
            "container's value. Update the config so uploads and retrieval agree.",
            source_id, configured.strip(), getattr(container, "id", "?"), actual,
        )
    return actual


def candidate_projection(partition_key_path: str) -> str:
    try:
        path = validate_partition_key_path(partition_key_path, "?")
    except ConfigurationError:
        raise ValueError(f"Invalid partition key path: {partition_key_path!r}") from None
    expression = "c" + "".join(f"[{json.dumps(segment)}]" for segment in path.split("/")[1:])
    return f"c.id, {expression} AS _partition_key"


def candidate_identity(candidate: dict[str, Any]) -> tuple[str, str]:
    document_id = candidate.get("id")
    if not isinstance(document_id, str) or not document_id:
        raise ValueError(f"Candidate has no valid document ID: {candidate!r}")
    partition_key = candidate.get("_partition_key", {})
    return document_id, json.dumps(partition_key, sort_keys=True, allow_nan=False)


def container_identity(container: Any) -> tuple[str, str, str] | tuple[str, int]:
    """Identify SDK proxies by account and database/container link, not object identity."""
    connection = getattr(container, "client_connection", None)
    endpoint = getattr(connection, "url_connection", None)
    link = getattr(container, "container_link", None)
    if isinstance(endpoint, str) and isinstance(link, str) and endpoint and link:
        url = urlsplit(endpoint)
        port = url.port
        authority = (url.hostname or "").lower()
        if port is not None and (url.scheme.lower(), port) not in {("https", 443), ("http", 80)}:
            authority += f":{port}"
        account = f"{url.scheme.lower()}://{authority}{url.path.rstrip('/')}"
        return "cosmos", account, link.strip("/")
    # Non-SDK adapters without physical-container metadata must remain distinct.
    return "object", id(container)


def extract_partition_key(document: dict[str, Any], partition_key_path: str) -> Any:
    """Distinguish an undefined key (including missing parents) from explicit JSON null."""
    from azure.cosmos.partition_key import NonePartitionKeyValue

    path = validate_partition_key_path(partition_key_path, "upload")
    value: Any = document
    for segment in path.split("/")[1:]:
        if not isinstance(value, dict) or segment not in value:
            return NonePartitionKeyValue
        value = value[segment]
    return value


async def fetch_candidate_results(
    references: list[tuple[Any, dict[str, Any]]],
) -> list[dict[str, Any] | None]:
    """Point-read each unique candidate; the result stays aligned with the de-duplicated input.

    A read that fails with HTTP 404 or 429 yields None and is logged, so one deleted
    or throttled document does not abort the search. Any other error propagates.
    """
    unique = []
    seen = set()
    for container, candidate in references:
        key = (container_identity(container), candidate_identity(candidate))
        if key not in seen:
            seen.add(key)
            unique.append((container, candidate))
    semaphore = asyncio.Semaphore(16)
    skipped: list[tuple[str, int]] = []

    async def fetch(container: Any, candidate: dict[str, Any]) -> dict[str, Any] | None:
        if "_partition_key" in candidate:
            partition_key = candidate["_partition_key"]
        else:
            from azure.cosmos.partition_key import NonePartitionKeyValue

            partition_key = NonePartitionKeyValue
        async with semaphore:
            try:
                document = dict(await container.read_item(
                    item=candidate["id"], partition_key=partition_key,
                ))
            except exceptions.CosmosHttpResponseError as error:
                if error.status_code not in _SKIPPABLE_READ_STATUSES:
                    raise
                skipped.append((candidate["id"], error.status_code))
                return None
        if "_score" in candidate:
            document["_score"] = candidate["_score"]
        return document

    results = await asyncio.gather(*(fetch(container, candidate) for container, candidate in unique))
    if skipped:
        sample = ", ".join(f"{doc_id!r} (HTTP {status})" for doc_id, status in skipped[:3])
        logging.warning(
            "Skipped %d of %d candidates whose point read failed: %s%s",
            len(skipped), len(unique), sample, ", ..." if len(skipped) > 3 else "",
        )
    return results


async def fetch_candidate_documents(
    references: list[tuple[Any, dict[str, Any]]],
) -> list[dict[str, Any]]:
    return [doc for doc in await fetch_candidate_results(references) if doc is not None]
