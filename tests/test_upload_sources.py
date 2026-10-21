import asyncio
import hashlib
import json
import sys
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from azure.identity import CredentialUnavailableError
from azure.cosmos.partition_key import NonePartitionKeyValue
from openai import APIStatusError, APITimeoutError

import cosmos_db_upload as uploader


@pytest.mark.parametrize("operation", ["upload", "exists"])
def test_cli_token_timeout_is_retried(monkeypatch, operation, capsys):
    container = AsyncMock()
    call = container.upsert_item if operation == "upload" else container.read_item
    call.side_effect = [CredentialUnavailableError("Timed out waiting for Azure CLI"), {}]
    sleep = AsyncMock()
    monkeypatch.setattr(uploader.asyncio, "sleep", sleep)
    document = {"id": "doc-1", "upc": "00123"}

    result = asyncio.run(
        uploader.upload_document(container, document)
        if operation == "upload"
        else uploader._doc_exists(container, document, "/upc")
    )

    assert result is True
    assert call.await_count == 2
    sleep.assert_awaited_once_with(1)
    assert "retrying" in capsys.readouterr().out


@pytest.mark.parametrize("operation", ["upload", "exists"])
@pytest.mark.parametrize(
    "message", ["Timed out waiting for Azure CLI", "Please run 'az login'"]
)
def test_cli_auth_failure_is_surfaced(monkeypatch, operation, message):
    container = AsyncMock()
    call = container.upsert_item if operation == "upload" else container.read_item
    call.side_effect = CredentialUnavailableError(message)
    monkeypatch.setattr(uploader.asyncio, "sleep", AsyncMock())
    document = {"id": "doc-1", "upc": "00123"}

    with pytest.raises(CredentialUnavailableError, match=message):
        asyncio.run(
            uploader.upload_document(container, document)
            if operation == "upload"
            else uploader._doc_exists(container, document, "/upc")
        )

    assert call.await_count == (5 if message.startswith("Timed out") else 1)


def test_cosmos_cli_credential_allows_slower_startup(monkeypatch):
    credential_factory = Mock()
    monkeypatch.setattr(uploader, "AsyncAzureCliCredential", credential_factory)
    monkeypatch.setattr(uploader, "CosmosClient", Mock())
    monkeypatch.setattr(uploader, "COSMOS_ENDPOINT", "https://example.invalid")
    monkeypatch.setattr(uploader, "CONFIG", {"cosmos": {"tenant_id": "test-tenant"}})

    uploader.get_cosmos_client(use_rbac_auth=True)

    credential_factory.assert_called_once_with(tenant_id="test-tenant", process_timeout=60)


@pytest.mark.parametrize("extension", [".jsonl", ".ndjson", ".jsonlines", ".NDJSON"])
def test_newline_delimited_file_extensions(tmp_path, extension):
    path = tmp_path / f"products{extension}"
    records = [{"upc": "00123", "product_id": 1}, {"upc": "00456", "product_id": 2}]
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    assert uploader._is_jsonl_file(str(path))
    loaded = uploader._load_jsonl_documents(str(path))
    assert [(doc["upc"], doc["product_id"]) for doc in loaded] == [
        ("00123", 1),
        ("00456", 2),
    ]
    assert len({doc["id"] for doc in loaded}) == 2
    assert loaded == uploader._load_jsonl_documents(str(path))


def test_newline_delimited_file_detection_rejects_other_inputs(tmp_path):
    json_path = tmp_path / "product.json"
    json_path.write_text("{}", encoding="utf-8")
    directory = tmp_path / "folder.ndjson"
    directory.mkdir()

    assert not uploader._is_jsonl_file(str(json_path))
    assert not uploader._is_jsonl_file(str(directory))
    assert not uploader._is_jsonl_file(str(tmp_path / "missing.ndjson"))


def test_directory_jsonl_ids_use_relative_paths_and_preserve_single_file_ids(tmp_path):
    documents = []
    for folder in ("a", "b"):
        path = tmp_path / folder / "part.ndjson"
        path.parent.mkdir()
        path.write_text('{"pk":"same"}\n', encoding="utf-8")
        relative = path.relative_to(tmp_path)
        docs = uploader._load_jsonl_documents(str(path), source_relative_path=str(relative))
        documents.extend(docs)
        assert docs == uploader._load_jsonl_documents(
            str(path), source_relative_path=relative.as_posix(),
        )
        assert docs == uploader._load_jsonl_documents(
            str(path), source_relative_path=str(relative).replace("/", "\\"),
        )
        assert docs[0]["id"] == "doc-" + hashlib.sha1(
            f"{folder}/part.ndjson:1".encode("utf-8")
        ).hexdigest()
        single = uploader._load_jsonl_documents(str(path))
        assert single[0]["id"] == "doc-" + hashlib.sha1(b"part.ndjson:1").hexdigest()
    assert documents[0]["id"] != documents[1]["id"]


@pytest.mark.parametrize("source_id", [None, "on-pack", "attributes"])
def test_upload_source_selection(monkeypatch, capsys, source_id):
    sources = [
        {"id": "on-pack", "container_name": "", "documents_root": "on-pack.ndjson"},
        {"id": "attributes", "container_name": "", "documents_root": "attributes.ndjson"},
    ]
    monkeypatch.setattr(uploader, "load_config", lambda path: None)
    monkeypatch.setattr(uploader, "SOURCE_CONFIGS", sources)
    monkeypatch.setattr(uploader, "CONFIG", {"cosmos": {"use_rbac_auth": True}})
    monkeypatch.setattr(uploader, "COSMOS_ENDPOINT", "https://example.invalid")
    monkeypatch.setattr(uploader, "EMBED_ENDPOINT", "https://example.invalid")
    monkeypatch.setattr(uploader, "EMBED_MODEL", "test-model")
    monkeypatch.setattr(uploader, "EMBED_USE_RBAC", True)
    args = ["cosmos_db_upload.py"]
    if source_id is not None:
        args.extend(["--source", source_id])
    monkeypatch.setattr(sys, "argv", args)

    asyncio.run(uploader.main_async())

    output = capsys.readouterr().out
    for source in sources:
        assert (f"Skipping {source['id']} upload:" in output) == (
            source_id is None or source_id == source["id"]
        )
    assert "No upload targets configured" in output


def test_unknown_upload_source_fails_before_connecting(monkeypatch, capsys):
    monkeypatch.setattr(uploader, "load_config", lambda path: None)
    monkeypatch.setattr(uploader, "SOURCE_CONFIGS", [{"id": "on-pack"}])
    monkeypatch.setattr(sys, "argv", ["cosmos_db_upload.py", "--source", "missing"])

    with pytest.raises(SystemExit) as exc:
        asyncio.run(uploader.main_async())

    assert exc.value.code == 2
    assert "Unknown source id: 'missing'" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("partition_path", "partition_value"),
    [("/upc", "00123"), ("/id", "doc-1"), ("/identity/upc", "00123")],
)
def test_existing_documents_use_configured_partition_key(partition_path, partition_value):
    container = AsyncMock()
    existing = {"id": "doc-1", "upc": "00123", "identity": {"upc": "00123"}}

    remaining = asyncio.run(
        uploader._filter_existing_docs(container, [existing], partition_path)
    )

    assert remaining == []
    container.read_item.assert_awaited_once_with(item="doc-1", partition_key=partition_value)


@pytest.mark.parametrize(
    ("path", "fields", "expected"),
    [
        ("/pk", {}, NonePartitionKeyValue),
        ("/parent/pk", {}, NonePartitionKeyValue),
        ("/parent/pk", {"parent": {}}, NonePartitionKeyValue),
        ("/parent/pk", {"parent": None}, NonePartitionKeyValue),
        ("/parent/pk", {"parent": "not-an-object"}, NonePartitionKeyValue),
        ("/pk", {"pk": None}, None),
        ("/parent/pk", {"parent": {"pk": None}}, None),
        ("/pk", {"pk": False}, False),
        ("/pk", {"pk": 0}, 0),
    ],
)
@pytest.mark.parametrize("exists", [False, True])
def test_existence_checks_preserve_undefined_null_and_falsy_keys(path, fields, expected, exists):
    container = AsyncMock()
    if not exists:
        container.read_item.side_effect = uploader.exceptions.CosmosResourceNotFoundError(
            status_code=404, message="Not found",
        )
    document = {"id": "doc-1", **fields}
    remaining = asyncio.run(uploader._filter_existing_docs(container, [document], path))
    assert remaining == ([] if exists else [document])
    container.read_item.assert_awaited_once_with(item="doc-1", partition_key=expected)


@pytest.mark.parametrize("path", ["", "pk", "/", "/parent//pk", "/pk/*"])
def test_invalid_upload_partition_path_fails_before_read(path):
    container = AsyncMock()
    with pytest.raises(ValueError, match="partition_key_path"):
        asyncio.run(uploader._doc_exists(container, {"id": "doc-1"}, path))
    container.read_item.assert_not_awaited()


def test_missing_document_is_not_skipped():
    container = AsyncMock()
    container.read_item.side_effect = uploader.exceptions.CosmosResourceNotFoundError(
        status_code=404, message="Not found"
    )
    document = {"id": "doc-1", "upc": "00123"}

    assert asyncio.run(uploader._filter_existing_docs(container, [document], "/upc")) == [
        document
    ]


def test_existence_check_does_not_hide_permission_errors():
    container = AsyncMock()
    container.read_item.side_effect = uploader.exceptions.CosmosHttpResponseError(
        status_code=403, message="Forbidden"
    )

    with pytest.raises(uploader.exceptions.CosmosHttpResponseError):
        asyncio.run(
            uploader._filter_existing_docs(
                container, [{"id": "doc-1", "upc": "00123"}], "/upc"
            )
        )


def test_embedding_timeout_is_retried(monkeypatch, capsys):
    timeout = APITimeoutError(request=httpx.Request("POST", "https://example.invalid"))
    embed = AsyncMock(side_effect=[timeout, [[0.1, 0.2]]])
    upload = AsyncMock(return_value=(1, 0))
    sleep = AsyncMock()
    monkeypatch.setattr(uploader, "generate_embeddings_batch", embed)
    monkeypatch.setattr(uploader, "upload_documents_batch", upload)
    monkeypatch.setattr(uploader.asyncio, "sleep", sleep)
    document = {"id": "doc-1", "upc": "00123"}

    result = asyncio.run(
        uploader._embed_and_upload_batch(
            [document], None, AsyncMock(), "e", ["upc"], skip_existing=False
        )
    )

    assert result[:2] == (1, 0)
    assert embed.await_count == 2
    sleep.assert_awaited_once_with(2)
    assert document["e"] == [0.1, 0.2]
    upload.assert_awaited_once()
    assert "retry" in capsys.readouterr().out.lower()


def test_embedding_timeout_exhaustion_reports_failed_batch(monkeypatch):
    timeout = APITimeoutError(request=httpx.Request("POST", "https://example.invalid"))
    embed = AsyncMock(side_effect=timeout)
    upload = AsyncMock()
    monkeypatch.setattr(uploader, "generate_embeddings_batch", embed)
    monkeypatch.setattr(uploader, "upload_documents_batch", upload)
    monkeypatch.setattr(uploader.asyncio, "sleep", AsyncMock())

    result = asyncio.run(
        uploader._embed_and_upload_batch(
            [{"id": "doc-1"}], None, AsyncMock(), "e", ["id"],
            max_embed_retries=2, skip_existing=False,
        )
    )

    assert result[:2] == (0, 1)
    assert embed.await_count == 2
    upload.assert_not_awaited()


def test_existing_batch_skips_embeddings_with_upc_partition(monkeypatch):
    container = AsyncMock()
    embed = AsyncMock()
    monkeypatch.setattr(uploader, "generate_embeddings_batch", embed)

    result = asyncio.run(
        uploader._embed_and_upload_batch(
            [{"id": "doc-1", "upc": "00123"}], None, container, "e", ["upc"],
            partition_key_path="/upc",
        )
    )

    assert result == (0, 0, 0.0, 0.0)
    container.read_item.assert_awaited_once_with(item="doc-1", partition_key="00123")
    embed.assert_not_awaited()


def test_missing_partition_key_document_reaches_upload(monkeypatch):
    container = AsyncMock()
    container.read_item.side_effect = uploader.exceptions.CosmosResourceNotFoundError(
        status_code=404, message="Not found",
    )
    monkeypatch.setattr(uploader, "generate_embeddings_batch", AsyncMock(return_value=[[0.1]]))
    document = {"id": "doc-1"}
    result = asyncio.run(uploader._embed_and_upload_batch(
        [document], None, container, "e", ["id"], partition_key_path="/upc",
    ))
    assert result[:2] == (1, 0)
    container.read_item.assert_awaited_once_with(
        item="doc-1", partition_key=NonePartitionKeyValue,
    )
    container.upsert_item.assert_awaited_once_with({"id": "doc-1", "e": [0.1]})


def test_resume_retries_only_missing_documents(monkeypatch):
    existing = {"id": "doc-existing", "upc": "00123"}
    missing = {"id": "doc-missing", "upc": "00456"}
    container = AsyncMock()
    container.read_item.side_effect = [
        existing,
        uploader.exceptions.CosmosResourceNotFoundError(status_code=404, message="Not found"),
    ]
    timeout = APITimeoutError(request=httpx.Request("POST", "https://example.invalid"))
    embed = AsyncMock(side_effect=[timeout, [[0.1, 0.2]]])
    upload = AsyncMock(return_value=(1, 0))
    monkeypatch.setattr(uploader, "generate_embeddings_batch", embed)
    monkeypatch.setattr(uploader, "upload_documents_batch", upload)
    monkeypatch.setattr(uploader.asyncio, "sleep", AsyncMock())

    result = asyncio.run(
        uploader._embed_and_upload_batch(
            [existing, missing], None, container, "e", ["upc"],
            partition_key_path="/upc",
        )
    )

    assert result[:2] == (1, 0)
    assert embed.await_count == 2
    for call in embed.await_args_list:
        assert call.args[1] == [uploader.generate_embedding_text(missing, ["upc"])]
    upload.assert_awaited_once_with(container, [missing])
    assert "e" not in existing


@pytest.mark.parametrize("status", [400, 401, 403, 404, 408, 409, 429, 500, 503])
def test_embedding_http_status_retry_policy(monkeypatch, status):
    response = httpx.Response(
        status, request=httpx.Request("POST", "https://example.invalid")
    )
    error = APIStatusError("Embedding request failed", response=response, body=None)
    embed = AsyncMock(side_effect=[error, [[0.1]]])
    upload = AsyncMock(return_value=(1, 0))
    monkeypatch.setattr(uploader, "generate_embeddings_batch", embed)
    monkeypatch.setattr(uploader, "upload_documents_batch", upload)
    monkeypatch.setattr(uploader.asyncio, "sleep", AsyncMock())

    result = asyncio.run(
        uploader._embed_and_upload_batch(
            [{"id": "doc-1"}], None, AsyncMock(), "e", ["id"], skip_existing=False,
        )
    )

    retryable = status in (408, 409, 429) or status >= 500
    assert embed.await_count == (2 if retryable else 1)
    assert result[:2] == ((1, 0) if retryable else (0, 1))


@pytest.mark.parametrize("failed", [0, 1])
@pytest.mark.parametrize("directory", [False, True])
def test_upload_exit_status_and_partition_key_wiring(monkeypatch, tmp_path, failed, directory):
    path = tmp_path / "products.ndjson"
    if directory:
        path = tmp_path / "nested" / "products.ndjson"
        path.parent.mkdir()
    path.write_text('{"upc":"00123","product_id":1}\n', encoding="utf-8")
    source = {
        "id": "on-pack",
        "container_name": "on-pack",
        "documents_root": str(tmp_path if directory else path),
        "partition_key_path": "/upc",
        "embedding_text_fields": ["upc"],
    }
    container = AsyncMock()
    database = Mock()
    database.get_container_client.return_value = container
    cosmos = Mock()
    cosmos.get_database_client.return_value = database
    cosmos.close = AsyncMock()
    embed_client = AsyncMock()
    batch = AsyncMock(return_value=(1 - failed, failed, 0.0, 0.0))
    monkeypatch.setattr(uploader, "load_config", lambda path: None)
    monkeypatch.setattr(uploader, "CONFIG", {"cosmos": {"use_rbac_auth": False}})
    monkeypatch.setattr(uploader, "SOURCE_CONFIGS", [source])
    monkeypatch.setattr(uploader, "COSMOS_ENDPOINT", "https://example.invalid")
    monkeypatch.setattr(uploader, "COSMOS_KEY", "test-key")
    monkeypatch.setattr(uploader, "EMBED_ENDPOINT", "https://example.invalid")
    monkeypatch.setattr(uploader, "EMBED_MODEL", "test-model")
    monkeypatch.setattr(uploader, "EMBED_API_FORMAT", "openai")
    monkeypatch.setattr(uploader, "AZURE_SUBSCRIPTION_ID", "")
    monkeypatch.setattr(uploader, "get_cosmos_client", Mock(return_value=cosmos))
    monkeypatch.setattr(uploader, "get_embedding_client", Mock(return_value=embed_client))
    monkeypatch.setattr(uploader, "_embed_and_upload_batch", batch)
    monkeypatch.setattr(sys, "argv", ["cosmos_db_upload.py", "--source", "on-pack"])

    if failed:
        with pytest.raises(SystemExit) as exc:
            asyncio.run(uploader.main_async())
        assert exc.value.code == 1
    else:
        asyncio.run(uploader.main_async())

    batch.assert_awaited_once()
    assert batch.await_args.kwargs["partition_key_path"] == "/upc"
    identity = "nested/products.ndjson:1" if directory else "products.ndjson:1"
    assert batch.await_args.args[0][0]["id"] == "doc-" + hashlib.sha1(
        identity.encode("utf-8")
    ).hexdigest()
    embed_client.close.assert_awaited_once()
    cosmos.close.assert_awaited_once()
