"""Unit tests for utils.cosmos_account (management-plane helpers extracted
from cosmos_db_upload.py)."""

from unittest.mock import MagicMock, patch

import pytest
from azure.core.exceptions import HttpResponseError

from utils.cosmos_account import (
    _build_dedicated_throughput_options,
    _field_to_cosmos_json_path,
    ensure_cosmos_account_exists,
    extract_account_name_from_endpoint,
)


class TestExtractAccountNameFromEndpoint:
    def test_extracts_account_name(self):
        assert (
            extract_account_name_from_endpoint("https://myaccount.documents.azure.com:443/")
            == "myaccount"
        )

    def test_extracts_account_name_without_port(self):
        assert (
            extract_account_name_from_endpoint("https://myaccount.documents.azure.com/")
            == "myaccount"
        )

    def test_raises_on_unrecognized_endpoint(self):
        with pytest.raises(ValueError, match="Could not extract account name"):
            extract_account_name_from_endpoint("https://example.com/")


class TestFieldToCosmosJsonPath:
    def test_simple_field(self):
        assert _field_to_cosmos_json_path("e") == "/e"

    def test_nested_field(self):
        assert _field_to_cosmos_json_path("embedding.vector") == "/embedding/vector"

    def test_empty_defaults_to_e(self):
        assert _field_to_cosmos_json_path("") == "/e"


class TestBuildDedicatedThroughputOptions:
    def test_autoscale_mode(self):
        options = _build_dedicated_throughput_options("autoscale", 4000)
        assert options.autoscale_settings.max_throughput == 4000
        assert options.throughput is None

    def test_manual_mode(self):
        options = _build_dedicated_throughput_options("manual", 400)
        assert options.throughput == 400
        assert options.autoscale_settings is None


class TestEnsureCosmosAccountExists:
    @patch("utils.cosmos_account.CosmosDBManagementClient")
    def test_passes_when_account_exists(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        ensure_cosmos_account_exists(
            credential=MagicMock(),
            subscription_id="sub-id",
            resource_group="my-rg",
            account_name="my-account",
        )

        mock_client.database_accounts.get.assert_called_once_with("my-rg", "my-account")

    @patch("utils.cosmos_account.CosmosDBManagementClient")
    def test_raises_clear_error_when_account_missing(self, mock_client_cls):
        mock_client = MagicMock()
        not_found = HttpResponseError(message="not found")
        not_found.status_code = 404
        mock_client.database_accounts.get.side_effect = not_found
        mock_client_cls.return_value = mock_client

        with pytest.raises(ValueError, match="was not found"):
            ensure_cosmos_account_exists(
                credential=MagicMock(),
                subscription_id="sub-id",
                resource_group="my-rg",
                account_name="my-account",
            )

    @patch("utils.cosmos_account.CosmosDBManagementClient")
    def test_reraises_non_404_errors(self, mock_client_cls):
        mock_client = MagicMock()
        forbidden = HttpResponseError(message="forbidden")
        forbidden.status_code = 403
        mock_client.database_accounts.get.side_effect = forbidden
        mock_client_cls.return_value = mock_client

        with pytest.raises(HttpResponseError):
            ensure_cosmos_account_exists(
                credential=MagicMock(),
                subscription_id="sub-id",
                resource_group="my-rg",
                account_name="my-account",
            )
