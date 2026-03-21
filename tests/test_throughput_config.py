"""Tests for cosmos_db_upload throughput configuration."""

import pytest
import textwrap
from pathlib import Path


# Minimal YAML that satisfies load_config requirements (cosmos.sources must exist)
_BASE_YAML = textwrap.dedent("""\
    embedding:
      embed_endpoint: "https://example.com"
      embed_model: "text-embedding-3-small"
      embed_dimensions: 1024
      embed_api_key: "fake-key"
    cosmos:
      uri: "https://example.documents.azure.com:443/"
      database_name: "testdb"
      embedding_batch_size: 10
      {throughput_block}
      sources:
        - id: "src1"
          container_name: "c1"
          partition_key_path: "/pk"
          embedding_field: "e"
          documents_root: "data/"
          embedding_text_fields:
            - title
""")


def _write_config(tmp_path: Path, throughput_block: str = "") -> Path:
    cfg = _BASE_YAML.format(throughput_block=throughput_block)
    p = tmp_path / "config.yaml"
    p.write_text(cfg)
    return p


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------

def test_defaults_when_throughput_not_specified(tmp_path):
    """Throughput defaults to autoscale / 1000 when omitted from config."""
    import cosmos_db_upload as mod
    mod.load_config(_write_config(tmp_path))
    assert mod.THROUGHPUT_MODE == "autoscale"
    assert mod.THROUGHPUT_VALUE == 1000


# ---------------------------------------------------------------------------
# Autoscale mode
# ---------------------------------------------------------------------------

def test_autoscale_mode_explicit(tmp_path):
    """Explicitly setting autoscale mode with a custom RU value."""
    import cosmos_db_upload as mod
    mod.load_config(_write_config(tmp_path,
        'throughput_mode: "autoscale"\n  throughput_value: 4000'))
    assert mod.THROUGHPUT_MODE == "autoscale"
    assert mod.THROUGHPUT_VALUE == 4000


# ---------------------------------------------------------------------------
# Manual mode
# ---------------------------------------------------------------------------

def test_manual_mode(tmp_path):
    """Setting manual throughput mode."""
    import cosmos_db_upload as mod
    mod.load_config(_write_config(tmp_path,
        'throughput_mode: "manual"\n  throughput_value: 800'))
    assert mod.THROUGHPUT_MODE == "manual"
    assert mod.THROUGHPUT_VALUE == 800


# ---------------------------------------------------------------------------
# Invalid mode
# ---------------------------------------------------------------------------

def test_invalid_mode_raises(tmp_path):
    """An unrecognised throughput_mode must raise ValueError."""
    import cosmos_db_upload as mod
    with pytest.raises(ValueError, match="Invalid cosmos.throughput_mode"):
        mod.load_config(_write_config(tmp_path,
            'throughput_mode: "serverless"'))


# ---------------------------------------------------------------------------
# Case insensitivity
# ---------------------------------------------------------------------------

def test_mode_is_case_insensitive(tmp_path):
    """throughput_mode should be normalised to lowercase."""
    import cosmos_db_upload as mod
    mod.load_config(_write_config(tmp_path,
        'throughput_mode: "Autoscale"\n  throughput_value: 2000'))
    assert mod.THROUGHPUT_MODE == "autoscale"
    assert mod.THROUGHPUT_VALUE == 2000
