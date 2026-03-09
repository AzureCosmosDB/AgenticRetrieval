#!/usr/bin/env python
"""Generate config.yaml from config.yaml.example by substituting environment variables.

Environment variables consumed
-------------------------------
COSMOS_URI                   – Cosmos DB account URI
COSMOS_KEY                   – Cosmos DB primary key
COSMOS_DATABASE_NAME         – database name (default: divdet)
COSMOS_SOURCE1_CONTAINER     – container name for source_1 (default: container_1)
COSMOS_SOURCE2_CONTAINER     – container name for source_2 (default: container_2)
LLM_ENDPOINT                 – Azure OpenAI LLM endpoint URL
LLM_API_KEY                  – Azure OpenAI LLM API key
LLM_MODEL                    – LLM deployment/model name
EMBED_ENDPOINT               – Azure OpenAI embedding endpoint URL
EMBED_API_KEY                – Azure OpenAI embedding API key
EMBED_MODEL                  – Embedding deployment/model name
EMBED_DIMENSIONS             – Embedding output dimensions (default: 1024)
"""
import os
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parent.parent
EXAMPLE_PATH = REPO_ROOT / "config.yaml.example"
OUTPUT_PATH = REPO_ROOT / "config.yaml"


def _require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        print(f"ERROR: required environment variable '{name}' is not set or empty.", file=sys.stderr)
        sys.exit(1)
    return value


def _optional(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip() or default


def main():
    if not EXAMPLE_PATH.exists():
        print(f"ERROR: {EXAMPLE_PATH} not found.", file=sys.stderr)
        sys.exit(1)

    with open(EXAMPLE_PATH) as fh:
        cfg = yaml.safe_load(fh)

    # --- Cosmos DB -----------------------------------------------------------
    cfg["cosmos"]["uri"] = _require("COSMOS_URI")
    cfg["cosmos"]["key"] = _require("COSMOS_KEY")
    cfg["cosmos"]["database_name"] = _optional("COSMOS_DATABASE_NAME", "divdet")

    # Patch container names inside sources list
    container_overrides = {
        "source_1": _optional("COSMOS_SOURCE1_CONTAINER", "container_1"),
        "source_2": _optional("COSMOS_SOURCE2_CONTAINER", "container_2"),
    }
    for source in cfg.get("cosmos", {}).get("sources", []):
        source_id = source.get("id", "")
        if source_id in container_overrides:
            source["container_name"] = container_overrides[source_id]

    # --- LLM -----------------------------------------------------------------
    cfg["llm"]["llm_endpoint"] = _require("LLM_ENDPOINT")
    cfg["llm"]["llm_api_key"] = _require("LLM_API_KEY")
    cfg["llm"]["llm_model"] = _optional("LLM_MODEL", cfg["llm"].get("llm_model", "model-router"))

    # --- Embedding -----------------------------------------------------------
    if "embedding" not in cfg:
        cfg["embedding"] = {}
    cfg["embedding"]["embed_endpoint"] = _require("EMBED_ENDPOINT")
    cfg["embedding"]["embed_api_key"] = _require("EMBED_API_KEY")
    cfg["embedding"]["embed_model"] = _require("EMBED_MODEL")
    cfg["embedding"]["embed_dimensions"] = int(_optional("EMBED_DIMENSIONS", "1024"))

    # Disable local LLM fallback in CI to avoid noise
    cfg.setdefault("local_llm", {})["use_local_fallback_for_subtasks"] = False

    # Write output
    with open(OUTPUT_PATH, "w") as fh:
        yaml.dump(cfg, fh, allow_unicode=True, sort_keys=False)

    print(f"config.yaml written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
