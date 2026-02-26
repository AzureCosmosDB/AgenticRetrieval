"""Shared pytest fixtures for integration tests."""
import sys
import asyncio
from pathlib import Path

import pytest
import yaml

# Ensure the repo root is on the path so rag_divdet can be imported
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Event-loop fixture (module-scoped so all fixtures share one loop)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def event_loop():
    """Create a module-scoped event loop shared by all fixtures and tests."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


# ---------------------------------------------------------------------------
# Config fixture: load config.yaml and expose as a dict
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def config():
    config_path = REPO_ROOT / "config.yaml"
    if not config_path.exists():
        pytest.skip("config.yaml not found – skipping integration tests")
    with open(config_path) as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# LLMClient fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def llm_client(config, event_loop):
    """Return an initialised LLMClient (module-scoped to avoid repeated setup)."""
    import rag_divdet as rd
    client = rd.LLMClient()
    yield client
    event_loop.run_until_complete(client.close())


# ---------------------------------------------------------------------------
# CombinedRetriever fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def retriever(config, event_loop):
    """Return an initialised CombinedRetriever (module-scoped)."""
    import rag_divdet as rd
    sources = rd._build_retrieval_sources(config)
    if not sources:
        pytest.skip("No retrieval sources configured – skipping retriever tests")
    ret = rd.CombinedRetriever(retrieval_sources=sources)
    event_loop.run_until_complete(ret.initialize())
    yield ret
    event_loop.run_until_complete(ret.close())
