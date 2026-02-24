"""
pytest configuration for AgenticRetrieval tests.

rag_divdet loads config.yaml at module-import time.  Because config.yaml is
intentionally git-ignored (it contains secrets), the tests redirect that open
call to config.yaml.example before any test module is imported.
"""

import builtins
import pathlib

_REPO = pathlib.Path(__file__).parent
_REAL_OPEN = builtins.open


def _open_redirect(path, *args, **kwargs):
    """Redirect config.yaml → config.yaml.example during tests."""
    p = pathlib.Path(str(path))
    if p.name == "config.yaml" and p.parent == _REPO:
        return _REAL_OPEN(_REPO / "config.yaml.example", *args, **kwargs)
    return _REAL_OPEN(path, *args, **kwargs)


builtins.open = _open_redirect
