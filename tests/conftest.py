"""Shared pytest fixtures.

Skips most heavy-weight tests when their dependencies are missing so that
``make test`` still succeeds on a fresh checkout (e.g. on CI) even before
the full requirements.txt has been installed.
"""

from __future__ import annotations

import os
import pathlib
import sys

import pytest

# Make ``import kgproweight`` work without ``pip install -e .``.
_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Tests must not touch the user's data; redirect KGPW_* dirs into pytest's tmp area.
os.environ.setdefault("KGPW_DATA_DIR", str(_ROOT / "outputs" / "_tests" / "data"))
os.environ.setdefault("KGPW_INDEX_DIR", str(_ROOT / "outputs" / "_tests" / "indexes"))
os.environ.setdefault("KGPW_CHECKPOINT_DIR", str(_ROOT / "outputs" / "_tests" / "ckpt"))
os.environ.setdefault("KGPW_OUTPUT_DIR", str(_ROOT / "outputs" / "_tests" / "out"))


def _need(module_name: str):
    """Return a pytest mark that skips when ``module_name`` is not importable."""
    try:
        __import__(module_name)
    except Exception as exc:  # ImportError or runtime ImportError chain
        return pytest.mark.skip(reason=f"{module_name} unavailable: {exc!s}")
    return None


@pytest.fixture(scope="session")
def need_torch():
    skip = _need("torch")
    if skip is not None:
        pytest.skip(skip.kwargs["reason"])
    import torch  # noqa: F401

    return torch
