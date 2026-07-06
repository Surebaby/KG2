"""Centralised path resolution.

Every other module imports paths from here. Hardcoded absolute paths are
forbidden everywhere else in the codebase. Resolution priority for any
path:

  1. Argument explicitly passed by the caller.
  2. Environment variable ``KGPW_*`` documented in ``.env.example``.
  3. Default relative to the project root (the directory containing
     ``pyproject.toml``).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

# Mapping from short logical name → environment variable for downloadable model paths.
# The reward / pipeline / scripts layer only references these short names.
DEFAULT_MODEL_ENV = {
    "llama3-8B-instruct": ("KGPW_LLAMA3_PATH", "meta-llama/Meta-Llama-3-8B-Instruct"),
    "e5": ("KGPW_E5_PATH", "intfloat/e5-base-v2"),
    "rearag": ("KGPW_REARAG_PATH", "THU-KEG/ReaRAG-9B"),
    "r1-searcher": ("KGPW_R1SEARCHER_PATH", "XXsongLALA/Qwen-2.5-7B-base-RAG-RL"),
    "selfrag": ("KGPW_SELFRAG_PATH", "selfrag/selfrag_llama2_7b"),
}


def _find_project_root() -> Path:
    """Walk upward from this file until we find a ``pyproject.toml``."""
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd().resolve()


@lru_cache(maxsize=1)
def project_root() -> Path:
    """Return the absolute project root.

    Override with the ``KGPW_PROJECT_ROOT`` env var if the install is
    relocated (e.g., the package is installed system-wide and the user
    keeps data under ``$HOME/kg-experiments``).
    """
    env = os.environ.get("KGPW_PROJECT_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return _find_project_root()


def _env_path(env_var: str, default_subdir: str) -> Path:
    env = os.environ.get(env_var)
    if env:
        return Path(env).expanduser().resolve()
    return project_root() / default_subdir


def data_dir() -> Path:
    """Directory containing ``hotpotqa/``, ``musique/``, ``silver_data/`` …"""
    return _env_path("KGPW_DATA_DIR", "data")


def index_dir() -> Path:
    """Directory containing ``e5_Flat.index``, ``bm25/``, ``kg_cache/`` …"""
    return _env_path("KGPW_INDEX_DIR", "indexes")


def checkpoint_dir() -> Path:
    """Directory containing training checkpoints."""
    return _env_path("KGPW_CHECKPOINT_DIR", "checkpoints")


def output_dir() -> Path:
    """Directory for evaluation outputs and summaries."""
    return _env_path("KGPW_OUTPUT_DIR", "outputs")


def config_dir() -> Path:
    """The ``configs/`` directory under the project root."""
    return project_root() / "configs"


def flashrag_root() -> Optional[Path]:
    """Return the FlashRAG-main source root if known.

    Resolution order:
      1. ``KGPW_FLASHRAG_ROOT`` env var.
      2. ``$PROJECT_ROOT/third_party/FlashRAG``.
      3. ``$HOME/flashrag/flashrag/FlashRAG-main`` (legacy convenience).
      4. None — assume FlashRAG is pip-installed.
    """
    env = os.environ.get("KGPW_FLASHRAG_ROOT")
    if env:
        path = Path(env).expanduser().resolve()
        if path.exists():
            return path

    candidates = [
        project_root() / "third_party" / "FlashRAG",
        project_root() / "third_party" / "FlashRAG-main",
        Path.home() / "flashrag" / "flashrag" / "FlashRAG-main",
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return None


# Common local model directories checked before falling back to HF.
_LOCAL_MODEL_DIRS = [
    # AutoDL standard (remote training server)
    "/root/autodl-tmp/models",
    # Project-local (dev machine)
    None,  # placeholder — filled at runtime with project_root() / "models"
]


def model_path(short_name: str) -> str:
    """Resolve a logical model name to a HF id or local checkout path.

    Resolution order:
      1. Environment variable (``KGPW_LLAMA3_PATH`` etc.).
      2. Local model directory lookup (``project_root/models/<name>``,
         ``/root/autodl-tmp/models/<name>``).
      3. HuggingFace default (``meta-llama/Meta-Llama-3-8B-Instruct`` etc.).
    """
    env_var, default = DEFAULT_MODEL_ENV.get(short_name, (None, short_name))
    if env_var is None:
        return short_name

    # 1. Env var override.
    val = os.environ.get(env_var)
    if val:
        return val

    # 2. Local model directories.
    model_dirs = list(_LOCAL_MODEL_DIRS)
    model_dirs[model_dirs.index(None)] = str(project_root() / "models")

    # Try common subdirectory names.
    # e.g. "llama3-8B-instruct" → try: llama3-8B-instruct, llama3-8b, Llama-3-8B
    name_no_instruct = short_name.replace("-Instruct", "").replace("-instruct", "")
    base = default.split("/")[-1] if "/" in default else default
    candidates = list({
        short_name,
        short_name.lower(),
        name_no_instruct,
        name_no_instruct.lower(),
        base,
        base.lower(),
    })
    for base_dir in model_dirs:
        if not base_dir:
            continue
        try:
            if not Path(base_dir).is_dir():
                continue
        except (PermissionError, OSError):
            continue
        for cand in set(candidates):
            cand_path = Path(base_dir) / cand
            try:
                if cand_path.is_dir():
                    return str(cand_path.resolve())
            except (PermissionError, OSError):
                continue

    # 3. HF default.
    return default
