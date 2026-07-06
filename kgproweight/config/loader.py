"""YAML + CLI config loader with path expansion.

Usage::

    from kgproweight.config import load_config, ProjectConfig

    cfg = load_config(
        "configs/training/phase3_ppo.yaml",
        overrides={"training": {"seed": 13}},
        validate=ProjectConfig,
    )

The loader supports an ``includes:`` list at the top of any YAML file
(processed depth-first; later files override earlier ones), which lets us
keep ``base.yaml`` as a shared root.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Type, TypeVar

import yaml

from kgproweight.utils.paths import (
    checkpoint_dir,
    data_dir,
    index_dir,
    output_dir,
    project_root,
)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------

def _deep_update(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(val, Mapping) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def merge_yaml(path: str | os.PathLike, seen: Optional[set[str]] = None) -> dict[str, Any]:
    """Read a YAML and recursively apply ``includes:`` directives.

    ``includes`` are resolved relative to the file that contains them.
    """
    seen = seen or set()
    path = Path(path).resolve()
    if str(path) in seen:
        return {}
    seen.add(str(path))

    with open(path, "r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}

    includes = doc.pop("includes", []) or []
    merged: dict[str, Any] = {}
    for inc in includes:
        inc_path = (path.parent / inc).resolve()
        merged = _deep_update(merged, merge_yaml(inc_path, seen))
    merged = _deep_update(merged, doc)
    return merged


# ---------------------------------------------------------------------------
# Path expansion: replace ``$KGPW_*`` and relative paths with absolute paths.
# ---------------------------------------------------------------------------

_PATH_KEYS = {
    "corpus_path",
    "dense_index",
    "sparse_index",
    "save_dir",
    "save_path",
    "model_path",
    "model2path",  # FlashRAG mapping
    "method2index",  # FlashRAG mapping
    "tokenizer_path",
    "output_dir",
    "log_dir",
    "data_dir",
    "index_dir",
    "checkpoint_dir",
    "silver_data_path",
    "input_jsonl",
    "output_jsonl",
    "config_path",
    "lora_path",
    "checkpoint",
    "prm_checkpoint",
    "sft_checkpoint",
    "kg_embedding_model",
    "text_reward_fallback_path",
}


def _expand_path_value(val: Any) -> Any:
    if isinstance(val, str):
        if val.startswith("~"):
            val = str(Path(val).expanduser())
        # Allow ${KGPW_DATA_DIR}/..., $KGPW_DATA_DIR/..., etc.
        return os.path.expandvars(val)
    return val


def expand_paths(node: Any) -> Any:
    """Walk a config tree and expand env-var-style path strings in place."""
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for k, v in node.items():
            if k in _PATH_KEYS:
                if isinstance(v, dict):
                    out[k] = {kk: _expand_path_value(vv) for kk, vv in v.items()}
                elif isinstance(v, list):
                    out[k] = [_expand_path_value(x) for x in v]
                else:
                    out[k] = _expand_path_value(v)
            else:
                out[k] = expand_paths(v)
        return out
    if isinstance(node, list):
        return [expand_paths(x) for x in node]
    return node


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def _inject_defaults(doc: dict[str, Any]) -> dict[str, Any]:
    """Backfill project paths from kgproweight.utils.paths if absent."""
    defaults = {
        "project_root": str(project_root()),
        "data_dir": str(data_dir()),
        "index_dir": str(index_dir()),
        "checkpoint_dir": str(checkpoint_dir()),
        "output_dir": str(output_dir()),
    }
    for k, v in defaults.items():
        doc.setdefault(k, v)
    return doc


def load_config(
    path: str | os.PathLike,
    overrides: Optional[Mapping[str, Any]] = None,
    validate: Optional[Type[T]] = None,
) -> Any:
    """Load a config tree from ``path`` and optionally validate it.

    Parameters
    ----------
    path:
        Top-level YAML file.
    overrides:
        Optional nested dict applied last (e.g. CLI overrides).
    validate:
        Pydantic model to coerce the result into. If ``None``, returns the
        raw dict.
    """
    raw = merge_yaml(path)
    if overrides:
        raw = _deep_update(raw, dict(overrides))
    raw = _inject_defaults(raw)
    raw = expand_paths(raw)

    if validate is not None:
        return validate(**raw)
    return raw
