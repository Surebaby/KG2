"""Logging helpers and reproducibility manifest dump."""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional


_DEFAULT_FORMAT = "[%(asctime)s] %(levelname)s %(name)s :: %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: str = "INFO", log_file: Optional[str | os.PathLike] = None) -> None:
    """Configure the root logger once.

    Call this at the start of every CLI entrypoint. Re-entrant: subsequent
    calls in the same process are no-ops.
    """
    if logging.getLogger().handlers:
        return

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=_DEFAULT_FORMAT,
        datefmt=_DEFAULT_DATEFMT,
        handlers=handlers,
    )

    # Hush the noisiest third-party loggers.
    for noisy in ("urllib3", "httpx", "filelock", "datasets", "transformers.tokenization_utils_base"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a logger; configure with defaults if no one has yet."""
    configure_logging()
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def _git_commit() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=str(Path(__file__).resolve().parent),
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _gpu_name() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip().splitlines()[0]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _cuda_version() -> Optional[str]:
    try:
        import torch

        return torch.version.cuda
    except ImportError:
        return None


def _pip_freeze(top_packages: tuple[str, ...] = ("torch", "transformers", "trl", "peft", "bitsandbytes", "datasets", "faiss-cpu", "bm25s")) -> dict[str, Optional[str]]:
    out: dict[str, Optional[str]] = {}
    for pkg in top_packages:
        try:
            from importlib.metadata import version  # py3.10+

            out[pkg] = version(pkg)
        except Exception:
            out[pkg] = None
    return out


def dump_manifest(checkpoint_dir: str | os.PathLike, extra: Optional[Mapping[str, Any]] = None) -> Path:
    """Write ``manifest.json`` recording everything needed to reproduce a run.

    Parameters
    ----------
    checkpoint_dir:
        Directory of the just-finished training run. ``manifest.json`` is
        written under this directory.
    extra:
        Additional key/value pairs (seed, config snapshot, dataset hash,
        hyperparameters, …).
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": _git_commit(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "gpu_name": _gpu_name(),
        "cuda_version": _cuda_version(),
        "packages": _pip_freeze(),
        "env": {
            "KGPW_PROJECT_ROOT": os.environ.get("KGPW_PROJECT_ROOT"),
            "KGPW_DATA_DIR": os.environ.get("KGPW_DATA_DIR"),
            "KGPW_INDEX_DIR": os.environ.get("KGPW_INDEX_DIR"),
            "KGPW_CHECKPOINT_DIR": os.environ.get("KGPW_CHECKPOINT_DIR"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }
    if extra is not None:
        manifest["run"] = dict(extra)

    target = checkpoint_dir / "manifest.json"
    with open(target, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False, default=str)
    return target
