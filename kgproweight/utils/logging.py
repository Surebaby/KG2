"""Logging helpers and reproducibility manifest dump."""

from __future__ import annotations

import json
import hashlib
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


def _git_status() -> list[str]:
    try:
        out = subprocess.check_output(
            ["git", "status", "--short"],
            stderr=subprocess.DEVNULL,
            cwd=str(Path(__file__).resolve().parent),
        )
        return [line for line in out.decode().splitlines() if line.strip()]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []


def _git_diff_sha256() -> Optional[str]:
    """Hash the tracked working-tree diff without embedding it in manifests."""
    try:
        out = subprocess.check_output(
            ["git", "diff", "--binary", "HEAD", "--"],
            stderr=subprocess.DEVNULL,
            cwd=str(Path(__file__).resolve().parent),
        )
        return hashlib.sha256(out).hexdigest()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _hash_source_inventory(root: Path, relative_paths: list[str]) -> str:
    digest = hashlib.sha256()
    for rel in sorted(set(relative_paths)):
        path = root / rel
        if not path.is_file():
            continue
        digest.update(rel.encode("utf-8") + b"\0")
        with path.open("rb") as fh:
            for block in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _filesystem_source_inventory(root: Path) -> list[str]:
    """Return the experiment-code inventory when a deployment has no ``.git``."""

    relative_paths: list[str] = []
    ignored_parts = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
    ignored_suffixes = {".pyc", ".pyo", ".swp", ".tmp"}
    for dirname in ("kgproweight", "configs", "scripts", "tests", "flashrag_src"):
        base = root / dirname
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(root)
            if any(part in ignored_parts for part in rel.parts):
                continue
            if path.suffix.lower() in ignored_suffixes:
                continue
            relative_paths.append(rel.as_posix())
    for name in ("pyproject.toml", "README.md", "RESEARCH_WORKFLOW.md", "AGENTS.md"):
        if (root / name).is_file():
            relative_paths.append(name)
    for pattern in ("launch*.sh", "check*.sh", "run*.sh"):
        relative_paths.extend(
            path.relative_to(root).as_posix()
            for path in root.glob(pattern)
            if path.is_file()
        )
    return sorted(set(relative_paths))


def _fallback_project_root() -> Optional[Path]:
    candidates = []
    configured = os.environ.get("KGPW_PROJECT_ROOT")
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend([Path.cwd(), Path(__file__).resolve().parents[2]])
    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / "kgproweight").is_dir() and (candidate / "configs").is_dir():
            return candidate
    return None


def _source_tree_provenance() -> dict[str, Any]:
    """Hash source/config files in Git checkouts and Git-less deployments."""

    try:
        root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL,
            cwd=str(Path(__file__).resolve().parent),
        ).decode().strip()
        out = subprocess.check_output(
            [
                "git", "ls-files", "--cached", "--others", "--exclude-standard",
                "--", "kgproweight", "configs", "scripts", "tests", "flashrag_src",
                "pyproject.toml", "README.md", "RESEARCH_WORKFLOW.md", "AGENTS.md",
                "*.sh",
            ],
            stderr=subprocess.DEVNULL,
            cwd=root,
        )
        paths = [line for line in out.decode().splitlines() if line]
        return {
            "sha256": _hash_source_inventory(Path(root), paths),
            "mode": "git_inventory",
            "root": str(Path(root).resolve()),
            "file_count": sum((Path(root) / rel).is_file() for rel in paths),
        }
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        root = _fallback_project_root()
        if root is None:
            return {"sha256": None, "mode": "unavailable", "root": None, "file_count": 0}
        try:
            paths = _filesystem_source_inventory(root)
            if not paths:
                return {"sha256": None, "mode": "unavailable", "root": str(root), "file_count": 0}
            return {
                "sha256": _hash_source_inventory(root, paths),
                "mode": "filesystem_fallback",
                "root": str(root),
                "file_count": len(paths),
            }
        except OSError:
            return {"sha256": None, "mode": "unavailable", "root": str(root), "file_count": 0}


def _source_tree_sha256() -> Optional[str]:
    """Backwards-compatible digest-only wrapper."""

    return _source_tree_provenance()["sha256"]


def file_md5(path: str | os.PathLike) -> str:
    digest = hashlib.md5()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_identity(path: str | os.PathLike) -> dict[str, Any]:
    """Return a compact, auditable identity for a file or model directory.

    Full data files and LoRA adapters are content-hashed. Multi-shard base
    models are identified by their config/index files plus shard inventory so a
    35 GB model pair is not re-read twice at every manifest update.
    """
    p = Path(path).expanduser().resolve()
    result: dict[str, Any] = {"path": str(p), "exists": p.exists()}
    if not p.exists():
        return result
    if p.is_file():
        result.update({"kind": "file", "size_bytes": p.stat().st_size, "md5": file_md5(p)})
        return result

    result["kind"] = "directory"
    inventory = []
    hash_names = {
        "manifest.json", "adapter_config.json", "adapter_model.safetensors",
        "config.json", "tokenizer_config.json", "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    }
    for child in sorted(x for x in p.iterdir() if x.is_file()):
        row: dict[str, Any] = {"name": child.name, "size_bytes": child.stat().st_size}
        if child.name in hash_names:
            row["md5"] = file_md5(child)
        inventory.append(row)
    result["files"] = inventory
    blob = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    result["inventory_sha256"] = hashlib.sha256(blob).hexdigest()
    return result


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


def dump_manifest(
    checkpoint_dir: str | os.PathLike,
    extra: Optional[Mapping[str, Any]] = None,
    *,
    status: str = "COMPLETE",
) -> Path:
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

    target = checkpoint_dir / "manifest.json"
    previous: dict[str, Any] = {}
    if target.exists():
        try:
            previous = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = {}
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    git_status = _git_status()
    source_provenance = _source_tree_provenance()
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "status": status,
        "started_at": previous.get("started_at", now),
        # Negative terminal outcomes are complete records too.  Otherwise a
        # preserved FAIL_STOP run is indistinguishable from a live process.
        "completed_at": previous.get("completed_at")
        or (now if status != "RUNNING" else None),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": _git_commit(),
        "git_dirty": bool(git_status),
        "git_status_short": git_status,
        "git_tracked_diff_sha256": _git_diff_sha256(),
        "source_tree_sha256": source_provenance["sha256"],
        "source_tree_hash_mode": source_provenance["mode"],
        "source_tree_root": source_provenance["root"],
        "source_tree_file_count": source_provenance["file_count"],
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

    with open(target, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False, default=str)
    return target


def prepare_new_run_dir(
    output: str | os.PathLike,
    *,
    experiment_id: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> tuple[Path, str]:
    """Atomically reserve a unique run directory and write a RUNNING manifest."""
    out = Path(output)
    exp_id = str(experiment_id or out.name).strip()
    if not exp_id:
        raise ValueError("A non-empty Experiment ID is required")
    try:
        out.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Refusing to reuse output_dir={out}. Formal experiments require a "
            "new Experiment ID/output directory; preserve the existing run, even "
            "if it failed."
        ) from exc
    run_extra = {"experiment_id": exp_id}
    if extra:
        run_extra.update(dict(extra))
    dump_manifest(out, extra=run_extra, status="RUNNING")
    return out, exp_id
