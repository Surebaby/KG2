#!/usr/bin/env python
"""Phase 2/3 deployment pre-flight check (AutoDL PRO 6000 / 96 GB).

Run this FIRST on a freshly-rented box, before `make phase2`. It verifies every
prerequisite that — if missing — would waste paid GPU time: model checkouts,
silver data, entity cache, Python deps, CUDA/VRAM, and config resolution.

    python scripts/deploy/preflight.py            # full check
    python scripts/deploy/preflight.py --quick    # skip the model-load probe

Exit code 0 = all green (safe to train). Non-zero = at least one BLOCKER.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

GREEN, RED, YEL, DIM, RST = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"
OK, BAD, WARN = f"{GREEN}✓{RST}", f"{RED}✗{RST}", f"{YEL}!{RST}"

_blockers: list[str] = []
_warnings: list[str] = []


def blocker(msg: str) -> None:
    print(f"  {BAD} {msg}")
    _blockers.append(msg)


def warn(msg: str) -> None:
    print(f"  {WARN} {msg}")
    _warnings.append(msg)


def ok(msg: str) -> None:
    print(f"  {OK} {msg}")


def section(title: str) -> None:
    print(f"\n{title}")


def _load_dotenv() -> None:
    """Best-effort: export KEY=VALUE lines from ./.env so KGPW_* resolve."""
    env = Path(".env")
    if not env.exists():
        warn(".env not found in CWD — relying on already-exported env vars")
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    ok(".env loaded")


# ── 1. Python dependencies ──────────────────────────────────────────────────
_REQUIRED = {
    "torch": "2.4.0",
    "transformers": "4.44.0",
    "peft": "0.12.0",
    "accelerate": "0.33.0",
    "trl": "0.11.4",
}


def _ver_tuple(v: str):
    out = []
    for part in v.split(".")[:3]:
        num = "".join(c for c in part if c.isdigit())
        out.append(int(num) if num else 0)
    return tuple(out)


def check_deps() -> None:
    section("1. Python dependencies")
    for pkg, want in _REQUIRED.items():
        try:
            mod = importlib.import_module(pkg)
            have = getattr(mod, "__version__", "?")
            if pkg == "trl" and have != "0.11.4":
                warn(f"{pkg}=={have} (pinned 0.11.4; other versions may break PPO model wiring)")
            elif have != "?" and _ver_tuple(have) < _ver_tuple(want):
                blocker(f"{pkg}=={have} < required {want}")
            else:
                ok(f"{pkg}=={have}")
        except ImportError:
            blocker(f"{pkg} NOT installed — run `pip install -r requirements.txt && pip install -e .`")


# ── 2. CUDA / GPU ───────────────────────────────────────────────────────────
def check_gpu() -> None:
    section("2. CUDA / GPU")
    try:
        import torch
    except ImportError:
        blocker("torch missing — cannot check GPU")
        return
    if not torch.cuda.is_available():
        blocker("CUDA not available — training needs a GPU")
        return
    n = torch.cuda.device_count()
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        gb = p.total_memory / 1024**3
        msg = f"GPU{i}: {p.name}, {gb:.0f} GB"
        if gb < 40:
            warn(f"{msg} — Phase 3 PPO (8B policy + 9B reward) wants ~50-60 GB; tight/insufficient")
        else:
            ok(msg)
    cc = torch.cuda.get_device_capability(0)
    ok(f"compute capability sm_{cc[0]}{cc[1]}, torch CUDA {torch.version.cuda}")


# ── 3. Models (Llama-3-8B + ReaRAG-9B) ──────────────────────────────────────
def _is_local_model_dir(p: str) -> bool:
    d = Path(p)
    if not d.is_dir():
        return False
    has_cfg = (d / "config.json").exists()
    has_wt = any(d.glob("*.safetensors")) or any(d.glob("*.bin")) or any(d.glob("**/*.safetensors"))
    return has_cfg and has_wt


def check_models(quick: bool) -> None:
    section("3. Models")
    from kgproweight.utils.paths import model_path

    for short, env_var, role in [
        ("llama3-8B-instruct", "KGPW_LLAMA3_PATH", "base model (all phases)"),
        ("rearag", "KGPW_REARAG_PATH", "PPO text reward (Phase 3b)"),
    ]:
        resolved = model_path(short)
        if _is_local_model_dir(resolved):
            ok(f"{short}: local dir {resolved}  [{role}]")
        elif "/" in resolved and not Path(resolved).exists():
            # looks like a HF id, not a local path
            warn(
                f"{short}: resolves to HF id '{resolved}' (no local checkout). "
                f"Set {env_var} to a downloaded dir, or it will stream/download at train time."
            )
        else:
            blocker(f"{short}: '{resolved}' is not a valid model dir (need config.json + weights). Set {env_var}.")


# ── 4. Silver data + entity cache ───────────────────────────────────────────
def check_data() -> None:
    section("4. Silver data + entity cache")
    from kgproweight.utils.paths import data_dir
    from kgproweight.retrieval.bootstrap import resolve_entity_cache_path

    silver = Path(data_dir()) / "silver_data" / "silver_trajectories.jsonl"
    if not silver.exists():
        blocker(f"silver data missing: {silver} — upload Phase 1 output here")
    else:
        try:
            from kgproweight.data.silver_dataset import SilverDatasetReader

            reader = SilverDatasetReader(silver)
            total = len(reader.trajectories)
            acc = len(reader.accepted())
            if acc == 0:
                blocker(f"silver data has 0 accepted trajectories ({total} total) — training has nothing to learn from")
            else:
                ok(f"silver data: {acc} accepted / {total} total @ {silver}")
        except Exception as exc:  # noqa: BLE001
            blocker(f"silver data unreadable: {exc}")

    ent = Path(resolve_entity_cache_path())
    if not ent.exists():
        warn(
            f"entity cache missing: {ent} — PPO's PRMAnnotator will run with an EMPTY cache "
            "(KG features degrade). Upload indexes/entity_cache.jsonl."
        )
    else:
        n = sum(1 for _ in ent.open())
        ok(f"entity cache: {n} entries @ {ent}")


# ── 5. Config files resolve ─────────────────────────────────────────────────
def check_configs() -> None:
    section("5. Training configs resolve")
    from kgproweight.config.loader import load_config
    from kgproweight.config.schemas import ProjectConfig

    for name in ["phase2_prm", "phase3_sft", "phase3_ppo"]:
        path = Path("configs/training") / f"{name}.yaml"
        if not path.exists():
            blocker(f"config missing: {path}")
            continue
        try:
            load_config(str(path), validate=ProjectConfig)
            ok(f"{name}.yaml parses + validates")
        except Exception as exc:  # noqa: BLE001
            blocker(f"{name}.yaml failed to load: {exc}")


# ── 6. Optional: probe a real model load (catches gated/corrupt checkouts) ───
def check_model_load() -> None:
    section("6. Model load probe (tokenizer + config only)")
    from kgproweight.utils.paths import model_path

    for short in ["llama3-8B-instruct", "rearag"]:
        resolved = model_path(short)
        if not _is_local_model_dir(resolved):
            warn(f"{short}: skipping probe (no local checkout)")
            continue
        try:
            from transformers import AutoConfig, AutoTokenizer

            AutoConfig.from_pretrained(resolved, trust_remote_code=True)
            AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
            ok(f"{short}: config + tokenizer load OK")
        except Exception as exc:  # noqa: BLE001
            blocker(f"{short}: load probe failed: {exc}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 2/3 deployment pre-flight check")
    ap.add_argument("--quick", action="store_true", help="skip the model-load probe (step 6)")
    args = ap.parse_args()

    print(f"{DIM}Phase 2/3 pre-flight — run on the rented box before `make phase2`{RST}")
    section("0. Environment")
    _load_dotenv()

    check_deps()
    check_gpu()
    check_models(args.quick)
    check_data()
    check_configs()
    if not args.quick:
        check_model_load()

    print("\n" + "=" * 60)
    if _blockers:
        print(f"{RED}BLOCKERS: {len(_blockers)}{RST} — fix before training:")
        for b in _blockers:
            print(f"  {BAD} {b}")
    if _warnings:
        print(f"{YEL}WARNINGS: {len(_warnings)}{RST} — review:")
        for w in _warnings:
            print(f"  {WARN} {w}")
    if not _blockers and not _warnings:
        print(f"{GREEN}ALL GREEN — safe to train.{RST}")
    elif not _blockers:
        print(f"{GREEN}No blockers.{RST} Warnings are non-fatal; train if you understand them.")
    print("=" * 60)
    return 1 if _blockers else 0


if __name__ == "__main__":
    sys.exit(main())





