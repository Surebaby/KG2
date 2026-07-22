#!/usr/bin/env python
"""R9 Pre-flight Check — run this on the rented server before launching PPO."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def check(msg: str, ok: bool, detail: str = ""):
    mark = "✅" if ok else "❌"
    line = f"  {mark} {msg}"
    if detail:
        line += f"  → {detail}"
    print(line)
    return ok


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("=" * 60)


def main():
    all_ok = True

    # ── 1. Python environment ──
    section("1. Python & Dependencies")
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    all_ok &= check("Python >= 3.10", sys.version_info >= (3, 10), py_ver)

    for pkg in ["torch", "transformers", "trl", "peft"]:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "?")
        except ImportError:
            ver = None
        all_ok &= check(f"import {pkg}", ver is not None, ver or "MISSING")

    # flashrag needs path injection
    try:
        from kgproweight.utils.flashrag_bootstrap import setup_flashrag
        setup_flashrag()
        import flashrag
        all_ok &= check("import flashrag", True, getattr(flashrag, "__version__", "ok"))
    except Exception as e:
        all_ok &= check("import flashrag", False, str(e)[:80])

    # ── 2. Project package ──
    section("2. Project Package (kgproweight)")
    try:
        from kgproweight.utils.paths import (
            project_root, data_dir, index_dir, model_path, checkpoint_dir, output_dir
        )
        root = project_root()
        all_ok &= check("project_root", root.exists(), str(root))
        all_ok &= check("data_dir", data_dir().exists(), str(data_dir()))
        all_ok &= check("index_dir", index_dir().exists(), str(index_dir()))
        all_ok &= check("output_dir", output_dir().exists(), str(output_dir()))
        all_ok &= check("checkpoint_dir", checkpoint_dir().exists(), str(checkpoint_dir()))
    except Exception as e:
        all_ok &= check("kgproweight import", False, str(e))
        print("   ⚠️ Skipping remaining checks — kgproweight not importable")
        return 1

    # ── 3. Models ──
    section("3. Model Checkpoints")
    models = [
        ("llama3-8B-instruct", "Llama-3-8B (base model)"),
        ("e5", "E5 retriever"),
        ("rearag", "ReaRAG-9B text reward"),
    ]
    for name, desc in models:
        path = model_path(name)
        exists = Path(path).exists()
        all_ok &= check(f"{name} ({desc})", exists, path)

    # ── 4. R9 Cache Files ──
    section("4. Cache Files")
    cache_checks = [
        ("question_kg_index.json", index_dir() / "kg_cache" / "question_kg_index.json"),
        ("entity_cache.jsonl", index_dir() / "entity_cache.jsonl"),
        ("kg_subgraph_cache.jsonl", index_dir() / "kg_cache" / "kg_subgraph_cache.jsonl"),
        ("e5_Flat.index", index_dir() / "e5_Flat.index"),
        ("corpus_flashrag.jsonl", index_dir() / "corpus_flashrag.jsonl"),
    ]
    for label, path in cache_checks:
        ok = path.exists()
        size = ""
        if ok:
            size_mb = path.stat().st_size / (1024 * 1024)
            size = f"{size_mb:.1f} MB"
        all_ok &= check(label, ok, size if ok else str(path))

    # ── 5. question_kg_index deep check ──
    kg_cache_path = index_dir() / "kg_cache" / "question_kg_index.json"
    if kg_cache_path.exists():
        section("5. question_kg_index Deep Check")
        raw = json.loads(kg_cache_path.read_text(encoding="utf-8"))
        all_ok &= check("entries > 0", len(raw) > 0, f"{len(raw)} entries")

        # Structure
        sample = raw[0]
        all_ok &= check('has "q" field', "q" in sample)
        all_ok &= check('has "t" field (triples)', "t" in sample)
        if "t" in sample:
            all_ok &= check("triples are list of lists", isinstance(sample["t"], list) and len(sample["t"]) > 0)
            if sample["t"]:
                all_ok &= check("triple format [subj, rel, obj]", len(sample["t"][0]) == 3, str(sample["t"][0]))

        # De-duplication
        questions = [e["q"] for e in raw]
        unique = set(questions)
        all_ok &= check("no duplicate questions", len(unique) == len(questions),
                        f"{len(unique)}/{len(questions)} unique")

        # Load speed
        t0 = time.time()
        q_kg_index = {e["q"]: e["t"] for e in raw}
        elapsed = time.time() - t0
        all_ok &= check("index build < 0.5s", elapsed < 0.5, f"{elapsed:.3f}s")

        # Triple stats
        avg_t = sum(len(e["t"]) for e in raw) / max(1, len(raw))
        all_ok &= check("avg triples/question > 5", avg_t > 5, f"{avg_t:.1f}")

    # ── 6. Datasets ──
    section("6. Datasets")
    for ds in ["hotpotqa", "2wikimultihopqa", "musique"]:
        ds_path = data_dir() / ds
        ok = ds_path.exists()
        files = ""
        if ok:
            files = ", ".join([f.name for f in ds_path.iterdir() if f.is_file()][:3])
        all_ok &= check(ds, ok, files if ok else str(ds_path))

    # ── 7. Silver Data ──
    section("7. Silver Data")
    silver_dir = data_dir() / "silver_data"
    if silver_dir.exists():
        for f in sorted(silver_dir.iterdir()):
            if f.suffix in (".jsonl", ".json"):
                size_mb = f.stat().st_size / (1024 * 1024)
                lines = len(f.read_text(encoding="utf-8").strip().split("\n"))
                all_ok &= check(f.name, True, f"{lines} lines, {size_mb:.1f} MB")
    else:
        all_ok &= check("silver_data dir", False, str(silver_dir))

    # ── 8. GPU ──
    section("8. GPU")
    try:
        import torch
        gpu_ok = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if gpu_ok else 0
        gpu_name = torch.cuda.get_device_name(0) if gpu_ok else "N/A"
        props = torch.cuda.get_device_properties(0) if gpu_ok else None
        gpu_mem = props.total_memory / (1024**3) if props else 0
        all_ok &= check("CUDA available", gpu_ok, f"{gpu_count}x {gpu_name} ({gpu_mem:.0f} GB)")
    except Exception as e:
        all_ok &= check("torch CUDA check", False, str(e))

    # ── Summary ──
    section("SUMMARY")
    if all_ok:
        print("  ✅ ALL CHECKS PASSED — Ready to launch R9 PPO training.")
    else:
        print("  ❌ SOME CHECKS FAILED — fix above before launching.")

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
