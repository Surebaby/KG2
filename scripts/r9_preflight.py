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
    # Prefer v2 (filtered), fall back to v1
    kg_cache_path = index_dir() / "kg_cache" / "question_kg_index_v2.json"
    if not kg_cache_path.exists():
        kg_cache_path = index_dir() / "kg_cache" / "question_kg_index.json"
    if kg_cache_path.exists():
        section("5. question_kg_index Deep Check")
        raw = json.loads(kg_cache_path.read_text(encoding="utf-8"))
        all_ok &= check("entries > 0", len(raw) > 0, f"{len(raw)} entries")

        # Detect v1 vs v2 format
        sample = raw[0]
        is_v2 = "builder_version" in sample
        has_q = "q" in sample or "question" in sample
        all_ok &= check("has question field", has_q, f"v{'2' if is_v2 else '1'} format")

        # Triple accessor
        def _get_triples(entry):
            if "triples" in entry:  # v2: list of dicts
                return entry["triples"]
            return entry.get("t", [])  # v1: list of lists

        def _get_relation(triple):
            if isinstance(triple, dict):
                return triple.get("r", "")
            return triple[1] if len(triple) >= 2 else ""

        triples_list = _get_triples(sample)
        all_ok &= check("has triples", len(triples_list) > 0, f"{len(triples_list)} triples")
        all_ok &= check("format ok", _get_relation(triples_list[0]) != "", str(triples_list[0])[:80])

        # De-duplication
        questions = [e.get("q", e.get("question", "")) for e in raw]
        unique = set(questions)
        all_ok &= check("no duplicate questions", len(unique) == len(questions),
                        f"{len(unique)}/{len(questions)} unique")

        # Load speed
        t0 = time.time()
        q_kg_index = {e.get("q", e.get("question", "")): _get_triples(e) for e in raw}
        elapsed = time.time() - t0
        all_ok &= check("index build < 0.5s", elapsed < 0.5, f"{elapsed:.3f}s")

        # Triple stats
        avg_t = sum(len(_get_triples(e)) for e in raw) / max(1, len(raw))
        all_ok &= check("avg triples/question > 5" if not is_v2 else "avg triples/question > 3",
                        avg_t > (5 if not is_v2 else 3), f"{avg_t:.1f}")

        # R9 v6: quality checks
        taxonomic = {"instance of", "subclass of"}
        tax_count = sum(1 for e in raw for t in _get_triples(e)
                        if _get_relation(t) in taxonomic)
        total_t = sum(len(_get_triples(e)) for e in raw)
        tax_ratio = tax_count / max(1, total_t) * 100
        threshold = 15 if is_v2 else 30
        all_ok &= check(f"taxonomic ratio < {threshold}%", tax_ratio < threshold,
                        f"{tax_ratio:.1f}% (target < {threshold}%)")

        # Builder/policy version
        if is_v2:
            all_ok &= check("builder_version", True, sample["builder_version"])
        else:
            all_ok &= check("v1→v2 rebuild recommended", True,
                          "run scripts/prepare/06_build_question_kg_index.py")

        # ── Eval-split coverage (R9 v6) ──
        # The single most consequential gap found in the v6 audit: the index was
        # built from TRAIN questions only, so it covered 0/100 hotpotqa dev,
        # 0/12576 2wiki dev and 0/2417 musique dev. Inference silently fell back
        # to raw SPARQL-order triples, so the filtered KG never reached eval.
        # Coverage is now a hard check, not an afterthought.
        for ds in ["hotpotqa", "2wikimultihopqa", "musique"]:
            ds_file = data_dir() / ds / "dev.jsonl"
            if not ds_file.exists():
                continue
            dev_qs = []
            with open(ds_file, encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        dev_qs.append(json.loads(line).get("question", ""))
                    except json.JSONDecodeError:
                        continue
            if not dev_qs:
                continue
            covered = sum(1 for q in dev_qs if q in q_kg_index)
            pct = covered / len(dev_qs) * 100
            all_ok &= check(
                f"{ds} dev coverage >= 90%", pct >= 90.0,
                f"{covered}/{len(dev_qs)} = {pct:.1f}%"
                + ("" if pct >= 90 else
                   "  [rebuild: 06_build_question_kg_index.py --datasets "
                   f"{ds} --split dev]"),
            )

        # KG token budget
        p95_triples = sorted(len(_get_triples(e)) for e in raw)[int(len(raw) * 0.95)]
        p95_tokens = p95_triples * 40 // 4
        all_ok &= check("P95 KG tokens < 1200", p95_tokens <= 1200,
                        f"P95={p95_triples} triples ≈ {p95_tokens} tokens")

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
