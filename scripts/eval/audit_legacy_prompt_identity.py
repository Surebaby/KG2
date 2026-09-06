#!/usr/bin/env python
"""Byte-exact identity audit of the legacy inference prompt path.

Rebuilds every question's prompt from the *stored* ``retrieval_result`` +
``kg_subgraphs`` in a historical n=300 ``intermediate_data.json``, using the
*current* ``build_inference_messages()`` and the same tokenizer chat template,
then compares it byte-for-byte (and by SHA256) against the prompt saved in that
artifact.

Purpose: prove that the step-1 ProofKG-v1 plumbing (``kg_supply_mode`` switch,
``parse_steps(..., known_kg=...)`` telemetry change) did NOT drift the legacy
input path.  This runs on CPU: it loads the tokenizer only, never the 8B
weights, and never re-runs retrieval.

Three computed layers per question:

1. messages       — ``build_inference_messages()`` output (deterministic, recorded as SHA256);
2. chat-template  — ``tokenizer.apply_chat_template(..., tokenize=False)`` full prompt, byte-compared;
3. SHA256         — digest of the rebuilt prompt vs the stored prompt.

If a prompt mismatches, ``details.jsonl`` records the first differing byte so the
drift can be localised to (a) the step-1 change, (b) an unrelated prompt/parser
edit, or (c) a tokenizer / chat-template version difference.  Note that
``parse_steps(..., known_kg=...)`` never enters this path, so it cannot be cited
as a prompt-mismatch explanation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from kgproweight.data.prompts import build_inference_messages
from kgproweight.utils.logging import dump_manifest, get_logger

logger = get_logger(__name__)

DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")
DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "outputs" / "sft_quota70_baseline_eval"
DEFAULT_OUT = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "audits"
    / "legacy_prompt_identity_n900_seed42_v1"
)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def discover_inputs(root: Path) -> List[Tuple[str, Path]]:
    """Return (dataset, run_dir) for the three historical n=300 eval runs."""
    found: List[Tuple[str, Path]] = []
    for ds in DATASETS:
        seed_dir = root / ds / "seed_42"
        if not seed_dir.is_dir():
            raise FileNotFoundError(f"missing seed dir: {seed_dir}")
        candidates = sorted(seed_dir.glob(f"{ds}_*_kg_proweight"))
        run_dirs = [c for c in candidates if (c / "intermediate_data.json").is_file()]
        if not run_dirs:
            raise FileNotFoundError(f"no kg_proweight run with intermediate_data.json under {seed_dir}")
        if len(run_dirs) > 1:
            raise ValueError(f"ambiguous: multiple runs under {seed_dir}: {run_dirs}")
        found.append((ds, run_dirs[0]))
    return found


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_yaml_config(run_dir: Path) -> Dict[str, Any]:
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"historical config missing: {cfg_path}")
    # The resolved FlashRAG config is YAML. Avoid a hard dependency on PyYAML in
    # this audit by loading via the project's config loader, which already
    # depends on it.
    import yaml

    return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}


def _first_diff(left: str, right: str) -> Optional[int]:
    if left == right:
        return None
    for i, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return i
    return min(len(left), len(right))


def _valid_retrieval_result(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    return all(
        isinstance(p, dict) and (p.get("contents") or p.get("text"))
        for p in value
    )


def _valid_kg_subgraphs(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    return all(isinstance(t, (list, tuple)) and len(t) == 3 for t in value)


def rebuild_prompt(
    question: str,
    retrieval_result: Any,
    kg_subgraphs: Any,
    tokenizer,
    *,
    retrieval_topk: int,
    max_kg_triples: int,
    max_input_len: int,
) -> Tuple[List[Dict[str, str]], str]:
    """Return (messages, full_prompt) using the current prompt-building code."""
    messages = build_inference_messages(
        question=question,
        retrieved_passages=retrieval_result,
        kg_triples=kg_subgraphs,
        top_k=retrieval_topk,
        max_kg_triples=max_kg_triples,
    )
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # Reproduce PromptTemplate.truncate_prompt's non-openai branch: it is a no-op
    # below max_input_len, but keep it so a future prompt that crosses the budget
    # is truncated exactly as eval would have truncated it.
    ids = tokenizer(prompt, truncation=False, return_tensors="pt")["input_ids"][0]
    if len(ids) >= max_input_len:
        half = int(max_input_len / 2) - 20
        prompt = tokenizer.decode(ids[:half], skip_special_tokens=True) + tokenizer.decode(
            ids[-half:], skip_special_tokens=True
        )
    return messages, prompt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="Root of the historical sft_quota70_baseline_eval runs.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="Output directory for details.jsonl/report.json/manifest.json.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    inputs = discover_inputs(args.root)

    # Load the tokenizer once. All three datasets share the same generator model.
    from transformers import AutoTokenizer

    first_dir = inputs[0][1]
    cfg = _read_yaml_config(first_dir)
    model_path = cfg.get("generator_model_path")
    if not model_path:
        raise ValueError(f"generator_model_path missing from {first_dir}/config.yaml")
    retrieval_topk = int(cfg.get("retrieval_topk") or 50)
    max_input_len = int(cfg.get("generator_max_input_len") or 6144)
    max_kg_triples = 12

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    logger.info("Loaded tokenizer %s (vocab=%d)", model_path, tokenizer.vocab_size)

    out = args.out
    if out.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing audit dir {out}; use a new --out."
        )
    out.mkdir(parents=True)

    total = 0
    matched = 0
    mismatch_rows: List[Dict[str, Any]] = []
    details_path = out / "details.jsonl"
    details_handle = details_path.open("w", encoding="utf-8")

    per_dataset: Dict[str, Dict[str, int]] = {}

    for ds, run_dir in inputs:
        # Verify each run's config points at the same model (tokenizer identity).
        ds_cfg = _read_yaml_config(run_dir)
        if ds_cfg.get("generator_model_path") != model_path:
            raise ValueError(
                f"{ds} config generator_model_path differs: "
                f"{ds_cfg.get('generator_model_path')!r} vs {model_path!r}"
            )

        rows = _read_json(run_dir / "intermediate_data.json")
        counters = {"qid_order": 0, "retrieval": 0, "kg": 0, "prompt": 0}
        seen_qids: set = set()

        for idx, row in enumerate(rows):
            total += 1
            qid = row.get("id")
            question = row.get("question") or ""
            output = row.get("output") or {}
            retrieval_result = output.get("retrieval_result")
            kg_subgraphs = output.get("kg_subgraphs")
            stored_prompt = output.get("prompt")

            qid_ok = bool(qid) and qid not in seen_qids
            if qid_ok:
                seen_qids.add(qid)
            retrieval_ok = _valid_retrieval_result(retrieval_result)
            kg_ok = _valid_kg_subgraphs(kg_subgraphs)
            counters["qid_order"] += int(qid_ok)
            counters["retrieval"] += int(retrieval_ok)
            counters["kg"] += int(kg_ok)

            if not (retrieval_ok and kg_ok):
                details_handle.write(json.dumps({
                    "dataset": ds, "qid": qid, "status": "input_invalid",
                    "retrieval_ok": retrieval_ok, "kg_ok": kg_ok,
                }, ensure_ascii=False) + "\n")
                continue

            messages, rebuilt = rebuild_prompt(
                question, retrieval_result, kg_subgraphs, tokenizer,
                retrieval_topk=retrieval_topk, max_kg_triples=max_kg_triples,
                max_input_len=max_input_len,
            )
            exact = rebuilt == stored_prompt
            if exact:
                matched += 1
                counters["prompt"] += 1

            detail = {
                "dataset": ds,
                "qid": qid,
                "status": "match" if exact else "mismatch",
                "messages_sha256": _sha256_json(messages),
                "stored_prompt_sha256": _sha256_text(stored_prompt or ""),
                "rebuilt_prompt_sha256": _sha256_text(rebuilt),
                "first_diff": None if exact else _first_diff(stored_prompt or "", rebuilt),
            }
            details_handle.write(json.dumps(detail, ensure_ascii=False) + "\n")
            if not exact:
                mismatch_rows.append(detail)

        per_dataset[ds] = counters
        logger.info("%s: %d/%d prompt exact", ds, counters["prompt"], len(rows))

    details_handle.close()

    gate_pass = matched == total
    report = {
        "schema_version": "legacy-prompt-identity-1",
        "seed": args.seed,
        "total": total,
        "prompt_exact_match": matched,
        "gate_pass": gate_pass,
        "model_path": model_path,
        "tokenizer": {
            "name_or_path": getattr(tokenizer, "name_or_path", model_path),
            "vocab_size": tokenizer.vocab_size,
        },
        "prompt_params": {
            "retrieval_topk": retrieval_topk,
            "max_kg_triples": max_kg_triples,
            "max_input_len": max_input_len,
        },
        "per_dataset": per_dataset,
        "inputs": [{"dataset": ds, "run_dir": str(run_dir)} for ds, run_dir in inputs],
        "mismatches": mismatch_rows[:20],
        "mismatch_count": len(mismatch_rows),
    }
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    dump_manifest(
        out,
        extra={
            "experiment_id": out.name,
            "phase": "audit",
            "gate": "legacy_prompt_identity",
            "gate_pass": gate_pass,
            "prompt_exact_match": f"{matched}/{total}",
        },
        status="COMPLETE",
    )

    logger.info("Wrote %s / %s", details_path, out / "report.json")
    print(f"qid/order match       = {sum(c['qid_order'] for c in per_dataset.values())}/{total}")
    print(f"retrieval input match = {sum(c['retrieval'] for c in per_dataset.values())}/{total}")
    print(f"KG input match        = {sum(c['kg'] for c in per_dataset.values())}/{total}")
    print(f"prompt exact match    = {matched}/{total}")
    print(f"gate: {'PASS' if gate_pass else 'FAIL'}")

    if not gate_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
