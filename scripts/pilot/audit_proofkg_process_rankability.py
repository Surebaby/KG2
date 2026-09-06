#!/usr/bin/env python
"""Zero-update rankability audit for automatic ProofKG process evidence.

The process score is gold-free and fixed before sampled candidates are generated.
Gold answers are used only after scoring to measure correct-vs-wrong ranking.
This script does not modify the production reward implementation.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import traceback
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from kgproweight.data.parsers import extract_final_answer, parse_steps
from kgproweight.data.prompts import build_rl_messages
from kgproweight.eval.metrics import compute_em, compute_f1
from kgproweight.training.reward_function import KGProWeightRewardFunction
from kgproweight.utils.logging import artifact_identity, dump_manifest, prepare_new_run_dir


SCORER_VERSION = "proofkg-grounded-process-score-1"
COMPARISON_CUES = re.compile(
    r"\b(which|who)\b.*\b(first|earlier|later|older|younger|more|less|higher|lower|"
    r"longer|shorter|before|after)\b|\b(first|earlier|later|older|younger)\b",
    re.IGNORECASE,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _norm(value: object) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").casefold()))


def _phrase_in(phrase: object, text: object) -> bool:
    needle = _norm(phrase)
    haystack = _norm(text)
    return bool(needle and re.search(rf"(?:^| ){re.escape(needle)}(?: |$)", haystack))


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _reachable_cited_edges(
    question: str,
    kg: Sequence[Sequence[str]],
    cited_by_step: Sequence[Sequence[tuple[str, str, str]]],
) -> tuple[set[tuple[str, str, str]], set[str]]:
    """Return cited edges reachable from a question-mentioned KG node."""

    edges = [tuple(str(value).strip() for value in edge) for edge in kg if len(edge) == 3]
    anchors = {
        node
        for head, _, tail in edges
        for node in (head, tail)
        if _phrase_in(node, question)
    }
    reachable_nodes = set(anchors)
    reachable_edges: set[tuple[str, str, str]] = set()
    for step_citations in cited_by_step:
        pending = [tuple(value) for value in step_citations]
        changed = True
        while changed:
            changed = False
            for edge in pending:
                if edge in reachable_edges:
                    continue
                head, _, tail = edge
                if head in reachable_nodes:
                    reachable_edges.add(edge)
                    reachable_nodes.add(tail)
                    changed = True
    return reachable_edges, reachable_nodes


def score_process_candidate(
    *,
    question: str,
    kg: Sequence[Sequence[str]],
    generation: str,
) -> dict[str, Any]:
    """Compute the frozen, gold-free ProofKG process score.

    Primary score (valid trajectories only):
      0.25 exact-citation precision
    + 0.25 conclusion grounding
    + 0.30 reachable ProofKG edge coverage
    + 0.20 final-answer/path alignment
    - 0.50 unknown-citation ratio
    - 0.15 duplicate-citation ratio

    Invalid trajectories receive -1.0.  No gold answer enters this function.
    """

    triples = [tuple(str(value).strip() for value in edge) for edge in kg if len(edge) == 3]
    steps = parse_steps(generation, known_kg=triples)
    answer = (extract_final_answer(generation) or "").split("\n", 1)[0].strip()
    indices = [step.index for step in steps]
    valid = KGProWeightRewardFunction._is_valid_trajectory(
        steps, generation, min_steps=3, min_reasoning_chars=20
    )

    known = [triple for step in steps for triple in step.cited_triples]
    unknown_count = sum(len(step.unknown_citation_surfaces) for step in steps)
    attempts = len(known) + unknown_count
    citation_precision = _ratio(len(known), attempts)

    grounded = 0
    for step in steps:
        conclusion = step.intermediate_conclusion or ""
        for head, _, tail in step.cited_triples:
            grounded += int(_phrase_in(head, conclusion) or _phrase_in(tail, conclusion))
    conclusion_grounding = _ratio(grounded, len(known))

    reachable_edges, reachable_nodes = _reachable_cited_edges(
        question, triples, [step.cited_triples for step in steps]
    )
    edge_coverage = _ratio(len(reachable_edges), len(set(triples)))

    outgoing = {head for head, _, _ in triples}
    terminal_nodes = {tail for _, _, tail in triples if tail not in outgoing}
    if COMPARISON_CUES.search(question):
        supported_answers = {
            head for head, _, _ in reachable_edges if _phrase_in(head, question)
        }
    else:
        supported_answers = terminal_nodes.intersection(reachable_nodes)
    answer_alignment = float(
        bool(answer)
        and any(_phrase_in(answer, value) or _phrase_in(value, answer) for value in supported_answers)
    )

    unknown_ratio = _ratio(unknown_count, attempts)
    duplicate_ratio = _ratio(len(known) - len(set(known)), len(known))
    score = (
        0.25 * citation_precision
        + 0.25 * conclusion_grounding
        + 0.30 * edge_coverage
        + 0.20 * answer_alignment
        - 0.50 * unknown_ratio
        - 0.15 * duplicate_ratio
        if valid
        else -1.0
    )
    return {
        "scorer_version": SCORER_VERSION,
        "score": float(score),
        "trajectory_valid": bool(valid),
        "prediction": answer,
        "n_steps": len(steps),
        "known_citations": len(known),
        "unknown_citations": unknown_count,
        "components": {
            "citation_precision": citation_precision,
            "conclusion_grounding": conclusion_grounding,
            "reachable_edge_coverage": edge_coverage,
            "answer_path_alignment": answer_alignment,
            "unknown_citation_ratio": unknown_ratio,
            "duplicate_citation_ratio": duplicate_ratio,
        },
    }


def _pairwise(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_qid: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in candidates:
        if row["candidate_type"] == "sampled":
            by_qid[str(row["qid"])].append(row)
    wins = ties = comparisons = 0
    mixed_qids = 0
    for rows in by_qid.values():
        local = 0
        for left_index in range(len(rows)):
            for right_index in range(left_index + 1, len(rows)):
                left, right = rows[left_index], rows[right_index]
                if float(left["em"]) == float(right["em"]):
                    continue
                correct, wrong = (left, right) if left["em"] > right["em"] else (right, left)
                comparisons += 1
                local += 1
                if correct["process"]["score"] > wrong["process"]["score"]:
                    wins += 1
                elif correct["process"]["score"] == wrong["process"]["score"]:
                    ties += 1
        mixed_qids += int(local > 0)
    return {
        "accuracy": _ratio(wins + 0.5 * ties, comparisons) if comparisons else None,
        "wins": wins,
        "ties": ties,
        "comparisons": comparisons,
        "mixed_outcome_qids": mixed_qids,
    }


def _bootstrap_delta(left: Sequence[float], right: Sequence[float], seed: int) -> dict[str, Any]:
    delta = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.empty(10_000, dtype=float)
    for index in range(len(draws)):
        positions = rng.integers(0, len(delta), len(delta))
        draws[index] = delta[positions].mean()
    return {
        "diff_mean": float(delta.mean()),
        "lower": float(np.quantile(draws, 0.025)),
        "upper": float(np.quantile(draws, 0.975)),
        "p_value": min(1.0, float(2 * min((draws <= 0).mean(), (draws >= 0).mean()))),
        "n": len(delta),
    }


def summarize(candidates: Sequence[Mapping[str, Any]], *, seed: int) -> dict[str, Any]:
    greedy = {str(row["qid"]): row for row in candidates if row["candidate_type"] == "greedy"}
    sampled: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in candidates:
        if row["candidate_type"] == "sampled":
            sampled[str(row["qid"])].append(row)
    qids = sorted(set(greedy).intersection(sampled))
    if len(qids) != 100 or any(len(sampled[qid]) != 4 for qid in qids):
        raise ValueError("Expected exactly 100 qids with one greedy and four sampled candidates")
    greedy_em = [float(greedy[qid]["em"]) for qid in qids]
    oracle_em = [max(float(row["em"]) for row in sampled[qid]) for qid in qids]
    selected = [
        max(
            sampled[qid],
            key=lambda row: (float(row["process"]["score"]), -int(row["candidate_index"])),
        )
        for qid in qids
    ]
    selected_em = [float(row["em"]) for row in selected]
    sampled_rows = [row for qid in qids for row in sampled[qid]]
    pair = _pairwise(candidates)
    metrics = {
        "n_qids": len(qids),
        "sampled_candidates": len(sampled_rows),
        "greedy_em": float(np.mean(greedy_em)),
        "oracle_at_4_em": float(np.mean(oracle_em)),
        "process_top1_em": float(np.mean(selected_em)),
        "sample_valid_rate": float(np.mean([row["process"]["trajectory_valid"] for row in sampled_rows])),
        "process_pairwise": pair,
        "oracle_minus_greedy_ci": _bootstrap_delta(oracle_em, greedy_em, seed),
        "process_top1_minus_greedy_ci": _bootstrap_delta(selected_em, greedy_em, seed + 1),
    }
    gates = {
        "exploration_headroom": metrics["oracle_at_4_em"] - metrics["greedy_em"] >= 0.05,
        "process_selected_gain": metrics["process_top1_em"] - metrics["greedy_em"] >= 0.02,
        "process_pairwise_accuracy": pair["accuracy"] is not None and pair["accuracy"] >= 0.60,
        "sample_valid_rate": metrics["sample_valid_rate"] >= 0.90,
    }
    return {"metrics": metrics, "gates": gates, "all_pass": all(gates.values())}


def _validate_hash(protocol: Mapping[str, Any], label: str, path: Path) -> None:
    expected = str(protocol["inputs"][label]["sha256"])
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA256 mismatch: expected {expected}, got {actual}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--proof_input", required=True)
    parser.add_argument("--question_kg_records", required=True)
    parser.add_argument("--greedy_predictions", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--experiment_id", required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol_path = Path(args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    paths = {
        "proof_input": Path(args.proof_input).resolve(),
        "question_kg_records": Path(args.question_kg_records).resolve(),
        "greedy_predictions": Path(args.greedy_predictions).resolve(),
        "adapter_model": Path(args.adapter).resolve() / "adapter_model.safetensors",
        "base_model_index": Path(args.base_model).resolve() / "model.safetensors.index.json",
    }
    for label, path in paths.items():
        _validate_hash(protocol, label, path)
    if protocol["process_scorer"]["version"] != SCORER_VERSION:
        raise ValueError("protocol scorer version does not match implementation")
    implementation_hash = _sha256(Path(__file__).resolve())
    if implementation_hash != protocol["process_scorer"]["implementation_sha256"]:
        raise ValueError(
            "frozen scorer implementation SHA256 mismatch: "
            f"expected {protocol['process_scorer']['implementation_sha256']}, "
            f"got {implementation_hash}"
        )

    proof_rows = _read_jsonl(paths["proof_input"])
    records = {str(row["qid"]): row for row in _read_jsonl(paths["question_kg_records"])}
    frozen_greedy = {
        str(row["qid"]): row
        for row in _read_jsonl(paths["greedy_predictions"])
        if row.get("arm") == "proof"
    }
    if len(proof_rows) != 100 or len(records) != 100 or len(frozen_greedy) != 100:
        raise ValueError("frozen audit requires 100 proof inputs, records, and proof-arm greedy rows")
    qids = [str(row["qid"]) for row in proof_rows]
    if set(qids) != set(records) or set(qids) != set(frozen_greedy):
        raise ValueError("qid mismatch across frozen inputs")

    run_record = {
        "phase": "proofkg_process_rankability_zero_update",
        "protocol": artifact_identity(protocol_path),
        "input_artifacts": {label: artifact_identity(path) for label, path in paths.items()},
        "production_reward_changed": False,
    }
    run_dir, experiment_id = prepare_new_run_dir(
        args.run_dir, experiment_id=args.experiment_id, extra=run_record
    )
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        seed = int(protocol["generation"]["seed"])
        max_new_tokens = int(protocol["generation"]["max_new_tokens"])
        rollouts_per_qid = int(protocol["generation"]["rollouts_per_qid"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        base_path = Path(args.base_model).resolve()
        adapter_path = Path(args.adapter).resolve()
        tokenizer = AutoTokenizer.from_pretrained(base_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            base_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()

        prompts: dict[str, str] = {}
        for row in proof_rows:
            messages = build_rl_messages(
                question=str(row["question"]),
                retrieved_passages=list(row["retrieved_passages"]),
                kg_triples=list(row["kg_subgraph"]),
                top_k=15,
            )
            prompts[str(row["qid"])] = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        candidates: list[dict[str, Any]] = []
        # Reuse deterministic frozen greedy text. If it exceeded the PPO cap,
        # token-truncate it exactly as max_new_tokens generation would.
        for row in proof_rows:
            qid = str(row["qid"])
            ids = tokenizer(
                str(frozen_greedy[qid]["generation"]),
                add_special_tokens=False,
            )["input_ids"]
            capped = len(ids) > max_new_tokens
            generation = tokenizer.decode(ids[:max_new_tokens], skip_special_tokens=True)
            process = score_process_candidate(
                question=str(row["question"]), kg=row["kg_subgraph"], generation=generation
            )
            golds = [str(value) for value in row["gold_answers"]]
            candidates.append({
                "qid": qid,
                "question": row["question"],
                "candidate_type": "greedy",
                "candidate_index": 0,
                "generation": generation,
                "generation_tokens": min(len(ids), max_new_tokens),
                "length_capped": capped,
                "process": process,
                "em": compute_em(process["prediction"], golds),
                "f1": compute_f1(process["prediction"], golds),
            })

        expanded = [
            (row, candidate_index)
            for row in proof_rows
            for candidate_index in range(rollouts_per_qid)
        ]
        completed = 0
        for start in range(0, len(expanded), args.batch_size):
            batch = expanded[start : start + args.batch_size]
            encoded = tokenizer(
                [prompts[str(row["qid"])] for row, _ in batch],
                return_tensors="pt",
                add_special_tokens=False,
                padding=True,
            ).to(model.device)
            prompt_length = int(encoded["input_ids"].shape[1])
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=float(protocol["generation"]["temperature"]),
                    top_p=float(protocol["generation"]["top_p"]),
                    top_k=0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            texts = tokenizer.batch_decode(
                generated[:, prompt_length:], skip_special_tokens=True
            )
            for (row, candidate_index), generation, token_ids in zip(
                batch, texts, generated[:, prompt_length:]
            ):
                qid = str(row["qid"])
                process = score_process_candidate(
                    question=str(row["question"]), kg=row["kg_subgraph"], generation=generation
                )
                golds = [str(value) for value in row["gold_answers"]]
                token_count = int((token_ids != tokenizer.pad_token_id).sum().item())
                candidates.append({
                    "qid": qid,
                    "question": row["question"],
                    "candidate_type": "sampled",
                    "candidate_index": candidate_index,
                    "generation": generation,
                    "generation_tokens": token_count,
                    "length_capped": token_count >= max_new_tokens,
                    "process": process,
                    "em": compute_em(process["prediction"], golds),
                    "f1": compute_f1(process["prediction"], golds),
                })
            completed += len(batch)
            if completed % 20 == 0 or completed == len(expanded):
                print(f"sampled {completed}/{len(expanded)}", flush=True)

        _write_jsonl(run_dir / "candidates.jsonl", candidates)
        summary = summarize(candidates, seed=int(protocol["bootstrap_seed"]))
        report = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "experiment_id": experiment_id,
            "status": "PASS" if summary["all_pass"] else "FAIL_STOP",
            "scope": protocol["scope"],
            "zero_update": True,
            "production_reward_changed": False,
            "summary": summary,
            "scientific_boundary": protocol["scientific_boundary"],
            "outputs": {"candidates": artifact_identity(run_dir / "candidates.jsonl")},
        }
        (run_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        dump_manifest(run_dir, status=report["status"], extra={**run_record, **report})
        print(json.dumps(report, ensure_ascii=False, indent=2))
    except Exception as exc:
        dump_manifest(
            run_dir,
            status="FAILED_RUNTIME",
            extra={
                **run_record,
                "failure": {"type": type(exc).__name__, "message": str(exc)},
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    main()
