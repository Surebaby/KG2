#!/usr/bin/env python
"""Train-only, zero-update audit of PPO exploration and reward rankability.

The audit draws multiple SFT-policy rollouts for a frozen cohort of training
questions.  It answers two questions before another paid PPO run is launched:

1. Does sampling expose answers that greedy decoding misses (oracle@K)?
2. Does the current, unchanged PPO reward rank correct rollouts above wrong ones?

No optimiser, critic, reference model, validation item, or test item is used.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch

from kgproweight.config import ProjectConfig, load_config
from kgproweight.data.parsers import parse_steps
from kgproweight.data.silver_dataset import SilverDatasetReader
from kgproweight.eval.metrics import compute_em, compute_f1
from kgproweight.eval.stats import paired_bootstrap
from kgproweight.kg.entity_linker import EntityLinker
from kgproweight.kg.wikidata_retriever import _QA_RELATION_FILTER, WikidataSubgraphRetriever
from kgproweight.retrieval.bootstrap import resolve_entity_cache_path
from kgproweight.reward.alpha_gate import AlphaGate
from kgproweight.reward.prm_annotator import PRMAnnotator
from kgproweight.reward.text_reward_model import build_text_reward_model
from kgproweight.training.phase3_ppo import (
    Phase3PPOConfig,
    _prepare_prompts,
    _step_logprobs_from_scores,
)
from kgproweight.training.reward_function import (
    KGProWeightRewardFunction,
    step_spans_over_ids,
)
from kgproweight.utils.logging import (
    artifact_identity,
    configure_logging,
    dump_manifest,
    get_logger,
    prepare_new_run_dir,
)
from kgproweight.utils.paths import index_dir, model_path
from kgproweight.utils.seed import set_seed


logger = get_logger(__name__)

STRATA = (
    "visible_kg",
    "visible_empty_kg",
    "hidden_kg",
    "hidden_empty_kg",
)
DEFAULT_QUOTAS = {
    "visible_kg": 30,
    "visible_empty_kg": 25,
    "hidden_kg": 25,
    "hidden_empty_kg": 20,
}


def _normalise(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).split())


def _gold_visible(gold: str, passages: Sequence[Mapping[str, Any]]) -> bool:
    needle = _normalise(gold)
    if not needle:
        return False
    haystack = " ".join(
        _normalise(str(p.get("contents") or p.get("text") or ""))
        for p in passages
        if isinstance(p, Mapping)
    )
    return bool(re.search(rf"(?:^| ){re.escape(needle)}(?: |$)", haystack))


def _stratum(row: Mapping[str, Any]) -> str:
    spec = row["spec"]
    visible = _gold_visible(spec.gold_answer, spec.retrieved_passages)
    has_kg = bool(spec.kg_subgraph)
    return (
        ("visible" if visible else "hidden")
        + ("_kg" if has_kg else "_empty_kg")
    )


def select_stratified(
    rows: Sequence[Mapping[str, Any]],
    quotas: Mapping[str, int],
    *,
    warmup_count: int,
    seed: int,
) -> Tuple[List[Mapping[str, Any]], List[Mapping[str, Any]], Dict[str, int]]:
    """Select a deterministic quota cohort plus disjoint warm-up questions."""

    grouped: Dict[str, List[Mapping[str, Any]]] = {key: [] for key in STRATA}
    seen: set[str] = set()
    for row in rows:
        qid = str(row["spec"].metadata.get("qid") or "")
        if not qid or qid in seen:
            raise ValueError(f"Every audit row needs one unique qid, got {qid!r}")
        seen.add(qid)
        grouped[_stratum(row)].append(row)

    cohort: List[Mapping[str, Any]] = []
    leftovers: List[Mapping[str, Any]] = []
    rng = random.Random(seed)
    for key in STRATA:
        items = sorted(grouped[key], key=lambda x: str(x["spec"].metadata["qid"]))
        rng.shuffle(items)
        need = int(quotas.get(key, 0))
        if len(items) < need:
            raise ValueError(
                f"Stratum {key} has {len(items)} rows, below requested quota {need}"
            )
        cohort.extend(items[:need])
        leftovers.extend(items[need:])

    # Shuffle the final order so scoring is not grouped by evidence condition.
    rng.shuffle(cohort)
    rng.shuffle(leftovers)
    if len(leftovers) < warmup_count:
        raise ValueError(
            f"Only {len(leftovers)} disjoint rows remain for {warmup_count} warm-ups"
        )
    warmup = leftovers[:warmup_count]
    availability = {key: len(grouped[key]) for key in STRATA}
    return cohort, warmup, availability


def _mean(values: Iterable[float]) -> float:
    vals = [float(x) for x in values]
    return sum(vals) / len(vals) if vals else 0.0


def _pairwise_correctness(candidates: Sequence[Mapping[str, Any]], reward_key: str) -> Dict[str, Any]:
    wins = ties = comparisons = 0
    by_qid: Dict[str, List[Mapping[str, Any]]] = {}
    for row in candidates:
        by_qid.setdefault(str(row["qid"]), []).append(row)
    for rows in by_qid.values():
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                if rows[i]["em"] == rows[j]["em"]:
                    continue
                correct, wrong = (rows[i], rows[j]) if rows[i]["em"] > rows[j]["em"] else (rows[j], rows[i])
                comparisons += 1
                if correct[reward_key] > wrong[reward_key]:
                    wins += 1
                elif correct[reward_key] == wrong[reward_key]:
                    ties += 1
    accuracy = (wins + 0.5 * ties) / comparisons if comparisons else None
    return {
        "accuracy": accuracy,
        "wins": wins,
        "ties": ties,
        "comparisons": comparisons,
    }


def _aggregate_rankability(rows: Sequence[Mapping[str, Any]], bootstrap_seed: int) -> Dict[str, Any]:
    greedy = {str(r["qid"]): r for r in rows if r["candidate_type"] == "greedy"}
    sampled: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        if row["candidate_type"] == "sampled":
            sampled.setdefault(str(row["qid"]), []).append(row)
    qids = sorted(set(greedy) & set(sampled))
    if not qids:
        raise ValueError("Rankability summary has no qids with greedy and sampled candidates")

    vectors: Dict[str, List[float]] = {
        "greedy_em": [], "greedy_f1": [], "sample_em": [], "sample_f1": [],
        "oracle_em": [], "oracle_f1": [], "full_em": [], "full_f1": [],
        "process_em": [], "process_f1": [],
    }
    unique_ratios: List[float] = []
    all_candidates: List[Mapping[str, Any]] = []
    for qid in qids:
        candidates = sorted(sampled[qid], key=lambda x: int(x["candidate_index"]))
        all_candidates.extend(candidates)
        g = greedy[qid]
        best_full = max(candidates, key=lambda x: (float(x["full_reward"]), -int(x["candidate_index"])))
        best_process = max(candidates, key=lambda x: (float(x["process_reward"]), -int(x["candidate_index"])))
        vectors["greedy_em"].append(float(g["em"]))
        vectors["greedy_f1"].append(float(g["f1"]))
        vectors["sample_em"].append(_mean(x["em"] for x in candidates))
        vectors["sample_f1"].append(_mean(x["f1"] for x in candidates))
        vectors["oracle_em"].append(max(float(x["em"]) for x in candidates))
        vectors["oracle_f1"].append(max(float(x["f1"]) for x in candidates))
        vectors["full_em"].append(float(best_full["em"]))
        vectors["full_f1"].append(float(best_full["f1"]))
        vectors["process_em"].append(float(best_process["em"]))
        vectors["process_f1"].append(float(best_process["f1"]))
        answers = {_normalise(str(x["predicted_answer"])) for x in candidates}
        unique_ratios.append(len(answers) / len(candidates))

    metrics = {key: _mean(value) for key, value in vectors.items()}
    metrics.update({
        "n_qids": len(qids),
        "sample_candidates": len(all_candidates),
        "valid_rate": _mean(float(x["trajectory_valid"]) for x in all_candidates),
        "answer_unique_ratio": _mean(unique_ratios),
        "full_pairwise": _pairwise_correctness(all_candidates, "full_reward"),
        "process_pairwise": _pairwise_correctness(all_candidates, "process_reward"),
        "oracle_em_minus_greedy_ci": paired_bootstrap(
            vectors["oracle_em"], vectors["greedy_em"], seed=bootstrap_seed,
        ),
        "full_em_minus_greedy_ci": paired_bootstrap(
            vectors["full_em"], vectors["greedy_em"], seed=bootstrap_seed + 1,
        ),
        "process_em_minus_greedy_ci": paired_bootstrap(
            vectors["process_em"], vectors["greedy_em"], seed=bootstrap_seed + 2,
        ),
    })
    return metrics


def summarize_rankability(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_seed: int = 42,
) -> Dict[str, Any]:
    """Summarise overall and pre-registered evidence strata."""

    overall = _aggregate_rankability(rows, bootstrap_seed)
    strata: Dict[str, Any] = {}
    for offset, key in enumerate(STRATA, start=10):
        subset = [row for row in rows if row["stratum"] == key]
        if subset:
            strata[key] = _aggregate_rankability(subset, bootstrap_seed + offset)

    pair_acc = overall["full_pairwise"]["accuracy"]
    gates = {
        "exploration_headroom": {
            "threshold": 0.05,
            "observed": overall["oracle_em"] - overall["greedy_em"],
        },
        "reward_selected_gain": {
            "threshold": 0.02,
            "observed": overall["full_em"] - overall["greedy_em"],
        },
        "reward_pairwise_accuracy": {
            "threshold": 0.60,
            "observed": pair_acc,
        },
        "sample_valid_rate": {
            "threshold": 0.90,
            "observed": overall["valid_rate"],
        },
    }
    for gate in gates.values():
        gate["passed"] = gate["observed"] is not None and gate["observed"] >= gate["threshold"]

    if not gates["exploration_headroom"]["passed"]:
        diagnosis = "EXPLORATION_OR_POLICY_BOTTLENECK"
    elif not (
        gates["reward_selected_gain"]["passed"]
        and gates["reward_pairwise_accuracy"]["passed"]
    ):
        diagnosis = "REWARD_RANKABILITY_BOTTLENECK"
    else:
        diagnosis = "AUDIT_GATES_PASS"
    return {"overall": overall, "strata": strata, "gates": gates, "diagnosis": diagnosis}


def _load_phase3_config(path: str) -> Tuple[Phase3PPOConfig, Any]:
    cfg_doc = load_config(path, validate=ProjectConfig)
    tcfg, ppo = cfg_doc.training, cfg_doc.training.ppo
    cfg = Phase3PPOConfig(
        silver_path=tcfg.silver_path,
        output_dir=tcfg.output_dir,
        base_model=tcfg.base_model,
        sft_checkpoint=tcfg.sft_checkpoint,
        alpha_gate_path=tcfg.alpha_gate_path,
        text_reward_backend=cfg_doc.reward.text_reward_backend,
        text_reward_fallback_path=cfg_doc.reward.text_reward_fallback_path,
        dtype=tcfg.dtype,
        seed=tcfg.seed,
        learning_rate=ppo.learning_rate,
        batch_size=ppo.batch_size,
        mini_batch_size=ppo.mini_batch_size,
        ppo_epochs=ppo.ppo_epochs,
        cliprange=ppo.cliprange,
        cliprange_value=ppo.cliprange_value,
        kl_coef=ppo.kl_coef,
        gamma=ppo.gamma,
        lam=ppo.lam,
        max_grad_norm=ppo.max_grad_norm,
        total_steps=ppo.total_ppo_steps,
        vf_coef=ppo.vf_coef,
        value_head_init=ppo.value_head_init,
        value_head_dropout=ppo.value_head_dropout,
        health_guard_after_steps=ppo.health_guard_after_steps,
        health_guard_window=ppo.health_guard_window,
        health_guard_min_valid_rate=ppo.health_guard_min_valid_rate,
        health_guard_max_length_capped_frac=ppo.health_guard_max_length_capped_frac,
        health_guard_max_mean_kl=ppo.health_guard_max_mean_kl,
        target_kl=ppo.target_kl,
        kl_horizon=ppo.kl_horizon,
        early_stopping=ppo.early_stopping,
        save_every_steps=ppo.save_every_steps,
        outcome_weight=ppo.outcome_weight,
        text_reward_scale=ppo.text_reward_scale,
        step_reward_scale=ppo.step_reward_scale,
        pure_em_reward=ppo.pure_em_reward,
        min_valid_steps=ppo.min_valid_steps,
        min_reasoning_chars=ppo.min_reasoning_chars,
        shortfall_coef=ppo.shortfall_coef,
        target_steps=ppo.target_steps,
        center_text_reward=ppo.center_text_reward,
        text_baseline_momentum=ppo.text_baseline_momentum,
        sft_anchor_weight=ppo.sft_anchor_weight,
        sft_anchor_interval=ppo.sft_anchor_interval,
        sft_replay_ratio=ppo.sft_replay_ratio,
        log_with=ppo.log_with,
        max_new_tokens=ppo.max_new_tokens,
        temperature=ppo.temperature,
        top_p=ppo.top_p,
        rollout_chunk_size=ppo.rollout_chunk_size,
        max_input_length=tcfg.max_input_length,
        max_steps=ppo.max_steps,
        ppo_max_passages=ppo.ppo_max_passages,
        ppo_min_kg_triples=ppo.ppo_min_kg_triples,
        ppo_max_kg_triples=ppo.ppo_max_kg_triples,
        question_kg_index_path=tcfg.question_kg_index_path,
        max_kg_index_miss_rate=tcfg.max_kg_index_miss_rate,
        require_exact_kg_index_alignment=tcfg.require_exact_kg_index_alignment,
        passage_overrides_path=tcfg.passage_overrides_path,
        rollout_schedule_path=tcfg.rollout_schedule_path,
        split=tcfg.split,
        split_allow_none=tcfg.split_allow_none,
        val_ratio=tcfg.val_ratio,
        test_ratio=tcfg.test_ratio,
        split_seed=tcfg.split_seed,
        alpha_override=tcfg.alpha_override,
        binary_labels_only=tcfg.binary_labels_only,
        use_real_logprobs=cfg_doc.reward.use_real_logprobs,
        use_lora=True,
        lora_r=tcfg.lora_r,
        lora_alpha=tcfg.lora_alpha,
        lora_dropout=tcfg.lora_dropout,
    )
    return cfg, cfg_doc


def _load_jsonl_by_qid(path: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row.get("qid") or "")
            if not qid or qid in out:
                raise ValueError(f"Invalid or duplicate override qid={qid!r}")
            if not isinstance(row.get("retrieved_passages"), list):
                raise ValueError(f"Override qid={qid} has no retrieved_passages list")
            out[qid] = row
    return out


def _load_question_kg(path: str) -> Dict[str, List[Tuple[str, str, str]]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    is_v2 = bool(raw and "builder_version" in raw[0])
    result: Dict[str, List[Tuple[str, str, str]]] = {}
    for entry in raw:
        question = str(entry.get("question", entry.get("q", "")))
        triples = entry.get("triples", []) if is_v2 else entry.get("t", [])
        if is_v2:
            result[question] = [(str(t["h"]), str(t["r"]), str(t["t"])) for t in triples]
        else:
            result[question] = [tuple(map(str, t)) for t in triples if len(t) == 3]
    return result


def _generate_mode(
    policy,
    tokenizer,
    prompts: Sequence[str],
    cfg: Phase3PPOConfig,
    *,
    do_sample: bool,
) -> Tuple[List[torch.Tensor], List[str], List[Any]]:
    """Generate like PPO, with an explicit greedy/sample mode."""

    response_tensors: List[torch.Tensor] = []
    response_texts: List[str] = []
    logprobs_per_step_list: List[Any] = []
    device = next(policy.parameters()).device
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    chunk = max(1, int(cfg.rollout_chunk_size))
    for lo in range(0, len(prompts), chunk):
        group = list(prompts[lo:lo + chunk])
        for prompt in group:
            n_tokens = len(tokenizer(prompt, truncation=False, add_special_tokens=False)["input_ids"])
            if n_tokens > cfg.max_input_length:
                raise ValueError(f"Prepared audit prompt exceeds budget: {n_tokens}>{cfg.max_input_length}")
        previous_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        try:
            encoded = tokenizer(
                group, return_tensors="pt", truncation=False, padding=True,
                add_special_tokens=False,
            )
        finally:
            tokenizer.padding_side = previous_side
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        prompt_len = input_ids.size(1)
        kwargs: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": cfg.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": pad_id,
            "use_cache": True,
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        if do_sample:
            kwargs.update(temperature=cfg.temperature, top_p=cfg.top_p, top_k=0)
        with torch.no_grad():
            output = policy.generate(**kwargs)
        for row in range(len(group)):
            response_ids = output.sequences[row][prompt_len:]
            non_pad = (response_ids != pad_id).nonzero()
            response_ids = (
                response_ids[: int(non_pad[-1].item()) + 1]
                if non_pad.numel() else response_ids[:1]
            )
            text = tokenizer.decode(response_ids, skip_special_tokens=True)
            n_steps = len(parse_steps(text)[: cfg.max_steps])
            spans = step_spans_over_ids(response_ids, tokenizer, n_steps)
            logprobs = _step_logprobs_from_scores(response_ids, output.scores, spans, row=row)
            response_tensors.append(response_ids.detach().cpu())
            response_texts.append(text)
            logprobs_per_step_list.append(logprobs)
        del output
    return response_tensors, response_texts, logprobs_per_step_list


def _score_one(
    reward_fn: KGProWeightRewardFunction,
    row: Mapping[str, Any],
    response_ids: torch.Tensor,
    response: str,
    logprobs: Any,
    *,
    candidate_type: str,
    candidate_index: int,
    stratum: str,
    generation_cap: int,
) -> Dict[str, Any]:
    spec = row["spec"]
    n_steps = len(parse_steps(response, known_kg=spec.kg_subgraph)[: reward_fn.max_steps])
    spans = step_spans_over_ids(response_ids, reward_fn.tokenizer, n_steps)
    info = reward_fn(
        prompt=row["prompt"], response=response, spec=spec,
        logprobs_per_step=logprobs, response_ids=response_ids, step_spans=spans,
    )
    records = list(info["per_step_records"])
    process_reward = sum(
        (
            float(record.alpha) * float(record.r_kg)
            + (1.0 - float(record.alpha))
            * float(record.r_text_used)
            * float(reward_fn.composite.text_reward_scale)
        )
        * float(reward_fn.composite.step_reward_scale)
        for record in records
    )
    steps = parse_steps(response, known_kg=spec.kg_subgraph)[: reward_fn.max_steps]
    valid = reward_fn._is_valid_trajectory(
        steps, response, min_steps=reward_fn.min_valid_steps,
        min_reasoning_chars=reward_fn.min_reasoning_chars,
    )
    predicted = str(info["predicted_answer"])
    em = compute_em(predicted, [spec.gold_answer])
    f1 = compute_f1(predicted, [spec.gold_answer])
    outcome = reward_fn.composite.outcome_weight * em if valid else 0.0
    invalid = 0.0 if valid else -reward_fn.composite.outcome_weight
    shortfall = -(
        reward_fn.composite.shortfall_coef
        * reward_fn.composite.outcome_weight
        * max(0, reward_fn.composite.target_steps - len(steps))
        / reward_fn.composite.target_steps
    )
    expected = process_reward + outcome + invalid + shortfall
    full_reward = float(info["trajectory_reward"])
    if abs(expected - full_reward) > 1e-4:
        raise AssertionError(
            f"Reward decomposition mismatch qid={spec.metadata.get('qid')}: "
            f"full={full_reward:.6f}, reconstructed={expected:.6f}"
        )
    return {
        "qid": str(spec.metadata["qid"]),
        "question": spec.query,
        "gold_answer": spec.gold_answer,
        "stratum": stratum,
        "candidate_type": candidate_type,
        "candidate_index": candidate_index,
        "response": response,
        "predicted_answer": predicted,
        "em": em,
        "f1": f1,
        "trajectory_valid": valid,
        "n_steps": len(steps),
        "response_tokens": int(response_ids.numel()),
        "length_capped": int(response_ids.numel()) >= int(generation_cap),
        "full_reward": full_reward,
        "process_reward": process_reward,
        "outcome_component": outcome,
        "invalid_component": invalid,
        "shortfall_component": shortfall,
        "per_step_records": [asdict(record) for record in records],
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--rollouts-per-qid", type=int, default=4)
    parser.add_argument("--warmup-qids", type=int, default=20)
    parser.add_argument("--cohort-seed", type=int, default=20260828)
    parser.add_argument("--generation-seed", type=int, default=42)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument(
        "--min-gpu-memory-gib", type=float, default=40.0,
        help="Fail before reserving the Experiment ID when policy+ReaRAG cannot safely co-reside.",
    )
    parser.add_argument("--quota-visible-kg", type=int, default=30)
    parser.add_argument("--quota-visible-empty", type=int, default=25)
    parser.add_argument("--quota-hidden-kg", type=int, default=25)
    parser.add_argument("--quota-hidden-empty", type=int, default=20)
    parser.add_argument(
        "--preflight-only", action="store_true",
        help="Validate files/CUDA/config without reserving an output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.rollouts_per_qid < 2:
        raise ValueError("rollouts-per-qid must be >=2 for a ranking audit")
    quotas = {
        "visible_kg": args.quota_visible_kg,
        "visible_empty_kg": args.quota_visible_empty,
        "hidden_kg": args.quota_hidden_kg,
        "hidden_empty_kg": args.quota_hidden_empty,
    }
    cfg, _ = _load_phase3_config(args.config)
    required = {
        "silver": cfg.silver_path,
        "sft_checkpoint": cfg.sft_checkpoint,
        "alpha_gate": cfg.alpha_gate_path,
        "question_kg_index": cfg.question_kg_index_path,
        "passage_overrides": cfg.passage_overrides_path,
    }
    for label, value in required.items():
        if not value or not Path(value).exists():
            raise FileNotFoundError(f"Required {label} artifact is missing: {value}")
    if cfg.split != "train":
        raise ValueError(f"Audit is train-only; config split must be 'train', got {cfg.split!r}")
    if not cfg.use_real_logprobs:
        raise ValueError("Audit requires reward.use_real_logprobs=true to match PPO alpha features")
    if not torch.cuda.is_available():
        raise RuntimeError("Rankability audit requires CUDA; no Experiment ID was reserved")
    if cfg.dtype == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Config requests bf16 but the active GPU does not support it")
    gpu_memory_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    if gpu_memory_gib < args.min_gpu_memory_gib:
        raise RuntimeError(
            f"Audit needs at least {args.min_gpu_memory_gib:.1f} GiB so the 8B policy, "
            f"9B ReaRAG reward model, KV cache and generation scores co-reside; "
            f"active GPU has {gpu_memory_gib:.1f} GiB. No Experiment ID was reserved."
        )

    base_path = model_path(cfg.base_model)
    rearag_path = model_path("rearag")
    for label, value in (("base model", base_path), ("ReaRAG", rearag_path)):
        if not Path(value).exists():
            raise FileNotFoundError(f"Audit requires a local {label}: {value}")
    if args.preflight_only:
        print(json.dumps({
            "status": "PREFLIGHT_OK", "split": cfg.split, "quotas": quotas,
            "gpu_memory_gib": gpu_memory_gib,
            "planned_generations": args.warmup_qids + sum(quotas.values()) * (1 + args.rollouts_per_qid),
            "config": str(Path(args.config).resolve()),
        }, indent=2, ensure_ascii=False))
        return

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    overrides = _load_jsonl_by_qid(str(cfg.passage_overrides_path))
    question_kg = _load_question_kg(str(cfg.question_kg_index_path))
    reader = SilverDatasetReader(
        cfg.silver_path, split=cfg.split,
        split_spec=cfg.build_split_spec(),
    )
    prepared = _prepare_prompts(
        reader, tokenizer, cfg, question_kg_index=question_kg,
        passage_overrides=overrides,
    )
    eligible = [row for row in prepared if row["spec"].metadata["passage_override_applied"]]
    cohort, warmup, availability = select_stratified(
        eligible, quotas, warmup_count=args.warmup_qids, seed=args.cohort_seed,
    )

    protocol = {
        "name": "ppo_reward_rankability_train_only_v1",
        "zero_update": True,
        "policy": "SFT adapter",
        "split": "train",
        "cohort_quotas": quotas,
        "cohort_seed": args.cohort_seed,
        "warmup_qids": args.warmup_qids,
        "rollouts_per_qid": args.rollouts_per_qid,
        "generation_seed": args.generation_seed,
        "sampling": {"temperature": cfg.temperature, "top_p": cfg.top_p, "top_k": 0},
        "max_new_tokens": cfg.max_new_tokens,
        "minimum_gpu_memory_gib": args.min_gpu_memory_gib,
        "text_baseline": "warm on disjoint greedy qids, then freeze",
        "gates": {"oracle_em_gain": 0.05, "reward_top1_em_gain": 0.02, "pairwise_accuracy": 0.60, "valid_rate": 0.90},
    }
    run_record = {
        "phase": "ppo_reward_rankability_audit",
        "protocol": protocol,
        "phase3_config": asdict(cfg),
        "stratum_availability": availability,
        "input_artifacts": {
            **{key: artifact_identity(value) for key, value in required.items()},
            "base_model": artifact_identity(base_path),
            "rearag": artifact_identity(rearag_path),
            "source_config": artifact_identity(args.config),
        },
    }
    out_dir, experiment_id = prepare_new_run_dir(
        args.output_dir, experiment_id=args.experiment_id, extra=run_record,
    )
    try:
        cohort_rows = [{
            "qid": row["spec"].metadata["qid"], "question": row["spec"].query,
            "gold_answer": row["spec"].gold_answer, "stratum": _stratum(row),
            "num_passages": row["num_passages"], "prompt_tokens": row["prompt_tokens"],
            "kg_triples": len(row["spec"].kg_subgraph),
        } for row in cohort]
        _write_jsonl(out_dir / "cohort.jsonl", cohort_rows)

        dtype = torch.bfloat16 if cfg.dtype == "bf16" else torch.float16
        base = AutoModelForCausalLM.from_pretrained(
            base_path, torch_dtype=dtype, device_map="auto",
        )
        policy = PeftModel.from_pretrained(base, cfg.sft_checkpoint, is_trainable=False)
        policy.eval()

        alpha_gate = AlphaGate()
        alpha_gate.load_state_dict(torch.load(cfg.alpha_gate_path, map_location="cpu"))
        alpha_gate.eval()
        linker = EntityLinker(cache_path=resolve_entity_cache_path(), offline=True)
        text_reward = build_text_reward_model(
            backend=cfg.text_reward_backend, fallback_head_path=cfg.text_reward_fallback_path,
            device="cuda", dtype=cfg.dtype,
        )
        reward_fn = KGProWeightRewardFunction(
            alpha_gate=alpha_gate,
            prm_annotator=PRMAnnotator(entity_linker=linker, verbose=False),
            text_reward_model=text_reward,
            tokenizer=tokenizer,
            outcome_weight=cfg.outcome_weight,
            discount=cfg.gamma,
            alpha_override=cfg.alpha_override,
            max_steps=cfg.max_steps,
            text_reward_scale=cfg.text_reward_scale,
            min_valid_steps=cfg.min_valid_steps,
            min_reasoning_chars=cfg.min_reasoning_chars,
            step_reward_scale=cfg.step_reward_scale,
            shortfall_coef=cfg.shortfall_coef,
            target_steps=cfg.target_steps,
            center_text_reward=cfg.center_text_reward,
            text_baseline_momentum=cfg.text_baseline_momentum,
            subgraph_retriever=WikidataSubgraphRetriever(
                max_hops=2, max_neighbors=30,
                cache_dir=str(Path(index_dir()) / "kg_cache"), offline=True,
                relation_filter=_QA_RELATION_FILTER,
            ),
            pure_em=cfg.pure_em_reward,
        )

        # Warm the causal EMA on disjoint train questions, then freeze it. The
        # cohort's score cannot depend on candidate scoring order after this.
        warm_ids, warm_texts, warm_lps = _generate_mode(
            policy, tokenizer, [row["prompt"] for row in warmup], cfg, do_sample=False,
        )
        warmup_log: List[Dict[str, Any]] = []
        for row, ids, text, lps in zip(warmup, warm_ids, warm_texts, warm_lps):
            _score_one(
                reward_fn, row, ids, text, lps, candidate_type="warmup",
                candidate_index=0, stratum=_stratum(row), generation_cap=cfg.max_new_tokens,
            )
            warmup_log.append({"qid": row["spec"].metadata["qid"], "stratum": _stratum(row)})
        baseline = reward_fn.composite.text_baseline
        baseline_observations = reward_fn.composite.text_baseline_n_obs
        reward_fn.composite.text_baseline_momentum = 1.0
        _write_jsonl(out_dir / "warmup.jsonl", warmup_log)

        scored: List[Dict[str, Any]] = []
        greedy_ids, greedy_texts, greedy_lps = _generate_mode(
            policy, tokenizer, [row["prompt"] for row in cohort], cfg, do_sample=False,
        )
        for row, ids, text, lps in zip(cohort, greedy_ids, greedy_texts, greedy_lps):
            scored.append(_score_one(
                reward_fn, row, ids, text, lps, candidate_type="greedy",
                candidate_index=0, stratum=_stratum(row), generation_cap=cfg.max_new_tokens,
            ))

        expanded = [(row, candidate_index) for row in cohort for candidate_index in range(args.rollouts_per_qid)]
        set_seed(args.generation_seed)
        sample_ids, sample_texts, sample_lps = _generate_mode(
            policy, tokenizer, [row["prompt"] for row, _ in expanded], cfg,
            do_sample=True,
        )
        for (row, candidate_index), ids, text, lps in zip(expanded, sample_ids, sample_texts, sample_lps):
            scored.append(_score_one(
                reward_fn, row, ids, text, lps, candidate_type="sampled",
                candidate_index=candidate_index, stratum=_stratum(row),
                generation_cap=cfg.max_new_tokens,
            ))
        _write_jsonl(out_dir / "rollouts.jsonl", scored)

        summary = summarize_rankability(scored, bootstrap_seed=args.bootstrap_seed)
        summary.update({
            "experiment_id": experiment_id,
            "protocol": protocol,
            "stratum_availability": availability,
            "text_baseline_frozen_value": baseline,
            "text_baseline_warmup_observations": baseline_observations,
        })
        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        dump_manifest(
            out_dir,
            extra={
                **run_record,
                "experiment_id": experiment_id,
                "text_baseline_frozen_value": baseline,
                "text_baseline_warmup_observations": baseline_observations,
                "summary": summary,
            },
            status="COMPLETE",
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    except Exception as exc:
        dump_manifest(
            out_dir,
            extra={
                **run_record,
                "experiment_id": experiment_id,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            },
            status="FAILED",
        )
        raise


if __name__ == "__main__":
    configure_logging("INFO")
    main()
