from __future__ import annotations

from types import SimpleNamespace

from scripts.pilot.audit_ppo_reward_rankability import (
    DEFAULT_QUOTAS,
    STRATA,
    _load_phase3_config,
    select_stratified,
    summarize_rankability,
)


def _selection_row(qid: str, *, visible: bool, has_kg: bool):
    gold = "Douglas County"
    passages = [{"contents": "Douglas County is here." if visible else "No answer here."}]
    spec = SimpleNamespace(
        gold_answer=gold,
        retrieved_passages=passages,
        kg_subgraph=[("a", "r", "b")] if has_kg else [],
        metadata={"qid": qid},
    )
    return {"spec": spec}


def test_stratified_selection_is_exact_disjoint_and_deterministic():
    rows = []
    conditions = {
        "visible_kg": (True, True),
        "visible_empty_kg": (True, False),
        "hidden_kg": (False, True),
        "hidden_empty_kg": (False, False),
    }
    for key, (visible, has_kg) in conditions.items():
        for index in range(DEFAULT_QUOTAS[key] + 6):
            rows.append(_selection_row(f"{key}_{index}", visible=visible, has_kg=has_kg))

    first, warmup, availability = select_stratified(
        rows, DEFAULT_QUOTAS, warmup_count=20, seed=17,
    )
    second, warmup_second, _ = select_stratified(
        rows, DEFAULT_QUOTAS, warmup_count=20, seed=17,
    )
    first_qids = [row["spec"].metadata["qid"] for row in first]
    warmup_qids = [row["spec"].metadata["qid"] for row in warmup]
    assert len(first_qids) == 100
    assert len(set(first_qids)) == 100
    assert set(first_qids).isdisjoint(warmup_qids)
    assert first_qids == [row["spec"].metadata["qid"] for row in second]
    assert warmup_qids == [row["spec"].metadata["qid"] for row in warmup_second]
    assert availability == {key: DEFAULT_QUOTAS[key] + 6 for key in STRATA}


def _metric_row(
    qid: str,
    candidate_type: str,
    candidate_index: int,
    *,
    em: float,
    f1: float,
    full: float,
    process: float,
    answer: str,
):
    return {
        "qid": qid,
        "candidate_type": candidate_type,
        "candidate_index": candidate_index,
        "em": em,
        "f1": f1,
        "full_reward": full,
        "process_reward": process,
        "predicted_answer": answer,
        "trajectory_valid": True,
        "stratum": "visible_kg",
    }


def test_rankability_summary_separates_oracle_and_reward_selection():
    rows = [
        _metric_row("q1", "greedy", 0, em=0, f1=0, full=0, process=0, answer="x"),
        _metric_row("q1", "sampled", 0, em=0, f1=0, full=3, process=3, answer="a"),
        _metric_row("q1", "sampled", 1, em=1, f1=1, full=2, process=4, answer="gold"),
        _metric_row("q1", "sampled", 2, em=0, f1=.5, full=1, process=1, answer="b"),
        _metric_row("q1", "sampled", 3, em=0, f1=0, full=0, process=0, answer="a"),
        _metric_row("q2", "greedy", 0, em=1, f1=1, full=0, process=0, answer="gold2"),
        _metric_row("q2", "sampled", 0, em=1, f1=1, full=5, process=5, answer="gold2"),
        _metric_row("q2", "sampled", 1, em=0, f1=0, full=4, process=4, answer="z"),
        _metric_row("q2", "sampled", 2, em=0, f1=0, full=3, process=3, answer="z"),
        _metric_row("q2", "sampled", 3, em=0, f1=0, full=2, process=2, answer="y"),
    ]

    summary = summarize_rankability(rows, bootstrap_seed=3)
    overall = summary["overall"]
    assert overall["greedy_em"] == 0.5
    assert overall["sample_em"] == 0.25
    assert overall["oracle_em"] == 1.0
    # Full reward picks q1's wrong candidate and q2's correct candidate.
    assert overall["full_em"] == 0.5
    # Process-only reward picks the correct candidate for both qids.
    assert overall["process_em"] == 1.0
    assert overall["full_pairwise"]["comparisons"] == 6
    assert overall["full_pairwise"]["accuracy"] == 5 / 6
    assert summary["diagnosis"] == "REWARD_RANKABILITY_BOTTLENECK"


def test_rankability_gate_reports_missing_correct_wrong_pairs_as_unknown():
    rows = [
        _metric_row("q", "greedy", 0, em=1, f1=1, full=0, process=0, answer="gold"),
        *[
            _metric_row("q", "sampled", i, em=1, f1=1, full=float(i), process=float(i), answer="gold")
            for i in range(4)
        ],
    ]
    summary = summarize_rankability(rows)
    assert summary["overall"]["full_pairwise"]["accuracy"] is None
    assert summary["gates"]["reward_pairwise_accuracy"]["passed"] is False


def test_rankability_manifest_config_forwards_hybrid_training_fields():
    cfg, _ = _load_phase3_config(
        "configs/training/phase3_ppo_legacy_repaired_v2_quota70_smoke600_hybrid_old10_bridge5_v3.yaml"
    )
    assert cfg.total_steps == 600
    assert cfg.batch_size == 4
    assert cfg.ppo_epochs == 2
    assert cfg.kl_coef == 0.25
    assert cfg.sft_replay_ratio == 0.10
    assert cfg.sft_anchor_weight == 0.10
    assert cfg.value_head_init == "zero"
