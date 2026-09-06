"""Tests for the deterministic silver train/val/test split.

The properties under test are the ones a split can violate silently — a broken
split does not raise, it just reports an optimistic number. So each test pins one
concrete failure mode rather than a generic invariant.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kgproweight.data.silver_dataset import SilverDatasetReader, SilverTrajectory
from kgproweight.data.silver_split import (
    SPLIT_NAMES,
    SplitSpec,
    assign_split,
    check_no_group_leak,
    group_key,
    split_trajectories,
    summarize_split,
)


def _traj(qid: str, question: str, accepted: bool = True) -> SilverTrajectory:
    return SilverTrajectory(
        qid=qid, question=question, answer="a", dataset="hotpotqa",
        steps=[], accepted=accepted,
    )


def _corpus(n: int = 2000, accept_every: int = 3) -> list[SilverTrajectory]:
    return [
        _traj(f"train_{i}", f"question number {i}?", accepted=(i % accept_every == 0))
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_assignment_is_stable_across_calls():
    spec = SplitSpec()
    items = _corpus(500)
    first = [assign_split(t, spec) for t in items]
    second = [assign_split(t, spec) for t in items]
    assert first == second


def test_assignment_does_not_depend_on_order_or_neighbours():
    """Position-based splits (shuffle-and-slice) fail this; hashing passes.

    This is the property that makes the split survive Phase 1 being re-run to
    append 2Wiki / MuSique data.
    """
    spec = SplitSpec()
    items = _corpus(800)
    baseline = {t.qid: assign_split(t, spec) for t in items}

    reversed_items = list(reversed(items))
    assert {t.qid: assign_split(t, spec) for t in reversed_items} == baseline

    # Appending new trajectories must not move any existing one.
    grown = items + [_traj(f"new_{i}", f"a brand new question {i}?") for i in range(400)]
    for t in grown[: len(items)]:
        assert assign_split(t, spec) == baseline[t.qid]

    # Nor must dropping trajectories.
    for t in items[::7]:
        assert assign_split(t, spec) == baseline[t.qid]


def test_assignment_survives_a_fresh_interpreter():
    """Guards against ``hash()``, which is salted per process.

    If the implementation ever switches to the builtin hash, the split silently
    differs between the training run and the evaluation run — i.e. evaluation on
    data the model trained on.
    """
    import subprocess
    import sys

    code = (
        "from kgproweight.data.silver_dataset import SilverTrajectory\n"
        "from kgproweight.data.silver_split import SplitSpec, assign_split\n"
        "ts = [SilverTrajectory(qid='q%d' % i, question='question number %d?' % i,\n"
        "                      answer='a', dataset='hotpotqa', steps=[],\n"
        "                      accepted=(i % 3 == 0)) for i in range(200)]\n"
        "print(','.join(assign_split(t, SplitSpec()) for t in ts))\n"
    )
    # cwd is pinned to the repo root: pytest's rootdir depends on how it was
    # invoked, and the child needs kgproweight importable regardless.
    repo_root = Path(__file__).resolve().parents[1]
    runs = set()
    for salt in ("0", "1"):
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, check=True, cwd=str(repo_root),
            env={"PYTHONHASHSEED": salt, "PATH": "/usr/bin:/bin",
                 "HOME": str(Path.home()), "PYTHONPATH": str(repo_root)},
        )
        runs.add(out.stdout.strip())
    assert len(runs) == 1, "split changed under a different PYTHONHASHSEED"


def test_different_seed_gives_a_different_split():
    items = _corpus(600)
    a = [assign_split(t, SplitSpec(seed=42)) for t in items]
    b = [assign_split(t, SplitSpec(seed=7)) for t in items]
    assert a != b


# ---------------------------------------------------------------------------
# Ratios
# ---------------------------------------------------------------------------

def test_defaults_are_ten_percent_each():
    """Pinned because the ratio was chosen from a power calculation, not taste.

    NEG is ~3.4% of held-out accepted steps, so 5% folds would leave ~57 NEG
    steps and a 95% CI of +/-0.12 on NEG recall. If someone lowers this, the
    headline metric silently loses precision.
    """
    s = SplitSpec()
    assert (s.val_ratio, s.test_ratio, s.seed) == (0.10, 0.10, 42)
    assert s.train_ratio == pytest.approx(0.80)


def test_every_phase_config_agrees_on_the_split():
    """A fold mismatch between phases is worse than having no split at all.

    Phase 3 training on Phase 2's test fold produces a number that looks
    held-out and is not, and nothing raises.
    """
    from kgproweight.training.phase2_prm import Phase2Config
    from kgproweight.training.phase3_grpo import Phase3GRPOConfig
    from kgproweight.training.phase3_ppo import Phase3PPOConfig
    from kgproweight.training.phase3_sft import Phase3SFTConfig

    specs = {
        name: C(silver_path="s", output_dir="o").build_split_spec()
        for name, C in (("phase2", Phase2Config), ("sft", Phase3SFTConfig),
                        ("ppo", Phase3PPOConfig), ("grpo", Phase3GRPOConfig))
    }
    assert len(set(specs.values())) == 1, specs


def test_formal_phase_configs_default_to_train_fold():
    from kgproweight.config import ProjectConfig, load_config

    root = Path(__file__).resolve().parents[1]
    for name in ("phase2_prm.yaml", "phase3_sft.yaml", "phase3_ppo.yaml"):
        cfg = load_config(str(root / "configs" / "training" / name), validate=ProjectConfig)
        assert cfg.training.split == "train", name


def test_quota70_frozen_configs_bind_all_cross_stage_artifacts():
    from kgproweight.config import ProjectConfig, load_config

    root = Path(__file__).resolve().parents[1]
    p2 = load_config(
        str(root / "configs/training/phase2_prm_legacy_repaired_v2_quota70.yaml"),
        validate=ProjectConfig,
    ).training
    sft = load_config(
        str(root / "configs/training/phase3_sft_legacy_repaired_v2_quota70.yaml"),
        validate=ProjectConfig,
    ).training
    ppo_doc = load_config(
        str(root / "configs/training/phase3_ppo_legacy_repaired_v2_quota70_smoke600.yaml"),
        validate=ProjectConfig,
    )
    ppo = ppo_doc.training

    assert "quota70" in p2.silver_path
    assert sft.silver_path == f"{p2.output_dir}/silver_with_logprobs.jsonl"
    assert ppo.silver_path == sft.silver_path
    assert ppo.sft_checkpoint == f"{sft.output_dir}/final"
    assert ppo.alpha_gate_path == f"{p2.output_dir}/alpha_gate.pt"
    assert "quota70" in ppo.question_kg_index_path
    assert ppo.require_exact_kg_index_alignment is True
    assert ppo.max_kg_index_miss_rate == 0.0
    assert ppo.ppo.total_ppo_steps == 600
    assert ppo.ppo.sft_replay_ratio == pytest.approx(0.10)
    assert ppo.ppo.sft_anchor_interval == 0
    assert ppo_doc.reward.text_reward_backend == "rearag"


def test_soft_alpha_config_changes_only_the_target_from_hard_config():
    from kgproweight.config import load_config

    root = Path(__file__).resolve().parents[1]
    hard = load_config(str(root / "configs/training/phase2_prm.yaml"))
    soft = load_config(str(root / "configs/training/phase2_prm_soft_alpha.yaml"))
    assert hard["training"]["alpha_target"] == "hard_verdict"
    assert soft["training"]["alpha_target"] == "soft_abs_rkg"
    hard["training"].pop("alpha_target")
    soft["training"].pop("alpha_target")
    assert hard == soft


def test_phase2_and_sft_refuse_implicit_whole_file_training(tmp_path: Path):
    from kgproweight.training.phase2_prm import Phase2Config, run_phase2
    from kgproweight.training.phase3_sft import Phase3SFTConfig, run_phase3_sft

    with pytest.raises(ValueError, match="split is None"):
        run_phase2(Phase2Config(silver_path="missing", output_dir=str(tmp_path / "p2")))
    with pytest.raises(ValueError, match="split is None"):
        run_phase3_sft(Phase3SFTConfig(silver_path="missing", output_dir=str(tmp_path / "sft")))


def test_split_seed_none_falls_back_to_training_seed():
    from kgproweight.training.phase2_prm import Phase2Config

    cfg = Phase2Config(silver_path="s", output_dir="o", seed=99, split_seed=None)
    assert cfg.build_split_spec().seed == 99


def test_fold_sizes_track_requested_ratios():
    spec = SplitSpec(val_ratio=0.1, test_ratio=0.1)
    folds = split_trajectories(_corpus(6000), spec)
    n = sum(len(v) for v in folds.values())
    assert n == 6000
    assert len(folds["val"]) / n == pytest.approx(0.1, abs=0.02)
    assert len(folds["test"]) / n == pytest.approx(0.1, abs=0.02)
    assert len(folds["train"]) / n == pytest.approx(0.8, abs=0.03)


def test_folds_partition_without_overlap():
    folds = split_trajectories(_corpus(1500), SplitSpec())
    ids = [{t.qid for t in folds[name]} for name in SPLIT_NAMES]
    assert ids[0] & ids[1] == set()
    assert ids[0] & ids[2] == set()
    assert ids[1] & ids[2] == set()
    assert sum(len(s) for s in ids) == 1500


def test_zero_ratio_means_empty_fold():
    folds = split_trajectories(_corpus(500), SplitSpec(val_ratio=0.0, test_ratio=0.1))
    assert folds["val"] == []
    assert folds["test"]


def test_invalid_ratios_rejected():
    with pytest.raises(ValueError):
        SplitSpec(val_ratio=0.6, test_ratio=0.5)
    with pytest.raises(ValueError):
        SplitSpec(val_ratio=-0.1)


# ---------------------------------------------------------------------------
# Stratification
# ---------------------------------------------------------------------------

def test_accepted_ratio_is_preserved_in_each_fold():
    """The real corpus is 39.4% accepted; each fold should match it.

    Without stratification a small val fold can drift far enough that val and
    train are not comparable, which makes an early-stopping signal read off val
    misleading.
    """
    items = _corpus(9000, accept_every=3)  # ~33% accepted
    spec = SplitSpec(val_ratio=0.05, test_ratio=0.05)
    folds = split_trajectories(items, spec)
    overall = sum(1 for t in items if t.accepted) / len(items)
    for name in SPLIT_NAMES:
        fold = folds[name]
        assert fold, name
        got = sum(1 for t in fold if t.accepted) / len(fold)
        assert got == pytest.approx(overall, abs=0.03), f"{name}: {got} vs {overall}"


def test_stratification_can_be_disabled():
    items = _corpus(1000)
    spec = SplitSpec(stratify_accepted=False)
    folds = split_trajectories(items, spec)
    assert sum(len(v) for v in folds.values()) == len(items)


# ---------------------------------------------------------------------------
# Group leakage
# ---------------------------------------------------------------------------

def test_duplicate_question_lands_in_one_fold():
    """The concrete case in the real file: train_3398 / train_22486 share text.

    Splitting on qid would put paraphrase-identical items on both sides.
    """
    q = "In what year was the head coach of the 2009-10 Oklahoma Sooners born?"
    a = _traj("train_3398", q, accepted=False)
    b = _traj("train_22486", q, accepted=False)
    spec = SplitSpec()
    assert assign_split(a, spec) == assign_split(b, spec)


def test_question_normalisation_catches_near_duplicates():
    spec = SplitSpec()
    variants = [
        _traj("a", "Who directed Inception?"),
        _traj("b", "who directed inception"),
        _traj("c", "  Who   directed Inception ?  "),
    ]
    folds = {assign_split(t, spec) for t in variants}
    assert len(folds) == 1
    assert len({group_key(t, spec) for t in variants}) == 1


def test_no_group_leak_across_folds():
    items = _corpus(3000)
    # Every question duplicated under a second qid.
    items += [_traj(t.qid + "_dup", t.question, t.accepted) for t in items]
    spec = SplitSpec()
    folds = split_trajectories(items, spec)
    assert check_no_group_leak(folds, spec) == {}


def test_empty_question_falls_back_to_qid():
    spec = SplitSpec()
    a, b = _traj("x1", ""), _traj("x2", "   ")
    assert group_key(a, spec) == "id:x1"
    assert group_key(b, spec) == "id:x2"


def test_group_by_qid_mode():
    spec = SplitSpec(group_by_question=False)
    t = _traj("q9", "some question")
    assert group_key(t, spec) == "id:q9"


# ---------------------------------------------------------------------------
# Reader integration
# ---------------------------------------------------------------------------

@pytest.fixture()
def silver_file(tmp_path: Path) -> Path:
    p = tmp_path / "silver.jsonl"
    with open(p, "w", encoding="utf-8") as fh:
        for i in range(1200):
            fh.write(json.dumps({
                "qid": f"train_{i}",
                "question": f"question number {i}?",
                "answer": "a",
                "dataset": "hotpotqa",
                "steps": [{"index": 0, "text": "Step 1: something.", "label": 0.8}],
                "accepted": i % 3 == 0,
            }) + "\n")
    return p


def test_reader_without_split_reads_whole_file(silver_file: Path):
    r = SilverDatasetReader(silver_file)
    assert len(r) == 1200
    assert r.split is None


def test_reader_split_folds_are_disjoint_and_complete(silver_file: Path):
    spec = SplitSpec()
    seen: set[str] = set()
    total = 0
    for name in SPLIT_NAMES:
        r = SilverDatasetReader(silver_file, split=name, split_spec=spec)
        qids = {t.qid for t in r}
        assert not (qids & seen), f"{name} overlaps an earlier fold"
        seen |= qids
        total += len(r)
        assert r.n_total_in_file == 1200
    assert total == 1200


def test_reader_accepted_respects_the_fold(silver_file: Path):
    """``accepted()`` must not quietly return the whole file once a split is set.

    This is the failure the reader-level filter exists to prevent: 12+ call sites
    treat ``accepted()`` as "the data".
    """
    full = SilverDatasetReader(silver_file)
    train = SilverDatasetReader(silver_file, split="train")
    assert len(train.accepted()) < len(full.accepted())
    train_qids = {t.qid for t in train}
    assert all(t.qid in train_qids for t in train.accepted())


def test_reader_subset_stays_inside_the_fold(silver_file: Path):
    train = SilverDatasetReader(silver_file, split="train")
    sub = train.subset(20, seed=1)
    fold_qids = {t.qid for t in train}
    assert len(sub) == 20
    assert all(t.qid in fold_qids for t in sub)


def test_reader_rejects_unknown_split(silver_file: Path):
    with pytest.raises(ValueError):
        SilverDatasetReader(silver_file, split="holdout")


def test_reader_splits_helper_matches_per_fold_readers(silver_file: Path):
    spec = SplitSpec()
    folds = SilverDatasetReader(silver_file).splits(spec)
    for name in SPLIT_NAMES:
        direct = SilverDatasetReader(silver_file, split=name, split_spec=spec)
        assert {t.qid for t in folds[name]} == {t.qid for t in direct}


def test_summarize_split_counts_add_up(silver_file: Path):
    spec = SplitSpec()
    r = SilverDatasetReader(silver_file)
    counts = summarize_split(r.splits(spec), spec)
    assert sum(counts.n.values()) == 1200
    assert sum(counts.n_accepted.values()) == len(r.accepted())
    d = counts.as_dict()
    assert set(d) == {"n", "n_accepted", "n_groups"}


# ---------------------------------------------------------------------------
# Data-efficiency subsets
# ---------------------------------------------------------------------------

def test_data_efficiency_subsets_stay_inside_the_fold(silver_file: Path, tmp_path: Path):
    """Every subset on the curve is TRAINING data for that point's model.

    If ``make_subset_file`` drew from the whole file, each point would train on
    val/test trajectories, so the curve's own held-out evaluation would be
    invalid — and no assertion anywhere else would catch it, because the subset
    file looks perfectly well-formed either way.
    """
    from kgproweight.eval.data_efficiency import make_subset_file

    spec = SplitSpec()
    out = tmp_path / "subset.jsonl"
    make_subset_file(silver_file, n=50, seed=7, output_path=out,
                     split="train", split_spec=spec)

    got = {json.loads(l)["qid"] for l in out.read_text(encoding="utf-8").splitlines() if l.strip()}
    assert len(got) == 50
    held_out = set()
    for name in ("val", "test"):
        held_out |= {t.qid for t in SilverDatasetReader(silver_file, split=name, split_spec=spec)}
    assert got & held_out == set()


def test_data_efficiency_subset_is_reproducible(silver_file: Path, tmp_path: Path):
    """Same (n, seed, fold) must give the same subset across runs.

    The curve is averaged over seeds, so a subset that silently differed between
    invocations would make the error bars meaningless.
    """
    from kgproweight.eval.data_efficiency import make_subset_file

    spec = SplitSpec()
    paths = []
    for i in (1, 2):
        p = tmp_path / f"subset_{i}.jsonl"
        make_subset_file(silver_file, n=40, seed=3, output_path=p,
                         split="train", split_spec=spec)
        paths.append(p)
    assert paths[0].read_text(encoding="utf-8") == paths[1].read_text(encoding="utf-8")


def test_inspect_split_stream_matches_eager(silver_file: Path):
    """``--stream`` must print exactly what the eager path prints.

    The streaming path exists because the AutoDL box in no-GPU mode has a 2 GiB
    cgroup memory cap and the real silver file is 1.37 GB, so the eager reader is
    OOM-killed there. That makes ``--stream`` the ONLY way the folds get verified
    on the machine that actually trains — if it ever drifted from the eager path,
    the pre-flight check would be confirming a split that no phase uses. It
    reconstructs only (qid, question, accepted), the three fields the split keys
    on, so any change to ``group_key``/``assign_split`` inputs breaks this.
    """
    import subprocess
    import sys as _sys

    script = Path(__file__).resolve().parents[1] / "scripts" / "utils" / "inspect_split.py"
    run = lambda extra: subprocess.run(
        [_sys.executable, str(script), str(silver_file), *extra],
        capture_output=True, text=True, check=True).stdout

    assert run(["--stream"]) == run([])
    # The qid dump feeds evaluator filtering, so it must agree too.
    for fold in ("val", "test"):
        a = sorted(run(["--dump-qids", fold]).split())
        b = sorted(run(["--stream", "--dump-qids", fold]).split())
        assert a == b and a, fold
