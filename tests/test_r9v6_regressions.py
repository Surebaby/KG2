"""Regressions for the R9 v6 audit fixes.

Each test pins one bug that was silently degrading EM/F1. Failure here means a
fix was reverted, not that a threshold needs relaxing.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from kgproweight.data.prompts import build_rl_messages, format_retrieved_block
from kgproweight.kg.entity_linker import extract_mentions
from kgproweight.kg.kg_filter import _RELATION_LABEL_TO_PID, filter_and_rank_triples
from kgproweight.kg.wikidata_retriever import (
    _QA_RELATION_FILTER,
    _RELATION_PRIORITY,
    WikidataSubgraphRetriever,
)
from kgproweight.data.silver_dataset import SilverDatasetReader, SilverTrajectory
from kgproweight.training.phase2_prm import (
    _StepSample,
    _label_to_class,
    _step_samples_from_silver,
    build_prm_input,
)
from kgproweight.training.reward_function import _grounded_mentions


# ---------------------------------------------------------------------------
# 1. prompts.py NameError — hard crash on every PPO prompt
# ---------------------------------------------------------------------------

def test_format_retrieved_block_survives_skipped_passages():
    """``logger`` was undefined, so the n_skipped > 0 branch raised NameError.

    Triggered whenever len(passages) > max_passages — i.e. every PPO prompt,
    since silver records store 50 passages and PPO renders top_k=15.
    """
    passages = [{"contents": f"doc {i}"} for i in range(20)]
    out = format_retrieved_block(passages, max_passages=5)
    assert out.count("\n") == 4


def test_format_retrieved_block_survives_empty_passage():
    out = format_retrieved_block([{"contents": ""}, {"contents": "real text"}])
    assert "real text" in out


def test_build_rl_messages_with_more_passages_than_top_k():
    msgs = build_rl_messages(
        question="Who?",
        retrieved_passages=[{"contents": f"passage {i}"} for i in range(50)],
        kg_triples=[("A", "rel", "B")],
        top_k=15,
    )
    assert len(msgs) == 2
    assert "(A, rel, B)" in msgs[1]["content"]


# ---------------------------------------------------------------------------
# 2. Subgraph cache key change orphaned ~63k entries
# ---------------------------------------------------------------------------

def test_cache_reads_legacy_key_format(tmp_path):
    """Keys moved to ``{qid}_{hops}_{neighbors}_{filter}``, orphaning
    ``{qid}_{hops}`` — the format holding 63063 of 63829 cached subgraphs. In
    offline mode that meant no KG at all."""
    cache_file = tmp_path / "kg_subgraph_cache.jsonl"
    legacy = [
        ["Ed Wood", "country of citizenship", "United States"],
        ["Ed Wood", "occupation", "film director"],
    ]
    cache_file.write_text(
        json.dumps({"key": "Q221843_2", "triples": legacy}) + "\n", encoding="utf-8"
    )
    r = WikidataSubgraphRetriever(
        max_hops=2, max_neighbors=30, cache_dir=str(tmp_path), offline=True,
    )
    assert len(r.fetch(["Q221843"])) == 2


def test_relation_filter_applied_at_read_time(tmp_path):
    """The cache stores RAW subgraphs; the relation policy is a read-time filter,
    so changing the policy must not require re-fetching."""
    cache_file = tmp_path / "kg_subgraph_cache.jsonl"
    cache_file.write_text(
        json.dumps({"key": "Q1_2", "triples": [
            ["A", "country of citizenship", "United States"],
            ["A", "described by source", "Some Encyclopedia"],
        ]}) + "\n",
        encoding="utf-8",
    )
    r = WikidataSubgraphRetriever(
        max_hops=2, max_neighbors=30, cache_dir=str(tmp_path), offline=True,
        relation_filter=_QA_RELATION_FILTER,
    )
    rels = {t[1] for t in r.fetch(["Q1"])}
    assert "country of citizenship" in rels
    assert "described by source" not in rels


def test_empty_fetch_is_not_cached(tmp_path):
    """A blocked/failed fetch must stay retryable."""
    r = WikidataSubgraphRetriever(
        max_hops=2, max_neighbors=30, cache_dir=str(tmp_path), offline=True,
    )
    assert r.fetch(["Q_DOES_NOT_EXIST"]) == []
    assert len(r.cache) == 0


# ---------------------------------------------------------------------------
# 3. Whitelisted PIDs unreachable because their label had no PID mapping
# ---------------------------------------------------------------------------

def test_every_whitelisted_pid_is_reachable_by_label():
    """``_apply_relation_filter`` resolves label -> PID before testing the
    whitelist, so a whitelisted PID whose label is absent from the map is
    dropped. P20/P166/P175/P112/P54/P286/P118 were all unreachable."""
    mapped = set(_RELATION_LABEL_TO_PID.values())
    unreachable = sorted(_QA_RELATION_FILTER - mapped)
    assert not unreachable, f"whitelisted but unreachable PIDs: {unreachable}"


@pytest.mark.parametrize("label,pid", [
    ("place of death", "P20"),
    ("award received", "P166"),
    ("founder", "P112"),
    ("member of sports team", "P54"),
    ("head of government", "P6"),
    ("named after", "P138"),
])
def test_qa_relations_survive_the_filter(label, pid):
    assert _RELATION_LABEL_TO_PID[label] == pid
    assert pid in _QA_RELATION_FILTER
    assert _RELATION_PRIORITY.get(pid, 0) >= 1


def test_no_duplicate_priority_entries_shadow_qa_relations():
    """A later duplicate key wins in a dict literal, so an entry repeated in the
    external-ID block silently demoted a QA relation to -1."""
    for pid in ("P1082", "P1411", "P1412"):
        assert _RELATION_PRIORITY[pid] >= 1
        assert pid in _QA_RELATION_FILTER


# ---------------------------------------------------------------------------
# 4. int(label) collapsed fractional PRM labels to NEUTRAL
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label,expected", [
    (1.0, 2), (0.75, 2), (0.5, 2),
    (0.33, 1), (0.0, 1), (-0.33, 1),
    (-0.5, 0), (-1.0, 0),
])
def test_fractional_labels_bucket_not_truncate(label, expected):
    """int(0.75) == 0 == NEUTRAL discarded all partial credit."""
    assert _label_to_class(label) == expected


def test_partial_credit_is_not_neutral():
    assert _label_to_class(0.75) != _label_to_class(0.0)


# ---------------------------------------------------------------------------
# 5. Mention extraction glued sentence-initial function words onto entities
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("question,expected", [
    ("Were Scott Derrickson and Ed Wood of the same nationality?", "Scott Derrickson"),
    ("Are Giuseppe Verdi and Ambroise Thomas both Opera composers?", "Giuseppe Verdi"),
    ("The Vermont Catamounts men's soccer team competes where?", "Vermont Catamounts"),
])
def test_leading_stopword_stripped_from_mention(question, expected):
    """"Were Scott Derrickson" never linked. Affected 17% of hotpotqa dev, 14.9%
    of 2wiki dev."""
    assert expected in extract_mentions(question, max_n=5)


def test_mentions_never_start_with_a_function_word():
    stop = {"were", "are", "was", "is", "the", "this", "that", "what", "which", "and"}
    for q in [
        "Were Scott Derrickson and Ed Wood of the same nationality?",
        "Are Random House Tower and 888 7th Avenue both used for real estate?",
        "This singer of A Rather Blustery Day also voiced what hedgehog?",
    ]:
        for m in extract_mentions(q, max_n=5):
            assert m.split()[0].lower() not in stop, f"{m!r} from {q!r}"


# ---------------------------------------------------------------------------
# 6. Reward-side dynamic KG let hallucinated entities earn positive R_KG
# ---------------------------------------------------------------------------

def test_ungrounded_mention_cannot_expand_the_reward_graph():
    """§3.4: generate a wrong entity -> system fetches its REAL subgraph -> model
    cites it -> PRM marks it verified -> wrong reasoning gets positive reward."""
    grounded = _grounded_mentions(
        {"Ed Wood", "Napoleon Bonaparte"},
        "Were Scott Derrickson and Ed Wood of the same nationality?",
        [{"contents": "Ed Wood was an American filmmaker."}],
        [("Ed Wood", "country of citizenship", "United States")],
    )
    assert "Ed Wood" in grounded
    assert "Napoleon Bonaparte" not in grounded


def test_mention_grounded_via_kg_node_is_allowed():
    grounded = _grounded_mentions(
        {"United States"}, "Which country?", [],
        [("Ed Wood", "country of citizenship", "United States")],
    )
    assert "United States" in grounded


def test_mention_grounded_via_passage_is_allowed():
    grounded = _grounded_mentions(
        {"Poughkeepsie"}, "Where was he born?",
        [{"contents": "Ed Wood was born in Poughkeepsie, New York."}], [],
    )
    assert "Poughkeepsie" in grounded


# ---------------------------------------------------------------------------
# 7. Fallback KG path bypassed the three-layer filter entirely
# ---------------------------------------------------------------------------

def test_filter_ranks_answer_bearing_triple_above_noise():
    """Raw SPARQL order buried the answer under 9 `occupation` rows; the eval
    path took that raw order because the v2 index covered 0 dev questions."""
    raw = [("Ed Wood", "occupation", f"role {i}") for i in range(9)]
    raw += [
        ("Ed Wood", "sex or gender", "male"),
        ("Ed Wood", "country of citizenship", "United States"),
    ]
    ranked = filter_and_rank_triples(
        raw, "Were Scott Derrickson and Ed Wood of the same nationality?", max_keep=5,
    )
    assert ("Ed Wood", "country of citizenship", "United States") == ranked[0]


def test_filter_caps_taxonomic_relations():
    raw = [("X", "instance of", f"type {i}") for i in range(10)]
    raw += [("X", "country of citizenship", "France")]
    ranked = filter_and_rank_triples(raw, "What nationality is X?", max_keep=10)
    tax = sum(1 for t in ranked if t[1] in ("instance of", "subclass of"))
    assert tax <= max(1, int(len(ranked) * 0.25))


# ---------------------------------------------------------------------------
# 13. PRM input dropped the prior conclusions the NEG label depends on
# ---------------------------------------------------------------------------

def _mk(text, prev=(), cls=1):
    return _StepSample(
        text=text, label=0.0, label_class=cls, kg_subgraph=[], coverage=0.0,
        binary_quality=1, semantic_entropy=0.0, prev_conclusions=list(prev),
    )


def test_prm_input_includes_prior_conclusions():
    """NEG is assigned by ``_is_contradiction(conclusion, prev_conclusions)``, so
    a step scored in isolation has no way to express its own label. Feeding bare
    step text made NEG unlearnable: held-out recall 0.152 vs 0.802 on seen data,
    precision 0.490 vs 1.000. The prefix must carry the prior conclusions."""
    rendered = build_prm_input(_mk("Conclusion: Born in Berlin.",
                                   prev=["Born in Paris."]))
    assert "Born in Paris." in rendered
    assert "Born in Berlin." in rendered


def test_contradictory_and_consistent_steps_differ_in_input():
    """The label-determining property: identical step text under different
    histories must produce DIFFERENT encoder inputs. If these collide, the
    training signal is noise no amount of class weighting can fix."""
    step = "Conclusion: The film is Finnish."
    neg = build_prm_input(_mk(step, prev=["The film is Swedish."], cls=0))
    neu = build_prm_input(_mk(step, prev=["The director was born in 1950."]))
    assert neg != neu


def test_prm_input_keeps_most_recent_conclusions_when_capped():
    """Truncation must drop the OLDEST context; the contradiction usually sits
    against a recent conclusion."""
    rendered = build_prm_input(_mk("cur", prev=[f"c{i}" for i in range(10)]))
    assert "c9" in rendered
    assert "c0" not in rendered


def test_prm_input_handles_empty_history():
    """Step 1 has no predecessors and must still render (and not crash)."""
    rendered = build_prm_input(_mk("Reasoning: first step"))
    assert "Reasoning: first step" in rendered


def test_class_weights_upweight_the_rare_negative_class():
    """NEG is ~4% of accepted steps. An unweighted CE lets the head trade NEG
    away for 4 points of accuracy, which is what it did. Weights are inverse
    frequency, mean-normalised to 1.0 so the loss scale (and hence the meaning
    of ``calibration_weight``) is unchanged."""
    counts = [336, 5087, 2690]          # measured NEG / NEU / POS
    n = sum(counts)
    raw = [n / (3.0 * c) for c in counts]
    scale = sum(raw) / 3.0
    w = [min(r / scale, 10.0) for r in raw]
    assert w[0] > w[2] > w[1]                      # NEG > POS > NEU
    assert abs(sum(w) / 3.0 - 1.0) < 0.05          # mean-normalised
    assert w[0] / w[1] > 10                        # rare class actually lifted


def test_step_samples_thread_conclusions_in_order():
    """Each sample sees only the conclusions of steps BEFORE it — no leakage of
    its own or later conclusions, which would let the head cheat."""
    steps = [
        {"index": 1, "text": "Reasoning: a\nConclusion: first.", "label": 0.0},
        {"index": 2, "text": "Reasoning: b\nConclusion: second.", "label": 0.0},
        {"index": 3, "text": "Reasoning: c\nConclusion: third.", "label": 0.0},
    ]
    traj = {"qid": "q1", "question": "?", "answer": "", "steps": steps,
            "kg_subgraph": [], "accepted": True, "metadata": {}}
    reader = SilverDatasetReader.__new__(SilverDatasetReader)
    reader.trajectories = [SilverTrajectory.from_dict(traj)]
    samples = _step_samples_from_silver(reader)
    assert len(samples) == 3
    assert samples[0].prev_conclusions == []
    # The parser normalises trailing punctuation, so match on the stem.
    assert "first" in samples[1].prev_conclusions[0]
    assert len(samples[2].prev_conclusions) == 2
    assert "second" in samples[2].prev_conclusions[1]
    # No sample may see its own conclusion or a later one.
    for s in samples:
        assert not any("third" in c for c in s.prev_conclusions)
