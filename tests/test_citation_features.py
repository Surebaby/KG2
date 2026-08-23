"""Tests for the α-gate's two per-step citation features (§14).

These are pinned against REAL silver steps as well as synthetic ones. The
``clean_entities`` bug is the reason: it was already a shared single-source
function, and the bug still shipped, because every unit test fed it synthetic
mentions and none fed it the mention lists the parser actually produces. So the
last test here reads the silver file and asserts on aggregate statistics.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from kgproweight.reward.citation_features import citation_features

KG = [
    ("Arthur's Magazine", "publication_date", "1844"),
    ("First for Women", "publisher", "Bauer Media Group"),
]


def test_no_citations_gives_zero_zero():
    """0.0 means 'made no KG claim', which is the honest value, not missing data."""
    assert citation_features([], KG) == (0.0, 0.0)
    assert citation_features(None, KG) == (0.0, 0.0)


def test_all_cited_triples_present():
    assert citation_features([KG[0]], KG) == (1.0, 1.0)
    assert citation_features(list(KG), KG) == (1.0, 1.0)


def test_partial_match_is_a_fraction():
    cited = [KG[0], ("Nonexistent", "rel", "Thing")]
    any_, match = citation_features(cited, KG)
    assert any_ == 1.0
    assert match == pytest.approx(0.5)


def test_citing_nothing_real_still_counts_as_citing():
    """cite_any and cite_match are independent: a step can cite yet match none."""
    any_, match = citation_features([("A", "b", "C")], KG)
    assert any_ == 1.0
    assert match == 0.0


def test_match_is_case_and_whitespace_insensitive():
    """Otherwise casing alone defeats the comparison on both sides."""
    cited = [("  arthur's MAGAZINE ", "Publication_Date", "1844")]
    assert citation_features(cited, KG) == (1.0, 1.0)


def test_empty_subgraph_means_no_match_but_still_cited():
    any_, match = citation_features([KG[0]], [])
    assert (any_, match) == (1.0, 0.0)


def test_malformed_triples_are_skipped():
    assert citation_features([("A", "b")], KG) == (0.0, 0.0)
    any_, match = citation_features([KG[0], ("A", "b")], KG)
    assert any_ == 1.0
    assert match == pytest.approx(1.0), "the malformed entry must not count as a miss"


SILVER = Path("data/silver_data/silver_v1_reannotated.jsonl")


@pytest.mark.skipif(not SILVER.exists(), reason="silver data not present")
def test_aggregate_statistics_on_real_silver_steps():
    """Guards the measured numbers §14's decision rests on.

    Measured 2026-08-23 over the 33,011 accepted steps: 51.08% cite anything and
    mean cite_match is 0.2081. If a parser or filter change moves these, the
    α-gate's fitted weights are no longer the ones §14 justified.
    """
    import itertools
    import json

    from kgproweight.data.parsers import parsed_step_from_silver_dict

    n = 0
    n_any = 0
    for line in itertools.islice(open(SILVER), 3000):
        d = json.loads(line)
        if not (d.get("accepted") or d.get("is_accepted")):
            continue
        kg = [tuple(t) for t in (d.get("kg_subgraph") or []) if len(t) >= 3]
        for i, step in enumerate(d.get("steps") or []):
            if not (step.get("text") or "").strip():
                continue
            p = parsed_step_from_silver_dict(step, fallback_index=i)
            any_, _ = citation_features(p.cited_triples, kg)
            n += 1
            n_any += int(any_ > 0)

    assert n > 500, f"expected a usable sample, got {n} steps"
    rate = n_any / n
    # Wide band: this is a regression tripwire, not a re-measurement.
    assert 0.35 < rate < 0.70, f"citation rate {rate:.3f} is far from the measured 0.511"
