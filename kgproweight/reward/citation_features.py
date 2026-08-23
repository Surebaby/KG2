"""The α-gate's two per-step CITATION features, in one place.

Why this module exists as a single source of truth: ``clean_entities`` was
written for exactly this reason and STILL diverged in effect, because a bug in
the shared function (multi-word scaffold bypassing the filter) hit training and
inference identically but was invisible from either side. The lesson taken is not
"share the code" -- that was already true -- but "share it AND test the shared
thing on real data". Both features here are therefore defined once, used by
Phase 2 (``phase2_prm._build_samples_accepted_only``) and by the PPO/inference
reward (``training/reward_function.py``), and pinned by tests that feed them the
same silver steps the gate is fitted on.

2026-08-23 (§14 of docs/retraining_plan.md). The 3-feature α-gate could not fit
its own BCE target: measured ceiling R^2 = +0.038 over a constant predictor,
while the shipped checkpoint scored WORSE than a constant. Adding these two
features raises the ceiling to +0.439.

    feature      mean    sd      corr(target)
    f_density    0.8764  0.2667  +0.172        <- original
    f_confidence 0.9070  0.0689  +0.130        <- original
    cite_any     0.5108  0.4999  +0.547        <- here
    cite_match   0.2081  0.3965  +0.593        <- here

Note ``cite_any``'s sd of 0.4999 against ``f_confidence``'s 0.0689: a near-binary
feature on a 51/49 split has ~7x the dynamic range of the almost-constant feature
that carried the gate's LARGEST weight. That is the mechanical reason the old
gate was effectively a function of density alone.
"""
from __future__ import annotations

from typing import Iterable, Sequence, Tuple


def _normalise(triple: Sequence[object]) -> str:
    """Canonical string form of a triple for set membership.

    Lower-cased and whitespace-collapsed on all three slots. Deliberately the
    same normalisation on both sides of the comparison -- matching a cited triple
    against the subgraph is otherwise defeated by nothing more than casing.
    """
    return " ".join(str(x).strip().lower() for x in triple[:3])


def citation_features(
    cited_triples: Iterable[Sequence[object]],
    kg_subgraph: Iterable[Sequence[object]],
) -> Tuple[float, float]:
    """Return ``(cite_any, cite_match)`` for one step.

    ``cite_any``   1.0 if the step cited at least one triple, else 0.0.
    ``cite_match`` fraction of the step's cited triples that appear in
                   ``kg_subgraph``; 0.0 when the step cited nothing.

    ``kg_subgraph`` must be the FILTERED subgraph (the one ``graph_density`` is
    computed from), so that every α-gate feature describes the same KG view.

    On a step with no citations both are 0.0 -- which is the honest value, not a
    missing one: 0 means "this step made no KG claim", the same neutral reading
    ``r_kg = 0`` carries in the reward (``prm_annotator`` returns NEUTRAL for
    exactly this case). A step that cites nothing should not be graded on the
    accuracy of citations it never made.
    """
    cited = [t for t in (cited_triples or []) if t is not None and len(tuple(t)) >= 3]
    if not cited:
        return 0.0, 0.0
    kg = {_normalise(t) for t in (kg_subgraph or []) if t is not None and len(tuple(t)) >= 3}
    matched = sum(1 for t in cited if _normalise(t) in kg)
    return 1.0, float(matched) / float(len(cited))
