from scripts.pilot.audit_query_planner_v2_a0_title_retrieval import (
    _aggregate,
    _coverage,
    _evaluate_gates,
)
from scripts.pilot.audit_query_planner_v2_a0_online_qid import _batch_resolve


def test_anchor_coverage_uses_corpus_titles_not_passage_ids():
    results = {
        "Death Stone": [
            {"id": "7", "contents": '"Der Stein des Todes"\nA film.'},
            {"id": "8", "contents": '"Other"\nOther text.'},
        ]
    }
    value = _coverage(["Der Stein des Todes"], ["Death Stone"], results, 1)
    assert value["complete"]
    assert value["recall"] == 1.0
    assert value["retrieved_titles"] == ["Der Stein des Todes"]


def test_a0_gate_requires_absolute_and_oracle_relative_effect():
    rows = []
    for hit in (True, True, True, False):
        rows.append({
            "predicted_surface_at_5": {"recall": float(hit), "any_hit": hit, "complete": hit},
            "predicted_surface_at_10": {"recall": float(hit), "any_hit": hit, "complete": hit},
            "predicted_surface_at_20": {"recall": float(hit), "any_hit": hit, "complete": hit},
            "gold_alias_oracle_at_5": {"recall": 1.0, "any_hit": True, "complete": True},
            "gold_alias_oracle_at_10": {"recall": 1.0, "any_hit": True, "complete": True},
            "gold_alias_oracle_at_20": {"recall": 1.0, "any_hit": True, "complete": True},
        })
    metrics = {
        source: {str(k): _aggregate(rows, source, k) for k in (5, 10, 20)}
        for source in ("predicted_surface", "gold_alias_oracle")
    }
    gates = {
        "oracle_complete_at_20_min": 0.9,
        "predicted_complete_at_5_min": 0.5,
        "predicted_complete_at_20_min": 0.6,
        "predicted_to_oracle_ratio_at_20_min": 0.7,
    }
    assert _evaluate_gates(metrics, gates)["pass"]


def test_batch_qid_resolver_applies_normalization_and_abstains_on_disambiguation(monkeypatch):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "query": {
                    "pages": [
                        {
                            "title": "Robert Shaw (Royal Navy officer)",
                            "pageprops": {"wikibase_item": "Q123"},
                        },
                        {
                            "title": "Two Timid Souls",
                            "pageprops": {"wikibase_item": "Q456", "disambiguation": ""},
                        },
                    ]
                }
            }

    monkeypatch.setattr(
        "scripts.pilot.audit_query_planner_v2_a0_online_qid.requests.get",
        lambda *args, **kwargs: Response(),
    )
    values = _batch_resolve(["Robert Shaw (Royal Navy Officer)", "Two Timid Souls"])
    assert values["Robert Shaw (Royal Navy Officer)"]["selected_qid"] == "Q123"
    assert values["Two Timid Souls"]["abstained"]
