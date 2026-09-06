from scripts.prepare.prefetch_historical_wikidata_revisions import runtime_qids


def test_runtime_qids_uses_only_frozen_execution_entities() -> None:
    rows = [{
        "execution": {
            "anchor_entities": {"A": {"qid": "Q1"}},
            "hops": [{
                "pids": ["P26"],
                "input_entities": [{"qid": "Q1"}],
                "output_entities": [{"qid": "Q2"}],
            }],
        }
    }]
    qids, qid_pids = runtime_qids(rows)
    assert qids == {"Q1", "Q2"}
    assert qid_pids == {"Q1": {"P26"}}
