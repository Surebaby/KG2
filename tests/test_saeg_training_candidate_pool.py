from scripts.prepare.prepare_saeg_v1_training_candidate_pool import build_candidate_rows


def _qpeg(qid, variant, dataset="2wikimultihopqa"):
    return {
        "dataset": dataset,
        "metadata": {
            "source_qid": qid,
            "source_question_sha256": f"hash-{qid}",
            "curriculum_variant": variant,
        },
    }


def _proof(qid, *, excluded=False, rebuildable=True):
    return {
        "dataset": "2wikimultihopqa",
        "qid": qid,
        "question_sha256": f"hash-{qid}",
        "family_sha256": f"family-{qid}",
        "passage_branch_rebuildable": rebuildable,
        "excluded_by_current_qpeg_v4_eval_family": excluded,
        "proof_gold_access_false": True,
        "context_title_set_exact": True,
        "context_body_set_exact": True,
    }


def test_candidate_pool_creates_four_modes_and_excludes_heldout_family():
    rows = build_candidate_rows(
        [_qpeg("p1", "qpeg")],
        [_qpeg("n1", "no_graph_replay")],
        [_proof("w1"), _proof("heldout", excluded=True), _proof("broken", rebuildable=False)],
    )
    assert {row["source_mode"] for row in rows} == {"P_ONLY", "N_REPLAY", "W_ONLY", "P_W_FUSED"}
    assert len(rows) == 4
    assert all(row["qid"] not in {"heldout", "broken"} for row in rows)


def test_candidate_ids_are_unique_across_variants_of_same_qid():
    rows = build_candidate_rows(
        [_qpeg("same", "qpeg")],
        [],
        [_proof("same")],
    )
    assert len(rows) == 3
    assert len({row["candidate_id"] for row in rows}) == 3
