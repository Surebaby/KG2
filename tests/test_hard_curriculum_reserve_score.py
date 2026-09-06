from scripts.pilot.score_hard_curriculum_reserve import score_candidates


def _candidate(qid, kind, index, em, score, valid=True):
    return {
        "qid": qid,
        "candidate_type": kind,
        "candidate_index": index,
        "em": em,
        "process": {
            "score": score,
            "trajectory_valid": valid,
            "components": {
                "P_precise_citation": score,
                "H_hop_coverage": score,
                "O_dependency_order": score,
                "G_conclusion_grounding": score,
                "A_answer_consistency": score,
            },
        },
    }


def test_mixed_rankability_metrics_are_within_sampled_candidates():
    rows = [_candidate("q1", "greedy", 0, 0, 0.0)]
    rows += [
        _candidate("q1", "sampled", 0, 0, 0.1),
        _candidate("q1", "sampled", 1, 1, 0.9),
        _candidate("q1", "sampled", 2, 0, 0.2),
        _candidate("q1", "sampled", 3, 1, 0.8),
    ]
    result = score_candidates(rows)
    assert result["mixed_outcome_qids"] == 1
    assert result["reward_pairwise_accuracy"] == 1.0
    assert result["mixed_reward_top1_em"] == 1.0
    assert result["mixed_random_sampled_em"] == 0.5
    assert result["recovery_qids"] == 1
