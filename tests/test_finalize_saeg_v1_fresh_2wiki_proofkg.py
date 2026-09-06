from scripts.prepare.finalize_saeg_v1_fresh_2wiki_proofkg import structural_gates


def _detail(i, *, nonempty=True, complete=True, recognized=True):
    return {
        "question_key": f"2wikimultihopqa::q{i}",
        "query_plan": {"recognized": recognized},
        "kg_subgraph": [["A", "r", "B"]] if nonempty else [],
        "execution": {"complete_plan_execution": complete},
        "runtime_error": None,
        "provenance": {"gold_access": False},
    }


def test_structural_gates_pass_at_frozen_thresholds():
    rows = [_detail(i) for i in range(7)] + [_detail(7, complete=False)] + [
        _detail(8, nonempty=False, complete=False), _detail(9, nonempty=False, complete=False)
    ]
    result = structural_gates(rows)
    assert result["values"]["nonempty"] == 0.8
    assert result["values"]["complete_execution"] == 0.7
    assert result["all_pass"]


def test_structural_gates_fail_closed_below_nonempty_threshold():
    rows = [_detail(i) for i in range(7)] + [_detail(i, nonempty=False, complete=False) for i in range(7, 10)]
    assert not structural_gates(rows)["all_pass"]
