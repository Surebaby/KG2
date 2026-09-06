from kgproweight.data.silver_dataset import SilverTrajectory
from scripts.pilot.build_disjoint_confirmation_cohort import select_disjoint


def _item(qid):
    return SilverTrajectory(
        qid=qid,
        question=f"question {qid}",
        answer="gold",
        dataset="hotpotqa",
        steps=[],
    )


def test_select_disjoint_is_deterministic_and_excludes_prior_qids():
    items = [_item(f"q{i:02d}") for i in range(10)]
    first = select_disjoint(items, {"q01", "q03"}, n=5, seed=7)
    second = select_disjoint(list(reversed(items)), {"q01", "q03"}, n=5, seed=7)

    assert [item.qid for item in first] == [item.qid for item in second]
    assert not {item.qid for item in first}.intersection({"q01", "q03"})


def test_select_disjoint_refuses_oversized_cohort():
    try:
        select_disjoint([_item("q1")], set(), n=2, seed=1)
    except ValueError as exc:
        assert "only 1 disjoint candidates" in str(exc)
    else:
        raise AssertionError("expected ValueError")
