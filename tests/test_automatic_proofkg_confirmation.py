from scripts.eval.generate_query_plans_unseen import _assert_question_only
from scripts.pilot.build_automatic_proofkg_from_plans import convert_predicted_target
from scripts.pilot.build_query_aware_proof_kg_pilot import _link_surface
from scripts.pilot.freeze_automatic_proofkg_confirmation import freeze_cohort


def _rows():
    rows = []
    assignments = {}
    for dataset in ("2wikimultihopqa", "musique"):
        for index in range(5):
            key = f"{dataset}::{index}"
            rows.append(
                {
                    "question_key": key,
                    "dataset": dataset,
                    "qid": str(index),
                    "question": f"question {dataset} {index}",
                    "target_type": "relation_graph" if dataset.startswith("2wiki") else "decomposition_graph",
                }
            )
            assignments[key] = {"family_sha256": f"{dataset}-family-{index}"}
    return rows, assignments


def test_freeze_cohort_excludes_historical_families_and_gold_fields():
    rows, assignments = _rows()
    old = [rows[0], rows[5]]
    selected, availability = freeze_cohort(
        dev_rows=rows,
        assignments=assignments,
        old_evaluated=old,
        excluded_keys={rows[1]["question_key"], rows[6]["question_key"]},
        per_dataset=2,
        seed=7,
    )
    assert len(selected) == 4
    assert {row["dataset"] for row in selected} == {"2wikimultihopqa", "musique"}
    assert all("target" not in row and "answer" not in row for row in selected)
    assert all(value["selected_families"] == 2 for value in availability.values())
    assert not ({row["question_key"] for row in selected} & {rows[0]["question_key"], rows[5]["question_key"]})


def test_freeze_cohort_supports_single_dataset():
    rows, assignments = _rows()
    selected, availability = freeze_cohort(
        dev_rows=rows,
        assignments=assignments,
        old_evaluated=[],
        excluded_keys=set(),
        per_dataset=3,
        seed=11,
        datasets=("2wikimultihopqa",),
    )
    assert len(selected) == 3
    assert {row["dataset"] for row in selected} == {"2wikimultihopqa"}
    assert set(availability) == {"2wikimultihopqa"}


def test_freeze_cohort_excludes_entire_explicit_family():
    rows, assignments = _rows()
    duplicate = dict(rows[0])
    duplicate["question_key"] = "2wikimultihopqa::duplicate"
    duplicate["qid"] = "duplicate"
    assignments[duplicate["question_key"]] = {
        "family_sha256": assignments[rows[0]["question_key"]]["family_sha256"]
    }
    selected, _ = freeze_cohort(
        dev_rows=rows + [duplicate],
        assignments=assignments,
        old_evaluated=[],
        excluded_keys={rows[0]["question_key"]},
        per_dataset=4,
        seed=9,
        datasets=("2wikimultihopqa",),
    )
    assert duplicate["question_key"] not in {row["question_key"] for row in selected}


def test_question_only_guard_rejects_nested_gold_fields():
    _assert_question_only({"question": "safe", "metadata": {"dataset": "x"}})
    try:
        _assert_question_only({"question": "unsafe", "metadata": {"answer": "gold"}})
    except ValueError as exc:
        assert "prohibited runtime fields" in str(exc)
    else:
        raise AssertionError("nested answer must be rejected")


def test_convert_2wiki_plan_preserves_exact_pid_chain():
    predicted = {
        "anchors": ["Example Film"],
        "steps": [
            {"subject": "Example Film", "pid": "P57", "output_slot": "hop_1", "dependencies": []},
            {"subject": "$hop_1", "pid": "P569", "output_slot": "hop_2", "dependencies": ["hop_1"]},
        ],
    }
    plan, diagnostic = convert_predicted_target("2wikimultihopqa", "When was its director born?", predicted)
    assert plan.recognized
    assert plan.anchors == ["Example Film"]
    assert [list(hop.pids) for hop in plan.hops] == [["P57"], ["P569"]]
    assert plan.hops[0].relation_role == "bridge"
    assert not [row for row in diagnostic if row["status"] == "ABSTAIN"]


def test_convert_musique_abstains_from_ambiguous_aggregation():
    predicted = {
        "steps": [
            {"subquery_template": "Example College >> country", "dependencies": [], "output_slot": "step_1"},
            {"subquery_template": "compare #1 and #2 >> instance of", "dependencies": ["step_1", "step_2"], "output_slot": "step_3"},
        ]
    }
    plan, diagnostic = convert_predicted_target("musique", "Where is Example College?", predicted)
    assert plan.recognized
    assert plan.anchors == ["Example College"]
    assert list(plan.hops[0].pids) == ["P17"]
    assert any(row.get("reason") == "multi_input_aggregation" for row in diagnostic)


def test_link_surface_restores_question_parenthetical_before_title_resolution():
    class Resolver:
        seen = None

        def resolve(self, value):
            self.seen = value
            return type(
                "Result",
                (),
                {
                    "selected_qid": "Q1",
                    "selected_label": value,
                    "score": 1.0,
                    "margin": 1.0,
                    "abstained": False,
                    "abstain_reason": "",
                },
            )()

    class Linker:
        def link_single(self, *_args, **_kwargs):
            raise AssertionError("fallback linker should not be used")

    resolver = Resolver()
    linked = _link_surface(
        Linker(),
        "Barcelona",
        "Who performed Barcelona (Freddie Mercury And Montserrat Caballé Song)?",
        resolver,
    )
    assert resolver.seen == "Barcelona (Freddie Mercury And Montserrat Caballé Song)"
    assert linked["resolved_surface"] == resolver.seen


def test_link_surface_fallback_search_uses_original_surface():
    class Resolver:
        def resolve(self, _value):
            return type("Result", (), {"abstained": True})()

    class Linker:
        seen = None

        def link_single(self, value, *, question):
            self.seen = (value, question)
            return type(
                "Result",
                (),
                {
                    "selected_qid": "Q2",
                    "selected_label": value,
                    "score": 0.9,
                    "margin": 0.2,
                    "abstained": False,
                    "abstain_reason": "",
                },
            )()

    linker = Linker()
    linked = _link_surface(
        linker,
        "Stephen Marley",
        "What is Stephen Marley (Musician)'s father's cause of death?",
        Resolver(),
    )
    assert linker.seen[0] == "Stephen Marley"
    assert linked["qid"] == "Q2"


def test_convert_2wiki_subject_anchor_normalization():
    # anchor "Who" and subject "Who?" must normalise to the same surface so the
    # resolved QID propagates to the first hop (dev_60 regression).
    predicted = {
        "anchors": ["Who"],
        "steps": [
            {"step": 1, "subject": "Who?", "relation_label": "occupation", "pid": "P106", "output_slot": "hop_1", "dependencies": []},
        ],
    }
    plan, diagnostic = convert_predicted_target("2wikimultihopqa", "q", predicted)
    assert plan.recognized
    assert plan.hops[0].subject == plan.anchors[0] == "Who"
    assert not [row for row in diagnostic if row["status"] == "ABSTAIN"]


def test_convert_2wiki_subject_strips_punct_quotes_whitespace():
    predicted = {
        "anchors": ['"Entity"?'],
        "steps": [
            {"step": 1, "subject": "  Entity  ", "relation_label": "occupation", "pid": "P106", "output_slot": "hop_1", "dependencies": []},
        ],
    }
    plan, _ = convert_predicted_target("2wikimultihopqa", "q", predicted)
    assert plan.hops[0].subject == plan.anchors[0] == "Entity"


def test_convert_2wiki_dependency_subject_unchanged_by_clean_anchor():
    predicted = {
        "anchors": ["Entity"],
        "steps": [
            {"step": 1, "subject": "Entity", "relation_label": "occupation", "pid": "P106", "output_slot": "hop_1", "dependencies": []},
            {"step": 2, "subject": "$hop_1", "relation_label": "country", "pid": "P17", "output_slot": "hop_2", "dependencies": ["hop_1"]},
        ],
    }
    plan, _ = convert_predicted_target("2wikimultihopqa", "q", predicted)
    assert plan.hops[1].subject == "$hop_1"  # dependency ref preserved exactly

