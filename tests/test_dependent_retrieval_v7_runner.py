"""CPU/fake-only tests for the staged Gold-free v7 retrieval runner."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

import scripts.pilot.materialize_paired_dependent_retrieval_v7 as runner
from scripts.pilot import generate_grounded_subanswers_v7 as generator_v7
from scripts.prepare import finalize_paired_dependent_retrieval_v7 as finalizer_v7
from kgproweight.retrieval.subanswer_v7 import build_subanswer_reader_messages


QUESTION = "Where is the birthplace of the author of Root Work located?"


def test_real_frozen_design_protocol_matches_runner_contract():
    protocol = json.loads(runner.DEFAULT_DESIGN_PROTOCOL.read_text(encoding="utf-8"))
    runner.validate_design_protocol(protocol)


def test_design_protocol_rejects_per_arm_query_variant_drift():
    protocol = json.loads(runner.DEFAULT_DESIGN_PROTOCOL.read_text(encoding="utf-8"))
    protocol["arms"]["C_verified_subanswer"][
        "max_query_variants_per_logical_hop"
    ] = 2
    with pytest.raises(runner.V7IntegrityError, match="query-variant cap"):
        runner.validate_design_protocol(protocol)


def _doc(doc_id: str, title: str | None = None, body: str | None = None):
    title = title or doc_id
    body = body or f"body for {doc_id}"
    return {"id": doc_id, "title": title, "contents": f"{title}\n{body}"}


def _arm_a(prefix: str = "original"):
    return [_doc(f"{prefix}-{index}") for index in range(1, 11)]


def _steps(depth: int = 2):
    result = [
        {
            "step": 1,
            "subject": "Root Work",
            "relation_label": "author",
            "output_slot": "hop_1",
            "dependencies": [],
        }
    ]
    if depth >= 2:
        result.append(
            {
                "step": 2,
                "subject": "$hop_1",
                "relation_label": "place of birth",
                "output_slot": "hop_2",
                "dependencies": ["hop_1"],
            }
        )
    if depth >= 3:
        result.append(
            {
                "step": 3,
                "subject": "$hop_2",
                "relation_label": "country",
                "output_slot": "hop_3",
                "dependencies": ["hop_2"],
            }
        )
    if depth >= 4:
        result.append(
            {
                "step": 4,
                "subject": "$hop_3",
                "relation_label": "continent",
                "output_slot": "hop_4",
                "dependencies": ["hop_3"],
            }
        )
    return result


def _row(*, qid: str = "synthetic", depth: int = 2, plan=None):
    question_sha = hashlib.sha256(QUESTION.encode("utf-8")).hexdigest()
    passages = _arm_a(qid)
    return {
        "question_key": f"hotpotqa::{qid}",
        "dataset": "hotpotqa",
        "qid": qid,
        "question": QUESTION,
        "question_sha256": question_sha,
        "family_sha256": f"family-{qid}",
        "role": "development_consumed",
        "arm_a_passages": passages,
        "arm_a_passages_sha256": runner._sha256_historical_json(passages),
        "plan": deepcopy(plan if plan is not None else {"steps": _steps(depth)}),
        "plan_row_sha256": "plan-row-lock",
        "gold_access": False,
    }


class _Retriever:
    def __init__(self, *, same_dependent=False):
        self.batch_calls = []
        self.same_dependent = same_dependent

    def batch_search(self, queries):
        queries = list(queries)
        self.batch_calls.append(queries)
        result = []
        for query in queries:
            if query == "Root Work author":
                result.append(
                    [
                        _doc(
                            "root-1",
                            "Entity Hint",
                            "The article names Verified Person as the author. "
                            + "x" * 1400,
                        ),
                        # Duplicate identity must be removed before C sees it.
                        _doc("root-1", "Entity Hint", "duplicate retrieval row"),
                    ]
                )
            elif query.endswith("Entity Hint place of birth"):
                result.append(
                    [
                        _doc(
                            "b-city",
                            "B City",
                            "The entity-hint branch reaches B City.",
                        )
                    ]
                )
            elif query.endswith("Verified Person place of birth"):
                result.append(
                    [
                        _doc(
                            "c-city",
                            "C City",
                            "The verified branch identifies Verified City.",
                        )
                    ]
                )
            elif query.endswith("B City country"):
                result.append([_doc("b-country", "B Country", "B final evidence")])
            elif query.endswith("Verified City country"):
                result.append([_doc("c-country", "C Country", "C final evidence")])
            elif query.endswith("B Country continent"):
                result.append([_doc("b-continent", "B Continent", "B last evidence")])
            elif query.endswith("C Country continent"):
                result.append([_doc("c-continent", "C Continent", "C last evidence")])
            elif self.same_dependent and query.endswith("Entity Hint place of birth"):
                result.append([_doc("same", "Same", "same query result")])
            else:
                result.append([])
        return result


class _CrossEncoder:
    def __init__(self, *, nonfinite=False):
        self.calls = []
        self.nonfinite = nonfinite

    def predict(self, pairs, show_progress_bar=False):
        assert show_progress_bar is False
        pairs = list(pairs)
        self.calls.append(pairs)
        if self.nonfinite:
            return [float("nan")] * len(pairs)
        values = []
        for question, text in pairs:
            if any(token in text for token in ("b-country", "B final evidence")):
                values.append(9.0)
            elif any(token in text for token in ("c-country", "C final evidence")):
                values.append(10.0)
            elif any(token in text for token in ("B City", "Verified City")):
                values.append(8.0)
            elif "original-9" in text or "original-10" in text:
                values.append(1.0)
            else:
                # Step reranking only needs deterministic finite scores.
                values.append(0.0)
        return values


def _answer(task, *, value="Verified Person", verified=True):
    if verified:
        passage = task["producer_passages"][0]
        sentence = (
            "The article names Verified Person as the author."
            if value == "Verified Person"
            else "The verified branch identifies Verified City."
        )
        telemetry = {
            "verifier_version": "extractive-subanswer-verifier-v7-development-1",
            "parser_version": "strict-subanswer-json-v7-development-1",
            "gold_access": False,
            "verification_scope": "surface_locality_not_semantic_entailment",
            "verified": True,
            "verified_answer": value,
            "reason": "verified",
            "answer_type": "entity",
            "cited_doc_ids": [str(passage["doc_id"])],
            "supporting_doc_id": str(passage["doc_id"]),
            "supporting_sentence": sentence,
            "supporting_sentence_sha256": hashlib.sha256(
                sentence.encode("utf-8")
            ).hexdigest(),
            "support_location": "text",
            "surface_match_mode": "nfkc_casefold_exact",
        }
        verified_answer = value
        response = json.dumps(
            {
                "answer": value,
                "cited_doc_ids": [str(passage["doc_id"])],
                "answer_type": "entity",
                "abstain": False,
            }
        )
    else:
        telemetry = {
            "verifier_version": "extractive-subanswer-verifier-v7-development-1",
            "parser_version": "strict-subanswer-json-v7-development-1",
            "gold_access": False,
            "verification_scope": "surface_locality_not_semantic_entailment",
            "verified": False,
            "verified_answer": None,
            "reason": "model_abstained",
            "answer_type": "entity",
            "cited_doc_ids": [],
            "supporting_doc_id": None,
            "supporting_sentence": None,
            "supporting_sentence_sha256": None,
            "support_location": None,
            "surface_match_mode": None,
        }
        verified_answer = None
        response = json.dumps(
            {
                "answer": "",
                "cited_doc_ids": [],
                "answer_type": "entity",
                "abstain": True,
            }
        )
    response_hash = hashlib.sha256(response.encode("utf-8")).hexdigest()
    telemetry["response_sha256"] = response_hash
    outer_telemetry = {
        "input_task_sha256": runner._sha256_json(task),
        "prompt_sha256": "a" * 64,
        "prompt_tokens": 10,
        "generation_tokens": 8,
        "raw_response": response,
        "raw_response_sha256": response_hash,
        "strict_parse": {"valid": True, "error_code": None},
        "verification": telemetry,
        "prompt_passages_sha256": task["producer_passages_sha256"],
        "verifier_passages_sha256": task["producer_passages_sha256"],
        "same_passage_bytes_for_prompt_and_verifier": True,
        "producer_passage_count": len(task["producer_passages"]),
        "generation": {
            "decode": "greedy",
            "do_sample": False,
            "max_new_tokens": 96,
            "retry_count": 0,
        },
        "gold_access": False,
    }
    return {
        **{field: task[field] for field in runner._TASK_ANSWER_IDENTITY_FIELDS},
        "verified": verified,
        "verified_answer": verified_answer,
        "telemetry": outer_telemetry,
        "gold_access": False,
    }


def _verified_title_answer(task):
    passage = task["producer_passages"][0]
    value = passage["title"]
    response = json.dumps(
        {
            "answer": value,
            "cited_doc_ids": [str(passage["doc_id"])],
            "answer_type": "entity",
            "abstain": False,
        }
    )
    verification = runner.parse_and_verify_subanswer(
        response,
        task["question"],
        task["step"],
        task["producer_passages"],
        target_type=task["target_type"],
    )
    assert verification["verified"] is True
    response_hash = hashlib.sha256(response.encode("utf-8")).hexdigest()
    return {
        **{field: task[field] for field in runner._TASK_ANSWER_IDENTITY_FIELDS},
        "verified": True,
        "verified_answer": value,
        "telemetry": {
            "input_task_sha256": runner._sha256_json(task),
            "prompt_sha256": "a" * 64,
            "prompt_tokens": 10,
            "generation_tokens": 8,
            "raw_response": response,
            "raw_response_sha256": response_hash,
            "strict_parse": {"valid": True, "error_code": None},
            "verification": verification,
            "prompt_passages_sha256": task["producer_passages_sha256"],
            "verifier_passages_sha256": task["producer_passages_sha256"],
            "same_passage_bytes_for_prompt_and_verifier": True,
            "producer_passage_count": len(task["producer_passages"]),
            "generation": {
                "decode": "greedy",
                "do_sample": False,
                "max_new_tokens": 96,
                "retry_count": 0,
            },
            "gold_access": False,
        },
        "gold_access": False,
    }


def test_root_stage_projects_a_shares_physical_query_and_freezes_exact_c_task():
    retriever = _Retriever()
    result = runner.execute_root_stage(
        [_row(depth=3)], retriever, cross_encoder=_CrossEncoder()
    )

    assert retriever.batch_calls == [["Root Work author"]]
    assert len(result.arm_a_rows) == 1
    arm_a = result.arm_a_rows[0]
    assert arm_a["arm"] == "A_canonical_one_shot"
    assert arm_a["kg_subgraph"] == []
    assert len(arm_a["retrieved_passages"]) == 10

    assert len(result.c_tasks) == 1
    task = result.c_tasks[0]
    assert frozenset(task) == runner.C_TASK_KEYS
    assert task["producer_slot"] == "slot_1"
    assert len(task["producer_passages"]) == 1  # duplicate root id was deduped
    assert set(task["producer_passages"][0]) == {"doc_id", "title", "text"}
    assert len(task["producer_passages"][0]["text"]) == 1200
    assert task["producer_passages_sha256"] == runner._sha256_json(
        task["producer_passages"]
    )
    messages = build_subanswer_reader_messages(
        task["question"],
        task["step"],
        task["producer_passages"],
        target_type=task["target_type"],
    )
    prompt_documents = json.loads(messages[1]["content"])["retrieved_documents"]
    assert prompt_documents == task["producer_passages"]
    assert runner._sha256_json(prompt_documents) == task[
        "producer_passages_sha256"
    ]
    assert "source" not in task["producer_passages"][0]
    assert result.states[0]["slot_values_B"] == {"slot_1": "Entity Hint"}

    budget = result.budget_rows[0]
    assert budget["B"]["logical_query_count"] == 1
    assert budget["C"]["logical_query_count"] == 1
    assert budget["B"]["physical_slot_id"] == budget["C"]["physical_slot_id"]
    assert budget["actual_shared_physical_search_count"] == 1


def test_globally_duplicate_root_query_has_one_physical_owner_and_two_ledgers():
    retriever = _Retriever()
    result = runner.execute_root_stage(
        [_row(qid="one"), _row(qid="two")],
        retriever,
        cross_encoder=_CrossEncoder(),
    )

    assert retriever.batch_calls == [["Root Work author"]]
    assert len(result.budget_rows) == 2
    physical_ids = {
        row["B"]["physical_slot_id"] for row in result.budget_rows
    }
    assert len(physical_ids) == 1
    assert all(
        row["B"]["physical_slot_id"] == row["C"]["physical_slot_id"]
        for row in result.budget_rows
    )
    assert sum(
        row["actual_shared_physical_search_count"] for row in result.budget_rows
    ) == 1


def test_non_idempotent_reader_title_edge_case_fails_closed_before_task_emission():
    passages = [
        {
            "id": "doc-1",
            "title": '""',
            "contents": "Different Heading\n" + "Ａ   value " + "x" * 1300,
        }
    ]
    projected, telemetry = runner._producer_passage_projection(passages)
    # The locked reader strips quote characters after accepting an explicit
    # non-empty title; it does not then fall back to the contents first line.
    assert projected[0]["title"] == ""
    assert len(projected[0]["text"]) == 1200
    messages = build_subanswer_reader_messages(
        QUESTION,
        _steps(1)[0],
        projected,
        target_type="relation_graph",
    )
    prompt_documents = json.loads(messages[1]["content"])["retrieved_documents"]
    assert prompt_documents != projected
    assert telemetry["ordered_producer_projection_sha256"] == runner._sha256_json(
        projected
    )
    state = {
        "question_key": "hotpotqa::title-edge",
        "dataset": "hotpotqa",
        "qid": "title-edge",
        "question": QUESTION,
        "question_sha256": hashlib.sha256(QUESTION.encode("utf-8")).hexdigest(),
        "target_type": "relation_graph",
    }
    record = {"slot": "hop_1", "step": _steps(1)[0]}
    with pytest.raises(runner.V7IntegrityError, match="differs byte-for-byte"):
        runner._make_c_task(state, record, passages)


def test_two_dependent_depths_use_verified_answers_and_keep_arm_searches_separate():
    retriever = _Retriever()
    ce = _CrossEncoder()
    roots = runner.execute_root_stage([_row(depth=3)], retriever, cross_encoder=ce)
    depth_2 = runner.execute_dependent_stage(
        roots.states,
        [_answer(roots.c_tasks[0], value="Verified Person")],
        producer_depth=1,
        retriever=retriever,
        cross_encoder=ce,
    )

    # Root was one shared call.  B and C are independent batch_search calls.
    assert retriever.batch_calls[1:] == [
        [f"{QUESTION}\nEntity Hint place of birth"],
        [f"{QUESTION}\nVerified Person place of birth"],
    ]
    assert depth_2.arm_b_rows is None
    assert len(depth_2.c_tasks) == 1
    assert depth_2.c_tasks[0]["producer_slot"] == "slot_2"
    assert depth_2.states[0]["slot_values_B"]["slot_2"] == "B City"

    final = runner.execute_dependent_stage(
        depth_2.states,
        [_answer(depth_2.c_tasks[0], value="Verified City")],
        producer_depth=2,
        retriever=retriever,
        cross_encoder=ce,
    )
    assert retriever.batch_calls[-2:] == [
        [f"{QUESTION}\nB City country"],
        [f"{QUESTION}\nVerified City country"],
    ]
    assert final.arm_b_rows is not None and final.arm_c_rows is not None
    arm_b, arm_c = final.arm_b_rows[0], final.arm_c_rows[0]
    original = roots.arm_a_rows[0]["retrieved_passages"]
    assert arm_b["retrieved_passages"][:8] == original[:8]
    assert arm_c["retrieved_passages"][:8] == original[:8]
    assert "b-country" in [row["id"] for row in arm_b["retrieved_passages"]]
    assert "c-country" in [row["id"] for row in arm_c["retrieved_passages"]]
    detail = final.execution_rows[0]
    assert detail["successful_paired_dependent_hops"] == 2
    assert detail["all_dependent_queries_start_with_exact_original_question"] is True
    assert detail["all_final_ce_pairs_use_exact_original_question"] is True
    assert all(
        row["question"] == QUESTION
        for arm in ("B", "C")
        for row in detail["merge"][arm]["selected_new"]
        if "question" in row
    )
    dependent_budgets = [
        row for row in detail["budget_ledger"] if not row["is_root"]
    ]
    assert len(dependent_budgets) == 2
    assert all(row["B"]["logical_query_count"] == 1 for row in dependent_budgets)
    assert all(row["C"]["logical_query_count"] == 1 for row in dependent_budgets)
    assert all(
        row["actual_independent_physical_search_count"] == 2
        for row in dependent_budgets
    )

    final_ce_calls = [
        call for call in ce.calls if call and all(pair[0] == QUESTION for pair in call)
    ]
    assert final_ce_calls


def test_mixed_depth_1_to_4_state_machine_reaches_final_and_descriptor_chain(tmp_path):
    """Regression for real batches containing terminal and recursive plans together."""

    retriever = _Retriever()
    ce = _CrossEncoder()
    roots = runner.execute_root_stage(
        [
            # Mirrors the terminal row that exposed the real-run regression.
            _row(qid="dev_5473", depth=1),
            _row(qid="depth-2", depth=2),
            _row(qid="depth-3-skip", depth=3),
            _row(qid="depth-4", depth=4),
        ],
        retriever,
        cross_encoder=ce,
    )
    assert len(roots.c_tasks) == 3
    root_by_qid = {state["qid"]: state for state in roots.states}
    assert root_by_qid["dev_5473"]["execution_status"] == "fallback_no_dependent_step"

    answers_depth_1 = []
    for task in roots.c_tasks:
        answers_depth_1.append(
            _answer(task, verified=task["qid"] != "depth-3-skip")
        )
    depth_2 = runner.execute_dependent_stage(
        roots.states,
        answers_depth_1,
        producer_depth=1,
        retriever=retriever,
        cross_encoder=ce,
    )
    assert depth_2.arm_b_rows is None
    assert len(depth_2.c_tasks) == 1
    assert depth_2.c_tasks[0]["qid"] == "depth-4"
    states_2 = {state["qid"]: state for state in depth_2.states}
    assert states_2["dev_5473"]["execution_status"] == "fallback_no_dependent_step"
    assert states_2["depth-2"]["execution_status"] == "dependent_retrieval_complete"
    assert states_2["depth-3-skip"]["execution_status"] == "depth_complete"
    skipped_depth_2 = [
        row
        for row in states_2["depth-3-skip"]["budget_ledger"]
        if row["dependency_depth"] == 2
    ]
    assert len(skipped_depth_2) == 1
    assert skipped_depth_2[0]["paired_active"] is False
    assert skipped_depth_2[0]["paired_skip_reason"].startswith(
        "paired_missing_dependency"
    )

    depth_3 = runner.execute_dependent_stage(
        depth_2.states,
        [_answer(depth_2.c_tasks[0], value="Verified City")],
        producer_depth=2,
        retriever=retriever,
        cross_encoder=ce,
    )
    assert depth_3.arm_b_rows is None
    assert len(depth_3.c_tasks) == 1
    assert depth_3.c_tasks[0]["qid"] == "depth-4"
    states_3 = {state["qid"]: state for state in depth_3.states}
    assert states_3["dev_5473"]["execution_status"] == "fallback_no_dependent_step"
    assert states_3["depth-3-skip"]["execution_status"] == (
        "dependent_retrieval_complete"
    )

    depth_4 = runner.execute_dependent_stage(
        depth_3.states,
        [_verified_title_answer(depth_3.c_tasks[0])],
        producer_depth=3,
        retriever=retriever,
        cross_encoder=ce,
    )
    assert depth_4.arm_b_rows is not None
    assert depth_4.arm_c_rows is not None
    assert depth_4.execution_rows is not None
    assert depth_4.c_tasks == []
    final_states = {state["qid"]: state for state in depth_4.states}
    assert final_states["dev_5473"]["execution_status"] == "fallback_no_dependent_step"
    assert final_states["depth-4"]["execution_status"] == "dependent_retrieval_complete"
    assert final_states["depth-4"]["successful_paired_dependent_hops"] == 3
    final_arms = {
        row["qid"]: row for row in depth_4.arm_b_rows
    }
    assert final_arms["dev_5473"]["retrieved_passages"] == root_by_qid[
        "dev_5473"
    ]["arm_a"]["retrieved_passages"]

    # Persist the same mixed-depth snapshots and prove the descriptor chain is
    # valid through the empty terminal C-task artifact at state depth four.
    output_dir = tmp_path / "materialization"
    output_dir.mkdir()
    runtime_locks = {
        "implementation_lock": {
            "path": "/impl",
            "size_bytes": 1,
            "sha256": "a" * 64,
        },
        "post_plan_execution_lock": {
            "path": "/plan",
            "size_bytes": 1,
            "sha256": "b" * 64,
        },
        "trajectory_semantics_addendum": {
            "path": "/trajectory-semantics",
            "size_bytes": 1,
            "sha256": "c" * 64,
        },
    }

    def write_rows(path, rows):
        runner._write_jsonl_new(path, rows)
        return runner._file_lock(path, allow_empty=not rows)

    root_state_lock = write_rows(output_dir / "root_state.jsonl", roots.states)
    root_tasks_lock = write_rows(output_dir / "c_tasks.depth_1.jsonl", roots.c_tasks)
    root_descriptor = {
        "schema_version": runner.STAGE_DESCRIPTOR_SCHEMA_VERSION,
        "runner_version": runner.RUNNER_VERSION,
        "experiment_id": "V7-MIXED-CHAIN",
        "stage": "roots",
        "state_depth": 1,
        "parent_stage_descriptor": None,
        "runtime_locks": runtime_locks,
        "outputs": {"root_state": root_state_lock, "c_tasks": root_tasks_lock},
        "gold_access": False,
    }
    parent_descriptor_path = output_dir / "roots_stage.json"
    runner._write_json_new(parent_descriptor_path, root_descriptor)
    previous_state_lock = root_state_lock
    for producer_depth, result in enumerate((depth_2, depth_3, depth_4), start=1):
        target_depth = producer_depth + 1
        state_lock = write_rows(
            output_dir / f"state.depth_{target_depth}.jsonl", result.states
        )
        task_lock = write_rows(
            output_dir / f"c_tasks.depth_{target_depth}.jsonl", result.c_tasks
        )
        descriptor = {
            "schema_version": runner.STAGE_DESCRIPTOR_SCHEMA_VERSION,
            "runner_version": runner.RUNNER_VERSION,
            "experiment_id": "V7-MIXED-CHAIN",
            "stage": "dependents",
            "producer_depth": producer_depth,
            "target_depth": target_depth,
            "state_depth": target_depth,
            "parent_stage_descriptor": runner._file_lock(parent_descriptor_path),
            "input_state": previous_state_lock,
            "runtime_locks": runtime_locks,
            "outputs": {"state": state_lock, "c_tasks": task_lock},
            "gold_access": False,
        }
        descriptor_path = output_dir / f"dependents_stage.depth_{producer_depth}.json"
        runner._write_json_new(descriptor_path, descriptor)
        parent_descriptor_path = descriptor_path
        previous_state_lock = state_lock

    _, _, terminal_state_lock = runner._validate_parent_stage_chain(
        output_dir,
        state_depth=4,
        experiment_id="V7-MIXED-CHAIN",
        runtime_locks=runtime_locks,
    )
    assert terminal_state_lock == previous_state_lock


def test_unverified_c_answer_causes_paired_zero_query_and_byte_exact_a():
    root_retriever = _Retriever()
    roots = runner.execute_root_stage(
        [_row(depth=2)], root_retriever, cross_encoder=_CrossEncoder()
    )
    dependent_retriever = _Retriever()
    final = runner.execute_dependent_stage(
        roots.states,
        [_answer(roots.c_tasks[0], verified=False)],
        producer_depth=1,
        retriever=dependent_retriever,
        cross_encoder=_CrossEncoder(),
    )
    assert dependent_retriever.batch_calls == []
    budget = final.budget_rows[0]
    assert budget["paired_active"] is False
    assert budget["B"]["logical_query_count"] == 0
    assert budget["C"]["logical_query_count"] == 0
    assert budget["B"]["query"] is None and budget["C"]["query"] is None
    original = roots.arm_a_rows[0]["retrieved_passages"]
    assert final.arm_b_rows[0]["retrieved_passages"] == original
    assert final.arm_c_rows[0]["retrieved_passages"] == original
    assert final.arm_b_rows[0]["passages_sha256"] == roots.arm_a_rows[0][
        "passages_sha256"
    ]
    assert final.arm_c_rows[0]["passages_sha256"] == roots.arm_a_rows[0][
        "passages_sha256"
    ]
    assert final.execution_rows[0]["fallback_reason"] == (
        "zero_successful_paired_dependent_hops"
    )


def test_same_depth_sibling_duplicate_query_is_skipped_before_batch_search():
    """A duplicate sibling is a paired skip, never retrieval-budget padding."""

    plan = {
        "steps": [
            _steps(2)[0],
            _steps(2)[1],
            {
                "step": 3,
                "subject": "$hop_1",
                "relation_label": "place of birth",
                "output_slot": "hop_3",
                "dependencies": ["hop_1"],
            },
        ]
    }
    retriever = _Retriever()
    roots = runner.execute_root_stage(
        [_row(qid="same-depth-siblings", plan=plan)],
        retriever,
        cross_encoder=_CrossEncoder(),
    )
    assert len(roots.c_tasks) == 1

    final = runner.execute_dependent_stage(
        roots.states,
        [_answer(roots.c_tasks[0])],
        producer_depth=1,
        retriever=retriever,
        cross_encoder=_CrossEncoder(),
    )
    # The root call is followed by exactly one B and one C physical query.
    assert retriever.batch_calls[1:] == [
        [f"{QUESTION}\nEntity Hint place of birth"],
        [f"{QUESTION}\nVerified Person place of birth"],
    ]
    detail = final.execution_rows[0]
    dependent = [
        row for row in detail["budget_ledger"] if row["dependency_depth"] == 2
    ]
    assert len(dependent) == 2
    active = [row for row in dependent if row["paired_active"]]
    skipped = [row for row in dependent if not row["paired_active"]]
    assert len(active) == 1 and len(skipped) == 1
    assert skipped[0]["paired_skip_reason"] == "paired_duplicate_or_noop_query"
    assert skipped[0]["B"]["logical_query_count"] == 0
    assert skipped[0]["C"]["logical_query_count"] == 0
    assert detail["successful_paired_dependent_hops"] == 1


def test_real_generator_fake_backend_output_is_consumed_without_schema_translation():
    roots = runner.execute_root_stage(
        [_row(depth=2)], _Retriever(), cross_encoder=_CrossEncoder()
    )
    task = roots.c_tasks[0]

    def fake_backend(messages, *, max_new_tokens, do_sample):
        assert max_new_tokens == 96 and do_sample is False
        response = json.dumps(
            {
                "answer": "Verified Person",
                "cited_doc_ids": [task["producer_passages"][0]["doc_id"]],
                "answer_type": "entity",
                "abstain": False,
            }
        )
        return generator_v7.GenerationOutput(
            prompt=json.dumps(messages, ensure_ascii=False),
            response_text=response,
            prompt_tokens=50,
            generation_tokens=12,
            runtime_telemetry={"backend": "fake"},
        )

    generated = generator_v7.generate_subanswer_rows(
        [task],
        generator=fake_backend,
        input_file_sha256="a" * 64,
        model_artifact={"adapter": {"sha256": "b" * 64}},
    )
    assert frozenset(generated[0]) == runner.C_ANSWER_KEYS
    dependent_retriever = _Retriever()
    final = runner.execute_dependent_stage(
        roots.states,
        generated,
        producer_depth=1,
        retriever=dependent_retriever,
        cross_encoder=_CrossEncoder(),
    )
    assert dependent_retriever.batch_calls == [
        [f"{QUESTION}\nEntity Hint place of birth"],
        [f"{QUESTION}\nVerified Person place of birth"],
    ]
    assert final.execution_rows[0]["successful_paired_dependent_hops"] == 1


def test_cross_arm_identical_query_still_uses_two_physical_searches():
    roots = runner.execute_root_stage(
        [_row(depth=2)], _Retriever(), cross_encoder=_CrossEncoder()
    )
    retriever = _Retriever()
    # Entity Hint is present in the cited root title, so this represents a
    # legitimate verified C value equal to B's entity surface.
    answer = _answer(roots.c_tasks[0], value="Verified Person")
    answer["verified_answer"] = "Entity Hint"
    verification = answer["telemetry"]["verification"]
    verification.update(
        {
            "verified_answer": "Entity Hint",
            "supporting_sentence": "Entity Hint",
            "supporting_sentence_sha256": hashlib.sha256(
                b"Entity Hint"
            ).hexdigest(),
            "support_location": "text",
        }
    )
    response = json.dumps(
        {
            "answer": "Entity Hint",
            "cited_doc_ids": [roots.c_tasks[0]["producer_passages"][0]["doc_id"]],
            "answer_type": "entity",
            "abstain": False,
        }
    )
    response_hash = hashlib.sha256(response.encode("utf-8")).hexdigest()
    answer["telemetry"]["raw_response"] = response
    answer["telemetry"]["raw_response_sha256"] = response_hash
    verification["response_sha256"] = response_hash
    result = runner.execute_dependent_stage(
        roots.states,
        [answer],
        producer_depth=1,
        retriever=retriever,
        cross_encoder=_CrossEncoder(),
    )
    identical = f"{QUESTION}\nEntity Hint place of birth"
    assert retriever.batch_calls == [[identical], [identical]]
    assert result.budget_rows[0]["cross_arm_query_strings_identical"] is True
    assert result.budget_rows[0]["actual_independent_physical_search_count"] == 2


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda answers: [], "cardinality mismatch"),
        (
            lambda answers: [dict(answers[0], question_sha256="wrong")],
            "identity/hash mismatch",
        ),
        (
            lambda answers: [dict(answers[0], extra="forbidden")],
            "fields differ",
        ),
        (
            lambda answers: [dict(answers[0], verified_answer="smuggled")],
            "unverified C answer carries",
        ),
    ],
)
def test_c_answer_integrity_errors_abort_the_stage(mutate, match):
    roots = runner.execute_root_stage(
        [_row(depth=2)], _Retriever(), cross_encoder=_CrossEncoder()
    )
    valid = [_answer(roots.c_tasks[0], verified=False)]
    with pytest.raises(runner.V7IntegrityError, match=match):
        runner.execute_dependent_stage(
            roots.states,
            mutate(valid),
            producer_depth=1,
            retriever=_Retriever(),
            cross_encoder=_CrossEncoder(),
        )


def test_gold_fields_are_rejected_before_projection_and_a_discards_side_inputs():
    row = _row(depth=2)
    cohort = [
        {
            "dataset": row["dataset"],
            "qid": row["qid"],
            "question_key": row["question_key"],
            "question": row["question"],
            "question_sha256": row["question_sha256"],
            "family_sha256": row["family_sha256"],
            "role": "development_consumed",
            "gold_access": False,
        }
    ]
    contexts = [
        {
            **cohort[0],
            "passages": row["arm_a_passages"],
            "passages_sha256": row["arm_a_passages_sha256"],
            "passage_evidence": {"safe_but_not_projected": True},
            "wikidata_graph": [["not", "projected", "either"]],
        }
    ]
    contexts.append(
        {
            "dataset": "2wikimultihopqa",
            "qid": "unselected-extra",
            "question_key": "2wikimultihopqa::unselected-extra",
            "question": "This row is outside the selected v7 datasets.",
            "question_sha256": "not-consumed",
            "gold_access": False,
            "passages": [],
        }
    )
    plans = [{**cohort[0], "predicted_target": row["plan"]}]
    assembled = runner.assemble_root_rows(cohort, contexts, plans)
    result = runner.execute_root_stage(
        assembled, _Retriever(), cross_encoder=_CrossEncoder()
    )
    serialised_a = runner._canonical_json(result.arm_a_rows[0])
    assert "passage_evidence" not in serialised_a
    assert "wikidata_graph" not in serialised_a
    assert result.arm_a_rows[0]["kg_subgraph"] == []

    poisoned = deepcopy(contexts)
    poisoned[0]["metadata"] = {"supporting_facts": [["hidden", 0]]}
    with pytest.raises(runner.V7IntegrityError, match="Gold/prohibited field"):
        runner.assemble_root_rows(cohort, poisoned, plans)


def test_invalid_plan_is_expected_exact_fallback_without_root_search():
    poisoned_plan = {
        "steps": [
            {
                "subject": "$hop_9",
                "relation_label": "author",
                "output_slot": "hop_1",
                "dependencies": ["hop_9"],
            }
        ]
    }
    retriever = _Retriever()
    roots = runner.execute_root_stage(
        [_row(plan=poisoned_plan)], retriever, cross_encoder=_CrossEncoder()
    )
    assert retriever.batch_calls == []
    state = roots.states[0]
    assert state["plan_executable"] is False
    assert state["execution_status"] == "fallback_plan_invalid"
    assert roots.c_tasks == []


def test_state_validation_rejects_terminal_or_partial_status_relabelling():
    terminal = runner.execute_root_stage(
        [_row(qid="dev_5473", depth=1)],
        _Retriever(),
        cross_encoder=_CrossEncoder(),
    ).states[0]
    poisoned_terminal = deepcopy(terminal)
    poisoned_terminal["execution_status"] = "dependent_retrieval_complete"
    poisoned_terminal["fallback_reason"] = None
    with pytest.raises(runner.V7IntegrityError, match="non-dependent plan"):
        runner._validate_state(poisoned_terminal)

    partial = runner.execute_root_stage(
        [_row(qid="partial-depth", depth=3)],
        _Retriever(),
        cross_encoder=_CrossEncoder(),
    ).states[0]
    poisoned_partial = deepcopy(partial)
    poisoned_partial["execution_status"] = "depth_complete"
    with pytest.raises(runner.V7IntegrityError, match="root-complete plan"):
        runner._validate_state(poisoned_partial)


def test_schema_invalid_locked_planner_row_becomes_expected_a_fallback():
    plan = runner._extract_plan(
        {
            "dataset": "hotpotqa",
            "qid": "invalid-planner-output",
            "predicted_target": None,
        }
    )
    assert plan == {"steps": []}
    retriever = _Retriever()
    roots = runner.execute_root_stage(
        [_row(qid="invalid-planner-output", plan=plan)],
        retriever,
        cross_encoder=_CrossEncoder(),
    )
    assert retriever.batch_calls == []
    assert roots.states[0]["plan_validation_errors"] == ["missing_steps"]
    assert roots.states[0]["fallback_reason"] == "plan_invalid"


def test_retriever_cardinality_and_nonfinite_ce_fail_closed_globally():
    class ShortRetriever:
        def batch_search(self, queries):
            return []

    with pytest.raises(runner.V7IntegrityError, match="returned 0/1"):
        runner.execute_root_stage(
            [_row()], ShortRetriever(), cross_encoder=_CrossEncoder()
        )
    with pytest.raises(runner.V7IntegrityError, match="non-finite"):
        runner.execute_root_stage(
            [_row()], _Retriever(), cross_encoder=_CrossEncoder(nonfinite=True)
        )


def test_append_only_writer_refuses_to_overwrite(tmp_path):
    path = tmp_path / "artifact.jsonl"
    runner._write_jsonl_new(path, [{"gold_access": False}])
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        runner._write_jsonl_new(path, [{"gold_access": False, "new": True}])
    assert path.read_bytes() == original


def test_historical_a_hash_matches_a_real_frozen_context_fixture():
    path = Path(
        "outputs/audits/qpeg_v4_schema_adaptation_eval_retrieval_v1/"
        "retrieval_contexts.jsonl"
    )
    if not path.is_file():
        pytest.skip("frozen canonical context fixture is not present")
    with path.open(encoding="utf-8") as handle:
        context = json.loads(next(handle))
    passages = context["passages"]
    assert runner._sha256_historical_json(passages) == context["passages_sha256"]
    # The producer commitment intentionally uses compact JSON and is a distinct
    # hash contract from the historical canonical-A context hash.
    assert runner._sha256_json(passages) != context["passages_sha256"]


def test_final_report_locks_five_outputs_and_freezes_gate_field_sets(tmp_path):
    roots = runner.execute_root_stage(
        [_row(depth=2)], _Retriever(), cross_encoder=_CrossEncoder()
    )
    final = runner.execute_dependent_stage(
        roots.states,
        [_answer(roots.c_tasks[0], value="Verified Person")],
        producer_depth=1,
        retriever=_Retriever(),
        cross_encoder=_CrossEncoder(),
    )
    output_paths = {}
    for name in ("arm_a", "arm_b", "arm_c", "execution_details", "budget_ledger"):
        path = tmp_path / f"{name}.jsonl"
        path.write_text("{}\n", encoding="utf-8")
        output_paths[name] = path
    dummy_lock = {"path": "/frozen", "size_bytes": 1, "sha256": "a" * 64}
    report = runner._report_from_final(
        final,
        experiment_id="V7-TEST",
        runtime_locks={
            "design_protocol": dummy_lock,
            "preregistration": dummy_lock,
            "truncation_addendum": dummy_lock,
            "trajectory_semantics_addendum": dummy_lock,
            "implementation_lock": dummy_lock,
            "post_plan_execution_lock": dummy_lock,
        },
        assets={"fake": True},
        output_paths=output_paths,
    )

    assert report["status"] == "COMPLETE_GOLD_FREE_MATERIALIZATION"
    assert set(report["outputs"]) == {
        "arm_a",
        "arm_b",
        "arm_c",
        "execution_details",
        "budget_ledger",
    }
    assert all(
        set(lock) == {"path", "size_bytes", "sha256"}
        for lock in report["outputs"].values()
    )
    assert set(report["safety_summary"]) == {
        "runtime_errors",
        "identity_join_rate",
        "recursive_forbidden_input_fields",
        "gold_access",
        "all_rows_and_arms_top10",
        "duplicate_output_documents",
        "unauthorized_A_prefix_displacements",
        "root_only_documents_injected",
        "all_dependent_queries_start_with_exact_original_question",
        "all_final_CE_pairs_use_exact_original_question",
        "B_C_query_budget_equal_every_question_depth_and_hop",
        "budget_padding_queries",
        "unverified_subanswers_used",
        "fallback_pair_and_A_byte_exact",
    }
    assert report["safety_summary"]["all_rows_and_arms_top10"] is True
    required_rates = {
        "plan_executable_rate",
        "strict_subanswer_json_parse_rate",
        "mechanically_verified_subanswer_rate",
        "paired_dependent_hop_activation_rate",
        "retained_new_dependent_document_question_rate_B",
        "retained_new_dependent_document_question_rate_C",
    }
    assert required_rates.issubset(report["by_dataset"]["hotpotqa"])
    assert report["preregistration"] == dummy_lock
    assert report["truncation_addendum"] == dummy_lock
    assert report["implementation_lock"] == dummy_lock
    assert report["plan_lock"] == dummy_lock


def test_runner_outputs_pass_independent_gold_free_finalizer_audit(tmp_path):
    hotpot = _row(qid="audit-hotpot", depth=2)
    musique = _row(
        qid="audit-musique",
        plan={
            "steps": [
                {
                    "step": 1,
                    "subquery_template": "Root Work >> author",
                    "output_slot": "step_1",
                    "dependencies": [],
                },
                {
                    "step": 2,
                    "subquery_template": "$step_1 >> place of birth",
                    "output_slot": "step_2",
                    "dependencies": ["step_1"],
                },
            ]
        },
    )
    musique["dataset"] = "musique"
    musique["question_key"] = "musique::audit-musique"

    retriever = _Retriever()
    ce = _CrossEncoder()
    roots = runner.execute_root_stage(
        [hotpot, musique], retriever, cross_encoder=ce
    )
    assert retriever.batch_calls == [["Root Work author"]]
    final = runner.execute_dependent_stage(
        roots.states,
        [_answer(task, value="Verified Person") for task in roots.c_tasks],
        producer_depth=1,
        retriever=retriever,
        cross_encoder=ce,
    )
    external_budget = [
        deepcopy(row)
        for state in final.states
        for row in state["budget_ledger"]
    ]
    observed = finalizer_v7.audit_materialization(
        final.arm_a_rows,
        final.arm_b_rows,
        final.arm_c_rows,
        final.execution_rows,
        external_budget,
        expected_per_dataset=1,
    )
    assert observed["n"] == 2
    assert observed["identity_join_rate"] == 1.0
    assert observed["B_C_query_budget_equal_every_question_depth_and_hop"] is True
    assert observed["root_only_documents_injected"] == 0

    rows_by_output = {
        "arm_a": final.arm_a_rows,
        "arm_b": final.arm_b_rows,
        "arm_c": final.arm_c_rows,
        "execution_details": final.execution_rows,
        "budget_ledger": external_budget,
    }
    output_paths = {}
    for name, rows in rows_by_output.items():
        path = tmp_path / f"{name}.jsonl"
        runner._write_jsonl_new(path, rows)
        output_paths[name] = path
    dummy_lock = {"path": "/frozen", "size_bytes": 1, "sha256": "a" * 64}
    report = runner._report_from_final(
        final,
        experiment_id="V7-TEST",
        runtime_locks={
            "design_protocol": dummy_lock,
            "preregistration": dummy_lock,
            "truncation_addendum": dummy_lock,
            "trajectory_semantics_addendum": dummy_lock,
            "implementation_lock": dummy_lock,
            "post_plan_execution_lock": dummy_lock,
        },
        assets={"fake": True},
        output_paths=output_paths,
    )
    finalizer_v7.enforce_gold_free_gates(
        report,
        observed,
        materialization_gates=finalizer_v7.v7_freeze.MATERIALIZATION_GATES,
        mechanism_gates=finalizer_v7.v7_freeze.MECHANISM_GATES,
        expected_per_dataset=1,
    )
    assert report["gate_decision"] == "PASS_READY_FOR_SEPARATE_GOLD_FINALIZER"


def test_c_answer_rejects_recursive_gold_field_and_model_content_drift():
    roots = runner.execute_root_stage(
        [_row(depth=2)], _Retriever(), cross_encoder=_CrossEncoder()
    )
    task = roots.c_tasks[0]
    leaked = _answer(task)
    leaked["telemetry"]["nested_payload"] = {"gold_answer": "forbidden"}
    with pytest.raises(runner.V7IntegrityError, match="Gold/prohibited field"):
        runner._validate_c_answer(task, leaked)

    expected_model = {
        "schema_version": runner.MODEL_ARTIFACT_SCHEMA,
        "base_model": {"tree_sha256": "a" * 64},
        "strong_sft_adapter": {"tree_sha256": "b" * 64},
    }
    answer = _answer(task)
    answer["telemetry"]["model_artifact"] = deepcopy(expected_model)
    runner._validate_c_answer(
        task, answer, expected_model_artifact=expected_model
    )
    answer["telemetry"]["model_artifact"]["strong_sft_adapter"][
        "tree_sha256"
    ] = "c" * 64
    with pytest.raises(runner.V7IntegrityError, match="model identity/content"):
        runner._validate_c_answer(
            task, answer, expected_model_artifact=expected_model
        )


def test_state_rechecks_locked_planner_row_and_schedule_hashes():
    source = _row(depth=2)
    locked_row = {
        "question_key": source["question_key"],
        "dataset": source["dataset"],
        "qid": source["qid"],
        "question": source["question"],
        "question_sha256": source["question_sha256"],
        "predicted_target": deepcopy(source["plan"]),
        "gold_access": False,
    }
    source["plan_row_sha256"] = runner._sha256_json(locked_row)
    roots = runner.execute_root_stage(
        [source], _Retriever(), cross_encoder=_CrossEncoder()
    )
    key = source["question_key"]
    runner._validate_state(
        roots.states[0], locked_plan_rows={key: locked_row}
    )

    changed_hash = deepcopy(roots.states[0])
    changed_hash["plan_row_sha256"] = "0" * 64
    with pytest.raises(runner.V7IntegrityError, match="planner-row hash"):
        runner._validate_state(changed_hash, locked_plan_rows={key: locked_row})

    changed_schedule = deepcopy(roots.states[0])
    changed_schedule["schedule"][0]["dependency_depth"] = 4
    with pytest.raises(runner.V7IntegrityError, match="schedule drift"):
        runner._validate_state(changed_schedule, locked_plan_rows={key: locked_row})


def test_stage_descriptor_chain_rejects_parent_or_state_tampering(tmp_path: Path):
    runtime_locks = {
        "implementation_lock": {
            "path": "/impl",
            "size_bytes": 1,
            "sha256": "a" * 64,
        },
        "post_plan_execution_lock": {
            "path": "/plan",
            "size_bytes": 1,
            "sha256": "b" * 64,
        },
    }
    root_state = tmp_path / "root_state.jsonl"
    root_state.write_text("{}\n", encoding="utf-8")
    tasks1 = tmp_path / "c_tasks.depth_1.jsonl"
    tasks1.write_text("{}\n", encoding="utf-8")
    root_descriptor = {
        "schema_version": runner.STAGE_DESCRIPTOR_SCHEMA_VERSION,
        "runner_version": runner.RUNNER_VERSION,
        "experiment_id": "V7-CHAIN",
        "stage": "roots",
        "state_depth": 1,
        "parent_stage_descriptor": None,
        "runtime_locks": runtime_locks,
        "outputs": {
            "root_state": runner._file_lock(root_state),
            "c_tasks": runner._file_lock(tasks1),
        },
        "gold_access": False,
    }
    root_descriptor_path = tmp_path / "roots_stage.json"
    runner._write_json_new(root_descriptor_path, root_descriptor)

    state2 = tmp_path / "state.depth_2.jsonl"
    state2.write_text("{}\n", encoding="utf-8")
    tasks2 = tmp_path / "c_tasks.depth_2.jsonl"
    tasks2.write_text("{}\n", encoding="utf-8")
    dependent_descriptor = {
        "schema_version": runner.STAGE_DESCRIPTOR_SCHEMA_VERSION,
        "runner_version": runner.RUNNER_VERSION,
        "experiment_id": "V7-CHAIN",
        "stage": "dependents",
        "producer_depth": 1,
        "target_depth": 2,
        "state_depth": 2,
        "parent_stage_descriptor": runner._file_lock(root_descriptor_path),
        "input_state": runner._file_lock(root_state),
        "runtime_locks": runtime_locks,
        "outputs": {
            "state": runner._file_lock(state2),
            "c_tasks": runner._file_lock(tasks2),
        },
        "gold_access": False,
    }
    dep_path = tmp_path / "dependents_stage.depth_1.json"
    runner._write_json_new(dep_path, dependent_descriptor)
    _, _, locked_state = runner._validate_parent_stage_chain(
        tmp_path,
        state_depth=2,
        experiment_id="V7-CHAIN",
        runtime_locks=runtime_locks,
    )
    assert locked_state == runner._file_lock(state2)

    root_state.write_text('{"tampered":true}\n', encoding="utf-8")
    with pytest.raises(runner.V7IntegrityError, match="(size|SHA256) differs"):
        runner._validate_parent_stage_chain(
            tmp_path,
            state_depth=2,
            experiment_id="V7-CHAIN",
            runtime_locks=runtime_locks,
        )


def test_identical_recursive_queries_require_identical_producer_passages():
    class NondeterministicRetriever(_Retriever):
        def __init__(self):
            super().__init__()
            self.dependent_call = 0

        def batch_search(self, queries):
            queries = list(queries)
            if queries and queries[0].endswith("Verified Person place of birth"):
                self.dependent_call += 1
                return [[_doc(f"different-{self.dependent_call}")]]
            return super().batch_search(queries)

    roots = runner.execute_root_stage(
        [_row(depth=2)], _Retriever(), cross_encoder=_CrossEncoder()
    )
    roots.states[0]["slot_values_B"]["hop_1"] = "Verified Person"
    with pytest.raises(runner.V7IntegrityError, match="identical B/C dependent queries"):
        runner.execute_dependent_stage(
            roots.states,
            [_answer(roots.c_tasks[0], value="Verified Person")],
            producer_depth=1,
            retriever=NondeterministicRetriever(),
            cross_encoder=_CrossEncoder(),
        )


def test_formal_cli_forbids_arbitrary_state_path_but_dry_run_may_inspect_it(tmp_path):
    def args(*, dry_run: bool):
        return type(
            "Args",
            (),
            {
                "stage": "dependents",
                "depth": 1,
                "state_path": tmp_path / "alternate.jsonl",
                "c_answers": None,
                "cohort": None,
                "contexts": None,
                "plans": None,
                "dry_run": dry_run,
            },
        )()

    with pytest.raises(SystemExit, match="forbids --state_path"):
        runner.validate_stage_cli_boundary(args(dry_run=False))
    runner.validate_stage_cli_boundary(args(dry_run=True))
