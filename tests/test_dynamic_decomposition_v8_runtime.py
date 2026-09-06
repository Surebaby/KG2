"""CPU-only tests for the v8 production-runtime wiring and frozen smoke scope."""

from __future__ import annotations

from copy import deepcopy
import ast
import hashlib
import inspect
import json

import pytest

from kgproweight.retrieval.dynamic_decomposition_v8 import NO_RELEVANT_ANSWER
import scripts.pilot.materialize_dynamic_decomposition_v8 as runtime


MODEL_IDENTITY = {
    "base_model_tree_sha256": "1" * 64,
    "adapter_tree_sha256": "2" * 64,
    "tokenizer_tree_sha256": "3" * 64,
}
RETRIEVAL_IDENTITY = {
    "corpus_sha256": "4" * 64,
    "dense_index_sha256": "5" * 64,
    "bm25_tree_sha256": "6" * 64,
    "e5_tree_sha256": "7" * 64,
    "bge_tree_sha256": "8" * 64,
}


def _docs(prefix: str, count: int, *, gold_poison: bool = False):
    return [
        {
            "id": f"{prefix}-{index:03d}",
            "title": f"{prefix} title {index}",
            "contents": f"{prefix} title {index}\nEvidence document {index}.",
            **({"gold_answer": "NEVER-EMIT-THIS"} if gold_poison else {}),
        }
        for index in range(1, count + 1)
    ]


class FakeRRF:
    def __init__(self, *, count=100):
        self.count = count
        self.calls = []

    def batch_search(self, queries):
        self.calls.append(list(queries))
        return [_docs("rrf", self.count) for _ in queries]


class FakeCrossEncoder:
    def __init__(self, *, nonfinite=False):
        self.nonfinite = nonfinite
        self.pairs = []

    def predict(self, pairs, show_progress_bar=False):
        assert show_progress_bar is False
        self.pairs.append(deepcopy(pairs))
        scores = list(range(len(pairs)))
        if self.nonfinite:
            scores[-1] = float("nan")
        return scores


def test_runtime_contract_locks_one_model_and_exact_canonical_retrieval():
    contract = runtime.runtime_contract()
    assert contract["gold_access"] is False
    assert contract["prospective_unlocked"] is False
    assert contract["production_staged"] is True
    assert contract["staged_retrieval_contract"] == {
        "stable_deduplicate_cache_misses": True,
        "backend_batch_stages": ["root_all", "q1_all", "q2_BC_all"],
        "engineering_smoke_logical_retrieval_requests": 84,
        "maximum_full_index_passes_per_attempt": 3,
    }
    assert contract["shared_hf_runtime"]["one_physical_model_instance_for_roles"] == [
        "controller",
        "subanswer_reader",
        "final_reader",
    ]
    assert contract["shared_hf_runtime"]["base_model_path"] == "models/llama3-8b"
    assert contract["shared_hf_runtime"]["strong_sft_adapter_path"].endswith(
        "sft_legacy_repaired_v2_quota70_hard_seed42_no_text_head/final"
    )
    assert contract["shared_hf_runtime"]["role_max_new_tokens"] == {
        "controller": 96,
        "subanswer_reader": 96,
        "final_reader": 512,
    }
    assert contract["canonical_retrieval"] == {
        "dense_top_k": 100,
        "bm25_top_k": 100,
        "rrf_k": 60,
        "rrf_output_k": 100,
        "bge_top_k": 10,
        "query_max_tokens": 128,
        "rerank_text_chars": 1200,
        "model_visible_passage_chars": 1200,
        "expected_documents": 21015324,
        "corpus_path": "indexes_wiki18/corpus_flashrag.jsonl",
        "dense_index_path": "indexes_wiki18/e5_fp16.dat",
        "bm25_index_path": "indexes_wiki18/bm25",
        "e5_model_path": "models/e5-base-v2",
        "bge_model_path": "models/bge-reranker-v2-m3",
        "silent_fallback_allowed": False,
    }


def test_production_runtime_contract_is_ast_literal_and_matches_public_copy():
    source = runtime.Path(runtime.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "PRODUCTION_RUNTIME_CONTRACT"
            for target in node.targets
        )
    )
    literal = ast.literal_eval(assignment.value)
    assert literal == runtime.runtime_contract()
    copied = runtime.runtime_contract()
    copied["seed"] = 0
    assert runtime.PRODUCTION_RUNTIME_CONTRACT["seed"] == 42


def test_canonical_retriever_reranks_exact_rrf100_to_stable_top10_and_binds_cache():
    raw, ce = FakeRRF(), FakeCrossEncoder()
    retriever = runtime.CanonicalRetrieverRuntime(
        rrf_retriever=raw,
        cross_encoder=ce,
        retrieval_asset_identity=RETRIEVAL_IDENTITY,
    )
    first = retriever.batch_search(["test query"])[0]
    assert [row["id"] for row in first] == [
        f"rrf-{index:03d}" for index in range(100, 90, -1)
    ]
    assert [row["rerank_score"] for row in first] == list(range(99, 89, -1))
    assert len(ce.pairs[0]) == 100
    key = retriever.cache_key_payload("test query")
    assert key["assets"] == RETRIEVAL_IDENTITY
    assert key["config"]["rrf_output_k"] == 100
    assert key["config"]["rerank_top_k"] == 10

    cached = runtime._CachedRetriever(retriever)
    cached.invoke("test query", logical_arms=(runtime.ARM_A,), slot="root_A")
    _, event = cached.invoke(
        "test query", logical_arms=(runtime.ARM_B, runtime.ARM_C), slot="root_BC"
    )
    assert event["cache_hit"] is True
    assert event["content_cache_key_mode"] == "asset_bound_query_and_retrieval_stack"
    assert cached.physical_calls == 1
    assert cached.logical_cache_hits == 2
    assert cached.logical_cache_misses == 1


def test_cached_retriever_batches_stable_unique_misses_without_changing_logical_semantics():
    backend = FakeAssetBoundTop10Retriever()
    backend.calls = []
    original_batch_search = backend.batch_search

    def recording_batch_search(queries):
        backend.calls.append(list(queries))
        return original_batch_search(queries)

    backend.batch_search = recording_batch_search
    cached = runtime._CachedRetriever(backend)
    outputs = cached.invoke_many(
        [
            {"query": "alpha", "logical_arms": (runtime.ARM_A,), "slot": "a"},
            {
                "query": "alpha",
                "logical_arms": (runtime.ARM_B, runtime.ARM_C),
                "slot": "bc",
            },
            {"query": "beta", "logical_arms": (runtime.ARM_B,), "slot": "b"},
            {"query": "alpha", "logical_arms": (runtime.ARM_C,), "slot": "c"},
        ],
        batch_stage="mixed",
    )
    assert backend.calls == [["alpha", "beta"]]
    assert [event["cache_hit"] for _, event in outputs] == [False, True, False, True]
    assert cached.physical_calls == 2
    assert cached.backend_batch_invocations == 1
    assert cached.full_index_passes == 1
    assert cached.logical_cache_misses == 2
    assert cached.logical_cache_hits == 3
    assert cached.logical_calls == {
        runtime.ARM_A: 1,
        runtime.ARM_B: 2,
        runtime.ARM_C: 2,
    }

    cached.invoke_many(
        [
            {"query": "beta", "logical_arms": (runtime.ARM_A,), "slot": "cached"},
            {"query": "alpha", "logical_arms": (runtime.ARM_B,), "slot": "cached"},
        ],
        batch_stage="cached",
    )
    assert backend.calls == [["alpha", "beta"]]
    assert cached.backend_batch_invocations == 1
    assert cached.full_index_passes == 1
    assert cached.stage_batch_telemetry == [
        {
            "stage": "mixed",
            "logical_request_groups": 4,
            "unique_miss_query_count": 2,
            "backend_invoked": True,
        },
        {
            "stage": "cached",
            "logical_request_groups": 2,
            "unique_miss_query_count": 0,
            "backend_invoked": False,
        },
    ]


def test_canonical_retriever_fails_closed_on_pool_or_bge_drift():
    short = runtime.CanonicalRetrieverRuntime(
        rrf_retriever=FakeRRF(count=99),
        cross_encoder=FakeCrossEncoder(),
        retrieval_asset_identity=RETRIEVAL_IDENTITY,
    )
    with pytest.raises(runtime.V8RunnerError, match="rather than 100"):
        short.batch_search(["query"])

    nonfinite = runtime.CanonicalRetrieverRuntime(
        rrf_retriever=FakeRRF(),
        cross_encoder=FakeCrossEncoder(nonfinite=True),
        retrieval_asset_identity=RETRIEVAL_IDENTITY,
    )
    with pytest.raises(runtime.V8RunnerError, match="non-finite"):
        nonfinite.batch_search(["query"])


class TinyTokenizer:
    chat_template = "tiny deterministic chat template"
    pad_token_id = 0
    eos_token_id = 2

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        assert add_generation_prompt is True
        return json.dumps(messages, ensure_ascii=False, sort_keys=True) + "<GEN>"

    def __call__(self, prompt, add_special_tokens=False, truncation=False):
        assert add_special_tokens is False
        assert truncation is False
        return {"input_ids": [byte + 3 for byte in prompt.encode("utf-8")]}

    def decode(self, token_ids, skip_special_tokens=True):
        assert skip_special_tokens is True
        values = token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids)
        return "".join(chr(value) for value in values)


class TinyModel:
    def __init__(self, torch_module):
        self.torch = torch_module
        self.calls = 0

    def generate(self, *, input_ids, **kwargs):
        self.calls += 1
        suffix = self.torch.tensor([[79, 75]], dtype=input_ids.dtype, device=input_ids.device)
        return self.torch.cat([input_ids, suffix], dim=1)


def test_shared_hf_wrapper_binds_role_assets_template_full_ids_and_reuses_one_model():
    torch = pytest.importorskip("torch")
    tokenizer = TinyTokenizer()
    model = TinyModel(torch)
    shared = runtime.SharedHuggingFaceRuntime._from_loaded_components_for_test(
        torch_module=torch,
        tokenizer=tokenizer,
        model=model,
        device=torch.device("cpu"),
        model_asset_identity=MODEL_IDENTITY,
    )
    controller = shared.bind_role("controller")
    subanswer = shared.bind_role("subanswer_reader")
    final = shared.bind_role("final_reader")
    assert controller.runtime is shared
    assert final.runtime is shared
    messages = [{"role": "user", "content": "hello"}]
    controller_key = controller.cache_key_payload(messages)
    final_key = final.cache_key_payload(messages)
    assert controller_key["role"] == "controller"
    assert final_key["role"] == "final_reader"
    assert controller_key["decoding"]["max_new_tokens"] == 96
    assert final_key["decoding"]["max_new_tokens"] == 512
    assert controller_key["base_model_tree_sha256"] == "1" * 64
    assert controller_key["chat_template_sha256"] == hashlib.sha256(
        tokenizer.chat_template.encode()
    ).hexdigest()
    assert controller_key["model_visible_prompt_token_ids"]
    assert controller_key != final_key

    invoker = runtime._CachedTextInvoker(controller, label="controller")
    response, event = invoker.invoke(
        messages, logical_arms=(runtime.ARM_B,), slot="q1"
    )
    cached_response, cached_event = invoker.invoke(
        messages, logical_arms=(runtime.ARM_C,), slot="q1"
    )
    assert response == cached_response == "OK"
    assert event["content_cache_key_mode"] == "asset_bound_model_visible_token_ids"
    assert cached_event["cache_hit"] is True
    subanswer_result = subanswer(messages)
    final_result = final(messages)
    object_ids = {
        event["shared_model_object_id"]
        for event in (
            event["runtime_telemetry"],
            subanswer_result.runtime_telemetry,
            final_result.runtime_telemetry,
        )
    }
    assert object_ids == {id(model)}
    assert model.calls == 3


def test_shared_hf_wrapper_fails_closed_instead_of_truncating_oversize_prompt():
    torch = pytest.importorskip("torch")

    class OversizeTokenizer(TinyTokenizer):
        def __call__(self, prompt, add_special_tokens=False, truncation=False):
            assert truncation is False
            return {"input_ids": [7] * (runtime.MODEL_MAX_INPUT_TOKENS + 1)}

    shared = runtime.SharedHuggingFaceRuntime._from_loaded_components_for_test(
        torch_module=torch,
        tokenizer=OversizeTokenizer(),
        model=TinyModel(torch),
        device=torch.device("cpu"),
        model_asset_identity=MODEL_IDENTITY,
    )
    with pytest.raises(runtime.V8RunnerError, match="truncation is forbidden"):
        shared.prepare(
            [{"role": "user", "content": "oversize"}],
            role="controller",
            max_new_tokens=runtime.CONTROLLER_MAX_NEW_TOKENS,
        )


class FakeRoleBackend:
    def __init__(self, shared, role):
        self.runtime = shared
        self.role = role

    def cache_key_payload(self, messages):
        prompt = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        return {
            "cache_schema": "test-asset-bound-generation",
            "role": self.role,
            "model": MODEL_IDENTITY,
            "model_visible_prompt_token_ids": list(prompt.encode("utf-8")),
        }

    def __call__(self, messages):
        if self.role == "controller":
            payload = json.loads(messages[1]["content"])
            if payload["task"] == "generate_q1":
                response = "Who directed Film Alpha?"
            else:
                response = "In which city was Film Alpha's director born?"
        elif self.role == "subanswer_reader":
            response = self.runtime.subanswer_response
        else:
            response = "[Step 1] Use the supplied passages.\n[Final Answer] Test answer"
        return runtime.TextGenerationResult(
            response_text=response,
            prompt_tokens=10,
            generation_tokens=5,
            runtime_telemetry={"shared_runtime_id": id(self.runtime)},
        )


class FakeSharedRuntime:
    def __init__(self, *, subanswer_response=NO_RELEVANT_ANSWER):
        self.calls = []
        self.subanswer_response = subanswer_response

    def bind_role(self, role):
        backend = FakeRoleBackend(self, role)
        original_call = backend.__call__

        class RecordingBackend(FakeRoleBackend):
            def __call__(recording_self, messages):
                self.calls.append(role)
                return original_call(messages)

        return RecordingBackend(self, role)


class FakeAssetBoundTop10Retriever:
    def cache_key_payload(self, query):
        return {
            "cache_schema": "test-asset-bound-retrieval",
            "query_utf8": query,
            "assets": RETRIEVAL_IDENTITY,
            "config": {"dense": 100, "bm25": 100, "rrf": 100, "bge": 10},
        }

    def batch_search(self, queries):
        rows = []
        for query in queries:
            prefix = hashlib.sha256(query.encode()).hexdigest()[:12]
            rows.append(_docs(prefix, 10, gold_poison=True))
        return rows


def test_locked_consumed_smoke_loader_is_fixed_4x3_and_has_no_path_override():
    assert not inspect.signature(runtime.load_locked_consumed_smoke4x3).parameters
    cohort = runtime.load_locked_consumed_smoke4x3()
    assert cohort["row_count"] == 12
    assert cohort["per_dataset_counts"] == {
        "hotpotqa": 4,
        "2wikimultihopqa": 4,
        "musique": 4,
    }
    assert cohort["gold_access"] is False
    assert cohort["prospective_unlocked"] is False
    assert all(set(row) == {"dataset", "qid", "question"} for row in cohort["rows"])


def test_complete_consumed_smoke_fake_runtime_closes_A_B_C_and_emits_no_gold():
    shared_runtime = FakeSharedRuntime()
    result = runtime.materialize_locked_consumed_smoke4x3_production(
        hf_runtime=shared_runtime,
        retriever_runtime=FakeAssetBoundTop10Retriever(),
    )
    assert result["row_count"] == 12
    assert result["scope"] == "LOCKED_CONSUMED_ENGINEERING_SMOKE4X3_ONLY"
    assert result["experiment_id"] == runtime.SMOKE_EXPERIMENT_ID
    assert result["logical_calls_by_arm"] == {
        runtime.ARM_A: {
            "retrieval": 12,
            "controller": 0,
            "subanswer_reader": 0,
            "final_reader": 12,
        },
        runtime.ARM_B: {
            "retrieval": 36,
            "controller": 24,
            "subanswer_reader": 12,
            "final_reader": 12,
        },
        runtime.ARM_C: {
            "retrieval": 36,
            "controller": 24,
            "subanswer_reader": 12,
            "final_reader": 12,
        },
    }
    for row in result["rows"]:
        assert set(row["arms"]) == {runtime.ARM_A, runtime.ARM_B, runtime.ARM_C}
        assert len(row["arms"][runtime.ARM_A]["final_passages"]) == 10
        assert len(row["shared"]["root_top10"]["documents"]) == 10
        assert len(row["shared"]["q1_top10"]["scores"]) == 10
        assert len(row["arms"][runtime.ARM_B]["q2_top10"]["documents"]) == 10
        assert row["counterfactual_identity"]["ineligible_c"] is True
        assert row["counterfactual_identity"]["b_c_q2_top10_byte_identical"] is True
        assert row["counterfactual_identity"]["b_c_final_prompt_byte_identical"] is True
        assert row["counterfactual_identity"]["b_c_prediction_byte_identical"] is True
        assert row["arms"][runtime.ARM_C]["final"]["reader_event"]["cache_hit"] is True
    serialized = json.dumps(result, ensure_ascii=False, sort_keys=True)
    assert "NEVER-EMIT-THIS" not in serialized
    assert '"gold_answer"' not in serialized
    assert result["gold_access"] is False
    assert result["prospective_unlocked"] is False
    assert sum(
        values["retrieval"] for values in result["logical_calls_by_arm"].values()
    ) == 84
    assert result["retrieval_batch_telemetry"] == {
        "backend_batch_invocations": 3,
        "full_index_passes": 3,
        "unique_query_count_by_batch": [12, 1, 1],
        "stage_batches": [
            {
                "stage": "root_all",
                "logical_request_groups": 24,
                "unique_miss_query_count": 12,
                "backend_invoked": True,
            },
            {
                "stage": "q1_all",
                "logical_request_groups": 12,
                "unique_miss_query_count": 1,
                "backend_invoked": True,
            },
            {
                "stage": "q2_BC_all",
                "logical_request_groups": 24,
                "unique_miss_query_count": 1,
                "backend_invoked": True,
            },
        ],
    }
    assert len(shared_runtime.calls) == sum(
        values["physical_executions"]
        for name, values in result["joint_cache_accounting"].items()
        if name != "retrieval"
    )


@pytest.mark.parametrize(
    "subanswer_response, expected_eligible",
    [(NO_RELEVANT_ANSWER, False), ("Evidence document 1", True)],
)
def test_staged_complete_executor_is_row_exact_to_sequential_fake_execution(
    subanswer_response, expected_eligible
):
    rows = [
        {
            "dataset": "hotpotqa",
            "qid": f"fake-{index}",
            "question": f"{index}: {runtime.SMOKE_EXPERIMENT_ID}?",
        }
        for index in range(2)
    ]

    # Use one fake physical runtime so runtime-only object identity telemetry is
    # also equal; invoker caches remain separate between the two executions.
    sequential_shared = FakeSharedRuntime(subanswer_response=subanswer_response)
    sequential_controller = runtime._CachedTextInvoker(
        sequential_shared.bind_role("controller"), label="controller"
    )
    sequential_subanswer = runtime._CachedTextInvoker(
        sequential_shared.bind_role("subanswer_reader"), label="subanswer_reader"
    )
    sequential_final = runtime._CachedTextInvoker(
        sequential_shared.bind_role("final_reader"), label="final_reader"
    )
    sequential_retriever = runtime._CachedRetriever(FakeAssetBoundTop10Retriever())
    sequential_rows = [
        runtime._run_complete_identity_row(
            row,
            controller=sequential_controller,
            subanswer_reader=sequential_subanswer,
            final_reader=sequential_final,
            retriever=sequential_retriever,
        )
        for row in rows
    ]
    sequential_generation_calls = len(sequential_shared.calls)

    staged_shared = sequential_shared
    staged_controller = runtime._CachedTextInvoker(
        staged_shared.bind_role("controller"), label="controller"
    )
    staged_subanswer = runtime._CachedTextInvoker(
        staged_shared.bind_role("subanswer_reader"), label="subanswer_reader"
    )
    staged_final = runtime._CachedTextInvoker(
        staged_shared.bind_role("final_reader"), label="final_reader"
    )
    staged_retriever = runtime._CachedRetriever(FakeAssetBoundTop10Retriever())
    staged_rows = runtime._run_complete_rows_staged(
        rows,
        controller=staged_controller,
        subanswer_reader=staged_subanswer,
        final_reader=staged_final,
        retriever=staged_retriever,
    )

    assert json.dumps(staged_rows, ensure_ascii=False, sort_keys=True) == json.dumps(
        sequential_rows, ensure_ascii=False, sort_keys=True
    )
    assert staged_retriever.backend_batch_invocations == 3
    assert staged_retriever.full_index_passes == 3
    assert all(
        row["arms"][runtime.ARM_C]["dynamic_eligible"] is expected_eligible
        for row in staged_rows
    )
    assert len(staged_shared.calls) - sequential_generation_calls == (
        sequential_generation_calls
    )


def test_formal_production_entrypoint_requests_only_complete_development90(monkeypatch):
    observed = []

    def stop_loader(*, role):
        observed.append(role)
        raise RuntimeError("stop before execution")

    monkeypatch.setattr(runtime, "load_frozen_v8_cohort", stop_loader)
    with pytest.raises(RuntimeError, match="before execution"):
        runtime.materialize_frozen_development_production(
            hf_runtime=FakeSharedRuntime(),
            retriever_runtime=FakeAssetBoundTop10Retriever(),
        )
    assert observed == [runtime.DEVELOPMENT_ROLE]
