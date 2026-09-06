#!/usr/bin/env python
"""Development-only execution core for dynamic decomposition v8.

The public entry point always obtains identities from the locked development
cohort loader.  It has no prospective role/path argument and performs no file
writes.  Model/retrieval backends are injectable so the complete state machine,
counterfactual prompt identity, and logical/physical budgets can be exercised
with CPU-only fakes before a production materializer is authorized.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import re
from typing import Any, Mapping, Protocol, Sequence

from kgproweight.data.prompts import build_inference_messages
from kgproweight.retrieval.dynamic_decomposition_v8 import (
    NO_VERIFIED_SUBANSWER,
    PASSAGE_TEXT_MAX_CHARS,
    QUERY_CONTRACT_VERSION,
    QueryParseError,
    build_dynamic_q2_action,
    build_dynamic_q2_state,
    build_static_q2_action,
    build_static_q2_state,
    merge_fixed_budget_passages,
    parse_and_bind_subanswer,
    parse_query_response,
    project_top10_passages_for_prompt,
)
from kgproweight.retrieval.dynamic_decomposition_v8_cohort import (
    DEVELOPMENT_ROLE,
    load_frozen_v8_cohort,
)


RUNNER_VERSION = "dynamic-decomposition-v8-two-controller-calls-development-1"
OUTPUT_SCHEMA_VERSION = "dynamic-decomposition-v8-development-materialization-row-1"
COMPLETE_OUTPUT_SCHEMA_VERSION = "dynamic-decomposition-v8-complete-gold-free-row-1"
PRODUCTION_RUNTIME_VERSION = "dynamic-decomposition-v8-production-runtime-1"
Q1_PROMPT_VERSION = "dynamic-decomposition-v8-q1-controller-prompt-1"
Q2_PROMPT_VERSION = "dynamic-decomposition-v8-q2-controller-prompt-1"
SUBANSWER_PROMPT_VERSION = "dynamic-decomposition-v8-subanswer-reader-prompt-1"

ARM_A = "A_canonical_one_shot"
ARM_B = "B_observation_blind"
ARM_C = "C_answer_conditioned"
ARMS = (ARM_B, ARM_C)
ALL_ARMS = (ARM_A, ARM_B, ARM_C)
IDENTITY_FIELDS = ("dataset", "qid", "question")
BOUND_EVIDENCE_FIELDS = frozenset(
    {
        "verified_answer",
        "supporting_document_key",
        "supporting_doc_id",
        "supporting_doc_rank",
        "supporting_sentence_sha256",
        "support_location",
        "support_unit_index",
        "bound_evidence_excerpt",
        "bound_evidence_excerpt_sha256",
        "supporting_document_prompt_sha256",
    }
)

# The formal runtime contract is deliberately literal so the separate
# implementation-freeze job can bind it without guessing defaults from helper
# modules.  This module never downloads a missing asset or silently substitutes
# another retriever/model.
BASE_MODEL_PATH = "models/llama3-8b"
STRONG_SFT_PATH = (
    "checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42_no_text_head/final"
)
E5_MODEL_PATH = "models/e5-base-v2"
BGE_MODEL_PATH = "models/bge-reranker-v2-m3"
CORPUS_PATH = "indexes_wiki18/corpus_flashrag.jsonl"
DENSE_INDEX_PATH = "indexes_wiki18/e5_fp16.dat"
BM25_INDEX_PATH = "indexes_wiki18/bm25"

SEED = 42
DENSE_TOP_K = 100
BM25_TOP_K = 100
RRF_K = 60
RRF_CANDIDATE_K = 100
RERANK_TOP_K = 10
QUERY_MAX_TOKENS = 128
RERANK_TEXT_CHARS = 1200
EXPECTED_WIKI18_DOCUMENTS = 21_015_324
MODEL_MAX_INPUT_TOKENS = 6144
CONTROLLER_MAX_NEW_TOKENS = 96
SUBANSWER_MAX_NEW_TOKENS = 96
FINAL_MAX_NEW_TOKENS = 512

SMOKE_EXPERIMENT_ID = (
    "SUBQUESTION-DECOMPOSITION-V8-RRF100-ENGINEERING-SMOKE4X3-SEED42-ATTEMPT001"
)
DEVELOPMENT_EXPERIMENT_ID = (
    "SUBQUESTION-DECOMPOSITION-V8-RRF100-DEVELOPMENT90-SEED42-ATTEMPT001"
)
SMOKE_OUTPUT_DIR = (
    "outputs/audits/"
    "subquestion_decomposition_v8_rrf100_engineering_smoke4x3_seed42_attempt001"
)
DEVELOPMENT_OUTPUT_DIR = (
    "outputs/audits/"
    "subquestion_decomposition_v8_rrf100_development90_seed42_attempt001"
)

SMOKE_COHORT_DIRECTORY = Path(
    "outputs/audits/"
    "subquestion_decomposition_v8_consumed_smoke4x3_seed20260904_v1"
)
SMOKE_COHORT_PATH = SMOKE_COHORT_DIRECTORY / "smoke.identity_only.jsonl"
SMOKE_REPORT_PATH = SMOKE_COHORT_DIRECTORY / "report.json"
SMOKE_MANIFEST_PATH = SMOKE_COHORT_DIRECTORY / "manifest.json"
SMOKE_COHORT_SHA256 = "3b2eb4da9abefc09c3df97083aa65462d6e51e7648cc811ae59f8e8266671606"
SMOKE_REPORT_SHA256 = "2156f59aebaeca8e34fbf0b1e40d46f66d54aa8349aef0af46c0b6ff8baa683b"
SMOKE_MANIFEST_SHA256 = "7eed188667cf8f9436c002705d3d11e0181c95f13b4986320c5a16695ddbf30b"

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Keep this assignment a pure Python literal: the independent implementation
# freezer reads it with ``ast.literal_eval`` and never imports this runner.
PRODUCTION_RUNTIME_CONTRACT = {
    "schema_version": "dynamic-decomposition-v8-production-runtime-contract-1",
    "runtime_version": "dynamic-decomposition-v8-production-runtime-1",
    "gold_access": False,
    "prospective_unlocked": False,
    "seed": 42,
    "production_staged": True,
    "staged_retrieval_contract": {
        "stable_deduplicate_cache_misses": True,
        "backend_batch_stages": ["root_all", "q1_all", "q2_BC_all"],
        "engineering_smoke_logical_retrieval_requests": 84,
        "maximum_full_index_passes_per_attempt": 3,
    },
    "shared_hf_runtime": {
        "one_physical_model_instance_for_roles": [
            "controller",
            "subanswer_reader",
            "final_reader",
        ],
        "base_model_path": "models/llama3-8b",
        "strong_sft_adapter_path": "checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42_no_text_head/final",
        "tokenizer_path": "models/llama3-8b",
        "tokenizer_source": "base_model_tokenizer_matching_legacy_SFT_evaluation",
        "pad_token_policy": "set_to_eos_when_missing",
        "torch_dtype": "bfloat16",
        "device": "cuda:0",
        "local_files_only": True,
        "peft_is_trainable": False,
        "chat_template_source": "base_tokenizer.chat_template",
        "chat_template_add_generation_prompt": True,
        "model_input_truncation": False,
        "max_input_tokens_fail_closed": 6144,
        "role_max_new_tokens": {
            "controller": 96,
            "subanswer_reader": 96,
            "final_reader": 512,
        },
        "decoding": {
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "seed": 42,
        },
    },
    "canonical_retrieval": {
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
    },
    "logical_budget_by_arm": {
        "A_canonical_one_shot": {
            "retrieval": 1,
            "controller": 0,
            "subanswer_reader": 0,
            "final_reader": 1,
        },
        "B_observation_blind": {
            "retrieval": 3,
            "controller": 2,
            "subanswer_reader": 1,
            "final_reader": 1,
        },
        "C_answer_conditioned": {
            "retrieval": 3,
            "controller": 2,
            "subanswer_reader": 1,
            "final_reader": 1,
        },
    },
    "cache_contract": {
        "scope": "in_memory_for_one_locked_materialization_attempt",
        "persistent_cache_in_this_runner": False,
        "outer_append_only_resume_required": True,
        "logical_requests_equal_cache_hits_plus_cache_misses": True,
        "physical_executions_equal_cache_misses": True,
        "generation_key_binds": [
            "role",
            "base_model_tree_sha256",
            "adapter_tree_sha256",
            "tokenizer_tree_sha256",
            "chat_template_sha256",
            "model_visible_prompt_utf8_sha256",
            "model_visible_prompt_token_ids",
            "decoding",
        ],
        "retrieval_key_binds": [
            "query_utf8",
            "corpus_sha256",
            "dense_index_sha256",
            "bm25_tree_sha256",
            "e5_tree_sha256",
            "bge_tree_sha256",
            "complete_retrieval_config",
        ],
        "key_forbidden_fields": ["arm_label", "outcome", "gold"],
    },
    "first_attempts": {
        "engineering_smoke": {
            "experiment_id": "SUBQUESTION-DECOMPOSITION-V8-RRF100-ENGINEERING-SMOKE4X3-SEED42-ATTEMPT001",
            "output_dir": "outputs/audits/subquestion_decomposition_v8_rrf100_engineering_smoke4x3_seed42_attempt001",
            "cohort_path": "outputs/audits/subquestion_decomposition_v8_consumed_smoke4x3_seed20260904_v1/smoke.identity_only.jsonl",
            "cohort_sha256": "3b2eb4da9abefc09c3df97083aa65462d6e51e7648cc811ae59f8e8266671606",
            "scope": "CONSUMED_IDENTITY_ONLY_4X3_NOT_FRESH_DEVELOPMENT",
        },
        "development": {
            "experiment_id": "SUBQUESTION-DECOMPOSITION-V8-RRF100-DEVELOPMENT90-SEED42-ATTEMPT001",
            "output_dir": "outputs/audits/subquestion_decomposition_v8_rrf100_development90_seed42_attempt001",
            "scope": "FROZEN_DEVELOPMENT90_ONLY",
        },
    },
}


class V8RunnerError(ValueError):
    """The injected backend or execution trace violates the v8 contract."""


class TextBackend(Protocol):
    def __call__(self, messages: Sequence[Mapping[str, str]]) -> str: ...


class RetrievalBackend(Protocol):
    def batch_search(self, queries: Sequence[str]) -> Sequence[Sequence[Mapping[str, Any]]]: ...


def runtime_contract() -> dict[str, Any]:
    """Return the complete immutable production contract as plain data."""

    _assert_runtime_contract_alignment()
    return deepcopy(PRODUCTION_RUNTIME_CONTRACT)


def _assert_runtime_contract_alignment() -> None:
    contract = PRODUCTION_RUNTIME_CONTRACT
    shared = contract["shared_hf_runtime"]
    retrieval = contract["canonical_retrieval"]
    attempts = contract["first_attempts"]
    expected_shared = {
        "base_model_path": BASE_MODEL_PATH,
        "strong_sft_adapter_path": STRONG_SFT_PATH,
        "tokenizer_path": BASE_MODEL_PATH,
        "max_input_tokens_fail_closed": MODEL_MAX_INPUT_TOKENS,
    }
    if any(shared.get(key) != value for key, value in expected_shared.items()):
        raise V8RunnerError("literal shared-HF contract drifted from runtime constants")
    if shared.get("role_max_new_tokens") != {
        "controller": CONTROLLER_MAX_NEW_TOKENS,
        "subanswer_reader": SUBANSWER_MAX_NEW_TOKENS,
        "final_reader": FINAL_MAX_NEW_TOKENS,
    }:
        raise V8RunnerError("literal role decoding limits drifted")
    expected_retrieval = {
        "dense_top_k": DENSE_TOP_K,
        "bm25_top_k": BM25_TOP_K,
        "rrf_k": RRF_K,
        "rrf_output_k": RRF_CANDIDATE_K,
        "bge_top_k": RERANK_TOP_K,
        "query_max_tokens": QUERY_MAX_TOKENS,
        "rerank_text_chars": RERANK_TEXT_CHARS,
        "model_visible_passage_chars": PASSAGE_TEXT_MAX_CHARS,
        "expected_documents": EXPECTED_WIKI18_DOCUMENTS,
        "corpus_path": CORPUS_PATH,
        "dense_index_path": DENSE_INDEX_PATH,
        "bm25_index_path": BM25_INDEX_PATH,
        "e5_model_path": E5_MODEL_PATH,
        "bge_model_path": BGE_MODEL_PATH,
    }
    if any(retrieval.get(key) != value for key, value in expected_retrieval.items()):
        raise V8RunnerError("literal retrieval contract drifted from runtime constants")
    if contract.get("runtime_version") != PRODUCTION_RUNTIME_VERSION or contract.get(
        "seed"
    ) != SEED:
        raise V8RunnerError("literal runtime version/seed drifted")
    if contract.get("production_staged") is not True or contract.get(
        "staged_retrieval_contract"
    ) != {
        "stable_deduplicate_cache_misses": True,
        "backend_batch_stages": ["root_all", "q1_all", "q2_BC_all"],
        "engineering_smoke_logical_retrieval_requests": 84,
        "maximum_full_index_passes_per_attempt": 3,
    }:
        raise V8RunnerError("literal staged retrieval contract drifted")
    if attempts["engineering_smoke"] != {
        "experiment_id": SMOKE_EXPERIMENT_ID,
        "output_dir": SMOKE_OUTPUT_DIR,
        "cohort_path": SMOKE_COHORT_PATH.as_posix(),
        "cohort_sha256": SMOKE_COHORT_SHA256,
        "scope": "CONSUMED_IDENTITY_ONLY_4X3_NOT_FRESH_DEVELOPMENT",
    } or attempts["development"] != {
        "experiment_id": DEVELOPMENT_EXPERIMENT_ID,
        "output_dir": DEVELOPMENT_OUTPUT_DIR,
        "scope": "FROZEN_DEVELOPMENT90_ONLY",
    }:
        raise V8RunnerError("literal first-attempt identities drifted")


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise V8RunnerError("value is not canonical-JSON serializable") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _messages_sha256(messages: Sequence[Mapping[str, str]]) -> str:
    return _sha256_bytes(_canonical_json_bytes(list(messages)))


def _validate_sha256_identity(
    value: Mapping[str, Any], *, required_fields: frozenset[str], label: str
) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != required_fields:
        raise V8RunnerError(
            f"{label} must contain exactly {sorted(required_fields)}"
        )
    result: dict[str, str] = {}
    for field in sorted(required_fields):
        digest = value[field]
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise V8RunnerError(f"{label}.{field} must be a lowercase SHA256")
        result[field] = digest
    return result


MODEL_ASSET_IDENTITY_FIELDS = frozenset(
    {
        "base_model_tree_sha256",
        "adapter_tree_sha256",
        "tokenizer_tree_sha256",
    }
)
RETRIEVAL_ASSET_IDENTITY_FIELDS = frozenset(
    {
        "corpus_sha256",
        "dense_index_sha256",
        "bm25_tree_sha256",
        "e5_tree_sha256",
        "bge_tree_sha256",
    }
)


@dataclass(frozen=True)
class TextGenerationResult:
    response_text: str
    prompt_tokens: int
    generation_tokens: int
    runtime_telemetry: Mapping[str, Any]


@dataclass(frozen=True)
class _PreparedHFRequest:
    role: str
    messages_sha256: str
    rendered_prompt: str
    prompt_token_ids: tuple[int, ...]
    max_new_tokens: int
    cache_key_payload: Mapping[str, Any]


class SharedHuggingFaceRuntime:
    """One real base+LoRA instance shared across all three generation roles."""

    _ROLE_LIMITS = {
        "controller": CONTROLLER_MAX_NEW_TOKENS,
        "subanswer_reader": SUBANSWER_MAX_NEW_TOKENS,
        "final_reader": FINAL_MAX_NEW_TOKENS,
    }

    def __init__(self, *, model_asset_identity: Mapping[str, Any]) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; refusing a CPU/disk model fallback")
        base_path = Path(BASE_MODEL_PATH).resolve()
        adapter_path = Path(STRONG_SFT_PATH).resolve()
        if not base_path.is_dir() or not adapter_path.is_dir():
            raise FileNotFoundError("frozen base model or strong-SFT adapter is missing")
        random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
        if tokenizer.eos_token_id is None:
            raise RuntimeError("base tokenizer has no EOS token")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            base_path,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            is_trainable=False,
            local_files_only=True,
        )
        model.eval()
        device = next(model.parameters()).device
        if device.type != "cuda":
            raise RuntimeError(f"model loaded on {device}; refusing a CPU/disk fallback")
        self._initialize_loaded(
            torch_module=torch,
            tokenizer=tokenizer,
            model=model,
            device=device,
            model_asset_identity=model_asset_identity,
        )

    @classmethod
    def _from_loaded_components_for_test(
        cls,
        *,
        torch_module: Any,
        tokenizer: Any,
        model: Any,
        device: Any,
        model_asset_identity: Mapping[str, Any],
    ) -> "SharedHuggingFaceRuntime":
        """Build around fakes without touching CUDA; never used by formal entrypoints."""

        instance = cls.__new__(cls)
        instance._initialize_loaded(
            torch_module=torch_module,
            tokenizer=tokenizer,
            model=model,
            device=device,
            model_asset_identity=model_asset_identity,
        )
        return instance

    def _initialize_loaded(
        self,
        *,
        torch_module: Any,
        tokenizer: Any,
        model: Any,
        device: Any,
        model_asset_identity: Mapping[str, Any],
    ) -> None:
        self._torch = torch_module
        self._tokenizer = tokenizer
        self._model = model
        self._device = device
        self._model_asset_identity = _validate_sha256_identity(
            model_asset_identity,
            required_fields=MODEL_ASSET_IDENTITY_FIELDS,
            label="model_asset_identity",
        )
        chat_template = getattr(tokenizer, "chat_template", None)
        if not isinstance(chat_template, str) or not chat_template:
            raise V8RunnerError("tokenizer has no explicit chat_template")
        self._chat_template_sha256 = _sha256_text(chat_template)

    def prepare(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        role: str,
        max_new_tokens: int,
    ) -> _PreparedHFRequest:
        if role not in self._ROLE_LIMITS:
            raise V8RunnerError(f"unsupported shared-HF role: {role!r}")
        if max_new_tokens != self._ROLE_LIMITS[role]:
            raise V8RunnerError(f"{role} max_new_tokens differs from frozen contract")
        canonical_messages = deepcopy(list(messages))
        rendered = self._tokenizer.apply_chat_template(
            canonical_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if not isinstance(rendered, str) or not rendered:
            raise V8RunnerError("chat template returned an empty/non-string prompt")
        encoded = self._tokenizer(
            rendered,
            add_special_tokens=False,
            truncation=False,
        )
        token_ids = encoded.get("input_ids") if isinstance(encoded, Mapping) else None
        if (
            isinstance(token_ids, (str, bytes))
            or not isinstance(token_ids, Sequence)
            or not token_ids
            or any(type(token) is not int or token < 0 for token in token_ids)
        ):
            raise V8RunnerError("tokenizer returned invalid prompt token IDs")
        token_tuple = tuple(token_ids)
        if len(token_tuple) > MODEL_MAX_INPUT_TOKENS:
            raise V8RunnerError(
                f"prompt has {len(token_tuple)} tokens; truncation is forbidden above "
                f"{MODEL_MAX_INPUT_TOKENS}"
            )
        cache_payload = {
            "cache_schema": "v8-hf-generation-content-key-1",
            "role": role,
            "base_model_path": BASE_MODEL_PATH,
            "strong_sft_adapter_path": STRONG_SFT_PATH,
            "tokenizer_path": BASE_MODEL_PATH,
            **self._model_asset_identity,
            "chat_template_sha256": self._chat_template_sha256,
            "add_generation_prompt": True,
            "model_visible_prompt_utf8_sha256": _sha256_text(rendered),
            # Full IDs, rather than only a digest, are part of the canonical
            # content-key payload.  They are not copied into row telemetry.
            "model_visible_prompt_token_ids": list(token_tuple),
            "decoding": {
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "temperature": None,
                "top_p": None,
                "seed": SEED,
            },
        }
        return _PreparedHFRequest(
            role=role,
            messages_sha256=_messages_sha256(canonical_messages),
            rendered_prompt=rendered,
            prompt_token_ids=token_tuple,
            max_new_tokens=max_new_tokens,
            cache_key_payload=cache_payload,
        )

    def generate(self, request: _PreparedHFRequest) -> TextGenerationResult:
        if not isinstance(request, _PreparedHFRequest):
            raise V8RunnerError("shared HF generate requires a prepared request")
        input_ids = self._torch.tensor(
            [list(request.prompt_token_ids)],
            dtype=self._torch.long,
            device=self._device,
        )
        attention_mask = self._torch.ones_like(input_ids)
        is_cuda = getattr(self._device, "type", None) == "cuda"
        if is_cuda:
            self._torch.cuda.reset_peak_memory_stats(self._device)
        with self._torch.inference_mode():
            generated = self._model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=request.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        new_ids = generated[0, len(request.prompt_token_ids) :]
        response = self._tokenizer.decode(new_ids, skip_special_tokens=True)
        if not isinstance(response, str):
            raise V8RunnerError("tokenizer decode returned a non-string response")
        telemetry: dict[str, Any] = {
            "execution_device": str(self._device),
            "torch_dtype": "bfloat16",
            "shared_model_instance": True,
            # Runtime-only identity (never part of a cache key or scientific
            # input) lets the outer mechanism gate prove that controller,
            # subanswer, and final-reader generations used one physical model
            # object rather than three separately loaded replicas.
            "shared_model_object_id": id(self._model),
        }
        if is_cuda:
            telemetry.update(
                {
                    "cuda_max_memory_allocated_bytes": int(
                        self._torch.cuda.max_memory_allocated(self._device)
                    ),
                    "cuda_max_memory_reserved_bytes": int(
                        self._torch.cuda.max_memory_reserved(self._device)
                    ),
                }
            )
        return TextGenerationResult(
            response_text=response,
            prompt_tokens=len(request.prompt_token_ids),
            generation_tokens=int(len(new_ids)),
            runtime_telemetry=telemetry,
        )

    def bind_role(self, role: str) -> "RoleBoundHuggingFaceBackend":
        if role not in self._ROLE_LIMITS:
            raise V8RunnerError(f"unsupported shared-HF role: {role!r}")
        return RoleBoundHuggingFaceBackend(
            runtime=self,
            role=role,
            max_new_tokens=self._ROLE_LIMITS[role],
        )


class RoleBoundHuggingFaceBackend:
    """Role/decoding view over one shared physical HF runtime."""

    def __init__(
        self,
        *,
        runtime: SharedHuggingFaceRuntime,
        role: str,
        max_new_tokens: int,
    ) -> None:
        self.runtime = runtime
        self.role = role
        self.max_new_tokens = max_new_tokens

    def cache_key_payload(
        self, messages: Sequence[Mapping[str, str]]
    ) -> Mapping[str, Any]:
        request = self.runtime.prepare(
            messages,
            role=self.role,
            max_new_tokens=self.max_new_tokens,
        )
        return request.cache_key_payload

    def __call__(
        self, messages: Sequence[Mapping[str, str]]
    ) -> TextGenerationResult:
        request = self.runtime.prepare(
            messages,
            role=self.role,
            max_new_tokens=self.max_new_tokens,
        )
        return self.runtime.generate(request)


class CanonicalRetrieverRuntime:
    """Strict RRF100 + BGE top-10 wrapper with no silent reranker fallback."""

    def __init__(
        self,
        *,
        rrf_retriever: Any,
        cross_encoder: Any,
        retrieval_asset_identity: Mapping[str, Any],
    ) -> None:
        if not hasattr(rrf_retriever, "batch_search"):
            raise V8RunnerError("canonical RRF retriever lacks batch_search")
        if not hasattr(cross_encoder, "predict"):
            raise V8RunnerError("canonical BGE cross encoder lacks predict")
        self._retriever = rrf_retriever
        self._cross_encoder = cross_encoder
        self._asset_identity = _validate_sha256_identity(
            retrieval_asset_identity,
            required_fields=RETRIEVAL_ASSET_IDENTITY_FIELDS,
            label="retrieval_asset_identity",
        )

    @classmethod
    def from_local_assets(
        cls, *, retrieval_asset_identity: Mapping[str, Any]
    ) -> "CanonicalRetrieverRuntime":
        """Load the exact frozen local stack; never download or downgrade."""

        for required in (E5_MODEL_PATH, BGE_MODEL_PATH, BM25_INDEX_PATH):
            if not Path(required).resolve().is_dir():
                raise FileNotFoundError(f"required retrieval directory is missing: {required}")
        for required in (CORPUS_PATH, DENSE_INDEX_PATH):
            if not Path(required).resolve().is_file():
                raise FileNotFoundError(f"required retrieval file is missing: {required}")

        from kgproweight.retrieval.reranker import get_cross_encoder
        from scripts.pilot.audit_iterative_bridge_retrieval import (
            _build_retriever,
            _validate_full_wiki18_assets,
        )

        _validate_full_wiki18_assets(
            CORPUS_PATH,
            DENSE_INDEX_PATH,
            BM25_INDEX_PATH,
            expected_docs=EXPECTED_WIKI18_DOCUMENTS,
        )
        retriever = _build_retriever(
            "hotpotqa",
            RRF_CANDIDATE_K,
            corpus_path=CORPUS_PATH,
            dense_index_path=DENSE_INDEX_PATH,
            bm25_index_path=BM25_INDEX_PATH,
        )
        cross_encoder = get_cross_encoder(BGE_MODEL_PATH)
        return cls(
            rrf_retriever=retriever,
            cross_encoder=cross_encoder,
            retrieval_asset_identity=retrieval_asset_identity,
        )

    def cache_key_payload(self, query: str) -> Mapping[str, Any]:
        if not isinstance(query, str) or not query:
            raise V8RunnerError("canonical retrieval cache key requires a query")
        return {
            "cache_schema": "v8-canonical-retrieval-content-key-1",
            "query_utf8": query,
            "assets": deepcopy(self._asset_identity),
            "paths": {
                "corpus": CORPUS_PATH,
                "dense_index": DENSE_INDEX_PATH,
                "bm25_index": BM25_INDEX_PATH,
                "e5_model": E5_MODEL_PATH,
                "bge_model": BGE_MODEL_PATH,
            },
            "config": {
                "dense_top_k": DENSE_TOP_K,
                "bm25_top_k": BM25_TOP_K,
                "rrf_k": RRF_K,
                "rrf_output_k": RRF_CANDIDATE_K,
                "rerank_top_k": RERANK_TOP_K,
                "query_max_tokens": QUERY_MAX_TOKENS,
                "rerank_text_chars": RERANK_TEXT_CHARS,
                "expected_documents": EXPECTED_WIKI18_DOCUMENTS,
            },
        }

    @staticmethod
    def _document_id(document: Mapping[str, Any], *, rank: int) -> str:
        values = [
            str(document[key])
            for key in ("id", "doc_id", "document_id")
            if key in document and document[key] is not None
        ]
        if not values or len(set(values)) != 1 or not values[0]:
            raise V8RunnerError(f"RRF candidate {rank} has no unique stable id")
        return values[0]

    @staticmethod
    def _document_text(document: Mapping[str, Any], *, rank: int) -> str:
        text = document.get("contents") or document.get("text")
        if not isinstance(text, str) or not text.strip():
            raise V8RunnerError(f"RRF candidate {rank} has no text")
        return text

    def batch_search(
        self, queries: Sequence[str]
    ) -> list[list[dict[str, Any]]]:
        if isinstance(queries, (str, bytes)) or not isinstance(queries, Sequence):
            raise V8RunnerError("canonical retriever queries must be a sequence")
        clean_queries = list(queries)
        if not clean_queries or any(not isinstance(query, str) or not query for query in clean_queries):
            raise V8RunnerError("canonical retriever received an empty/invalid query")
        raw_batches = self._retriever.batch_search(clean_queries)
        if (
            isinstance(raw_batches, (str, bytes))
            or not isinstance(raw_batches, Sequence)
            or len(raw_batches) != len(clean_queries)
        ):
            raise V8RunnerError("RRF retriever returned the wrong batch cardinality")
        results: list[list[dict[str, Any]]] = []
        for query, raw in zip(clean_queries, raw_batches):
            if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
                raise V8RunnerError("RRF retriever returned a non-sequence row")
            candidates = [deepcopy(dict(document)) for document in raw if isinstance(document, Mapping)]
            if len(candidates) != len(raw):
                raise V8RunnerError("RRF candidates contain a non-object document")
            if len(candidates) != RRF_CANDIDATE_K:
                raise V8RunnerError(
                    f"RRF returned {len(candidates)} rather than {RRF_CANDIDATE_K} candidates"
                )
            ids = [
                self._document_id(document, rank=rank)
                for rank, document in enumerate(candidates, start=1)
            ]
            if len(set(ids)) != len(ids):
                raise V8RunnerError("RRF candidate pool contains duplicate document ids")
            pairs = [
                (
                    query,
                    self._document_text(document, rank=rank)[:RERANK_TEXT_CHARS],
                )
                for rank, document in enumerate(candidates, start=1)
            ]
            raw_scores = self._cross_encoder.predict(pairs, show_progress_bar=False)
            try:
                scores = [float(score) for score in raw_scores]
            except TypeError:
                scores = [float(raw_scores)]
            if len(scores) != RRF_CANDIDATE_K or any(
                not math.isfinite(score) for score in scores
            ):
                raise V8RunnerError("BGE returned missing/non-finite candidate scores")
            order = sorted(range(len(candidates)), key=lambda index: (-scores[index], index))
            selected: list[dict[str, Any]] = []
            for index in order[:RERANK_TOP_K]:
                document = deepcopy(candidates[index])
                document["rerank_score"] = scores[index]
                selected.append(document)
            project_top10_passages_for_prompt(selected, role="canonical_bge_top10")
            results.append(selected)
        return results


def _messages(system: str, payload: Mapping[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": json.dumps(
                dict(payload),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        },
    ]


def build_q1_controller_messages(
    *,
    original_question: str,
    root_passages: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    documents = project_top10_passages_for_prompt(root_passages, role="root")
    payload = {
        "prompt_version": Q1_PROMPT_VERSION,
        "gold_access": False,
        "task": "generate_q1",
        "original_question": original_question,
        "root_documents": documents,
    }
    system = (
        "Generate one natural-language single-hop search question that makes "
        "progress toward the original question using the supplied root documents. "
        "Output only that one query line: no JSON, labels, PID, explanation, or "
        "placeholder."
    )
    return _messages(system, payload)


def build_subanswer_reader_messages(
    *,
    original_question: str,
    q1_query: str,
    q1_passages: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    documents = project_top10_passages_for_prompt(q1_passages, role="q1")
    payload = {
        "prompt_version": SUBANSWER_PROMPT_VERSION,
        "gold_access": False,
        "task": "answer_q1",
        "original_question": original_question,
        "q1_query": q1_query,
        "q1_documents": documents,
    }
    system = (
        "Answer only q1 from the supplied q1 documents. Output one concise "
        "extractive answer line, or exactly NO_RELEVANT_ANSWER when no unique "
        "document-grounded answer is available. Do not output JSON, type, "
        "citation, abstention flag, reasoning, or the final answer unless q1 asks it."
    )
    return _messages(system, payload)


def build_q2_controller_messages(state: Mapping[str, Any]) -> list[dict[str, str]]:
    """Serialize only a state returned by one of the two frozen core builders."""

    if not isinstance(state, Mapping):
        raise V8RunnerError("q2 controller state must be an object")
    mode = state.get("mode")
    if mode == "q2_no_verified_subanswer":
        required = {
            "state_version",
            "mode",
            "gold_access",
            "original_question",
            "q1_query",
            "verified_subanswer",
        }
        if set(state) != required or state.get("verified_subanswer") != NO_VERIFIED_SUBANSWER:
            raise V8RunnerError("observation-blind q2 state violates its exact schema")
    elif mode == "q2_dynamic":
        required = {
            "state_version",
            "mode",
            "gold_access",
            "original_question",
            "q1_query",
            "verified_subanswer",
            "bound_evidence",
        }
        evidence = state.get("bound_evidence")
        if set(state) != required or not isinstance(evidence, Mapping):
            raise V8RunnerError("dynamic q2 state violates its exact schema")
        if set(evidence) != BOUND_EVIDENCE_FIELDS:
            raise V8RunnerError("dynamic q2 bound evidence violates its exact allowlist")
        if state.get("verified_subanswer") != evidence.get("verified_answer"):
            raise V8RunnerError("dynamic q2 answer/provenance binding mismatch")
    else:
        raise V8RunnerError(f"unsupported q2 controller state mode: {mode!r}")
    if state.get("gold_access") is not False:
        raise V8RunnerError("q2 controller state gold_access must be false")
    payload = {
        "prompt_version": Q2_PROMPT_VERSION,
        "gold_access": False,
        "task": "generate_q2",
        "state": deepcopy(dict(state)),
    }
    system = (
        "Generate the next natural-language single-hop search question from the "
        "provided state. NO_VERIFIED_SUBANSWER means no q1 answer or observation "
        "is available and must not be inferred. Otherwise use only the verified "
        "subanswer and bound evidence supplied in the state. Output one query line "
        "only: no JSON, labels, PID, explanation, or placeholder."
    )
    return _messages(system, payload)


def build_final_reader_messages(
    *,
    original_question: str,
    final_passages: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Build the shared strong-SFT answer prompt with no KG or Gold input."""

    safe_passages = project_top10_passages_for_prompt(
        final_passages,
        role="final_reader",
    )
    return build_inference_messages(
        question=original_question,
        retrieved_passages=safe_passages,
        kg_triples=[],
        top_k=RERANK_TOP_K,
        max_kg_triples=0,
    )


def _safe_top10_snapshot(
    passages: Sequence[Mapping[str, Any]], *, role: str
) -> dict[str, Any]:
    """Record all model-visible documents plus non-semantic rank-score telemetry."""

    documents = project_top10_passages_for_prompt(passages, role=role)
    scores: list[float | None] = []
    for rank, passage in enumerate(passages, start=1):
        score: float | None = None
        for field in ("rerank_score", "score", "retrieval_score"):
            if field not in passage or passage[field] is None:
                continue
            raw = passage[field]
            if isinstance(raw, bool):
                raise V8RunnerError(f"{role} passage {rank} score is boolean")
            try:
                score = float(raw)
            except (TypeError, ValueError) as exc:
                raise V8RunnerError(f"{role} passage {rank} score is invalid") from exc
            if not math.isfinite(score):
                raise V8RunnerError(f"{role} passage {rank} score is non-finite")
            break
        scores.append(score)
    return {
        "documents": documents,
        "documents_sha256": _sha256_bytes(_canonical_json_bytes(documents)),
        "ranks": list(range(1, RERANK_TOP_K + 1)),
        "scores": scores,
        "gold_access": False,
    }


def _build_q1_action(response_text: str, *, original_question: str) -> dict[str, Any]:
    response_hash = _sha256_text(response_text) if isinstance(response_text, str) else None
    try:
        parsed = parse_query_response(response_text, previous_queries=(original_question,))
    except QueryParseError as exc:
        return {
            "contract_version": QUERY_CONTRACT_VERSION,
            "gold_access": False,
            "slot": "q1_shared",
            "response_sha256": response_hash,
            "proposal_valid": False,
            "proposal_query": None,
            "parse_error": exc.code,
            "selected_query": original_question,
            "selection_source": "original_question",
            "used_fallback": True,
            "fallback_reason": f"invalid_q1:{exc.code}",
        }
    return {
        "contract_version": QUERY_CONTRACT_VERSION,
        "gold_access": False,
        "slot": "q1_shared",
        "response_sha256": response_hash,
        "proposal_valid": True,
        "proposal_query": parsed["query"],
        "parse_error": None,
        "selected_query": parsed["query"],
        "selection_source": "q1",
        "used_fallback": False,
        "fallback_reason": None,
    }


class _CachedTextInvoker:
    def __init__(self, backend: TextBackend, *, label: str):
        if not callable(backend):
            raise V8RunnerError(f"{label} backend must be callable")
        self.backend = backend
        self.label = label
        self.cache: dict[str, tuple[bytes, str, dict[str, Any]]] = {}
        self.logical_calls: Counter[str] = Counter()
        self.logical_cache_hits = 0
        self.logical_cache_misses = 0
        self.physical_calls = 0
        self.events: list[dict[str, Any]] = []

    def invoke(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        logical_arms: Sequence[str],
        slot: str,
    ) -> tuple[str, dict[str, Any]]:
        if not logical_arms or any(arm not in ALL_ARMS for arm in logical_arms):
            raise V8RunnerError(f"invalid logical arms for {self.label}: {logical_arms}")
        prompt_bytes = _canonical_json_bytes(list(messages))
        prompt_sha = _sha256_bytes(prompt_bytes)
        cache_payload_builder = getattr(self.backend, "cache_key_payload", None)
        if callable(cache_payload_builder):
            cache_payload = cache_payload_builder(deepcopy(list(messages)))
            if not isinstance(cache_payload, Mapping):
                raise V8RunnerError(f"{self.label} asset-bound cache payload is not an object")
            cache_key_bytes = _canonical_json_bytes(dict(cache_payload))
            cache_key_mode = "asset_bound_model_visible_token_ids"
        else:
            # Test-only/injectable backends are scoped by role label and exact
            # messages.  Formal runtime backends must expose cache_key_payload.
            cache_key_bytes = _canonical_json_bytes(
                {
                    "cache_schema": "v8-injectable-test-content-key-1",
                    "role": self.label,
                    "messages": list(messages),
                }
            )
            cache_key_mode = "injectable_test_exact_messages"
        cache_key_sha = _sha256_bytes(cache_key_bytes)
        cache_hit = cache_key_sha in self.cache
        if cache_hit:
            cached_key, response, generation = self.cache[cache_key_sha]
            if cached_key != cache_key_bytes:
                raise V8RunnerError(f"{self.label} content-key SHA256 collision")
        else:
            raw_response = self.backend(deepcopy(list(messages)))
            if isinstance(raw_response, TextGenerationResult):
                response = raw_response.response_text
                generation = {
                    "prompt_tokens": raw_response.prompt_tokens,
                    "generation_tokens": raw_response.generation_tokens,
                    "runtime_telemetry": deepcopy(dict(raw_response.runtime_telemetry)),
                }
            elif isinstance(raw_response, str):
                response = raw_response
                generation = {
                    "prompt_tokens": None,
                    "generation_tokens": None,
                    "runtime_telemetry": {"mode": "injectable_test_backend"},
                }
            else:
                raise V8RunnerError(
                    f"{self.label} backend must return string/TextGenerationResult"
                )
            self.cache[cache_key_sha] = (
                cache_key_bytes,
                response,
                deepcopy(generation),
            )
            self.physical_calls += 1
        for arm in logical_arms:
            self.logical_calls[arm] += 1
        cache_misses = int(not cache_hit)
        cache_hits = len(logical_arms) - cache_misses
        self.logical_cache_hits += cache_hits
        self.logical_cache_misses += cache_misses
        event = {
            "backend": self.label,
            "slot": slot,
            "logical_arms": list(logical_arms),
            "prompt_sha256": prompt_sha,
            "content_cache_key_sha256": cache_key_sha,
            "content_cache_key_mode": cache_key_mode,
            "cache_hit": cache_hit,
            "logical_request_count": len(logical_arms),
            "logical_cache_hits": cache_hits,
            "logical_cache_misses": cache_misses,
            "physical_call_made": not cache_hit,
            "response_sha256": _sha256_text(response),
            **generation,
        }
        self.events.append(event)
        return response, deepcopy(event)


class _CachedRetriever:
    def __init__(self, backend: RetrievalBackend):
        if not hasattr(backend, "batch_search"):
            raise V8RunnerError("retriever backend must provide batch_search")
        self.backend = backend
        self.cache: dict[str, tuple[bytes, list[dict[str, Any]]]] = {}
        self.logical_calls: Counter[str] = Counter()
        self.logical_cache_hits = 0
        self.logical_cache_misses = 0
        self.physical_calls = 0
        # ``physical_calls`` counts individual unique query executions.  The
        # counters below deliberately count the more expensive full-backend
        # batch invocations.  Keeping both prevents a batched implementation
        # from changing the established logical/cache accounting semantics.
        self.backend_batch_invocations = 0
        self.full_index_passes = 0
        self.batch_unique_query_counts: list[int] = []
        self.stage_batch_telemetry: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []

    def invoke(
        self,
        query: str,
        *,
        logical_arms: Sequence[str],
        slot: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return self.invoke_many(
            [
                {
                    "query": query,
                    "logical_arms": tuple(logical_arms),
                    "slot": slot,
                }
            ]
        )[0]

    def invoke_many(
        self,
        requests: Sequence[Mapping[str, Any]],
        *,
        batch_stage: str | None = None,
    ) -> list[tuple[list[dict[str, Any]], dict[str, Any]]]:
        """Resolve a stable list of logical requests with one backend batch.

        The first occurrence of each previously unseen content key is a cache
        miss; later duplicates in the same call are cache hits, exactly as if
        the requests had been issued sequentially in the supplied order.  Only
        the stable, de-duplicated miss queries are sent to ``batch_search``.
        Thus logical requests, cache hits/misses, and physical *query*
        executions retain their old meaning while a whole cohort stage incurs
        at most one full index pass.
        """

        if isinstance(requests, (str, bytes)) or not isinstance(requests, Sequence):
            raise V8RunnerError("retrieval requests must be a sequence")
        if batch_stage is not None and (
            not isinstance(batch_stage, str) or not batch_stage
        ):
            raise V8RunnerError("retrieval batch_stage must be a non-empty string")
        if not requests:
            if batch_stage is not None:
                self.stage_batch_telemetry.append(
                    {
                        "stage": batch_stage,
                        "logical_request_groups": 0,
                        "unique_miss_query_count": 0,
                        "backend_invoked": False,
                    }
                )
            return []

        prepared: list[dict[str, Any]] = []
        pending_keys: dict[str, bytes] = {}
        pending_queries: list[str] = []
        pending_key_order: list[str] = []
        for raw_request in requests:
            if not isinstance(raw_request, Mapping) or set(raw_request) != {
                "query",
                "logical_arms",
                "slot",
            }:
                raise V8RunnerError(
                    "retrieval request must contain exactly query/logical_arms/slot"
                )
            query = raw_request["query"]
            logical_arms = raw_request["logical_arms"]
            slot = raw_request["slot"]
            if not isinstance(query, str) or not query:
                raise V8RunnerError("retrieval query must be a non-empty string")
            if (
                isinstance(logical_arms, (str, bytes))
                or not isinstance(logical_arms, Sequence)
                or not logical_arms
                or any(arm not in ALL_ARMS for arm in logical_arms)
            ):
                raise V8RunnerError(f"invalid logical retrieval arms: {logical_arms}")
            if not isinstance(slot, str) or not slot:
                raise V8RunnerError("retrieval slot must be a non-empty string")

            cache_payload_builder = getattr(self.backend, "cache_key_payload", None)
            if callable(cache_payload_builder):
                payload = cache_payload_builder(query)
                if not isinstance(payload, Mapping):
                    raise V8RunnerError(
                        "asset-bound retrieval cache payload is not an object"
                    )
                cache_key_bytes = _canonical_json_bytes(dict(payload))
                cache_key_mode = "asset_bound_query_and_retrieval_stack"
            else:
                cache_key_bytes = _canonical_json_bytes(
                    {
                        "cache_schema": "v8-injectable-test-retrieval-key-1",
                        "query_utf8": query,
                    }
                )
                cache_key_mode = "injectable_test_exact_query"
            cache_key_sha = _sha256_bytes(cache_key_bytes)
            if cache_key_sha in self.cache:
                cached_key, _ = self.cache[cache_key_sha]
                if cached_key != cache_key_bytes:
                    raise V8RunnerError("retrieval content-key SHA256 collision")
                first_miss = False
            elif cache_key_sha in pending_keys:
                if pending_keys[cache_key_sha] != cache_key_bytes:
                    raise V8RunnerError("retrieval content-key SHA256 collision")
                first_miss = False
            else:
                pending_keys[cache_key_sha] = cache_key_bytes
                pending_key_order.append(cache_key_sha)
                pending_queries.append(query)
                first_miss = True
            prepared.append(
                {
                    "query": query,
                    "logical_arms": tuple(logical_arms),
                    "slot": slot,
                    "query_sha256": _sha256_text(query),
                    "cache_key_bytes": cache_key_bytes,
                    "cache_key_sha": cache_key_sha,
                    "cache_key_mode": cache_key_mode,
                    "first_miss": first_miss,
                }
            )

        if pending_queries:
            raw_batches = self.backend.batch_search(pending_queries)
            if (
                isinstance(raw_batches, (str, bytes))
                or not isinstance(raw_batches, Sequence)
                or len(raw_batches) != len(pending_queries)
            ):
                raise V8RunnerError(
                    "retriever returned the wrong number of batched result lists"
                )
            materialized: list[list[dict[str, Any]]] = []
            for index, result in enumerate(raw_batches):
                if isinstance(result, (str, bytes)) or not isinstance(result, Sequence):
                    raise V8RunnerError("retriever result must be a passage sequence")
                passages = [
                    deepcopy(dict(value)) for value in result if isinstance(value, Mapping)
                ]
                if len(passages) != len(result):
                    raise V8RunnerError("retriever result contains a non-object passage")
                # Enforce exact top-10 and stable document identity before caching.
                project_top10_passages_for_prompt(
                    passages,
                    role=f"batched_miss_{index}",
                )
                materialized.append(passages)
            # Only commit after the entire backend response validates so a
            # failed batch cannot leave a partially populated in-memory cache.
            for key_sha, passages in zip(pending_key_order, materialized):
                self.cache[key_sha] = (
                    pending_keys[key_sha],
                    deepcopy(passages),
                )
            self.physical_calls += len(pending_queries)
            self.backend_batch_invocations += 1
            self.full_index_passes += 1
            self.batch_unique_query_counts.append(len(pending_queries))

        if batch_stage is not None:
            self.stage_batch_telemetry.append(
                {
                    "stage": batch_stage,
                    "logical_request_groups": len(requests),
                    "unique_miss_query_count": len(pending_queries),
                    "backend_invoked": bool(pending_queries),
                }
            )

        outputs: list[tuple[list[dict[str, Any]], dict[str, Any]]] = []
        for request in prepared:
            cached_key, cached_passages = self.cache[request["cache_key_sha"]]
            if cached_key != request["cache_key_bytes"]:
                raise V8RunnerError("retrieval content-key SHA256 collision")
            passages = deepcopy(cached_passages)
            logical_arms = request["logical_arms"]
            cache_misses = int(request["first_miss"])
            cache_hits = len(logical_arms) - cache_misses
            for arm in logical_arms:
                self.logical_calls[arm] += 1
            self.logical_cache_hits += cache_hits
            self.logical_cache_misses += cache_misses
            event = {
                "slot": request["slot"],
                "logical_arms": list(logical_arms),
                "query": request["query"],
                "query_sha256": request["query_sha256"],
                "content_cache_key_sha256": request["cache_key_sha"],
                "content_cache_key_mode": request["cache_key_mode"],
                "cache_hit": not request["first_miss"],
                "logical_request_count": len(logical_arms),
                "logical_cache_hits": cache_hits,
                "logical_cache_misses": cache_misses,
                "physical_call_made": bool(request["first_miss"]),
                "result_count": len(passages),
            }
            self.events.append(event)
            outputs.append((passages, deepcopy(event)))
        return outputs


def _validate_identity_row(row: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(row, Mapping) or tuple(row) != IDENTITY_FIELDS or set(row) != set(IDENTITY_FIELDS):
        raise V8RunnerError("development row must contain exactly dataset/qid/question")
    result: dict[str, str] = {}
    for field in IDENTITY_FIELDS:
        value = row[field]
        if not isinstance(value, str) or not value or value != value.strip():
            raise V8RunnerError(f"development row {field} must be a non-empty unpadded string")
        if "\n" in value or "\r" in value:
            raise V8RunnerError(f"development row {field} must be one line")
        result[field] = value
    return result


class _DuplicateJSONKey(ValueError):
    pass


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKey(key)
        result[key] = value
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise V8RunnerError(f"cannot hash locked smoke artifact: {path}") from exc
    return digest.hexdigest()


def _read_locked_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
        )
    except (_DuplicateJSONKey, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V8RunnerError(f"cannot read valid locked {label}: {path}") from exc
    if not isinstance(value, dict):
        raise V8RunnerError(f"locked {label} must be an object")
    return value


def load_locked_consumed_smoke4x3() -> dict[str, Any]:
    """Load only the fixed consumed smoke cohort; no path/role override exists."""

    manifest_path = (PROJECT_ROOT / SMOKE_MANIFEST_PATH).resolve()
    report_path = (PROJECT_ROOT / SMOKE_REPORT_PATH).resolve()
    cohort_path = (PROJECT_ROOT / SMOKE_COHORT_PATH).resolve()
    if _sha256_file(manifest_path) != SMOKE_MANIFEST_SHA256:
        raise V8RunnerError("consumed smoke manifest SHA mismatch")
    manifest = _read_locked_json(manifest_path, label="smoke manifest")
    if (
        manifest.get("schema_version")
        != "dynamic-decomposition-v8-consumed-smoke-manifest-1"
        or manifest.get("experiment_id")
        != "SUBQUESTION-DECOMPOSITION-V8-CONSUMED-SMOKE4X3-SEED20260904-V1"
        or manifest.get("status")
        != "COMPLETE_FROZEN_CONSUMED_IDENTITY_ONLY_SMOKE4X3"
        or manifest.get("gold_access") is not False
        or manifest.get("prospective_opened_or_hashed") is not False
    ):
        raise V8RunnerError("consumed smoke manifest contract mismatch")
    manifest_outputs = manifest.get("outputs")
    if (
        not isinstance(manifest_outputs, list)
        or len(manifest_outputs) != 2
        or any(not isinstance(item, Mapping) for item in manifest_outputs)
    ):
        raise V8RunnerError("consumed smoke manifest outputs are malformed")
    output_locks = {
        str(item.get("path")): str(item.get("sha256"))
        for item in manifest_outputs
    }
    if len(output_locks) != 2 or output_locks != {
        "smoke.identity_only.jsonl": SMOKE_COHORT_SHA256,
        "report.json": SMOKE_REPORT_SHA256,
    }:
        raise V8RunnerError("consumed smoke manifest output locks mismatch")
    if _sha256_file(report_path) != SMOKE_REPORT_SHA256:
        raise V8RunnerError("consumed smoke report SHA mismatch")
    report = _read_locked_json(report_path, label="smoke report")
    if (
        report.get("schema_version")
        != "dynamic-decomposition-v8-consumed-smoke-cohort-1"
        or report.get("status")
        != "COMPLETE_FROZEN_CONSUMED_IDENTITY_ONLY_SMOKE4X3"
        or report.get("scope")
        != "CONSUMED_ENGINEERING_SMOKE_ONLY_NOT_FRESH_DEVELOPMENT"
        or report.get("row_count") != 12
        or report.get("per_dataset_counts")
        != {"2wikimultihopqa": 4, "hotpotqa": 4, "musique": 4}
        or report.get("output_fields_exact") != list(IDENTITY_FIELDS)
        or report.get("source_fields_accessed") != list(IDENTITY_FIELDS)
        or report.get("fresh_development_or_prospective_overlap") != 0
        or report.get("gold_access") is not False
        or report.get("prospective_opened_or_hashed") is not False
    ):
        raise V8RunnerError("consumed smoke report contract mismatch")
    if _sha256_file(cohort_path) != SMOKE_COHORT_SHA256:
        raise V8RunnerError("consumed smoke cohort SHA mismatch")
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    counts: Counter[str] = Counter()
    try:
        handle = cohort_path.open("r", encoding="utf-8")
    except OSError as exc:
        raise V8RunnerError("cannot open locked consumed smoke cohort") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.endswith("\n") or not line.strip():
                raise V8RunnerError(f"smoke cohort line {line_number} framing mismatch")
            try:
                raw = json.loads(line, object_pairs_hook=_unique_json_object)
            except (_DuplicateJSONKey, json.JSONDecodeError) as exc:
                raise V8RunnerError(f"invalid smoke cohort JSON at line {line_number}") from exc
            identity = _validate_identity_row(raw)
            if identity["dataset"] not in {"hotpotqa", "2wikimultihopqa", "musique"}:
                raise V8RunnerError("smoke cohort contains an unsupported dataset")
            key = f"{identity['dataset']}::{identity['qid']}"
            if key in seen:
                raise V8RunnerError(f"duplicate smoke identity: {key}")
            seen.add(key)
            counts[identity["dataset"]] += 1
            rows.append(identity)
    if len(rows) != 12 or dict(counts) != {
        "hotpotqa": 4,
        "2wikimultihopqa": 4,
        "musique": 4,
    }:
        raise V8RunnerError("consumed smoke cohort must contain exactly 4x3 rows")
    return {
        "scope": "LOCKED_CONSUMED_ENGINEERING_SMOKE4X3_ONLY",
        "gold_access": False,
        "prospective_unlocked": False,
        "manifest_path": SMOKE_MANIFEST_PATH.as_posix(),
        "manifest_sha256": SMOKE_MANIFEST_SHA256,
        "report_path": SMOKE_REPORT_PATH.as_posix(),
        "report_sha256": SMOKE_REPORT_SHA256,
        "cohort_path": SMOKE_COHORT_PATH.as_posix(),
        "cohort_sha256": SMOKE_COHORT_SHA256,
        "row_count": 12,
        "per_dataset_counts": dict(counts),
        "rows": rows,
    }


def _counter_delta(
    after: Counter[str], before: Counter[str], *, arms: Sequence[str] = ARMS
) -> dict[str, int]:
    return {arm: int(after[arm] - before[arm]) for arm in arms}


def _run_identity_row(
    row: Mapping[str, Any],
    *,
    controller: _CachedTextInvoker,
    subanswer_reader: _CachedTextInvoker,
    retriever: _CachedRetriever,
) -> dict[str, Any]:
    """Run one identity-only row; intended for the locked public driver/tests."""

    identity = _validate_identity_row(row)
    question = identity["question"]
    logical_controller_before = controller.logical_calls.copy()
    logical_reader_before = subanswer_reader.logical_calls.copy()
    logical_retrieval_before = retriever.logical_calls.copy()
    physical_before = {
        "controller": controller.physical_calls,
        "subanswer_reader": subanswer_reader.physical_calls,
        "retrieval": retriever.physical_calls,
    }
    cache_before = {
        "controller": (controller.logical_cache_hits, controller.logical_cache_misses),
        "subanswer_reader": (
            subanswer_reader.logical_cache_hits,
            subanswer_reader.logical_cache_misses,
        ),
        "retrieval": (retriever.logical_cache_hits, retriever.logical_cache_misses),
    }

    root_passages, root_event = retriever.invoke(
        question,
        logical_arms=ARMS,
        slot="root_shared",
    )
    q1_messages = build_q1_controller_messages(
        original_question=question,
        root_passages=root_passages,
    )
    q1_response, q1_controller_event = controller.invoke(
        q1_messages,
        logical_arms=ARMS,
        slot="q1_shared",
    )
    q1_action = _build_q1_action(q1_response, original_question=question)
    q1_query = str(q1_action["selected_query"])
    q1_passages, q1_retrieval_event = retriever.invoke(
        q1_query,
        logical_arms=ARMS,
        slot="q1_shared",
    )
    reader_messages = build_subanswer_reader_messages(
        original_question=question,
        q1_query=q1_query,
        q1_passages=q1_passages,
    )
    subanswer_response, reader_event = subanswer_reader.invoke(
        reader_messages,
        logical_arms=ARMS,
        slot="q1_subanswer_shared",
    )
    binding = parse_and_bind_subanswer(
        subanswer_response,
        q1_query=q1_query,
        q1_passages=q1_passages,
    )

    blind_state = build_static_q2_state(
        original_question=question,
        q1_query=q1_query,
    )
    blind_messages = build_q2_controller_messages(blind_state)
    b_response, b_controller_event = controller.invoke(
        blind_messages,
        logical_arms=(ARM_B,),
        slot="q2_B_observation_blind",
    )
    b_action = build_static_q2_action(
        b_response,
        original_question=question,
        q1_query=q1_query,
    )

    dynamic_eligible = binding.get("verified") is True
    if dynamic_eligible:
        c_state = build_dynamic_q2_state(
            original_question=question,
            q1_query=q1_query,
            binding=binding,
        )
        c_messages = build_q2_controller_messages(c_state)
    else:
        c_state = blind_state
        c_messages = blind_messages
    c_response, c_controller_event = controller.invoke(
        c_messages,
        logical_arms=(ARM_C,),
        slot=("q2_C_dynamic" if dynamic_eligible else "q2_C_observation_blind"),
    )
    if dynamic_eligible:
        c_action = build_dynamic_q2_action(
            c_response,
            original_question=question,
            q1_query=q1_query,
            binding=binding,
        )
    else:
        c_action = build_static_q2_action(
            c_response,
            original_question=question,
            q1_query=q1_query,
        )
        if _canonical_json_bytes(c_messages) != _canonical_json_bytes(blind_messages):
            raise V8RunnerError("ineligible C q2 prompt differs from B byte-for-byte")
        if c_response != b_response or c_action["selected_query"] != b_action["selected_query"]:
            raise V8RunnerError("ineligible C did not reproduce B's observation-blind action")

    q2_b_passages, q2_b_event = retriever.invoke(
        str(b_action["selected_query"]),
        logical_arms=(ARM_B,),
        slot="q2_B",
    )
    q2_c_passages, q2_c_event = retriever.invoke(
        str(c_action["selected_query"]),
        logical_arms=(ARM_C,),
        slot="q2_C",
    )
    b_passages, b_merge = merge_fixed_budget_passages(
        root_passages,
        q1_passages,
        q2_b_passages,
        root_query=question,
        q1_query=q1_query,
        q2_query=str(b_action["selected_query"]),
        q1_binding=binding,
    )
    c_passages, c_merge = merge_fixed_budget_passages(
        root_passages,
        q1_passages,
        q2_c_passages,
        root_query=question,
        q1_query=q1_query,
        q2_query=str(c_action["selected_query"]),
        q1_binding=binding,
    )
    if not dynamic_eligible and _canonical_json_bytes(b_passages) != _canonical_json_bytes(c_passages):
        raise V8RunnerError("ineligible C final passages differ from B byte-for-byte")
    if not dynamic_eligible and _canonical_json_bytes(
        _safe_top10_snapshot(q2_b_passages, role="q2_identity")
    ) != _canonical_json_bytes(
        _safe_top10_snapshot(q2_c_passages, role="q2_identity")
    ):
        raise V8RunnerError("ineligible C q2 top-10 differs from B byte-for-byte")

    controller_delta = _counter_delta(controller.logical_calls, logical_controller_before)
    reader_delta = _counter_delta(subanswer_reader.logical_calls, logical_reader_before)
    retrieval_delta = _counter_delta(retriever.logical_calls, logical_retrieval_before)
    expected = {
        ARM_B: {"controller_calls": 2, "subanswer_reader_calls": 1, "retrieval_calls": 3},
        ARM_C: {"controller_calls": 2, "subanswer_reader_calls": 1, "retrieval_calls": 3},
    }
    observed = {
        arm: {
            "controller_calls": controller_delta[arm],
            "subanswer_reader_calls": reader_delta[arm],
            "retrieval_calls": retrieval_delta[arm],
        }
        for arm in ARMS
    }
    if observed != expected:
        raise V8RunnerError(f"logical budget mismatch: observed={observed}, expected={expected}")
    physical_delta = {
        "controller_calls": controller.physical_calls - physical_before["controller"],
        "subanswer_reader_calls": (
            subanswer_reader.physical_calls - physical_before["subanswer_reader"]
        ),
        "retrieval_calls": retriever.physical_calls - physical_before["retrieval"],
    }
    cache_delta = {
        "controller": {
            "logical_requests": sum(controller_delta.values()),
            "cache_hits": controller.logical_cache_hits - cache_before["controller"][0],
            "cache_misses": (
                controller.logical_cache_misses - cache_before["controller"][1]
            ),
        },
        "subanswer_reader": {
            "logical_requests": sum(reader_delta.values()),
            "cache_hits": (
                subanswer_reader.logical_cache_hits
                - cache_before["subanswer_reader"][0]
            ),
            "cache_misses": (
                subanswer_reader.logical_cache_misses
                - cache_before["subanswer_reader"][1]
            ),
        },
        "retrieval": {
            "logical_requests": sum(retrieval_delta.values()),
            "cache_hits": retriever.logical_cache_hits - cache_before["retrieval"][0],
            "cache_misses": (
                retriever.logical_cache_misses - cache_before["retrieval"][1]
            ),
        },
    }
    if any(
        values["logical_requests"] != values["cache_hits"] + values["cache_misses"]
        for values in cache_delta.values()
    ):
        raise V8RunnerError(f"cache accounting conservation failed: {cache_delta}")
    physical_by_backend = {
        "controller": physical_delta["controller_calls"],
        "subanswer_reader": physical_delta["subanswer_reader_calls"],
        "retrieval": physical_delta["retrieval_calls"],
    }
    if any(
        cache_delta[name]["cache_misses"] != physical_by_backend[name]
        for name in cache_delta
    ):
        raise V8RunnerError(
            "physical executions do not equal joint content-cache misses"
        )

    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "runner_version": RUNNER_VERSION,
        "gold_access": False,
        "identity": identity,
        "shared": {
            "q1_action": q1_action,
            "subanswer_binding": binding,
            "root_retrieval": root_event,
            "root_top10": _safe_top10_snapshot(root_passages, role="root_output"),
            "q1_controller": q1_controller_event,
            "q1_retrieval": q1_retrieval_event,
            "q1_top10": _safe_top10_snapshot(q1_passages, role="q1_output"),
            "subanswer_reader": reader_event,
        },
        "arms": {
            ARM_B: {
                "q2_state_mode": blind_state["mode"],
                "q2_prompt_sha256": _messages_sha256(blind_messages),
                "q2_controller": b_controller_event,
                "q2_action": b_action,
                "q2_retrieval": q2_b_event,
                "q2_top10": _safe_top10_snapshot(q2_b_passages, role="q2_B_output"),
                "final_passages": b_passages,
                "merge": b_merge,
            },
            ARM_C: {
                "dynamic_eligible": dynamic_eligible,
                "q2_state_mode": c_state["mode"],
                "q2_prompt_sha256": _messages_sha256(c_messages),
                "q2_controller": c_controller_event,
                "q2_action": c_action,
                "q2_retrieval": q2_c_event,
                "q2_top10": _safe_top10_snapshot(q2_c_passages, role="q2_C_output"),
                "final_passages": c_passages,
                "merge": c_merge,
            },
        },
        "counterfactual_identity": {
            "ineligible_c": not dynamic_eligible,
            "b_c_q2_prompt_byte_identical": (
                _canonical_json_bytes(blind_messages) == _canonical_json_bytes(c_messages)
            ),
            "b_c_q2_response_byte_identical": b_response == c_response,
            "b_c_q2_query_byte_identical": (
                b_action["selected_query"] == c_action["selected_query"]
            ),
            "b_c_q2_top10_byte_identical": (
                _canonical_json_bytes(
                    _safe_top10_snapshot(q2_b_passages, role="q2_identity")
                )
                == _canonical_json_bytes(
                    _safe_top10_snapshot(q2_c_passages, role="q2_identity")
                )
            ),
            "b_c_final_passages_byte_identical": (
                _canonical_json_bytes(b_passages) == _canonical_json_bytes(c_passages)
            ),
        },
        "budget": {
            "logical_by_arm": observed,
            "physical_for_row": physical_delta,
            "joint_cache_accounting": cache_delta,
            "controller_content_cache_enabled": True,
        },
    }


def _run_complete_identity_row(
    row: Mapping[str, Any],
    *,
    controller: _CachedTextInvoker,
    subanswer_reader: _CachedTextInvoker,
    final_reader: _CachedTextInvoker,
    retriever: _CachedRetriever,
) -> dict[str, Any]:
    """Execute Gold-free A/B/C inference for one locked identity row."""

    identity = _validate_identity_row(row)
    question = identity["question"]
    logical_before = {
        "controller": controller.logical_calls.copy(),
        "subanswer_reader": subanswer_reader.logical_calls.copy(),
        "final_reader": final_reader.logical_calls.copy(),
        "retrieval": retriever.logical_calls.copy(),
    }
    physical_before = {
        "controller": controller.physical_calls,
        "subanswer_reader": subanswer_reader.physical_calls,
        "final_reader": final_reader.physical_calls,
        "retrieval": retriever.physical_calls,
    }
    cache_before = {
        "controller": (controller.logical_cache_hits, controller.logical_cache_misses),
        "subanswer_reader": (
            subanswer_reader.logical_cache_hits,
            subanswer_reader.logical_cache_misses,
        ),
        "final_reader": (
            final_reader.logical_cache_hits,
            final_reader.logical_cache_misses,
        ),
        "retrieval": (retriever.logical_cache_hits, retriever.logical_cache_misses),
    }

    # Arm A owns the first physical root request.  B/C make their own logical
    # root requests inside the shared core and reuse the exact content key.
    arm_a_raw, arm_a_retrieval = retriever.invoke(
        question,
        logical_arms=(ARM_A,),
        slot="root_A",
    )
    arm_a_passages = project_top10_passages_for_prompt(arm_a_raw, role="arm_A")
    bc = _run_identity_row(
        identity,
        controller=controller,
        subanswer_reader=subanswer_reader,
        retriever=retriever,
    )
    if (
        arm_a_retrieval["content_cache_key_sha256"]
        != bc["shared"]["root_retrieval"]["content_cache_key_sha256"]
        or bc["shared"]["root_retrieval"]["cache_hit"] is not True
    ):
        raise V8RunnerError("A/B/C root retrieval was not byte-identical and shared")

    passages_by_arm: dict[str, list[dict[str, Any]]] = {
        ARM_A: arm_a_passages,
        ARM_B: deepcopy(bc["arms"][ARM_B]["final_passages"]),
        ARM_C: deepcopy(bc["arms"][ARM_C]["final_passages"]),
    }
    final_records: dict[str, dict[str, Any]] = {}
    final_messages: dict[str, list[dict[str, str]]] = {}
    final_responses: dict[str, str] = {}
    for arm in ALL_ARMS:
        messages = build_final_reader_messages(
            original_question=question,
            final_passages=passages_by_arm[arm],
        )
        response, event = final_reader.invoke(
            messages,
            logical_arms=(arm,),
            slot=f"final_{arm}",
        )
        final_messages[arm] = messages
        final_responses[arm] = response
        final_records[arm] = {
            "prompt_sha256": _messages_sha256(messages),
            "generation": response,
            "generation_sha256": _sha256_text(response),
            "reader_event": event,
        }

    ineligible_c = bc["counterfactual_identity"]["ineligible_c"] is True
    if ineligible_c:
        if not (
            _canonical_json_bytes(final_messages[ARM_B])
            == _canonical_json_bytes(final_messages[ARM_C])
            and final_responses[ARM_B] == final_responses[ARM_C]
            and final_records[ARM_C]["reader_event"]["cache_hit"] is True
        ):
            raise V8RunnerError("ineligible C did not remain identical to B through prediction")

    deltas = {
        name: _counter_delta(
            current.logical_calls,
            logical_before[name],
            arms=ALL_ARMS,
        )
        for name, current in (
            ("controller", controller),
            ("subanswer_reader", subanswer_reader),
            ("final_reader", final_reader),
            ("retrieval", retriever),
        )
    }
    observed = {
        arm: {
            "retrieval": deltas["retrieval"][arm],
            "controller": deltas["controller"][arm],
            "subanswer_reader": deltas["subanswer_reader"][arm],
            "final_reader": deltas["final_reader"][arm],
        }
        for arm in ALL_ARMS
    }
    expected = runtime_contract()["logical_budget_by_arm"]
    if observed != expected:
        raise V8RunnerError(
            f"complete logical budget mismatch: observed={observed}, expected={expected}"
        )
    invokers = {
        "controller": controller,
        "subanswer_reader": subanswer_reader,
        "final_reader": final_reader,
        "retrieval": retriever,
    }
    physical = {
        name: invoker.physical_calls - physical_before[name]
        for name, invoker in invokers.items()
    }
    cache_accounting: dict[str, dict[str, int]] = {}
    for name, invoker in invokers.items():
        hits = invoker.logical_cache_hits - cache_before[name][0]
        misses = invoker.logical_cache_misses - cache_before[name][1]
        logical = sum(deltas[name].values())
        if logical != hits + misses or physical[name] != misses:
            raise V8RunnerError(f"complete cache accounting mismatch for {name}")
        cache_accounting[name] = {
            "logical_requests": logical,
            "cache_hits": hits,
            "cache_misses": misses,
            "physical_executions": physical[name],
        }

    arms = {
        ARM_A: {
            "retrieval": arm_a_retrieval,
            "final_passages": arm_a_passages,
            "final": final_records[ARM_A],
        },
        ARM_B: {
            **deepcopy(bc["arms"][ARM_B]),
            "final": final_records[ARM_B],
        },
        ARM_C: {
            **deepcopy(bc["arms"][ARM_C]),
            "final": final_records[ARM_C],
        },
    }
    return {
        "schema_version": COMPLETE_OUTPUT_SCHEMA_VERSION,
        "runner_version": RUNNER_VERSION,
        "production_runtime_version": PRODUCTION_RUNTIME_VERSION,
        "gold_access": False,
        "identity": identity,
        "shared": deepcopy(bc["shared"]),
        "arms": arms,
        "counterfactual_identity": {
            **deepcopy(bc["counterfactual_identity"]),
            "b_c_final_prompt_byte_identical": (
                _canonical_json_bytes(final_messages[ARM_B])
                == _canonical_json_bytes(final_messages[ARM_C])
            ),
            "b_c_prediction_byte_identical": (
                final_responses[ARM_B] == final_responses[ARM_C]
            ),
        },
        "budget": {
            "logical_by_arm": observed,
            "joint_physical_executions": physical,
            "joint_cache_accounting": cache_accounting,
        },
    }


def _event_accounting(
    events: Sequence[Mapping[str, Any]],
) -> tuple[Counter[str], dict[str, int]]:
    """Reconstruct one row's logical/cache/physical accounting from events."""

    logical: Counter[str] = Counter()
    cache_hits = 0
    cache_misses = 0
    physical = 0
    for event in events:
        arms = event.get("logical_arms")
        if (
            isinstance(arms, (str, bytes))
            or not isinstance(arms, Sequence)
            or not arms
            or any(arm not in ALL_ARMS for arm in arms)
        ):
            raise V8RunnerError("staged event contains invalid logical arms")
        for arm in arms:
            logical[arm] += 1
        event_hits = event.get("logical_cache_hits")
        event_misses = event.get("logical_cache_misses")
        logical_count = event.get("logical_request_count")
        if (
            type(event_hits) is not int
            or type(event_misses) is not int
            or type(logical_count) is not int
            or event_hits < 0
            or event_misses < 0
            or logical_count != event_hits + event_misses
            or logical_count != len(arms)
        ):
            raise V8RunnerError("staged event cache accounting is malformed")
        call_made = event.get("physical_call_made")
        if type(call_made) is not bool or int(call_made) != event_misses:
            raise V8RunnerError("staged event physical/cache-miss accounting differs")
        cache_hits += event_hits
        cache_misses += event_misses
        physical += int(call_made)
    return logical, {
        "logical_requests": sum(logical.values()),
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "physical_executions": physical,
    }


def _assemble_staged_complete_row(state: Mapping[str, Any]) -> dict[str, Any]:
    """Assemble one already-executed staged row without making backend calls."""

    identity = state["identity"]
    arm_a_retrieval = state["arm_a_retrieval"]
    root_event = state["root_event"]
    q1_controller_event = state["q1_controller_event"]
    q1_retrieval_event = state["q1_retrieval_event"]
    reader_event = state["reader_event"]
    b_controller_event = state["b_controller_event"]
    c_controller_event = state["c_controller_event"]
    q2_b_event = state["q2_b_event"]
    q2_c_event = state["q2_c_event"]
    final_records = state["final_records"]

    event_groups = {
        "controller": [
            q1_controller_event,
            b_controller_event,
            c_controller_event,
        ],
        "subanswer_reader": [reader_event],
        "final_reader": [
            final_records[ARM_A]["reader_event"],
            final_records[ARM_B]["reader_event"],
            final_records[ARM_C]["reader_event"],
        ],
        "retrieval": [
            arm_a_retrieval,
            root_event,
            q1_retrieval_event,
            q2_b_event,
            q2_c_event,
        ],
    }
    logical_by_backend: dict[str, Counter[str]] = {}
    cache_accounting: dict[str, dict[str, int]] = {}
    for backend_name, events in event_groups.items():
        logical_by_backend[backend_name], cache_accounting[backend_name] = (
            _event_accounting(events)
        )
    observed = {
        arm: {
            "retrieval": logical_by_backend["retrieval"][arm],
            "controller": logical_by_backend["controller"][arm],
            "subanswer_reader": logical_by_backend["subanswer_reader"][arm],
            "final_reader": logical_by_backend["final_reader"][arm],
        }
        for arm in ALL_ARMS
    }
    expected = runtime_contract()["logical_budget_by_arm"]
    if observed != expected:
        raise V8RunnerError(
            f"staged complete logical budget mismatch: observed={observed}, "
            f"expected={expected}"
        )
    physical = {
        name: values["physical_executions"]
        for name, values in cache_accounting.items()
    }

    b_passages = state["b_passages"]
    c_passages = state["c_passages"]
    final_messages = state["final_messages"]
    final_responses = state["final_responses"]
    arms = {
        ARM_A: {
            "retrieval": arm_a_retrieval,
            "final_passages": state["arm_a_passages"],
            "final": final_records[ARM_A],
        },
        ARM_B: {
            "q2_state_mode": state["blind_state"]["mode"],
            "q2_prompt_sha256": _messages_sha256(state["blind_messages"]),
            "q2_controller": b_controller_event,
            "q2_action": state["b_action"],
            "q2_retrieval": q2_b_event,
            "q2_top10": _safe_top10_snapshot(
                state["q2_b_passages"], role="q2_B_output"
            ),
            "final_passages": b_passages,
            "merge": state["b_merge"],
            "final": final_records[ARM_B],
        },
        ARM_C: {
            "dynamic_eligible": state["dynamic_eligible"],
            "q2_state_mode": state["c_state"]["mode"],
            "q2_prompt_sha256": _messages_sha256(state["c_messages"]),
            "q2_controller": c_controller_event,
            "q2_action": state["c_action"],
            "q2_retrieval": q2_c_event,
            "q2_top10": _safe_top10_snapshot(
                state["q2_c_passages"], role="q2_C_output"
            ),
            "final_passages": c_passages,
            "merge": state["c_merge"],
            "final": final_records[ARM_C],
        },
    }
    return {
        "schema_version": COMPLETE_OUTPUT_SCHEMA_VERSION,
        "runner_version": RUNNER_VERSION,
        "production_runtime_version": PRODUCTION_RUNTIME_VERSION,
        "gold_access": False,
        "identity": identity,
        "shared": {
            "q1_action": state["q1_action"],
            "subanswer_binding": state["binding"],
            "root_retrieval": root_event,
            "root_top10": _safe_top10_snapshot(
                state["root_passages"], role="root_output"
            ),
            "q1_controller": q1_controller_event,
            "q1_retrieval": q1_retrieval_event,
            "q1_top10": _safe_top10_snapshot(
                state["q1_passages"], role="q1_output"
            ),
            "subanswer_reader": reader_event,
        },
        "arms": arms,
        "counterfactual_identity": {
            "ineligible_c": not state["dynamic_eligible"],
            "b_c_q2_prompt_byte_identical": (
                _canonical_json_bytes(state["blind_messages"])
                == _canonical_json_bytes(state["c_messages"])
            ),
            "b_c_q2_response_byte_identical": (
                state["b_response"] == state["c_response"]
            ),
            "b_c_q2_query_byte_identical": (
                state["b_action"]["selected_query"]
                == state["c_action"]["selected_query"]
            ),
            "b_c_q2_top10_byte_identical": (
                _canonical_json_bytes(
                    _safe_top10_snapshot(
                        state["q2_b_passages"], role="q2_identity"
                    )
                )
                == _canonical_json_bytes(
                    _safe_top10_snapshot(
                        state["q2_c_passages"], role="q2_identity"
                    )
                )
            ),
            "b_c_final_passages_byte_identical": (
                _canonical_json_bytes(b_passages) == _canonical_json_bytes(c_passages)
            ),
            "b_c_final_prompt_byte_identical": (
                _canonical_json_bytes(final_messages[ARM_B])
                == _canonical_json_bytes(final_messages[ARM_C])
            ),
            "b_c_prediction_byte_identical": (
                final_responses[ARM_B] == final_responses[ARM_C]
            ),
        },
        "budget": {
            "logical_by_arm": observed,
            "joint_physical_executions": physical,
            "joint_cache_accounting": cache_accounting,
        },
    }


def _run_complete_rows_staged(
    rows: Sequence[Mapping[str, Any]],
    *,
    controller: _CachedTextInvoker,
    subanswer_reader: _CachedTextInvoker,
    final_reader: _CachedTextInvoker,
    retriever: _CachedRetriever,
) -> list[dict[str, Any]]:
    """Execute a cohort in three retrieval batches without repeating generation."""

    states: list[dict[str, Any]] = [
        {"identity": _validate_identity_row(row)} for row in rows
    ]

    # Stage 1: A and B/C logically request each root.  The A request is ordered
    # first so the per-request cache events exactly match the old sequential
    # runner while all cache misses share one backend batch.
    root_requests: list[dict[str, Any]] = []
    for state in states:
        question = state["identity"]["question"]
        root_requests.extend(
            [
                {
                    "query": question,
                    "logical_arms": (ARM_A,),
                    "slot": "root_A",
                },
                {
                    "query": question,
                    "logical_arms": ARMS,
                    "slot": "root_shared",
                },
            ]
        )
    root_results = retriever.invoke_many(root_requests, batch_stage="root_all")
    for index, state in enumerate(states):
        arm_a_raw, arm_a_event = root_results[2 * index]
        root_passages, root_event = root_results[2 * index + 1]
        if (
            arm_a_event["content_cache_key_sha256"]
            != root_event["content_cache_key_sha256"]
            or root_event["cache_hit"] is not True
        ):
            raise V8RunnerError("staged A/B/C root retrieval was not shared")
        state.update(
            {
                "arm_a_passages": project_top10_passages_for_prompt(
                    arm_a_raw, role="arm_A"
                ),
                "arm_a_retrieval": arm_a_event,
                "root_passages": root_passages,
                "root_event": root_event,
            }
        )

    # Generate every q1 exactly once, then retrieve the complete q1 cohort in
    # the second full-index batch.
    for state in states:
        question = state["identity"]["question"]
        q1_messages = build_q1_controller_messages(
            original_question=question,
            root_passages=state["root_passages"],
        )
        q1_response, q1_event = controller.invoke(
            q1_messages,
            logical_arms=ARMS,
            slot="q1_shared",
        )
        q1_action = _build_q1_action(q1_response, original_question=question)
        state.update(
            {
                "q1_messages": q1_messages,
                "q1_response": q1_response,
                "q1_controller_event": q1_event,
                "q1_action": q1_action,
                "q1_query": str(q1_action["selected_query"]),
            }
        )
    q1_results = retriever.invoke_many(
        [
            {
                "query": state["q1_query"],
                "logical_arms": ARMS,
                "slot": "q1_shared",
            }
            for state in states
        ],
        batch_stage="q1_all",
    )
    for state, (q1_passages, q1_event) in zip(states, q1_results):
        state["q1_passages"] = q1_passages
        state["q1_retrieval_event"] = q1_event

    # Bind subanswers and generate B/C q2 once each.  Ineligible C calls reuse
    # B through the content cache and are still separate logical requests.
    for state in states:
        question = state["identity"]["question"]
        reader_messages = build_subanswer_reader_messages(
            original_question=question,
            q1_query=state["q1_query"],
            q1_passages=state["q1_passages"],
        )
        subanswer_response, reader_event = subanswer_reader.invoke(
            reader_messages,
            logical_arms=ARMS,
            slot="q1_subanswer_shared",
        )
        binding = parse_and_bind_subanswer(
            subanswer_response,
            q1_query=state["q1_query"],
            q1_passages=state["q1_passages"],
        )
        blind_state = build_static_q2_state(
            original_question=question,
            q1_query=state["q1_query"],
        )
        blind_messages = build_q2_controller_messages(blind_state)
        b_response, b_event = controller.invoke(
            blind_messages,
            logical_arms=(ARM_B,),
            slot="q2_B_observation_blind",
        )
        b_action = build_static_q2_action(
            b_response,
            original_question=question,
            q1_query=state["q1_query"],
        )
        dynamic_eligible = binding.get("verified") is True
        if dynamic_eligible:
            c_state = build_dynamic_q2_state(
                original_question=question,
                q1_query=state["q1_query"],
                binding=binding,
            )
            c_messages = build_q2_controller_messages(c_state)
        else:
            c_state = blind_state
            c_messages = blind_messages
        c_response, c_event = controller.invoke(
            c_messages,
            logical_arms=(ARM_C,),
            slot=("q2_C_dynamic" if dynamic_eligible else "q2_C_observation_blind"),
        )
        if dynamic_eligible:
            c_action = build_dynamic_q2_action(
                c_response,
                original_question=question,
                q1_query=state["q1_query"],
                binding=binding,
            )
        else:
            c_action = build_static_q2_action(
                c_response,
                original_question=question,
                q1_query=state["q1_query"],
            )
            if (
                _canonical_json_bytes(c_messages)
                != _canonical_json_bytes(blind_messages)
                or c_response != b_response
                or c_action["selected_query"] != b_action["selected_query"]
            ):
                raise V8RunnerError(
                    "staged ineligible C did not reproduce B byte-for-byte"
                )
        state.update(
            {
                "reader_messages": reader_messages,
                "subanswer_response": subanswer_response,
                "reader_event": reader_event,
                "binding": binding,
                "blind_state": blind_state,
                "blind_messages": blind_messages,
                "b_response": b_response,
                "b_controller_event": b_event,
                "b_action": b_action,
                "dynamic_eligible": dynamic_eligible,
                "c_state": c_state,
                "c_messages": c_messages,
                "c_response": c_response,
                "c_controller_event": c_event,
                "c_action": c_action,
            }
        )

    # B/C second-hop retrievals share the third and final full-index batch.
    q2_requests: list[dict[str, Any]] = []
    for state in states:
        q2_requests.extend(
            [
                {
                    "query": str(state["b_action"]["selected_query"]),
                    "logical_arms": (ARM_B,),
                    "slot": "q2_B",
                },
                {
                    "query": str(state["c_action"]["selected_query"]),
                    "logical_arms": (ARM_C,),
                    "slot": "q2_C",
                },
            ]
        )
    q2_results = retriever.invoke_many(q2_requests, batch_stage="q2_BC_all")
    for index, state in enumerate(states):
        q2_b_passages, q2_b_event = q2_results[2 * index]
        q2_c_passages, q2_c_event = q2_results[2 * index + 1]
        b_passages, b_merge = merge_fixed_budget_passages(
            state["root_passages"],
            state["q1_passages"],
            q2_b_passages,
            root_query=state["identity"]["question"],
            q1_query=state["q1_query"],
            q2_query=str(state["b_action"]["selected_query"]),
            q1_binding=state["binding"],
        )
        c_passages, c_merge = merge_fixed_budget_passages(
            state["root_passages"],
            state["q1_passages"],
            q2_c_passages,
            root_query=state["identity"]["question"],
            q1_query=state["q1_query"],
            q2_query=str(state["c_action"]["selected_query"]),
            q1_binding=state["binding"],
        )
        if not state["dynamic_eligible"]:
            if _canonical_json_bytes(b_passages) != _canonical_json_bytes(c_passages):
                raise V8RunnerError(
                    "staged ineligible C final passages differ from B byte-for-byte"
                )
            if _canonical_json_bytes(
                _safe_top10_snapshot(q2_b_passages, role="q2_identity")
            ) != _canonical_json_bytes(
                _safe_top10_snapshot(q2_c_passages, role="q2_identity")
            ):
                raise V8RunnerError(
                    "staged ineligible C q2 top-10 differs from B byte-for-byte"
                )
        state.update(
            {
                "q2_b_passages": q2_b_passages,
                "q2_b_event": q2_b_event,
                "q2_c_passages": q2_c_passages,
                "q2_c_event": q2_c_event,
                "b_passages": b_passages,
                "b_merge": b_merge,
                "c_passages": c_passages,
                "c_merge": c_merge,
            }
        )

    # Final-reader generation remains deterministic and unbatched.  Each arm
    # is generated once; ineligible C is served from B's content cache.
    for state in states:
        passages_by_arm = {
            ARM_A: state["arm_a_passages"],
            ARM_B: state["b_passages"],
            ARM_C: state["c_passages"],
        }
        final_records: dict[str, dict[str, Any]] = {}
        final_messages: dict[str, list[dict[str, str]]] = {}
        final_responses: dict[str, str] = {}
        for arm in ALL_ARMS:
            messages = build_final_reader_messages(
                original_question=state["identity"]["question"],
                final_passages=passages_by_arm[arm],
            )
            response, event = final_reader.invoke(
                messages,
                logical_arms=(arm,),
                slot=f"final_{arm}",
            )
            final_messages[arm] = messages
            final_responses[arm] = response
            final_records[arm] = {
                "prompt_sha256": _messages_sha256(messages),
                "generation": response,
                "generation_sha256": _sha256_text(response),
                "reader_event": event,
            }
        if not state["dynamic_eligible"] and not (
            _canonical_json_bytes(final_messages[ARM_B])
            == _canonical_json_bytes(final_messages[ARM_C])
            and final_responses[ARM_B] == final_responses[ARM_C]
            and final_records[ARM_C]["reader_event"]["cache_hit"] is True
        ):
            raise V8RunnerError(
                "staged ineligible C did not remain identical through prediction"
            )
        state["final_messages"] = final_messages
        state["final_responses"] = final_responses
        state["final_records"] = final_records

    return [_assemble_staged_complete_row(state) for state in states]


def materialize_frozen_development(
    *,
    controller_backend: TextBackend,
    subanswer_reader_backend: TextBackend,
    retriever_backend: RetrievalBackend,
) -> dict[str, Any]:
    """Run the entire locked development90 cohort; prospective is unreachable."""

    cohort = load_frozen_v8_cohort(role=DEVELOPMENT_ROLE)
    if (
        cohort.get("role") != DEVELOPMENT_ROLE
        or cohort.get("gold_access") is not False
        or cohort.get("prospective_unlocked") is not False
    ):
        raise V8RunnerError("development cohort loader returned an unauthorized scope")
    rows = cohort.get("rows")
    if not isinstance(rows, list) or len(rows) != 90:
        raise V8RunnerError("locked development cohort must contain exactly 90 rows")

    controller = _CachedTextInvoker(controller_backend, label="controller")
    reader = _CachedTextInvoker(subanswer_reader_backend, label="subanswer_reader")
    retriever = _CachedRetriever(retriever_backend)
    outputs = [
        _run_identity_row(
            row,
            controller=controller,
            subanswer_reader=reader,
            retriever=retriever,
        )
        for row in rows
    ]
    expected_logical = len(outputs) * 2
    if any(controller.logical_calls[arm] != expected_logical for arm in ARMS):
        raise V8RunnerError("aggregate controller logical-call budget mismatch")
    aggregate_cache = {
        "controller": {
            "logical_requests": sum(controller.logical_calls.values()),
            "cache_hits": controller.logical_cache_hits,
            "cache_misses": controller.logical_cache_misses,
        },
        "subanswer_reader": {
            "logical_requests": sum(reader.logical_calls.values()),
            "cache_hits": reader.logical_cache_hits,
            "cache_misses": reader.logical_cache_misses,
        },
        "retrieval": {
            "logical_requests": sum(retriever.logical_calls.values()),
            "cache_hits": retriever.logical_cache_hits,
            "cache_misses": retriever.logical_cache_misses,
        },
    }
    if any(
        values["logical_requests"] != values["cache_hits"] + values["cache_misses"]
        for values in aggregate_cache.values()
    ):
        raise V8RunnerError("aggregate cache accounting conservation failed")
    aggregate_physical = {
        "controller": controller.physical_calls,
        "subanswer_reader": reader.physical_calls,
        "retrieval": retriever.physical_calls,
    }
    if any(
        aggregate_cache[name]["cache_misses"] != aggregate_physical[name]
        for name in aggregate_cache
    ):
        raise V8RunnerError("aggregate physical/cache-miss accounting mismatch")
    return {
        "runner_version": RUNNER_VERSION,
        "scope": "FROZEN_DEVELOPMENT90_ONLY",
        "gold_access": False,
        "prospective_unlocked": False,
        "cohort_lock": {
            key: deepcopy(cohort[key])
            for key in (
                "loader_version",
                "role",
                "manifest_path",
                "manifest_sha256",
                "cohort_path",
                "cohort_sha256",
                "row_count",
                "per_dataset_counts",
            )
        },
        "row_count": len(outputs),
        "logical_calls": {
            "controller": dict(controller.logical_calls),
            "subanswer_reader": dict(reader.logical_calls),
            "retrieval": dict(retriever.logical_calls),
        },
        "physical_calls": aggregate_physical,
        "joint_cache_accounting": aggregate_cache,
        "rows": outputs,
    }


def _materialize_complete_locked_rows(
    *,
    rows: Sequence[Mapping[str, Any]],
    scope: str,
    experiment_id: str,
    intended_output_dir: str,
    cohort_lock: Mapping[str, Any],
    hf_runtime: Any,
    retriever_runtime: Any,
) -> dict[str, Any]:
    """Internal production wiring; callers must obtain rows from a locked loader."""

    if (
        isinstance(rows, (str, bytes))
        or not isinstance(rows, Sequence)
        or not rows
        or cohort_lock.get("row_count") != len(rows)
    ):
        raise V8RunnerError("locked runtime rows/cardinality do not match cohort lock")
    if not hasattr(hf_runtime, "bind_role"):
        raise V8RunnerError("production HF runtime lacks bind_role")
    if not callable(getattr(retriever_runtime, "cache_key_payload", None)):
        raise V8RunnerError("production retriever lacks an asset-bound cache key")
    role_backends = {
        role: hf_runtime.bind_role(role)
        for role in ("controller", "subanswer_reader", "final_reader")
    }
    if any(
        getattr(backend, "runtime", None) is not hf_runtime
        or not callable(getattr(backend, "cache_key_payload", None))
        for backend in role_backends.values()
    ):
        raise V8RunnerError("all generation roles must share one asset-bound HF runtime")
    controller = _CachedTextInvoker(role_backends["controller"], label="controller")
    reader = _CachedTextInvoker(
        role_backends["subanswer_reader"], label="subanswer_reader"
    )
    final_reader = _CachedTextInvoker(
        role_backends["final_reader"], label="final_reader"
    )
    retriever = _CachedRetriever(retriever_runtime)
    outputs = _run_complete_rows_staged(
        rows,
        controller=controller,
        subanswer_reader=reader,
        final_reader=final_reader,
        retriever=retriever,
    )
    if len(outputs) != len(rows):
        raise V8RunnerError("complete runtime output cardinality mismatch")
    n = len(outputs)
    expected_aggregate = {
        ARM_A: {
            name: count * n
            for name, count in runtime_contract()["logical_budget_by_arm"][ARM_A].items()
        },
        ARM_B: {
            name: count * n
            for name, count in runtime_contract()["logical_budget_by_arm"][ARM_B].items()
        },
        ARM_C: {
            name: count * n
            for name, count in runtime_contract()["logical_budget_by_arm"][ARM_C].items()
        },
    }
    observed_aggregate = {
        arm: {
            "retrieval": retriever.logical_calls[arm],
            "controller": controller.logical_calls[arm],
            "subanswer_reader": reader.logical_calls[arm],
            "final_reader": final_reader.logical_calls[arm],
        }
        for arm in ALL_ARMS
    }
    if observed_aggregate != expected_aggregate:
        raise V8RunnerError("aggregate A/B/C logical budget mismatch")
    invokers = {
        "controller": controller,
        "subanswer_reader": reader,
        "final_reader": final_reader,
        "retrieval": retriever,
    }
    aggregate_cache = {
        name: {
            "logical_requests": sum(invoker.logical_calls.values()),
            "cache_hits": invoker.logical_cache_hits,
            "cache_misses": invoker.logical_cache_misses,
            "physical_executions": invoker.physical_calls,
        }
        for name, invoker in invokers.items()
    }
    if any(
        values["logical_requests"] != values["cache_hits"] + values["cache_misses"]
        or values["physical_executions"] != values["cache_misses"]
        for values in aggregate_cache.values()
    ):
        raise V8RunnerError("aggregate A/B/C cache accounting mismatch")
    return {
        "schema_version": "dynamic-decomposition-v8-complete-gold-free-run-1",
        "runner_version": RUNNER_VERSION,
        "production_runtime_version": PRODUCTION_RUNTIME_VERSION,
        "scope": scope,
        "experiment_id": experiment_id,
        "intended_output_dir": intended_output_dir,
        "gold_access": False,
        "prospective_unlocked": False,
        "cohort_lock": deepcopy(dict(cohort_lock)),
        "runtime_contract": runtime_contract(),
        "row_count": n,
        "logical_calls_by_arm": observed_aggregate,
        "joint_cache_accounting": aggregate_cache,
        "retrieval_batch_telemetry": {
            "backend_batch_invocations": retriever.backend_batch_invocations,
            "full_index_passes": retriever.full_index_passes,
            "unique_query_count_by_batch": list(retriever.batch_unique_query_counts),
            "stage_batches": deepcopy(retriever.stage_batch_telemetry),
        },
        "rows": outputs,
    }


def materialize_locked_consumed_smoke4x3_production(
    *,
    hf_runtime: SharedHuggingFaceRuntime,
    retriever_runtime: CanonicalRetrieverRuntime,
) -> dict[str, Any]:
    """Run only the frozen consumed 4x3 engineering cohort."""

    cohort = load_locked_consumed_smoke4x3()
    rows = cohort.pop("rows")
    return _materialize_complete_locked_rows(
        rows=rows,
        scope="LOCKED_CONSUMED_ENGINEERING_SMOKE4X3_ONLY",
        experiment_id=SMOKE_EXPERIMENT_ID,
        intended_output_dir=SMOKE_OUTPUT_DIR,
        cohort_lock=cohort,
        hf_runtime=hf_runtime,
        retriever_runtime=retriever_runtime,
    )


def materialize_frozen_development_production(
    *,
    hf_runtime: SharedHuggingFaceRuntime,
    retriever_runtime: CanonicalRetrieverRuntime,
) -> dict[str, Any]:
    """Run exactly the complete frozen development90; prospective is unreachable."""

    cohort = load_frozen_v8_cohort(role=DEVELOPMENT_ROLE)
    if (
        cohort.get("role") != DEVELOPMENT_ROLE
        or cohort.get("gold_access") is not False
        or cohort.get("prospective_unlocked") is not False
        or cohort.get("row_count") != 90
    ):
        raise V8RunnerError("development loader returned an unauthorized scope")
    rows = cohort.pop("rows")
    return _materialize_complete_locked_rows(
        rows=rows,
        scope="FROZEN_DEVELOPMENT90_ONLY",
        experiment_id=DEVELOPMENT_EXPERIMENT_ID,
        intended_output_dir=DEVELOPMENT_OUTPUT_DIR,
        cohort_lock=cohort,
        hf_runtime=hf_runtime,
        retriever_runtime=retriever_runtime,
    )


__all__ = [
    "ARM_A",
    "ARM_B",
    "ARM_C",
    "CanonicalRetrieverRuntime",
    "COMPLETE_OUTPUT_SCHEMA_VERSION",
    "OUTPUT_SCHEMA_VERSION",
    "PRODUCTION_RUNTIME_CONTRACT",
    "PRODUCTION_RUNTIME_VERSION",
    "RUNNER_VERSION",
    "SharedHuggingFaceRuntime",
    "TextGenerationResult",
    "V8RunnerError",
    "build_final_reader_messages",
    "build_q1_controller_messages",
    "build_q2_controller_messages",
    "build_subanswer_reader_messages",
    "load_locked_consumed_smoke4x3",
    "materialize_frozen_development",
    "materialize_frozen_development_production",
    "materialize_locked_consumed_smoke4x3_production",
    "runtime_contract",
]
