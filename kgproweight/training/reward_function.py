"""PPO / GRPO reward function — shared.

Given a Student-generated trajectory string and the per-trajectory context
(query, gold answer, KG subgraph, retrieved passages), produce:

  - per-step rewards ``R_total(t) = α_t · R_KG(t) + (1 - α_t) · R_Text(t)``,
  - the outcome bonus ``R_outcome = EM(answer, gold)`` added to the last step,
  - step token-boundary indices so PPO can place per-step rewards on the
    correct token positions.

The function is intentionally self-contained: it instantiates the
``CompositeRewardModel`` on first use and reuses it across calls.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from kgproweight.data.entity_filter import clean_entities
from kgproweight.data.parsers import extract_final_answer, extract_step_token_spans, parse_steps
from kgproweight.data.prompts import build_sft_messages
from kgproweight.eval.pred_processing import extract_kg_proweight_answer
from kgproweight.kg.entity_linker import build_passage_text, build_passage_titles
from kgproweight.reward.alpha_gate import AlphaGate
from kgproweight.reward.answer_format_objective_v2 import (
    compose_answer_format_objective_v2,
    inspect_shortfall_salvage_v2,
)
from kgproweight.reward.composite_reward import CompositeRewardModel
from kgproweight.reward.prm_annotator import PRMAnnotator
from kgproweight.reward.proofkg_process import (
    canonical_answer_normalize,
    canonical_exact_match,
    canonical_token_f1,
    is_automatic_proofkg,
    is_identity_safe_automatic_proofkg,
    required_steps as proofkg_required_steps,
    score_grounded_process,
    token_f1,
)
from kgproweight.reward.proofkg_process_v2 import build_execution_trace, score_proofkg_v2
from kgproweight.reward.proofkg_process_v2_2 import (
    build_execution_trace_v2_2,
    score_proofkg_v2_2,
)
from kgproweight.reward.proofkg_process_v2_3 import (
    build_execution_trace_v2_3,
    score_proofkg_v2_3,
)
from kgproweight.reward.text_reward_model import TextRewardModel
from kgproweight.reward.source_quality_gate_v1 import (
    SourceQualityGateV1,
    compute_gate_features,
)
from kgproweight.reward.source_credit_gate_v1 import SourceCreditGateV1
from kgproweight.reward.source_credit_gate_v2 import SourceCreditGateV2
from kgproweight.reward.source_reward_normalization_v2 import normalize_text_steps_v2
from kgproweight.utils.logging import get_logger

logger = get_logger(__name__)


def _v2_format_contract_violations(response: str, all_steps: Sequence[Any], max_steps: int) -> List[str]:
    violations: List[str] = []
    if len(all_steps) > max_steps:
        violations.append("too_many_steps")
    final_fields = re.findall(
        r"\[\s*Final Answer\s*\]|^[ \t]*(?:\*\*)?Final Answer(?:\*\*)?[ \t]*[:：]",
        response, flags=re.IGNORECASE | re.MULTILINE,
    )
    if len(final_fields) != 1:
        violations.append("final_field_count_not_one")
    for step in all_steps:
        for label in ("Reasoning", "Knowledge Used", "Conclusion"):
            fields = re.findall(
                rf"^[ \t]*{re.escape(label)}[ \t]*:", step.raw_text,
                flags=re.IGNORECASE | re.MULTILINE,
            )
            if len(fields) != 1:
                violations.append(f"step_{step.index}_{label.lower().replace(' ', '_')}_count_not_one")
    return violations


def validate_source_gate_trajectory_v1(
    spec: "RewardSpec", response: str, *, max_steps: int = 5,
    min_valid_steps: int = 3, min_reasoning_chars: int = 20,
) -> Dict[str, Any]:
    """Gold-free shared PPO/bank validity, including the full output before cap."""
    all_steps = parse_steps(response, known_kg=spec.kg_subgraph)
    steps = all_steps[:max_steps]
    features = compute_gate_features(spec, steps, {})
    required = (
        proofkg_required_steps(spec.metadata.get("source_quality_record") or {})
        if features["m_graph"] else min_valid_steps
    )
    violations = _v2_format_contract_violations(response, all_steps, max_steps)
    structural_valid = KGProWeightRewardFunction._is_valid_trajectory(
        steps, response, min_steps=required, min_reasoning_chars=min_reasoning_chars,
    )
    if not structural_valid:
        violations.append("invalid_step_sequence_content_or_minimum")
    return {
        "valid": bool(structural_valid and not violations), "violations": violations,
        "steps": steps, "all_step_count": len(all_steps),
        "required_steps": required, "source_features": features,
        "contract_version": "source-gate-runtime-v2-format-v1",
    }


def source_gate_format_contract_version(format_version: str = "v1") -> str:
    """Name the selected training format without changing historical v1."""
    if format_version not in {"v1", "v2"}:
        raise ValueError("source_gate_format_version must be v1 or v2")
    return f"source-gate-runtime-v2-format-{format_version}"


def validate_source_gate_format_contract(
    gate: SourceQualityGateV1, format_version: str = "v1",
) -> None:
    """Reject calibration statistics fitted under another validity contract."""
    expected = source_gate_format_contract_version(format_version)
    actual = gate.artifact.get(
        "format_contract_version", source_gate_format_contract_version("v1"),
    )
    if actual != expected:
        raise ValueError(
            f"source gate format contract mismatch: artifact={actual!r}, selected={expected!r}"
        )


def validate_source_gate_source_integrity(
    gate: SourceQualityGateV1, format_version: str = "v1",
) -> None:
    """A diagnostic v2 calibration cannot authorize PPO on unresolved sources."""
    source_gate_format_contract_version(format_version)
    if format_version == "v2" and gate.artifact.get("source_integrity_clearance") is not True:
        raise ValueError(
            "source gate v2 requires source_integrity_clearance=true before PPO; "
            f"status={gate.artifact.get('source_integrity_status', 'MISSING')!r}"
        )


def validate_source_gate_credit_config(credit_version: str, reward_version: str,
                                      format_version: str) -> None:
    if credit_version not in {"disabled", "v1", "v2"}:
        raise ValueError("source_gate_credit_version must be disabled, v1 or v2")
    if credit_version in {"v1", "v2"} and (reward_version != "v1" or format_version != "v2"):
        raise ValueError(f"source credit {credit_version} requires source_gated_reward_version=v1 and source_gate_format_version=v2")


def validate_source_gate_runtime_contract(gate: SourceQualityGateV1 | SourceCreditGateV2,
                                         format_version: str,
                                         credit_version: str = "disabled") -> None:
    """Select explicit source-credit semantics without adapting legacy gates."""
    validate_source_gate_credit_config(credit_version, "v1", format_version)
    if credit_version in {"v1", "v2"}:
        expected_class = SourceCreditGateV1 if credit_version == "v1" else SourceCreditGateV2
        if not isinstance(gate, expected_class):
            raise TypeError(f"source credit {credit_version} requires a validated {expected_class.__name__}")
        validate_source_gate_format_contract(gate, format_version)
        bound = gate.artifact.get("source_credit_mask") or {}
        if (gate.artifact.get("source_credit_clearance") is not True
                or bound.get("sha256") != gate.mask.manifest_sha256
                or bound.get("payload_sha256") != gate.mask.payload_sha256):
            raise ValueError("source credit clearance or frozen mask binding mismatch")
    else:
        if isinstance(gate, (SourceCreditGateV1, SourceCreditGateV2)):
            raise TypeError("source_gate_credit_version=disabled rejects source-credit successor gates")
        if not isinstance(gate, SourceQualityGateV1):
            raise TypeError("source-gated v1 requires a validated SourceQualityGateV1")
        validate_source_gate_format_contract(gate, format_version)
        validate_source_gate_source_integrity(gate, format_version)


def load_source_gate_for_runtime(path: str, format_version: str,
                                 credit_version: str = "disabled") -> SourceQualityGateV1 | SourceCreditGateV2:
    validate_source_gate_credit_config(credit_version, "v1", format_version)
    loader = {"disabled": SourceQualityGateV1, "v1": SourceCreditGateV1, "v2": SourceCreditGateV2}[credit_version]
    gate = loader.load(path)
    validate_source_gate_runtime_contract(gate, format_version, credit_version)
    return gate


def validate_source_gate_trajectory_v2(
    spec: "RewardSpec", response: str, *, max_steps: int = 5,
    min_valid_steps: int = 3, min_reasoning_chars: int = 20,
) -> Dict[str, Any]:
    """v1 plus a nonempty literal Final field; the evaluation parser is unchanged.

    The historical parser can turn an empty ``[Final Answer]`` into ``]``.
    Inspect the actual suffix instead. Unicode letters/numbers admit short and
    non-English answers while whitespace and punctuation decorations are empty.
    """
    result = validate_source_gate_trajectory_v1(
        spec, response, max_steps=max_steps, min_valid_steps=min_valid_steps,
        min_reasoning_chars=min_reasoning_chars,
    )
    final_fields = list(re.finditer(
        r"\[\s*Final Answer\s*\]|^[ \t]*(?:\*\*)?Final Answer(?:\*\*)?[ \t]*[:：]",
        response, flags=re.IGNORECASE | re.MULTILINE,
    ))
    if len(final_fields) == 1 and not any(
        char.isalnum() for char in response[final_fields[0].end():]
    ):
        result["violations"].append("final_answer_empty_or_decoration_only")
        result["valid"] = False
    result["contract_version"] = source_gate_format_contract_version("v2")
    return result


def validate_source_gate_trajectory(
    spec: "RewardSpec", response: str, *, max_steps: int = 5,
    min_valid_steps: int = 3, min_reasoning_chars: int = 20,
    format_version: str = "v1",
) -> Dict[str, Any]:
    """Shared versioned dispatch for offline bank scoring and PPO rewards."""
    source_gate_format_contract_version(format_version)
    validator = (
        validate_source_gate_trajectory_v1 if format_version == "v1"
        else validate_source_gate_trajectory_v2
    )
    return validator(
        spec, response, max_steps=max_steps, min_valid_steps=min_valid_steps,
        min_reasoning_chars=min_reasoning_chars,
    )


def source_gate_text_inputs_v1(spec: "RewardSpec", steps: Sequence[Any]) -> Tuple[List[str], List[str]]:
    """Frozen passage-only ReaRAG context, shared with offline calibration.

    The current step is the scoring target; only earlier steps enter its prefix.
    KG triples are deliberately excluded from this source-specific channel.
    """
    messages = build_sft_messages(
        question=spec.query, retrieved_passages=spec.retrieved_passages,
        kg_triples=[], top_k=10,
    )
    base = "\n\n".join(message["content"] for message in messages)
    prompts = []
    for index in range(len(steps)):
        prefix = "\n".join(step.raw_text for step in steps[:index])
        prompts.append(base + ("\n\n" + prefix if prefix else ""))
    return prompts, [step.raw_text for step in steps]


def source_gate_text_budget_v1(text_model: Any, prompts: Sequence[str], texts: Sequence[str]) -> Dict[str, Any]:
    """Preflight every pair so ReaRAG cannot silently left-truncate context."""
    backend = getattr(text_model, "backend", None)
    tokenizer = getattr(backend, "tokenizer", None)
    limit = getattr(backend, "max_length", None)
    if tokenizer is None or not isinstance(limit, int) or limit <= 0:
        raise RuntimeError("source-gated ReaRAG requires an inspectable tokenizer and max_length")
    if limit > 4096:
        raise RuntimeError("source-gated ReaRAG max_length cannot exceed frozen 4096-token contract")
    if len(prompts) != len(texts):
        raise RuntimeError("source-gated text prompt/step count mismatch")
    lengths = []
    for index, (prompt, step) in enumerate(zip(prompts, texts)):
        prompt_n = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        step_n = len(tokenizer(step, add_special_tokens=False)["input_ids"])
        row = {"step": index + 1, "prompt_tokens": prompt_n, "step_tokens": step_n,
               "total_tokens": prompt_n + step_n, "truncated_tokens": 0}
        if step_n == 0 or prompt_n + step_n > limit:
            raise RuntimeError(
                "source-gated ReaRAG token budget violation; implicit truncation forbidden: "
                f"step={index + 1} prompt_tokens={prompt_n} step_tokens={step_n} "
                f"max_length={limit} overflow_tokens={max(0, prompt_n + step_n - limit)}"
            )
        lengths.append(row)
    return {"policy": "fail_before_implicit_truncation", "max_length": limit,
            "truncated_tokens": 0, "step_lengths": lengths}


def _canonical_gold_surfaces(primary: object, aliases: object) -> List[str]:
    """Return nonempty canonical-unique answer surfaces, primary first.

    Frozen mixed data stores aliases as a list of strings.  Treat a lone string
    as one alias for forward compatibility, ignore malformed containers/items,
    and always retain a valid primary answer.  Deduplication uses the same
    normalizer as canonical EM/F1, so punctuation/case-only variants cannot
    silently change tie-breaking or alias counts.
    """

    candidates: List[object] = [primary]
    if isinstance(aliases, str):
        candidates.append(aliases)
    elif isinstance(aliases, (list, tuple)):
        candidates.extend(value for value in aliases if isinstance(value, str))
    surfaces: List[str] = []
    seen: set[str] = set()
    for value in candidates:
        if not isinstance(value, str):
            continue
        surface = value.strip()
        normalized = canonical_answer_normalize(surface)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        surfaces.append(surface)
    return surfaces


def _decode_len(tokenizer, ids: List[int]) -> int:
    """Char length of the decoded prefix ``ids`` (special tokens kept so the
    offset is measured in the SAME stream the trainer scatters onto)."""
    if not ids:
        return 0
    try:
        return len(tokenizer.decode(ids, skip_special_tokens=False))
    except TypeError:  # fakes / tokenizers without the kwarg
        return len(tokenizer.decode(ids))


def _norm_text(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower())


def _grounded_mentions(
    mentions,
    question: str,
    retrieved_passages: Sequence[Dict[str, Any]],
    question_kg: Sequence[Tuple[str, str, str]],
) -> List[str]:
    """Subset of ``mentions`` that the question's own evidence supports (§3.4).

    A generated mention qualifies when it appears in the question, in a retrieved
    passage, or as a node of the question-anchored KG. Mentions the model
    invented out of nothing are excluded, so they cannot pull their real-but-
    irrelevant Wikidata subgraph into the reward graph.
    """
    haystack = _norm_text(question)
    for p in list(retrieved_passages)[:20]:
        if isinstance(p, dict):
            haystack += " " + _norm_text(str(p.get("contents") or p.get("text") or ""))
        else:
            haystack += " " + _norm_text(str(p))
    kg_nodes = set()
    for t in question_kg:
        if len(t) == 3:
            kg_nodes.add(_norm_text(str(t[0])).strip())
            kg_nodes.add(_norm_text(str(t[2])).strip())

    out: List[str] = []
    for m in mentions:
        key = _norm_text(m).strip()
        if not key:
            continue
        if key in kg_nodes or re.search(rf"\b{re.escape(key)}\b", haystack):
            out.append(m)
    return out


def _supports_skip(tokenizer) -> bool:
    try:
        tokenizer.decode([], skip_special_tokens=False)
        return True
    except TypeError:
        return False
    except Exception:
        return True


def step_spans_over_ids(
    response_ids: Sequence[int],
    tokenizer,
    n_steps: Optional[int] = None,
) -> List[Tuple[int, int]]:
    """Step ``(start, end)`` token spans in **response_ids coordinates** (#6).

    The previous path re-tokenised ``decode(response_ids, skip_special_tokens=
    True)`` and computed spans in that re-tokenised space, which does NOT align
    with the raw ``response_ids`` the PPO trainer scatters rewards onto (special
    tokens stripped + decode∘encode drift). Each per-step reward therefore
    landed a few tokens off its true position, smearing the per-step credit
    assignment that the whole StepRewardPPOTrainer exists to provide.

    Here we locate each ``[Step N]`` header directly in the decode of the
    actual ``response_ids`` stream and map its char offset back to a token index
    by binary-searching the monotonically-increasing prefix-decode length. Only
    the handful of step boundaries are searched (≈n_steps·log₂T decodes), not
    every token.
    """
    ids = [int(x) for x in (response_ids.tolist() if hasattr(response_ids, "tolist") else response_ids)]
    n = len(ids)
    if n == 0:
        return []
    full_text = tokenizer.decode(ids, skip_special_tokens=False) if _supports_skip(tokenizer) else tokenizer.decode(ids)
    headers = [m.start() for m in re.finditer(r"\[Step\s+\d+\]", full_text)]
    if not headers:
        return [(0, n)]

    def char_to_token(char_pos: int) -> int:
        """Smallest token index k such that decode(ids[:k]) reaches char_pos."""
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) // 2
            if _decode_len(tokenizer, ids[:mid]) >= char_pos:
                hi = mid
            else:
                lo = mid + 1
        return lo

    bounds = [char_to_token(h) for h in headers] + [n]
    spans: List[Tuple[int, int]] = []
    for i in range(len(headers)):
        start, end = bounds[i], bounds[i + 1]
        spans.append((start, max(end, start + 1)))
    if n_steps is not None:
        spans = spans[:n_steps]
    return spans


def _mixed_text_step_spans_over_ids(
    response_ids: Sequence[int],
    tokenizer,
    n_steps: int,
) -> List[Tuple[int, int]]:
    """Locate reasoning-step spans with Final Answer as a terminal boundary.

    The historical generic span intentionally lets the final step extend to the
    end of the response. Mixed ReaRAG needs a stricter credit boundary: its last
    step score belongs before ``[Final Answer]``, while outcome and global KG
    reward belong on the actual final generated token. This helper is private to
    the default-off mixed route so legacy token placement cannot drift.
    """

    ids = [
        int(value)
        for value in (
            response_ids.tolist()
            if hasattr(response_ids, "tolist") else response_ids
        )
    ]
    if not ids or n_steps <= 0:
        return []
    full_text = (
        tokenizer.decode(ids, skip_special_tokens=False)
        if _supports_skip(tokenizer) else tokenizer.decode(ids)
    )
    headers = [m.start() for m in re.finditer(r"\[Step\s+\d+\]", full_text)]
    headers = headers[:n_steps]
    if len(headers) != n_steps:
        return []
    final_candidates = [
        match.start()
        for match in re.finditer(
            r"\[Final Answer\]|(?:^|\n)\s*\*{0,3}Final Answer\*{0,3}\s*[:：]?",
            full_text,
            flags=re.IGNORECASE,
        )
        if match.start() > headers[-1]
    ]
    if not final_candidates:
        return []
    final_start = min(final_candidates)

    def char_to_token(char_pos: int) -> int:
        lo, hi = 0, len(ids)
        while lo < hi:
            mid = (lo + hi) // 2
            if _decode_len(tokenizer, ids[:mid]) >= char_pos:
                hi = mid
            else:
                lo = mid + 1
        return lo

    end_chars = headers[1:] + [final_start]
    spans = [
        (char_to_token(start), char_to_token(end))
        for start, end in zip(headers, end_chars)
    ]
    if any(end <= start for start, end in spans):
        return []
    return spans


@dataclass
class RewardSpec:
    """Per-sample inputs to :func:`KGProWeightRewardFunction.__call__`."""

    query: str
    gold_answer: str
    kg_subgraph: List[Tuple[str, str, str]]
    retrieved_passages: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Frozen evaluation-equivalent answer surfaces. Kept last so every existing
    # positional RewardSpec construction remains backward compatible.
    gold_answer_aliases: List[str] = field(default_factory=list)


class KGProWeightRewardFunction:
    """PPO/GRPO reward callable.

    Parameters
    ----------
    alpha_gate, prm_annotator, text_reward_model:
        Components composed inside a :class:`CompositeRewardModel`.
    tokenizer:
        Used to align per-step reward to token spans for PPO.
    outcome_weight, discount:
        Hyperparameters from :class:`CompositeRewardModel`.
    alpha_override:
        ``None`` (default) or one of ``{0.0, 0.5, 1.0}`` for the alpha
        ablations. The trained α-gate is bypassed if set.
    """

    def __init__(
        self,
        alpha_gate: AlphaGate,
        prm_annotator: PRMAnnotator,
        text_reward_model: TextRewardModel,
        tokenizer,
        outcome_weight: float = 1.0,
        discount: float = 0.95,
        alpha_override: Optional[float] = None,
        max_steps: int = 7,
        pure_em: bool = False,
        text_reward_scale: float = 1.0,
        # R7: minimum number of parsed [Step N] blocks for a trajectory to be
        # considered "valid" and eligible for the outcome reward.
        min_valid_steps: int = 3,
        # R8: minimum characters of actual reasoning content per step.
        min_reasoning_chars: int = 20,
        step_reward_scale: float = 1.0,
        subgraph_retriever = None,
        # retraining_plan §9.4-3 / R-1b: step-shortfall penalty. 0.0 = disabled
        # (reproduces every pre-2026-08-22 run).
        shortfall_coef: float = 0.0,
        target_steps: int = 3,
        # retraining_plan §9.4-1 (量纲): remove R_Text's DC offset before mixing.
        # False = pre-2026-08-23 behaviour, bit-for-bit.
        center_text_reward: bool = False,
        text_baseline_momentum: float = 0.99,
        proofkg_process_reward: bool = False,
        proofkg_outcome_only_reward: bool = False,
        proofkg_process_version: str = "v1",
        proofkg_process_weight: float = 1.0,
        proofkg_f1_weight: float = 0.0,
        proofkg_dynamic_validity: bool = False,
        # Mixed-dataset PPO fast path.  When enabled, *every* row receives the
        # same outcome reward; only identity-safe complete automatic ProofKG
        # rows may receive the optional v2.1 process term.  Default-off keeps
        # every historical configuration on its original reward path.
        mixed_outcome_reward: bool = False,
        # Add the same frozen ReaRAG step-quality signal to every valid mixed
        # trajectory. It is length-normalised and assigned to each corresponding
        # step end; no legacy alpha/PRM mixture is involved. Default-off
        # preserves old experiments.
        mixed_text_reward: bool = False,
        runtime_contract_version: str = "legacy",
        source_gated_reward_version: str = "disabled",
        source_gate_format_version: str = "v1",
        source_gate_credit_version: str = "disabled",
        source_gate_mode: str = "learned",
        source_gate_calibration_path: Optional[str] = None,
        source_quality_gate: Optional[SourceQualityGateV1 | SourceCreditGateV2] = None,
        answer_format_reward_version: str = "legacy",
    ) -> None:
        self.composite = CompositeRewardModel(
            alpha_gate=alpha_gate,
            prm_annotator=prm_annotator,
            text_reward_model=text_reward_model,
            outcome_weight=outcome_weight,
            discount=discount,
            text_reward_scale=text_reward_scale,
            step_reward_scale=step_reward_scale,
            shortfall_coef=shortfall_coef,
            target_steps=target_steps,
            center_text_reward=center_text_reward,
            text_baseline_momentum=text_baseline_momentum,
        )
        self.tokenizer = tokenizer
        self.alpha_override = alpha_override
        self.max_steps = max_steps
        # R7: minimum valid step count for trajectory validity gating.
        self.min_valid_steps = min_valid_steps
        # R8: minimum reasoning content per step (content-aware gate).
        self.min_reasoning_chars = min_reasoning_chars
        # R9: step reward scale for KG/Text composite.
        self.step_reward_scale = step_reward_scale
        # R9: dynamic KG subgraph from model output, not static silver data
        self.subgraph_retriever = subgraph_retriever
        # Pure EM reward mode (ablation): ignore R_KG and R_text entirely;
        # reward is EM × outcome_weight on the final step (no per-step bonuses).
        # This is the upper bound for "what PPO can achieve when reward is
        # perfectly aligned with the evaluation metric".
        self.pure_em = pure_em
        self.proofkg_process_reward = bool(proofkg_process_reward)
        self.proofkg_outcome_only_reward = bool(proofkg_outcome_only_reward)
        self.proofkg_process_version = str(proofkg_process_version)
        self.proofkg_process_weight = float(proofkg_process_weight)
        self.proofkg_f1_weight = float(proofkg_f1_weight)
        self.proofkg_dynamic_validity = bool(proofkg_dynamic_validity)
        self.mixed_outcome_reward = bool(mixed_outcome_reward)
        self.mixed_text_reward = bool(mixed_text_reward)
        if runtime_contract_version not in {"legacy", "v2"}:
            raise ValueError("runtime_contract_version must be 'legacy' or 'v2'")
        self.runtime_contract_version = runtime_contract_version
        if source_gated_reward_version not in {"disabled", "v1"}:
            raise ValueError("source_gated_reward_version must be disabled or v1")
        self.source_gated_reward_version = source_gated_reward_version
        self.source_gate_format_version = source_gate_format_version
        self.source_gate_format_contract = source_gate_format_contract_version(source_gate_format_version)
        validate_source_gate_credit_config(source_gate_credit_version,
                                          source_gated_reward_version,
                                          source_gate_format_version)
        self.source_gate_credit_version = source_gate_credit_version
        if answer_format_reward_version not in {"legacy", "v2"}:
            raise ValueError("answer_format_reward_version must be legacy or v2")
        if answer_format_reward_version == "v2" and (
            source_gated_reward_version != "v1" or source_gate_format_version != "v2"
            or source_gate_credit_version != "v2"
        ):
            raise ValueError("answer format reward v2 requires source-gated v1, format v2 and source credit v2")
        self.answer_format_reward_version = answer_format_reward_version
        if source_gate_credit_version == "disabled" and isinstance(source_quality_gate, (SourceCreditGateV1, SourceCreditGateV2)):
            raise TypeError("source_gate_credit_version=disabled rejects source-credit successor gates")
        self.source_gate_mode = source_gate_mode
        self.source_quality_gate = None
        if source_gated_reward_version == "v1":
            if source_gate_mode not in {"text", "fixed", "learned"}:
                raise ValueError("source_gate_mode must be text, fixed or learned")
            if not (self.mixed_outcome_reward and self.mixed_text_reward):
                raise ValueError("source-gated v1 requires mixed outcome and text rewards")
            if runtime_contract_version != "v2" or center_text_reward:
                raise ValueError("source-gated v1 requires runtime v2 and frozen text stats")
            if not proofkg_process_reward or proofkg_process_version != "v2_3":
                raise ValueError("source-gated v1 requires shared ProofKG v2_3")
            if alpha_override is not None or pure_em or proofkg_outcome_only_reward:
                raise ValueError("source-gated v1 cannot combine historical reward overrides")
            for value, expected, label in (
                (outcome_weight, 4.0, "outcome_weight"),
                (text_reward_scale, .30, "text_reward_scale"),
                (proofkg_process_weight, .20, "proofkg_process_weight"),
                (proofkg_f1_weight, .10, "proofkg_f1_weight"),
            ):
                if not math.isclose(value, expected, rel_tol=0, abs_tol=1e-12):
                    raise ValueError(f"source-gated v1 requires {label}={expected}")
            if not proofkg_dynamic_validity:
                raise ValueError("source-gated v1 requires shared dynamic validity")
            if source_quality_gate is None:
                if not source_gate_calibration_path:
                    raise ValueError("source-gated v1 requires a frozen calibration artifact")
                source_quality_gate = load_source_gate_for_runtime(
                    source_gate_calibration_path, source_gate_format_version,
                    source_gate_credit_version,
                )
            validate_source_gate_runtime_contract(source_quality_gate,
                                                 source_gate_format_version,
                                                 source_gate_credit_version)
            self.source_quality_gate = source_quality_gate
        if self.mixed_text_reward and not self.mixed_outcome_reward:
            raise ValueError("mixed_text_reward requires mixed_outcome_reward")
        if (self.mixed_text_reward and not self.composite.center_text_reward
                and source_gated_reward_version == "disabled"):
            raise ValueError("mixed_text_reward requires causal text centering")
        if self.proofkg_process_weight < 0 or self.proofkg_f1_weight < 0:
            raise ValueError("ProofKG reward weights must be non-negative")
        if self.proofkg_process_version not in {"v1", "v2_1", "v2_2", "v2_3"}:
            raise ValueError(
                "proofkg_process_version must be 'v1', 'v2_1', 'v2_2', or 'v2_3', got "
                f"{self.proofkg_process_version!r}"
            )

    @staticmethod
    def _is_valid_trajectory(
        steps: list,
        response: str,
        min_steps: int = 3,
        min_reasoning_chars: int = 20,
    ) -> bool:
        """R7: Check whether a generated trajectory meets format requirements.

        A trajectory is "valid" (eligible for the outcome reward) when ALL of
        the following hold:

        1. At least ``min_steps`` parseable ``[Step N]`` blocks.
        2. A ``Final Answer`` can be extracted.
        3. Step indices are sequential (1, 2, 3, …).
        4. Every step has non-empty text.
        5. (R8) Every step's ``Reasoning:`` section contains at least
           ``min_reasoning_chars`` characters of actual content (excluding
           whitespace and subsequent Knowledge/Conclusion/Final Answer sections).

        This is a FORMAT constraint — it does NOT judge factual correctness.
        Its sole purpose is to make the outcome reward *conditional* on
        producing a well-structured reasoning trace, so PPO cannot collect the
        "grand prize" by emitting a bare answer or an empty "Reasoning:" block.
        """
        if not steps or len(steps) < min_steps:
            return False
        if extract_final_answer(response) is None:
            return False
        expected = 1
        for s in steps:
            if s.index != expected:
                return False
            if not s.raw_text or not s.raw_text.strip():
                return False
            # R8: content-aware gate — ``Reasoning:`` must not be empty.
            # PPO was exploiting the old check (just non-empty raw_text) by
            # writing ``[Step 1]\\nReasoning: \\nFinal Answer: X``, which
            # parses as raw_text="Reasoning: \\nFinal Answer: X" → non-empty
            # → gate passes. Now we extract the actual reasoning body and
            # require >= min_reasoning_chars of substantive content.
            body = s.raw_text.strip()
            if "Reasoning:" in body:
                after = body.split("Reasoning:", 1)[1]
                # Stop at the next structural label.
                reasoning = re.split(
                    r'Knowledge Used:|Conclusion:|Final Answer:', after
                )[0].strip()
                if len(reasoning) < min_reasoning_chars:
                    return False
            else:
                # R9 v6: step without "Reasoning:" field is invalid
                return False
            expected += 1
        return True

    def _score_mixed_text_process(
        self,
        steps: Sequence[Any],
        spec: RewardSpec,
    ) -> Dict[str, Any]:
        """Score valid steps with frozen ReaRAG and a causal EMA baseline.

        Per-step scores are causally centered and clipped.  Each receives
        ``text_reward_scale / n_steps`` credit at its own step-end token, so the
        trajectory sum is exactly the approved mean and writing more steps
        cannot mechanically increase reward.  Outcome and optional ProofKG-v2.1
        remain trajectory-level terms on the final generated token.
        """

        messages = build_sft_messages(
            question=spec.query,
            retrieved_passages=spec.retrieved_passages,
            kg_triples=spec.kg_subgraph,
        )
        rendered_prompt = "\n\n".join(message["content"] for message in messages)
        prompts: List[str] = []
        for index, _step in enumerate(steps):
            prefix = "\n".join(step.raw_text for step in steps[:index])
            prompts.append(rendered_prompt + ("\n\n" + prefix if prefix else ""))
        raw_scores = [
            float(value)
            for value in self.composite.text_reward_model.score_steps(
                prompts, [step.raw_text for step in steps]
            )
        ]
        if len(raw_scores) != len(steps):
            raise RuntimeError(
                "ReaRAG text scorer returned a different number of scores than steps"
            )
        baselines: List[float] = []
        residuals: List[float] = []
        centered_scores: List[float] = []
        for raw in raw_scores:
            if not math.isfinite(raw):
                raise RuntimeError(f"ReaRAG text scorer returned non-finite value: {raw}")
            baseline = float(self.composite._update_text_baseline(raw))
            baselines.append(baseline)
            residual = raw - baseline
            residuals.append(residual)
            centered_scores.append(max(-1.0, min(1.0, residual)))
        mean_centered = (
            sum(centered_scores) / len(centered_scores) if centered_scores else 0.0
        )
        weighted_step_rewards = [
            float(self.composite.text_reward_scale * value / len(centered_scores))
            for value in centered_scores
        ] if centered_scores else []
        return {
            "raw_step_scores": raw_scores,
            "baseline_before_step": baselines,
            "centered_clipped_step_scores": centered_scores,
            "mean_centered_clipped": float(mean_centered),
            "mean_abs_centered_clipped": float(
                sum(abs(value) for value in centered_scores) / len(centered_scores)
                if centered_scores else 0.0
            ),
            "clip_frac": float(
                sum(abs(value) > 1.0 for value in residuals) / len(residuals)
                if residuals else 0.0
            ),
            "weighted_step_rewards": weighted_step_rewards,
            "weighted_reward": float(sum(weighted_step_rewards)),
        }

    def _score_source_gated_text_v1(
        self, steps: Sequence[Any], spec: RewardSpec, alpha_eff: float,
    ) -> Dict[str, Any]:
        prompts, texts = source_gate_text_inputs_v1(spec, steps)
        budget = source_gate_text_budget_v1(self.composite.text_reward_model, prompts, texts)
        raw = [float(value) for value in self.composite.text_reward_model.score_steps(prompts, texts)]
        if len(raw) != len(steps) or any(not math.isfinite(value) or not -1 <= value <= 1 for value in raw):
            raise RuntimeError("source-gated ReaRAG requires one finite [-1,1] score per step")
        stats = self.source_quality_gate.normalization
        center, scale = stats["text_center"], stats["text_scale"]
        text_v2 = None
        if self.source_gate_credit_version == "v2":
            text_v2 = normalize_text_steps_v2(raw, stats["text_v2"])
            normalized = text_v2["normalized_unclipped_step_scores"]
            clipped = text_v2["bounded_step_scores"]
        else:
            normalized = [(value - center) / scale for value in raw]
            clipped = [max(-1.0, min(1.0, value)) for value in normalized]
        weighted = [.30 * (1.0 - alpha_eff) * value / len(raw) for value in clipped]
        return {
            "raw_step_scores": raw, "baseline_before_step": [center] * len(raw),
            "centered_clipped_step_scores": clipped,
            "normalized_unclipped_step_scores": normalized,
            "mean_centered_clipped": text_v2["mean_bounded"] if text_v2 else sum(clipped) / len(raw),
            "mean_abs_centered_clipped": sum(abs(value) for value in clipped) / len(raw),
            "clip_frac": text_v2["hard_clip_frac"] if text_v2 else sum(abs(value) > 1 for value in normalized) / len(raw),
            "weighted_step_rewards": weighted, "weighted_reward": sum(weighted),
            "token_budget": budget,
            **({"text_normalization_v2": text_v2} if text_v2 else {}),
        }

    def __call__(
        self,
        prompt: str,
        response: str,
        spec: RewardSpec,
        logprobs_per_step: Optional[Sequence[Optional[Sequence[float]]]] = None,
        response_ids: Optional[Sequence[int]] = None,
        step_spans: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> Dict[str, Any]:
        """Compute per-step + per-token rewards for one rollout.

        ``logprobs_per_step`` (P1-1): optional list aligned to the parsed steps,
        each entry the token logprobs of that step (or ``None``). When absent we
        fall back to ``None`` per step, matching the inference path.

        ``response_ids`` / ``step_spans`` (#6): when the caller passes the raw
        generated token ids (and, optionally, precomputed spans in those same
        coordinates), the per-token reward tensor is built in ``response_ids``
        space so it aligns EXACTLY with what the PPO trainer scatters onto. When
        omitted (e.g. the offline test), we fall back to re-tokenising the
        decoded ``response`` — correct in isolation, but only used outside the
        trainer loop.
        """
        all_steps = parse_steps(response, known_kg=spec.kg_subgraph)
        steps = all_steps[: self.max_steps]
        # Finding-2 follow-up: strip reasoning-scaffold mentions ("Reasoning",
        # "Conclusion", …) so link_confidence reflects real entities only. MUST
        # match Phase 2's _build_samples_accepted_only, which applies the same
        # clean_entities to the same parser output.
        for _s in steps:
            _s.mentioned_entities = clean_entities(_s.mentioned_entities)
        predicted_answer = extract_final_answer(response) or ""
        proofkg_record = spec.metadata.get("question_kg_runtime") or {}
        proofkg_eligible = is_automatic_proofkg(proofkg_record, spec.kg_subgraph)
        identity_safe_proofkg_eligible = is_identity_safe_automatic_proofkg(
            proofkg_record,
            spec.kg_subgraph,
            dataset=spec.metadata.get("dataset"),
            qid=spec.metadata.get("qid"),
        )
        # Historical automatic-ProofKG experiments retain their original
        # structural eligibility predicate.  The new mixed route additionally
        # requires the exact dataset::qid join performed by the versioned loader.
        routed_proofkg_eligible = (
            identity_safe_proofkg_eligible
            if self.mixed_outcome_reward
            else proofkg_eligible
        )
        source_features = None
        if self.source_gated_reward_version == "v1":
            source_features = compute_gate_features(spec, steps, {})
            routed_proofkg_eligible = bool(source_features["m_graph"])
            proofkg_record = spec.metadata.get("source_quality_record") or {}
        validity_min_steps = (
            proofkg_required_steps(proofkg_record)
            if self.proofkg_dynamic_validity and routed_proofkg_eligible
            else self.min_valid_steps
        )

        # R7: gate the outcome reward on trajectory validity.
        # Per-step composite rewards are still computed regardless — PPO gets
        # signal from step-level KG/Text quality even for incomplete traces.
        trajectory_valid = self._is_valid_trajectory(
            steps, response, min_steps=validity_min_steps,
            min_reasoning_chars=self.min_reasoning_chars,
        )
        format_contract_violations: List[str] = []
        if self.runtime_contract_version == "v2":
            format_contract_violations = _v2_format_contract_violations(
                response, all_steps, self.max_steps,
            )
            trajectory_valid = trajectory_valid and not format_contract_violations
        if self.source_gated_reward_version == "v1":
            source_validity = validate_source_gate_trajectory(
                spec, response, max_steps=self.max_steps,
                min_valid_steps=self.min_valid_steps, min_reasoning_chars=self.min_reasoning_chars,
                format_version=self.source_gate_format_version,
            )
            trajectory_valid = source_validity["valid"]
            format_contract_violations = source_validity["violations"]
            validity_min_steps = source_validity["required_steps"]

        shortfall_salvage = None
        if self.answer_format_reward_version == "v2":
            shortfall_salvage = inspect_shortfall_salvage_v2(
                response, steps=source_validity["steps"],
                required_steps=source_validity["required_steps"],
                violations=source_validity["violations"],
                known_passage_ids=list(range(1, min(10, len(spec.retrieved_passages)) + 1)),
            )

        # ── Automatic ProofKG fast-path (experimental, config gated) ──
        # The scorer is deterministic and Gold-free until the explicit outcome
        # term.  Eligible rows bypass the legacy PRM/alpha/text mixture, whose
        # old rankability was near random and whose KG term was zero ~85% of the
        # time.  In the historical route, ineligible rows retain the old path.
        # In the default-off mixed route, every row receives the same outcome
        # and optional shared ReaRAG channels, while ineligible rows can never
        # touch ProofKG process reward or the legacy PRM/alpha path.
        historical_proofkg_fast_path = (
            (self.proofkg_process_reward or self.proofkg_outcome_only_reward)
            and proofkg_eligible
        )
        if self.mixed_outcome_reward or historical_proofkg_fast_path:
            answer = predicted_answer.split("\n", 1)[0].strip()
            process_applied = bool(
                self.proofkg_process_reward and routed_proofkg_eligible
            )
            if not process_applied:
                process = {
                    "score": 0.0,
                    "scorer_version": (
                        "mixed-outcome-only-ineligible"
                        if self.mixed_outcome_reward and not routed_proofkg_eligible
                        else "outcome-only"
                    ),
                }
            elif trajectory_valid and self.proofkg_process_version == "v2_1":
                plan = proofkg_record.get("query_plan") or {}
                execution = proofkg_record.get("execution") or {}
                if not execution.get("hops"):
                    raise ValueError(
                        "ProofKG process-v2.1 requires execution.hops in the "
                        "identity-safe question-KG record"
                    )
                process = score_proofkg_v2(
                    question=spec.query,
                    generation=response,
                    kg_triples=spec.kg_subgraph,
                    execution_trace=build_execution_trace(plan, execution),
                    planned_hops=len(plan.get("hops") or []),
                )
            elif trajectory_valid and self.proofkg_process_version == "v2_2":
                plan = proofkg_record.get("query_plan") or {}
                execution = proofkg_record.get("execution") or {}
                if not execution.get("hops"):
                    raise ValueError(
                        "ProofKG process-v2.2 requires execution.hops in the "
                        "identity-safe question-KG record"
                    )
                process = score_proofkg_v2_2(
                    question=spec.query,
                    generation=response,
                    kg_triples=spec.kg_subgraph,
                    execution_trace=build_execution_trace_v2_2(plan, execution),
                    planned_hops=len(plan.get("hops") or []),
                )
            elif trajectory_valid and self.proofkg_process_version == "v2_3":
                plan = proofkg_record.get("query_plan") or {}
                execution = proofkg_record.get("execution") or {}
                if not execution.get("hops"):
                    raise ValueError(
                        "ProofKG structural-answer-v2.3 requires execution.hops "
                        "in the identity-safe question-KG record"
                    )
                process = score_proofkg_v2_3(
                    question=spec.query,
                    generation=response,
                    kg_triples=spec.kg_subgraph,
                    execution_trace=build_execution_trace_v2_3(plan, execution),
                    planned_hops=len(plan.get("hops") or []),
                )
            elif trajectory_valid:
                process = score_grounded_process(
                    question=spec.query,
                    kg=spec.kg_subgraph,
                    steps=steps,
                    predicted_answer=answer,
                )
            else:
                process = (
                    {
                        "score": 0.0,
                        "scorer_version": "invalid-not-scored",
                    }
                    if self.mixed_outcome_reward
                    else {
                        # Preserve the historical fast-path telemetry exactly;
                        # its invalid score was diagnostic only and never added
                        # to the hard invalid reward.
                        "score": -1.0,
                        "citation_precision": 0.0,
                        "conclusion_grounding": 0.0,
                        "reachable_edge_coverage": 0.0,
                        "answer_path_alignment": 0.0,
                        "unknown_citation_ratio": 0.0,
                        "duplicate_citation_ratio": 0.0,
                    }
                )
            gold_surfaces = (
                _canonical_gold_surfaces(spec.gold_answer, spec.gold_answer_aliases)
                if self.mixed_outcome_reward else [spec.gold_answer]
            ) or [spec.gold_answer]
            em_matched_alias = spec.gold_answer
            f1_matched_alias = spec.gold_answer
            answer_signal_eligible = trajectory_valid or bool(
                shortfall_salvage is not None and shortfall_salvage.eligible
            )
            if answer_signal_eligible and answer and spec.gold_answer:
                if self.mixed_outcome_reward:
                    em_values = [
                        canonical_exact_match(answer, gold) for gold in gold_surfaces
                    ]
                    f1_values = [
                        canonical_token_f1(answer, gold) for gold in gold_surfaces
                    ]
                    em_index = max(range(len(gold_surfaces)), key=em_values.__getitem__)
                    f1_index = max(range(len(gold_surfaces)), key=f1_values.__getitem__)
                    em, f1 = em_values[em_index], f1_values[f1_index]
                    em_matched_alias = gold_surfaces[em_index]
                    f1_matched_alias = gold_surfaces[f1_index]
                else:
                    # Historical fast-path remains single-Gold and uses its
                    # original normalisers bit-for-bit.
                    em = float(self.composite._em(answer, spec.gold_answer))
                    f1 = token_f1(answer, spec.gold_answer)
            else:
                em, f1 = 0.0, 0.0
            if trajectory_valid:
                outcome_component = self.composite.outcome_weight * (
                    em + self.proofkg_f1_weight * f1
                )
                source_details = None
                if self.source_gated_reward_version == "v1":
                    gate = self.source_quality_gate
                    source_features = (
                        gate.compute_features(spec, steps, process if process_applied else {})
                        if self.source_gate_credit_version == "v2" else
                        compute_gate_features(spec, steps, process if process_applied else {})
                    )
                    if self.source_gate_credit_version in {"v1", "v2"}:
                        source_features = gate.mask_features(spec, source_features)
                    alpha_pred = (
                        0.0 if self.source_gate_mode == "text" else
                        gate.normalization["fixed_alpha"] if self.source_gate_mode == "fixed" else
                        gate.predict(source_features)
                    )
                    if not math.isfinite(alpha_pred) or not 0 <= alpha_pred <= 1:
                        raise RuntimeError("source gate returned invalid alpha")
                    alpha_eff = float(source_features["m_graph"]) * alpha_pred
                    text_details = self._score_source_gated_text_v1(steps, spec, alpha_eff)
                    graph_raw = float(process["score"])
                    if not math.isfinite(graph_raw) or not 0 <= graph_raw <= .85 + 1e-12:
                        raise RuntimeError("source-gated Graph score outside v2_3 range")
                    graph_normalized = ((graph_raw - gate.normalization["graph_center"])
                                        / gate.normalization["graph_scale"]) if process_applied else 0.0
                    graph_clipped = max(-1.0, min(1.0, graph_normalized))
                    source_details = {
                        "version": "v1", "mode": self.source_gate_mode,
                        "format_contract_version": self.source_gate_format_contract,
                        "alpha_predicted": float(alpha_pred), "alpha_effective": alpha_eff,
                        "features": source_features, "m_graph": source_features["m_graph"],
                        "graph_raw": graph_raw, "graph_normalized_unclipped": graph_normalized,
                        "graph_normalized": graph_clipped,
                        "text_normalized": text_details["mean_centered_clipped"],
                        "text_normalized_unclipped_steps": text_details["normalized_unclipped_step_scores"],
                        "normalization": dict(gate.normalization),
                        "text_aggregation": (
                            text_details["text_normalization_v2"]["application_contract"]
                            if self.source_gate_credit_version == "v2" else "step_normalize_then_clip_then_mean_v1"
                        ),
                        "text_score_contract": "rearag-passage-only-raw-tanh-nll-v1",
                        "token_budget": text_details["token_budget"],
                        "artifact_payload_sha256": gate.artifact["payload_sha256"],
                        **({"source_credit_version": self.source_gate_credit_version,
                            "source_credit_mask": dict(source_features["source_credit_mask"])}
                           if self.source_gate_credit_version in {"v1", "v2"} else {}),
                        **({"text_normalization_v2": text_details["text_normalization_v2"],
                            "text_normalization_version": gate.normalization["text_v2"]["version"]}
                           if self.source_gate_credit_version == "v2" else {}),
                    }
                else:
                    text_details = (
                    self._score_mixed_text_process(steps, spec)
                    if self.mixed_text_reward
                    else {
                        "raw_step_scores": [],
                        "baseline_before_step": [],
                        "centered_clipped_step_scores": [],
                        "mean_centered_clipped": 0.0,
                        "mean_abs_centered_clipped": 0.0,
                        "clip_frac": 0.0,
                        "weighted_step_rewards": [],
                        "weighted_reward": 0.0,
                    }
                    )
                text_component = float(text_details["weighted_reward"])
                process_component = (
                    self.proofkg_process_weight * float(process["score"])
                    if process_applied else 0.0
                )
                if source_details is not None:
                    process_component = .20 * alpha_eff * graph_clipped
                    source_details["text_component"] = text_component
                    source_details["graph_component"] = process_component
                final_reward = outcome_component + text_component + process_component
            else:
                # Invalid trajectories never receive either process channel.
                # The opt-in objective below may salvage a bounded answer term
                # for the narrowly specified complete two-step shortfall only.
                outcome_component = -self.composite.outcome_weight
                text_component = 0.0
                process_component = 0.0
                text_details = {
                    "raw_step_scores": [],
                    "baseline_before_step": [],
                    "centered_clipped_step_scores": [],
                    "mean_centered_clipped": 0.0,
                    "mean_abs_centered_clipped": 0.0,
                    "clip_frac": 0.0,
                    "weighted_step_rewards": [],
                    "weighted_reward": 0.0,
                }
                final_reward = outcome_component
                if self.source_gated_reward_version == "v1" and self.source_gate_credit_version in {"v1", "v2"}:
                    if self.source_gate_credit_version == "v2":
                        source_features = self.source_quality_gate.compute_features(spec, steps, {})
                    source_features = self.source_quality_gate.mask_features(spec, source_features)
                source_details = ({
                    "version": "v1", "mode": self.source_gate_mode,
                    "format_contract_version": self.source_gate_format_contract,
                    "alpha_predicted": 0.0, "alpha_effective": 0.0,
                    "features": source_features, "m_graph": source_features["m_graph"],
                    "graph_raw": 0.0, "graph_normalized": 0.0, "text_normalized": 0.0,
                    "text_component": 0.0, "graph_component": 0.0,
                    "invalid_not_scored": True,
                    "artifact_payload_sha256": self.source_quality_gate.artifact["payload_sha256"],
                    **({"source_credit_version": self.source_gate_credit_version,
                        "source_credit_mask": dict(source_features["source_credit_mask"])}
                       if self.source_gate_credit_version in {"v1", "v2"} else {}),
                    **({"text_normalization_version": self.source_quality_gate.normalization["text_v2"]["version"]}
                       if self.source_gate_credit_version == "v2" else {}),
                } if self.source_gated_reward_version == "v1" else None)
            objective_details = None
            if shortfall_salvage is not None:
                objective = compose_answer_format_objective_v2(
                    trajectory_valid=bool(trajectory_valid), salvage_contract=shortfall_salvage,
                    outcome_em=em, outcome_f1=f1,
                    text_component=text_component, graph_component=process_component,
                )
                # Preserve the historical valid path and its arithmetic exactly.
                # Invalid rows have no process scores and change only this term.
                if not trajectory_valid:
                    outcome_component = objective.outcome_component
                    final_reward = outcome_component
                objective_details = {**objective.telemetry(), **shortfall_salvage.telemetry(),
                                     "strict_valid": bool(trajectory_valid)}
                # Unconditional frozen evaluation metrics are diagnostics only.
                # They must not be confused with format-gated training EM/F1.
                canonical_answer = extract_kg_proweight_answer(extract_kg_proweight_answer(response))
                objective_details.update(
                    canonical_em=float(max(canonical_exact_match(canonical_answer, gold) for gold in gold_surfaces)),
                    canonical_f1=float(max(canonical_token_f1(canonical_answer, gold) for gold in gold_surfaces)),
                )
            text_step_rewards = list(text_details["weighted_step_rewards"])
            per_step_rewards = (
                list(text_step_rewards)
                if trajectory_valid and text_step_rewards
                else [0.0] * max(1, len(steps))
            )
            # Keep the logical per-step total auditable: the final list element
            # includes trajectory-level outcome/ProofKG, while token placement
            # below separates those from ReaRAG's reasoning-step boundaries.
            per_step_rewards[-1] += float(outcome_component + process_component)
            if not math.isclose(
                sum(per_step_rewards), final_reward, rel_tol=0.0, abs_tol=1e-9
            ):
                raise RuntimeError(
                    "mixed reward token allocation does not sum to trajectory reward"
                )

            if response_ids is not None:
                ids = [int(x) for x in (
                    response_ids.tolist() if hasattr(response_ids, "tolist") else response_ids
                )]
                n_tokens = len(ids)
                spans = (
                    list(step_spans) if step_spans is not None
                    else step_spans_over_ids(ids, self.tokenizer, len(steps))
                )
            else:
                spans = extract_step_token_spans(response, self.tokenizer)
                ids = [
                    int(value) for value in self.tokenizer(
                        response, add_special_tokens=False
                    )["input_ids"]
                ]
                n_tokens = len(ids)
            # A format-invalid response may have no parseable step.  Its -W
            # penalty must still reach the final generated token.
            if not spans and n_tokens:
                spans = [(0, n_tokens)]
            token_rewards = torch.zeros(n_tokens, dtype=torch.float32)
            text_step_spans = (
                _mixed_text_step_spans_over_ids(ids, self.tokenizer, len(steps))
                if text_step_rewards else []
            )
            if text_step_rewards and len(text_step_spans) != len(text_step_rewards):
                raise RuntimeError(
                    "mixed ReaRAG reasoning spans do not align with parsed steps: "
                    f"spans={len(text_step_spans)} scores={len(text_step_rewards)}"
                )
            for span, reward in zip(text_step_spans, text_step_rewards):
                start, end = span
                if end > 0 and start < n_tokens:
                    token_rewards[min(end - 1, n_tokens - 1)] += float(reward)
            # Outcome and KG-v2.1 are trajectory-level.  Put them on the actual
            # final generated token, independent of any step-boundary decoder
            # approximation. Invalid rows receive their net answer/format term.
            trajectory_level_term = float(outcome_component + process_component)
            if n_tokens:
                token_rewards[n_tokens - 1] += trajectory_level_term
            elif trajectory_level_term:
                raise RuntimeError("non-zero mixed trajectory reward has no response token")
            return {
                "per_step_rewards": per_step_rewards,
                "per_step_records": [],
                "token_rewards": token_rewards,
                "step_spans": spans,
                "text_step_spans": text_step_spans,
                "predicted_answer": answer,
                "trajectory_reward": float(final_reward),
                "trajectory_valid": bool(trajectory_valid),
                **({"answer_format_reward": objective_details} if objective_details is not None else {}),
                **({"source_gate": source_details} if source_details is not None else {}),
                **({
                    "runtime_contract_version": self.runtime_contract_version,
                    "format_contract_violations": format_contract_violations,
                } if self.runtime_contract_version == "v2" else {}),
                "proofkg_process": {
                    "eligible": bool(routed_proofkg_eligible),
                    "identity_safe": bool(identity_safe_proofkg_eligible),
                    "mixed_outcome_reward": bool(self.mixed_outcome_reward),
                    "process_applied": bool(process_applied and trajectory_valid),
                    "scorer_version": str(
                        process.get("scorer_version") or self.proofkg_process_version
                    ),
                    "required_steps": int(validity_min_steps),
                    "process_weight": (
                        self.proofkg_process_weight
                        if process_applied and (
                            trajectory_valid or not self.mixed_outcome_reward
                        ) else 0.0
                    ),
                    "process_score": float(process["score"]),
                    "outcome_em": em,
                    "outcome_f1": f1,
                    "gold_alias_count": int(len(gold_surfaces)),
                    "outcome_em_matched_alias": str(em_matched_alias),
                    "outcome_f1_matched_alias": str(f1_matched_alias),
                    "components": dict(process.get("components") or {}),
                    **{
                        key: float(value)
                        for key, value in process.items()
                        if key != "score" and isinstance(value, (int, float, bool))
                    },
                },
                **({
                    "mixed_reward": {
                        "dataset": str(spec.metadata.get("dataset") or ""),
                        "outcome": float(outcome_component),
                        "text": float(text_component),
                        "process": float(process_component),
                        "total": float(final_reward),
                        "text_enabled": bool(self.mixed_text_reward),
                        "text_raw_step_mean": float(
                            sum(text_details["raw_step_scores"])
                            / len(text_details["raw_step_scores"])
                            if text_details["raw_step_scores"] else 0.0
                        ),
                        "text_mean_centered_clipped": float(
                            text_details["mean_centered_clipped"]
                        ),
                        "text_raw_step_scores": list(text_details["raw_step_scores"]),
                        "text_baseline_before_step": list(
                            text_details["baseline_before_step"]
                        ),
                        "text_centered_clipped_step_scores": list(
                            text_details["centered_clipped_step_scores"]
                        ),
                        "text_centered_abs_mean": float(
                            text_details["mean_abs_centered_clipped"]
                        ),
                        "text_clip_frac": float(text_details["clip_frac"]),
                        "text_ema_baseline": float(self.composite.text_baseline),
                        "text_ema_n_obs": int(self.composite.text_baseline_n_obs),
                        "text_weighted_step_rewards": list(
                            text_details["weighted_step_rewards"]
                        ),
                        "proofkg_eligible": bool(routed_proofkg_eligible),
                        "gold_alias_count": int(len(gold_surfaces)),
                        "outcome_em_matched_alias": str(em_matched_alias),
                        "outcome_f1_matched_alias": str(f1_matched_alias),
                        "outcome_em_matched_nonprimary": bool(
                            canonical_answer_normalize(em_matched_alias)
                            != canonical_answer_normalize(spec.gold_answer)
                        ),
                        "outcome_f1_matched_nonprimary": bool(
                            canonical_answer_normalize(f1_matched_alias)
                            != canonical_answer_normalize(spec.gold_answer)
                        ),
                    }
                } if self.mixed_outcome_reward else {}),
            }

        # ── Pure EM reward fast-path (ablation) ──
        # When enabled, skip the entire composite reward pipeline (R_KG, R_text,
        # α-gate). Reward = EM × outcome_weight on the last step ONLY when the
        # trajectory is valid. (R7: outcome now gated on trajectory validity.)
        if self.pure_em:
            per_step_rewards: List[float] = (
                [0.0] * len(steps) if steps else [0.0] * max(len(steps), 1)
            )
            outcome = 0.0
            if trajectory_valid and predicted_answer and spec.gold_answer:
                outcome = float(self.composite._em(predicted_answer, spec.gold_answer))
            if per_step_rewards and outcome:
                per_step_rewards[-1] += self.composite.outcome_weight * outcome
            # NOTE (retraining_plan §9.4-3): the step-shortfall penalty is
            # deliberately NOT applied here. pure_em is the "reward is exactly the
            # eval metric" upper-bound ablation; adding a process-shaping term
            # would stop it measuring that. Expect this arm to collapse to short
            # trajectories -- that is the finding, not a bug.

            # token mapping (shared with the main path)
            if response_ids is not None:
                ids = [int(x) for x in (response_ids.tolist() if hasattr(response_ids, "tolist") else response_ids)]
                n_tokens = len(ids)
                spans = list(step_spans) if step_spans is not None else step_spans_over_ids(ids, self.tokenizer, len(steps))
            else:
                spans = extract_step_token_spans(response, self.tokenizer)
                n_tokens = len(self.tokenizer(response, add_special_tokens=False)["input_ids"])

            token_rewards = torch.zeros(n_tokens, dtype=torch.float32)
            for span, r in zip(spans, per_step_rewards):
                start, end = span
                if end <= 0 or start >= n_tokens:
                    continue
                token_rewards[min(end - 1, n_tokens - 1)] += float(r)

            return {
                "per_step_rewards": per_step_rewards,
                "per_step_records": [],
                "token_rewards": token_rewards,
                "step_spans": spans,
                "predicted_answer": predicted_answer,
                "trajectory_reward": float(sum(per_step_rewards)),
                "trajectory_valid": bool(trajectory_valid),
            }

        # Build per-step text-reward prompts that align with the SFT prompt
        # so the ReaRAG/Llama-head reward model evaluates each step in its
        # actual context.
        text_reward_prompts = []
        msgs = build_sft_messages(
            question=spec.query,
            retrieved_passages=spec.retrieved_passages,
            kg_triples=spec.kg_subgraph,
        )
        rendered_prompt = "\n\n".join(m["content"] for m in msgs)
        for i, _ in enumerate(steps):
            prefix = "\n".join(s.raw_text for s in steps[:i])
            text_reward_prompts.append(rendered_prompt + ("\n\n" + prefix if prefix else ""))

        # P1-1: pass real per-step logprobs through to the α-gate's entropy
        # feature. If none supplied, use None per step (entropy→1.0 fallback,
        # matching the inference path).
        if logprobs_per_step is None:
            logprobs_list: List[Optional[Sequence[float]]] = [None] * len(steps)
        else:
            logprobs_list = list(logprobs_per_step[: len(steps)])
            if len(logprobs_list) < len(steps):
                logprobs_list += [None] * (len(steps) - len(logprobs_list))

        # R9: dynamic KG subgraph — extract entities from the MODEL OUTPUT,
        # not from the static silver-data spec. The old path used spec.kg_subgraph
        # which contains entities relevant to the silver *question*, but the
        # model's *generated text* mentions different entities — so graph_density
        # was always 0 and α was stuck at 0.02.
        dynamic_kg: List[Tuple[str, str, str]] = []
        if self.subgraph_retriever is not None:
            try:
                all_mentions = set()
                for _s in steps:
                    for m in (_s.mentioned_entities or []):
                        if m:
                            all_mentions.add(m)
                # Only mentions that are GROUNDED in the question's own evidence
                # may expand the reward graph (§3.4). Without this, the model can
                # hallucinate an entity, the system fetches that entity's REAL
                # Wikidata subgraph, the model cites it, the PRM marks the
                # citation verified — and wrong reasoning earns positive R_KG.
                grounded = _grounded_mentions(
                    all_mentions, spec.query, spec.retrieved_passages, spec.kg_subgraph
                )
                if grounded:
                    linker = self.composite.prm_annotator.entity_linker
                    _titles = build_passage_titles(spec.retrieved_passages)
                    _ptext = build_passage_text(spec.retrieved_passages)
                    qids = []
                    for m in grounded:
                        # Pass the question + retrieved passage context so the
                        # linker's full context scoring runs (R9 v7). Matches the
                        # inference path so train and eval disambiguate identically.
                        r = linker.link_single(
                            m, question=spec.query,
                            retrieved_titles=_titles,
                            passage_text=_ptext,
                        )
                        if r.selected_qid and not r.abstained:
                            qids.append(r.selected_qid)
                    if qids:
                        dynamic_kg = self.subgraph_retriever.fetch(qids) or []
            except Exception as e:
                logger.warning("Dynamic KG fetch failed: %s", e)

        # Merge the grounded expansion with the question-anchored KG; never let
        # generated entities REPLACE it.
        kg_for_reward = list(set(tuple(t) for t in (list(spec.kg_subgraph) + dynamic_kg)))

        records = self.composite.compute_trajectory_rewards(
            steps=steps,
            kg_subgraph=kg_for_reward,
            text_reward_prompts=text_reward_prompts,
            logprobs_list=logprobs_list,
            predicted_answer=predicted_answer,
            gold_answer=spec.gold_answer,
            alpha_override=self.alpha_override,
            trajectory_valid=trajectory_valid,
        )

        per_step_rewards = [r.r_total for r in records]
        # R7: no per-step format bonus. Format is a CONSTRAINT (enforced via
        # trajectory_valid gating the outcome reward), not a reward target.
        # See problem_and_solutions.md for the rationale.
        # NOTE: per_step_rewards already embeds r_kg (KG citation quality) via
        # composite.compute_step_reward(). Adding PRM annotate_trajectory() on
        # top would double-count the same signal — annotate_trajectory() calls
        # the same label() method that produces r_kg.
        # Map each step's reward to the last token of its [Step N] span.
        # #6: prefer response_ids coordinates so placement aligns with the
        # trainer's scatter; fall back to re-tokenising the decoded response.
        if response_ids is not None:
            ids = [int(x) for x in (response_ids.tolist() if hasattr(response_ids, "tolist") else response_ids)]
            n_tokens = len(ids)
            spans = list(step_spans) if step_spans is not None else step_spans_over_ids(ids, self.tokenizer, len(steps))
        else:
            spans = extract_step_token_spans(response, self.tokenizer)
            n_tokens = len(self.tokenizer(response, add_special_tokens=False)["input_ids"])

        token_rewards = torch.zeros(n_tokens, dtype=torch.float32)
        for span, r in zip(spans, per_step_rewards):
            start, end = span
            if end <= 0 or start >= n_tokens:
                continue
            token_rewards[min(end - 1, n_tokens - 1)] += float(r)

        # R7: outcome fallback is now also gated on trajectory_valid.
        # When the policy emits a bare correct answer with no [Step N] markers,
        # it does NOT receive the outcome reward — the "grand prize" requires
        # a well-structured reasoning trace.  (The #6b fallback was originally
        # added to prevent zero task signal early in training; R7 replaces it
        # with the combination of valid-trajectory gating + SFT anchor.)
        outcome_fallback = 0.0
        if (
            trajectory_valid
            and not records
            and predicted_answer
            and spec.gold_answer
            and n_tokens > 0
        ):
            outcome_fallback = float(self.composite._em(predicted_answer, spec.gold_answer))
            if outcome_fallback:
                token_rewards[n_tokens - 1] += self.composite.outcome_weight * outcome_fallback

        return {
            "per_step_rewards": per_step_rewards,
            "per_step_records": records,
            "token_rewards": token_rewards,
            "step_spans": spans,
            "predicted_answer": predicted_answer,
            "trajectory_reward": float(sum(per_step_rewards) + outcome_fallback),
            "trajectory_valid": bool(trajectory_valid),
        }
