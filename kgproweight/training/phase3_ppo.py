"""Phase 3b — PPO + GAE + Critic + Reference Model (default on Pro 6000 96 GB).

Fixes bugs #8 and #9: the legacy script used a single scalar reward per
trajectory and never invoked the critic head. Here we run a full TRL
``PPOTrainer`` with:

- a frozen reference model (the Phase 3a SFT checkpoint),
- a policy model (also initialised from SFT, then PEFT-LoRA-tuned),
- a value head attached to the policy (TRL provides it in
  ``AutoModelForCausalLMWithValueHead``),
- a per-step composite reward function via
  :class:`kgproweight.training.reward_function.KGProWeightRewardFunction`,
- per-token reward shaping that places the step reward on the last token
  of the corresponding ``[Step N]`` span.

We also support an ``alpha_override`` for the alpha-ablations
(``α=0`` / ``α=0.5`` / ``α=1``) by passing the value straight to the
reward function.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from kgproweight.data.parsers import parse_steps
from kgproweight.data.prompts import build_rl_messages, build_sft_messages
from kgproweight.data.silver_dataset import SilverDatasetReader
from kgproweight.kg.wikidata_retriever import WikidataSubgraphRetriever
from kgproweight.reward.alpha_gate import AlphaGate
from kgproweight.reward.prm_annotator import PRMAnnotator
from kgproweight.reward.text_reward_model import build_text_reward_model
from kgproweight.training.reward_function import (
    KGProWeightRewardFunction,
    RewardSpec,
    step_spans_over_ids,
)
from kgproweight.training.step_reward_ppo_trainer import StepRewardPPOTrainer
from kgproweight.utils.logging import dump_manifest, get_logger
from kgproweight.utils.paths import index_dir, model_path
from kgproweight.utils.seed import set_seed

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Phase3PPOConfig:
    silver_path: str
    output_dir: str
    base_model: str = "llama3-8B-instruct"
    sft_checkpoint: Optional[str] = None
    alpha_gate_path: Optional[str] = None
    text_reward_backend: str = "auto"  # rearag | llama_head | auto | dummy
    text_reward_fallback_path: Optional[str] = None
    dtype: str = "bf16"
    seed: int = 42

    # PPO
    learning_rate: float = 1.0e-5
    batch_size: int = 64
    mini_batch_size: int = 8
    ppo_epochs: int = 4
    cliprange: float = 0.2
    cliprange_value: float = 0.2  # value-function clip range (TRL default 0.2)
    # KL anchor: 0.1 is the midpoint. 0.2 locked the policy to SFT
    # (only rephrasing, not changing answers); 0.05 was too loose (policy
    # dropped [Step N] structure to chase reward). Combined with
    # step_format_bonus, 0.1 gives room to change answers safely.
    kl_coef: float = 0.1
    gamma: float = 0.95
    lam: float = 0.95
    max_grad_norm: float = 1.0
    total_steps: int = 5000
    vf_coef: float = 0.5
    # Adaptive-controller target KL (TRL's `target`), NOT the early-stop knob.
    target_kl: float = 8.0
    # Horizon for the adaptive KL controller.
    kl_horizon: float = 2000.0
    early_stopping: bool = False
    # EM bonus weight: when the predicted answer matches the gold AND the
    # trajectory is valid, the final step gets +outcome_weight.
    # R7: outcome is now CONDITIONAL on trajectory validity — no more
    # unconditional answer reward (see problem_and_solutions.md).
    outcome_weight: float = 10.0  # R7: strong conditional EM signal
    text_reward_scale: float = 0.3  # R6: scale down ReaRAG text reward
    pure_em_reward: bool = False
    # R7: minimum number of parsed [Step N] blocks for trajectory validity.
    # Trajectories with fewer steps cannot receive the outcome reward.
    min_valid_steps: int = 3
    # R8: minimum characters of actual reasoning content per step. Empty
    # "Reasoning:\n" blocks are treated as invalid (content-aware gate).
    min_reasoning_chars: int = 20
    # R9: scale up per-step composite reward to cover KL token cost.
    # With KG online, max R_step ≈ 0.8; KL cost ≈ 5 per 100 tokens.
    # Scale ×5 brings max R_step to ~4.0 so multi-step reasoning is net positive.
    step_reward_scale: float = 5.0
    # R8: SFT Replay ratio — fraction of PPO prompts (0.0–1.0) sourced from
    # the SFT anchor data instead of the silver retrieval pool. Each SFT-replay
    # prompt includes the gold trajectory as the assistant response, so the
    # model periodically sees well-formatted examples during rollouts.
    sft_replay_ratio: float = 0.15
    # R7: SFT Anchor — small weight for cross-entropy loss on silver trajectories
    # to prevent policy drift away from the [Step N] reasoning format.
    sft_anchor_weight: float = 0.02  # λ: small, for format prior only
    # R7: run one SFT anchor step every N PPO steps.
    sft_anchor_interval: int = 50
    log_with: Optional[str] = None
    # Save a recoverable adapter checkpoint every N trajectories seen (0 = only
    # at the end). Lets a collapsed run roll back to the last healthy step.
    save_every_steps: int = 256

    # Generation
    max_new_tokens: int = 256
    # PPO ROLLOUT SAMPLING MUST MATCH TRL's logprob recomputation distribution.
    # TRL's batched_forward_pass scores responses from RAW logits — i.e.
    # temperature=1.0, no top_p, no top_k. If we sample at temperature=0.7 /
    # top_p=0.9 (a sharper, truncated distribution) the recomputed logp_old is
    # not the distribution the tokens were actually drawn from, so the PPO ratio
    # baseline is wrong and the KL estimate drifts NEGATIVE (the 2026-06-23
    # symptom: KL swung 60→0.34→-20). Keep these at the no-op values and disable
    # top_k in the generate() call so rollout == scoring distribution.
    temperature: float = 1.0
    top_p: float = 1.0
    max_input_length: int = 4096
    # R9 v3: cap parsed [Step N] blocks per rollout (reduced to save VRAM).
    max_steps: int = 5
    # B3: the prompt template orders Question → Passages → KG, and the tokenizer
    # right-truncates. With the package defaults (50 passages) the KG block — the
    # whole point of KG-grounding — gets truncated away before the model sees it
    # during rollouts. Cap passages so the KG always fits.
    # Unified with SFT/eval (DEFAULT_TOPK=15) so PPO rolls out in the SAME
    # passage context the policy is later evaluated in (no train/inference
    # mismatch). 15 passages + KG fit max_input_length=6144; on a 96GB card the
    # policy (8B) + frozen SFT ref + ReaRAG-9B reward co-reside (~50GB) with room
    # for activations at batch_size=8 / mini_batch_size=1 + gradient checkpointing.
    # If OOM: drop batch_size 8→4, then ppo_max_kg_triples 30→20, then passages.
    ppo_max_passages: int = 15
    ppo_max_kg_triples: int = 30

    # LoRA on policy
    use_lora: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05

    # Ablation hooks
    alpha_override: Optional[float] = None
    binary_labels_only: bool = False
    # P1-1: use real per-step token logprobs for the α-gate's entropy feature.
    use_real_logprobs: bool = True

    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_models(cfg: Phase3PPOConfig):
    """Build policy+value, reference, and tokenizer.

    BUGFIX (2026-06-22): the previous version passed ``base_id`` = SFT-adapter
    dir together with a fresh ``peft_config``. TRL's ``from_pretrained`` ignores
    ``peft_config`` when an ``adapter_config.json`` is present and loads the
    trained adapter with ``is_trainable=False`` (the default) — so the SFT LoRA
    was loaded FROZEN, PPO produced zero gradient on it, and the saved checkpoint
    was byte-identical to SFT. We now load the SFT adapter as TRAINABLE and anchor
    the KL reference to a frozen SFT copy (not the bare base, which is what
    adapter-disabling would give and which would penalise SFT-acquired behaviour).
    """
    import os
    import torch as _torch
    from transformers import AutoTokenizer
    from trl import AutoModelForCausalLMWithValueHead, create_reference_model

    dtype_map = {"bf16": _torch.bfloat16, "fp16": _torch.float16, "fp32": _torch.float32}
    torch_dtype = dtype_map.get(cfg.dtype, _torch.bfloat16)
    base_id = cfg.sft_checkpoint or model_path(cfg.base_model)

    tokenizer = AutoTokenizer.from_pretrained(base_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    has_adapter = os.path.exists(os.path.join(str(base_id), "adapter_config.json"))
    policy_kwargs: Dict[str, Any] = {"torch_dtype": torch_dtype}
    is_peft = False

    if has_adapter:
        # base_id is a trained PEFT adapter dir (the SFT student). TRL will load
        # base + this adapter; force is_trainable=True so PPO can update it and
        # save_pretrained writes the UPDATED weights (SFT+PPO in one adapter).
        policy_kwargs["is_trainable"] = True
        is_peft = True
    elif cfg.use_lora:
        # Fresh start from the bare base model: attach a new trainable LoRA.
        try:
            from peft import LoraConfig

            policy_kwargs["peft_config"] = LoraConfig(
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                bias="none",
                task_type="CAUSAL_LM",
            )
            is_peft = True
        except ImportError:
            logger.warning("peft not installed; PPO will fine-tune all parameters.")

    policy = AutoModelForCausalLMWithValueHead.from_pretrained(base_id, **policy_kwargs)

    # Sanity: confirm at least one LoRA parameter is actually trainable, so a
    # frozen-adapter regression can never silently return (the original bug).
    if is_peft:
        n_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        logger.info("Policy trainable params: %d", n_trainable)
        if n_trainable == 0:
            raise RuntimeError(
                "Policy has 0 trainable parameters — the LoRA adapter loaded frozen. "
                "PPO would be a no-op (this was the 2026-06-22 bug). Aborting."
            )

    # Activation memory: enable gradient checkpointing on the POLICY so the 8B
    # policy + frozen SFT reference + ReaRAG-9B reward co-reside on one 96GB card
    # (the 2026-06-22 run sat at ~93/96GB without it — one long rollout from OOM).
    # The value-head wrapper proxies to .pretrained_model; checkpoint that. With
    # LoRA the base is frozen, so enable_input_require_grads() is REQUIRED — else
    # the checkpointed segments have no grad-requiring input and the LoRA grad
    # never flows (silent no-learn). We set config.use_cache=False for the
    # training forward/backward (KV-cache is incompatible with checkpointing);
    # the rollout generate() below passes use_cache=True explicitly to override
    # this, so generation stays fast (it runs under no_grad — no activation cost).
    inner = getattr(policy, "pretrained_model", policy)
    if hasattr(inner, "gradient_checkpointing_enable"):
        inner.gradient_checkpointing_enable()
        if hasattr(inner, "enable_input_require_grads"):
            inner.enable_input_require_grads()
        if hasattr(inner, "config"):
            inner.config.use_cache = False
        logger.info("Gradient checkpointing enabled on policy (use_cache=False).")

    # KL reference = frozen snapshot of the SFT-initialised policy. We always
    # build an explicit reference (deepcopy, ~+1 model of VRAM) rather than
    # ref_model=None: adapter-disabling would anchor KL to the BARE BASE, pulling
    # the policy away from SFT. Anchoring to SFT is the standard PPO setup.
    ref_model = create_reference_model(policy)
    return policy, ref_model, tokenizer


def _step_logprobs_from_scores(
    response_ids: torch.Tensor,
    scores: Sequence[torch.Tensor],
    spans,
) -> List[Optional[List[float]]]:
    """Slice per-step token logprobs from generation ``scores`` by token span.

    ``scores`` is the tuple from ``generate(output_scores=True)``: one
    ``(1, vocab)`` logit tensor per *generated* token. We convert each to the
    logprob of the actually-sampled token, then bucket those into the step
    spans (P1-1 feeds these to the α-gate's entropy feature).
    """
    if not scores:
        return [None] * len(spans)
    # logprob of the sampled token at each generated position.
    tok_logprobs: List[float] = []
    n_gen = min(len(scores), response_ids.size(0))
    for t in range(n_gen):
        logits = scores[t][0]
        lp = torch.log_softmax(logits.float(), dim=-1)
        tok_id = int(response_ids[t].item())
        tok_logprobs.append(float(lp[tok_id].item()))
    out: List[Optional[List[float]]] = []
    for start, end in spans:
        s = max(0, start)
        e = min(end, len(tok_logprobs))
        out.append(tok_logprobs[s:e] if e > s else None)
    return out


def _prepare_prompts(reader: SilverDatasetReader, tokenizer, cfg: Phase3PPOConfig,
                     question_kg_index: dict = None):
    """Build PPO prompts.  When ``question_kg_index`` is provided, the Knowledge
    Graph block is populated from the pre-built Q→KG lookup (instant, 100% hit).
    """
    rows = []
    skipped_no_gold = 0
    dyn_kg_hits = 0
    for traj in reader.accepted():
        kg_triples = list(traj.kg_subgraph)
        # R9: instant Q→KG lookup from pre-built index (0.2s, 100% coverage)
        if question_kg_index is not None:
            dyn = question_kg_index.get(traj.question)
            if dyn:
                dyn_kg_hits += 1
                kg_triples = list(dyn)

        msgs = build_rl_messages(
            question=traj.question,
            retrieved_passages=traj.retrieved_passages,
            kg_triples=kg_triples,
            top_k=cfg.ppo_max_passages,
            max_kg_triples=cfg.ppo_max_kg_triples,
        )
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        else:
            text = "\n\n".join(m["content"] for m in msgs)
        gold = str(traj.metadata.get("gold_answer") or "").strip()
        if not gold:
            skipped_no_gold += 1
            continue
        spec = RewardSpec(
            query=traj.question,
            gold_answer=gold,
            kg_subgraph=kg_triples,  # R9: may be dynamic, may be silver fallback
            retrieved_passages=list(traj.retrieved_passages),
            metadata={"qid": traj.qid, "dataset": traj.dataset},
        )
        rows.append({"prompt": text, "spec": spec})
    if skipped_no_gold:
        logger.warning(
            "Skipped %d accepted trajectories with no gold_answer (A4: no teacher fallback).",
            skipped_no_gold,
        )
    if dyn_kg_hits > 0:
        logger.info("R9 prompt KG: %d/%d samples got subgraphs from pre-built index", dyn_kg_hits, len(rows))
        # Print one example: show KG block from the first dynamic-KG prompt
        for row in rows:
            prompt = row["prompt"]
            if "(empty)" not in prompt and "Knowledge Graph:" in prompt:
                kg_start = prompt.index("Knowledge Graph:")
                kg_end = prompt.index("\n\n", kg_start) if "\n\n" in prompt[kg_start:] else len(prompt)
                logger.info("R9 KG example:\n%s", prompt[kg_start:kg_end])
                break
    return rows


def _prepare_sft_anchor_data(
    silver_path: str,
    tokenizer,
    cfg: Phase3PPOConfig,
    max_samples: int = 2000,
) -> List[Dict[str, Any]]:
    """R7: Build tokenised SFT samples for the format-preservation anchor.

    Each sample is a ``(prompt, full_trajectory)`` pair tokenised as a single
    sequence with the prompt portion masked in the labels.  The anchor loss is
    cross-entropy on the trajectory tokens only — it nudges the policy to
    maintain the ``[Step N] ... [Final Answer]`` output format without forcing
    specific answers.

    We create a fresh reader (the PPO reader is already consumed by
    ``_prepare_prompts``) and accept ALL trajectories, including those without
    a gold answer — the anchor only cares about the format, not correctness.
    """
    reader2 = SilverDatasetReader(silver_path)
    sft_samples: List[Dict[str, Any]] = []
    for traj in reader2.accepted():
        answer_text = (traj.answer or "").strip()
        if not answer_text:
            continue
        # Build the SAME prompt format as RL so the anchor sees the identical
        # context distribution.
        msgs = build_sft_messages(
            question=traj.question,
            retrieved_passages=traj.retrieved_passages,
            kg_triples=traj.kg_subgraph,
            top_k=cfg.ppo_max_passages,
            max_kg_triples=cfg.ppo_max_kg_triples,
        )
        if hasattr(tokenizer, "apply_chat_template"):
            prompt_text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )
        else:
            prompt_text = "\n\n".join(m["content"] for m in msgs)

        full_text = prompt_text + answer_text
        max_total = cfg.max_input_length + cfg.max_new_tokens

        # Tokenise the prompt alone to find the mask boundary.
        prompt_ids = tokenizer(
            prompt_text, truncation=True, max_length=cfg.max_input_length,
        )["input_ids"]
        # Tokenise the full sequence.
        full_enc = tokenizer(
            full_text, truncation=True, max_length=max_total,
        )
        full_ids = full_enc["input_ids"]

        # Labels = full_ids, but mask the prompt portion so the loss is
        # computed only on the trajectory tokens (the format prior).
        labels = [-100] * min(len(prompt_ids), len(full_ids))
        labels += full_ids[len(labels):]
        # Defensively align lengths.
        if len(labels) < len(full_ids):
            labels += full_ids[len(labels):]
        elif len(labels) > len(full_ids):
            labels = labels[: len(full_ids)]

        sft_samples.append({"input_ids": full_ids, "labels": labels})
        if len(sft_samples) >= max_samples:
            break

    return sft_samples


def _generate(policy, tokenizer, prompts: Sequence[str], cfg: Phase3PPOConfig, device: str):
    """Generate one response per prompt.

    SCALE: when ``use_real_logprobs`` is on we convert each prompt's generation
    ``scores`` to small per-step logprob lists *immediately* inside the loop and
    discard the raw logit tensors, so the whole batch's logits (≈GBs at
    batch_size=64) are never held in memory at once.
    """
    query_tensors, response_tensors, response_texts, logprobs_per_step_list = [], [], [], []
    # Read the device LIVE from the model: PPOTrainer/accelerate move the policy
    # to CUDA *after* run_phase3_ppo computed its `device` string, so the passed
    # `device` can be a stale "cpu". Trust where the params actually are now.
    try:
        device = next(p for p in policy.parameters()).device
    except StopIteration:
        pass
    for prompt in prompts:
        # B3: check length BEFORE truncation so a prompt that would lose its KG
        # block is a loud warning, not a silent grounding failure. With passages
        # capped in _prepare_prompts this should not trigger; if it does, lower
        # cfg.ppo_max_passages / ppo_max_kg_triples.
        full_len = len(tokenizer(prompt, truncation=False)["input_ids"])
        if full_len > cfg.max_input_length:
            logger.warning(
                "PPO prompt is %d tokens > max_input_length=%d; right-truncation will drop the "
                "trailing KG block. Lower ppo_max_passages (now %d) / ppo_max_kg_triples (now %d).",
                full_len, cfg.max_input_length, cfg.ppo_max_passages, cfg.ppo_max_kg_triples,
            )
        enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=cfg.max_input_length,
        )
        query_ids = enc["input_ids"].to(device).squeeze(0)
        with torch.no_grad():
            out = policy.generate(
                input_ids=query_ids.unsqueeze(0),
                max_new_tokens=cfg.max_new_tokens,
                do_sample=True,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                # top_k=0 disables the default top-50 truncation: any truncation
                # (top_p<1 or top_k>0) makes the rollout distribution differ from
                # TRL's raw-logit logp recomputation and pushes KL negative.
                top_k=0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                # Override the config.use_cache=False set for gradient
                # checkpointing: rollout runs under no_grad, so KV-cache is free
                # memory-wise and ~Nx faster than recomputing every step.
                use_cache=True,
                return_dict_in_generate=cfg.use_real_logprobs,
                output_scores=cfg.use_real_logprobs,
            )
        if cfg.use_real_logprobs:
            gen = out.sequences[0]
            response_ids = gen[query_ids.size(0):]
            # SCALE: collapse this prompt's raw scores into the small per-step
            # logprob lists RIGHT HERE, then drop the logits before the next
            # prompt — never accumulate raw scores across the batch.
            resp_text = tokenizer.decode(response_ids, skip_special_tokens=True)
            n_parsed = len(parse_steps(resp_text)[: cfg.max_steps])
            spans = step_spans_over_ids(response_ids, tokenizer, n_parsed)
            logprobs_per_step = _step_logprobs_from_scores(response_ids, out.scores, spans)
            del out
        else:
            gen = out[0]
            response_ids = gen[query_ids.size(0):]
            resp_text = tokenizer.decode(response_ids, skip_special_tokens=True)
            logprobs_per_step = None
        query_tensors.append(query_ids)
        response_tensors.append(response_ids)
        response_texts.append(resp_text)
        logprobs_per_step_list.append(logprobs_per_step)
    return query_tensors, response_tensors, response_texts, logprobs_per_step_list


def _count_reasoning_content(response_texts: Sequence[str], min_chars: int = 20) -> Dict[str, Any]:
    """R8: Count how many responses have substantive reasoning content per step."""
    import re

    n_with_steps = 0
    n_with_final_answer = 0
    n_with_reasoning = 0
    total_steps = 0
    steps_with_content = 0

    for text in response_texts:
        if not text:
            continue
        if "[Step" in text or "Step " in text:
            n_with_steps += 1
        if "Final Answer" in text:
            n_with_final_answer += 1

        step_bodies = re.findall(
            r'\[Step \d+\]\s*(.*?)(?=\[Step|\Z)', text, re.DOTALL,
        )
        for body in step_bodies:
            total_steps += 1
            if "Reasoning:" in body:
                after = body.split("Reasoning:", 1)[1]
                reasoning = re.split(
                    r'Knowledge Used:|Conclusion:|Final Answer:', after,
                )[0].strip()
                if len(reasoning) >= min_chars:
                    steps_with_content += 1

    return {
        "n_samples": len(response_texts),
        "n_with_steps": n_with_steps,
        "n_with_final_answer": n_with_final_answer,
        "total_steps": total_steps,
        "steps_with_content": steps_with_content,
        "step_rate": n_with_steps / max(1, len(response_texts)),
        "final_answer_rate": n_with_final_answer / max(1, len(response_texts)),
        "reasoning_content_rate": (
            steps_with_content / max(1, total_steps) if total_steps > 0 else 0.0
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_phase3_ppo(cfg: Phase3PPOConfig) -> Dict[str, Any]:
    set_seed(cfg.seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from trl import PPOConfig

    policy, ref_model, tokenizer = _build_models(cfg)
    # NOTE: at this point the policy still lives on CPU — PPOTrainer/accelerate
    # move it to CUDA later. So don't read the device off the (CPU) policy here;
    # pick the real training device directly. Reward components (text_reward etc.)
    # built below must land on the SAME device the policy ends up on.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Build reward components ----------------------------------------
    alpha_gate = AlphaGate()
    if cfg.alpha_gate_path and Path(cfg.alpha_gate_path).exists():
        alpha_gate.load_state_dict(torch.load(cfg.alpha_gate_path, map_location="cpu"))
        logger.info("Loaded α-gate from %s", cfg.alpha_gate_path)
    alpha_gate.eval()

    # P0-2 / Finding 2: the (improved) PRMAnnotator with a LIVE entity cache so
    # link_confidence (the α-gate's middle feature) matches Phase 2 training. A
    # bare PRMAnnotator() builds an EntityLinker with an EMPTY cache →
    # link_confidence ≡ 0, which would make the gate's middle input dead at
    # inference and mismatch Phase 2.
    from kgproweight.retrieval.bootstrap import resolve_entity_cache_path
    from kgproweight.kg.entity_linker import EntityLinker

    _entity_linker = EntityLinker(cache_path=resolve_entity_cache_path(), offline=True)
    logger.info(
        "PPO link_confidence: EntityLinker cache=%s (%d entries)",
        resolve_entity_cache_path(), len(list(_entity_linker.cache.items())),
    )
    annotator = PRMAnnotator(entity_linker=_entity_linker, verbose=False)
    text_reward = build_text_reward_model(
        backend=cfg.text_reward_backend,
        fallback_head_path=cfg.text_reward_fallback_path,
        device=str(device),
        dtype=cfg.dtype,
    )
    reward_fn = KGProWeightRewardFunction(
        alpha_gate=alpha_gate,
        prm_annotator=annotator,
        text_reward_model=text_reward,
        tokenizer=tokenizer,
        outcome_weight=cfg.outcome_weight,
        discount=cfg.gamma,
        alpha_override=cfg.alpha_override,
        max_steps=cfg.max_steps,
        text_reward_scale=cfg.text_reward_scale,
        min_valid_steps=cfg.min_valid_steps,
        min_reasoning_chars=cfg.min_reasoning_chars,
        step_reward_scale=cfg.step_reward_scale,
        subgraph_retriever=WikidataSubgraphRetriever(max_hops=2, max_neighbors=30, offline=True),
        pure_em=cfg.pure_em_reward,
    )

    # ---- Data ------------------------------------------------------------
    reader = SilverDatasetReader(cfg.silver_path)
    if cfg.binary_labels_only:
        for traj in reader.trajectories:
            for step in traj.steps:
                if step.label == 0:
                    step.label = -1  # collapse neutral into negative for this ablation

    # R9: dynamic prompt KG disabled for speed (9839 prompts).
    # EntityLinker needed for reward-side dynamic KG (below).
    # R9: prompt KG uses silver data (instant). Reward-side dynamic KG provides
    # the KG signal (already works — α=0.85, r_kg broke zero). Prompt-side
    # injection needs pre-built entity index for speed; TODO separately.
    # R9 v6: pre-built Q→KG index with filtered & ranked triples.
    _q_kg_index = {}
    _q_kg_path = Path(index_dir()) / "kg_cache" / "question_kg_index_v2.json"
    if not _q_kg_path.exists():
        _q_kg_path = Path(index_dir()) / "kg_cache" / "question_kg_index.json"
    if _q_kg_path.exists():
        import json as _json
        _q_kg_raw = _json.loads(_q_kg_path.read_text(encoding="utf-8"))
        is_v2 = "builder_version" in (_q_kg_raw[0] if _q_kg_raw else {})
        for _entry in _q_kg_raw:
            _q = _entry.get("question", _entry.get("q", ""))
            if is_v2:
                # v2 rich format: triples is list of dicts {h, pid, r, t, score}
                _q_kg_index[_q] = [(t["h"], t["r"], t["t"]) for t in _entry["triples"]]
            else:
                # v1 format: t is list of lists [h, r, t]
                _q_kg_index[_q] = _entry["t"]
        logger.info("R9 v6: Loaded %d question→KG entries from %s (v%s)",
                    len(_q_kg_index), _q_kg_path, "2" if is_v2 else "1")
    samples = _prepare_prompts(reader, tokenizer, cfg, question_kg_index=_q_kg_index)
    if not samples:
        raise ValueError(f"No PPO samples derived from {cfg.silver_path}")

    # R7: prepare SFT anchor data from silver trajectories for format
    # preservation. We use accepted trajectories (including those without
    # gold answers — the anchor only cares about output format).
    sft_anchor_data: List[Dict[str, Any]] = []
    if cfg.sft_anchor_weight > 0:
        sft_anchor_data = _prepare_sft_anchor_data(
            silver_path=cfg.silver_path,
            tokenizer=tokenizer,
            cfg=cfg,
            max_samples=2000,
        )
        logger.info(
            "R7 SFT anchor: %d samples prepared (weight=%.3f, interval=%d)",
            len(sft_anchor_data), cfg.sft_anchor_weight, cfg.sft_anchor_interval,
        )

    # R8: prepare SFT replay prompts — prompts that INCLUDE the well-formatted
    # gold trajectory in the context, so the model periodically sees correct
    # format examples during PPO rollouts. Unlike SFT anchor (separate backward
    # pass), replay is mixed directly into the PPO prompt pool.
    sft_replay_prompts: List[str] = []
    if cfg.sft_replay_ratio > 0 and sft_anchor_data:
        for item in sft_anchor_data[:max(1, int(len(samples) * cfg.sft_replay_ratio * 2))]:
            # Reconstruct: input_ids = prompt + trajectory. Decode to get the
            # full text, then strip the trajectory portion to get the prompt-only.
            # We want the prompt WITH the gold trajectory as context for the model.
            full_text = tokenizer.decode(item["input_ids"], skip_special_tokens=False)
            sft_replay_prompts.append(full_text)
        logger.info(
            "R8 SFT replay: %d prompts prepared (ratio=%.2f, mixed into PPO batches)",
            len(sft_replay_prompts), cfg.sft_replay_ratio,
        )
    # SCALE: total_steps counts trajectories SEEN (n_seen += batch_size per
    # iteration), not epochs. For one full pass over the data it should be
    # >= ceil(len(samples)/batch_size) * batch_size. We do NOT change the
    # semantics here; just report so under-coverage is visible.
    full_coverage_steps = math.ceil(len(samples) / max(1, cfg.batch_size)) * cfg.batch_size
    logger.info(
        "Phase 3b PPO with %d prompts; batch_size=%d, total_steps=%d "
        "(>= %d needed for one full pass over the data).",
        len(samples), cfg.batch_size, cfg.total_steps, full_coverage_steps,
    )

    # ---- Trainer ---------------------------------------------------------
    ppo_cfg = PPOConfig(
        learning_rate=cfg.learning_rate,
        batch_size=cfg.batch_size,
        mini_batch_size=cfg.mini_batch_size,
        ppo_epochs=cfg.ppo_epochs,
        cliprange=cfg.cliprange,
        cliprange_value=cfg.cliprange_value,
        kl_penalty="kl",
        # KL control. init_kl_coef is the initial penalty coefficient for the
        # adaptive controller; cfg.target_kl is that controller's TARGET KL —
        # TRL's `target`, NOT TRL's `target_kl` (the latter is an early-stop
        # threshold that only fires when early_stopping=True). The previous code
        # wired cfg.target_kl into the early-stop knob (inert here) and left the
        # adaptive `target` at its default. Route it to the correct knob and make
        # adaptive control explicit.
        adap_kl_ctrl=True,
        init_kl_coef=cfg.kl_coef,
        target=cfg.target_kl,
        horizon=cfg.kl_horizon,
        gamma=cfg.gamma,
        lam=cfg.lam,
        max_grad_norm=cfg.max_grad_norm,
        vf_coef=cfg.vf_coef,
        early_stopping=cfg.early_stopping,
        log_with=None,  # R7: use custom tb_writer, not TRL's tracker
        seed=cfg.seed,
    )
    trainer = StepRewardPPOTrainer(
        config=ppo_cfg,
        model=policy,
        ref_model=ref_model,
        tokenizer=tokenizer,
    )

    # R7: TensorBoard writer for custom metrics not covered by TRL's built-in
    # logging (valid_rate, sft_anchor_loss, trajectory_reward distribution).
    # TRL handles PPO losses/KL/clipfrac when log_with="tensorboard".
    tb_writer = None
    if cfg.log_with == "tensorboard":
        from torch.utils.tensorboard import SummaryWriter

        tb_log_dir = Path("/root/tf-logs")
        tb_log_dir.mkdir(parents=True, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=str(tb_log_dir))
        logger.info("TensorBoard logging to %s", tb_log_dir)

    # ---- Loop ------------------------------------------------------------
    rng = torch.Generator().manual_seed(cfg.seed)
    n_seen = 0
    history: List[Dict[str, float]] = []
    while n_seen < cfg.total_steps:
        # R9: SFT replay WITH REAL SPECS.  For sft_replay_ratio of the batch,
        # substitute prompts with SFT replay (prompts that include the gold
        # trajectory).  Unlike the old code that used dummy specs, we pair
        # each replay prompt with a real silver-data spec so the reward
        # distribution is consistent with exploration samples — no more
        # advantage contamination.
        n_replay = (
            int(cfg.batch_size * cfg.sft_replay_ratio)
            if sft_replay_prompts else 0
        )
        n_explore = cfg.batch_size - n_replay

        batch_samples: List[Dict[str, Any]] = []
        prompts: List[str] = []
        specs: List[RewardSpec] = []

        if n_explore > 0:
            explore_idx = torch.randint(0, len(samples), (n_explore,), generator=rng).tolist()
            for i in explore_idx:
                batch_samples.append(samples[i])
                prompts.append(samples[i]["prompt"])
                specs.append(samples[i]["spec"])

        if n_replay > 0:
            replay_idx = torch.randint(0, len(sft_replay_prompts), (n_replay,), generator=rng).tolist()
            # Pair each replay prompt with a real silver spec so reward
            # distribution matches exploration samples.
            spec_idx = torch.randint(0, len(samples), (n_replay,), generator=rng).tolist()
            for ri, si in zip(replay_idx, spec_idx):
                batch_samples.append({"prompt": samples[si]["prompt"]})
                prompts.append(sft_replay_prompts[ri])
                specs.append(samples[si]["spec"])

        query_tensors, response_tensors, response_texts, logprobs_per_step_list = _generate(
            policy, tokenizer, prompts, cfg, device
        )

        token_reward_list: List[torch.Tensor] = []
        traj_rewards: List[float] = []
        all_per_step_records = []  # R9: collect from all responses
        for resp_text, response_ids, spec, logprobs_per_step in zip(
            response_texts, response_tensors, specs, logprobs_per_step_list
        ):
            # #6: compute step spans in response_ids coordinates ONCE; the reward
            # fn places per-step rewards on those same spans, so the placement
            # matches the trainer's scatter (no decode∘re-tokenise drift). The
            # entropy logprobs were already bucketed onto these spans in
            # _generate (SCALE: done per-prompt to avoid holding raw logits).
            n_parsed = len(parse_steps(resp_text)[: cfg.max_steps])
            aligned_spans = step_spans_over_ids(response_ids, tokenizer, n_parsed)
            info = reward_fn(
                prompt="",
                response=resp_text,
                spec=spec,
                logprobs_per_step=logprobs_per_step,
                response_ids=response_ids,
                step_spans=aligned_spans,
            )
            traj_rewards.append(info["trajectory_reward"])
            all_per_step_records.extend(info.get("per_step_records", []))
            # token_rewards is already built in response_ids space (#6), so it
            # matches the response tensor length exactly — but stay defensive.
            tr = info["token_rewards"]
            n = response_ids.size(0)
            if tr.size(0) != n:
                tr = (torch.cat([tr, torch.zeros(n - tr.size(0), dtype=tr.dtype)])
                      if tr.size(0) < n else tr[:n])
            token_reward_list.append(tr)

        # ── R9: extract reward components from all responses for diagnostics ──
        reward_rc = {"alpha_mean": 0.0, "r_kg_mean": 0.0, "r_text_mean": 0.0, "r_total_mean": 0.0, "n_steps": 0}
        if all_per_step_records:
            reward_rc["alpha_mean"] = float(sum(r.alpha for r in all_per_step_records) / len(all_per_step_records))
            reward_rc["r_kg_mean"] = float(sum(r.r_kg for r in all_per_step_records) / len(all_per_step_records))
            reward_rc["r_text_mean"] = float(sum(r.r_text for r in all_per_step_records) / len(all_per_step_records))
            reward_rc["r_total_mean"] = float(sum(r.r_total for r in all_per_step_records) / len(all_per_step_records))
            reward_rc["n_steps"] = len(all_per_step_records)

        # P0-1 / #6: hand the per-token step rewards to the trainer so GAE runs
        placeholder_scores = [torch.zeros((), dtype=torch.float32) for _ in token_reward_list]
        trainer.set_pending_step_rewards(token_reward_list)
        stats = trainer.step(query_tensors, response_tensors, placeholder_scores)
        n_seen += cfg.batch_size

        # ── R7 SFT Anchor step ──
        # After every sft_anchor_interval PPO steps, add a lightweight SFT
        # gradient to preserve the [Step N] ... [Final Answer] output format.
        # This is Plan B: alternating optimisation — the SFT anchor runs as a
        # separate forward+backward+step, equivalent to adding λ·L_sft to the
        # PPO objective with λ ≈ sft_anchor_weight / sft_anchor_interval.
        sft_loss_val = 0.0
        if (
            cfg.sft_anchor_weight > 0
            and sft_anchor_data
            and n_seen % (cfg.batch_size * cfg.sft_anchor_interval) == 0
        ):
            inner = getattr(policy, "pretrained_model", policy)
            # Read live device (trainer may have moved the policy).
            try:
                _dev = next(p for p in policy.parameters()).device
            except StopIteration:
                _dev = torch.device(device)

            # Sample one SFT anchor item (mini_batch_size=1 — simple, no
            # padding needed, and the anchor weight is already small).
            sft_idx = torch.randint(
                0, len(sft_anchor_data), (1,), generator=rng,
            ).item()
            item = sft_anchor_data[sft_idx]
            sft_input_ids = torch.tensor(
                [item["input_ids"]], dtype=torch.long, device=_dev,
            )
            sft_labels = torch.tensor(
                [item["labels"]], dtype=torch.long, device=_dev,
            )
            sft_out = inner(input_ids=sft_input_ids, labels=sft_labels)
            sft_loss_val = float(sft_out.loss.item())
            weighted = cfg.sft_anchor_weight * sft_out.loss
            weighted.backward()
            trainer.optimizer.step()
            trainer.optimizer.zero_grad()
            logger.info(
                "SFT anchor step=%d loss=%.4f weighted=%.4f",
                n_seen, sft_loss_val, float(weighted.item()),
            )

        # Collect trajectory validity stats for monitoring (R8: content-aware).
        n_valid = sum(
            1 for info_text in response_texts
            if reward_fn._is_valid_trajectory(
                parse_steps(info_text)[: cfg.max_steps],
                info_text,
                min_steps=cfg.min_valid_steps,
                min_reasoning_chars=cfg.min_reasoning_chars,
            )
        )

        # R9: advantage variance directly from our custom trainer's last batch.
        adv_var = getattr(trainer, "_last_adv_var", 0.0)

        def _stat(key):
            """Pull a TRL stat as a python float (handles numpy scalars/arrays)."""
            v = stats.get(key)
            if v is None:
                return None
            try:
                import numpy as _np
                return float(_np.asarray(v).mean())
            except Exception:
                return None

        history.append(
            {
                "step": n_seen,
                "mean_reward": float(sum(traj_rewards) / max(1, len(traj_rewards))),
                "ppo_mean_kl": float(stats.get("objective/kl", 0.0)),
                "advantage_var": adv_var,
                # PPO losses (so loss/clip curves are recoverable from history.jsonl).
                "loss_total": _stat("ppo/loss/total"),
                "loss_policy": _stat("ppo/loss/policy"),
                "loss_value": _stat("ppo/loss/value"),
                "policy_clipfrac": _stat("ppo/policy/clipfrac"),
                "policy_entropy": _stat("ppo/policy/entropy"),
                "policy_approxkl": _stat("ppo/policy/approxkl"),
                # R7: trajectory validity monitoring.
                "n_valid": n_valid,
                "valid_rate": n_valid / max(1, cfg.batch_size),
                "sft_anchor_loss": sft_loss_val,
                # R9: reward component diagnostics
                "alpha_mean": reward_rc["alpha_mean"],
                "r_kg_mean": reward_rc["r_kg_mean"],
                "r_text_mean": reward_rc["r_text_mean"],
                "r_total_mean": reward_rc["r_total_mean"],
                "n_steps_sample": reward_rc["n_steps"],
            }
        )
        if n_seen % (cfg.batch_size * 4) == 0:
            logger.info(
                "step=%d reward=%.2f kl=%.1f clip=%.3f valid=%d/%d α=%.3f r_kg=%.3f r_text=%.3f n_steps=%d",
                n_seen,
                history[-1]["mean_reward"],
                history[-1]["ppo_mean_kl"],
                history[-1]["policy_clipfrac"] if history[-1]["policy_clipfrac"] is not None else float("nan"),
                n_valid,
                cfg.batch_size,
                reward_rc["alpha_mean"],
                reward_rc["r_kg_mean"],
                reward_rc["r_text_mean"],
                reward_rc["n_steps"],
            )
            # R7: dump one sample response for qualitative monitoring.
            if response_texts:
                sample = response_texts[0][:500]
                logger.info("  [sample] %s", sample.replace("\n", "\\n"))

        # ── R8: Periodic reasoning-content sampling ──
        # Every save_every_steps, sample 20 responses and check Reasoning
        # content rate so we can see whether the content gate is working.
        sample_log: Optional[Dict[str, Any]] = None
        if (
            cfg.save_every_steps > 0
            and (n_seen // cfg.save_every_steps) != ((n_seen - cfg.batch_size) // cfg.save_every_steps)
        ):
            sample_log = _count_reasoning_content(
                response_texts, min_chars=cfg.min_reasoning_chars,
            )
            sample_log["step"] = n_seen
            logger.info(
                "R8 sample step=%d: step_rate=%.2f fa_rate=%.2f reasoning_content=%.2f "
                "(%d/%d steps have >=%d chars)",
                n_seen,
                sample_log["step_rate"],
                sample_log["final_answer_rate"],
                sample_log["reasoning_content_rate"],
                sample_log["steps_with_content"],
                sample_log["total_steps"],
                cfg.min_reasoning_chars,
            )
            # Dump samples to disk for offline inspection.
            sample_dir = out_dir / "samples"
            sample_dir.mkdir(parents=True, exist_ok=True)
            sample_path = sample_dir / f"step_{n_seen:05d}.txt"
            with open(sample_path, "w", encoding="utf-8") as sfh:
                sfh.write(f"# Step {n_seen} — Reasoning content sampling\n")
                sfh.write(f"# step_rate={sample_log['step_rate']:.2f} ")
                sfh.write(f"fa_rate={sample_log['final_answer_rate']:.2f} ")
                sfh.write(f"reasoning_content={sample_log['reasoning_content_rate']:.2f}\n\n")
                for i, text in enumerate(response_texts[:20], 1):
                    sfh.write(f"--- Sample {i} ---\n{text}\n\n")
            logger.info("  Saved %d samples → %s", min(20, len(response_texts)), sample_path)

        # ── TensorBoard custom metrics (R7 + R8) ──
        # TRL logs: ppo/loss/total, ppo/loss/policy, ppo/loss/value,
        #           ppo/policy/clipfrac, ppo/policy/approxkl,
        #           ppo/policy/entropy, objective/kl.
        # We add our own signals below.
        if tb_writer is not None:
            global_step = n_seen
            # --- Core training signals ---
            tb_writer.add_scalar("custom/mean_reward", history[-1]["mean_reward"], global_step)
            tb_writer.add_scalar("custom/valid_rate", history[-1]["valid_rate"], global_step)
            tb_writer.add_scalar("custom/n_valid", n_valid, global_step)
            if sft_loss_val > 0:
                tb_writer.add_scalar("custom/sft_anchor_loss", sft_loss_val, global_step)
            tb_writer.add_scalar("custom/advantage_var", adv_var, global_step)
            # --- KL divergence (TRL's objective/kl) ---
            kl_val = history[-1]["ppo_mean_kl"]
            if kl_val is not None:
                tb_writer.add_scalar("custom/kl_divergence", kl_val, global_step)
            # --- Clip fraction ---
            clip_val = history[-1]["policy_clipfrac"]
            if clip_val is not None:
                tb_writer.add_scalar("custom/clip_fraction", clip_val, global_step)
            # --- Policy entropy (diversity signal) ---
            ent_val = history[-1]["policy_entropy"]
            if ent_val is not None:
                tb_writer.add_scalar("custom/policy_entropy", ent_val, global_step)
            # --- Reward distribution ---
            if traj_rewards:
                import numpy as _np
                tb_writer.add_scalar("custom/reward_std", float(_np.std(traj_rewards)), global_step)
                tb_writer.add_scalar("custom/reward_max", float(max(traj_rewards)), global_step)
                tb_writer.add_scalar("custom/reward_min", float(min(traj_rewards)), global_step)
            # --- R8: Reasoning content quality ---
            if sample_log is not None:
                tb_writer.add_scalar("r8/step_rate", sample_log["step_rate"], global_step)
                tb_writer.add_scalar("r8/final_answer_rate", sample_log["final_answer_rate"], global_step)
                tb_writer.add_scalar("r8/reasoning_content_rate", sample_log["reasoning_content_rate"], global_step)
                tb_writer.add_scalar("r8/total_steps", sample_log["total_steps"], global_step)
                tb_writer.add_scalar("r8/steps_with_content", sample_log["steps_with_content"], global_step)
            # ── R9: Reward component diagnostics ──
            tb_writer.add_scalar("reward/alpha_mean", reward_rc["alpha_mean"], global_step)
            tb_writer.add_scalar("reward/r_kg_mean", reward_rc["r_kg_mean"], global_step)
            tb_writer.add_scalar("reward/r_text_mean", reward_rc["r_text_mean"], global_step)
            tb_writer.add_scalar("reward/r_total_mean", reward_rc["r_total_mean"], global_step)
            tb_writer.add_scalar("reward/n_steps", reward_rc["n_steps"], global_step)

        # Intermediate checkpoint: save the (PEFT) adapter whenever n_seen crosses
        # a save_every_steps boundary, so a run that collapses can be rolled back
        # to the last healthy step. The crossing test (rather than `% == 0`) is
        # robust to a batch_size that does not divide save_every_steps. The final
        # save below writes a separate `final/` dir, so there is no double-save.
        if (
            cfg.save_every_steps > 0
            and n_seen < cfg.total_steps
            and (n_seen // cfg.save_every_steps) != ((n_seen - cfg.batch_size) // cfg.save_every_steps)
        ):
            ckpt_dir = out_dir / f"step_{n_seen}"
            trainer.save_pretrained(str(ckpt_dir))
            tokenizer.save_pretrained(ckpt_dir)
            # Persist history incrementally so a killed run keeps its curves and
            # the saved step's metrics are recoverable alongside the weights.
            with open(out_dir / "history.jsonl", "w", encoding="utf-8") as fh:
                for h in history:
                    fh.write(json.dumps(h) + "\n")
            logger.info("Saved intermediate PPO checkpoint at %s (step %d)", ckpt_dir, n_seen)

    final_dir = out_dir / "final"
    trainer.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(final_dir)

    history_path = out_dir / "history.jsonl"
    with open(history_path, "w", encoding="utf-8") as fh:
        for h in history:
            fh.write(json.dumps(h) + "\n")

    dump_manifest(
        out_dir,
        extra={
            "phase": "phase3_ppo",
            "config": asdict(cfg),
            "history_tail": history[-5:],
        },
    )
    if tb_writer is not None:
        tb_writer.close()
        logger.info("TensorBoard writer closed.")

    logger.info("Phase 3b PPO done. Final checkpoint at %s", final_dir)
    return {"output_dir": str(out_dir), "final_checkpoint": str(final_dir)}
