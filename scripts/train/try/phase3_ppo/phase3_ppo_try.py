#!/usr/bin/env python
"""Phase 3b PPO (try variant) — per-step reward into GAE + 4 fixes.

Standalone copy of the package's ``phase3_ppo`` that applies the fixes in
``scripts/train/try/README_ppo.md``. The package's ``kgproweight/`` is left
untouched; this reuses everything except the four changed pieces:

* **P0-1** per-step ``R_total(t)`` actually reaches GAE — via
  :class:`StepRewardPPOTrainer` (overrides ``compute_rewards`` only).
* **P0-2** ``ImprovedPRMAnnotator`` for ``R_KG`` (no filler-+1 / -1 misfires).
* **P0-3** outcome EM compares to the *real* gold (``metadata['gold_answer']``),
  not the teacher's own answer.
* **P1-1** the α-gate entropy feature uses *real* per-step token logprobs from
  the rollout.
* **P1-2** the reference model shares the policy's base via
  ``create_reference_model`` instead of loading a second full 8B.

Run from the project root (the CLI inserts this dir on sys.path)::

    python scripts/train/try/phase3_ppo_try.py \
        --silver scripts/train/try/outputs/silver_try_50b.jsonl \
        --sft_checkpoint <path> --output_dir <dir> [--total_steps 64]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

# Make sibling try-modules importable regardless of CWD.
# Make sibling try-modules importable regardless of CWD/subdir layout.
_TRY_ROOT = Path(__file__).resolve().parent.parent
for _d in (_TRY_ROOT, _TRY_ROOT / "shared", _TRY_ROOT / "phase1_distill",
           _TRY_ROOT / "phase2_prm", _TRY_ROOT / "phase3_sft", _TRY_ROOT / "phase3_ppo"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

# --- reused, unchanged, from the package -----------------------------------
from kgproweight.data.parsers import parse_steps
from kgproweight.data.prompts import build_rl_messages
from kgproweight.data.silver_dataset import SilverDatasetReader
from kgproweight.reward.alpha_gate import AlphaGate
from kgproweight.reward.text_reward_model import build_text_reward_model
from kgproweight.utils.logging import dump_manifest, get_logger
from kgproweight.utils.paths import model_path
from kgproweight.utils.seed import set_seed

# --- changed logic, local to the try variant -------------------------------
from ppo_reward_try import ImprovedRewardFunction, RewardSpec, step_spans_over_ids
from ppo_trainer_try import StepRewardPPOTrainer
from prm_annotator_try import ImprovedPRMAnnotator

logger = get_logger(__name__)


@dataclass
class Phase3PPOTryConfig:
    silver_path: str
    output_dir: str
    base_model: str = "llama3-8B-instruct"
    sft_checkpoint: Optional[str] = None
    alpha_gate_path: Optional[str] = None
    text_reward_backend: str = "auto"
    text_reward_fallback_path: Optional[str] = None
    dtype: str = "bf16"
    seed: int = 42

    # PPO
    learning_rate: float = 1.0e-5
    batch_size: int = 64
    mini_batch_size: int = 8
    ppo_epochs: int = 4
    cliprange: float = 0.2
    kl_coef: float = 0.01
    gamma: float = 0.95
    lam: float = 0.95
    max_grad_norm: float = 1.0
    total_steps: int = 5000
    vf_coef: float = 0.5
    target_kl: float = 6.0
    early_stopping: bool = False

    # Generation
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    max_input_length: int = 4096
    max_steps: int = 7  # cap parsed [Step N] blocks per rollout (matches reward_fn)
    # B3: the prompt template orders Question → Passages → KG, and the tokenizer
    # right-truncates. With the package defaults (50 passages ≈ 10k+ tokens) the
    # KG block — the whole point of KG-grounding — gets truncated away before the
    # model sees it during rollouts. Cap passages so the KG always fits, and (in
    # _generate) left-truncate as a safety net so the question survives too.
    ppo_max_passages: int = 8
    ppo_max_kg_triples: int = 50
    max_steps: int = 7  # cap parsed [Step N] blocks per rollout (matches reward_fn)

    # LoRA on policy
    use_lora: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    # Load the base in 4-bit (QLoRA) so 8B fits on a 24GB card (paper §4.3.3).
    use_4bit: bool = False

    # Ablation hooks (paper §7)
    alpha_override: Optional[float] = None
    binary_labels_only: bool = False
    # P1-1: use real per-step logprobs for the entropy feature.
    use_real_logprobs: bool = True

    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Models  (P1-2: shared reference via create_reference_model)
# ---------------------------------------------------------------------------

def _build_models(cfg: Phase3PPOTryConfig):
    import torch as _torch
    from transformers import AutoTokenizer
    from trl import AutoModelForCausalLMWithValueHead, create_reference_model

    dtype_map = {"bf16": _torch.bfloat16, "fp16": _torch.float16, "fp32": _torch.float32}
    torch_dtype = dtype_map.get(cfg.dtype, _torch.bfloat16)
    base_id = cfg.sft_checkpoint or model_path(cfg.base_model)

    tokenizer = AutoTokenizer.from_pretrained(base_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy_kwargs: Dict[str, Any] = {"torch_dtype": torch_dtype}
    # Optional 4-bit base (QLoRA) so an 8B policy + value head fits on a 24GB
    # card. The paper allows this as the low-VRAM fallback (§4.3.3).
    if cfg.use_4bit:
        from transformers import BitsAndBytesConfig

        policy_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )
        policy_kwargs["device_map"] = {"": 0}
    is_peft = False
    if cfg.use_lora:
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
    # P1-2: shared-base reference.
    #  * With LoRA, ref_model=None tells TRL to use the policy with adapters
    #    DISABLED as the reference — zero extra memory (the previous
    #    create_reference_model copied a second full base and caused OOM on 24GB).
    #  * Without LoRA there is no adapter to disable, so we need an explicit
    #    frozen copy.
    if is_peft:
        ref_model = None
    else:
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


# ---------------------------------------------------------------------------
# Data  (P0-3: gold from metadata, not the teacher's own answer)
# ---------------------------------------------------------------------------

def _prepare_prompts(reader: SilverDatasetReader, tokenizer, cfg: "Phase3PPOTryConfig"):
    rows = []
    skipped_no_gold = 0
    for traj in reader.accepted():
        # B3: cap passages (and triples) so the KG block survives prompt budgeting.
        msgs = build_rl_messages(
            question=traj.question,
            retrieved_passages=traj.retrieved_passages,
            kg_triples=traj.kg_subgraph,
            top_k=cfg.ppo_max_passages,
            max_kg_triples=cfg.ppo_max_kg_triples,
        )
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        else:
            text = "\n\n".join(m["content"] for m in msgs)
        # A4: require a real dataset gold. The teacher answer is NOT a valid EM
        # target — rewarding it would teach PPO to match the teacher's mistakes
        # (reward hacking). Skip trajectories with no gold rather than fall back.
        gold = str(traj.metadata.get("gold_answer") or "").strip()
        if not gold:
            skipped_no_gold += 1
            continue
        spec = RewardSpec(
            query=traj.question,
            gold_answer=gold,
            kg_subgraph=list(traj.kg_subgraph),
            retrieved_passages=list(traj.retrieved_passages),
            metadata={"qid": traj.qid, "dataset": traj.dataset},
        )
        rows.append({"prompt": text, "spec": spec})
    if skipped_no_gold:
        logger.warning("Skipped %d accepted trajectories with no gold_answer (A4: no teacher fallback).", skipped_no_gold)
    return rows


def _generate(policy, tokenizer, prompts: Sequence[str], cfg: Phase3PPOTryConfig, device: str):
    """Generate one response per prompt; also return per-token scores (P1-1)."""
    query_tensors, response_tensors, response_texts, scores_list = [], [], [], []
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
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=cfg.max_input_length)
        query_ids = enc["input_ids"].to(device).squeeze(0)
        with torch.no_grad():
            out = policy.generate(
                input_ids=query_ids.unsqueeze(0),
                max_new_tokens=cfg.max_new_tokens,
                do_sample=True,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                return_dict_in_generate=cfg.use_real_logprobs,
                output_scores=cfg.use_real_logprobs,
            )
        if cfg.use_real_logprobs:
            gen = out.sequences[0]
            scores_list.append(out.scores)
        else:
            gen = out[0]
            scores_list.append(None)
        response_ids = gen[query_ids.size(0):]
        query_tensors.append(query_ids)
        response_tensors.append(response_ids)
        response_texts.append(tokenizer.decode(response_ids, skip_special_tokens=True))
    return query_tensors, response_tensors, response_texts, scores_list


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_phase3_ppo_try(cfg: Phase3PPOTryConfig) -> Dict[str, Any]:
    set_seed(cfg.seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from trl import PPOConfig

    policy, ref_model, tokenizer = _build_models(cfg)
    device = next(policy.parameters()).device.type if next(policy.parameters()).is_cuda else "cpu"

    # ---- reward components (P0-2: ImprovedPRMAnnotator) -----------------
    alpha_gate = AlphaGate()
    if cfg.alpha_gate_path and Path(cfg.alpha_gate_path).exists():
        alpha_gate.load_state_dict(torch.load(cfg.alpha_gate_path, map_location="cpu"))
        logger.info("Loaded α-gate from %s", cfg.alpha_gate_path)
    alpha_gate.eval()

    # Finding 2: load the entity cache so link_confidence (the α-gate's middle
    # feature) is live and matches Phase 2 training. A bare ImprovedPRMAnnotator()
    # builds an EntityLinker with an EMPTY cache → link_confidence ≡ 0, which would
    # make the gate's middle input dead at inference and mismatch Phase 2.
    from kgproweight.retrieval.bootstrap import resolve_entity_cache_path
    from kgproweight.kg.entity_linker import EntityLinker as _EntityLinker
    _entity_linker = _EntityLinker(cache_path=resolve_entity_cache_path())
    logger.info("PPO link_confidence: EntityLinker cache=%s (%d entries)",
                resolve_entity_cache_path(), len(list(_entity_linker.cache.items())))
    annotator = ImprovedPRMAnnotator(entity_linker=_entity_linker, verbose=False)
    text_reward = build_text_reward_model(
        backend=cfg.text_reward_backend,
        fallback_head_path=cfg.text_reward_fallback_path,
        device=str(device),
        dtype=cfg.dtype,
    )
    reward_fn = ImprovedRewardFunction(
        alpha_gate=alpha_gate,
        prm_annotator=annotator,
        text_reward_model=text_reward,
        tokenizer=tokenizer,
        outcome_weight=1.0,
        discount=cfg.gamma,
        alpha_override=cfg.alpha_override,
    )

    # ---- data ------------------------------------------------------------
    reader = SilverDatasetReader(cfg.silver_path)
    if cfg.binary_labels_only:
        for traj in reader.trajectories:
            for step in traj.steps:
                if step.label == 0:
                    step.label = -1
    samples = _prepare_prompts(reader, tokenizer, cfg)
    if not samples:
        raise ValueError(f"No PPO samples derived from {cfg.silver_path}")
    logger.info("Phase 3b PPO (try) with %d prompts (target steps=%d)", len(samples), cfg.total_steps)

    # ---- trainer (StepRewardPPOTrainer) ----------------------------------
    # TRL requires batch_size to be a multiple of mini_batch_size; clamp the
    # mini-batch down for small (smoke) batches so a tiny run never trips the
    # exact-division check.
    mini_batch_size = min(cfg.mini_batch_size, cfg.batch_size)
    if cfg.batch_size % mini_batch_size != 0:
        mini_batch_size = cfg.batch_size  # fall back to single mini-batch
    ppo_cfg = PPOConfig(
        learning_rate=cfg.learning_rate,
        batch_size=cfg.batch_size,
        mini_batch_size=mini_batch_size,
        ppo_epochs=cfg.ppo_epochs,
        cliprange=cfg.cliprange,
        kl_penalty="kl",
        init_kl_coef=cfg.kl_coef,
        gamma=cfg.gamma,
        lam=cfg.lam,
        max_grad_norm=cfg.max_grad_norm,
        vf_coef=cfg.vf_coef,
        target_kl=cfg.target_kl,
        early_stopping=cfg.early_stopping,
        log_with=None,
        seed=cfg.seed,
    )
    trainer = StepRewardPPOTrainer(config=ppo_cfg, model=policy, ref_model=ref_model, tokenizer=tokenizer)

    # ---- loop ------------------------------------------------------------
    rng = torch.Generator().manual_seed(cfg.seed)
    n_seen = 0
    history: List[Dict[str, float]] = []
    while n_seen < cfg.total_steps:
        batch_idx = torch.randint(0, len(samples), (cfg.batch_size,), generator=rng).tolist()
        batch = [samples[i] for i in batch_idx]
        prompts = [s["prompt"] for s in batch]
        specs: List[RewardSpec] = [s["spec"] for s in batch]

        query_tensors, response_tensors, response_texts, scores_list = _generate(
            policy, tokenizer, prompts, cfg, device
        )

        token_reward_list: List[torch.Tensor] = []
        traj_rewards: List[float] = []
        for resp_text, response_ids, spec, gen_scores in zip(
            response_texts, response_tensors, specs, scores_list
        ):
            # #6: compute step spans in response_ids coordinates ONCE, and use
            # them for BOTH the entropy logprob bucketing and the reward
            # placement, so neither is misaligned by decode∘re-tokenise drift.
            n_parsed = len(parse_steps(resp_text)[: cfg.max_steps])
            aligned_spans = step_spans_over_ids(response_ids, tokenizer, n_parsed)
            logprobs_per_step = None
            if cfg.use_real_logprobs and gen_scores is not None:
                logprobs_per_step = _step_logprobs_from_scores(response_ids, gen_scores, aligned_spans)
            info = reward_fn(
                prompt="", response=resp_text, spec=spec,
                logprobs_per_step=logprobs_per_step,
                response_ids=response_ids, step_spans=aligned_spans,
            )
            traj_rewards.append(info["trajectory_reward"])
            # token_rewards is already built in response_ids space (#6), so it
            # matches the response tensor length exactly — no pad/truncate.
            tr = info["token_rewards"]
            n = response_ids.size(0)
            if tr.size(0) != n:  # defensive: should not happen with aligned spans
                tr = (torch.cat([tr, torch.zeros(n - tr.size(0), dtype=tr.dtype)])
                      if tr.size(0) < n else tr[:n])
            token_reward_list.append(tr)

        # P0-1: hand the per-token step rewards to the trainer so GAE runs on
        # them. The scalar `scores` arg is a placeholder (ignored by our
        # override when pending rewards are set) but must satisfy TRL's shape
        # checks: one scalar tensor per sample.
        placeholder_scores = [torch.zeros((), dtype=torch.float32) for _ in token_reward_list]
        trainer.set_pending_step_rewards(token_reward_list)
        stats = trainer.step(query_tensors, response_tensors, placeholder_scores)

        n_seen += cfg.batch_size
        history.append({
            "step": n_seen,
            "mean_reward": float(sum(traj_rewards) / max(1, len(traj_rewards))),
            "ppo_mean_kl": float(stats.get("objective/kl", 0.0)),
            "advantage_var": float(stats.get("ppo/policy/advantages_var", 0.0)),
        })
        if n_seen % (cfg.batch_size * 4) == 0:
            logger.info("step=%d mean_reward=%.4f kl=%.4f adv_var=%.4f", n_seen,
                        history[-1]["mean_reward"], history[-1]["ppo_mean_kl"], history[-1]["advantage_var"])

    final_dir = out_dir / "final"
    trainer.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(final_dir)
    with open(out_dir / "history.jsonl", "w", encoding="utf-8") as fh:
        for h in history:
            fh.write(json.dumps(h) + "\n")
    dump_manifest(out_dir, extra={"phase": "phase3_ppo_try", "config": asdict(cfg), "history_tail": history[-5:]})
    logger.info("Phase 3b PPO (try) done. Final checkpoint at %s", final_dir)
    return {"output_dir": str(out_dir), "final_checkpoint": str(final_dir)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None, help="optional YAML (configs/training/phase3_ppo.yaml)")
    p.add_argument("--silver", required=True, help="silver trajectory JSONL")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--sft_checkpoint", default=None)
    p.add_argument("--alpha_gate_path", default=None)
    p.add_argument("--text_reward_backend", default=None, choices=["auto", "rearag", "llama_head", "dummy"])
    p.add_argument("--total_steps", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--alpha_override", type=float, default=None)
    p.add_argument("--binary_labels_only", action="store_true")
    p.add_argument("--no_real_logprobs", action="store_true", help="disable P1-1 (entropy→1.0 fallback)")
    p.add_argument("--use_4bit", action="store_true", help="load base in 4-bit (QLoRA) for 24GB cards")
    p.add_argument("--max_new_tokens", type=int, default=None, help="generation length (default from config: 512)")
    p.add_argument("--mini_batch_size", type=int, default=None, help="PPO forward-pass chunk (smaller = less VRAM)")
    p.add_argument("--max_input_length", type=int, default=None, help="prompt truncation length")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _cfg_from_args(args: argparse.Namespace) -> Phase3PPOTryConfig:
    cfg = Phase3PPOTryConfig(silver_path=args.silver, output_dir=args.output_dir, seed=args.seed)
    # Layer 1: YAML config (reuse the package loader; read the ppo block defensively).
    if args.config:
        from kgproweight.config import ProjectConfig, load_config

        pc = load_config(args.config, validate=ProjectConfig)
        tr = pc.training
        cfg.base_model = getattr(tr, "base_model", cfg.base_model)
        cfg.dtype = getattr(tr, "dtype", cfg.dtype)
        cfg.use_lora = not getattr(tr, "use_qlora", False) if hasattr(tr, "use_qlora") else cfg.use_lora
        cfg.lora_r = getattr(tr, "lora_r", cfg.lora_r)
        cfg.lora_alpha = getattr(tr, "lora_alpha", cfg.lora_alpha)
        cfg.lora_dropout = getattr(tr, "lora_dropout", cfg.lora_dropout)
        ppo = getattr(tr, "ppo", None)
        if ppo is not None:
            for k in ("learning_rate", "batch_size", "mini_batch_size", "ppo_epochs",
                      "cliprange", "kl_coef", "gamma", "lam", "max_grad_norm", "vf_coef"):
                if hasattr(ppo, k):
                    setattr(cfg, k, getattr(ppo, k))
            if hasattr(ppo, "total_ppo_steps"):
                cfg.total_steps = ppo.total_ppo_steps
        cfg.max_input_length = getattr(tr, "max_input_length", cfg.max_input_length)
    # Layer 2: explicit CLI wins.
    if args.sft_checkpoint is not None:
        cfg.sft_checkpoint = args.sft_checkpoint
    if args.alpha_gate_path is not None:
        cfg.alpha_gate_path = args.alpha_gate_path
    if args.text_reward_backend is not None:
        cfg.text_reward_backend = args.text_reward_backend
    if args.total_steps is not None:
        cfg.total_steps = args.total_steps
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.alpha_override is not None:
        cfg.alpha_override = args.alpha_override
    cfg.binary_labels_only = args.binary_labels_only
    if args.no_real_logprobs:
        cfg.use_real_logprobs = False
    if args.use_4bit:
        cfg.use_4bit = True
    if args.max_new_tokens is not None:
        cfg.max_new_tokens = args.max_new_tokens
    if args.mini_batch_size is not None:
        cfg.mini_batch_size = args.mini_batch_size
    if args.max_input_length is not None:
        cfg.max_input_length = args.max_input_length
    return cfg


def main() -> None:
    args = parse_args()
    cfg = _cfg_from_args(args)
    stats = run_phase3_ppo_try(cfg)
    logger.info("Phase 3b PPO (try) stats: %s", stats)


if __name__ == "__main__":
    main()
