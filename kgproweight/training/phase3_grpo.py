"""Phase 3b alternative — GRPO (Group Relative Policy Optimisation).

Lighter than PPO: no critic, no value head, no reference model — uses a
group of K rollouts per prompt and standardises the rewards within the
group as the advantage estimate. Useful on 24 GB cards where a frozen
reference + value head would not fit.

We keep the same reward function (composite per-step + outcome) as PPO.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import os
import torch
import torch.nn.functional as F

from collections import defaultdict

from kgproweight.data.prompts import build_rl_messages
from kgproweight.data.silver_dataset import SilverDatasetReader
from kgproweight.kg.kg_filter import filter_and_rank_triples
from kgproweight.data.silver_split import (
    DEFAULT_SPLIT_SEED,
    DEFAULT_TEST_RATIO,
    DEFAULT_VAL_RATIO,
)
from kgproweight.kg.wikidata_retriever import WikidataSubgraphRetriever
from kgproweight.reward.alpha_gate import AlphaGate
from kgproweight.reward.prm_annotator import PRMAnnotator
from kgproweight.reward.text_reward_model import build_text_reward_model
from kgproweight.training.reward_function import KGProWeightRewardFunction, RewardSpec
from kgproweight.utils.logging import dump_manifest, get_logger
from kgproweight.utils.paths import index_dir, model_path
from kgproweight.utils.seed import set_seed

logger = get_logger(__name__)


@dataclass
class Phase3GRPOConfig:
    silver_path: str
    output_dir: str
    base_model: str = "llama3-8B-instruct"
    sft_checkpoint: Optional[str] = None
    alpha_gate_path: Optional[str] = None
    text_reward_backend: str = "auto"
    text_reward_fallback_path: Optional[str] = None
    dtype: str = "bf16"
    seed: int = 42

    group_size: int = 4
    learning_rate: float = 5.0e-6
    batch_size: int = 16  # number of prompts per update
    total_steps: int = 3000

    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    max_input_length: int = 4096

    use_lora: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05

    alpha_override: Optional[float] = None
    binary_labels_only: bool = False
    kl_coef: float = 0.05
    # R9 v6: reward calibration (Plan B).
    # High outcome weight + low step scale = "final answer is what matters".
    # The noisy per-step r_kg signal is down-weighted; GRPO's group-comparison
    # advantage amplifies the consistent EM signal across rollouts.
    outcome_weight: float = 8.0
    step_reward_scale: float = 0.5
    text_reward_scale: float = 0.3
    # retraining_plan §9.4-1 (量纲): R_Text DC removal. GRPO is NOT the retraining
    # path (PPO is), so this defaults OFF and no GRPO run changes. It is plumbed
    # anyway because GRPO shares KGProWeightRewardFunction and therefore shares
    # the bug: the +0.63 offset makes d r_total/d alpha negative here too. Note
    # GRPO's group-relative advantage already cancels the offset in the OUTCOME
    # channel -- but not in the alpha interaction, which is the part that matters.
    center_text_reward: bool = False
    text_baseline_momentum: float = 0.99
    discount: float = 0.95
    min_valid_steps: int = 2  # aligned with PPO
    min_reasoning_chars: int = 20
    max_steps: int = 7
    kg_offline: bool = True  # offline by default for training (use cached KG)
    max_kg_triples: int = 12  # R9 v6: filter noisy silver KG during training
    # Fold to roll out on; see Phase3PPOConfig.split. Must match the fold the
    # earlier phases used. ``None`` reproduces the pre-split whole-file behaviour.
    split: Optional[str] = None
    val_ratio: float = DEFAULT_VAL_RATIO
    test_ratio: float = DEFAULT_TEST_RATIO
    split_seed: Optional[int] = DEFAULT_SPLIT_SEED
    extra: Dict[str, Any] = field(default_factory=dict)

    def build_split_spec(self):
        from kgproweight.data.silver_split import SplitSpec

        return SplitSpec(
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            seed=self.seed if self.split_seed is None else self.split_seed,
        )


def _build_policy(cfg: Phase3GRPOConfig):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_id = cfg.sft_checkpoint or model_path(cfg.base_model)
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map.get(cfg.dtype, torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(base_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # R9 v6: limit policy model to 50 GiB max so ReaRAG + activations fit.
    # Without this, accelerate's device_map="auto" reserves ~90% of GPU (85 GiB
    # on a 96 GiB card), leaving no room for training activations or reward models.
    max_mem = os.getenv("KGPW_POLICY_MAX_MEMORY", "50GiB")
    model = AutoModelForCausalLM.from_pretrained(
        base_id, torch_dtype=torch_dtype,
        device_map="auto",
        max_memory={0: max_mem},
    )

    if cfg.use_lora:
        try:
            from peft import LoraConfig, TaskType, get_peft_model

            lcfg = LoraConfig(
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, lcfg)
        except ImportError:
            logger.warning("peft not installed; GRPO will train all parameters.")
    return model, tokenizer


def run_phase3_grpo(cfg: Phase3GRPOConfig) -> Dict[str, Any]:
    set_seed(cfg.seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = _build_policy(cfg)
    device = next(model.parameters()).device

    alpha_gate = AlphaGate()
    if cfg.alpha_gate_path and Path(cfg.alpha_gate_path).exists():
        alpha_gate.load_state_dict(torch.load(cfg.alpha_gate_path, map_location="cpu"))
    alpha_gate.eval()

    annotator = PRMAnnotator(verbose=False)
    text_reward = build_text_reward_model(
        backend=cfg.text_reward_backend,
        fallback_head_path=cfg.text_reward_fallback_path,
        device=str(device),
        dtype=cfg.dtype,
    )
    # R9 v6: build subgraph retriever for dynamic KG reward (same as PPO).
    # Without this, the reward function uses only static silver KG → graph_density
    # is always zero → α gate can't compute meaningful r_kg.
    _kg_cache_dir = str(Path(index_dir()) / "kg_cache")
    subgraph_retriever = WikidataSubgraphRetriever(
        max_hops=2, max_neighbors=30, cache_dir=_kg_cache_dir,
        offline=cfg.kg_offline,
    )

    reward_fn = KGProWeightRewardFunction(
        alpha_gate=alpha_gate,
        prm_annotator=annotator,
        text_reward_model=text_reward,
        tokenizer=tokenizer,
        outcome_weight=cfg.outcome_weight,
        step_reward_scale=cfg.step_reward_scale,
        text_reward_scale=cfg.text_reward_scale,
        discount=cfg.discount,
        alpha_override=cfg.alpha_override,
        min_valid_steps=cfg.min_valid_steps,
        min_reasoning_chars=cfg.min_reasoning_chars,
        max_steps=cfg.max_steps,
        subgraph_retriever=subgraph_retriever,
        center_text_reward=cfg.center_text_reward,
        text_baseline_momentum=cfg.text_baseline_momentum,
    )

    reader = SilverDatasetReader(
        cfg.silver_path,
        split=cfg.split,
        split_spec=cfg.build_split_spec() if cfg.split else None,
    )
    if cfg.split is None:
        logger.warning(
            "Phase 3 GRPO split: NONE — rolling out over the whole file (%d "
            "trajectories, %d accepted). Nothing is held back.",
            len(reader.trajectories), len(reader.accepted()),
        )
    else:
        logger.info(
            "Phase 3 GRPO split: fold=%s -> %d/%d trajectories, %d accepted.",
            cfg.split, len(reader.trajectories), reader.n_total_in_file,
            len(reader.accepted()),
        )
    if cfg.binary_labels_only:
        for traj in reader.trajectories:
            for step in traj.steps:
                # Labels are continuous r_kg; anything not clearly positive
                # collapses to negative for this ablation.
                if float(step.label) < 0.5:
                    step.label = -1.0

    prompts_pool: List[Dict[str, Any]] = []
    for traj in reader.accepted():
        # R9 v6: filter silver KG to match inference quality (max_keep=12,
        # hard-delete noise, score threshold). Without this, training sees
        # 105 triples/q of unfiltered noise while inference sees 10.3 — the
        # distribution mismatch that collapsed PPO.
        filtered_kg = filter_and_rank_triples(
            list(traj.kg_subgraph),
            question=traj.question,
            max_keep=cfg.max_kg_triples,
            min_keep=5,
        )
        msgs = build_rl_messages(
            question=traj.question,
            retrieved_passages=traj.retrieved_passages,
            kg_triples=filtered_kg,
        )
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        else:
            text = "\n\n".join(m["content"] for m in msgs)
        prompts_pool.append(
            {
                "prompt": text,
                "spec": RewardSpec(
                    query=traj.question,
                    gold_answer=traj.answer or "",
                    kg_subgraph=filtered_kg,  # R9 v6: use filtered KG for reward
                    retrieved_passages=list(traj.retrieved_passages),
                    metadata={"qid": traj.qid},
                ),
            }
        )

    if not prompts_pool:
        raise ValueError(f"No GRPO samples derived from {cfg.silver_path}")

    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.learning_rate)

    rng = torch.Generator().manual_seed(cfg.seed)
    n_seen = 0
    history = []
    _batch_metrics: List[Dict[str, float]] = []
    while n_seen < cfg.total_steps:
        idx = torch.randint(0, len(prompts_pool), (cfg.batch_size,), generator=rng).tolist()
        batch = [prompts_pool[i] for i in idx]

        loss_total = torch.zeros((), device=device, dtype=torch.float32)
        for sample in batch:
            prompt = sample["prompt"]
            spec = sample["spec"]
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=cfg.max_input_length).to(device)
            query_ids = enc["input_ids"][0]

            # Rollout K times — must be in eval mode (dropout OFF).
            model.eval()
            responses_ids = []
            responses_text = []
            for _ in range(cfg.group_size):
                with torch.no_grad():
                    gen = model.generate(
                        input_ids=query_ids.unsqueeze(0),
                        attention_mask=enc["attention_mask"],
                        max_new_tokens=cfg.max_new_tokens,
                        do_sample=True,
                        temperature=cfg.temperature,
                        top_p=cfg.top_p,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    )[0]
                response_ids = gen[query_ids.size(0) :]
                responses_ids.append(response_ids)
                responses_text.append(tokenizer.decode(response_ids, skip_special_tokens=True))

            # Compute rewards and collect per-step metrics.
            reward_results = [reward_fn(prompt, t, spec) for t in responses_text]
            rewards = [r["trajectory_reward"] for r in reward_results]
            rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
            advantages = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-6)

            # Collect per-sample metrics for logging.
            _step_records = []
            for r in reward_results:
                recs = r.get("per_step_records", [])
                if recs:
                    _step_records.extend(recs)
            _alphas = [float(sr.alpha) for sr in _step_records if hasattr(sr, 'alpha')]
            _rkgs = [float(sr.r_kg) for sr in _step_records if hasattr(sr, 'r_kg')]
            _rtexts = [float(sr.r_text) for sr in _step_records if hasattr(sr, 'r_text')]
            _outcomes = [1.0 if r.get("predicted_answer") and r.get("trajectory_reward", 0) > 0 else 0.0
                         for r in reward_results]
            _resp_lens = [len(ids) for ids in responses_ids]

            _batch_metrics.append({
                "reward": float(rewards_t.mean()),
                "reward_std": float(rewards_t.std()),
                "advantage_std": float(advantages.std()),
                "alpha": sum(_alphas) / len(_alphas) if _alphas else 0.0,
                "r_kg": sum(_rkgs) / len(_rkgs) if _rkgs else 0.0,
                "r_text": sum(_rtexts) / len(_rtexts) if _rtexts else 0.0,
                "outcome_rate": sum(_outcomes) / len(_outcomes) if _outcomes else 0.0,
                "response_len": sum(_resp_lens) / len(_resp_lens) if _resp_lens else 0,
            })

            # Policy loss: -E[A * log π(response | prompt)]
            model.train()  # enable dropout for training
            sample_loss = torch.zeros((), device=device, dtype=torch.float32)
            for adv, r_ids in zip(advantages, responses_ids):
                concat = torch.cat([query_ids, r_ids]).unsqueeze(0)
                labels = concat.clone()
                labels[:, : query_ids.size(0)] = -100
                out = model(input_ids=concat, labels=labels)
                # out.loss is mean NLL over response tokens.
                sample_loss = sample_loss + adv.detach() * out.loss
            sample_loss = sample_loss / cfg.group_size

            loss_total = loss_total + sample_loss

        loss = loss_total / cfg.batch_size
        # NaN guard: skip update if loss is not finite (prevents silent corruption).
        if not torch.isfinite(loss):
            logger.warning("GRPO step %d: loss is not finite (%.4f) — skipping update", n_seen, float(loss))
            _batch_metrics = _batch_metrics[:-cfg.batch_size] if len(_batch_metrics) >= cfg.batch_size else _batch_metrics
            continue
        loss.backward()
        _grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optim.step()
        optim.zero_grad(set_to_none=True)
        n_seen += cfg.batch_size

        # Log detailed metrics periodically (every ~4 update steps).
        _log_every = max(cfg.batch_size * 4, 16)
        if n_seen % _log_every == 0 and _batch_metrics:
            _window = _batch_metrics[-_log_every:]
            _avg = {k: sum(m[k] for m in _window) / len(_window) for k in _window[0]}
            logger.info(
                "GRPO step=%d loss=%.4f grad=%.2f | reward=%.2f±%.2f adv_std=%.2f | "
                "α=%.3f r_kg=%.3f r_text=%.3f | EM=%.1f%% len=%d",
                n_seen, float(loss), float(_grad_norm),
                _avg["reward"], _avg["reward_std"], _avg["advantage_std"],
                _avg["alpha"], _avg["r_kg"], _avg["r_text"],
                _avg["outcome_rate"] * 100, int(_avg["response_len"]),
            )
            _batch_metrics = []  # reset for next window

        history.append({"step": n_seen, "loss": float(loss.detach().cpu().item())})

    final_dir = out_dir / "final"
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(final_dir)

    dump_manifest(out_dir, extra={"phase": "phase3_grpo", "config": asdict(cfg), "history_tail": history[-5:]})
    logger.info("Phase 3b GRPO done. Final checkpoint at %s", final_dir)
    return {"output_dir": str(out_dir), "final_checkpoint": str(final_dir)}
