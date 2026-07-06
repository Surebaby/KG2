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
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from kgproweight.data.prompts import build_rl_messages
from kgproweight.data.silver_dataset import SilverDatasetReader
from kgproweight.reward.alpha_gate import AlphaGate
from kgproweight.reward.prm_annotator import PRMAnnotator
from kgproweight.reward.text_reward_model import build_text_reward_model
from kgproweight.training.reward_function import KGProWeightRewardFunction, RewardSpec
from kgproweight.utils.logging import dump_manifest, get_logger
from kgproweight.utils.paths import model_path
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
    extra: Dict[str, Any] = field(default_factory=dict)


def _build_policy(cfg: Phase3GRPOConfig):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_id = cfg.sft_checkpoint or model_path(cfg.base_model)
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map.get(cfg.dtype, torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(base_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch_dtype, device_map="auto")

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
    reward_fn = KGProWeightRewardFunction(
        alpha_gate=alpha_gate,
        prm_annotator=annotator,
        text_reward_model=text_reward,
        tokenizer=tokenizer,
        outcome_weight=1.0,
        discount=0.95,
        alpha_override=cfg.alpha_override,
    )

    reader = SilverDatasetReader(cfg.silver_path)
    if cfg.binary_labels_only:
        for traj in reader.trajectories:
            for step in traj.steps:
                if step.label == 0:
                    step.label = -1

    prompts_pool: List[Dict[str, Any]] = []
    for traj in reader.accepted():
        msgs = build_rl_messages(
            question=traj.question,
            retrieved_passages=traj.retrieved_passages,
            kg_triples=traj.kg_subgraph,
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
                    kg_subgraph=list(traj.kg_subgraph),
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
    while n_seen < cfg.total_steps:
        idx = torch.randint(0, len(prompts_pool), (cfg.batch_size,), generator=rng).tolist()
        batch = [prompts_pool[i] for i in idx]

        loss_total = torch.zeros((), device=device, dtype=torch.float32)
        for sample in batch:
            prompt = sample["prompt"]
            spec = sample["spec"]
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=cfg.max_input_length).to(device)
            query_ids = enc["input_ids"][0]

            # Rollout K times
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

            rewards = [reward_fn(prompt, t, spec)["trajectory_reward"] for t in responses_text]
            rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
            advantages = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-6)

            # Policy loss: -E[A * log π(response | prompt)]
            sample_loss = torch.zeros((), device=device, dtype=torch.float32)
            for adv, r_ids in zip(advantages, responses_ids):
                concat = torch.cat([query_ids, r_ids]).unsqueeze(0)
                labels = concat.clone()
                labels[:, : query_ids.size(0)] = -100
                out = model(input_ids=concat, labels=labels)
                # out.loss is mean NLL over response tokens.
                sample_loss = sample_loss + adv.detach() * out.loss
            sample_loss = sample_loss / cfg.group_size

            # Light-weight KL anchor toward base via cross-entropy regularisation on response.
            # Skipped here; PPO-style anchor is in :mod:`phase3_ppo`.
            loss_total = loss_total + sample_loss

        loss = loss_total / cfg.batch_size
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optim.step()
        optim.zero_grad(set_to_none=True)
        n_seen += cfg.batch_size

        history.append({"step": n_seen, "loss": float(loss.detach().cpu().item())})
        if n_seen % (cfg.batch_size * 4) == 0:
            logger.info("GRPO step=%d loss=%.4f", n_seen, history[-1]["loss"])

    final_dir = out_dir / "final"
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(final_dir)

    dump_manifest(out_dir, extra={"phase": "phase3_grpo", "config": asdict(cfg), "history_tail": history[-5:]})
    logger.info("Phase 3b GRPO done. Final checkpoint at %s", final_dir)
    return {"output_dir": str(out_dir), "final_checkpoint": str(final_dir)}
