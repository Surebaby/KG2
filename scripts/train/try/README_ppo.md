# Phase 3b PPO (try variant) — per-step reward into GAE + 4 fixes

A standalone PPO implementation under `scripts/train/try/` that fixes four
defects in the package's `kgproweight/training/phase3_ppo.py`. **The package is
left completely untouched** — this reuses every unchanged piece and overrides
only what's broken, mirroring the Phase-1 try-variant convention.

## Why this exists

The package PPO path diverges from the paper design (`docs/paper_design.md`,
the source-of-truth) in four places. One is paper-critical.

| # | Defect (package) | Fix (here) |
|---|------------------|------------|
| **P0-1** | per-step rewards `.sum()`'d to a scalar → TRL 0.11.4 puts it on the **last token only** → GAE sees an outcome-only signal. **Theorem 2 (advantage-variance reduction) loses its code basis** and "process supervision" degrades to outcome RL. | `StepRewardPPOTrainer` overrides `compute_rewards` to place each step's `R_total(t)` on its `[Step N]` end-token; GAE then runs on the real per-step signal. |
| **P0-2** | `R_KG` from the original `PRMAnnotator` (24% filler-+1, -1 misfires) → reward-hackable. | `ImprovedPRMAnnotator` (relevance + abstention guards). |
| **P0-3** | outcome EM vs `traj.answer` (the **teacher's** answer) → rewards copying teacher errors. | EM vs `metadata["gold_answer"]` (the real dataset gold). |
| **P1-1** | `logprobs=[None]` → α-gate entropy feature constant at 1.0. | real per-step token logprobs sliced from `generate(output_scores=True)`. |

Non-issue (verified, not changed): `R_KG ∈ {-1,0,+1}` vs `R_text` — the text
reward is already `tanh`'d to `[-1,1]`, so the scales match.

Bonus (**P1-2**): the reference model uses `create_reference_model(policy)`
(shared base, adapters disabled) instead of loading a second full 8B.

## Files

| File | Role |
|------|------|
| `ppo_reward_try.py` | `ImprovedRewardFunction` — composite per-step reward via the package's `CompositeRewardModel`, but with `ImprovedPRMAnnotator` and optional real per-step logprobs. Returns a per-token reward tensor. |
| `ppo_trainer_try.py` | `StepRewardPPOTrainer(PPOTrainer)` — overrides **only** `compute_rewards` to scatter per-token step rewards over the masked region while keeping TRL's exact per-token KL-penalty channel. Everything downstream (GAE, clip, minibatch, KL controller) is upstream TRL. |
| `phase3_ppo_try.py` | Training entry-point. P0-2/P0-3/P1-1/P1-2 wired in; CLI + YAML config; ablation hooks (`--alpha_override`, `--binary_labels_only`) preserved. |
| `test_ppo_offline.py` | Offline unit tests (no API/GPU/model). |

## How P0-1 works (the key design)

TRL 0.11.4 `step()` internally does:
`compute_rewards(scores, logprobs, ref_logprobs, masks)` → per-token reward
tensor → `compute_advantages(values, rewards, masks)` (GAE) → `train_minibatch`.

We subclass and override **only** `compute_rewards`. The per-token step rewards
are passed through a side channel (`set_pending_step_rewards`, one tensor per
response, called right before each `step`). The override reproduces TRL's KL
penalty per token, then adds our step rewards onto the masked (response) token
positions instead of TRL's "scalar on last token". The `scores` arg becomes a
placeholder (zeros) — when no pending rewards are set the override falls back to
upstream last-token scalar behaviour, so the subclass is drop-in safe.

**Coupling note:** the override copies TRL 0.11.4's `_kl_penalty` + `kl_ctl`
bookkeeping. If TRL is upgraded, re-sync `compute_rewards` with the new upstream.

## Usage

```bash
cd /home/ai/flashrag/kgpaper
conda activate kgpw
source .env
export KGPW_FLASHRAG_ROOT=/home/ai/flashrag/flashrag/FlashRAG-main

# offline tests (fast, no GPU)
python scripts/train/try/test_ppo_offline.py

# a tiny real PPO smoke (needs GPU + an SFT checkpoint); 1 update only
python scripts/train/try/phase3_ppo_try.py \
    --silver scripts/train/try/outputs/silver_try_50b.jsonl \
    --sft_checkpoint <path-to-sft> \
    --output_dir scripts/train/try/outputs/ppo_smoke \
    --total_steps 8 --batch_size 8 \
    --text_reward_backend dummy        # skip the 9B reward model for a smoke

# ablation example (α≡0, paper §7) — retrained, not patched at inference
python scripts/train/try/phase3_ppo_try.py ... --alpha_override 0.0 --total_steps 1000
```

CLI knobs: `--alpha_gate_path` (load trained gate), `--alpha_override
{0,0.5,1}`, `--binary_labels_only`, `--no_real_logprobs` (disable P1-1 →
entropy 1.0 fallback), `--config configs/training/phase3_ppo.yaml` (YAML
defaults, CLI wins).

## Notes / caveats

- `try` is a keyword → this dir is **not** an importable package; the CLI
  inserts its own dir on `sys.path` and imports siblings flat. Run as a script.
- P1-1 logprobs are sliced from `generate` scores by `[Step N]` token span; if
  span extraction fails for a degenerate response the step falls back to
  `None` (entropy 1.0), never crashing.
- The package `phase3_ppo.py` is intentionally left as-is — useful as a
  scalar-reward baseline to contrast against P0-1 (see the paper's Theorem-2
  `variance_validation` comparison).
- Not done here: full-scale training (10h-class), TRL upgrade, GRPO (its
  package impl lacks a KL anchor and would drop Theorem 2 — PPO stays the main
  line).
