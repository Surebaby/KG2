# Reproducibility Policy

Goal: any user with a Pro 6000 (96 GB) machine and an OpenAI / DeepSeek key
can reproduce every table in `paper_design.md` within ±0.2 F1.

## 1. Random seeds

Every script accepts `--seed`. Default seeds are read from
`configs/<phase>.yaml`. Paper results use three seeds: `{13, 42, 2024}`.

`kgproweight.utils.seed.set_seed(int)` calls (in order):

```python
random.seed(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
transformers.set_seed(seed)
```

For PPO we also set `torch.use_deterministic_algorithms(False)` because
TRL's GAE relies on non-deterministic CUDA kernels; we accept small
run-to-run noise inside one seed and average over the 3 seeds instead.

## 2. Software versions

The pinned versions live in `requirements.txt`. Critical pins:

| Package | Version | Reason |
|---------|---------|--------|
| `torch` | `>=2.4` | Pro 6000 Blackwell support |
| `transformers` | `>=4.44,<4.50` | TRL 0.11.4 compatibility |
| `trl` | `==0.11.4` | API stability for PPOTrainer/Critic |
| `peft` | `>=0.12` | LoRA save/load contract |
| `bm25s` | `==0.2.1` | reproducible index serialisation |

`kgproweight.utils.logging.dump_manifest(checkpoint_dir)` writes a
`manifest.json` after every successful phase with:

- git commit hash
- list of pip-frozen packages
- `torch.version.cuda`
- GPU name (`nvidia-smi --query-gpu=name --format=csv,noheader`)
- seed used
- timestamp
- input data fingerprints (sha256 of `silver_trajectories.jsonl`)

## 3. External API versions

- **Teacher (GPT-4o)**: pin `model="gpt-4o-2024-08-06"` in
  `configs/training/phase1_silver.yaml` (the date suffix freezes the
  weights even when OpenAI rolls the alias).
- **Teacher (DeepSeek-V3)**: `model="deepseek-chat"`. Costs ≈ ¥180 for 25k
  queries.
- **IHR Judge (GPT-4o)**: pin to the same date as the Teacher.

## 4. Hyperparameters

All hyperparameters live in `configs/`. The PPO learning rate, batch size,
KL coefficient, etc. are the values quoted in `paper_design.md` Table §4.3.

## 5. Data fingerprints

We track the sha256 of every released data artefact so a downstream user
can verify that they downloaded the right file:

```
silver_trajectories.jsonl   <sha256>
d_dropout/dev.jsonl         <sha256>
indexes/e5_Flat.index       <sha256>
indexes/corpus_flashrag.jsonl <sha256>
```

These are written into `outputs/manifests/data_fingerprints.json` after
data preparation.

## 6. Reproducibility checklist

Before publishing results, run:

```bash
make test                    # unit tests pass
python scripts/eval/summarize_results.py --check
```

`summarize_results.py --check` verifies:

- every `metric_score.json` it references exists for ≥ 3 seeds;
- every `manifest.json` carries matching git commit hashes;
- paired bootstrap p-values are computed against ReaRAG.
