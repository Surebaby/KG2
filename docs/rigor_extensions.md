# Rigour Extensions

The paper's experimental design (paper_design.md §5–§7) requires several
analyses that the legacy `kg2` code base does not implement. This document
describes each extension, the script that produces it, and the artefact it
contributes to the final paper.

| Extension | Script | Section | Output |
|-----------|--------|---------|--------|
| GPT-4o LLM-as-Judge IHR | `scripts/eval/run_ihr_judge.py` | §5.5 indicator 3 | `outputs/rigor/ihr/` |
| Multi-seed aggregation  | `scripts/eval/summarize_results.py` | §5.4–§5.5 | `outputs/summary/` |
| Data-efficiency curve   | `scripts/eval/run_data_efficiency.py` | §5.5 indicator 4 | `outputs/rigor/data_eff/` |
| α distribution analysis | `scripts/eval/run_kg_proweight.py` + `kgproweight.eval.alpha_analysis` | §5.5 indicator 5 | `outputs/rigor/alpha_compare/` |
| Theorem 2 variance log  | `scripts/eval/run_variance_validation.py` | §6.2 | `outputs/rigor/variance/` |
| Significance testing    | `kgproweight.eval.stats` | §5.5 | embedded into summary tables |

---

## 1. IHR with GPT-4o LLM-as-Judge

**Why** the heuristic IHR (PRMAnnotator flagging hallucinations from regex
parsing) overestimates hallucination rates on natural-language steps that
omit triple citations. The paper requires GPT-4o-as-Judge to score each step
on a binary scale, with Cohen κ ≥ 0.7 between human and LLM judges.

**How**

```bash
make rigor-ihr
# or:
python scripts/eval/run_ihr_judge.py \
    --datasets hotpotqa 2wikimultihopqa musique \
    --split dev \
    --sample 200 \
    --predictions outputs/kg_proweight/<dataset>/<run>/intermediate_data.json \
    --judge_model gpt-4o \
    --output outputs/rigor/ihr/<dataset>.json
```

The script:

1. Loads recorded reasoning traces.
2. Parses them with `kgproweight.data.parsers.parse_steps`.
3. Sends a structured-JSON request to GPT-4o per step.
4. Aggregates IHR per dataset.
5. Optionally takes a `--human_csv` with hand labels and reports Cohen κ.

---

## 2. Multi-seed aggregation

**Why** the paper claims 2–4 F1 gain over ReaRAG; with single-seed results
this is below noise.

**How**

- Every training script accepts `--seed <int>`; the default is the value in
  `configs/training/<phase>.yaml`.
- Every evaluation script accepts `--seed`.
- `make eval-kgpw SEEDS="13 42 2024"` runs the eval three times with three
  seeds and stores results in
  `outputs/kg_proweight/<dataset>/seed_<seed>/`.
- `make summarize` collects all per-seed `metric_score.json` and emits
  mean ± std plus 95 % CI.

---

## 3. Data-efficiency curve

**Why** paper §5.5 indicator 4 demands F1 vs. silver-data size to claim
"≥30 % less data than ReaRAG".

**How**

```bash
make rigor-data-eff
# or:
python scripts/eval/run_data_efficiency.py \
    --sizes 1000 2000 5000 10000 15000 \
    --seeds 13 42 \
    --base_config configs/training/phase3_ppo.yaml
```

For each size, the script:

1. Takes a random subset of `silver_with_logprobs.jsonl`.
2. Trains a Phase 3b PPO model (reduced step budget).
3. Evaluates on HotpotQA dev.
4. Writes `outputs/rigor/data_eff/size_<N>_seed_<S>/metric_score.json`.

`summarize_results.py` plots the F1-vs-N curve into `outputs/summary/`.

---

## 4. α distribution analysis

**Why** §5.5 indicator 5: the dynamic α must shrink on D_dropout, validating
the graceful-fallback claim.

**How** `scripts/eval/run_kg_proweight.py` already records per-step α to
`alpha_distribution.jsonl`. `kgproweight.eval.alpha_analysis` provides:

- `compare_alpha(d_std_path, d_dropout_path)` — Welch's t-test of α means.
- Histograms + bar charts of α distributions.

---

## 5. Theorem 2 — empirical variance check

**Why** Theorem 2 predicts that dynamic α reduces the variance of the PPO
advantage estimate. We log it during training and analyse offline.

**How**

```bash
make rigor-variance
# or:
python scripts/eval/run_variance_validation.py \
    --max_steps 500 \
    --alpha_strategies dynamic fixed_0.0 fixed_0.5 fixed_1.0
```

The script reuses `kgproweight.training.phase3_ppo` but with a
`VarianceMonitorCallback` that logs the empirical variance of the GAE
advantage per update step. Output:

- `outputs/rigor/variance/<strategy>.csv` (step, advantage_var, kl, reward)
- `outputs/summary/theorem2_variance.png`

---

## 6. Significance testing

`kgproweight.eval.stats.paired_bootstrap(pred_a, pred_b, gold, n=10000)`
returns the 95 % CI of `F1(a) - F1(b)`. Used by `summarize_results.py`
to print a "p < 0.05 against ReaRAG" column in the main table.

---

## 7. Future-work additions (not in this refactor)

- **GENRE checkpoint download** — currently the entity linker calls
  Wikidata Search; integrating GENRE weights requires `fairseq` which we
  keep as an optional extra (`pip install -e .[genre]`).
- **Multilingual Wikidata coverage** — paper §8.2 direction 1.
- **vLLM-accelerated PPO** — vLLM/TRL integration is still moving; we keep
  vLLM only as an evaluation back-end via `framework: vllm` in the
  generator config.
