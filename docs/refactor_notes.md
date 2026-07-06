# Refactor Notes: kg2 → kgpaper

A traceable mapping from the legacy `kg2` layout to the refactored
`kgpaper` package, plus the list of semantic bugs that were fixed.

---

## 1. Path mapping

| kg2 file | kgpaper destination | Notes |
|----------|---------------------|-------|
| `src/__init__.py` | `kgproweight/__init__.py` | reorganised, no top-level torch import |
| `src/alpha_gate.py` | `kgproweight/reward/alpha_gate.py` | `semantic_entropy` accepts real logprobs |
| `src/kg_retriever.py` | split: `kgproweight/kg/{entity_linker,wikidata_retriever,coverage,cache}.py` | per-concern files |
| `src/prm_annotator.py` | `kgproweight/reward/prm_annotator.py` | unified prompt schema |
| `src/composite_reward.py` | split: `kgproweight/reward/{composite_reward,prm_value_head,text_reward_model}.py` | R_Text actually wired |
| `src/kg_proweight_pipeline.py` | `kgproweight/pipeline/kg_proweight_pipeline.py` | D_dropout honoured |
| `src/qlora_inference.py` | `kgproweight/pipeline/generators.py` | bf16 primary, QLoRA fallback |
| `scripts/step0_convert_corpus.py` | `scripts/prepare/00_convert_corpus.py` | same logic |
| `scripts/step1_build_dense_index.sh` | `scripts/prepare/01_build_dense_index.sh` | Pro 6000 batch_size 1024 |
| `scripts/step2_build_bm25_index.sh` | `scripts/prepare/02_build_bm25_index.sh` | unchanged |
| `scripts/step3_download_datasets.py` | `scripts/prepare/03_download_datasets.py` | unchanged |
| `scripts/step4_generate_silver_data.py` | `scripts/train/phase1_generate_silver.py` + `kgproweight/training/phase1_distill.py` | retrieval really invoked |
| `scripts/step5_train_prm.py` | `scripts/train/phase2_train_prm.py` + `kgproweight/training/phase2_prm.py` | real logprobs |
| `scripts/step6_train_ppo.py` | split: `scripts/train/{phase3_sft,phase3_ppo,phase3_grpo}.py` + matching modules | SFT pre-step, GAE+Critic |
| `scripts/step7_eval_baselines.py` and its three siblings | `scripts/eval/run_baselines.py` (single source) | RRF top-50 default |
| `scripts/step8_eval_kg_proweight.py` | `scripts/eval/run_kg_proweight.py` | bf16 default |
| `scripts/step8_add_ablations.py` | folded into `scripts/eval/run_ablations.py` | no_kg/e5_only variants |
| `scripts/step9_ablation.py` | `scripts/eval/run_ablations.py` | alpha variants retrain PPO |
| `scripts/step10_build_dropout_test.py` | `scripts/prepare/05_build_d_dropout.py` | same logic |
| `scripts/reannotate_silver_jsonl.py` | `scripts/utils/reannotate_silver.py` | unchanged |
| `scripts/prewarm_wikidata_cache.py` | `scripts/prepare/04_prewarm_wikidata_cache.py` | unchanged |
| `configs/base.yaml` | `configs/base.yaml` | KG2_ROOT removed, env-driven paths |
| `configs/{hotpotqa,2wikimultihopqa,musique}.yaml` | `configs/datasets/*.yaml` | + `d_dropout.yaml` |
| `data/silver_data/*` | (not migrated; regenerate via Phase 1) | code-only refactor |
| `indexes/*` | (not migrated; rebuild via Steps 0–2) | code-only refactor |
| `checkpoints/*` | (not migrated; retrain) | code-only refactor |

---

## 2. Semantic bugs fixed

The numbers below cross-reference §3 of the original refactor plan.

1. **R_Text consumed in PPO reward.** `kgproweight/training/reward_function.py`
   now computes `R_total = α·R_KG + (1-α)·R_Text`. Previously
   `kg2/scripts/step6_train_ppo.py:112` dropped R_Text entirely.

2. **R_Text model is real.** `kgproweight/reward/text_reward_model.py`
   loads ReaRAG-9B for prompt scoring (primary) or trains a Llama-3-8B
   reward head on silver data (fallback). The legacy `TextRewardHead` was
   an untrained `nn.Linear`.

3. **`semantic_entropy` from real logprobs.** Phase 2 now runs a logprob
   pre-pass over silver data and `silver_with_logprobs.jsonl` carries
   `steps[].token_logprobs`. Previously hardcoded to 0.5.

4. **Phase 1 Teacher sees retrieved text.** `phase1_distill.py` calls the
   FlashRAG hybrid retriever; the legacy `get_retrieved_text_placeholder`
   returned a literal placeholder string.

5. **D_dropout honoured at inference.** `kgproweight/pipeline/kg_proweight_pipeline.py`
   reads `item.metadata.dropout.modified_kg` before calling SPARQL.
   Previously the dropout subgraph was written to disk and ignored.

6. **Unified prompt schema.** `kgproweight/data/prompts.py` is the only
   source of prompts; Teacher/SFT/RL/inference/PRM annotator all share
   `[Step N] ... [Final Answer]`. Previously three schemas drifted.

7. **SFT before PPO.** `scripts/train/phase3_sft.py` exists. Previously
   PPO ran straight on the PRM checkpoint and the student never learned
   the target format.

8. **PPO complete.** `scripts/train/phase3_ppo.py` uses TRL `PPOTrainer`
   with a reference model, critic, per-token GAE, and `R_total` exposed at
   step boundaries. Previously the PPO path scored full trajectories with
   a single scalar.

9. **Critic uses PRMValueHead.** `kgproweight/reward/prm_value_head.py`
   is attached to the policy in PPO. Previously it sat unused.

10. **Link confidence from KG embeddings.** `kgproweight/kg/kg_embeddings.py`
    loads optional TransE/RotatE checkpoints and computes the cosine
    between TransE entity and LLM context embeddings. Falls back to fuzzy
    matching with a clear log warning.

11. **Ablations retrained.** `scripts/eval/run_ablations.py` triggers a
    short PPO retrain for α=0, α=1, α=0.5, binary_labels. Previously the
    α value was monkey-patched at inference.

12. **Baselines under RRF top-50 by default.** `scripts/eval/run_baselines.py`
    always builds `multi_retriever_setting` via
    `kgproweight.retrieval.hybrid.build_rrf_setting()`. The legacy
    `step7_add_rrf_baselines.py` patch is no longer necessary.

13. **GPT-4o LLM-as-Judge IHR implemented.**
    `kgproweight/reward/ihr_judge.py` + `scripts/eval/run_ihr_judge.py`.
    Previously only the heuristic IHR existed.

14. **Path & import hygiene.** Every module imports
    `from kgproweight.utils.flashrag_bootstrap import setup_flashrag`
    rather than mutating `sys.path` ad-hoc. Hardcoded `KG2_ROOT /
    "Meta-Llama-3-8B-Instruct"` is gone — paths are env-driven through
    `kgproweight.utils.paths`.

15. **Inference aligned with SFT schema.** `KGProWeightPipeline` now uses
    `build_inference_messages()` + single-pass generation +
    `extract_final_answer()` instead of FlashRAG `ReasoningPipeline`'s
    `<answer>` protocol.

16. **Naive RAG baseline prompts include `{reference}`.** Retrieved
    passages are injected into the LLM prompt under RRF top-50.

17. **Training/eval passage count unified at top-50.** Phase 1 silver
    generation and all prompt builders default to `DEFAULT_TOPK = 50`.

---

## 3. Rigour additions (new in kgpaper)

- `kgproweight/eval/stats.py` — paired bootstrap CI + paired t-test.
- `kgproweight/eval/data_efficiency.py` + `scripts/eval/run_data_efficiency.py`.
- `kgproweight/eval/variance_validation.py` + `scripts/eval/run_variance_validation.py`.
- `kgproweight/eval/alpha_analysis.py` for D_std vs D_dropout α comparison.
- `kgproweight/reward/ihr_judge.py` + `scripts/eval/run_ihr_judge.py`.
- Multi-seed support across every train and eval script.
- `kgproweight/utils/logging.py:dump_manifest()` for reproducibility.

---

## 4. What was *not* migrated

- 15M-line Wikipedia corpus (`indexes/corpus_flashrag.jsonl`).
- 45 GB dense index (`indexes/e5_Flat.index`).
- BM25 directory (`indexes/bm25/`).
- Existing checkpoints under `checkpoints/`.
- Pre-existing `outputs/`.
- Junk files: `data/silver_data/combine.py`,
  `data/silver_data/silver_trajectories copy.jsonl`,
  `data/silver_data/__pycache__/`.

Rebuild instructions live in
[`operation_guide.md`](operation_guide.md) §1 and §2.
