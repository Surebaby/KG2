# Phase 1 (try variant) — improved trajectory distillation

A standalone copy of the Phase-1 silver-generation pipeline that applies five
fixes discussed for the low yield (original run: 250 attempts → 19 accepted,
~7.6%). **The original code under `kgproweight/` and `scripts/train/` is left
completely untouched.** Everything here lives under `scripts/train/try/` and
*reuses* the unchanged original modules.

## Files

| File | Role |
|------|------|
| `distill_helpers_try.py` | The changed logic: lenient answer matching, robust mention extraction, stratified filter. |
| `phase1_distill_try.py` | Orchestration (`run_phase1`): reuses the original Teacher/retriever/annotator, rewrites `_process_one` + the accept loop. |
| `phase1_generate_silver_try.py` | CLI entry-point + cold-cache guard. |
| `../../../configs/training/phase1_silver_try.yaml` | Config for this variant. |

## The five changes

**1. Lenient answer matching** (`answer_match_score` in `distill_helpers_try.py`)
Replaces the strict `token_f1 >= 0.5` gate. Before scoring it `clean_final_answer`s
the prediction (drops "the answer is" lead-ins, trailing clauses like
"…, a physicist", keeps the first line). The score is the max of:
exact-match → 1.0, gold-is-substring-of-pred → 1.0, token-recall of gold in
pred (handles verbosity), and alias-tolerant token-F1. Default accept threshold
`min_answer_score = 0.3`. This rescues "correct but verbose / aliased" answers
that strict F1 rejected.

**2. Stratified acceptance** (`StratifiedSilverFilter`)
The hard `triple_rate`/`coverage` rejection is gone. Trajectories are bucketed
by `triple_rate`:
- `kg_rich`  (≥ 0.5): always accepted,
- `kg_medium`(≥ 0.15): accepted up to `medium_quota` (35%) of the pool,
- `kg_sparse`(< 0.15): accepted up to `sparse_quota` (25%) of the pool.

Keeping a real `kg_sparse` slice is the point: the α-Gate needs low
density/coverage examples to learn the **α→0 fallback**, which is exactly the
behaviour `D_dropout` is meant to validate. The original filter systematically
deleted these.

**3. Robust mention extraction + soft coverage** (`extract_mentions_robust`)
Combines (a) optional spaCy NER on the question, (b) the capitalised-phrase
regex, and (c) **titles of the top retrieved passages** as anchors (in
HotpotQA/2Wiki the gold supporting docs are titled by their key entity).
`coverage` is recorded in `metadata` as a soft signal and **never rejects** a
query — low-coverage items are exactly the α→0 negatives we want.

**4. SPARQL graceful degradation + prewarm guard**
`kg_retriever.fetch` is wrapped in try/except; on failure the subgraph is empty
and the trajectory simply lands in `kg_sparse` instead of being dropped. The
CLI also runs a **cold-cache check**: it refuses to start a big run unless the
Wikidata subgraph cache has ≥ `--min_cached_subgraphs` (default 50) entries, or
you pass `--allow_cold_cache`. Prewarm first with
`scripts/prepare/04_prewarm_wikidata_cache.py`.

**5. Format retry preserved**
The one-shot corrective retry (`_needs_format_retry` / `_build_retry_messages`,
imported unchanged) still runs — it targets *format* compliance. Yield gains
come from changes 1–2, not from retrying.

## What gets written

Every processed candidate is written to the output JSONL (nothing silently
dropped), with `metadata.accepted`, `metadata.bucket`, `metadata.triple_rate`,
`metadata.coverage`, `metadata.answer_score`, and `metadata.reject_reason`.
Downstream training should filter on `accepted == true`
(`SilverDatasetReader.accepted()` already does this); the rejected pool stays
available for analysis. The manifest records per-bucket counts.

## Usage

```bash
cd /home/ai/flashrag/kgpaper
conda activate kgpw
source .env
export KGPW_FLASHRAG_ROOT=/home/ai/flashrag/flashrag/FlashRAG-main

# 0) prewarm the Wikidata cache (change 4 prerequisite)
python scripts/prepare/04_prewarm_wikidata_cache.py \
    --datasets hotpotqa --split train --limit_per_dataset 25000

# 1) generate silver trajectories (try variant)
python scripts/train/try/phase1_generate_silver_try.py \
    --config configs/training/phase1_silver_try.yaml \
    --dataset hotpotqa --split train --max_workers 8

# knobs without editing the yaml:
#   --min_answer_score 0.25 --sparse_quota 0.3 --medium_quota 0.35
#   --allow_cold_cache   (skip the prewarm guard)
```

Output defaults to `data/silver_data/silver_trajectories_try.jsonl` (distinct
from the original `silver_trajectories.jsonl`, so nothing is overwritten).

## Notes / caveats

- `try` is a Python keyword, so this directory is **not** an importable
  package. The CLI adds its own dir to `sys.path` and imports the siblings
  flat (`from phase1_distill_try import ...`). Run the CLI as a script, not as
  `python -m`.
- spaCy is optional. Without `en_core_web_sm` installed, mention extraction
  falls back to regex + passage titles (no error).
- The stratified quotas are streaming/greedy (decided as items arrive), so the
  final mix approximates the target ratios rather than hitting them exactly.
  For exact ratios, run once then resample offline from the written pool.
