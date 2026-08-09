#!/bin/bash
# Full silver generation — 7:1.5:1.5 ratio
# WARNING: ~10577 API calls, ~$10, ~4-6 hours runtime
set -e

export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:?set it in .env, never hardcode: git history scrubbed 2026-08-23}"
source /home/zjulab/anaconda3/bin/activate kgpaper
export KGPW_FLASHRAG_ROOT=/home/zjulab/kgpaper/flashrag_src

OUTDIR=/home/zjulab/kgpaper/data/silver_data
TIMESTAMP=$(date +%Y%m%d_%H%M)

echo "=== Phase 1/3: HotpotQA (7405 questions, 70%) ==="
python scripts/train/phase1_generate_silver.py \
  --config configs/training/phase1_silver.yaml \
  --dataset hotpotqa \
  --split dev \
  --allow_eval_split \
  --max_queries 7405 \
  --rerank 10 \
  --offline on \
  --output ${OUTDIR}/silver_hotpotqa_${TIMESTAMP}.jsonl \
  --seed 42 \
  2>&1 | tail -5

echo "=== Phase 2/3: 2WikiMultihopQA (1586 questions, 15%) ==="
python scripts/train/phase1_generate_silver.py \
  --config configs/training/phase1_silver.yaml \
  --dataset 2wikimultihopqa \
  --split dev \
  --allow_eval_split \
  --max_queries 1586 \
  --rerank 10 \
  --offline on \
  --output ${OUTDIR}/silver_2wiki_${TIMESTAMP}.jsonl \
  --seed 42 \
  2>&1 | tail -5

echo "=== Phase 3/3: Musique (1586 questions, 15%) ==="
python scripts/train/phase1_generate_silver.py \
  --config configs/training/phase1_silver.yaml \
  --dataset musique \
  --split dev \
  --allow_eval_split \
  --max_queries 1586 \
  --rerank 10 \
  --offline on \
  --output ${OUTDIR}/silver_musique_${TIMESTAMP}.jsonl \
  --seed 42 \
  2>&1 | tail -5

echo "=== Merging ==="
cat ${OUTDIR}/silver_hotpotqa_${TIMESTAMP}.jsonl \
    ${OUTDIR}/silver_2wiki_${TIMESTAMP}.jsonl \
    ${OUTDIR}/silver_musique_${TIMESTAMP}.jsonl \
    > ${OUTDIR}/silver_v6_full_${TIMESTAMP}.jsonl

TOTAL=$(wc -l < ${OUTDIR}/silver_v6_full_${TIMESTAMP}.jsonl)
echo "Done! Total: $TOTAL trajectories → ${OUTDIR}/silver_v6_full_${TIMESTAMP}.jsonl"

# Quality check
python3 -c "
import json
with open('${OUTDIR}/silver_v6_full_${TIMESTAMP}.jsonl') as f:
    data = [json.loads(l) for l in f.read().strip().split('\n') if l.strip()]
acc = [d for d in data if d.get('accepted')]
print(f'Accepted: {len(acc)}/{len(data)} ({100*len(acc)//len(data)}%)')
for ds in ['hotpotqa','2wikimultihopqa','musique']:
    ds_acc = [d for d in acc if d.get('dataset')==ds]
    has_kg = sum(1 for d in ds_acc if 'Knowledge Used: [(' in d.get('teacher_output',''))
    print(f'  {ds}: {len(ds_acc)} accepted, KG ref {has_kg}/{max(1,len(ds_acc))} ({100*has_kg//max(1,len(ds_acc))}%)')
"
