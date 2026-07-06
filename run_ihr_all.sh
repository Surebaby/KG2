#!/bin/bash
set -euo pipefail
export OPENAI_API_KEY="sk-d48bff8b1c3b4376b849570ab44a3e62"
export OPENAI_BASE_URL="https://api.deepseek.com"
cd /root/autodl-tmp/kgpaper
MODEL="deepseek-v4-pro"
N=50
SEED=42

for model in eval_base eval_sft eval_ppo; do
  for ds in hotpotqa_smoke 2wikimultihopqa musique; do
    pred=$(find outputs/$model/$ds/seed_42 -name intermediate_data.json 2>/dev/null | head -1)
    if [ -z "$pred" ]; then
      echo "===== $model / $ds ====="
      echo "  SKIP"
      continue
    fi
    outdir=$(dirname "$pred")
    out="$outdir/ihr_result.json"
    echo "===== $model / $ds ====="
    /root/autodl-tmp/kgpw_env/bin/python -u scripts/eval/run_ihr_judge.py \
      --predictions "$pred" --sample "$N" --seed "$SEED" \
      --judge_model "$MODEL" --output "$out" 2>&1 | tail -3
    sleep 2
  done
done

echo ""
echo "========== SUMMARY =========="
for model in eval_base eval_sft eval_ppo; do
  echo "--- $model ---"
  for ds in hotpotqa_smoke 2wikimultihopqa musique; do
    f=$(find outputs/$model/$ds/seed_42 -name ihr_result.json 2>/dev/null | head -1)
    if [ -f "$f" ]; then
      /root/autodl-tmp/kgpw_env/bin/python3 -c "import json; d=json.load(open('$f')); print(f'  {d.get(\"n_items\",\"?\")} items  IHR={d.get(\"overall_ihr\",\"?\"):.3f}')" 2>/dev/null || echo "  parse error"
    else
      echo "  $ds: (no result)"
    fi
  done
done
