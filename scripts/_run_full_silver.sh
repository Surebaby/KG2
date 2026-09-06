#!/bin/bash
# Full silver generation — 7:1.5:1.5 ratio
# WARNING: ~10577 API calls, ~$10, ~4-6 hours runtime
set -euo pipefail

export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:?set it in .env, never hardcode: git history scrubbed 2026-08-23}"
source /home/zjulab/anaconda3/bin/activate kgpaper
export KGPW_FLASHRAG_ROOT=/home/zjulab/kgpaper/flashrag_src

# Retrieval MUST point at the real 21M-passage wiki18 corpus. Without these,
# resolve_corpus_path/resolve_dense_index_path/resolve_bm25_index_path fall back
# to $INDEX_DIR, and indexes/{corpus_flashrag.jsonl,e5_Flat.index,bm25} are
# SYMLINKS to indexes_smoke/ -- a 989-document corpus. The run would complete
# normally and produce silver distilled from 989 documents instead of 21M.
# Same block as launch_baselines.sh:23-25; the existing 24,998-record silver was
# built against wiki18, so a rebuild without these is not comparable to it.
# The BM25 copy contains the same 21,015,324 JSON documents in the same id
# order and ships an adjacent corpus.mmindex.json for low-memory random access.
# Using the root JSONL through HuggingFace Dataset reached 52.3 GB RSS and was
# OOM-killed on a 62 GB host before the first query.
export KGPW_CORPUS_PATH=/home/zjulab/kgpaper/indexes_wiki18/bm25/corpus.jsonl
export KGPW_DENSE_INDEX_PATH=/home/zjulab/kgpaper/indexes_wiki18/e5_fp16.dat
export KGPW_BM25_INDEX_PATH=/home/zjulab/kgpaper/indexes_wiki18/bm25
export KGPW_CORPUS_MMAP=1
export KGPW_REQUIRE_BATCH_RETRIEVAL=1
for _p in "$KGPW_CORPUS_PATH" "$KGPW_DENSE_INDEX_PATH" "$KGPW_BM25_INDEX_PATH"; do
  [ -e "$_p" ] || { echo "FATAL: missing $_p -- run scripts/prepare/07_fetch_wiki18.sh" >&2; exit 1; }
done

OUTDIR=/home/zjulab/kgpaper/data/silver_data
TIMESTAMP=$(date +%Y%m%d_%H%M)
LOGDIR="${OUTDIR}/logs/${TIMESTAMP}"
mkdir -p "$LOGDIR"
python scripts/prepare/check_wiki18_assets.py \
  --corpus "$KGPW_CORPUS_PATH" \
  --dense "$KGPW_DENSE_INDEX_PATH" \
  --bm25 "$KGPW_BM25_INDEX_PATH" \
  --output "$LOGDIR/wiki18_asset_preflight.json"

echo "=== Phase 1/3: HotpotQA (7405 questions, 70%) ==="
python scripts/train/phase1_generate_silver.py \
  --config configs/training/phase1_silver.yaml \
  --dataset hotpotqa \
  --split train \
  --max_queries 7405 \
  --sample_strategy random \
  --rerank 10 \
  --offline on \
  --output "${OUTDIR}/silver_hotpotqa_${TIMESTAMP}.jsonl" \
  --seed 42 \
  > "${LOGDIR}/hotpotqa.log" 2>&1 || {
    tail -50 "${LOGDIR}/hotpotqa.log"
    exit 1
  }
tail -5 "${LOGDIR}/hotpotqa.log"

echo "=== Phase 2/3: 2WikiMultihopQA (1586 questions, 15%) ==="
python scripts/train/phase1_generate_silver.py \
  --config configs/training/phase1_silver.yaml \
  --dataset 2wikimultihopqa \
  --split train \
  --max_queries 1586 \
  --sample_strategy random \
  --rerank 10 \
  --offline on \
  --output "${OUTDIR}/silver_2wiki_${TIMESTAMP}.jsonl" \
  --seed 42 \
  > "${LOGDIR}/2wiki.log" 2>&1 || {
    tail -50 "${LOGDIR}/2wiki.log"
    exit 1
  }
tail -5 "${LOGDIR}/2wiki.log"

echo "=== Phase 3/3: Musique (1586 questions, 15%) ==="
python scripts/train/phase1_generate_silver.py \
  --config configs/training/phase1_silver.yaml \
  --dataset musique \
  --split train \
  --max_queries 1586 \
  --sample_strategy random \
  --rerank 10 \
  --offline on \
  --output "${OUTDIR}/silver_musique_${TIMESTAMP}.jsonl" \
  --seed 42 \
  > "${LOGDIR}/musique.log" 2>&1 || {
    tail -50 "${LOGDIR}/musique.log"
    exit 1
  }
tail -5 "${LOGDIR}/musique.log"

echo "=== Merging ==="
cat "${OUTDIR}/silver_hotpotqa_${TIMESTAMP}.jsonl" \
    "${OUTDIR}/silver_2wiki_${TIMESTAMP}.jsonl" \
    "${OUTDIR}/silver_musique_${TIMESTAMP}.jsonl" \
    > "${OUTDIR}/silver_v6_full_${TIMESTAMP}.jsonl"

TOTAL=$(wc -l < "${OUTDIR}/silver_v6_full_${TIMESTAMP}.jsonl")
echo "Done! Total: $TOTAL trajectories → ${OUTDIR}/silver_v6_full_${TIMESTAMP}.jsonl"

# Quality gate + run manifest. Every attempted item must produce a persisted
# record (accepted or rejected); missing records mean teacher/worker failures and
# must stop a formal run rather than silently changing the dataset mixture.
python3 -c "
import hashlib, json, pathlib, subprocess

specs = [
    (pathlib.Path('${OUTDIR}/silver_hotpotqa_${TIMESTAMP}.jsonl'), 'hotpotqa', 7405),
    (pathlib.Path('${OUTDIR}/silver_2wiki_${TIMESTAMP}.jsonl'), '2wikimultihopqa', 1586),
    (pathlib.Path('${OUTDIR}/silver_musique_${TIMESTAMP}.jsonl'), 'musique', 1586),
]

def md5(path):
    h = hashlib.md5()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

source_stats, data = [], []
for path, dataset, expected in specs:
    rows = []
    with path.open() as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f'FATAL: {path}:{lineno}: invalid JSON: {exc}')
            if row.get('dataset') != dataset:
                raise SystemExit(f'FATAL: {path}:{lineno}: dataset={row.get(\"dataset\")!r}, expected {dataset!r}')
            extra = (row.get('metadata') or {}).get('extra') or {}
            if extra.get('source_split') != 'train':
                raise SystemExit(f'FATAL: {path}:{lineno}: source_split={extra.get(\"source_split\")!r}')
            rows.append(row)
    if len(rows) != expected:
        raise SystemExit(f'FATAL: {path}: wrote {len(rows)}/{expected}; refusing partial merge')
    source_stats.append({
        'path': str(path), 'dataset': dataset, 'split': 'train',
        'records': len(rows), 'accepted': sum(bool(r.get('accepted')) for r in rows),
        'md5': md5(path),
    })
    data.extend(rows)

merged = pathlib.Path('${OUTDIR}/silver_v6_full_${TIMESTAMP}.jsonl')
if len(data) != 10577:
    raise SystemExit(f'FATAL: merged row count {len(data)} != 10577')
acc = [d for d in data if d.get('accepted')]
print(f'Accepted: {len(acc)}/{len(data)} ({100*len(acc)//len(data)}%)')
for ds in ['hotpotqa','2wikimultihopqa','musique']:
    ds_acc = [d for d in acc if d.get('dataset')==ds]
    has_kg = sum(1 for d in ds_acc if 'Knowledge Used: [(' in d.get('teacher_output',''))
    print(f'  {ds}: {len(ds_acc)} accepted, KG ref {has_kg}/{max(1,len(ds_acc))} ({100*has_kg//max(1,len(ds_acc))}%)')

manifest = {
    'experiment_id': 'silver_v6_${TIMESTAMP}',
    'source_split': 'train',
    'seed': 42,
    'sample_strategy': 'random',
    'config': 'configs/training/phase1_silver.yaml',
    'sources': source_stats,
    'merged': {'path': str(merged), 'records': len(data), 'accepted': len(acc), 'md5': md5(merged)},
    'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'git_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()),
}
manifest_path = pathlib.Path('${LOGDIR}/manifest.json')
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
print(f'Manifest: {manifest_path}')
"
