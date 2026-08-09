#!/usr/bin/env bash
# ===========================================================================
# Fetch the wiki18 corpus + PREBUILT e5 Flat index.
#
#   bash scripts/prepare/07_fetch_wiki18.sh
#
# WHY prebuilt: encoding 21M passages with e5 would take 20-30 h on this
# RTX 4090 (01_build_dense_index.sh's own header says ~6 h for 15M on a
# Pro 6000 96 GB at batch 1024; a 24 GB card must quarter that batch).
# PeterJinGo/wiki-18-e5-index is the same e5-base-v2 / IndexFlatIP artefact,
# already built, so this skips the encode entirely.
#
# DOWNLOAD MECHANICS, measured 2026-08-07 on this box:
#   single curl connection ....  6.0-7.7 MB/s
#   6 parallel ranged curls ... 15.7 MB/s aggregate
# So parallelism buys 2.3x, not 6x — we are near the link ceiling. 70 GB at
# 15.7 MB/s = ~1.2 h. More than 6 streams is unlikely to help.
#
# Two tools that do NOT work here, do not retry them:
#   - huggingface_hub (hf_hub_download): raises LocalEntryNotFoundError even
#     though `requests.head` on the same URL returns 200 with a correct
#     content-length. Fails with hf_transfer on/off and HF_HUB_DISABLE_XET=1.
#   - aria2c: hf-mirror answers 308 -> huggingface.co, and aria2c then dies
#     with "SSL/TLS handshake failure: The TLS connection was non-properly
#     terminated." curl follows the same redirect fine.
# curl with byte ranges is what actually works, hence the loop below.
#
# MEMORY — the reason this is worth doing at all. The index is 64.6 GB of
# fp32 (21M x 768 x 4 B) and this box has 62 GB total / ~56 GB available, so
# it does NOT fit in RAM. It is still usable because faiss can mmap it:
#   faiss.read_index(path, faiss.IO_FLAG_MMAP)
# Verified on the smoke index — RSS grew 0.004 GB and search returned
# correct results. Requires the mmap patch to retriever.py:381; without it
# read_index loads eagerly and this WILL OOM.
#
# Rejected alternative, kept because the measurement is useful: converting
# fp32 -> IndexScalarQuantizer QT_fp16 halves it to ~33 GB and was lossless
# on the smoke index (top-10 overlap 1.000, top-1 1.000, max distance delta
# 0.00000). Not used because the conversion must first read all 64.6 GB into
# RAM to call add() — the exact thing we cannot do. A chunked converter would
# work but adds a moving part, and mmap already solves it at no precision cost.
#
# DISK: needs ~70 GB (5.1 corpus + 64.6 index) and 132 GB is free, leaving
# ~62 GB. Check df before starting; the index lands as two ~43/22 GB parts
# that are then concatenated, so peak usage is the same as final (cat streams).
# ===========================================================================
set -euo pipefail

DEST="${KGPW_WIKI18_DIR:-/home/zjulab/kgpaper/indexes_wiki18}"
STREAMS="${STREAMS:-6}"
MIRROR="${HF_ENDPOINT:-https://hf-mirror.com}"
CORPUS_REPO="datasets/PeterJinGo/wiki-18-corpus"
INDEX_REPO="datasets/PeterJinGo/wiki-18-e5-index"

mkdir -p "$DEST"
avail=$(df -BG --output=avail "$DEST" | tail -1 | tr -dc '0-9')
if [ "$avail" -lt 80 ]; then
  echo "ERROR: only ${avail} GB free at $DEST, need ~80 GB with slack"; exit 1
fi
echo "disk free: ${avail} GB   dest: $DEST   streams: $STREAMS"
echo

# Parallel ranged fetch of one file. curl is the only client that survives the
# mirror's 308; see header. Ranges are split evenly and concatenated in order.
fetch () {
  local url="$1" out="$2"
  if [ -f "$out" ]; then echo "  exists, skipping: $(basename "$out")"; return 0; fi
  local size
  size=$(python3 -c "
import requests,sys
r=requests.head('$url',allow_redirects=True,timeout=30)
r.raise_for_status()
print(r.headers['content-length'])
")
  echo "  $(basename "$out"): $(python3 -c "print(f'{$size/1e9:.2f} GB')")"
  local chunk=$(( (size + STREAMS - 1) / STREAMS ))
  local pids=()
  for i in $(seq 0 $((STREAMS-1))); do
    local off=$(( i * chunk )) end=$(( (i+1) * chunk - 1 ))
    [ "$end" -ge "$size" ] && end=$(( size - 1 ))
    [ "$off" -gt "$end" ] && continue

    # Resume by ADJUSTING THE RANGE, never with `-C -`. Measured 2026-08-07:
    # `-C - -r 0-20971519` on a 20 MB part produced a 213 MB file, because
    # -C - rewrites the header to `Range: <have>-` and drops the upper bound,
    # so curl downloads to end-of-file and appends. Parts silently overshoot
    # and the concatenated index is byte-misaligned.
    local part="${out}.part${i}" have=0
    [ -f "$part" ] && have=$(stat -c%s "$part")
    local want=$(( end - off + 1 ))
    if [ "$have" -ge "$want" ]; then
      # Already complete (or overshot from an earlier buggy run) — truncate to
      # exactly the wanted length so a bad part can never poison the assembly.
      truncate -s "$want" "$part"
      continue
    fi
    # Append only the missing tail: shift the start, keep the hard upper bound.
    curl -sSL --retry 5 --retry-delay 3 -r "$(( off + have ))-${end}" \
         -o - "$url" >> "$part" &
    pids+=($!)
  done
  local rc=0
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  [ "$rc" -eq 0 ] || { echo "  FAILED: a stream errored, parts kept for resume"; return 1; }
  cat "${out}".part* > "$out"
  local got
  got=$(stat -c%s "$out")
  if [ "$got" -ne "$size" ]; then
    echo "  SIZE MISMATCH: got $got want $size — keeping parts"; return 1
  fi
  rm -f "${out}".part*
  echo "  done: $(basename "$out")"
}

echo "[1/3] corpus (5.1 GB)"
fetch "$MIRROR/$CORPUS_REPO/resolve/main/wiki-18.jsonl.gz" "$DEST/wiki-18.jsonl.gz"

echo "[2/3] prebuilt e5 Flat index, 2 parts (64.6 GB total)"
fetch "$MIRROR/$INDEX_REPO/resolve/main/part_aa" "$DEST/part_aa"
fetch "$MIRROR/$INDEX_REPO/resolve/main/part_ab" "$DEST/part_ab"

echo "[3/3] assembling e5_Flat.index"
# Do NOT `cat part_aa part_ab > index` — that leaves the parts in place, so two
# copies of 64.6 GB coexist: 64.6 (parts) + 64.6 (index) + 5.1 (corpus) + ~13
# (extracted jsonl) = 147 GB against 132 GB free. It would fail at the very last
# step, after downloading all 70 GB.
# Instead rename part_aa into place and append part_ab, deleting it right after.
# Peak is then 86.2 GB (index 64.6 + part_ab 21.6), total ~104 GB with corpus
# and extracted jsonl. Trade-off: the parts do not survive, so a failure here
# means re-downloading — acceptable, they have no independent use.
if [ ! -f "$DEST/e5_Flat.index" ]; then
  mv "$DEST/part_aa" "$DEST/e5_Flat.index"
  cat "$DEST/part_ab" >> "$DEST/e5_Flat.index"
  rm -f "$DEST/part_ab"
  echo "  assembled: $(du -h "$DEST/e5_Flat.index" | cut -f1)  (parts consumed)"
else
  echo "  exists, skipping"
fi

echo
echo "downloaded. NEXT, in this order:"
echo "  1. verify it opens under mmap WITHOUT loading 64.6 GB into RAM:"
echo "       python3 -c \"import faiss,resource;"
echo "       ix=faiss.read_index('$DEST/e5_Flat.index', faiss.IO_FLAG_MMAP);"
echo "       print(ix.ntotal, ix.d, type(ix).__name__,"
echo "       resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6, 'GB RSS')\""
echo "     Expect ~21M x 768, IndexFlatIP, RSS well under 1 GB. If RSS jumps to"
echo "     tens of GB the mmap flag is not taking effect — stop, do not proceed."
echo "  2. convert the corpus to FlashRAG format (id/contents):"
echo "       python scripts/prepare/00_convert_corpus.py --input <decompressed jsonl>"
echo "     Row order MUST match the index; the index was built from this corpus."
echo "  3. patch retriever.py:381 to pass IO_FLAG_MMAP (env-gated so the smoke"
echo "     index keeps its current eager-load behaviour)."
echo "  4. TIME a small run FIRST (n=20). Every query scans 64.6 GB from NVMe;"
echo "     the page cache holds ~51 GB so steady state should be mostly cached,"
echo "     but the first queries will be slow. If n=20 extrapolates badly,"
echo "     revisit chunked fp16 (~33 GB, fits in RAM) before running n=300."
