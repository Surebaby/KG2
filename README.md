# KG-ProWeight

> **Adaptive Process Supervision for Agentic RAG via Knowledge-Graph-Constrained Distillation and Dynamic Confidence Weighting.**

Reference implementation accompanying the paper *KG-ProWeight*. The repository
provides:

1. A self-contained Python package `kgproweight` that implements every method
   component described in [`docs/paper_design.md`](docs/paper_design.md).
2. A reproducible end-to-end pipeline (Phase 1 / Phase 2 / Phase 3) that runs
   on a single **RTX PRO 6000 Blackwell (96 GB)** GPU in `bf16`.
3. Evaluation runners for all baselines under a unified hybrid retrieval
   (E5 + BM25 → RRF top-50) configuration, together with the paper's ablation
   studies and rigorous-extension experiments (IHR LLM-as-Judge, multi-seed
   significance testing, data efficiency curve, theorem-2 empirical
   verification, α distribution analysis).

The legacy implementation lives in [`../kg2`](../kg2) (operational guide only —
do **not** depend on it for new experiments). This refactor preserves every
result-producing component while:

- fixing 14 semantic bugs documented in [`docs/refactor_notes.md`](docs/refactor_notes.md);
- enforcing a single prompt schema across Teacher / SFT / PPO / inference;
- adding the rigour pieces required by the paper (LLM-as-Judge IHR, paired
  bootstrap, multi-seed, data-efficiency curve, theorem-2 variance log).

---

## 1. Quickstart (Pro 6000 Blackwell · 96 GB)

```bash
# 1) Clone (and FlashRAG, an external dependency)
git clone <your repo url> kgpaper
cd kgpaper
git clone https://github.com/RUC-NLPIR/FlashRAG.git third_party/FlashRAG

# 2) Create env & install deps
conda create -n kgpw python=3.10 -y && conda activate kgpw
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
pip install -e third_party/FlashRAG
pip install -e ".[dev]"

# 3) Environment variables (see .env.example)
export KGPW_PROJECT_ROOT=$(pwd)
export KGPW_DATA_DIR=$KGPW_PROJECT_ROOT/data
export KGPW_INDEX_DIR=$KGPW_PROJECT_ROOT/indexes
export KGPW_CHECKPOINT_DIR=$KGPW_PROJECT_ROOT/checkpoints
export KGPW_OUTPUT_DIR=$KGPW_PROJECT_ROOT/outputs
export KGPW_FLASHRAG_ROOT=$KGPW_PROJECT_ROOT/third_party/FlashRAG
export KGPW_LLAMA3_PATH=meta-llama/Meta-Llama-3-8B-Instruct
export KGPW_E5_PATH=intfloat/e5-base-v2
export KGPW_REARAG_PATH=THU-KEG/ReaRAG-9B
export OPENAI_API_KEY=sk-...   # required for IHR LLM-as-Judge
export DEEPSEEK_API_KEY=sk-... # optional Teacher backend

# 4) Build corpora and indices (one-time)
make prepare-corpus
make build-dense-index            # ~6h on Pro 6000 96GB
make build-bm25-index             # ~1h
make download-datasets

# 5) End-to-end training (Phase 1 → 2 → 3)
make phase1                       # ~6h with --max_workers 8 (DeepSeek-V3 backend)
make phase2                       # ~3h
make phase3-sft                   # ~2h
make phase3-ppo                   # ~10h

# 6) Evaluation
make dropout                      # build D_dropout robustness set
make eval-baselines               # ~24h serial; can parallelise across GPUs
make eval-kgpw                    # 4 datasets + d_dropout
make eval-ablations               # five paper ablations

# 7) Rigor extensions
make rigor-ihr                    # GPT-4o IHR + Cohen κ vs heuristic
make rigor-data-eff               # F1-vs-N learning curve
make rigor-variance               # advantage variance under fixed vs dynamic α

# 8) Aggregate paper artefacts
make summarize
```

All outputs land under `outputs/` and `checkpoints/`; both directories are
gitignored.

---

## 2. Project structure

```
kgpaper/
├── kgproweight/              # importable Python package
│   ├── config/               # YAML+CLI loader + pydantic schemas
│   ├── kg/                   # entity linker / Wikidata SPARQL / KG embeddings
│   ├── reward/               # alpha-gate / PRM / text reward / IHR judge
│   ├── data/                 # prompts, silver datasets, parsers, dropout loader
│   ├── retrieval/            # hybrid RRF top-50 setting
│   ├── pipeline/             # FlashRAG pipeline subclasses
│   ├── training/             # phase1 distillation / phase2 PRM / phase3 SFT+PPO
│   ├── eval/                 # metrics, alpha analysis, stats, variance, data-eff
│   └── utils/                # paths, seeds, logging, FlashRAG bootstrap
│
├── configs/                  # YAML configs for every command
├── scripts/{prepare,train,eval,utils}/  # thin CLI wrappers around the package
├── docs/                     # paper design, operation guide, architecture
└── tests/                    # pytest suite (alpha-gate, PRM annotator, …)
```

See [`docs/architecture.md`](docs/architecture.md) for the full dataflow.

---

## 3. Documentation

| File | Purpose |
|------|---------|
| [`docs/paper_design.md`](docs/paper_design.md) | Full methodology (formerly `KG-ProWeight_paper_design.txt`). |
| [`docs/operation_guide.md`](docs/operation_guide.md) | Step-by-step commands tuned for Pro 6000 96 GB. |
| [`docs/architecture.md`](docs/architecture.md) | Module diagram + dataflow (mermaid). |
| [`docs/rigor_extensions.md`](docs/rigor_extensions.md) | Purpose of every rigour-extension script. |
| [`docs/reproducibility.md`](docs/reproducibility.md) | Seed / version / API-pinning policy. |
| [`docs/refactor_notes.md`](docs/refactor_notes.md) | Mapping kg2 → kgpaper with each fixed bug. |

---

## 4. Citation

```bibtex
@article{kgproweight2026,
  title  = {KG-ProWeight: Adaptive Process Supervision for Agentic RAG via Knowledge-Graph-Constrained Distillation and Dynamic Confidence Weighting},
  author = {Anonymous},
  year   = {2026}
}
```

---

## 5. License

Apache 2.0. See [`LICENSE`](LICENSE).
