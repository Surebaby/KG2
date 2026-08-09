"""Does triple_in_subgraph verify what the policy cites?

The relevance factor measured 0.556, so the collapse in r_kg (~0.012) has to be
precision: cited triples not matching the subgraph. This replays the real KG
blocks from the real PPO prompts against the citations the policy produced.

Unlike _rkg_diag.py, this reconstructs the subgraph, so precision is measurable.
"""
from __future__ import annotations

import os
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kgproweight.data.parsers import parse_steps
from kgproweight.reward.prm_annotator import triple_in_subgraph

OPEN, CLOSE = "[Knowledge Graph Context]", "[End of Knowledge Graph]"

sample_path = Path(sys.argv[1] if len(sys.argv) > 1
                   else "outputs/split_ppo_smoke/samples/step_00040.txt")

# The sample dump has responses but NOT the prompt each came from, so we cannot
# pair them exactly. Instead measure the question the batch mean actually asks:
# for a citation drawn from these rollouts, how often does it verify against the
# KG block of ANY prompt in the pool? A near-zero rate under the loose "any"
# test proves the matcher, not the pairing, is what fails.
from kgproweight.config import ProjectConfig, load_config
from kgproweight.data import SilverDatasetReader
from kgproweight.data.silver_split import SplitSpec
from kgproweight.training.phase3_ppo import Phase3PPOConfig, _prepare_prompts
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
tcfg = load_config(str(ROOT / "configs/training/phase3_ppo.yaml"),
                   validate=ProjectConfig).training
cfg = Phase3PPOConfig(silver_path="", output_dir="",
                      max_input_length=getattr(tcfg, "max_input_length", 4096))

sft = os.environ.get("KGPW_SFT", str(ROOT / "checkpoints/sft_student_split/final"))
tok = AutoTokenizer.from_pretrained(sft, trust_remote_code=True)

silver = os.environ.get("KGPW_SILVER",
                        str(ROOT / "data/silver_data/silver_v1_reannotated.jsonl"))
spec = SplitSpec(val_ratio=tcfg.val_ratio, test_ratio=tcfg.test_ratio,
                 seed=tcfg.split_seed)
reader = SilverDatasetReader(silver, split="train", split_spec=spec)

acc = list(reader.accepted())
random.seed(42)
keep = {(t.qid, t.question) for t in random.sample(acc, min(300, len(acc)))}
reader.trajectories = [t for t in reader.trajectories
                       if (t.qid, t.question) in keep]

import json
from kgproweight.utils.paths import index_dir

q_kg = {}
p = Path(index_dir()) / "kg_cache" / "question_kg_index_v2.json"
if not p.exists():
    p = Path(index_dir()) / "kg_cache" / "question_kg_index.json"
if p.exists():
    raw = json.loads(p.read_text(encoding="utf-8"))
    v2 = "builder_version" in (raw[0] if raw else {})
    for e in raw:
        q = e.get("question", e.get("q", ""))
        q_kg[q] = ([(t["h"], t["r"], t["t"]) for t in e["triples"]] if v2
                   else e["t"])

rows = _prepare_prompts(reader, tok, cfg, question_kg_index=q_kg or None)

# Pool every triple that appears in any prompt's KG block.
pool: list[tuple[str, str, str]] = []
for r in rows:
    pr = r["prompt"]
    if OPEN not in pr:
        continue
    body = pr.split(OPEN, 1)[1].split(CLOSE, 1)[0]
    for ln in body.split("\n"):
        ln = ln.strip().strip("()")
        parts = [x.strip() for x in ln.split(",")]
        if len(parts) >= 3:
            pool.append((parts[0], parts[1], ",".join(parts[2:])))
print(f"KG pool: {len(pool)} triples from {len(rows)} prompts")

# Also keep each prompt's own block, to compare paired vs pooled matching.
per_prompt = {}
for r in rows:
    pr = r["prompt"]
    if OPEN not in pr:
        continue
    body = pr.split(OPEN, 1)[1].split(CLOSE, 1)[0]
    tl = []
    for ln in body.split("\n"):
        ln = ln.strip().strip("()")
        parts = [x.strip() for x in ln.split(",")]
        if len(parts) >= 3:
            tl.append((parts[0], parts[1], ",".join(parts[2:])))
    per_prompt[r["spec"].query] = tl

text = sample_path.read_text(encoding="utf-8", errors="replace")
blocks = text.split("--- Sample ")[1:]
cited: list[tuple[str, str, str]] = []
for blk in blocks:
    body = blk.split("\n", 1)[1] if "\n" in blk else ""
    for st in parse_steps(body):
        cited.extend(st.cited_triples)
print(f"policy citations: {len(cited)}\n")

hits = sum(1 for c in cited if triple_in_subgraph(c, pool, fuzzy_threshold=0.85))
print(f"verified against the POOLED subgraph ({len(pool)} triples):")
print(f"  {hits}/{len(cited)} = {100.0 * hits / max(1, len(cited)):.1f}%")
print("  (a loose upper bound -- the real reward pairs one prompt to one rollout)")

# Sensitivity to the fuzzy threshold: if a lower threshold recovers most of the
# citations, the matcher is too strict rather than the citations being wrong.
print("\nsensitivity to fuzzy_threshold:")
for th in (0.95, 0.85, 0.75, 0.60, 0.50):
    h = sum(1 for c in cited if triple_in_subgraph(c, pool, fuzzy_threshold=th))
    print(f"  {th:.2f} -> {h:4d}/{len(cited)} ({100.0 * h / max(1, len(cited)):5.1f}%)")

miss = [c for c in cited if not triple_in_subgraph(c, pool, fuzzy_threshold=0.85)]
print(f"\n{len(miss)} unverified citations, first 8:")
for c in miss[:8]:
    print(f"  {c}")
