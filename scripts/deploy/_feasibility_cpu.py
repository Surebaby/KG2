#!/usr/bin/env python
"""Phase 3b PPO feasibility check that needs NO GPU.

Runs on an AutoDL no-card instance. It exercises every step of the PPO setup
that is CPU-reachable, so the paid GPU session starts with these already known:

  1. config loads and the R10 values are live
  2. the silver train fold loads and the accepted pool is the expected size
  3. the Q->KG index covers those questions
  4. prompts BUILD, and their real token lengths sit under max_input_length
     -- this pre-answers smoke-check #1 (KG right-truncation) without a card
  5. the tokenizer/chat template of the SFT checkpoint is loadable and its
     LoRA adapter points at a base model that is present on disk
  6. disk headroom is enough for the checkpoints the schedule will write

It deliberately does NOT load model weights: that is the part which needs the
GPU. Exit code is nonzero if any hard check fails.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

FAIL: list[str] = []
WARN: list[str] = []


def ok(msg: str) -> None:
    print(f"  ok    {msg}")


def bad(msg: str) -> None:
    print(f"  FAIL  {msg}")
    FAIL.append(msg)


def warn(msg: str) -> None:
    print(f"  warn  {msg}")
    WARN.append(msg)


# ---------------------------------------------------------------- 1. config
print("=== 1. config + R10 values ===")
from kgproweight.config import load_config  # noqa: E402
from kgproweight.training.phase3_ppo import Phase3PPOConfig  # noqa: E402

from kgproweight.config import ProjectConfig  # noqa: E402

cfg_path = ROOT / "configs/training/phase3_ppo.yaml"
tcfg = load_config(str(cfg_path), validate=ProjectConfig).training
ppo = tcfg.ppo  # NOT tcfg.phase3_ppo -- match scripts/train/phase3_ppo.py:64

EXPECT = {
    "total_ppo_steps": 16000,
    "save_every_steps": 2000,
    "target_kl": 40.0,
    "outcome_weight": 4.0,
    "step_reward_scale": 1.5,
}
for k, want in EXPECT.items():
    got = getattr(ppo, k, None)
    (ok if got == want else bad)(f"{k} = {got} (want {want})")

# Assemble the config the way scripts/train/phase3_ppo.py does, so section 5
# measures against the REAL caps. The bare dataclass has max_input_length=4096
# and batch_size=64; the YAML raises them to 6144 and lowers bs to 8, and
# measuring prompt lengths against 4096 would invent a truncation problem.
cfg = Phase3PPOConfig(
    silver_path="", output_dir="",
    batch_size=ppo.batch_size,
    total_steps=ppo.total_ppo_steps,
    save_every_steps=ppo.save_every_steps,
    outcome_weight=ppo.outcome_weight,
    step_reward_scale=getattr(ppo, "step_reward_scale", 1.0),
    target_kl=ppo.target_kl,
    max_input_length=getattr(tcfg, "max_input_length", 4096),
)

# ppo_max_kg_triples lives on the dataclass, not the YAML (schemas.py extra=allow
# would silently swallow a YAML key here -- see the NOTE in phase3_ppo.yaml).
(ok if cfg.ppo_max_kg_triples == 30 else bad)(
    f"ppo_max_kg_triples = {cfg.ppo_max_kg_triples} (want 30)")
(ok if cfg.max_input_length == 6144 else bad)(
    f"max_input_length = {cfg.max_input_length} (want 6144, = SFT max_length)")

print(f"  info  schedule: {cfg.total_steps} traj / bs {cfg.batch_size} "
      f"= {cfg.total_steps // cfg.batch_size} optimiser updates")
print(f"  info  ppo_max_passages={cfg.ppo_max_passages} "
      f"ppo_max_kg_triples={cfg.ppo_max_kg_triples}")

# ------------------------------------------------------- 2. silver train fold
print("\n=== 2. silver train fold ===")
from kgproweight.data import SilverDatasetReader  # noqa: E402
from kgproweight.data.silver_split import SplitSpec  # noqa: E402

silver = Path(os.environ.get(
    "KGPW_SILVER", ROOT / "data/silver_data/silver_v1_reannotated.jsonl"))
if not silver.exists():
    bad(f"silver data missing: {silver}")
    print("\nCannot continue without the silver file.")
    sys.exit(1)

spec = SplitSpec(val_ratio=tcfg.val_ratio, test_ratio=tcfg.test_ratio,
                 seed=tcfg.split_seed)
reader = SilverDatasetReader(silver, split="train", split_spec=spec)
accepted = list(reader.accepted())
ok(f"train fold: {len(reader.trajectories)} trajectories, {len(accepted)} accepted")
if len(accepted) != 7913:
    warn(f"accepted = {len(accepted)}, manifest recorded 7913 -- split drifted?")

n_gold = sum(1 for t in accepted
             if str(t.metadata.get("gold_answer") or "").strip())
(ok if n_gold == len(accepted) else warn)(
    f"gold_answer present on {n_gold}/{len(accepted)} "
    f"({len(accepted) - n_gold} would be skipped at A4)")

# ----------------------------------------------------------- 3. Q->KG index
print("\n=== 3. Q->KG index coverage ===")
from kgproweight.utils.paths import index_dir  # noqa: E402

q_kg: dict = {}
kg_path = Path(index_dir()) / "kg_cache" / "question_kg_index_v2.json"
is_v2 = kg_path.exists()
if not is_v2:
    kg_path = Path(index_dir()) / "kg_cache" / "question_kg_index.json"

if kg_path.exists():
    # The index is a LIST of entries, not a dict keyed by question. Parse it
    # exactly as run_phase3_ppo does (phase3_ppo.py:714-724), and detect v2 by
    # the builder_version marker on the first entry rather than by filename.
    raw = json.loads(kg_path.read_text(encoding="utf-8"))
    is_v2 = "builder_version" in (raw[0] if raw else {})
    for entry in raw:
        q = entry.get("question", entry.get("q", ""))
        if is_v2:
            q_kg[q] = [(t["h"], t["r"], t["t"]) for t in entry["triples"]]
        else:
            q_kg[q] = entry["t"]
    ok(f"index v{'2' if is_v2 else '1'}: {len(q_kg)} questions ({kg_path.name})")
    hits = sum(1 for t in accepted if q_kg.get(t.question))
    pct = 100.0 * hits / max(1, len(accepted))
    (ok if pct >= 95 else warn)(
        f"covers {hits}/{len(accepted)} of the train fold ({pct:.1f}%)")
    sizes = [len(q_kg[t.question]) for t in accepted if q_kg.get(t.question)]
    if sizes:
        sizes.sort()
        print(f"  info  triples/question: median {sizes[len(sizes) // 2]}, "
              f"p95 {sizes[int(0.95 * (len(sizes) - 1))]}, max {sizes[-1]}")
else:
    bad(f"no Q->KG index at {kg_path.parent} -- every prompt falls back to "
        "the raw silver kg_subgraph")

# ------------------------------------- 4. SFT checkpoint: tokenizer + adapter
print("\n=== 4. SFT anchor: tokenizer + LoRA adapter (no weights loaded) ===")
sft = Path(os.environ.get("KGPW_SFT", ROOT / "checkpoints/sft_student_split/final"))
if not sft.is_dir():
    bad(f"SFT checkpoint missing: {sft}")
    print("\nCannot build prompts without a tokenizer.")
    sys.exit(1)

base_path = None
adapter_cfg = sft / "adapter_config.json"
if adapter_cfg.exists():
    base_path = json.loads(adapter_cfg.read_text()).get("base_model_name_or_path")
    ok(f"LoRA adapter -> base {base_path}")
    if base_path and Path(base_path).is_dir():
        n_shard = len(list(Path(base_path).glob("*.safetensors")))
        gb = sum(f.stat().st_size for f in Path(base_path).rglob("*")
                 if f.is_file()) / 1e9
        (ok if gb > 1 else warn)(
            f"base model on disk: {gb:.1f} GB, {n_shard} safetensors shards")
    elif base_path:
        warn(f"base model path is not a local dir: {base_path} "
             "(PPO would try to download it)")
else:
    warn("no adapter_config.json -- treating as a full checkpoint")

from transformers import AutoTokenizer  # noqa: E402

tok = None
for cand in [str(sft), base_path]:
    if not cand:
        continue
    try:
        tok = AutoTokenizer.from_pretrained(cand, trust_remote_code=True)
        ok(f"tokenizer loaded from {cand}")
        break
    except Exception as exc:  # noqa: BLE001
        warn(f"tokenizer load failed from {cand}: {type(exc).__name__}: {exc}")
if tok is None:
    bad("no tokenizer could be loaded")
    sys.exit(1)
(ok if hasattr(tok, "apply_chat_template") and tok.chat_template else warn)(
    "chat template present" if getattr(tok, "chat_template", None)
    else "no chat template -- prompts fall back to plain join")

# ------------------------------ 5. prompt lengths vs max_input_length (KEY)
# This is smoke-check #1, answered without a GPU. run_phase3_ppo warns per
# rollout when a prompt exceeds max_input_length, because right-truncation
# drops the trailing KG block. Here we measure the whole fold up front.
print("\n=== 5. prompt token lengths vs max_input_length (pre-answers smoke #1) ===")
from kgproweight.training.phase3_ppo import _prepare_prompts  # noqa: E402

# Build prompts from a random subset of the SAME accepted pool PPO would use.
# Sampling the reader (not the output) keeps chat-template work proportional to
# KGPW_FEAS_N -- the full 7913 is fine but slow on a no-card CPU.
import random  # noqa: E402

n_sample = int(os.environ.get("KGPW_FEAS_N", "400"))
random.seed(42)
sub = SilverDatasetReader(silver, split="train", split_spec=spec)
if len(accepted) > n_sample:
    # qid is not unique on its own (one question can yield several accepted
    # trajectories), so key on (qid, question) and keep whatever it matches.
    keep = {(t.qid, t.question) for t in random.sample(accepted, n_sample)}
    sub.trajectories = [t for t in sub.trajectories
                        if (t.qid, t.question) in keep]

rows = _prepare_prompts(sub, tok, cfg, question_kg_index=q_kg or None)
print(f"  info  {len(rows)} prompts built (sampled from {len(accepted)} accepted)")

probe = rows
if not probe:
    bad("_prepare_prompts returned 0 rows -- PPO would have nothing to train on")
    print("\n" + "=" * 60)
    print("FEASIBILITY: HARD FAILURE -- no prompts built")
    sys.exit(1)

lens = [len(tok(r["prompt"], truncation=False)["input_ids"]) for r in probe]
lens.sort()
cap = cfg.max_input_length
over = [n for n in lens if n > cap]
p50, p95, p99 = (lens[len(lens) // 2], lens[int(0.95 * (len(lens) - 1))],
                 lens[int(0.99 * (len(lens) - 1))])
print(f"  info  n={len(lens)}  median {p50}  p95 {p95}  p99 {p99}  max {lens[-1]}"
      f"  cap {cap}")
if over:
    bad(f"{len(over)}/{len(lens)} prompts exceed max_input_length={cap} "
        f"(worst {lens[-1]}, +{lens[-1] - cap}) -- KG block WILL be truncated")
else:
    ok(f"0/{len(lens)} exceed the cap; headroom at max is {cap - lens[-1]} tokens")

# The KG block must survive with room for the generation anchor after it.
# RL_USER_TEMPLATE = SFT_USER_TEMPLATE, so the block is delimited by
# "[Knowledge Graph Context]" ... "[End of Knowledge Graph]" (prompts.py:120).
KG_OPEN, KG_CLOSE = "[Knowledge Graph Context]", "[End of Knowledge Graph]"
kg_present = sum(1 for r in probe if KG_OPEN in r["prompt"])
(ok if kg_present == len(probe) else bad)(
    f"KG block delimiters present in {kg_present}/{len(probe)} prompts")

# Count triples actually inside the block. A present-but-empty block would mean
# the KG channel has nothing to reward, which no length check would reveal.
tri_counts = []
for r in probe:
    p = r["prompt"]
    if KG_OPEN not in p or KG_CLOSE not in p:
        continue
    body = p.split(KG_OPEN, 1)[1].split(KG_CLOSE, 1)[0]
    # One triple per line. Do NOT count "(" -- entity and relation labels
    # frequently contain parentheses ("rel (x)", "Foo (disambiguation)"), which
    # overcounts and invented a bogus "61 blocks exceed the cap" warning.
    tri_counts.append(len([ln for ln in body.split("\n") if ln.strip()]))
if tri_counts:
    tri_counts.sort()
    empty = sum(1 for c in tri_counts if c == 0)
    print(f"  info  triples in block: median {tri_counts[len(tri_counts) // 2]}, "
          f"max {tri_counts[-1]}, empty {empty}/{len(tri_counts)}")
    (ok if empty == 0 else warn)(
        "every prompt carries at least one triple" if empty == 0
        else f"{empty} prompts have an EMPTY KG block -- no KG signal to reward")
    over_cap = sum(1 for c in tri_counts if c > cfg.ppo_max_kg_triples)
    (ok if over_cap == 0 else warn)(
        f"all blocks within ppo_max_kg_triples={cfg.ppo_max_kg_triples}"
        if over_cap == 0 else
        f"{over_cap} blocks exceed the {cfg.ppo_max_kg_triples} cap")
(ok if kg_present == len(probe) else warn)(
    f"KG block present in {kg_present}/{len(probe)} prompts")

# max_input_length + max_new_tokens is the real ceiling PPO tokenizes to.
print(f"  info  rollout ceiling: {cap} + {cfg.max_new_tokens} new "
      f"= {cap + cfg.max_new_tokens} tokens")

# ------------------------------------------------------------- 6. disk budget
print("\n=== 6. disk headroom for the checkpoint schedule ===")
n_ckpt = cfg.total_steps // cfg.save_every_steps
adapter_mb = 0.0
if adapter_cfg.exists():
    adapter_mb = sum(f.stat().st_size for f in sft.rglob("*") if f.is_file()) / 1e6
free_gb = shutil.disk_usage(ROOT).free / 1e9
need_gb = n_ckpt * adapter_mb / 1e3
print(f"  info  {n_ckpt} checkpoints x {adapter_mb:.0f} MB = {need_gb:.1f} GB; "
      f"{free_gb:.0f} GB free")
(ok if free_gb > need_gb * 2 + 5 else warn)(
    "headroom fine" if free_gb > need_gb * 2 + 5 else "disk may be tight")

# ------------------------------------------------------------------- verdict
print("\n" + "=" * 60)
if FAIL:
    print(f"FEASIBILITY: {len(FAIL)} HARD FAILURE(S) -- fix before opening a card")
    for m in FAIL:
        print(f"  - {m}")
    sys.exit(1)
print("FEASIBILITY: PASS -- everything CPU-reachable is in order.")
if WARN:
    print(f"{len(WARN)} warning(s) worth a look:")
    for m in WARN:
        print(f"  - {m}")
print("\nStill needs a GPU (cannot be checked here): model weight loading, "
      "peak VRAM, rollout sampling, objective/kl, actual kg_reward_share.")
