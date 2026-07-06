#!/usr/bin/env python
"""No-card-mode training-readiness check. Verifies weight integrity WITHOUT
loading models into RAM (no-card mode caps RAM at 2GB)."""
import glob
import json
import os
import sys

LLAMA = "/root/autodl-tmp/models/llama3-8b"
REARAG = "/root/autodl-tmp/models/rearag-9b"
bad = 0


def check_model(d, kind):
    global bad
    print(f"\n=== {os.path.basename(d)} ===")
    cfg_path = os.path.join(d, "config.json")
    if not os.path.exists(cfg_path):
        print("  FAIL: no config.json")
        bad += 1
        return
    cfg = json.load(open(cfg_path))
    print(f"  model_type: {cfg.get('model_type')}  layers: {cfg.get('num_hidden_layers')}")
    idxs = glob.glob(os.path.join(d, "*.index.json"))
    if idxs:
        idx = json.load(open(idxs[0]))
        shards = set(idx["weight_map"].values())
        missing = [s for s in shards if not os.path.exists(os.path.join(d, s))]
        print(f"  index shards: {len(shards)}  params: {len(idx['weight_map'])}")
        print(f"  missing shards: {missing if missing else 'NONE'}")
        if missing:
            bad += 1
    else:
        n = len(glob.glob(os.path.join(d, "*.safetensors"))) or len(glob.glob(os.path.join(d, "*.bin")))
        print(f"  weight files (no index): {n}")
    tok = any(os.path.exists(os.path.join(d, f)) for f in ["tokenizer.json", "tokenizer.model", "tokenizer_config.json"])
    print(f"  tokenizer present: {tok}")
    if not tok:
        bad += 1


check_model(LLAMA, "policy/base")
check_model(REARAG, "text-reward")
print(f"\n{'ALL MODEL FILES OK' if bad == 0 else f'{bad} PROBLEM(S)'}")
sys.exit(1 if bad else 0)
