#!/usr/bin/env bash
# KG-ProWeight — environment setup for Phase 2/3 on a fresh AutoDL box.
#
# Usage:
#   bash scripts/deploy/setup_env.sh          # auto-detect GPU arch, install matching torch
#   GPU_ARCH=blackwell bash scripts/deploy/setup_env.sh   # force a torch build
#
# After this finishes: configure .env, then `python scripts/deploy/preflight.py`.
set -euo pipefail

ENV_NAME="${ENV_NAME:-kgpw}"
PY_VER="${PY_VER:-3.10}"

echo "==> 1/5  conda env '${ENV_NAME}' (python ${PY_VER})"
if ! conda env list | grep -qE "^${ENV_NAME}\s"; then
  conda create -n "${ENV_NAME}" "python=${PY_VER}" -y
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# ── Detect GPU architecture to pick the right torch build ───────────────────
echo "==> 2/5  detecting GPU"
ARCH="${GPU_ARCH:-}"
if [[ -z "${ARCH}" ]]; then
  NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo '')"
  echo "    GPU: ${NAME:-unknown}"
  if echo "${NAME}" | grep -qiE "PRO 6000|B200|B100|GB200|5090|RTX 50"; then
    ARCH="blackwell"
  else
    ARCH="ampere_ada"   # A100 / 4090 / etc.
  fi
fi
echo "    -> arch class: ${ARCH}"

# ── Install CUDA-matched torch FIRST ────────────────────────────────────────
echo "==> 3/5  installing torch for ${ARCH}"
if [[ "${ARCH}" == "blackwell" ]]; then
  # Blackwell (sm_120) needs CUDA 12.8 kernels → torch 2.7+ cu128.
  pip install --index-url https://download.pytorch.org/whl/cu128 \
    "torch>=2.7.0"
else
  # Ampere/Ada (sm_80/sm_89) → the verified cu124 build.
  pip install --index-url https://download.pytorch.org/whl/cu124 \
    "torch==2.4.1"
fi

echo "==> 4/5  installing pinned deps + package"
pip install -r scripts/deploy/requirements-lock.txt
pip install -e .

echo "==> 5/5  verifying torch sees the GPU"
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not available after install"
p = torch.cuda.get_device_properties(0)
cc = torch.cuda.get_device_capability(0)
print(f"    torch {torch.__version__} | CUDA {torch.version.cuda} | {p.name} sm_{cc[0]}{cc[1]} {p.total_memory/1024**3:.0f}GB")
# Smoke a real kernel — catches 'no kernel image for this device' on arch mismatch.
x = torch.randn(256, 256, device="cuda"); (x @ x).sum().item()
print("    GPU matmul OK — kernels match this device")
PY

echo ""
echo "Done. Next:"
echo "  1) edit .env  (KGPW_LLAMA3_PATH, KGPW_REARAG_PATH, KGPW_* dirs)"
echo "  2) python scripts/deploy/preflight.py"
echo "  3) make phase2 && make phase3-sft && (PPO command)"
