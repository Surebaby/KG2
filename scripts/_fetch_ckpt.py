#!/usr/bin/env python
"""Pull a Phase 2 checkpoint from the AutoDL box to the local disk.

Separate from ``scripts/deploy/_ssh.py`` because that module only shells out
(``run()``) and cannot move files. G5 needs the LoRA adapter + PRM head (~129 MB);
the goal test additionally needs ``silver_with_logprobs.jsonl`` (~1.37 GB), which
is why the big file is opt-in via --with-enriched rather than always fetched.

Usage:
  python scripts/_fetch_ckpt.py CKPT_NAME [--with-enriched]

Reads KGPW_SSH_* from the environment, same as _ssh.py.
"""

from __future__ import annotations

import os
import posixpath
import stat
import sys
import time
from pathlib import Path

import paramiko

HOST = os.environ.get("KGPW_SSH_HOST", "connect.bjb1.seetacloud.com")
PORT = int(os.environ.get("KGPW_SSH_PORT", "41354"))
USER = os.environ.get("KGPW_SSH_USER", "root")
PASS = os.environ.get("KGPW_SSH_PASS", "")

# G5 needs only these. Ordered small-first so a failure surfaces fast.
SMALL = [
    "alpha_gate.pt",
    "text_reward_head.pt",
    "manifest.json",
    "history.jsonl",
]
ENRICHED = "silver_with_logprobs.jsonl"

CKPT = sys.argv[1] if len(sys.argv) > 1 else "prm_alpha_gate_v1reann_negfix"
WITH_ENRICHED = "--with-enriched" in sys.argv

# The training run was launched with a RELATIVE --output_dir, so the checkpoint
# lives under whatever cwd the launcher used. Probe rather than hardcode: a wrong
# guess would silently fetch nothing and make the eval look like it ran on the
# new checkpoint when it actually reused the old local one.
ROOTS = [
    "/root/kgpaper",
    "/root/autodl-tmp/kgpaper",
    "/root/autodl-tmp",
]


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return "%.1f %s" % (n, unit)
        n /= 1024
    return "%.1f GB" % n


def main() -> int:
    if not PASS:
        print("KGPW_SSH_PASS 未设置 — 无法登录")
        return 2

    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)
    sftp = cli.open_sftp()

    remote_dir = None
    for root in ROOTS:
        cand = posixpath.join(root, "checkpoints", CKPT)
        try:
            sftp.stat(posixpath.join(cand, "prm_head"))
            remote_dir = cand
            break
        except IOError:
            continue
    if remote_dir is None:
        print("在以下位置都找不到 checkpoints/%s/prm_head:" % CKPT)
        for r in ROOTS:
            print("  %s" % r)
        sftp.close()
        cli.close()
        return 1
    print("远程目录: %s" % remote_dir)

    local_dir = Path("checkpoints") / CKPT
    (local_dir / "prm_head").mkdir(parents=True, exist_ok=True)

    todo: list[tuple[str, Path]] = []
    # prm_head/ — the LoRA adapter, PRM head weights, tokenizer.
    for attr in sftp.listdir_attr(posixpath.join(remote_dir, "prm_head")):
        if stat.S_ISDIR(attr.st_mode or 0):
            continue
        todo.append((posixpath.join(remote_dir, "prm_head", attr.filename),
                     local_dir / "prm_head" / attr.filename))
    for name in SMALL:
        todo.append((posixpath.join(remote_dir, name), local_dir / name))
    if WITH_ENRICHED:
        todo.append((posixpath.join(remote_dir, ENRICHED), local_dir / ENRICHED))

    total = 0
    for rpath, lpath in todo:
        try:
            size = sftp.stat(rpath).st_size or 0
        except IOError:
            print("  跳过 (远程不存在): %s" % posixpath.basename(rpath))
            continue
        t0 = time.time()
        sftp.get(rpath, str(lpath))
        dt = max(time.time() - t0, 1e-6)
        total += size
        print("  %-28s %10s  %5.1f MB/s"
              % (posixpath.basename(rpath), human(size),
                 size / dt / 1024 / 1024))

    sftp.close()
    cli.close()
    print("共 %s -> %s" % (human(total), local_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
