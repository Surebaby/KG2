#!/usr/bin/env python
"""R7 AutoDL deploy: upload files → preflight → (optionally) train."""
import os, sys, io, tarfile, time
from pathlib import Path

import paramiko

HOST = "connect.bjb1.seetacloud.com"
PORT = 36491
USER = "root"
PASS = "ZIpMP0LAQGZ7"
PROJECT_DIR = "/root/autodl-tmp/kgpaper"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

FILES_TO_UPLOAD = [
    "kgproweight/reward/composite_reward.py",
    "kgproweight/training/reward_function.py",
    "kgproweight/training/phase3_ppo.py",
    "kgproweight/config/schemas.py",
    "configs/training/phase3_ppo.yaml",
    "scripts/train/phase3_ppo.py",
    "scripts/preflight_R7.py",
    "schemas.py",
    "phase3_ppo.py",
    "R7_setup.sh",
]

GREEN = "\033[32m"; RED = "\033[31m"; YEL = "\033[33m"; RST = "\033[0m"

def ssh_exec(ssh, cmd, timeout=120):
    """Execute and return (exit_code, stdout_str)."""
    chan = ssh.get_transport().open_session()
    chan.settimeout(timeout)
    chan.exec_command(cmd)
    out = io.BytesIO()
    while True:
        if chan.recv_ready():
            out.write(chan.recv(65536))
        if chan.recv_stderr_ready():
            out.write(chan.recv_stderr(65536))
        if chan.exit_status_ready() and not chan.recv_ready() and not chan.recv_stderr_ready():
            break
    time.sleep(0.3)
    while chan.recv_ready():
        out.write(chan.recv(65536))
    while chan.recv_stderr_ready():
        out.write(chan.recv_stderr(65536))
    code = chan.recv_exit_status()
    return code, out.getvalue().decode("utf-8", "replace")


def main():
    print(f"{GREEN}=== R7 AutoDeploy ==={RST}")
    print(f"Target: {USER}@{HOST}:{PORT}")
    print(f"Project: {PROJECT_DIR}")

    # 1. Connect
    print("\n[1/5] Connecting...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)
    sftp = ssh.open_sftp()
    print(f"  Connected")

    # 2. Backup server files
    print("\n[2/5] Backing up current files on server...")
    backup_dir = f"{PROJECT_DIR}/backups/R7_rollback"
    ssh_exec(ssh, f"mkdir -p {backup_dir}")
    for f in [
        "kgproweight/reward/composite_reward.py",
        "kgproweight/training/reward_function.py",
        "kgproweight/training/phase3_ppo.py",
        "configs/training/phase3_ppo.yaml",
        "kgproweight/config/schemas.py",
        "scripts/train/phase3_ppo.py",
        "schemas.py", "phase3_ppo.py",
    ]:
        remote_path = f"{PROJECT_DIR}/{f}"
        bak_name = f.replace("/", "_") + ".bak"
        code, out = ssh_exec(ssh, f"cp {remote_path} {backup_dir}/{bak_name} 2>&1")
        if code == 0:
            print(f"  {GREEN}OK{RST} backed up {f}")
        else:
            print(f"  {YEL}SKIP{RST} {f} — {out.strip()}")
    print(f"  Backups in {backup_dir}/")

    # 3. Upload files
    print("\n[3/5] Uploading R7 files...")
    for f in FILES_TO_UPLOAD:
        local_path = PROJECT_ROOT / f
        remote_path = f"{PROJECT_DIR}/{f}"
        if not local_path.exists():
            print(f"  {RED}MISS{RST} {f} — not found locally")
            continue
        try:
            sftp.put(str(local_path), remote_path)
            print(f"  {GREEN}OK{RST} {f} ({local_path.stat().st_size} bytes)")
        except Exception as e:
            print(f"  {RED}FAIL{RST} {f} — {e}")

    # 4. Ensure preflight script executable
    sftp.chmod(f"{PROJECT_DIR}/R7_setup.sh", 0o755)

    # 5. Run preflight
    print("\n[4/5] Running preflight check...")
    code, out = ssh_exec(
        ssh,
        f"cd '{PROJECT_DIR}' && python scripts/preflight_R7.py",
        timeout=120,
    )
    print(out)
    if code != 0:
        print(f"\n  {RED}PREFLIGHT FAILED{RST} — fix issues before training.")
        print(f"  SSH in and run: cd {PROJECT_DIR} && bash R7_setup.sh train")
    else:
        print(f"\n  {GREEN}PREFLIGHT PASSED{RST} — ready to train!")

    # 6. Ask about training
    print(f"\n[5/5] Training")
    print(f"  To start training, run on server:")
    print(f"    cd {PROJECT_DIR} && bash R7_setup.sh train")
    print(f"  Or let me start it now...")

    # Start training automatically
    print(f"\n  Starting training in background...")
    code, out = ssh_exec(
        ssh,
        f"cd '{PROJECT_DIR}' && "
        f"OUTPUT_DIR='{PROJECT_DIR}/checkpoints/kg_proweight_R7A' && "
        f"export OUTPUT_DIR && "
        f"mkdir -p logs && "
        f"nohup python scripts/train/phase3_ppo.py "
        f"--config configs/training/phase3_ppo.yaml "
        f"--sft_checkpoint checkpoints/sft_student_elite/final "
        f"--alpha_gate_path checkpoints/prm_alpha_gate/alpha_gate.pt "
        f"--silver_data checkpoints/prm_alpha_gate/silver_with_logprobs.jsonl "
        f"--output_dir $OUTPUT_DIR "
        f"--seed 42 "
        f"> logs/R7A_train.log 2>&1 & "
        f"echo PID=\\$!",
        timeout=30,
    )
    print(out)

    # Wait a moment and check
    time.sleep(3)
    code, out = ssh_exec(ssh, f"cd '{PROJECT_DIR}' && tail -5 logs/R7A_train.log", timeout=10)
    print(f"  First log lines:\n{out}")

    sftp.close()
    ssh.close()
    print(f"\n{GREEN}=== Deploy complete ==={RST}")
    print(f"Monitor:  ssh -p {PORT} root@{HOST}")
    print(f"  cd {PROJECT_DIR} && bash R7_setup.sh monitor")
    print(f"  tensorboard --logdir checkpoints/kg_proweight_R7A/tensorboard --port 6006 --bind_all")


if __name__ == "__main__":
    main()
