#!/usr/bin/env python3
"""Check + run R9 PPO test on remote AutoDL server."""
import paramiko, sys, json, time

HOST = "connect.bjb1.seetacloud.com"
PORT = 41354
USER = "root"
PASS = "JfJszmekRbvJ"
PROJ = "/root/autodl-tmp/kgpaper"
ENV_PYTHON = "/root/autodl-tmp/kgpw_env/bin/python3.10"

def ssh():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)
    return c

def run(c, cmd, timeout=30):
    _, out, err = c.exec_command(cmd, timeout=timeout)
    return out.read().decode().strip(), err.read().decode().strip()

c = ssh()

# ── 1. Preflight checks ──
print("=" * 60)
print("1. HARDWARE")
print("=" * 60)
o, _ = run(c, 'nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader')
print("GPU:", o)
o, _ = run(c, 'nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null || echo IDLE')
print("Processes:", o)
o, _ = run(c, 'df -h /root/autodl-tmp | tail -1')
print("Disk:", o)

print()
print("=" * 60)
print("2. CODE CHECK")
print("=" * 60)
for key, path in [("composite_reward", f"{PROJ}/kgproweight/reward/composite_reward.py"),
                   ("prm_annotator", f"{PROJ}/kgproweight/reward/prm_annotator.py"),
                   ("phase3_ppo", f"{PROJ}/kgproweight/training/phase3_ppo.py"),
                   ("YAML", f"{PROJ}/configs/training/phase3_ppo.yaml")]:
    marks = {
        "step_reward_scale": "step_reward_scale",
        "precision": "precision",
        "question_kg_index": "question_kg_index",
        "invalid_penalty": "invalid_penalty",
    }
    found = []
    for mark_key, mark_val in marks.items():
        o, _ = run(c, f"grep -c '{mark_val}' {path} 2>/dev/null || echo 0")
        if o.strip() != "0":
            found.append(mark_key)
    print(f"  {key}: {' '.join(found) if found else 'MISSING'}")

print()
print("=" * 60)
print("3. CACHE CHECK")
print("=" * 60)
cache_path = f"{PROJ}/indexes/kg_cache/question_kg_index.json"
o, _ = run(c, f"ls -la {cache_path} 2>/dev/null")
print("Cache:", o[:120] if o else "MISSING!")
if o:
    o2, _ = run(c, f"python3 -c \"import json;d=json.load(open('{cache_path}'));print(len(d),'entries,',sum(len(e.get('t',[]))for e in d),'triples')\"")
    print("Content:", o2)
    o3, _ = run(c, f"md5sum {cache_path}")
    print("MD5:", o3.split()[0] if o3 else "ERR")

print()
print("=" * 60)
print("4. MODELS")
print("=" * 60)
o, _ = run(c, 'ls /root/autodl-tmp/models/ 2>/dev/null')
print("Models:", o)

print()
print("=" * 60)
print("5. PREFLIGHT (quick)")
print("=" * 60)
o, _ = run(c, f'cd {PROJ} && {ENV_PYTHON} -c "from kgproweight.utils.paths import index_dir; print(index_dir())" 2>&1')
print("Import test:", o)

# ── 2. Launch training ──
print()
print("=" * 60)
print("6. LAUNCH PPO 500 STEPS")
print("=" * 60)

# First: find which python has kgproweight installed
o, _ = run(c, f"find /root -name 'python*' -type f 2>/dev/null | head -10")
print("Python candidates:", o[:300])
o, _ = run(c, "which python3 python 2>/dev/null; ls /root/miniconda3/envs/ 2>/dev/null")
print("which & envs:", o)
# Try each to find kgproweight
for py in ["/root/autodl-tmp/kgpw_env/bin/python3.10", "/root/miniconda3/bin/python3", "/root/miniconda3/bin/python", "python3", "python"]:
    o, _ = run(c, f"{py} -c 'import kgproweight; print(kgproweight.__file__)' 2>&1")
    if "kgproweight" in o and "Error" not in o:
        print(f"CORRECT PYTHON: {py} -> {o}")
        ENV_PYTHON = py.strip()
        break
else:
    print("Could not find kgproweight-compatible python")
    # Last resort: look at the previous training logs
    o, _ = run(c, "head -5 /root/autodl-tmp/train_all.log 2>/dev/null")
    print("Previous train log:", o[:300])

# Launch: write script then run with setsid to detach
script = f"""#!/bin/bash
cd {PROJ}
nohup {ENV_PYTHON} kgproweight/training/phase3_ppo.py --config configs/training/phase3_ppo.yaml > /root/autodl-tmp/ppo_r9_500.log 2>&1 &
"""
sftp = c.open_sftp()
with sftp.file('/tmp/run_ppo.sh', 'w') as f:
    f.write(script)
sftp.close()
run(c, 'chmod +x /tmp/run_ppo.sh')
# Use setsid to fully detach
o, _ = run(c, 'setsid bash /tmp/run_ppo.sh < /dev/null > /dev/null 2>&1 &')
print("Launched via setsid")
time.sleep(20)
o, _ = run(c, "ps aux | grep phase3_ppo | grep -v grep")
if o:
    print("RUNNING:", o[:300])
    o, _ = run(c, "tail -20 /root/autodl-tmp/ppo_r9_500.log 2>/dev/null")
    print("=== Log ===")
    print(o[:800])
else:
    print("Not running - check log")
    o, _ = run(c, "cat /root/autodl-tmp/ppo_r9_500.log 2>/dev/null | wc -l")
    print("Log lines:", o)
    o, _ = run(c, "head -30 /root/autodl-tmp/ppo_r9_500.log 2>/dev/null")
    print("=== Log ===")
    print(o[:800] if o else "EMPTY LOG")

print()
print("=" * 60)
print("DONE — Training launched. Monitor:")
print(f"  ssh -p {PORT} root@{HOST}")
print("  tail -f /root/autodl-tmp/ppo_r9_500.log")
print("=" * 60)

c.close()
