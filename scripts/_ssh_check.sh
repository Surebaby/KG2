#!/bin/bash
sshpass -p "MUvlntYq1UfZ" ssh -o StrictHostKeyChecking=no -p 49996 root@connect.bjb1.seetacloud.com "
echo '=== HOST ==='
hostname
echo '=== GPU ==='
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo '=== DISK ==='
df -h / | tail -1
echo '=== /root ==='
ls /root/ | head -20
"
