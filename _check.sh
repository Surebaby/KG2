#!/bin/bash
echo '=== 1. Code ==='
echo 'eval:' 0 'files'
echo 'scripts:' 0 'files'
echo 'configs:' 
echo ''
echo '=== 2. Conda env ==='
/home/zjulab/anaconda3/envs/kgpaper/bin/python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available())' 2>&1
/home/zjulab/anaconda3/envs/kgpaper/bin/python -c 'import transformers; print(transformers.__version__)' 2>&1
echo ''
echo '=== 3. Models ==='
du -sh /home/zjulab/kgpaper/models/ 2>/dev/null || echo 'NO models dir'
ls /home/zjulab/kgpaper/models/ 2>/dev/null
echo ''
echo '=== 4. Indexes & Data ==='
du -sh /home/zjulab/kgpaper/indexes/ /home/zjulab/kgpaper/indexes_smoke/ /home/zjulab/kgpaper/data/ /home/zjulab/kgpaper/checkpoints/ 2>/dev/null || echo 'missing some'
echo ''
echo '=== 5. GPU ==='
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader 2>&1
