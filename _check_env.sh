#!/bin/bash
export PYTHONPATH=/home/zjulab/kgpaper/flashrag_src:/home/zjulab/kgpaper
cd /home/zjulab/kgpaper
/home/zjulab/anaconda3/envs/kgpaper/bin/python -c '
try:
    from kgproweight.eval.runner import run_evaluation
    print("kgproweight OK")
except Exception as e:
    print("kgproweight FAIL:", str(e)[:100])
try:
    from flashrag.pipeline import SequentialPipeline
    print("flashrag OK")
except Exception as e:
    print("flashrag FAIL:", str(e)[:100])
try:
    import torch
    print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
except:
    print("torch FAIL")
'
