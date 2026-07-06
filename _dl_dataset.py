import json, os
from datasets import load_dataset

target = "/home/zjulab/kgpaper/data/hotpotqa"
os.makedirs(target, exist_ok=True)
print("Downloading hotpotqa dev from HF...")
ds = load_dataset("RUC-NLPIR/FlashRAG_datasets", name="hotpotqa", split="dev")
out = os.path.join(target, "dev.jsonl")
n = 0
with open(out, "w", encoding="utf-8") as f:
    for i, item in enumerate(ds):
        ans = item.get("golden_answers") or item.get("answer") or []
        if isinstance(ans, str):
            ans = [ans]
        rec = {
            "id": str(item.get("id", i)),
            "question": item["question"],
            "golden_answers": [str(a) for a in ans],
            "metadata": {},
        }
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n += 1
print(f"Done: {n} items -> {out}")
