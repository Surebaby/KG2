#!/usr/bin/env python
"""Mixed smoke test: 40 HotpotQA + 5 2Wiki + 5 Musique = 50 questions."""
import json, os, random, time

from kgproweight.utils.flashrag_bootstrap import setup_flashrag
setup_flashrag(os.environ.get("KGPW_FLASHRAG_ROOT", "/home/zjulab/kgpaper/flashrag_src"))
from flashrag.utils import get_retriever

from kgproweight.retrieval.hybrid import build_flashrag_config, DEFAULT_RRF_CANDIDATE_TOPK
from kgproweight.data.flashrag_loader import flashrag_config
from kgproweight.kg.wikidata_retriever import WikidataSubgraphRetriever, _QA_RELATION_FILTER
from kgproweight.kg.entity_linker import EntityLinker, extract_mentions
from kgproweight.kg.kg_filter import filter_and_rank_triples
from kgproweight.data.prompts import build_teacher_messages
from kgproweight.training.phase1_distill import TeacherClient
from kgproweight.utils.paths import data_dir, index_dir
from sentence_transformers import CrossEncoder

random.seed(42)
N_PER = {"hotpotqa": 40, "2wikimultihopqa": 5, "musique": 5}
OUTPUT = "data/silver_data/_smoke_mixed_DO_NOT_TRAIN.jsonl"

# Collect questions
all_items = []
for ds_name, n in N_PER.items():
    path = f"{data_dir()}/{ds_name}/dev.jsonl"
    items = [json.loads(l) for l in open(path).read().strip().split("\n")]
    picked = random.sample(items, min(n, len(items)))
    for p in picked:
        p["_dataset"] = ds_name
    all_items.extend(picked)
random.shuffle(all_items)

# Components
cfg = flashrag_config(build_flashrag_config("hotpotqa", "test", "/tmp/smoke_mix", topk=DEFAULT_RRF_CANDIDATE_TOPK))
retriever = get_retriever(cfg)
ce = CrossEncoder('/home/zjulab/kgpaper/models/bge-reranker-v2-m3')

linker = EntityLinker(cache_path=f"{index_dir()}/entity_cache.jsonl", offline=True)
kg_retr = WikidataSubgraphRetriever(max_hops=2, max_neighbors=30, cache_dir=f"{index_dir()}/kg_cache", offline=True, relation_filter=_QA_RELATION_FILTER)
teacher = TeacherClient(model="deepseek-chat", backend="deepseek")

entries = []
for i, item in enumerate(all_items):
    q = item['question']; ds = item['_dataset']

    # Passages
    raw = retriever.batch_search([q])[0]
    candidates = raw[:50]
    if candidates:
        pairs = [(q, (c.get("contents","") or c.get("text",""))[:1200]) for c in candidates]
        scores = ce.predict(pairs, show_progress_bar=False)
        scored = list(zip(scores, candidates)); scored.sort(key=lambda x: x[0], reverse=True)
        passages = [c for _, c in scored[:10]]
    else:
        passages = []

    # KG
    mentions = extract_mentions(q, max_n=5)
    qids = [linker.link_single(m).selected_qid for m in mentions if linker.link_single(m).selected_qid]
    raw_kg = kg_retr.fetch(qids) if qids else []
    filt_kg = filter_and_rank_triples(raw_kg, q, max_keep=30)

    # Teacher
    msgs = build_teacher_messages(question=q, retrieved_passages=passages, kg_triples=filt_kg)
    try:
        raw_output = teacher.chat(msgs); time.sleep(0.3)
    except Exception as e:
        raw_output = f"[ERROR: {e}]"

    has_kg = 'Knowledge Used: [(' in (raw_output or '')
    steps = (raw_output or '').count('[Step')
    accepted = steps >= 2 and has_kg

    entries.append({
        'qid': f'{ds}_{i}', 'question': q, 'answer': item.get('golden_answers', []),
        'dataset': ds, 'kg_subgraph': [list(t) for t in filt_kg],
        'retrieved_passages': passages, 'teacher_output': raw_output,
        'accepted': accepted,
        'metadata': {'steps': steps, 'kg_cites': (raw_output or '').count('Knowledge Used: [(')}
    })

    rate = sum(1 for e in entries if e['accepted']) / len(entries) * 100
    print(f"#{i} [{ds}]: steps={steps} kg={has_kg} acc={rate:.0f}% | {q[:55]}")

with open(OUTPUT, 'w') as f:
    for e in entries: f.write(json.dumps(e, ensure_ascii=False) + '\n')

acc = [e for e in entries if e['accepted']]
print(f"\nDone. {len(acc)}/{len(entries)} accepted. Output: {OUTPUT}")
for ds in N_PER:
    ds_acc = [e for e in acc if e['dataset'] == ds]
    print(f"  {ds}: {len(ds_acc)} accepted")
