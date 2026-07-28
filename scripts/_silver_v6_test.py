#!/usr/bin/env python
"""Standalone Phase 1 v6 test: cross-encoder passages + v2 KG + DeepSeek Teacher."""
from __future__ import annotations

import json, os, random, time
from collections import Counter
from pathlib import Path

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

random.seed(42)

# ── Config ──
N = 50
DATASET = "hotpotqa"
OUTPUT = f"data/silver_data/silver_v6_{N}.jsonl"

# ── Load questions ──
ds = [json.loads(l) for l in Path(data_dir(), f"{DATASET}/dev.jsonl").read_text().strip().split("\n")]
samples = random.sample(ds, N)

# ── Components ──
cfg = flashrag_config(build_flashrag_config(DATASET, "test", "/tmp/sil_v6", topk=DEFAULT_RRF_CANDIDATE_TOPK))
retriever = get_retriever(cfg)

from sentence_transformers import CrossEncoder
ce = CrossEncoder('/home/zjulab/kgpaper/models/bge-reranker-v2-m3')

kg_cache_dir = str(Path(index_dir()) / 'kg_cache')
linker = EntityLinker(cache_path=str(Path(index_dir()) / 'entity_cache.jsonl'), offline=True)
kg_retr = WikidataSubgraphRetriever(max_hops=2, max_neighbors=30, cache_dir=kg_cache_dir, offline=True, relation_filter=_QA_RELATION_FILTER)

teacher = TeacherClient(model="deepseek-chat", backend="deepseek")

# ── Process ──
entries = []
noise_rels = {'instance of', 'subclass of', 'has part(s)', 'part of', 'different from', 'said to be the same as', 'properties for this type', 'topic\'s main category', 'described by source'}

for i, item in enumerate(samples):
    q = item['question']
    gold = item.get('golden_answers', [])

    # Cross-encoder passages
    raw = retriever.batch_search([q])[0]
    candidates = raw[:50]
    if candidates:
        pairs = [(q, (c.get("contents","") or c.get("text",""))[:1200]) for c in candidates]
        scores = ce.predict(pairs, show_progress_bar=False)
        scored = list(zip(scores, candidates))
        scored.sort(key=lambda x: x[0], reverse=True)
        passages = [c for _, c in scored[:10]]
    else:
        passages = []

    # v2 KG
    mentions = extract_mentions(q, max_n=5)
    qids = [linker.link_single(m).selected_qid for m in mentions if linker.link_single(m).selected_qid]
    raw_kg = kg_retr.fetch(qids) if qids else []
    filtered_kg = filter_and_rank_triples(raw_kg, q, max_keep=30)

    # Teacher
    msgs = build_teacher_messages(question=q, retrieved_passages=passages, kg_triples=filtered_kg)
    try:
        raw_output = teacher.chat(msgs)
        time.sleep(0.3)
    except Exception as e:
        raw_output = ""
        print(f"Q{i} ERROR: {e}")

    # Parse steps
    has_kg = 'Knowledge Used: [(' in (raw_output or '')
    steps_count = (raw_output or '').count('[Step')
    kg_count = (raw_output or '').count('Knowledge Used: [(')
    empty_kg = (raw_output or '').count('Knowledge Used: []')
    noise_kg = sum(1 for t in filtered_kg if t[1] in noise_rels)

    # Accept: has at least 2 steps and uses KG at least once
    accepted = steps_count >= 2 and has_kg

    entry = {
        'qid': f'test_{i}',
        'question': q,
        'answer': gold,
        'dataset': DATASET,
        'kg_subgraph': [list(t) for t in filtered_kg],
        'retrieved_passages': passages,
        'teacher_output': raw_output,
        'accepted': accepted,
        'metadata': {
            'raw_kg': len(raw_kg), 'filt_kg': len(filtered_kg),
            'noise_kg': noise_kg, 'passages': len(passages),
            'steps': steps_count, 'kg_citations': kg_count, 'empty_kg': empty_kg,
        }
    }
    entries.append(entry)

    rate = sum(1 for e in entries if e['accepted']) / len(entries) * 100
    print(f"#{i}: steps={steps_count} kg={kg_count} accepted={accepted} ({rate:.0f}%) | {q[:60]}")

# ── Write ──
with open(OUTPUT, 'w') as f:
    for e in entries:
        f.write(json.dumps(e, ensure_ascii=False) + '\n')

# ── Report ──
acc = [e for e in entries if e['accepted']]
print(f"\n{'='*55}")
print(f"  V6 Silver Test — {len(entries)} questions")
print(f"{'='*55}")
print(f"  Accepted: {len(acc)}/{len(entries)}")
print(f"  Traj w/ KG: {sum(1 for e in acc if 'Knowledge Used: [(' in (e['teacher_output'] or ''))}/{len(acc)}")
print(f"  Avg steps: {sum(e['metadata']['steps'] for e in acc)/max(1,len(acc)):.0f}")
print(f"  Avg KG citations: {sum(e['metadata']['kg_citations'] for e in acc)/max(1,len(acc)):.1f}")
print(f"  Avg raw KG: {sum(e['metadata']['raw_kg'] for e in entries)/len(entries):.0f}")
print(f"  Avg filt KG: {sum(e['metadata']['filt_kg'] for e in entries)/len(entries):.0f}")
print(f"  KG noise: {sum(e['metadata']['noise_kg'] for e in entries)/max(1,sum(e['metadata']['filt_kg'] for e in entries))*100:.0f}%")
print(f"  Passages: {sum(e['metadata']['passages'] for e in entries)/len(entries):.0f}/q")
print(f"  Output: {OUTPUT}")
