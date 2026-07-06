#!/usr/bin/env python
"""Offline test for the `try` distillation logic — NO API, NO network, NO GPU.

It injects fake Teacher / retriever / entity-linker / KG-retriever objects so
that the *real* distillation code path runs end-to-end:

    run_phase1 -> _process_one -> robust mentions -> (fake) link/fetch ->
    build_teacher_messages -> (fake) Teacher -> parse/annotate ->
    answer_match_score -> StratifiedSilverFilter.decide -> write + manifest

Run:
    python scripts/train/try/test_distill_offline.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

# Make sibling try-modules importable regardless of CWD/subdir layout.
_TRY_ROOT = Path(__file__).resolve().parent.parent
for _d in (_TRY_ROOT, _TRY_ROOT / "shared", _TRY_ROOT / "phase1_distill",
           _TRY_ROOT / "phase2_prm", _TRY_ROOT / "phase3_sft", _TRY_ROOT / "phase3_ppo"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

from distill_helpers_try import StratifiedSilverFilter
from phase1_distill_try import Phase1TryConfig, run_phase1


# ---------------------------------------------------------------------------
# Fakes — each mimics only the methods the distillation code actually calls.
# ---------------------------------------------------------------------------

# A valid, schema-compliant teacher trajectory that cites the fake triples.
_RICH_TRACE = """[Step 1]
Reasoning: We need the birthplace of Albert Einstein.
Knowledge Used: [(Albert Einstein, place of birth, Ulm)]
Conclusion: Einstein was born in Ulm.

[Step 2]
Reasoning: Now find which country Ulm is in.
Knowledge Used: [(Ulm, country, Germany)]
Conclusion: Ulm is in Germany.

[Step 3]
Reasoning: Combine the two facts.
Knowledge Used: [(Germany, currency, Euro)]
Conclusion: The country is Germany.

[Final Answer]
The answer is Germany, a country in Europe.
"""

# A sparse trajectory: no triples cited (simulates KG-empty / long-tail).
_SPARSE_TRACE = """[Step 1]
Reasoning: The passages mention a small indie studio.
Knowledge Used: []
Conclusion: The studio was founded recently.

[Step 2]
Reasoning: Its founder previously worked elsewhere.
Knowledge Used: []
Conclusion: The founder came from another company.

[Step 3]
Reasoning: Combining the passage hints.
Knowledge Used: []
Conclusion: The previous employer is Acme.

[Final Answer]
Acme
"""


class FakeTeacher:
    """Content-addressed fake: pick the trace by the *question line* only.

    Keyed on the ``Question:`` line of the user message (not the whole prompt,
    which also contains KG triples), so behaviour is deterministic and not
    accidentally triggered by entities in the KG block.
    """

    model = "fake-teacher"

    def chat(self, messages):
        import re as _re
        user = " ".join(m.get("content", "") for m in messages if m.get("role") == "user")
        m = _re.search(r"Question:\s*(.+)", user)
        question = (m.group(1) if m else user).lower()
        return _RICH_TRACE if "einstein" in question else _SPARSE_TRACE


class FakeRetriever:
    def search(self, query):
        return [
            {"contents": "Ulm\nUlm is a city in Germany on the Danube."},
            {"contents": "Germany\nGermany is a country in central Europe."},
        ]


class FakeLinker:
    """Links capitalised mentions to QIDs, but only for the Einstein question.

    The indie-studio question's mentions are treated as long-tail / unlinkable
    so its trajectory exercises the KG-sparse path.
    """

    def link(self, mentions):
        if any("einstein" in m.lower() for m in mentions):
            return {m: f"Q{1000 + i}" for i, m in enumerate(mentions)}
        return {m: None for m in mentions}


class FakeKG:
    """Returns triples only when given QIDs (i.e. the Einstein question)."""

    def fetch(self, qids):
        if not qids:
            return []
        return [
            ("Albert Einstein", "place of birth", "Ulm"),
            ("Ulm", "country", "Germany"),
            ("Germany", "currency", "Euro"),
        ]


def main() -> None:
    items = [
        {"id": "q1", "question": "What country is the birthplace of Albert Einstein in?",
         "golden_answers": ["Germany"]},
        {"id": "q2", "question": "Who founded the indie studio mentioned in the passages?",
         "golden_answers": ["Acme"]},
        {"id": "q3", "question": "What country is the birthplace of Albert Einstein in?",
         "golden_answers": ["Germany"]},
        {"id": "q4", "question": "Where did the founder previously work?",
         "golden_answers": ["Acme"]},
    ]

    tmp = Path(tempfile.mkdtemp(prefix="kgpw_try_test_"))
    out_path = tmp / "silver_try_test.jsonl"

    flt = StratifiedSilverFilter(min_answer_score=0.3, sparse_quota=0.5, medium_quota=0.5)
    cfg = Phase1TryConfig(
        dataset_name="hotpotqa",
        items=items,
        output_path=str(out_path),
        teacher_client=FakeTeacher(),
        retriever_factory=FakeRetriever(),
        entity_linker=FakeLinker(),
        kg_retriever=FakeKG(),
        prm_annotator=None,   # built internally with the (fake) linker
        max_workers=1,
        accept_filter=flt,
    )

    stats = run_phase1(cfg)

    print("\n=== run_phase1 stats ===")
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    print("\n=== written records (per-item summary) ===")
    rows = [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in rows:
        m = r["metadata"]
        print(f"  {r['qid']}: accepted={r['accepted']!s:5} bucket={m.get('bucket'):9} "
              f"triple_rate={m.get('triple_rate'):.2f} answer_score={m.get('answer_score'):.2f} "
              f"coverage={m.get('coverage'):.2f} kg_empty={m.get('kg_empty')} "
              f"n_steps={len(r['steps'])} reason={m.get('reject_reason')!r}")

    # ---- assertions ----
    assert stats["total"] == 4, stats
    assert stats["written"] == 4, "every candidate should be written"
    by_id = {r["qid"]: r for r in rows}
    # rich items (q1, q3) must be accepted into kg_rich with score 1.0
    assert by_id["q1"]["accepted"] and by_id["q1"]["metadata"]["bucket"] == "kg_rich", by_id["q1"]
    assert by_id["q1"]["metadata"]["answer_score"] == 1.0, by_id["q1"]["metadata"]
    # change 1: verbose "The answer is Germany, a country..." still matched gold "Germany"
    assert by_id["q1"]["metadata"]["answer_score"] >= 0.9
    # change 2: sparse trace (Knowledge Used: []) -> triple_rate 0 -> kg_sparse bucket,
    # and it is ACCEPTED (not hard-rejected) since answer matches gold "Acme".
    assert by_id["q2"]["metadata"]["bucket"] == "kg_sparse", by_id["q2"]["metadata"]
    assert by_id["q2"]["metadata"]["triple_rate"] == 0.0, by_id["q2"]["metadata"]
    assert by_id["q2"]["metadata"]["kg_empty"] is True, by_id["q2"]["metadata"]
    assert by_id["q2"]["metadata"]["coverage"] == 0.0, by_id["q2"]["metadata"]
    assert by_id["q2"]["accepted"] is True, "sparse-but-correct trace must be kept (change 2+3+4)"
    print("\nALL ASSERTIONS PASSED ✅")
    print(f"(output written to {out_path})")


if __name__ == "__main__":
    main()
