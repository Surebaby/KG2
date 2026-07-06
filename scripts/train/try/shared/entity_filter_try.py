"""Shared entity-mention cleaner for the try pipeline (Finding 2 follow-up).

The package parser ``ParsedStep.from_text`` extracts ``mentioned_entities`` with a
broad capitalised-phrase regex (``ENTITY_RE``). On our trace format that regex also
grabs reasoning SCAFFOLD words that happen to start a line — "Reasoning",
"Conclusion", "Therefore", "Step", "Final Answer", etc. — which then fuzzy-match
cache keys and inflate ``link_confidence`` even on steps with no real entity (e.g. a
pure abstention step scored 0.82 purely from "Reasoning"/"Conclusion").

This filter strips that scaffold so ``link_confidence`` reflects only genuine entity
mentions. It MUST be applied identically wherever step entities feed the α-gate —
both Phase 2 training (``_build_samples_accepted_only``) and PPO inference
(``ppo_reward_try`` after ``parse_steps``) — or the Finding-2 train/inference
alignment breaks. Keep this the single source of truth for both call sites.
"""
from __future__ import annotations

from typing import List

# Reasoning-scaffold / template tokens that ENTITY_RE captures but are never
# real-world entities. Lower-cased; matched case-insensitively against the WHOLE
# mention (so multi-word mentions containing a real entity are NOT dropped).
_SCAFFOLD: frozenset[str] = frozenset(
    """
    reasoning conclusion therefore thus hence so step final answer
    knowledge used question reasoning steps given evidence passage passages
    retrieved fact facts triple triples context note observation observe
    first second third fourth fifth next then finally also however
    """.split()
)


def clean_entities(entities: List[str]) -> List[str]:
    """Drop scaffold mentions; keep order and de-dup. Pure, no side effects.

    A mention is dropped only if its ENTIRE lower-cased form is a scaffold token,
    so "Reasoning" → dropped but "Albert Einstein" / "Reasoning Museum" are kept.
    """
    out: List[str] = []
    seen = set()
    for e in entities:
        key = e.strip()
        if not key:
            continue
        if key.lower() in _SCAFFOLD:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out
