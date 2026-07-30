"""Entity-mention cleaner shared by Phase 2 training and the PPO reward.

The parser ``ParsedStep.from_text`` extracts ``mentioned_entities`` with a broad
capitalised-phrase regex (``ENTITY_RE``). On the trace format that regex also
grabs reasoning SCAFFOLD words that start a line — "Reasoning", "Conclusion",
"Therefore", "Step", "Final Answer", etc. — which then fuzzy-match entity-cache
keys and inflate ``link_confidence`` even on steps with no real entity.

This filter strips that scaffold so ``link_confidence`` reflects only genuine
entity mentions. It MUST be applied identically wherever step entities feed the
α-gate — both Phase 2 training and PPO inference — or the train/inference
distribution of the link_confidence feature diverges. This is the single source
of truth for both call sites.
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
    so "Reasoning" -> dropped but "Albert Einstein" / "Reasoning Museum" are kept.
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
