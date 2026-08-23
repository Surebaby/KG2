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

    A mention is dropped when EVERY one of its tokens is a scaffold word, so
    "Reasoning" and "Knowledge Used" -> dropped, while "Albert Einstein" and
    "Reasoning Museum" (one non-scaffold token) are kept.

    The whole-token test the earlier version used compared the joined mention
    against the single-token set, so every MULTI-WORD scaffold phrase bypassed
    it. The header of this module lists "Knowledge Used" and "Final Answer" as
    exactly what must be stripped, and both were being kept: measured over the
    33,011 accepted silver steps, "Knowledge Used" survived on 32,986 of them
    (99.9%) and pure-scaffold mentions were 17.2% of all kept mentions
    (32,989/191,400). Each contributed a spurious link_confidence of 0.667 (its
    fuzzy match against the entity cache), which is precisely the inflation this
    module was written to prevent.

    Both live call sites -- Phase 2 (``phase2_prm._build_samples_accepted_only``)
    and the PPO reward (``reward_function``) -- go through this one function, so
    the fix lands on training and inference together and the feature
    distributions stay aligned. It does, however, change the feature the shipped
    α-gate was FITTED with, so the gate must be re-fitted (Phase 2 re-run) rather
    than reused across this change.
    """
    out: List[str] = []
    seen = set()
    for e in entities:
        key = e.strip()
        if not key:
            continue
        toks = key.lower().split()
        if toks and all(t in _SCAFFOLD for t in toks):
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out
