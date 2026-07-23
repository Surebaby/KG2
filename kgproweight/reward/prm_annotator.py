"""Automatic three-class step labeller used in Phase 1 (and the PPO reward).

Labels:
    +1 (POSITIVE) — every cited triple verified AND relevant AND consistent.
     0 (NEUTRAL)  — KG can neither confirm nor deny (transitional, sparse
                    subgraph, unverifiable citation, honest abstention).
    -1 (NEGATIVE) — a cited/claimed step that directly contradicts a verified
                    prior conclusion.

Decision policy (per step)
--------------------------
1. discourse/transition with no cited triples            -> NEUTRAL (0)
2. no cited triples:
     - contradiction with a prior conclusion             -> NEGATIVE (-1)
     - otherwise (KG can't confirm/deny)                 -> NEUTRAL (0)
3. cited triples:
     - subgraph empty/too-sparse to verify               -> NEUTRAL (0)
     - contradiction with a verified prior conclusion    -> NEGATIVE (-1)
     - all verified AND at least one relevant            -> POSITIVE (+1)
     - all verified but none relevant (filler citation)  -> NEUTRAL (0)
     - cited but unverifiable (absent from subgraph)     -> NEUTRAL (0)

Rationale (correctness fixes folded in from the validated `try` variant):

* Entity-drift is **no longer** a stand-alone negative trigger. The original
  fired -1 whenever a step's capitalised mentions did not fuzzy-match the
  (often noisy) subgraph, mislabelling legitimate world-knowledge steps.
* A cited triple that is *correct but simply absent* from a sparse/incomplete
  subgraph is NEUTRAL, not NEGATIVE — paper pain-point C2 ("do not punish where
  Wikidata is incomplete").
* Honest abstentions ("the KG has no info", "cannot determine", "not found")
  are vetoed from the contradiction path (``_ABSTENTION_RE``) — they report a
  gap, they do not hallucinate a contradiction.
* A verified citation only earns +1 if at least one cited triple is lexically
  relevant to the step's conclusion, killing the "filler triple" hack.
"""

from __future__ import annotations

import re
import string
from typing import List, Optional, Set, Tuple

from kgproweight.data.parsers import (
    DISCOURSE_RE,
    ENTITY_RE,
    ParsedStep,
    parsed_step_from_silver_dict,
)
from kgproweight.kg.coverage import triple_in_subgraph
from kgproweight.kg.entity_linker import EntityLinker
from kgproweight.utils.logging import get_logger

logger = get_logger(__name__)

POSITIVE = 1
NEUTRAL = 0
NEGATIVE = -1


# Conclusions that *honestly abstain* ("the KG has no info", "cannot determine",
# "no matching X found", "not about Y") must NOT be read as contradictions —
# they are the model correctly reporting a gap, not hallucinating.
#
# A BARE copula negation ("X is not a scientist") is a genuine counter-assertion,
# NOT an abstention — so the copula forms require an explicit abstention OBJECT
# (available / found / mentioned / specified / clear / identified / the same /
# in the (graph|kg|...)) rather than matching any "is not". The POSITIVE
# alternatives `identified in` / `is identified` are intentionally absent — they
# match assertions like "Einstein is identified as a physicist"; only the negated
# `not identified` is an abstention (covered standalone and via the copula clause).
_ABSTENTION_RE = re.compile(
    r"\b(no information|no info|not available|not contain|does not contain|"
    r"doesn't contain|no matching|not found|cannot (be )?(determine|found|answer)|"
    r"can't (determine|find|answer)|unknown|insufficient|lacks?|lacking|"
    r"no (relevant |retrieved )?(passage|information|data|triple|entry)|"
    r"not (mention|state|provide|specify|about|the same|directly)|no suitable|"
    r"no evidence|not identified|"
    r"does not (directly )?(link|relate|connect|indicate)|"
    r"unable to|"
    r"(is|are|was|were) not "
    r"(available|found|mentioned|specified|present|clear|known|listed|given|"
    r"identified|provided|stated|the same|directly|in (the |this )?"
    r"(graph|kg|knowledge graph|subgraph|passage|context|data)))\b",
    re.IGNORECASE,
)

_PUNCT = str.maketrans("", "", string.punctuation)
_REL_STOP: frozenset[str] = frozenset(
    """the a an of in on at to for and or is are was were be been being that this these
    those which who whom whose what when where why how from with as by into over""".split()
)


def _content_tokens(text: str) -> Set[str]:
    """Lower-cased content tokens (len>=3, non-stopword) for relevance checks.

    The threshold is len>=3 (not >=4): >=4 silently dropped short proper nouns
    ("Ulm", "USA", "UK") and 1-3 digit numbers from BOTH the conclusion and the
    triple, so a genuinely verified+relevant step whose entity is short shared no
    token and got demoted POSITIVE->NEUTRAL. The word-boundary phrase match in
    ``_triple_relevant`` covers the len<3 tail.
    """
    text = text.lower().translate(_PUNCT)
    return {w for w in text.split() if len(w) >= 3 and w not in _REL_STOP}


# ---------------------------------------------------------------------------
# Annotator
# ---------------------------------------------------------------------------

class PRMAnnotator:
    """Conservative three-class step labeller.

    Labels parsed steps against a KG subgraph and prior conclusions. NEUTRAL is
    the default whenever the KG can neither confirm nor deny a claim.

    Parameters
    ----------
    entity_linker:
        Kept for API-compatibility; not required by the labelling policy.
    min_subgraph_for_verify:
        A subgraph with fewer than this many triples is too sparse to *disprove*
        a cited triple, so unverifiable citations fall back to NEUTRAL.
    triple_fuzzy_threshold:
        Threshold passed to ``triple_in_subgraph``.
    require_triple_relevance:
        When True, a verified citation only earns +1 if at least one cited triple
        is lexically relevant to the step conclusion (kills the "filler triple"
        hack where the teacher cites any in-subgraph triple).
    guard_abstention:
        When True, conclusions that honestly abstain are never labelled -1 by the
        contradiction path.
    entity_drift_threshold:
        Deprecated/ignored. Entity drift is no longer a negative trigger; kept
        only so existing call sites that pass it do not break.
    """

    def __init__(
        self,
        entity_linker: Optional[EntityLinker] = None,
        min_subgraph_for_verify: int = 3,
        triple_fuzzy_threshold: float = 80.0,
        neutral_pattern_match: bool = True,
        require_triple_relevance: bool = True,
        guard_abstention: bool = True,
        entity_drift_threshold: float = 70.0,  # deprecated, ignored
        verbose: bool = False,
    ) -> None:
        self.entity_linker = entity_linker or EntityLinker()
        self.min_subgraph_for_verify = min_subgraph_for_verify
        self.triple_fuzzy_threshold = triple_fuzzy_threshold
        self.neutral_pattern_match = neutral_pattern_match
        self.require_triple_relevance = require_triple_relevance
        self.guard_abstention = guard_abstention
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def label(
        self,
        step: ParsedStep,
        kg_subgraph: List[Tuple[str, str, str]],
        prev_conclusions: List[str],
    ) -> int:
        text = step.raw_text
        subgraph_usable = len(kg_subgraph) >= self.min_subgraph_for_verify

        # 1) discourse/transition with no cited triples -> neutral
        if self.neutral_pattern_match and self._is_discourse(text) and not step.cited_triples:
            return self._log(step.index, NEUTRAL, "discourse, no triples")

        # 2) no cited triples
        if not step.cited_triples:
            # contradiction is the ONLY negative trigger without citations
            if self._is_contradiction(step.intermediate_conclusion, prev_conclusions):
                return self._log(step.index, NEGATIVE, "contradiction (no triples)")
            # otherwise the KG cannot confirm or deny -> neutral
            return self._log(step.index, NEUTRAL, "no verifiable triples")

        # 3) cited triples present
        # 3a) subgraph too sparse to verify/refute -> neutral (C2: don't punish KG gaps)
        if not subgraph_usable:
            return self._log(step.index, NEUTRAL, f"subgraph too sparse ({len(kg_subgraph)})")

        # 3b) verify each cited triple, count precision
        verified_count = 0
        for triple in step.cited_triples:
            if triple_in_subgraph(triple, kg_subgraph, fuzzy_threshold=self.triple_fuzzy_threshold):
                verified_count += 1

        # 3c) contradiction always dominates (abstentions excluded)
        if self._is_contradiction(step.intermediate_conclusion, prev_conclusions):
            return self._log(step.index, NEGATIVE, "contradiction (with triples)")

        # R9: precision-based R_KG = verified / total. Breaks the zero-signal
        # deadlock: even partially-verified citations give non-zero reward,
        # which provides gradient for the model to improve citation accuracy.
        precision = verified_count / len(step.cited_triples) if step.cited_triples else 0.0

        # R9 v5: KG Utility = Precision × Relevance.
        # Precision alone rewards "real triples" but ignores whether they help
        # answer the question. Relevance measures how many cited triples actually
        # touch the step's conclusion. A triple that is real but irrelevant
        # (e.g. "Ed Wood, instance of, human" for a nationality question) should
        # not earn the same reward as a directly useful triple.
        # R9 v6: strip Knowledge Used section from text used for relevance.
        # The triple was extracted from "Knowledge Used:" which is part of raw_text,
        # so phrase-matching against raw_text always succeeds (self-citation).
        # Extract only the reasoning body to judge whether the triple's content
        # actually appears in the model's independent reasoning.
        reasoning_only = step.raw_text
        if "Knowledge Used:" in reasoning_only:
            reasoning_only = reasoning_only.split("Knowledge Used:", 1)[0]

        relevance_ratio = 1.0  # default: assume relevant
        if self.require_triple_relevance and step.cited_triples:
            relevant_count = sum(
                1 for triple in step.cited_triples
                if self._triple_relevant(
                    [triple],
                    reasoning=reasoning_only,
                    conclusion=step.intermediate_conclusion,
                )
            )
            relevance_ratio = relevant_count / len(step.cited_triples)

        r_kg = precision * relevance_ratio

        if precision >= 1.0 and relevance_ratio >= 1.0:
            return self._log(step.index, POSITIVE,
                f"all {verified_count} verified + relevant, r_kg={r_kg:.2f}")
        elif precision > 0:
            return self._log(step.index, r_kg,
                f"precision={precision:.2f} relevance={relevance_ratio:.2f} r_kg={r_kg:.2f}")
        else:
            return self._log(step.index, NEUTRAL, "cited triple absent from subgraph")

    def annotate_trajectory(
        self,
        steps: List[ParsedStep],
        kg_subgraph: List[Tuple[str, str, str]],
    ) -> List[int]:
        labels: List[int] = []
        prev: List[str] = []
        for step in steps:
            labels.append(self.label(step, kg_subgraph, prev))
            if step.intermediate_conclusion:
                prev.append(step.intermediate_conclusion)
        return labels

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_discourse(text: str) -> bool:
        return bool(DISCOURSE_RE.match(text.strip()))

    @staticmethod
    def _has_factual_claim(text: str) -> bool:
        return bool(ENTITY_RE.search(text))

    @staticmethod
    def _contradicts(conclusion: str, prev_conclusions: List[str]) -> bool:
        neg_patterns = [r"\bnot\b", r"\bnever\b", r"\bno\b", r"\bcannot\b", r"\bdoes not\b"]
        cl = conclusion.lower()
        for prev in prev_conclusions:
            pl = prev.lower()
            tokens_c = set(re.findall(r"\b\w{4,}\b", cl))
            tokens_p = set(re.findall(r"\b\w{4,}\b", pl))
            if len(tokens_c & tokens_p) >= 2:
                c_neg = any(re.search(p, cl) for p in neg_patterns)
                p_neg = any(re.search(p, pl) for p in neg_patterns)
                if c_neg != p_neg:
                    return True
        return False

    def _is_contradiction(self, conclusion: Optional[str], prev_conclusions: List[str]) -> bool:
        """Guarded contradiction test.

        Wraps the raw ``_contradicts`` heuristic but first vetoes *honest
        abstentions* — conclusions that report a gap or decline to commit. These
        share words + a negation with a prior conclusion and so trip the bare
        heuristic, but they are the model correctly reporting absence. A genuine
        -1 is a positive counter-assertion about the same fact, which carries no
        abstention marker and so survives the guard.
        """
        if not conclusion or not prev_conclusions:
            return False
        if self.guard_abstention and _ABSTENTION_RE.search(conclusion):
            return False
        return self._contradicts(conclusion, prev_conclusions)

    @staticmethod
    def _phrase_match(phrase: str, text: str) -> bool:
        """Normalized phrase match: lower-case, strip punctuation, word-boundary."""
        p = phrase.strip().lower().translate(_PUNCT).strip()
        t = text.lower().translate(_PUNCT)
        if not p:
            return False
        return re.search(rf"\b{re.escape(p)}\b", t) is not None

    def _triple_relevant(
        self,
        cited_triples: List[Tuple[str, str, str]],
        reasoning: Optional[str] = None,
        conclusion: Optional[str] = None,
    ) -> bool:
        """Lightweight surface-level evidence-overlap estimator.

        Returns True if at least one cited triple's entities or relation appear
        in the reasoning trajectory or final conclusion. This measures *lexical
        evidence grounding* — whether the cited KG fact left a trace in the
        model's output — not semantic entailment.

        Scoring (per triple):
          head OR tail phrase-match in text    → +1 (entity evidence)
          relation phrase-match in text        → +0.5 (relation evidence, weaker)

        A triple scores > 0 → relevant.
        """
        text = " ".join(s for s in (reasoning, conclusion) if s)
        if not text:
            return True  # cannot judge → default relevant

        for h, r, t in cited_triples:
            score = 0.0
            # entity evidence (primary)
            if self._phrase_match(h, text):
                score += 1.0
            if self._phrase_match(t, text):
                score += 1.0
            # relation evidence (secondary, lower weight)
            if self._phrase_match(r, text):
                score += 0.5
            if score > 0:
                return True
        return False

    def _log(self, idx: int, label, reason: str):
        if self.verbose:
            if label == POSITIVE:
                name = "+1"
            elif label == NEGATIVE:
                name = "-1"
            elif label == NEUTRAL:
                name = "0"
            else:
                name = f"{label:.3f}"
            logger.debug("step %d -> %s (%s)", idx, name, reason)
        return label
