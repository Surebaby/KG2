"""Improved 3-class PRM annotator for the `try` variant.

Why this exists
---------------
The original ``kgproweight.reward.prm_annotator.PRMAnnotator`` collapses to a
**binary** labeller in practice: on a 30-trajectory real run it produced
51.5% +1 / 0% NEUTRAL / 48.5% -1, and 40 steps that cited *no* triple were
still labelled -1. Root causes:

* The entity-drift branch fires whenever a step's capitalised mentions do not
  fuzzy-match the (often noisy / wrongly-linked) KG subgraph, so legitimate
  world-knowledge steps are flagged as "drift" hallucinations.
* A cited triple that is *correct but simply absent* from a sparse/incomplete
  subgraph is treated as a hallucination — the exact opposite of the paper's
  pain-point C2 ("do not punish where Wikidata is incomplete").
* NEUTRAL never gets assigned, so the three-valued signal degrades to binary.

This class keeps the original *positive* and *contradiction* logic but makes
the negative path conservative, so NEUTRAL is the default when the KG cannot
confirm or deny a claim.

Decision policy (per step)
--------------------------
1. discourse/transition with no cited triples           -> NEUTRAL (0)
2. no cited triples, no factual entity claim            -> NEUTRAL (0)
3. cited triples:
     - if subgraph is empty/too-sparse to verify        -> NEUTRAL (0)
     - all cited triples verified in subgraph + no
       contradiction with prior conclusions             -> POSITIVE (+1)
     - a cited triple directly *contradicts* a verified
       prior conclusion                                  -> NEGATIVE (-1)
     - cited but unverifiable (absent from subgraph)     -> NEUTRAL (0)
4. no cited triples but factual claim present:
     - contradiction with a prior conclusion             -> NEGATIVE (-1)
     - otherwise (KG can't confirm/deny)                 -> NEUTRAL (0)

Entity-drift is **no longer** a stand-alone negative trigger; it only
contributes when the subgraph is dense AND a hard contradiction is present.
This removes the false-negative cascade while still catching genuine
contradictions.
"""

from __future__ import annotations

import re
import string
from typing import List, Optional, Set, Tuple

from kgproweight.data.parsers import DISCOURSE_RE, ENTITY_RE, ParsedStep
from kgproweight.kg.coverage import triple_in_subgraph
from kgproweight.kg.entity_linker import EntityLinker
from kgproweight.reward.prm_annotator import (
    NEGATIVE,
    NEUTRAL,
    POSITIVE,
)
from kgproweight.utils.logging import get_logger

logger = get_logger(__name__)


# Conclusions that *honestly abstain* ("the KG has no info", "cannot determine",
# "no matching X found", "not about Y") must NOT be read as contradictions —
# they are the model correctly reporting a gap, not hallucinating. On the 80-item
# run, 11/16 of the original -1 labels were abstentions misfired by _contradicts.
#
# NOTE (#abstention-tighten): a BARE copula negation ("X is not a scientist") is a
# genuine counter-assertion, NOT an abstention — matching it here wrongly vetoes
# real -1 labels and thins an already-sparse class. So the copula forms require an
# explicit abstention OBJECT (available / found / mentioned / specified / clear /
# identified / the same / in the (graph|kg|...)) rather than matching any "is not".
# Also dropped the POSITIVE alternatives `identified in` / `is identified` — those
# match assertions like "Einstein is identified as a physicist" and would veto a
# real contradiction built on them; only the negated `not identified` is an
# abstention (covered both standalone and via the copula clause).
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

    A2c: the threshold was len>=4, which silently dropped short proper nouns
    ("Ulm", "USA", "UK") and 1-3 digit numbers from BOTH the conclusion and the
    triple, so a genuinely verified+relevant step whose entity is short shared no
    token and got demoted POSITIVE→NEUTRAL. len>=3 recovers most; the
    word-boundary phrase match in ``_triple_relevant`` covers the len<3 tail.
    """
    text = text.lower().translate(_PUNCT)
    return {w for w in text.split() if len(w) >= 3 and w not in _REL_STOP}


class ImprovedPRMAnnotator:
    """Conservative three-class step labeller (try variant).

    Parameters
    ----------
    entity_linker:
        Reused only for API-compatibility with the original; not required by
        the labelling policy here.
    min_subgraph_for_verify:
        A subgraph with fewer than this many triples is considered too sparse
        to *disprove* a cited triple, so unverifiable citations fall back to
        NEUTRAL instead of NEGATIVE.
    triple_fuzzy_threshold:
        Threshold passed to ``triple_in_subgraph``.
    verbose:
        Emit per-step debug logs.
    """

    def __init__(
        self,
        entity_linker: Optional[EntityLinker] = None,
        min_subgraph_for_verify: int = 3,
        triple_fuzzy_threshold: float = 80.0,
        neutral_pattern_match: bool = True,
        require_triple_relevance: bool = True,
        guard_abstention: bool = True,
        verbose: bool = False,
    ) -> None:
        self.entity_linker = entity_linker or EntityLinker()
        self.min_subgraph_for_verify = min_subgraph_for_verify
        self.triple_fuzzy_threshold = triple_fuzzy_threshold
        self.neutral_pattern_match = neutral_pattern_match
        # When True, a verified citation only earns +1 if at least one cited
        # triple is lexically relevant to the step conclusion (kills the
        # "filler triple" hack where the teacher cites any in-subgraph triple).
        self.require_triple_relevance = require_triple_relevance
        # When True, conclusions that honestly abstain ("KG has no info",
        # "cannot determine") are never labelled -1 by the contradiction path.
        self.guard_abstention = guard_abstention
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Public API (mirrors the original signature)
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
            #   (this is where the original wrongly fired entity-drift -> -1)
            return self._log(step.index, NEUTRAL, "no verifiable triples")

        # 3) cited triples present
        # 3a) subgraph too sparse to verify/refute -> neutral (C2: don't punish KG gaps)
        if not subgraph_usable:
            return self._log(step.index, NEUTRAL, f"subgraph too sparse ({len(kg_subgraph)})")

        # 3b) verify each cited triple
        verified_all = True
        for triple in step.cited_triples:
            if not triple_in_subgraph(triple, kg_subgraph, fuzzy_threshold=self.triple_fuzzy_threshold):
                verified_all = False
                break

        # 3c) contradiction always dominates (abstentions excluded)
        if self._is_contradiction(step.intermediate_conclusion, prev_conclusions):
            return self._log(step.index, NEGATIVE, "contradiction (with triples)")

        if verified_all:
            # 3d) a verified citation only earns +1 if at least one cited triple
            # is actually relevant to the conclusion — otherwise it is a filler
            # citation (triple is real & in-subgraph but supports nothing here).
            if self.require_triple_relevance and not self._triple_relevant(
                step.cited_triples, step.intermediate_conclusion
            ):
                return self._log(step.index, NEUTRAL, "verified but filler (triple unrelated to conclusion)")
            return self._log(step.index, POSITIVE, "all triples verified")

        # cited but unverifiable in an otherwise usable subgraph:
        # this is an incompleteness case, not a proven hallucination -> neutral
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
    # Helpers (kept identical in spirit to the original)
    # ------------------------------------------------------------------

    @staticmethod
    def _is_discourse(text: str) -> bool:
        return bool(DISCOURSE_RE.match(text.strip()))

    @staticmethod
    def _has_factual_claim(text: str) -> bool:
        return bool(ENTITY_RE.search(text))

    @staticmethod
    def _contradicts(conclusion: str, prev_conclusions: List[str]) -> bool:
        import re

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

    @staticmethod
    def _entities(text: str) -> Set[str]:
        """Proper-noun entity surface forms (lower-cased) found in ``text``.

        Kept for API parity / future use; not currently consulted by the
        contradiction guard (see ``_is_contradiction`` for why entity overlap
        proved unreliable for subject matching).
        """
        return {e.lower() for e in ENTITY_RE.findall(text or "")}

    def _is_contradiction(self, conclusion: Optional[str], prev_conclusions: List[str]) -> bool:
        """Guarded contradiction test.

        Wraps the raw ``_contradicts`` heuristic but first vetoes *honest
        abstentions* — conclusions that report a gap or decline to commit
        ("no evidence that…", "does not directly link…", "no X is identified…",
        "cannot determine…"). These share words + a negation with a prior
        conclusion and so trip the bare heuristic, but they are the model
        correctly reporting absence, not a hallucinated contradiction.

        A genuine -1 is a *positive counter-assertion* about the same fact
        ("X is an architect, not a computer scientist" against an earlier
        "X is a computer scientist"). Those carry no abstention marker and so
        survive the guard.

        (An earlier attempt also required the contradicting pair to share a
        proper-noun entity, but ``ENTITY_RE`` extracts phrase fragments like
        "Rock"/"Roll Hall"/"Fame"/"May" rather than the grammatical subject, so
        the entity overlap was driven by object/venue names and the guard both
        kept false positives and dropped the true Levin case. Reliable subject
        extraction needs parsing; the abstention-phrasing guard is the robust
        signal here.)
        """
        if not conclusion or not prev_conclusions:
            return False
        if self.guard_abstention and _ABSTENTION_RE.search(conclusion):
            return False
        return self._contradicts(conclusion, prev_conclusions)

    def _triple_relevant(
        self,
        cited_triples: List[Tuple[str, str, str]],
        conclusion: Optional[str],
    ) -> bool:
        """True if at least one cited triple's head/tail entity touches the conclusion.

        A filler citation (real, in-subgraph, but supporting nothing in this step)
        shares no content token with the conclusion. If we cannot read the
        conclusion at all, default to True so we never *increase* false negatives.
        """
        if not conclusion:
            return True
        ctoks = _content_tokens(conclusion)
        concl_lc = conclusion.lower()
        for h, _r, t in cited_triples:
            # token-overlap (handles len>=3 entities)
            if ctoks and (_content_tokens(h) | _content_tokens(t)) & ctoks:
                return True
            # A2c tail: short/numeric entities (len<3, e.g. "UK", years) won't
            # survive _content_tokens, so also match the full entity as a phrase.
            for ent in (h, t):
                e = ent.strip().lower()
                if e and (re.search(rf"\b{re.escape(e)}\b", concl_lc)):
                    return True
        # If we cannot read any content token from the conclusion AND no entity
        # phrase matched, default to True so we never *increase* false negatives.
        if not ctoks:
            return True
        return False

    def _log(self, idx: int, label: int, reason: str) -> int:
        if self.verbose:
            name = {POSITIVE: "+1", NEUTRAL: "0", NEGATIVE: "-1"}[label]
            logger.debug("step %d -> %s (%s)", idx, name, reason)
        return label
