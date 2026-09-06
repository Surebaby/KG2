# mixed3-v4 identity protection and cohort integration audit

> Audit date: 2026-09-05  
> Scope: identity leakage prevention and answer-free cohort freezing only.  
> No planner, retrieval, GPU generation, training, Gold-label modification, or old-artifact deletion was performed by this audit.

## 1. Authoritative never-train ledger

The single authoritative release is:

`outputs/audits/mixed_ppo_v4_protected_identity_ledger_v2/`

The machine-readable input for all future PPO cohort freezers is:

`protected_identities.question_only.jsonl`

Its SHA256 is
`2fca962d48fa12fbc4e64eb479ba354e4fc8a319253b5879b80c3b8d90bf40ae`.
The bound `report.json` SHA256 is
`4866c39298280e36ee46d813cde2db39b209f0d014bfe0e97f1003490f3425ab`.

The release has `complete=true` and `current_family_recomputed=true`. Stored
family hashes from upstream files are provenance only; isolation uses the
current `answer-free-lexical-family-v1` recomputation.

| Dataset | unique dataset::qid | unique question hash | unique current family |
|---|---:|---:|---:|
| 2WikiMultiHopQA | 2,408 | 2,408 | 1,360 |
| HotpotQA | 1,140 | 1,140 | 1,132 |
| MuSiQue | 1,142 | 1,142 | 1,139 |
| **Total** | **4,690** | **4,690** | **3,631** |

The ledger combines 34 source files / 6,545 source rows and protects all
identified QPEG, SAEG, subquestion, controller development/confirmation,
reward-rankability, learned-verifier train/dev/confirmation/reserve, and
ProofKG planner/supply confirmation cohorts. It emits only identity fields and
source roles. Some upstream JSON objects contain Gold/outcome keys, but those
values were neither used to select identities nor emitted.

Several later artifacts reproduce identities from an earlier canonical freeze
and are therefore not duplicated as ledger inputs. Direct coverage checks found:

- QPEG-v4 development/confirmation: 150/150 and 300/300 protected;
- SAEG 2Wiki planner subset: 150/150 protected;
- inference-ProofKG confirmation: 810/810 protected;
- subquestion v8 development, v8 smoke, v9 and v9.1 executed rows: all protected
  (90/90 or 12/12 as applicable);
- L0 verifier train/dev/confirmation qids: 261/261, 60/60, and 100/100
  protected; fresh verifier candidates: 221/221 protected.

The actually trained controller v4.2/v4.3 identities are reproduced by v4.4
and are therefore covered by the v4.4 source. Earlier original/v4.1 protocols
were frozen but have no corresponding training or evaluation output in the
workspace; they are not treated as consumed cohorts. If those abandoned
protocols are ever reactivated, their identities must be added in a new ledger
version before any use.

### Required isolation rule

PPO training candidates must have zero dataset-scoped overlap with this ledger
at all three levels:

1. `dataset::qid`;
2. `dataset::question_sha256`;
3. `dataset::current_family_sha256`.

Future scripts must consume the complete release directory and validate the
ledger, report, and manifest hashes. Hand-maintained path subsets are not a
safe replacement.

## 2. Why the previous protected-file defaults were insufficient

The previous v4 path list protected 2,790 unique qids. The complete ledger
adds 1,900 unique qids. Recomputing all identities gives the following impact:

| Candidate asset | rows | overlap under old defaults (any) | overlap under complete ledger (any) |
|---|---:|---:|---:|
| mixed-v2 population | 1,799 | 37 | 366 |
| mixed-v2 ordinary subset | 200 | 2 | 14 |
| old automatic Proof cohort | 1,500 | 685 | 1,315 |
| old strict automatic Proof rows | 1,299 | 589 | 1,168 |
| Proof extension | 350 | 15 | 165 |
| interim H/M population | 2,000 | 0 | 11 |
| historical replay v1c | 2,000 | 69 | 138 |

“Any” is the union of qid, exact-question-hash, and current-family overlap.
These counts explain why neither the old 1,299 strict Proof rows nor the old
ordinary200/HM selections may be reused unchanged.

## 3. Evidence-store and old-Proof consequences

The superseded v5 store used a partial exclusion list. The complete-ledger v6
store is:

`indexes/versioned_2wiki_evidence_store_v6_mixed3_v4_complete_ledger_seed42/`

Against the 4,690-row complete ledger, the exclusions recorded by v5 covered
3,939 exact qids/hashes and 4,197 current families; 493 protected identities
had no qid/hash/family coverage at all. By dataset, v5 any-level coverage was
2,283/2,408 for 2Wiki, 934/1,140 for HotpotQA, and 980/1,142 for MuSiQue. V6
replaces that incomplete path collection with the single bound ledger release.

It excludes the complete protected ledger plus the relevant training cohorts.
Its store manifest records 5,887 eligible aligned official rows, 14,007 stored
evidence hops, 23,271 alias keys, and 10,526 edge keys. This lower capacity than
v5 is expected: v6 applies the larger leakage boundary.

After complete-ledger isolation, only 131 of the old 1,299 strict automatic
Proof rows remain safe:

| question type | safe old strict rows |
|---|---:|
| bridge/comparison | 94 |
| comparison | 25 |
| compositional | 7 |
| inference | 5 |

All 477 executed edges for those 131 questions were independently reproduced.
Root identities are reusable for 58 questions; the remaining 73 require clean
root re-resolution. This is an attestation/worklist result, not a new Proof
supply and not permission to reuse unaudited old traces.

## 4. Safe 2Wiki capacity and official-raw candidate pool

The existing `proofkg_curriculum_mix_v1` has 2,200 rows after excluding the old
automatic-1500 and extension-350 exact identities. After complete-ledger family
isolation, only 439 rows remain:

| type | safe rows | unique current families |
|---|---:|---:|
| bridge/comparison | 277 | 185 |
| comparison | 105 | 65 |
| compositional | 57 | 42 |
| inference | 0 | 0 |

Therefore this curriculum cannot supply the planned strict Proof800 cohort.

The answer-free replacement candidate pool is frozen at:

`outputs/audits/2wiki_proofkg_official_raw_v2_candidate_pool_n1500_seed42_preregistration/`

It contains 1,500 official-raw train identities and no answer, evidence,
passage, decomposition, or Gold field. It excludes the complete ledger, replay,
and the old automatic-1500 plus extension-350 exact identities.

| type | eligible capacity | eligible families | frozen candidates | frozen families |
|---|---:|---:|---:|---:|
| bridge/comparison | 15,141 | 3,256 | 390 | 390 |
| comparison | 13,332 | 3,109 | 390 | 390 |
| compositional | 8,882 | 1,416 | 389 | 389 |
| inference | 331 | 158 | 331 | 158 |
| **Total** | **37,686** | **7,939** | **1,500** | **1,327** |

Inference has only 158 unique families, so 173 repeated-family rows are
unavoidable if all 331 safe inference identities are retained. The other three
types use one row per current family. The 1,500 rows are a candidate pool for
yield and ablations; the balanced main PPO schedule remains 1,000 2Wiki groups
with strict ProofKG=800 and ordinary=200. The 1,500 candidates must not all be
inserted into the main schedule.

The strict post-planner yield is still `UNKNOWN`. In particular, selecting 200
strict inference rows from 331 candidates requires at least a 60.4% strict-pass
yield for that type. Failure to reach that yield must stop or trigger a new
pre-registered source strategy; it must not be repaired by silently changing
the main 200-per-type quota after seeing results.

## 5. H/M and ordinary successor cohorts

### HotpotQA/MuSiQue

The full-ledger-safe successor is:

`outputs/audits/mixed_ppo_v4_hm_full_ledger_reconciliation_v2_seed42_preregistration/`

It freezes H=1,000 and M=1,000. Eleven overlapping MuSiQue identities were
deterministically replaced. Of 818 previously materialized retrieval contexts,
812 can be reused, six are retired, and only 11 new retrieval requests remain.
The final context requirement is 823 rows: HotpotQA=417 and MuSiQue=406.

No retrieval was run in this reconciliation stage. The next retrieval release
must merge the 812 reused contexts with the 11 newly retrieved contexts and
contain exactly the frozen 823 identities.

### 2Wiki ordinary200

The successor is:

`outputs/audits/2wiki_ordinary200_full_ledger_v2_seed42_preregistration/`

It retains 148 parent ordinary rows and deterministically replaces 52. The
larger replacement count is intentional: the selection is isolated not only
from the complete ledger and replay but also from all Proof candidate pools,
including the new official-raw n=1,500 pool. The final 200 rows have 200 unique
qids, hashes, and current families, and each row binds its source file, source
line, source-record hash, and passage hash without emitting answer or passage
text.

## 6. Replay boundary

Replay is a separate training anchor rather than a never-train evaluation
ledger. The authoritative clean successor is:

`data/silver_data/sft_replay_legacy_strong_train_rendered3to5_n2000_seed42_v2/`

It has 2,000 accepted HotpotQA trajectories and zero qid/hash/current-family
overlap with the complete protected ledger, the reconciled H/M population, the
ordinary200 successor, and the new 2Wiki Proof candidate pool. Final v4 freezer
and materializer defaults must bind this v2 release, not historical v1c.

The three successor cohort artifacts above were originally frozen while v1c
was still the available replay and therefore retain that historical input
binding. Their identities were subsequently checked explicitly against replay
v2 with zero overlap at all three levels. The final v4 protocol is required to
repeat this live check and bind replay v2; the historical child protocols are
not rewritten in place.

## 7. Current readiness and remaining work

Completed without training:

- authoritative full protected ledger;
- complete-ledger v6 2Wiki evidence store;
- n=1,500 Gold-free official-raw Proof candidate pool;
- H/M deterministic replacements and 11-request retrieval delta;
- isolated ordinary200 successor with source provenance;
- main v4 freezer integration and focused regression tests.

Still required before the v4 dataset can be labelled ready for PPO:

1. execute only the 11 frozen H/M retrieval requests, then build an exactly
   823-row canonical merged retrieval release;
2. run planner/executor/closure over the n=1,500 2Wiki candidate pool;
3. apply strict Graph eligibility and select exactly 800 Proof rows;
4. materialize the final 3,000-question / K=4 schedule using replay v2 and the
   ordinary200 source-provenance join;
5. require every identity, retrieval, strict-Graph, schedule, and replay gate
   to pass before changing status from `NOT_READY_FOR_TRAINING`.

No PPO configuration should be finalized or launched until those five items
are complete.
