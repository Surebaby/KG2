# KG Build Report — question_kg_index_v2
Generated: 2026-07-30 23:57
Builder: r9v6-kg-2 | Relation policy: rel-2

## Summary
| Metric | Before | After | Change |
|---|---|---|---|
| Entries | 40090 | 40085 | — |
| Total triples | 1913892 | 826200 | 1087692 removed (56.8%) |
| Hard deleted | — | 29351 | — |
| Quota/score dropped | — | 1058341 | — |
| Avg triples/question | 47.7 | 20.6 | — |

## Coverage per dataset
| Dataset | Questions | Triples before | Triples after | Empty KG |
|---|---|---|---|---|
| 2wikimultihopqa | 12576 | 631406 | 315466 | 314 (2.5%) |
| hotpotqa | 25097 | 1175415 | 461452 | 5845 (23.3%) |
| musique | 2417 | 107071 | 49282 | 299 (12.4%) |

## Entity linking
| Metric | Value |
|---|---|
| Mentions processed | 91107 |
| Linked (high confidence) | 65219 (71.6%) |
| Abstained | 25888 (28.4%) |

## Top 20 Relations: Before
| Relation | Count |
|---|---|
| instance of | 337651 |
| contains the administrative territorial entity | 164801 |
| subclass of | 155248 |
| country | 120101 |
| has part(s) | 113255 |
| part of | 71239 |
| member of | 59555 |
| occupation | 53944 |
| cast member | 42429 |
| shares border with | 38946 |
| sex or gender | 37734 |
| country of citizenship | 32212 |
| language of work or name | 31373 |
| genre | 31018 |
| place of birth | 30451 |
| named after | 26557 |
| head of government | 24116 |
| located in the administrative territorial entity | 22695 |
| child | 21304 |
| country of origin | 18242 |

## Top 20 Relations: After
| Relation | Count |
|---|---|
| country | 64431 |
| instance of | 40519 |
| contains the administrative territorial entity | 35910 |
| occupation | 35864 |
| has part(s) | 30035 |
| cast member | 29236 |
| part of | 29083 |
| member of | 27589 |
| country of citizenship | 22486 |
| genre | 22079 |
| place of birth | 21463 |
| language of work or name | 21140 |
| located in the administrative territorial entity | 17563 |
| sex or gender | 17315 |
| shares border with | 16233 |
| head of government | 16012 |
| named after | 15958 |
| country of origin | 15753 |
| child | 14729 |
| subclass of | 13990 |

## Taxonomic Relation Ratio
| Metric | Before | After |
|---|---|---|
| instance_of + subclass_of | 25.8% | 6.6% |
| Target | — | < 25% |