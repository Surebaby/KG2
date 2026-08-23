# KG Build Report — question_kg_index_v2
Generated: 2026-08-19 15:11
Builder: r9v6-kg-2 | Relation policy: rel-2

## Summary
| Metric | Before | After | Change |
|---|---|---|---|
| Entries | 22398 | 22393 | — |
| Total triples | 1188553 | 355395 | 833158 removed (70.1%) |
| Hard deleted | — | 70883 | — |
| Quota/score dropped | — | 762275 | — |
| Avg triples/question | 53.1 | 15.9 | — |

## Coverage per dataset
| Dataset | Questions | Triples before | Triples after | Empty KG |
|---|---|---|---|---|
| 2wikimultihopqa | 12576 | 634150 | 199581 | 395 (3.1%) |
| hotpotqa | 7405 | 447599 | 124341 | 407 (5.5%) |
| musique | 2417 | 106804 | 31473 | 299 (12.4%) |

## Entity linking
| Metric | Value |
|---|---|
| Mentions processed | 46073 |
| Linked (high confidence) | 41984 (91.1%) |
| Abstained | 4089 (8.9%) |

## Top 20 Relations: Before
| Relation | Count |
|---|---|
| instance of | 219180 |
| subclass of | 104841 |
| country | 66809 |
| contains the administrative territorial entity | 52007 |
| has part(s) | 44452 |
| occupation | 43708 |
| cast member | 39167 |
| part of | 37041 |
| member of | 32791 |
| sex or gender | 31227 |
| country of citizenship | 26318 |
| place of birth | 24610 |
| shares border with | 23073 |
| genre | 22595 |
| child | 18713 |
| language of work or name | 16374 |
| head of government | 15075 |
| located in the administrative territorial entity | 14933 |
| place of death | 14152 |
| spouse | 13694 |

## Top 20 Relations: After
| Relation | Count |
|---|---|
| cast member | 26037 |
| country | 20507 |
| occupation | 20369 |
| genre | 15269 |
| place of birth | 12083 |
| country of citizenship | 12075 |
| child | 10507 |
| language of work or name | 9455 |
| country of origin | 9174 |
| contains the administrative territorial entity | 8730 |
| member of | 8127 |
| instance of | 8063 |
| spouse | 7635 |
| located in the administrative territorial entity | 7609 |
| part of | 7512 |
| father | 7122 |
| director | 7053 |
| has part(s) | 6736 |
| place of death | 6636 |
| sibling | 6390 |

## Taxonomic Relation Ratio
| Metric | Before | After |
|---|---|---|
| instance_of + subclass_of | 27.3% | 3.1% |
| Target | — | < 25% |