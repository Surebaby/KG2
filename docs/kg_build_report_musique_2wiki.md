# KG Build Report — question_kg_index_v2
Generated: 2026-08-19 14:56
Builder: r9v6-kg-2 | Relation policy: rel-2

## Summary
| Metric | Before | After | Change |
|---|---|---|---|
| Entries | 14993 | 22393 | — |
| Total triples | 740947 | 190291 | 550656 removed (74.3%) |
| Hard deleted | — | 47831 | — |
| Quota/score dropped | — | 502825 | — |
| Avg triples/question | 49.4 | 8.5 | — |

## Coverage per dataset
| Dataset | Questions | Triples before | Triples after | Empty KG |
|---|---|---|---|---|
| 2wikimultihopqa | 12576 | 634150 | 167908 | 1041 (8.3%) |
| musique | 2417 | 106797 | 22383 | 481 (19.9%) |

## Entity linking
| Metric | Value |
|---|---|
| Mentions processed | 46073 |
| Linked (high confidence) | 41984 (91.1%) |
| Abstained | 4089 (8.9%) |

## Top 20 Relations: Before
| Relation | Count |
|---|---|
| instance of | 135273 |
| subclass of | 69475 |
| cast member | 35618 |
| country | 35600 |
| occupation | 29649 |
| sex or gender | 23451 |
| part of | 21661 |
| has part(s) | 21070 |
| country of citizenship | 19613 |
| place of birth | 18372 |
| genre | 16019 |
| child | 15926 |
| member of | 15899 |
| contains the administrative territorial entity | 15805 |
| shares border with | 12507 |
| place of death | 11993 |
| language of work or name | 10760 |
| spouse | 10685 |
| country of origin | 9531 |
| given name | 8724 |

## Top 20 Relations: After
| Relation | Count |
|---|---|
| cast member | 26079 |
| country | 22619 |
| occupation | 20309 |
| genre | 14952 |
| place of birth | 12416 |
| country of citizenship | 12024 |
| child | 9983 |
| member of | 9792 |
| country of origin | 9517 |
| contains the administrative territorial entity | 9379 |
| language of work or name | 8906 |
| located in the administrative territorial entity | 8236 |
| instance of | 7863 |
| spouse | 7492 |
| part of | 7460 |
| father | 7099 |
| director | 7052 |
| place of death | 6474 |
| has part(s) | 6394 |
| educated at | 6390 |

## Taxonomic Relation Ratio
| Metric | Before | After |
|---|---|---|
| instance_of + subclass_of | 27.6% | 5.5% |
| Target | — | < 25% |