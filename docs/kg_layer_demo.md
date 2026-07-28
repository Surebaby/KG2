# KG 三层过滤效果示例
> 10 个 HotpotQA 问题 | L1 = SPARQL + Python QA filter | L2 = v2 scoring
---
## Example 1
**Q: The Album Against the Wind was the 11th Album of a Rock singer Robert C Seger born may 6 1945. What was the Rock singers stage name ?**
Gold: ['Bob Seger']

### Layer 0 (原始 SPARQL): 104 triples
```
  ⚠️ (Wind, instance of, family name)
  ⚠️ (Wind, different from, Wind)
     (Wind, writing system, Latin script)
     (Wind, language of work or name, English)
     (Wind, language of work or name, Danish)
  ⚠️ (Wind, said to be the same as, Vind)
     (Wind, attested in, 2010 United States Census surn)
     (family name, named after, family)
     (family name, named after, name)
  ⚠️ (family name, instance of, name particle)
  ⚠️ (family name, described by source, Brockhaus and Efron Encycloped)
  ⚠️ (family name, described by source, Armenian Soviet Encyclopedia)
  ... (+92 more)
```

### Layer 1 (QA filter): 30 triples
```
  ⚠️ (Wind, instance of, family name)
  ✅ (Wind, language of work or name, English)
  ✅ (Wind, language of work or name, Danish)
  ⚠️ (family name, instance of, name particle)
  ⚠️ (album, instance of, music release type)
  ⚠️ (music release type, instance of, form of art)
  ⚠️ (music release type, subclass of, class)
  ⚠️ (Template:Infobox album, instance of, Wikimedia infobox template)
  ✅ (Napster release ID, country, United States)
  ✅ (Qobuz album ID, country, France)
  ... (+20 more)
```

### Layer 2 (v2 scoring + filter): 6 triples
```
  ✅ (Wind, language of work or name, English)
  ✅ (Wind, language of work or name, Danish)
     (Qobuz album ID, country, France)
     (Acclaimed Music album ID, country, Sweden)
  ✅ (Wind, instance of, family name)
     (Napster release ID, country, United States)
```

| 层 | 数量 | 噪音 |
|---|---|---|
| L0 原始 | 104 | 67 (64%) |
| L1 QA filter | 30 | 16 (53%) |
| L2 v2 scoring | 6 | — |

---
## Example 2
**Q: The football manager who recruited David Beckham managed Manchester United during what timeframe?**
Gold: ['from 1986 to 2013']

### Layer 0 (原始 SPARQL): 94 triples
```
     (David Beckham, field of work, entrepreneurship)
     (David Beckham, field of work, literary activity)
     (David Beckham, field of work, association football)
     (David Beckham, field of work, autobiography)
     (David Beckham, native language, English)
     (David Beckham, occupation, model)
     (David Beckham, occupation, actor)
     (David Beckham, occupation, association football player)
     (David Beckham, occupation, businessperson)
     (David Beckham, occupation, entrepreneur)
     (David Beckham, occupation, blogger)
     (David Beckham, occupation, sports executive)
  ... (+82 more)
```

### Layer 1 (QA filter): 29 triples
```
  ✅ (David Beckham, occupation, model)
  ✅ (David Beckham, occupation, actor)
  ✅ (David Beckham, occupation, association football player)
  ✅ (David Beckham, occupation, businessperson)
  ✅ (David Beckham, occupation, entrepreneur)
  ✅ (David Beckham, occupation, blogger)
  ✅ (David Beckham, occupation, sports executive)
  ✅ (David Beckham, employer, UNICEF)
  ⚠️ (autobiography, instance of, literary genre)
  ⚠️ (autobiography, instance of, book form)
  ... (+19 more)
```

### Layer 2 (v2 scoring + filter): 8 triples
```
  ✅ (David Beckham, occupation, model)
  ✅ (David Beckham, occupation, actor)
  ✅ (David Beckham, employer, UNICEF)
  ✅ (Manchester United, country, United Kingdom)
     (Glory Glory, genre, traditional folk music)
     (Glory Glory, country, United Kingdom)
     (Glory Glory, language of work or name, English)
     (association football, instance of, team sport)
```

| 层 | 数量 | 噪音 |
|---|---|---|
| L0 原始 | 94 | 28 (30%) |
| L1 QA filter | 29 | 16 (55%) |
| L2 v2 scoring | 8 | — |

---
## Example 3
**Q: Are the Laleli Mosque and Esma Sultan Mansion located in the same neighborhood?**
Gold: ['no']

### Layer 0 (原始 SPARQL): 84 triples
```
     (Laleli Mosque, located in the administrative territorial entity, Istanbul)
     (Laleli Mosque, located in the administrative territorial entity, Fatih)
     (Laleli Mosque, religion or worldview, Islam)
     (Laleli Mosque, architectural style, Ottoman architecture)
     (Laleli Mosque, country, Turkey)
     (Laleli Mosque, made from material, granite)
  ⚠️ (Laleli Mosque, instance of, historic building)
  ⚠️ (Laleli Mosque, instance of, mosque)
     (Laleli Mosque, architect, Mimar Mehmet Tahir)
     (Laleli Mosque, commissioned by, Mustafa III)
     (Laleli Mosque, heritage designation, cultural property requiring pr)
  ⚠️ (Laleli Mosque, part of, Laleli Külliyesi)
  ... (+72 more)
```

### Layer 1 (QA filter): 24 triples
```
  ✅ (Laleli Mosque, located in the administrative territorial entity, Istanbul)
  ✅ (Laleli Mosque, located in the administrative territorial entity, Fatih)
  ✅ (Laleli Mosque, country, Turkey)
  ⚠️ (Laleli Mosque, instance of, historic building)
  ⚠️ (Laleli Mosque, instance of, mosque)
  ✅ (Istanbul, located in the administrative territorial entity, Istanbul Province)
  ✅ (Fatih, located in the administrative territorial entity, Istanbul Province)
  ✅ (Istanbul, country, Turkey)
  ✅ (Fatih, country, Turkey)
  ✅ (Esma Sultan Mansion, located in the administrative territorial entity, Beşiktaş)
  ... (+14 more)
```

### Layer 2 (v2 scoring + filter): 5 triples
```
  ✅ (Laleli Mosque, located in the administrative territorial entity, Istanbul)
  ✅ (Laleli Mosque, located in the administrative territorial entity, Fatih)
  ✅ (Laleli Mosque, country, Turkey)
  ✅ (Laleli Mosque, instance of, historic building)
     (Istanbul, country, Turkey)
```

| 层 | 数量 | 噪音 |
|---|---|---|
| L0 原始 | 84 | 13 (15%) |
| L1 QA filter | 24 | 10 (42%) |
| L2 v2 scoring | 5 | — |

---
## Example 4
**Q: In what month is the annual documentary film festival, that is presented by the fortnightly published British journal of literary essays, held? **
Gold: ['March and April']

### Layer 0 (原始 SPARQL): 38 triples
```
     (British Journal of Urolog, publisher, Wiley-Blackwell)
  ⚠️ (British Journal of Urolog, instance of, scientific journal)
     (British Journal of Urolog, replaced by, BJU International)
     (British Journal of Urolog, language of work or name, English)
     (British Journal of Urolog, archives at, CLOCKSS)
     (British Journal of Urolog, archives at, Portico)
     (British Journal of Urolog, country of origin, United Kingdom)
     (British Journal of Urolog, indexed in bibliographic review, Scopus)
     (Wiley-Blackwell, headquarters location, Hoboken)
     (Wiley-Blackwell, country, United Kingdom)
     (Wiley-Blackwell, country, United States)
  ⚠️ (Wiley-Blackwell, instance of, book publisher)
  ... (+26 more)
```

### Layer 1 (QA filter): 10 triples
```
  ⚠️ (British Journal of Urolog, instance of, scientific journal)
  ✅ (British Journal of Urolog, language of work or name, English)
  ✅ (Wiley-Blackwell, headquarters location, Hoboken)
  ✅ (Wiley-Blackwell, country, United Kingdom)
  ✅ (Wiley-Blackwell, country, United States)
  ⚠️ (Wiley-Blackwell, instance of, book publisher)
  ⚠️ (Wiley-Blackwell, instance of, publishing house)
  ⚠️ (scientific journal, instance of, magazine genre)
  ⚠️ (scientific journal, subclass of, journal)
  ⚠️ (scientific journal, subclass of, scientific publication)
```

### Layer 2 (v2 scoring + filter): 4 triples
```
     (British Journal of Urolog, instance of, scientific journal)
     (British Journal of Urolog, language of work or name, English)
     (Wiley-Blackwell, headquarters location, Hoboken)
     (Wiley-Blackwell, country, United Kingdom)
```

| 层 | 数量 | 噪音 |
|---|---|---|
| L0 原始 | 38 | 20 (53%) |
| L1 QA filter | 10 | 6 (60%) |
| L2 v2 scoring | 4 | — |

---
## Example 5
**Q: Alexander Kerensky was defeated and destroyed by the Bolsheviks in the course of a civil war that ended when ?**
Gold: ['October 1922']

### Layer 0 (原始 SPARQL): 82 triples
```
     (Alexander Kerensky, genre, portrait)
     (Alexander Kerensky, creator, Ilya Repin)
     (Alexander Kerensky, depicts, Alexander Kerensky)
     (Alexander Kerensky, made from material, linoleum)
     (Alexander Kerensky, made from material, oil paint)
  ⚠️ (Alexander Kerensky, instance of, painting)
     (Alexander Kerensky, copyright status, public domain)
     (Alexander Kerensky, main subject, Alexander Kerensky)
     (portrait, depicts, person)
  ⚠️ (portrait, instance of, form of art)
  ⚠️ (portrait, instance of, art genre)
     (portrait, facet of, figurative art)
  ... (+70 more)
```

### Layer 1 (QA filter): 20 triples
```
  ✅ (Alexander Kerensky, genre, portrait)
  ✅ (Alexander Kerensky, creator, Ilya Repin)
  ⚠️ (Alexander Kerensky, instance of, painting)
  ⚠️ (portrait, instance of, form of art)
  ⚠️ (portrait, instance of, art genre)
  ⚠️ (portrait, subclass of, image)
  ⚠️ (portrait, subclass of, work)
  ✅ (Ilya Repin, occupation, teacher)
  ✅ (Ilya Repin, occupation, painter)
  ✅ (Ilya Repin, occupation, graphic artist)
  ... (+10 more)
```

### Layer 2 (v2 scoring + filter): 6 triples
```
  ✅ (Alexander Kerensky, genre, portrait)
  ✅ (Alexander Kerensky, creator, Ilya Repin)
  ✅ (Alexander Kerensky, instance of, painting)
     (Ilya Repin, occupation, teacher)
     (Ilya Repin, occupation, painter)
     (Ilya Repin, employer, Higher Art School at the Imper)
```

| 层 | 数量 | 噪音 |
|---|---|---|
| L0 原始 | 82 | 51 (62%) |
| L1 QA filter | 20 | 12 (60%) |
| L2 v2 scoring | 6 | — |

---
## Example 6
**Q: Hayden is a singer-songwriter from Canada, but where does Buck-Tick hail from?**
Gold: ['Fujioka, Gunma']

### Layer 0 (原始 SPARQL): 112 triples
```
     (Hayden, located in the administrative territorial entity, Kootenai County)
     (Hayden, country, United States of America)
     (Hayden, located in or next to body of water, Lake Hayden)
  ⚠️ (Hayden, instance of, city in the United States)
     (Hayden, category of associated people, Category:People from Hayden, I)
  ⚠️ (Hayden, different from, Hayden Lake)
  ⚠️ (Hayden, topic's main category, Category:Hayden, Idaho)
     (Kootenai County, located in the administrative territorial entity, Idaho)
     (Kootenai County, named after, Kootenai Tribe of Idaho)
     (Kootenai County, contains the administrative territorial entity, Coeur d'Alene)
     (Kootenai County, contains the administrative territorial entity, Athol)
     (Kootenai County, country, United States of America)
  ... (+100 more)
```

### Layer 1 (QA filter): 21 triples
```
  ✅ (Hayden, located in the administrative territorial entity, Kootenai County)
  ✅ (Hayden, country, United States of America)
  ⚠️ (Hayden, instance of, city in the United States)
  ✅ (Kootenai County, located in the administrative territorial entity, Idaho)
  ✅ (Kootenai County, country, United States of America)
  ⚠️ (Kootenai County, instance of, county of Idaho)
  ✅ (Kootenai County, capital, Coeur d'Alene)
  ✅ (Kootenai County, shares border with, Bonner County)
  ✅ (Kootenai County, shares border with, Spokane County)
  ✅ (Kootenai County, shares border with, Benewah County)
  ... (+11 more)
```

### Layer 2 (v2 scoring + filter): 8 triples
```
  ✅ (Hayden, located in the administrative territorial entity, Kootenai County)
  ✅ (Hayden, country, United States of America)
  ✅ (Hayden, instance of, city in the United States)
     (Kootenai County, located in the administrative territorial entity, Idaho)
     (Kootenai County, country, United States of America)
     (Kootenai County, capital, Coeur d'Alene)
     (Kootenai County, shares border with, Bonner County)
     (Kootenai County, shares border with, Spokane County)
```

| 层 | 数量 | 噪音 |
|---|---|---|
| L0 原始 | 112 | 15 (13%) |
| L1 QA filter | 21 | 6 (29%) |
| L2 v2 scoring | 8 | — |

---
## Example 7
**Q: Kaiser Ventures corporation was founded by an American industrialist who became known as the father of modern American shipbuilding?**
Gold: ['Henry J. Kaiser']

### Layer 0 (原始 SPARQL): 86 triples
```
     (Kaiser Ventures, founder, Henry J. Kaiser)
     (Kaiser Ventures, country, United States)
  ⚠️ (Kaiser Ventures, instance of, business)
     (Kaiser Ventures, industry, iron and steel industry)
     (Henry J. Kaiser, field of work, shipyard)
     (Henry J. Kaiser, field of work, industry)
     (Henry J. Kaiser, occupation, engineer)
     (Henry J. Kaiser, occupation, industrialist)
     (Henry J. Kaiser, occupation, inventor)
     (Henry J. Kaiser, occupation, entrepreneur)
     (Henry J. Kaiser, occupation, philanthropist)
     (Henry J. Kaiser, place of burial, Mountain View Cemetery)
  ... (+74 more)
```

### Layer 1 (QA filter): 17 triples
```
  ✅ (Kaiser Ventures, country, United States)
  ⚠️ (Kaiser Ventures, instance of, business)
  ✅ (Henry J. Kaiser, occupation, engineer)
  ✅ (Henry J. Kaiser, occupation, industrialist)
  ✅ (Henry J. Kaiser, occupation, inventor)
  ✅ (Henry J. Kaiser, occupation, entrepreneur)
  ✅ (Henry J. Kaiser, occupation, philanthropist)
  ✅ (Henry J. Kaiser, place of birth, Sprout Brook)
  ✅ (Henry J. Kaiser, sex or gender, male)
  ✅ (Henry J. Kaiser, country of citizenship, United States)
  ... (+7 more)
```

### Layer 2 (v2 scoring + filter): 10 triples
```
     (Henry J. Kaiser, country of citizenship, United States)
  ✅ (Kaiser Ventures, country, United States)
  ✅ (Henry J. Kaiser, occupation, industrialist)
     (Henry J. Kaiser, occupation, engineer)
     (Henry J. Kaiser, place of birth, Sprout Brook)
     (Henry J. Kaiser, sex or gender, male)
     (Henry J. Kaiser, child, Edgar Kaiser, Sr)
     (American English, country, United States)
  ... (+2 more)
```

| 层 | 数量 | 噪音 |
|---|---|---|
| L0 原始 | 86 | 27 (31%) |
| L1 QA filter | 17 | 6 (35%) |
| L2 v2 scoring | 10 | — |

---
## Example 8
**Q: Are Giuseppe Verdi and Ambroise Thomas both Opera composers ?**
Gold: ['yes']

### Layer 0 (原始 SPARQL): 174 triples
```
     (Giuseppe Verdi, field of work, music)
     (Giuseppe Verdi, field of work, opera)
     (Giuseppe Verdi, field of work, art music)
     (Giuseppe Verdi, field of work, conducting)
     (Giuseppe Verdi, field of work, politics)
     (Giuseppe Verdi, member of political party, Historical Right)
     (Giuseppe Verdi, occupation, writer)
     (Giuseppe Verdi, occupation, conductor)
     (Giuseppe Verdi, occupation, composer)
     (Giuseppe Verdi, occupation, politician)
     (Giuseppe Verdi, place of burial, Casa di Riposo per Musicisti)
     (Giuseppe Verdi, movement, Romantic music)
  ... (+162 more)
```

### Layer 1 (QA filter): 62 triples
```
  ✅ (Giuseppe Verdi, member of political party, Historical Right)
  ✅ (Giuseppe Verdi, occupation, writer)
  ✅ (Giuseppe Verdi, occupation, conductor)
  ✅ (Giuseppe Verdi, occupation, composer)
  ✅ (Giuseppe Verdi, occupation, politician)
  ✅ (Giuseppe Verdi, genre, opera)
  ✅ (Giuseppe Verdi, genre, classical music)
  ✅ (Giuseppe Verdi, place of birth, Le Roncole)
  ✅ (Giuseppe Verdi, sex or gender, male)
  ✅ (Giuseppe Verdi, spouse, Giuseppina Strepponi)
  ... (+52 more)
```

### Layer 2 (v2 scoring + filter): 28 triples
```
  ✅ (Ambroise Thomas, country of citizenship, France)
  ✅ (Giuseppe Verdi, genre, opera)
  ✅ (Ambroise Thomas, genre, opera)
  ✅ (Giuseppe Verdi, member of political party, Historical Right)
  ✅ (Giuseppe Verdi, occupation, writer)
  ✅ (Giuseppe Verdi, occupation, conductor)
  ✅ (Giuseppe Verdi, occupation, composer)
  ✅ (Giuseppe Verdi, occupation, politician)
  ... (+20 more)
```

| 层 | 数量 | 噪音 |
|---|---|---|
| L0 原始 | 174 | 82 (47%) |
| L1 QA filter | 62 | 36 (58%) |
| L2 v2 scoring | 28 | — |

---
## Example 9
**Q: Are Random House Tower and 888 7th Avenue both used for real estate?**
Gold: ['no']

### Layer 0 (原始 SPARQL): 44 triples
```
  ⚠️ (avenue, instance of, road type)
  ⚠️ (avenue, described by source, Great Soviet Encyclopedia (192)
  ⚠️ (avenue, described by source, Encyclopædia Britannica 11th e)
  ⚠️ (avenue, described by source, Meyers Konversations-Lexikon, )
     (avenue, has characteristic, roadside tree)
     (avenue, has characteristic, large)
  ⚠️ (avenue, different from, avenue)
  ⚠️ (avenue, subclass of, street)
  ⚠️ (avenue, said to be the same as, boulevard)
  ⚠️ (avenue, said to be the same as, avenue)
  ⚠️ (avenue, said to be the same as, prospekt)
  ⚠️ (avenue, said to be the same as, třída)
  ... (+32 more)
```

### Layer 1 (QA filter): 14 triples
```
  ⚠️ (avenue, instance of, road type)
  ⚠️ (avenue, subclass of, street)
  ⚠️ (road type, instance of, second-order class)
  ⚠️ (road type, subclass of, type of physical object)
  ✅ (Encyclopædia Britannica 1, genre, encyclopedia)
  ✅ (Meyers Konversations-Lexi, genre, lexicon)
  ⚠️ (Encyclopædia Britannica 1, instance of, version, edition or translatio)
  ⚠️ (Meyers Konversations-Lexi, instance of, version, edition or translatio)
  ⚠️ (Great Soviet Encyclopedia, instance of, version, edition or translatio)
  ✅ (Meyers Konversations-Lexi, author, group of authors)
  ... (+4 more)
```

### Layer 2 (v2 scoring + filter): 6 triples
```
  ✅ (avenue, instance of, road type)
     (Encyclopædia Britannica 1, genre, encyclopedia)
     (Meyers Konversations-Lexi, genre, lexicon)
     (Meyers Konversations-Lexi, author, group of authors)
     (Meyers Konversations-Lexi, language of work or name, German)
     (Meyers Konversations-Lexi, language of work or name, Czech)
```

| 层 | 数量 | 噪音 |
|---|---|---|
| L0 原始 | 44 | 17 (39%) |
| L1 QA filter | 14 | 7 (50%) |
| L2 v2 scoring | 6 | — |

---
## Example 10
**Q: In what year was the novel that Lourenço Mutarelli based "Nina" on based first published?**
Gold: ['1866']

### Layer 0 (原始 SPARQL): 92 triples
```
  ⚠️ (Mutarelli, instance of, family name)
     (Mutarelli, writing system, Latin script)
     (Mutarelli, language of work or name, Italian)
     (family name, named after, family)
     (family name, named after, name)
  ⚠️ (family name, instance of, name particle)
  ⚠️ (family name, described by source, Brockhaus and Efron Encycloped)
  ⚠️ (family name, described by source, Armenian Soviet Encyclopedia)
  ⚠️ (family name, described by source, Small Brockhaus and Efron Ency)
     (family name, replaces, disc number)
     (family name, partially coincident with, house name)
     (family name, partially coincident with, first name)
  ... (+80 more)
```

### Layer 1 (QA filter): 17 triples
```
  ⚠️ (Mutarelli, instance of, family name)
  ✅ (Mutarelli, language of work or name, Italian)
  ⚠️ (family name, instance of, name particle)
  ⚠️ (Nina, instance of, female given name)
  ✅ (Nina, language of work or name, Spanish)
  ✅ (Nina, language of work or name, Polish)
  ✅ (Nina, language of work or name, Dutch)
  ✅ (Nina, language of work or name, Czech)
  ✅ (Nina, language of work or name, Atayal)
  ✅ (Nina, language of work or name, Norwegian)
  ... (+7 more)
```

### Layer 2 (v2 scoring + filter): 5 triples
```
  ✅ (Mutarelli, language of work or name, Italian)
  ✅ (Nina, language of work or name, Spanish)
  ✅ (Mutarelli, instance of, family name)
     (Wiktionary, owned by, Wikimedia Foundation)
     (Wiktionary, genre, dictionary wiki)
```

| 层 | 数量 | 噪音 |
|---|---|---|
| L0 原始 | 92 | 43 (47%) |
| L1 QA filter | 17 | 7 (41%) |
| L2 v2 scoring | 5 | — |

---
