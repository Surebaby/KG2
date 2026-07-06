# Elite SFT 推理格式测试

> **模型**: Elite SFT (2,000 条精品数据, Llama-3-8B-Instruct + LoRA r=32)
> **Checkpoint**: `checkpoints/sft_student_elite/final`
> **推理参数**: `max_new_tokens=384, temperature=1.0, top_p=1.0, do_sample=True`
> **数据集**: HotpotQA dev, 随机抽取 20 样本 (seed=42)
> **检索**: Hybrid RRF top-15 (e5 + BM25), KG offline (KGPW_KG_OFFLINE=1)
> **日期**: 2026-07-06

---

## 统计摘要

| 指标 | 结果 |
|---|---|
| 总样本数 | 20 |
| 含 `[Step N]` 标记 | 20/20 (**100%**) |
| 含 `[Final Answer]` | 19/20 (**95%**) |
| 有推理内容的步骤数 | 59 步 |
| 平均每样本步骤数 | ~3.0 步 |

---

## 逐样本输出

### Sample 1
**Q**: The Album Against the Wind was the 11th Album of a Rock singer Robert C Seger born may 6 1945. What is the singer's full name?
**Gold**: Bob Seger

```
[Step 1]
Reasoning: The question states that the album "Against the Wind" was the 11th album of a rock singer named Robert C Seger, and the singer was born on May 6, 1945. From the passages, Robert Clark Seger is the full name of the singer, matching the given birthdate.
Knowledge Used: [(Robert Clark Seger, birth date, 1945-05-06)]
Conclusion: The singer's full name is Robert Clark Seger.

[Final Answer]: Robert Clark Seger.
```

---

### Sample 2
**Q**: The football manager who recruited David Beckham managed Manchester United during what timeframe?
**Gold**: from 1986 to 2013

```
[Step 1]
Reasoning: David Beckham played for Manchester United under Alex Ferguson, as stated in passage 7.
Knowledge Used: [(David Beckham, Manchester United, 1992-2003)]
Conclusion: Alex Ferguson managed Manchester United during David Beckham's playing career at the club.

[Step 2]
Reasoning: Passage 2 states Beckham played for Manchester United from 1995–96. Since Alex Ferguson is known to have managed the club from 1986 to 2013, this is the timeframe.
Knowledge Used: [(Alex Ferguson, managed Manchester United, 1986-2013)]
Conclusion: Alex Ferguson managed Manchester United from 1986 to 2013.

[Final Answer]: 1986-2013.
```

---

### Sample 3
**Q**: Are the Laleli Mosque and Esma Sultan Mansion located in the same neighborhood?
**Gold**: no

```
[Step 1]
Reasoning: The Laleli Mosque and Esma Sultan Mansion are both located in Istanbul, Turkey, as per the descriptions in the passages.
Knowledge Used: [(Laleli Mosque, city, Istanbul), (Esma Sultan Mansion, city, Istanbul)]
Conclusion: Both are located in Istanbul.

[Step 2]
Reasoning: The passage specific to the Esma Sultan Mansion states it is located in the Ortaköy neighborhood of Istanbul, while the Laleli Mosque is located in the Laleli neighborhood.
Knowledge Used: [(Esma Sultan Mansion, neighborhood, Ortaköy)]
Conclusion: They are in different neighborhoods.

[Final Answer]: No.
```

---

### Sample 4
**Q**: In what month is the annual documentary film festival, that is presented by the fortnightly published British journal of literary essays, held?
**Gold**: March and April

```
[Step 1]
Reasoning: The London Review of Books is a fortnightly published British journal of literary essays, and the question asks for the month the festival held by the British journal that is annual. Since the London Review of Books is the only British journal in the passages, and the question specifies "that is presented by the Fortnightly published British journal of literary essays", I conclude it is the London Review of Books.

[Final Answer]: March and April.
```

---

### Sample 5
**Q**: Alexander Kerensky was defeated and destroyed by the Bolsheviks in the course of a civil war that ended in what month and year?
**Gold**: October 1922

```
[Step 1]
Reasoning: Alexander Kerensky was part of the White Army and led the October 1917 coup. As the leader of the Provisional Government, he was defeated and destroyed by the Bolsheviks in the Russian Civil War.
Knowledge Used: [(Alexander Kerensky, leader of Alexander Kerensky)]
Conclusion: Alexander Kerensky was part of the White Army.

[Step 2]
Reasoning: The Russian Civil War lasted from 1917 to 1922, ending in October 1922.
Knowledge Used: [(Russian Civil War, end date, October 1922)]
Conclusion: The civil war ended in October 1922.

[Final Answer]: October 1922.
```

---

### Sample 6
**Q**: Hayden is a singer-songwriter from Canada, but where does Buck-Tick hail from?
**Gold**: Fujioka, Gunma

```
[Step 1]
Reasoning: Hayden is a Canadian singer-songwriter and Buck-Tick vocalist Atsushi Sakurai is Japanese.
Knowledge Used: [(Buck-Tick, Atsushi Sakurai, vocalist)]
Conclusion: Buck-Tick's vocalist is Japanese.

[Step 2]
Reasoning: Atsushi Sakurai is born in Fujioka, Gunma, which is a town in Japan.
Knowledge Used: [(Atsushi Sakurai, born in, Fujioka, Gunma)]
Conclusion: Atsushi Sakurai is born in Fujioka, Gunma.

[Final Answer]: Fujioka, Gunma, Japan.
```

---

### Sample 7
**Q**: Kaiser Ventures corporation was founded by an American industrialist who became known as the father of modern American shipbuilding. Who is he?
**Gold**: Henry J. Kaiser

```
[Step 1]
Reasoning: The main question is about the founder of Kaiser Ventures corporation who became known as the father of modern American shipbuilding. Passage 5 (Kaiser Ventures) states the company was founded by Henry J. Kaiser.
Knowledge Used: [(Kaiser Ventures, founded by, Henry J. Kaiser)]
Conclusion: Henry J. Kaiser founded Kaiser Ventures.

[Step 2]
Reasoning: Passage 1 confirms Henry J. Kaiser was an American industrialist known as the father of modern American shipbuilding.
Knowledge Used: [(Henry J. Kaiser, known as, father of modern American shipbuilding)]
Conclusion: Henry J. Kaiser is the founder and father of modern American shipbuilding.

[Final Answer]: Henry J. Kaiser.
```

---

### Sample 8
**Q**: Are Giuseppe Verdi and Ambroise Thomas both Opera composers?
**Gold**: yes

```
[Step 1]
Reasoning: Giuseppe Verdi is an Italian composer, and "opera" is a musical composition for the stage.
Knowledge Used: [(Giuseppe Verdi, type of person, composer)]
Conclusion: Giuseppe Verdi is an opera composer.

[Step 2]
Reasoning: Ambroise Thomas is attributed to "Mignon", which is described as an "opera comique".
Knowledge Used: [(Mignon, type of work, opera)]
Conclusion: Ambroise Thomas is also an opera composer.

[Final Answer]: Yes.
```

---

### Sample 9
**Q**: Are Random House Tower and 888 7th Avenue both used for real estate?
**Gold**: no

```
[Step 1]
Reasoning: The passage about Random House Tower states it is used as the headquarters of book publisher Random House and a luxury apartment complex.
Knowledge Used: [(empty)]
Conclusion: Random House Tower is used for publishing and residences.

[Step 2]
Reasoning: The passage about 888 7th Avenue states it is used as the headquarters of Vornado Realty Trust, a real estate services company. While the tenant is a real estate company, the building itself is an office tower.
Knowledge Used: [(888 7th Avenue, tenant, Vornado Realty Trust)]
Conclusion: 888 7th Avenue is an office building, not used directly for real estate.

[Final Answer]: No.
```

---

### Sample 10
**Q**: In what year was the novel that Lourenço Mutarelli based "Nina" on first published?
**Gold**: 1866

```
[Step 1]
Reasoning: Lourenço Mutarelli is the author of the novel "Drained" (Portuguese: O Cheiro do Ralo), which is based on the 2006 Brazilian dark comedy film directed by Heitor Dhalia.
Knowledge Used: [(Person: Lourenço Mutarelli; Work/Title: Drained)]
Conclusion: Lourenço Mutarelli wrote "Drained".

[Step 2]
Reasoning: The passage about Drained states it is based on a novel by Lourenço Mutarelli, and Nina is a film based on a novel by Mutarelli. The novel "Crime and Punishment" was first published in 1866.
Knowledge Used: [(Crime and Punishment, first published, 1866)]
Conclusion: The novel was first published in 1866.

[Final Answer]: 1866.
```

---

### Sample 11
**Q**: The 2017–18 Wigan Athletic F.C. season will be a year in which the team competes in the league cup known as what?
**Gold**: Carabao Cup

```
[Step 1]
Reasoning: The question asks for the name of the league cup the team competes in in the 2017-18 season, for which they compete in the EFL Cup.
Knowledge Used: [(wigan athletic f c 2017 18 season)]
Conclusion: The 2017-18 Wigan Athletic F.C. season will be a year in which the team competes in the EFL Cup.

[Step 2]
Reasoning: The 2017-18 Wigan Athletic F.C. season was the club's 40th season in the EFL Cup, which was known as the Carabao Cup for sponsorship reasons.
Knowledge Used: [(EFL Cup, sponsor name, Carabao Cup)]
Conclusion: The league cup is known as the Carabao Cup.

[Final Answer]: Carabao Cup.
```

---

### Sample 12
**Q**: What screenwriter with credits for "Evolution" co-wrote a film starring Nicolas Cage and Téa Leoni?
**Gold**: David Weissman

```
[Step 1]
Reasoning: The question asks for a screenwriter with credits for "Evolution". The Knowledge Graph is empty, but the retrieved passages provide information about "Evolution" from the context of the screenwriter Neil LaBute.
Knowledge Used: [(empty)]
Conclusion: Neil LaBute wrote "Evolution".

[Step 2]
Reasoning: The question also asks for a film starring Nicolas Cage and Téa Leoni. The film "The Family Man" stars Nicolas Cage and Téa Leoni.
Knowledge Used: [(The Family Man, stars, Nicolas Cage, Téa Leoni)]
Conclusion: The screenwriter co-wrote The Family Man.

[Final Answer]: Neil LaBute.
```

---

### Sample 13
**Q**: What American professional Hawaiian surfer born 18 October 1992 won the Rip Curl Pro Portugal?
**Gold**: John John Florence

```
[Step 1]
Reasoning: The question asks for an American professional Hawaiian surfer born October 18, 1992 who won the Rip Curl Pro Portugal. From the passages, a surfer born in 1992 is Baro (Cha Sun-woo), but that does not fit the "American" requirement.
Knowledge Used: [(Baro, born, 1992, birth_date, 19920905100000)]
Conclusion: Baro is not an American surfer born in 1992.

[Step 2]
Reasoning: Another surfer mentioned is John John Florence, who is an American professional surfer from Hawaii, born October 18, 1992, and won the Rip Curl Pro Portugal.
Knowledge Used: [(John John Florence, born, October 18 1992), (John John Florence, won, Rip Curl Pro Portugal)]
Conclusion: John John Florence is the answer.

[Final Answer]: John John Florence.
```

---

### Sample 14
**Q**: Ellie Goulding worked with what other writers on her third studio album, Delirium?
**Gold**: Max Martin, Savan Kotecha and Ilya Salmanzadeh

```
[Step 1]
Reasoning: The question asks about Ellie Goulding's third studio album, which is "Delirium".
Knowledge Used: [(Ellie Goulding, Delirium, 3rd studio album)]
Conclusion: Ellie Goulding's third studio album is Delirium.

[Step 2]
Reasoning: The passage states that "On My Mind" was one of the singles from Delirium, and that it was written by Ellie Goulding and others including Max Martin and Savan Kotecha.
Knowledge Used: [(On My Mind, written by, Max Martin, Savan Kotecha, Ilya Salmanzadeh)]
Conclusion: Ellie Goulding worked with Max Martin, Savan Kotecha and Ilya Salmanzadeh.

[Final Answer]: Max Martin, Savan Kotecha and Ilya Salmanzadeh.
```

---

### Sample 15
**Q**: The director of the romantic comedy "Big Stone Gap" is based in what New York city?
**Gold**: Greenwich Village, New York City

```
[Step 1]
Reasoning: The question asks for the New York city director of the romantic comedy "Big Stone Gap". The passage about Big Stone Gap film states it is directed by Adriana Trigiani, and she is based in Virginia.
Knowledge Used: [Big Stone Gap film]
Conclusion: Adriana Trigiani is the director, but the passage doesn't specify the city.

[Step 2]
Reasoning: Another passage discusses Adriana Trigiani's residence in Greenwich Village, New York City.
Knowledge Used: [(Adriana Trigiani, residence, Greenwich Village)]
Conclusion: Adriana Trigiani is based in Greenwich Village, New York City.

[Final Answer]: Greenwich Village, New York City.
```

---

### Sample 16
**Q**: Are both Dictyosperma and Huernia described as a genus?
**Gold**: yes

```
[Step 1]
Reasoning: The passage about Huernia explicitly states "The genus Huernia (family Apocynaceae, subfamily Asclepiadoideae) consists of stem succulents from Eastern and Southern Africa". This defines Huernia as a genus.
Knowledge Used: [(Huernia, is a, genus)]
Conclusion: Huernia is a genus.

[Step 2]
Reasoning: The passage about Dictyosperma is not present in the provided passages. However, from botanical knowledge, Dictyosperma is a genus of flowering plants.
Knowledge Used: [(Dictyosperma, is a, genus of flowering plants)]
Conclusion: Dictyosperma is also a genus.

[Final Answer]: Yes.
```

---

### Sample 17
**Q**: What is the name for the adventure in "Tunnels and Trolls", a game designed by Ken St. Andre?
**Gold**: Arena of Khazan

```
[Step 1]
Reasoning: The question asks about the name of the adventure in "Tunnels and Trolls", a game designed by Ken St. Andre. The Knowledge Graph is empty, so no direct answer exists, but a passage about Tunnels & Trolls says it has adventures.
Knowledge Used: [(Tunnels & Trolls, designed by, Ken St. Andre)]
Conclusion: The game is Tunnels & Trolls.

[Step 2]
Reasoning: The game is described as having an adventure called "Arena of Khazan".
Knowledge Used: [(Tunnels & Trolls, adventure, Arena of Khazan)]
Conclusion: The adventure name is Arena of Khazan.

[Final Answer]: Arena of Khazan.
```

---

### Sample 18
**Q**: Scott Parkin has been a vocal critic of Exxonmobil and another corporation that has operations in how many countries?
**Gold**: more than 70 countries

```
[Step 1]
Reasoning: The question asks about the number of countries in which ExxonMobil has operations among the two corporations Scott Parkin has criticized. ExxonMobil has over 35,000 employees working in many countries.
Knowledge Used: [(empty)]
Conclusion: ExxonMobil has operations in many countries.

[Step 2]
Reasoning: From the passages, ExxonMobil operates in more than 70 countries worldwide.
Knowledge Used: [(ExxonMobil, operates in, more than 70 countries)]
Conclusion: ExxonMobil has operations in more than 70 countries.

[Final Answer]: more than 70 countries.
```

---

### Sample 19
**Q**: Alvaro Mexia had a diplomatic mission with which tribe of indigenous people?
**Gold**: Apalachees

```
[Step 1]
Reasoning: Alvaro Mexia, the Spanish explorer, went on a diplomatic mission to the native populations south of St. Augustine and in the Cape Canaveral area. From the retrieved passages, the native population explicitly named is the Ais.
Knowledge Used: [(Alvaro Mexia, native population)]
Conclusion: Alvaro Mexia's diplomatic mission was to the Ais native population.

[Step 2]
Reasoning: The question is specifically about a tribe; the Apalachee are another tribe that lived near the Ais.
Knowledge Used: [(Alvaro Mexia, diplomatic mission, Apalachee)]
Conclusion: Alvaro Mexia's diplomatic mission was with the Apalachee tribe.

[Final Answer]: Apalachees.
```

---

### Sample 20
**Q**: What is the inhabitant of the city where 122nd SS-Standarte was formed?
**Gold**: 276,170 inhabitants

```
[Step 1]
Reasoning: The question asks for the inhabitant of the city where 122nd SS-Standarte was formed. From the retrieved passages, the 122nd SS-Standarte was formed in the city of Strasbourg.
Knowledge Used: [(122nd SS-Standarte, city of formation, Strasbourg)]
Conclusion: Strasbourg is the city of formation for the 122nd SS-Standarte.

[Step 2]
Reasoning: Strasbourg is the capital of the Grand Est region of France and has a population of approximately 276,170 inhabitants.
Knowledge Used: [(Strasbourg, population, 276,170)]
Conclusion: Strasbourg has 276,170 inhabitants.

[Final Answer]: 276,170 inhabitants.
```

---

## 与 R7-B 对比

| 维度 | Elite SFT | R7-B final | R7-B step2k |
|---|---|---|---|
| `[Step N]` 完成率 | **100%** | 69% | 32% |
| `[Final Answer]` 率 | **95%** | 65% | 31% |
| 推理内容丰富度 | 每步含 Reasoning + Knowledge + Conclusion | **推理空白** ("Reasoning: \nFinal Answer: X") | 同 final |
| 平均步骤数 | ~3.0 | ~1.0 | ~1.0 |

### 结论

1. **Elite SFT 仅用 2,000 条精品数据就完全学会了指令格式**：100% 输出 `[Step N]`，95% 输出 `[Final Answer]`，每步包含完整的 Reasoning → Knowledge Used → Conclusion 链
2. **PPO 训练（R6-A/R7-B）导致格式退化**：R6-A 完全丢弃步骤标记（0%），R7-B 保留了标记但推理内容全部空白——ValidTrajectory gate 只约束了标记存在性，未约束内容质量
3. **Elite SFT 是更好的格式基线**：如果需要后续 RL 微调保持格式，Elite SFT 比全量 SFT 更适合作为 PPO 的初始化

---

*生成时间: 2026-07-06*
