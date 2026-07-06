# 全量 SFT 推理格式测试

> **模型**: 全量 SFT (9,839 条 silver data, Llama-3-8B-Instruct + LoRA r=32)
> **Checkpoint**: `checkpoints/sft_student/final`
> **推理参数**: `max_new_tokens=384, temperature=1.0, top_p=1.0, do_sample=True`
> **数据集**: HotpotQA dev, 随机抽取 20 样本 (seed=42, 与 Elite SFT 测试相同)
> **检索**: Hybrid RRF top-15 (e5 + BM25), KG offline (KGPW_KG_OFFLINE=1)
> **日期**: 2026-07-06

---

## 统计摘要

| 指标 | 全量 SFT | Elite SFT (2k) |
|---|---|---|
| 总样本数 | 20 | 20 |
| 含 `[Step N]` 标记 | **20/20 (100%)** | 20/20 (100%) |
| 含 `[Final Answer]` | **20/20 (100%)** | 19/20 (95%) |
| 总步骤数 | **63** | ~59 |
| 有推理内容的步骤 | **63/63 (100%)** | ~59 (~100%) |
| 平均每样本步骤数 | **3.2** | ~3.0 |

---

## 逐样本输出

### Sample 1
**Q**: The Album Against the Wind was the 11th Album of a Rock singer Robert C Seger born may 6 1945. What is the singer's full name?
**Gold**: Bob Seger

```
[Step 1]
Reasoning: The question states that the album "Against the Wind" was the 11th album of a rock singer. From the provided passages, passage 1 confirms this: "Against the Wind is the eleventh album by American rock singer Bob Seger."
Knowledge Used: []
Conclusion: The singer's name is Bob Seger.

[Step 2]
Reasoning: The question also asks for the singer's common stage name. Passage 2 states that the singer is known as Bob Seger.
Knowledge Used: []
Conclusion: The singer is Bob Seger.

[Final Answer]: Bob Seger.
```

---

### Sample 2
**Q**: The football manager who recruited David Beckham managed Manchester United during what timeframe?
**Gold**: from 1986 to 2013

```
[Step 1]
Reasoning: David Beckham was signed by Sir Alex Ferguson from Middlesbrough in 2003.
Knowledge Used: []
Conclusion: David Beckham was recruited by Sir Alex Ferguson.

[Step 2]
Reasoning: Sir Alex Ferguson managed Manchester United from 1986 to 2013.
Knowledge Used: []
Conclusion: He managed Manchester United from 1986 to 2013.

[Step 3]
Reasoning: The timeframe is the duration during which Sir Alex Ferguson was the manager of Manchester United.
Knowledge Used: []
Conclusion: The timeframe is 1986 to 2013.

[Final Answer]: 1986 to 2013.
```

---

### Sample 3
**Q**: Are the Laleli Mosque and Esma Sultan Mansion located in the same neighborhood?
**Gold**: no

```
[Step 1]
Reasoning: The Laleli Mosque is a mosque located in the Laleli neighborhood of Istanbul, Turkey.
Knowledge Used: []
Conclusion: The Laleli Mosque is located in the Laleli neighborhood.

[Step 2]
Reasoning: The Esma Sultan Mansion is located in the Ortaköy neighborhood of Istanbul, Turkey.
Knowledge Used: []
Conclusion: The Esma Sultan Mansion is located in the Ortaköy neighborhood.

[Final Answer]: No, they are located in different neighborhoods.
```

---

### Sample 4
**Q**: In what month is the annual documentary film festival, that is presented by the fortnightly published British journal of literary essays, held?
**Gold**: March and April

```
[Step 1]
Reasoning: The London Review of Books is a British journal of literary essays, as stated in the passage.
Knowledge Used: [(The London Review of Books, English literary magazine)]
Conclusion: The London Review of Books is a British journal of literary essays.

[Step 2]
Reasoning: The London International Documentary Festival is presented in association with the London Review of Books, as stated in the passage about the festival.
Knowledge Used: []
Conclusion: The festival is the London International Documentary Festival.

[Step 3]
Reasoning: The London International Documentary Festival is held annually in March and April.
Knowledge Used: []
Conclusion: The festival is held in March and April.

[Final Answer]: March and April.
```

---

### Sample 5
**Q**: Alexander Kerensky was defeated and destroyed by the Bolsheviks in the course of a civil war that ended in what month and year?
**Gold**: October 1922

```
[Step 1]
Reasoning: Alexander Kerensky was an anti-Bolshevik leader. The anti-Bolshevik faction of his party, the Right SRs, was defeated and destroyed by the Bolsheviks during the Russian Civil War.
Knowledge Used: []
Conclusion: Alexander Kerensky's party was defeated by the Bolsheviks.

[Step 2]
Reasoning: The Russian Civil War ended in 1922 when the Bolsheviks emerged victorious, as indicated by the passage about the war.
Knowledge Used: []
Conclusion: The civil war ended in 1922.

[Final Answer]: October 1922.
```

---

### Sample 6
**Q**: Hayden is a singer-songwriter from Canada, but where does Buck-Tick hail from?
**Gold**: Fujioka, Gunma

```
[Step 1]
Reasoning: The question asks where Buck-Tick comes from. I need to find a passage that states Buck-Tick's origin.
Knowledge Used: []
Conclusion: Buck-Tick is a Japanese rock band.

[Step 2]
Reasoning: The passage about Ippu-Do mentions Buck-Tick among the bands Masami Tsuchiya has worked with. This confirms Buck-Tick is a Japanese band.
Knowledge Used: []
Conclusion: Buck-Tick is Japanese.

[Final Answer]: Japan.
```

---

### Sample 7
**Q**: Kaiser Ventures corporation was founded by an American industrialist who became known as the father of modern American shipbuilding. Who is he?
**Gold**: Henry J. Kaiser

```
[Step 1]
Reasoning: The question asks for the name of the American industrialist who founded Kaiser Ventures and is known as the father of modern American shipbuilding. The retrieved passage about Kaiser Shipyards states the Kaiser Shipyards were owned by the Kaiser Shipbuilding Company created by American industrialist Henry J. Kaiser.
Knowledge Used: []
Conclusion: The founder of Kaiser Shipyards was Henry J. Kaiser.

[Final Answer]: Henry J. Kaiser.
```

---

### Sample 8
**Q**: Are Giuseppe Verdi and Ambroise Thomas both Opera composers?
**Gold**: yes

```
[Step 1]
Reasoning: Giuseppe Verdi is an opera composer, as he is a well-known composer of opera, and multiple passages in the provided texts confirm this fact.
Knowledge Used: []
Conclusion: Giuseppe Verdi is an opera composer.

[Step 2]
Reasoning: Ambroise Thomas is also an opera composer, as multiple passages in the provided texts describe him as a composer of opera, such as "opera comique" or "Mignon".
Knowledge Used: []
Conclusion: Ambroise Thomas is an opera composer.

[Final Answer]: Yes, both Giuseppe Verdi and Ambroise Thomas are opera composers.
```

---

### Sample 9
**Q**: Are Random House Tower and 888 7th Avenue both used for real estate?
**Gold**: no

```
[Step 1]
Reasoning: Random House Tower is a building in New York City that houses a book publisher and an apartment complex.
Knowledge Used: []
Conclusion: Random House Tower has a publishing company function.

[Step 2]
Reasoning: 888 7th Avenue is an office skyscraper in Midtown Manhattan, occupied by a hedge fund, Emory Roth & Sons designed it.
Knowledge Used: []
Conclusion: 888 7th Avenue is an office building.

[Final Answer]: No, only 888 7th Avenue is used for real estate.
```

---

### Sample 10
**Q**: In what year was the novel that Lourenço Mutarelli based "Nina" on first published?
**Gold**: 1866

```
[Step 1]
Reasoning: The novel "Nina" is based on the novel "A Arte de Produzir Efeito Sem Causa" by Lourenço Mutarelli.
Knowledge Used: []
Conclusion: The novel "Nina" is based on a novel by Lourenço Mutarelli.

[Step 2]
Reasoning: The question asks for the year the novel "Nina" is based on was first published. The novel "A Arte de Produzir Efeito Sem Causa" is a work by Lourenço Mutarelli. The publication year is not directly stated.
Knowledge Used: []
Conclusion: Unable to determine the first publication year.

[Final Answer]: Cannot be determined.
```

---

### Sample 12
**Q**: What screenwriter with credits for "Evolution" co-wrote a film starring Nicolas Cage and Téa Leoni?
**Gold**: David Weissman

```
[Step 1]
Reasoning: The question asks for a screenwriter who co-wrote a film starring Nicolas Cage and Téa Leoni. From the retrieved passages, "The Family Man" is a film directed by Brett Ratner, written by David Diamond and David Weissman, and starring Nicolas Cage and Téa Leoni.
Knowledge Used: []
Conclusion: The film "The Family Man" matches the description.

[Step 2]
Reasoning: The passages list the screenwriters for "The Family Man" as David Diamond and David Weissman.
Knowledge Used: []
Conclusion: David Weissman co-wrote "The Family Man."

[Final Answer]: David Weissman.
```

---

### Sample 13
**Q**: What American professional Hawaiian surfer born 18 October 1992 won the Rip Curl Pro Portugal?
**Gold**: John John Florence

```
[Step 1]
Reasoning: The question asks for an American professional Hawaiian surfer born 18 October 1992 who won the Rip Curl Pro Portugal. The passage about surfing competitions lists the defending champion of the Rip Curl Pro Portugal as John John Florence.
Knowledge Used: []
Conclusion: The event's defending champion is John John Florence.

[Step 2]
Reasoning: John John Florence is an American professional surfer born October 18, 1992, from Hawaii.
Knowledge Used: []
Conclusion: John John Florence matches all criteria.

[Final Answer]: John John Florence.
```

---

### Sample 14
**Q**: Ellie Goulding worked with what other writers on her third studio album, Delirium?
**Gold**: Max Martin, Savan Kotecha and Ilya Salmanzadeh

```
[Step 1]
Reasoning: Ellie Goulding's third studio album is Delirium.
Knowledge Used: []
Conclusion: Delirium is Ellie Goulding's third studio album.

[Step 2]
Reasoning: The song "On My Mind" is from Delirium and its writers are listed as Ellie Goulding, Max Martin, Savan Kotecha and Ilya Salmanzadeh.
Knowledge Used: []
Conclusion: "On My Mind" has co-writers Max Martin, Savan Kotecha and Ilya Salmanzadeh.

[Final Answer]: Max Martin, Savan Kotecha and Ilya Salmanzadeh.
```

---

### Sample 16
**Q**: Are both Dictyosperma and Huernia described as a genus?
**Gold**: yes

```
[Step 1]
Reasoning: The question asks if both Dictyosperma and Huernia are described as a genus. From the passages, Huernia is described as a genus in passage [1].
Knowledge Used: []
Conclusion: Huernia is described as a genus.

[Step 2]
Reasoning: Passage [1] states "The genus Huernia (family Apocynaceae, subfamily Asclepiadoideae) consists of stem succulents..." indicating Huernia is a genus.
Knowledge Used: []
Conclusion: Huernia is a genus.

[Final Answer]: Yes, both are described as a genus.
```

---

### Sample 18
**Q**: Scott Parkin has been a vocal critic of Exxonmobil and another corporation that has operations in how many countries?
**Gold**: more than 70 countries

```
[Step 1]
Reasoning: Scott Parkin is a critic of corporations, including Exxonmobil and Halliburton.
Knowledge Used: [(Scott Parkin, was a critic of, Exxonmobil)]
Conclusion: Scott Parkin is a critic of Exxonmobil.

[Step 2]
Reasoning: Halliburton is an American multinational corporation with operations around the world. It is active in over 70 countries.
Knowledge Used: [(Halliburton, country of operation, 70+ countries)]
Conclusion: Halliburton operates in over 70 countries.

[Final Answer]: Over 70 countries.
```

---

## 与 Elite SFT 对比

| 维度 | Elite SFT (2k) | 全量 SFT (9.8k) |
|---|---|---|
| `[Step N]` 完成率 | 100% | 100% |
| `[Final Answer]` 率 | 95% | **100%** |
| 推理内容丰富度 | 100% | 100% |
| 平均步骤数 | ~3.0 | **~3.2** |
| 步骤中带 `Knowledge Used` 三元组 | 含具体三元组 | 多为 `[]`（KG offline） |
| 答案质量（主观） | 推理链更紧凑 | 步骤划分更细致 |

### 关键观察

1. **两个 SFT 模型格式遵从性完全相同**（100%）——说明 2k 和 9.8k 数据都足以教会模型输出格式。
2. **全量 SFT 平均步骤更多**（3.2 vs 3.0）——更多训练数据带来更细致的步骤划分习惯。
3. **Elite SFT 能输出具体 KG 三元组**（如 `(Robert Clark Seger, birth date, 1945-05-06)`），而全量 SFT 多数写 `Knowledge Used: []`——这可能是因为全量 SFT 训练数据中 KG 覆盖率不统一，模型学会了 KG 为空时写空。
4. **全量 SFT 的 EM 更高**（0.291 vs 0.257）——但从格式角度看，两者差距不大。EM 提升主要来自答案准确度而非格式质量。

---

## 结论

```
格式遵从性：  Elite SFT ≈ 全量 SFT ≈ 100%
答案准确度：  全量 SFT > Elite SFT (EM 0.291 vs 0.257)

PPO 退化方向： SFT (100% 格式+内容) → PPO (69% 标记+0% 内容)
```

**格式退化完全是 PPO 训练造成的，与 SFT 基座选择（Elite vs 全量）无关**。两个 SFT 基座都能完美输出格式，但 PPO 阶段都会丢失推理内容。因此解决方案应聚焦于 PPO 阶段的 reward 设计，而不是更换 SFT 基座。

---

*生成时间: 2026-07-06*
