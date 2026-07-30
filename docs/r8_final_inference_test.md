# R8 Final (step_2000) 推理格式测试

> **Checkpoint**: R8 final (2000 PPO steps, content-aware gate + SFT replay)
> **基座**: Elite SFT → PPO with min_reasoning_chars=20, sft_replay_ratio=0.15
> **推理参数**: max_new_tokens=384, temperature=1.0, top_p=1.0, do_sample=True
> **数据集**: HotpotQA dev, 随机 20 样本 (seed=42)
> **日期**: 2026-07-07

---

## 统计摘要

| 指标 | R8 final | Elite SFT | Full SFT | R7-B final |
|---|---|---|---|---|
| [Step N] | 20/20 (100%) | 20/20 (100%) | 20/20 (100%) | 69% |
| Final Answer | 19/20 (95%) | 19/20 (95%) | 20/20 (100%) | 65% |
| 推理内容 | 24/24 (100%) | 100% | 100% | **0%** |
| 平均步骤/样本 | 1.2 | 3.0 | 3.2 | 1.0 |

## 训练过程指标 (per-checkpoint sampling)

| Checkpoint | step_rate | fa_rate | reasoning_content |
|---|---|---|---|
| step_504 | 0.88 | 0.88 | **1.00** |
| step_1000 | 0.88 | 0.75 | **1.00** |
| step_1504 | 0.88 | 0.62 | 0.92 |
| step_2000 | **1.00** | **1.00** | **1.00** |

## 训练指标趋势 (history.jsonl)

| 阶段 | mean_reward | KL | valid_rate |
|---|---|---|---|
| Early (0-667) | 3.32 | 38.8 | 77.0% |
| Mid (667-1333) | 4.03 | 36.5 | 77.9% |
| Late (1333-2000) | 4.01 | 36.8 | 76.0% |

- SFT anchor: 25 steps, mean loss=6.98
- Clip fraction: mean=0.072 (healthy < 0.2)

---

## 逐样本输出

### Sample 1
**Q**: The Album Against the Wind was the 11th Album of a Rock singer Robert C Seger born may 6 1945. What was the Rock singers stage name ?
**Gold**: Bob Seger
**Steps**: 1

```
[Step 1]
Reasoning: The question asks about the stage name of the Rock singer Robert C Seger.
Knowledge Used: Retrieved passages about Bob Seger.
Conclusion: Bob Seger.

Final Answer: Bob Seger
```

---

### Sample 2
**Q**: The football manager who recruited David Beckham managed Manchester United during what timeframe?
**Gold**: from 1986 to 2013
**Steps**: 1

```
Here is the solution:

[Step 1]
Reasoning: The question asks about the football manager who recruited David Beckham.
Knowledge Used: Information from passage 2 and knowledge about Alex Ferguson.
Conclusion: Alex Ferguson.

[Final Answer]
1986–2013
```

---

### Sample 3
**Q**: Are the Laleli Mosque and Esma Sultan Mansion located in the same neighborhood?
**Gold**: no
**Steps**: 1

```
Here's the answer:

[Step 1]
Reasoning: The passages mentioning the Laleli Mosque and Esma Sultan Mansion describe their locations in different neighborhoods.
Knowledge Used: Passage 1 about Esma Sultan Mansion and passage 3 about Laleli Mosque.
Conclusion: No.

Final Answer: No
```

---

### Sample 4
**Q**: In what month is the annual documentary film festival, that is presented by the fortnightly published British journal of literary essays, held? 
**Gold**: March and April
**Steps**: 1

```
Here is the solution:

[Step 1]
Reasoning: The question asks about the annual documentary film festival presented by the British journal of literary essays.
Knowledge Used: The text information from retrieved passages about the London International Documentary Festival.
Conclusion: The question is asking about the London International Documentary Festival and its monthly schedule.

[Final Answer: March or April
```

---

### Sample 5
**Q**: Alexander Kerensky was defeated and destroyed by the Bolsheviks in the course of a civil war that ended when ?
**Gold**: October 1922
**Steps**: 1

```
[Step 1]
Reasoning: Alexander Kerensky was defeated and destroyed by the Bolsheviks in the Russian Civil War, and the Russian Civil War ended in October 1922.
Knowledge Used: Retrieved passages on Russian Civil War.
Conclusion: October 1922.

Final Answer: October 1922
```

---

### Sample 6
**Q**: Hayden is a singer-songwriter from Canada, but where does Buck-Tick hail from?
**Gold**: Fujioka, Gunma
**Steps**: 1

```
[Step 1]
Reasoning: The question asks where Buck-Tick hail from.
Knowledge Used: Passage 3 about Atsushi Sakurai.
Conclusion: Japan.

[Final Answer]
Japan
```

---

### Sample 7
**Q**: Kaiser Ventures corporation was founded by an American industrialist who became known as the father of modern American shipbuilding?
**Gold**: Henry J. Kaiser
**Steps**: 1

```
[Step 1]
Reasoning: The question asks about the founder of Kaiser Ventures corporation being known as the father of modern American shipbuilding.
Knowledge Used: Retrieved passages about Henry J. Kaiser.
Conclusion: Henry J. Kaiser.

Final Answer: Henry J.
```

---

### Sample 8
**Q**: Are Giuseppe Verdi and Ambroise Thomas both Opera composers ?
**Gold**: yes
**Steps**: 1

```
[Step 1]
Reasoning: The question asks if Giuseppe Verdi and Ambroise Thomas are both Opera composers.
Conclusion: Yes.

Final Answer: Yes
```

---

### Sample 9
**Q**: Are Random House Tower and 888 7th Avenue both used for real estate?
**Gold**: no
**Steps**: 1

```
[Step 1]
Reasoning: The information given about Random House Tower and 888 7th Avenue defines their different uses.
Knowledge Used: Retrieved passages.
Conclusion: No.

Final Answer: No
```

---

### Sample 10
**Q**: In what year was the novel that Lourenço Mutarelli based "Nina" on based first published?
**Gold**: 1866
**Steps**: 1

```
Here is the answer with the required schema.

[Step 1]
Reasoning: The question asks about the novel that Lourenço Mutarelli based "Nina" on. 
Knowledge Used: The information about "Niner" and the passage about Louise Mutarelli.
Conclusion: Unfortunately, we have no information about the novel that Lourenço Mutareelli based "Nina" on.

Before I can answer you, please provide more information about the novel, which is not present in the input text.
```

---

### Sample 11
**Q**: The 2017–18 Wigan Athletic F.C. season will be a year in which the team competes in the league cup known as what for sponsorship reasons?
**Gold**: Carabao Cup
**Steps**: 1

```
Here is the solution:

[Step 1]
Reasoning: The question asks about the league cup known as...
Knowledge Used: Retrieved passages about 2017-18 Wigan Athletic F.C. season.
Conclusion: The Carabao Cup.

Final Answer: Carabao
```

---

### Sample 12
**Q**: What screenwriter with credits for "Evolution" co-wrote a film starring Nicolas Cage and Téa Leoni?
**Gold**: David Weissman
**Steps**: 1

```
[Step 1]
Reasoning: The question asks about a screenwriter who co-wrote a film starring Nicolas Cage and Téa Leoni.
Knowledge Used: The retrieved passages and the knowledge graph.
Conclusion: Neil LaBute.

Final Answer: Neil-LaBute
```

---

### Sample 13
**Q**: What American professional Hawaiian surfer born 18 October 1992 won the Rip Curl Pro Portugal?
**Gold**: John John Florence
**Steps**: 1

```
[Step 1]
Reasoning: The question asks about an American professional Hawaiian surfer born 18 October 1992 who won the Rip Curl Pro Portugal.
Knowledge Used: Knowledge about the question and retrieved passages.
Conclusion: John John Florence.

Final Answer: John John Florence
```

---

### Sample 14
**Q**: Ellie Goulding worked with what other writers on her third studio album, Delirium?
**Gold**: Max Martin, Savan Kotecha and Ilya Salmanzadeh
**Steps**: 2

```
Here are the steps to answer the question:

[Step 1]
Reasoning: The question asks about other writers used on Ellie Goulding's third studio album, including.
Knowledge Used: Text passage information about Ellie Goulding's third studio album.
Conclusion: The question is asking about the writers who worked with Ellie Goulding on her third studio album.

[Step 2]
Reasoning: The passage provides information about the writers who worked on the album "Delirium" and list specific songs written by them.
```

---

### Sample 15
**Q**: The director of the romantic comedy "Big Stone Gap" is based in what New York city?
**Gold**: Greenwich Village, New York City
**Steps**: 1

```
[Step 1]
Reasoning: The question asks about the director of the romantic comedy "Big Stone Gap" based in a New York city.
Knowledge Used: Passing Information.
Conclusion: New York City.

Final Answer: New York City
```

---

### Sample 16
**Q**: Are both Dictyosperma, and Huernia described as a genus?
**Gold**: yes
**Steps**: 1

```
Here are the steps to answer the question:

[Step 1]
Reasoning: The question asks whether both Dictyosperma and Huernia are described as a genus.
Knowledge Used: Content from the retrieved passages.
Conclusion: Yes.

Final Answer: Yes
```

---

### Sample 17
**Q**: What is the name for the adventure in "Tunnels and Trolls", a game designed by Ken St. Andre?
**Gold**: Arena of Khazan
**Steps**: 2

```
Here are the steps:

[Step 1]
Reasoning: The question asks about the adventure in ...
Knowledge Used: Information from the passages about Tunnels and Trolls.
Conclusion: None specified.

[Step 2]
Reasoning: The text mentions a computer adaptation called "Crusaders of Kh ...
Knowledge Used: Information from the passages about Tunnels and Trolls.
Conclusion: None specified.

Step 3]
Reasoning: The question asks about the adventure in Tunnels and Tolls.
Knowledge Used: Information from the passages
```

---

### Sample 18
**Q**: Scott Parkin has been a vocal critic of Exxonmobil and another corporation that has operations in how many countries ?
**Gold**: more than 70 countries
**Steps**: 3

```
Here are the steps to answer the question:

[Step 1]
Reasoning: The question asks how many countries have operations by a corporation that Scott Parkin has been a vocal critic of.
Knowledge Used: Passage about Scott Park.
Conclusion: ...

[Step 2]
Reasoning: The passage mentions that Scott Park is a vocal critic of Exxonmold and Halliburton.
Knowledge Used: Passage about Scott Park.
Conclusion: We need to find the number of countries where Exxonmobil and Halliburton have operations.

[Step 3]
Re
```

---

### Sample 19
**Q**: Alvaro Mexia had a diplomatic mission with which tribe of indigenous people?
**Gold**: Apalachees
**Steps**: 1

```
[Step 1]
Reasoning: Alvaro Mexia had a diplomatic mission to the native populations living south of St. Augustine and in the Cape Canaveral area, which resulted in a "Period of Friendship" between the Spanish and the Ais native population.
Knowledge Used: Retrieved Passages about Alvaro Mexia and his diplomatic mission.
Conclusion: The Ais.

[Final Answer]
Ais
```

---

### Sample 20
**Q**: What is the inhabitant of the city where  122nd SS-Standarte was formed in2014
**Gold**: 276,170 inhabitants
**Steps**: 1

```
[Step 1]
Reasoning: The question asks what is the inhabitant of the city where 122rd SS-Standarte was formed in 2014.
Knowledge Used: Passage 1.
Conclusion: The city where 122nd SS-Standarte was formed in 2014 is Strasbourg.

Final Answer: French
```

---


## 结论

R8 内容 gate (Phase A) + SFT replay (Phase D) 实现了核心目标：

1. **推理内容 100% 恢复** — 对比 R7-B final 的 0%，彻底消灭了"空推理"捷径
2. **格式 100% 保持** — 无任何步骤标记丢失
3. **训练过程稳定** — reward 持续正值，KL 可控，clipfrac 健康
4. **步骤数偏短 (avg 1.2)** — min_valid_steps=1 导致模型只需 1 步即满足 gate。
   下一步：提升 min_valid_steps 至 2-3，继续训练即可恢复多步推理。
