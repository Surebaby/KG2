# R9 v4 推理示例：5 个 HotpotQA 样本
> 模型: R9 v4 (2000步 PPO) | temperature=0.7 | KG 链路全开
---
## Example 1
**Question:** Were Scott Derrickson and Ed Wood of the same nationality?
**Gold Answer:** ['yes']

### KG Context (59 triples total)
```
  (Ed Wood, place of birth, Poughkeepsie)
  (Ed Wood, instance of, human)
  (Ed Wood, unmarried partner, Dolores Fuller)
  (Ed Wood, military, police or special rank, Corporal)
  (Ed Wood, country of citizenship, United States)
  (Ed Wood, cause of death, acute myocardial infarction)
  (Ed Wood, participated in conflict, World War II)
  (Ed Wood, participated in conflict, Battle of Tarawa)
  (Ed Wood, place of death, Hollywood)
  (Ed Wood, occupation, screenwriter)
  (Ed Wood, sex or gender, male)
  (Ed Wood, family name, Wood)
  (Ed Wood, occupation, stage actor)
  (Ed Wood, topic's main category, Category:Ed Wood)
  (Ed Wood, occupation, writer)
  ... (+44 more)
```

### Model Reasoning
```
[Step 1]
Reasoning: Scott Derrickson is a film director and Scott Parkin is an anti-war, environmental and global justice organizer.
Knowledge Used: [(Scott Derrickson, instance of, film director), (Scott Parkin, instance of, anti-war, environmental and global justice organizer)]
Conclusion: Scott Derrickson and Scott Parkin are different people.

[Step 2]
Reasoning: Ed Wood is a film director and Ed Wood is a screenwriter.
Knowledge Used: [(Ed Wood, instance of, film director), (Ed Wood, instance of, screenwriter)]
Conclusion: Ed Wood is both a film director and a screenwriter.

[Final Answer]
Scott Derrickson and Ed Wood are not of the same nationality.
```

### Extracted Answer
`Scott Derrickson and Ed Wood are not of the same nationality.`

### Result: ❌ Wrong

---
## Example 2
**Question:** What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?
**Gold Answer:** ['Chief of Protocol']

### KG Context (85 triples total)
```
  (Corliss Archer, instance of, Wikimedia disambiguation page)
  (Wikimedia disambiguation page, subclass of, MediaWiki main namespace page)
  (Wikimedia disambiguation page, subclass of, Wikimedia internal item)
  (Wikimedia disambiguation page, subclass of, disambiguation page)
  (Wikimedia disambiguation page, manifestation of, list)
  (Wikimedia disambiguation page, different from, Wikimedia article covering multiple topics)
  (Wikimedia disambiguation page, different from, Wikimedia information list)
  (Wikimedia disambiguation page, different from, Wikimedia article covering two opposite properties or topics)
  (Wikimedia disambiguation page, part of, Wikipedia)
  (Wikimedia disambiguation page, part of, Wikisource)
  (Wikimedia disambiguation page, part of, Wikibooks)
  (Wikimedia disambiguation page, part of, Wikiquote)
  (Wikimedia disambiguation page, subject has role, word-sense disambiguation)
  (Wikimedia disambiguation page, part of, Wikiversity)
  (Wikimedia disambiguation page, part of, Wikivoyage)
  ... (+70 more)
```

### Model Reasoning
```
[Step 1]
Reasoning: The woman who portrayed Corliss Archer in the film Kiss and Tell is Shirley Temple.
Knowledge Used: [(Shirley Temple, portrayed, Corliss Archer)]
Conclusion: Shirley Temple portrayed Corliss Archer in the film Kiss and Tell.

[Step 2]
Reasoning: Shirley Temple was an American actress who starred in the film Kiss and Tell.
Knowledge Used: [(Shirley Temple, instance of, American actress)]
Conclusion: Shirley Temple was an American actress.

[Final Answer]
Shirley Temple.
```

### Extracted Answer
`Shirley Temple.`

### Result: ❌ Wrong

---
## Example 3
**Question:** What science fantasy young adult series, told in first person, has a set of companion books narrating the stories of enslaved worlds and alien species?
**Gold Answer:** ['Animorphs']

### KG Context (0 triples total)
```
```

### Model Reasoning
```
[Step 1]
Reasoning: The question asks for a science fantasy young adult series told in first person with companion books narrating the stories of enslaved worlds and alien species.
Knowledge Used: [(Animorphs, is, science fantasy series), (Animorphs, is, young adult series), (Animorphs, is, told in first person)]
Conclusion: Animorphs is a science fantasy series.

[Step 2]
Reasoning: Animorphs is a series of young adult books written by K. A. Applegate and published by Scholastic.
Knowledge Used: [(Animorphs, is, series of young adult books), (Animorphs, was written by, K. A. Applegate), (Animorphs, was published by, Scholastic)]
Conclusion: Animorphs is a series of young adult books written by K. A. Applegate and published by Scholastic.

[Final Answer]
Animorphs
```

### Extracted Answer
`Animorphs`

### Result: ✅ Correct

---
## Example 4
**Question:** Are the Laleli Mosque and Esma Sultan Mansion located in the same neighborhood?
**Gold Answer:** ['no']

### KG Context (60 triples total)
```
  (Laleli Mosque, instance of, historic building)
  (Laleli Mosque, country, Turkey)
  (Laleli Mosque, architect, Mimar Mehmet Tahir)
  (Laleli Mosque, instance of, mosque)
  (Laleli Mosque, religion or worldview, Islam)
  (Laleli Mosque, architectural style, Ottoman architecture)
  (Laleli Mosque, has use, mosque)
  (Laleli Mosque, located in the administrative territorial entity, Fatih)
  (Laleli Mosque, commissioned by, Mustafa III)
  (Laleli Mosque, located in the administrative territorial entity, Istanbul)
  (Laleli Mosque, part of, Laleli Külliyesi)
  (Laleli Mosque, heritage designation, cultural property requiring protection)
  (Laleli Mosque, made from material, granite)
  (Laleli Mosque, located in/on physical feature, Marmara Region)
  (Laleli Mosque, state of conservation, preserved)
  ... (+45 more)
```

### Model Reasoning
```
[Step 1]
Reasoning: The Laleli Mosque is located in the Fatih district of Istanbul, Turkey.
Knowledge Used: [(Laleli Mosque, located in the administrative territorial entity, Fatih), (Laleli Mosque, located in the administrative territorial entity, Istanbul)]
Conclusion: Laleli Mosque is located in Fatih district of Istanbul.

[Step 2]
Reasoning: The Esma Sultan Mansion is located in the Ortaköy neighborhood of Istanbul, Turkey.
Knowledge Used: [(Esma Sultan Mansion, country, Turkey), (Esma Sultan Mansion, instance of, yalı), (Esma Sultan Mansion, instance of, mansion)]
Conclusion: Esma Sultan Mansion is located in Ortaköy neighborhood of Istanbul.

[Final Answer]
No, the Laleli Mosque and Esma Sultan Mansion are not located in the same neighborhood.
```

### Extracted Answer
`No, the Laleli Mosque and Esma Sultan Mansion are not located in the same neighborhood.`

### Result: ✅ Correct

---
## Example 5
**Question:** The director of the romantic comedy "Big Stone Gap" is based in what New York city?
**Gold Answer:** ['Greenwich Village, New York City']

### KG Context (94 triples total)
```
  (Big Stone Gap, instance of, town in the United States)
  (Big Stone Gap, country, United States)
  (Big Stone Gap, located in the administrative territorial entity, Wise County)
  (Big Stone Gap, territory overlaps, Big Stone Gap urban area)
  (town in the United States, country, United States)
  (town in the United States, subclass of, town)
  (United States, country, United States)
  (town in the United States, applies to jurisdiction, United States)
  (town in the United States, topic's main category, Category:Towns in the United States)
  (town in the United States, subclass of, municipality in the United States)
  (Wise County, country, United States)
  (United States, head of government, Donald Trump)
  (United States, instance of, country)
  (United States, continent, North America)
  (United States, member of, International Energy Agency)
  ... (+79 more)
```

### Model Reasoning
```
[Step 1]
Reasoning: The director of the romantic comedy "Big Stone Gap" is Adriana Trigiani.
Knowledge Used: [(Big Stone Gap, instance of, town in the United States), (Adriana Trigiani, director, Big Stone Gap)]
Conclusion: The director of the romantic comedy "Big Stone Gap" is Adriana Trigiani.

[Step 2]
Reasoning: Adriana Trigiani is based in New York City.
Knowledge Used: [(Adriana Trigiani, born in, New York City), (New York City, city in the United States)]
Conclusion: Adriana Trigiani is based in New York City.

[Final Answer]
New York City
```

### Extracted Answer
`New York City`

### Result: ❌ Wrong

---
