# 03 - Parent-child chunking

> Week 4 · Topic 3 of 7 · Time: ~60 min · Runs offline (toy embedder), real Gemini embeddings when a key is set

## 🎯 What you will learn

- You will be able to explain the **small-vs-large chunk dilemma** and how parent-child retrieval resolves it.
- You will be able to build parents (sections) and children (sentences) with **links and deterministic ids**.
- You will be able to **search children but return parents**, with dedupe and a size budget.
- You will be able to implement **sentence-window retrieval** as a cheaper middle path.
- You will be able to measure whether the extra context actually lets the LLM answer.

## 🧠 The big idea in plain words

Small chunks are great for *finding* (one precise fact, one clean vector) but poor for *answering*: "It must be returned within 7 days" does not say *what* must be returned. Big chunks are great for answering but blurry for finding.

**Parent-child chunking** keeps both: you **index small children** (precise matching) but **return their larger parent** (enough context) to the LLM.

Laravel analogy: an Eloquent `hasMany`/`belongsTo`. You search the `comments` table (children) for a match, then use `$comment->post` to load the whole post (parent) for display.

## 📚 Concepts, step by step

### 1. The dilemma

| | Small chunk | Large chunk |
|---|---|---|
| Matching precision | high | low (diluted) |
| Context for the LLM | poor | good |
| Typical failure | "it", "this", "the form" with no antecedent | right passage retrieved but ranked low |

### 2. The structure

```
parent  expenses#p : "Employees claim travel costs using Form X-12. The form must be filed within 30 days ... "
 ├─ child c0 : "Employees claim travel costs using Form X-12."
 ├─ child c1 : "The form must be filed within 30 days of the trip."   <- matched by the question
 ├─ child c2 : "Late claims need director approval."
 └─ child c3 : "Receipts above 500 rupees must be attached."
```
Only children are embedded and searched. Each child stores `parent_id` (and its position). Parents are stored as plain text in a key-value store (a dict, Redis, a SQL table or a vector DB payload).

### 3. Query time

1. Embed question, search children (over-fetch, say `3 × k`).
2. Map each hit to its `parent_id`.
3. **Dedupe** parents; rank each parent by its **best child score**.
4. Respect a **token/character budget**; stop adding parents when full.
5. Send parents to the LLM; cite by parent/section.

### 4. What is a "parent"?

Choose by your document structure: a **section** (best), a page, a paragraph, or a fixed window of 1,500-3,000 characters. Children: a sentence, or 100-300 characters. Rule of thumb: child ≈ one idea, parent ≈ the smallest unit that makes sense on its own. Too-large parents re-create the noise problem; too-small ones do not add context.

### 5. Sentence-window retrieval (the cheap variant)

Index single sentences, then return the matched sentence **± n neighbours** (inside the same parent). No parent store is needed beyond the ordered sentences, and the context is smaller than a whole section (Q3: window ±1 gave 131 chars vs 171 for the parent). Choose `n` by measuring.

### 6. Costs and trade-offs

- **More vectors:** a section of 4 sentences becomes 4 vectors (versus 1). Storage and embedding cost grow with the number of children.
- **Bigger prompts:** parents are larger than children, so budget tokens (topic 05, Week 3).
- **Simpler ingestion than it sounds:** one extra dict and a `parent_id` field.
- Works with any vector DB: store `parent_id` in metadata; keep parents in a separate collection or table.

### 7. Variants

- **Multi-level (hierarchy):** document → section → paragraph → sentence; retrieve at the leaf, return a chosen ancestor.
- **Small-to-big:** the same idea under another name (LlamaIndex `AutoMergingRetriever` merges children into their parent when enough siblings are retrieved).
- **Summary parents:** index children, but return an LLM summary of the parent plus the matched child.

### 8. Evaluating

Count how often the returned context contains **both** the fact and what the fact is about (Q2). Also record average context size. Pick the smallest context that keeps answerability high.

## 💻 Code walkthrough

### Stage 1 - build the structure (Q1)

```python
for name, text in docs.items():
    pid = f"{name}#p"                       # deterministic parent id
    parents[pid] = text
    for i, sent in enumerate(sentences(text)):
        children.append(Child(f"{pid}.c{len(children)}", pid, i, sent))
```

### Stage 2 - search children, return parents (Q2)

```python
for c, score in child_hits(question, k * 3):     # over-fetch children
    if c.parent_id not in seen: out.append(parents[c.parent_id])   # collapse to unique parents
```

### Stage 3 - sentence window (Q3)

Find the matched sentence's index among its siblings and slice `[i - n, i + n]`, staying inside one parent.

### Stage 4 - rank parents by best child (Q4)

```python
best[c.parent_id] = max(best.get(c.parent_id, 0), score)
for pid in sorted(best, key=lambda p: -best[p]): add while used + len(parent) <= max_chars
```

### Stage 5 - idempotent ingestion (Q5)

Deterministic ids mean ingesting twice yields the same counts (4 parents, 16 children): updates overwrite instead of duplicating.

### With a real vector DB

Chroma/Qdrant: store children with `metadata={"parent_id": pid}`; after the query read the unique `parent_id`s and fetch parents from your own store. Delete by `source` metadata when a document changes.

## ⚠️ Common mistakes & debugging

| Symptom | Cause | Fix |
|---|---|---|
| Same section repeated in the prompt | One parent per child hit | Dedupe by `parent_id` |
| Wrong parent ranked first | Ranking by count or first hit | Rank by best child score |
| Prompt too long | Parents huge, many returned | Size budget; smaller parents |
| Orphan children after re-ingestion | Parent ids not deterministic | Stable ids; delete-then-insert by source |
| Answers still lack context | Child too small / parent too small | Larger parent (section) or window ±2 |
| Retrieval quality unchanged | Child ≈ parent size already | Make children noticeably smaller |
| Child from section A, parent of B | Misaligned `parent_id` mapping | Test the mapping; store ids not list positions |
| Cross-section windows | Neighbours taken across parent boundary | Restrict to siblings with same `parent_id` |

## 🚀 Pro-level insights (what the top 2% know)

- **Decouple "what you search" from "what you read".** This is the most useful sentence in RAG design: index for matching, return for answering.
- **Dedupe + budget are as important as the idea itself.** Without them parent-child inflates prompts.
- **Combine with reranking:** rerank the *child* hits (precise) before expanding to parents.
- **Combine with metadata filters:** filter on children, return parents; ACL stays enforced because both share the same tenant/ACL.
- **Choose parents from the document's structure,** not fixed lengths, whenever structure exists (topic 07, Week 3).
- **Measure answerability, not only hit@k:** a hit on a child that lacks context counts as a failure for the LLM.
- **Storage cost is small** relative to quality gains; children vectors dominate, parents are plain text.
- **Cache parent text** in memory/Redis (topic 05): they are read on every query.

## 🧪 Mini lab (10-15 min)

1. Take 3 pages of a real document; split into sections (parents) and sentences (children).
2. Write 6 questions whose answers need the surrounding context.
3. Compare child-only, window ±1, and parent on "answerable" and context size.
4. Add a size budget (800 characters) to the parent retrieval and see which questions fail.
5. Re-ingest twice and confirm the counts do not change.

## ✅ Recap checklist

- [ ] I can explain why small chunks find well but answer poorly.
- [ ] I store `parent_id` on every child and ids are deterministic.
- [ ] I over-fetch children, dedupe parents, rank by best child, apply a budget.
- [ ] I can build sentence-window retrieval.
- [ ] I measure answerability and context size.

## 🎤 Interview questions

**1. What is parent-child retrieval?**
Embed small child chunks for precise matching, but return their larger parent chunk (or section) to the LLM for context.

**2. Why dedupe parents?**
Several children of one parent usually match together; without deduping you send the same text multiple times.

**3. Parent-child vs sentence-window?**
Window returns neighbouring sentences (cheap, smaller context); parent-child returns a structural unit (section) and needs a parent store.

**4. How do you size parents and children?**
Child ≈ one idea (sentence or 100-300 characters); parent ≈ the smallest self-contained section; validate with answerability and context-size measurements.

**5. Downsides?**
More vectors to store and embed, larger prompts, extra mapping logic and re-ingestion care.

## 🔗 Next

**04 - RAG evaluation with RAGAs:** measure faithfulness, answer relevancy, context precision and recall so every change in this course (chunking, HyDE, parent-child) can be judged with numbers.
