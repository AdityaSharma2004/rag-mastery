# 07 - Handling long documents

> Week 3 · Topic 7 of 7 · Time: ~90 min · Pure Python, runs offline (toy summariser/embedder; real Gemini used when a key is set)

## 🎯 What you will learn

- You will be able to decide between **stuffing, long-context, retrieval and map-reduce** for a given document and question.
- You will be able to split structured documents by **headings with breadcrumb paths**.
- You will be able to build a **map-reduce (tree) summary** and compute its cost.
- You will be able to implement **hierarchical (two-level) retrieval**.
- You will be able to avoid the "silent truncation" bug and build a **strategy router with a cost model**.

## 🧠 The big idea in plain words

So far a document was a few pages. Real life brings a 300-page contract, a 2,000-page manual, a 10,000-document archive. Three things break:

1. It does not fit in the prompt (or fits but is slow, expensive and noisy).
2. Plain chunk search finds *similar-looking* passages, not necessarily the right *section*.
3. Questions come in two flavours: **lookups** ("what is the notice period?") and **big-picture** ("summarise the risks"). They need different tools.

The cure is to give the system *structure and levels*, like a book with a table of contents, chapter summaries and pages. You do not read 500 pages to find one clause: you use the index, jump to the chapter, read the page.

Laravel/MySQL analogy: a **composite index and pagination**. You do not `SELECT *` from a 10 million-row table; you use an index to narrow to a partition, then fetch a page. Hierarchical retrieval is that for text.

## 📚 Concepts, step by step

### 1. Four strategies

| Strategy | What it does | Good for | Cost / risk |
|---|---|---|---|
| **Stuff** | put the whole document in the prompt | tiny docs (a few thousand tokens) | cheap if small |
| **Long context** | use a huge window (Gemini 2.5 Flash advertises ~1M tokens) | one big doc, one-off questions, whole-doc reasoning | slow, costly per query, quality can drop in the middle, free-tier limits |
| **Retrieve (RAG)** | embed chunks, fetch top-k | fact lookup in large corpora | misses answers that need many parts |
| **Map-reduce / hierarchical** | summarise pieces, then summarise summaries | "summarise", "main themes" | many LLM calls, details lost |

Rule of thumb: **lookup → retrieve; global question → map-reduce or long context; tiny → stuff.** Q5 turns this into code.

### 2. Long context is not a free pass

Even with a million-token window:
- You pay (time and quota) for every token **on every question**.
- Accuracy tends to **drop for facts buried in the middle** of huge prompts.
- You cannot cite precisely, filter by permissions, or update one paragraph cheaply.
- It is great for a *single* document analysed a few times; RAG wins when many questions hit a large, changing corpus.

Often the best answer is **both**: retrieve the right sections, then give the model *whole sections* with generous context.

### 3. Structure-aware splitting with breadcrumbs

Use the document's own structure (Markdown `#`, DOCX heading styles, HTML `h1-h3`, PDF bookmarks/table of contents). Keep a **path** for each piece:

```
Acme Employee Handbook > Leave > Sick leave
```

Embed `path + ": " + text`. The breadcrumb tells the vector what the passage is *about* even if the sentence says "it expires after 3 days". Implementation: keep a stack of `(level, title)`; when a heading of level L arrives, pop everything ≥ L, then push (Q2).

### 4. Map-reduce summarisation (a tree)

```
40 chunks --map--> 40 summaries --reduce (groups of 5)--> 8 --> 2 --> 1 final summary
```
Calls = n_map + n/g + n/g² + ... With 40 chunks and groups of 5: 40 + 8 + 2 + 1 = **51 calls**, depth 3. Bigger groups mean fewer calls but longer reduce prompts. Group size must be ≥ 2 or the tree never shrinks (Q1 raises an error). Weakness: details are lost at each level; ask the map step to keep **numbers, names and dates**.

Variants: **refine** (carry a running summary through the chunks in order; sequential, cannot parallelise) and **parallel map-reduce** (fast, use threads with rate-limit care on the free tier).

### 5. Hierarchical (two-level) retrieval

Index two things: a short **summary of each section** and the **chunks** inside it.

```
question -> compare with section summaries -> top 2 sections -> compare with chunks only inside them -> top k chunks
```
At scale (100 sections × 100 chunks): flat search scores 10,000 vectors; two-level scores 100 + 2×100 = 300. Benefits: fewer distractors from unrelated sections and lower cost. Risks: if a summary omits the detail, stage 1 picks the wrong sections; mitigate with `top_sections ≥ 2-3`, summaries that keep key facts, and a flat-search fallback. This idea is the heart of **RAPTOR**-style trees and of "document → section → chunk" retrieval.

### 6. The truncation trap

```python
context = doc[:limit]      # looks harmless
```
Everything after the limit silently disappears, so questions about the end of the document are unanswerable and the model may guess. Always choose context by **relevance under a budget**, not by position (Q4).

### 7. Routing by size and question type

```python
if doc_tokens <= 8_000:           "stuff"
elif question is broad:           "long_context" if it fits comfortably else "map_reduce"
else:                             "retrieve"
```
Classify "broad" with keywords (summarise, overview, main points) or one cheap LLM call. Then use a cost model (LLM calls, prompt tokens) to choose; Q5 shows a 500-page set needing ~1,900 calls for map-reduce versus 1 call (≈2k tokens) for a lookup.

### 8. Other long-document tools

- **Page/section expansion:** retrieve a small chunk, then return its whole section (parent-child, Week 4 topic 03).
- **Table of contents routing:** let the LLM read the TOC and choose which sections to open.
- **Multi-hop / agentic reading:** the model issues several searches and reads incrementally.
- **Context compression:** extract only sentences relevant to the question before stuffing.
- **Caching:** cache section summaries and embeddings by content hash, as summarising is the expensive part.

## 💻 Code walkthrough

### Stage 1 - tree reduce

```python
def tree_reduce(items, combine, group_size=4):
    if group_size < 2: raise ValueError("group_size must be >= 2 or the tree never shrinks")
    calls = levels = 0
    while len(items) > 1:
        items = [combine(items[i:i + group_size]) for i in range(0, len(items), group_size)]
        calls += len(items); levels += 1
    return items[0], calls, levels
```
`combine` is any function from a list to one item: with Gemini it is `lambda g: generate("Merge these summaries...\n" + "\n".join(g))`.

### Stage 2 - markdown to breadcrumb sections

```python
while stack and stack[-1][0] >= level: stack.pop()     # leaving siblings / deeper sections
stack.append((level, title))
path = " > ".join(t for _, t in stack)
```

### Stage 3 - two-level search

Stage 1 ranks section summaries; stage 2 ranks chunks inside the top sections only. `scored` counts show the saving.

### Stage 4 - budgeted context

```python
for i in np.argsort(-scores):                 # most relevant first
    if used + tokens(secs[i]["text"]) + 8 <= limit: chosen.append(i)
context = "\n".join(... for i in sorted(chosen))   # restore document order for readability
```

### Stage 5 - the router

`route()` returns a strategy name; `cost()` estimates `llm_calls` and `prompt_tokens`. Plug the strategies into your Week 3 pipeline.

## ⚠️ Common mistakes & debugging

| Symptom | Cause | Fix |
|---|---|---|
| Cannot answer questions about the end of a doc | Position-based truncation | Select context by relevance within a token budget |
| "Summarise" answered from 3 random chunks | Using top-k RAG for a global question | Map-reduce or long-context for broad questions |
| Map-reduce summary lacks numbers | Summaries too abstract | Instruct "keep numbers, names, dates"; refine pass |
| Infinite reduction loop | `group_size = 1` | Require ≥ 2 |
| Wrong section chosen in hierarchical retrieval | Summaries omit details | Richer summaries, `top_sections` 2-3, flat fallback |
| Snippets lack context ("it expires after 3 days") | No breadcrumb/heading | Embed `path: text` |
| Cost explodes on summarisation | Re-summarising unchanged sections | Cache by content hash |
| Free-tier 429s during map step | Parallel calls without limits | Batch, throttle, retry with backoff |
| Long-context answers miss facts mid-document | "Lost in the middle" | Retrieve + put relevant sections at edges |

## 🚀 Pro-level insights (what the top 2% know)

- **Decide with numbers.** Estimate tokens per question for each strategy; if a long-context call costs 100x a retrieval call and you ask 1,000 questions, the router pays for itself in a day.
- **Section summaries are an index, not a deliverable.** Generate them at ingestion, store them with `level=section` metadata, and refresh only changed sections.
- **Two retrievals beat one big one** for manuals and contracts: find the section, then answer from the whole section (parent-child).
- **Evaluate global questions separately.** Lookup metrics (hit@k) do not measure summary quality; use checklist-style rubrics ("mentions the 3 key risks") with an LLM judge.
- **Parallelism:** the map step is embarrassingly parallel, but respect rate limits; a small thread pool plus exponential backoff is enough.
- **Keep provenance in summaries:** include section paths so the final summary can cite where each point came from.
- **Streaming UX:** for long jobs show progress ("summarised 18/40 sections") and cache partial results so failures resume.

## 🧪 Mini lab (10-15 min)

1. Take a real 20+ page document (PDF → Markdown with headings).
2. Run `split_markdown` and print the first 10 breadcrumb paths. Fix any heading-level surprises.
3. Ask 5 lookup questions with flat search and with two-level search; count how often the gold section is found.
4. Compute the map-reduce call count for group sizes 3, 5, 8 and pick one.
5. Use `route()` on three questions and note which strategy each gets and why.

## ✅ Recap checklist

- [ ] I can name the four strategies and when each fits.
- [ ] I split by headings and keep breadcrumb paths.
- [ ] I can compute the number of LLM calls in a map-reduce tree.
- [ ] I can build two-level retrieval and name its failure mode.
- [ ] I never truncate by position; I select by relevance under a budget.
- [ ] I route by document size and question type.

## 🎤 Interview questions

**1. Gemini has a ~1M-token window. Why still use RAG?**
Cost and latency per question, quality loss for facts in the middle, no fine-grained citations/permissions, and corpora larger than any window. Use long context for single-document, low-volume analysis.

**2. How do you summarise a 500-page document?**
Map-reduce: summarise chunks (map), then summarise groups of summaries repeatedly (reduce) until one remains; keep numbers and names; cache and parallelise within rate limits.

**3. What is hierarchical retrieval?**
Retrieve the relevant sections via section summaries first, then search chunks only inside them; it reduces noise and compute, but can miss answers if summaries omit details.

**4. Why prepend heading breadcrumbs to chunks?**
They give each chunk its topical context so vectors and the LLM know what an otherwise ambiguous sentence refers to.

**5. How would you choose between RAG and map-reduce for a question?**
Lookup questions need a few precise passages (RAG); global questions need coverage of the whole document (map-reduce or long context). A router classifies the question and uses a cost model.

## 🔗 Next

**Week 4 · 01 - Query expansion:** improve retrieval by rewriting the user's question into several better queries (multi-query, step-back, decomposition) before searching.
