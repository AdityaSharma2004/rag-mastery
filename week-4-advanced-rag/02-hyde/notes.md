# 02 - HyDE: Hypothetical Document Embeddings

> Week 4 · Topic 2 of 7 · Time: ~75 min · Runs offline with toy stand-ins; real Gemini when a key is set

## 🎯 What you will learn

- You will be able to explain **why a question's vector is far from its answer's vector** and how HyDE bridges that gap.
- You will be able to implement HyDE with `gemini-2.5-flash` + `gemini-embedding-001` (using the **document** task type for the hypothesis).
- You will be able to **blend** the question and hypothesis vectors and use several hypotheses.
- You will be able to recognise when HyDE **hurts** (hallucinated hypotheses) and guard against it with a confidence gate and fallbacks.
- You will be able to measure whether HyDE helps on *your* data.

## 🧠 The big idea in plain words

A user types: *"my page makes hundreds of database calls"*. The documentation says: *"Use eager loading with the `with` method to avoid the N+1 query problem."* To a human these match; to an embedding they share almost no words, and a question *looks* different from an explanation.

**HyDE** (Hypothetical Document Embeddings, Gao et al. 2022) says: do not search with the question. First ask an LLM to **write a plausible answer** (even if some details are wrong), then embed *that* and search. An answer-shaped text lands close to real answer-shaped chunks.

The hypothetical document is **never shown to the user**. It is only a better search key.

Laravel/MySQL analogy: instead of `WHERE title LIKE '%my page is slow%'`, you first translate the user's words into the vocabulary of your schema ("N+1", "eager loading") and *then* query. HyDE is the LLM doing that translation, in vector space.

## 📚 Concepts, step by step

### 1. Why questions and answers are far apart

Embedding models are trained so that a query lands near relevant passages, and `task_type="RETRIEVAL_QUERY"` helps. But when the user's wording is vague, colloquial or symptom-based, the vocabulary gap remains. In Q1 the question has cosine ≈ 0.0-0.1 to its gold chunk, while a hypothetical answer scores ≈ 0.7 (toy embedder; real models give smaller gaps but the same direction).

```
question  ──(far)──►  gold chunk
question ──LLM──► hypothetical answer ──(close)──► gold chunk
```

### 2. The HyDE algorithm

```
1. hypo   = LLM("Write a short documentation passage that answers: <question>")
2. vec    = embed(hypo, task_type="RETRIEVAL_DOCUMENT")      # NOT RETRIEVAL_QUERY: it should look like a document
3. chunks = vector_search(vec, k)
4. answer = LLM(question + chunks)                             # the normal RAG generation step
```
Steps 1-3 replace "embed the question and search". Nothing else in your pipeline changes.

Why `RETRIEVAL_DOCUMENT`? The hypothesis plays the role of a stored passage, and Gemini's document embeddings are what your index holds. Using the query type would push it toward "question space" again.

### 3. Writing a good hypothesis prompt

- Ask for **2-3 sentences** (short = cheaper, less hallucination).
- Say it is from **documentation** of your domain; mention the expected style ("Laravel docs", "HR policy").
- Tell the model it can be imprecise: *"some details may be wrong; use the vocabulary the documentation would use"*. We want vocabulary, not truth.
- Use `temperature` 0.2-0.4 for one hypothesis, 0.7 when sampling several.
- Use `thinking_budget=0` (our helper's default) for speed.

```python
prompt = ("Write a short passage (2-3 sentences) from technical documentation that would answer the question. "
          "It is fine if some details are imprecise; use the vocabulary the documentation would use.\n\n"
          f"Question: {query}\nPassage:")
hypo = generate(prompt, temperature=0.3, thinking_budget=0)
vec = embed_texts([hypo], task_type="RETRIEVAL_DOCUMENT")[0]
```

### 4. Improvements

**a) Blend with the original query.** `vec = normalise(α·q + (1−α)·mean(h))`. The question still carries real information (names, numbers) that a hypothesis may lose. Typical α = 0.2-0.5; Q3 sweeps it.

**b) Several hypotheses.** Sample 3-5 hypotheses at temperature ~0.7 and average their vectors (or fuse the result lists with RRF from topic 01). Diversity reduces the damage of one bad hypothesis.

**c) Confidence gate.** If the plain query already matches strongly (top-1 cosine above a threshold you tune), skip HyDE: no extra LLM call, no risk (Q4/Q5).

**d) Fallback.** If the LLM call fails (429, timeout, safety block), fall back to plain query search. A retrieval feature must never turn into an outage.

### 5. When HyDE helps and when it hurts

| Helps | Hurts |
|---|---|
| Short, vague, colloquial queries ("my page is slow") | Queries with exact names, ids, numbers ("error E-4521", "section 7.2(b)") |
| Cold-start domain with no labelled data | Niche/private facts the LLM cannot know: it **hallucinates a confident wrong answer** and retrieval follows it |
| Big vocabulary gap between users and docs | Already well-phrased queries (no gain, extra cost) |
| Cross-lingual or paraphrase-heavy corpora | Latency-critical paths (adds an LLM round trip, often 0.5-2 s) |

Rule: the hypothesis must contain **the vocabulary of the corpus, not its facts**. If the LLM knows nothing about your private product it will invent plausible wrong content; blending and gating contain the damage.

### 6. Related idea: index-time questions (doc2query / HyQE)

HyDE moves the query toward the documents *at query time*. The mirror trick moves documents toward queries at **ingestion time**: for each chunk, ask the LLM to write 3-5 questions it answers, embed those questions (storing a pointer to the chunk), and match user questions to them. Cost is paid once at ingestion (and cached), query latency is unchanged. Often combined with HyDE-free retrieval for latency-sensitive apps.

### 7. Cost and latency on the free tier

Per query HyDE adds one `generate` call (≈ 60-150 output tokens) and one embed call. With a confidence gate maybe only 30-60% of queries pay it. Cache hypotheses by `sha256(query)` (Q5). Remember free-tier rate limits: batch/cache, retry on 429 (our helper does), never loop over thousands of queries without throttling.

### 8. How to evaluate (the only honest way)

Build a gold set of 30-100 real user queries mapped to the chunks that answer them. Compare baseline vs HyDE vs blended on **recall@k and MRR**, per query, and read the losses: they tell you which kinds of queries to route away from HyDE. A single improved example proves nothing.

## 💻 Code walkthrough

### Stage 1 - setup and the offline toys

`solutions.py` defines a 12-chunk Laravel corpus, eight vague gold queries plus one well-phrased query, and the plug points `EMBED_DOCS`, `EMBED_QUERIES`, `HYPOTHETICAL`. Without a key they are toys (hashed words and a lookup table that plays the LLM); with `GEMINI_API_KEY` they call the real models.

### Stage 2 - plain HyDE (Q2)

```python
def hyde_search(query, k=5):
    hypo = HYPOTHETICAL(query, n=1)[0]
    vec = EMBED_DOCS([hypo])[0]          # document task type
    return search_vec(vec, k)            # vectors are unit length, so dot product = cosine
```

### Stage 3 - blending and multiple hypotheses (Q3)

```python
qv = EMBED_QUERIES([query])[0]
hv = EMBED_DOCS(HYPOTHETICAL(query, n=3))
vec = unit(alpha * qv + (1 - alpha) * hv.mean(axis=0))
```
`unit` re-normalises, because a blend of unit vectors is not unit length.

### Stage 4 - gate (Q4)

```python
top1 = float((DOC_VECS @ qv).max())
if top1 >= threshold: return search_vec(qv, k)      # confident: skip the LLM
```

### Stage 5 - production policy (Q5)

Gate → cache → HyDE → blend → fall back on exceptions. Count `llm_calls` and `fallbacks`; both belong on your dashboard.

### Expected shape of results with real Gemini (≈)

```text
baseline   recall@3 ≈ 0.55   mrr ≈ 0.50
HyDE       recall@3 ≈ 0.70   mrr ≈ 0.62     (vague queries gain, exact-match queries tie or lose)
blend 0.3  recall@3 ≈ 0.72   mrr ≈ 0.66
```
These are illustrative; the offline toy shows much bigger gains because its "LLM" was written for the queries.

## ⚠️ Common mistakes & debugging

| Symptom | Cause | Fix |
|---|---|---|
| HyDE results worse than baseline | Hypothesis hallucinated the wrong topic/framework | Blend with query (α 0.3-0.5), several hypotheses, gate, better prompt |
| Exact ids/error codes stop matching | Hypothesis dropped the literal | Blend, or use hybrid/BM25 for such queries (Week 2 topic 06) |
| No improvement, slower | Queries already well phrased | Confidence gate; skip HyDE for clear queries |
| Hypothesis embedded as a question | Used `RETRIEVAL_QUERY` | Embed with `RETRIEVAL_DOCUMENT` |
| Blended vector gives odd scores | Forgot to re-normalise | `unit()` after blending |
| Random failures in production | LLM 429/timeout/safety block | try/except fallback to plain query |
| Costs creeping up | Re-generating hypotheses each time | Cache by query hash; gate |
| Long rambling hypotheses | Prompt asks for an essay | "2-3 sentences", lower max tokens |
| Gains only on 3 hand-picked demos | No gold set | Evaluate recall@k / MRR on 30+ queries |

## 🚀 Pro-level insights (what the top 2% know)

- **HyDE is a query-side vocabulary translator.** If your gap is domain jargon, consider *fixing the index side* (doc2query, glossaries, synonyms) which costs nothing at query time.
- **Match the hypothesis style to the corpus** (a legal clause vs a README vs a ticket). Few-shot one example in the prompt for big gains.
- **Never rely on the hypothesis for facts.** It is a search key. Your final answer must come from retrieved chunks only (topic 05 of Week 3).
- **Route by query type:** ids/codes/quotes → keyword/hybrid; vague natural language → HyDE; clear questions → plain vector. A tiny classifier (or the confidence gate) pays for itself.
- **Combine with reranking:** HyDE broadens recall; a cross-encoder reranker (Week 2 topic 07) restores precision. This pairing is a common high-quality recipe.
- **Watch latency budgets:** HyDE's LLM call is serial before retrieval; run it concurrently with a plain-query search and use whichever finishes, or merge both with RRF.
- **Safety blocks:** hypothetical texts on sensitive topics may be refused by the model; always have the fallback.
- **Report it honestly:** "+X recall@5 on N=60 queries, +1.1 s p50" is a good internal claim; "HyDE improves RAG" is not.

## 🧪 Mini lab (10-15 min)

1. Write 10 vague queries about a topic you know and mark the chunk that answers each (use the Week 3 chunks of one of your own documents).
2. Measure baseline recall@3.
3. Implement HyDE with `generate` + `embed_texts(..., task_type="RETRIEVAL_DOCUMENT")` and re-measure.
4. Add the blend (try α = 0.3) and compare.
5. Find one query where HyDE loses and write down why.

## ✅ Recap checklist

- [ ] I can explain the question/answer vector gap in one sentence.
- [ ] I embed the hypothesis with the document task type.
- [ ] I blend with the original query and re-normalise.
- [ ] I have a gate and a fallback so HyDE can never break retrieval.
- [ ] I cache hypotheses and track LLM calls.
- [ ] I evaluate on a gold set and read the losses.

## 🎤 Interview questions

**1. What is HyDE?**
Generate a hypothetical answer with an LLM, embed that, and use it to search the vector store instead of the raw question, so the query vector lands near answer-shaped passages.

**2. Why can HyDE hurt?**
If the LLM does not know the domain it hallucinates confident wrong text, and retrieval follows it; exact tokens (ids, numbers) in the question can be lost.

**3. How do you make HyDE safer?**
Blend the query vector with the hypothesis, average several hypotheses, gate on the baseline's confidence, fall back to plain search on errors, and measure on a gold set.

**4. HyDE vs multi-query expansion?**
Multi-query writes several *queries* with different wording and fuses results; HyDE writes an *answer-like passage* and embeds it. They can be combined; HyDE addresses the question/answer style gap, multi-query the vocabulary gap.

**5. Which task type do you use for the hypothesis embedding and why?**
`RETRIEVAL_DOCUMENT`, because the hypothesis stands in for a stored passage and should live in the same space as the indexed chunks.

## 🔗 Next

**03 - Parent-child chunking:** retrieve with small, precise child chunks but hand the LLM their larger parent sections, so you get precise matching and enough context.
