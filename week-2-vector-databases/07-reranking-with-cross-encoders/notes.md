# 07 - Re-ranking with cross-encoders

> Week 2 · Topic 7 of 7 · Time: ~75 min · Local cross-encoder (free, `sentence-transformers`) **and** an LLM reranker with `gemini-2.5-flash` (free tier)

## 🎯 What you will learn

- You will be able to explain the **two-stage retrieve-then-rerank** architecture and why it beats either stage alone.
- You will be able to explain **bi-encoder vs cross-encoder** from first principles (what each computes and why one is fast and the other accurate).
- You will be able to run a local cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) and an LLM-as-reranker, behind one interface.
- You will be able to choose the candidate pool size using a quality-vs-latency measurement.
- You will be able to avoid the classic bugs: wrong sort order, wrong pair order, logits treated as probabilities, pool too small.

## 🧠 The big idea in plain words

Imagine hiring. Step one: a recruiter skims 10,000 CVs in an hour using keywords and a quick gut feel and picks 50 (**fast, approximate**). Step two: a hiring manager reads each of those 50 CVs *next to the job description*, carefully (**slow, precise**). You would never give the manager all 10,000, and you would never trust the recruiter's gut for the final call.

Search works the same way:

- **Stage 1: retrieval** (BM25 / vectors / hybrid) looks at a million chunks and returns ~50. It must be fast, so each chunk was pre-digested into a vector *without ever seeing the question*.
- **Stage 2: reranking** reads the question and each of the 50 chunks **together** and gives each a precise relevance score. It is slow per item, but 50 items is nothing.

Reranking is routinely the largest single quality jump after hybrid search: a model that actually reads question and passage side by side catches nuance (negation, "which of two", specific entities) that independent vectors blur.

## 📚 Concepts, step by step

### 1. Bi-encoder (what your embedding model is)

```
question ──► encoder ──► vector q      ┐
passage  ──► encoder ──► vector p      ┘  score = cosine(q, p)
```

- Question and passage are encoded **separately**.
- Passage vectors are computed **once, offline**, and stored.
- A query = 1 encoding + a fast nearest-neighbour search.
- Cost: O(1) model runs per query. **Scales to billions.**
- Weakness: the passage vector is a fixed summary made without knowing the question, so fine detail is lost.

### 2. Cross-encoder (the reranker)

```
[CLS] question [SEP] passage [SEP] ──► one transformer ──► one number (relevance)
```

- Question and passage go in **together**; every word of one can attend to every word of the other.
- Nothing can be precomputed: one forward pass **per (question, passage) pair**.
- Cost: O(N) model runs per query, so it is applied only to a short list.
- Strength: far more accurate relevance judgments.

| | Bi-encoder | Cross-encoder |
|---|---|---|
| Input | text alone | (query, passage) pair |
| Precompute doc side? | Yes | No |
| Cost per query | ~constant | grows with candidates |
| Accuracy | good | **best** |
| Role | stage 1 (recall) | stage 2 (precision) |

Q1 does the arithmetic: running a cross-encoder over 1,000,000 chunks at ~8 ms/pair is ~2 hours per query; ANN top-50 plus rerank is ~0.4 s. That 19,000× difference *is* the reason for two stages.

### 3. Recall vs precision: who does what

- Stage 1 must maximise **recall@N**: is the right chunk *somewhere* in the top N? If it is not, nothing downstream can find it.
- Stage 2 maximises **precision@k / MRR**: is the right chunk at the very top (what the LLM will see)?

So make N generous (20-100) and k small (3-8). Q5 shows recall@pool rising with N while rerank latency rises linearly.

### 4. Local cross-encoder with `sentence-transformers`

```python
from sentence_transformers import CrossEncoder
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")      # ~90 MB, downloaded once, runs on CPU
scores = model.predict([(query, p) for p in passages])             # list of (query, passage) pairs
```

- `ms-marco-MiniLM-L-6-v2` is small and fast (milliseconds per pair on CPU) and trained for web-style passage relevance in English.
- Output is a **logit** (any real number, e.g. -8 to +10), not a probability. Higher = more relevant. Compare scores *within one query*; do not compare across queries.
- Also available: `model.rank(query, documents, top_k=...)`, which returns sorted results for you.
- Larger/multilingual rerankers exist (e.g. BGE-reranker family) if you need Hindi or higher accuracy; test before switching.

### 5. LLM as a reranker (Gemini, free tier)

Ask `gemini-2.5-flash` to grade passages. Two styles:

- **Pointwise:** one call per passage ("score 0-10"). Simple but N calls: too slow/expensive for free tier.
- **Listwise (used in Q2):** one call with all N passages numbered, asking for a JSON list of N scores. One request, and the model compares passages against each other.

```python
prompt = f"""Question: {query}

Passages:
[0] ...
[1] ...

Rate each passage 0-10 for how well it answers the question.
Return ONLY JSON: {{"scores": [n0, n1, ...]}} with exactly N numbers."""
scores = generate_json(prompt, temperature=0.0)["scores"]
```

Safeguards: temperature 0, validate the list length, clamp to numbers, fall back to stage-1 order on any parse failure. Trade-offs vs a cross-encoder: more flexible (can follow instructions like "prefer recent policies") and multilingual, but slower (hundreds of ms to seconds), rate-limited on the free tier, and not perfectly deterministic.

### 6. Putting it together

```python
def retrieve_then_rerank(query, n_candidates=30, top_n=5):
    cand = hybrid_search(query, k=n_candidates)           # stage 1 (topics 05-06)
    scores = reranker.score(query, [chunk.text for chunk in cand])
    order = np.argsort(-np.array(scores))                 # DESCENDING
    return [cand[i] for i in order[:top_n]]
```

A single `Reranker` interface (`score(query, docs) -> list[float]`) lets you swap a cross-encoder, an LLM, or a stub in tests without touching the pipeline.

### 7. Choosing the pool size N

| N | Effect |
|---|---|
| Too small (3-5) | the right chunk often is not even in the pool; reranking cannot recover it |
| 20-50 | the usual sweet spot: recall plateaus, latency still OK |
| 100+ | diminishing returns, 2-5× latency |

Measure: for your labelled queries compute recall@N of stage 1 and MRR after reranking; pick the smallest N where both flatten (Q5 simulates the curve).

### 8. Evaluating the reranker

Run the labelled set through stage 1 only and through stage 1 + rerank: compare **MRR**, **hit@1**, **hit@3**. A reranker is worth its latency if hit@1 improves materially. Also check it does not *demote* correct results on code/ID queries (a general-purpose model can be worse than BM25 on exact tokens; keep hybrid scores as a tie-breaker or fuse them).

## 💻 Code walkthrough

### Stage 1 - one interface, three scorers

```python
class CrossEncoderReranker:
    def __init__(self, name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(name)
    def score(self, query, docs):
        return [float(s) for s in self.model.predict([(query, d) for d in docs])]

class LexicalStubReranker:          # offline fallback and for tests
    def score(self, query, docs):
        q = words(query); return [len(q & words(d)) / (len(q) or 1) for d in docs]
```

`best_available_reranker()` in the solutions tries the cross-encoder and falls back to the stub if the package or model download is unavailable.

### Stage 2 - the LLM reranker

See `llm_rerank(query, passages)` in Q2: build a numbered prompt, call `generate_json`, validate the length, return floats.

### Stage 3 - the pipeline and its evaluation

```python
top, cand = retrieve_then_rerank(query, D, DOCS, reranker, n_candidates=6, top_n=3)
```

then `mrr(before, truth)` vs `mrr(after, truth)` over `LABELED` (Q3).

### Stage 4 - timing

```python
import time
t = time.perf_counter(); reranker.score(query, docs[:30]); print((time.perf_counter() - t) * 1000, "ms")
```

Measure on *your* hardware; the cross-encoder cost scales linearly with candidates and passage length.

## ⚠️ Common mistakes & debugging

| Symptom | Cause | Fix |
|---|---|---|
| Worst result ranked first | `np.argsort(scores)` ascending | `np.argsort(-scores)` |
| Rerank makes things worse | Pairs passed as `(passage, query)` | Always `(query, passage)` |
| "Threshold 0.5 keeps nothing/everything" | Cross-encoder outputs logits | Apply sigmoid, then calibrate a threshold on labelled data |
| Reranker cannot find the answer | Stage-1 pool too small | Raise N to 20-50; improve stage 1 (hybrid) |
| Very slow | Reranking hundreds of long chunks on CPU | Cap N, truncate passages (model max ~512 tokens), batch, or use GPU |
| LLM reranker returns wrong number of scores | Free-form JSON | Validate length; retry once; fall back to stage-1 order |
| 429 on the LLM reranker | Free-tier rate limit | Listwise (1 call), cache by (query, candidate ids), back-off |
| Model download fails | Offline / blocked network | Pre-download on a machine with access; set `cache_folder`; use LLM reranker or stub meanwhile |
| Good on English, poor on Hindi | English-only model | Multilingual reranker or LLM reranker |

## 🚀 Pro-level insights (what the top 2% know)

- **Budget the pipeline end to end:** embed query (~50-300 ms API), vector/hybrid search (~10-50 ms), rerank (N × per-pair cost), LLM generation (seconds). Rerank is usually the cheapest quality win per millisecond.
- **Truncate smartly.** Cross-encoders have a max length (~512 tokens). Rerank on the most informative part of each chunk (or on child chunks, see parent-child in Week 4).
- **Cache** rerank results keyed by `(query_normalised, tuple(candidate_ids))`; repeated questions are common.
- **Combine signals:** final score = rerank score plus a small prior from the stage-1 rank (or RRF of both) to protect exact-match queries.
- **Score thresholds enable "I don't know":** if even the best reranked passage scores below a calibrated cut-off, answer "not found" rather than hallucinate.
- **Distil your LLM reranker:** use Gemini to label a few thousand (query, passage) pairs and fine-tune a small cross-encoder for your domain; you keep quality and cut latency/cost.
- **A/B test with real traffic and logs;** a reranker that wins on a benchmark can lose on your jargon.
- **When NOT to rerank:** tiny corpora where stage 1 already gets hit@1 ≈ 1.0, or hard latency budgets under ~100 ms.

## 🧪 Mini lab (10-15 min)

1. Take the 10 `DOCS` and 8 `LABELED` queries. Compute stage-1 MRR (dense, top 6).
2. Add the LLM reranker (`llm_rerank`) and recompute MRR. Count Gemini calls (should be 8).
3. Install `sentence-transformers`, run the cross-encoder, and compare MRR and time per query.
4. Shrink the pool to 3. Which queries break? Why?
5. Pick a score cut-off below which you would answer "I don't know"; justify it with the labelled data.

## ✅ Recap checklist

- [ ] I can draw bi-encoder vs cross-encoder and say why each exists.
- [ ] I can state the two-stage architecture and who optimises recall vs precision.
- [ ] I pass `(query, passage)` pairs and sort descending.
- [ ] I know cross-encoder outputs are logits, not probabilities.
- [ ] I can run a listwise LLM reranker with validation and fallback.
- [ ] I choose N by measuring recall@pool, MRR and latency.

## 🎤 Interview questions

**1. Why not use a cross-encoder for the whole corpus?**
It must process every (query, document) pair together with nothing precomputable, so cost is linear in corpus size per query; use a bi-encoder or hybrid search to shortlist first.

**2. What is the difference between recall@N and MRR after reranking?**
recall@N measures whether the correct chunk is in the candidate pool (stage 1's job); MRR after reranking measures how high it is placed (stage 2's job).

**3. A reranker made results worse for queries containing error codes. Why, and what do you do?**
A general semantic model can undervalue exact tokens; fuse its score with the BM25/hybrid rank or keep the keyword leg's top results protected.

**4. What are the trade-offs of an LLM reranker vs a cross-encoder?**
LLM: instruction-following and multilingual, but slower, rate-limited and less deterministic; cross-encoder: fast, cheap, deterministic, but fixed behaviour and often English-centric.

**5. How do you pick the number of candidates to rerank?**
Plot recall@pool and final MRR against N alongside latency; choose the smallest N after which both flatten, typically 20-50.

## 🔗 Next

**Week 3 - 01 Document loading:** turn real PDFs, Word files, web pages and CSVs into clean text with metadata, which unlocks building the full RAG pipeline on your own documents.
