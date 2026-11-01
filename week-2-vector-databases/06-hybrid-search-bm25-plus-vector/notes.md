# 06 - Hybrid search: BM25 + vector, fused

> Week 2 · Topic 6 of 7 · Time: ~75 min · Builds on Week 1 topic 05 (BM25 and RRF basics); this topic is the production version

## 🎯 What you will learn

- You will be able to build a **hybrid retriever** (keyword BM25 + dense vectors) with candidate pools and weighted **Reciprocal Rank Fusion**.
- You will be able to explain why raw-score fusion fails (scale, outliers) and compare min-max, z-score and rank-based fusion.
- You will be able to make the keyword leg robust with **tokenisation** that handles codes like `E-4012`.
- You will be able to run hybrid search **inside Postgres** (tsvector + GIN + pgvector, fused with RRF in one SQL query).
- You will be able to tune and evaluate hybrid search (pool size, `k`, leg weights) with MRR/hit@1 on labelled queries.

## 🧠 The big idea in plain words

Two searchers, two superpowers:

- The **keyword** searcher (BM25) is a stickler for exact words. It nails `E4012`, product codes, names, legal section numbers. It is blind to meaning: "money back" finds nothing about refunds (or finds something silly: a doc about "back-off" because it contains the word *back*, see Q1).
- The **vector** searcher understands meaning, so "money back" finds the refund article. It is fuzzy about exact tokens and may return a generic error article for `E4012`.

**Hybrid search** asks both and merges their rankings. A document both like rises to the top; a document only one loves still gets a chance. In practice this is the single most effective upgrade after good chunking, and it is cheap: BM25 costs microseconds.

Laravel analogy: Scout with a driver that can do both `MATCH ... AGAINST` (keyword) and a "similar items" lookup, whose two result lists you merge before rendering the page.

## 📚 Concepts, step by step

### 1. Two ranked lists, one merge

```
keyword ranking  : [2, 8, 9]          (doc ids, best first)
dense ranking    : [2, 6, 3, 1, ...]
hybrid ranking   : fuse(keyword, dense)  ->  [2, ...]
```

Never assume one list is "right". Retrieve a **pool** (top 20-50) from each, fuse, then keep the top 3-10 (or send them to a reranker, topic 07).

### 2. Why not just add the scores?

BM25 scores are unbounded and depend on the query (0-30 or more), cosine similarity is 0-1. Adding them means BM25 dominates (Q2). Normalising does not fully fix it:

| Method | Formula | Problem |
|---|---|---|
| min-max | `(x - min)/(max - min)` | one outlier (an exact-ID hit) squashes every other score toward 0 |
| z-score | `(x - mean)/std` | assumes a bell curve; unstable on 5-20 candidates |
| **RRF (rank-based)** | `Σ w_i / (k + rank_i)` | ignores magnitudes entirely: robust, no tuning of scales |

In Q2, an exact-ID BM25 outlier makes all three score-based methods pick document 0, while RRF picks document 1, which *both* methods rank highly.

### 3. RRF in detail

```
RRF(d) = Σ over lists i :  w_i / (k + rank_i(d))        rank starts at 1, default k = 60, default w_i = 1
```

Worked example. Keyword `[B, A, C]`, dense `[A, C, D]`:

| Doc | keyword rank | dense rank | score |
|---|---|---|---|
| A | 2 | 1 | 1/62 + 1/61 = 0.03252 |
| C | 3 | 2 | 1/63 + 1/62 = 0.03200 |
| B | 1 | - | 1/61 = 0.01639 |
| D | - | 3 | 1/63 = 0.01587 |

Order: **A, C, B, D**. Knobs:
- **`k`**: small `k` (e.g. 5) rewards rank 1 strongly; large `k` flattens. 60 is a robust default.
- **weights `w_i`**: give the keyword leg 1.5-2× when your queries are code-heavy; the dense leg more for natural language.
- **pool size**: if the correct doc is rank 35 in a leg and you only take the top 20, fusion cannot find it.
- **Drop zero-score keyword results** before fusing: a doc that matched no query term has no business earning rank points.

### 4. The keyword leg lives or dies by tokenisation

BM25 matches **tokens**. If the query and document are tokenised differently, nothing matches:

| Text | `split()` | alnum split | code-aware |
|---|---|---|---|
| `Error E-4012: timed out` | `["Error","E-4012:",...]` | `["error","e","4012"]` | `[..., "e","4012","e4012"]` |

Fixes: lowercase, strip punctuation, handle hyphenated codes (index both parts and the glued form), consider stemming (`refunds`→`refund`), keep numbers, remove stop-words, and for non-English use an appropriate analyser. Test your tokenizer with the real identifiers in your domain (Q4).

### 5. Where hybrid lives in each system

| System | How |
|---|---|
| **Your own code** (this topic) | `rank_bm25` + numpy/Chroma vectors + RRF function: simplest, fine for ≤ ~1M chunks in RAM |
| **PostgreSQL** | `tsvector` generated column + GIN index for keyword, pgvector for dense, RRF in SQL (Q5) |
| **Qdrant** | store a dense and a **sparse** vector per point (named vectors) and fuse server-side with the Query API (prefetch + RRF fusion) |
| **Pinecone** | sparse-dense vectors in one index with weighting at query time |
| **Elasticsearch / OpenSearch** | native BM25 + kNN with RRF |
| **Chroma** | no built-in BM25 fusion in the basic API: do it yourself as in this topic |

### 6. Sparse vectors, SPLADE (preview)

BM25 is "sparse": a vector as long as the vocabulary, almost all zeros. **Learned sparse models** (e.g. SPLADE) keep that format but learn which words matter and add related terms, giving keyword-style exactness with some semantic expansion. Qdrant and Pinecone store sparse vectors natively. Start with BM25; consider learned sparse when BM25 recall is the bottleneck.

### 7. Evaluate, do not guess

Collect 30-100 real queries with the correct chunk. Report MRR and hit@1/hit@5 for keyword, dense and hybrid. Then tune the leg weight and `k` and pool size and **stop when the metric stops moving**. Always include exact-ID and paraphrase queries; a test set of only one kind hides the failure of the other leg.

## 💻 Code walkthrough

### Stage 1 - tokenizer and BM25

```python
import re, numpy as np
from rank_bm25 import BM25Okapi

STOP = {"the", "a", "to", "of", "is", "are", "you", "your", "can", "on", "in", "if", "i", "my", "how", "what", "does"}
def tokenize(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]

bm25 = BM25Okapi([tokenize(d) for d in DOCS])
scores = bm25.get_scores(tokenize("error E4012"))
keyword_rank = [int(i) for i in np.argsort(-scores)[:20] if scores[i] > 0]
```

*(Note: `rank_bm25` uses an IDF variant that can be 0 or negative for terms present in half or more of a tiny corpus. Test with a realistic corpus size.)*

### Stage 2 - dense ranking

```python
from gemini_utils import embed_texts, embed_query
D = embed_texts(DOCS, task_type="RETRIEVAL_DOCUMENT", dim=768)
dense_rank = [int(i) for i in np.argsort(-(D @ embed_query("error E4012", dim=768)))[:20]]
```

### Stage 3 - weighted RRF

```python
def rrf(rank_lists, weights=None, k=60):
    weights = weights or [1.0] * len(rank_lists)
    scores = {}
    for w, ranking in zip(weights, rank_lists):
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + w / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])

top = rrf([keyword_rank, dense_rank], weights=[1.5, 1.0])[:5]
```

### Stage 4 - the class

`HybridRetriever(docs, pool=20, k_rrf=60, w_kw=1.0, w_dense=1.0)` wraps the two rankers; `search(q, k, mode="hybrid"|"keyword"|"dense")` lets you evaluate each leg and the fusion with one object (Q3).

### Stage 5 - hybrid in SQL (Postgres)

```sql
ALTER TABLE chunks ADD COLUMN tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;
CREATE INDEX ON chunks USING gin (tsv);

WITH semantic AS (SELECT id, row_number() OVER (ORDER BY embedding <=> :q) rnk FROM chunks ORDER BY embedding <=> :q LIMIT 20),
     keyword  AS (SELECT id, row_number() OVER (ORDER BY ts_rank_cd(tsv, query) DESC) rnk
                  FROM chunks, websearch_to_tsquery('english', :text) query WHERE tsv @@ query LIMIT 20)
SELECT COALESCE(s.id, k.id) id, COALESCE(1.0/(60+s.rnk),0) + COALESCE(1.0/(60+k.rnk),0) AS score
FROM semantic s FULL OUTER JOIN keyword k ON s.id = k.id ORDER BY score DESC LIMIT 5;
```

`FULL OUTER JOIN` keeps documents found by only one leg; `COALESCE(..., 0)` gives a missing leg zero points. One round trip, one database.

## ⚠️ Common mistakes & debugging

| Symptom | Cause | Fix |
|---|---|---|
| Hybrid ≈ BM25 | Raw scores added; BM25 dominates | RRF, or normalise per query |
| Exact code query fails | Tokeniser splits/keeps punctuation differently for query vs doc | One shared tokenizer; test with real codes |
| Keyword leg returns silly docs ("money back" → "back-off") | Lexical false positives | Expected: that is why the dense leg and a reranker exist |
| BM25 scores all 0 on a small test | Tiny corpus + IDF variant | Test on ≥ 50-100 docs |
| Correct doc missing from fused list | Pool too small | Increase pool to 30-50 per leg |
| Ties in RRF scores | Few candidates per leg | Larger pools; tie-break by dense score |
| Different results between runs | Query embeddings not cached / non-deterministic tie-breaks | Cache query embeddings, sort stably |
| Postgres `tsquery` syntax error | Raw user text in `to_tsquery` | Use `websearch_to_tsquery` or `plainto_tsquery` |

## 🚀 Pro-level insights (what the top 2% know)

- **Hybrid first, reranker second.** Fuse a generous pool (30-50), then rerank the top 20-30 with a cross-encoder (topic 07). That two-step pipeline is the standard high-quality RAG retriever.
- **Query-adaptive weights:** if the query contains digits, quotes or CamelCase identifiers, boost the keyword leg; for long natural-language questions boost the dense leg. A 10-line heuristic often beats fixed weights.
- **Filters apply to both legs.** Tenant/date filters must be in the keyword query *and* the vector query, or one leg leaks. Test it (topic 05).
- **Cache** query embeddings and BM25 tokenisation; both are pure functions.
- **Chunking affects BM25 more than dense:** keyword search likes shorter, focused chunks; very long chunks dilute term frequency.
- **BM25 parameters** `k1` (1.2-2.0) and `b` (0.75) rarely need tuning; tokenisation and stemming matter far more.
- **Observe in production:** log each leg's top-5 and the fused result. When an answer is wrong you can see *which leg failed*.
- **When NOT to bother:** tiny corpora where dense alone already scores near-perfect hit@1 on your labelled queries.

## 🧪 Mini lab (10-15 min)

1. Take `DOCS` (10 chunks) and add 5 chunks containing product codes (`SKU-8841`, `E-9001`).
2. Write 8 labelled queries: 3 code lookups, 3 paraphrases, 2 mixed.
3. Evaluate keyword, dense, hybrid (MRR/hit@1).
4. Try weights `(2,1)` and `(1,2)` and `k = 10, 60, 200`.
5. Write the failing queries for each mode and say why.

## ✅ Recap checklist

- [ ] I can build keyword + dense rankings with pools and fuse them with weighted RRF.
- [ ] I can explain why score addition and even min-max/z-score fail on outliers.
- [ ] I tested my tokenizer on codes, hyphens and case.
- [ ] I can write the Postgres hybrid SQL (tsvector + pgvector + RRF).
- [ ] I know how Qdrant/Pinecone expose hybrid (sparse + dense).
- [ ] I evaluate with labelled exact-ID and paraphrase queries.

## 🎤 Interview questions

**1. Why is hybrid search usually better than vector search alone?**
Dense retrieval misses exact tokens (IDs, names, numbers); BM25 misses paraphrases. Fusing the two rankings covers both failure modes.

**2. Why RRF instead of weighted score sums?**
Scores from different systems are on incomparable, query-dependent scales; ranks are comparable, so RRF needs no normalisation and resists outliers.

**3. How do you implement hybrid search in Postgres?**
A `tsvector` generated column with a GIN index for keyword ranking, pgvector for semantic ranking, then fuse the ranks with RRF in one SQL query using a `FULL OUTER JOIN`.

**4. Your hybrid retriever misses `E-4012` but finds `E4012`. Diagnosis?**
Tokenisation mismatch (hyphen handling); use a code-aware tokenizer applied identically to documents and queries.

**5. How do you tune hybrid search?**
Build a labelled query set including exact and paraphrase queries; vary pool size, RRF `k` and leg weights; track MRR/hit@k; stop when gains flatten.

## 🔗 Next

**07 - Re-ranking with cross-encoders:** take the fused candidates and reorder them with a model that reads query and chunk together, which unlocks the largest single precision gain in the retrieval stack.
