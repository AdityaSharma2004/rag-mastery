# 05 - Semantic vs keyword vs hybrid search

> Week 1 · Topic 5 of 6 · Time: ~75 min · Pure-Python BM25 and fusion need no API; the semantic part uses the free Gemini embedding API

## 🎯 What you will learn

- You will be able to explain how keyword search (`LIKE`, FULLTEXT, BM25) and semantic search (embeddings) work, and where each one fails.
- You will be able to implement **BM25** from scratch and read its formula.
- You will be able to implement **hybrid search** with **Reciprocal Rank Fusion (RRF)** and explain why raw score addition is a bug.
- You will be able to measure the three methods on your own data with **MRR** and **hit@1**, instead of arguing from opinion.
- You will be able to decide which method a given product needs.

## 🧠 The big idea in plain words

You have a support knowledge base. A customer types: **"I want my money back."** The correct article says: *"Refunds are issued to the original payment method within 5 business days."*

- The two texts share **no words**. A keyword engine finds nothing.
- An embedding model knows "money back" ≈ "refund" and finds it.

Now the customer types **"error E4012"**. The right article contains the token `E4012`.

- A keyword engine finds it instantly because the token is rare and exact.
- An embedding model may blur "E4012" into "some error code" and return a generic error article.

So **each method has a blind spot the other covers**. Hybrid search uses both and merges the results.

MySQL analogy, which you already know:

| In MySQL | In RAG | Finds by |
|---|---|---|
| `WHERE body LIKE '%refund%'` | substring match | exact characters, in order |
| `MATCH(body) AGAINST('refund')` (FULLTEXT) | keyword / BM25 | shared *words*, ranked by rarity |
| *(no built-in)* | semantic / vector search | shared *meaning* |

FULLTEXT is already a step above `LIKE` (it tokenises, ranks and ignores stop-words), but it is still **lexical**: it never knows that "refund" and "money back" are related.

## 📚 Concepts, step by step

### 1. `LIKE` search

```sql
SELECT * FROM kb WHERE body LIKE '%reset password%';
```

Matches only if the exact phrase appears. "To reset **your** password" fails because "your" is in the middle. No ranking, no stemming, no synonyms, and `%...%` cannot use a normal index (full table scan).

### 2. Keyword search (inverted index + BM25)

**Inverted index:** instead of "document → words", store "word → list of documents". Looking up a word is instant, like the index at the back of a book.

```
password → [doc0, doc3]
refund   → [doc1]
e4012    → [doc2]
```

A query becomes a few lookups. But which matching document is *best*? That is **ranking**, and the industry standard is **BM25** (Best Matching 25; used by Elasticsearch, OpenSearch, Lucene, SQLite FTS5, Postgres `ts_rank` variants).

### 3. BM25 explained from first principles

Three intuitions:

1. **Rare words matter more.** "the" appears everywhere, so it says nothing; "E4012" appears once, so a match is a strong signal. This is **IDF** (inverse document frequency).
2. **More occurrences help, with diminishing returns.** Mentioning "refund" 3 times is better than once, but 30 times is not 10× better. This is **term-frequency saturation** (parameter `k1`).
3. **Long documents are not automatically better.** A match inside a 10-word doc is more meaningful than inside a 1,000-word doc. This is **length normalisation** (parameter `b`).

The formula, for query term `w` and document `d`:

```
score(d) = Σ over query words w:  IDF(w) × [ tf(w,d) × (k1 + 1) ] / [ tf(w,d) + k1 × (1 - b + b × |d| / avgdl) ]

IDF(w) = ln( (N - n(w) + 0.5) / (n(w) + 0.5) + 1 )
```

| Symbol | Meaning |
|---|---|
| `N` | number of documents |
| `n(w)` | number of documents containing `w` |
| `tf(w,d)` | how many times `w` appears in `d` |
| `|d|`, `avgdl` | length of this doc, average doc length |
| `k1 ≈ 1.2-2.0` | saturation speed (default 1.5) |
| `b ≈ 0.75` | strength of length normalisation (0 = off, 1 = full) |

Worked mini example, `N = 8` docs, query word `e4012` appears in 1 doc: `IDF = ln((8-1+0.5)/(1+0.5)+1) = ln(6) ≈ 1.79`. The word `payment` appears in 2 docs: `IDF = ln((8-2+0.5)/(2+0.5)+1) = ln(3.6) ≈ 1.28`. So a match on the rarer token is worth more, exactly as intuition says.

**Strengths:** exact terms, IDs, names, codes, rare jargon, very fast, no model, cheap, explainable.
**Weaknesses:** synonyms ("money back" vs "refund"), paraphrase, multilingual, typos, word order, intent.

### 4. Semantic search (dense vectors)

Embed every chunk once; embed the query; rank by cosine similarity (topics 01-04). It understands meaning, so "lost access to my profile" finds the account-recovery article. **Weaknesses:** it can miss exact tokens and numbers (`E4012`, `Section 21(2)`), rare names, and negation ("with" vs "without"), and it always returns *something* (the least-bad match), even when nothing is relevant.

### 5. Hybrid search

Run both, **fuse** the two ranked lists into one. Fusion options:

**(a) Weighted sum of scores.** `final = α × bm25 + (1-α) × cosine`. Looks easy; it is a trap, because BM25 is unbounded (0-30) and cosine is 0-1. Without normalising, BM25 silently dominates (Q4). You must scale both first (min-max), and the scale depends on each query. Fragile.

**(b) Reciprocal Rank Fusion (RRF).** Ignore scores, use only **ranks**:

```
RRF(d) = Σ over rankings:  1 / (k + rank(d))        (k = 60 is the standard constant, rank starts at 1)
```

Worked example. Keyword ranking: `[B, A, C]`. Semantic ranking: `[A, C, D]`.

| Doc | Keyword rank | Semantic rank | RRF score |
|---|---|---|---|
| A | 2 | 1 | 1/62 + 1/61 = 0.01613 + 0.01639 = **0.03252** |
| B | 1 | - | 1/61 = 0.01639 |
| C | 3 | 2 | 1/63 + 1/62 = 0.01587 + 0.01613 = 0.03200 |
| D | - | 3 | 1/63 = 0.01587 |

Final order: **A, C, B, D**. A document that both methods like beats a document only one loves, with no tuning of scales. That is why RRF is the default in Elasticsearch, OpenSearch, Qdrant, Weaviate and most RAG stacks. The constant `k` softens the advantage of rank 1 over rank 2; larger `k` = flatter.

**(c) Learned fusion / reranker** (Week 2): fuse top candidates, then let a cross-encoder reorder. Best quality, more latency.

### 6. When to use which

| Situation | Best choice |
|---|---|
| Natural-language questions, paraphrase, multilingual | Semantic |
| IDs, error codes, SKUs, legal section numbers, names | Keyword / BM25 |
| Mixed real-world RAG (most products) | **Hybrid (BM25 + dense, RRF)**, then optionally rerank |
| Tiny corpus (< a few hundred docs), prototype | Semantic alone is fine |
| Structured filters (date, tenant) | Metadata filtering (Week 2/3), not either method |

### 7. Evaluating objectively

Write 20-50 labelled queries: `(query, id_of_correct_doc)`. Compute:
- **hit@1:** fraction where the correct doc is ranked first.
- **MRR** (mean reciprocal rank): average of `1 / rank_of_correct_doc` (rank 1 → 1.0, rank 2 → 0.5, missing → 0).

Compare keyword, semantic and hybrid on *your* queries. Q5 builds this harness.

## 💻 Code walkthrough

### Stage 1 - tokenizer and a toy corpus

```python
import re, math
import numpy as np
from collections import Counter

KB = [
    "To reset your password click Forgot password on the login page.",
    "Refunds are issued to the original payment method within 5 business days.",
    "Error E4012 means the payment gateway timed out; retry after one minute.",
    "If you cannot access your account, use the recovery link sent to your email.",
]
STOP = {"the", "a", "to", "your", "on", "you", "if", "is", "are", "of", "in", "means"}

def tokenize(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]
```

`[a-z0-9]+` keeps `e4012` as one token. Real engines also stem ("refunds" → "refund"); we skip that for clarity.

### Stage 2 - BM25 in ~25 lines

```python
class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.toks = [tokenize(d) for d in docs]
        self.N = len(docs)
        self.avgdl = sum(len(t) for t in self.toks) / self.N
        df = Counter(w for t in self.toks for w in set(t))        # document frequency
        self.idf = {w: math.log((self.N - n + 0.5) / (n + 0.5) + 1) for w, n in df.items()}

    def score(self, query):
        out = np.zeros(self.N)
        for i, t in enumerate(self.toks):
            tf, dl = Counter(t), len(t)
            for w in tokenize(query):
                if w in tf:
                    num = tf[w] * (self.k1 + 1)
                    den = tf[w] + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                    out[i] += self.idf[w] * num / den
        return out
```

Line by line: `df` counts in how many documents each word appears (`set(t)` so repeated words count once); `idf` is the formula above; `score` adds one term per query word that the document contains.

Run: `BM25(KB).score("payment gateway error")` → the E4012 doc (index 2) scores highest, the refund doc (shares "payment") second, the rest 0.

### Stage 3 - semantic scores

```python
from gemini_utils import embed_texts
D = embed_texts(KB, task_type="RETRIEVAL_DOCUMENT", dim=768)
q = embed_texts(["I want my money back"], task_type="RETRIEVAL_QUERY", dim=768)[0]
sem = D @ q                       # cosine, since vectors are unit length
print(np.argsort(-sem))           # the refund doc (1) should be first
```

### Stage 4 - RRF

```python
def rrf(rank_lists, k=60):
    scores = {}
    for ranking in rank_lists:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])

kw = BM25(KB).score("I want my money back")
kw_rank = [int(i) for i in np.argsort(-kw) if kw[i] > 0]     # drop zero-score docs: they matched nothing
sem_rank = [int(i) for i in np.argsort(-sem)]
print(rrf([kw_rank, sem_rank])[:3])
```

Dropping zero-score documents from the keyword list matters: a document that matched nothing should not earn rank points.

Expected (≈): refund doc first. For the query `"error E4012"` the E4012 doc is first, supported by *both* lists.

## ⚠️ Common mistakes & debugging

| Symptom | Cause | Fix |
|---|---|---|
| Hybrid is no better than BM25 | Added raw BM25 + cosine; BM25 scale dominates | Normalise per query (min-max) or use RRF |
| Keyword list full of junk | Zero-score docs included in the ranking | Filter `score > 0` before fusing |
| Exact ID query returns wrong doc in semantic mode | Dense models blur rare tokens | Add BM25 leg (hybrid) |
| Paraphrase query returns nothing from keyword | No lexical overlap | Expected; rely on the semantic leg |
| BM25 ignores "Refunds" for query "refund" | No stemming/lemmatisation | Add a stemmer (Porter/Snowball) or n-grams |
| Stop-words dominate scores | No stop-word list / IDF too flat on tiny corpus | Add a stop-word list; use larger corpus |
| RRF output has ties | Few candidates in each list | Use a larger pool (top 20-50 each) |

## 🚀 Pro-level insights (what the top 2% know)

- **Always evaluate before choosing.** On product search and legal/technical text, hybrid typically beats either alone by a clear margin; on conversational FAQ, semantic alone may be enough. Your labelled set decides.
- **Fuse deep, return shallow.** Retrieve top 20-50 from each leg, RRF, then keep the top 5-10 (or rerank them). Fusing only the top 3 of each throws away overlap.
- **Tune `k` and weights last.** RRF with `k=60` is a strong default. Weighted RRF (`w_kw / (k+rank)`) is the next lever if one leg is clearly stronger.
- **Keyword leg needs good analysis.** Lowercasing, accent folding, stemming, and handling of codes (`E-4012` vs `E4012`) change recall more than tuning `k1`/`b`.
- **Sparse learned models** (SPLADE) and **BM25 inside your vector DB** (Qdrant sparse vectors, pgvector + Postgres FTS) give hybrid without running a separate Elasticsearch. Week 2 covers this.
- **Dense search always returns something.** Add a minimum score or a "no answer" path; hybrid does not fix that.
- **Latency:** BM25 is sub-millisecond on 100k docs; embedding the query is the main cost (one API call). Cache query embeddings (Week 4).
- **When NOT to go hybrid:** when you have no exact-match needs and a tiny corpus; the added complexity is not worth it.

## 🧪 Mini lab (10-15 min)

1. Add 4 more documents to `KB` (include one with a product code like `SKU-8841`).
2. Write 6 labelled queries: 2 exact-token, 2 paraphrase, 2 mixed.
3. Print MRR and hit@1 for keyword, semantic, hybrid.
4. Change RRF `k` from 60 to 5 and to 500. Which queries change?
5. Write down which method you would ship for this KB and why, in two sentences.

## ✅ Recap checklist

- [ ] I can explain why `LIKE` and FULLTEXT miss "money back" → "refund".
- [ ] I can write the BM25 formula and name the role of IDF, `k1` and `b`.
- [ ] I can compute RRF by hand for two ranked lists.
- [ ] I know why adding BM25 and cosine scores directly is a bug.
- [ ] I can compute MRR and hit@1.
- [ ] I can say when keyword, semantic or hybrid is the right choice.

## 🎤 Interview questions

**1. Why can't pure semantic search replace keyword search?**
Embeddings blur rare exact tokens (IDs, codes, names, numbers) and may miss them; BM25 matches them precisely. Also BM25 is cheap, explainable and model-free.

**2. What does BM25 add over TF-IDF?**
Term-frequency saturation (diminishing returns for repeated words) and document-length normalisation, both tunable with `k1` and `b`.

**3. Why use RRF instead of a weighted sum of scores?**
BM25 and cosine scores are on different, query-dependent scales; RRF fuses ranks, which are comparable, with no normalisation or tuning.

**4. A query returns relevant results in keyword search but not in vector search. What do you check?**
Whether the query depends on exact tokens (IDs, codes), task types used for query/document embeddings, chunking that split the relevant text, and then add a BM25 leg.

**5. How do you decide if hybrid is worth it?**
Build a labelled query set from real traffic, compare hit@k/MRR for keyword, semantic and hybrid, and weigh the gain against added latency and complexity.

## 🔗 Next

**06 - Embedding dimensionality trade-offs:** measure what you gain and lose going from 3072 to 1536 or 768 dimensions, which unlocks sizing memory and cost for the vector databases in Week 2.
