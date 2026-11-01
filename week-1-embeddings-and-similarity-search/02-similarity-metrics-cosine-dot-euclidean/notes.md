# 02 - Similarity metrics: cosine, dot product, Euclidean

> Week 1 · Topic 2 of 6 · Time: ~60 min · Code: runs offline, no API key needed for the maths

## 🎯 What you will learn

- You will be able to compute dot product, vector length, cosine similarity and Euclidean distance **by hand** and in numpy.
- You will be able to explain, in one sentence each, what every metric actually measures (angle, angle × length, straight-line gap).
- You will be able to prove why all three give the **same ranking** on unit-length vectors, and say when they do not.
- You will be able to convert a score between "cosine similarity", "cosine distance", "squared L2" and "dot" so you never mis-set a threshold in Chroma, Qdrant, Pinecone or pgvector.
- You will be able to pick the right metric for a given embedding model without guessing.

## 🧠 The big idea in plain words

In topic 01 you saw that an embedding turns a sentence into a list of numbers, i.e. a **point in space**. Sentences with similar meaning land close together. Search is therefore one question: *"which stored points are closest to my query point?"*

But "closest" can be measured in more than one way. Imagine two arrows drawn from the same starting point (the origin):

```
        ^ y
        |    b
        |   /
        |  /   a
        | /  /
        |/ /
        +-------------> x
```

You could ask:
1. **How far apart are the arrow tips?** (Euclidean distance: a ruler between the tips.)
2. **Do the arrows point the same way?** (Cosine similarity: the angle between them, ignoring how long they are.)
3. **Do they point the same way AND are both long?** (Dot product: angle and length mixed together.)

For text, *direction is meaning* and *length is mostly noise* (a long paragraph is not "more cat" than a short sentence about a cat). That is why **cosine similarity is the default language of semantic search**.

Laravel analogy: think of three ways to compare two `products` rows. `WHERE price = price` is an exact match, `ABS(a.price - b.price)` is a distance, and "same category and similar tags" is closer to a direction. Different questions, different SQL. Choosing the metric is choosing the question.

## 📚 Concepts, step by step

### 1. The vector and its length

A vector is just a list of numbers: `a = [3, 4]`. Its **length** (also called the *norm* or *magnitude*) comes from Pythagoras:

```
|a| = sqrt(3² + 4²) = sqrt(25) = 5
```

For a 768-number vector, the same idea: square every number, add them, take the square root. In numpy: `np.linalg.norm(a)`.

A **unit vector** has length exactly 1. Turning any vector into a unit vector is called **normalising**: divide every number by the length. `[3, 4] / 5 = [0.6, 0.8]`.

### 2. Dot product

Multiply matching positions, then add:

```
a = [3, 4]
b = [4, 3]
a · b = 3×4 + 4×3 = 12 + 12 = 24
```

The dot product has a geometric meaning:

```
a · b = |a| × |b| × cos(θ)
```

where θ is the angle between the arrows. Read it as: *"how much do these point the same way, scaled by how long they both are."* Double the length of `b` and the dot product doubles, although the meaning did not change. That is both its strength (fast, one multiply-add per number) and its danger (length contaminates the score).

### 3. Cosine similarity

Remove the length effect by dividing it out:

```
cos(a, b) = (a · b) / (|a| × |b|)
          = 24 / (5 × 5) = 0.96
```

Range: **-1 to +1**.

| Value | Meaning |
|---|---|
| +1 | exactly the same direction (same meaning) |
| 0 | perpendicular (unrelated) |
| -1 | exactly opposite direction |

In practice, text embeddings from modern models rarely go below 0. Typical numbers with Gemini embeddings: ≈0.8-0.95 for paraphrases, ≈0.5-0.7 for same-topic, ≈0.3-0.5 for unrelated. **These are not universal**: they shift by model and by `task_type`, so never hard-code a threshold you did not measure.

**Cosine distance** is just `1 - cosine similarity`. Databases like Chroma and pgvector (`<=>`) report this because "distance" must be *small = close*. Same information, flipped.

### 4. Euclidean distance (L2)

The straight-line ruler between the two tips:

```
|a - b| = sqrt( (3-4)² + (4-3)² ) = sqrt(2) ≈ 1.414
```

Smaller = more similar. Many databases report the **squared** L2 (`1² + 1² = 2`) because it skips the square root and ranks identically. Chroma's default metric is squared L2 for exactly that reason.

### 5. The key theorem: on unit vectors, all three agree

If `|a| = |b| = 1`:

```
|a - b|²  =  |a|² - 2(a·b) + |b|²  =  1 - 2(a·b) + 1  =  2 - 2·cos(a, b)
a · b     =  cos(a, b)        (because both lengths are 1)
```

So: *dot product = cosine similarity*, and *squared L2 = 2 - 2 × cosine*. They are the same number wearing different clothes, and **sorting by any of them gives the identical ranking**.

This is the single most useful fact in this topic. Gemini's native 3072-dim vector is already normalised, and our helper `embed_texts` re-normalises truncated vectors, so on our data **dot = cosine**, and the fastest option (dot) is also the correct one.

### 6. When the metrics disagree (un-normalised vectors)

If vectors have different lengths, the three metrics *can* rank differently:

| Metric | Reacts to length? | Failure mode |
|---|---|---|
| Dot product | Yes, rewards long vectors | A huge irrelevant vector beats a small relevant one |
| Euclidean | Yes | A short vector near the origin looks "close" to everything short |
| Cosine | No | None for direction-only meaning |

Worked example (Q4 in the practice file): query `[1, 0]`, doc A `[0.9, 0.1]` (relevant), doc C `[5, 6]` (unrelated but big). Dot: A = 0.9, C = 5.0 → **C wins, wrongly**. Cosine: A = 0.99, C = 0.64 → A wins, correctly.

### 7. When is dot product *meant* to include length?

Some systems deliberately encode "importance" in the vector length, e.g. recommendation models trained with **maximum inner product search (MIPS)**. There, dot is the trained objective. For text embedding models you are told: *"use cosine"* (or normalise and use dot). Rule: **use the metric the model was trained with**. Gemini embeddings → cosine/dot on normalised vectors.

### 8. Distance vs similarity: the direction trap

| Metric | Better means | Threshold looks like |
|---|---|---|
| Cosine similarity | **higher** | keep `score >= 0.75` |
| Dot (unit vectors) | **higher** | keep `score >= 0.75` |
| Cosine distance | **lower** | keep `distance <= 0.25` |
| Squared L2 (unit) | **lower** | keep `distance <= 0.50` |
| L2 (unit) | **lower** | keep `distance <= 0.707` |
| pgvector `<#>` | lower (it returns the **negative** dot) | keep `score <= -0.75` |

Conversion on unit vectors (Q5 implements this):

```
cosine_distance = 1 - cos
squared_L2      = 2 - 2·cos
L2              = sqrt(2 - 2·cos)
```

### 9. A note on high dimensions

In 768+ dimensions, random vectors are almost always nearly perpendicular (cosine ≈ 0), so *any* strong similarity is meaningful signal. But the spread between "relevant" and "irrelevant" scores can be narrow (e.g. 0.62 vs 0.55), which is why ranking (top-k) is usually safer than absolute thresholds, and why rerankers (Week 2) matter.

## 💻 Code walkthrough

### Stage 1 - the three metrics in numpy

```python
import numpy as np

a = np.array([3.0, 4.0])
b = np.array([4.0, 3.0])

dot = a @ b                                   # 24.0
cos = dot / (np.linalg.norm(a) * np.linalg.norm(b))   # 0.96
l2 = np.linalg.norm(a - b)                    # 1.414
print(dot, cos, l2)
```

`@` is numpy's matrix/dot operator. Nothing here is Gemini-specific; it is plain maths.

### Stage 2 - double b: which metrics notice?

```python
b2 = 2 * b
print(a @ b2)                                              # 48.0  (changed)
print(a @ b2 / (np.linalg.norm(a) * np.linalg.norm(b2)))   # 0.96  (same)
print(np.linalg.norm(a - b2))                              # 5.385 (changed)
```

Cosine is the only one that says "same meaning".

### Stage 3 - many documents at once

Search never compares one pair; it compares one query against thousands of rows. Use matrix multiplication:

```python
rng = np.random.default_rng(0)
docs = rng.normal(size=(1000, 768))
docs /= np.linalg.norm(docs, axis=1, keepdims=True)   # normalise every row
query = rng.normal(size=768); query /= np.linalg.norm(query)

scores = docs @ query          # shape (1000,)  <- 1000 cosine similarities in ONE line
top3 = np.argsort(-scores)[:3] # indexes of the 3 best
print(top3, scores[top3])
```

`axis=1, keepdims=True` means "one length per row, keep it as a column so the division broadcasts". `np.argsort(-scores)` sorts descending (negative flips the order).

### Stage 4 - real Gemini vectors

```python
import sys; sys.path.insert(0, "common")
from gemini_utils import embed_texts, embed_query, cosine_sim, SAMPLE_DOCS

D = embed_texts(SAMPLE_DOCS, task_type="RETRIEVAL_DOCUMENT", dim=768)
q = embed_query("kitten sleeping", dim=768)

scores = D @ q                 # dot == cosine, because the helper returns unit vectors
for i in scores.argsort()[::-1][:3]:
    print(f"{scores[i]:.3f}  {SAMPLE_DOCS[i]}")
```

Expected (≈): the two cat/kitten sentences score highest (≈0.65-0.80), database sentences far lower (≈0.35-0.5).

### Stage 5 - verifying the unit-vector theorem

```python
a, b = rng.normal(size=(2, 768))
a /= np.linalg.norm(a); b /= np.linalg.norm(b)
print(np.sum((a - b) ** 2), 2 - 2 * (a @ b))   # identical to ~1e-16
```

## ⚠️ Common mistakes & debugging

| Symptom | Cause | Fix |
|---|---|---|
| Longest chunk always ranks first | Dot product on un-normalised vectors | Normalise rows, or use cosine |
| Scores look tiny after truncating dims | Sliced vector no longer length 1 | Re-normalise after slicing |
| Threshold keeps *everything* (or nothing) | Treated a *distance* like a *similarity* | Check "lower or higher is better" per DB; convert with the table |
| Different ranking in Chroma vs my numpy | Chroma default is squared L2, not cosine | Create the collection with `hnsw:space = cosine` (Week 2) |
| Cosine = NaN | Zero vector (empty text) | Skip/guard empty strings; clip the norm |
| pgvector scores negative | `<#>` returns negative inner product | Multiply by -1 |

## 🚀 Pro-level insights (what the top 2% know)

- **Pre-normalise once at ingestion, then use dot.** It is one multiply-add per number: the cheapest operation, SIMD/GPU friendly, and what ANN indexes (HNSW) are optimised for. Cosine computed at query time re-divides by norms for every row.
- **The metric must match the index.** An HNSW index built with L2 does not answer cosine queries correctly. Pick the metric when you *create* the collection; it is usually not changeable later (Laravel analogy: changing a column's collation after indexing).
- **Scores are not probabilities.** 0.8 does not mean "80% relevant". Calibrate with labelled queries (Q5 of topic 01 and later RAGAs in Week 4).
- **Relative gaps beat absolute thresholds.** "Top score minus second score" or "score ≥ 0.9 × best score" is often more stable than a fixed number across query types.
- **Task type changes the geometry.** Query vs document embeddings from `RETRIEVAL_QUERY`/`RETRIEVAL_DOCUMENT` are trained to sit near each other, so cosine between them is not comparable with `SEMANTIC_SIMILARITY` scores.
- **When NOT to use cosine alone:** when keywords/IDs matter ("error 0x80070057", SKU codes). Dense similarity blurs exact tokens; add BM25 (topic 05, Week 2 hybrid).

## 🧪 Mini lab (10-15 min)

1. Embed 8 sentences of your own (4 about one topic, 4 about another) with `embed_texts`.
2. Print the 8×8 cosine matrix with `np.round(cosine_sim(E, E), 2)`.
3. Find the **lowest within-topic** score and the **highest across-topic** score. Is there a clean gap? What threshold would separate the topics?
4. Multiply one embedding by 5 and recompute cosine vs dot. Which changes?
5. Write one sentence on what threshold you would NOT trust.

## ✅ Recap checklist

- [ ] I can compute dot, norm, cosine and L2 for `[3,4]` and `[4,3]` without a computer.
- [ ] I can state `a·b = |a||b|cosθ` and what each part means.
- [ ] I can prove `|a-b|² = 2 - 2cos(a,b)` for unit vectors.
- [ ] I know why dot = cosine for Gemini vectors from our helper.
- [ ] I can convert a similarity threshold into a distance threshold for Chroma, Qdrant and pgvector.
- [ ] I know when un-normalised dot product ranks wrongly.

## 🎤 Interview questions

**1. Why is cosine similarity preferred for text embeddings?**
Meaning is encoded in the direction of the vector, not its length. Length varies with text length and model quirks, so ignoring it removes noise.

**2. If vectors are normalised, which metric should I use?**
Any: they produce identical rankings. Use dot product because it is the cheapest.

**3. What is the relationship between cosine similarity and squared Euclidean distance?**
For unit vectors, `squared L2 = 2 - 2·cos`, a decreasing linear function, so sorting by one is sorting by the other.

**4. Your vector DB returns "distance 0.18" using cosine. What is the similarity?**
`1 - 0.18 = 0.82`.

**5. A teammate sets `similarity >= 0.8` as a hard cut-off across all queries. What is your concern?**
Scores are not calibrated: they vary by model, task type and query type. Measure on labelled data, or use a relative cut-off, and keep top-k plus reranking.

## 🔗 Next

**03 - Generating embeddings with Gemini:** call `gemini-embedding-001` yourself (task types, batching, rate limits, caching), which unlocks building real vectors for the search code in topic 04.
