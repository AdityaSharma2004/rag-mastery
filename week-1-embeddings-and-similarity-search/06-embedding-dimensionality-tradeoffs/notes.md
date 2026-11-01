# 06 - Embedding dimensionality trade-offs (3072 vs 1536 vs 768)

> Week 1 · Topic 6 of 6 · Time: ~60 min · The memory maths and the truncation experiment run offline; Q2 uses the free Gemini API

## 🎯 What you will learn

- You will be able to explain what "dimensionality" means and why more dimensions can capture more meaning but cost more everywhere.
- You will be able to explain **Matryoshka (MRL)** embeddings and why Gemini lets you cut a vector to 768 or 1536 numbers.
- You will be able to **truncate correctly** (slice, then re-normalise) and avoid the classic silent bug.
- You will be able to compute memory and cost for millions of vectors, and apply quantisation (float16, int8).
- You will be able to measure the quality loss on **your own data** and choose a dimension with evidence.

## 🧠 The big idea in plain words

A vector with 3072 numbers is like describing a person with 3072 questions: "height, hair colour, accent, favourite food..." More questions = a more precise description. But every question costs you:

- **Storage:** each number is 4 bytes. 3072 numbers = 12 KB per chunk.
- **Speed:** comparing two vectors multiplies 3072 pairs of numbers.
- **Money:** a managed vector database charges (directly or indirectly) by memory.

The question is **how many questions you really need**. Gemini's embedding model was trained with a trick called **Matryoshka Representation Learning** (named after Russian nesting dolls): the *most important* information is packed into the *first* numbers, the next-most-important into the next, and so on. So the first 768 numbers are already a good, smaller embedding by themselves. You can throw away the tail and keep most of the quality.

Laravel analogy: choosing the dimension is like choosing between `VARCHAR(3072)` and `VARCHAR(768)` for an indexed column, except the "length" is the number of floats, and the price of getting it wrong shows up as RAM bills and slower queries, not an error message.

## 📚 Concepts, step by step

### 1. What is a dimension?

One **dimension** is one number in the vector. A "768-dimensional embedding" is a list of 768 numbers; a point in a 768-dimensional space. We cannot picture it, but the maths of topic 02 (dot product, cosine) works identically in any dimension.

More dimensions give the model more room to separate subtle meanings ("bank" the river edge vs "bank" the building). But there are diminishing returns: the first few hundred numbers carry most of the signal.

### 2. Gemini embedding sizes

| Option | Value |
|---|---|
| Native size | **3072** |
| Recommended sizes | **768, 1536, 3072** |
| How to choose | `output_dimensionality=...` in `EmbedContentConfig` |
| Normalised by the API? | **Only 3072.** Smaller sizes must be re-normalised by you |

For comparison in prose: OpenAI's `text-embedding-3-small` is 1536-d and `text-embedding-3-large` is 3072-d, and OpenAI's models also support shortening. The idea is the same across vendors; do not mix vendors in one index.

### 3. Matryoshka: why truncation works

Normal embedding models spread information evenly across all numbers, so deleting any number damages the meaning. MRL training adds a pressure: *the first 128 numbers alone must work, and the first 256 alone, and the first 512...* So prefixes are nested valid embeddings, like dolls.

```
[ n1 n2 ... n768 | n769 ... n1536 | n1537 ... n3072 ]
  \__ good enough for most tasks __/
  \________ a bit better ________________/
  \______________ best ______________________________/
```

Therefore, two equivalent ways to get a 768-d vector:
1. Ask the API for it: `output_dimensionality=768`.
2. Embed once at 3072, keep the first 768 numbers yourself.

Either way you **must re-normalise** (divide by the vector's length), because after cutting off numbers the length is no longer 1. Q2 proves the two routes agree.

### 4. The cost side: memory

```
bytes = number_of_vectors × dimensions × bytes_per_number
```

| Vectors | 3072 float32 | 1536 float32 | 768 float32 | 768 int8 |
|---|---|---|---|---|
| 100 K | 1.1 GiB | 0.57 GiB | 0.29 GiB | 0.07 GiB |
| 1 M | 11.4 GiB | 5.7 GiB | 2.9 GiB | 0.72 GiB |
| 5 M | 57 GiB | 29 GiB | 14.3 GiB | 3.6 GiB |

Vector databases (HNSW) additionally keep a graph and metadata, so budget roughly **1.3-2× the raw vector size** in RAM. Going 3072 → 768 cuts every one of these costs 4×, and also makes each similarity computation 4× cheaper.

### 5. Quantisation: shrinking the *numbers* instead of the *count*

Dimensions are the number of floats; **precision** is bytes per float.

| Type | Bytes | Idea |
|---|---|---|
| float32 | 4 | default |
| float16 | 2 | half precision, almost no loss for search |
| int8 (scalar quantisation) | 1 | map each float to an integer -127..127 using a scale factor |
| binary | 1 bit | keep only the sign of each number (32× smaller), used with re-scoring |

Q5 implements int8: `scale = max(|x|)/127; q = round(x/scale)`. Typical outcome: 4× smaller storage and a small recall drop that **re-scoring** the top candidates with the full vectors can recover. Qdrant, Pinecone and pgvector (halfvec) expose these natively (Week 2).

Combining both: 3072 float32 → 768 int8 is **16× smaller**.

### 6. How much quality do you lose?

It depends on the data, so *measure it*. The test (Q3):

1. Take real chunks and real questions.
2. For each question, find the top-k with the full 3072 vectors → this is your reference.
3. Truncate to 1536, 768, 512, 256 (slice + re-normalise), search again.
4. Report **overlap with the reference top-k** (and, better, hit-rate against human-labelled answers).

Typical pattern: quality falls slowly until a knee (often below 512), then drops faster. Published model cards show 768/1536 within a small margin of 3072 on standard benchmarks; **do not trust a benchmark for your corpus, trust your own measurement.**

### 7. Choosing a dimension

| Situation | Choose |
|---|---|
| Prototype, laptop, free tier, < 1 M chunks | **768** (our course default) |
| Need best accuracy, small corpus, hard domain (legal/medical) | 1536 or 3072 |
| Very large corpus, RAM-bound | 768 or 512 + int8, then rerank |
| Two-stage: cheap recall then precise rerank | Small dim for stage 1, full vector for re-scoring |

Rule of thumb for RAG: **improve chunking, hybrid search and reranking before paying for more dimensions.** They usually move quality more.

### 8. Other consequences of dimension

- **Index build time and query latency** grow with dimension.
- **Network/serialisation cost:** returning 3072 floats per hit in JSON is heavy; request IDs + payload, not vectors.
- **Compatibility:** an index has one fixed dimension. Changing it means re-embedding everything (a migration).
- **The "curse of dimensionality"** (distances concentrate) is real but mild with text embeddings, because they live on a lower-dimensional manifold.

## 💻 Code walkthrough

### Stage 1 - memory calculator

```python
def memory_gib(n_vectors, dim, bytes_per_number=4):
    return n_vectors * dim * bytes_per_number / 1024**3

print(memory_gib(5_000_000, 3072))      # ≈ 57.2
print(memory_gib(5_000_000, 768))       # ≈ 14.3
print(memory_gib(5_000_000, 768, 1))    # ≈ 3.6 (int8)
```

### Stage 2 - truncate correctly

```python
import numpy as np
from gemini_utils import embed_texts

full = embed_texts(["some chunk of text"], task_type="RETRIEVAL_DOCUMENT", dim=3072)   # (1, 3072), unit length

cut = full[:, :768]                                    # slice: first 768 numbers
print(np.linalg.norm(cut, axis=1))                     # NOT 1.0 (clearly below 1, e.g. 0.6-0.8)
cut = cut / np.linalg.norm(cut, axis=1, keepdims=True) # re-normalise: now 1.0
```

The one line `cut / norm` is the difference between correct similarity scores and quietly wrong ones.

### Stage 3 - measure what you lose

```python
def overlap_at_k(D_full, Q_full, d, k=5):
    truth = np.argsort(-(Q_full @ D_full.T), axis=1)[:, :k]
    Dd = D_full[:, :d] / np.linalg.norm(D_full[:, :d], axis=1, keepdims=True)
    Qd = Q_full[:, :d] / np.linalg.norm(Q_full[:, :d], axis=1, keepdims=True)
    got = np.argsort(-(Qd @ Dd.T), axis=1)[:, :k]
    return np.mean([len(set(t) & set(g)) / k for t, g in zip(truth, got)])

for d in (128, 256, 512, 768, 1536):
    print(d, round(overlap_at_k(D, Q, d), 2))
```

Expected on real text (≈): overlap near 0.9-1.0 for 768+ and falling for 128-256. With only 10 sample documents the numbers are coarse; use hundreds of chunks for conclusions.

### Stage 4 - int8 quantisation

```python
scale = np.abs(D).max() / 127.0
Dq = np.round(D / scale).astype(np.int8)         # 1 byte per number
scores = (Q @ Dq.T.astype(np.float32)) * scale   # approximate similarity
```

Re-score the top-50 with the original float vectors if you need exact ordering.

## ⚠️ Common mistakes & debugging

| Symptom | Cause | Fix |
|---|---|---|
| `ValueError: matmul ... mismatch in core dimension` | Query and index built with different dims | Pin dim in config; store it with the index; add a guard on insert |
| Scores much lower than expected after truncating | Forgot to re-normalise | Divide by the norm |
| Retrieval got worse after switching dim | Old and new vectors mixed in one index | Re-embed everything; version your index |
| Memory blow-up in production | Planned for raw vectors only | Budget 1.3-2× for index graph + payload |
| Truncated to 64 dims to "save money" and recall collapsed | Went below the knee of the curve | Measure; stay at ≥ 256-512 unless measured |
| Benchmarks say 768 is fine but my domain suffers | Benchmarks ≠ your data | Build a labelled set; test 768 vs 1536 vs 3072 |

## 🚀 Pro-level insights (what the top 2% know)

- **Treat dimension as a hyper-parameter you tune with a labelled set**, exactly like chunk size. Pick the smallest dimension within ~1% of your best hit-rate.
- **Two-stage retrieval:** store 768-d (cheap, fast) for candidate search; keep 3072-d (or float32) on disk and re-score the top 50-100. You get near-full accuracy at a fraction of RAM.
- **Combine tools in this order of impact:** better chunks → hybrid search → reranker → bigger dimension. Dimension is usually the *last* lever.
- **Budget the whole pipeline.** The embedding API call is a few milliseconds; the dominant latency for large indexes is the ANN search, which scales with dimension.
- **Binary quantisation + rescoring** is how some systems search 100M+ vectors on modest hardware; learn it once you outgrow int8.
- **Version your embeddings.** Store `model`, `dim`, `task_type`, and `normalised` as metadata; refuse to insert a mismatching vector (Q4's `VectorIndex` guard).
- **When NOT to shrink:** small corpora (< 100 K chunks). RAM is cheap at that scale; take the quality.

## 🧪 Mini lab (10-15 min)

1. Take 30 sentences (3 topics × 10) and 6 questions (2 per topic).
2. Embed at 3072. Compute top-3 overlap for dims 128, 256, 512, 768, 1536.
3. Plot (or print) overlap vs dimension. Where is the knee?
4. Quantise the 768-d version to int8. Does overlap drop? By how much?
5. Write a one-line rule: "For my data I would use ___ dims because ___."

## ✅ Recap checklist

- [ ] I can explain Matryoshka embeddings in two sentences.
- [ ] I always re-normalise after slicing.
- [ ] I can compute RAM for N vectors at any dimension and precision.
- [ ] I can implement int8 quantisation and measure recall loss.
- [ ] I can run a truncation experiment on my own data.
- [ ] I know dimension is a migration-level decision (changing it = re-embedding).

## 🎤 Interview questions

**1. What is a Matryoshka embedding?**
A model trained so that the first N numbers of the vector are themselves a good embedding for several values of N, so the vector can be truncated with graceful quality loss.

**2. You cut a 3072-d Gemini embedding to 768. What must you do before using dot product as cosine?**
Re-normalise (divide by the L2 norm), because only the 3072-d output is unit length.

**3. How much RAM do 10 million 768-d float32 vectors need, before index overhead?**
`10e6 × 768 × 4 B ≈ 30.7 GB` (≈28.6 GiB); with int8 about 7.7 GB. Budget extra for the index graph and metadata.

**4. How would you choose between 768 and 3072 for a legal-document RAG system?**
Build a labelled question set, measure hit-rate/MRR at both sizes (and with a reranker), choose the smallest that is within tolerance, and check memory/latency budgets.

**5. What is the cheapest way to recover accuracy lost to compression?**
Two-stage retrieval: search compressed vectors for candidates, then re-score the top few dozen with full-precision vectors or a cross-encoder reranker.

## 🔗 Next

**Week 2 - 01 ChromaDB:** put these vectors into a real (local) vector database and let it do the search, which unlocks persistence, metadata filtering and approximate nearest-neighbour indexes.
