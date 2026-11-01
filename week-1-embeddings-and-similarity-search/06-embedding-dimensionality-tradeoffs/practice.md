# Practice - Embedding dimensionality trade-offs

Five questions, rising difficulty. Attempt each before opening the hint, and the hint before the solution. Every solution is the exact code in `solutions.py`. **Q1, Q4 and Q5 run offline with no API key.** Q2 needs the free Gemini key. Q3 uses Gemini when a key is present and otherwise falls back to synthetic "Matryoshka-like" data, so it always runs.

---

### Q1. Warm-up - a memory table

**Problem**

For 1,000,000 vectors, print the raw storage (GiB) for dimensions 256, 512, 768, 1536, 3072 in float32, float16 and int8. How many times smaller is 768 than 3072?

<details><summary><b>Hint</b></summary>

`bytes = N × dim × bytes_per_number`; divide by `1024**3` for GiB. float32 = 4, float16 = 2, int8 = 1.

</details>

<details><summary><b>Solution</b></summary>

```python
# --------------------------------------------------------------------------
# Q1. Warm-up: a memory table for every dimension size
# --------------------------------------------------------------------------
def memory_gib(n_vectors: int, dim: int, bytes_per_number: int = 4) -> float:
    return n_vectors * dim * bytes_per_number / 1024**3


def q1():
    n = 1_000_000
    print(f"{n:,} vectors, raw storage (no index overhead)")
    print(f"{'dim':>6} {'float32':>10} {'float16':>10} {'int8':>10}")
    for dim in (256, 512, 768, 1536, 3072):
        print(f"{dim:>6} {memory_gib(n, dim, 4):>9.2f}G {memory_gib(n, dim, 2):>9.2f}G {memory_gib(n, dim, 1):>9.2f}G")
    ratio = memory_gib(n, 3072) / memory_gib(n, 768)
    print(f"\n3072 -> 768 saves {ratio:.0f}x memory")
    return ratio
```

</details>

**Why it works / what to learn**
Memory is linear in dimension and in bytes per number, so 3072 → 768 is exactly 4× smaller, and float32 → int8 another 4×. Always do this arithmetic *before* choosing a database plan.

**Expected output (≈)**
```text
1,000,000 vectors, raw storage (no index overhead)
   dim    float32    float16       int8
   256      0.95G      0.48G      0.24G
   512      1.91G      0.95G      0.48G
   768      2.86G      1.43G      0.72G
  1536      5.72G      2.86G      1.43G
  3072     11.44G      5.72G      2.86G

3072 -> 768 saves 4x memory
```

---

### Q2. Apply - truncate by hand vs ask the API

**Problem**

Embed the 10 `SAMPLE_DOCS` at 3072 dims and at 768 dims. Build your own 768-d version by slicing the 3072 vector and re-normalising. (a) What is the norm of the raw slice? (b) How close is your slice to the API's 768-d vector (cosine per text)?

<details><summary><b>Hint</b></summary>

`full[:, :768]`, then divide by `np.linalg.norm(..., axis=1, keepdims=True)`. Cosine between two unit vectors is just their row-wise dot product.

</details>

<details><summary><b>Solution</b></summary>

```python
# --------------------------------------------------------------------------
# Q2. Apply: Matryoshka truncation by hand (slice + re-normalise) vs asking the API for 768
# --------------------------------------------------------------------------
def q2():
    texts = SAMPLE_DOCS
    full = embed_texts(texts, task_type="SEMANTIC_SIMILARITY", dim=3072)     # native size, unit length
    asked = embed_texts(texts, task_type="SEMANTIC_SIMILARITY", dim=768)     # API-side truncation (+ our normalise)

    sliced = unit(full[:, :768])                       # our own slice + re-normalise

    sliced_raw_norm = np.linalg.norm(full[:, :768], axis=1)
    print("norm of a raw 768-slice of a 3072 vector:", np.round(sliced_raw_norm[:3], 3), "(not 1.0)")

    agreement = np.sum(sliced * asked, axis=1)         # cosine between the two 768-d versions of each text
    print("cosine(our slice, API 768) per text:", np.round(agreement, 4))
    return float(agreement.min())
```

</details>

**Why it works / what to learn**
Because of Matryoshka training the first 768 numbers *are* the 768-d embedding, so your manual slice and the API's output point in nearly the same direction (cosine very close to 1.0). The raw slice has length well below 1, which is the trap: always re-normalise.

**Expected output (≈)**
```text
norm of a raw 768-slice of a 3072 vector: [values clearly below 1.0] (not 1.0)
cosine(our slice, API 768) per text: [≈0.99+ ...]
```

---

### Q3. Build - how much quality survives truncation?

**Problem**

Write `truncation_report(D_full, Q_full, dims, k)` that, for each smaller size, computes the average fraction of the top-k results (found with the full vectors) that are still found after slicing and re-normalising. Run it on real Gemini vectors if you have a key, otherwise on the synthetic corpus.

<details><summary><b>Hint</b></summary>

Reference ranking: `np.argsort(-(Q @ D.T), axis=1)[:, :k]`. For each dimension `d`, slice `[:, :d]`, re-normalise, rank again, then compare sets per query.

</details>

<details><summary><b>Solution</b></summary>

```python
# --------------------------------------------------------------------------
# Q3. Build: how much retrieval quality survives truncation?
# --------------------------------------------------------------------------
def truncation_report(D_full: np.ndarray, Q_full: np.ndarray, dims=(128, 256, 512, 768, 1536), k: int = 3):
    """For each smaller size, how many of the TOP-k results (found with the full vectors) are still found?"""
    truth = np.argsort(-(Q_full @ D_full.T), axis=1)[:, :k]
    rows = []
    for d in dims:
        Dd, Qd = unit(D_full[:, :d]), unit(Q_full[:, :d])
        got = np.argsort(-(Qd @ Dd.T), axis=1)[:, :k]
        overlap = np.mean([len(set(t) & set(g)) / k for t, g in zip(truth, got)])
        rows.append((d, float(overlap)))
    return rows


def synthetic_corpus(n_docs=400, n_queries=50, dim=1536, seed=0):
    """Fake 'Matryoshka-like' data: early coordinates carry most of the variance, like a real MRL model."""
    rng = np.random.default_rng(seed)
    scale = 1.0 / np.sqrt(np.arange(1, dim + 1))               # decaying importance per coordinate
    centers = rng.normal(size=(20, dim)) * scale                # 20 topics
    topic = rng.integers(0, 20, size=n_docs)
    D = unit(centers[topic] + 0.35 * rng.normal(size=(n_docs, dim)) * scale)
    qtopic = rng.integers(0, 20, size=n_queries)
    Q = unit(centers[qtopic] + 0.35 * rng.normal(size=(n_queries, dim)) * scale)
    return D, Q


def q3():
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        docs = SAMPLE_DOCS
        queries = ["kitten sleeping", "keyword search in MySQL", "how are fundamental rights protected",
                   "in-memory cache", "managed database service"]
        D = embed_texts(docs, task_type="RETRIEVAL_DOCUMENT", dim=3072)
        Q = embed_texts(queries, task_type="RETRIEVAL_QUERY", dim=3072)
        dims, k, source = (128, 256, 512, 768, 1536), 3, "Gemini 3072-d (10 docs, 5 queries)"
    else:
        D, Q = synthetic_corpus()
        dims, k, source = (32, 64, 128, 256, 512, 768), 5, "synthetic Matryoshka-like data (no API key found)"
    print("data:", source)
    print(f"{'dim':>6} {'top-%d overlap with full vectors' % k}")
    rows = truncation_report(D, Q, dims, k)
    for d, o in rows:
        print(f"{d:>6} {o:>8.2f}")
    return rows
```

</details>

**Why it works / what to learn**
You get a quality-vs-size curve instead of an opinion. Typical shape: flat, then a knee, then a drop. With the synthetic data (decaying variance per coordinate, like a real MRL model) overlap climbs smoothly with dimension. With real Gemini vectors and only 10 documents the numbers are coarse, so repeat with hundreds of real chunks before deciding.

**Expected output (≈, synthetic fallback)**
```text
data: synthetic Matryoshka-like data (no API key found)
   dim top-5 overlap with full vectors
    32     0.82
    64     0.88
   128     0.91
   256     0.94
   512     0.96
   768     0.99
```

---

### Q4. Debug - shape crash and silently wrong scores

**Problem**

After a refactor, search crashes with `ValueError: matmul: Input operand 1 has a mismatch in its core dimension`. Elsewhere a teammate sliced vectors to 768 and scores dropped. Reproduce both problems, then write a `VectorIndex.add()` guard that rejects wrong-dimension and non-normalised vectors.

<details><summary><b>Hint</b></summary>

An index created at 768 dims receives a query produced at 3072. For the second bug print `np.linalg.norm` of the slice. The guard checks `v.shape` and `abs(norm - 1)`.

</details>

<details><summary><b>Solution</b></summary>

```python
# --------------------------------------------------------------------------
# Q4. Debug: "ValueError: shapes (768,) and (3072,) not aligned" and "scores too small"
# --------------------------------------------------------------------------
class VectorIndex:
    """Tiny guard rail: remember the model, dim and normalisation of what you stored."""
    def __init__(self, dim: int, model: str = "gemini-embedding-001"):
        self.dim, self.model = dim, model
        self.rows: list[np.ndarray] = []

    def add(self, v: np.ndarray):
        if v.shape != (self.dim,):
            raise ValueError(f"index expects dim={self.dim} ({self.model}), got {v.shape}")
        if abs(float(np.linalg.norm(v)) - 1.0) > 1e-3:
            raise ValueError(f"vector is not unit length (norm={np.linalg.norm(v):.3f}); normalise first")
        self.rows.append(v.astype(np.float32))


def q4():
    rng = np.random.default_rng(1)
    stored_768 = unit(rng.normal(size=(5, 768)))                # what was indexed last month
    query_3072 = unit(rng.normal(size=3072))[0]                # what the new code produces, shape (3072,)

    # the reported crash
    try:
        stored_768 @ query_3072
    except ValueError as e:
        print("CRASH reproduced:", type(e).__name__, "-", str(e)[:70])

    # the silent bug: truncating but forgetting to normalise
    sliced = rng.normal(size=3072); sliced /= np.linalg.norm(sliced)
    bad = sliced[:768]
    print("norm of un-normalised slice:", round(float(np.linalg.norm(bad)), 3))

    idx = VectorIndex(dim=768)
    for name, vec in (("good 768 unit", stored_768[0]), ("wrong dim 3072", query_3072), ("slice not normalised", bad)):
        try:
            idx.add(vec)
            print(f"  add({name}): ok")
        except ValueError as e:
            print(f"  add({name}): REFUSED -> {e}")
    return len(idx.rows)
```

</details>

**Why it works / what to learn**
Dimension mismatch is a *loud* bug (a crash), un-normalised slices are a *silent* one (bad scores). Fail fast at insert time: record model, dimension and normalisation next to the vectors, and refuse anything that does not match, like a foreign-key constraint for embeddings.

**Expected output (≈)**
```text
CRASH reproduced: ValueError - matmul: Input operand 1 has a mismatch in its core dimension 0, ...
norm of un-normalised slice: ≈0.5 (random data; real vectors are higher)
  add(good 768 unit): ok
  add(wrong dim 3072): REFUSED -> index expects dim=768 (gemini-embedding-001), got (3072,)
  add(slice not normalised): REFUSED -> vector is not unit length (norm=0.502); normalise first
```

---

### Q5. Production - size 5M chunks and test int8

**Problem**

You must index 5,000,000 chunks. (a) Print RAM for 3072/1536/768 dims in float32, float16 and int8. (b) Implement symmetric int8 scalar quantisation and measure `recall@10` against the float32 ranking on a synthetic corpus. (c) State your recommendation for a 16 GiB machine.

<details><summary><b>Hint</b></summary>

`scale = np.abs(M).max()/127`, `Mq = np.round(M/scale).astype(np.int8)`. Approximate scores = `(Q @ Mq.T.astype(np.float32)) * scale`. Recall = overlap of top-10 sets.

</details>

<details><summary><b>Solution</b></summary>

```python
# --------------------------------------------------------------------------
# Q5. Production: size a 5M-chunk index and test int8 quantisation
# --------------------------------------------------------------------------
def quantize_int8(M: np.ndarray):
    """Symmetric scalar quantisation: float32 -> int8 with ONE scale for the whole matrix."""
    scale = np.abs(M).max() / 127.0
    return np.round(M / scale).astype(np.int8), np.float32(scale)


def q5():
    n = 5_000_000
    print("Sizing 5,000,000 chunks (vectors only, GiB):")
    for dim in (3072, 1536, 768):
        print(f"  {dim:>4}d float32 {memory_gib(n, dim, 4):6.1f} | float16 {memory_gib(n, dim, 2):6.1f} | int8 {memory_gib(n, dim, 1):6.1f}")
    print("  (an HNSW graph typically adds tens of percent on top; budget 1.3-2x)")

    # Does int8 quantisation hurt ranking? test on clustered synthetic data
    D, Q = synthetic_corpus(n_docs=2000, n_queries=100, dim=768, seed=3)
    truth = np.argsort(-(Q @ D.T), axis=1)[:, :10]
    Dq, scale = quantize_int8(D)
    approx_scores = (Q @ Dq.T.astype(np.float32)) * scale       # dequantise on the fly
    got = np.argsort(-approx_scores, axis=1)[:, :10]
    recall = float(np.mean([len(set(t) & set(g)) / 10 for t, g in zip(truth, got)]))
    print(f"\nint8 storage: {D.nbytes/1e6:.1f} MB -> {Dq.nbytes/1e6:.1f} MB, recall@10 vs float32 = {recall:.2f}")
    return recall
```

</details>

**Why it works / what to learn**
768-d float32 needs ≈14.3 GiB of raw vectors, which will not fit a 16 GiB machine once the index graph is added; 768-d int8 needs ≈3.6 GiB and fits comfortably, at a small recall cost that re-scoring the top candidates can win back. A sensible plan: 768-d int8 for search + keep float32 vectors on disk for re-scoring or a cross-encoder reranker. Always confirm the recall loss on your own data, not on synthetic numbers.

**Expected output (≈)**
```text
Sizing 5,000,000 chunks (vectors only, GiB):
  3072d float32   57.2 | float16   28.6 | int8   14.3
  1536d float32   28.6 | float16   14.3 | int8    7.2
   768d float32   14.3 | float16    7.2 | int8    3.6
  (an HNSW graph typically adds tens of percent on top; budget 1.3-2x)

int8 storage: 6.1 MB -> 1.5 MB, recall@10 vs float32 = ≈0.9
```
