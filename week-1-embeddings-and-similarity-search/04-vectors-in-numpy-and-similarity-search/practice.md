# Practice - Vectors in numpy & similarity search

Five questions, rising difficulty. Try each one on paper or in your editor **before** opening the hint, and the hint before the solution. Every solution below is the exact code in `solutions.py` (`python solutions.py` runs them all).

### Q1. Memory and compute budget for a big corpus

**Problem**

Your company wants to embed 2,500,000 support-ticket chunks with `gemini-embedding-001` at 768 dimensions and keep them in RAM for brute-force search.
1. How much RAM (MiB and GiB) do the vectors need in float32? In float16? In int8?
2. How many multiply-adds does ONE query cost?
3. Prove your formula on a real numpy array instead of trusting it.

<details><summary><b>Hint</b></summary>

`bytes = N × d × bytes_per_number`. float32 = 4, float16 = 2, int8 = 1. Divide by 1024² for MiB. A query multiplies every stored number once. For the proof, `np.zeros(shape, dtype=...).nbytes`.

</details>

<details><summary><b>Solution</b></summary>

```python
def memory_mb(n_vectors: int, dim: int, bytes_per_number: int = 4) -> float:
    return n_vectors * dim * bytes_per_number / 1024**2


def q1():
    n, d = 2_500_000, 768
    f32 = memory_mb(n, d, 4)
    f16 = memory_mb(n, d, 2)
    i8 = memory_mb(n, d, 1)
    print(f"{n:,} chunks x {d} dims")
    print(f"  float32 : {f32:,.0f} MiB  (~{f32/1024:.2f} GiB)")
    print(f"  float16 : {f16:,.0f} MiB  (~{f16/1024:.2f} GiB)")
    print(f"  int8    : {i8:,.0f} MiB  (~{i8/1024:.2f} GiB)")
    mults = n * d  # one multiply-add per stored number, per query
    print(f"  one brute-force query = {mults:,} multiply-adds (O(N*d))")
    # Sanity check the formula on a real array:
    arr = np.zeros((1000, d), dtype=np.float32)
    assert arr.nbytes == 1000 * d * 4
    print("  check: np.zeros((1000, 768), float32).nbytes =", arr.nbytes)
    return f32
```

`memory_mb` is the formula from notes section 2. The driver prints three storage options and checks the formula against `ndarray.nbytes`. 2.5 M × 768 × 4 B ≈ 7.15 GiB: too big for a small server, which is the practical argument for lower dims, float16/int8, or a vector DB.

</details>

**Why it works / what to learn**

Memory is `N × d × bytes`, nothing more (metadata and index overhead come on top). Shrinking either `d` or bytes-per-number is the cheapest scaling lever, which is exactly what topic 06 and quantisation exploit.

**Expected output** (≈; numbers vary by machine)

```text
2,500,000 chunks x 768 dims
  float32 : 7,324 MiB  (~7.15 GiB)
  float16 : 3,662 MiB  (~3.58 GiB)
  int8    : 1,831 MiB  (~1.79 GiB)
  one brute-force query = 1,920,000,000 multiply-adds (O(N*d))
  check: np.zeros((1000, 768), float32).nbytes = 3072000
```

---

### Q2. Many queries in one matrix multiplication

**Problem**

Write `batch_search(matrix, queries, k)` that answers a whole batch of queries (shape `(Q, d)`) against a `(N, d)` unit-vector matrix with ONE matrix multiplication and returns `(indices, scores)`, each of shape `(Q, k)`, best first per row. Verify it against the one-query-at-a-time `top_k` helper from `gemini_utils`.

<details><summary><b>Hint</b></summary>

`queries @ matrix.T` has shape `(Q, N)`. `argpartition` and `argsort` accept `axis=1`. `np.take_along_axis` is how you gather per-row indices.

</details>

<details><summary><b>Solution</b></summary>

```python
def make_unit_vectors(n: int, d: int, seed: int = 0) -> np.ndarray:
    """n random unit vectors, float32. Stand-in for real embeddings."""
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((n, d)).astype(np.float32)
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    return m


def batch_search(matrix: np.ndarray, queries: np.ndarray, k: int):
    """matrix (N,d) unit rows, queries (Q,d) unit rows -> (idx (Q,k), scores (Q,k))."""
    scores = queries @ matrix.T                      # (Q, N): one BLAS call
    k = min(k, scores.shape[1])
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]          # (Q, k) unsorted
    part_scores = np.take_along_axis(scores, part, axis=1)          # (Q, k)
    order = np.argsort(-part_scores, axis=1)                        # sort only k per row
    idx = np.take_along_axis(part, order, axis=1)
    return idx, np.take_along_axis(part_scores, order, axis=1)


def q2():
    matrix = make_unit_vectors(20_000, 256, seed=1)
    queries = make_unit_vectors(3, 256, seed=2)
    idx, sc = batch_search(matrix, queries, k=5)
    for qi in range(len(queries)):
        truth = top_k(queries[qi], matrix, k=5)          # helper from gemini_utils
        assert [i for i, _ in truth] == idx[qi].tolist(), "mismatch vs brute force"
        print(f"query {qi}: top ids {idx[qi].tolist()}  best score {sc[qi, 0]:.3f}")
    print("batch result == one-at-a-time result  OK")
    return idx
```

Scores are `(Q, N)`. `argpartition(-scores, k-1, axis=1)[:, :k]` gets the k winners per row in O(N) each; we then sort only those k per row. `take_along_axis` is the row-wise equivalent of `scores[idx]`. The assert compares every row with the helper `top_k` (a full sort), so equality proves the partition trick is correct.

</details>

**Why it works / what to learn**

Batching reuses each loaded block of the matrix for many queries, so throughput per query improves. The `-scores` and `k-1` details from the notes generalise to 2-D with `axis=1`.

**Expected output** (≈; numbers vary by machine)

```text
query 0: top ids [12925, 8561, 16186, 18294, 1162]  best score 0.255
query 1: top ids [13995, 18939, 10708, 19359, 3408]  best score 0.260
query 2: top ids [5277, 15263, 5688, 13874, 15828]  best score 0.259
batch result == one-at-a-time result  OK
```

---

### Q3. Add metadata filtering and deletes to a store

**Problem**

Extend the idea of `MiniVectorStore` into `FilterableStore` with:
- `add(vectors, ids, metadata)`
- `delete(ids)` that does **not** shift rows (use a boolean "alive" mask, i.e. tombstones)
- `search(query, k, where=callable)` that applies the filter **before** picking the top-k, so you never get fewer than k results when k matching rows exist.
Test with 6 documents tagged `php` / `python`.

<details><summary><b>Hint</b></summary>

Compute all scores, then set the score of excluded rows to `-np.inf` so they can never win. Build the filter mask by calling `where(meta)` for each row. Cap `k` at the number of rows still allowed.

</details>

<details><summary><b>Solution</b></summary>

```python
def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest scores, best first. O(N) select + O(k log k) sort."""
    k = min(k, scores.shape[-1])
    if k == scores.shape[-1]:
        return np.argsort(-scores)
    part = np.argpartition(-scores, k - 1)[:k]
    return part[np.argsort(-scores[part])]


class FilterableStore:
    """Tiny store: add / search(where=...) / delete(ids). Deletes are 'tombstones'."""

    def __init__(self, dim: int):
        self.dim = dim
        self.matrix = np.empty((0, dim), dtype=np.float32)
        self.ids: list[str] = []
        self.meta: list[dict] = []
        self.alive = np.empty((0,), dtype=bool)

    def add(self, vectors, ids, metadata):
        v = np.asarray(vectors, dtype=np.float32)
        if v.ndim == 1:
            v = v[None, :]
        v = v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)
        self.matrix = np.vstack([self.matrix, v])
        self.ids.extend(ids)
        self.meta.extend(metadata)
        self.alive = np.concatenate([self.alive, np.ones(len(v), dtype=bool)])

    def delete(self, ids) -> int:
        wanted = set(ids)
        n = 0
        for row, _id in enumerate(self.ids):
            if _id in wanted and self.alive[row]:
                self.alive[row] = False
                n += 1
        return n

    def search(self, query, k=3, where=None):
        q = np.asarray(query, dtype=np.float32)
        q = q / max(float(np.linalg.norm(q)), 1e-12)
        scores = self.matrix @ q
        mask = self.alive.copy()
        if where is not None:
            mask &= np.fromiter((where(m) for m in self.meta), dtype=bool, count=len(self.meta))
        scores = np.where(mask, scores, -np.inf)          # excluded rows can never win
        take = min(k, int(mask.sum()))
        if take == 0:
            return []
        order = top_k_indices(scores, take)
        return [
            {"id": self.ids[i], "score": float(scores[i]), "meta": self.meta[i]}
            for i in order
        ]


def q3():
    d = 64
    vecs = make_unit_vectors(6, d, seed=3)
    store = FilterableStore(d)
    langs = ["php", "php", "python", "python", "php", "python"]
    store.add(vecs, [f"doc{i}" for i in range(6)], [{"lang": l} for l in langs])

    query = vecs[2] + 0.01 * make_unit_vectors(1, d, seed=9)[0]   # nearly doc2 (python)
    print("no filter      :", [r["id"] for r in store.search(query, k=3)])
    print("lang == php    :", [r["id"] for r in store.search(query, k=3, where=lambda m: m["lang"] == "php")])
    print("deleted doc2   :", store.delete(["doc2"]))
    print("after delete   :", [r["id"] for r in store.search(query, k=3)])
    assert store.search(query, k=1)[0]["id"] != "doc2"
    assert all(r["meta"]["lang"] == "php" for r in store.search(query, k=5, where=lambda m: m["lang"] == "php"))
    return store
```

Masking with `-inf` is **pre-filtering**: excluded rows are removed from the competition before ranking. The tombstone keeps row i aligned with `ids[i]`/`meta[i]` (the key invariant). Real databases compact tombstones later in a background job, like `VACUUM` / `OPTIMIZE TABLE`.

</details>

**Why it works / what to learn**

Post-filtering (top-k first, filter after) can return fewer than k hits; pre-filtering cannot. Deleting by list surgery on only one structure silently desynchronises ids and vectors; tombstones avoid that.

**Expected output** (≈; numbers vary by machine)

```text
no filter      : ['doc2', 'doc1', 'doc5']
lang == php    : ['doc1', 'doc0', 'doc4']
deleted doc2   : 1
after delete   : ['doc1', 'doc5', 'doc0']
```

---

### Q4. Debug: three bugs in one search function

**Problem**

This looks fine and runs without error, but returns wrong results. Find all three bugs, explain each, and fix them.
```python
def buggy_search(matrix, query, k):
    scores = matrix @ query
    idx = np.argpartition(scores, k)[:k]
    return idx
```
Test data: 1,000 vectors whose lengths vary from 0.2 to 5. The query is the exact direction of row 10, so row 10 must be ranked first.

<details><summary><b>Hint</b></summary>

Ask three questions: (1) are the vectors the same length? (2) which end of the order does `argpartition` put in front, and what is the pivot index for k winners? (3) does `argpartition` promise an order inside the first k?

</details>

<details><summary><b>Solution</b></summary>

```python
def buggy_search(matrix, query, k):
    scores = matrix @ query                       # BUG 1: nobody normalised matrix/query
    idx = np.argpartition(scores, k)[:k]          # BUG 2: partitions by SMALLEST, off-by-one pivot
    return idx                                    # BUG 3: result is not sorted best-first


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest scores, best first. O(N) select + O(k log k) sort."""
    k = min(k, scores.shape[-1])
    if k == scores.shape[-1]:
        return np.argsort(-scores)
    part = np.argpartition(-scores, k - 1)[:k]
    return part[np.argsort(-scores[part])]


def fixed_search(matrix, query, k):
    matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    query = query / np.linalg.norm(query)
    scores = matrix @ query
    return top_k_indices(scores, k)


def q4():
    rng = np.random.default_rng(4)
    matrix = rng.standard_normal((1000, 32)).astype(np.float32)
    matrix *= rng.uniform(0.2, 5.0, size=(1000, 1)).astype(np.float32)   # wildly different lengths
    query = matrix[10] / np.linalg.norm(matrix[10])                       # exact direction of row 10
    bad = buggy_search(matrix, query, 3)
    good = fixed_search(matrix, query, 3)
    print("buggy :", bad.tolist(), "(row 10 present?", 10 in bad.tolist(), ")")
    print("fixed :", good.tolist(), "(best first, row 10 first?", good[0] == 10, ")")
    assert good[0] == 10
    # Bug 2 in isolation: argpartition(scores, k) puts the k SMALLEST first
    s = np.array([0.9, 0.1, 0.5, 0.7, 0.3])
    print("argpartition(s, 2)[:2]  ->", np.argpartition(s, 2)[:2], "(low scores!)")
    print("argpartition(-s, 1)[:2] ->", np.argpartition(-s, 1)[:2], "(the two highest, unsorted)")
    return good
```

1. **Not normalised**: a long row wins the dot product just by being long, so row 10 (unit direction match) loses to long vectors. Normalise matrix and query.
2. **Wrong end, wrong pivot**: `argpartition` is ascending, so `scores` puts the k *smallest* first. Negate, and use pivot `k-1`.
3. **No ordering**: the k winners come back in arbitrary order; sort those k.

</details>

**Why it works / what to learn**

All three bugs are silent: no exception, plausible-looking output. The fix is the exact recipe `top_k_indices`: select O(N), sort only k. Test search code with a planted answer (a query equal to a stored row must return that row first).

**Expected output** (≈; numbers vary by machine)

```text
buggy : [342, 250, 794] (row 10 present? False )
fixed : [10, 65, 84] (best first, row 10 first? True )
argpartition(s, 2)[:2]  -> [1 4] (low scores!)
argpartition(-s, 1)[:2] -> [0 3] (the two highest, unsorted)
```

---

### Q5. Search a matrix bigger than RAM with mmap and blocks

**Problem**

You have `vectors.npy` that you cannot (or do not want to) load fully. Write `blockwise_search(npy_path, query, k, block_rows)` that memory-maps the file, scores one block of rows at a time, and keeps only a running list of the best k. Prove the result equals the in-RAM answer and report how small each block is compared with the full matrix.

<details><summary><b>Hint</b></summary>

`np.load(path, mmap_mode='r')`. For each block compute `block @ query`, take its local top-k, add `start` to turn local row numbers into global ones, concatenate with the running winners, and trim back to k.

</details>

<details><summary><b>Solution</b></summary>

```python
def make_unit_vectors(n: int, d: int, seed: int = 0) -> np.ndarray:
    """n random unit vectors, float32. Stand-in for real embeddings."""
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((n, d)).astype(np.float32)
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    return m


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest scores, best first. O(N) select + O(k log k) sort."""
    k = min(k, scores.shape[-1])
    if k == scores.shape[-1]:
        return np.argsort(-scores)
    part = np.argpartition(-scores, k - 1)[:k]
    return part[np.argsort(-scores[part])]


def blockwise_search(npy_path, query, k=5, block_rows=5_000):
    """
    Memory-map the matrix (OS pages rows in on demand), score one block at a time,
    keep only a running list of the best k. Peak extra RAM = one block of scores.
    """
    mat = np.load(npy_path, mmap_mode="r")                 # no data read yet
    best_scores = np.empty(0, dtype=np.float32)
    best_idx = np.empty(0, dtype=np.int64)
    for start in range(0, mat.shape[0], block_rows):
        block = np.asarray(mat[start : start + block_rows])        # reads just this slice
        s = block @ query
        kk = min(k, s.shape[0])
        loc = np.argpartition(-s, kk - 1)[:kk]
        best_scores = np.concatenate([best_scores, s[loc]])
        best_idx = np.concatenate([best_idx, loc + start])
        if best_scores.shape[0] > k:                               # trim the running winners
            keep = np.argpartition(-best_scores, k - 1)[:k]
            best_scores, best_idx = best_scores[keep], best_idx[keep]
    order = np.argsort(-best_scores)
    return best_idx[order], best_scores[order]


def q5():
    n, d, k = 60_000, 128, 5
    mat = make_unit_vectors(n, d, seed=5)
    query = make_unit_vectors(1, d, seed=6)[0]
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "vectors.npy"
        np.save(path, mat)
        t0 = time.perf_counter()
        idx, sc = blockwise_search(path, query, k=k, block_rows=7_000)
        t_block = (time.perf_counter() - t0) * 1000
    full = top_k_indices(mat @ query, k)
    print(f"blockwise ids: {idx.tolist()}  ({t_block:.1f} ms)")
    print(f"in-RAM ids   : {full.tolist()}")
    assert idx.tolist() == full.tolist()
    print(f"matrix is {mat.nbytes/1024**2:.1f} MB; each block only ~{7_000*d*4/1024**2:.1f} MB")
    return idx
```

Memory-mapping defers reading until a slice is touched, so peak extra RAM is one block plus a few k-sized arrays. The merge step is correct because the global top-k must be inside the union of each block's top-k. The assert checks equality with the in-RAM answer.

</details>

**Why it works / what to learn**

This is the 'map-reduce' shape of all exact search: local top-k, then merge. It is also how sharded vector databases combine results from many machines.

**Expected output** (≈; numbers vary by machine)

```text
blockwise ids: [47166, 40630, 11554, 9406, 17633]  (2.5 ms)
in-RAM ids   : [47166, 40630, 11554, 9406, 17633]
matrix is 29.3 MB; each block only ~3.4 MB
```

---
