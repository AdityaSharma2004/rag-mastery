# 04 - Vectors in numpy & similarity search

> Syllabus line: *Storing vectors in numpy, doing basic similarity search.*
> By the end you will have written your own tiny vector database (about 60 lines) and you will know exactly what
> Chroma, Qdrant and pgvector are doing under the hood.

## 🎯 What you will learn

- You will be able to **store thousands of embeddings in one numpy matrix** and explain what each row, column and byte means.
- You will be able to **normalise vectors** and explain why that turns cosine similarity into a plain dot product.
- You will be able to **search with one matrix multiplication** and pick the top-k results with `np.argpartition` (and know when `argsort` is fine).
- You will be able to **save and load** a store using `.npy` / `.npz` plus a metadata JSON file, including memory-mapped loading.
- You will be able to **calculate memory and compute cost** (`N × d × 4 bytes`, `O(N·d)` per query) in your head for any corpus.
- You will be able to **rebuild a reusable `MiniVectorStore` class** from scratch and say precisely when brute force stops being enough (the reason vector databases exist).

## 🧠 The big idea in plain words

In topic 03 you learned that an embedding model turns a sentence into a list of numbers (a **vector**), and that
sentences with similar meaning get vectors that point in similar directions. Fine. But a real app has *thousands or
millions* of chunks. Where do those vectors live, and how do you find the closest ones to a question, fast?

Story: you run a Laravel app with a `documents` table of 50,000 help-centre paragraphs. A user asks "how do I reset my password?".
You embed that question into 768 numbers. Now you want the 5 paragraphs whose 768 numbers are closest to the question's.

The simplest honest approach:

1. Keep **all** paragraph vectors stacked in one big grid (a matrix) in memory: one row per paragraph.
2. Compare the question vector to **every** row. (That is a "brute-force" or "exact" search.)
3. Sort by score and take the top 5.

That is literally it. A "vector database" is this idea plus clever shortcuts (indexes) so step 2 does not have to
touch every row once you have millions of rows.

**Laravel/MySQL mapping (only where it genuinely fits):**

| Our numpy store | The familiar thing |
|---|---|
| `matrix` (N × d) | The table's rows. Each row = one record, each column = one *numeric* feature (no meaning individually). |
| `ids` list + `metadata` list | The non-vector columns: `id`, `title`, `source_url`, `created_at`. |
| `matrix @ query` over all rows | A full table scan (`SELECT ... ORDER BY score` with **no index**). Correct, but `O(N)`. |
| ANN index (later: HNSW) | A B-tree/FULLTEXT index: extra structure that trades exactness/space for speed. |
| `save()` / `load()` | `php artisan` dump / restore. Here it is a `.npy` file plus `meta.json`. |

## 📚 Concepts, step by step

### 1. What a "stored vector" actually is

An embedding is a 1-D numpy array of floats. For `gemini-embedding-001` the native length is **3072**
(you can ask for 768 or 1536 via Matryoshka truncation; our helper defaults to 768). That length is called the **dimension**, written `d`.

Many embeddings stacked on top of each other make a **2-D array** (a matrix) with shape `(N, d)`:

```
                 d = 4 (toy; real = 768)
              col0   col1   col2   col3
row 0 doc A  [ 0.12, -0.40,  0.55,  0.71 ]
row 1 doc B  [ 0.10, -0.38,  0.60,  0.69 ]     <- similar to A: numbers look alike
row 2 doc C  [-0.62,  0.30, -0.11,  0.72 ]     <- different topic
        shape = (3, 4)  ->  N = 3 vectors, d = 4
```

Important: **a column has no human meaning.** Column 2 is not "is about databases". Meaning lives in the *whole row*, in its direction.

**Order matters.** Row `i` of the matrix must always line up with `ids[i]` and `metadata[i]`. This is the single most
important invariant of your store. (Think: `array index = primary key position`. If you delete from one list and not the others, everything silently breaks.)

### 2. dtype and memory maths: `N × d × 4 bytes`

numpy's default float is `float64` (8 bytes). Embeddings are stored as `float32` (4 bytes): half the memory and faster
matrix multiplication, with no practical loss in search quality.

```
memory (bytes) = N × d × bytes_per_number
```

Worked examples (float32, 4 bytes; sizes in MiB/GiB, i.e. divide by 1024², what Task Manager and `htop` show):

| Corpus | N | d | Size |
|---|---|---|---|
| Small blog | 10,000 | 768 | 10,000 × 768 × 4 = 30,720,000 B ≈ **29 MB** |
| Help-centre | 100,000 | 768 | ≈ **293 MB** |
| Big docs site | 1,000,000 | 768 | ≈ **2.86 GiB** |
| Same, native dims | 1,000,000 | 3072 | ≈ **11.4 GiB** |
| Same, but accidentally float64 | 1,000,000 | 768 | ≈ **5.7 GiB** (double!) |

Quick mental shortcut: `1M × 768 × 4 B = 3.07 GB (decimal)`, `≈ 2.86 GiB`. Remember "**about 3 GB per million 768-dim vectors**".

Two things *not* in that formula: the metadata (ids, text, JSON: often bigger than the vectors if you keep the chunk text) and, for real vector DBs, the index (HNSW adds
maybe 20-100% on top). Topic 06 digs into cost; here just remember the formula.

### 3. Normalisation: why unit vectors make life easy

The **length** (L2 norm) of a vector is `sqrt(x₁² + x₂² + … + x_d²)`. **Normalising** means dividing every number by that length, so the
new vector has length exactly 1 (a "unit vector").

Cosine similarity compares only the *angle* between two vectors:

```
cos(a, b) = (a · b) / (‖a‖ × ‖b‖)
```

If `‖a‖ = ‖b‖ = 1`, the denominator is 1, so:

```
cos(a, b) = a · b          (just the dot product!)
```

Tiny hand example. `a = [3, 4]`, `b = [4, 3]`:

- `‖a‖ = 5`, `‖b‖ = 5`
- normalised: `a' = [0.6, 0.8]`, `b' = [0.8, 0.6]`
- `a' · b' = 0.6×0.8 + 0.8×0.6 = 0.96` → cosine similarity = 0.96. Same as `(3×4+4×3)/(5×5) = 24/25 = 0.96`. ✔

**Why it matters for search:** do the normalising work **once at insert time**, and every query is then just
`matrix @ query`: one matrix multiplication, no per-row division. A dot product on unit vectors is also a monotone transform of Euclidean distance
(`‖a-b‖² = 2 - 2·a·b`), so cosine, dot-product and L2 all give the **same ranking**. That is why vector DBs let you choose any of them and why people stop worrying about it.

Gemini detail you must remember: only the native **3072** vector comes pre-normalised. Truncated 768/1536 vectors are **not**, so we normalise ourselves
(the course helper `embed_texts` already does; your store should do it too, as a safety net: normalising an already-unit vector is harmless).

### 4. Search = one matrix multiplication

Matrix `M` has shape `(N, d)`. Query vector `q` has shape `(d,)`. Then

```
scores = M @ q          # shape (N,)   scores[i] = dot(M[i], q)
```

Tiny example with `d = 2`, `N = 3`:

```
M = [[1.0, 0.0],        q = [0.8, 0.6]
     [0.6, 0.8],
     [0.0, 1.0]]

scores = [1.0×0.8 + 0.0×0.6,  0.6×0.8 + 0.8×0.6,  0.0×0.8 + 1.0×0.6]
       = [0.80, 0.96, 0.60]    -> best is row 1
```

Why is this fast? `@` calls **BLAS**, a heavily optimised C/Fortran library that uses CPU SIMD instructions and multiple cores. A Python `for` loop over 100k rows would be ~100× slower.
It is the same reason `SUM()` in MySQL beats summing rows in a PHP `foreach`.

You can also search **many queries at once**: `Q @ M.T` gives shape `(num_queries, N)`: one call, all answers (practice Q2).

### 5. Picking the top-k: `argsort` vs `argpartition`

After scoring you have N numbers and want the indices of the k largest.

- `np.argsort(-scores)[:k]`: sorts everything, `O(N log N)`.
- `np.argpartition(-scores, k-1)[:k]`: only guarantees that the k best are in the first k slots, **in arbitrary order**, `O(N)`. Then sort just those k (`O(k log k)`).

```python
part = np.argpartition(-scores, k - 1)[:k]     # k best, unordered
top  = part[np.argsort(-scores[part])]         # now ordered best-first
```

Why `-scores`? Both functions work in *ascending* order (smallest first). Negating turns "largest" into "smallest". Why `k - 1`? `argpartition(a, kth)` puts the element
that would be at sorted position `kth` in place; everything before it is ≤ it. We want positions `0..k-1` to be the winners, so the pivot position is `k-1`.

Honest note (we measure it below): at 100k-1M rows, the *scoring* (`M @ q`) is the expensive part because it reads the whole matrix from RAM; `argsort` of 200k floats is only a few milliseconds. `argpartition`
is the right habit and wins big at millions of rows or with many queries, but do not expect miracles at small N.

Ties: equal scores can come out in any order. Fine for search.

### 6. Complexity: `O(N·d)` per query

Each query does `N × d` multiply-adds (every number in the matrix is touched once). So:

| N | d | multiply-adds per query | rough time (one modern CPU, memory-bound) |
|---|---|---|---|
| 10,000 | 768 | 7.7 M | ≈ 1-3 ms |
| 200,000 | 768 | 154 M | ≈ 30-60 ms (we measure ≈ 47 ms below) |
| 1,000,000 | 768 | 768 M | ≈ 0.2-0.4 s |
| 10,000,000 | 768 | 7.7 B | seconds + 29 GB RAM: **too slow / too big** |

Time grows linearly with N and with d. Two levers exist: lower `d` (topic 06) or avoid touching every row (ANN, next section).

### 7. Brute force vs ANN: preview of why vector DBs exist

- **Exact / brute-force / "flat" search**: compare against everything. 100% correct. `O(N·d)`.
- **ANN (Approximate Nearest Neighbour)**: build an index up front (a graph such as **HNSW**, or clusters such as **IVF**) so a query only visits a tiny fraction of the rows,
  e.g. a few thousand out of 10 million. It may occasionally miss a true neighbour; quality is measured by **recall@k** ("of the true top-k, how many did we return?"). Typical settings reach 95-99% recall.

Rule of thumb you can quote:

| Rows (N) | Reasonable choice |
|---|---|
| up to ~100k (768-dim) | Brute-force numpy is perfectly fine (tens of ms). Do not over-engineer. |
| ~100k - 1M | Brute force still works if latency budget allows (hundreds of ms); an index becomes attractive. |
| > ~1M, or high QPS, or need filtering + persistence + concurrent writes | Use a vector DB (week 2): Chroma / Qdrant / pgvector. |

What a real vector DB adds beyond our 60 lines: ANN indexes, **concurrent** writes and crash-safe persistence, metadata **pre-filtering** inside the index, deletes/updates,
sharding, auth, and a network API. Building the mini store first means none of that will feel like magic later.

### 8. Persisting: `.npy`, `.npz`, and metadata JSON

- `np.save("vectors.npy", matrix)` / `np.load(...)`: one array, raw bytes plus a tiny header. Fastest, simplest. Supports **memory-mapping**.
- `np.savez("store.npz", vectors=matrix, other=...)`: several arrays in one zip container; `np.savez_compressed` also compresses. Float embeddings compress poorly (~5-10%), so compression is mostly wasted CPU; use plain `.npy` for vectors.
- **Metadata does not belong in numpy.** Ids, text, URLs, page numbers: save them as JSON (or JSONL, or a SQLite/MySQL table). Pickle is risky; never `np.load(..., allow_pickle=True)` on files you did not create.

Our on-disk layout:

```
my_store/
├── vectors.npy      # (N, d) float32, rows already normalised
└── meta.json        # {"dim": 768, "count": N, "ids": [...], "metadata": [...], ...}
```

**Memory-mapping** (`np.load(path, mmap_mode="r")`) does not read the file into RAM; the OS maps it into your address space and loads pages from disk when you touch them. You can open a
10 GB file on an 8 GB laptop and search it (slowly on first touch, fast once the OS cache is warm). Think of it as a lazy-loaded Eloquent cursor for arrays.

**Safety:** save the `dim` and `count` in `meta.json` and check them on load; if `vectors.npy` has 1000 rows and `meta.json` lists 999 ids, a half-finished save corrupted your store, so fail loudly.
Our `save()` writes temp files and `os.replace()`s them into place, so a crash cannot leave a half-written file (the two files can still disagree if the crash lands between the two replaces, which is why `load()` re-checks).

## 💻 Code walkthrough

Install once: `pip install -r requirements.txt` and put `GEMINI_API_KEY` in `.env`. Stages 1-5 are pure numpy and need **no** key.

### Stage 1: the matrix and its memory footprint

```python
import numpy as np

rng = np.random.default_rng(0)
N, d = 5, 8
M = rng.standard_normal((N, d)).astype(np.float32)   # stand-in for 5 embeddings
print(M.shape, M.dtype, M.nbytes)                    # nbytes = N*d*4
```

Expected: `(5, 8) float32 160` (5 × 8 × 4 = 160).

Line by line: `default_rng(0)` is a seeded random generator (reproducible); `.astype(np.float32)` forces 4-byte floats, because `standard_normal` returns float64 by default. `.nbytes` proves the formula.

### Stage 2: normalise

```python
def normalise(m: np.ndarray) -> np.ndarray:
    m = np.atleast_2d(np.asarray(m, dtype=np.float32))
    norms = np.linalg.norm(m, axis=1, keepdims=True)   # shape (N, 1): one length per row
    norms[norms == 0] = 1.0                            # avoid 0/0 for an all-zero vector
    return m / norms                                   # broadcasting divides each row

U = normalise(M)
print(np.linalg.norm(U, axis=1))                       # ≈ [1. 1. 1. 1. 1.]
```

`axis=1` means "collapse columns, keep one value per row". `keepdims=True` keeps the shape `(N,1)` so the division broadcasts row-wise. Without it you would get shape `(N,)` and broadcasting would try to divide the wrong axis (or error).

### Stage 3: search by matrix multiplication

```python
query = normalise(M[2] + 0.05 * rng.standard_normal(d))[0]   # "almost row 2"
scores = U @ query                                           # shape (5,)
print(np.round(scores, 3))
print("best row:", int(np.argmax(scores)))                   # 2
```

Row 2 scores near 1.0 (the query is a slightly noisy copy of it); the others are random-looking values around 0 ± 0.4 at d=8 (at d=768 they would cluster tightly near 0).

### Stage 4: top-k and a benchmark you can run

```python
import time

rng = np.random.default_rng(42)
N, d, k = 200_000, 768, 5
M = normalise(rng.standard_normal((N, d)))
q = M[123]                                    # use a stored row as the "query" -> must come back as #1

def ms(fn, reps=5):
    fn()                                      # warm-up
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t) / reps * 1000

scores = M @ q
def partition_topk():
    part = np.argpartition(-scores, k - 1)[:k]
    return part[np.argsort(-scores[part])]

print(f"matrix: {M.nbytes/1024**2:.0f} MB")
print(f"score  (M @ q)     : {ms(lambda: M @ q):7.2f} ms")
print(f"argsort top-k      : {ms(lambda: np.argsort(-scores)[:k]):7.2f} ms")
print(f"argpartition top-k : {ms(partition_topk):7.2f} ms")
print("top-1 is the query row itself:", partition_topk()[0] == 123)
```

Expected shape of the output (numbers vary by machine; ratios are the lesson, and these are ≈):

```
matrix: 586 MB
score  (M @ q)     :   47.21 ms
argsort top-k      :    3.41 ms
argpartition top-k :    0.61 ms
top-1 is the query row itself: True
```

Takeaways: (1) scoring dominates because it must stream 586 MB from RAM (it is memory-bandwidth bound); (2) `argpartition` is ~5× faster than a full sort here; (3) the same matrix as float64 would take twice the RAM and roughly twice the time.

### Stage 5: save and load with metadata

```python
import json
from pathlib import Path

folder = Path("demo_store")
folder.mkdir(exist_ok=True)
np.save(folder / "vectors.npy", U)
(folder / "meta.json").write_text(json.dumps({
    "dim": int(U.shape[1]), "count": int(U.shape[0]), "dtype": "float32",
    "ids": [f"doc{i}" for i in range(len(U))],
    "metadata": [{"source": "demo", "page": i} for i in range(len(U))],
}, indent=2))

U2 = np.load(folder / "vectors.npy", mmap_mode="r")   # lazy: nothing read yet
print(type(U2).__name__, U2.shape)                    # memmap (5, 8)
print(np.allclose(U, U2))                             # True
```

Bonus `.npz` (several arrays, one file): `np.savez_compressed("bundle.npz", vectors=U, row_ids=np.arange(len(U)))`, then `data = np.load("bundle.npz"); data["vectors"]`. (`.npz` loads lazily per key but cannot be memory-mapped.)

### Stage 6: the reusable `MiniVectorStore` class

Copy this into `mini_vector_store.py`; it is the thing you will reuse in topics 05/06 and the Week 2 comparisons.

```python
from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np


class MiniVectorStore:
    """A brute-force vector store: a (N, d) float32 matrix + ids + metadata."""

    def __init__(self, dim: int):
        self.dim = dim
        self.matrix = np.empty((0, dim), dtype=np.float32)   # unit-length rows
        self.ids: list[str] = []
        self.metadata: list[dict] = []

    def __len__(self) -> int:
        return self.matrix.shape[0]

    @staticmethod
    def _unit(v: np.ndarray) -> np.ndarray:
        v = np.atleast_2d(np.asarray(v, dtype=np.float32))
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return v / norms

    def add(self, vectors, ids=None, metadata=None) -> None:
        v = self._unit(vectors)
        if v.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {v.shape[1]}")
        n = v.shape[0]
        ids = list(ids) if ids is not None else [str(len(self) + i) for i in range(n)]
        metadata = list(metadata) if metadata is not None else [{} for _ in range(n)]
        if not (len(ids) == len(metadata) == n):
            raise ValueError("vectors, ids and metadata must have the same length")
        if set(ids) & set(self.ids) or len(set(ids)) != n:
            raise ValueError("duplicate ids")
        self.matrix = np.vstack([self.matrix, v])
        self.ids.extend(ids)
        self.metadata.extend(metadata)

    def search(self, query, k: int = 5) -> list[dict]:
        if len(self) == 0:
            return []
        q = self._unit(query)[0]
        scores = self.matrix @ q
        k = min(k, len(self))
        if k < len(self):
            part = np.argpartition(-scores, k - 1)[:k]
        else:
            part = np.arange(len(self))
        top = part[np.argsort(-scores[part])]
        return [
            {"rank": r + 1, "id": self.ids[i], "score": float(scores[i]),
             "metadata": self.metadata[i]}
            for r, i in enumerate(top)
        ]

    def save(self, folder: str | Path) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        info = {"dim": self.dim, "count": len(self), "dtype": "float32",
                "normalised": True, "ids": self.ids, "metadata": self.metadata}
        # write to temp names first, then atomically swap in (a crash never leaves a half file)
        np.save(folder / "vectors.tmp.npy", self.matrix)
        (folder / "meta.tmp.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
        os.replace(folder / "vectors.tmp.npy", folder / "vectors.npy")
        os.replace(folder / "meta.tmp.json", folder / "meta.json")

    @classmethod
    def load(cls, folder: str | Path, mmap: bool = False) -> "MiniVectorStore":
        folder = Path(folder)
        info = json.loads((folder / "meta.json").read_text())
        matrix = np.load(folder / "vectors.npy", mmap_mode="r" if mmap else None)
        if matrix.shape != (info["count"], info["dim"]) or len(info["ids"]) != info["count"]:
            raise ValueError("vectors.npy and meta.json disagree - partial save?")
        store = cls(info["dim"])
        store.matrix, store.ids, store.metadata = matrix, info["ids"], info["metadata"]
        return store
```

Design decisions, and why:

- **Normalise inside `add` and `search`**: callers cannot forget. It is idempotent on unit vectors.
- **`dim` check**: mixing a 768 vector with a 1536 store is the #1 real-world bug; fail at the door with a clear message.
- **Duplicate-id check**: silently allowing two rows with the same id makes later deletes and updates ambiguous.
- **`np.vstack` on every `add`** copies the whole matrix: `O(N)` per call. Fine for batches; for streaming inserts collect chunks in a list and stack once (a real DB uses segments / append-only logs).
- **`mmap=True`** returns a read-only memmap. Calling `add` after a mmap load would copy it into RAM through `vstack`; treat mmap stores as read-only.

### Stage 7: use it with real Gemini embeddings

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parents[1] / "common"))   # adjust to wherever common/ is
from gemini_utils import embed_texts, embed_query, SAMPLE_DOCS
from mini_vector_store import MiniVectorStore

vecs = embed_texts(SAMPLE_DOCS, task_type="RETRIEVAL_DOCUMENT", dim=768)   # (10, 768), cached on disk
store = MiniVectorStore(dim=768)
store.add(vecs, ids=[f"sample-{i}" for i in range(len(SAMPLE_DOCS))],
          metadata=[{"text": t} for t in SAMPLE_DOCS])
store.save("sample_store")

loaded = MiniVectorStore.load("sample_store")
q = embed_query("Which database stores data in memory for caching?", dim=768)   # RETRIEVAL_QUERY
for hit in loaded.search(q, k=3):
    print(f'{hit["rank"]}. {hit["score"]:.3f}  {hit["metadata"]["text"]}')
```

Expected (≈; exact scores differ, order should look like this):

```
1. 0.7x  Redis is an in-memory key-value store often used for caching and queues.
2. 0.6x  AWS RDS is a managed relational database service for MySQL and PostgreSQL.
3. 0.6x  MySQL FULLTEXT indexes allow fast keyword search on text columns.
```

Note the two different task types: documents with `RETRIEVAL_DOCUMENT`, the question with `RETRIEVAL_QUERY` (embed_query does this for you). Embeddings are cached in `.embed_cache`, so re-running costs zero API calls, and with only 10 docs you use a single batched request. Free tier has rate limits: always batch and cache.

## ⚠️ Common mistakes & debugging

| Symptom | Cause | Fix |
|---|---|---|
| Top results look random or the best hit is not the obvious one | Vectors not normalised, or only the query was normalised; long vectors dominate dot products | Normalise at insert **and** query time (our `_unit`) |
| Truncated (768) results are slightly worse than expected | Forgot to re-normalise after truncating a 3072 vector | Normalise after every truncation |
| `ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0 ...` (with `np.dot`: `shapes (N,768) and (3072,) not aligned`) | Query embedded at a different `dim` than the stored matrix | Keep `dim` in one config constant; store it in `meta.json` and assert on load |
| Memory is double what you computed | float64 arrays (numpy default) | `.astype(np.float32)` as soon as data arrives |
| `argpartition` returns the *worst* items | Forgot the minus sign (ascending order) | `np.argpartition(-scores, k-1)[:k]` |
| Top-k list not sorted best-first | `argpartition` guarantees membership, not order | Sort the k winners with `argsort` |
| Searches slow down with every `add` | Calling `np.vstack` per vector (O(N) copy each time) | Batch your adds, or buffer then stack once |
| `IndexError` / wrong text shown for a hit | `ids`/`metadata` drifted out of sync with matrix rows after a manual delete | Always mutate all three together; never delete from one list only |
| `meta.json` says 1000 items, `.npy` has 999 | Crash mid-save, or the files come from different runs | Verify counts on load (we do); write to temp file then `os.replace` |
| `np.load` of a `.npz` prints a pickle warning/error | Object arrays inside; `allow_pickle` required | Do not store Python objects in numpy files; metadata goes to JSON |
| Every score is nearly identical (e.g. all ≈ 0.99, or all ≈ 0) | Random/placeholder vectors, an all-duplicate corpus, or zero vectors from a failed embed call | Print `matrix.std(axis=0).mean()` and a few raw texts; check for duplicates; assert no zero rows |

## 🚀 Pro-level insights (what the top 2% know)

- **Flat search is a real baseline, not a toy.** Every ANN index is judged against exact search; the ground truth for "recall@k" *is* the brute-force result. Keep a brute-force path in your test-suite so you can measure what your index loses.
- **Scoring is memory-bandwidth bound.** Time is dominated by streaming the matrix from RAM, not by arithmetic. So halving bytes (float16, int8, lower `d`) speeds things up almost proportionally. That is why quantisation and Matryoshka truncation (topic 06) matter.
- **Float16 storage, float32 compute** is a common trick: store at 2 bytes, cast blocks to float32 for the multiply. Quality drop is negligible for retrieval.
- **Block your search for big matrices**: process 10k-50k rows at a time and merge top-k lists (practice Q5). Peak memory = one block of scores, not N.
- **Pre-filtering vs post-filtering.** "Only docs of tenant 7" can be applied *before* scoring (mask the matrix) or *after* (take top-k, then drop). Post-filtering can leave you with fewer than k results; always over-fetch (`k × 5`) or pre-filter. Real DBs implement this inside the index and it is a headline difference between them.
- **Matrix multiply gives you batch for free.** 32 simultaneous users = a `(32, d)` query matrix; throughput per query often improves 3-10× vs one-at-a-time because BLAS reuses the loaded matrix. Web servers can micro-batch concurrent requests.
- **Threads:** numpy releases the GIL inside BLAS, so a thread pool serving searches is effective. Memory-mapped stores can be shared across worker processes (each `gunicorn`/`uvicorn` worker maps the same file: the OS caches it once).
- **When NOT to use this:** >1-5M vectors, frequent updates/deletes, multi-writer concurrency, rich filtering, or strict latency (<20 ms) at scale. Then go to a vector DB.
- **When it is the *best* choice:** a few thousand to ~200k chunks, a single process, read-mostly. No server to run, no network hop, trivially testable, deployable as one file. Many production RAG apps live here forever.
- **Numbers worth memorising:** `1M × 768 × 4 B ≈ 3 GB`; `1M × 3072 × 4 B ≈ 12 GB`; brute force ≈ `N·d` multiply-adds; HNSW typically ≈ 95-99% recall at 1-10 ms for millions of vectors.
- **Evaluation:** for exact search, results are exact; the thing to evaluate is the *embeddings* (retrieval quality: recall@k / MRR on labelled questions). For ANN, evaluate recall@k against your flat store.

## 🧪 Mini lab

**Goal (10-15 min, no API key needed):** find where brute force starts to hurt on *your* laptop.

1. Paste `MiniVectorStore` into `mini_vector_store.py`.
2. Create a script that, for `N` in `[10_000, 50_000, 200_000, 500_000]` with `d = 768`, adds `N` random vectors (generate with `rng.standard_normal((N, 768)).astype(np.float32)`) and times `store.search(random_query, k=10)` averaged over 20 runs.
3. Print a table `N | MB | ms per query`. Plot or just eyeball: is time roughly linear in N?
4. Repeat for `d = 256`. Compare the ms and MB; does time scale with `d` as the theory says?
5. Stretch: save the 500k store, load it with `mmap=True`, and time the **first** search versus the **second**. The first is slower; explain why (page cache).

Success looks like: a roughly straight line in N, ~3× faster at d=256 than d=768, and a clear first-vs-second search gap under mmap.

## ✅ Recap checklist

- [ ] I can draw an `(N, d)` matrix and say what a row and a column mean (and that a column has no meaning by itself).
- [ ] I can compute memory as `N × d × bytes` for float32/float16/int8 without a calculator.
- [ ] I can explain why unit vectors turn cosine similarity into a dot product, and why cosine/dot/L2 rank identically on them.
- [ ] I know why truncated Gemini vectors need re-normalising and the 3072 one does not.
- [ ] I can write `matrix @ query` search and the `argpartition` + sort top-k from memory, including the `-scores` and `k-1` details.
- [ ] I can state the complexity `O(N·d)` per query and why it is memory-bound.
- [ ] I can save/load vectors with `.npy` plus metadata JSON, and explain mmap and why metadata stays out of numpy.
- [ ] I can say, with numbers, when brute force is enough and when I need ANN / a vector DB.
- [ ] I have rebuilt `MiniVectorStore` (add / search / save / load) without copying.

## 🎤 Interview questions

**1. Why normalise embeddings before storing them?**
So cosine similarity equals the dot product. You pay the division once at insert time and every query becomes a single matrix multiplication. It also makes cosine, dot product and Euclidean distance produce the same ranking.

**2. A million 768-dim embeddings: how much RAM, and what changes at 3072 dims?**
`1,000,000 × 768 × 4 B ≈ 3 GB` in float32; at 3072 dims it is four times that, ≈ 12 GB. Plus metadata and, in a real DB, index overhead (HNSW graph).

**3. What is the time complexity of brute-force search and why does a vector DB beat it at scale?**
`O(N·d)` per query: every stored number is touched. Vector DBs build an ANN index (HNSW/IVF) so each query visits a small fraction of rows, giving sub-linear time at the cost of approximate results (recall < 100%) and extra memory and index-build time.

**4. `argsort` vs `argpartition` for top-k?**
`argsort` is `O(N log N)` and sorts everything; `argpartition` is `O(N)` and only guarantees the k best are in the first k slots, unordered, so you sort those k afterwards. It matters more as N grows or when you serve many queries.

**5. Where would you keep metadata, and what goes wrong if row order drifts?**
Outside the numpy matrix (JSON, SQLite, MySQL), aligned by row index or, better, by a stable id. If lists drift out of sync (e.g. a delete applied to one list), the app returns the wrong text for a hit with no error, which is a silent data-corruption bug. Fix with an invariant check on load and mutating all structures together.

## 🔗 Next

Topic 05 uses this store to compare semantic search against keyword search and a hybrid of both (like `FULLTEXT` vs `LIKE` in MySQL), which tells you *when* vector search alone is not enough.
