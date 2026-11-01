"""
Solutions for practice.md - Topic 04: Vectors in numpy & similarity search.

Run everything:   python solutions.py
Run one question: python -c "import solutions; solutions.q3()"

All five solutions are pure numpy (synthetic unit vectors), so they run offline
and cost 0 API calls. The helper import is kept so you can swap in real
Gemini embeddings (embed_texts) whenever you like.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))
from gemini_utils import *   # noqa: F401,F403  (cosine_sim, top_k, embed_texts, ...)

import json
import tempfile
import time

import numpy as np


# --------------------------------------------------------------------------
# Shared helper: fake-but-realistic data (random unit vectors)
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# Q1. Memory and compute maths (warm-up)
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# Q2. Batch search: many queries in ONE matrix multiply
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# Q3. Build: add metadata filtering + delete on top of a store
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# Q4. Debug: three bugs in one innocent-looking snippet
# --------------------------------------------------------------------------
def buggy_search(matrix, query, k):
    scores = matrix @ query                       # BUG 1: nobody normalised matrix/query
    idx = np.argpartition(scores, k)[:k]          # BUG 2: partitions by SMALLEST, off-by-one pivot
    return idx                                    # BUG 3: result is not sorted best-first


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


# --------------------------------------------------------------------------
# Q5. Production: search a .npy bigger than RAM, in blocks, with mmap
# --------------------------------------------------------------------------
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


if __name__ == "__main__":
    for fn in (q1, q2, q3, q4, q5):
        print(f"\n===== {fn.__name__} =====")
        fn()
