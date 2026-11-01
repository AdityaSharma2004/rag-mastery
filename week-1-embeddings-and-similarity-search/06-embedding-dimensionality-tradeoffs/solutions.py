"""
Solutions for 06-embedding-dimensionality-tradeoffs/practice.md
Run everything:   python solutions.py
Q1, Q4, Q5 are pure numpy (no API key). Q2 needs the free Gemini key. Q3 uses Gemini when a key
is present and otherwise falls back to a synthetic corpus so the code always runs.
"""
import sys, pathlib, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))
from gemini_utils import *   # embed_texts, cosine_sim, SAMPLE_DOCS, np ...


def unit(M: np.ndarray) -> np.ndarray:
    M = np.atleast_2d(M).astype(np.float32)
    return M / np.clip(np.linalg.norm(M, axis=1, keepdims=True), 1e-12, None)


# region q1
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
# endregion q1


# region q2
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
# endregion q2


# region q3
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
# endregion q3


# region q4
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
# endregion q4


# region q5
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
# endregion q5


if __name__ == "__main__":
    for fn in (q1, q2, q3, q4, q5):
        print(f"\n===== {fn.__name__} =====")
        fn()
