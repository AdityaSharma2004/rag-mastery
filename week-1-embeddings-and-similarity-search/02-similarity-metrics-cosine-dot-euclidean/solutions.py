"""
Solutions for 02-similarity-metrics-cosine-dot-euclidean/practice.md
All five solutions are pure numpy: they run offline with NO API key.
Run everything:   python solutions.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))
from gemini_utils import *   # cosine_sim, top_k, np ...

import numpy as np


# region q1
# --------------------------------------------------------------------------
# Q1. Warm-up: compute all three metrics by hand-sized numbers
# --------------------------------------------------------------------------
def q1():
    a = np.array([3.0, 4.0])
    b = np.array([4.0, 3.0])

    dot = float(a @ b)                              # 3*4 + 4*3 = 24
    len_a, len_b = np.linalg.norm(a), np.linalg.norm(b)   # both 5.0
    cos = dot / (len_a * len_b)                     # 24 / 25 = 0.96
    euclid = float(np.linalg.norm(a - b))           # sqrt(1 + 1) = 1.414

    print(f"dot product      : {dot}")
    print(f"lengths          : {len_a}, {len_b}")
    print(f"cosine similarity: {cos:.2f}")
    print(f"euclidean dist   : {euclid:.3f}")

    # Now double b's length. Direction is unchanged.
    b2 = 2 * b
    print("\nafter doubling b:")
    print(f"dot product      : {float(a @ b2)}   (changed!)")
    print(f"cosine similarity: {float(a @ b2 / (np.linalg.norm(a) * np.linalg.norm(b2))):.2f}   (unchanged)")
    print(f"euclidean dist   : {float(np.linalg.norm(a - b2)):.3f}   (changed!)")
    return cos
# endregion q1


# region q2
# --------------------------------------------------------------------------
# Q2. Apply: prove numerically that on unit vectors  |a-b|^2 = 2 - 2*cos(a,b)
# --------------------------------------------------------------------------
def q2():
    rng = np.random.default_rng(42)
    A = rng.normal(size=(1000, 768))
    B = rng.normal(size=(1000, 768))
    A /= np.linalg.norm(A, axis=1, keepdims=True)     # make them unit vectors
    B /= np.linalg.norm(B, axis=1, keepdims=True)

    cos = np.sum(A * B, axis=1)                        # row-wise dot == cosine here
    sq_l2 = np.sum((A - B) ** 2, axis=1)
    max_err = float(np.abs(sq_l2 - (2 - 2 * cos)).max())
    print(f"max |  |a-b|^2 - (2 - 2cos)  | over 1000 pairs: {max_err:.2e}")
    assert max_err < 1e-5

    # So the three metrics induce the SAME ranking on unit vectors:
    q = A[0]
    by_cos = np.argsort(-(B @ q))
    by_l2 = np.argsort(np.linalg.norm(B - q, axis=1))
    print("same ranking by cosine and by euclidean:", bool((by_cos == by_l2).all()))
    return max_err
# endregion q2


# region q3
# --------------------------------------------------------------------------
# Q3. Build: a metric-comparison harness
# --------------------------------------------------------------------------
def rank_all(query: np.ndarray, matrix: np.ndarray, k: int = 3) -> dict[str, list[int]]:
    """Top-k row indexes under dot, cosine and euclidean (best first)."""
    dot = matrix @ query
    cos = cosine_sim(query, matrix)
    l2 = np.linalg.norm(matrix - query, axis=1)
    return {
        "dot": [int(i) for i in np.argsort(-dot)[:k]],
        "cosine": [int(i) for i in np.argsort(-cos)[:k]],
        "euclid": [int(i) for i in np.argsort(l2)[:k]],
    }


def q3():
    rng = np.random.default_rng(7)
    M = rng.normal(size=(50, 64))
    q = rng.normal(size=64)

    raw = rank_all(q, M)
    print("UN-normalised vectors:")
    for name, idx in raw.items():
        print(f"  {name:<7}: {idx}")

    Mn = M / np.linalg.norm(M, axis=1, keepdims=True)
    qn = q / np.linalg.norm(q)
    unit = rank_all(qn, Mn)
    print("UNIT vectors:")
    for name, idx in unit.items():
        print(f"  {name:<7}: {idx}")

    agree = unit["dot"] == unit["cosine"] == unit["euclid"]
    print("all three agree on unit vectors:", agree)
    return raw, unit
# endregion q3


# region q4
# --------------------------------------------------------------------------
# Q4. Debug: "the longest document always wins my dot-product search"
# --------------------------------------------------------------------------
def q4():
    # Three documents. Doc 0 is the genuinely relevant one (same direction as the query).
    # Doc 2 points somewhere unrelated but its vector is HUGE (imagine an un-normalised
    # model that outputs bigger numbers for longer text).
    query = np.array([1.0, 0.0])
    docs = np.array([
        [0.9, 0.1],    # doc 0: relevant, length ~0.9
        [0.1, 0.9],    # doc 1: irrelevant
        [5.0, 6.0],    # doc 2: irrelevant-ish direction, but enormous
    ])
    print("dot scores   :", np.round(docs @ query, 2), "-> winner doc", int(np.argmax(docs @ query)))

    # FIX: compare directions only (cosine), i.e. normalise first.
    cos = cosine_sim(query, docs)
    print("cosine scores:", np.round(cos, 2), "-> winner doc", int(np.argmax(cos)))
    return int(np.argmax(cos))
# endregion q4


# region q5
# --------------------------------------------------------------------------
# Q5. Production: translate thresholds between the three "languages" databases speak
# --------------------------------------------------------------------------
# On UNIT vectors:   cosine_distance = 1 - cos    and    squared_l2 = 2 - 2*cos
def to_cosine_similarity(value: float, metric: str) -> float:
    """Convert a raw DB score into cosine similarity (unit vectors only)."""
    if metric == "cosine_distance":      # Chroma 'cosine', pgvector <=>
        return 1.0 - value
    if metric == "squared_l2":           # Chroma default 'l2'
        return 1.0 - value / 2.0
    if metric == "l2":                   # plain euclidean, pgvector <->, Qdrant Euclid
        return 1.0 - (value ** 2) / 2.0
    if metric == "dot":                  # Qdrant Dot, pgvector <#> returns the NEGATIVE of this
        return value
    raise ValueError(metric)


def from_cosine_similarity(cos: float, metric: str) -> float:
    if metric == "cosine_distance":
        return 1.0 - cos
    if metric == "squared_l2":
        return 2.0 - 2.0 * cos
    if metric == "l2":
        return float(np.sqrt(max(2.0 - 2.0 * cos, 0.0)))
    if metric == "dot":
        return cos
    raise ValueError(metric)


def q5():
    wanted_cos = 0.75     # business rule: "only keep chunks with cosine similarity >= 0.75"
    print(f"business rule: cosine similarity >= {wanted_cos}")
    for m in ("cosine_distance", "squared_l2", "l2", "dot"):
        print(f"  in a DB that reports {m:<16}: keep results with score "
              f"{'<=' if m != 'dot' else '>='} {from_cosine_similarity(wanted_cos, m):.4f}")

    # round-trip check on real random unit vectors
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(2, 768))
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    cos = float(a @ b)
    for m, raw in {
        "cosine_distance": 1 - cos,
        "squared_l2": float(np.sum((a - b) ** 2)),
        "l2": float(np.linalg.norm(a - b)),
        "dot": cos,
    }.items():
        assert abs(to_cosine_similarity(raw, m) - cos) < 1e-6, m
    print("round-trip conversions verified on random unit vectors")
    return from_cosine_similarity(wanted_cos, "squared_l2")
# endregion q5


if __name__ == "__main__":
    for fn in (q1, q2, q3, q4, q5):
        print(f"\n===== {fn.__name__} =====")
        fn()
