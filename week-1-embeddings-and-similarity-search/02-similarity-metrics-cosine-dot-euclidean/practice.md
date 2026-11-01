# Practice - Similarity metrics: cosine, dot product, Euclidean

Five questions, rising difficulty. Attempt each before opening the hint, and the hint before the solution. Every solution is the exact code in `solutions.py`. **All five run offline with no API key** (`python solutions.py`).

---

### Q1. Warm-up - compute the three metrics by hand

**Problem**

For `a = [3, 4]` and `b = [4, 3]` compute the dot product, both lengths, the cosine similarity and the Euclidean distance. Then double `b` and say which metrics change.

<details><summary><b>Hint</b></summary>

cos = (a·b) / (|a||b|). Lengths come from Pythagoras: |[3,4]| = 5. Doubling b stretches it without turning it.

</details>

<details><summary><b>Solution</b></summary>

```python
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
```

</details>

**Why it works / what to learn**
Cosine measures only the angle, so stretching a vector cannot change it. Dot product and Euclidean distance both react to length. Embedding search depends on angle, so length-sensitive metrics need normalised inputs.

**Expected output (≈)**
```text
dot product      : 24.0
lengths          : 5.0, 5.0
cosine similarity: 0.96
euclidean dist   : 1.414

after doubling b:
dot product      : 48.0   (changed!)
cosine similarity: 0.96   (unchanged)
euclidean dist   : 5.385   (changed!)
```

---

### Q2. Apply - prove the unit-vector identity numerically

**Problem**

Generate 1000 random pairs of 768-d vectors, normalise them, and show that `|a-b|² = 2 - 2·cos(a, b)`. Then show cosine and Euclidean produce the same ranking.

<details><summary><b>Hint</b></summary>

Expand `|a-b|²` as `|a|² - 2a·b + |b|²`. For the ranking, sort one query against many rows two ways and compare the index arrays.

</details>

<details><summary><b>Solution</b></summary>

```python
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
```

</details>

**Why it works / what to learn**
`2 - 2cos` is a decreasing straight-line function of `cos`, so the order of results never changes. That is why a vector DB can use whichever metric is cheapest as long as vectors are unit length.

**Expected output (≈)**
```text
max |  |a-b|^2 - (2 - 2cos)  | over 1000 pairs: ≈1e-15
same ranking by cosine and by euclidean: True
```

---

### Q3. Build - a metric-comparison harness

**Problem**

Write `rank_all(query, matrix, k)` that returns the top-k row indexes under dot, cosine and Euclidean. Run it on random un-normalised vectors, then on the normalised versions, and show when the three agree.

<details><summary><b>Hint</b></summary>

Sort descending for dot/cosine (`argsort(-scores)`), ascending for Euclidean distance. Normalise rows with `M / np.linalg.norm(M, axis=1, keepdims=True)`.

</details>

<details><summary><b>Solution</b></summary>

```python
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
```

</details>

**Why it works / what to learn**
On raw vectors, three metrics give three different answers (a long vector wins under dot, a short one under Euclidean). After normalising, they collapse into one. Building the harness yourself is the fastest way to build intuition for what each metric rewards.

**Expected output (≈)**
```text
UN-normalised vectors:
  dot    : [17, 20, 35]
  cosine : [17, 35, 20]
  euclid : [4, 35, 17]
UNIT vectors:
  dot / cosine / euclid : [17, 35, 20]  (identical)
all three agree on unit vectors: True
```

---

### Q4. Debug - "the biggest document always wins"

**Problem**

Query `[1, 0]`; documents `[0.9, 0.1]` (truly relevant), `[0.1, 0.9]`, and `[5, 6]`. A dot-product search returns document 2 first. Explain why and fix it.

<details><summary><b>Hint</b></summary>

Print the dot scores and compare the document lengths. What does cosine ignore that dot does not?

</details>

<details><summary><b>Solution</b></summary>

```python
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
```

</details>

**Why it works / what to learn**
Dot product multiplies by length, so a huge vector pointing in a vaguely similar direction out-scores a small perfect match. Real-world cause: un-normalised embeddings, or vectors you truncated/averaged and forgot to re-normalise. Fix: cosine, or normalise once at ingestion.

**Expected output (≈)**
```text
dot scores   : [0.9 0.1 5. ] -> winner doc 2
cosine scores: [0.99 0.11 0.64] -> winner doc 0
```

---

### Q5. Production - translate a threshold between databases

**Problem**

Product rule: *"keep only chunks with cosine similarity ≥ 0.75."* Write converters between cosine similarity and the four scores databases return (cosine distance, squared L2, L2, raw dot) for unit vectors. State the correct cut-off for each, and verify the converters on random vectors.

<details><summary><b>Hint</b></summary>

Use `cosine_distance = 1 - cos` and `squared_L2 = 2 - 2cos`. Distances keep results *below* the cut-off; similarities keep results *above*.

</details>

<details><summary><b>Solution</b></summary>

```python
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
```

</details>

**Why it works / what to learn**
The most common silent RAG bug is a threshold written for one scale and applied to another (e.g. `distance >= 0.75` keeps the *worst* results). Always convert explicitly, name variables `*_similarity` vs `*_distance`, and test the conversion with an assertion like the one above.

**Expected output (≈)**
```text
business rule: cosine similarity >= 0.75
  in a DB that reports cosine_distance : keep results with score <= 0.2500
  in a DB that reports squared_l2      : keep results with score <= 0.5000
  in a DB that reports l2              : keep results with score <= 0.7071
  in a DB that reports dot             : keep results with score >= 0.7500
round-trip conversions verified on random unit vectors
```
