# Practice - Vector embeddings: from text to numbers

Five questions, rising difficulty. Attempt each one **before** opening the hint, and the hint before the solution. Every solution below is the exact code in `solutions.py` (`python solutions.py` runs them all; Q1 needs no API key, Q2-Q5 call the free Gemini embedding API and cache results on disk).

---

### Q1. Warm-up - a hand-made "meaning map"

**Problem**

Invent a 2-D meaning map with axes *(fruit-ness, vehicle-ness)* and place `apple`, `banana`, `car`, `bus` on it. Add a query word `scooter = (1, 6)`.
1. Compute cosine similarity of apple~banana and apple~car.
2. Which word is nearest to `scooter`?
3. Show that cosine similarity ignores vector length.

<details><summary><b>Hint</b></summary>

cosine(a, b) = (a · b) / (|a| |b|). For part 3 compare `scooter` with `10 * scooter`.

</details>

<details><summary><b>Solution</b></summary>

```python
# --------------------------------------------------------------------------
# Q1. Warm-up: a hand-made 2D "meaning map" for fruits and vehicles
# --------------------------------------------------------------------------
def q1():
    # axes: (fruit-ness, vehicle-ness). WE chose these labels - real models never do.
    words = {
        "apple":   np.array([9.0, 0.0]),
        "banana":  np.array([8.0, 1.0]),
        "car":     np.array([1.0, 9.0]),
        "bus":     np.array([0.0, 8.0]),
        "scooter": np.array([1.0, 6.0]),   # the query word
    }

    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    print("apple  ~ banana :", round(cos(words["apple"], words["banana"]), 3))
    print("apple  ~ car    :", round(cos(words["apple"], words["car"]), 3))

    # nearest word to the query "scooter"
    q = words["scooter"]
    ranked = sorted(
        ((cos(q, v), w) for w, v in words.items() if w != "scooter"), reverse=True
    )
    for score, w in ranked:
        print(f"scooter ~ {w:<7}: {score:.3f}")

    # cosine ignores length: a vector 10x longer points the same way
    print("scooter ~ 10*scooter:", round(cos(q, 10 * q), 3))
    return ranked[0][1]
```

</details>

**Why it works / what to learn**
An embedding is just coordinates on a map where *distance = difference in meaning*. Real models use 768-3072 axes nobody can name, but the geometry is identical. Cosine looks only at direction, so a long vector and a short one pointing the same way are "the same meaning".

**Expected output (≈)**
```text
apple  ~ banana : 0.992
apple  ~ car    : 0.11
scooter ~ car    : 0.999
scooter ~ bus    : 0.986
scooter ~ banana : 0.285
scooter ~ apple  : 0.164
scooter ~ 10*scooter: 1.0
```

---

### Q2. Apply - why keyword vectors cannot see synonyms

**Problem**

Take "The car is fast", "The automobile is quick", "Bananas are yellow fruit".
1. Build bag-of-words count vectors and print the cosine matrix.
2. Embed the same three sentences with Gemini (`SEMANTIC_SIMILARITY`) and print the cosine matrix.
3. Explain the difference in one sentence.

<details><summary><b>Hint</b></summary>

Build a vocabulary of all non-stop-words, then count each word per sentence. `cosine_sim(M, M)` gives the full matrix.

</details>

<details><summary><b>Solution</b></summary>

```python
# --------------------------------------------------------------------------
# Q2. Bag-of-words cannot see synonyms; dense embeddings can
# --------------------------------------------------------------------------
def q2():
    sentences = [
        "The car is fast",
        "The automobile is quick",
        "Bananas are yellow fruit",
    ]
    stop = {"the", "is", "are", "a"}
    tokens = [[w for w in s.lower().split() if w not in stop] for s in sentences]
    vocab = sorted({w for t in tokens for w in t})
    bow = np.array([[t.count(w) for w in vocab] for t in tokens], dtype=float)

    print("vocab:", vocab)
    print("BoW cosine matrix (all zero off-diagonal = blind to meaning):")
    print(np.round(cosine_sim(bow, bow), 2))

    dense = embed_texts(sentences, task_type="SEMANTIC_SIMILARITY", dim=768)
    print("Gemini cosine matrix:")
    sim = cosine_sim(dense, dense)
    print(np.round(sim, 2))
    return sim
```

</details>

**Why it works / what to learn**
Bag-of-words gives each *word* its own axis, so "car" and "automobile" are perpendicular (similarity 0). A trained embedding model learned from billions of sentences that those words appear in the same contexts, so it places them almost on top of each other.

**Expected output (≈)**
```text
BoW cosine matrix: off-diagonals all 0.0 (no shared words between sentences)
Gemini cosine matrix: car~automobile ≈ 0.8-0.9, car~banana ≈ 0.4-0.6
```

---

### Q3. Build - PCA from scratch and cluster check

**Problem**

You have 12 sentences about cooking, cricket and programming (4 each).
1. Embed them (`CLUSTERING`, 768-d).
2. Implement PCA yourself with `np.linalg.svd` and project to 2-D.
3. Verify against scikit-learn, and compare nearest-centroid accuracy in 768-D vs 2-D.

<details><summary><b>Hint</b></summary>

Centre the data (subtract the column mean), run SVD, project onto the first two rows of `Vt`. Signs of axes may flip versus scikit-learn, so compare absolute values.

</details>

<details><summary><b>Solution</b></summary>

```python
# --------------------------------------------------------------------------
# Q3. Build PCA from scratch (SVD) and check topic clusters survive in 2D
# --------------------------------------------------------------------------
TOPIC_SENTENCES = {
    "cooking": [
        "Simmer the onions in ghee until they turn golden brown.",
        "Knead the dough and let it rest for thirty minutes.",
        "Add a pinch of turmeric and cumin to the hot oil.",
        "Bake the cake at 180 degrees until a toothpick comes out clean.",
    ],
    "cricket": [
        "The opening batsman hit a cover drive for four runs.",
        "The spinner took three wickets in the final over.",
        "India won the toss and chose to bowl first.",
        "The umpire raised his finger and the batsman had to walk.",
    ],
    "programming": [
        "A null pointer exception crashed the backend service.",
        "Use a foreign key to link orders to customers in the database.",
        "The unit tests failed after we refactored the controller.",
        "Deploy the Docker container to the production server.",
    ],
}


def pca_from_scratch(X: np.ndarray, n_components: int = 2):
    """Centre the data, SVD it, project on the top directions of variance."""
    Xc = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    coords = Xc @ Vt[:n_components].T
    explained = (S ** 2) / (S ** 2).sum()
    return coords, explained[:n_components]


def _nearest_centroid_accuracy(X: np.ndarray, labels: list[str]) -> float:
    uniq = sorted(set(labels))
    cents = np.vstack([X[[i for i, l in enumerate(labels) if l == u]].mean(axis=0) for u in uniq])
    correct = 0
    for x, l in zip(X, labels):
        d = np.linalg.norm(cents - x, axis=1)
        correct += uniq[int(d.argmin())] == l
    return correct / len(labels)


def q3():
    texts, labels = [], []
    for topic, sents in TOPIC_SENTENCES.items():
        texts += sents
        labels += [topic] * len(sents)

    X = embed_texts(texts, task_type="CLUSTERING", dim=768)
    coords, explained = pca_from_scratch(X, 2)
    print("explained variance of PC1, PC2:", np.round(explained, 3))

    # cross-check against scikit-learn (equal up to the sign of each axis)
    from sklearn.decomposition import PCA
    sk = PCA(n_components=2).fit_transform(X)
    match = all(
        np.allclose(np.abs(coords[:, k]), np.abs(sk[:, k]), atol=1e-3) for k in range(2)
    )
    print("matches scikit-learn (up to sign):", match)

    acc_full = _nearest_centroid_accuracy(X, labels)
    acc_2d = _nearest_centroid_accuracy(coords, labels)
    print(f"nearest-centroid accuracy  768-D: {acc_full:.2f}   2-D: {acc_2d:.2f}")
    return acc_full, acc_2d
```

</details>

**Why it works / what to learn**
PCA finds the directions where the data varies most. Topic clusters in a 768-D space are so strong that they survive being squashed into 2 dimensions, which is why 2-D scatter plots of embeddings are a useful debugging tool, even though they lose information.

**Expected output (≈)**
```text
explained variance of PC1, PC2: [≈0.15 ≈0.11]
matches scikit-learn (up to sign): True
nearest-centroid accuracy  768-D: 1.00   2-D: ≈0.92-1.00
```

---

### Q4. Debug - "I truncated my vectors and the scores look wrong"

**Problem**

A teammate sliced the first 256 numbers out of the 768-d vectors to save memory, kept using a plain dot product, and set a threshold of 0.5. Results look wrong. Find the bug and fix it.

<details><summary><b>Hint</b></summary>

Print `np.linalg.norm` of the sliced vectors. A dot product equals cosine similarity only when both vectors have length 1.

</details>

<details><summary><b>Solution</b></summary>

```python
# --------------------------------------------------------------------------
# Q4. Debug: "I truncated the vectors and my scores look wrong"
# --------------------------------------------------------------------------
def q4():
    docs = [
        "How do I reset my password?",
        "Steps to change your account password",
        "Our office is closed on public holidays",
        "Invoice payment is due within 30 days",
    ]
    query = "forgot my login password"

    D = embed_texts(docs, task_type="RETRIEVAL_DOCUMENT", dim=768)
    qv = embed_query(query, dim=768)

    def unit(M):
        M = np.atleast_2d(M)
        return M / np.linalg.norm(M, axis=1, keepdims=True)

    # BUG: slice off the first 256 numbers and use a plain dot product
    D_bad, q_bad = D[:, :256], qv[:256]
    print("norms after slicing (should be 1.0 but are not):", np.round(np.linalg.norm(D_bad, axis=1), 2))
    print("bad dot scores :", np.round(D_bad @ q_bad, 3))
    THRESHOLD = 0.5   # "only return passages scoring >= 0.5"
    print("docs passing threshold  bad:", int((D_bad @ q_bad >= THRESHOLD).sum()),
          " fixed:", int((unit(D_bad) @ unit(q_bad)[0] >= THRESHOLD).sum()))

    # FIX: re-normalise after slicing (or just request dim=256 from the API)
    D_ok, q_ok = unit(D_bad), unit(q_bad)[0]
    print("fixed dot scores:", np.round(D_ok @ q_ok, 3))
    print("ranking bad  :", list(np.argsort(-(D_bad @ q_bad))))
    print("ranking fixed:", list(np.argsort(-(D_ok @ q_ok))))
    return D_ok @ q_ok
```

</details>

**Why it works / what to learn**
Only the full 3072-d Gemini vector comes pre-normalised. Slicing a unit vector makes it shorter than 1, so dot products shrink and a fixed threshold silently filters everything out. Always re-normalise after truncation (or request `output_dimensionality=256` and normalise), exactly as our `embed_texts` helper does.

**Expected output (≈)**
```text
norms after slicing (should be 1.0 but are not): [≈0.6 ≈0.6 ≈0.6 ≈0.6]
docs passing threshold  bad: 0   fixed: ≈2
ranking bad vs fixed: may differ, because every doc shrank by a different amount, so raw dot products are not comparable between docs
```

---

### Q5. Production - near-duplicate support tickets

**Problem**

A support inbox has 12 tickets, some of which are the same issue worded differently. Group duplicates automatically:
1. Embed all tickets, compute the cosine matrix.
2. Cluster by a similarity threshold (union-find, so A~B and B~C puts all three together).
3. Sweep thresholds 0.60-0.95, score with pairwise precision/recall/F1 against the known groups, and pick the best.

<details><summary><b>Hint</b></summary>

Too low a threshold merges unrelated tickets (precision drops); too high splits real duplicates (recall drops). Pairwise F1 balances both.

</details>

<details><summary><b>Solution</b></summary>

```python
# --------------------------------------------------------------------------
# Q5. Production: near-duplicate support tickets with a tuned threshold
# --------------------------------------------------------------------------
TICKETS = [
    # (text, true_group)
    ("I can't log in to my account", "login"),
    ("Unable to sign in, login keeps failing", "login"),
    ("Login page says my credentials are invalid", "login"),
    ("Please refund my last payment", "refund"),
    ("I want my money back for the last charge", "refund"),
    ("Request for refund of duplicate payment", "refund"),
    ("The mobile app crashes when I open settings", "crash"),
    ("App closes unexpectedly on the settings screen", "crash"),
    ("How do I export my data to CSV?", "export"),
    ("Where can I download my data as a spreadsheet?", "export"),
    ("Do you offer a student discount?", "discount"),
    ("What are your office hours?", "hours"),
]


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def cluster_by_threshold(sim: np.ndarray, threshold: float) -> list[list[int]]:
    n = len(sim)
    uf = UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= threshold:
                uf.union(i, j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(uf.find(i), []).append(i)
    return list(groups.values())


def pair_f1(groups: list[list[int]], truth: list[str]) -> tuple[float, float, float]:
    """Pairwise precision / recall / F1 of predicted duplicate pairs vs ground truth."""
    n = len(truth)
    pred = {(i, j) for g in groups for i in g for j in g if i < j}
    true = {(i, j) for i in range(n) for j in range(i + 1, n) if truth[i] == truth[j]}
    tp = len(pred & true)
    p = tp / len(pred) if pred else 1.0
    r = tp / len(true) if true else 1.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def q5():
    texts = [t for t, _ in TICKETS]
    truth = [g for _, g in TICKETS]
    E = embed_texts(texts, task_type="SEMANTIC_SIMILARITY", dim=768)
    sim = cosine_sim(E, E)

    best = (-1.0, 0.0)
    print("threshold  precision  recall   F1")
    for th in np.arange(0.60, 0.96, 0.05):
        groups = cluster_by_threshold(sim, float(th))
        p, r, f1 = pair_f1(groups, truth)
        print(f"  {th:.2f}      {p:.2f}      {r:.2f}   {f1:.2f}")
        if f1 > best[0]:
            best = (f1, float(th))

    f1, th = best
    print(f"\nbest threshold = {th:.2f} (F1 {f1:.2f})")
    for g in cluster_by_threshold(sim, th):
        if len(g) > 1:
            print("  duplicates:", [texts[i] for i in g])
    return th
```

</details>

**Why it works / what to learn**
There is no universal "correct" similarity threshold: it depends on the model, the task type and your data. The professional move is to label a small set, sweep the threshold, and choose by a metric. The same habit is used later for retrieval cut-offs and semantic caching (Week 4).

**Expected output (≈)**
```text
threshold  precision  recall   F1
  0.60 ... 0.95 ...   (best F1 around 0.75-0.85)
duplicates: login group, refund group, crash group, export group
```
