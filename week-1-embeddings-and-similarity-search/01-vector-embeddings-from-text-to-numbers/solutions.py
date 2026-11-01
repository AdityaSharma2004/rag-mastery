"""
Solutions for 01-vector-embeddings-from-text-to-numbers/practice.md
Run everything:   python solutions.py      (q2-q5 call the free Gemini embedding API)
Run one:          python -c "import solutions; solutions.q1()"
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))
from gemini_utils import *   # embed_texts, embed_query, cosine_sim, top_k, SAMPLE_DOCS, np ...

import numpy as np  # (explicit, even though the star-import already brings it in)


# region q1
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
# endregion q1


# region q2
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
# endregion q2


# region q3
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
# endregion q3


# region q4
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
# endregion q4


# region q5
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
# endregion q5


if __name__ == "__main__":
    for fn in (q1, q2, q3, q4, q5):
        print(f"\n===== {fn.__name__} =====")
        fn()
