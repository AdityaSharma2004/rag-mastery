"""
Solutions for 05-semantic-vs-keyword-vs-hybrid-search/practice.md
Run everything:   python solutions.py
Q1, Q2, Q4 are pure Python/numpy (no API key). Q3 and Q5 embed text with the free Gemini API.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))
from gemini_utils import *   # embed_texts, embed_query, cosine_sim, np ...

import math
import re
from collections import Counter

# A tiny support knowledge base shared by every question.
KB = [
    "To reset your password click Forgot password on the login page.",              # 0
    "Refunds are issued to the original payment method within 5 business days.",     # 1
    "Error E4012 means the payment gateway timed out; retry after one minute.",     # 2
    "Two-factor authentication adds a second step when you sign in.",               # 3
    "You can cancel your subscription anytime from the billing settings.",          # 4
    "Invoices are emailed on the first day of every month as PDF files.",           # 5
    "If you cannot access your account, use the recovery link sent to your email.", # 6
    "The mobile app supports dark mode under Display settings.",                    # 7
]

STOP = {"the", "a", "an", "to", "of", "is", "are", "you", "your", "can", "on", "in", "if", "as", "or", "and", "from", "i", "my", "do", "how", "when", "for", "it", "by", "at", "be", "means"}


def tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]


# region q1
# --------------------------------------------------------------------------
# Q1. Warm-up: WHERE LIKE vs token keyword search vs (spoiler) meaning
# --------------------------------------------------------------------------
def like_search(query: str, docs: list[str]) -> list[int]:
    """SQL:  WHERE doc LIKE '%query%'   (the whole phrase must appear)"""
    return [i for i, d in enumerate(docs) if query.lower() in d.lower()]


def keyword_search(query: str, docs: list[str]) -> list[int]:
    """Any shared token counts (very rough FULLTEXT ... IN BOOLEAN MODE with OR). Ranked by overlap."""
    q = set(tokenize(query))
    scored = [(len(q & set(tokenize(d))), i) for i, d in enumerate(docs)]
    return [i for s, i in sorted(scored, reverse=True) if s > 0]


def q1():
    for query in ["reset password", "forgot my login credentials", "money back"]:
        print(f"\nquery: {query!r}")
        print("  LIKE    ->", like_search(query, KB))
        print("  keyword ->", keyword_search(query, KB))
    # 'money back' should find the refund doc (1) but shares no words with it.
    return keyword_search("money back", KB)
# endregion q1


# region q2
# --------------------------------------------------------------------------
# Q2. Apply: BM25 from scratch
# --------------------------------------------------------------------------
class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.toks = [tokenize(d) for d in docs]
        self.N = len(docs)
        self.avgdl = sum(len(t) for t in self.toks) / self.N
        df = Counter(w for t in self.toks for w in set(t))        # in how many docs does w appear?
        # idf: rare words matter more. The "+1" keeps it positive.
        self.idf = {w: math.log((self.N - n + 0.5) / (n + 0.5) + 1) for w, n in df.items()}

    def score(self, query: str) -> np.ndarray:
        q = tokenize(query)
        out = np.zeros(self.N)
        for i, t in enumerate(self.toks):
            tf = Counter(t)
            dl = len(t)
            for w in q:
                if w not in tf:
                    continue
                num = tf[w] * (self.k1 + 1)
                den = tf[w] + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                out[i] += self.idf[w] * num / den
        return out


def q2():
    bm = BM25(KB)
    query = "payment gateway error"
    s = bm.score(query)
    order = np.argsort(-s)
    print("BM25 ranking for", repr(query))
    for i in order[:3]:
        print(f"  {s[i]:.3f}  [{i}] {KB[i]}")
    # why is 'e4012' worth more than 'payment'?  rarity:
    print("idf('e4012') =", round(bm.idf["e4012"], 3), "  idf('payment') =", round(bm.idf["payment"], 3))

    # cross-check the *ranking* against the rank_bm25 library (numbers differ: different idf variant)
    try:
        from rank_bm25 import BM25Okapi
        lib = BM25Okapi([tokenize(d) for d in KB]).get_scores(tokenize(query))
        print("library top-3:", [int(i) for i in np.argsort(-lib)[:3]], " ours:", [int(i) for i in order[:3]])
    except ImportError:
        print("(pip install rank-bm25 to cross-check)")
    return int(order[0])
# endregion q2


# region q3
# --------------------------------------------------------------------------
# Q3. Build: hybrid search with Reciprocal Rank Fusion (RRF)
# --------------------------------------------------------------------------
def rrf(rank_lists: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """score(doc) = sum over lists of 1/(k + rank).  rank starts at 1. Only ranks matter, not raw scores."""
    scores: dict[int, float] = {}
    for ranking in rank_lists:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


def hybrid_search(query: str, docs: list[str], top_n: int = 3, pool: int = 5):
    bm = BM25(docs)
    kw = bm.score(query)
    kw_rank = [int(i) for i in np.argsort(-kw)[:pool] if kw[i] > 0]

    D = embed_texts(docs, task_type="RETRIEVAL_DOCUMENT", dim=768)
    qv = embed_texts([query], task_type="RETRIEVAL_QUERY", dim=768)[0]
    sem = D @ qv
    sem_rank = [int(i) for i in np.argsort(-sem)[:pool]]

    fused = rrf([kw_rank, sem_rank])[:top_n]
    return kw_rank, sem_rank, fused


def q3():
    for query in ["I forgot my login credentials", "error E4012"]:
        kw, sem, fused = hybrid_search(query, KB)
        print(f"\nquery: {query!r}")
        print("  keyword :", kw)
        print("  semantic:", sem)
        print("  hybrid  :", [(i, round(s, 4)) for i, s in fused])
    return fused
# endregion q3


# region q4
# --------------------------------------------------------------------------
# Q4. Debug: "I added BM25 and cosine and hybrid got WORSE"
# --------------------------------------------------------------------------
def minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    return np.zeros_like(x) if hi == lo else (x - lo) / (hi - lo)


def q4():
    # Made-up but realistic scales: BM25 scores are unbounded (here 0-14), cosine is 0-1.
    bm25 = np.array([11.0, 14.0, 1.0, 0.0])    # keyword engine: doc 1 slightly ahead of doc 0
    cosine = np.array([0.82, 0.55, 0.80, 0.30])  # embeddings: doc 0 clearly best (doc 0 is good on BOTH)

    naive = 0.5 * bm25 + 0.5 * cosine            # BUG: different scales
    print("naive sum   :", np.round(naive, 2), "-> winner", int(naive.argmax()))
    print("  (BM25 contributes ~", round(float(0.5 * bm25[1] / naive[1]) * 100), "% of doc 1's score)")

    # FIX A: normalise both to 0-1 first, then weight
    fixed = 0.5 * minmax(bm25) + 0.5 * minmax(cosine)
    print("min-max sum :", np.round(fixed, 2), "-> winner", int(fixed.argmax()))

    # FIX B: ignore scales entirely, fuse RANKS (RRF)
    r_bm = [int(i) for i in np.argsort(-bm25)]
    r_co = [int(i) for i in np.argsort(-cosine)]
    print("RRF         :", [(i, round(s, 4)) for i, s in rrf([r_bm, r_co])])
    return int(fixed.argmax())
# endregion q4


# region q5
# --------------------------------------------------------------------------
# Q5. Production: an evaluation harness - keyword vs semantic vs hybrid
# --------------------------------------------------------------------------
EVAL = [  # (query, index of the correct KB doc)
    ("error E4012", 2),                          # exact token: keyword should win
    ("what does E4012 mean", 2),
    ("I want my money back", 1),                 # no shared words: semantic should win
    ("lost access to my profile", 6),
    ("how do I stop paying every month", 4),
    ("change app theme to dark", 7),             # shares words: all should work
    ("when do I get my bill by email", 5),
    ("extra security step at login", 3),
]


def mrr_and_hit1(rankings: list[list[int]], truth: list[int]) -> tuple[float, float]:
    rr, hit = [], []
    for ranking, t in zip(rankings, truth):
        pos = ranking.index(t) + 1 if t in ranking else None
        rr.append(1 / pos if pos else 0.0)
        hit.append(1.0 if pos == 1 else 0.0)
    return float(np.mean(rr)), float(np.mean(hit))


def q5():
    bm = BM25(KB)
    D = embed_texts(KB, task_type="RETRIEVAL_DOCUMENT", dim=768)
    kw_all, sem_all, hyb_all = [], [], []
    for query, _ in EVAL:
        kw = bm.score(query)
        kw_r = [int(i) for i in np.argsort(-kw)]
        qv = embed_texts([query], task_type="RETRIEVAL_QUERY", dim=768)[0]
        sem_r = [int(i) for i in np.argsort(-(D @ qv))]
        hyb_r = [i for i, _ in rrf([[i for i in kw_r if kw[i] > 0], sem_r])]
        kw_all.append(kw_r); sem_all.append(sem_r); hyb_all.append(hyb_r)

    truth = [t for _, t in EVAL]
    print(f"{'method':<10} {'MRR':>6} {'hit@1':>7}")
    for name, r in (("keyword", kw_all), ("semantic", sem_all), ("hybrid", hyb_all)):
        mrr, h1 = mrr_and_hit1(r, truth)
        print(f"{name:<10} {mrr:>6.2f} {h1:>7.2f}")
    return mrr_and_hit1(hyb_all, truth)
# endregion q5


if __name__ == "__main__":
    for fn in (q1, q2, q3, q4, q5):
        print(f"\n===== {fn.__name__} =====")
        fn()
