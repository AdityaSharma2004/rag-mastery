"""
Week 2 · 06 Hybrid search (BM25 + vector) - solutions to practice.md (q1 ... q5)
Run all: python solutions.py     Run one: python solutions.py 3
Q1, Q2, Q4 are pure Python/numpy. Q3 embeds with the free Gemini API. Q5 needs Postgres+pgvector
(docker run -d -e POSTGRES_PASSWORD=postgres -p 5432:5432 pgvector/pgvector:pg16; DATABASE_URL in .env).
"""
import sys, pathlib, os, re, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))
from gemini_utils import *   # embed_texts, embed_query, np ...

from rank_bm25 import BM25Okapi

DIM = 768
DOCS = [
    "To reset your password click Forgot password on the login page.",                 # 0
    "Refunds are issued to the original payment method within 5 business days.",        # 1
    "Error E4012 means the payment gateway timed out; retry after one minute.",         # 2
    "Two-factor authentication adds a second step when you sign in.",                   # 3
    "You can cancel your subscription anytime from the billing settings.",              # 4
    "Invoices are emailed on the first day of every month as PDF files.",               # 5
    "If you cannot access your account, use the recovery link sent to your email.",     # 6
    "The mobile app supports dark mode under Display settings.",                        # 7
    "Error E5001 means your API key is invalid or has been revoked.",                   # 8
    "Webhooks retry failed deliveries three times with exponential back-off.",          # 9
]
LABELED = [  # (query, correct doc)
    ("error E4012", 2), ("what does E5001 mean", 8), ("I want my money back", 1),
    ("lost access to my profile", 6), ("how do I stop paying every month", 4),
    ("change the app theme to dark", 7), ("when will I receive my bill by email", 5),
    ("extra security step at sign in", 3),
]
STOP = {"the", "a", "an", "to", "of", "is", "are", "you", "your", "can", "on", "in", "if", "as", "or", "and", "from",
        "i", "my", "do", "how", "when", "for", "it", "by", "at", "be", "what", "does", "will"}


# region helper
def tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]


def rrf(rank_lists: list[list[int]], weights: list[float] | None = None, k: int = 60) -> list[tuple[int, float]]:
    """score(d) = sum_i  w_i / (k + rank_i(d))  - ranks start at 1; only ranks matter, never raw scores."""
    weights = weights or [1.0] * len(rank_lists)
    scores: dict[int, float] = {}
    for w, ranking in zip(weights, rank_lists):
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + w / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


def mrr_hit1(rankings, truth):
    rr = [1 / (r.index(t) + 1) if t in r else 0.0 for r, t in zip(rankings, truth)]
    h1 = [1.0 if r and r[0] == t else 0.0 for r, t in zip(rankings, truth)]
    return round(float(np.mean(rr)), 3), round(float(np.mean(h1)), 3)
# endregion helper


# region q1
# --------------------------------------------------------------------------
# Q1. Warm-up: BM25 with the rank_bm25 library; why tokenisation decides everything
# --------------------------------------------------------------------------
def q1():
    bm = BM25Okapi([tokenize(d) for d in DOCS])
    for query in ["error E4012", "payment gateway timeout", "money back"]:
        s = bm.get_scores(tokenize(query))
        top = [int(i) for i in np.argsort(-s)[:2] if s[i] > 0]
        print(f"{query!r:<28} -> {top}  scores {[round(float(s[i]), 2) for i in top]}")
    # a naive split() tokenizer keeps punctuation and case: "E4012" and "e4012;" never match
    naive = BM25Okapi([d.split() for d in DOCS])
    s = naive.get_scores("error e4012".split())
    print("naive split() tokenizer, 'error e4012' ->", [int(i) for i in np.argsort(-s)[:2] if s[i] > 0])
    return True
# endregion q1


# region q2
# --------------------------------------------------------------------------
# Q2. Apply: why fusing raw scores fails - three normalisations vs RRF, on made-up score lists
# --------------------------------------------------------------------------
def q2():
    bm25 = np.array([14.0, 3.0, 2.5, 2.0, 0.0])        # one outlier: an exact ID match
    cosine = np.array([0.55, 0.81, 0.80, 0.62, 0.30])   # embeddings: docs 1 and 2 are the real answers
    minmax = lambda x: (x - x.min()) / (x.max() - x.min())
    zscore = lambda x: (x - x.mean()) / x.std()

    def winner(s): return int(np.argmax(s))
    print("raw sum        :", np.round(bm25 + cosine, 2), "-> winner", winner(bm25 + cosine))
    print("min-max (0.5/0.5):", np.round(0.5 * minmax(bm25) + 0.5 * minmax(cosine), 2),
          "-> winner", winner(minmax(bm25) + minmax(cosine)))
    print("z-score        :", np.round(zscore(bm25) + zscore(cosine), 2), "-> winner", winner(zscore(bm25) + zscore(cosine)))
    r_bm = [int(i) for i in np.argsort(-bm25)]
    r_co = [int(i) for i in np.argsort(-cosine)]
    fused = rrf([r_bm, r_co])
    print("RRF (k=60)     :", [(d, round(s, 4)) for d, s in fused], "-> winner", fused[0][0])
    # hand check: doc 1 -> bm25 rank 2, cosine rank 1: 1/62 + 1/61
    print("hand check doc 1:", round(1 / 62 + 1 / 61, 4))
    return fused[0][0]
# endregion q2


# region q3
# --------------------------------------------------------------------------
# Q3. Build: a HybridRetriever (BM25 + dense, weighted RRF, candidate pools) and an evaluation
# --------------------------------------------------------------------------
class HybridRetriever:
    def __init__(self, docs, dim=DIM, pool=20, k_rrf=60, w_kw=1.0, w_dense=1.0):
        self.docs, self.pool, self.k_rrf, self.w = docs, pool, k_rrf, (w_kw, w_dense)
        self.bm25 = BM25Okapi([tokenize(d) for d in docs])
        self.D = embed_texts(docs, task_type="RETRIEVAL_DOCUMENT", dim=dim)
        self.dim = dim

    def keyword_ranking(self, q):
        s = self.bm25.get_scores(tokenize(q))
        return [int(i) for i in np.argsort(-s)[: self.pool] if s[i] > 0]          # drop docs that matched nothing

    def dense_ranking(self, q):
        s = self.D @ embed_query(q, dim=self.dim)
        return [int(i) for i in np.argsort(-s)[: self.pool]]

    def search(self, q, k=3, mode="hybrid"):
        kw, de = self.keyword_ranking(q), self.dense_ranking(q)
        if mode == "keyword": return kw[:k]
        if mode == "dense": return de[:k]
        return [d for d, _ in rrf([kw, de], weights=list(self.w), k=self.k_rrf)[:k]]


def q3():
    hr = HybridRetriever(DOCS)
    truth = [t for _, t in LABELED]
    print(f"{'mode':<9} {'MRR':>6} {'hit@1':>7}")
    for mode in ("keyword", "dense", "hybrid"):
        ranks = [hr.search(q, k=len(DOCS), mode=mode) for q, _ in LABELED]
        print(f"{mode:<9} {mrr_hit1(ranks, truth)[0]:>6} {mrr_hit1(ranks, truth)[1]:>7}")
    for w_kw in (0.5, 1.0, 2.0):                                                  # tune the leg weights
        hr.w = (w_kw, 1.0)
        ranks = [hr.search(q, k=len(DOCS)) for q, _ in LABELED]
        print(f"weighted RRF  keyword weight {w_kw}: MRR {mrr_hit1(ranks, truth)[0]}")
    return True
# endregion q3


# region q4
# --------------------------------------------------------------------------
# Q4. Debug: "BM25 never finds E-4012 / e4012 even though the doc contains it"
# --------------------------------------------------------------------------
def q4():
    # (a corpus of 10: with only 2 docs, rank_bm25 gives a term that is in half of them an IDF of 0)
    docs = ["Error E-4012: the payment gateway timed out."] + [d for i, d in enumerate(DOCS) if i != 2]

    def naive_tok(t): return t.split()                                           # case + punctuation kept
    def basic_tok(t): return re.findall(r"[a-z0-9]+", t.lower())                 # splits E-4012 into 'e' and '4012'
    def code_tok(t):                                                              # also keep the glued form 'e4012'
        toks = basic_tok(t)
        glued = re.findall(r"[a-z]+-\d+", t.lower())
        return toks + [g.replace("-", "") for g in glued]

    for name, tok in (("naive split()", naive_tok), ("alnum split", basic_tok), ("code-aware", code_tok)):
        bm = BM25Okapi([tok(d) for d in docs])
        for query in ("E-4012", "e4012", "4012"):
            s = bm.get_scores(tok(query))
            print(f"{name:<14} query {query!r:<9} -> {'FOUND' if s[0] > 0 else 'missed'}")
    return True
# endregion q4


# region q5
# --------------------------------------------------------------------------
# Q5. Production: hybrid search INSIDE Postgres (tsvector full-text + pgvector, fused with RRF in SQL)
# --------------------------------------------------------------------------
def q5():
    import psycopg
    from pgvector.psycopg import register_vector
    url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/ragdb")
    try:
        conn = psycopg.connect(url, autocommit=True)
    except psycopg.OperationalError as e:
        print("Postgres not reachable - start it (see header) and re-run Q5.", e); return None
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector"); register_vector(conn)
    conn.execute("DROP TABLE IF EXISTS w2_hybrid")
    conn.execute(f"""CREATE TABLE w2_hybrid (
        id int PRIMARY KEY, content text, embedding vector({DIM}),
        tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED)""")   # keyword column, auto-maintained
    conn.execute("CREATE INDEX w2_hybrid_tsv ON w2_hybrid USING gin (tsv)")               # GIN = inverted index
    vecs = embed_texts(DOCS, dim=DIM)
    with conn.cursor() as cur:
        cur.executemany("INSERT INTO w2_hybrid (id, content, embedding) VALUES (%s,%s,%s)",
                        [(i, t, v) for i, (t, v) in enumerate(zip(DOCS, vecs))])

    SQL = """
    WITH semantic AS (
        SELECT id, row_number() OVER (ORDER BY embedding <=> %(q)s) AS rnk
        FROM w2_hybrid ORDER BY embedding <=> %(q)s LIMIT 20),
    keyword AS (
        SELECT id, row_number() OVER (ORDER BY ts_rank_cd(tsv, query) DESC) AS rnk
        FROM w2_hybrid, websearch_to_tsquery('english', %(text)s) query
        WHERE tsv @@ query ORDER BY ts_rank_cd(tsv, query) DESC LIMIT 20)
    SELECT COALESCE(s.id, k.id) AS id,
           COALESCE(1.0 / (60 + s.rnk), 0) + COALESCE(1.0 / (60 + k.rnk), 0) AS rrf_score
    FROM semantic s FULL OUTER JOIN keyword k ON s.id = k.id
    ORDER BY rrf_score DESC LIMIT 3"""

    for text in ("error E4012", "I want my money back"):
        rows = conn.execute(SQL, {"q": embed_query(text, dim=DIM), "text": text}).fetchall()
        print(f"{text!r:<24} -> {[(r[0], round(float(r[1]), 4)) for r in rows]}")
    conn.execute("DROP TABLE w2_hybrid")
    return True
# endregion q5


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3", "4", "5"]
    for n in which:
        print(f"\n===== Q{n} =====")
        globals()[f"q{n}"]()
