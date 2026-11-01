"""
Week 2 · 07 Re-ranking with cross-encoders - solutions to practice.md (q1 ... q5)
Run all: python solutions.py     Run one: python solutions.py 3
* Q1, Q4, Q5 are pure Python/numpy (no key, no model).
* Q2 uses gemini-2.5-flash as the reranker (free Gemini key). Q3 uses Gemini embeddings for stage 1.
* The local cross-encoder needs `pip install sentence-transformers` and downloads
  cross-encoder/ms-marco-MiniLM-L-6-v2 (~90 MB) once. If that is unavailable (offline / blocked network) the code
  falls back to `LexicalStubReranker`, a crude word-overlap scorer, so every question still runs.
"""
import sys, pathlib, re, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))
from gemini_utils import *   # embed_texts, embed_query, generate_json, np ...

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
LABELED = [("error E4012", 2), ("what does E5001 mean", 8), ("I want my money back", 1),
           ("lost access to my profile", 6), ("how do I stop paying every month", 4),
           ("change the app theme to dark", 7), ("when will I receive my bill by email", 5),
           ("extra security step at sign in", 3)]


# region helper
def _words(t): return set(re.findall(r"[a-z0-9]+", t.lower()))


class LexicalStubReranker:
    """Offline stand-in: scores a (query, doc) PAIR by word overlap. Same interface as the real rerankers."""
    name = "lexical-stub"

    def score(self, query: str, docs: list[str]) -> list[float]:
        q = _words(query)
        return [len(q & _words(d)) / (len(q) or 1) for d in docs]


class CrossEncoderReranker:
    """Local cross-encoder: reads query + passage TOGETHER and outputs one relevance score (a logit)."""
    name = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder          # imported lazily: heavy (torch)
        self.model = CrossEncoder(model_name)

    def score(self, query: str, docs: list[str]) -> list[float]:
        pairs = [(query, d) for d in docs]                       # ORDER MATTERS: (query, passage)
        return [float(s) for s in self.model.predict(pairs)]


def best_available_reranker():
    try:
        return CrossEncoderReranker()
    except Exception as e:                                       # no package, no network, no disk...
        print(f"[cross-encoder unavailable: {type(e).__name__}] using LexicalStubReranker for the demo")
        return LexicalStubReranker()


def mrr(rankings, truth):
    return round(float(np.mean([1 / (r.index(t) + 1) if t in r else 0.0 for r, t in zip(rankings, truth)])), 3)
# endregion helper


# region q1
# --------------------------------------------------------------------------
# Q1. Warm-up: why two stages? A cost model for bi-encoder vs cross-encoder
# --------------------------------------------------------------------------
def q1():
    n_docs, n_candidates = 1_000_000, 50
    bi_ms_per_query = 20.0          # ANN search over precomputed vectors (order of magnitude)
    cross_ms_per_pair = 8.0         # one cross-encoder forward pass on CPU (order of magnitude)

    all_pairs_ms = n_docs * cross_ms_per_pair
    two_stage_ms = bi_ms_per_query + n_candidates * cross_ms_per_pair
    print(f"cross-encoder on ALL {n_docs:,} docs : {all_pairs_ms / 1000 / 3600:,.1f} hours per query")
    print(f"two-stage (ANN top {n_candidates} + rerank): {two_stage_ms:,.0f} ms per query")
    print(f"speed-up: {all_pairs_ms / two_stage_ms:,.0f}x")
    print("\nbi-encoder : embeds each doc ONCE (offline); query = 1 embedding + vector search")
    print("cross-enc. : no precomputation possible; must read (query, doc) together for EVERY candidate")
    return two_stage_ms
# endregion q1


# region q2
# --------------------------------------------------------------------------
# Q2. Apply: an LLM-as-reranker with gemini-2.5-flash (listwise, one call, JSON scores)
# --------------------------------------------------------------------------
def llm_rerank(query: str, passages: list[str]) -> list[float]:
    numbered = "\n".join(f"[{i}] {p}" for i, p in enumerate(passages))
    prompt = (
        "You are a search relevance judge.\n"
        f"Question: {query}\n\nPassages:\n{numbered}\n\n"
        "Rate how well each passage answers the question from 0 (irrelevant) to 10 (directly answers it). "
        'Return ONLY JSON like {"scores": [<number for [0]>, <number for [1]>, ...]} with exactly '
        f"{len(passages)} numbers, in order."
    )
    data = generate_json(prompt, temperature=0.0)
    scores = data["scores"] if isinstance(data, dict) else data
    if len(scores) != len(passages):                              # never trust the model's list length
        raise ValueError(f"expected {len(passages)} scores, got {len(scores)}")
    return [float(s) for s in scores]


def q2():
    query = "I want my money back"
    D = embed_texts(DOCS, dim=DIM)
    q = embed_query(query, dim=DIM)
    cand = [int(i) for i in np.argsort(-(D @ q))[:6]]                       # stage 1: dense top 6
    print("stage 1 (dense) :", cand)
    scores = llm_rerank(query, [DOCS[i] for i in cand])                      # stage 2: ONE Gemini call
    order = [cand[j] for j in np.argsort(-np.array(scores), kind="stable")]
    print("LLM scores      :", scores)
    print("stage 2 (rerank):", order, "->", DOCS[order[0]][:60])
    return order
# endregion q2


# region q3
# --------------------------------------------------------------------------
# Q3. Build: retrieve -> rerank pipeline with a pluggable reranker, evaluated before/after
# --------------------------------------------------------------------------
def retrieve_then_rerank(query, D, docs, reranker, n_candidates=6, top_n=3, dim=DIM):
    q = embed_query(query, dim=dim)
    cand = [int(i) for i in np.argsort(-(D @ q))[:n_candidates]]             # stage 1: cheap, recall-oriented
    scores = reranker.score(query, [docs[i] for i in cand])                  # stage 2: precise, on few docs
    order = [cand[j] for j in np.argsort(-np.array(scores), kind="stable")]  # DESCENDING: higher = more relevant
    return order[:top_n], cand


def q3():
    D = embed_texts(DOCS, dim=DIM)
    rr = best_available_reranker()
    truth = [t for _, t in LABELED]
    before, after = [], []
    for query, _ in LABELED:
        top, cand = retrieve_then_rerank(query, D, DOCS, rr, n_candidates=6, top_n=6)
        before.append(cand); after.append(top)
    print(f"reranker: {rr.name}")
    print(f"MRR  stage-1 only: {mrr(before, truth)}   after rerank: {mrr(after, truth)}")
    return mrr(before, truth), mrr(after, truth)
# endregion q3


# region q4
# --------------------------------------------------------------------------
# Q4. Debug: four classic reranking mistakes (pure Python)
# --------------------------------------------------------------------------
def q4():
    query = "payment gateway error"
    cands = [DOCS[1], DOCS[2], DOCS[9]]                       # truth is DOCS[2]
    rr = LexicalStubReranker()
    scores = np.array(rr.score(query, cands))
    print("scores:", np.round(scores, 2), "for", [c[:20] for c in cands])

    # BUG 1: sorting ascending puts the WORST first
    bad = [cands[i][:20] for i in np.argsort(scores)]
    good = [cands[i][:20] for i in np.argsort(-scores)]
    print("bug1 ascending sort :", bad[0], "<- worst first")
    print("fix  descending sort:", good[0])

    # BUG 2: reranking a pool that does not contain the answer (pool of 1 from a weak first stage)
    pool_too_small = [DOCS[1]]
    print("bug2 pool of 1 can only return:", pool_too_small[0][:20], "-> fix: rerank 20-50 candidates")

    # BUG 3: treating cross-encoder LOGITS as probabilities with a fixed 0.5 threshold
    logits = np.array([-3.2, 4.1, -8.0])                       # typical ms-marco logits: any real number
    print("bug3 logits:", logits, "-> `> 0.5` keeps", int((logits > 0.5).sum()), "but they are not probabilities")
    probs = 1 / (1 + np.exp(-logits))
    print("fix  sigmoid(logits) :", np.round(probs, 3), " (and still calibrate a threshold on labelled data)")

    # BUG 4: swapped argument order (passage, query) silently degrades a cross-encoder
    print("bug4 pairs must be (query, passage):", [(query[:12], c[:12]) for c in cands[:1]])
    return good[0]
# endregion q4


# region q5
# --------------------------------------------------------------------------
# Q5. Production: how many candidates should you rerank? (quality vs latency, simulated)
# --------------------------------------------------------------------------
def q5(n_queries=300, n_docs=500, seed=0):
    rng = np.random.default_rng(seed)
    cost_ms_per_pair = 8.0                                     # CPU cross-encoder, order of magnitude
    results = {}
    for n_cand in (5, 10, 20, 50, 100):
        in_pool, rr_ranks = 0, []
        for _ in range(n_queries):
            # stage-1 (bi-encoder) is noisy: the true doc's rank is heavy-tailed (often 1-5, sometimes 40+)
            true_rank = int(min(n_docs, max(1, rng.lognormal(mean=1.2, sigma=1.1))))
            if true_rank <= n_cand:
                in_pool += 1
                # cross-encoder is far more accurate: true doc usually ends up 1st within the pool
                new_rank = 1 if rng.random() < 0.8 else int(rng.integers(2, 4))
                rr_ranks.append(1 / new_rank)
            else:
                rr_ranks.append(0.0)                           # can never be recovered: not in the pool
        results[n_cand] = (in_pool / n_queries, float(np.mean(rr_ranks)), n_cand * cost_ms_per_pair)

    print(f"{'candidates':>10} {'recall@pool':>12} {'MRR after':>10} {'rerank ms':>10}")
    for n_cand, (rec, m, ms) in results.items():
        print(f"{n_cand:>10} {rec:>12.2f} {m:>10.2f} {ms:>10.0f}")
    return results
# endregion q5


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3", "4", "5"]
    for n in which:
        print(f"\n===== Q{n} =====")
        globals()[f"q{n}"]()
