"""
Week 4 · Topic 01 - Query expansion: solutions to practice.md (Q1..Q5).

Run everything:   python solutions.py
OFFLINE by default: a toy embedder (stop-word-free, crudely stemmed hashed words) and a toy "query rewriter" (a hand-made
synonym table that plays the role of the LLM) let you watch the *mechanics* (variants -> embed -> search -> RRF -> metrics).
With GEMINI_API_KEY set, the same code uses gemini-embedding-001 and gemini-2.5-flash (generate_json) instead.
The expected outputs in practice.md come from the offline run; real-model numbers differ.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))

import hashlib, json, os, re, tempfile

import numpy as np

CORPUS = [
    "Time to First Byte (TTFB) above 600 ms usually means the origin server is slow; enable server-side caching and check PHP-FPM worker saturation.",  # 0
    "To reduce latency for visitors in Asia, attach the CDN add-on and select the Singapore edge region.",                                              # 1
    "Invoices are generated on the 1st of each month and can be downloaded as PDF from Billing > Invoices.",                                           # 2
    "A 502 Bad Gateway error means the upstream PHP-FPM process crashed or timed out; restart the service from the Control Panel.",                    # 3
    "Daily automatic backups are kept for 14 days; restore a backup from Backups > Restore, which takes about 10 minutes.",                            # 4
    "To add a custom domain, create a CNAME record pointing to sites.acmecloud.example and then verify ownership in Domains.",                         # 5
    "Two-factor authentication can be enforced for all team members under Security > Authentication.",                                                 # 6
    "Autoscaling adds an instance when average CPU stays above 70% for five minutes; the maximum is set per plan.",                                    # 7
    "Database connections are capped at 150 per plan; exceeding this causes 'Too many connections' errors in MySQL.",                                  # 8
    "You can cancel your subscription anytime; unused credit is refunded pro rata within 7 business days.",                                            # 9
    "Free SSL certificates are issued by Let's Encrypt and renew automatically 30 days before expiry.",                                                # 10
    "Slow MySQL queries are logged when they exceed 2 seconds; view them under Databases > Slow Query Log.",                                           # 11
    "Object storage buckets are private by default; make a file public with a signed URL valid for up to 7 days.",                                     # 12
    "Support tickets are answered within 4 hours on Business plans and 24 hours on Starter plans.",                                                    # 13
]

GOLD = {   # messy user query -> ids of chunks that answer it
    "my website feels sluggish": {0, 1, 11},
    "site down with gateway error": {3},
    "how do I get my money back": {9},
    "stop my site crashing when traffic spikes": {7, 8},
    "browser shows a not-secure padlock warning": {10},
    "can I make my team log in more safely": {6},
}

# --- offline stand-ins ---------------------------------------------------------
_STOP = set("a an the is are of to in on for and or my i me do does how what when it be at can with from by up under".split())


def toy_embed(texts, dim=512):
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in re.findall(r"[a-z0-9]+", t.lower()):
            if w in _STOP:
                continue
            w = w[:-1] if len(w) > 3 and w.endswith("s") else w
            w = w[:-2] if len(w) > 5 and w.endswith("ed") else w
            out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1.0
    n = np.linalg.norm(out, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return out / n


# The "LLM": maps symptom words to technical vocabulary, the way a good rewriter would.
SYNONYMS = {
    "sluggish": ["slow ttfb latency", "server caching response", "cdn edge region", "mysql slow query log"],
    "gateway": ["502 bad gateway", "php-fpm crashed restart service", "upstream timed out"],
    "down": ["502 error service restart", "php-fpm crashed"],
    "money": ["refund subscription cancel", "refunded pro rata"],
    "crashing": ["autoscaling instance cpu", "database connections capped mysql", "too many connections"],
    "traffic": ["autoscaling cpu average", "connections plan"],
    "padlock": ["ssl certificate let's encrypt", "certificate renew expiry"],
    "not-secure": ["ssl certificate", "https certificate expiry"],
    "safely": ["two-factor authentication security", "enforced authentication team members"],
    "log": ["authentication security"],
}


def toy_expand(query, n=4):
    words = re.findall(r"[a-z\-]+", query.lower())
    hits = [w for w in words if w in SYNONYMS]
    if not hits:
        return [query + " documentation"]
    return [" ".join(SYNONYMS[w][j % len(SYNONYMS[w])] for w in hits) for j in range(n)]


if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
    from gemini_utils import embed_texts, generate_json

    def EMBED_DOCS(texts):
        return embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")

    def EMBED_QUERIES(texts):
        return embed_texts(texts, task_type="RETRIEVAL_QUERY")

    def EXPAND(query, n=4):
        prompt = (f"Write {n} different search queries (different vocabulary: symptom, technical cause, product names, "
                  f'formal docs style) that could retrieve documents answering this question. Return JSON {{"queries": [...]}}.'
                  f"\n\nQuestion: {query}")
        data = generate_json(prompt, system="You rewrite search queries for a hosting support knowledge base.", temperature=0.4)
        return [v for v in (data["queries"] if isinstance(data, dict) else data) if isinstance(v, str)]
    MODE = "gemini"
else:
    EMBED_DOCS = toy_embed
    EMBED_QUERIES = toy_embed
    EXPAND = toy_expand
    MODE = "offline toys"

DOC_VECS = EMBED_DOCS(CORPUS)


def search_one(qvec, k=5, min_score=0.01):
    """Top-k chunk ids by cosine (unit vectors). Chunks scoring ~0 are not 'matches', so they are not returned."""
    scores = DOC_VECS @ qvec
    return [int(i) for i in np.argsort(-scores)[:k] if scores[i] > min_score]


def recall_at_k(ranked, relevant, k):
    return len(set(ranked[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranked, relevant):
    for pos, d in enumerate(ranked, 1):
        if d in relevant:
            return 1.0 / pos
    return 0.0


def evaluate(strategy, k=3):
    rec, rr, per = [], [], {}
    for q, rel in GOLD.items():
        ranked = strategy(q)
        per[q] = recall_at_k(ranked, rel, k)
        rec.append(per[q])
        rr.append(reciprocal_rank(ranked, rel))
    return {"recall@k": float(np.mean(rec)), "mrr": float(np.mean(rr)), "per_query": per}


def baseline(query, k=5):
    return search_one(EMBED_QUERIES([query])[0], k)


# region q1
# --- Q1: Reciprocal Rank Fusion from scratch ----------------------------------
def rrf(rank_lists, k=60, top=5):
    """score(d) = sum over lists of 1 / (k + rank), rank starting at 1. Ties broken by id (deterministic)."""
    scores = {}
    for lst in rank_lists:
        for rank, doc_id in enumerate(lst, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda d: (-scores[d], d))[:top], scores


def q1():
    lists = [[3, 7, 1], [7, 3, 9], [7, 5, 3]]
    ranked, scores = rrf(lists, top=5)
    print("fused ranking:", ranked)
    for d in ranked:
        print(f"  doc {d}: {scores[d]:.5f}")
    for k in (1, 60, 1000):
        print(f"k={k:<4} ->", rrf(lists, k=k, top=3)[0])
# endregion q1


# region q2
# --- Q2: the baseline - one query, one vector, metrics ------------------------
def q2():
    res = evaluate(baseline, k=3)
    print(f"mode: {MODE}")
    print(f"baseline recall@3={res['recall@k']:.2f} mrr={res['mrr']:.2f}")
    for q, r in res["per_query"].items():
        print(f"  recall@3={r:.2f}  {q}")
# endregion q2


# region q3
# --- Q3: multi-query retrieval with the 'suggestion box' rules ----------------
def normalize(s):
    return re.sub(r"\W+", " ", s.lower()).strip()


def clean_variants(original, variants, max_variants=5):
    """Original first (the anchor), then unique variants, capped. The LLM suggests; we enforce."""
    seen, out = {normalize(original)}, [original]
    for v in variants:
        key = normalize(v)
        if key and key not in seen:
            seen.add(key)
            out.append(v.strip())
    return out[: max_variants + 1]


def multi_query(query, expand=None, n=4, per_query_k=5, top=5):
    expand = expand or EXPAND
    queries = clean_variants(query, expand(query, n))
    vecs = EMBED_QUERIES(queries)                          # ONE batched embedding call for all variants
    return rrf([search_one(v, per_query_k) for v in vecs], top=top)[0]


def q3():
    base, multi = evaluate(baseline), evaluate(multi_query)
    print(f"baseline    recall@3={base['recall@k']:.2f} mrr={base['mrr']:.2f}")
    print(f"multi-query recall@3={multi['recall@k']:.2f} mrr={multi['mrr']:.2f}")
    for q in GOLD:
        b, m = base["per_query"][q], multi["per_query"][q]
        print(f"  {'WIN ' if m > b else 'LOSS' if m < b else 'tie '} {b:.2f} -> {m:.2f}  {q}")
# endregion q3


# region q4
# --- Q4: expansion made things worse - add an anchor weight and a drift filter -
def drifting_expand(query, n=4):
    """A bad rewriter: invents topics (what a high-temperature LLM sometimes does)."""
    return ["pricing plans and billing invoices", "support ticket response time", "custom domain cname record"]


def weighted_rrf(rank_lists, weights, k=60, top=5):
    scores = {}
    for lst, w in zip(rank_lists, weights):
        for rank, d in enumerate(lst, 1):
            scores[d] = scores.get(d, 0.0) + w / (k + rank)
    return sorted(scores, key=lambda d: (-scores[d], d))[:top]


def drift_filter(original_vec, variants, variant_vecs, min_sim=0.15):
    """Drop variants whose embedding is almost unrelated to the original question."""
    return [(v, vec) for v, vec in zip(variants, variant_vecs) if float(original_vec @ vec) >= min_sim]


def safe_multi_query(query, expand, anchor_weight=3.0, min_sim=0.0, per_query_k=5, top=5):
    variants = clean_variants(query, expand(query))[1:]
    qv = EMBED_QUERIES([query])[0]
    vv = EMBED_QUERIES(variants) if variants else []
    kept = drift_filter(qv, variants, vv, min_sim) if min_sim > 0 else list(zip(variants, vv))
    lists = [search_one(qv, per_query_k)] + [search_one(vec, per_query_k) for _, vec in kept]
    weights = [anchor_weight] + [1.0] * len(kept)
    return weighted_rrf(lists, weights, top=top), [v for v, _ in kept]


def q4():
    print("baseline          :", {k: round(v, 2) for k, v in evaluate(baseline).items() if k != "per_query"})
    naive = evaluate(lambda q: multi_query(q, expand=drifting_expand))
    print("naive + drifting  :", {k: round(v, 2) for k, v in naive.items() if k != "per_query"})
    fixed = evaluate(lambda q: safe_multi_query(q, drifting_expand, anchor_weight=3.0, min_sim=0.05)[0])
    print("anchor + filter   :", {k: round(v, 2) for k, v in fixed.items() if k != "per_query"})
    _, kept = safe_multi_query("how do I get my money back", drifting_expand, min_sim=0.05)
    print("variants kept for 'how do I get my money back':", kept)
# endregion q4


# region q5
# --- Q5: cached expansion, strategy routing and a cost report -----------------
class ExpansionService:
    def __init__(self, cache_path):
        self.path = pathlib.Path(cache_path)
        self.cache = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.llm_calls = 0

    def expand(self, query, n=4):
        key = hashlib.sha256(f"{n}|{query}".encode()).hexdigest()
        if key not in self.cache:
            self.llm_calls += 1                                   # a real (paid-in-quota) call
            self.cache[key] = EXPAND(query, n)
            self.path.write_text(json.dumps(self.cache))
        return self.cache[key]


def plan(query):
    """Cheap routing: multi-part questions are decomposed; everything else gets multi-query."""
    parts = [p.strip() for p in re.split(r"\band\b|\?|;", query) if len(p.split()) >= 3]
    return ("decompose", parts) if len(parts) >= 2 else ("multi_query", [query])


def run_all(service):
    out = {}
    for q in GOLD:
        strategy, parts = plan(q)
        queries = clean_variants(q, service.expand(q) if strategy == "multi_query" else parts)
        out[q] = rrf([search_one(v, 5) for v in EMBED_QUERIES(queries)], top=5)[0]
    return out


def q5():
    with tempfile.TemporaryDirectory() as d:
        cache = pathlib.Path(d) / "expansions.json"
        s1 = ExpansionService(cache)
        r1 = run_all(s1)
        s2 = ExpansionService(cache)                              # a fresh process: reads the cache file
        r2 = run_all(s2)
        print("run 1 LLM calls:", s1.llm_calls, "| run 2 LLM calls:", s2.llm_calls, "| same results:", r1 == r2)
    print("routing:", {q: plan(q)[0] for q in ["why is my site slow and how do I add a domain", "how do I get my money back"]})
    print("cost per query ≈ 1 LLM call + 1 batched embed call of", 1 + 4, "texts, paid once thanks to the cache")
# endregion q5


if __name__ == "__main__":
    for fn in (q1, q2, q3, q4, q5):
        print(f"\n===== {fn.__name__} =====")
        fn()
