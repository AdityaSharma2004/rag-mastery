"""
Week 4 · Topic 02 - HyDE (Hypothetical Document Embeddings): solutions to practice.md (Q1..Q5).

Run everything:   python solutions.py
OFFLINE by default: a toy embedder and a toy "hypothetical answer writer" (a small lookup table that plays the part of
gemini-2.5-flash) let you watch the mechanics. With GEMINI_API_KEY set, the same code uses gemini-embedding-001 and
gemini-2.5-flash. The toy writer was hand-written for these queries, so its big gains are an ARTEFACT; real gains are
smaller and vary by corpus. Expected outputs in practice.md are from the offline run.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))

import hashlib, os, re

import numpy as np

CORPUS = [
    "Queued jobs run in the background through a queue worker; start one with php artisan queue:work, and failed jobs are stored in the failed_jobs table.",   # 0
    "Use Cache::remember to store the result of an expensive query for a number of seconds, so repeated requests skip the database.",                          # 1
    "Migrations are version control for your database schema; run them with php artisan migrate and roll back with migrate:rollback.",                          # 2
    "Form request classes validate incoming data with rules such as required, email and max, and return errors to the user automatically.",                    # 3
    "Middleware filters HTTP requests entering your application, for example to check that a user is authenticated before a route runs.",                       # 4
    "Policies organise authorisation logic around a model, letting you decide whether a user may update or delete a post.",                                    # 5
    "Eloquent relationships such as hasMany and belongsTo let you load related models; use eager loading with the with method to avoid the N+1 query problem.", # 6
    "Configuration values that differ per environment live in the .env file and are read through the config helper, never directly with env outside config files.",  # 7
    "The task scheduler lets you define scheduled jobs in code and needs only a single cron entry that runs php artisan schedule:run every minute.",           # 8
    "Laravel writes logs through channels such as single, daily and slack, configured in config/logging.php.",                                                 # 9
    "Rate limiting uses the RateLimiter and the throttle middleware to restrict how many requests a client can make per minute.",                              # 10
    "The Storage facade provides a filesystem abstraction so the same code can save files to local disk or Amazon S3.",                                        # 11
]

GOLD = {   # vague user phrasing -> the chunk that answers it
    "my page makes hundreds of database calls": {6},
    "run something later without making the user wait": {0},
    "stop people from hammering my API": {10},
    "keep passwords and keys out of git, different per server": {7},
    "run a task every night automatically": {8},
    "only let the author edit their own article": {5},
    "check input before saving the form": {3},
    "remember an answer so I do not ask the database again": {1},
    "php artisan migrate to update the database schema": {2},      # already well phrased: HyDE not needed
}

_STOP = set("a an the is are of to in on for and or my i me do does how what when it be at can with from by up under so as that this".split())


def toy_embed(texts, dim=512):
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in re.findall(r"[a-z0-9_]+", t.lower()):
            if w in _STOP:
                continue
            w = w[:-1] if len(w) > 3 and w.endswith("s") else w
            w = w[:-2] if len(w) > 5 and w.endswith("ed") else w
            out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1.0
    n = np.linalg.norm(out, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return out / n


# The toy "LLM": writes a documentation-style answer for a user question (paraphrased, not copied from the corpus).
TOY_ANSWERS = {
    "hundreds": "Load related models with eager loading using the with method in Eloquent, so relationships come in one query and the N+1 query problem disappears.",
    "later": "Dispatch the work as a queued job; a queue worker processes it in the background, and any failed jobs are recorded in the failed_jobs table.",
    "hammering": "Protect the endpoint with rate limiting: the throttle middleware restricts how many requests each client may make per minute.",
    "git": "Keep configuration that differs per environment in the .env file and read it through the config helper instead of hard-coding secrets.",
    "night": "Use the task scheduler to define a scheduled job; a single cron entry runs php artisan schedule:run each minute.",
    "author": "Write a policy for the model that decides whether a user may update or delete the post, which handles authorisation.",
    "input": "Validate incoming data with a form request class and rules such as required and email; errors go back to the user.",
    "remember": "Wrap the expensive query in Cache::remember so the result is stored and repeated requests skip the database.",
}


def toy_hypothetical(query, n=1):
    q = query.lower()
    for key, ans in TOY_ANSWERS.items():
        if key in q:
            return [ans] * n
    return [query] * n


if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
    from gemini_utils import embed_texts, generate

    def EMBED_DOCS(texts):
        return embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")

    def EMBED_QUERIES(texts):
        return embed_texts(texts, task_type="RETRIEVAL_QUERY")

    def HYPOTHETICAL(query, n=1):
        prompt = ("Write a short passage (2-3 sentences) from technical documentation that would answer the question. "
                  "It is fine if some details are imprecise; use the vocabulary the documentation would use.\n\n"
                  f"Question: {query}\nPassage:")
        return [generate(prompt, temperature=0.7 if n > 1 else 0.3, thinking_budget=0) for _ in range(n)]
    MODE = "gemini"
else:
    EMBED_DOCS = toy_embed
    EMBED_QUERIES = toy_embed
    HYPOTHETICAL = toy_hypothetical
    MODE = "offline toys"

DOC_VECS = EMBED_DOCS(CORPUS)


def search_vec(vec, k=5, min_score=0.01):
    scores = DOC_VECS @ vec
    return [int(i) for i in np.argsort(-scores)[:k] if scores[i] > min_score]


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n else v


def recall_at_k(ranked, rel, k):
    return len(set(ranked[:k]) & rel) / len(rel)


def evaluate(strategy, k=3):
    rec, rr = [], []
    for q, rel in GOLD.items():
        ranked = strategy(q)
        rec.append(recall_at_k(ranked, rel, k))
        rr.append(next((1 / p for p, d in enumerate(ranked, 1) if d in rel), 0.0))
    return round(float(np.mean(rec)), 2), round(float(np.mean(rr)), 2)


def baseline(q, k=5):
    return search_vec(EMBED_QUERIES([q])[0], k)


# region q1
# --- Q1: questions and answers live in different places ------------------------
def q1():
    print(f"mode: {MODE}")
    print(f"{'query':<52} sim(query, gold)  sim(hypothetical, gold)")
    for q, rel in list(GOLD.items())[:4]:
        gold_vec = DOC_VECS[next(iter(rel))]
        qv = EMBED_QUERIES([q])[0]
        hv = EMBED_DOCS(HYPOTHETICAL(q))[0]
        print(f"{q:<52} {float(qv @ gold_vec):>10.2f} {float(hv @ gold_vec):>17.2f}")
# endregion q1


# region q2
# --- Q2: HyDE retrieval: embed a hypothetical ANSWER instead of the question ---
def hyde_search(query, k=5):
    hypo = HYPOTHETICAL(query, n=1)[0]
    vec = EMBED_DOCS([hypo])[0]                  # DOCUMENT task type: it should look like a document, not a question
    return search_vec(vec, k)


def q2():
    print("baseline  recall@3, mrr:", evaluate(baseline))
    print("HyDE      recall@3, mrr:", evaluate(hyde_search))
    print("example:", GOLD and list(GOLD)[0])
    print("  hypothetical answer:", HYPOTHETICAL(list(GOLD)[0])[0])
# endregion q2


# region q3
# --- Q3: several hypotheses + blending with the original query ------------------
def hyde_blend(query, alpha=0.3, n=3, k=5):
    """vec = normalise(alpha * query_vec + (1 - alpha) * mean(hypothesis_vecs))"""
    qv = EMBED_QUERIES([query])[0]
    hv = EMBED_DOCS(HYPOTHETICAL(query, n=n))
    return search_vec(unit(alpha * qv + (1 - alpha) * hv.mean(axis=0)), k)


def q3():
    print("alpha  recall@3  mrr     (alpha = weight of the ORIGINAL query vector)")
    for a in (0.0, 0.3, 0.5, 0.8, 1.0):
        r, m = evaluate(lambda q, a=a: hyde_blend(q, alpha=a))
        print(f"{a:>4}   {r:>6}   {m:>5}")
# endregion q3


# region q4
# --- Q4: HyDE hurt retrieval - a hallucinated answer; add a confidence gate ------
def bad_hypothetical(query, n=1):
    """A hallucinating writer: confident answer about the WRONG framework."""
    return ["In Django, define a custom management command and register it with the manage.py utility and settings module."] * n


def hyde_with_gate(query, writer, threshold=0.25, alpha=0.5, k=5):
    """Use HyDE only when the plain query looks weak (low top-1 score); blend so the question still votes."""
    qv = EMBED_QUERIES([query])[0]
    top1 = float((DOC_VECS @ qv).max())
    if top1 >= threshold:
        return search_vec(qv, k)                                  # query already matches well: skip the LLM
    hv = EMBED_DOCS(writer(query, 1))[0]
    return search_vec(unit(alpha * qv + (1 - alpha) * hv), k)


def q4():
    print("baseline                 :", evaluate(baseline))
    bad = evaluate(lambda q: search_vec(EMBED_DOCS(bad_hypothetical(q))[0]))
    print("pure HyDE, bad writer    :", bad)
    guarded = evaluate(lambda q: hyde_with_gate(q, bad_hypothetical, threshold=0.0, alpha=0.5))
    print("blend(0.5), bad writer   :", guarded)
    good = evaluate(lambda q: hyde_with_gate(q, toy_hypothetical, threshold=0.45, alpha=0.3))
    print("gate+blend, good writer  :", good)
# endregion q4


# region q5
# --- Q5: a production policy: gate, cache, fall back on LLM failure -------------
class HydeRetriever:
    def __init__(self, writer, threshold=0.45, alpha=0.3):
        self.writer, self.threshold, self.alpha = writer, threshold, alpha
        self.cache, self.llm_calls, self.fallbacks = {}, 0, 0

    def hypothesis(self, q):
        if q not in self.cache:
            self.llm_calls += 1
            self.cache[q] = self.writer(q, 1)[0]
        return self.cache[q]

    def retrieve(self, q, k=5):
        qv = EMBED_QUERIES([q])[0]
        if float((DOC_VECS @ qv).max()) >= self.threshold:
            return search_vec(qv, k)                               # confident: no LLM call
        try:
            hv = EMBED_DOCS([self.hypothesis(q)])[0]
        except Exception:                                         # 429 / timeout / blocked: degrade gracefully
            self.fallbacks += 1
            return search_vec(qv, k)
        return search_vec(unit(self.alpha * qv + (1 - self.alpha) * hv), k)


def flaky_writer(query, n=1):
    if "night" in query:
        raise TimeoutError("simulated LLM timeout")
    return toy_hypothetical(query, n)


def q5():
    r = HydeRetriever(toy_hypothetical)
    print("normal  : recall@3, mrr =", evaluate(r.retrieve), "| LLM calls:", r.llm_calls, "of", len(GOLD), "queries")
    evaluate(r.retrieve)
    print("2nd pass: LLM calls total:", r.llm_calls, "(cached)")
    f = HydeRetriever(flaky_writer)
    print("flaky   : recall@3, mrr =", evaluate(f.retrieve), "| fallbacks:", f.fallbacks)
# endregion q5


if __name__ == "__main__":
    for fn in (q1, q2, q3, q4, q5):
        print(f"\n===== {fn.__name__} =====")
        fn()
