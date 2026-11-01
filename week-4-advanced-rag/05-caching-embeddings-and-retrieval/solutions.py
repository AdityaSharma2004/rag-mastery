"""
Week 4 · Topic 05 - Caching embeddings and retrieval: solutions to practice.md (Q1..Q5).

Run everything:   python solutions.py
OFFLINE: embeddings/LLM are counted fakes so you can SEE how many "API calls" the cache saves. Redis is optional:
set REDIS_URL (e.g. redis://localhost:6379/0) to use a real server; otherwise the in-process `fakeredis` package (same API)
is used, and if that is missing too, a plain dict cache. With GEMINI_API_KEY the real helpers can be plugged in at the
two marked places (RAW_EMBED / RAW_LLM).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))

import hashlib, json, os, re, threading, time

import numpy as np

_STOP = set("a an the is are of to in on for and or my i me do does how what when it be at can with from by up so as that this you your".split())


def _toy_vec(text, dim=256):
    v = np.zeros(dim, dtype=np.float32)
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if w in _STOP:
            continue
        w = w[:-1] if len(w) > 3 and w.endswith("s") else w
        v[int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1.0
    n = np.linalg.norm(v)
    return v / n if n else v


class Counter:
    """Counts the 'API calls' that actually happen (cache misses)."""
    def __init__(self):
        self.embed_calls = self.embed_texts = self.llm_calls = 0


COUNT = Counter()


def RAW_EMBED(texts):                      # <- replace with embed_texts(texts, task_type="RETRIEVAL_DOCUMENT") for Gemini
    COUNT.embed_calls += 1
    COUNT.embed_texts += len(texts)
    return np.vstack([_toy_vec(t) for t in texts])


def RAW_LLM(prompt):                       # <- replace with generate(prompt) for Gemini
    COUNT.llm_calls += 1
    time.sleep(0.01)                       # pretend latency
    return "ANSWER[" + prompt[:40] + "]"


def make_redis():
    url = os.getenv("REDIS_URL")
    try:
        if url:
            import redis
            r = redis.Redis.from_url(url)
            r.ping()
            return r, "redis server"
        import fakeredis
        return fakeredis.FakeRedis(), "fakeredis (in-process)"
    except Exception:
        return None, "dict fallback"


# region q1
# --- Q1: cache keys and a TTL cache with hit/miss statistics --------------------
def make_key(*parts):
    """Stable key from everything that changes the result. Hashing keeps keys short and safe."""
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()


class TTLCache:
    def __init__(self, ttl_seconds=60, clock=time.monotonic):
        self.ttl, self.clock, self.data, self.hits, self.misses = ttl_seconds, clock, {}, 0, 0

    def get(self, key):
        item = self.data.get(key)
        if item and item[0] > self.clock():
            self.hits += 1
            return item[1]
        self.data.pop(key, None)             # expired or absent
        self.misses += 1
        return None

    def set(self, key, value):
        self.data[key] = (self.clock() + self.ttl, value)

    def stats(self):
        total = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses, "hit_rate": round(self.hits / total, 2) if total else 0.0}


def q1():
    now = [0.0]
    cache = TTLCache(ttl_seconds=10, clock=lambda: now[0])
    k = make_key("gemini-embedding-001", "RETRIEVAL_DOCUMENT", 768, "hello")
    print("key length:", len(k), "| same inputs same key:", k == make_key("gemini-embedding-001", "RETRIEVAL_DOCUMENT", 768, "hello"))
    print("different task type -> different key:", k != make_key("gemini-embedding-001", "RETRIEVAL_QUERY", 768, "hello"))
    print("get before set:", cache.get(k))
    cache.set(k, "vector")
    print("get right after set:", cache.get(k))
    now[0] = 11.0
    print("get after TTL expiry:", cache.get(k))
    print(cache.stats())
# endregion q1


# region q2
# --- Q2: an embedding cache that embeds ONLY the misses, in one batch ------------
class EmbeddingCache:
    def __init__(self, raw_embed, model="gemini-embedding-001", task="RETRIEVAL_DOCUMENT", dim=256):
        self.raw, self.model, self.task, self.dim = raw_embed, model, task, dim
        self.store = {}                        # key -> vector (swap for Redis / disk, see Q5)

    def key(self, text):
        return make_key(self.model, self.task, self.dim, text)

    def embed(self, texts):
        keys = [self.key(t) for t in texts]
        missing = [i for i, k in enumerate(keys) if k not in self.store]
        uniq = list(dict.fromkeys(texts[i] for i in missing))      # dedupe inside the batch too
        if uniq:
            for t, v in zip(uniq, self.raw(uniq)):                 # ONE call for all misses
                self.store[self.key(t)] = v
        return np.vstack([self.store[k] for k in keys])


def q2():
    COUNT.__init__()
    cache = EmbeddingCache(RAW_EMBED)
    docs = ["refund policy", "shipping policy", "refund policy", "warranty terms"]
    cache.embed(docs)
    print(f"first pass : calls={COUNT.embed_calls} texts sent={COUNT.embed_texts} (4 requested, 1 duplicate)")
    cache.embed(docs + ["new policy"])
    print(f"second pass: calls={COUNT.embed_calls} texts sent={COUNT.embed_texts} (only 'new policy' was missing)")
    cache.embed(docs)
    print(f"third pass : calls={COUNT.embed_calls} texts sent={COUNT.embed_texts} (all hits, zero API calls)")
# endregion q2


# region q3
# --- Q3: cached answers with corpus-version invalidation -------------------------
class CachedRAG:
    def __init__(self, ttl=300):
        self.version = 1                       # bump when documents change -> all old keys become unreachable
        self.cache = TTLCache(ttl)
        self.docs = {"refund": "Refunds are possible within 14 days."}

    def update_docs(self, new_docs):
        self.docs.update(new_docs)
        self.version += 1                      # cheap, safe invalidation: no need to delete anything

    def ask(self, question, k=3):
        key = make_key("answer", self.version, k, question.strip().lower())
        cached = self.cache.get(key)
        if cached is not None:
            return cached + " (cached)"
        answer = RAW_LLM(question + " | " + " ".join(self.docs.values()))
        self.cache.set(key, answer)
        return answer


def q3():
    COUNT.__init__()
    rag = CachedRAG()
    rag.ask("How long is the refund window?")
    rag.ask("how long is the refund window?  ")        # same after normalisation: hit
    print("LLM calls after 2 identical questions:", COUNT.llm_calls)
    rag.update_docs({"refund": "Refunds are possible within 30 days."})
    rag.ask("How long is the refund window?")
    print("LLM calls after the documents changed:", COUNT.llm_calls, "(version bump forced a fresh answer)")
# endregion q3


# region q4
# --- Q4: two silent cache bugs: cross-tenant leak and stale answers --------------
class BuggyCache:
    def __init__(self):
        self.d = {}

    def ask(self, tenant, question, docs):
        key = question.lower()                    # BUG: key ignores tenant (and docs version, k, model, ...)
        if key not in self.d:
            self.d[key] = f"[{tenant}] " + RAW_LLM(question + " ".join(docs))
        return self.d[key]


class FixedCache:
    def __init__(self):
        self.d, self.versions = {}, {}

    def ask(self, tenant, question, docs):
        version = hashlib.sha256(" ".join(docs).encode()).hexdigest()[:8]    # content hash = automatic invalidation
        key = make_key(tenant, version, question.strip().lower())
        if key not in self.d:
            self.d[key] = f"[{tenant}] " + RAW_LLM(question + " ".join(docs))
        return self.d[key]


def q4():
    for name, cache in (("buggy", BuggyCache()), ("fixed", FixedCache())):
        COUNT.__init__()
        cache.ask("acme", "what is our discount code?", ["acme code is SAVE10"])
        b = cache.ask("globex", "what is our discount code?", ["globex code is VIP20"])
        before = COUNT.llm_calls
        cache.ask("acme", "what is our discount code?", ["acme code is SAVE15 (updated)"])
        print(f"{name}: globex is served by {b.split(']')[0]}] | recomputed after acme's docs changed: {COUNT.llm_calls > before}")
# endregion q4


# region q5
# --- Q5: Redis-backed semantic cache + single-flight (cache stampede control) -----
class SemanticCache:
    """Exact-match first, then 'close enough' questions via cosine similarity. Stored in Redis (or a dict)."""
    def __init__(self, threshold=0.85, ttl=3600):
        self.r, self.backend = make_redis()
        self.mem = {}                                          # used only when Redis is unavailable
        self.threshold, self.ttl = threshold, ttl
        self.locks, self.guard = {}, threading.Lock()
        self.hits_exact = self.hits_semantic = self.misses = 0

    def _get(self, k):
        if self.r:
            v = self.r.get(k)
            return v.decode() if v else None
        return self.mem.get(k)

    def _set(self, k, v):
        if self.r:
            self.r.set(k, v, ex=self.ttl)
        else:
            self.mem[k] = v

    def _index(self):
        raw = self._get("sem:index")
        return json.loads(raw) if raw else []

    def ask(self, tenant, question, compute):
        norm = question.strip().lower()
        exact = f"ans:{tenant}:{make_key(norm)}"
        if (v := self._get(exact)) is not None:
            self.hits_exact += 1
            return v
        qv = _toy_vec(norm)
        for item in self._index():                              # semantic lookup (tenant-scoped!)
            if item["tenant"] == tenant and float(qv @ np.array(item["vec"], dtype=np.float32)) >= self.threshold:
                self.hits_semantic += 1
                return self._get(item["key"])
        with self.guard:                                        # single-flight: one lock per key
            lock = self.locks.setdefault(exact, threading.Lock())
        with lock:
            if (v := self._get(exact)) is not None:             # someone else computed it while we waited
                self.hits_exact += 1
                return v
            self.misses += 1
            value = compute(question)
            self._set(exact, value)
            idx = self._index()
            idx.append({"tenant": tenant, "key": exact, "vec": [round(float(x), 4) for x in qv]})
            self._set("sem:index", json.dumps(idx))
            return value


def q5():
    COUNT.__init__()
    sc = SemanticCache()
    print("backend:", sc.backend)
    llm = lambda q: RAW_LLM(q)
    stream = ["How long is the refund window?", "how long is the refund window", "refund window how long?",
              "What does the Growth plan cost?", "Growth plan cost?", "How long is the refund window?"]
    for q in stream:
        sc.ask("acme", q, llm)
    print(f"stream of {len(stream)} questions -> exact hits={sc.hits_exact} semantic hits={sc.hits_semantic} LLM calls={COUNT.llm_calls}")
    sc.ask("globex", "How long is the refund window?", llm)
    print("same question, other tenant -> LLM calls:", COUNT.llm_calls, "(tenant-scoped: no cross-tenant hit)")
    # stampede: 10 threads ask the same NEW question at once; only ONE should call the LLM
    before = COUNT.llm_calls
    ts = [threading.Thread(target=lambda: sc.ask("acme", "Where do I download my invoices?", llm)) for _ in range(10)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    print("10 concurrent identical requests -> LLM calls:", COUNT.llm_calls - before)
# endregion q5


if __name__ == "__main__":
    for fn in (q1, q2, q3, q4, q5):
        print(f"\n===== {fn.__name__} =====")
        fn()
