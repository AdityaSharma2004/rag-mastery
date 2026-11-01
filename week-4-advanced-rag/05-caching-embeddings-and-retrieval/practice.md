# Practice - Caching embeddings and retrieval

Five questions, rising difficulty. Attempt each before opening the hint, and the hint before the solution. Every solution is the exact code in `solutions.py` (`python solutions.py` runs all). **Runs offline, no API key:** `RAW_EMBED` and `RAW_LLM` are counted fakes, so you can *see* how many "API calls" a cache saves (swap them for `embed_texts` / `generate` to use Gemini). Redis is optional: set `REDIS_URL` for a real server; otherwise the in-process `fakeredis` package (same API) is used, with a plain-dict fallback if that is missing as well. `make_redis`, `COUNT` (the call counter) and `_toy_vec` are defined at the top of `solutions.py`.

---

### Q1. Warm-up - cache keys and a TTL cache

**Problem**

Write `make_key(*parts)` (SHA-256 of the joined parts) and a `TTLCache(ttl_seconds, clock)` with `get`, `set` and hit/miss statistics. Show that identical inputs give the same key, that a different task type gives a different key, and that an entry expires after its TTL (use an injectable clock instead of `sleep`).

<details><summary><b>Hint</b></summary>

Store `(expires_at, value)`. In `get`, if the entry exists and `expires_at > clock()` it is a hit; otherwise delete it and count a miss.

</details>

<details><summary><b>Solution</b></summary>

```python
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
```

</details>

**Why it works / what to learn**
The key includes model, task type and dimension, so changing any of them cannot return a stale vector from another configuration. The injectable clock makes expiry testable without waiting.

**Expected output**
```text
key length: 64 | same inputs same key: True
different task type -> different key: True
get before set: None
get right after set: vector
get after TTL expiry: None
{'hits': 1, 'misses': 2, 'hit_rate': 0.33}
```

---

### Q2. Apply - an embedding cache that embeds only the misses

**Problem**

Write `EmbeddingCache.embed(texts)` that computes keys, finds missing texts, removes duplicates, calls the raw embedder **once** for all misses, stores the results and returns vectors in the original order. Use a call counter to show the effect over three passes.

<details><summary><b>Hint</b></summary>

`uniq = list(dict.fromkeys(texts[i] for i in missing))` dedupes while keeping order; return `np.vstack([store[k] for k in keys])`.

</details>

<details><summary><b>Solution</b></summary>

```python
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
```

</details>

**Why it works / what to learn**
Pass 1 requests 4 texts but sends 3 (one duplicate); pass 2 sends only the one new text; pass 3 makes no call at all. Re-running an ingestion after a crash or a deploy costs nothing for text you already embedded, which matters a lot on rate-limited free tiers.

**Expected output**
```text
first pass : calls=1 texts sent=3 (4 requested, 1 duplicate)
second pass: calls=2 texts sent=4 (only 'new policy' was missing)
third pass : calls=2 texts sent=4 (all hits, zero API calls)
```

---

### Q3. Build - answer caching with version-based invalidation

**Problem**

Build `CachedRAG.ask(question)` that caches answers under a key containing the **corpus version**, `k` and the normalised question. Provide `update_docs()` that bumps the version. Show that two differently-formatted copies of the same question cost one LLM call, and that after a document update the question is recomputed.

<details><summary><b>Hint</b></summary>

Normalise with `.strip().lower()`. Do not delete anything on update: just bump `self.version`, and the old keys are never looked up again (they expire by TTL).

</details>

<details><summary><b>Solution</b></summary>

```python
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
```

</details>

**Why it works / what to learn**
Versioned keys make invalidation O(1) and impossible to forget. Normalisation raises the hit rate. Remember to include every other input that changes the answer (prompt version, model, tenant) in real systems.

**Expected output**
```text
LLM calls after 2 identical questions: 1
LLM calls after the documents changed: 2 (version bump forced a fresh answer)
```

---

### Q4. Debug - two silent cache bugs

**Problem**

A cache keyed only by `question.lower()` is deployed in a multi-tenant app. Demonstrate (1) the **cross-tenant leak** (tenant `globex` receives tenant `acme`'s cached answer) and (2) the **stale answer** (acme's documents changed but the old answer is served). Fix both with a key that includes the tenant and a content hash of the documents used.

<details><summary><b>Hint</b></summary>

Ask: what inputs does the answer depend on? The tenant, the retrieved documents' content (hash it), and the question. All must be in the key.

</details>

<details><summary><b>Solution</b></summary>

```python
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
```

</details>

**Why it works / what to learn**
The buggy cache serves globex an acme answer (a data breach) and never recomputes after the documents change. Including `tenant` and a content hash fixes both: a changed document changes the hash, so the old entry is never hit. Always test caches with two tenants asking identical questions.

**Expected output**
```text
buggy: globex is served by [acme] | recomputed after acme's docs changed: False
fixed: globex is served by [globex] | recomputed after acme's docs changed: True
```

---

### Q5. Production - a Redis-backed semantic cache with single-flight

**Problem**

Build `SemanticCache.ask(tenant, question, compute)` that (1) tries an exact key, (2) tries a **semantic match** among earlier questions of the **same tenant** (cosine ≥ threshold), (3) otherwise computes under a **per-key lock** with a re-check (single-flight), then stores the answer in Redis with a TTL and adds the question to a semantic index. Show hit counts for a stream of paraphrases, tenant isolation, and that 10 concurrent identical requests cause one LLM call.

<details><summary><b>Hint</b></summary>

Take the lock, then check the cache again: another thread may have filled it while you waited. For multi-server setups the lock would be `redis.set(key, "1", nx=True, ex=30)`.

</details>

<details><summary><b>Solution</b></summary>

```python
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
```

</details>

**Why it works / what to learn**
Six questions (three paraphrases of one and two of another) cost only 2 LLM calls; the other tenant gets its own answer; and the stampede test shows 1 call for 10 simultaneous requests. The semantic threshold is the dangerous dial: tune it on labelled question pairs, and skip semantic reuse for questions with numbers or negations. The backend line tells you whether a real Redis server, `fakeredis` or the dict fallback was used.

**Expected output** (backend line depends on your setup)
```text
backend: fakeredis (in-process)
stream of 6 questions -> exact hits=1 semantic hits=3 LLM calls=2
same question, other tenant -> LLM calls: 3 (tenant-scoped: no cross-tenant hit)
10 concurrent identical requests -> LLM calls: 1
```
