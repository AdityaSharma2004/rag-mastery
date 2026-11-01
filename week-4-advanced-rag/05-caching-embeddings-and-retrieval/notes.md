# 05 - Caching embeddings and retrieval

> Week 4 · Topic 5 of 7 · Time: ~75 min · Runs offline (counted fake "API calls"); Redis optional (uses `fakeredis` if no server)

## 🎯 What you will learn

- You will be able to decide **what to cache** in a RAG pipeline (embeddings, retrieval results, answers) and what never to cache.
- You will be able to design **correct cache keys** (model, task type, dimension, tenant, corpus version, parameters).
- You will be able to build an **embedding cache that embeds only the misses** in one batch.
- You will be able to **invalidate** caches safely when documents change.
- You will be able to add a **semantic cache** and **single-flight** (stampede) protection, using Redis or an in-memory fallback.

## 🧠 The big idea in plain words

Most RAG cost and latency is waste: the same chunk gets embedded again after every deploy, the same question is asked all day, and the same expensive LLM call is paid every time. A **cache** remembers results so repeat work is free.

Free-tier angle: Gemini calls have rate limits. A cache turns "429 Too Many Requests" into "instant answer".

Laravel analogy: this is exactly `Cache::remember('key', $ttl, fn () => expensive())`, backed by Redis, with `Cache::lock()` to avoid stampedes and `Cache::tags()`/versioned keys to invalidate. The new part is only *what the keys must contain* in a RAG system.

## 📚 Concepts, step by step

### 1. What to cache (from safest to riskiest)

| Layer | Key contains | Value | Risk | Typical saving |
|---|---|---|---|---|
| **Embeddings** | model + task type + dim + text hash | vector | none (deterministic) | re-ingestion, repeated queries: ~100% of embed calls |
| **Retrieval results** | tenant + corpus version + k + filters + query | ids/scores | stale after updates | vector search latency |
| **Answers** | tenant + corpus version + prompt version + model + normalised question | text + citations | stale, leaks if key incomplete | biggest: skips the LLM |
| **Semantic cache** | as above, plus similarity threshold | answer of a *similar* question | wrong answer for near-miss questions | catches paraphrases |
| **Document loading / summaries** | file hash | text/summary | none | ingestion time |

Never cache: anything personalised without the user/tenant in the key, or answers to questions where freshness is critical (live prices) without a short TTL.

### 2. Cache keys are the whole game

A key must contain **everything that changes the result**:

```python
make_key("gemini-embedding-001", "RETRIEVAL_DOCUMENT", 768, text)             # embeddings
make_key("answer", tenant, corpus_version, prompt_version, model, k, normalised_question)
```
Missing one input = a silent wrong answer (Q4: tenant missing → cross-tenant leak). Extra inputs only lower the hit rate. Hash with SHA-256 for short, safe keys. Normalise questions (trim, lowercase, collapse spaces) to raise hit rate, but do not normalise away meaning.

### 3. Embedding cache: embed only the misses

```
requested texts → keys → which are missing? → dedupe → ONE batched API call → store → return in original order
```
Properties: deterministic (no staleness), the same text embedded twice in a batch is sent once, and a re-run after a crash resumes where it stopped. Our `gemini_utils.embed_texts` already caches on disk; this lesson shows how it works and how to put it in Redis.

### 4. TTL and invalidation

- **TTL (time to live):** entries expire automatically (e.g. answers 1-24 h). Simple, but stale data can live until expiry.
- **Versioned keys:** put a `corpus_version` in the key and bump it when documents change; old entries become unreachable and expire. No deletion needed.
- **Content-hash version:** hash the retrieved chunk texts (or document hashes) into the key so any change invalidates exactly the affected answers (Q4 fix).
- **Explicit delete:** delete by tag/prefix on document update (Redis `SCAN` + `DEL`, or tags).
There are only two hard things in computer science... one of them is this. Prefer versioned keys.

### 5. Semantic cache

Users paraphrase: "refund window?" vs "how long do I have to ask for a refund?". Embed the incoming question and compare it with embeddings of previously answered questions; above a threshold (say cosine ≥ 0.90) reuse that answer.

Risks and rules:
- Tune the **threshold on labelled pairs**; too low returns wrong answers ("refund in 14 days" vs "refund in 40 days").
- Scope by **tenant** and by corpus version.
- Keep it for FAQ-like traffic; skip it for questions with numbers/ids/negations (they change meaning with tiny wording changes).
- For many entries use a vector index (Redis vector search, Chroma, Qdrant); a linear scan is fine for thousands.

### 6. Redis in 10 lines

```python
import redis
r = redis.Redis.from_url("redis://localhost:6379/0")
r.set("ans:acme:ab12", "14 days", ex=3600)        # ex = TTL seconds
r.get("ans:acme:ab12")                            # -> b"14 days" (bytes!)
r.delete("ans:acme:ab12")
r.set("lock:k", "1", nx=True, ex=30)              # NX = only if absent: a lock
```
Configure `maxmemory` and `maxmemory-policy allkeys-lru` so Redis evicts old entries instead of failing. Store vectors compactly (`np.float32.tobytes()` / `np.frombuffer`) rather than JSON. Local dev without Redis: `fakeredis.FakeRedis()` (same API), or a dict.

### 7. Cache stampede and single-flight

A popular question expires; 100 requests arrive in the same second; all 100 miss and call the LLM. Protect with **single-flight**: one request computes while the others wait for its result (a lock per key). Single process: `threading.Lock`. Multiple servers: a Redis lock (`SET key NX EX`) with a short TTL, then re-check the cache after acquiring (Q5 does a per-key lock and re-check). Also add small random jitter to TTLs so keys do not all expire together.

### 8. Measure it

Track: hit rate per layer, LLM calls saved, p50/p95 latency for hit vs miss, cache size, evictions. A cache nobody measures is a source of invisible wrong answers.

## 💻 Code walkthrough

### Stage 1 - keys and a TTL cache (Q1)

```python
def make_key(*parts): return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
class TTLCache:  # data[key] = (expires_at, value); get() pops expired entries and counts hits/misses
```
The injectable `clock` makes TTL behaviour testable without sleeping.

### Stage 2 - embedding cache (Q2)

```python
missing = [i for i, k in enumerate(keys) if k not in self.store]
uniq = list(dict.fromkeys(texts[i] for i in missing))        # dedupe, keep order
for t, v in zip(uniq, self.raw(uniq)): self.store[self.key(t)] = v     # ONE call
return np.vstack([self.store[k] for k in keys])
```
Swap `RAW_EMBED` for `lambda t: embed_texts(t, task_type="RETRIEVAL_DOCUMENT")`.

### Stage 3 - versioned answers (Q3)

The key includes `self.version`; `update_docs` bumps it, so every old answer is simply never looked up again.

### Stage 4 - the two bugs (Q4)

Buggy key `question.lower()`; fixed key `make_key(tenant, content_hash(docs), question)`.

### Stage 5 - semantic cache + single-flight (Q5)

Order of lookup: exact key → semantic scan (same tenant) → lock → re-check → compute → store → add to index. The 10-thread test proves only one LLM call occurs.

## ⚠️ Common mistakes & debugging

| Symptom | Cause | Fix |
|---|---|---|
| One tenant sees another's answer | Key lacks tenant | Include tenant/user/permission scope in every key |
| Users see outdated answers | No invalidation, long TTL | Versioned or content-hash keys; shorter TTL |
| Hit rate near zero | Key includes noise (timestamps, request ids) or no normalisation | Remove noise; normalise questions |
| Wrong embeddings after a model change | Key omits model/dim/task type | Include them; old vectors become unreachable |
| LLM called 100× for one popular question | Stampede on expiry | Single-flight lock, TTL jitter |
| Semantic cache returns wrong answer | Threshold too low, near-miss questions | Tune on labelled pairs; skip numeric/negated questions |
| Redis returns `b'...'` | Bytes from redis-py | `.decode()` or `decode_responses=True` |
| Redis memory grows forever | No TTL/maxmemory policy | TTLs; `maxmemory` + `allkeys-lru` |
| Cache hides a bug while debugging | Stale hit | Add a bypass flag/header and log hit/miss |
| Secrets/PII cached | Caching full prompts | Cache outputs you can store; encrypt or skip sensitive data |

## 🚀 Pro-level insights (what the top 2% know)

- **Cache the deterministic things aggressively (embeddings), the stochastic things carefully (answers).**
- **Version everything that affects the output:** corpus, chunker, embedding model, prompt, LLM. Put a combined `pipeline_version` string in keys.
- **Two-level cache:** in-process LRU (microseconds) in front of Redis (shared across servers).
- **Provider-side prompt caching** (where available) reduces cost of long static prefixes; it complements, not replaces, your own cache.
- **Warm the cache:** pre-embed the corpus at ingestion; pre-answer the top 100 FAQ questions nightly.
- **Negative caching:** cache "no relevant document" results briefly to avoid repeated expensive misses.
- **Security:** the cache is part of your trust boundary; apply the same ACL scoping as retrieval, and never share semantic-cache hits across tenants.
- **Observability:** emit `cache=hit|miss|semantic` in the request trace so quality issues can be correlated with cached answers.

## 🧪 Mini lab (10-15 min)

1. Wrap your Week 3 `embed_texts` in an `EmbeddingCache` backed by Redis (or fakeredis) with TTL 7 days.
2. Ingest the same 100 chunks twice and compare API calls.
3. Add an answer cache keyed with tenant + corpus version; change one document and confirm the answer is recomputed.
4. Fire 20 concurrent identical requests and verify one LLM call.
5. Record hit rate over a 50-question session.

## ✅ Recap checklist

- [ ] I can list the cache layers and the risk of each.
- [ ] My keys include every input that changes the result.
- [ ] I embed only cache misses, in one batch.
- [ ] I invalidate with versioned/content-hash keys.
- [ ] I scope caches by tenant.
- [ ] I protect against stampedes and measure hit rate.

## 🎤 Interview questions

**1. What would you cache in a RAG system?**
Embeddings (deterministic), retrieval results and answers (keyed by tenant, corpus version, parameters), plus document-derived artefacts such as summaries; with TTL and versioned invalidation.

**2. How do you invalidate answer caches when documents change?**
Include a corpus version or content hash in the key so old entries become unreachable; optionally delete by prefix; keep TTLs short for volatile data.

**3. What is a semantic cache and its risk?**
It reuses answers for questions whose embeddings are within a similarity threshold; the risk is returning an answer to a *different* question, so tune the threshold, scope by tenant, and avoid it for numeric/negated queries.

**4. What is a cache stampede and how do you prevent it?**
Many concurrent misses for the same expired key; use single-flight locks (per-key lock or Redis `SET NX`), TTL jitter, and stale-while-revalidate.

**5. How do you make sure a cache never leaks data across users?**
Put tenant/user/permission scope in every key, test with two tenants asking identical questions, and apply the same ACL filtering before caching results.

## 🔗 Next

**06 - Multimodal RAG:** extend retrieval to images and scanned pages by captioning them with `gemini-2.5-flash` and indexing the captions.
