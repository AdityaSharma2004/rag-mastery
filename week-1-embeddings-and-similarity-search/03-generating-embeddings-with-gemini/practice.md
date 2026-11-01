# Practice - Generating embeddings with Gemini

Five questions, rising difficulty. Attempt each before opening the hint, and the hint before the solution. Every solution is the exact code in `solutions.py`. Q1, Q2, Q4 and Q5 call the free Gemini embedding API (set `GEMINI_API_KEY` in `.env`); Q3's class takes the API call as an argument, so it can also be unit-tested with a fake. Results are cached on disk, so re-runs are free.

---

### Q1. Warm-up - call the raw API once

**Problem**

Without our helper, embed two sentences with `gemini-embedding-001` using the default settings. Print how many vectors came back, their shape, the first five numbers of the first vector, and its length. What is the native dimension?

<details><summary><b>Hint</b></summary>

`client.models.embed_content(model=..., contents=[...])` returns `resp.embeddings`; each item has `.values`. Convert to `np.array(..., dtype=np.float32)` and use `np.linalg.norm`.

</details>

<details><summary><b>Solution</b></summary>

```python
# --------------------------------------------------------------------------
# Q1. Warm-up: call the raw API once and inspect the response
# --------------------------------------------------------------------------
def q1():
    client = get_client()
    resp = client.models.embed_content(
        model="gemini-embedding-001",
        contents=["What is RAG?", "Retrieval-Augmented Generation explained"],
    )
    vecs = np.array([e.values for e in resp.embeddings], dtype=np.float32)
    print("number of vectors :", len(resp.embeddings))
    print("shape             :", vecs.shape)          # (2, 3072) - native size
    print("first 5 numbers   :", np.round(vecs[0][:5], 4))
    print("length of vec 0   :", round(float(np.linalg.norm(vecs[0])), 4))   # ~1.0 at 3072
    return vecs.shape
```

</details>

**Why it works / what to learn**
A list of strings in = a list of vectors out, in the same order. With no `output_dimensionality` you get the native **3072** numbers, and that full-size vector is already unit length.

**Expected output (≈)**
```text
number of vectors : 2
shape             : (2, 3072)
first 5 numbers   : [≈ small floats like 0.01, -0.02 ...]
length of vec 0   : 1.0
```

---

### Q2. Apply - does `task_type` really matter?

**Problem**

Embed five help-centre sentences and the question *"How can I get back into my account if I forgot my login?"* three ways: (a) documents `RETRIEVAL_DOCUMENT` + query `RETRIEVAL_QUERY`; (b) everything `SEMANTIC_SIMILARITY`; (c) everything `RETRIEVAL_DOCUMENT`. Print the top-3 (index, score) for each and decide which is the proper RAG setup.

<details><summary><b>Hint</b></summary>

Our `embed_texts(texts, task_type=..., dim=768)` returns unit vectors, so `D @ q` is cosine similarity. Sort with `np.argsort(-scores)`.

</details>

<details><summary><b>Solution</b></summary>

```python
# --------------------------------------------------------------------------
# Q2. Apply: does task_type change retrieval? Compare three settings
# --------------------------------------------------------------------------
def q2():
    corpus = [
        "To reset your password, click 'Forgot password' on the login page.",
        "Our refund policy allows returns within 30 days of purchase.",
        "The API rate limit is 100 requests per minute per key.",
        "Invoices are emailed on the first day of every month.",
        "Two-factor authentication can be enabled under Security settings.",
    ]
    question = "How can I get back into my account if I forgot my login?"

    def rank(doc_type: str, query_type: str):
        D = embed_texts(corpus, task_type=doc_type, dim=768)
        q = embed_texts([question], task_type=query_type, dim=768)[0]
        scores = D @ q
        order = np.argsort(-scores)
        return [(int(i), round(float(scores[i]), 3)) for i in order[:3]]

    print("RETRIEVAL_DOCUMENT + RETRIEVAL_QUERY :", rank("RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"))
    print("SEMANTIC_SIMILARITY both sides       :", rank("SEMANTIC_SIMILARITY", "SEMANTIC_SIMILARITY"))
    print("both as RETRIEVAL_DOCUMENT (misuse)  :", rank("RETRIEVAL_DOCUMENT", "RETRIEVAL_DOCUMENT"))
    return rank("RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY")[0][0]
```

</details>

**Why it works / what to learn**
All three probably put the password sentence (index 0) first on an easy question, but the *score gap* and the ordering of the rest differ, and on hard questions the proper pair wins more often. Use (a) for RAG; use symmetric `SEMANTIC_SIMILARITY` only when comparing two texts of the same kind.

**Expected output (≈)**
```text
RETRIEVAL_DOCUMENT + RETRIEVAL_QUERY : [(0, ≈0.7), (4, ≈0.6), ...]
SEMANTIC_SIMILARITY both sides       : [(0, ≈0.6), (4, ≈0.5), ...]
both as RETRIEVAL_DOCUMENT (misuse)  : [(0, ≈0.7), ...]
```
(exact numbers vary; index 0 first in all.)

---

### Q3. Build - your own embedding client

**Problem**

Write `EmbeddingStore` with: (1) a disk cache keyed by `sha256(model|task|dim|text)`; (2) batching; (3) retry with exponential back-off; (4) L2 normalisation; (5) de-duplication of identical texts within a call; (6) a `stats` dict counting API calls, cache hits and texts embedded. The API call must be injectable so it can be tested offline. Run `["alpha","beta","gamma","alpha"]` twice with `batch_size=2`.

<details><summary><b>Hint</b></summary>

Split into three phases: look up the cache, group missing texts by unique text, send the unique texts in batches. Write each vector to disk right after normalising.

</details>

<details><summary><b>Solution</b></summary>

```python
# --------------------------------------------------------------------------
# Q3. Build: your own embedding client (batching + disk cache + retry + stats)
#     The API call is injected, so you can unit-test it with a fake (no key needed).
# --------------------------------------------------------------------------
class EmbeddingStore:
    def __init__(self, api_call=None, cache_dir=".q3_cache", batch_size=50, dim=768, max_tries=5):
        self.api_call = api_call or self._gemini_call
        self.cache = pathlib.Path(cache_dir)
        self.cache.mkdir(exist_ok=True)
        self.batch_size, self.dim, self.max_tries = batch_size, dim, max_tries
        self.stats = {"api_calls": 0, "cache_hits": 0, "texts_embedded": 0}

    @staticmethod
    def _gemini_call(batch, task_type, dim):
        resp = get_client().models.embed_content(
            model="gemini-embedding-001",
            contents=batch,
            config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=dim),
        )
        return np.array([e.values for e in resp.embeddings], dtype=np.float32)

    def _key(self, text, task_type):
        raw = f"gemini-embedding-001|{task_type}|{self.dim}|{text}"
        return self.cache / (hashlib.sha256(raw.encode()).hexdigest() + ".npy")

    def _call_with_retry(self, batch, task_type):
        delay = 1.0
        for attempt in range(1, self.max_tries + 1):
            try:
                return self.api_call(batch, task_type, self.dim)
            except Exception as e:                     # in real code catch the SDK's APIError (429/5xx) only
                if attempt == self.max_tries:
                    raise
                time.sleep(delay)
                delay *= 2

    def embed(self, texts, task_type="RETRIEVAL_DOCUMENT"):
        out = [None] * len(texts)
        missing = []                                   # (index, text) pairs not on disk
        for i, t in enumerate(texts):
            p = self._key(t, task_type)
            if p.exists():
                out[i] = np.load(p)
                self.stats["cache_hits"] += 1
            else:
                missing.append((i, t))

        # de-duplicate identical texts inside this call: embed each unique text once
        unique = {}
        for i, t in missing:
            unique.setdefault(t, []).append(i)
        uniq_texts = list(unique)

        for s in range(0, len(uniq_texts), self.batch_size):
            batch = uniq_texts[s : s + self.batch_size]
            raw = self._call_with_retry(batch, task_type)
            raw = raw / np.clip(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12, None)  # L2-normalise
            self.stats["api_calls"] += 1
            self.stats["texts_embedded"] += len(batch)
            for t, v in zip(batch, raw):
                np.save(self._key(t, task_type), v)
                for i in unique[t]:
                    out[i] = v
        return np.vstack(out)


def q3():
    store = EmbeddingStore(cache_dir=".q3_cache", batch_size=2)
    texts = ["alpha", "beta", "gamma", "alpha"]       # note the duplicate
    v = store.embed(texts)
    print("shape:", v.shape, " stats after run 1:", store.stats)
    v2 = store.embed(texts)
    print("stats after run 2 (all cached):", store.stats)
    return store.stats
```

</details>

**Why it works / what to learn**
Four texts but one duplicate -> 3 unique -> 2 batches of size ≤2 -> **2 API calls**; the second run is 100% cache (stats show 4 cache hits and no new calls). Dependency injection (`api_call=`) is the same trick as binding a fake service in a Laravel test.

**Expected output (≈)**
```text
shape: (4, 768)  stats after run 1: {'api_calls': 2, 'cache_hits': 0, 'texts_embedded': 3}
stats after run 2 (all cached): {'api_calls': 2, 'cache_hits': 4, 'texts_embedded': 3}
```

---

### Q4. Debug - three bugs in one snippet

**Problem**

The snippet below embeds 10,000 chunks for a search feature. It is slow, hits 429 errors, and gives poor results. Find **three** bugs and fix them.

```python
docs = load_10000_chunks()
vectors = []
for d in docs:
    r = client.models.embed_content(model="gemini-embedding-001", contents=d,
        config=types.EmbedContentConfig(output_dimensionality=768))
    vectors.append(r.embeddings[0].values)
q = client.models.embed_content(model="gemini-embedding-001", contents=question,
    config=types.EmbedContentConfig(output_dimensionality=768)).embeddings[0].values
```

<details><summary><b>Hint</b></summary>

Count the API requests. Print `np.linalg.norm` of a 768-d vector. Which `task_type` should the query use?

</details>

<details><summary><b>Solution</b></summary>

```python
# --------------------------------------------------------------------------
# Q4. Debug: three bugs hiding in one snippet
# --------------------------------------------------------------------------
BROKEN = '''
docs = load_10000_chunks()
vectors = []
for d in docs:                                              # BUG 1: one API call per chunk
    r = client.models.embed_content(
        model="gemini-embedding-001", contents=d,
        config=types.EmbedContentConfig(output_dimensionality=768))   # BUG 2: no normalisation after
    vectors.append(r.embeddings[0].values)
q = client.models.embed_content(                            # BUG 3: query embedded with the default/doc task
    model="gemini-embedding-001", contents=question,
    config=types.EmbedContentConfig(output_dimensionality=768)).embeddings[0].values
'''


def q4():
    print("BROKEN snippet:\n", BROKEN)
    texts = ["How do I reset my password?", "Our office is closed on public holidays",
             "Invoices are due in 30 days"]

    # FIX 2 + 1: batch the call (one request for the list) and normalise the 768-d vectors
    client = get_client()
    resp = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts,                                                         # batch: 1 call, not 3
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT", output_dimensionality=768),
    )
    raw = np.array([e.values for e in resp.embeddings], dtype=np.float32)
    print("raw 768-d norms        :", np.round(np.linalg.norm(raw, axis=1), 3))   # not 1.0 -> must normalise
    D = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    print("normalised norms       :", np.round(np.linalg.norm(D, axis=1), 3))

    # FIX 3: embed the user's question with RETRIEVAL_QUERY
    q = embed_query("forgot my login password", dim=768)
    print("top match:", texts[int(np.argmax(D @ q))])
    return D.shape
```

</details>

**Why it works / what to learn**
Bug 1: one request per chunk (10,000 requests) instead of batches of ~50 (200 requests) -> 429s. Bug 2: 768-d vectors are not unit length, so dot products are not cosine -> normalise. Bug 3: no `task_type` on documents and the question embedded like a document -> use `RETRIEVAL_DOCUMENT` / `RETRIEVAL_QUERY`. (A bonus fourth: no cache or retry.)

**Expected output (≈)**
```text
raw 768-d norms        : [≈0.6-1.0 each, not exactly 1.0]
normalised norms       : [1. 1. 1.]
top match: How do I reset my password?
```

---

### Q5. Production - an idempotent ingestion job

**Problem**

You ingest 11 chunks, one of which is an exact duplicate, into an index keyed by chunk id. Write `ingest(chunks, store)` so that (a) the duplicate text is embedded only once, (b) running the job a second time (e.g. after a crash and restart) makes **zero** API calls, and (c) it returns `{chunk_id: vector}`.

<details><summary><b>Hint</b></summary>

Reuse `EmbeddingStore` from Q3. The job is idempotent because the cache key depends only on the text, not on the run.

</details>

<details><summary><b>Solution</b></summary>

```python
# --------------------------------------------------------------------------
# Q5. Production: an idempotent ingestion job that never pays twice
# --------------------------------------------------------------------------
def ingest(chunks: list[dict], store: EmbeddingStore) -> dict:
    """chunks = [{'id':..., 'text':...}]. Returns {id: vector}. Re-running costs 0 API calls."""
    texts = [c["text"] for c in chunks]
    vecs = store.embed(texts, task_type="RETRIEVAL_DOCUMENT")
    return {c["id"]: v for c, v in zip(chunks, vecs)}


def q5():
    chunks = [{"id": f"doc-{i}", "text": t} for i, t in enumerate(SAMPLE_DOCS)]
    chunks.append({"id": "doc-dup", "text": SAMPLE_DOCS[0]})        # an exact duplicate chunk

    store = EmbeddingStore(cache_dir=".q5_cache", batch_size=50)
    index = ingest(chunks, store)
    first = dict(store.stats)
    print(f"run 1: {len(index)} vectors, stats={first}")            # 10 unique texts -> 1 API call

    ingest(chunks, store)
    second = {k: store.stats[k] - first[k] for k in first}
    print(f"run 2: extra stats={second}")                            # 0 api_calls, 11 cache hits
    return first, second
```

</details>

**Why it works / what to learn**
Idempotent = safe to run twice. Content-hash caching gives you that for free and turns a failed 5,000-chunk job into "resume from where it stopped". In production you would store the vectors in a database with the same content hash as a column, so unchanged chunks are never re-embedded when a document is re-uploaded.

**Expected output (≈)**
```text
run 1: 11 vectors, stats={'api_calls': 1, 'cache_hits': 0, 'texts_embedded': 10}
run 2: extra stats={'api_calls': 0, 'cache_hits': 11, 'texts_embedded': 0}
```
