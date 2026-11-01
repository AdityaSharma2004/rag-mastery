# Practice - Pinecone

Five questions, rising difficulty. Attempt each before opening the hint, and the hint before the solution. Every solution is the exact code in `solutions.py` (`python solutions.py` runs all; `python solutions.py 3` runs one).

**Two ways to run:** with `PINECONE_API_KEY` set the code uses a real serverless index (`rag-course-w2`); with no key it runs against `FakeIndex`, an in-memory stand-in with the same method names and keyword arguments. The fake is for learning the API shape and testing logic offline, it is not Pinecone (no network, no eventual consistency). All embeddings use the free Gemini API (768-d).

Solutions use `get_index()`, `FakeIndex` and `wait_for_count()` from the top of `solutions.py`; they are not repeated below.

---

### Q1. Warm-up - create, upsert, query

**Problem**

Create (or reuse) a 768-d cosine index, upsert the 10 `SAMPLE_DOCS` with their text as metadata into namespace `warmup`, wait until they are visible, and query *"a young cat sleeping"*. Print the top 3 with score, id and text.

<details><summary><b>Hint</b></summary>

`index.upsert(vectors=[{"id":..., "values": vec.tolist(), "metadata": {...}}], namespace=...)`, then `index.query(vector=..., top_k=3, namespace=..., include_metadata=True)`. Results are in `res.matches`.

</details>

<details><summary><b>Solution</b></summary>

```python
# --------------------------------------------------------------------------
# Q1. Warm-up: create an index, upsert 10 vectors with metadata, query, read the matches
# --------------------------------------------------------------------------
def q1():
    index, real = get_index()
    vectors = [
        {"id": f"doc-{i}", "values": v.tolist(), "metadata": {"text": t, "n": i}}
        for i, (t, v) in enumerate(zip(SAMPLE_DOCS, embed_texts(SAMPLE_DOCS, dim=DIM)))
    ]
    index.upsert(vectors=vectors, namespace="warmup")          # keyword-only arguments
    print("visible vectors:", wait_for_count(index, len(vectors), "warmup"))

    q = embed_query("a young cat sleeping", dim=DIM).tolist()
    res = index.query(vector=q, top_k=3, namespace="warmup", include_metadata=True)
    for m in res.matches:                                      # best first, higher cosine score = closer
        print(f"{m.score:.3f}  {m.id}  {m.metadata['text'][:55]}")
    return [m.id for m in res.matches]
```

</details>

**Why it works / what to learn**
All data-plane arguments are keyword-only. Scores are cosine **similarity** (higher = closer). Without `include_metadata=True` you would get ids and scores only. On real serverless indexes the `wait_for_count` poll matters because upserts are eventually consistent.

**Expected output (≈)**
```text
visible vectors: 10
≈0.7  doc-7  Kittens are young cats that sleep most of the day...
≈0.6  doc-6  A cat sat quietly on the warm windowsill...
≈0.3  doc-...  (something unrelated)
```

---

### Q2. Apply - namespaces as tenants + a metadata filter

**Problem**

Two customers, `acme` and `globex`, each get their own namespace. Both have a record with id `a1` (on purpose). (1) Ask the same question in each namespace. (2) In `globex`, restrict to `year >= 2024` with a metadata filter. (3) Print per-namespace counts. (4) Delete the whole `globex` namespace and confirm `acme` survives.

<details><summary><b>Hint</b></summary>

`namespace=` on both upsert and query. Filter: `{"year": {"$gte": 2024}}`. Wipe: `index.delete(delete_all=True, namespace="globex")`.

</details>

<details><summary><b>Solution</b></summary>

```python
# --------------------------------------------------------------------------
# Q2. Apply: namespaces as tenants + metadata filters
# --------------------------------------------------------------------------
def q2():
    index, real = get_index()
    docs = {
        "acme": [("a1", "Acme refunds take 7 days.", 2024), ("a2", "Acme ships within 3 working days.", 2022)],
        "globex": [("a1", "Globex refunds take 14 days.", 2023), ("g2", "Globex support is open 24x7.", 2024)],
    }
    for tenant, rows in docs.items():                          # NOTE: id "a1" exists in BOTH namespaces on purpose
        index.upsert(namespace=tenant, vectors=[
            {"id": i, "values": v.tolist(), "metadata": {"text": t, "year": y}}
            for (i, t, y), v in zip(rows, embed_texts([r[1] for r in rows], dim=DIM))
        ])
    for tenant in docs:
        wait_for_count(index, len(docs[tenant]), tenant)

    q = embed_query("how long do refunds take", dim=DIM).tolist()
    for tenant in docs:                                        # a query only ever searches ONE namespace
        r = index.query(vector=q, top_k=2, namespace=tenant, include_metadata=True)
        print(f"{tenant:>6}:", [(m.id, m.metadata["text"][:28]) for m in r.matches])

    newer = index.query(vector=q, top_k=5, namespace="globex", include_metadata=True,
                        filter={"year": {"$gte": 2024}})        # metadata filter inside the namespace
    print("globex year>=2024:", [m.id for m in newer.matches])

    stats = index.describe_index_stats()
    print("per-namespace counts:", {n: s.vector_count for n, s in stats.namespaces.items() if n in docs})

    index.delete(delete_all=True, namespace="globex")           # offboard one tenant, others untouched
    time.sleep(2 if real else 0)
    print("after deleting globex:", sorted(n for n in index.describe_index_stats().namespaces if n in docs))
    return True
```

</details>

**Why it works / what to learn**
A query searches exactly one namespace, so tenants cannot see each other, and the same id may exist in several namespaces independently. Filters narrow results inside a namespace. Deleting a tenant (e.g. for a data-erasure request) is one call.

**Expected output (≈)**
```text
  acme: [('a1', 'Acme refunds take 7 days.'), ('a2', 'Acme ships within 3 working ')]
globex: [('a1', 'Globex refunds take 14 days.'), ('g2', 'Globex support is open 24x7.')]
globex year>=2024: ['g2']
per-namespace counts: {'acme': 2, 'globex': 2}
after deleting globex: ['acme']
```

---

### Q3. Build - a small store with batching, ids and a score threshold

**Problem**

Write `PineconeStore(index, namespace, dim, batch_size)` with `ingest(source, texts)` and `search(question, k, flt, min_score)`. Requirements: deterministic ids (re-ingesting the same text overwrites, never duplicates); upsert in batches; chunk text stored in metadata with a size cap; results below `min_score` are dropped so off-topic questions return `[]`.

<details><summary><b>Hint</b></summary>

`id = sha1(source|index|text)[:20]`. Slice records into `batch_size` groups. Use `m.score >= min_score` on cosine similarity.

</details>

<details><summary><b>Solution</b></summary>

```python
# --------------------------------------------------------------------------
# Q3. Build: a small store with batching, deterministic ids, score threshold, metadata text
# --------------------------------------------------------------------------
class PineconeStore:
    """Pinecone keeps vectors + metadata (no separate 'documents' field), so the chunk text lives in metadata."""
    MAX_TEXT_BYTES = 8_000           # metadata is capped (tens of KB per record); keep chunk text small

    def __init__(self, index, namespace: str = "", dim: int = DIM, batch_size: int = 100):
        self.index, self.ns, self.dim, self.batch = index, namespace, dim, batch_size

    @staticmethod
    def make_id(source: str, i: int, text: str) -> str:
        return hashlib.sha1(f"{source}|{i}|{text}".encode()).hexdigest()[:20]

    def ingest(self, source: str, texts: list[str]) -> int:
        vecs = embed_texts(texts, dim=self.dim)
        records = [{
            "id": self.make_id(source, i, t),
            "values": v.tolist(),
            "metadata": {"source": source, "chunk": i, "text": t.encode()[: self.MAX_TEXT_BYTES].decode(errors="ignore")},
        } for i, (t, v) in enumerate(zip(texts, vecs))]
        for s in range(0, len(records), self.batch):           # upsert in batches, never one giant request
            self.index.upsert(vectors=records[s : s + self.batch], namespace=self.ns)
        return len(records)

    def search(self, question: str, k: int = 3, flt: dict | None = None, min_score: float = 0.5):
        res = self.index.query(vector=embed_query(question, dim=self.dim).tolist(), top_k=k,
                               namespace=self.ns, filter=flt, include_metadata=True)
        return [{"id": m.id, "score": round(m.score, 3), "text": m.metadata["text"], "source": m.metadata["source"]}
                for m in res.matches if m.score >= min_score]


def q3():
    index, real = get_index()
    store = PineconeStore(index, namespace="kb", batch_size=4)
    n1 = store.ingest("laravel.md", ["Eloquent maps tables to PHP classes.",
                                     "Queues run slow jobs in the background.",
                                     "Migrations version your database schema.",
                                     "Blade is the Laravel templating engine.",
                                     "Horizon monitors Redis queues."])
    n2 = store.ingest("laravel.md", ["Eloquent maps tables to PHP classes."])   # same id again -> overwrite
    wait_for_count(index, 5, "kb")
    print("ingested", n1, "then re-ingested", n2, "-> vectors in namespace:", wait_for_count(index, 5, "kb"))
    print("relevant :", [(h["score"], h["text"][:35]) for h in store.search("background jobs", min_score=0.4)])
    print("off-topic:", store.search("capital of France", min_score=0.75))
    return store
```

</details>

**Why it works / what to learn**
Pinecone has no separate document field, so the chunk text rides in metadata. Deterministic ids give idempotent ingestion: the second `ingest` overwrote one record instead of adding a sixth. The score floor gives your RAG layer a clean "I don't know" path; the right threshold must be measured on your data.

**Expected output (≈)**
```text
ingested 5 then re-ingested 1 -> vectors in namespace: 5
relevant : [(≈0.7, 'Queues run slow jobs in the backgro')]
off-topic: []
```

---

### Q4. Debug - four classic Pinecone mistakes

**Problem**

Reproduce and fix: (1) upserting a 1536-d vector into a 768-d index; (2) `match.metadata` comes back empty; (3) a query returns zero matches even though data was upserted; (4) a test that queries immediately after upserting is flaky.

<details><summary><b>Hint</b></summary>

(1) Compare `len(values)` with the index dimension. (2) A query option. (3) Which namespace did you upsert into, and which are you querying? (4) Serverless is eventually consistent.

</details>

<details><summary><b>Solution</b></summary>

```python
# --------------------------------------------------------------------------
# Q4. Debug: four classic Pinecone mistakes
# --------------------------------------------------------------------------
def q4():
    index, real = get_index()
    v768 = embed_texts(["hello"], dim=768)[0].tolist()
    v1536 = embed_texts(["hello"], dim=1536)[0].tolist()
    index.upsert(vectors=[{"id": "h", "values": v768, "metadata": {"text": "hello"}}], namespace="bugs")
    wait_for_count(index, 1, "bugs")

    # BUG 1: vector size differs from the index dimension
    try:
        index.upsert(vectors=[{"id": "bad", "values": v1536}], namespace="bugs")
    except Exception as e:
        print("bug1 ->", type(e).__name__, "(index is 768-d, vector was 1536-d)")

    # BUG 2: forgot include_metadata -> matches carry no text
    r = index.query(vector=v768, top_k=1, namespace="bugs")
    print("bug2 metadata without include_metadata:", r.matches[0].metadata)
    r = index.query(vector=v768, top_k=1, namespace="bugs", include_metadata=True)
    print("fix  with include_metadata=True       :", r.matches[0].metadata)

    # BUG 3: wrong namespace -> empty result, no error
    r = index.query(vector=v768, top_k=1, include_metadata=True)            # default namespace is ""
    print("bug3 query in the default namespace  :", len(r.matches), "matches (data is in 'bugs')")

    # BUG 4: reading before the write is visible (eventual consistency on serverless)
    visible = wait_for_count(index, 1, "bugs")
    print("fix  poll describe_index_stats until count is visible:", visible)
    index.delete(delete_all=True, namespace="bugs")
    return True
```

</details>

**Why it works / what to learn**
These are the four things a new Pinecone user hits in the first hour: index settings are immutable, metadata is opt-in on queries, the default namespace is `""`, and writes become visible shortly after being accepted. Fix them with a pinned `DIM`, `include_metadata=True`, a single namespace constant, and a readiness poll.

**Expected output (≈)**
```text
bug1 -> ValueError (index is 768-d, vector was 1536-d)      (real Pinecone raises an API exception)
bug2 metadata without include_metadata: None                (or an empty value on some versions)
fix  with include_metadata=True       : {'text': 'hello'}
bug3 query in the default namespace  : 0 matches (data is in 'bugs')
fix  poll describe_index_stats until count is visible: 1
```

---

### Q5. Production - resumable, retrying bulk ingestion

**Problem**

Ingest 50 chunks in batches of 20 with: retry and exponential back-off on transient failures (the fake injects two 503s), a **checkpoint file** recording progress after each batch, and a second run that does nothing because everything is already done. Report how many vectors are visible.

<details><summary><b>Hint</b></summary>

Loop `for start in range(done, len(records), batch_size)`; after each successful batch write `{"next": start + batch_size}` to a JSON file; on start read it to resume. Retry only the failing batch.

</details>

<details><summary><b>Solution</b></summary>

```python
# --------------------------------------------------------------------------
# Q5. Production: resumable, retrying bulk ingestion with a checkpoint file
# --------------------------------------------------------------------------
def upsert_with_retry(index, batch, namespace, max_tries=5):
    delay = 0.2
    for attempt in range(1, max_tries + 1):
        try:
            return index.upsert(vectors=batch, namespace=namespace)
        except Exception as e:
            if attempt == max_tries:
                raise
            print(f"  upsert failed ({e}); retry {attempt}/{max_tries - 1} in {delay:.1f}s")
            time.sleep(delay)
            delay *= 2


def bulk_ingest(index, records: list[dict], namespace: str, checkpoint: pathlib.Path, batch_size: int = 50) -> int:
    """Upsert in batches; write progress after each batch so a crash resumes instead of restarting."""
    done = json.loads(checkpoint.read_text())["next"] if checkpoint.exists() else 0
    for start in range(done, len(records), batch_size):
        upsert_with_retry(index, records[start : start + batch_size], namespace)
        checkpoint.write_text(json.dumps({"next": start + batch_size}))
    return len(records)


def q5():
    index, real = get_index()
    texts = [f"{t} (variant {i})" for i in range(5) for t in SAMPLE_DOCS]          # 50 chunks
    vecs = embed_texts(texts, dim=DIM)
    records = [{"id": f"bulk-{i}", "values": v.tolist(), "metadata": {"text": t}}
               for i, (t, v) in enumerate(zip(texts, vecs))]
    ckpt = pathlib.Path(tempfile.mkdtemp()) / "progress.json"

    if isinstance(index, FakeIndex):
        index.fail_next_upserts = 2                      # first two requests fail with a simulated 503
    bulk_ingest(index, records, "bulk", ckpt, batch_size=20)
    print("vectors visible:", wait_for_count(index, len(records), "bulk"), "checkpoint:", ckpt.read_text())

    bulk_ingest(index, records, "bulk", ckpt, batch_size=20)                          # re-run: nothing left to do
    print("second run was a no-op; still", wait_for_count(index, len(records), "bulk"), "vectors")
    index.delete(delete_all=True, namespace="bulk")
    return True
```

</details>

**Why it works / what to learn**
Bulk loads fail halfway in real life (rate limits, deploys, laptop sleep). Per-batch retry handles transient errors; the checkpoint makes the job **resumable**; deterministic ids make a re-run **idempotent**. These three habits turn a fragile script into a reliable ingestion job.

**Expected output (≈)**
```text
  upsert failed (simulated 503 from Pinecone); retry 1/4 in 0.2s
  upsert failed (simulated 503 from Pinecone); retry 2/4 in 0.4s
vectors visible: 50 checkpoint: {"next": 60}
second run was a no-op; still 50 vectors
```
