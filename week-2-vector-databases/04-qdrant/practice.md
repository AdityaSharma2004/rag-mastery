# Practice - Qdrant

Five questions, rising difficulty. Attempt each before opening the hint, and the hint before the solution. Every solution is the exact code in `solutions.py` (`python solutions.py` runs all). They run on Qdrant's **embedded mode** (no server needed) and embed with the free Gemini API (768-d). Shared helper:

```python
def point_id(text: str) -> str:
    """Qdrant ids must be unsigned ints or UUIDs. A UUID derived from the content = idempotent upserts."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, text))
```

(Imports used throughout: `from qdrant_client import QdrantClient, models` and `from gemini_utils import *`.)

---

### Q1. Warm-up - collection, points, query

**Problem**

Create an in-memory cosine collection (768-d), upsert the 10 `SAMPLE_DOCS` with their text in the payload (ids derived from the text), and query *"a young cat sleeping"*. Print the count and top 3 with scores.

<details><summary><b>Hint</b></summary>

`PointStruct(id=..., vector=v.tolist(), payload={...})`; `client.query_points(name, query=vector, limit=3, with_payload=True).points`.

</details>

<details><summary><b>Solution</b></summary>

```python
# --------------------------------------------------------------------------
# Q1. Warm-up: collection, points with payload, nearest-neighbour query
# --------------------------------------------------------------------------
def q1():
    client = QdrantClient(":memory:")
    client.create_collection("warmup", vectors_config=models.VectorParams(size=DIM, distance=models.Distance.COSINE))
    vecs = embed_texts(SAMPLE_DOCS, dim=DIM)
    client.upsert("warmup", points=[
        models.PointStruct(id=point_id(t), vector=v.tolist(), payload={"text": t, "n": i})
        for i, (t, v) in enumerate(zip(SAMPLE_DOCS, vecs))
    ])
    print("points stored:", client.count("warmup").count)

    q = embed_query("a young cat sleeping", dim=DIM).tolist()
    res = client.query_points("warmup", query=q, limit=3, with_payload=True)      # .points, best first
    for p in res.points:
        print(f"{p.score:.3f}  {p.payload['text'][:55]}")                          # COSINE: score = similarity (higher = closer)
    return [p.payload["n"] for p in res.points]
```

</details>

**Why it works / what to learn**
With cosine distance Qdrant's score is the similarity (higher = closer). Deterministic ids (uuid5) mean re-running the script overwrites instead of duplicating.

**Expected output (≈)**
```text
points stored: 10
≈0.7  Kittens are young cats that sleep most of the day ...
≈0.6  A cat sat quietly on the warm windowsill ...
≈0.3  (unrelated)
```

---

### Q2. Apply - payload filters

**Problem**

Store four chunks with `tenant`, `year`, `type` payload. Query "how long do refunds take" with (a) tenant = acme, (b) acme **and** year ≥ 2024, (c) acme but **not** type faq. Also create payload indexes on `tenant` and `year`.

<details><summary><b>Hint</b></summary>

`models.Filter(must=[...], must_not=[...])` with `FieldCondition(key=..., match=MatchValue(value=...))` and `range=models.Range(gte=2024)`.

</details>

<details><summary><b>Solution</b></summary>

```python
# --------------------------------------------------------------------------
# Q2. Apply: payload filters (must / should / must_not, match, range) + a payload index
# --------------------------------------------------------------------------
def q2():
    client = QdrantClient(":memory:")
    client.create_collection("kb", vectors_config=models.VectorParams(size=DIM, distance=models.Distance.COSINE))
    rows = [("Acme refunds take 7 days.", "acme", 2024, "policy"),
            ("Acme ships within 3 working days.", "acme", 2022, "faq"),
            ("Acme password reset links expire in 15 minutes.", "acme", 2024, "faq"),
            ("Globex refunds take 14 days.", "globex", 2024, "policy")]
    vecs = embed_texts([r[0] for r in rows], dim=DIM)
    client.upsert("kb", points=[models.PointStruct(id=point_id(r[0]), vector=v.tolist(),
                                                   payload={"text": r[0], "tenant": r[1], "year": r[2], "type": r[3]})
                                for r, v in zip(rows, vecs)])
    # On a real server, index the fields you filter on (no-op / ignored in embedded mode):
    client.create_payload_index("kb", field_name="tenant", field_schema=models.PayloadSchemaType.KEYWORD)
    client.create_payload_index("kb", field_name="year", field_schema=models.PayloadSchemaType.INTEGER)

    q = embed_query("how long do refunds take", dim=DIM).tolist()

    def ask(flt, k=3):
        r = client.query_points("kb", query=q, query_filter=flt, limit=k, with_payload=True)
        return [(p.payload["text"][:26], round(p.score, 2)) for p in r.points]

    acme = models.Filter(must=[models.FieldCondition(key="tenant", match=models.MatchValue(value="acme"))])
    acme_new = models.Filter(must=[models.FieldCondition(key="tenant", match=models.MatchValue(value="acme")),
                                   models.FieldCondition(key="year", range=models.Range(gte=2024))])
    not_faq = models.Filter(must=[models.FieldCondition(key="tenant", match=models.MatchValue(value="acme"))],
                            must_not=[models.FieldCondition(key="type", match=models.MatchValue(value="faq"))])
    print("acme            :", ask(acme))
    print("acme and >=2024 :", ask(acme_new))
    print("acme, not faq   :", ask(not_faq))
    return ask(acme_new)
```

</details>

**Why it works / what to learn**
Filters compose like boolean logic: `must` = AND, `should` = OR, `must_not` = NOT. On a real server payload indexes make these filters fast; in embedded mode the call is accepted but has no effect (a warning is printed).

**Expected output (≈)**
```text
acme            : [('Acme refunds take 7 days.', ≈0.7), ('Acme password reset links ', ≈0.3), ('Acme ships within 3 workin', ≈0.2)]
acme and >=2024 : [('Acme refunds take 7 days.', ≈0.7), ('Acme password reset links ', ≈0.3)]
acme, not faq   : [('Acme refunds take 7 days.', ≈0.7)]
```

---

### Q3. Build - an idempotent, persistent store

**Problem**

Write `QdrantStore` with `upsert(source, texts)` (deterministic ids), `replace_source(source, texts)` (delete by filter, then upsert), `search(question, k, min_similarity)` using `score_threshold`, and persistence in a local folder. Prove: upserting twice adds nothing; replacing a shrunken file leaves no stale chunk; data survives a "restart"; off-topic questions return `[]`.

<details><summary><b>Hint</b></summary>

`point_id(f"{source}|{i}")`; delete with `points_selector=models.FilterSelector(filter=...)`; `client.close()` before reopening the same folder.

</details>

<details><summary><b>Solution</b></summary>

```python
# --------------------------------------------------------------------------
# Q3. Build: an idempotent, persistent store with replace-by-source and a score threshold
# --------------------------------------------------------------------------
class QdrantStore:
    def __init__(self, path=None, name="knowledge", dim=DIM):
        self.name, self.dim = name, dim
        self.client = QdrantClient(path=str(path)) if path else QdrantClient(":memory:")
        if not self.client.collection_exists(name):
            self.client.create_collection(name, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE))

    def _src(self, source):
        return models.Filter(must=[models.FieldCondition(key="source", match=models.MatchValue(value=source))])

    def upsert(self, source: str, texts: list[str]) -> int:
        vecs = embed_texts(texts, dim=self.dim)
        self.client.upsert(self.name, points=[
            models.PointStruct(id=point_id(f"{source}|{i}"), vector=v.tolist(),
                               payload={"source": source, "chunk": i, "text": t})
            for i, (t, v) in enumerate(zip(texts, vecs))])
        return len(texts)

    def replace_source(self, source: str, texts: list[str]) -> int:
        self.client.delete(self.name, points_selector=models.FilterSelector(filter=self._src(source)))
        return self.upsert(source, texts)

    def search(self, question: str, k: int = 3, min_similarity: float = 0.5):
        r = self.client.query_points(self.name, query=embed_query(question, dim=self.dim).tolist(),
                                     limit=k, score_threshold=min_similarity, with_payload=True)
        return [{"text": p.payload["text"], "source": p.payload["source"], "similarity": round(p.score, 3)} for p in r.points]

    def count(self) -> int:
        return self.client.count(self.name).count


def q3():
    tmp = pathlib.Path(tempfile.mkdtemp())
    try:
        s = QdrantStore(path=tmp / "qdb")
        v1 = ["Queues run slow jobs in the background.", "Horizon monitors Redis queues."]
        s.upsert("queues.md", v1); s.upsert("queues.md", v1)
        print("points after upserting twice:", s.count())
        s.replace_source("queues.md", ["Queues can use Redis, SQS or the database."])
        print("points after replace_source :", s.count())
        s.client.close(); del s                                   # "restart"
        s = QdrantStore(path=tmp / "qdb")
        print("after restart:", s.count())
        print("relevant :", s.search("background jobs and queues", min_similarity=0.3))
        print("off-topic:", s.search("capital city of France", min_similarity=0.75))
        s.client.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
```

</details>

**Why it works / what to learn**
Same source + chunk number gives the same UUID, so upsert overwrites. A file that gets shorter would otherwise leave stale tail chunks, hence delete-by-filter first. `score_threshold` gives a clean "no answer" path.

**Expected output (≈)**
```text
points after upserting twice: 2
points after replace_source : 1
after restart: 1
relevant : [{'text': 'Queues can use Redis, SQS or the database.', 'source': 'queues.md', 'similarity': ≈0.6}]
off-topic: []
```

---

### Q4. Debug - four classic Qdrant mistakes

**Problem**

Reproduce and fix: (1) upserting with id `"doc-1"`; (2) a 1536-d vector into a 768-d collection; (3) a tenant filter written with `should` that returns another tenant's points; (4) confusion about score direction for cosine vs Euclid collections.

<details><summary><b>Hint</b></summary>

(1) ids are ints or UUIDs. (3) Which clause means AND? (4) Query `[1,0]` against `[1,0]` and `[0,1]` in both kinds of collection and read the scores.

</details>

<details><summary><b>Solution</b></summary>

```python
# --------------------------------------------------------------------------
# Q4. Debug: four classic Qdrant mistakes
# --------------------------------------------------------------------------
def q4():
    client = QdrantClient(":memory:")
    client.create_collection("bugs", vectors_config=models.VectorParams(size=DIM, distance=models.Distance.COSINE))
    v = embed_texts(["hello"], dim=DIM)[0].tolist()

    # BUG 1: an arbitrary string is not a valid point id
    try:
        client.upsert("bugs", points=[models.PointStruct(id="doc-1", vector=v, payload={})])
    except Exception as e:
        print("bug1 ->", type(e).__name__, "(ids must be unsigned ints or UUIDs)")
    client.upsert("bugs", points=[models.PointStruct(id=point_id("doc-1"), vector=v, payload={"tenant": "acme"})])

    # BUG 2: wrong vector size
    try:
        client.upsert("bugs", points=[models.PointStruct(id=1, vector=[0.1] * 1536, payload={})])
    except Exception as e:
        print("bug2 ->", type(e).__name__, "(collection is 768-d, vector was 1536-d)")

    # BUG 3: tenant filter written with `should` (OR) instead of `must` (AND) -> leaks other tenants
    client.upsert("bugs", points=[models.PointStruct(id=2, vector=v, payload={"tenant": "globex"}),
                                  models.PointStruct(id=3, vector=v, payload={"tenant": "globex", "year": 2024})])
    leaky = models.Filter(should=[models.FieldCondition(key="tenant", match=models.MatchValue(value="acme")),
                                  models.FieldCondition(key="year", range=models.Range(gte=2024))])
    safe = models.Filter(must=[models.FieldCondition(key="tenant", match=models.MatchValue(value="acme")),
                               models.FieldCondition(key="year", range=models.Range(gte=2024))])
    n_leaky = len(client.query_points("bugs", query=v, query_filter=leaky, limit=10).points)
    n_safe = len(client.query_points("bugs", query=v, query_filter=safe, limit=10).points)
    print(f"bug3 should (OR) returns {n_leaky} points incl. another tenant; must (AND) returns {n_safe}")

    # BUG 4: reading the score with the wrong direction. Cosine/Dot: higher = closer. Euclid: lower = closer.
    for dist in (models.Distance.COSINE, models.Distance.EUCLID):
        c = QdrantClient(":memory:")
        c.create_collection("d", vectors_config=models.VectorParams(size=2, distance=dist))
        c.upsert("d", points=[models.PointStruct(id=1, vector=[1.0, 0.0], payload={}),
                              models.PointStruct(id=2, vector=[0.0, 1.0], payload={})])
        r = c.query_points("d", query=[1.0, 0.0], limit=2).points
        print(f"bug4 {dist.value:<9} scores best-first: {[round(p.score, 2) for p in r]}")
    return n_safe
```

</details>

**Why it works / what to learn**
`should` is an OR, so "tenant = acme OR year ≥ 2024" matches a different tenant's 2024 point: a data leak. Use `must` and unit-test isolation. For Euclid the best score is the **smallest**; for cosine and dot the **largest**.

**Expected output (≈)**
```text
bug1 -> ValueError (ids must be unsigned ints or UUIDs)
bug2 -> ValueError (collection is 768-d, vector was 1536-d)
bug3 should (OR) returns 2 points incl. another tenant; must (AND) returns 0
bug4 Cosine    scores best-first: [1.0, 0.0]
bug4 Euclid    scores best-first: [0.0, 1.41]
```
(`must` returns 0 because the only `acme` point has no `year`.)

---

### Q5. Production - batch load, scroll, filtered count/delete, quantisation

**Problem**

Create a collection with HNSW settings and int8 scalar quantisation. Upload 60 points in batches of 25, paginate through **all** of them with `scroll`, count and delete by a payload filter, and compute the RAM for 1 M vectors at float32 vs int8.

<details><summary><b>Hint</b></summary>

`scroll` returns `(points, next_offset)`; loop until `next_offset is None`. Delete with `models.FilterSelector(filter=...)`. `n × dim × bytes / 1024**3`.

</details>

<details><summary><b>Solution</b></summary>

```python
# --------------------------------------------------------------------------
# Q5. Production: batch load, paginate with scroll, filtered count/delete, quantization config
# --------------------------------------------------------------------------
def q5():
    client = QdrantClient(":memory:")
    client.create_collection(
        "prod",
        vectors_config=models.VectorParams(size=DIM, distance=models.Distance.COSINE),
        # server-side options (stored in embedded mode, enforced on a real server):
        hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100),
        quantization_config=models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8, quantile=0.99, always_ram=True)),
    )
    texts = [f"{t} (variant {i})" for i in range(6) for t in SAMPLE_DOCS]                 # 60 chunks
    vecs = embed_texts(texts, dim=DIM)
    points = [models.PointStruct(id=point_id(t), vector=v.tolist(), payload={"text": t, "variant": i % 6})
              for i, (t, v) in enumerate(zip(texts, vecs))]
    for s in range(0, len(points), 25):                                                    # batches of 25
        client.upsert("prod", points=points[s : s + 25])
    print("total points:", client.count("prod").count)

    seen, offset, pages = 0, None, 0                                                       # scroll = paginate everything
    while True:
        batch, offset = client.scroll("prod", limit=25, offset=offset, with_payload=False, with_vectors=False)
        seen += len(batch); pages += 1
        if offset is None:
            break
    print(f"scrolled {seen} points in {pages} pages")

    flt = models.Filter(must=[models.FieldCondition(key="variant", match=models.MatchValue(value=0))])
    print("count where variant=0:", client.count("prod", count_filter=flt, exact=True).count)
    client.delete("prod", points_selector=models.FilterSelector(filter=flt))
    print("after deleting variant 0:", client.count("prod").count)

    gb = lambda n, d, b: n * d * b / 1024**3
    print(f"1M x {DIM}d float32 = {gb(1_000_000, DIM, 4):.2f} GiB; int8 quantised copy = {gb(1_000_000, DIM, 1):.2f} GiB")
    return seen
```

</details>

**Why it works / what to learn**
Scroll is how you export or audit a collection without a vector query. Filtered `count`/`delete` are the building blocks of GDPR-style erasure ("delete everything for tenant X"). Scalar quantisation cuts the RAM for 1 M × 768-d from ≈2.9 GiB to ≈0.7 GiB; on a real server keep the originals on disk for re-scoring.

**Expected output (≈)**
```text
total points: 60
scrolled 60 points in 3 pages
count where variant=0: 10
after deleting variant 0: 50
1M x 768d float32 = 2.86 GiB; int8 quantised copy = 0.72 GiB
```
