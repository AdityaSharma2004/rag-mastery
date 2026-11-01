# Practice - CRUD, namespaces & metadata filtering

Five questions, rising difficulty. Attempt each before opening the hint, and the hint before the solution. Every solution is the exact code in `solutions.py` (`python solutions.py` runs all). They use Chroma (in-memory) and Qdrant (embedded) and embed with the free Gemini API (768-d); **Q4 is pure numpy** (no key). Shared imports and helper at the top of `solutions.py`:

```python
import chromadb, uuid
from qdrant_client import QdrantClient, models
from gemini_utils import *
DIM = 768
def uid(key): return str(uuid.uuid5(uuid.NAMESPACE_URL, key))     # Qdrant needs UUID/int ids
```

---

### Q1. Warm-up - the same CRUD cycle in two databases

**Problem**

For two chunks, do create, read-by-id, update-metadata-only and delete-by-id in **Chroma** and in **Qdrant**. Print what each returns after the update and the final counts.

<details><summary><b>Hint</b></summary>

Chroma: `upsert`, `get(ids=)`, `update(ids, metadatas=)`, `delete(ids=)`. Qdrant: `upsert`, `retrieve(ids=)`, `set_payload`, `delete(points_selector=[ids])`.

</details>

<details><summary><b>Solution</b></summary>

```python
# --------------------------------------------------------------------------
# Q1. Warm-up: the same CRUD cycle in Chroma and in Qdrant
# --------------------------------------------------------------------------
def q1():
    text_a, text_b = "Refunds take 7 days.", "Parcels ship in 3 days."
    va, vb = embed_texts([text_a, text_b], dim=DIM)

    # ---------- Chroma ----------
    ch = chromadb.EphemeralClient().create_collection("crud", metadata={"hnsw:space": "cosine"})
    ch.upsert(ids=["a", "b"], embeddings=[va, vb], documents=[text_a, text_b],            # CREATE
              metadatas=[{"topic": "refund", "v": 1}, {"topic": "shipping", "v": 1}])
    got = ch.get(ids=["a"])                                                               # READ by id
    ch.update(ids=["a"], metadatas=[{"topic": "refund", "v": 2}])                         # UPDATE metadata only
    ch.delete(ids=["b"])                                                                  # DELETE by id
    print("chroma :", got["documents"][0], "| after update:", ch.get(ids=["a"])["metadatas"][0], "| count:", ch.count())

    # ---------- Qdrant ----------
    qd = QdrantClient(":memory:")
    qd.create_collection("crud", vectors_config=models.VectorParams(size=DIM, distance=models.Distance.COSINE))
    qd.upsert("crud", points=[                                                            # CREATE
        models.PointStruct(id=uid("a"), vector=va.tolist(), payload={"text": text_a, "topic": "refund", "v": 1}),
        models.PointStruct(id=uid("b"), vector=vb.tolist(), payload={"text": text_b, "topic": "shipping", "v": 1})])
    got = qd.retrieve("crud", ids=[uid("a")], with_payload=True)                          # READ by id
    qd.set_payload("crud", payload={"v": 2}, points=[uid("a")])                           # UPDATE payload only
    qd.delete("crud", points_selector=[uid("b")])                                         # DELETE by id
    print("qdrant :", got[0].payload["text"], "| after update:",
          qd.retrieve("crud", ids=[uid("a")])[0].payload, "| count:", qd.count("crud").count)
    return ch.count(), qd.count("crud").count
```

</details>

**Why it works / what to learn**
The verbs are the same, the spellings differ. Metadata updates do not touch the vector; changing the text requires re-embedding and upserting.

**Expected output (≈)**
```text
chroma : Refunds take 7 days. | after update: {'topic': 'refund', 'v': 2} | count: 1
qdrant : Refunds take 7 days. | after update: {'text': 'Refunds take 7 days.', 'topic': 'refund', 'v': 2} | count: 1
```

---

### Q2. Apply - two multi-tenancy designs and an isolation test

**Problem**

Implement (A) one Chroma collection per tenant and (B) one Qdrant collection with a `tenant` payload filter. Query as `acme` in both and assert that no `globex` text ever appears.

<details><summary><b>Hint</b></summary>

A: `create_collection(f"tenant_{t}")` and query only that collection. B: `Filter(must=[FieldCondition(key="tenant", match=MatchValue(value="acme"))])`. The test is just `all(x.startswith("Acme") for x in results)`.

</details>

<details><summary><b>Solution</b></summary>

```python
# --------------------------------------------------------------------------
# Q2. Apply: three multi-tenancy designs and an isolation test
# --------------------------------------------------------------------------
TENANT_TEXTS = {
    "acme": ["Acme refunds take 7 days.", "Acme ships in 3 days."],
    "globex": ["Globex refunds take 14 days.", "Globex has 24x7 support."],
}


def q2():
    q = embed_query("how long do refunds take", dim=DIM)
    vecs = {t: embed_texts(txts, dim=DIM) for t, txts in TENANT_TEXTS.items()}

    # Design A - one Chroma COLLECTION per tenant (hard isolation, like one table per customer)
    ch = chromadb.EphemeralClient()
    for t, txts in TENANT_TEXTS.items():
        col = ch.create_collection(f"tenant_{t}", metadata={"hnsw:space": "cosine"})
        col.add(ids=[f"{t}{i}" for i in range(len(txts))], embeddings=vecs[t], documents=txts)
    a = ch.get_collection("tenant_acme").query(query_embeddings=[q], n_results=2)["documents"][0]

    # Design B - ONE Qdrant collection + `tenant` payload field (like a tenant_id column)
    qd = QdrantClient(":memory:")
    qd.create_collection("shared", vectors_config=models.VectorParams(size=DIM, distance=models.Distance.COSINE))
    qd.upsert("shared", points=[models.PointStruct(id=uid(f"{t}{i}"), vector=v.tolist(), payload={"tenant": t, "text": txt})
                                for t, txts in TENANT_TEXTS.items() for i, (txt, v) in enumerate(zip(txts, vecs[t]))])
    flt = lambda t: models.Filter(must=[models.FieldCondition(key="tenant", match=models.MatchValue(value=t))])
    b = [p.payload["text"] for p in qd.query_points("shared", query=q.tolist(), query_filter=flt("acme"), limit=4).points]

    # Isolation test: no text of another tenant may ever appear
    assert all(x.startswith("Acme") for x in a), a
    assert all(x.startswith("Acme") for x in b), b
    print("design A (collection per tenant):", a)
    print("design B (payload filter)        :", b)
    print("isolation test passed")
    return a, b
```

</details>

**Why it works / what to learn**
Design A isolates physically (you cannot forget a filter) but multiplies collections; design B scales to many tenants but its safety depends on the filter being applied everywhere, which is why the test matters.

**Expected output (≈)**
```text
design A (collection per tenant): ['Acme refunds take 7 days.', 'Acme ships in 3 days.']
design B (payload filter)        : ['Acme refunds take 7 days.', 'Acme ships in 3 days.']
isolation test passed
```

---

### Q3. Build - one interface, two backends, one contract test

**Problem**

Write `ChromaBackend` and `QdrantBackend` with the same methods (`upsert(items)`, `search(question, k, where)`, `delete_where(where)`, `count()`), where `where` is a neutral dict like `{"tenant": "acme"}`. Write a single `contract_test(backend)` asserting: upserting twice is idempotent; the tenant filter never leaks; a two-condition filter works; delete-by-filter removes the right rows. Run it against both.

<details><summary><b>Hint</b></summary>

Translate the neutral dict inside each backend: Chroma needs `{"$and": [...]}` for two keys; Qdrant uses several `must` conditions. Qdrant ids must be UUIDs, so store the original id in the payload.

</details>

<details><summary><b>Solution</b></summary>

```python
# --------------------------------------------------------------------------
# Q3. Build: ONE interface, two backends, ONE shared test-suite  (a "repository" / Laravel contract)
# --------------------------------------------------------------------------
class ChromaBackend:
    def __init__(self):
        self.col = chromadb.EphemeralClient().create_collection(f"kb_{uuid.uuid4().hex[:8]}", metadata={"hnsw:space": "cosine"})   # names: 3-512 chars

    @staticmethod
    def _where(where):
        if not where:
            return None
        conds = [{k: v} for k, v in where.items()]
        return conds[0] if len(conds) == 1 else {"$and": conds}          # Chroma: ONE operator per level

    def upsert(self, items):
        self.col.upsert(ids=[i["id"] for i in items], documents=[i["text"] for i in items],
                        embeddings=embed_texts([i["text"] for i in items], dim=DIM),
                        metadatas=[i["meta"] for i in items])

    def search(self, question, k=3, where=None):
        r = self.col.query(query_embeddings=[embed_query(question, dim=DIM)], n_results=k, where=self._where(where))
        return [(i, round(1 - d, 3), t, m) for i, d, t, m in zip(r["ids"][0], r["distances"][0], r["documents"][0], r["metadatas"][0])]

    def delete_where(self, where):
        self.col.delete(where=self._where(where))

    def count(self):
        return self.col.count()


class QdrantBackend:
    def __init__(self):
        self.c = QdrantClient(":memory:")
        self.c.create_collection("kb", vectors_config=models.VectorParams(size=DIM, distance=models.Distance.COSINE))

    @staticmethod
    def _flt(where):
        if not where:
            return None
        return models.Filter(must=[models.FieldCondition(key=k, match=models.MatchValue(value=v)) for k, v in where.items()])

    def upsert(self, items):
        vecs = embed_texts([i["text"] for i in items], dim=DIM)
        self.c.upsert("kb", points=[models.PointStruct(id=uid(i["id"]), vector=v.tolist(),
                                                       payload={**i["meta"], "_id": i["id"], "_text": i["text"]})
                                    for i, v in zip(items, vecs)])

    def search(self, question, k=3, where=None):
        r = self.c.query_points("kb", query=embed_query(question, dim=DIM).tolist(), query_filter=self._flt(where),
                                limit=k, with_payload=True).points
        return [(p.payload["_id"], round(p.score, 3), p.payload["_text"],
                 {a: b for a, b in p.payload.items() if not a.startswith("_")}) for p in r]

    def delete_where(self, where):
        self.c.delete("kb", points_selector=models.FilterSelector(filter=self._flt(where)))

    def count(self):
        return self.c.count("kb").count


ITEMS = [
    {"id": "a1", "text": "Acme refunds take 7 days.", "meta": {"tenant": "acme", "type": "policy"}},
    {"id": "a2", "text": "Acme ships in 3 days.", "meta": {"tenant": "acme", "type": "faq"}},
    {"id": "g1", "text": "Globex refunds take 14 days.", "meta": {"tenant": "globex", "type": "policy"}},
]


def contract_test(backend, name: str):
    backend.upsert(ITEMS); backend.upsert(ITEMS)                                   # idempotent
    assert backend.count() == 3, "upsert twice must not duplicate"
    hits = backend.search("how long do refunds take", k=3, where={"tenant": "acme"})
    assert hits and all(h[3]["tenant"] == "acme" for h in hits), "tenant filter leaked"
    two = backend.search("refunds", k=3, where={"tenant": "acme", "type": "faq"})   # two conditions at once
    assert [h[0] for h in two] == ["a2"], two
    backend.delete_where({"tenant": "globex"})
    assert backend.count() == 2
    print(f"  {name:<7} passed: upsert idempotent, tenant filter, 2-condition filter, delete-by-filter")


def q3():
    for name, cls in (("chroma", ChromaBackend), ("qdrant", QdrantBackend)):
        contract_test(cls(), name)
    return True
```

</details>

**Why it works / what to learn**
This is the repository pattern: callers never see database-specific syntax, and one test-suite proves the implementations behave identically. Adding pgvector later means one more class, passing the same test.

**Expected output (≈)**
```text
  chroma  passed: upsert idempotent, tenant filter, 2-condition filter, delete-by-filter
  qdrant  passed: upsert idempotent, tenant filter, 2-condition filter, delete-by-filter
```

---

### Q4. Debug - "my filtered search returns fewer than k results"

**Problem**

A corpus of 2,000 vectors; tenant `acme` owns only ~3%. A search takes the global top-20 and then keeps `acme` rows, returning too few (even zero). Reproduce it in numpy, show that pre-filtering fixes it, and write a fallback that widens the candidate pool until `k` results survive.

<details><summary><b>Hint</b></summary>

Pre-filter: `allowed = np.where(tenant == "acme")[0]` then rank only those. Fallback: double the pool size until you have `k` matches or the pool covers the corpus.

</details>

<details><summary><b>Solution</b></summary>

```python
# --------------------------------------------------------------------------
# Q4. Debug: "my filtered search returns fewer than k results" (post-filter vs pre-filter)
# --------------------------------------------------------------------------
def q4():
    rng = np.random.default_rng(3)
    n, d, k = 2000, 16, 5
    X = rng.normal(size=(n, d)); X /= np.linalg.norm(X, axis=1, keepdims=True)
    tenant = np.where(rng.random(n) < 0.03, "acme", "other")         # acme owns only ~3% of the data
    q = rng.normal(size=d); q /= np.linalg.norm(q)
    scores = X @ q

    # POST-filter: take the global top-k candidates first, THEN keep the acme ones
    cand = np.argsort(-scores)[:k * 4]                                 # even with 4x over-fetch
    post = [i for i in cand if tenant[i] == "acme"][:k]

    # PRE-filter: restrict to acme rows first, then rank
    allowed = np.where(tenant == "acme")[0]
    pre = allowed[np.argsort(-scores[allowed])[:k]]

    print(f"acme rows in corpus: {len(allowed)} of {n}")
    print(f"post-filter returned {len(post)} results (wanted {k})   <- the bug")
    print(f"pre-filter  returned {len(pre)} results")
    # Mitigation when you can only post-filter: widen the candidate pool until k survive (or give up)
    pool = k
    while True:
        cand = np.argsort(-scores)[:pool]
        got = [i for i in cand if tenant[i] == "acme"][:k]
        if len(got) >= k or pool >= n:
            break
        pool *= 2
    print(f"adaptive widening needed a candidate pool of {pool} to find {len(got)} results")
    return len(post), len(pre)
```

</details>

**Why it works / what to learn**
Post-filtering assumes matching rows are spread evenly among the nearest neighbours; for a small tenant they are not. Use a filter-aware index (Qdrant, Pinecone, Chroma `where`), or widen the pool, or partition by tenant.

**Expected output (≈)**
```text
acme rows in corpus: 50 of 2000
post-filter returned 0 results (wanted 5)   <- the bug
pre-filter  returned 5 results
adaptive widening needed a candidate pool of 320 to find 5 results
```
(Exact numbers depend on the random seed.)

---

### Q5. Production - provable tenant erasure on both backends

**Problem**

Load items for two tenants into each backend (7 items, 5 belong to `globex`). Write `erase_tenant(backend, tenant)` that deletes by filter and returns a report: count before, count after, and the number of search hits still returned for that tenant. Assert the erasure is complete.

<details><summary><b>Hint</b></summary>

Reuse `delete_where` from Q3. For proof, run a broad `search(..., k=50, where={"tenant": tenant})` after the delete and assert it returns `[]`.

</details>

<details><summary><b>Solution</b></summary>

```python
# --------------------------------------------------------------------------
# Q5. Production: tenant erasure (GDPR-style) with proof, on both backends
# --------------------------------------------------------------------------
def erase_tenant(backend, tenant: str) -> dict:
    before = backend.count()
    backend.delete_where({"tenant": tenant})
    leftovers = backend.search("refunds policy shipping support", k=50, where={"tenant": tenant})
    return {"before": before, "after": backend.count(), "leftover_hits_for_tenant": len(leftovers)}


def q5():
    items = ITEMS + [{"id": f"g{n}", "text": f"Globex note {n} about refunds", "meta": {"tenant": "globex", "type": "faq"}}
                     for n in range(2, 6)]
    for name, cls in (("chroma", ChromaBackend), ("qdrant", QdrantBackend)):
        b = cls(); b.upsert(items)
        report = erase_tenant(b, "globex")
        ok = report["leftover_hits_for_tenant"] == 0 and report["after"] == 2
        print(f"{name:<7} {report}  -> erasure verified: {ok}")
        assert ok
    return True
```

</details>

**Why it works / what to learn**
"Deleted" must be verifiable: counts change *and* a tenant-scoped search finds nothing. In production also purge derived copies (answer caches, backups, logs), otherwise the data lives on elsewhere.

**Expected output (≈)**
```text
chroma  {'before': 7, 'after': 2, 'leftover_hits_for_tenant': 0}  -> erasure verified: True
qdrant  {'before': 7, 'after': 2, 'leftover_hits_for_tenant': 0}  -> erasure verified: True
```
