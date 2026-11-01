# 05 - CRUD, namespaces & metadata filtering

> Week 2 · Topic 5 of 7 · Time: ~70 min · Chroma + Qdrant side by side; Pinecone and pgvector in a comparison table

## 🎯 What you will learn

- You will be able to perform create / read / update / delete on vectors in Chroma and Qdrant and translate each call to Pinecone and pgvector.
- You will be able to design **multi-tenant** retrieval: collection per tenant vs namespace vs payload/metadata filter, and pick the right one.
- You will be able to explain **pre-filtering vs post-filtering** and why post-filtering can silently return fewer than `k` results.
- You will be able to write one storage interface with several backends and prove they behave identically with a shared contract test.
- You will be able to implement verifiable tenant erasure (GDPR-style delete) and an isolation test.

## 🧠 The big idea in plain words

A vector database is a table where one column is "meaning". Everything else you do with it is the same set of verbs you use in Eloquent: **create, read, update, delete**, plus `where`. The twist is that the main read is not `find($id)` but "give me the 5 rows *nearest* to this vector, but only among rows where `tenant = 'acme'`."

Two things make this topic matter in real products:

1. **Idempotency.** Pipelines re-run (crashes, re-uploads). Each write must be safe to repeat: same id, same row, overwritten not duplicated. This is `updateOrCreate`, not `create`.
2. **Isolation.** A RAG feature for many customers must never show customer A a chunk from customer B. That is a security property you design and test, not hope for.

## 📚 Concepts, step by step

### 1. The verbs, side by side

| Verb | Eloquent | Chroma | Qdrant | Pinecone | pgvector (SQL) |
|---|---|---|---|---|---|
| Create / upsert | `updateOrCreate` | `col.upsert(ids, embeddings, documents, metadatas)` | `client.upsert(name, points=[PointStruct(...)])` | `index.upsert(vectors=[...], namespace=)` | `INSERT ... ON CONFLICT DO UPDATE` |
| Insert only if new | `create` | `col.add(...)` (existing ids ignored) | *(upsert only)* | *(upsert only)* | `INSERT` (error on duplicate key) |
| Read by id | `find($id)` | `col.get(ids=[...])` | `client.retrieve(name, ids=[...])` | `index.fetch(ids=[...])` | `SELECT ... WHERE id = ...` |
| Similarity search | n/a | `col.query(query_embeddings, n_results, where)` | `client.query_points(name, query, limit, query_filter)` | `index.query(vector, top_k, filter)` | `ORDER BY embedding <=> q LIMIT k` |
| Update metadata | `update` | `col.update(ids, metadatas)` | `client.set_payload(name, payload, points)` | `index.update(id, set_metadata=...)` | `UPDATE ... SET meta = ...` |
| Update the vector | `update` | `col.update(ids, embeddings=...)` / `upsert` | `upsert` same id | `upsert` same id | `UPDATE ... SET embedding = ...` |
| Delete by id | `delete` | `col.delete(ids=[...])` | `client.delete(name, points_selector=[ids])` | `index.delete(ids=[...])` | `DELETE ... WHERE id = ...` |
| Delete by filter | `where()->delete()` | `col.delete(where={...})` | `delete(points_selector=FilterSelector(filter=...))` | limited on serverless (use ids or namespace) | `DELETE ... WHERE ...` |
| Count | `count()` | `col.count()` | `client.count(name, count_filter=...)` | `describe_index_stats()` | `SELECT count(*)` |

**Important gotcha:** changing a chunk's *text* means its *vector* is stale. Always re-embed and **upsert** the new vector with the new text. Updating only the text/metadata leaves the old meaning in the index.

### 2. Idempotent ids

Same input must produce the same id:

```
id = sha1(f"{source}|{chunk_index}|{text}")      # or uuid5(...) for Qdrant, which requires UUIDs/ints
```

- Key on `source|chunk_index` if you want an *edit* to overwrite the old version of that chunk.
- Key on the text hash if you want identical text to deduplicate across files.
- When a document **shrinks**, chunks past the new end are not overwritten, so **delete by `source` first** (replace-by-source), then insert.

### 3. Metadata (payload) design

Store what you will **filter** on (`tenant`, `source`, `doc_type`, `year`, `language`, `acl_group`) and what you need to **cite** (`title`, `page`, `url`). Rules:
- Use flat, simple types (string / number / bool).
- Do not stuff whole documents into metadata.
- Decide the field names once; filter keys are case-sensitive and a typo silently matches nothing.

### 4. Filter syntax in each system

| Condition | Chroma `where` | Qdrant `Filter` | Pinecone `filter` |
|---|---|---|---|
| Equals | `{"tenant": "acme"}` | `must=[FieldCondition(key="tenant", match=MatchValue(value="acme"))]` | `{"tenant": "acme"}` |
| AND (two fields) | `{"$and": [{"a": 1}, {"b": 2}]}` (**required**) | two items in `must` | `{"a": 1, "b": 2}` or `$and` |
| OR | `{"$or": [...]}` | `should=[...]` | `{"$or": [...]}` |
| Range | `{"year": {"$gte": 2024}}` | `range=Range(gte=2024)` | `{"year": {"$gte": 2024}}` |
| In list | `{"type": {"$in": ["a","b"]}}` | `MatchAny(any=[...])` | `{"type": {"$in": [...]}}` |
| NOT | `{"type": {"$ne": "faq"}}` | `must_not=[...]` | `{"type": {"$ne": "faq"}}` |

### 5. Three multi-tenancy designs

| Design | How | Isolation | Cost / scale | Use when |
|---|---|---|---|---|
| **A. Collection / index per tenant** | one collection each | physical, strongest | many small indexes: overhead grows with tenant count | few large tenants, regulatory separation |
| **B. Namespace per tenant** (Pinecone) | partition inside one index | strong (a query searches one namespace) | cheap, hundreds+ fine | Pinecone apps |
| **C. Shared collection + tenant field** | `tenant` metadata + mandatory filter | logical: **only as safe as your filter** | best scale (thousands of tenants) | most SaaS apps, Qdrant (with a tenant payload index), pgvector (+ row-level security) |

Rule for design C: the filter is added in **one** place (a repository class), never by each caller, and an automated test proves isolation. Our Q3 does exactly that.

### 6. Pre-filtering vs post-filtering

- **Post-filter:** fetch the global top-N nearest, then drop rows that do not match. Fast and simple, but if the tenant owns 3% of the data you may get **zero results** from a pool of 20 (Q4).
- **Pre-filter:** restrict to matching rows first, then find the nearest. Always returns up to `k` (if that many exist) but needs index support to stay fast.
- **Filtered ANN** (Qdrant, Pinecone, Chroma, newer pgvector with iterative scans) applies the filter *during* the graph traversal, giving pre-filter correctness at near-ANN speed.

How to detect the bug: results count < `k` while more matching rows exist. Mitigations: widen candidates (`ef_search`, `n_results × factor`), use a filter-aware engine, partition by tenant, or fall back to exact search for small filtered subsets.

### 7. The repository pattern for vectors

Your app should not call a vector database directly everywhere. Define a small interface (a Laravel *contract*):

```python
class VectorStore(Protocol):
    def upsert(self, items): ...
    def search(self, question, k, where): ...
    def delete_where(self, where): ...
    def count(self): ...
```

Implement it for each database and write **one contract test-suite** that every implementation must pass (idempotent upsert, tenant filter, multi-condition filter, delete-by-filter). Swapping Chroma → Qdrant → pgvector then becomes a config change, and the tests tell you if behaviour drifted.

### 8. Deletion and erasure

"Delete customer X's data" must be **provable**:
1. delete by tenant filter (or drop the namespace/collection),
2. verify the count dropped,
3. run a search restricted to that tenant and assert zero results,
4. remember copies: caches (Week 4), backups, logs, and the source store.

## 💻 Code walkthrough

### Stage 1 - the same cycle in Chroma and Qdrant

```python
import chromadb, uuid
from qdrant_client import QdrantClient, models
from gemini_utils import embed_texts

va, vb = embed_texts(["Refunds take 7 days.", "Parcels ship in 3 days."], dim=768)
uid = lambda k: str(uuid.uuid5(uuid.NAMESPACE_URL, k))

# Chroma
ch = chromadb.EphemeralClient().create_collection("crud_demo", metadata={"hnsw:space": "cosine"})
ch.upsert(ids=["a", "b"], embeddings=[va, vb], documents=["Refunds take 7 days.", "Parcels ship in 3 days."],
          metadatas=[{"topic": "refund"}, {"topic": "shipping"}])
ch.update(ids=["a"], metadatas=[{"topic": "refund", "v": 2}])
ch.delete(ids=["b"])

# Qdrant
qd = QdrantClient(":memory:")
qd.create_collection("crud", vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE))
qd.upsert("crud", points=[models.PointStruct(id=uid("a"), vector=va.tolist(), payload={"topic": "refund"})])
qd.set_payload("crud", payload={"v": 2}, points=[uid("a")])
qd.delete("crud", points_selector=[uid("a")])
```

### Stage 2 - translating one filter

```python
# Chroma:  two conditions need $and
where = {"$and": [{"tenant": "acme"}, {"year": {"$gte": 2024}}]}
# Qdrant:
flt = models.Filter(must=[
    models.FieldCondition(key="tenant", match=models.MatchValue(value="acme")),
    models.FieldCondition(key="year", range=models.Range(gte=2024))])
```

### Stage 3 - one interface, two backends

```python
class ChromaBackend:
    def search(self, question, k=3, where=None):
        r = self.col.query(query_embeddings=[embed_query(question, dim=768)], n_results=k, where=self._where(where))
        ...
class QdrantBackend:
    def search(self, question, k=3, where=None):
        r = self.c.query_points("kb", query=..., query_filter=self._flt(where), limit=k, with_payload=True).points
        ...
```

Callers pass the same neutral `where={"tenant": "acme"}` to either; each backend translates it (Q3 has the full class and the contract test).

### Stage 4 - the post-filter bug in numpy

```python
cand = np.argsort(-scores)[:k * 4]                    # global top candidates
post = [i for i in cand if tenant[i] == "acme"][:k]   # may be EMPTY when acme is 3% of the data
```

versus ranking only `np.where(tenant == "acme")`. Q4 runs it.

## ⚠️ Common mistakes & debugging

| Symptom | Cause | Fix |
|---|---|---|
| Chroma `where` raises with two keys | Multiple conditions need an operator | Wrap in `$and` |
| Fewer than `k` results with a filter | Post-filtering / too small candidate pool | Filter-aware ANN, bigger `ef`/pool, partition, exact fallback |
| Old text still retrieved after edit | Only metadata/text updated; vector stale | Re-embed and upsert |
| Stale tail chunks after a file shrinks | Overwrote chunks 0..n-1 but not n..old | Delete by `source` before re-inserting |
| Another tenant's chunk appears | Filter missing in one code path, or `should` vs `must` | Single repository enforcing the filter; isolation test |
| Same id in two tenants collides | Shared id space in one collection | Prefix ids with tenant, or use namespaces |
| Qdrant rejects `"a1"` id | Ids must be UUID/int | `uuid5` from the key; keep the original id in the payload |
| Chroma rejects name `kb` | Collection names need 3-512 chars | Use descriptive names |

## 🚀 Pro-level insights (what the top 2% know)

- **Tenant id belongs in the primary key or id prefix**, not only in metadata: it prevents cross-tenant overwrites.
- **Make isolation a CI test:** ingest two tenants with similar content, query as A, assert nothing from B ever appears, for every code path (search, hybrid, rerank, cache).
- **Cache keys must include the tenant** (and ACL set). A semantic cache without it is a data leak (Week 4).
- **Version your chunks:** store `doc_version` and `content_hash` so you can re-embed only changed chunks and roll back.
- **Soft delete first** (a `deleted_at` metadata and a default filter), hard delete later, when you need an undo window; but erasure requests need hard delete + proof.
- **ACL filters:** store `allowed_groups` as a list and filter with `$in`/`MatchAny` using the caller's groups; never trust the client to send them.
- **Watch update cost:** in HNSW systems frequent updates fragment the index; batch changes and monitor.
- **When NOT to use shared-collection multi-tenancy:** contractual physical separation or tenants with billion-scale data; use a collection/cluster per tenant.

## 🧪 Mini lab (10-15 min)

1. Implement the `VectorStore` interface for Chroma and Qdrant (or reuse Q3).
2. Ingest 6 chunks for tenant `a` and 6 for `b` with deliberately similar text.
3. Run the contract tests; add one more: updating a chunk's text changes what is retrieved.
4. Erase tenant `b`; assert count and zero tenant-`b` search hits.
5. Make a deliberate bug (remove the filter in `search`) and watch the isolation test fail.

## ✅ Recap checklist

- [ ] I can map any CRUD verb across Chroma, Qdrant, Pinecone and pgvector.
- [ ] My ids are deterministic and my writes idempotent.
- [ ] I re-embed when text changes and delete-by-source when a file shrinks.
- [ ] I know when to use collection-per-tenant, namespaces, or a tenant field.
- [ ] I can explain pre- vs post-filtering and detect the "fewer than k" bug.
- [ ] I have a contract test and an isolation test.

## 🎤 Interview questions

**1. How do you make vector ingestion idempotent?**
Deterministic ids (hash of source, chunk index/text) and upsert semantics, plus delete-by-source before re-inserting a changed file.

**2. Pre-filter vs post-filter?**
Post-filter ranks globally then filters (may return too few); pre-filter restricts first (always up to k but needs index support); filtered ANN does it during traversal.

**3. Three ways to do multi-tenancy in a vector DB, and the trade-offs?**
Collection per tenant (strongest isolation, overhead), namespace per tenant (cheap, strong), shared collection with a tenant field (best scale, safety depends on the enforced filter).

**4. How would you prove tenant data was erased?**
Delete by filter/namespace, check counts, search restricted to that tenant returns nothing, and also purge caches, backups and logs.

**5. Why write a repository/interface around the vector DB?**
To swap backends, centralise the tenant filter, and run one contract test-suite against every implementation.

## 🔗 Next

**06 - Hybrid search (BM25 + vector):** combine keyword and semantic ranking with Reciprocal Rank Fusion, which unlocks exact-term recall on top of the multi-tenant search you just built.
