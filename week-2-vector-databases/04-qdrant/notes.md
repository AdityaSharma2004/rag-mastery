# 04 - Qdrant: an open-source, self-hostable vector database

> Week 2 · Topic 4 of 7 · Time: ~75 min · Free: runs embedded in Python (no server) or in Docker · Embeddings: free Gemini API

## 🎯 What you will learn

- You will be able to run Qdrant three ways: in memory, in a local folder, and as a Docker server, with identical code.
- You will be able to create collections, upsert **points** (id + vector + payload), and query with `query_points`.
- You will be able to write payload filters with `must`, `should`, `must_not`, `MatchValue`, `Range`, and know why a payload index matters.
- You will be able to make ingestion idempotent (UUID ids from content), paginate with `scroll`, and delete or count by filter.
- You will be able to explain HNSW settings and quantisation, and when Qdrant beats Chroma, Pinecone or pgvector.

## 🧠 The big idea in plain words

Qdrant (say "quadrant") is a vector database written in Rust, **open source**, which you can run on your laptop, on your own server, or rent as a cloud service. It sits between the two extremes you already met:

- **Chroma** = easy, local, great for development.
- **Pinecone** = fully managed, you rent it, you cannot see inside.
- **Qdrant** = you can start with `QdrantClient(":memory:")` in a unit test, ship the *same code* to a Docker container, and move to Qdrant Cloud later. Free, controllable cost, serious filtering and compression features.

Laravel analogy: Qdrant is like **MySQL itself**: you can run it locally, on EC2 (self-host, cheapest at scale), or on RDS (managed). The client code barely changes between the three.

## 📚 Concepts, step by step

### 1. Vocabulary

| Qdrant | Meaning | Analogy |
|---|---|---|
| **Collection** | set of points sharing vector size + distance | table |
| **Point** | `id` + `vector` + `payload` | row |
| **Payload** | JSON metadata attached to a point | columns / `json` column |
| **Distance** | `COSINE`, `DOT`, `EUCLID`, `MANHATTAN` | fixed at collection creation |
| **Payload index** | index on a payload field to speed filters | `CREATE INDEX` |
| **Named vectors** | several vectors per point (e.g. dense + sparse, text + image) | several embedding columns |

### 2. Three ways to run it

```python
from qdrant_client import QdrantClient
QdrantClient(":memory:")                       # RAM only: tests, notebooks
QdrantClient(path="./qdrant_data")             # embedded, persisted to a folder (single process)
QdrantClient(url="http://localhost:6333")      # real server: docker run -p 6333:6333 qdrant/qdrant
```

The embedded modes are implemented in Python and are for development/small data; they accept server-only options (like quantisation) but may ignore them. For production use the server (or Qdrant Cloud), which runs the real Rust engine with HNSW, payload indexes and quantisation.

### 3. Collections

```python
client.create_collection(
    "docs",
    vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
)
```

Size and distance are fixed. With `COSINE`/`DOT`, **score = similarity (higher is closer)**. With `EUCLID` the score is a distance (**lower is closer**) (Q4). Qdrant normalises vectors itself for cosine.

### 4. Points and ids

```python
models.PointStruct(id="3f1c...-uuid", vector=[...768 floats...], payload={"text": "...", "tenant": "acme", "year": 2024})
```

**Ids must be unsigned integers or UUIDs. Arbitrary strings like `"doc-1"` are rejected.** The professional pattern: derive a UUID from the content/key, `uuid.uuid5(uuid.NAMESPACE_URL, "source|chunk_no")`. Same input, same id, so re-ingestion **overwrites** instead of duplicating.

### 5. Searching: `query_points`

```python
res = client.query_points("docs", query=vector, limit=5, with_payload=True, score_threshold=0.5)
for p in res.points:
    print(p.score, p.payload["text"])
```

Older tutorials use `client.search(...)`. It is deprecated; `query_points` is the current, more general API (it also does recommendations, fusion, grouping). `score_threshold` drops results worse than the cut-off.

### 6. Payload filters

```python
flt = models.Filter(
    must=[                                                        # AND
        models.FieldCondition(key="tenant", match=models.MatchValue(value="acme")),
        models.FieldCondition(key="year",   range=models.Range(gte=2024)),
    ],
    must_not=[models.FieldCondition(key="type", match=models.MatchValue(value="faq"))],
)
client.query_points("docs", query=v, query_filter=flt, limit=5)
```

| Clause | Meaning |
|---|---|
| `must` | all conditions true (AND) |
| `should` | at least one true (OR) |
| `must_not` | none true (NOT) |
| `MatchValue` / `MatchAny` | equality / value in list |
| `Range` | `gt, gte, lt, lte` for numbers |

**`should` where you meant `must` is a security bug** (it is an OR; Q4 shows a tenant leak).

Qdrant applies the filter **during** the HNSW traversal ("filterable HNSW") instead of after, so you still get your `limit` results even for selective filters. That is a major advantage over naive post-filtering (compare topic 03).

### 7. Payload indexes

On a server, create an index for every field you filter on:

```python
client.create_payload_index("docs", field_name="tenant", field_schema=models.PayloadSchemaType.KEYWORD)
client.create_payload_index("docs", field_name="year",   field_schema=models.PayloadSchemaType.INTEGER)
```

Without it Qdrant must inspect each candidate's payload (slow on big collections). For multi-tenant apps there is also a dedicated "tenant index" option for the tenant field. *(In embedded mode the call works but is a no-op and prints a warning.)*

### 8. HNSW and quantisation (what the server does for you)

- **HNSW settings**: `m` (graph links per node, default 16) and `ef_construct` (build candidate list, default 100) via `hnsw_config`; at query time `search_params=models.SearchParams(hnsw_ef=128)` trades speed for recall. Same ideas as pgvector (topic 03).
- **Scalar quantisation (int8)**: store each number as 1 byte instead of 4, keeping the original vectors on disk for optional re-scoring. ≈4× less RAM with small recall loss. **Binary quantisation** goes to 1 bit per number (≈32×), best for high-dimensional vectors.
- `on_disk` options keep vectors/payload on disk and only the hot index in RAM.

```python
quantization_config=models.ScalarQuantization(
    scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8, quantile=0.99, always_ram=True))
```

### 9. Housekeeping calls

| Task | Call |
|---|---|
| Count (optionally filtered) | `client.count("docs", count_filter=flt, exact=True)` |
| Paginate everything | `client.scroll("docs", limit=100, offset=offset)` returns `(points, next_offset)`; stop when `next_offset is None` |
| Get by id | `client.retrieve("docs", ids=[...])` |
| Update payload only | `client.set_payload("docs", payload={...}, points=[...])` |
| Delete by id or filter | `client.delete("docs", points_selector=[ids] or models.FilterSelector(filter=flt))` |
| Drop | `client.delete_collection("docs")` |

### 10. Qdrant vs the others

| | Chroma | Pinecone | pgvector | **Qdrant** |
|---|---|---|---|---|
| Hosting | local/self | managed only | your Postgres | **self-host or cloud** |
| Filtering | simple `where` | metadata filters | full SQL | **rich, filterable HNSW** |
| Quantisation | limited | managed | halfvec/bit | **scalar, product, binary** |
| Best for | prototypes | zero-ops | have Postgres, <~10M | **cost control, big filtered search** |

## 💻 Code walkthrough

### Stage 1 - collection and points

```python
import uuid
from qdrant_client import QdrantClient, models
from gemini_utils import embed_texts, embed_query, SAMPLE_DOCS

client = QdrantClient(":memory:")
client.create_collection("warmup", vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE))

def point_id(text): return str(uuid.uuid5(uuid.NAMESPACE_URL, text))

vecs = embed_texts(SAMPLE_DOCS, dim=768)
client.upsert("warmup", points=[
    models.PointStruct(id=point_id(t), vector=v.tolist(), payload={"text": t, "n": i})
    for i, (t, v) in enumerate(zip(SAMPLE_DOCS, vecs))])
```

`v.tolist()` converts numpy to plain floats. `upsert` is "insert or replace by id".

### Stage 2 - query

```python
q = embed_query("a young cat sleeping", dim=768).tolist()
for p in client.query_points("warmup", query=q, limit=3, with_payload=True).points:
    print(f"{p.score:.3f}  {p.payload['text'][:55]}")
```

Expected (≈): kitten and cat sentences first, scores ≈0.6-0.8.

### Stage 3 - filter

```python
flt = models.Filter(must=[models.FieldCondition(key="n", range=models.Range(lt=5))])
client.query_points("warmup", query=q, query_filter=flt, limit=3)
```

### Stage 4 - persist and reopen

```python
c = QdrantClient(path="./qdrant_data"); ...; c.close()
c = QdrantClient(path="./qdrant_data")          # data is still there
```

An embedded folder can be opened by one process at a time; use the Docker server for several apps.

### Stage 5 - the server

```bash
docker run -d -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

Dashboard at `http://localhost:6333/dashboard`. Only the constructor changes: `QdrantClient(url="http://localhost:6333")`.

## ⚠️ Common mistakes & debugging

| Symptom | Cause | Fix |
|---|---|---|
| `... is not a valid point ID` | String id like `"doc-1"` | `uuid5(...)` from your key, or an integer |
| `Vector dimension error` | Size mismatch with the collection | Pin `DIM`; create collection from the same constant |
| Other tenant's data appears | `should` used instead of `must` | `must` for AND; add a test for tenant isolation |
| `AttributeError: 'QdrantClient' object has no attribute 'search'` | Newer client removed it | Use `query_points(...).points` |
| Euclid scores "backwards" | Distance semantics: lower = closer | Sort/threshold accordingly, or use COSINE |
| Slow filtered queries on a server | No payload index | `create_payload_index` for each filtered field |
| `Storage folder already accessed by another instance` | Two processes on one embedded folder | Close the first client, or use the server |
| Duplicates after re-ingest | Random UUIDs | Deterministic `uuid5` ids |
| Warning "payload indexes have no effect in local mode" | Embedded mode | Harmless; effective on a server |

## 🚀 Pro-level insights (what the top 2% know)

- **Deterministic UUIDs + `wait=True` upserts** make ingestion idempotent and read-your-writes consistent (default `wait=True`).
- **Multi-tenancy:** one collection + a `tenant` payload field with a tenant payload index scales to thousands of tenants; one collection per tenant does not. Always filter with `must`, and add a unit test that proves isolation.
- **Quantise + re-score:** keep int8 vectors in RAM for the search, original float vectors on disk for rescoring the top candidates (`oversampling`). Most of the accuracy at a fraction of the RAM.
- **Named vectors** let one point hold a dense and a sparse (keyword) vector for hybrid search with server-side fusion (topic 06).
- **Tune per workload:** measure recall against exact search (`SearchParams(exact=True)`) before and after changing `m`, `ef_construct`, `hnsw_ef`.
- **Snapshots** give you backup/restore and shard transfer; combine with deterministic ids so a restore is verifiable.
- **When NOT to use Qdrant:** you already run Postgres and have under a few million vectors with heavy relational joins (pgvector), or you want zero ops (a fully managed service).

## 🧪 Mini lab (10-15 min)

1. Create an in-memory collection; upsert 12 chunks across two `tenant` values and two `year` values.
2. Query with: no filter, tenant filter, tenant + year range, and `must_not` on a type.
3. Change a `must` to a `should` and observe the leak.
4. Persist to a folder, close, reopen, and confirm the count.
5. Compute the RAM for 2 M vectors at 768-d float32 vs int8.

## ✅ Recap checklist

- [ ] I can run Qdrant in memory, embedded on disk, and in Docker.
- [ ] I use UUID/int ids derived from my own keys.
- [ ] I use `query_points`, and know cosine = higher-is-closer.
- [ ] I write filters with `must` / `should` / `must_not` correctly and create payload indexes on servers.
- [ ] I can scroll, count and delete by filter.
- [ ] I can explain scalar/binary quantisation and HNSW parameters.

## 🎤 Interview questions

**1. Why does Qdrant reject string ids like "doc-1"?**
Point ids are unsigned integers or UUIDs for compact, fast storage; derive a deterministic UUID from your own key.

**2. What is "filterable HNSW"?**
The payload filter is applied while traversing the graph, so selective filters still return `limit` results instead of being applied afterwards to a fixed candidate list.

**3. `should` vs `must`?**
`must` = AND, `should` = OR (at least one), `must_not` = NOT. Mixing them up in tenant filters leaks data.

**4. How does scalar quantisation save memory?**
Stores each float as one int8 byte (4× smaller); original vectors can stay on disk for re-scoring top candidates.

**5. When pick Qdrant over pgvector?**
Large filtered workloads, need for quantisation tiers, independent scaling of the vector tier, or self-hosted cost control beyond what Postgres comfortably handles.

## 🔗 Next

**05 - CRUD, namespaces & metadata filtering:** the same operations side by side across Chroma and Qdrant (with Pinecone and pgvector in a table), which unlocks designing multi-tenant retrieval correctly.
