# 02 - Pinecone: a managed cloud vector database

> Week 2 · Topic 2 of 7 · Time: ~75 min · Needs: a free Pinecone Starter account + `PINECONE_API_KEY` (optional: every exercise also runs offline against a built-in fake)

## 🎯 What you will learn

- You will be able to explain what a *managed* vector database is, and when renting one beats running your own.
- You will be able to create a serverless index (dimension + metric), upsert vectors with metadata, query, fetch and delete.
- You will be able to use **namespaces** for multi-tenancy and **metadata filters** for scoping.
- You will be able to handle the real-world traps: eventual consistency, dimension mismatch, metadata limits, batching, retries.
- You will be able to judge cost, latency (network distance!) and lock-in before choosing Pinecone.

## 🧠 The big idea in plain words

In Week 1 you stored vectors in a numpy array and in Chroma on your own laptop. That works until you need: "my app runs on three servers", "the index must survive deploys", "10 million vectors", "somebody else gets paged at 3 a.m. when the database is down".

**Pinecone is a vector database you rent.** You do not install anything. You sign up, create an *index* (a named collection with a fixed vector size), and talk to it over HTTPS with an API key. Pinecone handles storage, indexing (the approximate nearest-neighbour search from topic 04), replication and scaling.

Laravel analogy: **Chroma is SQLite, Pinecone is hosted MySQL on AWS RDS.** With RDS you give up some control and pay a bill, and in return you never patch, back up or scale a server. You also gain a network hop: every query now crosses the internet to a data centre, so *where* the database lives matters for latency.

What Pinecone is **not**: it does not create embeddings for you in our setup (we use Gemini), it does not store your original documents as a separate field (the chunk text goes into metadata), and it is not a general SQL database.

## 📚 Concepts, step by step

### 1. Vocabulary

| Pinecone term | Meaning | Laravel / MySQL analogy |
|---|---|---|
| **Project** | Your account's workspace, owns an API key | RDS account/project |
| **Index** | One vector collection; fixed `dimension` and `metric` | a table |
| **Namespace** | A partition inside an index | `WHERE tenant_id = ?` baked into the storage layer |
| **Record / vector** | `id` + `values` (floats) + optional `metadata` | a row |
| **Metadata** | key-value pairs you can filter on | columns |
| **Serverless spec** | Cloud + region where the index lives | RDS region |

### 2. Serverless vs pod-based

Pinecone offers *serverless* indexes (you pay for storage and read/write operations; capacity scales automatically) and older *pod-based* indexes (you reserve hardware). For learning and most new apps use **serverless**, which is what the free Starter plan provides. Starter currently restricts you to one cloud/region (AWS `us-east-1` at the time of writing) and has small limits on storage and number of indexes. **Always read the current limits on the pricing page**: they change, and inactive Starter indexes may be paused.

### 3. Index settings you cannot change later

```python
pc.create_index(
    name="rag-course-w2",
    dimension=768,          # must equal your embedding size (Gemini at 768)
    metric="cosine",        # cosine | dotproduct | euclidean
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)
```

- **dimension** is fixed. Sending a 1536-d vector to a 768-d index is rejected (Q4).
- **metric** is fixed. Use `cosine` for our unit vectors (scores come back as cosine similarity, **higher = closer**, unlike Chroma which returns distances).
- Changing either means creating a new index and re-embedding (the blue/green migration from topic 01).

### 4. Records: id, values, metadata

```python
{"id": "doc-12",
 "values": [0.013, -0.027, ...],               # 768 floats
 "metadata": {"text": "Queues run jobs...", "source": "laravel.md", "year": 2024, "tags": ["queue", "redis"]}}
```

Metadata rules to remember: values may be **string, number, boolean, or list of strings**; no nested objects; no `null`; and the total size per record is capped (tens of KB). Because Pinecone has no separate "documents" field, **store the chunk text in metadata** (keep it reasonably small) so the search result is self-contained.

### 5. Upsert, query, fetch, delete

All data-plane methods take **keyword arguments only** in the current SDK.

```python
index.upsert(vectors=[...], namespace="acme")                       # insert or overwrite by id
res = index.query(vector=q, top_k=5, namespace="acme",
                  include_metadata=True,
                  filter={"year": {"$gte": 2024}})                  # nearest neighbours
index.fetch(ids=["doc-12"], namespace="acme")                       # get by id
index.delete(ids=["doc-12"], namespace="acme")                      # delete by id
index.delete(delete_all=True, namespace="acme")                     # wipe one namespace
index.describe_index_stats()                                        # counts per namespace
```

`res.matches` is a list sorted best-first; each match has `.id`, `.score`, and `.metadata` (only if you asked with `include_metadata=True`).

### 6. Namespaces = cheap multi-tenancy

A query searches **exactly one namespace**. Put each customer (or each document set, or each environment) in its own namespace and tenants can never see each other's vectors, with no filter that someone can forget to add. The same `id` can exist in two namespaces independently (Q2). Namespaces are created implicitly the first time you upsert into them. Deleting a whole tenant is one call.

When to use a namespace vs a metadata filter:

| Need | Use |
|---|---|
| Hard isolation (customers, environments) | **Namespace** |
| Soft scoping inside one corpus (year, doc type, language) | **Metadata filter** |
| Both | Namespace per tenant + filters inside it |

### 7. Metadata filter syntax

Mongo-like operators: `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`, `$exists`, and logical `$and` / `$or`.

```python
filter={"source": "laravel.md", "year": {"$gte": 2023}}                       # several keys = AND
filter={"$or": [{"type": "faq"}, {"type": "policy"}]}
filter={"tags": {"$in": ["redis", "queue"]}}
```

Filters are applied *during* the nearest-neighbour search (so you still get `top_k` results if enough records match), not after. Only filter on fields you actually need; very high-cardinality metadata can slow indexing.

### 8. Eventual consistency (the thing that surprises everyone)

On serverless, a successful `upsert` does **not** guarantee the record is immediately visible to `query`. Writes are acknowledged, then become searchable within seconds. Tests like "upsert then immediately assert the result" will be flaky. Solution: poll `describe_index_stats()` until the count you expect appears (our `wait_for_count`), or design the app to tolerate a short delay.

### 9. Batching and limits

Do not upsert one vector per request, and do not send everything at once. Typical guidance: batches of **~100 vectors** (request size is capped at a few MB and a maximum count per call). The SDK also accepts `batch_size=` and can parallelise. Wrap each batch in retry with back-off (429/5xx), and checkpoint progress so a crash resumes (Q5).

### 10. Cost and latency model

- **Storage** is proportional to `vectors × dimension × 4 bytes` (plus metadata). 768-d instead of 3072-d is 4× cheaper to store.
- **Reads/writes** are billed in "units" per operation; top-k size and filters affect read cost. Check the pricing page; Starter is free but capped.
- **Latency = network + search.** Search itself is typically tens of milliseconds; the round trip from India to AWS us-east-1 can add well over 200 ms. Co-locate your app and index in the same region in production; for a demo, accept it.

### 11. Lock-in and portability

Your vectors come from your own embedding model, so you can always re-index elsewhere (Qdrant, pgvector). Keep the **source text and a content hash** in your own system (database or object storage) as the source of truth, and treat the Pinecone index as a rebuildable *derived* store.

## 💻 Code walkthrough

### Stage 0 - setup

```bash
pip install -r requirements.txt            # includes pinecone
# .env:  GEMINI_API_KEY=...   PINECONE_API_KEY=...
```

### Stage 1 - connect and create an index (once)

```python
import os, time
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
name = "rag-course-w2"
if not pc.has_index(name):
    pc.create_index(name=name, dimension=768, metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"))
    while not pc.describe_index(name).status["ready"]:      # takes ~30-60 s
        time.sleep(3)
index = pc.Index(name)
```

`has_index` makes this idempotent, like `Schema::hasTable()` before `create`. *(Version note: the SDK has changed shape across major versions; if `.status["ready"]` errors on yours, use `getattr(status, "ready")`. The solutions file handles both.)*

### Stage 2 - embed and upsert

```python
from gemini_utils import embed_texts, embed_query, SAMPLE_DOCS

vecs = embed_texts(SAMPLE_DOCS, dim=768)                   # unit vectors, float32
index.upsert(
    vectors=[{"id": f"doc-{i}", "values": v.tolist(), "metadata": {"text": t}}
             for i, (t, v) in enumerate(zip(SAMPLE_DOCS, vecs))],
    namespace="warmup",
)
```

`v.tolist()` converts numpy floats to plain Python floats (the HTTP client needs JSON-serialisable values).

### Stage 3 - wait, then query

```python
while index.describe_index_stats().namespaces.get("warmup", None) is None:
    time.sleep(1)                                           # simple readiness check

q = embed_query("a young cat sleeping", dim=768).tolist()
res = index.query(vector=q, top_k=3, namespace="warmup", include_metadata=True)
for m in res.matches:
    print(f"{m.score:.3f}  {m.id}  {m.metadata['text'][:55]}")
```

Expected (≈): the kitten and cat sentences first, scores around 0.6-0.8.

### Stage 4 - a filtered query

```python
res = index.query(vector=q, top_k=5, namespace="warmup", include_metadata=True,
                  filter={"n": {"$lt": 5}})                 # only records whose metadata n < 5
```

### Stage 5 - clean up

```python
index.delete(delete_all=True, namespace="warmup")
# pc.delete_index(name)        # when you are done with the whole index
```

## ⚠️ Common mistakes & debugging

| Symptom | Cause | Fix |
|---|---|---|
| `Vector dimension 1536 does not match the dimension of the index 768` | Index and embeddings disagree | Create the index with the dimension you embed at; pin `DIM` in one config |
| Query returns `[]` right after upsert | Eventual consistency | Poll `describe_index_stats()`; add a short wait in tests |
| Query returns `[]` although data exists | Wrong namespace (default is `""`) | Pass the same `namespace=` to upsert and query |
| `match.metadata` is `None`/empty | `include_metadata=True` missing | Add it |
| `Metadata ... exceeds size limit` | Whole document stuffed into metadata | Store only the chunk text (a few KB); keep the full doc elsewhere |
| 400 error on metadata value | `None`, nested dict, or list of non-strings | Clean metadata to str/num/bool/list[str] |
| 429 / timeouts during bulk load | Too many requests | Batch (~100), retry with back-off, lower concurrency |
| Scores look "reversed" vs Chroma | Pinecone returns *similarity* for cosine, Chroma returns *distance* | Do not copy thresholds between databases without converting |
| Index disappears / is paused | Starter inactivity policy | Treat the index as rebuildable; keep source data and a re-ingest script |

## 🚀 Pro-level insights (what the top 2% know)

- **Treat Pinecone as a derived cache of your embeddings, not your system of record.** Keep chunks, hashes and metadata in Postgres/MySQL/S3; a re-index job rebuilds everything. That makes migration between vector stores a one-day task, not a crisis.
- **Namespace per tenant** is the cleanest multi-tenant pattern here, and deleting a tenant (GDPR erasure!) is a single call. Hundreds of namespaces are fine; thousands need design thought.
- **Region matters more than index size** for perceived speed. Put your API next to the index.
- **Use deterministic ids** (content hash or `doc_id#chunk_no`) so re-ingestion overwrites instead of duplicating, and deletion of a document's chunks is possible by id list.
- **Delete-by-metadata is limited on serverless** (support has varied by index type); keep a list of chunk ids per document in your own database so you can delete precisely.
- **Hybrid search:** Pinecone supports sparse vectors alongside dense for keyword+semantic fusion; evaluate if exact-term recall matters. Otherwise do BM25 yourself (topic 06).
- **Cost control:** smaller dimension, avoid huge `top_k`, avoid fetching `include_values=True`, cache frequent queries (Week 4).
- **When NOT to use Pinecone:** data cannot leave your VPC/country, you already run Postgres and have < a few million vectors (use pgvector), or you want zero vendor bills (self-host Qdrant).

## 🧪 Mini lab (10-15 min)

1. Create the index (or run with the fake). Upsert 10 chunks into namespace `lab-a` and 10 different chunks into `lab-b`.
2. Ask one question in each namespace; confirm results never cross.
3. Add `year` metadata to the records and run a query with `{"year": {"$gte": 2024}}`.
4. Print `describe_index_stats()` and delete `lab-b` entirely. Confirm `lab-a` is untouched.
5. Time one query with `time.perf_counter()`. How much of it is network?

## ✅ Recap checklist

- [ ] I can explain Chroma vs Pinecone with the SQLite vs RDS analogy.
- [ ] I know dimension and metric are fixed at creation.
- [ ] I can write upsert / query / fetch / delete with keyword arguments.
- [ ] I use namespaces for isolation and filters for scoping.
- [ ] I handle eventual consistency, batching and retries.
- [ ] I keep my source of truth outside Pinecone.

## 🎤 Interview questions

**1. When would you pick a managed vector DB over a local one?**
When several app servers need shared, durable, scalable storage and you do not want to operate infrastructure; accept cost, network latency and some lock-in.

**2. What is a namespace and why use it?**
A partition inside an index that a query searches exclusively; ideal for tenant isolation and fast bulk deletion without writing filters that could be forgotten.

**3. Your test upserts a vector and queries it immediately but gets nothing. Why?**
Serverless indexes are eventually consistent; poll index stats until the count appears or wait briefly.

**4. How do you migrate from a 768-d to a 1536-d index?**
Create a new index with the new dimension, re-embed from the source text, verify retrieval quality, switch the app's pointer, then delete the old index (blue/green).

**5. Why store chunk text in metadata?**
Pinecone returns ids, scores and metadata, not separate documents; keeping the chunk text in metadata (within size limits) lets one query return everything the prompt needs.

## 🔗 Next

**03 - pgvector:** the same ideas inside Postgres, so you can use SQL joins and transactions with vectors, which unlocks "no extra database" production setups.
