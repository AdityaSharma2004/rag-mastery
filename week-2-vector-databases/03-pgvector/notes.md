# 03 - pgvector: vector search inside PostgreSQL

> Week 2 · Topic 3 of 7 · Time: ~90 min · Needs: Docker (or any Postgres with the pgvector extension) · Free, local, production-grade

## 🎯 What you will learn

- You will be able to run Postgres with pgvector, create a `vector(n)` column and insert/query embeddings with plain SQL.
- You will be able to explain the three distance operators (`<->`, `<=>`, `<#>`) and the index families (exact, IVFFlat, **HNSW**).
- You will be able to combine vector search with `JOIN`, `WHERE`, transactions, `ON CONFLICT` upserts and row-level security, which dedicated vector DBs cannot match.
- You will be able to read `EXPLAIN` to prove an index is used, and tune HNSW (`m`, `ef_construction`, `ef_search`) with a recall-vs-latency measurement.
- You will be able to decide when pgvector beats a dedicated vector database.

## 🧠 The big idea in plain words

You already run a relational database for your app: users, orders, documents. pgvector adds **one new column type, `vector`**, and **three new operators** to PostgreSQL. Now your embeddings live in the same database as the rest of your data, behind the same backups, permissions, transactions and SQL.

Laravel analogy: you have Eloquent models over MySQL. pgvector means "store the embedding as another column on the `chunks` table, and `ORDER BY` distance". There is no second system to deploy, secure, back up or keep in sync. A `Document` that is deleted cascades to its chunks and embeddings, **atomically**, because it is one database.

The honest trade-off: Postgres is a general-purpose engine. It is brilliant up to a few million vectors on a decent machine, and dedicated engines (Qdrant, Pinecone) pull ahead at hundreds of millions of vectors or when you need specialised features. For most RAG apps, especially one that already has a Postgres, **pgvector is the boring, correct choice**.

(Your existing knowledge is MySQL. The SQL below is standard; the differences you will notice are `serial/bigserial` instead of `AUTO_INCREMENT`, `%s` placeholders in psycopg, `ON CONFLICT ... DO UPDATE` instead of `ON DUPLICATE KEY UPDATE`, and `jsonb`.)

## 📚 Concepts, step by step

### 1. Installing it

Easiest: the official Docker image that already contains Postgres + pgvector.

```bash
docker run -d --name pgvector \
  -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=ragdb \
  -p 5432:5432 pgvector/pgvector:pg16
```

Then, once per database:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

On AWS (your stack): **RDS for PostgreSQL and Aurora PostgreSQL both support pgvector**; so do Supabase, Neon and most managed Postgres services. Connection string used in this course: `DATABASE_URL=postgresql://postgres:postgres@localhost:5432/ragdb`.

### 2. The `vector` type

```sql
CREATE TABLE chunks (
  id        bigserial PRIMARY KEY,
  doc_id    int REFERENCES documents(id) ON DELETE CASCADE,
  content   text NOT NULL,
  embedding vector(768)          -- dimension is part of the type, like VARCHAR(255)
);
```

Inserting a wrong-size vector fails immediately (`expected 768 dimensions, not 1536`), a built-in guard that Chroma lacks. Vectors are 4-byte floats, so storage is `dims × 4` bytes per row plus overhead. *(Newer pgvector versions (0.7+) also have `halfvec` (2-byte) and `bit`/`sparsevec` types.)*

**Important limit:** a plain `vector` column can only be **indexed up to 2,000 dimensions**. Gemini's native 3072-d vectors cannot get an ANN index as `vector`. That is one more practical reason to use **768 or 1536** dimensions (topic 06 of Week 1). (With `halfvec` the indexable limit is higher, if your pgvector is new enough.)

### 3. Distance operators

| Operator | Meaning | `ORDER BY ...` returns closest first when value is | Convert to cosine similarity (unit vectors) |
|---|---|---|---|
| `<=>` | **cosine distance** (`1 - cos`) | smallest | `1 - (a <=> b)` |
| `<->` | Euclidean (L2) distance | smallest | `1 - (a <-> b)^2 / 2` |
| `<#>` | **negative** inner product | most negative | `-(a <#> b)` |

Why is `<#>` negative? Postgres index scans only know "ascending order = nearest". To make "largest dot product" look like "smallest value", pgvector returns the dot product times -1. For our unit vectors we use **`<=>`** (cosine), the clearest.

### 4. A query

```sql
SELECT id, content, 1 - (embedding <=> $1) AS similarity
FROM chunks
ORDER BY embedding <=> $1          -- keep the bare operator here (see "index rules")
LIMIT 5;
```

`$1` is the query vector (psycopg sends it as a parameter, after `register_vector(conn)`).

### 5. Without an index: exact search

With no index, Postgres computes the distance to every row (a **sequential scan**): 100% accurate, O(N). That is perfectly fine for thousands of rows and often up to ~100K. Never add an index before you need one; exact search is your ground truth for measuring recall.

### 6. Approximate indexes: IVFFlat vs HNSW

| | **HNSW** | IVFFlat |
|---|---|---|
| Idea | multi-layer "small-world" graph; hop towards the query | k-means clusters ("lists"); search only the nearest few clusters ("probes") |
| Build | slower, more memory | fast, small |
| Needs data before building? | no (can build on empty table) | **yes** (clusters learned from existing rows) |
| Query speed/recall | better recall at the same speed | good, needs tuning of `probes` |
| Tuning knobs | `m`, `ef_construction` (build); `hnsw.ef_search` (query) | `lists` (build); `ivfflat.probes` (query) |
| Default choice for RAG | ✅ | when build time/memory is the bottleneck |

```sql
CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

SET hnsw.ef_search = 100;       -- per session: higher = better recall, slower queries
```

- `m` = links per node (default 16): higher = better recall, bigger index.
- `ef_construction` = candidate list while building (default 64): higher = better graph, slower build.
- `ef_search` = candidate list while querying (default 40): the main runtime recall/latency dial (Q5 measures it).

### 7. The index rules (the #1 source of "why is it slow?")

An index is used only if **the query matches it exactly**:

1. **Operator class must match the operator.** Index built with `vector_cosine_ops` serves `<=>`. A query using `<->` or `<#>` will not use it (Q4).
2. **`ORDER BY <bare distance expression> LIMIT k`.** Wrapping the distance (`ORDER BY 1 - (embedding <=> q) DESC`) hides it from the planner, so it falls back to a sequential scan (Q4).
3. **You need `LIMIT`.** The index is a nearest-neighbour index, not a filter.
4. **Check with `EXPLAIN`:** look for `Index Scan using ... hnsw` rather than `Seq Scan`.

Operator classes: `vector_cosine_ops`, `vector_l2_ops`, `vector_ip_ops`.

### 8. Filtering: the subtle part

```sql
SELECT ... FROM chunks c JOIN documents d ON d.id = c.doc_id
WHERE d.tenant = 'acme' AND d.year >= 2024
ORDER BY c.embedding <=> $1 LIMIT 5;
```

This is where pgvector shines: the full power of SQL (joins, `IN`, ranges, `jsonb` filters). But with an approximate index the engine first fetches roughly `ef_search` nearest candidates, **then** applies `WHERE`. If the filter is very selective you can get **fewer than `LIMIT` rows back**. Mitigations:

- raise `hnsw.ef_search`,
- use **partial indexes** (`CREATE INDEX ... WHERE tenant = 'acme'`) or **partitioning** per tenant for big tenants,
- on newer pgvector (0.8+) enable **iterative index scans** (`SET hnsw.iterative_scan = relaxed_order`) so the index keeps searching until enough rows pass the filter,
- fall back to exact search for tiny filtered subsets (the planner often does this itself).

### 9. SQL superpowers you get for free

| Need | pgvector answer |
|---|---|
| Upsert idempotently | `INSERT ... ON CONFLICT (source, chunk_index) DO UPDATE` |
| Replace a file's chunks atomically | `BEGIN; DELETE ...; INSERT ...; COMMIT;` (Q3) |
| Delete a document + its chunks | `ON DELETE CASCADE` |
| Tenant isolation | **Row-Level Security** policies, or `WHERE tenant_id = ...` |
| Keyword + vector hybrid | `tsvector` + GIN full-text index, fuse with RRF (topic 06) |
| Backup, replication, PITR | all the Postgres tooling you already trust |
| Joins with business data | ordinary `JOIN` |

### 10. When pgvector is (and is not) the right tool

| Choose pgvector when | Choose a dedicated DB when |
|---|---|
| You already run Postgres/RDS/Aurora | Hundreds of millions of vectors / billions |
| < ~5-10 M vectors | Heavy QPS needing independent vector scaling |
| You need joins, transactions, RLS | You need specialised features (quantisation tiers, multi-vector, built-in sparse+dense, cluster sharding) |
| One system to operate | A separate team operating the vector tier |

## 💻 Code walkthrough

### Stage 0 - start Postgres, connect from Python

```bash
docker run -d --name pgvector -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=ragdb -p 5432:5432 pgvector/pgvector:pg16
pip install "psycopg[binary]" pgvector
```

```python
import os, psycopg, numpy as np
from pgvector.psycopg import register_vector

conn = psycopg.connect("postgresql://postgres:postgres@localhost:5432/ragdb", autocommit=True)
conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
register_vector(conn)          # must come AFTER the extension exists
```

`register_vector` teaches psycopg to convert numpy arrays to/from the `vector` type. Forgetting it gives confusing type errors.

### Stage 1 - table and data

```python
from gemini_utils import embed_texts, embed_query, SAMPLE_DOCS

conn.execute("DROP TABLE IF EXISTS items")
conn.execute("CREATE TABLE items (id bigserial PRIMARY KEY, content text, embedding vector(768))")

vecs = embed_texts(SAMPLE_DOCS, dim=768)
with conn.cursor() as cur:
    cur.executemany("INSERT INTO items (content, embedding) VALUES (%s, %s)",
                    [(t, v) for t, v in zip(SAMPLE_DOCS, vecs)])
```

### Stage 2 - nearest neighbours

```python
q = embed_query("a young cat sleeping", dim=768)
rows = conn.execute("""SELECT content, 1 - (embedding <=> %s) AS similarity
                       FROM items ORDER BY embedding <=> %s LIMIT 3""", (q, q)).fetchall()
for content, sim in rows:
    print(f"{sim:.3f}  {content[:50]}")
```

Note `q` appears twice (one per placeholder). Expected (≈): the kitten and cat sentences, similarity ≈0.6-0.8.

### Stage 3 - add an HNSW index and prove it is used

```python
conn.execute("CREATE INDEX ON items USING hnsw (embedding vector_cosine_ops)")
plan = conn.execute("EXPLAIN SELECT id FROM items ORDER BY embedding <=> %s LIMIT 3", (q,)).fetchall()
print("\n".join(r[0] for r in plan))
```

With only 10 rows Postgres may still choose a sequential scan (cheaper). To see the index in action use thousands of rows (Q4 builds 20,000).

### Stage 4 - bulk load fast

For many rows use `COPY` instead of row-by-row `INSERT` (Q4/Q5 show it): it can be 10-100× faster. Build the index **after** bulk loading; building on a loaded table is faster than maintaining it per insert. Raise `maintenance_work_mem` for big builds.

### Stage 5 - tune `ef_search`

```python
for ef in (10, 40, 100, 200):
    conn.execute(f"SET hnsw.ef_search = {ef}")
    # run your queries, compare against exact search, record recall and latency  (Q5)
```

## ⚠️ Common mistakes & debugging

| Symptom | Cause | Fix |
|---|---|---|
| `expected 768 dimensions, not 1536` | Column and embeddings disagree | Fix the pipeline dim, or migrate the column (new table + re-embed) |
| `can't adapt type 'numpy.ndarray'` | Forgot `register_vector(conn)` (or called before the extension existed) | `CREATE EXTENSION` first, then `register_vector` |
| Query is slow, `EXPLAIN` shows `Seq Scan` | Operator ≠ index opclass, or `ORDER BY` wraps the distance, or no `LIMIT` | Match `<=>` with `vector_cosine_ops`; `ORDER BY embedding <=> q LIMIT k` |
| `column cannot have more than 2000 dimensions` when indexing | 3072-d `vector` column | Use 768/1536, or `halfvec` on pgvector ≥ 0.7 |
| Fewer rows than `LIMIT` with a `WHERE` | Index returned `ef_search` candidates, then filtered | Raise `ef_search`, partial indexes, iterative scans (0.8+), or partition |
| IVFFlat recall is terrible | Index built on an empty/tiny table | Build after loading data; set `probes` ~ √lists |
| Insert is slow | Row-by-row inserts with a live HNSW index | `COPY`, or drop index → load → recreate |
| Similarity > 1 or negative | Mixed up `<#>` (negative dot) with similarity | Convert per the operator table |
| Out-of-memory / slow after growth | HNSW index no longer fits RAM | More RAM, smaller dim, `halfvec`, or a dedicated DB |

## 🚀 Pro-level insights (what the top 2% know)

- **Keep the index in RAM.** HNSW is fast because the graph stays cached; give Postgres enough `shared_buffers` / OS cache for `table + index`. A "slow pgvector" is usually "index spilled to disk".
- **Embedding dimension is an infrastructure decision:** 768-d at 1 M rows ≈ 3 GB raw; 3072-d ≈ 12 GB *and* unindexable as `vector`. Choose 768/1536.
- **Always keep an exact-search path** (`SET enable_indexscan = off` in a test session) to compute true neighbours and measure recall. If you cannot measure recall, you are guessing.
- **One transaction per logical change.** Re-ingesting a document = delete old chunks + insert new ones in one transaction; readers never see a half-updated file. Do the (slow) embedding call **outside** the transaction.
- **Row-Level Security** is the safest multi-tenant design: even a buggy query cannot leak another tenant's chunks.
- **Hybrid in one database:** `to_tsvector` + GIN for keyword, `<=>` for semantic, fuse with RRF in SQL or Python (topic 06). No Elasticsearch needed.
- **Version-check features** (halfvec, iterative scans, sparsevec depend on the pgvector version). `SELECT extversion FROM pg_extension WHERE extname='vector';`
- **Vacuum and autovacuum matter:** frequent updates/deletes on HNSW tables create bloat; monitor and `REINDEX` occasionally.
- **When NOT to use it:** you already know you will exceed ~50-100 M vectors, or you cannot give the database enough RAM.

## 🧪 Mini lab (10-15 min)

1. Start the Docker container; create the table; insert 10 chunks.
2. Run the same question with `<=>`, `<->`, `<#>`. Confirm the **order** is identical for unit vectors and the values relate as in the table above.
3. Add a `tenant` column and a `WHERE tenant = 'a'` filter; check the results are scoped.
4. Insert 20,000 random 32-d vectors, build HNSW, and compare `EXPLAIN` before/after the index.
5. Change `ef_search` from 10 to 100 and write down recall and time.

## ✅ Recap checklist

- [ ] I can create a `vector(n)` column and query with `<=>`.
- [ ] I know `<#>` is the *negative* inner product.
- [ ] I can explain HNSW vs IVFFlat and the roles of `m`, `ef_construction`, `ef_search`.
- [ ] I know the index rules: matching opclass, bare `ORDER BY`, `LIMIT`; and I check `EXPLAIN`.
- [ ] I understand the filtered-search pitfall and its mitigations.
- [ ] I know the 2,000-dimension indexing limit and why I use 768/1536.

## 🎤 Interview questions

**1. Why would you choose pgvector over a dedicated vector database?**
You already run Postgres; you gain transactions, joins, RLS, one backup/ops story and no sync between systems, which is ideal up to a few million vectors.

**2. A pgvector query is slow. What do you check first?**
`EXPLAIN`: is it using the HNSW index? Operator class matches the operator, `ORDER BY` uses the bare distance expression, `LIMIT` present, and the index fits in memory.

**3. Explain `ef_search`.**
Size of the candidate list during HNSW traversal at query time: larger gives higher recall and higher latency; it is the main runtime accuracy dial.

**4. Why can a filtered pgvector query return fewer than `LIMIT` rows?**
The ANN index returns a fixed number of nearest candidates first, and the `WHERE` filter is applied afterwards; a selective filter leaves too few. Raise `ef_search`, partial indexes, partitioning or iterative scans.

**5. Why is `<#>` negative?**
Postgres ordering operators are ascending-nearest; negating the inner product makes the largest dot product sort first.

## 🔗 Next

**04 - Qdrant:** a purpose-built, open-source vector database you can self-host, which unlocks payload indexes, quantisation and filtering that scales beyond what a general database handles comfortably.
