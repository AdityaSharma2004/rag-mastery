# 01 · ChromaDB — a local vector database ("SQLite for vectors")

> **One line:** Chroma is a vector database you `pip install` and use inside your Python process. No server, no Docker, no account. It is to vectors what SQLite is to rows: perfect for development, tests, prototypes and small apps.

## 🎯 What you will learn

- You will be able to explain what a vector database actually does (and what it does *not* do) and decide when plain numpy is enough.
- You will be able to describe, from first principles, how HNSW finds nearest neighbours without scanning everything — and even build a tiny version yourself.
- You will be able to create in-memory and persistent Chroma clients, create collections, and choose the distance metric (`hnsw:space`).
- You will be able to `add`, `query`, `get`, `update`, `upsert` and `delete` data while passing **your own Gemini embeddings** explicitly.
- You will be able to filter with `where` / `where_document`, and convert Chroma's *distances* back into similarity scores correctly.
- You will be able to list Chroma's limits so you know when to graduate to Pinecone, pgvector or Qdrant (the next three topics).

---

## 🧠 The big idea in plain words

In week 1 you turned sentences into lists of numbers (**embeddings**) and found the closest ones with a few lines of numpy. That worked because you had 10 sentences.

Now imagine your Laravel app has 200,000 help-centre paragraphs. A user asks a question, and you need the 5 most similar paragraphs in under 100 ms, *while* the app keeps adding, editing and deleting paragraphs, *and* the data must still be there tomorrow after a server restart, *and* you only want paragraphs belonging to `tenant_id = 7`.

A numpy array in memory gives you none of that: no persistence, no updates by id, no filtering, no safety when two requests write at once. A **vector database** is the thing that gives you all of that around the "find the nearest vectors" operation.

**The Laravel/MySQL mapping that genuinely holds:**

| Relational world | Chroma world |
|---|---|
| SQLite file for dev/tests, MySQL in production | Chroma for dev/tests, a bigger DB in production |
| `:memory:` SQLite database | `chromadb.Client()` (in-memory) |
| `database.sqlite` file on disk | `chromadb.PersistentClient(path="./chroma_db")` |
| Table | **Collection** |
| Primary key | `id` (a string you choose) |
| Columns | `metadata` (a flat dict) + the `document` text + the **embedding** |
| `WHERE category = 'x'` | `where={"category": "x"}` |
| `ORDER BY ... LIMIT 5` | `n_results=5` (ordered by distance) |
| `updateOrCreate()` | `upsert()` |
| A B-tree index (exact) | An **HNSW** index (approximate!) |

The one place the analogy breaks: a MySQL index always returns the *exact* right answer, only faster. A vector index returns an *almost always right* answer, much faster. That trade (a tiny loss of accuracy for a huge gain in speed) is the heart of this lesson.

---

## 📚 Concepts, step by step

### 1. What a vector database really is

Strip away the marketing and a vector DB is four things glued together:

1. **A store** — records of `(id, vector, document text, metadata)`, saved to disk.
2. **An ANN index** — *Approximate Nearest Neighbour* index, so "give me the 5 closest vectors" does not scan every row.
3. **A filter engine** — "…but only among records where `lang = 'en'`".
4. **CRUD + concurrency + persistence** — add, read, update, delete by id, survive restarts, handle several callers.

The vector index is the exotic part; everything else you already know from regular databases.

### 2. Why not just numpy? (an honest answer)

Brute force = compare the query against **every** stored vector. With unit-length vectors, that is one matrix-vector product: `scores = matrix @ query`.

Cost for `N` vectors of `d` dimensions: about `N × d` multiply-adds, and you must keep `N × d × 4` bytes in RAM (float32).

| N (vectors) | RAM at d=768 | One brute-force query (≈, laptop) |
|---|---|---|
| 10,000 | 31 MB | ~1-3 ms |
| 100,000 | 307 MB | ~20-30 ms |
| 1,000,000 | 3 GB | ~250 ms+ |
| 10,000,000 | 30 GB | seconds, and may not fit in RAM |

So the honest rule: **up to ~50-100k vectors, numpy brute force is perfectly fine** — and it is *exact*. Do not reach for a vector DB just because it sounds professional. You reach for one when you need (a) many more vectors, (b) persistence and CRUD by id, (c) metadata filters, (d) multiple processes/users. Most real apps need (b), (c) and (d) long before they hit (a).

### 3. HNSW intuition: how to avoid scanning everything

Suppose you must find the café nearest to you in a country with a million cafés. You don't check all of them. You (1) take a flight to the right *region*, (2) drive to the right *city*, (3) walk the *streets*. Big hops first, small hops later.

**HNSW** (*Hierarchical Navigable Small World*) builds exactly that for vectors:

```text
Layer 2 (few nodes, long-range links)     A ─────────────── H
                                          │                 │
Layer 1 (more nodes, medium links)        A ───── D ─────── H ─── K
                                          │       │         │     │
Layer 0 (ALL nodes, short links)          A-B-C-D-E-F-G-H-I-J-K-L-M
```

- Every vector is a **node**. Each node links to a handful of *nearby* nodes (its neighbours).
- Each node is also randomly "promoted" to higher layers; higher layers have fewer nodes and longer hops (like a skip list).
- **Search:** enter at the top layer, greedily walk to whichever neighbour is closer to the query, drop one layer when you can't get closer, repeat; at layer 0 explore a small "beam" of candidates and return the best.

You touch maybe a few hundred nodes instead of a million. Because the walk is greedy, it can miss the true best neighbour occasionally — hence *approximate*. Three knobs control the trade-off (names differ per database, ideas are the same):

| Knob | Meaning | Bigger value means |
|---|---|---|
| `M` (Chroma: `max_neighbors`) | links per node | better recall, more RAM, slower build |
| `ef_construction` | beam width while *building* | better graph quality, slower inserts |
| `ef_search` | beam width while *querying* | better recall, slower queries |

**Recall@k** is the standard score: of the true top-k neighbours (found by brute force), what fraction did the ANN index return? Production systems typically target 0.95-0.99.

You'll build a one-layer toy version in the code walkthrough so this stops being magic.

### 4. Chroma's building blocks

- **Client** — the entry point. `chromadb.Client()` is in-memory (gone when the process exits). `chromadb.PersistentClient(path=...)` saves to a folder. (There is also a client/server mode and a hosted Chroma Cloud — see the current docs; we stay local.)
- **Collection** — a named set of records with *one* embedding dimension and *one* distance metric. Names must be 3-512 characters from `[a-zA-Z0-9._-]` (a 2-letter name like `kb` errors out — a classic first-day surprise).
- **Record** — `id` (string, unique in the collection) + `embedding` + optional `document` (the raw text) + optional `metadata` (flat dict whose values are `str`, `int`, `float` or `bool`).

### 5. Distance metrics and the "distance vs similarity" trap

A collection's metric is chosen **at creation** with `metadata={"hnsw:space": ...}` (the long-standing way) — newer Chroma versions also accept `configuration={"hnsw": {"space": ...}}`. It cannot be changed afterwards; you would create a new collection and re-add the data.

| `hnsw:space` | What Chroma returns as `distance` | Convert to similarity |
|---|---|---|
| `"l2"` (**default!**) | **squared** Euclidean distance `‖a−b‖²` | for unit vectors: `cos = 1 − d/2` |
| `"cosine"` | `1 − cos(a, b)` | `cos = 1 − d` |
| `"ip"` | `1 − (a·b)` | `dot = 1 − d` |

Three facts to burn in:

1. **Lower distance = more similar.** Results come back best-first, i.e. *ascending* distance. Never treat the number as a score where bigger is better.
2. Floating-point noise can give a cosine distance like `-1e-7` for an identical vector; clip with `max(0, d)` before displaying.
3. **Gemini embeddings are unit-length (our helper normalises them), so all three metrics rank results identically.** Still, set `"cosine"` explicitly: the distances become readable (`0.0` identical, `~1` unrelated) and you won't be bitten when someone later adds un-normalised vectors.

By hand: two unit vectors with cosine similarity 0.8. Cosine distance = 0.2. Squared L2 = `2 − 2×0.8 = 0.4`. Inner-product distance = `1 − 0.8 = 0.2`.

### 6. `add` vs `upsert` vs `update` (the silent-ignore trap)

| Method | Id exists? | Id missing? |
|---|---|---|
| `add` | **silently ignored** (logs a warning, old data stays!) | inserted |
| `update` | modified | ignored |
| `upsert` | overwritten | inserted |
| `delete` | removed | no-op |

(`get()` returns `embeddings: None` unless you ask with `include=["embeddings"]`; `query()` returns documents, metadatas and distances by default.)

`add()` on an existing id does **not** error and does **not** overwrite. If you "re-ingest" an edited document with `add`, your index keeps serving the stale version. Default to `upsert` in ingestion code (it's `updateOrCreate`).

### 7. Metadata filters

`where` filters on metadata *during* the nearest-neighbour search (not after, so you still get `n_results` rows when enough match).

```text
{"lang": "en"}                                   equality (shorthand for $eq)
{"year": {"$gte": 2023}}                         comparison: $eq $ne $gt $gte $lt $lte
{"topic": {"$in": ["db", "cache"]}}              membership: $in $nin
{"$and": [{"lang": "en"}, {"year": {"$gte": 2023}}]}
{"$or":  [{"lang": "en"}, {"lang": "hi"}]}
```

`where_document={"$contains": "Redis"}` filters on the stored text (substring) — a crude keyword filter, no ranking. Two rules: a bare dict with two keys is **not** an implicit AND (wrap in `$and`), and filtering on a key that no record has just returns nothing.

### 8. Where the embeddings come from — do NOT use the default

If you call `collection.add(documents=[...])` with no embeddings, Chroma silently embeds the text with its **default embedding function**: a small local model (`all-MiniLM-L6-v2`, 384 dims) that is downloaded on first use. That would mean your index uses *different, weaker* vectors than the Gemini ones used everywhere else in this course, and later mixing dimensions raises errors.

Two clean fixes:

- **Explicit (preferred in this course):** compute vectors with `embed_texts(...)` / `embed_query(...)` and pass `embeddings=` and `query_embeddings=`. Nothing is hidden; the same code ports to every other database.
- **Embedding-function class:** wrap Gemini in a class implementing Chroma's `EmbeddingFunction` interface and give it to the collection, so `query_texts=[...]` works. Convenient, but it hides the document/query `task_type` distinction unless you implement `embed_query` — the code walkthrough does it correctly.

### 9. Persistence

`PersistentClient(path="./chroma_db")` creates a folder (SQLite metadata + HNSW index files). Re-opening the same path gives you your collections back. One writer process per folder — treat it like SQLite: great for one app, not for ten workers writing concurrently. (Need multi-process? Run Chroma's server mode, or graduate to a real server DB.)

### 10. Limitations (know them before you commit)

- No joins, no SQL, no transactions across collections. Metadata is flat — no nested objects.
- One distance metric and dimension per collection, fixed at creation.
- Single-node open source; horizontal scaling is not the point (managed offerings exist; check current docs).
- Filtering is basic (no full-text ranking, no aggregations).
- The default embedding function is a trap, as above.
- API details changed between 0.4 → 0.5 → 1.x (embedding-function protocol, config). **Version note:** every Chroma call here was executed against Chroma 1.5.9 with fake embeddings (Gemini calls could not be run offline); if something errors, run `pip show chromadb` and compare with the docs.

---

## 💻 Code walkthrough

Save everything below as `walkthrough.py` inside this folder. The blocks continue one another (same file). You need a Gemini key in `.env` (see the repo's `.env.example`) for stages that embed text; stages 1-2 need no key.

### Stage 0 — header (same as every lesson)

```python
import sys, pathlib, time, heapq, shutil, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import numpy as np
from gemini_utils import embed_texts, embed_query, SAMPLE_DOCS
```

### Stage 1 — brute force: the baseline everything must beat

```python
rng = np.random.default_rng(0)
N, D = 100_000, 768
big = rng.normal(size=(N, D)).astype("float32")
big /= np.linalg.norm(big, axis=1, keepdims=True)      # unit vectors, like Gemini's

def brute_force(q, matrix, k=10):
    scores = matrix @ q                                  # cosine == dot for unit vectors
    idx = np.argpartition(-scores, k)[:k]                # cheap partial sort: top k, unordered
    return idx[np.argsort(-scores[idx])]                 # now order just those k

t = time.perf_counter()
for _ in range(5):
    brute_force(big[5], big)
print(f"brute force over {N:,} vectors: {(time.perf_counter()-t)/5*1000:.0f} ms/query, "
      f"{big.nbytes/1e6:.0f} MB RAM")
```

`argpartition` avoids sorting all 100,000 scores (a full sort is `O(N log N)`; partition is `O(N)`). Expected: **≈ 20-40 ms/query, ≈ 307 MB** (varies by CPU).

### Stage 2 — a baby HNSW (one layer) to see the idea work

```python
X = rng.normal(size=(3000, 32)).astype("float32")
X /= np.linalg.norm(X, axis=1, keepdims=True)

M = 8
sims = X @ X.T
nearest = np.argsort(-sims, axis=1)[:, 1:M + 1]          # each node's M closest (exact, fine for a toy)
graph = {i: set(map(int, nearest[i])) for i in range(len(X))}
for i in range(len(X)):                                  # make links two-way so we can walk both ways
    for j in nearest[i]:
        graph[int(j)].add(i)

def graph_search(q, ef=20, k=5, entry=0):
    """Greedy beam search. Returns (top-k ids, number of distance computations)."""
    dist = lambda i: 1.0 - float(X[i] @ q)               # cosine distance for unit vectors
    visited = {entry}
    d0 = dist(entry)
    candidates = [(d0, entry)]                           # min-heap: closest unexplored first
    best = [(-d0, entry)]                                # max-heap (negated): the ef best so far
    while candidates:
        d, node = heapq.heappop(candidates)
        if len(best) >= ef and d > -best[0][0]:          # closest candidate is worse than our worst keeper
            break
        for nb in graph[node]:
            if nb in visited:
                continue
            visited.add(nb)
            dn = dist(nb)
            if len(best) < ef or dn < -best[0][0]:
                heapq.heappush(candidates, (dn, nb))
                heapq.heappush(best, (-dn, nb))
                if len(best) > ef:
                    heapq.heappop(best)                  # drop the worst keeper
    top = sorted((-nd, n) for nd, n in best)[:k]
    return [n for _, n in top], len(visited)

queries = rng.normal(size=(200, 32)).astype("float32")
queries /= np.linalg.norm(queries, axis=1, keepdims=True)
for ef in (5, 20, 80):
    hits, touched = 0, 0
    for q in queries:
        truth = set(np.argsort(-(X @ q))[:5])
        found, n_touched = graph_search(q, ef=ef)
        hits += len(truth & set(found)); touched += n_touched
    print(f"ef={ef:>3}  recall@5={hits/(5*len(queries)):.2f}  "
          f"avg vectors touched={touched/len(queries):.0f} of {len(X)}")
```

Why it works: `candidates` is the to-do list (closest first), `best` is the leaderboard of size `ef`. We stop when even the most promising unexplored node is worse than the worst node on the leaderboard. `ef` is exactly HNSW's `ef_search`. Expected (≈, measured with this exact seed): `ef=5 recall≈0.23, touched≈82` · `ef=20 recall≈0.58, touched≈215` · `ef=80 recall≈0.93, touched≈580` of 3000. Recall climbs with `ef` while you still touch only a fraction of the data. (Random 32-d vectors are a *hard* case with no clusters; real text embeddings are friendlier, and real HNSW adds the upper layers that give a much better entry point.) Real HNSW adds the upper "express" layers so the entry point is already near the answer — that is what keeps recall high at huge scale.

### Stage 3 — your first Chroma collection (in-memory, explicit Gemini embeddings)

```python
import chromadb

client = chromadb.Client()                                # in-memory: gone when the process ends
col = client.create_collection(
    name="sample_docs",                                   # >= 3 chars!
    metadata={"hnsw:space": "cosine"},                    # set the metric NOW; immutable later
)

doc_vecs = embed_texts(SAMPLE_DOCS)                       # (10, 768) float32, RETRIEVAL_DOCUMENT
col.add(
    ids=[f"doc{i}" for i in range(len(SAMPLE_DOCS))],
    embeddings=doc_vecs,                                  # numpy array is accepted
    documents=SAMPLE_DOCS,                                # stored so we can show the text back
    metadatas=[{"n_chars": len(t), "is_db": ("database" in t.lower() or "mysql" in t.lower())}
               for t in SAMPLE_DOCS],
)
print("count:", col.count())

q = "Which tool stores data in memory for caching?"
res = col.query(query_embeddings=[embed_query(q)], n_results=3)   # RETRIEVAL_QUERY vector
for doc, dist in zip(res["documents"][0], res["distances"][0]):
    print(f"  distance={dist:.3f}  similarity={1 - dist:.3f}  {doc[:60]}")
```

`query` returns **lists of lists** (one inner list per query vector) — hence `[0]`. Expected (≈): Redis is first, with cosine distance around 0.2-0.4 (similarity 0.6-0.8); the unrelated cat sentences sit near 0.6+.

### Stage 4 — get / update / upsert / delete

```python
print(col.get(ids=["doc3"], include=["documents", "metadatas"]))      # fetch by id, like find()
print(col.get(limit=2, offset=2)["ids"])                               # paging, like LIMIT/OFFSET

col.add(ids=["doc3"], embeddings=embed_texts(["THIS IS IGNORED"]), documents=["THIS IS IGNORED"])
print("after add() on existing id :", col.get(ids=["doc3"])["documents"][0][:40])   # still the OLD text

new_text = "Qdrant is an open-source vector database written in Rust and easy to self-host."
col.upsert(ids=["doc3"], embeddings=embed_texts([new_text]), documents=[new_text],
           metadatas=[{"n_chars": len(new_text), "is_db": True}])
print("after upsert()             :", col.get(ids=["doc3"])["documents"][0][:40])

col.update(ids=["doc3"], metadatas=[{"n_chars": len(new_text), "is_db": True, "reviewed": True}])
col.delete(ids=["doc6", "doc7"])                                       # the two cat sentences
print("count after delete:", col.count())
```

Note: `update` with only `metadatas` changes only metadata; the stored embedding is untouched. Chroma 1.5 *merges* the dict you pass into the existing metadata (keys you omit are kept), so you cannot remove a key with `update`; use `upsert` with the full metadata dict if you need an exact replacement. If you change the document **text**, you must also pass a new embedding (or an embedding function) — otherwise text and vector disagree.

### Stage 5 — filters

```python
qv = embed_query("databases")
only_db = col.query(query_embeddings=[qv], n_results=10, where={"is_db": True})
print("is_db only:", [d[:25] for d in only_db["documents"][0]])

combo = col.query(
    query_embeddings=[qv], n_results=10,
    where={"$and": [{"is_db": True}, {"n_chars": {"$lt": 70}}]},
)
print("db AND short:", [d[:25] for d in combo["documents"][0]])

kw = col.query(query_embeddings=[qv], n_results=10, where_document={"$contains": "Redis"})
print("contains 'Redis':", [d[:25] for d in kw["documents"][0]])
```

You asked for 10 results but fewer records match, so you get fewer — that's normal, not a bug.

### Stage 6 — persistence ("restart" the app)

```python
db_dir = pathlib.Path(tempfile.mkdtemp()) / "chroma_db"
p = chromadb.PersistentClient(path=str(db_dir))
pc = p.get_or_create_collection("notes", metadata={"hnsw:space": "cosine"})   # get_or_create = idempotent
pc.upsert(ids=["a", "b"], embeddings=embed_texts(["first note", "second note"]),
          documents=["first note", "second note"])
del p, pc                                                 # pretend the process died

p2 = chromadb.PersistentClient(path=str(db_dir))          # new process, same folder
print("after restart:", p2.get_collection("notes").count(), "records;",
      "collections:", [c.name for c in p2.list_collections()])
shutil.rmtree(db_dir.parent, ignore_errors=True)
```

`get_or_create_collection` is what you want at app boot (like `Schema::hasTable` guards in a migration). Expected: `after restart: 2 records; collections: ['notes']`.

### Stage 7 — a Gemini embedding-function class (optional convenience)

```python
from chromadb import Documents, EmbeddingFunction
from chromadb.utils.embedding_functions import register_embedding_function

@register_embedding_function
class GeminiEmbeddingFunction(EmbeddingFunction[Documents]):
    """Lets you call col.add(documents=...) / col.query(query_texts=...) with Gemini vectors."""
    def __init__(self, dim: int = 768):
        self.dim = dim
    def __call__(self, input: Documents):                 # parameter MUST be called `input`
        return embed_texts(list(input), task_type="RETRIEVAL_DOCUMENT", dim=self.dim)
    def embed_query(self, input: Documents):              # used for query_texts= in recent Chroma
        return embed_texts(list(input), task_type="RETRIEVAL_QUERY", dim=self.dim)
    @staticmethod
    def name() -> str:                                    # identifies the class when a collection is re-opened
        return "gemini_rag_course"
    def get_config(self) -> dict:                         # what gets saved with the collection
        return {"dim": self.dim}
    @staticmethod
    def build_from_config(config: dict) -> "GeminiEmbeddingFunction":
        return GeminiEmbeddingFunction(**config)
    def default_space(self):
        return "cosine"

ef_col = client.create_collection("with_ef", embedding_function=GeminiEmbeddingFunction(),
                                  metadata={"hnsw:space": "cosine"})
ef_col.add(ids=["x", "y"], documents=["Eloquent maps tables to classes", "Redis caches data in RAM"])
print(ef_col.query(query_texts=["ORM for PHP"], n_results=1)["documents"][0])
```

This follows the current protocol (`__call__(self, input)`, static `name`, `get_config`, `build_from_config`; `embed_query` splits document vs query embedding). If your Chroma version complains about the interface, fall back to explicit `embeddings=` — the stage 3-6 style — which works on every version.

### Stage 8 — wrap it so the rest of your app never sees Chroma

```python
class ChromaStore:
    def __init__(self, path=None, name="docs"):
        self.client = chromadb.PersistentClient(path=path) if path else chromadb.Client()
        self.col = self.client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})

    def upsert(self, ids, texts, metadatas=None):
        self.col.upsert(ids=ids, embeddings=embed_texts(texts), documents=texts, metadatas=metadatas)

    def search(self, question, k=3, where=None):
        r = self.col.query(query_embeddings=[embed_query(question)], n_results=k, where=where)
        return [{"id": i, "text": d, "score": 1 - dist, "meta": m}
                for i, d, dist, m in zip(r["ids"][0], r["documents"][0], r["distances"][0], r["metadatas"][0])]

store = ChromaStore()
store.upsert(["1", "2"], ["Laravel queues run jobs in the background", "Cats sleep a lot"])
print(store.search("background jobs in PHP", k=1))
```

The wrapper converts distance → similarity once, in one place. Topic 02 formalises this idea into an adapter that also fits Pinecone.

---

## ⚠️ Common mistakes & debugging

| Symptom | Cause | Fix |
|---|---|---|
| `Expected a name containing 3-512 characters` | Collection name too short or has illegal characters | Use ≥3 chars from `[a-zA-Z0-9._-]` |
| `Collection expecting embedding with dimension of 768, got 3072` | Mixed `dim` between add and query | Use one `DEFAULT_DIM` everywhere; re-create the collection if you change it |
| Search results look weak / dimension 384 appears | You passed `documents=` only, so Chroma used its default MiniLM model | Pass `embeddings=` or a Gemini `embedding_function` |
| Edited a document but search shows old text | `add()` ignores existing ids | Use `upsert()` |
| Treated `distance` as a score (higher = better) | Chroma returns distance | `similarity = 1 − distance` (cosine) and sort ascending |
| Distances up to 2 or weird values with `l2` | Default metric is *squared* L2 | Create the collection with `hnsw:space = "cosine"` |
| Changed `hnsw:space` on existing collection, nothing happened / error | Metric is fixed at creation | New collection + re-add (keep source data!) |
| `where` with two keys returns an error | Needs an explicit operator | `{"$and": [{...}, {...}]}` |
| Fewer than `n_results` rows returned | Collection (or filtered subset) is smaller | Normal; check `col.count()` |
| `include` of embeddings returns a big payload | Embeddings are excluded by default for a reason | Only request `include=["embeddings"]` when you need them |
| Two processes writing the same folder corrupt/lock | Embedded mode is single-writer | One writer, or run Chroma in server mode / use a server DB |
| Updated text, but search or dimensions go wrong | `update(documents=...)` without `embeddings=` makes Chroma re-embed with its own default 384-d model (or fail on a dimension mismatch) | Always pass your own Gemini `embeddings=` with the new text (or use `upsert`) |

---

## 🚀 Pro-level insights (what the top 2% know)

- **Ids are your idempotency key.** Derive ids deterministically (e.g. `sha1(source_path + chunk_index)` or a content hash) so re-running ingestion `upsert`s instead of duplicating. This single habit prevents most "why are there 3 copies of this chunk" bugs.
- **Store enough metadata to delete and cite:** `source`, `page`, `chunk_index`, `doc_version`, `tenant_id`. `col.delete(where={"source": "handbook_v1.pdf"})` is how you remove a replaced file cleanly.
- **Distance → relevance thresholds are model-specific.** "Cosine distance < 0.35 means relevant" is something you calibrate on *your* data with a few labelled queries; never copy a magic number from a tutorial. Returning "nothing found" is a feature.
- **Recall vs latency is tunable:** raising `ef_search` (Chroma's `hnsw:search_ef` / configuration `ef_search`) buys recall at the cost of latency; raising `M` and `ef_construction` costs RAM and index-build time. Measure recall@k against brute force on a sample — that is how ANN indexes are evaluated.
- **Dimension is a cost lever.** 768-dim vectors take a quarter of the memory of 3072 with only a small quality drop (Matryoshka truncation, re-normalised). 1M × 768 × 4 B ≈ 3 GB of raw vectors, plus the HNSW graph overhead.
- **Batch writes.** Adding 50,000 rows one call at a time is slow; add in batches of a few thousand (Chroma enforces a maximum batch size — check `client.get_max_batch_size()`).
- **Backups:** the persistent folder is the database. Stop writers and copy the folder (or snapshot the volume) — same discipline as a SQLite file.
- **When NOT to use Chroma in production:** multi-writer workloads, strict SLAs/HA, hundreds of millions of vectors, or when you need SQL joins with your business data. That is pgvector (topic 03) or a managed/clustered engine (02, 04).
- **Test strategy:** unit tests use `chromadb.Client()` (in-memory) with a deterministic fake embedder — fast, free, no API key. Only a few integration tests hit Gemini.

---

## 🧪 Mini lab

**Goal (≈ 12 min): measure how much recall HNSW gives up, on the real thing.**

1. Generate 20,000 random unit vectors of dimension 64 with numpy (fixed seed).
2. Add them to a Chroma collection with `hnsw:space = "cosine"` in batches of 2,000.
3. Take 100 random unit query vectors. For each, compute the exact top-10 with `matrix @ q` and the Chroma top-10.
4. Report mean recall@10 and mean query time for both approaches.
5. Re-create the collection with `configuration={"hnsw": {"space": "cosine", "ef_search": 10}}` and again with `ef_search: 200`. How do recall and time change?

Stretch: do the same with `metadata={"hnsw:space": "l2"}` and confirm the ranking is identical on unit vectors (distances differ, order does not).

---

## ✅ Recap checklist

- [ ] I can say what a vector DB adds over a numpy array (persistence, CRUD by id, filters, ANN index, concurrency).
- [ ] I know brute force is exact and fine up to roughly 50-100k vectors.
- [ ] I can explain HNSW as "express layers on top, short roads at the bottom, beam search at the end" and name `M`, `ef_construction`, `ef_search`.
- [ ] I know Chroma's default metric is squared L2 and set `hnsw:space = "cosine"` at creation.
- [ ] I convert distance to similarity correctly (`1 − d` for cosine) and sort ascending.
- [ ] I always pass Gemini embeddings explicitly (or a proper `EmbeddingFunction`) and never rely on the default model.
- [ ] I use `upsert` rather than `add` for re-ingestion and know why `add` can silently keep stale data.
- [ ] I can write `where` filters with `$and`, `$or`, `$in`, comparison operators and `where_document`.
- [ ] I can persist, "restart", and re-open a collection with `PersistentClient`.
- [ ] I can list at least four Chroma limitations.

---

## 🎤 Interview questions

**1. What is a vector database and why not just keep embeddings in a numpy array?**
A vector DB stores vectors with ids, text and metadata, builds an approximate-nearest-neighbour index, supports filtered search, CRUD and persistence. Numpy is exact and fine to ~100k vectors, but gives no persistence, no updates by id, no filtering and no concurrency, and a full scan gets slow at millions.

**2. Explain HNSW in two minutes.**
A layered proximity graph. Top layers are sparse with long links, bottom layer contains every vector with short links. A query enters at the top, greedily hops to closer nodes, descends layer by layer, then runs a beam search of width `ef_search` on layer 0. It visits a tiny fraction of nodes; it is approximate because greedy search can miss the true nearest neighbour. `M`, `ef_construction`, `ef_search` trade recall against RAM and speed.

**3. Chroma returns `distance`. How do you turn it into similarity and what is the default metric?**
Default is squared L2. With `hnsw:space="cosine"`, `distance = 1 − cosine_similarity`, so similarity = `1 − distance`; lower distance is better and results are sorted ascending. For unit vectors squared L2 = `2 − 2·cos`.

**4. A teammate re-ingested an updated document with `collection.add` but users still see the old answer. Why?**
`add` silently ignores ids that already exist. Use `upsert` (and derive deterministic ids so re-ingestion is idempotent).

**5. When would you move off Chroma?**
When you need several writer processes or HA, very large scale, SQL joins/transactions with business data (→ pgvector), a fully managed service (→ Pinecone), or fine-grained cost control and advanced filtering with self-hosting (→ Qdrant).

---

## 🔗 Next

**02 · Pinecone** — the same operations on a managed cloud index (the "hosted RDS" of vector search), plus a small adapter interface so your app code can swap Chroma and Pinecone without changes.
