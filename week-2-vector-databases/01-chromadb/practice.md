# Practice - ChromaDB

Five questions, rising difficulty. Attempt each before opening the hint, and the hint before the solution. Every solution is the exact code in `solutions.py` (`python solutions.py` runs all, `python solutions.py 3` runs one). **Q1 needs no API key**; Q2-Q5 embed with the free Gemini API. Verified against ChromaDB 1.x; if a call errors on your version, check the "version note" in the notes.

All solutions share this header and helper:

```python
import sys, pathlib, hashlib, json, os, shutil, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))
from gemini_utils import *          # embed_texts, embed_query, SAMPLE_DOCS, DEFAULT_DIM ...
import chromadb, numpy as np
```

```python
def _fresh_client() -> "chromadb.ClientAPI":
    """In-memory client. Each call = a brand-new empty database (like sqlite ':memory:')."""
    return chromadb.EphemeralClient() if hasattr(chromadb, "EphemeralClient") else chromadb.Client()
```

---

### Q1. Warm-up - three metrics, one truth

**Problem**

Create three Chroma collections with `hnsw:space` = `l2`, `cosine` and `ip`, each holding the unit vectors `A=[1,0]`, `B=[0.8,0.6]`, `C=[0,1]`. Query with `A`. Show the distances Chroma returns and convert each back to cosine similarity. (True values: cos(A,B)=0.8, cos(A,C)=0.)

<details><summary><b>Hint</b></summary>

For unit vectors: cosine distance = `1 - cos`; squared L2 = `2 - 2cos`; inner-product "distance" in Chroma = `1 - dot`. Invert each formula.

</details>

<details><summary><b>Solution</b></summary>

```python
# ---------------------------------------------------------------------------
# Q1 (warm-up): one pair of vectors, three metrics, convert distance -> cosine
# ---------------------------------------------------------------------------
def q1():
    # Three unit vectors in 2-D. cos(A,B)=0.8 and cos(A,C)=0 because C is at 90 degrees.
    vecs = {"A": [1.0, 0.0], "B": [0.8, 0.6], "C": [0.0, 1.0]}
    client = _fresh_client()
    results = {}
    for space in ("l2", "cosine", "ip"):
        col = client.create_collection(f"metric_{space}", metadata={"hnsw:space": space})
        col.add(ids=list(vecs), embeddings=list(vecs.values()))
        r = col.query(query_embeddings=[vecs["A"]], n_results=3)    # query = A itself
        results[space] = dict(zip(r["ids"][0], r["distances"][0]))

    def to_cos(space, d):
        return {"l2": 1 - d / 2, "cosine": 1 - d, "ip": 1 - d}[space]   # unit vectors only

    for space, dists in results.items():
        print(f"{space:>6}: distances " + ", ".join(f"{k}={max(v, 0):.2f}" for k, v in dists.items())
              + "   -> cosine " + ", ".join(f"{k}={to_cos(space, v):.2f}" for k, v in dists.items()))
    for space in results:                                            # all three must agree
        assert abs(to_cos(space, results[space]["B"]) - 0.8) < 1e-3
        assert abs(to_cos(space, results[space]["C"]) - 0.0) < 1e-3
    print("All three metrics recover cos(A,B)=0.8 and cos(A,C)=0.0")
```

</details>

**Why it works / what to learn**
Chroma always returns a **distance** (smaller = closer), and its default metric is squared L2, not cosine. On unit vectors all three metrics carry the same information, but the numbers differ, so a threshold must be converted. Choose the metric when you create the collection; it cannot be changed later.

**Expected output (≈)**
```text
    l2: distances A=0.00, B=0.40, C=2.00   -> cosine A=1.00, B=0.80, C=0.00
cosine: distances A=0.00, B=0.20, C=1.00   -> cosine A=1.00, B=0.80, C=0.00
    ip: distances A=0.00, B=0.20, C=1.00   -> cosine A=1.00, B=0.80, C=0.00
All three metrics recover cos(A,B)=0.8 and cos(A,C)=0.0
```

---

### Q2. Apply - multi-tenant knowledge base with metadata filters

**Problem**

Store five documents for two customers (`acme`, `globex`) with `tenant` and `year` metadata. (1) Ask the same question restricted to each tenant. (2) Ask for password-reset info restricted to `acme` **and** `year >= 2024`. (3) Offboard `globex` by deleting all its records in one call.

<details><summary><b>Hint</b></summary>

`col.query(..., where={"tenant": "acme"})`. For two conditions wrap them in `{"$and": [ ... ]}`. Delete with `col.delete(where=...)`.

</details>

<details><summary><b>Solution</b></summary>

```python
# ---------------------------------------------------------------------------
# Q2 (apply): multi-tenant KB with metadata filters + delete by metadata
# ---------------------------------------------------------------------------
TENANT_DOCS = [
    # id, text, tenant, year
    ("a1", "Acme refunds are processed within 7 days after an order is cancelled.", "acme", 2024),
    ("a2", "Acme parcels are delivered across India in 3 to 5 working days.", "acme", 2022),
    ("a3", "Acme password reset links expire after 15 minutes.", "acme", 2024),
    ("g1", "Globex refunds take 14 days and require the original receipt.", "globex", 2024),
    ("g2", "Globex password reset emails expire after 30 minutes.", "globex", 2023),
]


def q2():
    client = _fresh_client()
    col = client.create_collection("tenant_kb", metadata={"hnsw:space": "cosine"})
    col.upsert(
        ids=[d[0] for d in TENANT_DOCS],
        embeddings=embed_texts([d[1] for d in TENANT_DOCS]),
        documents=[d[1] for d in TENANT_DOCS],
        metadatas=[{"tenant": d[2], "year": d[3]} for d in TENANT_DOCS],
    )

    def ask(question, where=None, k=3):
        r = col.query(query_embeddings=[embed_query(question)], n_results=k, where=where)
        return [(i, round(max(0.0, 1 - dist), 2)) for i, dist in zip(r["ids"][0], r["distances"][0])]

    q = "how long does a refund take"
    print("acme   :", ask(q, {"tenant": "acme"}))
    print("globex :", ask(q, {"tenant": "globex"}))
    # two conditions MUST be wrapped in $and (a dict with two keys raises an error)
    print("acme>=2024 password:", ask("password reset link expiry",
                                      {"$and": [{"tenant": "acme"}, {"year": {"$gte": 2024}}]}, k=5))
    col.delete(where={"tenant": "globex"})                      # offboard a customer in one call
    print("after deleting globex, count =", col.count(), "->", sorted(col.get()["ids"]))
    assert col.count() == 3 and all(i.startswith("a") for i in col.get()["ids"])
```

</details>

**Why it works / what to learn**
The filter runs inside the database together with the vector search, like an Eloquent `->where('tenant', ...)` on a similarity query, so one customer can never see another's chunks. `upsert` (not `add`) makes loading idempotent. `year >= 2024` uses the `$gte` operator.

**Expected output (≈)**
```text
acme   : [('a1', ≈0.7), ('a2', ≈0.4), ('a3', ≈0.3)]   (a1 = Acme refunds first)
globex : [('g1', ≈0.7), ('g2', ≈0.3)]
acme>=2024 password: [('a3', ≈0.7), ('a1', ≈0.3)]
after deleting globex, count = 3 -> ['a1', 'a2', 'a3']
```

---

### Q3. Build - an idempotent knowledge store

**Problem**

Write a `KnowledgeStore` wrapping a persistent Chroma collection with: (a) deterministic ids so re-ingesting the same file creates **no duplicates**; (b) `replace_source()` that removes all old chunks of a file before adding the edited ones; (c) `search()` that applies a minimum similarity and returns an empty list for off-topic questions; (d) survival across a restart.

<details><summary><b>Hint</b></summary>

`id = sha1(source|index|text)[:16]`; store `source` and `chunk_index` in metadata; `delete(where={"source": ...})` before re-adding. Similarity = `1 - distance` on a cosine collection.

</details>

<details><summary><b>Solution</b></summary>

```python
# ---------------------------------------------------------------------------
# Q3 (build): idempotent knowledge store (content-hash ids, threshold, replace-by-source)
# ---------------------------------------------------------------------------
class KnowledgeStore:
    """Small Chroma wrapper: deterministic ids, cosine similarity scores, 'no answer' threshold."""

    def __init__(self, path=None, name="knowledge", dim=DEFAULT_DIM):
        self.dim = dim
        self.client = chromadb.PersistentClient(path=str(path)) if path else _fresh_client()
        self.col = self.client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})

    @staticmethod
    def make_id(source: str, index: int, text: str) -> str:
        # same source + position + text  ->  same id  ->  re-ingest is an UPSERT, never a duplicate
        return hashlib.sha1(f"{source}|{index}|{text}".encode()).hexdigest()[:16]

    def ingest(self, source: str, texts: list[str]) -> int:
        """Upsert chunks of one source. Returns how many NEW records were created."""
        before = self.col.count()
        ids = [self.make_id(source, i, t) for i, t in enumerate(texts)]
        self.col.upsert(
            ids=ids,
            embeddings=embed_texts(texts, dim=self.dim),
            documents=texts,
            metadatas=[{"source": source, "chunk_index": i} for i in range(len(texts))],
        )
        return self.col.count() - before

    def replace_source(self, source: str, texts: list[str]) -> int:
        """A file changed: remove every old chunk of it first, so no stale chunk survives."""
        self.col.delete(where={"source": source})
        return self.ingest(source, texts)

    def search(self, question: str, k: int = 3, min_similarity: float = 0.5):
        r = self.col.query(query_embeddings=[embed_query(question, dim=self.dim)], n_results=k)
        hits = []
        for i, doc, dist, meta in zip(r["ids"][0], r["documents"][0], r["distances"][0], r["metadatas"][0]):
            sim = max(0.0, 1 - dist)
            if sim >= min_similarity:                    # below the line = "I don't know"
                hits.append({"id": i, "text": doc, "similarity": round(sim, 3), "source": meta["source"]})
        return hits


def q3():
    tmp = pathlib.Path(tempfile.mkdtemp())
    try:
        v1 = ["Laravel queues run jobs in the background using Redis or the database.",
              "Horizon is a dashboard for monitoring Laravel queues."]
        store = KnowledgeStore(path=tmp / "db")
        print("first ingest  new records:", store.ingest("queues.md", v1))
        print("second ingest new records:", store.ingest("queues.md", v1), "(idempotent)")
        assert store.col.count() == 2

        v2 = ["Laravel queues run jobs in the background using Redis, SQS or the database."]   # file shrank + edited
        print("replace_source new records:", store.replace_source("queues.md", v2), "-> total", store.col.count())
        assert store.col.count() == 1

        del store                                         # "restart": reopen the same folder
        store = KnowledgeStore(path=tmp / "db")
        print("after restart count:", store.col.count())
        hits = store.search("background jobs queues in Laravel", k=3, min_similarity=0.5)
        print("relevant question ->", [(h["similarity"], h["text"][:40]) for h in hits])
        print("off-topic question ->", store.search("capital city of France", k=3, min_similarity=0.75))
        assert hits and store.col.count() == 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
```

</details>

**Why it works / what to learn**
Same input → same id → `upsert` overwrites instead of duplicating (idempotent ingestion). Editing a file can *shrink* it, so you must delete the old chunks first or stale paragraphs keep being retrieved. A similarity floor gives the RAG layer a clean "I don't know" path.

**Expected output (≈)**
```text
first ingest  new records: 2
second ingest new records: 0 (idempotent)
replace_source new records: 1 -> total 1
after restart count: 1
relevant question -> [(≈0.7, 'Laravel queues run jobs in the backgroun')]
off-topic question -> []
```

---

### Q4. Debug - four bugs in one snippet

**Problem**

A teammate's Chroma code has four problems: (1) `create_collection("kb")` raises an error; (2) "perfect matches" are rejected by `if distance > 0.8`; (3) updating a document by calling `add()` again changes nothing; (4) a filter `where={"tenant": "acme", "year": 2024}` raises an error. Reproduce each and fix it.

<details><summary><b>Hint</b></summary>

(1) Collection names must be 3-512 characters. (2) A distance of 0 means identical; convert with `1 - distance` and mind the metric. (3) `add` ignores existing ids; there is another method for "insert or replace". (4) Multiple conditions need an explicit operator.

</details>

<details><summary><b>Solution</b></summary>

```python
# ---------------------------------------------------------------------------
# Q4 (debug): four bugs in one snippet - reproduce each, then fix it
# (short name, default l2 metric, add() ignoring existing ids, two-key where filter)
# ---------------------------------------------------------------------------
def q4():
    client = _fresh_client()

    # BUG 1: collection name shorter than 3 characters
    try:
        client.create_collection("kb")
    except Exception as e:
        print("bug1 ->", type(e).__name__, "(name must be 3-512 chars)")

    # (default metric = squared L2 -> BUG 2 below)
    bad = client.create_collection("kb_docs")
    good = client.create_collection("kb_docs_fixed", metadata={"hnsw:space": "cosine"})
    old_text = "Refunds are accepted within 7 days of delivery."
    new_text = "Parcels ship from our Delhi warehouse every morning."
    for c in (bad, good):
        c.add(ids=["p1"], embeddings=embed_texts([old_text]), documents=[old_text])

    # BUG 3: add() on an existing id is silently ignored
    bad.add(ids=["p1"], embeddings=embed_texts([new_text]), documents=[new_text])
    print("bug3 add()    ->", bad.get(ids=["p1"])["documents"][0][:30], "(still OLD)")
    good.upsert(ids=["p1"], embeddings=embed_texts([new_text]), documents=[new_text])
    print("fix  upsert() ->", good.get(ids=["p1"])["documents"][0][:30], "(NEW)")

    # BUG 2: reading `distance` as a score ("bigger is better") and using the default l2 metric
    r = bad.query(query_embeddings=[embed_texts([old_text])[0]], n_results=1)
    d = r["distances"][0][0]
    print(f"bug2 identical text has distance {max(d, 0):.2f}; `if distance > 0.8: relevant` rejects the PERFECT match")
    r = good.query(query_embeddings=[embed_texts([new_text])[0]], n_results=1)
    print(f"fix  similarity = 1 - distance = {1 - max(r['distances'][0][0], 0):.2f}")

    # BUG 4: a metadata filter with two conditions in one dict
    c4 = client.create_collection("kb_docs_filter", metadata={"hnsw:space": "cosine"})
    c4.add(ids=["x1", "x2"], embeddings=embed_texts([old_text, new_text]),
           metadatas=[{"tenant": "acme", "year": 2024}, {"tenant": "acme", "year": 2022}])
    q = embed_texts([old_text])
    try:
        c4.query(query_embeddings=q, n_results=2, where={"tenant": "acme", "year": 2024})
    except Exception as e:
        print("bug4 ->", type(e).__name__, "(two keys in one where dict are rejected)")
    ok = c4.query(query_embeddings=q, n_results=2,
                  where={"$and": [{"tenant": "acme"}, {"year": {"$gte": 2024}}]})
    print("fix  $and ->", ok["ids"][0], "(only the 2024 record)")
    assert ok["ids"][0] == ["x1"]
```

</details>

**Why it works / what to learn**
These are the four most common first-week Chroma mistakes. `add` is "insert if new" (silently skipping existing ids), `upsert` is "insert or replace". Distances are *smaller = better*. Filters take exactly one operator per level, so combine conditions with `$and` / `$or`.

**Expected output (≈)**
```text
bug1 -> InvalidArgumentError (name must be 3-512 chars)
bug3 add()    -> Refunds are accepted within 7  (still OLD)
fix  upsert() -> Parcels ship from our Delhi wa (NEW)
bug2 identical text has distance 0.00; `if distance > 0.8: relevant` rejects the PERFECT match
fix  similarity = 1 - distance = 1.00
bug4 -> ValueError (two keys in one where dict are rejected)
fix  $and -> ['x1'] (only the 2024 record)
```

---

### Q5. Production - zero-downtime re-embedding migration (blue/green)

**Problem**

Your live collection holds 768-d vectors. You decide to move to 1536-d. Write `migrate_collection(...)` that copies documents and metadata page by page into a new collection while **re-embedding**, a pointer file that the app reads to know which collection is live, and a switch procedure: verify top-1 agreement on spot-check questions, atomically flip the pointer, then delete the old collection.

<details><summary><b>Hint</b></summary>

Never mutate the live collection. `old.get(limit=..., offset=..., include=["documents","metadatas"])` pages through rows. Write the pointer to a temp file and `os.replace` it for an atomic swap.

</details>

<details><summary><b>Solution</b></summary>

```python
# ---------------------------------------------------------------------------
# Q5 (production): blue/green re-embedding migration 768 -> 1536 dims, with a pointer file
# ---------------------------------------------------------------------------
def _read_active(pointer: pathlib.Path) -> dict:
    return json.loads(pointer.read_text())


def _write_active(pointer: pathlib.Path, name: str, dim: int) -> None:
    tmp = pointer.with_suffix(".tmp")
    tmp.write_text(json.dumps({"collection": name, "dim": dim}))
    os.replace(tmp, pointer)                              # atomic swap: readers never see a half-written file


def migrate_collection(client, old_name: str, new_name: str, new_dim: int, batch: int = 100) -> int:
    """Copy documents + metadata from old to new collection, re-embedding at new_dim. Returns rows copied."""
    old = client.get_collection(old_name)
    new = client.create_collection(new_name, metadata={"hnsw:space": "cosine"})
    copied, offset = 0, 0
    while offset < old.count():
        page = old.get(limit=batch, offset=offset, include=["documents", "metadatas"])
        if not page["ids"]:
            break
        new.upsert(ids=page["ids"], documents=page["documents"], metadatas=page["metadatas"],
                   embeddings=embed_texts(page["documents"], dim=new_dim))
        copied += len(page["ids"])
        offset += batch
    return copied


def q5():
    tmp = pathlib.Path(tempfile.mkdtemp())
    pointer = tmp / "active_collection.json"
    try:
        client = chromadb.PersistentClient(path=str(tmp / "db"))
        texts = SAMPLE_DOCS
        ids = [f"d{i}" for i in range(len(texts))]

        # --- blue: the live collection at 768 dims
        blue = client.create_collection("docs_v1_d768", metadata={"hnsw:space": "cosine"})
        blue.upsert(ids=ids, documents=texts, embeddings=embed_texts(texts, dim=768),
                    metadatas=[{"source": "sample", "n": i} for i in range(len(texts))])
        _write_active(pointer, "docs_v1_d768", 768)

        def search(question, k=1):
            cfg = _read_active(pointer)                       # app always asks the pointer, never hard-codes a name
            col = client.get_collection(cfg["collection"])
            r = col.query(query_embeddings=[embed_query(question, dim=cfg["dim"])], n_results=k)
            return cfg["collection"], r["ids"][0]

        questions = ["Which tool is used for caching in memory?", "What is an ORM in Laravel?",
                     "managed relational database on AWS"]
        before = [search(q)[1] for q in questions]
        print("live (blue):", search(questions[0]))

        # --- green: build the new collection while blue keeps serving
        copied = migrate_collection(client, "docs_v1_d768", "docs_v2_d1536", 1536, batch=4)
        green = client.get_collection("docs_v2_d1536")
        assert copied == blue.count() == green.count()

        # --- verify BEFORE switching: same top-1 on the spot-check questions
        green_top = [green.query(query_embeddings=[embed_query(q, dim=1536)], n_results=1)["ids"][0] for q in questions]
        agree = sum(a == b for a, b in zip(before, green_top))
        print(f"copied {copied} rows; top-1 agreement old vs new on {len(questions)} spot checks: {agree}/{len(questions)}")

        # --- switch (atomic) and keep blue for a rollback window, then delete it
        _write_active(pointer, "docs_v2_d1536", 1536)
        print("live (green):", search(questions[0]))
        client.delete_collection("docs_v1_d768")
        print("collections now:", [c.name for c in client.list_collections()])
        assert _read_active(pointer)["collection"] == "docs_v2_d1536"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
```

</details>

**Why it works / what to learn**
Embeddings are a *derived* column: changing model or dimension means rebuilding the index. Blue/green keeps the old index serving until the new one is verified, and the atomic pointer swap means readers never see a half-written state; keep the old collection for a rollback window before deleting it (like a database migration with a rollback plan).

**Expected output (≈)**
```text
live (blue): ('docs_v1_d768', ['d2'])
copied 10 rows; top-1 agreement old vs new on 3 spot checks: ≈3/3
live (green): ('docs_v2_d1536', ['d2'])
collections now: ['docs_v2_d1536']
```
(`d2` is the Redis sentence for the caching question; ids vary with the live model.)
