"""
Week 2 · 01 ChromaDB — solutions to practice.md (q1 ... q5).

Run all:   python solutions.py          (q1 needs no API key; q2-q5 embed with Gemini)
Run one:   python solutions.py 3
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))
from gemini_utils import *   # embed_texts, embed_query, SAMPLE_DOCS, ...

import hashlib
import json
import os
import shutil
import tempfile

import chromadb
import numpy as np


# region helper
def _fresh_client() -> "chromadb.ClientAPI":
    """In-memory client. Each call = a brand-new empty database (like sqlite ':memory:')."""
    return chromadb.EphemeralClient() if hasattr(chromadb, "EphemeralClient") else chromadb.Client()
# endregion helper


# region q1
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
# endregion q1


# region q2
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
# endregion q2


# region q3
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
# endregion q3


# region q4
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
# endregion q4


# region q5
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
# endregion q5


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3", "4", "5"]
    for n in which:
        print(f"\n===== Q{n} =====")
        globals()[f"q{n}"]()
