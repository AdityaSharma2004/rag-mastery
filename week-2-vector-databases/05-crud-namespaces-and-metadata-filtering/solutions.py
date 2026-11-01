"""
Week 2 · 05 CRUD, namespaces & metadata filtering - solutions to practice.md (q1 ... q5)
Run all: python solutions.py     Run one: python solutions.py 3
Chroma (in-memory) and Qdrant (embedded) side by side; embeddings from the free Gemini API (768-d).
Q4 is pure numpy (no key).
"""
import sys, pathlib, uuid
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))
from gemini_utils import *   # embed_texts, embed_query, np ...

import chromadb
from qdrant_client import QdrantClient, models

DIM = 768


def uid(key: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


# region q1
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
# endregion q1


# region q2
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
# endregion q2


# region q3
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
# endregion q3


# region q4
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
# endregion q4


# region q5
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
# endregion q5


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3", "4", "5"]
    for n in which:
        print(f"\n===== Q{n} =====")
        globals()[f"q{n}"]()
