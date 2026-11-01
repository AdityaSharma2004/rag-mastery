"""
Week 2 · 04 Qdrant - solutions to practice.md (q1 ... q5)
Run all: python solutions.py     Run one: python solutions.py 3
Uses Qdrant's embedded mode (':memory:' or a local folder): no server, no Docker, no key for Qdrant itself.
Embeddings use the free Gemini API (768-d). Same code works against a real server:
    QdrantClient(url="http://localhost:6333")      # docker run -p 6333:6333 qdrant/qdrant
"""
import sys, pathlib, uuid, hashlib, tempfile, shutil
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))
from gemini_utils import *   # embed_texts, embed_query, SAMPLE_DOCS, np ...

from qdrant_client import QdrantClient, models

DIM = 768


# region helper
def point_id(text: str) -> str:
    """Qdrant ids must be unsigned ints or UUIDs. A UUID derived from the content = idempotent upserts."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, text))
# endregion helper


# region q1
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
# endregion q1


# region q2
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
# endregion q2


# region q3
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
# endregion q3


# region q4
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
# endregion q4


# region q5
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
# endregion q5


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3", "4", "5"]
    for n in which:
        print(f"\n===== Q{n} =====")
        globals()[f"q{n}"]()
