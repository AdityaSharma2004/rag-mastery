"""
Week 2 · 02 Pinecone - solutions to practice.md (q1 ... q5)

Run all:   python solutions.py          Run one:  python solutions.py 3

* With PINECONE_API_KEY set (free Starter plan, https://app.pinecone.io) every question talks to a real
  serverless index called "rag-course-w2".
* WITHOUT a key every question still runs against `FakeIndex`, a tiny in-memory stand-in that has the same
  method names and keyword arguments as the real `Index` (upsert / query / fetch / delete /
  describe_index_stats). It lets you learn the API shape and test your logic offline, but it is NOT Pinecone:
  it has no network, no eventual consistency and no billing.
* All embeddings use the free Gemini API (768 dims).
"""
import sys, pathlib, os, time, json, hashlib, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))
from gemini_utils import *   # embed_texts, embed_query, SAMPLE_DOCS, np ...

from types import SimpleNamespace

INDEX_NAME = "rag-course-w2"
DIM = 768


# region helper
# ---------------------------------------------------------------------------
# Offline stand-in with the same call shapes as pinecone.Index (data-plane methods are keyword-only)
# ---------------------------------------------------------------------------
def _match(meta: dict, flt: dict | None) -> bool:
    """Pinecone filter syntax: {"year": {"$gte": 2024}, "tenant": "acme"} (several keys = AND)."""
    if not flt:
        return True
    for key, cond in flt.items():
        if key == "$and":
            if not all(_match(meta, c) for c in cond):
                return False
        elif key == "$or":
            if not any(_match(meta, c) for c in cond):
                return False
        else:
            val = meta.get(key)
            ops = cond if isinstance(cond, dict) else {"$eq": cond}
            for op, arg in ops.items():
                ok = {
                    "$eq": lambda: val == arg, "$ne": lambda: val != arg,
                    "$gt": lambda: val is not None and val > arg, "$gte": lambda: val is not None and val >= arg,
                    "$lt": lambda: val is not None and val < arg, "$lte": lambda: val is not None and val <= arg,
                    "$in": lambda: val in arg, "$nin": lambda: val not in arg,
                }[op]()
                if not ok:
                    return False
    return True


class FakeIndex:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.ns: dict[str, dict[str, dict]] = {}
        self.fail_next_upserts = 0                         # for the retry exercise (Q5)

    def upsert(self, *, vectors, namespace: str = "", **_):
        if self.fail_next_upserts > 0:
            self.fail_next_upserts -= 1
            raise RuntimeError("simulated 503 from Pinecone")
        store = self.ns.setdefault(namespace, {})
        for v in vectors:
            vid, values, meta = v["id"], list(v["values"]), v.get("metadata", {})
            if len(values) != self.dimension:
                raise ValueError(f"Vector dimension {len(values)} does not match the dimension of the index {self.dimension}")
            store[vid] = {"values": np.array(values, dtype=np.float32), "metadata": dict(meta)}
        return SimpleNamespace(upserted_count=len(vectors))

    def query(self, *, top_k: int, vector=None, namespace: str = "", filter=None,
              include_metadata: bool = False, include_values: bool = False, **_):
        q = np.array(vector, dtype=np.float32)
        rows = [(vid, r) for vid, r in self.ns.get(namespace, {}).items() if _match(r["metadata"], filter)]
        scored = sorted(((float(r["values"] @ q), vid, r) for vid, r in rows), key=lambda x: -x[0])[:top_k]
        matches = [SimpleNamespace(id=vid, score=s,
                                   metadata=r["metadata"] if include_metadata else None,
                                   values=list(r["values"]) if include_values else None)
                   for s, vid, r in scored]
        return SimpleNamespace(matches=matches, namespace=namespace)

    def fetch(self, *, ids, namespace: str = "", **_):
        store = self.ns.get(namespace, {})
        return SimpleNamespace(vectors={i: SimpleNamespace(id=i, metadata=store[i]["metadata"])
                                        for i in ids if i in store})

    def delete(self, *, ids=None, delete_all: bool = False, namespace: str = "", **_):
        if delete_all:
            self.ns.pop(namespace, None)
        else:
            for i in ids or []:
                self.ns.get(namespace, {}).pop(i, None)

    def describe_index_stats(self, **_):
        ns = {n: SimpleNamespace(vector_count=len(s)) for n, s in self.ns.items() if s}
        return SimpleNamespace(namespaces=ns, dimension=self.dimension,
                               total_vector_count=sum(len(s) for s in self.ns.values()))


def _is_ready(pc, name: str) -> bool:
    status = pc.describe_index(name).status
    return bool(status["ready"] if isinstance(status, dict) else getattr(status, "ready", False))


def get_index(name: str = INDEX_NAME, dim: int = DIM):
    """Return (index, is_real). Creates the serverless index on first use (takes ~30-60 s)."""
    key = os.getenv("PINECONE_API_KEY")
    if not key:
        print(f"[no PINECONE_API_KEY] using the in-memory FakeIndex (dimension {dim}) - same API shape, no network")
        return FakeIndex(dim), False
    from pinecone import Pinecone, ServerlessSpec
    pc = Pinecone(api_key=key)
    if not pc.has_index(name):
        print(f"creating serverless index {name!r} ...")
        # Starter plan: AWS us-east-1. cosine = the metric our unit vectors are meant for.
        pc.create_index(name=name, dimension=dim, metric="cosine",
                        spec=ServerlessSpec(cloud="aws", region="us-east-1"))
        while not _is_ready(pc, name):
            time.sleep(3)
    return pc.Index(name), True


def wait_for_count(index, expected: int, namespace: str = "", timeout: float = 60.0) -> int:
    """Serverless indexes are eventually consistent: a fresh upsert may not be queryable for a moment."""
    t0, count = time.time(), -1
    while time.time() - t0 < timeout:
        stats = index.describe_index_stats()
        nsd = stats.namespaces
        count = (nsd[namespace].vector_count if namespace in nsd else 0) if nsd else 0
        if count >= expected:
            return count
        time.sleep(1.5)
    return count
# endregion helper


# region q1
# --------------------------------------------------------------------------
# Q1. Warm-up: create an index, upsert 10 vectors with metadata, query, read the matches
# --------------------------------------------------------------------------
def q1():
    index, real = get_index()
    vectors = [
        {"id": f"doc-{i}", "values": v.tolist(), "metadata": {"text": t, "n": i}}
        for i, (t, v) in enumerate(zip(SAMPLE_DOCS, embed_texts(SAMPLE_DOCS, dim=DIM)))
    ]
    index.upsert(vectors=vectors, namespace="warmup")          # keyword-only arguments
    print("visible vectors:", wait_for_count(index, len(vectors), "warmup"))

    q = embed_query("a young cat sleeping", dim=DIM).tolist()
    res = index.query(vector=q, top_k=3, namespace="warmup", include_metadata=True)
    for m in res.matches:                                      # best first, higher cosine score = closer
        print(f"{m.score:.3f}  {m.id}  {m.metadata['text'][:55]}")
    return [m.id for m in res.matches]
# endregion q1


# region q2
# --------------------------------------------------------------------------
# Q2. Apply: namespaces as tenants + metadata filters
# --------------------------------------------------------------------------
def q2():
    index, real = get_index()
    docs = {
        "acme": [("a1", "Acme refunds take 7 days.", 2024), ("a2", "Acme ships within 3 working days.", 2022)],
        "globex": [("a1", "Globex refunds take 14 days.", 2023), ("g2", "Globex support is open 24x7.", 2024)],
    }
    for tenant, rows in docs.items():                          # NOTE: id "a1" exists in BOTH namespaces on purpose
        index.upsert(namespace=tenant, vectors=[
            {"id": i, "values": v.tolist(), "metadata": {"text": t, "year": y}}
            for (i, t, y), v in zip(rows, embed_texts([r[1] for r in rows], dim=DIM))
        ])
    for tenant in docs:
        wait_for_count(index, len(docs[tenant]), tenant)

    q = embed_query("how long do refunds take", dim=DIM).tolist()
    for tenant in docs:                                        # a query only ever searches ONE namespace
        r = index.query(vector=q, top_k=2, namespace=tenant, include_metadata=True)
        print(f"{tenant:>6}:", [(m.id, m.metadata["text"][:28]) for m in r.matches])

    newer = index.query(vector=q, top_k=5, namespace="globex", include_metadata=True,
                        filter={"year": {"$gte": 2024}})        # metadata filter inside the namespace
    print("globex year>=2024:", [m.id for m in newer.matches])

    stats = index.describe_index_stats()
    print("per-namespace counts:", {n: s.vector_count for n, s in stats.namespaces.items() if n in docs})

    index.delete(delete_all=True, namespace="globex")           # offboard one tenant, others untouched
    time.sleep(2 if real else 0)
    print("after deleting globex:", sorted(n for n in index.describe_index_stats().namespaces if n in docs))
    return True
# endregion q2


# region q3
# --------------------------------------------------------------------------
# Q3. Build: a small store with batching, deterministic ids, score threshold, metadata text
# --------------------------------------------------------------------------
class PineconeStore:
    """Pinecone keeps vectors + metadata (no separate 'documents' field), so the chunk text lives in metadata."""
    MAX_TEXT_BYTES = 8_000           # metadata is capped (tens of KB per record); keep chunk text small

    def __init__(self, index, namespace: str = "", dim: int = DIM, batch_size: int = 100):
        self.index, self.ns, self.dim, self.batch = index, namespace, dim, batch_size

    @staticmethod
    def make_id(source: str, i: int, text: str) -> str:
        return hashlib.sha1(f"{source}|{i}|{text}".encode()).hexdigest()[:20]

    def ingest(self, source: str, texts: list[str]) -> int:
        vecs = embed_texts(texts, dim=self.dim)
        records = [{
            "id": self.make_id(source, i, t),
            "values": v.tolist(),
            "metadata": {"source": source, "chunk": i, "text": t.encode()[: self.MAX_TEXT_BYTES].decode(errors="ignore")},
        } for i, (t, v) in enumerate(zip(texts, vecs))]
        for s in range(0, len(records), self.batch):           # upsert in batches, never one giant request
            self.index.upsert(vectors=records[s : s + self.batch], namespace=self.ns)
        return len(records)

    def search(self, question: str, k: int = 3, flt: dict | None = None, min_score: float = 0.5):
        res = self.index.query(vector=embed_query(question, dim=self.dim).tolist(), top_k=k,
                               namespace=self.ns, filter=flt, include_metadata=True)
        return [{"id": m.id, "score": round(m.score, 3), "text": m.metadata["text"], "source": m.metadata["source"]}
                for m in res.matches if m.score >= min_score]


def q3():
    index, real = get_index()
    store = PineconeStore(index, namespace="kb", batch_size=4)
    n1 = store.ingest("laravel.md", ["Eloquent maps tables to PHP classes.",
                                     "Queues run slow jobs in the background.",
                                     "Migrations version your database schema.",
                                     "Blade is the Laravel templating engine.",
                                     "Horizon monitors Redis queues."])
    n2 = store.ingest("laravel.md", ["Eloquent maps tables to PHP classes."])   # same id again -> overwrite
    wait_for_count(index, 5, "kb")
    print("ingested", n1, "then re-ingested", n2, "-> vectors in namespace:", wait_for_count(index, 5, "kb"))
    print("relevant :", [(h["score"], h["text"][:35]) for h in store.search("background jobs", min_score=0.4)])
    print("off-topic:", store.search("capital of France", min_score=0.75))
    return store
# endregion q3


# region q4
# --------------------------------------------------------------------------
# Q4. Debug: four classic Pinecone mistakes
# --------------------------------------------------------------------------
def q4():
    index, real = get_index()
    v768 = embed_texts(["hello"], dim=768)[0].tolist()
    v1536 = embed_texts(["hello"], dim=1536)[0].tolist()
    index.upsert(vectors=[{"id": "h", "values": v768, "metadata": {"text": "hello"}}], namespace="bugs")
    wait_for_count(index, 1, "bugs")

    # BUG 1: vector size differs from the index dimension
    try:
        index.upsert(vectors=[{"id": "bad", "values": v1536}], namespace="bugs")
    except Exception as e:
        print("bug1 ->", type(e).__name__, "(index is 768-d, vector was 1536-d)")

    # BUG 2: forgot include_metadata -> matches carry no text
    r = index.query(vector=v768, top_k=1, namespace="bugs")
    print("bug2 metadata without include_metadata:", r.matches[0].metadata)
    r = index.query(vector=v768, top_k=1, namespace="bugs", include_metadata=True)
    print("fix  with include_metadata=True       :", r.matches[0].metadata)

    # BUG 3: wrong namespace -> empty result, no error
    r = index.query(vector=v768, top_k=1, include_metadata=True)            # default namespace is ""
    print("bug3 query in the default namespace  :", len(r.matches), "matches (data is in 'bugs')")

    # BUG 4: reading before the write is visible (eventual consistency on serverless)
    visible = wait_for_count(index, 1, "bugs")
    print("fix  poll describe_index_stats until count is visible:", visible)
    index.delete(delete_all=True, namespace="bugs")
    return True
# endregion q4


# region q5
# --------------------------------------------------------------------------
# Q5. Production: resumable, retrying bulk ingestion with a checkpoint file
# --------------------------------------------------------------------------
def upsert_with_retry(index, batch, namespace, max_tries=5):
    delay = 0.2
    for attempt in range(1, max_tries + 1):
        try:
            return index.upsert(vectors=batch, namespace=namespace)
        except Exception as e:
            if attempt == max_tries:
                raise
            print(f"  upsert failed ({e}); retry {attempt}/{max_tries - 1} in {delay:.1f}s")
            time.sleep(delay)
            delay *= 2


def bulk_ingest(index, records: list[dict], namespace: str, checkpoint: pathlib.Path, batch_size: int = 50) -> int:
    """Upsert in batches; write progress after each batch so a crash resumes instead of restarting."""
    done = json.loads(checkpoint.read_text())["next"] if checkpoint.exists() else 0
    for start in range(done, len(records), batch_size):
        upsert_with_retry(index, records[start : start + batch_size], namespace)
        checkpoint.write_text(json.dumps({"next": start + batch_size}))
    return len(records)


def q5():
    index, real = get_index()
    texts = [f"{t} (variant {i})" for i in range(5) for t in SAMPLE_DOCS]          # 50 chunks
    vecs = embed_texts(texts, dim=DIM)
    records = [{"id": f"bulk-{i}", "values": v.tolist(), "metadata": {"text": t}}
               for i, (t, v) in enumerate(zip(texts, vecs))]
    ckpt = pathlib.Path(tempfile.mkdtemp()) / "progress.json"

    if isinstance(index, FakeIndex):
        index.fail_next_upserts = 2                      # first two requests fail with a simulated 503
    bulk_ingest(index, records, "bulk", ckpt, batch_size=20)
    print("vectors visible:", wait_for_count(index, len(records), "bulk"), "checkpoint:", ckpt.read_text())

    bulk_ingest(index, records, "bulk", ckpt, batch_size=20)                          # re-run: nothing left to do
    print("second run was a no-op; still", wait_for_count(index, len(records), "bulk"), "vectors")
    index.delete(delete_all=True, namespace="bulk")
    return True
# endregion q5


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3", "4", "5"]
    for n in which:
        print(f"\n===== Q{n} =====")
        globals()[f"q{n}"]()
