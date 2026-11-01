"""
Week 2 · 03 pgvector - solutions to practice.md (q1 ... q5)

Run all:   python solutions.py          Run one:  python solutions.py 3

Needs a Postgres with the pgvector extension (see notes.md, "Stage 0"):
    docker run -d --name pgvector -e POSTGRES_PASSWORD=postgres -p 5432:5432 pgvector/pgvector:pg16
    DATABASE_URL=postgresql://postgres:postgres@localhost:5432/ragdb      (put it in .env)
Q1-Q3 embed text with the free Gemini API. Q4 and Q5 use synthetic vectors (no API key).
"""
import sys, pathlib, os, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))
from gemini_utils import *   # embed_texts, embed_query, SAMPLE_DOCS, DEFAULT_DIM, np ...

import psycopg
from pgvector.psycopg import register_vector

DIM = 768
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/ragdb")


# region helper
def connect() -> psycopg.Connection:
    """Open a connection, make sure the extension exists, teach psycopg about the `vector` type."""
    try:
        conn = psycopg.connect(DATABASE_URL, autocommit=True)
    except psycopg.OperationalError as e:
        raise SystemExit(f"Cannot reach Postgres at {DATABASE_URL}\n"
                         f"Start it with:  docker run -d --name pgvector -e POSTGRES_PASSWORD=postgres "
                         f"-p 5432:5432 pgvector/pgvector:pg16\n({e})")
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")      # once per database
    register_vector(conn)                                      # numpy arrays <-> vector columns
    return conn
# endregion helper


# region q1
# --------------------------------------------------------------------------
# Q1. Warm-up: a vector column, three distance operators, similarity = 1 - distance
# --------------------------------------------------------------------------
def q1():
    conn = connect()
    conn.execute("DROP TABLE IF EXISTS w2_items")
    conn.execute(f"CREATE TABLE w2_items (id bigserial PRIMARY KEY, content text NOT NULL, embedding vector({DIM}))")

    vecs = embed_texts(SAMPLE_DOCS, dim=DIM)
    with conn.cursor() as cur:
        cur.executemany("INSERT INTO w2_items (content, embedding) VALUES (%s, %s)",
                        [(t, v) for t, v in zip(SAMPLE_DOCS, vecs)])

    q = embed_query("a young cat sleeping", dim=DIM)
    for op, name in (("<=>", "cosine distance"), ("<->", "euclidean (L2) distance"), ("<#>", "NEGATIVE inner product")):
        rows = conn.execute(
            f"SELECT id, content, embedding {op} %s AS d FROM w2_items ORDER BY embedding {op} %s LIMIT 3", (q, q)
        ).fetchall()
        print(f"\nORDER BY embedding {op} q    ({name}, smaller = closer)")
        for id_, text, d in rows:
            print(f"  {d:8.4f}  id={id_}  {text[:45]}")

    # similarity = 1 - cosine distance; for <#> similarity = -value (unit vectors)
    best = conn.execute("SELECT content, 1 - (embedding <=> %s) AS sim FROM w2_items ORDER BY embedding <=> %s LIMIT 1",
                        (q, q)).fetchone()
    print(f"\nbest match: {best[0][:45]!r} cosine similarity = {best[1]:.3f}")
    conn.execute("DROP TABLE w2_items")
    return best[1]
# endregion q1


# region q2
# --------------------------------------------------------------------------
# Q2. Apply: vector search + SQL JOIN + filters in ONE query
# --------------------------------------------------------------------------
def q2():
    conn = connect()
    conn.execute("DROP TABLE IF EXISTS w2_chunks; DROP TABLE IF EXISTS w2_documents")
    conn.execute("""CREATE TABLE w2_documents (
                       id serial PRIMARY KEY, title text, tenant text, year int)""")
    conn.execute(f"""CREATE TABLE w2_chunks (
                       id bigserial PRIMARY KEY,
                       doc_id int REFERENCES w2_documents(id) ON DELETE CASCADE,
                       content text, embedding vector({DIM}))""")

    docs = [("Acme Refund Policy", "acme", 2024, ["Refunds are processed within 7 days.", "Gift cards are not refundable."]),
            ("Acme Shipping 2022", "acme", 2022, ["Parcels ship in 3 to 5 working days."]),
            ("Globex Refund Policy", "globex", 2024, ["Refunds take 14 days and need a receipt."])]
    for title, tenant, year, chunks in docs:
        doc_id = conn.execute("INSERT INTO w2_documents (title, tenant, year) VALUES (%s,%s,%s) RETURNING id",
                              (title, tenant, year)).fetchone()[0]
        for c, v in zip(chunks, embed_texts(chunks, dim=DIM)):
            conn.execute("INSERT INTO w2_chunks (doc_id, content, embedding) VALUES (%s,%s,%s)", (doc_id, c, v))

    q = embed_query("how long do refunds take", dim=DIM)
    rows = conn.execute("""
        SELECT d.title, c.content, 1 - (c.embedding <=> %(q)s) AS similarity
        FROM w2_chunks c
        JOIN w2_documents d ON d.id = c.doc_id
        WHERE d.tenant = %(tenant)s AND d.year >= %(year)s
        ORDER BY c.embedding <=> %(q)s
        LIMIT 3""", {"q": q, "tenant": "acme", "year": 2024}).fetchall()
    for title, content, sim in rows:
        print(f"{sim:.3f}  [{title}]  {content}")

    conn.execute("DELETE FROM w2_documents WHERE tenant = 'globex'")            # cascades to its chunks
    n = conn.execute("SELECT count(*) FROM w2_chunks").fetchone()[0]
    print("chunks left after deleting globex documents (cascade):", n)
    conn.execute("DROP TABLE w2_chunks; DROP TABLE w2_documents")
    return rows
# endregion q2


# region q3
# --------------------------------------------------------------------------
# Q3. Build: an idempotent store - upsert with ON CONFLICT, atomic replace, similarity threshold
# --------------------------------------------------------------------------
class PgVectorStore:
    def __init__(self, table="w2_store", dim=DIM):
        self.conn, self.table, self.dim = connect(), table, dim
        self.conn.execute(f"""CREATE TABLE IF NOT EXISTS {table} (
            source text NOT NULL, chunk_index int NOT NULL, content text NOT NULL,
            embedding vector({dim}) NOT NULL, PRIMARY KEY (source, chunk_index))""")

    def upsert(self, source: str, texts: list[str]) -> int:
        vecs = embed_texts(texts, dim=self.dim)
        with self.conn.cursor() as cur:
            cur.executemany(f"""INSERT INTO {self.table} (source, chunk_index, content, embedding)
                                VALUES (%s,%s,%s,%s)
                                ON CONFLICT (source, chunk_index)
                                DO UPDATE SET content = EXCLUDED.content, embedding = EXCLUDED.embedding""",
                            [(source, i, t, v) for i, (t, v) in enumerate(zip(texts, vecs))])
        return len(texts)

    def replace_source(self, source: str, texts: list[str]) -> int:
        """Delete old chunks + insert new ones in ONE transaction: readers never see a half-updated file."""
        vecs = embed_texts(texts, dim=self.dim)                       # embed first: no network call inside the transaction
        with self.conn.transaction():
            self.conn.execute(f"DELETE FROM {self.table} WHERE source = %s", (source,))
            with self.conn.cursor() as cur:
                cur.executemany(f"INSERT INTO {self.table} (source, chunk_index, content, embedding) VALUES (%s,%s,%s,%s)",
                                [(source, i, t, v) for i, (t, v) in enumerate(zip(texts, vecs))])
        return len(texts)

    def search(self, question: str, k: int = 3, min_similarity: float = 0.5):
        q = embed_query(question, dim=self.dim)
        rows = self.conn.execute(f"""SELECT source, content, 1 - (embedding <=> %s) AS sim
                                     FROM {self.table} ORDER BY embedding <=> %s LIMIT %s""", (q, q, k)).fetchall()
        return [{"source": s, "text": c, "similarity": round(float(sim), 3)} for s, c, sim in rows if sim >= min_similarity]

    def count(self) -> int:
        return self.conn.execute(f"SELECT count(*) FROM {self.table}").fetchone()[0]


def q3():
    conn = connect(); conn.execute("DROP TABLE IF EXISTS w2_store"); conn.close()
    store = PgVectorStore()
    v1 = ["Queues run slow jobs in the background.", "Horizon monitors Redis queues."]
    store.upsert("queues.md", v1)
    store.upsert("queues.md", v1)                                   # same keys again -> updated in place
    print("rows after upserting the same file twice:", store.count())
    store.replace_source("queues.md", ["Queues can use Redis, SQS or the database."])
    print("rows after replace_source (file shrank):", store.count())
    print("relevant :", store.search("background jobs and queues", min_similarity=0.3))
    print("off-topic:", store.search("capital city of France", min_similarity=0.75))
    store.conn.execute("DROP TABLE w2_store")
    return True
# endregion q3


# region q4
# --------------------------------------------------------------------------
# Q4. Debug: four reasons "my pgvector search is wrong or slow"
# --------------------------------------------------------------------------
def q4():
    conn = connect()
    rng = np.random.default_rng(0)
    conn.execute("DROP TABLE IF EXISTS w2_demo")
    conn.execute("CREATE TABLE w2_demo (id serial PRIMARY KEY, embedding vector(32))")
    data = rng.normal(size=(20000, 32)).astype(np.float32)
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    with conn.cursor() as cur, cur.copy("COPY w2_demo (embedding) FROM STDIN WITH (FORMAT TEXT)") as cp:
        for v in data:
            cp.write_row(["[" + ",".join(f"{x:.6f}" for x in v) + "]"])     # fast bulk load
    conn.execute("CREATE INDEX w2_demo_hnsw ON w2_demo USING hnsw (embedding vector_cosine_ops)")
    conn.execute("ANALYZE w2_demo")
    q = data[0]

    def plan(sql: str) -> str:
        rows = conn.execute("EXPLAIN " + sql, (q,)).fetchall()
        text = " | ".join(r[0].strip() for r in rows[:3])
        return "INDEX (hnsw)" if "w2_demo_hnsw" in text else "SEQ SCAN (exact, slow on big tables)"

    # BUG 1: dimension mismatch
    try:
        conn.execute("INSERT INTO w2_demo (embedding) VALUES (%s)", (np.zeros(768, dtype=np.float32),))
    except Exception as e:
        print("bug1 ->", type(e).__name__, str(e).splitlines()[0])

    # BUG 2: ordering by a computed similarity -> planner cannot use the index
    print("bug2  ORDER BY 1 - (embedding <=> q) DESC  ->", plan(
        "SELECT id FROM w2_demo ORDER BY 1 - (embedding <=> %s) DESC LIMIT 5"))
    print("fix   ORDER BY embedding <=> q            ->", plan(
        "SELECT id FROM w2_demo ORDER BY embedding <=> %s LIMIT 5"))

    # BUG 3: operator does not match the index operator class (index is vector_cosine_ops)
    print("bug3  ORDER BY embedding <-> q (L2)        ->", plan(
        "SELECT id FROM w2_demo ORDER BY embedding <-> %s LIMIT 5"))

    # BUG 4: threshold written as similarity but compared to a distance
    d = conn.execute("SELECT embedding <=> %s FROM w2_demo ORDER BY embedding <=> %s LIMIT 1", (q, q)).fetchone()[0]
    print(f"bug4  the identical vector has cosine DISTANCE {max(d, 0):.2f}; `WHERE distance > 0.8` would exclude it")
    conn.execute("DROP TABLE w2_demo")
    return True
# endregion q4


# region q5
# --------------------------------------------------------------------------
# Q5. Production: measure HNSW recall and latency vs ef_search (exact search = ground truth)
# --------------------------------------------------------------------------
def q5(n: int = 20000, dim: int = 32, n_queries: int = 40):
    conn = connect()
    rng = np.random.default_rng(1)
    centers = rng.normal(size=(50, dim))
    data = centers[rng.integers(0, 50, n)] + 0.6 * rng.normal(size=(n, dim))
    data = (data / np.linalg.norm(data, axis=1, keepdims=True)).astype(np.float32)
    queries = data[rng.choice(n, n_queries, replace=False)] + 0.05 * rng.normal(size=(n_queries, dim)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    conn.execute("DROP TABLE IF EXISTS w2_ann")
    conn.execute(f"CREATE TABLE w2_ann (id int PRIMARY KEY, embedding vector({dim}))")
    with conn.cursor() as cur, cur.copy("COPY w2_ann (id, embedding) FROM STDIN WITH (FORMAT TEXT)") as cp:
        for i, v in enumerate(data):
            cp.write_row([i, "[" + ",".join(f"{x:.6f}" for x in v) + "]"])
    t0 = time.perf_counter()
    conn.execute("CREATE INDEX w2_ann_hnsw ON w2_ann USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)")
    conn.execute("ANALYZE w2_ann")
    print(f"built HNSW on {n:,} vectors in {time.perf_counter() - t0:.1f}s (m=16, ef_construction=64)")

    truth = np.argsort(-(queries @ data.T), axis=1)[:, :10]             # exact top-10 in numpy

    print(f"{'ef_search':>9} {'recall@10':>10} {'ms/query':>9}")
    out = []
    for ef in (10, 40, 100, 200):
        conn.execute(f"SET hnsw.ef_search = {ef}")
        hits, t0 = 0, time.perf_counter()
        for qv, tr in zip(queries, truth):
            ids = [r[0] for r in conn.execute("SELECT id FROM w2_ann ORDER BY embedding <=> %s LIMIT 10", (qv,)).fetchall()]
            hits += len(set(ids) & set(tr.tolist()))
        ms = (time.perf_counter() - t0) / n_queries * 1000
        rec = hits / (10 * n_queries)
        out.append((ef, rec))
        print(f"{ef:>9} {rec:>10.2f} {ms:>9.2f}")
    conn.execute("DROP TABLE w2_ann")
    return out
# endregion q5


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3", "4", "5"]
    for n in which:
        print(f"\n===== Q{n} =====")
        globals()[f"q{n}"]()
