"""
Solutions for 03-generating-embeddings-with-gemini/practice.md
Run everything:   python solutions.py        (needs GEMINI_API_KEY in .env - free tier)
Run one:          python -c "import solutions; solutions.q1()"
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))
from gemini_utils import *   # get_client, EMBED_MODEL, embed_texts, embed_query, cosine_sim, top_k, SAMPLE_DOCS, np ...

import hashlib
import time
from google.genai import types


# region q1
# --------------------------------------------------------------------------
# Q1. Warm-up: call the raw API once and inspect the response
# --------------------------------------------------------------------------
def q1():
    client = get_client()
    resp = client.models.embed_content(
        model="gemini-embedding-001",
        contents=["What is RAG?", "Retrieval-Augmented Generation explained"],
    )
    vecs = np.array([e.values for e in resp.embeddings], dtype=np.float32)
    print("number of vectors :", len(resp.embeddings))
    print("shape             :", vecs.shape)          # (2, 3072) - native size
    print("first 5 numbers   :", np.round(vecs[0][:5], 4))
    print("length of vec 0   :", round(float(np.linalg.norm(vecs[0])), 4))   # ~1.0 at 3072
    return vecs.shape
# endregion q1


# region q2
# --------------------------------------------------------------------------
# Q2. Apply: does task_type change retrieval? Compare three settings
# --------------------------------------------------------------------------
def q2():
    corpus = [
        "To reset your password, click 'Forgot password' on the login page.",
        "Our refund policy allows returns within 30 days of purchase.",
        "The API rate limit is 100 requests per minute per key.",
        "Invoices are emailed on the first day of every month.",
        "Two-factor authentication can be enabled under Security settings.",
    ]
    question = "How can I get back into my account if I forgot my login?"

    def rank(doc_type: str, query_type: str):
        D = embed_texts(corpus, task_type=doc_type, dim=768)
        q = embed_texts([question], task_type=query_type, dim=768)[0]
        scores = D @ q
        order = np.argsort(-scores)
        return [(int(i), round(float(scores[i]), 3)) for i in order[:3]]

    print("RETRIEVAL_DOCUMENT + RETRIEVAL_QUERY :", rank("RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"))
    print("SEMANTIC_SIMILARITY both sides       :", rank("SEMANTIC_SIMILARITY", "SEMANTIC_SIMILARITY"))
    print("both as RETRIEVAL_DOCUMENT (misuse)  :", rank("RETRIEVAL_DOCUMENT", "RETRIEVAL_DOCUMENT"))
    return rank("RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY")[0][0]
# endregion q2


# region q3
# --------------------------------------------------------------------------
# Q3. Build: your own embedding client (batching + disk cache + retry + stats)
#     The API call is injected, so you can unit-test it with a fake (no key needed).
# --------------------------------------------------------------------------
class EmbeddingStore:
    def __init__(self, api_call=None, cache_dir=".q3_cache", batch_size=50, dim=768, max_tries=5):
        self.api_call = api_call or self._gemini_call
        self.cache = pathlib.Path(cache_dir)
        self.cache.mkdir(exist_ok=True)
        self.batch_size, self.dim, self.max_tries = batch_size, dim, max_tries
        self.stats = {"api_calls": 0, "cache_hits": 0, "texts_embedded": 0}

    @staticmethod
    def _gemini_call(batch, task_type, dim):
        resp = get_client().models.embed_content(
            model="gemini-embedding-001",
            contents=batch,
            config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=dim),
        )
        return np.array([e.values for e in resp.embeddings], dtype=np.float32)

    def _key(self, text, task_type):
        raw = f"gemini-embedding-001|{task_type}|{self.dim}|{text}"
        return self.cache / (hashlib.sha256(raw.encode()).hexdigest() + ".npy")

    def _call_with_retry(self, batch, task_type):
        delay = 1.0
        for attempt in range(1, self.max_tries + 1):
            try:
                return self.api_call(batch, task_type, self.dim)
            except Exception as e:                     # in real code catch the SDK's APIError (429/5xx) only
                if attempt == self.max_tries:
                    raise
                time.sleep(delay)
                delay *= 2

    def embed(self, texts, task_type="RETRIEVAL_DOCUMENT"):
        out = [None] * len(texts)
        missing = []                                   # (index, text) pairs not on disk
        for i, t in enumerate(texts):
            p = self._key(t, task_type)
            if p.exists():
                out[i] = np.load(p)
                self.stats["cache_hits"] += 1
            else:
                missing.append((i, t))

        # de-duplicate identical texts inside this call: embed each unique text once
        unique = {}
        for i, t in missing:
            unique.setdefault(t, []).append(i)
        uniq_texts = list(unique)

        for s in range(0, len(uniq_texts), self.batch_size):
            batch = uniq_texts[s : s + self.batch_size]
            raw = self._call_with_retry(batch, task_type)
            raw = raw / np.clip(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12, None)  # L2-normalise
            self.stats["api_calls"] += 1
            self.stats["texts_embedded"] += len(batch)
            for t, v in zip(batch, raw):
                np.save(self._key(t, task_type), v)
                for i in unique[t]:
                    out[i] = v
        return np.vstack(out)


def q3():
    store = EmbeddingStore(cache_dir=".q3_cache", batch_size=2)
    texts = ["alpha", "beta", "gamma", "alpha"]       # note the duplicate
    v = store.embed(texts)
    print("shape:", v.shape, " stats after run 1:", store.stats)
    v2 = store.embed(texts)
    print("stats after run 2 (all cached):", store.stats)
    return store.stats
# endregion q3


# region q4
# --------------------------------------------------------------------------
# Q4. Debug: three bugs hiding in one snippet
# --------------------------------------------------------------------------
BROKEN = '''
docs = load_10000_chunks()
vectors = []
for d in docs:                                              # BUG 1: one API call per chunk
    r = client.models.embed_content(
        model="gemini-embedding-001", contents=d,
        config=types.EmbedContentConfig(output_dimensionality=768))   # BUG 2: no normalisation after
    vectors.append(r.embeddings[0].values)
q = client.models.embed_content(                            # BUG 3: query embedded with the default/doc task
    model="gemini-embedding-001", contents=question,
    config=types.EmbedContentConfig(output_dimensionality=768)).embeddings[0].values
'''


def q4():
    print("BROKEN snippet:\n", BROKEN)
    texts = ["How do I reset my password?", "Our office is closed on public holidays",
             "Invoices are due in 30 days"]

    # FIX 2 + 1: batch the call (one request for the list) and normalise the 768-d vectors
    client = get_client()
    resp = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts,                                                         # batch: 1 call, not 3
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT", output_dimensionality=768),
    )
    raw = np.array([e.values for e in resp.embeddings], dtype=np.float32)
    print("raw 768-d norms        :", np.round(np.linalg.norm(raw, axis=1), 3))   # not 1.0 -> must normalise
    D = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    print("normalised norms       :", np.round(np.linalg.norm(D, axis=1), 3))

    # FIX 3: embed the user's question with RETRIEVAL_QUERY
    q = embed_query("forgot my login password", dim=768)
    print("top match:", texts[int(np.argmax(D @ q))])
    return D.shape
# endregion q4


# region q5
# --------------------------------------------------------------------------
# Q5. Production: an idempotent ingestion job that never pays twice
# --------------------------------------------------------------------------
def ingest(chunks: list[dict], store: EmbeddingStore) -> dict:
    """chunks = [{'id':..., 'text':...}]. Returns {id: vector}. Re-running costs 0 API calls."""
    texts = [c["text"] for c in chunks]
    vecs = store.embed(texts, task_type="RETRIEVAL_DOCUMENT")
    return {c["id"]: v for c, v in zip(chunks, vecs)}


def q5():
    chunks = [{"id": f"doc-{i}", "text": t} for i, t in enumerate(SAMPLE_DOCS)]
    chunks.append({"id": "doc-dup", "text": SAMPLE_DOCS[0]})        # an exact duplicate chunk

    store = EmbeddingStore(cache_dir=".q5_cache", batch_size=50)
    index = ingest(chunks, store)
    first = dict(store.stats)
    print(f"run 1: {len(index)} vectors, stats={first}")            # 10 unique texts -> 1 API call

    ingest(chunks, store)
    second = {k: store.stats[k] - first[k] for k in first}
    print(f"run 2: extra stats={second}")                            # 0 api_calls, 11 cache hits
    return first, second
# endregion q5


if __name__ == "__main__":
    for fn in (q1, q2, q3, q4, q5):
        print(f"\n===== {fn.__name__} =====")
        fn()
