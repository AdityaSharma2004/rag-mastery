# 03 - Generating embeddings with Gemini (free tier)

> Week 1 · Topic 3 of 6 · Time: ~75 min · Needs: free `GEMINI_API_KEY` from https://aistudio.google.com/apikey

## 🎯 What you will learn

- You will be able to call `gemini-embedding-001` directly with the `google-genai` SDK and read the response.
- You will be able to choose the right `task_type` for documents, queries, classification and clustering, and explain why it matters.
- You will be able to control vector size with `output_dimensionality` and **re-normalise** correctly.
- You will be able to embed thousands of texts without hitting free-tier limits: batching, retry with back-off, and a disk cache.
- You will be able to rebuild the course helper `common/gemini_utils.py` from memory.

## 🧠 The big idea in plain words

An embedding model is a **web service that turns text into a list of numbers**. You send strings, you get back one vector per string. That is all. Everything else in this topic is *being a good API citizen*: asking for the right kind of vector, not paying twice for the same text, and surviving rate limits.

Laravel analogy: this is the `GeminiService` class in `app/Services`. Controllers (your search code) should never think about API keys, retries or caching. They call `$embeddings->embed($texts)`. Our `common/gemini_utils.py` is that service.

Two facts shape every design decision:

1. **Embedding is cheap but not free of limits.** On the free tier you are limited by requests per minute and per day (the exact numbers change, so check the AI Studio dashboard). Batching and caching are not optimisations, they are survival.
2. **Documents and questions are different kinds of text.** "How do I reset my password?" and "Click *Forgot password* on the login page" mean related things but look nothing alike. The model has a dial (`task_type`) that says *"this is a question"* or *"this is an answer-bearing passage"* so that they land near each other.

## 📚 Concepts, step by step

### 1. The model

| Item | Value |
|---|---|
| Model name | `gemini-embedding-001` (also written `models/gemini-embedding-001`; both accepted) |
| Native size | **3072** numbers per text |
| Other sizes | any via `output_dimensionality`; recommended **768, 1536, 3072** |
| Input limit | about **2048 tokens** per text (roughly 1,500 English words); longer text is cut off unless you chunk |
| Languages | 100+ (including Hindi), so a Hindi question can match an English passage |
| Cost | free tier (rate-limited); paid tier exists but we never need it here |
| Output | text only: images are **not** embedded by this model |

For comparison (prose only, we do not call it): OpenAI's `text-embedding-3-small` is 1536 dims and `text-embedding-3-large` is 3072 dims. Same idea, different vendor. **Never mix vectors from two different models in one index**: their number spaces are unrelated.

### 2. Anatomy of a call

```python
resp = client.models.embed_content(
    model="gemini-embedding-001",
    contents=["text one", "text two"],      # a LIST = one batch = one request
    config=types.EmbedContentConfig(
        task_type="RETRIEVAL_DOCUMENT",
        output_dimensionality=768,
    ),
)
vectors = [e.values for e in resp.embeddings]   # list of list-of-floats
```

`resp.embeddings` is a list with one entry per input text, **in the same order**. Each has `.values`.

`EmbedContentConfig` in the current SDK exposes: `task_type`, `output_dimensionality`, `title` (an optional document title that can improve `RETRIEVAL_DOCUMENT` quality) and `auto_truncate`.

### 3. Task types: telling the model *why* you embed

| `task_type` | Use it for | Side of the search |
|---|---|---|
| `RETRIEVAL_DOCUMENT` | chunks you **store** | index side |
| `RETRIEVAL_QUERY` | the user's **question** | query side |
| `SEMANTIC_SIMILARITY` | comparing two texts as equals (duplicates, paraphrase) | symmetric |
| `CLASSIFICATION` | features for a classifier | - |
| `CLUSTERING` | grouping texts by topic | - |
| `QUESTION_ANSWERING`, `FACT_VERIFICATION`, `CODE_RETRIEVAL_QUERY` | specialised query-side variants | query side |

The golden RAG rule: **chunks = `RETRIEVAL_DOCUMENT`, question = `RETRIEVAL_QUERY`**. The two vectors are trained to sit close together when the chunk answers the question. Using the wrong pair does not crash, it quietly lowers recall, which is why it is such a common hidden bug. Vectors made with different task types are **not interchangeable** (our cache key includes the task type for that reason).

### 4. Dimensions and normalisation (the trap)

Gemini uses *Matryoshka* training: the first N numbers of the 3072-vector are themselves a valid, smaller embedding. So you can ask for 768 and get a usable vector.

**But:** only the 3072-dim vector comes already length-1. A 768 or 1536 vector does not. Before you use dot product as cosine you must divide by the norm:

```python
v = v / np.linalg.norm(v)
```

Skipping this is Q4 in the practice file. Topic 06 measures how much quality you lose going 3072 → 768 (usually very little) while memory drops 4×.

### 5. Batching

One request can carry many strings (the course helper uses 50 per call). Why batch?
- Free-tier limits count *requests* as well as tokens: 50 texts in 1 request costs 1 request, 50 requests cost 50.
- Network latency is paid once, not 50 times.

Do not exceed the API's per-request limits. If an error says the batch is too large, lower the batch size.

### 6. Rate limits and retries

Free tier = **HTTP 429** ("quota exceeded") will happen. Also 500/503 are transient. Strategy: **exponential back-off**:

```
try → fail (429) → wait 2s → try → fail → wait 4s → try → fail → wait 8s ... (cap at 60s)
```

Retry only retryable codes (429, 500, 502, 503, 504). A 400 (bad request) or 403 (bad key) will never succeed by waiting.

### 7. Caching: pay once per unique text

Embeddings are **deterministic for the same model + task type + dimension + text**. So cache them forever:

```
key  = sha256(model | task_type | dim | text)
file = .embed_cache/<key>.npy
```

Laravel analogy: `Cache::remember($key, ..., fn() => $api->embed($text))`. Benefits: re-running a notebook costs 0 requests, tweaking your RAG code does not burn quota, and a crashed ingestion job resumes where it stopped. Invalidate by changing the model/dim/task (they are in the key), not by deleting files.

### 8. De-duplication

If 40 of your 1,000 chunks are identical (page footers, boilerplate), embed the unique text once and reuse the vector. Our `EmbeddingStore` in Q3 does this inside each call.

### 9. What the API does *not* do

- It does not chunk long text for you. Over 2048 tokens is truncated or rejected (`auto_truncate`). Week 3 covers chunking.
- It does not remember anything: no index, no search. Storage and search are your job (topic 04 and Week 2).

## 💻 Code walkthrough

### Stage 0 - setup

```bash
pip install -r requirements.txt
cp .env.example .env        # paste your key: GEMINI_API_KEY=...
```

### Stage 1 - the raw call (no helper)

```python
import os
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

resp = client.models.embed_content(
    model="gemini-embedding-001",
    contents=["What is RAG?", "Retrieval-Augmented Generation explained"],
)
vecs = np.array([e.values for e in resp.embeddings], dtype=np.float32)
print(vecs.shape)                          # (2, 3072)
print(np.linalg.norm(vecs, axis=1))        # ≈ [1. 1.]  (3072 is pre-normalised)
```

`dtype=np.float32` halves memory vs numpy's default float64 and is what vector databases store.

### Stage 2 - ask for 768 dims and fix the length

```python
resp = client.models.embed_content(
    model="gemini-embedding-001",
    contents=["What is RAG?"],
    config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY", output_dimensionality=768),
)
v = np.array(resp.embeddings[0].values, dtype=np.float32)
print(v.shape, np.linalg.norm(v))          # (768,) and a norm that is NOT 1.0
v /= np.linalg.norm(v)                     # now it is 1.0
```

### Stage 3 - the whole helper in 40 lines

```python
import hashlib, time
from pathlib import Path

CACHE = Path(".embed_cache"); CACHE.mkdir(exist_ok=True)

def _path(text, task_type, dim):
    return CACHE / (hashlib.sha256(f"gemini-embedding-001|{task_type}|{dim}|{text}".encode()).hexdigest() + ".npy")

def embed_texts(texts, task_type="RETRIEVAL_DOCUMENT", dim=768, batch=50):
    out, todo = [None] * len(texts), []
    for i, t in enumerate(texts):                       # 1) cache lookup
        p = _path(t, task_type, dim)
        if p.exists(): out[i] = np.load(p)
        else: todo.append(i)

    for s in range(0, len(todo), batch):                # 2) batch the misses
        idx = todo[s:s + batch]
        delay = 2
        for attempt in range(6):                        # 3) retry with back-off
            try:
                r = client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=[texts[i] for i in idx],
                    config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=dim))
                break
            except Exception:
                if attempt == 5: raise
                time.sleep(delay); delay = min(delay * 2, 60)
        M = np.array([e.values for e in r.embeddings], dtype=np.float32)
        M /= np.linalg.norm(M, axis=1, keepdims=True)   # 4) normalise
        for i, v in zip(idx, M):
            out[i] = v; np.save(_path(texts[i], task_type, dim), v)
        time.sleep(0.7)                                  # 5) be polite
    return np.vstack(out)
```

Read it as five numbered steps; the real `common/gemini_utils.py` is this plus a narrower `except` (only `errors.APIError` with a retryable status code).

### Stage 4 - documents vs query in a mini search

```python
docs = ["Click 'Forgot password' on the login page.",
        "Refunds are available within 30 days.",
        "API calls are limited to 100 per minute."]
D = embed_texts(docs, task_type="RETRIEVAL_DOCUMENT")
q = embed_texts(["I can't sign in"], task_type="RETRIEVAL_QUERY")[0]
print(docs[int(np.argmax(D @ q))])         # the password sentence
```

Expected: the password sentence, with a score around 0.6-0.8. The raw numbers vary by task type, so rely on the ranking.

## ⚠️ Common mistakes & debugging

| Symptom | Cause | Fix |
|---|---|---|
| `429 RESOURCE_EXHAUSTED` | Free-tier rate limit | Batch, back off, cache; pause between runs |
| Search quality quietly poor | Query embedded with `RETRIEVAL_DOCUMENT` (or no task type) | Query = `RETRIEVAL_QUERY`, chunks = `RETRIEVAL_DOCUMENT` |
| Scores/threshold behave oddly at 768 dims | Forgot to re-normalise | Divide by the norm (or use the helper) |
| Shape mismatch when adding to the DB | Mixed dimensions (some 768, some 3072) | One dimension per collection; put dim in the cache key |
| `API key not valid` / 403 | Key missing or `.env` not loaded | `load_dotenv()`; check the variable name `GEMINI_API_KEY` |
| Different vectors for "same" text | Different task type or dim, or trailing whitespace | Include all in the cache key; strip text consistently |
| Very long text gives weak vector | Over 2048-token limit truncated | Chunk it (Week 3) |
| Re-running costs quota every time | No cache | Disk cache keyed by hash |

## 🚀 Pro-level insights (what the top 2% know)

- **Embed once, at ingestion; embed only the query at runtime.** A query is one tiny request. If your app embeds documents at query time, the architecture is wrong.
- **Store the model name and dimension next to your vectors.** When you re-embed with a new model you need to know which rows are stale. Treat embeddings as a *derived, versioned* column (like a materialised view).
- **Re-embedding is a migration.** Changing model = re-embed everything and rebuild the index. Plan blue/green: build the new index beside the old one, switch, delete.
- **Use the `title` field for documents** when you have a natural title (page or section heading): it gives the model extra context at no cost.
- **Prepend context for fragments.** A chunk like "It was approved in 2019." is meaningless alone. Prepend "Document: Privacy Policy. Section: Data retention." before embedding (contextual chunking, Week 3).
- **Free-tier limits are per project and per minute/day**, not per key. Spread a big ingestion across time, persist progress, and make jobs idempotent (Q5).
- **Measure; do not trust.** Evaluate retrieval on 20-50 real questions (hit-rate@k), and re-check when you change dimension, task type or chunking.
- **When NOT to embed:** exact-match lookups (IDs, emails, SKU codes) belong in SQL/BM25, not vectors.

## 🧪 Mini lab (10-15 min)

1. Write 6 short help-centre sentences (2 about passwords, 2 billing, 2 shipping).
2. Embed them as `RETRIEVAL_DOCUMENT`. Embed 3 real-sounding questions as `RETRIEVAL_QUERY`.
3. Print the top-1 for each question.
4. Redo with *everything* as `RETRIEVAL_DOCUMENT`. Did any top-1 change? Did the score gap between top-1 and top-2 shrink?
5. Run the script twice. Confirm the second run makes zero API calls (add a counter).

## ✅ Recap checklist

- [ ] I can write a correct `embed_content` call with `task_type` and `output_dimensionality`.
- [ ] I know which task type goes on chunks and which on questions, and why.
- [ ] I re-normalise vectors that are not 3072-dim.
- [ ] I batch, retry only retryable errors with back-off, and cache by hash.
- [ ] I know the model is text-only, with a ~2048-token input limit.
- [ ] I never mix vectors from different models/dims/task types in one index.

## 🎤 Interview questions

**1. Why do embedding APIs have separate query and document modes?**
Questions and answer-bearing passages are phrased differently. Task-specific training places a question near the passages that answer it, improving recall over treating both identically.

**2. You reduced Gemini embeddings to 768 dims and your similarity scores dropped. Why?**
Only the 3072-dim output is normalised; truncated vectors need re-normalising, otherwise dot products no longer equal cosine.

**3. How do you handle API rate limits in an ingestion pipeline?**
Batch inputs, retry 429/5xx with exponential back-off, cache by content hash, make jobs resumable/idempotent, and spread load over time.

**4. What belongs in an embedding cache key?**
Model name, task type, dimensionality and the exact text, since changing any of them yields a different vector.

**5. When would you re-embed your corpus?**
When the embedding model, dimension, task type or chunking strategy changes, or when the source text changes (re-embed only the changed chunks).

## 🔗 Next

**04 - Vectors in numpy & similarity search:** store these vectors in a numpy matrix and build brute-force top-k search yourself, which unlocks understanding why vector databases exist.
