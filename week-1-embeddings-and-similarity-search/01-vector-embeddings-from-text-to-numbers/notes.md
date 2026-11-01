# 01 - Vector embeddings: from text to numbers

> Syllabus line: *What embeddings are, how text becomes numbers.*
> By the end you will be able to explain an embedding to a non-ML colleague, build a tiny one from scratch with numpy,
> and call a production model (`gemini-embedding-001`) knowing exactly what comes back.

## 🎯 What you will learn

- You will be able to **explain what an embedding is** (a fixed-length list of floats whose *direction* encodes meaning) and why every RAG system starts with one.
- You will be able to **say why keyword counting fails** (synonyms, paraphrase) and show it with numbers.
- You will be able to **build a tiny embedding model from scratch** (co-occurrence, PPMI, SVD) and see "car" land next to "automobile".
- You will be able to **describe the real pipeline** (tokenizer, transformer, pooling, normalise) and how contrastive training shapes the space.
- You will be able to **call `embed_texts`, inspect shape, dtype and norm**, and explain `task_type` and the 2048-token limit.
- You will be able to **name the limits** (negation, exact IDs, uncalibrated scores, model lock-in) so you know when embeddings alone are not enough.

## 🧠 The big idea in plain words

A computer cannot compare the *meaning* of two sentences. It can only compare numbers. So we need a machine that reads
a sentence and outputs numbers such that **sentences that mean similar things produce similar numbers**. That list of numbers is an
**embedding** (also called a **vector**, just "an ordered list of numbers").

Story. A user types **"my car won't start"**. Your help-centre article is titled **"Automobile engine failure troubleshooting"**.
They share zero words. A MySQL `LIKE '%car%'` finds nothing. An embedding model, however, has read a huge amount of text
and learned that "car" and "automobile" show up in the same kinds of sentences, so it places both near each other in a
"meaning space". Searching becomes: *turn the question into a point in that space, find the closest article points.*

Picture a map. Cities that are close on the map are close in distance. An embedding is a map of *meaning*: the sentence
is a point, nearby points mean similar things. Real maps have 2 numbers per place (latitude, longitude); Gemini's map
has 768, 1536 or 3072 numbers per sentence (you choose; 3072 is native).

**Laravel/MySQL mapping (only where it fits):**

| Embedding world | Familiar thing |
|---|---|
| Embedding model | A deterministic pure function, like a `Hash::make()` that preserves *similarity* instead of destroying it. Same text in, same vector out. |
| Vector column | A `JSON`/`BLOB` column holding `d` floats per row (pgvector gives it a real type, `vector(768)`). |
| "Switch embedding model" | A destructive migration: every stored vector must be regenerated, and old and new vectors cannot be compared. |
| Embedding cache | `Cache::remember(sha256($text), ...)`. Same text, skip the API call. Our helper does this on disk. |

## 📚 Concepts, step by step

### 1. Vector, dimension, point

A **vector** is an ordered list of numbers, e.g. `[0.2, -0.7, 0.5]`. The count is its **dimension** (`d = 3` here).
You can read it as a point in `d`-dimensional space, or as an arrow from the origin to that point. For `d = 2` you can
draw it on paper; for `d = 768` you cannot, but all the maths (add, multiply, angle) works identically.

In numpy: `np.array([0.2, -0.7, 0.5]).shape == (3,)`. A table of `N` embeddings is an `(N, d)` matrix, one row per text.

### 2. Why not just count words?

The oldest trick is **bag-of-words (BoW)**: build a vocabulary of all words, then represent each sentence as counts.

```
vocab:         car  automobile  fast  quick  banana
"car fast"      1       0        1     0      0
"automobile quick" 0    1        0     1      0
```

These two sentences mean the same, yet their vectors share **no non-zero position**, so their cosine similarity
(the angle between arrows, covered in topic 02) is exactly **0**. BoW only knows whether the *same string* appeared.
Two further problems: vectors are huge and mostly zeros (one slot per vocabulary word, easily 100,000 slots), and word order
is thrown away.

An embedding fixes both: it is **dense** (every slot is a meaningful non-zero float), **short** (hundreds, not
hundreds of thousands), and **learned so that meaning, not spelling, drives the geometry**.

### 3. A 2D meaning map you can verify by hand

Pretend a model only had two knobs: *(pet-ness, money-ness)*. We (not a model) assign:

| word | pet-ness | money-ness |
|---|---|---|
| cat | 9 | 1 |
| dog | 8 | 2 |
| invoice | 1 | 9 |
| loan | 0 | 8 |

Similarity of two arrows is the cosine of their angle: `cos(a, b) = (a . b) / (|a| |b|)`, where `a . b` multiplies
matching positions and adds, and `|a|` is the arrow length `sqrt(sum of squares)`.

- `cos(cat, dog) = (9*8 + 1*2) / (sqrt(82) * sqrt(68)) = 74 / (9.055 * 8.246) ≈ 0.991`
- `cos(cat, invoice) = (9*1 + 1*9) / (9.055 * 9.055) = 18 / 82 ≈ 0.22`
- `cos(cat, loan) = (9*0 + 1*8) / (9.055 * 8) ≈ 0.11`

So cat is near dog and far from money words. **Real models do exactly this, but with hundreds of knobs nobody labelled.**
A single dimension of Gemini's vector does *not* mean "pet-ness"; the meaning lives in the pattern across all of them.
Do not try to read individual coordinates.

### 4. Where do the numbers come from? Co-occurrence ("you shall know a word by its company")

The core idea, from the 1950s linguist J.R. Firth and used by every embedding model since: **words that appear in
similar contexts have similar meanings.** "Car" and "automobile" both appear near "fast", "fuel", "engine"; "banana"
appears near "yellow", "ripe", "sweet".

Recipe for a tiny embedding model (you will run it in the code walkthrough):

1. Build a table **C** (words by words): how often does word *i* appear with word *j* in the same sentence?
2. Re-weight it with **PPMI** (positive pointwise mutual information): keep a pair only if it co-occurs *more than
   chance*. This stops "the" and "is" from dominating.
3. Compress with **SVD** (singular value decomposition, a matrix factorisation you can think of as "find the few
   directions that carry most of the pattern"). Each word shrinks from a 13-number row to a 3-number dense vector.

Two words with the same company end up with the same row, hence nearly the same dense vector. No one told the program
that cars are automobiles; it *discovered* it from usage.

### 5. How modern models do it: transformers and contrastive training

`gemini-embedding-001` replaces the counting table with a **transformer** (a neural network that reads all tokens of a text
at once and lets each token look at all the others). The pipeline:

```
"My car won't start"
   |  1. tokenizer: split into sub-word pieces ("My", " car", " won", "'t", " start") -> integer ids
   v
[ids]  2. embedding layer: each id -> its own vector
   v
   3. transformer layers: every token's vector is rewritten using context
      ("bank" next to "river" differs from "bank" next to "loan")
   v
   4. pooling: squash N token vectors into ONE vector for the whole text
   v
   5. (optional) projection to the target size, then L2-normalise
   v
[0.012, -0.044, ... 3072 floats]   <- the embedding
```

**Training** is usually *contrastive*: show the network millions of (question, relevant passage) pairs. For each pair,
push their two vectors **together**; push the vectors of unrelated passages in the same batch (**negatives**) **apart**.
After enough pairs the geometry organises itself so that "things that answer each other" sit close. That is why
retrieval-trained models are great at matching a *question* to its *answer passage*, not only a sentence to its twin.

**Context matters (contextual vs static).** Older models (word2vec, GloVe) gave one fixed vector per word. Transformers
give a vector per *whole text* that depends on the surrounding words, so "apple pie recipe" and "Apple earnings report"
land in different places.

### 6. What the real vector looks like

| Property | `gemini-embedding-001` fact |
|---|---|
| Native dimension | **3072** floats |
| Smaller sizes | **768 / 1536 / 3072** recommended via `output_dimensionality` (Matryoshka training: the first numbers carry most meaning, so truncating works) |
| Normalisation | Only the **3072** vector arrives pre-normalised (length 1). Truncated vectors **must be re-normalised**; our helper does it. |
| Max input | about **2048 tokens** per text (a token is roughly 3/4 of an English word). Longer text must be split (chunking, week 3). |
| dtype | numbers are float; we store `float32` (4 bytes each) |
| Output | `(N, d)` matrix for `N` texts; row order = input order |

**Memory arithmetic worth memorising:** `bytes = N x d x 4`. One 768-d vector = 3,072 bytes (about 3 KB). One million
vectors at 768-d = **about 3.1 GB**; at 3072-d = **about 12.3 GB**. This single formula drives most database sizing decisions (topic 06).

Comparison only, prose: OpenAI's `text-embedding-3-small/large` work the same way (1536 / 3072 native dims, also
shortenable). We do not call them in this course; everything runs on the free Gemini tier.

### 7. The `task_type` knob

The same model can specialise its output for the job you tell it about:

| `task_type` | Use it for |
|---|---|
| `RETRIEVAL_DOCUMENT` | chunks you **store** in your index |
| `RETRIEVAL_QUERY` | the user's **question** at search time |
| `SEMANTIC_SIMILARITY` | comparing two texts of the same kind (duplicates, paraphrase) |
| `CLASSIFICATION` | features for a classifier |
| `CLUSTERING` | grouping texts by topic |

Why separate query and document types? A question ("how to reset password?") and its answer ("Go to Settings, click
Security...") do not *look* alike; the model is trained so that a `QUERY`-flavoured vector lands near the `DOCUMENT`-flavoured
vector that answers it. Mixing them (both as documents) usually works a little worse. **Rule: store with DOCUMENT,
search with QUERY, and never mix model, task type or dimension between the two sides' comparisons you care about.**

### 8. What embeddings are bad at (know this before you trust them)

- **Negation:** "I love this phone" and "I don't love this phone" come out *very close*.
- **Exact identifiers:** error codes (`ERR_4021`), SKUs, invoice numbers, rare names. The model sees them as odd token soup. Keyword search wins here (topic 05).
- **Numbers and units:** "30 days" vs "90 days" are nearly identical to the model.
- **Uncalibrated scores:** 0.62 does not mean "62% relevant". Scores are comparable *within one model and one query*, not across models or even across queries.
- **Language/domain drift:** a model that never saw your legal or medical jargon places it poorly. Always evaluate on *your* data.
- **Lock-in:** vectors from model A are meaningless next to vectors from model B. Changing models, dimension or task type means re-embedding the whole corpus (a "migration").

## 💻 Code walkthrough

Setup once: `pip install -r requirements.txt`, then copy `.env.example` to `.env` and paste your free key from
https://aistudio.google.com/apikey. All Gemini calls below go through `common/gemini_utils.py`.

### Stage 1 - Build a mini embedding model from scratch (no API, pure numpy)

```python
import numpy as np

corpus = [
    "the car is fast", "the automobile is fast", "the car is quick", "the automobile is quick",
    "a fast car needs fuel", "a quick automobile needs fuel",
    "the banana is yellow", "the apple is red", "a ripe banana is sweet", "a ripe apple is sweet",
    "the banana needs sunshine", "the apple needs sunshine",
]
STOP = {"the", "a", "is"}
tokens = [[w for w in s.split() if w not in STOP] for s in corpus]
vocab = sorted({w for t in tokens for w in t})
ix = {w: i for i, w in enumerate(vocab)}
V = len(vocab)

# 1) how often does word i share a sentence with word j?
C = np.zeros((V, V))
for t in tokens:
    for i, w in enumerate(t):
        for j, c in enumerate(t):
            if i != j:
                C[ix[w], ix[c]] += 1

# 2) PPMI: keep only pairs that co-occur MORE than chance
total = C.sum()
p_w = C.sum(axis=1, keepdims=True) / total
p_c = C.sum(axis=0, keepdims=True) / total
with np.errstate(divide="ignore", invalid="ignore"):
    pmi = np.log((C / total) / (p_w * p_c))
ppmi = np.where(np.isfinite(pmi) & (pmi > 0), pmi, 0.0)

# 3) SVD squeezes each 13-number row into 3 dense numbers
U, S, _ = np.linalg.svd(ppmi)
k = 3
E = U[:, :k] * S[:k]
E = E / np.linalg.norm(E, axis=1, keepdims=True)      # unit length

def sim(a, b):
    return float(E[ix[a]] @ E[ix[b]])

print("vocab size:", V, "-> embedding shape:", E.shape)
for a, b in [("car", "automobile"), ("fast", "quick"), ("apple", "banana"), ("car", "banana")]:
    raw = ppmi[ix[a]] @ ppmi[ix[b]] / (np.linalg.norm(ppmi[ix[a]]) * np.linalg.norm(ppmi[ix[b]]))
    print(f"{a:>10} ~ {b:<10} raw-counts {raw:.2f}   dense {sim(a, b):.2f}")
```

Why each step: step 1 turns text into statistics; step 2 removes noise from frequent words; step 3 forces the model to
**generalise**, because 3 numbers cannot memorise 13 columns, so words with the same company must share coordinates.
Expected output (this is real, deterministic output):

```
vocab size: 13 -> embedding shape: (13, 3)
       car ~ automobile raw-counts 0.87   dense 1.00
      fast ~ quick      raw-counts 0.87   dense 1.00
     apple ~ banana     raw-counts 0.37   dense 1.00
       car ~ banana     raw-counts 0.01   dense 0.01
```

Notice "apple ~ banana": the raw rows only overlap 0.37 (they share "ripe", "sweet", "sunshine" but not "red"/"yellow"),
yet after compression they are identical. **Compression is what creates generalisation.** Try `k = 2` and `k = 4`
to see what happens when you squeeze too much or too little (it is the same dimensionality trade-off as topic 06).

### Stage 2 - The real thing: one call, inspect everything

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))
from gemini_utils import embed_texts, embed_query, cosine_sim, top_k, SAMPLE_DOCS
import numpy as np

vecs = embed_texts(SAMPLE_DOCS, task_type="RETRIEVAL_DOCUMENT", dim=768)
print("shape:", vecs.shape, "dtype:", vecs.dtype)
print("first 5 numbers:", np.round(vecs[0][:5], 4))
print("length of row 0:", float(np.linalg.norm(vecs[0])))
print("bytes for these vectors:", vecs.nbytes)
```

- `embed_texts` takes a *list* and returns an `(N, dim)` `float32` numpy array, one row per text, same order.
- It batches the list into API calls (up to 50 texts each), retries on rate-limit errors and caches every vector on disk
  (`.embed_cache/`), so running this twice costs zero extra API calls.
- `norm` prints about `1.0` because the helper L2-normalises (required for 768 because only 3072 comes normalised).

Expected output (≈; the five numbers will differ):

```
shape: (10, 768) dtype: float32
first 5 numbers: [ 0.0123 -0.0291  0.0457 ...]
length of row 0: 1.0
bytes for these vectors: 30720
```

`30720 = 10 x 768 x 4`. The arithmetic from section 6 checks out.

### Stage 3 - Ask a question, read the neighbours

```python
query = "Which tool is good for caching?"
qv = embed_query(query, dim=768)                 # RETRIEVAL_QUERY under the hood

for idx, score in top_k(qv, vecs, k=3):
    print(f"{score:.3f}  {SAMPLE_DOCS[idx]}")
```

`embed_query` returns shape `(768,)`. `top_k` computes cosine similarity against every row and returns the best `k`.
Expected (scores ≈; the *order* is what matters):

```
0.7xx  Redis is an in-memory key-value store often used for caching and queues.
0.5xx  MySQL FULLTEXT indexes allow fast keyword search on text columns.
0.5xx  AWS RDS is a managed relational database service for MySQL and PostgreSQL.
```

The question never says "Redis", yet the Redis sentence wins: meaning, not spelling.

### Stage 4 - Paraphrase beats vocabulary

```python
pairs = [
    ("A cat sat quietly on the warm windowsill.", "A kitten is napping in a sunny spot."),   # same idea, few shared words
    ("A cat sat quietly on the warm windowsill.", "The Indian Constitution guarantees fundamental rights."),
]
for a, b in pairs:
    E = embed_texts([a, b], task_type="SEMANTIC_SIMILARITY", dim=768)
    print(round(float(cosine_sim(E[0], E[1:2])[0]), 3), "|", b)
```

(`cosine_sim(E[0], E[1:2])` returns shape `(1,)` because the second argument is a one-row matrix.) Expect the first pair
clearly higher than the second (≈ 0.6-0.8 versus ≈ 0.3-0.5). The gap, not the absolute number, is the signal.

### Stage 5 - See the geometry yourself (preview)

Topic 02 turns "angle between arrows" into code. A visual preview: embed 12 sentences on 3 topics and project 768
numbers down to 2 with PCA (practice Q3 builds this from scratch). Topic clusters stay visible even after throwing
away 99.7% of the numbers. Another reminder that *direction carries meaning*.

## ⚠️ Common mistakes & debugging

| Symptom | Cause | Fix |
|---|---|---|
| `RuntimeError: GEMINI_API_KEY missing` | `.env` not created, or script run from a folder where `load_dotenv()` cannot find it | Copy `.env.example` to `.env`, run from the course root or export the variable |
| Scores for relevant docs are suspiciously low or ranking is random | Stored documents with one `task_type`/`dim`, queried with another, or mixed two models | Same model and same `dim` for both sides; `RETRIEVAL_DOCUMENT` for store, `RETRIEVAL_QUERY` for search |
| `ValueError` on `@` / shapes `(768,)` vs `(1536,)` | Vectors created with different `dim` | Pick one dimension per index; re-embed if you change it |
| Dot product scores not in [-1, 1] after truncating | Sliced a 3072 vector with `v[:768]` without re-normalising | `v / np.linalg.norm(v)` after slicing (or request `dim=768` from the API) |
| 429 errors / slow first run | Free-tier rate limit | Embed lists, not single strings; keep the on-disk cache; sleep and retry (helper already does) |
| "Negated" sentence ranks as the best match | Embeddings do not model logic well | Add a reranker or an LLM check (week 4); do not rely on cosine for negation |
| Long document embedded but only the start matters | Input limit about 2048 tokens; text beyond is cut or errors | Chunk before embedding (week 3) |
| Top result is a near-duplicate of the query itself | Query is also stored in the index | Exclude self-matches or dedupe before indexing |
| Different results after "just" upgrading the model name | New model, new geometry | Treat as a migration: re-embed everything, keep the model name next to each stored vector |

## 🚀 Pro-level insights (what the top 2% know)

- **Embeddings are a lossy summary.** One vector per text cannot hold every fact. If your chunk contains five unrelated
  ideas the vector is a blur of all five. The fix is small, single-topic chunks (week 3), not a bigger model.
- **Rank, don't threshold.** Absolute cosine values drift between models, languages and text lengths. Pipelines that say
  "keep anything above 0.8" break silently when the model changes. Use top-k, then rerank; if you must threshold,
  calibrate on labelled pairs from your own data (practice Q5 does this).
- **Store provenance with every vector:** `model`, `task_type`, `dim`, `text_hash`, `created_at`. It is the vector
  equivalent of a migration table; it lets you detect and rebuild a stale index.
- **Cache by `sha256(model|task|dim|text)`.** That is precisely what the helper does. Re-indexing a 100k-chunk corpus
  after a code change should cost zero API calls for unchanged chunks.
- **The "embedding gap".** A query and the passage that answers it are often *far* in embedding space (a question vs a
  statement). Two mitigations: use `RETRIEVAL_QUERY`/`RETRIEVAL_DOCUMENT`, or HyDE (week 4: have an LLM write a
  hypothetical answer and embed that).
- **Anisotropy.** Many models squeeze vectors into a narrow cone, so even unrelated texts score 0.3-0.5. Hence "compare
  relative scores, not raw ones", and why mean-centring sometimes helps.
- **Evaluate with your own question set.** Public benchmarks (MTEB) tell you which models to *shortlist*; your 50-200
  labelled (question, correct-chunk) pairs decide the winner. Metrics: recall@k, MRR (week 4).
- **When NOT to use embeddings:** exact lookups (IDs, SKUs, emails), strict filters (price < 500), and tiny corpora
  where a `LIKE`/`FULLTEXT` query is enough. Embeddings add latency, cost and non-determinism of ranking.
- **Cost/latency levers:** batch many texts per call; reduce `dim` (768 is a strong default); cache; embed offline at
  index time so the only online embedding call is the short user query.
- **Privacy:** text you embed through a hosted API leaves your server. Check data policies for confidential content.

## 🧪 Mini lab

Time: 10-15 minutes. No API needed for parts A-B; part C needs your key.

**A. Change the corpus.** In the Stage 1 script add six sentences about a third topic (say "the dog is loyal",
"a loyal puppy needs a walk", ...). Print `sim("dog", "puppy")` and `sim("dog", "car")`.
**B. Break it.** Set `k = 1`, then `k = 6`. Write down which pairs become too similar or too different and explain why
using the phrase "compression forces generalisation".
**C. Probe the real model.** Embed these four with `SEMANTIC_SIMILARITY`: `"I love this phone"`, `"I hate this phone"`,
`"I adore this phone"`, `"The weather is rainy"`. Print the 4x4 `cosine_sim` matrix. Which pair do you *expect* to be the least similar,
and what does the matrix say? (Typical surprise: love and hate score higher than love and rainy, because
the model groups by **topic**, not sentiment polarity.)

## ✅ Recap checklist

- [ ] I can define embedding, vector, dimension and dense vs sparse in one sentence each.
- [ ] I can show by hand that "car fast" vs "automobile quick" has BoW cosine 0.
- [ ] I can compute cosine of two 2D vectors on paper.
- [ ] I can explain co-occurrence, PPMI and SVD and why compression creates generalisation.
- [ ] I can draw the pipeline: tokenizer, transformer, pooling, normalise.
- [ ] I can explain contrastive training (pull positives together, push negatives apart).
- [ ] I know the Gemini facts: 3072 native, 768/1536/3072 via MRL, only 3072 pre-normalised, ~2048 tokens.
- [ ] I can compute memory for N vectors (`N x d x 4` bytes).
- [ ] I know when to use `RETRIEVAL_DOCUMENT` vs `RETRIEVAL_QUERY` vs `SEMANTIC_SIMILARITY`.
- [ ] I can list four things embeddings are bad at and what to use instead.

## 🎤 Interview questions

**1. What is an embedding and why does RAG need one?**
A fixed-length vector of floats produced by a model so that texts with similar meaning get nearby vectors. RAG needs
it to retrieve relevant chunks by meaning (not exact words) before asking the LLM to answer.

**2. Why does bag-of-words fail where embeddings succeed?**
BoW is sparse and tied to exact tokens: synonyms have orthogonal vectors (cosine 0). Embeddings are dense and learned
from context, so "car" and "automobile" are close.

**3. How is an embedding model trained?**
Typically contrastively: pairs of related texts (question and answer passage) are pulled together, in-batch unrelated
texts are pushed apart, so geometry encodes relevance.

**4. A colleague stores `v[:768]` of a 3072-d Gemini vector and uses a dot product. What is wrong?**
Only the full 3072-d vector is pre-normalised. A slice has length below 1, so dot products are scaled down and uneven
across documents, which can change rankings and break thresholds. Re-normalise (or request `output_dimensionality=768`).

**5. When would you refuse to use embeddings alone?**
For exact identifiers, numbers, negation-sensitive queries and hard filters. Use keyword or metadata filtering, hybrid
search and reranking (topic 05 and week 4).

## 🔗 Next

**02 - Similarity metrics:** cosine, dot product and Euclidean distance, when each is correct, and why normalised vectors make them agree, which unlocks writing your own search in topic 04.
