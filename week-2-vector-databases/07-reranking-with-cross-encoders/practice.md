# Practice - Re-ranking with cross-encoders

Five questions, rising difficulty. Attempt each before opening the hint, and the hint before the solution. Every solution is the exact code in `solutions.py` (`python solutions.py` runs all). **Q1, Q4, Q5 are pure Python (no key, no model).** Q2 uses `gemini-2.5-flash` as the reranker; Q3 uses Gemini embeddings for stage 1 and the local cross-encoder **if** `sentence-transformers` can download `cross-encoder/ms-marco-MiniLM-L-6-v2`, otherwise a lexical stub so it still runs. Documents: the 10-chunk support KB `DOCS` and 8 labelled queries `LABELED` at the top of `solutions.py`.

Shared helpers (rerankers with one interface, `score(query, docs) -> list[float]`):

```python
def _words(t): return set(re.findall(r"[a-z0-9]+", t.lower()))


class LexicalStubReranker:
    """Offline stand-in: scores a (query, doc) PAIR by word overlap. Same interface as the real rerankers."""
    name = "lexical-stub"

    def score(self, query: str, docs: list[str]) -> list[float]:
        q = _words(query)
        return [len(q & _words(d)) / (len(q) or 1) for d in docs]


class CrossEncoderReranker:
    """Local cross-encoder: reads query + passage TOGETHER and outputs one relevance score (a logit)."""
    name = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder          # imported lazily: heavy (torch)
        self.model = CrossEncoder(model_name)

    def score(self, query: str, docs: list[str]) -> list[float]:
        pairs = [(query, d) for d in docs]                       # ORDER MATTERS: (query, passage)
        return [float(s) for s in self.model.predict(pairs)]


def best_available_reranker():
    try:
        return CrossEncoderReranker()
    except Exception as e:                                       # no package, no network, no disk...
        print(f"[cross-encoder unavailable: {type(e).__name__}] using LexicalStubReranker for the demo")
        return LexicalStubReranker()


def mrr(rankings, truth):
    return round(float(np.mean([1 / (r.index(t) + 1) if t in r else 0.0 for r, t in zip(rankings, truth)])), 3)
```

---

### Q1. Warm-up - a cost model for two-stage retrieval

**Problem**

A corpus has 1,000,000 chunks. A cross-encoder costs about 8 ms per (query, chunk) pair; an ANN vector search costs about 20 ms. Compare reranking the whole corpus with retrieving 50 and reranking those. Report time per query and the speed-up.

<details><summary><b>Hint</b></summary>

All-pairs: `N × 8 ms`. Two-stage: `20 ms + 50 × 8 ms`. Convert milliseconds to hours for the first.

</details>

<details><summary><b>Solution</b></summary>

```python
# --------------------------------------------------------------------------
# Q1. Warm-up: why two stages? A cost model for bi-encoder vs cross-encoder
# --------------------------------------------------------------------------
def q1():
    n_docs, n_candidates = 1_000_000, 50
    bi_ms_per_query = 20.0          # ANN search over precomputed vectors (order of magnitude)
    cross_ms_per_pair = 8.0         # one cross-encoder forward pass on CPU (order of magnitude)

    all_pairs_ms = n_docs * cross_ms_per_pair
    two_stage_ms = bi_ms_per_query + n_candidates * cross_ms_per_pair
    print(f"cross-encoder on ALL {n_docs:,} docs : {all_pairs_ms / 1000 / 3600:,.1f} hours per query")
    print(f"two-stage (ANN top {n_candidates} + rerank): {two_stage_ms:,.0f} ms per query")
    print(f"speed-up: {all_pairs_ms / two_stage_ms:,.0f}x")
    print("\nbi-encoder : embeds each doc ONCE (offline); query = 1 embedding + vector search")
    print("cross-enc. : no precomputation possible; must read (query, doc) together for EVERY candidate")
    return two_stage_ms
```

</details>

**Why it works / what to learn**
The numbers are order-of-magnitude assumptions, but the shape is universal: a cross-encoder's cost grows with corpus size because nothing can be precomputed, so it can only run on a short list. The speed-up (~19,000×) is why every serious retrieval stack is two-stage.

**Expected output (≈)**
```text
cross-encoder on ALL 1,000,000 docs : 2.2 hours per query
two-stage (ANN top 50 + rerank): 420 ms per query
speed-up: 19,048x
```

---

### Q2. Apply - an LLM as reranker (one Gemini call, JSON scores)

**Problem**

Take the dense top-6 for *"I want my money back"* and rerank them with `gemini-2.5-flash` in a **single listwise call**: number the passages, ask for a JSON list of scores 0-10, validate that exactly 6 numbers came back, and sort descending.

<details><summary><b>Hint</b></summary>

Use `generate_json(prompt, temperature=0.0)`. Raise an error if `len(scores) != len(passages)`; in production fall back to the stage-1 order instead of crashing.

</details>

<details><summary><b>Solution</b></summary>

```python
# --------------------------------------------------------------------------
# Q2. Apply: an LLM-as-reranker with gemini-2.5-flash (listwise, one call, JSON scores)
# --------------------------------------------------------------------------
def llm_rerank(query: str, passages: list[str]) -> list[float]:
    numbered = "\n".join(f"[{i}] {p}" for i, p in enumerate(passages))
    prompt = (
        "You are a search relevance judge.\n"
        f"Question: {query}\n\nPassages:\n{numbered}\n\n"
        "Rate how well each passage answers the question from 0 (irrelevant) to 10 (directly answers it). "
        'Return ONLY JSON like {"scores": [<number for [0]>, <number for [1]>, ...]} with exactly '
        f"{len(passages)} numbers, in order."
    )
    data = generate_json(prompt, temperature=0.0)
    scores = data["scores"] if isinstance(data, dict) else data
    if len(scores) != len(passages):                              # never trust the model's list length
        raise ValueError(f"expected {len(passages)} scores, got {len(scores)}")
    return [float(s) for s in scores]


def q2():
    query = "I want my money back"
    D = embed_texts(DOCS, dim=DIM)
    q = embed_query(query, dim=DIM)
    cand = [int(i) for i in np.argsort(-(D @ q))[:6]]                       # stage 1: dense top 6
    print("stage 1 (dense) :", cand)
    scores = llm_rerank(query, [DOCS[i] for i in cand])                      # stage 2: ONE Gemini call
    order = [cand[j] for j in np.argsort(-np.array(scores), kind="stable")]
    print("LLM scores      :", scores)
    print("stage 2 (rerank):", order, "->", DOCS[order[0]][:60])
    return order
```

</details>

**Why it works / what to learn**
Listwise grading uses **one** request instead of six (important on the free tier) and lets the model compare passages against each other. Temperature 0 and strict validation keep it stable. Expect the refund passage (doc 1) to score highest even though it shares no words with the query, which is the kind of semantic judgment a lexical scorer cannot make.

**Expected output (≈, with the real Gemini model)**
```text
stage 1 (dense) : [≈1, ≈4, ...]  (six candidate ids)
LLM scores      : [9.0, 1.0, 0.0, ...]
stage 2 (rerank): [1, ...] -> Refunds are issued to the original payment method within 5 b
```

---

### Q3. Build - a retrieve-then-rerank pipeline with a pluggable reranker

**Problem**

Write `retrieve_then_rerank(query, D, docs, reranker, n_candidates, top_n)`: dense top-N candidates, score with the reranker, sort **descending**, return the top. Evaluate MRR on `LABELED` before and after reranking, using the cross-encoder when available and the stub otherwise.

<details><summary><b>Hint</b></summary>

Keep stage 1's candidate order to compute the "before" MRR. For the "after", map the sorted score indexes back to document ids: `order = [cand[j] for j in np.argsort(-scores)]`.

</details>

<details><summary><b>Solution</b></summary>

```python
# --------------------------------------------------------------------------
# Q3. Build: retrieve -> rerank pipeline with a pluggable reranker, evaluated before/after
# --------------------------------------------------------------------------
def retrieve_then_rerank(query, D, docs, reranker, n_candidates=6, top_n=3, dim=DIM):
    q = embed_query(query, dim=dim)
    cand = [int(i) for i in np.argsort(-(D @ q))[:n_candidates]]             # stage 1: cheap, recall-oriented
    scores = reranker.score(query, [docs[i] for i in cand])                  # stage 2: precise, on few docs
    order = [cand[j] for j in np.argsort(-np.array(scores), kind="stable")]  # DESCENDING: higher = more relevant
    return order[:top_n], cand


def q3():
    D = embed_texts(DOCS, dim=DIM)
    rr = best_available_reranker()
    truth = [t for _, t in LABELED]
    before, after = [], []
    for query, _ in LABELED:
        top, cand = retrieve_then_rerank(query, D, DOCS, rr, n_candidates=6, top_n=6)
        before.append(cand); after.append(top)
    print(f"reranker: {rr.name}")
    print(f"MRR  stage-1 only: {mrr(before, truth)}   after rerank: {mrr(after, truth)}")
    return mrr(before, truth), mrr(after, truth)
```

</details>

**Why it works / what to learn**
Because every reranker implements the same `score()` method, you can swap cross-encoder, LLM or stub without touching the pipeline, and evaluate each with the same harness. With the real cross-encoder expect MRR to rise (typically a clear jump); with the crude lexical stub the effect is small or none, which is itself a lesson: a reranker is only as good as its relevance model.

**Expected output (≈)**
```text
reranker: cross-encoder/ms-marco-MiniLM-L-6-v2      (or: lexical-stub if the model cannot be downloaded)
MRR  stage-1 only: ≈0.7-0.9   after rerank: ≈0.9-1.0   (real cross-encoder)
```

---

### Q4. Debug - four classic reranking mistakes

**Problem**

Reproduce and fix: (1) results sorted ascending so the worst is first; (2) a candidate pool of 1 that cannot contain the answer; (3) cross-encoder logits treated as probabilities with a fixed `> 0.5` threshold; (4) arguments passed as `(passage, query)` instead of `(query, passage)`.

<details><summary><b>Hint</b></summary>

(1) `np.argsort(-scores)`. (3) Logits are unbounded; `sigmoid(x) = 1/(1+e^-x)` maps them to 0-1 but the cut-off still needs calibrating. (4) The model was trained on `(query, passage)` pairs.

</details>

<details><summary><b>Solution</b></summary>

```python
# --------------------------------------------------------------------------
# Q4. Debug: four classic reranking mistakes (pure Python)
# --------------------------------------------------------------------------
def q4():
    query = "payment gateway error"
    cands = [DOCS[1], DOCS[2], DOCS[9]]                       # truth is DOCS[2]
    rr = LexicalStubReranker()
    scores = np.array(rr.score(query, cands))
    print("scores:", np.round(scores, 2), "for", [c[:20] for c in cands])

    # BUG 1: sorting ascending puts the WORST first
    bad = [cands[i][:20] for i in np.argsort(scores)]
    good = [cands[i][:20] for i in np.argsort(-scores)]
    print("bug1 ascending sort :", bad[0], "<- worst first")
    print("fix  descending sort:", good[0])

    # BUG 2: reranking a pool that does not contain the answer (pool of 1 from a weak first stage)
    pool_too_small = [DOCS[1]]
    print("bug2 pool of 1 can only return:", pool_too_small[0][:20], "-> fix: rerank 20-50 candidates")

    # BUG 3: treating cross-encoder LOGITS as probabilities with a fixed 0.5 threshold
    logits = np.array([-3.2, 4.1, -8.0])                       # typical ms-marco logits: any real number
    print("bug3 logits:", logits, "-> `> 0.5` keeps", int((logits > 0.5).sum()), "but they are not probabilities")
    probs = 1 / (1 + np.exp(-logits))
    print("fix  sigmoid(logits) :", np.round(probs, 3), " (and still calibrate a threshold on labelled data)")

    # BUG 4: swapped argument order (passage, query) silently degrades a cross-encoder
    print("bug4 pairs must be (query, passage):", [(query[:12], c[:12]) for c in cands[:1]])
    return good[0]
```

</details>

**Why it works / what to learn**
Most reranking disappointment is plumbing: sort direction, pair order, pool size and score scale. Add unit tests for them: a known-relevant passage must outrank a known-irrelevant one, and the pool must contain the gold chunk in a labelled set.

**Expected output (≈)**
```text
scores: [0.33 1.   0.  ] for ['Refunds are issued t', 'Error E4012 means th', 'Webhooks retry faile']
bug1 ascending sort : Webhooks retry faile <- worst first
fix  descending sort: Error E4012 means th
bug2 pool of 1 can only return: Refunds are issued t -> fix: rerank 20-50 candidates
bug3 logits: [-3.2  4.1 -8. ] -> `> 0.5` keeps 1 but they are not probabilities
fix  sigmoid(logits) : [0.039 0.984 0.   ]  (and still calibrate a threshold on labelled data)
bug4 pairs must be (query, passage): [('payment gate', 'Refunds are ')]
```

---

### Q5. Production - how many candidates should you rerank?

**Problem**

Simulate 300 queries where stage 1's rank for the true chunk is heavy-tailed and the reranker puts the true chunk first 80% of the time *if it is in the pool*. For pools of 5, 10, 20, 50, 100 report recall@pool, MRR after reranking and rerank latency (8 ms per pair). Recommend a pool size.

<details><summary><b>Hint</b></summary>

If `true_rank <= n_cand` the doc is in the pool and can be reranked; otherwise it is lost (contributes 0). Latency is `n_cand × 8` ms.

</details>

<details><summary><b>Solution</b></summary>

```python
# --------------------------------------------------------------------------
# Q5. Production: how many candidates should you rerank? (quality vs latency, simulated)
# --------------------------------------------------------------------------
def q5(n_queries=300, n_docs=500, seed=0):
    rng = np.random.default_rng(seed)
    cost_ms_per_pair = 8.0                                     # CPU cross-encoder, order of magnitude
    results = {}
    for n_cand in (5, 10, 20, 50, 100):
        in_pool, rr_ranks = 0, []
        for _ in range(n_queries):
            # stage-1 (bi-encoder) is noisy: the true doc's rank is heavy-tailed (often 1-5, sometimes 40+)
            true_rank = int(min(n_docs, max(1, rng.lognormal(mean=1.2, sigma=1.1))))
            if true_rank <= n_cand:
                in_pool += 1
                # cross-encoder is far more accurate: true doc usually ends up 1st within the pool
                new_rank = 1 if rng.random() < 0.8 else int(rng.integers(2, 4))
                rr_ranks.append(1 / new_rank)
            else:
                rr_ranks.append(0.0)                           # can never be recovered: not in the pool
        results[n_cand] = (in_pool / n_queries, float(np.mean(rr_ranks)), n_cand * cost_ms_per_pair)

    print(f"{'candidates':>10} {'recall@pool':>12} {'MRR after':>10} {'rerank ms':>10}")
    for n_cand, (rec, m, ms) in results.items():
        print(f"{n_cand:>10} {rec:>12.2f} {m:>10.2f} {ms:>10.0f}")
    return results
```

</details>

**Why it works / what to learn**
This is a *simulation with stated assumptions*, not a benchmark, but it shows the real trade-off: recall@pool and MRR climb steeply then flatten, while latency grows linearly. Choose the smallest pool where the curve flattens (here about 20-50). Repeat the experiment with your real stage 1, your labelled queries and your measured per-pair latency.

**Expected output (≈, seeded simulation)**
```text
candidates  recall@pool  MRR after  rerank ms
         5         0.70       0.59         40
        10         0.87       0.77         80
        20         0.94       0.83        160
        50         0.99       0.88        400
       100         1.00       0.89        800
```
Recommendation: N ≈ 20-50.
