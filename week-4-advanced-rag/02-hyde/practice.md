# Practice - HyDE (Hypothetical Document Embeddings)

Five questions, rising difficulty. Attempt each before opening the hint, and the hint before the solution. Every solution is the exact code in `solutions.py` (`python solutions.py` runs all).

**Offline by default.** Without an API key the file uses a *toy embedder* (hashed words) and a *toy answer writer* (a lookup table playing `gemini-2.5-flash`). **The toy writer was hand-written for these queries, so its perfect scores are an artefact**; with real Gemini, HyDE typically gives modest gains on vague queries and can lose on exact-match queries. Set `GEMINI_API_KEY` and the same code uses `gemini-embedding-001` and `gemini-2.5-flash` (numbers then differ). `CORPUS` (12 Laravel doc chunks), `GOLD` (9 user queries with their answering chunk), `search_vec`, `unit`, `evaluate` and `baseline` are defined at the top of `solutions.py`.

---

### Q1. Warm-up - see the question/answer gap

**Problem**

For four vague queries, compute the cosine similarity between (a) the *question* embedding and its gold chunk, and (b) the embedding of a *hypothetical answer* and the same chunk. Which is higher, and why does that matter for retrieval?

<details><summary><b>Hint</b></summary>

Embed the question with the query task type and the hypothesis with the **document** task type. Vectors are unit length, so `a @ b` is the cosine.

</details>

<details><summary><b>Solution</b></summary>

```python
# --- Q1: questions and answers live in different places ------------------------
def q1():
    print(f"mode: {MODE}")
    print(f"{'query':<52} sim(query, gold)  sim(hypothetical, gold)")
    for q, rel in list(GOLD.items())[:4]:
        gold_vec = DOC_VECS[next(iter(rel))]
        qv = EMBED_QUERIES([q])[0]
        hv = EMBED_DOCS(HYPOTHETICAL(q))[0]
        print(f"{q:<52} {float(qv @ gold_vec):>10.2f} {float(hv @ gold_vec):>17.2f}")
```

</details>

**Why it works / what to learn**
The user's wording shares almost no vocabulary with the documentation, so the question lands far from its answer. An answer-shaped text uses the documentation's words ("eager loading", "queue worker", "throttle") and lands close. (The toy embedder exaggerates the gap; real models show a smaller but real one.)

**Expected output** (offline toys)
```text
mode: offline toys
query                                                sim(query, gold)  sim(hypothetical, gold)
my page makes hundreds of database calls                   0.10              0.68
run something later without making the user wait           0.08              0.69
stop people from hammering my API                          0.00              0.79
keep passwords and keys out of git, different per server       0.07              0.69
```

---

### Q2. Apply - HyDE retrieval vs the baseline

**Problem**

Implement `hyde_search(query)`: generate one hypothetical answer, embed it with the **document** task type, and search. Compare recall@3 and MRR with the plain-query baseline on all nine gold queries, and print one hypothetical answer.

<details><summary><b>Hint</b></summary>

Replace `embed(query)` with `embed(HYPOTHETICAL(query))` and keep everything else the same. Evaluate both strategies with `evaluate(strategy)`.

</details>

<details><summary><b>Solution</b></summary>

```python
# --- Q2: HyDE retrieval: embed a hypothetical ANSWER instead of the question ---
def hyde_search(query, k=5):
    hypo = HYPOTHETICAL(query, n=1)[0]
    vec = EMBED_DOCS([hypo])[0]                  # DOCUMENT task type: it should look like a document, not a question
    return search_vec(vec, k)


def q2():
    print("baseline  recall@3, mrr:", evaluate(baseline))
    print("HyDE      recall@3, mrr:", evaluate(hyde_search))
    print("example:", GOLD and list(GOLD)[0])
    print("  hypothetical answer:", HYPOTHETICAL(list(GOLD)[0])[0])
```

</details>

**Why it works / what to learn**
Only one line of the pipeline changed, yet retrieval changed completely. The hypothesis is never shown to the user; it is only a better search key. Eight of nine queries gain here; the ninth was already well phrased (Q4 and Q5 skip HyDE for it).

**Expected output** (offline toys)
```text
baseline  recall@3, mrr: (0.44, 0.39)
HyDE      recall@3, mrr: (1.0, 1.0)
example: my page makes hundreds of database calls
  hypothetical answer: Load related models with eager loading using the with method in Eloquent, so relationships come in one query and the N+1 query problem disappears.
```

---

### Q3. Build - blend the query with several hypotheses

**Problem**

Write `hyde_blend(query, alpha, n)` that embeds `n` hypotheses, averages them, mixes with the question vector as `alpha * q + (1 - alpha) * mean(h)`, re-normalises, and searches. Sweep `alpha` over 0, 0.3, 0.5, 0.8 and 1.0 and explain the curve.

<details><summary><b>Hint</b></summary>

`alpha = 1` is the plain baseline, `alpha = 0` is pure HyDE. A blend of two unit vectors is not unit length, so divide by its norm.

</details>

<details><summary><b>Solution</b></summary>

```python
# --- Q3: several hypotheses + blending with the original query ------------------
def hyde_blend(query, alpha=0.3, n=3, k=5):
    """vec = normalise(alpha * query_vec + (1 - alpha) * mean(hypothesis_vecs))"""
    qv = EMBED_QUERIES([query])[0]
    hv = EMBED_DOCS(HYPOTHETICAL(query, n=n))
    return search_vec(unit(alpha * qv + (1 - alpha) * hv.mean(axis=0)), k)


def q3():
    print("alpha  recall@3  mrr     (alpha = weight of the ORIGINAL query vector)")
    for a in (0.0, 0.3, 0.5, 0.8, 1.0):
        r, m = evaluate(lambda q, a=a: hyde_blend(q, alpha=a))
        print(f"{a:>4}   {r:>6}   {m:>5}")
```

</details>

**Why it works / what to learn**
At `alpha=1` you get the baseline (0.44); a little hypothesis already gives large gains; too much weight on the question brings the vocabulary gap back (MRR dips at 0.8). In practice choose a moderate alpha (0.2-0.5) so literal tokens from the question (names, numbers) still count, and tune it on your gold set rather than copying a number.

**Expected output** (offline toys)
```text
alpha  recall@3  mrr     (alpha = weight of the ORIGINAL query vector)
 0.0      1.0     1.0
 0.3      1.0     1.0
 0.5      1.0     1.0
 0.8      1.0    0.94
 1.0     0.44    0.39
```

---

### Q4. Debug - "HyDE made things worse"

**Problem**

With a hallucinating writer (it confidently answers about Django when asked about Laravel), pure HyDE scores *below* the baseline. Reproduce it, then fix it: (1) blend the question vector back in, and (2) add a **confidence gate** that skips HyDE when the plain query already matches strongly. Show the effect with both a bad and a good writer.

<details><summary><b>Hint</b></summary>

Gate: `top1 = max(DOC_VECS @ qv)`; if `top1 >= threshold`, return the plain search. Otherwise embed the hypothesis and search with `unit(alpha*qv + (1-alpha)*hv)`.

</details>

<details><summary><b>Solution</b></summary>

```python
# --- Q4: HyDE hurt retrieval - a hallucinated answer; add a confidence gate ------
def bad_hypothetical(query, n=1):
    """A hallucinating writer: confident answer about the WRONG framework."""
    return ["In Django, define a custom management command and register it with the manage.py utility and settings module."] * n


def hyde_with_gate(query, writer, threshold=0.25, alpha=0.5, k=5):
    """Use HyDE only when the plain query looks weak (low top-1 score); blend so the question still votes."""
    qv = EMBED_QUERIES([query])[0]
    top1 = float((DOC_VECS @ qv).max())
    if top1 >= threshold:
        return search_vec(qv, k)                                  # query already matches well: skip the LLM
    hv = EMBED_DOCS(writer(query, 1))[0]
    return search_vec(unit(alpha * qv + (1 - alpha) * hv), k)


def q4():
    print("baseline                 :", evaluate(baseline))
    bad = evaluate(lambda q: search_vec(EMBED_DOCS(bad_hypothetical(q))[0]))
    print("pure HyDE, bad writer    :", bad)
    guarded = evaluate(lambda q: hyde_with_gate(q, bad_hypothetical, threshold=0.0, alpha=0.5))
    print("blend(0.5), bad writer   :", guarded)
    good = evaluate(lambda q: hyde_with_gate(q, toy_hypothetical, threshold=0.45, alpha=0.3))
    print("gate+blend, good writer  :", good)
```

</details>

**Why it works / what to learn**
A pure HyDE vector follows the hallucination (recall 0.33 < baseline 0.44). Blending keeps the question's evidence and restores baseline performance even with a terrible writer, so the *worst case is bounded*. With a good writer, the gated blend keeps the full gain. Never ship HyDE without a safety net, and never evaluate it on a single example.

**Expected output** (offline toys)
```text
baseline                 : (0.44, 0.39)
pure HyDE, bad writer    : (0.33, 0.23)
blend(0.5), bad writer   : (0.44, 0.39)
gate+blend, good writer  : (1.0, 1.0)
```

---

### Q5. Production - gate, cache and fail safe

**Problem**

Build `HydeRetriever` that (1) skips HyDE when the plain query is confident, (2) caches hypotheses per query, (3) counts LLM calls, and (4) falls back to plain search when the LLM raises an exception. Run all queries twice, then run again with a "flaky" writer that times out on one query.

<details><summary><b>Hint</b></summary>

Wrap only the LLM+embed step in `try/except Exception`, increment a `fallbacks` counter in the handler, and `return search_vec(qv, k)` there.

</details>

<details><summary><b>Solution</b></summary>

```python
# --- Q5: a production policy: gate, cache, fall back on LLM failure -------------
class HydeRetriever:
    def __init__(self, writer, threshold=0.45, alpha=0.3):
        self.writer, self.threshold, self.alpha = writer, threshold, alpha
        self.cache, self.llm_calls, self.fallbacks = {}, 0, 0

    def hypothesis(self, q):
        if q not in self.cache:
            self.llm_calls += 1
            self.cache[q] = self.writer(q, 1)[0]
        return self.cache[q]

    def retrieve(self, q, k=5):
        qv = EMBED_QUERIES([q])[0]
        if float((DOC_VECS @ qv).max()) >= self.threshold:
            return search_vec(qv, k)                               # confident: no LLM call
        try:
            hv = EMBED_DOCS([self.hypothesis(q)])[0]
        except Exception:                                         # 429 / timeout / blocked: degrade gracefully
            self.fallbacks += 1
            return search_vec(qv, k)
        return search_vec(unit(self.alpha * qv + (1 - self.alpha) * hv), k)


def flaky_writer(query, n=1):
    if "night" in query:
        raise TimeoutError("simulated LLM timeout")
    return toy_hypothetical(query, n)


def q5():
    r = HydeRetriever(toy_hypothetical)
    print("normal  : recall@3, mrr =", evaluate(r.retrieve), "| LLM calls:", r.llm_calls, "of", len(GOLD), "queries")
    evaluate(r.retrieve)
    print("2nd pass: LLM calls total:", r.llm_calls, "(cached)")
    f = HydeRetriever(flaky_writer)
    print("flaky   : recall@3, mrr =", evaluate(f.retrieve), "| fallbacks:", f.fallbacks)
```

</details>

**Why it works / what to learn**
The confident query ("php artisan migrate...") never triggers an LLM call (8 of 9 queries use one), the second pass costs nothing (cache), and a timeout degrades to plain search instead of failing the request. Track `llm_calls` and `fallbacks` as metrics: a rising fallback rate means rate limiting or a prompt that triggers safety blocks.

**Expected output** (offline toys)
```text
normal  : recall@3, mrr = (1.0, 1.0) | LLM calls: 8 of 9 queries
2nd pass: LLM calls total: 8 (cached)
flaky   : recall@3, mrr = (1.0, 1.0) | fallbacks: 1
```
