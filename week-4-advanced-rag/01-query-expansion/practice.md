# Practice - Query expansion

Five questions, rising difficulty. Attempt each before opening the hint, and the hint before the solution. Every solution is the exact code in `solutions.py` (`python solutions.py` runs all).

**Offline by default.** Without an API key the file uses a *toy embedder* (hashed, stop-word-free, crudely stemmed words) and a *toy rewriter* (a hand-made synonym table playing the part of `gemini-2.5-flash`). This lets you watch the mechanics (variants → batched embedding → per-variant search → RRF → recall/MRR) with no setup. **The toy rewriter was written for these six queries, so its perfect scores are an artefact;** with real Gemini expect gains of roughly 10-25 recall points on messy queries and ties on clear ones (set `GEMINI_API_KEY` and the same code switches to `gemini-embedding-001` and `generate_json`; numbers then differ from the expected outputs here). `CORPUS` (14 hosting-support chunks), `GOLD` (6 messy queries with the chunk ids that answer them), `search_one`, `recall_at_k`, `reciprocal_rank`, `evaluate` and `baseline` are defined at the top of `solutions.py`.

---

### Q1. Warm-up - Reciprocal Rank Fusion by hand and in code

**Problem**

Three searches returned these ranked id lists: `[3, 7, 1]`, `[7, 3, 9]`, `[7, 5, 3]`. Write `rrf(rank_lists, k=60, top=5)` where `score(d) = Σ 1 / (k + rank)` with ranks starting at 1 (break ties by id). What is the fused ranking? Does changing `k` to 1 or 1000 change the top 3 here?

<details><summary><b>Hint</b></summary>

Doc 7 is rank 2, 1, 1. Doc 3 is rank 1, 2, 3. Compute `1/61 + 1/62 + 1/62` for doc 7 by hand and compare.

</details>

<details><summary><b>Solution</b></summary>

```python
# --- Q1: Reciprocal Rank Fusion from scratch ----------------------------------
def rrf(rank_lists, k=60, top=5):
    """score(d) = sum over lists of 1 / (k + rank), rank starting at 1. Ties broken by id (deterministic)."""
    scores = {}
    for lst in rank_lists:
        for rank, doc_id in enumerate(lst, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda d: (-scores[d], d))[:top], scores


def q1():
    lists = [[3, 7, 1], [7, 3, 9], [7, 5, 3]]
    ranked, scores = rrf(lists, top=5)
    print("fused ranking:", ranked)
    for d in ranked:
        print(f"  doc {d}: {scores[d]:.5f}")
    for k in (1, 60, 1000):
        print(f"k={k:<4} ->", rrf(lists, k=k, top=3)[0])
```

</details>

**Why it works / what to learn**
RRF uses only *ranks*, never raw scores, so lists from different queries (or from BM25 and vectors) can be combined without normalising scores. Documents that several variants agree on rise to the top. The constant `k` softens the advantage of rank 1 over rank 2; here all three settings agree, but with few lists and big rank gaps it can matter. Docs 1 and 9 tie (each appears once at rank 3), and the id breaks the tie.

**Expected output**
```text
fused ranking: [7, 3, 5, 1, 9]
  doc 7: 0.04892
  doc 3: 0.04840
  doc 5: 0.01613
  doc 1: 0.01587
  doc 9: 0.01587
k=1    -> [7, 3, 5]
k=60   -> [7, 3, 5]
k=1000 -> [7, 3, 5]
```

---

### Q2. Apply - measure the baseline first

**Problem**

Before improving retrieval, measure it. Using the gold set, compute `recall@3` and MRR for the plain one-query-one-vector baseline and print recall per query.

<details><summary><b>Hint</b></summary>

`recall@k = |top-k ∩ relevant| / |relevant|`. MRR is the mean over queries of `1 / rank of the first relevant result` (0 if none). `evaluate(strategy)` loops over `GOLD`.

</details>

<details><summary><b>Solution</b></summary>

```python
# --- Q2: the baseline - one query, one vector, metrics ------------------------
def q2():
    res = evaluate(baseline, k=3)
    print(f"mode: {MODE}")
    print(f"baseline recall@3={res['recall@k']:.2f} mrr={res['mrr']:.2f}")
    for q, r in res["per_query"].items():
        print(f"  recall@3={r:.2f}  {q}")
```

</details>

**Why it works / what to learn**
Messy user phrasing ("my website feels sluggish", "money back") shares few words with the documentation ("TTFB", "refunded pro rata"), the *vocabulary gap*, so the baseline misses four of six queries. Without this number you cannot tell whether any expansion technique helped. Notice `search_one` drops chunks whose score is about zero: they are not matches, and returning them would pollute later fusion.

**Expected output** (offline toys)
```text
mode: offline toys
baseline recall@3=0.33 mrr=0.33
  recall@3=0.00  my website feels sluggish
  recall@3=1.00  site down with gateway error
  recall@3=0.00  how do I get my money back
  recall@3=0.00  stop my site crashing when traffic spikes
  recall@3=0.00  browser shows a not-secure padlock warning
  recall@3=1.00  can I make my team log in more safely
```

---

### Q3. Build - multi-query retrieval with guard rails

**Problem**

Build `multi_query(query)`: ask the rewriter for variants, apply `clean_variants` (keep the **original first** as an anchor, drop duplicates after normalising, cap the count), embed all queries in **one batch**, search with each, and fuse with RRF. Compare against the baseline per query and label each WIN / LOSS / tie.

<details><summary><b>Hint</b></summary>

`clean_variants(original, variants)` keeps a `seen` set of normalised strings (lower-case, punctuation removed). The LLM suggests; your code enforces uniqueness, the anchor and the cap.

</details>

<details><summary><b>Solution</b></summary>

```python
# --- Q3: multi-query retrieval with the 'suggestion box' rules ----------------
def normalize(s):
    return re.sub(r"\W+", " ", s.lower()).strip()


def clean_variants(original, variants, max_variants=5):
    """Original first (the anchor), then unique variants, capped. The LLM suggests; we enforce."""
    seen, out = {normalize(original)}, [original]
    for v in variants:
        key = normalize(v)
        if key and key not in seen:
            seen.add(key)
            out.append(v.strip())
    return out[: max_variants + 1]


def multi_query(query, expand=None, n=4, per_query_k=5, top=5):
    expand = expand or EXPAND
    queries = clean_variants(query, expand(query, n))
    vecs = EMBED_QUERIES(queries)                          # ONE batched embedding call for all variants
    return rrf([search_one(v, per_query_k) for v in vecs], top=top)[0]


def q3():
    base, multi = evaluate(baseline), evaluate(multi_query)
    print(f"baseline    recall@3={base['recall@k']:.2f} mrr={base['mrr']:.2f}")
    print(f"multi-query recall@3={multi['recall@k']:.2f} mrr={multi['mrr']:.2f}")
    for q in GOLD:
        b, m = base["per_query"][q], multi["per_query"][q]
        print(f"  {'WIN ' if m > b else 'LOSS' if m < b else 'tie '} {b:.2f} -> {m:.2f}  {q}")
```

</details>

**Why it works / what to learn**
Each variant uses different vocabulary, so at least one lands near the right chunk; RRF rewards chunks that several variants retrieve. Keeping the original query protects the queries that already worked (the two ties). One batched embedding call keeps latency and free-tier quota under control. Cap the number of variants: more is not better, since each adds cost and noise.

**Expected output** (offline toys; real Gemini gains are smaller)
```text
baseline    recall@3=0.33 mrr=0.33
multi-query recall@3=1.00 mrr=1.00
  WIN  0.00 -> 1.00  my website feels sluggish
  tie  1.00 -> 1.00  site down with gateway error
  WIN  0.00 -> 1.00  how do I get my money back
  WIN  0.00 -> 1.00  stop my site crashing when traffic spikes
  WIN  0.00 -> 1.00  browser shows a not-secure padlock warning
  tie  1.00 -> 1.00  can I make my team log in more safely
```

---

### Q4. Debug - "expansion made retrieval worse"

**Problem**

After switching to a more "creative" rewriter (high temperature) your recall and MRR *dropped* below the baseline. Simulate it with `drifting_expand`, which returns off-topic variants (pricing, tickets, domains). Explain why this hurts, then fix it with (1) an **anchor weight** in a weighted RRF so the original query counts more and (2) a **drift filter** that drops variants whose embedding is unrelated to the original question.

<details><summary><b>Hint</b></summary>

In plain RRF every variant list has the same vote as the original. If three of four lists are off-topic, they outvote it. Weighted RRF: `score += weight / (k + rank)`. The drift filter compares `original_vec @ variant_vec` with a minimum similarity.

</details>

<details><summary><b>Solution</b></summary>

```python
# --- Q4: expansion made things worse - add an anchor weight and a drift filter -
def drifting_expand(query, n=4):
    """A bad rewriter: invents topics (what a high-temperature LLM sometimes does)."""
    return ["pricing plans and billing invoices", "support ticket response time", "custom domain cname record"]


def weighted_rrf(rank_lists, weights, k=60, top=5):
    scores = {}
    for lst, w in zip(rank_lists, weights):
        for rank, d in enumerate(lst, 1):
            scores[d] = scores.get(d, 0.0) + w / (k + rank)
    return sorted(scores, key=lambda d: (-scores[d], d))[:top]


def drift_filter(original_vec, variants, variant_vecs, min_sim=0.15):
    """Drop variants whose embedding is almost unrelated to the original question."""
    return [(v, vec) for v, vec in zip(variants, variant_vecs) if float(original_vec @ vec) >= min_sim]


def safe_multi_query(query, expand, anchor_weight=3.0, min_sim=0.0, per_query_k=5, top=5):
    variants = clean_variants(query, expand(query))[1:]
    qv = EMBED_QUERIES([query])[0]
    vv = EMBED_QUERIES(variants) if variants else []
    kept = drift_filter(qv, variants, vv, min_sim) if min_sim > 0 else list(zip(variants, vv))
    lists = [search_one(qv, per_query_k)] + [search_one(vec, per_query_k) for _, vec in kept]
    weights = [anchor_weight] + [1.0] * len(kept)
    return weighted_rrf(lists, weights, top=top), [v for v, _ in kept]


def q4():
    print("baseline          :", {k: round(v, 2) for k, v in evaluate(baseline).items() if k != "per_query"})
    naive = evaluate(lambda q: multi_query(q, expand=drifting_expand))
    print("naive + drifting  :", {k: round(v, 2) for k, v in naive.items() if k != "per_query"})
    fixed = evaluate(lambda q: safe_multi_query(q, drifting_expand, anchor_weight=3.0, min_sim=0.05)[0])
    print("anchor + filter   :", {k: round(v, 2) for k, v in fixed.items() if k != "per_query"})
    _, kept = safe_multi_query("how do I get my money back", drifting_expand, min_sim=0.05)
    print("variants kept for 'how do I get my money back':", kept)
```

</details>

**Why it works / what to learn**
Query expansion can *add* wrong evidence. Off-topic variants retrieve off-topic chunks and RRF happily fuses them. Guard rails: always include the original, give it extra weight, filter variants by similarity to the original, cap the count, use low-to-moderate temperature, and *measure* against the baseline on your gold set. Here the guarded version recovers to baseline level (recall 0.33, MRR slightly higher) instead of collapsing to 0.25 / 0.14. (With the real model, tune `min_sim` on your own data: the similarity scale differs.)

**Expected output** (offline toys)
```text
baseline          : {'recall@k': 0.33, 'mrr': 0.33}
naive + drifting  : {'recall@k': 0.25, 'mrr': 0.14}
anchor + filter   : {'recall@k': 0.33, 'mrr': 0.38}
variants kept for 'how do I get my money back': []
```

---

### Q5. Production - cached expansions, routing and cost

**Problem**

Expansion costs an LLM call per query. Build an `ExpansionService` that caches expansions on disk keyed by `sha256(n|query)` and counts real LLM calls. Add a cheap `plan(query)` router that decomposes multi-part questions ("... and ...") and uses multi-query otherwise. Run all gold queries twice, the second time from a *new* service instance that only has the cache file, and report calls and result equality.

<details><summary><b>Hint</b></summary>

Check `key in self.cache` before calling the LLM and write the file after each new entry. A fresh instance simulates a restarted server.

</details>

<details><summary><b>Solution</b></summary>

```python
# --- Q5: cached expansion, strategy routing and a cost report -----------------
class ExpansionService:
    def __init__(self, cache_path):
        self.path = pathlib.Path(cache_path)
        self.cache = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.llm_calls = 0

    def expand(self, query, n=4):
        key = hashlib.sha256(f"{n}|{query}".encode()).hexdigest()
        if key not in self.cache:
            self.llm_calls += 1                                   # a real (paid-in-quota) call
            self.cache[key] = EXPAND(query, n)
            self.path.write_text(json.dumps(self.cache))
        return self.cache[key]


def plan(query):
    """Cheap routing: multi-part questions are decomposed; everything else gets multi-query."""
    parts = [p.strip() for p in re.split(r"\band\b|\?|;", query) if len(p.split()) >= 3]
    return ("decompose", parts) if len(parts) >= 2 else ("multi_query", [query])


def run_all(service):
    out = {}
    for q in GOLD:
        strategy, parts = plan(q)
        queries = clean_variants(q, service.expand(q) if strategy == "multi_query" else parts)
        out[q] = rrf([search_one(v, 5) for v in EMBED_QUERIES(queries)], top=5)[0]
    return out


def q5():
    with tempfile.TemporaryDirectory() as d:
        cache = pathlib.Path(d) / "expansions.json"
        s1 = ExpansionService(cache)
        r1 = run_all(s1)
        s2 = ExpansionService(cache)                              # a fresh process: reads the cache file
        r2 = run_all(s2)
        print("run 1 LLM calls:", s1.llm_calls, "| run 2 LLM calls:", s2.llm_calls, "| same results:", r1 == r2)
    print("routing:", {q: plan(q)[0] for q in ["why is my site slow and how do I add a domain", "how do I get my money back"]})
    print("cost per query ≈ 1 LLM call + 1 batched embed call of", 1 + 4, "texts, paid once thanks to the cache")
```

</details>

**Why it works / what to learn**
The first run pays 6 LLM calls (one per query); the second run pays 0 and returns identical results. Cache every deterministic expensive step (expansions, embeddings), and consider routing: clean single-topic questions often need no expansion at all, so classify first and expand only when the baseline looks weak (for example low top-1 score). On the free tier, fewer calls means fewer 429 errors.

**Expected output**
```text
run 1 LLM calls: 6 | run 2 LLM calls: 0 | same results: True
routing: {'why is my site slow and how do I add a domain': 'decompose', 'how do I get my money back': 'multi_query'}
cost per query ≈ 1 LLM call + 1 batched embed call of 5 texts, paid once thanks to the cache
```
