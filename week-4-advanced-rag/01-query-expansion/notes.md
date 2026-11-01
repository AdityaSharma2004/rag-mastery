# 01 - Query Expansion: Ask the Question Five Ways

## 🎯 What you will learn
- You will be able to explain *why* a perfectly good vector search still misses relevant chunks (the "vocabulary gap") and say when query expansion is the right fix.
- You will be able to turn one user query into 3-5 useful variations with `gemini-2.5-flash`, using structured JSON output and a prompt that actually produces *different* queries.
- You will be able to fuse several ranked result lists into one with **Reciprocal Rank Fusion (RRF)** and compute it by hand.
- You will be able to implement the **multi-query**, **step-back** and **decomposition** variants and know which one fits which failure.
- You will be able to measure the recall gain with `recall@k` and `MRR`, and count the extra LLM calls, embeddings and seconds you paid for it on the free tier.
- You will be able to dedupe, cap and cache variations so the technique stays cheap and safe in production.

## 🧠 The big idea in plain words

A user types: **"my website feels sluggish"**.

Your knowledge base has a chunk titled: *"Time to First Byte (TTFB) above 600 ms usually means the origin server is slow..."*. That chunk is the answer. But the user never wrote "TTFB", "origin" or "latency". They wrote "sluggish". The embedding model is smart, yet it is being asked to match a *casual complaint* against a *technical statement*. Sometimes it succeeds, sometimes the right chunk lands at rank 9 and your top-3 never sees it.

Humans fix this by rephrasing: "site slow", "high latency", "page load time", "server response time". Each rephrasing "knocks on a different door" of the vector space. **Query expansion** automates this: ask an LLM to produce several rephrasings, search with each, and merge the results.

**Laravel/MySQL analogy.** Think of a MySQL `FULLTEXT` search or Laravel Scout with Meilisearch/Algolia. Those engines let you configure a *synonyms list* (`sluggish -> slow, latency`) so one search term matches more rows. Query expansion is the same idea, except the synonym list is generated on the fly by an LLM per query instead of being hand-maintained. And "merge the result lists" is like running several `SELECT ... ORDER BY score LIMIT 5` queries and then doing a smart `UNION` that rewards rows appearing near the top of multiple queries.

The price: **one extra LLM call and a few extra embeddings per user question**. The gain: higher recall (the right chunk is more likely *somewhere* in your candidate set). That trade-off is the whole topic.

## 📚 Concepts, step by step

### 1. Recall vs precision (define before using)
- **Retrieval** = given a query, return the k chunks most likely to contain the answer.
- **Recall@k** = of all chunks that *should* be found, what fraction appear in the top k? If 2 chunks are relevant and 1 is in your top 3, recall@3 = 0.5.
- **Precision@k** = of the k chunks you returned, what fraction are relevant?
- **MRR (Mean Reciprocal Rank)** = for each query, take 1 / (rank of the first relevant chunk), then average. First hit at rank 1 gives 1.0, rank 2 gives 0.5, rank 4 gives 0.25, a miss gives 0.

Query expansion mostly buys **recall**. Reranking (week 3) mostly buys **precision**. A top-tier pipeline uses both: expand to cast a wide net, rerank to pick the best fish.

### 2. Why a single query fails: the vocabulary gap
An embedding places text in a 768-dimensional space (our default) where similar *meaning* is close. In theory "sluggish" and "high latency" are close. In practice:

1. **Query/document style mismatch.** Queries are short and informal; chunks are long and formal.
2. **Under-specified queries.** "How do I get my money back" never mentions "refund", "pro rata" or "cancel subscription".
3. **Multi-faceted queries.** "Stop my site crashing when traffic spikes" needs *autoscaling* AND *database connection limits*. One vector can only sit in one place.
4. **Over-specific queries.** "Why does my Laravel app on plan Starter return 502 after deploy?" is so specific that no chunk matches the whole thing; a more *general* query ("what causes 502 Bad Gateway") would match.

Each failure suggests a different expansion flavour:

| Failure | Technique | One-line idea |
|---|---|---|
| Vocabulary gap / ambiguity | **Multi-query** | Paraphrase the query 3-5 ways |
| Query too specific | **Step-back** | Also ask the more general question |
| Several facets in one question | **Decomposition** | Split into sub-questions, search each |
| Query too short / answer-shaped gap | **HyDE** (next topic) | Search with an imagined answer |

### 3. Multi-query expansion
The flow:

```
user query
    |
    v
[LLM] -> 4 rewrites            (1 generate call)
    |
    v
original + 4 rewrites = 5 queries
    |
    v
[embed all 5 in ONE batched call]   (1 embed call, 5 vectors)
    |
    v
5 x vector search -> 5 ranked lists   (local numpy: milliseconds)
    |
    v
RRF fusion -> one ranked list
    |
    v
top-k chunks -> LLM answer
```

Important design rules, each learned from real failures:
- **Always keep the original query** in the set. The LLM may drift; the original is your anchor.
- Ask for variations that differ in *vocabulary and angle*, not just word order. A prompt that says "rephrase" gets near-duplicates. A prompt that says "use different technical vocabulary, one as a symptom, one as a likely cause" gets diversity.
- Use **structured JSON output** so you never regex-parse a numbered list. (`generate_json` from our helper.)
- Embed queries with `task_type="RETRIEVAL_QUERY"`. The chunks were embedded as `RETRIEVAL_DOCUMENT`; mixing this up silently hurts quality.
- 3-5 variations is the sweet spot. Beyond ~5 you add cost and noise faster than recall.

### 4. Reciprocal Rank Fusion (RRF), from scratch
You now have N ranked lists of chunk ids. How do you merge them? You could add up cosine scores, but scores from different queries are **not comparable** (one query may produce generally higher similarities than another). Ranks *are* comparable.

RRF gives each chunk, for each list it appears in, the score `1 / (k + rank)` where `rank` starts at 1 and `k` is a smoothing constant (60 is the classic default from the original paper; it works fine and you rarely tune it). Then sum across lists.

By hand. Three lists:

```
L1 (query A): [d3, d7, d1]
L2 (query B): [d7, d3, d9]
L3 (query C): [d7, d5, d3]
```
With k = 60:
- d7: 1/(60+2) + 1/(60+1) + 1/(60+1) = 0.01613 + 0.01639 + 0.01639 = **0.04891**
- d3: 1/61 + 1/62 + 1/63 = 0.01639 + 0.01613 + 0.01587 = **0.04839**
- d1: 1/63 = 0.01587
- d9: 1/63 = 0.01587
- d5: 1/62 = 0.01613

Fused order: d7, d3, d5, d1/d9 (tie). d7 wins because it is near the top *everywhere*, which is exactly the signal "many phrasings agree this is relevant".

Why `k = 60`? It dampens the difference between rank 1 and rank 2. With small k, rank 1 dominates; with large k, all ranks count almost equally. 60 is a robust middle.

RRF is also how **hybrid search** (BM25 + vectors) is fused in most production systems. You are learning one tool that you will reuse.

### 5. Step-back prompting
Prompt the LLM: "Write a more general question that covers the *background knowledge* needed to answer this." Example:

- Original: *"Why do I get 'Too many connections' on the Starter plan after enabling autoscaling?"*
- Step-back: *"How are database connections limited and shared across instances?"*

You then search with both the original and the step-back (usually 1 extra query, not 4). It shines for over-specific questions and "why" questions where the answer lives in a conceptual paragraph.

### 6. Decomposition (sub-questions)
Prompt: "If the question has several distinct parts, split it into independent sub-questions; otherwise return it unchanged."

- Original: *"Stop my site crashing when traffic spikes"*
- Sub-questions: *"How does autoscaling trigger?"*, *"What are the database connection limits?"*, *"How do I restart PHP-FPM after a crash?"*

Each sub-question is searched separately and the lists are merged. Two flavours exist:
- **Parallel decomposition** (what we build): sub-questions are independent, so searches run together and merge with RRF.
- **Sequential / multi-hop** ("who founded the company that makes X?"): sub-question 2 depends on the answer to 1. This needs an agent loop (LLM answers step 1, then queries step 2). Mention it, don't confuse it with parallel decomposition.

### 7. Dedupe and caps
Three layers of dedupe:
1. **Query dedupe**: LLMs often return "site slow" and "Site slow." Normalise (lowercase, strip punctuation) and drop repeats. Optionally drop variants whose embedding cosine to an existing variant is > ~0.97: you pay for a search that cannot change the result.
2. **Result dedupe**: RRF dedupes by design (a chunk id appears once in the output, with a combined score).
3. **Chunk-text dedupe**: if your corpus has duplicated text under different ids (re-ingested files), dedupe by a hash of the text *before* sending to the LLM.

Cap the number of variants (e.g. 5) regardless of what the LLM returns. The model is a *suggestion box*; your code enforces limits. Same mindset as validating `$request->input()` in Laravel.

### 8. Cost and latency on the free tier
Count every call:

| Item | Without expansion | With multi-query (4 variants) |
|---|---|---|
| LLM calls for retrieval | 0 | **1** (expansion) |
| Embedding calls | 1 (1 text) | **1** (5 texts, batched) - same number of *requests*, 5x the *tokens* |
| Vector searches | 1 | 5 (local numpy, ~ms; in Qdrant/pgvector, 5 round-trips, still ms) |
| Added latency | - | ≈ 0.5-2 s (the LLM call dominates) |

The key trick: **embed all variants in one `embed_texts` call**. Free-tier limits are per *request* (RPM/RPD) and per tokens; batching makes N texts cost one request. Our helper batches up to 50 texts per call and caches embeddings on disk, so repeat queries are free.

Also: the expansion call itself is **cacheable** (same question, same expansions). An in-memory dict or Redis `Cache::remember()` equivalent saves the LLM request on repeats. We build a tiny JSON-file cache below. Never quote exact free-tier numbers to colleagues; they change. Check Google AI Studio's rate-limit page.

Set `thinking_budget=0` (the helper default) for the expansion call. Thinking tokens add seconds and gain nothing for rewriting.

### 9. How to measure the recall gain (the only honest way)
Intuition is not evidence. You need a **golden set**: queries + the ids of chunks that answer them (topic 04 shows how to generate one). Then:

1. Run the baseline (single query) and compute recall@k and MRR.
2. Run the expansion strategy, same corpus, same k.
3. Compare **per query** (wins / ties / losses), not just the average. Expansion usually helps messy queries and does nothing, or slightly hurts, on already clear queries.

Beware: 6-10 queries is a smoke test, not proof. Aim for 30-100 for decisions.

## 💻 Code walkthrough

We use a tiny "Acme Cloud" hosting-support knowledge base (14 chunks) with 6 messy user queries and the chunk ids that answer them. Save everything below, in order, as one file `expansion_demo.py` inside this folder. Needs `GEMINI_API_KEY` in `.env`.

### Stage 1 - setup, corpus, gold set, index

```python
import sys, pathlib, json, hashlib, re, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))
from gemini_utils import embed_texts, generate_json
import numpy as np

CORPUS = [
    "Time to First Byte (TTFB) above 600 ms usually means the origin server is slow; enable server-side caching and check PHP-FPM worker saturation.",  # 0
    "To reduce latency for visitors in Asia, attach the CDN add-on and select the Singapore edge region.",                                              # 1
    "Invoices are generated on the 1st of each month and can be downloaded as PDF from Billing > Invoices.",                                           # 2
    "A 502 Bad Gateway error means the upstream PHP-FPM process crashed or timed out; restart the service from the Control Panel.",                    # 3
    "Daily automatic backups are kept for 14 days; restore a backup from Backups > Restore, which takes about 10 minutes.",                            # 4
    "To add a custom domain, create a CNAME record pointing to sites.acmecloud.example and then verify ownership in Domains.",                         # 5
    "Two-factor authentication can be enforced for all team members under Security > Authentication.",                                                 # 6
    "Autoscaling adds an instance when average CPU stays above 70% for five minutes; the maximum is set per plan.",                                    # 7
    "Database connections are capped at 150 per plan; exceeding this causes 'Too many connections' errors in MySQL.",                                  # 8
    "You can cancel your subscription anytime; unused credit is refunded pro rata within 7 business days.",                                            # 9
    "Free SSL certificates are issued by Let's Encrypt and renew automatically 30 days before expiry.",                                                # 10
    "Slow MySQL queries are logged when they exceed 2 seconds; view them under Databases > Slow Query Log.",                                           # 11
    "Object storage buckets are private by default; make a file public with a signed URL valid for up to 7 days.",                                     # 12
    "Support tickets are answered within 4 hours on Business plans and 24 hours on Starter plans.",                                                    # 13
]

# query -> set of chunk ids that answer it (our "golden set")
GOLD = {
    "my website feels sluggish": {0, 1, 11},
    "site down with gateway error": {3},
    "how do I get my money back": {9},
    "stop my site crashing when traffic spikes": {7, 8},
    "browser shows a not-secure padlock warning": {10},
    "can I make my team log in more safely": {6},
}

# Embed the chunks ONCE (RETRIEVAL_DOCUMENT is the default task type). Shape (14, 768), unit vectors.
DOC_VECS = embed_texts(CORPUS, task_type="RETRIEVAL_DOCUMENT")


def search_one(query_vec: np.ndarray, k: int = 5) -> list[int]:
    """Vectors are unit length, so dot product == cosine similarity."""
    scores = DOC_VECS @ query_vec
    return [int(i) for i in np.argsort(-scores)[:k]]
```

Line notes: `DOC_VECS @ query_vec` is a matrix-vector product: 14 dot products in one go (the same math `top_k` in our helper does, minus the sorting). We return **ids**, not text, because ids are what RRF and evaluation need.

### Stage 2 - metrics and the baseline

```python
def recall_at_k(ranked: list[int], relevant: set[int], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranked: list[int], relevant: set[int]) -> float:
    for pos, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            return 1.0 / pos
    return 0.0


def baseline(query: str, k: int = 5) -> list[int]:
    qv = embed_texts([query], task_type="RETRIEVAL_QUERY")[0]
    return search_one(qv, k)


def evaluate(strategy, k: int = 3) -> dict:
    """strategy(query) -> ranked list of chunk ids."""
    recalls, rrs, per_query = [], [], {}
    for q, rel in GOLD.items():
        ranked = strategy(q)
        r = recall_at_k(ranked, rel, k)
        recalls.append(r)
        rrs.append(reciprocal_rank(ranked, rel))
        per_query[q] = r
    return {"recall@k": float(np.mean(recalls)), "mrr": float(np.mean(rrs)), "per_query": per_query}
```

### Stage 3 - generate variations (multi-query)

```python
EXPAND_SYSTEM = "You rewrite search queries for a hosting-provider support knowledge base."


def normalize(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def expand_query(query: str, n: int = 4) -> list[str]:
    prompt = f"""Write {n} different search queries that could retrieve documents answering the user's question.
Rules:
- Use different vocabulary in each: one describing the symptom, one naming the likely technical cause,
  one using product/feature names, one in formal documentation style.
- Each query is a short phrase or sentence, no explanations.
- Do not repeat the original wording.
Return JSON exactly like: {{"queries": ["...", "..."]}}

User question: {query}"""
    data = generate_json(prompt, system=EXPAND_SYSTEM, temperature=0.4)
    variants = data["queries"] if isinstance(data, dict) else data
    return [v for v in variants if isinstance(v, str)]


def clean_variants(original: str, variants: list[str], max_variants: int = 5) -> list[str]:
    """Original first (the anchor), then unique variants, capped. The LLM suggests; we enforce."""
    seen = {normalize(original)}
    out = [original]
    for v in variants:
        key = normalize(v)
        if key and key not in seen:
            seen.add(key)
            out.append(v.strip())
    return out[: max_variants + 1]
```

`temperature=0.4` is a little higher than the helper's default 0.2: we *want* variety. `clean_variants` implements the "suggestion box" rule: dedupe, anchor, cap.

### Stage 4 - RRF and the multi-query retriever

```python
def rrf(rank_lists: list[list[int]], k: int = 60, top: int = 5) -> list[int]:
    scores: dict[int, float] = {}
    for lst in rank_lists:
        for rank, doc_id in enumerate(lst, start=1):      # rank starts at 1
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    # sort by score desc; tie-break on id so results are deterministic
    return sorted(scores, key=lambda d: (-scores[d], d))[:top]


def multi_query(query: str, n: int = 4, per_query_k: int = 5, top: int = 5) -> list[int]:
    queries = clean_variants(query, expand_query(query, n))
    vecs = embed_texts(queries, task_type="RETRIEVAL_QUERY")   # ONE batched request for all variants
    lists = [search_one(v, per_query_k) for v in vecs]
    return rrf(lists, top=top)
```

Check your understanding with the by-hand example: `rrf([[3,7,1],[7,3,9],[7,5,3]], top=5)` returns `[7, 3, 5, 1, 9]` (1 and 9 tie, id breaks the tie).

### Stage 5 - step-back and decomposition

```python
def step_back(query: str) -> str:
    prompt = ("Write ONE more general question about the underlying concept needed to answer this question. "
              'Return JSON: {"query": "..."}\n\nQuestion: ' + query)
    return generate_json(prompt, system=EXPAND_SYSTEM)["query"]


def decompose(query: str, max_sub: int = 3) -> list[str]:
    prompt = (f"If the question contains several distinct parts, split it into at most {max_sub} independent "
              "sub-questions. If it is already a single focused question, return it unchanged as the only item. "
              'Return JSON: {"sub_questions": ["..."]}\n\nQuestion: ' + query)
    subs = generate_json(prompt, system=EXPAND_SYSTEM)["sub_questions"]
    return [s for s in subs if isinstance(s, str)][:max_sub]


def search_many(queries: list[str], per_query_k: int = 5, top: int = 5) -> list[int]:
    vecs = embed_texts(queries, task_type="RETRIEVAL_QUERY")
    return rrf([search_one(v, per_query_k) for v in vecs], top=top)


def step_back_search(query: str) -> list[int]:
    return search_many([query, step_back(query)])


def decomposition_search(query: str) -> list[int]:
    return search_many([query] + decompose(query))
```

Decomposition keeps the original query *plus* the sub-questions, again as an anchor.

### Stage 6 - cache the expansions and count the cost

```python
CACHE_FILE = pathlib.Path(".expansion_cache.json")


def cached_expand(query: str, n: int = 4) -> list[str]:
    cache = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}
    key = hashlib.sha256(f"{n}|{query}".encode()).hexdigest()
    if key not in cache:
        cache[key] = expand_query(query, n)
        CACHE_FILE.write_text(json.dumps(cache, indent=2))
    return cache[key]


def multi_query_cached(query: str) -> list[int]:
    queries = clean_variants(query, cached_expand(query))
    return search_many(queries)


if __name__ == "__main__":
    t0 = time.perf_counter()
    base = evaluate(baseline)
    t1 = time.perf_counter()
    multi = evaluate(multi_query_cached)
    t2 = time.perf_counter()
    print(f"baseline    recall@3={base['recall@k']:.2f} mrr={base['mrr']:.2f}  ({t1 - t0:.1f}s)")
    print(f"multi-query recall@3={multi['recall@k']:.2f} mrr={multi['mrr']:.2f}  ({t2 - t1:.1f}s)")
    for q in GOLD:
        b, m = base["per_query"][q], multi["per_query"][q]
        tag = "WIN " if m > b else "LOSS" if m < b else "tie "
        print(f"  {tag} {b:.2f} -> {m:.2f}  {q}")
    print("expansions for first query:", cached_expand("my website feels sluggish"))
```

Expected output (≈, because real embeddings and LLM output vary run to run):

```text
baseline    recall@3=0.61 mrr=0.72  (≈ 4s on first run, cached after)
multi-query recall@3=0.83 mrr=0.86  (≈ 10s first run: 6 LLM calls + 6 embed calls)
  WIN  0.33 -> 0.67  my website feels sluggish
  tie  1.00 -> 1.00  site down with gateway error
  ...
expansions for first query: ['website slow response time', 'high TTFB server latency', 'reduce page load time CDN caching', 'slow database queries performance']
```

Second run: expansions come from the cache and embeddings from the disk cache, so it costs **0 API calls**. That is the habit: *every expensive deterministic step gets cached*.

How to read the result: the numbers above are an illustration of the typical shape (recall up 10-25 points on messy queries, ties on clear ones). Your numbers will differ; what matters is that you computed them on *your* gold set.

## ⚠️ Common mistakes & debugging

| Symptom | Cause | Fix |
|---|---|---|
| Variants are near-identical ("site slow", "slow site", "website is slow") | Prompt only said "rephrase" | Demand different *angles* (symptom, cause, feature name, doc style); dedupe by embedding cosine > 0.97 |
| Recall gets *worse* on clear queries | Drifted variants pull in unrelated chunks and outvote the right one | Keep the original; weight it (add its list twice in RRF) or only expand when the first retrieval looks weak |
| `JSONDecodeError` / `KeyError: 'queries'` | Model returned a bare list or other key | Accept both shapes (see `expand_query`), wrap in try/except and fall back to the original query |
| Summing cosine scores across lists gives odd results | Scores from different queries are not on the same scale | Fuse by **rank** (RRF), not raw score |
| Quota errors (429) after a few test queries | One embed request *per variant* in a loop | Embed all variants in one `embed_texts([...])` call; cache; add sleeps in eval loops |
| Variants embedded with `RETRIEVAL_DOCUMENT` | Wrong task type | Queries are always `RETRIEVAL_QUERY` |
| Answers contain info from irrelevant chunks | Too many fused chunks (e.g. 20) sent to the LLM | Fuse wide (top 20), then **rerank or cut** to the top 4-6 before generating |
| Latency doubled | Sequential LLM calls (expand, then step-back, then decompose) | Do only one expansion flavour per query, or run calls concurrently with a thread pool |
| Results change every run | LLM temperature > 0 on the expansion | Cache expansions by query hash; lower temperature when debugging |

## 🚀 Pro-level insights (what the top 2% know)

1. **Expansion is a recall tool; measure recall, then check what happens after reranking.** If you already rerank the top 50 with a cross-encoder, a wider candidate pool from expansion helps; if your first-stage retriever already gets recall@20 ≈ 0.98, expansion is wasted money.
2. **Route, don't always expand.** A cheap gate: skip expansion for queries longer than ~12 words or containing IDs/error codes (those are keyword-like, where BM25 does the job). Expand short, vague, conversational queries. Some teams let a tiny LLM call classify "needs expansion?" but that costs a call too; heuristics are free.
3. **Weight the original.** Common production tweak: add the original query's ranked list twice to RRF (weight 2). It keeps precision while still gaining recall.
4. **Hybrid beats clever.** Fusing BM25 + dense with RRF often delivers more recall gain than LLM expansion, at zero LLM cost. Use expansion when vocabulary mismatch is the diagnosed problem, not as a default.
5. **Cheap candidate pool, expensive judgement.** Fuse top 5 per variant -> ~15-20 unique chunks -> rerank to 5. Don't hand 20 chunks to the generator.
6. **Conversation context.** In chat, a follow-up ("what about the Starter plan?") has no meaning alone. The same LLM call that expands should also *rewrite* it as a standalone question using the last 2-3 turns. This "contextualise then expand" is the single most valuable query-side fix for chatbots.
7. **Failure modes:** drift (variants wander to a neighbouring topic), hallucinated product names in variants, and over-fusion (chunks matching only one weird variant). Mitigation: cap at 5, keep original, rerank, and log variants in production so you can inspect bad cases.
8. **Observability.** Log (query, variants, per-variant top ids, fused ids). When an answer is wrong, you can see whether expansion helped or hurt. Same as logging SQL in Laravel Telescope.
9. **Latency math.** The expansion call is serial and blocks everything (≈ 0.5-2 s). Hide it: start the *original-query* retrieval immediately in parallel while the LLM generates variants, then merge when variants are ready. If expansion times out (say 2 s), return the original-only result: graceful degradation.
10. **Evaluate on a golden set in CI** (topic 04). Query expansion changes with the model version; a silent regression is common.

## 🧪 Mini lab
Time: 10-15 minutes. Use `expansion_demo.py`.

1. Add two **new** gold queries of your own that use informal wording (e.g. "my emails from the site never arrive" needs a new chunk, so add a chunk about SMTP limits to `CORPUS` first; renumber your gold ids carefully).
2. Run `evaluate(baseline)` vs `evaluate(multi_query_cached)`.
3. Implement a **weighted RRF**: add a `weights: list[float]` parameter to `rrf` so the original list counts double. Compare recall@3 and MRR with weights `[2,1,1,1,1]` vs `[1,1,1,1,1]`.
4. Print, for each query, the chunk ids retrieved by each variant. Find a variant that *hurt* (retrieved only irrelevant ids) and write down why you think the LLM produced it.
5. Stretch: run `evaluate(decomposition_search)` and `evaluate(step_back_search)`; which query in the gold set benefits from which technique?

## ✅ Recap checklist
- [ ] I can explain the vocabulary gap with an example from my own domain.
- [ ] I can compute RRF for three small ranked lists by hand (k = 60).
- [ ] I know why fusing by rank is safer than adding cosine scores.
- [ ] I always keep the original query and cap the number of variants.
- [ ] I embed all variants in one batched call with `RETRIEVAL_QUERY`.
- [ ] I can write the multi-query, step-back and decomposition prompts from memory.
- [ ] I can say which failure each variant fixes (vocabulary / over-specific / multi-facet).
- [ ] I can compute recall@k and MRR and compare strategies per query.
- [ ] I know the cost: +1 LLM call, +N embeddings in 1 request, +≈ 0.5-2 s, and how caching removes repeats.
- [ ] I know when NOT to expand (precise/ID-like queries, already high recall, tight latency).

## 🎤 Interview questions

**1. What problem does query expansion solve, and what does it cost?**
It addresses the vocabulary gap: user wording differs from document wording, so the relevant chunk may rank too low. It raises recall by searching with several rewrites. Cost: one extra LLM call (≈ 0.5-2 s), more embedding tokens, and risk of query drift.

**2. Why use Reciprocal Rank Fusion instead of summing similarity scores?**
Scores from different queries (and different retrievers like BM25 vs cosine) are on different scales and not comparable; ranks are. RRF adds `1/(k+rank)` per list, rewarding chunks that appear near the top of many lists, needs no score normalisation and is robust. k ≈ 60 is the usual default.

**3. Multi-query, step-back and decomposition: when do you use each?**
Multi-query for vocabulary mismatch and vague queries; step-back when the query is too specific and the answer is conceptual; decomposition when one question has several facets or needs several documents. Multi-hop questions where step 2 depends on step 1 need an agentic loop instead.

**4. How would you prove expansion improved your system?**
Build a golden set of queries with labelled relevant chunks, compute recall@k and MRR for baseline and expansion on the same corpus, compare per-query wins/losses, and also compare end-to-end answer quality (faithfulness/correctness) because better recall does not guarantee better answers. Use enough queries (dozens) to be meaningful.

**5. How do you keep expansion cheap on a free-tier API?**
Batch all variants in one embedding request, cache expansions by query hash and embeddings on disk, cap variants (3-5), skip expansion for precise queries, use `thinking_budget=0`, run the original retrieval in parallel, and fall back to the original query if the LLM call fails or times out.

## 🔗 Next
**02 - HyDE**: instead of rewriting the *question*, imagine a fake *answer* and search with that, which attacks the same vocabulary gap from the other direction.
