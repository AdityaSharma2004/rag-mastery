# Practice - Semantic vs keyword vs hybrid search

Five questions, rising difficulty. Attempt each before opening the hint, and the hint before the solution. Every solution is the exact code in `solutions.py`. **Q1, Q2 and Q4 run offline with no API key.** Q3 and Q5 embed text with the free Gemini API (set `GEMINI_API_KEY`). All use the same 8-document support knowledge base `KB` defined at the top of `solutions.py`:

```text
0 To reset your password click Forgot password on the login page.
1 Refunds are issued to the original payment method within 5 business days.
2 Error E4012 means the payment gateway timed out; retry after one minute.
3 Two-factor authentication adds a second step when you sign in.
4 You can cancel your subscription anytime from the billing settings.
5 Invoices are emailed on the first day of every month as PDF files.
6 If you cannot access your account, use the recovery link sent to your email.
7 The mobile app supports dark mode under Display settings.
```

---

### Q1. Warm-up - `LIKE` vs keyword search

**Problem**

Implement `like_search` (the whole phrase must appear, like `LIKE '%q%'`) and `keyword_search` (any shared token, ranked by overlap). Run both on `"reset password"`, `"forgot my login credentials"` and `"money back"`. Which queries return nothing, and why?

<details><summary><b>Hint</b></summary>

Lower-case everything, tokenise with a regex, drop stop-words, and compare sets of tokens for the keyword version.

</details>

<details><summary><b>Solution</b></summary>

```python
# --------------------------------------------------------------------------
# Q1. Warm-up: WHERE LIKE vs token keyword search vs (spoiler) meaning
# --------------------------------------------------------------------------
def like_search(query: str, docs: list[str]) -> list[int]:
    """SQL:  WHERE doc LIKE '%query%'   (the whole phrase must appear)"""
    return [i for i, d in enumerate(docs) if query.lower() in d.lower()]


def keyword_search(query: str, docs: list[str]) -> list[int]:
    """Any shared token counts (very rough FULLTEXT ... IN BOOLEAN MODE with OR). Ranked by overlap."""
    q = set(tokenize(query))
    scored = [(len(q & set(tokenize(d))), i) for i, d in enumerate(docs)]
    return [i for s, i in sorted(scored, reverse=True) if s > 0]


def q1():
    for query in ["reset password", "forgot my login credentials", "money back"]:
        print(f"\nquery: {query!r}")
        print("  LIKE    ->", like_search(query, KB))
        print("  keyword ->", keyword_search(query, KB))
    # 'money back' should find the refund doc (1) but shares no words with it.
    return keyword_search("money back", KB)
```

</details>

**Why it works / what to learn**
`LIKE` fails on "reset password" because the document says "reset **your** password". Token search is more forgiving but still needs a shared word: "money back" shares none with the refund document, so it returns nothing. Only meaning-based search bridges that gap.

**Expected output (≈)**
```text
query: 'reset password'
  LIKE    -> []
  keyword -> [0]
query: 'forgot my login credentials'
  LIKE    -> []
  keyword -> [0]
query: 'money back'
  LIKE    -> []
  keyword -> []
```

---

### Q2. Apply - BM25 from scratch

**Problem**

Implement BM25 (`k1=1.5`, `b=0.75`) and rank `KB` for `"payment gateway error"`. Print the top three, explain why `e4012` has a higher IDF than `payment`, and cross-check the ranking against the `rank_bm25` library.

<details><summary><b>Hint</b></summary>

`idf = ln((N - n + 0.5)/(n + 0.5) + 1)`. For the length term use `1 - b + b * len(doc)/avgdl`. The library uses a slightly different IDF, so compare *rankings*, not raw numbers.

</details>

<details><summary><b>Solution</b></summary>

```python
# --------------------------------------------------------------------------
# Q2. Apply: BM25 from scratch
# --------------------------------------------------------------------------
class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.toks = [tokenize(d) for d in docs]
        self.N = len(docs)
        self.avgdl = sum(len(t) for t in self.toks) / self.N
        df = Counter(w for t in self.toks for w in set(t))        # in how many docs does w appear?
        # idf: rare words matter more. The "+1" keeps it positive.
        self.idf = {w: math.log((self.N - n + 0.5) / (n + 0.5) + 1) for w, n in df.items()}

    def score(self, query: str) -> np.ndarray:
        q = tokenize(query)
        out = np.zeros(self.N)
        for i, t in enumerate(self.toks):
            tf = Counter(t)
            dl = len(t)
            for w in q:
                if w not in tf:
                    continue
                num = tf[w] * (self.k1 + 1)
                den = tf[w] + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                out[i] += self.idf[w] * num / den
        return out


def q2():
    bm = BM25(KB)
    query = "payment gateway error"
    s = bm.score(query)
    order = np.argsort(-s)
    print("BM25 ranking for", repr(query))
    for i in order[:3]:
        print(f"  {s[i]:.3f}  [{i}] {KB[i]}")
    # why is 'e4012' worth more than 'payment'?  rarity:
    print("idf('e4012') =", round(bm.idf["e4012"], 3), "  idf('payment') =", round(bm.idf["payment"], 3))

    # cross-check the *ranking* against the rank_bm25 library (numbers differ: different idf variant)
    try:
        from rank_bm25 import BM25Okapi
        lib = BM25Okapi([tokenize(d) for d in KB]).get_scores(tokenize(query))
        print("library top-3:", [int(i) for i in np.argsort(-lib)[:3]], " ours:", [int(i) for i in order[:3]])
    except ImportError:
        print("(pip install rank-bm25 to cross-check)")
    return int(order[0])
```

</details>

**Why it works / what to learn**
`e4012` appears in 1 of 8 documents, `payment` in 2, so the rarer token carries more weight (IDF 1.79 vs 1.28). The doc that matches all three query words wins by a wide margin, and the doc matching only "payment" is a distant second.

**Expected output (≈)**
```text
BM25 ranking for 'payment gateway error'
  4.302  [2] Error E4012 means the payment gateway timed out; retry after one minute.
  1.194  [1] Refunds are issued to the original payment method within 5 business days.
  0.000  [0] To reset your password click Forgot password on the login page.
idf('e4012') = 1.792   idf('payment') = 1.281
library top-3: [2, 1, 0]  ours: [2, 1, 0]
```

---

### Q3. Build - hybrid search with RRF

**Problem**

Write `rrf(rank_lists, k=60)` and `hybrid_search(query, docs)` that runs BM25 and Gemini semantic search, then fuses them. Try `"I forgot my login credentials"` and `"error E4012"`. Show each leg's ranking and the fused result.

<details><summary><b>Hint</b></summary>

`RRF(d) = Σ 1/(k + rank)`, ranks start at 1. Drop zero-score BM25 documents before fusing. Use `RETRIEVAL_DOCUMENT` for docs and `RETRIEVAL_QUERY` for the query.

</details>

<details><summary><b>Solution</b></summary>

```python
# --------------------------------------------------------------------------
# Q3. Build: hybrid search with Reciprocal Rank Fusion (RRF)
# --------------------------------------------------------------------------
def rrf(rank_lists: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """score(doc) = sum over lists of 1/(k + rank).  rank starts at 1. Only ranks matter, not raw scores."""
    scores: dict[int, float] = {}
    for ranking in rank_lists:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


def hybrid_search(query: str, docs: list[str], top_n: int = 3, pool: int = 5):
    bm = BM25(docs)
    kw = bm.score(query)
    kw_rank = [int(i) for i in np.argsort(-kw)[:pool] if kw[i] > 0]

    D = embed_texts(docs, task_type="RETRIEVAL_DOCUMENT", dim=768)
    qv = embed_texts([query], task_type="RETRIEVAL_QUERY", dim=768)[0]
    sem = D @ qv
    sem_rank = [int(i) for i in np.argsort(-sem)[:pool]]

    fused = rrf([kw_rank, sem_rank])[:top_n]
    return kw_rank, sem_rank, fused


def q3():
    for query in ["I forgot my login credentials", "error E4012"]:
        kw, sem, fused = hybrid_search(query, KB)
        print(f"\nquery: {query!r}")
        print("  keyword :", kw)
        print("  semantic:", sem)
        print("  hybrid  :", [(i, round(s, 4)) for i, s in fused])
    return fused
```

</details>

**Why it works / what to learn**
For the exact-ID query both legs agree, so document 2 gets the maximum RRF score. For the paraphrase query the keyword leg is weak and the semantic leg does the work. Fusion lets each method contribute where it is strong, without comparing their raw scores.

**Expected output (≈)**
```text
query: 'I forgot my login credentials'
  keyword : [0]
  semantic: [0, 6, 3, ...]
  hybrid  : [(0, 0.0328), (6, ...), ...]
query: 'error E4012'
  keyword : [2]
  semantic: [2, ...]
  hybrid  : [(2, 0.0328), ...]
```
(The top RRF score for a document ranked first by both legs is 2/61 ≈ 0.0328. Semantic ranks beyond the first depend on the live model.)

---

### Q4. Debug - "I added BM25 and cosine and hybrid got worse"

**Problem**

BM25 scores `[11, 14, 1, 0]` and cosine scores `[0.82, 0.55, 0.80, 0.30]` for four documents. A teammate fuses with `0.5*bm25 + 0.5*cosine` and document 1 wins. Explain the bug and fix it two ways.

<details><summary><b>Hint</b></summary>

Compare the *range* of the two score arrays. What fraction of document 1's final score comes from BM25?

</details>

<details><summary><b>Solution</b></summary>

```python
# --------------------------------------------------------------------------
# Q4. Debug: "I added BM25 and cosine and hybrid got WORSE"
# --------------------------------------------------------------------------
def minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    return np.zeros_like(x) if hi == lo else (x - lo) / (hi - lo)


def q4():
    # Made-up but realistic scales: BM25 scores are unbounded (here 0-14), cosine is 0-1.
    bm25 = np.array([11.0, 14.0, 1.0, 0.0])    # keyword engine: doc 1 slightly ahead of doc 0
    cosine = np.array([0.82, 0.55, 0.80, 0.30])  # embeddings: doc 0 clearly best (doc 0 is good on BOTH)

    naive = 0.5 * bm25 + 0.5 * cosine            # BUG: different scales
    print("naive sum   :", np.round(naive, 2), "-> winner", int(naive.argmax()))
    print("  (BM25 contributes ~", round(float(0.5 * bm25[1] / naive[1]) * 100), "% of doc 1's score)")

    # FIX A: normalise both to 0-1 first, then weight
    fixed = 0.5 * minmax(bm25) + 0.5 * minmax(cosine)
    print("min-max sum :", np.round(fixed, 2), "-> winner", int(fixed.argmax()))

    # FIX B: ignore scales entirely, fuse RANKS (RRF)
    r_bm = [int(i) for i in np.argsort(-bm25)]
    r_co = [int(i) for i in np.argsort(-cosine)]
    print("RRF         :", [(i, round(s, 4)) for i, s in rrf([r_bm, r_co])])
    return int(fixed.argmax())
```

</details>

**Why it works / what to learn**
BM25 is unbounded (here up to 14) while cosine lives in 0-1, so BM25 contributes ~96% of the fused score and the semantic leg is nearly ignored. Fix A (min-max normalise both to 0-1 before weighting) or Fix B (RRF on ranks) both make document 0, which is strong on both legs, win. RRF is preferred because it needs no per-query normalisation.

**Expected output (≈)**
```text
naive sum   : [5.91 7.28 0.9  0.15] -> winner 1
  (BM25 contributes ~ 96 % of doc 1's score)
min-max sum : [0.89 0.74 0.52 0.  ] -> winner 0
RRF         : [(0, 0.0325), (1, 0.0323), (2, 0.032), (3, 0.0312)]
```

---

### Q5. Production - evaluate keyword vs semantic vs hybrid

**Problem**

Given 8 labelled queries (`(query, correct_doc_id)`), compute **MRR** and **hit@1** for keyword (BM25), semantic (Gemini) and hybrid (RRF). Which method would you ship for this knowledge base?

<details><summary><b>Hint</b></summary>

For each query get a *full* ranking per method. The reciprocal rank is `1/position` of the correct id (0 if absent). hit@1 is 1 when that position is 1.

</details>

<details><summary><b>Solution</b></summary>

```python
# --------------------------------------------------------------------------
# Q5. Production: an evaluation harness - keyword vs semantic vs hybrid
# --------------------------------------------------------------------------
EVAL = [  # (query, index of the correct KB doc)
    ("error E4012", 2),                          # exact token: keyword should win
    ("what does E4012 mean", 2),
    ("I want my money back", 1),                 # no shared words: semantic should win
    ("lost access to my profile", 6),
    ("how do I stop paying every month", 4),
    ("change app theme to dark", 7),             # shares words: all should work
    ("when do I get my bill by email", 5),
    ("extra security step at login", 3),
]


def mrr_and_hit1(rankings: list[list[int]], truth: list[int]) -> tuple[float, float]:
    rr, hit = [], []
    for ranking, t in zip(rankings, truth):
        pos = ranking.index(t) + 1 if t in ranking else None
        rr.append(1 / pos if pos else 0.0)
        hit.append(1.0 if pos == 1 else 0.0)
    return float(np.mean(rr)), float(np.mean(hit))


def q5():
    bm = BM25(KB)
    D = embed_texts(KB, task_type="RETRIEVAL_DOCUMENT", dim=768)
    kw_all, sem_all, hyb_all = [], [], []
    for query, _ in EVAL:
        kw = bm.score(query)
        kw_r = [int(i) for i in np.argsort(-kw)]
        qv = embed_texts([query], task_type="RETRIEVAL_QUERY", dim=768)[0]
        sem_r = [int(i) for i in np.argsort(-(D @ qv))]
        hyb_r = [i for i, _ in rrf([[i for i in kw_r if kw[i] > 0], sem_r])]
        kw_all.append(kw_r); sem_all.append(sem_r); hyb_all.append(hyb_r)

    truth = [t for _, t in EVAL]
    print(f"{'method':<10} {'MRR':>6} {'hit@1':>7}")
    for name, r in (("keyword", kw_all), ("semantic", sem_all), ("hybrid", hyb_all)):
        mrr, h1 = mrr_and_hit1(r, truth)
        print(f"{name:<10} {mrr:>6.2f} {h1:>7.2f}")
    return mrr_and_hit1(hyb_all, truth)
```

</details>

**Why it works / what to learn**
The set mixes exact-token queries (`E4012`), pure paraphrases (`money back`) and overlapping queries. A typical real run shows keyword strong on the ID queries and weak on paraphrases, semantic the reverse, and hybrid at or near the top on both, which is the reason hybrid is the default for mixed traffic. The exact figures depend on the live embedding model, so run it and read your own numbers. The habit to take away: ship the method your labelled queries justify, not the one that sounds smartest.

**Expected output (≈)**
```text
method        MRR   hit@1
keyword      ≈0.5-0.7  ≈0.4-0.6     (misses paraphrase queries)
semantic     ≈0.8-1.0  ≈0.7-1.0     (strong on paraphrases)
hybrid       ≈0.8-1.0  ≈0.7-1.0     (usually >= the better single method)
```
