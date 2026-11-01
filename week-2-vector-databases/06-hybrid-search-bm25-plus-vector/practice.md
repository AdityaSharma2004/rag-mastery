# Practice - Hybrid search (BM25 + vector)

Five questions, rising difficulty. Attempt each before opening the hint, and the hint before the solution. Every solution is the exact code in `solutions.py` (`python solutions.py` runs all). **Q1, Q2 and Q4 need no API key.** Q3 embeds with the free Gemini API; Q5 also needs Postgres with pgvector (`docker run -d -e POSTGRES_PASSWORD=postgres -p 5432:5432 pgvector/pgvector:pg16`, `DATABASE_URL` in `.env`). All use the 10-document support knowledge base `DOCS` and the 8 labelled queries `LABELED` at the top of `solutions.py`, plus this shared helper:

```python
def tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]


def rrf(rank_lists: list[list[int]], weights: list[float] | None = None, k: int = 60) -> list[tuple[int, float]]:
    """score(d) = sum_i  w_i / (k + rank_i(d))  - ranks start at 1; only ranks matter, never raw scores."""
    weights = weights or [1.0] * len(rank_lists)
    scores: dict[int, float] = {}
    for w, ranking in zip(weights, rank_lists):
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + w / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


def mrr_hit1(rankings, truth):
    rr = [1 / (r.index(t) + 1) if t in r else 0.0 for r, t in zip(rankings, truth)]
    h1 = [1.0 if r and r[0] == t else 0.0 for r, t in zip(rankings, truth)]
    return round(float(np.mean(rr)), 3), round(float(np.mean(h1)), 3)
```

---

### Q1. Warm-up - BM25 with `rank_bm25` and why tokenisation matters

**Problem**

Build a `BM25Okapi` index over `DOCS` with the `tokenize` helper. Search `"error E4012"`, `"payment gateway timeout"` and `"money back"`, printing the top matches with scores. Then rebuild the index with a naive `str.split()` tokenizer and search `"error e4012"`.

<details><summary><b>Hint</b></summary>

`bm.get_scores(tokenize(query))` returns one score per document; sort with `np.argsort(-scores)` and ignore zeros. `split()` keeps case and punctuation (`"E4012"` vs `"e4012"`; `"E4012"` vs `"E4012;"`).

</details>

<details><summary><b>Solution</b></summary>

```python
# --------------------------------------------------------------------------
# Q1. Warm-up: BM25 with the rank_bm25 library; why tokenisation decides everything
# --------------------------------------------------------------------------
def q1():
    bm = BM25Okapi([tokenize(d) for d in DOCS])
    for query in ["error E4012", "payment gateway timeout", "money back"]:
        s = bm.get_scores(tokenize(query))
        top = [int(i) for i in np.argsort(-s)[:2] if s[i] > 0]
        print(f"{query!r:<28} -> {top}  scores {[round(float(s[i]), 2) for i in top]}")
    # a naive split() tokenizer keeps punctuation and case: "E4012" and "e4012;" never match
    naive = BM25Okapi([d.split() for d in DOCS])
    s = naive.get_scores("error e4012".split())
    print("naive split() tokenizer, 'error e4012' ->", [int(i) for i in np.argsort(-s)[:2] if s[i] > 0])
    return True
```

</details>

**Why it works / what to learn**
Exact codes are found instantly. `"money back"` returns the *webhook back-off* document: a lexical false positive, because the only shared word is "back" (the refund article never says "back"). That is the keyword leg's blind spot, and why we fuse with a semantic leg. The naive tokenizer finds nothing at all: case and punctuation must be normalised.

**Expected output (≈)**
```text
'error E4012'                -> [2, 8]  scores [≈2.7, ≈1.2]
'payment gateway timeout'    -> [2, 1]  scores [≈2.7, ≈1.2]
'money back'                 -> [9]  scores [≈1.7]
naive split() tokenizer, 'error e4012' -> []
```

---

### Q2. Apply - why fusing raw scores fails

**Problem**

Given BM25 scores `[14, 3, 2.5, 2, 0]` (one exact-ID outlier at doc 0) and cosine scores `[0.55, 0.81, 0.80, 0.62, 0.30]` (docs 1 and 2 are the truly relevant ones), compute the winner under: raw sum, min-max, z-score, and RRF. Verify one RRF value by hand.

<details><summary><b>Hint</b></summary>

RRF uses the *rank* of each doc in each list: `1/(60+rank_bm25) + 1/(60+rank_cosine)`. Doc 1 is rank 2 in BM25 and rank 1 in cosine.

</details>

<details><summary><b>Solution</b></summary>

```python
# --------------------------------------------------------------------------
# Q2. Apply: why fusing raw scores fails - three normalisations vs RRF, on made-up score lists
# --------------------------------------------------------------------------
def q2():
    bm25 = np.array([14.0, 3.0, 2.5, 2.0, 0.0])        # one outlier: an exact ID match
    cosine = np.array([0.55, 0.81, 0.80, 0.62, 0.30])   # embeddings: docs 1 and 2 are the real answers
    minmax = lambda x: (x - x.min()) / (x.max() - x.min())
    zscore = lambda x: (x - x.mean()) / x.std()

    def winner(s): return int(np.argmax(s))
    print("raw sum        :", np.round(bm25 + cosine, 2), "-> winner", winner(bm25 + cosine))
    print("min-max (0.5/0.5):", np.round(0.5 * minmax(bm25) + 0.5 * minmax(cosine), 2),
          "-> winner", winner(minmax(bm25) + minmax(cosine)))
    print("z-score        :", np.round(zscore(bm25) + zscore(cosine), 2), "-> winner", winner(zscore(bm25) + zscore(cosine)))
    r_bm = [int(i) for i in np.argsort(-bm25)]
    r_co = [int(i) for i in np.argsort(-cosine)]
    fused = rrf([r_bm, r_co])
    print("RRF (k=60)     :", [(d, round(s, 4)) for d, s in fused], "-> winner", fused[0][0])
    # hand check: doc 1 -> bm25 rank 2, cosine rank 1: 1/62 + 1/61
    print("hand check doc 1:", round(1 / 62 + 1 / 61, 4))
    return fused[0][0]
```

</details>

**Why it works / what to learn**
The BM25 outlier dominates every score-based fusion (raw, min-max, z-score all pick doc 0). RRF throws away magnitudes and asks "who ranks well in *both*?", picking doc 1. Rank fusion is the robust default.

**Expected output (≈)**
```text
raw sum        : [14.55  3.81  3.3   2.62  0.3 ] -> winner 0
min-max (0.5/0.5): [0.75 0.61 0.58 0.39 0.  ] -> winner 0
z-score        : [ 1.61  0.77  0.62 -0.44 -2.55] -> winner 0
RRF (k=60)     : [(1, 0.0325), (0, 0.032), (2, 0.032), ...] -> winner 1
hand check doc 1: 0.0325
```

---

### Q3. Build - a `HybridRetriever` and an evaluation

**Problem**

Write `HybridRetriever(docs, pool, k_rrf, w_kw, w_dense)` with `search(q, k, mode)` for `keyword`, `dense` and `hybrid`. Evaluate MRR and hit@1 for each mode on `LABELED`, then try keyword weights 0.5, 1 and 2.

<details><summary><b>Hint</b></summary>

Keyword ranking: top `pool` with score > 0. Dense ranking: `np.argsort(-(D @ q))[:pool]`. Fuse with `rrf([kw, dense], weights=[w_kw, w_dense])`. MRR = mean of `1/rank` of the correct doc.

</details>

<details><summary><b>Solution</b></summary>

```python
# --------------------------------------------------------------------------
# Q3. Build: a HybridRetriever (BM25 + dense, weighted RRF, candidate pools) and an evaluation
# --------------------------------------------------------------------------
class HybridRetriever:
    def __init__(self, docs, dim=DIM, pool=20, k_rrf=60, w_kw=1.0, w_dense=1.0):
        self.docs, self.pool, self.k_rrf, self.w = docs, pool, k_rrf, (w_kw, w_dense)
        self.bm25 = BM25Okapi([tokenize(d) for d in docs])
        self.D = embed_texts(docs, task_type="RETRIEVAL_DOCUMENT", dim=dim)
        self.dim = dim

    def keyword_ranking(self, q):
        s = self.bm25.get_scores(tokenize(q))
        return [int(i) for i in np.argsort(-s)[: self.pool] if s[i] > 0]          # drop docs that matched nothing

    def dense_ranking(self, q):
        s = self.D @ embed_query(q, dim=self.dim)
        return [int(i) for i in np.argsort(-s)[: self.pool]]

    def search(self, q, k=3, mode="hybrid"):
        kw, de = self.keyword_ranking(q), self.dense_ranking(q)
        if mode == "keyword": return kw[:k]
        if mode == "dense": return de[:k]
        return [d for d, _ in rrf([kw, de], weights=list(self.w), k=self.k_rrf)[:k]]


def q3():
    hr = HybridRetriever(DOCS)
    truth = [t for _, t in LABELED]
    print(f"{'mode':<9} {'MRR':>6} {'hit@1':>7}")
    for mode in ("keyword", "dense", "hybrid"):
        ranks = [hr.search(q, k=len(DOCS), mode=mode) for q, _ in LABELED]
        print(f"{mode:<9} {mrr_hit1(ranks, truth)[0]:>6} {mrr_hit1(ranks, truth)[1]:>7}")
    for w_kw in (0.5, 1.0, 2.0):                                                  # tune the leg weights
        hr.w = (w_kw, 1.0)
        ranks = [hr.search(q, k=len(DOCS)) for q, _ in LABELED]
        print(f"weighted RRF  keyword weight {w_kw}: MRR {mrr_hit1(ranks, truth)[0]}")
    return True
```

</details>

**Why it works / what to learn**
One object exposes all three modes, so evaluation is a loop. Expect keyword to win on the code queries (`E4012`, `E5001`) and lose on paraphrases; dense the reverse; hybrid at or above the better single method. The weight sweep shows whether tuning matters on your data; if the metric does not move, keep the defaults.

**Expected output (≈, with the real Gemini model)**
```text
mode         MRR   hit@1
keyword    ≈0.6   ≈0.6      (strong on codes, weak on paraphrases)
dense      ≈0.8-1.0 ≈0.7-1.0
hybrid     ≈0.9-1.0 ≈0.8-1.0 (usually >= both)
weighted RRF  keyword weight 0.5 / 1.0 / 2.0: MRR close to each other
```
(Exact figures depend on the live embedding model; the pattern is what to check.)

---

### Q4. Debug - "BM25 never finds E-4012 / e4012 / 4012"

**Problem**

A document says `"Error E-4012: the payment gateway timed out."`. With a naive tokenizer, users searching `E-4012`, `e4012` or `4012` find nothing. Compare three tokenizers (naive `split`, alnum split, code-aware) on the three queries and fix it.

<details><summary><b>Hint</b></summary>

`"E-4012:".split()` gives `"E-4012:"`, which never equals `"e-4012"`. A regex `[a-z0-9]+` after lower-casing splits into `e` and `4012`; to also match `e4012` index the glued form as an extra token.

</details>

<details><summary><b>Solution</b></summary>

```python
# --------------------------------------------------------------------------
# Q4. Debug: "BM25 never finds E-4012 / e4012 even though the doc contains it"
# --------------------------------------------------------------------------
def q4():
    # (a corpus of 10: with only 2 docs, rank_bm25 gives a term that is in half of them an IDF of 0)
    docs = ["Error E-4012: the payment gateway timed out."] + [d for i, d in enumerate(DOCS) if i != 2]

    def naive_tok(t): return t.split()                                           # case + punctuation kept
    def basic_tok(t): return re.findall(r"[a-z0-9]+", t.lower())                 # splits E-4012 into 'e' and '4012'
    def code_tok(t):                                                              # also keep the glued form 'e4012'
        toks = basic_tok(t)
        glued = re.findall(r"[a-z]+-\d+", t.lower())
        return toks + [g.replace("-", "") for g in glued]

    for name, tok in (("naive split()", naive_tok), ("alnum split", basic_tok), ("code-aware", code_tok)):
        bm = BM25Okapi([tok(d) for d in docs])
        for query in ("E-4012", "e4012", "4012"):
            s = bm.get_scores(tok(query))
            print(f"{name:<14} query {query!r:<9} -> {'FOUND' if s[0] > 0 else 'missed'}")
    return True
```

</details>

**Why it works / what to learn**
Documents and queries must pass through the *same* analysis, and that analysis must understand your identifiers. Log failing exact-match queries and turn each into a tokenizer test case. (The demo uses a 10-document corpus: with only 2 documents `rank_bm25` assigns a term in half of them an IDF of 0 and would show "missed" regardless.)

**Expected output (≈)**
```text
naive split()  query 'E-4012'  -> missed      (all three queries missed)
alnum split    query 'E-4012'  -> FOUND
alnum split    query 'e4012'   -> missed
alnum split    query '4012'    -> FOUND
code-aware     query 'E-4012'  -> FOUND       (all three found)
```

---

### Q5. Production - hybrid search inside Postgres

**Problem**

In Postgres, add a generated `tsvector` column with a GIN index next to a pgvector column, and write **one SQL query** that runs the semantic and keyword searches, ranks each, and fuses them with RRF (`k = 60`) using a `FULL OUTER JOIN`. Run it for `"error E4012"` and `"I want my money back"`.

<details><summary><b>Hint</b></summary>

`tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED`; keyword rank via `ts_rank_cd(tsv, query)` and `websearch_to_tsquery`; `row_number() OVER (ORDER BY ...)` gives the rank; `COALESCE(1.0/(60+rnk), 0)` for a missing leg.

</details>

<details><summary><b>Solution</b></summary>

```python
# --------------------------------------------------------------------------
# Q5. Production: hybrid search INSIDE Postgres (tsvector full-text + pgvector, fused with RRF in SQL)
# --------------------------------------------------------------------------
def q5():
    import psycopg
    from pgvector.psycopg import register_vector
    url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/ragdb")
    try:
        conn = psycopg.connect(url, autocommit=True)
    except psycopg.OperationalError as e:
        print("Postgres not reachable - start it (see header) and re-run Q5.", e); return None
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector"); register_vector(conn)
    conn.execute("DROP TABLE IF EXISTS w2_hybrid")
    conn.execute(f"""CREATE TABLE w2_hybrid (
        id int PRIMARY KEY, content text, embedding vector({DIM}),
        tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED)""")   # keyword column, auto-maintained
    conn.execute("CREATE INDEX w2_hybrid_tsv ON w2_hybrid USING gin (tsv)")               # GIN = inverted index
    vecs = embed_texts(DOCS, dim=DIM)
    with conn.cursor() as cur:
        cur.executemany("INSERT INTO w2_hybrid (id, content, embedding) VALUES (%s,%s,%s)",
                        [(i, t, v) for i, (t, v) in enumerate(zip(DOCS, vecs))])

    SQL = """
    WITH semantic AS (
        SELECT id, row_number() OVER (ORDER BY embedding <=> %(q)s) AS rnk
        FROM w2_hybrid ORDER BY embedding <=> %(q)s LIMIT 20),
    keyword AS (
        SELECT id, row_number() OVER (ORDER BY ts_rank_cd(tsv, query) DESC) AS rnk
        FROM w2_hybrid, websearch_to_tsquery('english', %(text)s) query
        WHERE tsv @@ query ORDER BY ts_rank_cd(tsv, query) DESC LIMIT 20)
    SELECT COALESCE(s.id, k.id) AS id,
           COALESCE(1.0 / (60 + s.rnk), 0) + COALESCE(1.0 / (60 + k.rnk), 0) AS rrf_score
    FROM semantic s FULL OUTER JOIN keyword k ON s.id = k.id
    ORDER BY rrf_score DESC LIMIT 3"""

    for text in ("error E4012", "I want my money back"):
        rows = conn.execute(SQL, {"q": embed_query(text, dim=DIM), "text": text}).fetchall()
        print(f"{text!r:<24} -> {[(r[0], round(float(r[1]), 4)) for r in rows]}")
    conn.execute("DROP TABLE w2_hybrid")
    return True
```

</details>

**Why it works / what to learn**
Two CTEs produce ranked candidate lists; the `FULL OUTER JOIN` keeps documents found by only one leg; `COALESCE` gives the missing leg zero points. You get hybrid search, transactions and filters in the single database you already operate, no Elasticsearch needed.

**Expected output (≈, with the real Gemini model)**
```text
'error E4012'            -> [(2, 0.0328), (8, ≈0.0161), ...]      (doc 2 found by both legs: top RRF score 2/61)
'I want my money back'   -> [(1, ≈0.0164), ...]                    (refund doc via the semantic leg)
```
