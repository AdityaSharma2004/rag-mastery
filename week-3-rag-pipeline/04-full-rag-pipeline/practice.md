# Practice - The full RAG pipeline

Five questions, rising difficulty. Attempt each before opening the hint, and the hint before the solution. Every solution is the exact code in `solutions.py` (`python solutions.py` runs all).

**Offline by default.** Without an API key, `solutions.py` uses a *toy embedder* (word hashing with a stop-word list and 3 hand-made synonyms) and a *toy extractive "LLM"* that returns the best-matching source sentence with a citation. That lets you watch every moving part with no setup. Set `GEMINI_API_KEY` and the same code switches to `gemini-embedding-001` and `gemini-2.5-flash` (scores and wording then differ from the expected outputs below, which are from the offline run). The helpers `Chunk`, `Hit`, `chunk_text`, `KIRANA_DOCS` (4 tiny support docs of a fictional accounting app), `REFUSAL` and `SYSTEM` live at the top of `solutions.py` and are explained in the notes.

---

### Q1. Warm-up - build the prompt and verify the citations

**Problem**

Given two retrieved `Hit`s, write `build_prompt(question, hits)` that numbers the sources `[1]`, `[2]` and appends the question, and `extract_citations(answer, hits)` that maps the `[n]` markers in a model answer back to real sources. The model may invent a number that does not exist (`[9]`): drop it and report it.

<details><summary><b>Hint</b></summary>

`re.findall(r"\[(\d+)\]", answer)` finds the numbers; keep those with `1 <= n <= len(hits)`; the source of `[n]` is `hits[n-1]`.

</details>

<details><summary><b>Solution</b></summary>

```python
# --- Q1: build the prompt and verify the citations the model returns --------
CITE = re.compile(r"\[(\d+)\]")


def build_prompt(question, hits):
    blocks = [f"[{h.rank}] (source: {h.chunk.source})\n{h.chunk.text}" for h in hits]
    return "SOURCES:\n" + "\n\n".join(blocks) + f"\n\nQUESTION: {question}\nANSWER:"


def extract_citations(answer, hits):
    """Turn '[1][3]' into real sources; invented numbers like [9] are dropped, not trusted."""
    used = sorted({int(n) for n in CITE.findall(answer)})
    valid = [n for n in used if 1 <= n <= len(hits)]
    return [{"n": n, "source": hits[n - 1].chunk.source, "chunk_id": hits[n - 1].chunk.id} for n in valid], \
           sorted(set(used) - set(valid))


def q1():
    hits = [
        Hit(Chunk("refunds.md#0", "Refunds are possible within 14 days.", "refunds.md"), 0.81, 1),
        Hit(Chunk("plans.md#0", "The Growth plan costs 1,299 rupees per month.", "plans.md"), 0.55, 2),
    ]
    prompt = build_prompt("How long is the refund window?", hits)
    print(prompt)
    print("---")
    good, bad = extract_citations("Refunds are possible for 14 days [1]. Growth is 1,299 rupees [2][9].", hits)
    print("valid citations:", good)
    print("invented citation numbers dropped:", bad)
```

</details>

**Why it works / what to learn**
The model only *writes* `[1]`; your code decides what that means, so citations are checkable and the user can click through to the real chunk. Never trust a citation number without validating it against what you actually sent.

**Expected output**
```text
SOURCES:
[1] (source: refunds.md)
Refunds are possible within 14 days.

[2] (source: plans.md)
The Growth plan costs 1,299 rupees per month.

QUESTION: How long is the refund window?
ANSWER:
---
valid citations: [{'n': 1, 'source': 'refunds.md', 'chunk_id': 'refunds.md#0'}, {'n': 2, 'source': 'plans.md', 'chunk_id': 'plans.md#0'}]
invented citation numbers dropped: [9]
```

---

### Q2. Apply - ingest, store, retrieve

**Problem**

Write a tiny `VectorStore` (list of chunks + one numpy matrix), an `ingest(store, docs)` that chunks every document, embeds all chunks in one batch with the *document* task type and stores them, and `retrieve(store, question, k, min_score)` that embeds the question with the *query* task type and returns ranked `Hit`s. Run it on two questions.

<details><summary><b>Hint</b></summary>

Row `i` of the matrix must be the vector of `chunks[i]`. With unit-length vectors, `matrix @ query_vec` is the cosine similarity for every chunk at once; `np.argsort(-scores)[:k]` gives the best rows.

</details>

<details><summary><b>Solution</b></summary>

```python
# --- Q2: ingest -> store -> retrieve on the KiranaBooks docs ----------------
class VectorStore:
    """Smallest useful vector DB: a list of chunks + one matrix. Row i of the matrix = vector of chunks[i]."""
    def __init__(self):
        self.chunks, self.matrix = [], np.zeros((0, 0), dtype=np.float32)

    def add(self, chunks, vectors):
        assert len(chunks) == len(vectors), "one vector per chunk"
        self.matrix = vectors if not self.chunks else np.vstack([self.matrix, vectors])
        self.chunks.extend(chunks)


def ingest(store, docs, max_chars=300):
    chunks = [Chunk(f"{src}#{i}", piece, src, {"pos": i})
              for src, text in docs.items() for i, piece in enumerate(chunk_text(text, max_chars))]
    store.add(chunks, EMBED_DOCS([c.text for c in chunks]))
    return len(chunks)


def retrieve(store, question, k=3, min_score=0.0):
    scores = store.matrix @ EMBED_QUERY(question)            # unit vectors: dot product = cosine
    order = np.argsort(-scores)[:k]
    return [Hit(store.chunks[i], float(scores[i]), r) for r, i in enumerate(order, 1) if scores[i] >= min_score]


def q2():
    store = VectorStore()
    print("mode:", MODE, "| chunks stored:", ingest(store, KIRANA_DOCS))
    for q in ["Can I get my money back after 3 weeks?", "How long does the export download link last?"]:
        print("\nQ:", q)
        for h in retrieve(store, q, k=2):
            print(f"  #{h.rank} {h.score:.2f} {h.chunk.id}")
```

</details>

**Why it works / what to learn**
Ingestion is offline and never touches the LLM; the online path is embed-query → dot product → top-k. Scores here are low because the toy embedder only matches shared words; real Gemini embeddings give much higher scores for paraphrases ("money back" vs "refund"). The structure is identical.

**Expected output** (offline toy embedder)
```text
mode: offline toys | chunks stored: 4

Q: Can I get my money back after 3 weeks?
  #1 0.21 refunds.md#0
  #2 0.00 plans.md#0

Q: How long does the export download link last?
  #1 0.40 export.md#0
  #2 0.17 plans.md#0
```

---

### Q3. Build - the pipeline with a no-answer gate

**Problem**

Wrap Q1-Q2 in `RAGPipeline(store, k, min_score)` with `ask(question, k=None)` that: retrieves, **skips the LLM entirely** when nothing scores above `min_score` (returning the exact refusal sentence), otherwise builds the prompt, calls the LLM, extracts citations, records timings, and flags refusals. Prove the gate saves LLM calls by counting them.

<details><summary><b>Hint</b></summary>

`if not hits: return Answer(..., refused=True)` before calling the LLM. Keep a counter `self.llm_calls`. Pass `k` as an argument to `ask` instead of mutating shared state.

</details>

<details><summary><b>Solution</b></summary>

```python
# --- Q3: the pipeline with a no-answer gate, timings and a trace ------------
@dataclass
class Answer:
    question: str
    text: str
    citations: list
    hits: list
    timings_ms: dict
    refused: bool = False


class RAGPipeline:
    def __init__(self, store, k=3, min_score=0.2, llm=None):
        self.store, self.k, self.min_score = store, k, min_score
        self.llm = llm or LLM
        self.llm_calls = 0

    def ask(self, question, k=None):
        k = k or self.k
        t0 = time.perf_counter()
        hits = retrieve(self.store, question, k=k, min_score=self.min_score)
        t1 = time.perf_counter()
        if not hits:                                           # gate: no relevant chunk -> do NOT call the LLM
            return Answer(question, REFUSAL, [], [], {"retrieve": round((t1 - t0) * 1000, 2)}, refused=True)
        self.llm_calls += 1
        text = self.llm(build_prompt(question, hits), system=SYSTEM, temperature=0.1).strip()
        t2 = time.perf_counter()
        cites, _ = extract_citations(text, hits)
        return Answer(question, text, cites, hits,
                      {"retrieve": round((t1 - t0) * 1000, 2), "generate": round((t2 - t1) * 1000, 2)},
                      refused=REFUSAL in text)


def q3():
    store = VectorStore()
    ingest(store, KIRANA_DOCS)
    pipe = RAGPipeline(store, k=3, min_score=0.2)
    for q in ["How many days do I have to ask for a refund?", "Who won the cricket world cup?"]:
        a = pipe.ask(q)
        print(f"Q: {q}\nA: {a.text}\n   refused={a.refused} cited={[c['source'] for c in a.citations]} "
              f"llm_calls_so_far={pipe.llm_calls}")
```

</details>

**Why it works / what to learn**
A cheap vector search decides whether an LLM call is worthwhile. The off-topic question never reaches the LLM (`llm_calls_so_far` stays at 1), which saves latency and free-tier quota and removes a chance to hallucinate.

**Expected output**
```text
Q: How many days do I have to ask for a refund?
A: KiranaBooks refunds a subscription within 14 days of purchase, provided you have not exported more than 50 invoices. [1]
   refused=False cited=['refunds.md'] llm_calls_so_far=1
Q: Who won the cricket world cup?
A: I don't know based on the provided documents.
   refused=True cited=[] llm_calls_so_far=1
```

---

### Q4. Debug - "answers cite the wrong document"

**Problem**

After a "tidy-up" that sorts `store.chunks` by id, the pipeline answers a refund question from the wrong source. Reproduce it, explain the cause, fix it, and write `diagnose(pipe, question, gold_phrase)` that tells you whether a failure is a **retrieval** problem or a **generation** problem.

<details><summary><b>Hint</b></summary>

What is the one invariant of the store? Row `i` of the matrix belongs to `chunks[i]`. Sorting one list but not the other breaks it silently: no exception, just wrong answers.

</details>

<details><summary><b>Solution</b></summary>

```python
# --- Q4: "answers cite the wrong document" - find the bug, add a diagnose() -
def buggy_sort(store):
    """BUG: sorts the chunk list by id (to look tidy) but leaves the matrix rows in the old order."""
    store.chunks.sort(key=lambda c: c.id)


def fixed_sort(store):
    """FIX: permute chunks AND matrix with the same index order (or just never sort)."""
    order = sorted(range(len(store.chunks)), key=lambda i: store.chunks[i].id)
    store.chunks = [store.chunks[i] for i in order]
    store.matrix = store.matrix[order]


def diagnose(pipe, question, gold_phrase):
    hits = retrieve(pipe.store, question, k=pipe.k, min_score=0.0)
    rank = next((h.rank for h in hits if gold_phrase.lower() in h.chunk.text.lower()), None)
    if rank is None:
        return "RETRIEVAL problem: gold chunk not in top-k"
    if gold_phrase.lower() not in pipe.ask(question).text.lower():
        return f"GENERATION problem: gold chunk at rank {rank} but answer lacks it"
    return f"OK: gold chunk at rank {rank}, answer contains it"


def q4():
    q, gold = "How many days do I have to ask for a refund?", "14 days"
    good = VectorStore(); ingest(good, KIRANA_DOCS)
    print("healthy      :", diagnose(RAGPipeline(good, k=1), q, gold))
    broken = VectorStore(); ingest(broken, KIRANA_DOCS)
    buggy_sort(broken)
    print("after bug    :", diagnose(RAGPipeline(broken, k=1), q, gold))
    repaired = VectorStore(); ingest(repaired, KIRANA_DOCS)
    fixed_sort(repaired)
    print("after fix    :", diagnose(RAGPipeline(repaired, k=1), q, gold))
```

</details>

**Why it works / what to learn**
`buggy_sort` reorders only the chunk list, so every score is attached to the wrong text. The fix applies the *same permutation* to both. `diagnose` is the first thing to run on any bad answer: if the gold phrase is not in the retrieved chunks, fix retrieval (chunking, k, filters); if it is there but the answer lacks it, fix the prompt/model. Vector databases hide this bookkeeping, which is one reason to use them past a few thousand chunks.

**Expected output**
```text
healthy      : OK: gold chunk at rank 1, answer contains it
after bug    : RETRIEVAL problem: gold chunk not in top-k
after fix    : OK: gold chunk at rank 1, answer contains it
```

---

### Q5. Production - a mini evaluation harness

**Problem**

Create an evaluation set of 7 questions (5 answerable with a "gold phrase" that must appear in the retrieved chunk, 2 off-topic that must be refused). Write `evaluate(store, k, min_score)` returning `hit@k`, the fraction answered, correct refusals and the number of LLM calls. Sweep `min_score` over 0.0, 0.1, 0.2, 0.3 (and one run with `k=1`). Which threshold would you pick?

<details><summary><b>Hint</b></summary>

A "hit" = the gold phrase is inside at least one retrieved chunk. Off-topic questions are handled correctly when `answer.refused` is true. Count LLM calls from the pipeline counter.

</details>

<details><summary><b>Solution</b></summary>

```python
# --- Q5: a mini evaluation harness: hit@k, refusal accuracy, latency --------
EVAL_SET = [
    # (question, gold phrase that must be in the retrieved chunk, should_answer)
    ("How many days do I have to ask for a refund?", "14 days", True),
    ("What does the Growth plan cost per month?", "1,299", True),
    ("How do I turn on two-factor authentication?", "Settings > Security", True),
    ("When does the export download link expire?", "24 hours", True),
    ("How many backup codes do I get?", "8 backup codes", True),
    ("Who won the cricket world cup?", None, False),
    ("What is the capital of France?", None, False),
]


def evaluate(store, k, min_score):
    pipe = RAGPipeline(store, k=k, min_score=min_score)
    hit, answered_ok, refused_ok = 0, 0, 0
    n_ans = sum(1 for *_, s in EVAL_SET if s)
    for q, gold, should in EVAL_SET:
        a = pipe.ask(q, k=k)
        if should:
            hit += any(gold.lower() in h.chunk.text.lower() for h in a.hits)
            answered_ok += (not a.refused)
        else:
            refused_ok += a.refused
    return {"k": k, "min_score": min_score, "hit@k": round(hit / n_ans, 2), "answered": round(answered_ok / n_ans, 2),
            "correct_refusals": f"{refused_ok}/{len(EVAL_SET) - n_ans}", "llm_calls": pipe.llm_calls}


def q5():
    store = VectorStore(); ingest(store, KIRANA_DOCS)
    for k, ms in [(3, 0.0), (3, 0.1), (3, 0.2), (3, 0.3), (1, 0.1)]:
        print(evaluate(store, k, ms))
```

</details>

**Why it works / what to learn**
`min_score` is a precision/recall dial: at 0.0 every question reaches the LLM (7 calls) and you rely on the prompt to refuse; at 0.3 you save calls but start refusing answerable questions (hit@k falls to 0.6). A threshold of 0.1 gives full hit rate here while saving one LLM call. You must choose it by *looking at scores on known-good and known-bad questions*, never copy it from a tutorial, because scores depend on the model, dimension and corpus. This 7-question set is the seed of the evaluation work in Week 4.

**Expected output** (offline toy embedder)
```text
{'k': 3, 'min_score': 0.0, 'hit@k': 1.0, 'answered': 1.0, 'correct_refusals': '2/2', 'llm_calls': 7}
{'k': 3, 'min_score': 0.1, 'hit@k': 1.0, 'answered': 1.0, 'correct_refusals': '2/2', 'llm_calls': 6}
{'k': 3, 'min_score': 0.2, 'hit@k': 0.8, 'answered': 0.8, 'correct_refusals': '2/2', 'llm_calls': 4}
{'k': 3, 'min_score': 0.3, 'hit@k': 0.6, 'answered': 0.6, 'correct_refusals': '2/2', 'llm_calls': 3}
{'k': 1, 'min_score': 0.1, 'hit@k': 1.0, 'answered': 1.0, 'correct_refusals': '2/2', 'llm_calls': 6}
```
