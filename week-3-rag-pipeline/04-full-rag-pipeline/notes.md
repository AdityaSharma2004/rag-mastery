# 04 - The Full RAG Pipeline (from scratch)

> Week 3 · Topic 4 of 7 · Prerequisite: loading, chunking, overlap (topics 01-03) and the embeddings/vector-search work from weeks 1-2.

## 🎯 What you will learn

- You will be able to build a complete RAG system (ingest, retrieve, generate) from scratch in about 150 lines, with no framework.
- You will be able to explain, step by step, what happens between "user types a question" and "answer appears", including which embedding `task_type` is used at each side.
- You will be able to return answers with verifiable citations and stream them token by token.
- You will be able to expose the pipeline as a small FastAPI service with a normal request/response endpoint and a streaming endpoint.
- You will be able to log every retrieval so that any bad answer can be debugged in two minutes.
- You will be able to decide, for any bad answer, whether it is a **retrieval problem** or a **generation problem**, which is the single most valuable RAG debugging skill.

---

## 🧠 The big idea in plain words

An LLM like `gemini-2.5-flash` knows a lot about the world, but it has never read **your** documents: your company's refund policy, your client's ERP manual, last month's invoices. If you ask it about them, it either says "I don't know" or, worse, invents a confident answer.

RAG (Retrieval-Augmented Generation) fixes this with a trick that is almost embarrassingly simple:

1. **Before** the LLM answers, go and *find* the few paragraphs of your documents that are relevant to the question.
2. **Paste** those paragraphs into the prompt, next to the question.
3. Tell the LLM: "answer using only these paragraphs".

That's it. The LLM is not retrained. It just gets an open-book exam instead of a closed-book exam.

**Story.** A customer of a small accounting app asks: *"Can I get my money back after 3 weeks?"*

- A closed-book LLM guesses: "Most companies offer 30-day refunds..." (wrong for *this* company).
- A RAG system searches the help-center, finds the paragraph "Refunds are possible within 14 days...", puts it in the prompt, and the LLM answers: "No, refunds are only possible within 14 days of purchase [1]."

**Laravel analogy (the useful one).** Think of a controller action:

```
Request  ->  Controller  ->  Repository (MySQL query)  ->  View/Blade (template + data)  ->  Response
```

RAG has the same shape:

```
Question ->  Retriever   ->  Vector store (similarity search) ->  Prompt template (+ chunks) ->  LLM -> Answer
```

The vector store is your repository/database, the retriever is the query, the prompt template is the Blade view that renders the data, and the LLM is the final "render" step. If a Laravel page shows wrong data, you do not blame Blade first, you `dd()` the query result. RAG debugging is the same: **look at what was retrieved before you blame the model**.

---

## 📚 Concepts, step by step

### 1. The two phases: offline ingest and online query

RAG has two completely separate lifecycles. Mixing them up is the most common beginner confusion.

```
OFFLINE  (run once, or when documents change  ~ like `php artisan migrate && db:seed`)
  documents -> split into chunks -> embed each chunk (RETRIEVAL_DOCUMENT) -> store vectors + text + metadata

ONLINE   (run on every user request  ~ like a controller action)
  question -> embed (RETRIEVAL_QUERY) -> find top-k similar chunks -> build prompt -> LLM -> answer + citations
```

- **Ingest** is slow and can be done in the background (a queued job in Laravel terms). Cost: one embedding per chunk.
- **Query** must be fast. Cost: one embedding for the question, one search, one LLM call.

### 2. The five online steps in detail

| # | Step | What happens | Typical cost (≈) |
|---|------|--------------|-------------------|
| 1 | Embed the question | `embed_query(q)` calls `gemini-embedding-001` with `task_type="RETRIEVAL_QUERY"` | 100-400 ms, 1 API call |
| 2 | Vector search | dot product of the query vector against all stored chunk vectors, sort, keep the top-k | < 5 ms for thousands of chunks |
| 3 | Retrieve | map the winning row numbers back to the chunk text + source | ~0 ms |
| 4 | Build the prompt | system rules + numbered sources + the question | ~0 ms |
| 5 | Generate | `gemini-2.5-flash` writes the answer | 1-5 s, 1 API call |

Notice that **generation dominates latency** and the search is nearly free. That is why we care so much about not calling the LLM when retrieval found nothing useful (see "no-answer gate" below).

### 3. Why the two `task_type`s differ

Gemini's embedding model gives a *different* vector depending on why you embed:

- Chunks (the "documents") are embedded with `RETRIEVAL_DOCUMENT`.
- Questions are embedded with `RETRIEVAL_QUERY`.

Why? A question ("how do I get a refund?") and its answer ("Refunds are possible within 14 days...") look very different as text. The model was trained so that a *query-flavoured* vector lands close to the *document-flavoured* vector that answers it. If you embed both sides with the same task type, search still works, but recall is measurably worse. The helper `embed_texts()` defaults to `RETRIEVAL_DOCUMENT`; `embed_query()` sets `RETRIEVAL_QUERY` for you.

Also remember from week 1: the helper returns **unit vectors** (L2-normalised). For unit vectors, `dot product == cosine similarity`, so search is just `matrix @ query_vector`. Both sides must use the **same dimension** (we use the helper's default 768).

### 4. Worked example with numbers (follow it by hand)

Pretend our embedding has only 3 dimensions and we stored three chunks:

| chunk | text | vector |
|-------|------|--------|
| A | "Refunds within 14 days" | (0.9, 0.1, 0.4) |
| B | "Plans cost 499 / 1299" | (0.1, 0.9, 0.4) |
| C | "Enable 2FA in settings" | (0.4, 0.1, 0.9) |

(Real vectors are unit length; we skip normalising to keep numbers small.) The question "money back after 3 weeks?" embeds to `(0.8, 0.2, 0.5)`.

Dot products: A = 0.72+0.02+0.20 = **0.94**, B = 0.08+0.18+0.20 = 0.46, C = 0.32+0.02+0.45 = 0.79.

Ranking: A (0.94), C (0.79), B (0.46). With `k=2`, we retrieve A and C. Notice C is *not* about refunds, it just "points in a similar direction". Vector search returns the **closest** chunks, not necessarily **relevant** ones. That is why the LLM must be told "if the sources don't contain the answer, say so", and why we may add a minimum-score gate.

### 5. Choosing `k` (how many chunks to retrieve)

- Too small (k=1): you miss answers that live in a neighbouring chunk or in a second document.
- Too large (k=20): you stuff noise into the prompt, cost goes up, and the model can get distracted (more on this in topic 05).
- Good defaults: **k = 3 to 5** for chunks of 200-500 tokens. Raise k only if your evaluation shows the right chunk is often at rank 6-10 (and then consider a re-ranker in week 4 instead).

### 6. Citations: making answers checkable

We number the sources `[1]`, `[2]`, ... in the prompt and ask the model to write the number after each fact. After generation we **parse** the numbers with a regex and map them back to real sources:

```
LLM says:   "Refunds are possible within 14 days [1]. Money arrives in 5-7 working days [1]."
We parse:   {1}  ->  refunds.md, chunk r-0, score 0.83
```

Two rules for trustworthy citations:
1. The *model* only writes numbers. **Your code** turns them into file names/URLs, so it can never invent a source name.
2. Validate: a number that is not in `1..len(hits)` is a hallucinated citation. Drop it and log it.

### 7. Streaming

`generate()` waits for the whole answer, which feels slow (2-5 s of blank screen). `generate_stream()` yields small pieces of text as they are produced. In a web app you send those pieces to the browser immediately (like Laravel's `response()->stream()` / Server-Sent Events). The user sees the first words in under a second. Streaming changes *perceived* latency, not total latency. Citations can be appended once the stream ends, because that is when we know which numbers the model used.

### 8. The debugging mindset: retrieval problem or generation problem?

Every bad RAG answer has exactly one of two root causes (sometimes both):

| | Gold chunk was retrieved | Gold chunk was NOT retrieved |
|---|---|---|
| **Answer is correct** | Healthy | Lucky / model used its own memory (dangerous!) |
| **Answer is wrong** | **Generation problem**: prompt, model, too much noise, wrong instructions | **Retrieval problem**: chunking, embeddings, k, filters, query wording |

("Gold chunk" = the chunk that actually contains the right answer.)

The 2-minute procedure:

1. **Log** the question and the retrieved chunks with their scores (we build this below).
2. **Look**: is the gold chunk in the list? Search the text for the key fact.
3. Not there? It's retrieval. Check, in order: was the document ingested at all? is the fact split across two chunks? is the query phrased very differently from the document? is `k` too small? is a filter hiding it?
4. There? It's generation. Check: is the instruction clear? is the chunk buried under irrelevant ones? did the prompt get truncated? is the question ambiguous?

This is the RAG equivalent of `dd($query->toSql(), $rows)` before touching the Blade file. Teams that skip it spend days "tweaking the prompt" for what is really a chunking bug.

---

## 💻 Code walkthrough

We build one file, `mini_rag.py`, in stages. The sample company is a fictional accounting app called **KiranaBooks**. All blocks below belong to the same file, in order. You need `pip install -r requirements.txt` and a `GEMINI_API_KEY` in `.env`.

### Stage 1 - Imports, logging, data shapes

```python
import sys, pathlib, time, json, logging, re
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))
from gemini_utils import embed_texts, embed_query, generate, generate_stream, top_k

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s | %(message)s")
log = logging.getLogger("rag")


@dataclass
class Chunk:
    """One searchable piece of a document. Like an Eloquent row: id, body, and a meta bag."""
    id: str
    text: str
    source: str                       # file name or URL, shown to the user in citations
    meta: dict = field(default_factory=dict)


@dataclass
class Hit:
    """A chunk that the retriever returned, plus how well it matched."""
    chunk: Chunk
    score: float                      # cosine similarity, higher = closer
    rank: int                         # 1 = best
```

Why dataclasses? They give us typed, printable records for free, so logs and tests are readable. We separate `Chunk` (what is stored) from `Hit` (what a *search* returned) because the score only exists in the context of one question.

### Stage 2 - Chunking and the vector store

```python
def chunk_text(text: str, max_chars: int = 300) -> list[str]:
    """Pack whole sentences into chunks of at most ~max_chars (topics 02-03 go deeper)."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, current = [], ""
    for s in sentences:
        if current and len(current) + len(s) + 1 > max_chars:
            chunks.append(current)
            current = s
        else:
            current = f"{current} {s}".strip()
    if current:
        chunks.append(current)
    return chunks


class VectorStore:
    """The smallest useful vector database: a list of chunks + one numpy matrix."""

    def __init__(self):
        self.chunks: list[Chunk] = []
        self.matrix = np.zeros((0, 0), dtype=np.float32)   # shape (n_chunks, dim)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        assert len(chunks) == len(vectors), "one vector per chunk"
        self.matrix = vectors if len(self.chunks) == 0 else np.vstack([self.matrix, vectors])
        self.chunks.extend(chunks)

    def __len__(self) -> int:
        return len(self.chunks)
```

The store is deliberately dumb: row `i` of `matrix` is the vector for `chunks[i]`. That row-alignment is the *only* invariant, and it is the source of the classic bug "answers come from the wrong document" if you ever reorder one list and not the other. (Chroma/Qdrant/pgvector do this bookkeeping for you; we use numpy to see the machinery. Up to ~100k chunks a numpy matrix is perfectly fast.)

### Stage 3 - Ingestor (offline phase)

```python
class Ingestor:
    def __init__(self, store: VectorStore):
        self.store = store

    def ingest(self, docs: dict[str, str]) -> int:
        """docs = {source_name: full_text}. Returns number of chunks stored."""
        chunks: list[Chunk] = []
        for source, text in docs.items():
            for i, piece in enumerate(chunk_text(text)):
                chunks.append(Chunk(id=f"{source}#{i}", text=piece, source=source, meta={"pos": i}))
        # ONE batched call; chunks are 'documents' so task_type = RETRIEVAL_DOCUMENT
        vectors = embed_texts([c.text for c in chunks], task_type="RETRIEVAL_DOCUMENT")
        self.store.add(chunks, vectors)
        log.info("ingested %d chunks from %d documents", len(chunks), len(docs))
        return len(chunks)
```

`embed_texts` batches and caches on disk, so re-running costs zero API calls, the same idea as caching an expensive query in Redis. Notice that **ingestion never touches the LLM**.

### Stage 4 - Retriever (online phase, step 1-3)

```python
class Retriever:
    def __init__(self, store: VectorStore, k: int = 4, min_score: float = 0.0):
        self.store, self.k, self.min_score = store, k, min_score

    def retrieve(self, question: str) -> list[Hit]:
        qvec = embed_query(question)                               # RETRIEVAL_QUERY
        pairs = top_k(qvec, self.store.matrix, k=self.k)           # [(row_index, score), ...] best first
        hits = [
            Hit(self.store.chunks[i], score, rank)
            for rank, (i, score) in enumerate(pairs, start=1)
            if score >= self.min_score
        ]
        # The debugging log: THE most useful 5 lines in the whole project.
        log.info("RETRIEVE q=%r k=%d returned=%d", question, self.k, len(hits))
        for h in hits:
            log.info("  #%d score=%.3f %s | %s", h.rank, h.score, h.chunk.id, h.chunk.text[:70])
        return hits
```

`top_k` comes from the shared helper and returns `(row_index, score)` pairs, best first. We convert row numbers back into `Chunk`s here, and apply `min_score` so very weak matches are dropped. Choose `min_score` by *looking at your logs* for known-bad questions; never copy a number from a blog post (it depends on dimension, model and domain).

### Stage 5 - Generator (online phase, step 4-5)

```python
REFUSAL = "I don't know based on the provided documents."

SYSTEM = (
    "You are a support assistant. Answer the question using ONLY the numbered sources provided.\n"
    "After every fact, cite the source number like [1]. Several sources: [1][2].\n"
    f'If the sources do not contain the answer, reply exactly: "{REFUSAL}"\n'
    "Be concise. Never use outside knowledge."
)


def build_prompt(question: str, hits: list[Hit]) -> str:
    blocks = [f"[{h.rank}] (source: {h.chunk.source})\n{h.chunk.text}" for h in hits]
    return "SOURCES:\n" + "\n\n".join(blocks) + f"\n\nQUESTION: {question}\nANSWER:"


CITE = re.compile(r"\[(\d+)\]")


def extract_citations(answer: str, hits: list[Hit]) -> list[dict]:
    """Turn '[1][3]' in the answer into real sources. Invented numbers are dropped and logged."""
    used = sorted({int(n) for n in CITE.findall(answer)})
    valid = [n for n in used if 1 <= n <= len(hits)]
    if len(valid) != len(used):
        log.warning("model cited non-existent sources: %s", sorted(set(used) - set(valid)))
    return [
        {"n": n, "source": hits[n - 1].chunk.source, "chunk_id": hits[n - 1].chunk.id,
         "score": round(hits[n - 1].score, 3)}
        for n in valid
    ]
```

The system prompt does four jobs: restricts knowledge to the sources, demands citations, defines the exact refusal sentence (so code can detect it), and keeps the answer short. Topic 05 goes much deeper on prompt design. Because hits are numbered by rank, `hits[n - 1]` is the chunk the model meant.

### Stage 6 - The pipeline class (glue)

```python
@dataclass
class Answer:
    question: str
    text: str
    citations: list[dict]
    hits: list[Hit]
    timings_ms: dict
    refused: bool = False


class RAGPipeline:
    def __init__(self, store: VectorStore, k: int = 4, min_score: float = 0.0):
        self.retriever = Retriever(store, k=k, min_score=min_score)

    def ask(self, question: str) -> Answer:
        t0 = time.perf_counter()
        hits = self.retriever.retrieve(question)
        t1 = time.perf_counter()

        if not hits:                                     # no-answer gate: skip the LLM, save time + quota
            return Answer(question, REFUSAL, [], [], {"retrieve": round((t1 - t0) * 1000)}, refused=True)

        text = generate(build_prompt(question, hits), system=SYSTEM, temperature=0.1)
        t2 = time.perf_counter()

        ans = Answer(
            question=question,
            text=text.strip(),
            citations=extract_citations(text, hits),
            hits=hits,
            timings_ms={"retrieve": round((t1 - t0) * 1000), "generate": round((t2 - t1) * 1000)},
            refused=REFUSAL in text,
        )
        log.info("TRACE %s", json.dumps({
            "q": question, "refused": ans.refused, "cited": [c["n"] for c in ans.citations],
            "chunks": [h.chunk.id for h in hits], "ms": ans.timings_ms,
        }))
        return ans

    def ask_stream(self, question: str):
        """Yield text pieces as they arrive; finish with a 'Sources:' footer built by OUR code."""
        hits = self.retriever.retrieve(question)
        if not hits:
            yield REFUSAL
            return
        pieces = []
        for piece in generate_stream(build_prompt(question, hits), system=SYSTEM, temperature=0.1):
            pieces.append(piece)
            yield piece
        cites = extract_citations("".join(pieces), hits)
        if cites:
            yield "\n\nSources: " + ", ".join(f"[{c['n']}] {c['source']}" for c in cites)
```

`temperature=0.1` keeps answers factual and repeatable. The one-line `TRACE` log (JSON) is what you would ship to your log aggregator: one searchable record per request, exactly like a Laravel request log.

### Stage 7 - Run it

```python
KIRANA_DOCS = {
    "refunds.md": (
        "KiranaBooks refunds a subscription within 14 days of purchase, provided you have not exported "
        "more than 50 invoices. After 14 days payments are non-refundable, but you can cancel at any time "
        "to stop the next renewal. Approved refunds reach your bank account in 5 to 7 working days."
    ),
    "plans.md": (
        "The Starter plan costs 499 rupees per month and includes one user and 100 invoices per month. "
        "The Growth plan costs 1,299 rupees per month and includes five users and unlimited invoices. "
        "Choosing annual billing gives you two months free."
    ),
    "security.md": (
        "You can enable two-factor authentication under Settings > Security. Admins on the Growth plan "
        "can force two-factor authentication for every user. If you lose your phone, use one of the 8 "
        "backup codes that were shown when you first set it up."
    ),
    "export.md": (
        "Owners can export all company data as a ZIP of CSV files from Settings > Data. Exports are ready "
        "within 10 minutes and the download link expires after 24 hours. Backups are kept for 30 days."
    ),
}


def build_pipeline() -> RAGPipeline:
    store = VectorStore()
    Ingestor(store).ingest(KIRANA_DOCS)
    return RAGPipeline(store, k=3)


if __name__ == "__main__":
    pipe = build_pipeline()
    for q in ["Can I get my money back after 3 weeks?", "How long do exports stay downloadable?"]:
        a = pipe.ask(q)
        print("\nQ:", q, "\nA:", a.text, "\nCitations:", a.citations, "\nTimings:", a.timings_ms)
```

Expected output (scores and wording ≈):

```
rag | ingested 4 chunks from 4 documents      # each doc fits in 1-2 chunks, so 4-8 in total
rag | RETRIEVE q='Can I get my money back after 3 weeks?' k=3 returned=3
rag |   #1 score=0.7x refunds.md#0 | KiranaBooks refunds a subscription within 14 days of purchase, ...
rag |   #2 score=0.5x plans.md#0 | ...
Q: Can I get my money back after 3 weeks?
A: No. Refunds are only possible within 14 days of purchase; after that you can only cancel to stop renewal [1].
Citations: [{'n': 1, 'source': 'refunds.md', 'chunk_id': 'refunds.md#0', 'score': 0.7..}]
Timings: {'retrieve': ~250, 'generate': ~1500}
```

### Stage 8 - A debugging helper: retrieval or generation?

```python
def diagnose(pipe: RAGPipeline, question: str, gold_phrase: str) -> str:
    """gold_phrase = a short string that MUST appear in the chunk holding the answer (e.g. '14 days')."""
    hits = pipe.retriever.retrieve(question)
    gold_rank = next((h.rank for h in hits if gold_phrase.lower() in h.chunk.text.lower()), None)
    if gold_rank is None:
        return "RETRIEVAL problem: the gold chunk was not in the top-k. Check chunking, k, query wording, filters."
    answer = pipe.ask(question).text
    if gold_phrase.lower() not in answer.lower():
        return f"GENERATION problem: gold chunk was retrieved at rank {gold_rank} but the answer lacks it."
    return f"OK: gold chunk at rank {gold_rank} and the answer contains it."
```

Run `diagnose(pipe, "How long is the refund window?", "14 days")` across 20-30 known questions and you have the seed of an evaluation set (week 4 formalises this).

### Stage 9 - A tiny FastAPI service

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="KiranaBooks RAG")
_pipe: RAGPipeline | None = None


def get_pipe() -> RAGPipeline:
    """Lazy singleton: build + ingest once, reuse for every request (like a service-container binding)."""
    global _pipe
    if _pipe is None:
        _pipe = build_pipeline()
    return _pipe


class AskBody(BaseModel):
    question: str
    k: int = 3


@app.post("/ask")
def ask(body: AskBody):
    pipe = get_pipe()
    pipe.retriever.k = body.k
    a = pipe.ask(body.question)
    return {
        "answer": a.text,
        "refused": a.refused,
        "citations": a.citations,
        "retrieved": [{"rank": h.rank, "score": round(h.score, 3), "chunk_id": h.chunk.id} for h in a.hits],
        "timings_ms": a.timings_ms,
    }


@app.get("/ask/stream")
def ask_stream(q: str):
    return StreamingResponse(get_pipe().ask_stream(q), media_type="text/plain")
```

Run: `uvicorn mini_rag:app --reload`, then:

```bash
curl -s localhost:8000/ask -H "Content-Type: application/json" -d '{"question":"What does Growth cost?"}'
curl -N "localhost:8000/ask/stream?q=What%20does%20Growth%20cost%3F"     # -N = don't buffer; watch tokens arrive
```

The response includes `retrieved`, the ranked chunk ids and scores. Returning this in development (hide it in production if the sources are sensitive) lets you debug from the browser's network tab. Note `pipe.retriever.k = body.k` mutates a shared object, which is fine for a demo but not thread-safe; production code passes `k` as an argument (see practice Q5).

> Version note: `generate_content_stream` and the `google-genai` SDK evolve quickly; we only use the helper's `generate_stream`, so SDK changes stay in one file.

---

## ⚠️ Common mistakes & debugging

| Symptom | Cause | Fix |
|---------|-------|-----|
| Answers cite the wrong document | `chunks` list and `matrix` rows got out of sync (re-ordered or filtered one of them) | Keep one structure; assert `len(chunks) == len(vectors)`; never sort one without the other |
| Search quality is mediocre on every question | Query embedded with `RETRIEVAL_DOCUMENT` (or chunks with `RETRIEVAL_QUERY`) | Documents: `RETRIEVAL_DOCUMENT`. Questions: `RETRIEVAL_QUERY` (use `embed_query`) |
| Best chunk is never returned, the worst is | `np.argsort(scores)` (ascending) instead of `np.argsort(-scores)` | Use the helper `top_k`; it sorts best-first |
| `ValueError: shapes not aligned` | Index built with `dim=768`, query embedded with another `dim` | Use the same `dim` everywhere (helper default) |
| Model says "I don't know" though the fact is in the docs | Fact was never retrieved (retrieval problem), or was cut off by `[:300]`-style truncation | Print the hits; run `diagnose`; fix chunking/k |
| Model answers from its own memory, ignoring sources | Prompt does not forbid outside knowledge, or no refusal sentence | Strict system prompt + exact refusal text (topic 05) |
| Invented citations like `[7]` with 3 sources | Model hallucinated a number | `extract_citations` validates the range; drop + log |
| First request is slow, later ones fast | Index built lazily on first call | Build at startup (lifespan hook) or in a worker |
| `429` errors while ingesting | Free-tier embedding rate limit | Batch (helper does), cache (helper does), ingest in the background, retry |
| Whole answer arrives at once even with streaming | Proxy/Nginx buffering | `curl -N` to test; set `X-Accel-Buffering: no` header behind Nginx |

---

## 🚀 Pro-level insights (what the top 2% know)

- **Log retrieval for every request.** The cheapest, highest-value habit in RAG. Store question, chunk ids, scores, answer, latency. When a user complains, you replay the exact retrieval. Without this you are guessing.
- **Measure retrieval separately from generation.** Retrieval metrics (hit-rate@k, recall@k, MRR) need no LLM and are free to run on every commit. A golden set of 30 questions, each with the id of its gold chunk, catches most regressions. Teams that only eyeball final answers cannot tell which half broke.
- **Most RAG failures (roughly 70-80% in practice reports) are retrieval failures**, not model failures. Upgrading the LLM rarely fixes a chunking problem.
- **The no-answer gate saves money and prevents hallucinations.** If the best score is far below what known-good questions score, do not call the LLM at all. Calibrate the threshold on real examples, per embedding model and dimension.
- **Latency budget.** Embedding the question (~100-400 ms) + search (ms) + first token (~0.5-1.5 s). Stream so the user sees progress; cache embeddings of repeated questions; keep chunks small enough that the prompt stays short.
- **Free-tier discipline.** One user question = 2 API calls (embed + generate). Ingestion = ceil(chunks / batch) calls. Cache embeddings on disk (the helper does) and never re-embed unchanged documents; store a content hash per document and skip if it matches (like Laravel's `updateOrCreate` with a checksum column).
- **Answer vs. sources trust.** Show citations in the UI. Users forgive "I don't know" far more than a confident wrong answer.
- **When NOT to use RAG:** when the whole corpus fits comfortably in the prompt and changes rarely (see topic 07), when questions need aggregation across all rows ("how many invoices were over 10k?" is SQL, not RAG), or when you need exact lookups (use a normal database query).
- **Swap-ability.** Because ingest / retrieve / generate are separate classes, you can replace numpy with Chroma/Qdrant/pgvector, add BM25 hybrid search, or add a re-ranker by changing only the retriever. This separation is the real lesson of this topic.
- **Evaluate with fixed seeds and low temperature**; LLM output varies, so judge on many questions, not one.

---

## 🧪 Mini lab (10-15 minutes)

Goal: *feel* the difference between retrieval and generation failures.

1. Run `mini_rag.py` as is and ask "How long do exports stay downloadable?". Confirm the log shows `export.md` at rank 1.
2. Add a fifth document to `KIRANA_DOCS`, e.g. `"invoicing.md"`, containing one sentence with a distinctive fact ("Invoice numbers restart every April 1 and look like KB/2025-26/0001.").
3. Ask "When do invoice numbers reset?" and call `diagnose(pipe, question, "April 1")`. It should say OK.
4. **Break retrieval on purpose:** set `k=1` and ask a question whose answer needs two documents ("Which plan lets admins force 2FA and how much is it?"). Run `diagnose` for each gold phrase. Which one fails and why?
5. **Break generation on purpose:** keep `k=3` but edit `SYSTEM` to delete the line "Never use outside knowledge." and ask something outside the documents ("What is GST?"). Does the model answer from memory? Restore the line and compare.
6. Write down in one sentence which log line told you each failure type.

---

## ✅ Recap checklist

- [ ] I can draw the offline (ingest) and online (query) pipelines from memory.
- [ ] I know which `task_type` is used for chunks and which for questions, and why.
- [ ] I can explain why dot product equals cosine similarity for unit vectors.
- [ ] I can write `top_k` retrieval with numpy and the helper without looking.
- [ ] My prompt numbers sources, forbids outside knowledge and defines an exact refusal sentence.
- [ ] My code (not the model) maps citation numbers to real sources and drops invalid ones.
- [ ] I can stream tokens and append a sources footer afterwards.
- [ ] I log question, chunk ids, scores and timings for every request.
- [ ] Given a wrong answer, I can say in under two minutes whether retrieval or generation caused it.
- [ ] I can serve the pipeline with FastAPI and test it with `curl`.

---

## 🎤 Interview questions

**1. Walk me through what happens when a user asks a question in a RAG system.**
Embed the question with the query task type, search the vector store for the top-k chunks, build a prompt containing numbered chunks and the question, call the LLM with instructions to answer only from the sources, then parse citations and return the answer plus sources. Ingestion (chunk, embed, store) happened earlier and separately.

**2. The answer is wrong. How do you debug it?**
Look at the logged retrieved chunks. If the chunk containing the right fact is missing, it is a retrieval problem (chunking, k, embeddings, filters, query phrasing). If it is present but the answer is wrong, it is a generation problem (prompt, noise, truncation, ambiguity). Fix the right half; measure with a golden question set.

**3. Why use different task types for documents and queries?**
Questions and answers are phrased differently. Task-specific embeddings are trained so a query vector lands near the vectors of the passages that answer it, which improves recall compared with embedding both sides identically.

**4. How do you make citations trustworthy?**
Number the sources in the prompt, have the model emit only numbers, and let application code translate numbers into sources, validating that each number exists. Optionally verify the cited chunk actually supports the sentence (an LLM or NLI check).

**5. Why stream, and what does it not fix?**
Streaming reduces time-to-first-token so the UI feels responsive. It does not reduce total generation time, cost, or retrieval latency, and anything that needs the full answer (citation validation) must run after the stream ends.

---

## 🔗 Next

Next: **05 - Prompt stuffing & grounding**, where you learn exactly how to put retrieved chunks into the prompt (templates, delimiters, ordering, token budgets) so the model stays faithful and resists prompt injection.
