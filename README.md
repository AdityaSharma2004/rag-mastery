# RAG Mastery - a 4-week, hands-on course (free Gemini models only)

Learn Retrieval-Augmented Generation from zero to production-minded, with Python. Built for a web developer (PHP/Laravel/MySQL) who is new to ML: every idea is explained in plain words, mapped to things you already know, and then **built from scratch** so you can rebuild it without looking.

**Models used (both free tier):** generation `gemini-2.5-flash` · embeddings `models/gemini-embedding-001`.

## How each topic is organised

Every topic folder has exactly three files:

| File | What it is |
|---|---|
| `notes.md` | The full lesson: big idea, concepts, code walkthrough, mistakes table, pro insights, mini lab, recap checklist, interview Q&A |
| `practice.md` | 5 solved questions of rising difficulty (warm-up → apply → build → debug → production), each with hint, full solution, explanation and expected output |
| `solutions.py` | The same 5 solutions as runnable code (`q1()` ... `q5()`); `python solutions.py` runs them all |

**Suggested method:** read `notes.md` → try each practice question *before* opening the hint → compare with the solution → do the mini lab → tick the checklist → answer the interview questions aloud.

## Setup (5 minutes)

```bash
pip install -r requirements.txt
cp .env.example .env          # paste your free key from https://aistudio.google.com/apikey
python common/gemini_utils.py # smoke test (needs the key)
cd week-1-embeddings-and-similarity-search/01-vector-embeddings-from-text-to-numbers
python solutions.py
```
All `solutions.py` files add `common/` to the path themselves, so you can run them from their own folder.

**Offline mode.** Weeks 3 and 4 run *without any API key*: where a Gemini call would happen, a small deterministic stand-in is used (marked clearly in each file), so you can watch the mechanics. With `GEMINI_API_KEY` set, the same code uses the real models, and printed numbers then differ from the expected outputs (which come from the offline run and are marked as such). Weeks 1 and 2 call the embedding API (results are cached on disk; free tier has rate limits, the helper retries and batches).

## Course map

### Week 1 - Embeddings & similarity search
1. `01-vector-embeddings-from-text-to-numbers`
2. `02-similarity-metrics-cosine-dot-euclidean`
3. `03-generating-embeddings-with-gemini`
4. `04-vectors-in-numpy-and-similarity-search`
5. `05-semantic-vs-keyword-vs-hybrid-search`
6. `06-embedding-dimensionality-tradeoffs`

### Week 2 - Vector databases
1. `01-chromadb` 2. `02-pinecone` 3. `03-pgvector` 4. `04-qdrant`
5. `05-crud-namespaces-and-metadata-filtering`
6. `06-hybrid-search-bm25-plus-vector`
7. `07-reranking-with-cross-encoders`

### Week 3 - The RAG pipeline
1. `01-document-loading` (PDF, DOCX, HTML, CSV, cleaning, incremental ingestion)
2. `02-text-chunking-strategies`
3. `03-chunk-overlap`
4. `04-full-rag-pipeline`
5. `05-prompt-stuffing-and-grounding` (citations, token budgets, prompt injection)
6. `06-metadata-filtering-in-rag` (multi-tenancy, ACLs, leak tests)
7. `07-handling-long-documents`

### Week 4 - Advanced RAG
1. `01-query-expansion` 2. `02-hyde` 3. `03-parent-child-chunking`
4. `04-rag-evaluation-with-ragas` (metrics written by hand with an LLM judge; notes explain the `ragas` library)
5. `05-caching-embeddings-and-retrieval` (Redis / fakeredis / dict)
6. `06-multimodal-rag`
7. `07-llamaindex`

## Capstone (do this last)

From a blank folder, without looking: load a set of PDFs → clean and chunk (parent-child) → embed with `gemini-embedding-001` (cached) → store in a vector database with tenant and ACL metadata → retrieve with hybrid search + reranking and query expansion → answer with grounded, cited prompts using `gemini-2.5-flash` → gate every change with an evaluation harness. If you can build it and explain each choice, you are in the top tier of RAG engineers.

## What was verified, and what was not

- Every `solutions.py` compiles; Week 3 and 4 solutions were executed offline and the expected outputs in `practice.md` were checked against real runs.
- Week 2 was tested against real Chroma, Qdrant and PostgreSQL 16 + pgvector (HNSW). Pinecone was tested only through an offline fake (it needs an account and key).
- **Not run in the authoring environment (no API key / no network):** real Gemini embedding and generation calls (including the vision call in Week 4 · 06 and the `GoogleGenAI` / `GoogleGenAIEmbedding` constructors in Week 4 · 07), real Pinecone, and the Hugging Face cross-encoder download (a stub is used when it cannot download). Those code paths were reviewed against the installed SDK signatures. Exact similarity scores with real embeddings will differ from the offline numbers.
- Free-tier rate limits change; the course never relies on exact quota numbers.

`STYLE.md` is the internal writer brief used to keep the topics consistent; you can ignore or delete it.
