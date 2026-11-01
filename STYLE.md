# Writer's Brief (internal - read fully before writing any topic)

## Who we write for
Aditya-style learner: a working full-stack dev (PHP/Laravel/MySQL/AWS, ~4 yrs), strong at web backends,
**zero knowledge of ML/RAG**. Goal: after finishing this course they are a top-2% RAG engineer who can
*rebuild every concept from scratch without looking*. So we TEACH, we don't just list.
Beginner-proof: never use a term before defining it in plain words. Map ideas to Laravel/PHP/MySQL
analogies where they genuinely help (Eloquent, migrations, queues, Redis cache, FULLTEXT vs LIKE, RDS...).
No fake analogies just to have one.

## Tech constraints (non-negotiable)
* ONLY free Gemini models via the `google-genai` SDK: generation = `gemini-2.5-flash`,
  embeddings = `gemini-embedding-001` (same as `models/gemini-embedding-001`). NO paid APIs, no OpenAI calls in
  runnable code. OpenAI/others may be *mentioned* as comparisons in prose only.
* ALL runnable code imports the shared helper `common/gemini_utils.py` (read it first!). Use its functions:
  `embed_texts`, `embed_query`, `generate`, `generate_json`, `generate_stream`, `cosine_sim`, `top_k`, `SAMPLE_DOCS`.
  Do not re-implement client setup. Don't edit common/ (if you need a helper, define it locally in the topic file).
* Gemini embedding facts (use exactly): native 3072 dims; Matryoshka (MRL) lets you truncate via
  `output_dimensionality` (recommended 768 / 1536 / 3072); ONLY the 3072 vector is pre-normalised, so truncated
  vectors must be L2-normalised (our helper does it); `task_type` matters (RETRIEVAL_DOCUMENT for chunks,
  RETRIEVAL_QUERY for questions, SEMANTIC_SIMILARITY, CLASSIFICATION, CLUSTERING); ~2048 input-token limit.
  Free tier has rate limits (RPM/RPD) - tell learners to batch + cache, never quote exact numbers (they change).
* Free/local tooling for databases: Chroma (local), Qdrant (`:memory:` or local path, optional docker),
  pgvector (docker `pgvector/pgvector:pg16`), Pinecone (free Starter tier, needs key - always give a graceful
  skip when PINECONE_API_KEY is missing). Rerankers: local sentence-transformers cross-encoder
  (`cross-encoder/ms-marco-MiniLM-L-6-v2`) AND an LLM-as-reranker using gemini-2.5-flash. RAGAs must be configured
  to use Gemini (wrap with the langchain/llamaindex Gemini wrappers) - if you are unsure of a library's exact current
  API, say so in a short "version note" and prefer writing the metric yourself with `generate_json` so it works.
* Never invent API signatures. If unsure, keep to the common helper, numpy, and stable well-known library calls.

## Folder contract - for EVERY topic folder you create
```
<week-folder>/<NN-topic-slug>/
    notes.md       # the full beginner-to-pro lesson
    practice.md    # exactly 5 questions, each fully SOLVED (inline)
    solutions.py   # all 5 solutions as runnable code: q1()...q5(), `if __name__ == "__main__"` runs them
```
`solutions.py` must start with:
```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))
from gemini_utils import *   # or explicit imports
```
(depth is rag-mastery/<week>/<topic>/solutions.py).

## notes.md - required structure (use these exact H2 headings, in this order)
1. `## 🎯 What you will learn` - 4-6 outcome bullets ("you will be able to ...")
2. `## 🧠 The big idea in plain words` - zero jargon, a concrete story/example, one analogy to Laravel/MySQL if it fits
3. `## 📚 Concepts, step by step` - the real teaching. Many H3s. Define every term. Small examples with actual numbers
   you can follow by hand. ASCII diagrams / tables where they clarify. Build from first principles to the real thing.
4. `## 💻 Code walkthrough` - complete runnable Gemini-free-tier code, built up in stages with line-level explanation of
   WHY. Show expected output (real shape/behaviour; label approximate numbers as "≈").
5. `## ⚠️ Common mistakes & debugging` - table: Symptom | Cause | Fix
6. `## 🚀 Pro-level insights (what the top 2% know)` - production trade-offs, numbers worth remembering, when NOT to use it,
   cost/latency/quality levers, failure modes, how it is evaluated
7. `## 🧪 Mini lab` - a 10-15 minute hands-on exercise (different from the practice questions)
8. `## ✅ Recap checklist` - checkbox list the learner ticks to confirm mastery
9. `## 🎤 Interview questions` - 5 short Q&As with model answers
10. `## 🔗 Next` - one line pointing to the next topic and what it unlocks

Length: 2,500-4,500 words per notes.md. Depth over breadth: a reader should NOT need another tutorial.
Tone: friendly, direct, precise. Use correct formatting (fenced code with language tags, tables, no walls of text).
Do not pad. Every code block must be correct, copy-paste runnable after `pip install -r requirements.txt` and a .env key.

## practice.md - required structure
Title, then 5 questions with rising difficulty:
Q1 Warm-up (understand) · Q2 Apply · Q3 Build · Q4 Debug/Analyse (a broken snippet or surprising result to explain) ·
Q5 Production challenge (realistic, mini-project flavour).
For each question: `### Q1. <title>` → **Problem** → **Hint** (collapsed in <details>) → **Solution** (full code + explanation,
in <details> or plainly) → **Why it works / what to learn** (2-4 lines) → **Expected output** (≈).
Questions must be genuinely DIFFERENT from the notes' examples. Solutions in practice.md must match solutions.py.

## Verification you MUST do before finishing
* `python -m py_compile` on every solutions.py.
* Where logic is pure (numpy, chunking, BM25, RRF, math), actually RUN it offline. For Gemini-dependent paths you cannot call
  the API (no key here): test with a deterministic fake by monkeypatching `embed_texts`/`generate`, or at least read the code
  twice against common/gemini_utils.py signatures.
* Every folder has all 3 files. No placeholder text like TODO/lorem/"left as an exercise" in solutions.
* Do NOT touch folders outside your assignment. Don't create READMEs for weeks (the lead does that).

## Final reply to lead
Max 120 words: list of folders written, any version caveats you were unsure about, anything not verifiable offline.
