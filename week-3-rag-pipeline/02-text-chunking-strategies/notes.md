# 02 - Text chunking strategies

> Week 3 · Topic 2 of 7 · Time: ~90 min · Runs offline (pure Python) · Next to embeddings, chunking is the biggest quality lever in RAG

## 🎯 What you will learn

- You will be able to explain **why** documents must be split into chunks and what goes wrong with chunks that are too big or too small.
- You will be able to implement **fixed-size, recursive, sentence-based and semantic** chunking from scratch.
- You will be able to carry **metadata, offsets and deterministic ids** onto every chunk so answers can cite sources.
- You will be able to **choose a chunk size with a retrieval benchmark**, not by guessing.
- You will be able to spot and fix classic chunker bugs (infinite loops, dropped tails, orphaned headings).

## 🧠 The big idea in plain words

An embedding turns a piece of text into **one** vector. If you embed a whole 40-page handbook as one vector, that vector is a blurry average of everything: it matches nothing well. If you embed every word separately, each vector has no context ("30" means nothing alone).

So we cut documents into **chunks**: pieces big enough to hold one complete idea, small enough that the vector is about *that* idea. At question time we retrieve the few best chunks and give them to the LLM.

Laravel analogy: database **normalisation**. One giant `TEXT` column holding the whole book is unqueryable; one row per character is useless. A good chunk is like a well-designed row: one coherent record that you can fetch by key and understand on its own.

## 📚 Concepts, step by step

### 1. Why chunk at all?

1. **Embedding quality:** one vector per idea beats one vector per book.
2. **Model limits:** `gemini-embedding-001` accepts roughly 2048 input tokens; longer text is truncated or rejected.
3. **Cost and focus:** you send only the relevant 3-5 chunks to `gemini-2.5-flash`, not the whole document. Less noise, lower latency, fewer hallucinations.
4. **Citations:** a chunk maps to a page/section, so you can show the source.

### 2. The trade-off (memorise this table)

| | Small chunks (~100-200 tokens) | Large chunks (~800+ tokens) |
|---|---|---|
| Precision of match | High: vector is about one fact | Low: vector is a blur of topics |
| Context for the LLM | Often too little ("it expires after 1 hour" - what expires?) | Plenty, but with noise |
| Number of vectors / cost | More | Fewer |
| Typical failure | Answer split across two chunks | Right chunk found, but diluted and ranked low |

A good starting point for prose is **200-500 tokens (≈ 800-2000 characters)** with 10-15% overlap, then *measure*. (Topic 03 covers overlap in depth.)

### 3. Strategy 1 - fixed-size

Slide a window of N characters (or tokens), step = size - overlap.

```
text:  |-------- 100 --------|
                      |-------- 100 --------|     (overlap 20)
```

Pros: trivial, predictable size. Cons: it cuts sentences and even words in half (`"...not been"` | `"downloaded."`). Use only as a baseline or for text with no structure (logs, transcripts without punctuation).

### 4. Strategy 2 - recursive (the default you should reach for)

Try the **largest natural boundary first**, and only fall back to smaller ones for pieces that are still too long:

```
separators = ["\n\n" (paragraph), "\n" (line), ". " (sentence), " " (word), "" (character)]
```

Algorithm: split on the first separator that exists in the text; any piece still longer than `max_chars` is split again with the *next* separator; finally small neighbours are merged back together up to `max_chars`. Result: chunks that respect paragraphs and sentences whenever possible. This is the idea behind LangChain's `RecursiveCharacterTextSplitter`; you will write it yourself in Q2.

Known weakness: a heading can get glued to the *previous* paragraph's chunk instead of the next one (you will see this in Q2's output: `'...downloaded.\n\nShipping policy'`). Fix with structure-aware chunking (strategy 5).

### 5. Strategy 3 - sentence-based

Split into sentences, then group sentences until a token/character budget is hit. Never cuts a sentence. Use a real sentence splitter for production (`nltk`, `spaCy`, `pysbd`) because naive `". "` splitting breaks on `Dr. Rao`, `e.g.`, `3.14`, and Indian number formats like `Rs. 1,00,000.`

### 6. Strategy 4 - semantic chunking

Embed every sentence; compute similarity between **neighbouring** sentences; where similarity drops sharply, the topic changed: start a new chunk.

```
sentence:   S1   S2   S3   S4   S5   S6
sim(i,i+1):   0.8  0.7  0.2  0.8  0.3
                        ^ cut       ^ cut
```

Choose breakpoints with a **percentile** of the similarity distribution (lowest 30% become cuts) rather than a fixed number, because absolute similarity values differ per model. Cons: costs one embedding per sentence at ingest, and sizes are irregular. Benchmarks often show it beats recursive only modestly, so *measure before adopting*. In our offline demo the fake embedder is not truly semantic, so Q3 only shows the mechanics.

### 7. Strategy 5 - structure-aware (headings, pages, code)

When the document has structure, **use it**: split Markdown at `#`/`##` headings, DOCX by heading style, HTML by `h1-h3` (topic 01 already did that), code by function/class, PDFs by page. Then apply recursive splitting *inside* sections that are too long. Prepend the heading to each chunk's text ("Shipping policy: ...") so the vector knows what the sentence is about.

### 8. Measuring tokens vs characters

LLMs and embedding models count **tokens**, not characters. English averages ~4 characters per token (≈ 0.75 words); Hindi/Devanagari and code use more tokens per character. For exact counts use the API's `client.models.count_tokens(model="gemini-2.5-flash", contents=text)`. A rough budget: `tokens ≈ len(text)/4` for English. The course uses characters for speed and offline testing.

### 9. What every chunk must carry

```python
Document(text=chunk_text, metadata={
    "source": "faq.pdf",       # copied from the parent document
    "page": 3,                 # copied
    "chunk": 7,                # index within the document
    "start": 1840,             # character offset in the parent text
    "chunk_id": "faq.pdf#7",   # deterministic: re-ingesting overwrites, never duplicates
})
```

Without `source`/`page` you cannot cite; without deterministic ids you get duplicates on every re-run; without offsets you cannot highlight the passage.

## 💻 Code walkthrough

### Stage 1 - fixed windows (Q1)

```python
def chunk_fixed(text, size=100, overlap=20):
    assert 0 <= overlap < size
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step) if text[i:i + size].strip()]
```
`range(0, len, step)` gives every window start. The assert prevents `step <= 0` (an infinite loop in `while` versions, see Q4).

### Stage 2 - recursive split (Q2)

```python
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
def recursive_split(text, max_chars=200, separators=SEPARATORS):
    if len(text) <= max_chars: return [text]
    sep = next((s for s in separators if s == "" or s in text), "")
    parts = list(text) if sep == "" else text.split(sep)
    ...  # recurse into over-long parts with the remaining separators, then merge small neighbours
```
The merge pass is what turns "one chunk per sentence" into "as many sentences as fit": without it you get tiny fragments.

### Stage 3 - semantic breakpoints (Q3)

```python
E = embed_fn(sentences)
sims = [E[i] @ E[i+1] for i in range(len(sentences) - 1)]      # unit vectors: dot = cosine
cutoff = np.percentile(sims, 100 - percentile)
```
Swap `fake_embed` for `lambda s: embed_texts(s, task_type="SEMANTIC_SIMILARITY")`. Batch all sentences in one call (the helper batches and caches).

### Stage 4 - the production wrapper (Q5)

`chunk_documents(docs, max_chars)` runs the splitter per document and writes `source`, `chunk`, `start`, `chunk_id` onto each chunk. Then `hit_rate(size)` embeds all chunks, retrieves top-1 for each benchmark question and checks that the retrieved chunk contains the answer. Expected shape of the result (offline stand-in embedder):

```text
size  chunks  hit@1  chars sent to LLM per question
  60      10  0.80      50
 140       5  1.00     105
 300       3  1.00     194
1000       1  1.00     537
```
Read it carefully: hit@1 of 1.00 at size 1000 is **trivial** (one chunk holds everything) - note the cost column: 5x more text sent to the LLM. Always report quality *and* cost.

## ⚠️ Common mistakes & debugging

| Symptom | Cause | Fix |
|---|---|---|
| Answers are vague or half-right | Chunks too small, answer split across chunks | Larger chunks, overlap, or parent-child retrieval (Week 4) |
| Right document, wrong passage ranked top | Chunks too big, diluted vectors | Smaller chunks; add headings |
| Infinite loop in chunker | `overlap >= size` so the pointer never advances | Validate `0 <= overlap < size` |
| Last sentences missing | Loop condition `i < len - size` drops the tail | Loop `while i < len(text)` and stop after the final window |
| Cannot cite sources | Metadata not copied to chunks | Copy parent metadata, add `chunk`, `start`, `chunk_id` |
| Duplicates after re-ingestion | Random/auto ids | Deterministic `source#n` ids + delete-then-insert |
| "It expires after 1 hour" retrieved without context | Chunk lacks heading | Prepend section heading to chunk text |
| Words cut in half | Fixed-size on characters | Use recursive/sentence splitter |
| Sentence splitter breaks on "Dr." or "Rs. 5." | Naive `". "` | `pysbd`/`spaCy`, or protect abbreviations |
| Embedding call errors on huge chunk | > ~2048 tokens | Cap `max_chars` (≈ 6000-8000 for English) |

## 🚀 Pro-level insights (what the top 2% know)

- **Evaluate, don't guess.** Build 30-100 real question→answer-passage pairs, then compare hit@k / MRR across chunk sizes. Gains of 10-20 points from chunking changes are common; most embedding-model swaps gain less.
- **Chunk for the question, not the document.** FAQ-like content wants small chunks (one Q+A each); legal/technical reasoning wants larger sections.
- **Prefix context to chunks** ("Document: Refund policy > Section 2") before embedding, and keep the clean text for display. Cheap, large gain.
- **Parent-child retrieval:** embed small chunks, return their larger parent to the LLM (Week 4 topic 03). Solves the small-vs-large dilemma.
- **Tables and code are special:** keep a table or function intact; convert table rows to sentences (topic 01).
- **Re-chunking = re-embedding**, so version your chunker (`chunker="recursive-v2-500"` in metadata) and use blue/green re-indexing (Week 2).
- **Do not over-engineer:** recursive + 10-15% overlap + heading prefix beats most fancy strategies on typical corpora. Add complexity only when your benchmark shows it helps.
- **Embedding cost scales with total tokens**, not chunk count, but *overlap* increases total tokens (15% overlap = 15% more embedding cost and storage).

## 🧪 Mini lab (10-15 min)

1. Take a real 5-page PDF or Markdown file you own and load it with topic 01's loaders.
2. Chunk with sizes 200, 500, 1000 characters using `chunk_documents`. Print counts and average length.
3. Write 8 questions whose answers you can find by hand; record the gold chunk text snippet.
4. Compute hit@1 and hit@3 for each size (use `embed_texts` if you have a key, `fake_embed` otherwise) and chars sent per question.
5. Write two sentences: which size would you ship and why?

## ✅ Recap checklist

- [ ] I can explain why chunks that are too big or too small both hurt.
- [ ] I can write fixed, recursive and sentence chunkers without looking.
- [ ] My chunker validates `overlap < size` and never drops the tail.
- [ ] Every chunk carries `source`, locator, `chunk`, `start` and a deterministic `chunk_id`.
- [ ] I choose chunk size using a benchmark that reports quality **and** cost.
- [ ] I know when to use structure-aware or semantic chunking instead.

## 🎤 Interview questions

**1. Why not embed the whole document as one vector?**
One vector averages all topics, so it matches no specific question well, may exceed the model's token limit, and forces you to send huge context to the LLM.

**2. How does recursive chunking work?**
Split on the biggest separator first (paragraph), recurse into pieces still too long with smaller separators (line, sentence, word), then merge small neighbours up to the limit. It keeps natural boundaries intact.

**3. How do you pick chunk size?**
Start ~300-500 tokens with 10-15% overlap, then benchmark hit@k/MRR on real questions across sizes, also tracking the tokens sent to the LLM.

**4. What is semantic chunking and its downside?**
It starts a new chunk where neighbouring-sentence embedding similarity drops. It follows topics but costs an embedding per sentence and gives irregular sizes; gains over recursive are often small.

**5. What metadata should chunks have?**
Parent `source` and page/heading, chunk index, character offsets, and a deterministic id; this enables citations, filtering, idempotent re-ingestion and deletion by source.

## 🔗 Next

**03 - Chunk overlap:** why neighbouring chunks should share a little text, how much is right, and how to measure what overlap buys you (and costs).
