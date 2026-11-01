# 06 - Multimodal RAG: images, charts and scanned pages

> Week 4 · Topic 6 of 7 · Time: ~90 min · Offline demo uses Pillow-drawn images, real OCR (if `tesseract` is installed) and a stand-in for the Gemini vision call

## 🎯 What you will learn

- You will be able to explain **why `gemini-embedding-001` cannot embed images** and the pattern that works instead (image → text → embed).
- You will be able to validate uploaded images and turn each into a **structured description** (type, caption, visible text, key facts) with `gemini-2.5-flash`.
- You will be able to index images by their text, retrieve them with text questions, and **attach the original image** for the final answer.
- You will be able to handle **scanned PDFs, charts, tables and screenshots**, and know the limits (OCR vs vision, cost, hallucinated numbers).
- You will be able to enforce **budgets and safety** (size, count, types, caching).

## 🧠 The big idea in plain words

Real documents are full of things that are not text: charts, pricing tables saved as images, screenshots of errors, scanned pages, diagrams. A text-only RAG system is blind to all of them: the answer to "what does the Q3 revenue chart show?" is in a picture.

Our embedding model (`models/gemini-embedding-001`) only understands **text**. So we use a simple, robust trick:

```
image ──► gemini-2.5-flash (vision) ──► text: caption + visible text + key facts ──► embed text ──► search
                                                                        (at answer time, show the image to the model again)
```

Laravel analogy: you cannot `FULLTEXT`-index a JPEG. So when a user uploads an image you store a **text column** next to it (`alt_text`/`ocr_text`) written by a job, and search that column. Multimodal RAG is the same idea with an LLM writing the text.

## 📚 Concepts, step by step

### 1. Three ways to make images searchable

| Approach | How | Pros | Cons |
|---|---|---|---|
| **Caption + OCR → text embedding** (this lesson) | vision LLM describes; embed text | works with any text embedder, cheap search, explainable, supports metadata filters | caption quality limits recall; one LLM call per image at ingestion |
| **Native multimodal embeddings** | one model embeds text and images into the same space | search by image similarity, no captioning | needs a multimodal embedding model (not `gemini-embedding-001`), less explainable |
| **Late interaction / page-image models** (ColPali-style) | embed page screenshots patch by patch | great for visually rich PDFs | heavier infrastructure |

With the free Gemini models in this course, approach 1 is the right one.

### 2. What to extract from an image

Ask the vision model for **structured JSON**, not a vague sentence:

```json
{"type": "table",
 "caption": "Slide listing subscription plans with monthly prices",
 "visible_text": "Pricing plans Starter 499 rupees per month ...",
 "key_facts": ["Starter plan 499 rupees", "Growth plan 1299 rupees"]}
```
- `type` lets you route (attach charts/tables to the final prompt, ignore decorative photos).
- `visible_text` is verbatim text for exact-match style questions.
- `key_facts` includes **every number**, because questions often hit numbers.
- Index text = caption + visible_text + key_facts (Q3).

A lazy caption ("a slide") makes the image unfindable (Q4). The prompt is the quality lever.

### 3. OCR vs vision LLM

| | OCR (Tesseract, etc.) | Vision LLM (`gemini-2.5-flash`) |
|---|---|---|
| Reads clean printed text | excellent, free, local | excellent |
| Understands charts/layout/meaning | no | yes |
| Handwriting, messy scans | poor | good |
| Cost/latency | none/low | API call, quota |
| Hallucination risk | none (but garbles) | possible (invented numbers) |

Pragmatic pipeline: run OCR first (cheap); use the vision model when the image is a chart/diagram/photo or OCR text is short/garbled; combine both (our `visible_text` comes from OCR offline, from the model with a key).

### 4. Scanned PDFs (connecting to Week 3 topic 01)

A scanned page has no text layer, so `extract_text()` returns nothing and the loader flags `needs_ocr`. Route those pages: render page → PNG (e.g. `pdf2image`/PyMuPDF) → OCR or vision model → `Document(text, metadata={"source", "page", "ocr": True})`. Cache by page-image hash because vision calls are your most expensive ingestion step.

### 5. Calling Gemini with an image

```python
from google.genai import types
resp = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[types.Part.from_bytes(data=image_bytes, mime_type="image/png"), PROMPT],
    config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.0),
)
data = json.loads(resp.text)
```
`Part.from_bytes(data=..., mime_type=...)` is keyword-only in the current SDK. Supported formats include PNG, JPEG and WEBP. Large images are downscaled by the service; resize big files yourself (e.g. longest side ≤ 2000 px) to save bandwidth. The offline course code cannot execute this call without a key; it was reviewed against the SDK signature.

### 6. Answering with the images

At question time retrieve image records by text. Then decide what to send to the LLM:

- Always send the **captions/visible text** as source text (cheap, citable).
- Attach the **original image** only when it adds value (charts, tables, screenshots), with a strong enough match score, and within **budgets**: max images (2-3), max total bytes, allowed types (Q5).
- Cite the image by name/page so users can open it.

### 7. Metadata and ids

Store per image: `source` (file or PDF + page), `sha` (content hash), `type`, `width/height`, `caption_version`, `tenant`/ACL. Deterministic ids from the content hash make ingestion idempotent and allow caching (Q2).

### 8. Safety and quality

- **Validate uploads:** real image format, size/pixel limits, no executable content. Never trust the file extension (Q1 uses `Image.verify`).
- **Prompt injection in images:** text inside an image can say "ignore your instructions". Treat `visible_text` as untrusted data (escape and label, topic 05 of Week 3).
- **PII:** screenshots can contain personal data; apply the same access control and retention as text.
- **Numbers in charts:** vision models can misread values. For high-stakes figures keep the underlying data table and prefer it over chart reading; show the image to the user.
- **Evaluation:** build a gold set of (question → image) pairs; measure hit@k like text retrieval (topic 04).

## 💻 Code walkthrough

### Stage 1 - make sample images (no downloads)

`make_sample_images(folder)` draws a pricing slide and an error screenshot (text on white) and a text-free "photo" with Pillow, plus a corrupt `broken.png`.

### Stage 2 - validate (Q1)

```python
img = Image.open(io.BytesIO(raw)); img.verify()          # corrupt? raises
img = Image.open(io.BytesIO(raw))                        # verify() consumes the object: reopen
if img.format not in ALLOWED: raise ValueError(...)
```

### Stage 3 - describe with caching (Q2)

`Captioner.describe` loads/validates, looks up `sha` in its cache, otherwise calls `caption_with_gemini` (key set) or the offline stand-in (a lookup table for the caption + **real OCR** for `visible_text`).

### Stage 4 - index and search (Q3)

```python
text = " ".join([desc["caption"], desc["visible_text"], " ".join(desc["key_facts"])])
vec = embed_texts([text], task_type="RETRIEVAL_DOCUMENT")        # offline: toy_embed
```
Search embeds the question with `RETRIEVAL_QUERY` and returns image names + scores.

### Stage 5 - build the request (Q5)

`build_request` returns text parts plus image parts for the top matches that pass type, score, count and byte budgets. Real code replaces the placeholder part with `types.Part.from_bytes`.

## ⚠️ Common mistakes & debugging

| Symptom | Cause | Fix |
|---|---|---|
| Images never retrieved | Lazy captions ("a chart") | Structured prompt: caption + verbatim text + all numbers |
| Embedding call fails on an image | `gemini-embedding-001` is text-only | Embed the **text** description instead |
| Wrong numbers in answers | Model misread a chart | Keep source data; cite image; verify; lower temperature |
| Ingestion costs explode | Captioning the same image repeatedly | Cache by content hash; skip duplicates; skip decorative images |
| 400 error on upload | Unsupported/corrupt/huge file | Validate format and size; resize |
| Scanned PDF gives empty text | No text layer | Detect `needs_ocr`; render + OCR/vision |
| OCR returns garbage | Low resolution, skew, colour | Upscale, grayscale, deskew; or use the vision model |
| Prompt too big when attaching images | Too many/large images | Limit count/bytes; attach only charts/tables above a score |
| Model obeys text found in an image | Prompt injection via OCR/caption text | Escape, label as untrusted data, least privilege |
| 429 during bulk captioning | Free-tier rate limits | Throttle, retry with backoff, cache, process overnight |

## 🚀 Pro-level insights (what the top 2% know)

- **Index two representations:** a short caption for semantic search and verbatim OCR text for exact search; use hybrid retrieval (Week 2 topic 06).
- **Page-level thinking for PDFs:** keep the page image; text chunks cite `page`; if a text chunk mentions "Figure 3", pull the figure's record too.
- **Route by cost:** OCR for text-heavy images, vision for charts/diagrams, skip logos and stock photos (classify cheaply with the first caption call).
- **Batch and parallelise ingestion with care:** vision calls are slower; use a small pool, retries and a resumable queue.
- **Keep captions versioned:** when you improve the prompt, re-caption only images whose `caption_version` is old.
- **Combine structure:** for tables, ask the model for Markdown/CSV and index rows as sentences ("Plan: Growth; Price: 1299").
- **Be explicit about uncertainty:** instruct the model to write "unreadable" instead of guessing; low-confidence images get a human review queue.
- **Accessibility bonus:** the same captions are usable as `alt` text.

## 🧪 Mini lab (10-15 min)

1. Collect 6 images (a chart, a screenshot, a table, a photo, a diagram, a scanned page).
2. Describe each with the Gemini prompt (or the offline stand-in + OCR) and read the JSON; fix the prompt where captions are vague.
3. Index them and ask 6 questions, one per image; compute hit@1.
4. Add a size/type budget to the final request and test with a 5 MB image.
5. Add the content-hash cache and re-run ingestion; count model calls.

## ✅ Recap checklist

- [ ] I know `gemini-embedding-001` is text-only and why that shapes the design.
- [ ] I validate images before processing.
- [ ] I extract caption + verbatim text + key facts as JSON.
- [ ] I cache descriptions by content hash.
- [ ] I only attach original images within type, score, count and size budgets.
- [ ] I treat image text as untrusted data.

## 🎤 Interview questions

**1. How do you do RAG over images with a text-only embedding model?**
Convert each image to text with a vision LLM (caption, verbatim text, key facts), embed and index that text with metadata, retrieve with text queries, and optionally pass the original image to the LLM for the final answer.

**2. OCR or a vision model?**
OCR for clean text (free, deterministic); a vision model for charts, diagrams, layout and messy scans (understands meaning, costs quota and can hallucinate). Often combine them.

**3. How do you keep multimodal ingestion affordable?**
Cache by content hash, skip duplicates and decorative images, resize, batch with rate limits, and re-process only when the caption prompt version changes.

**4. What are the risks of showing images to the model?**
Misreading numbers, prompt injection via text in the image, PII exposure, and prompt size; mitigate with validation, escaping, access control and budgets.

**5. How do you evaluate it?**
A gold set mapping questions to images/pages; hit@k for retrieval, plus answer-level faithfulness checks on chart/table questions.

## 🔗 Next

**07 - LlamaIndex:** see how a framework packages loading, chunking, embedding, indexing and querying (with Gemini) in a few lines, and when to prefer your own code.
