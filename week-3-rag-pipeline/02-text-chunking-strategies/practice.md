# Practice - Text chunking strategies

Five questions, rising difficulty. Attempt each before opening the hint, and the hint before the solution. Every solution is the exact code in `solutions.py` (`python solutions.py` runs all). **Everything runs offline, no API key:** a hashed bag-of-words `fake_embed` stands in for Gemini in Q3 and Q5 (it is not semantic, so exact numbers will differ with the real model). `Document`, `fake_embed` and `SAMPLE` (a two-section policy text) are defined at the top of `solutions.py`.

---

### Q1. Warm-up - fixed-size chunks and the broken-word problem

**Problem**

Write `chunk_fixed(text, size, overlap)` that slides a window over `SAMPLE`. Print the first three chunks and count how many chunks end in the middle of a word. Reject an invalid overlap.

<details><summary><b>Hint</b></summary>

Window starts are `range(0, len(text), size - overlap)`. A chunk ends mid-word if its last character is alphanumeric and it is not the end of the text.

</details>

<details><summary><b>Solution</b></summary>

```python
# --- Q1: fixed-size chunking - and why it cuts words in half ----------------
def chunk_fixed(text, size=100, overlap=20):
    """Slide a window of `size` characters, stepping by size - overlap."""
    assert 0 <= overlap < size, "overlap must be smaller than size"
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step) if text[i:i + size].strip()]


def q1():
    chunks = chunk_fixed(SAMPLE, size=100, overlap=20)
    print("chunks:", len(chunks))
    for c in chunks[:3]:
        print(repr(c))
    cut = sum(1 for c in chunks if c[-1].isalnum() and not c.endswith(SAMPLE[-1]))
    print("chunks that end in the middle of a word:", cut)
    return chunks
```

</details>

**Why it works / what to learn**
Fixed windows are simple and predictable but ignore language: they cut words and sentences. That is why they are a baseline, not a default.

**Expected output**
```text
chunks: 5
'Refund policy\n\nCustomers may request a refund within 30 days of purchase. Refunds are issued to the '
's are issued to the original payment method. Digital goods are refundable only if they have not been'
'f they have not been downloaded.\n\nShipping policy\n\nOrders ship within 2 business days. Delivery to m'
chunks that end in the middle of a word: 3
```

---

### Q2. Apply - a recursive splitter from scratch

**Problem**

Implement `recursive_split(text, max_chars)` with separators `["\n\n", "\n", ". ", " ", ""]`: split on the biggest separator present, recurse only into over-long pieces, then merge small neighbours up to `max_chars`. Assert no chunk exceeds the limit. Look at the output and point out one weakness.

<details><summary><b>Hint</b></summary>

Pick `sep = next(s for s in separators if s == "" or s in text)`, split, recurse with the *remaining* separators, and finish with a merge loop that appends pieces while the combined length fits.

</details>

<details><summary><b>Solution</b></summary>

```python
# --- Q2: recursive splitter built from scratch ------------------------------
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def recursive_split(text, max_chars=200, separators=SEPARATORS):
    """Split on the biggest separator first; recurse only into pieces that are still too long;
    then merge small neighbours back together up to max_chars."""
    if len(text) <= max_chars:
        return [text]
    sep = next((s for s in separators if s == "" or s in text), "")
    parts = list(text) if sep == "" else text.split(sep)
    rest = separators[separators.index(sep) + 1:] if sep else []
    pieces = []
    for i, p in enumerate(parts):
        p = p + sep if (sep and sep != "\n\n" and sep != "\n" and i < len(parts) - 1) else p
        if not p.strip():
            continue
        if len(p) > max_chars and rest:
            pieces.extend(recursive_split(p, max_chars, rest))
        else:
            pieces.append(p)
    # merge pass
    joiner = "\n\n" if sep == "\n\n" else ("\n" if sep == "\n" else "")
    merged, cur = [], ""
    for p in pieces:
        cand = (cur + joiner + p) if cur else p
        if len(cand) <= max_chars:
            cur = cand
        else:
            if cur:
                merged.append(cur.strip())
            cur = p
    if cur.strip():
        merged.append(cur.strip())
    return merged


def q2():
    chunks = recursive_split(SAMPLE, max_chars=170)
    for i, c in enumerate(chunks):
        print(f"[{i}] ({len(c)} chars) {c!r}")
    assert all(len(c) <= 170 for c in chunks)
    return chunks
```

</details>

**Why it works / what to learn**
Natural boundaries are tried first, so chunks follow paragraphs and sentences. The weakness is visible in chunk 1: the heading `Shipping policy` got glued to the *previous* section. Structure-aware chunking (split on headings first) fixes it.

**Expected output**
```text
[0] (124 chars) 'Refund policy\n\nCustomers may request a refund within 30 days of purchase. Refunds are issued to the original payment method.'
[1] (84 chars) 'Digital goods are refundable only if they have not been downloaded.\n\nShipping policy'
[2] (154 chars) 'Orders ship within 2 business days. Delivery to metro cities takes 3 days. Remote areas may take up to 10 days. Tracking links are emailed after dispatch.'
```

---

### Q3. Build - semantic breakpoints

**Problem**

Write `semantic_chunks(text, embed_fn, percentile, max_sentences)`: split into sentences, embed them, compute cosine similarity between neighbours, and start a new chunk wherever similarity falls in the lowest `100 - percentile` percent. Try it on a text mixing cricket, programming and cooking sentences.

<details><summary><b>Hint</b></summary>

With unit vectors, `E[i] @ E[i+1]` is the cosine. `np.percentile(sims, 100 - percentile)` is the cut-off; start a new chunk when `sims[i] <= cutoff`.

</details>

<details><summary><b>Solution</b></summary>

```python
# --- Q3: sentence grouping + semantic breakpoints ---------------------------
def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
    return [p.strip() for p in parts if p.strip()]


def semantic_chunks(text, embed_fn=fake_embed, percentile=70, max_sentences=6):
    """Start a new chunk where the similarity between neighbouring sentences drops the most."""
    sents = split_sentences(text)
    if len(sents) < 3:
        return [" ".join(sents)]
    E = embed_fn(sents)
    sims = np.array([float(E[i] @ E[i + 1]) for i in range(len(sents) - 1)])
    cutoff = np.percentile(sims, 100 - percentile)      # lowest (100-p)% similarities = breakpoints
    chunks, cur = [], [sents[0]]
    for i, s in enumerate(sents[1:]):
        if sims[i] <= cutoff or len(cur) >= max_sentences:
            chunks.append(" ".join(cur))
            cur = []
        cur.append(s)
    chunks.append(" ".join(cur))
    return chunks


def q3():
    text = (
        "Cricket is played between two teams of eleven players. The batsman tries to hit the ball and score runs. "
        "A bowler delivers the ball from the other end of the pitch. "
        "Python is a popular programming language. A programmer writes code and runs the code in a terminal. "
        "Programming languages have functions and classes. "
        "Banana bread needs ripe bananas and flour. Mix the bananas with sugar and bake the bread."
    )
    for i, c in enumerate(semantic_chunks(text, percentile=30)):
        print(f"[{i}] {c}")
    return semantic_chunks(text, percentile=30)
```

</details>

**Why it works / what to learn**
Using a percentile adapts to whatever similarity scale the embedding model produces. Note the offline stand-in only matches shared words, so it over-splits here; with real Gemini embeddings you would expect roughly one chunk per topic. Treat these numbers as a demo of the mechanism.

**Expected output** (stand-in embedder; real Gemini output will group by topic)
```text
[0] Cricket is played between two teams of eleven players.
[1] The batsman tries to hit the ball and score runs. A bowler delivers the ball from the other end of the pitch.
[2] Python is a popular programming language.
[3] A programmer writes code and runs the code in a terminal.
[4] Programming languages have functions and classes.
[5] Banana bread needs ripe bananas and flour. Mix the bananas with sugar and bake the bread.
```

---

### Q4. Debug - three bugs in one chunker

**Problem**

```python
def bad_chunker(text, size, overlap):
    chunks, i = [], 0
    while i < len(text) - size:
        chunks.append(text[i:i + size])
        i += size - overlap
    return chunks
```
Find every bug (there are three: one loses data, one can hang forever, one makes citing impossible) and write `good_chunker`.

<details><summary><b>Hint</b></summary>

Check what happens to the last `size` characters, what happens when `overlap >= size`, and what each chunk knows about where it came from.

</details>

<details><summary><b>Solution</b></summary>

```python
# --- Q4: debug a broken chunker ---------------------------------------------
def bad_chunker(text, size, overlap):
    chunks, i = [], 0
    while i < len(text) - size:            # BUG 1: drops the tail (<= size chars left)
        chunks.append(text[i:i + size])
        i += size - overlap                # BUG 2: overlap >= size means i never advances -> infinite loop
    return chunks                          # BUG 3: no metadata, no offsets -> cannot cite


def good_chunker(text, size, overlap, metadata=None):
    if not 0 <= overlap < size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")
    out, i, n = [], 0, 0
    while i < len(text):
        piece = text[i:i + size]
        if piece.strip():
            out.append(Document(piece, {**(metadata or {}), "chunk": n, "start": i, "end": i + len(piece)}))
            n += 1
        if i + size >= len(text):          # last window reached the end: stop
            break
        i += size - overlap
    return out


def q4():
    text = "abcdefghij" * 3                  # 30 chars
    print("bad  :", [c for c in bad_chunker(text, 10, 2)])
    print("tail lost:", "".join(bad_chunker(text, 10, 2))[-1] != text[-1] or len(bad_chunker(text, 10, 2)) < 3)
    good = good_chunker(text, 10, 2, {"source": "demo.txt"})
    print("good :", [(c.metadata["start"], c.metadata["end"]) for c in good])
    try:
        good_chunker(text, 10, 10)
    except ValueError as e:
        print("guard:", e)
    return good
```

</details>

**Why it works / what to learn**
(1) `i < len - size` stops early and drops the tail; (2) with `overlap >= size` the step is ≤ 0 so `i` never advances; (3) plain strings carry no metadata or offsets. The fix validates input, loops until the final window reaches the end, and records `start`/`end` plus parent metadata.

**Expected output**
```text
bad  : ['abcdefghij', 'ijabcdefgh', 'ghijabcdef']
tail lost: True
good : [(0, 10), (8, 18), (16, 26), (24, 30)]
guard: overlap must satisfy 0 <= overlap < size
```

---

### Q5. Production - metadata-preserving chunker + chunk-size benchmark

**Problem**

Write `chunk_documents(docs, max_chars)` that splits each `Document` and copies its metadata, adding `chunk`, `start` and a deterministic `chunk_id` (`source#n`). Then, for a 5-section FAQ and 5 questions, measure hit@1 and the average characters sent to the LLM for chunk sizes 60, 140, 300 and 1000. Which size would you ship?

<details><summary><b>Hint</b></summary>

A "hit" means the top-1 retrieved chunk contains the sentence that answers the question. Also record `len(chunk.text)` of what you retrieve, because quality without cost is half the story.

</details>

<details><summary><b>Solution</b></summary>

```python
# --- Q5: production chunker + choose the size with a retrieval benchmark ----
def chunk_documents(docs, max_chars=300, overlap_chars=0):
    """Split each Document recursively, COPY its metadata, add deterministic ids and offsets."""
    out = []
    for d in docs:
        pieces = recursive_split(d.text, max_chars)
        pos = 0
        for n, p in enumerate(pieces):
            start = d.text.find(p[:40], pos)
            start = pos if start < 0 else start
            pos = start + 1
            meta = {**d.metadata, "chunk": n, "start": start,
                    "chunk_id": f"{d.metadata.get('source', 'doc')}#{n}"}
            out.append(Document(p, meta))
    return out


FAQ = Document(
    "Refunds: customers may request a refund within 30 days of purchase. Refunds go back to the original payment method.\n\n"
    "Shipping: orders ship within 2 business days. Metro delivery takes 3 days and remote areas up to 10 days.\n\n"
    "Warranty: all laptops carry a 12 month warranty covering manufacturing defects but not water damage.\n\n"
    "Passwords: reset your password from the login page using the forgot password link. The link expires after 1 hour.\n\n"
    "Invoices: invoices are emailed after payment and can be downloaded as PDF from the billing page.",
    {"source": "faq.txt"},
)
QUESTIONS = [
    ("how long do I have to ask for a refund", "faq.txt#0"),
    ("how many days does delivery take to remote areas", "faq.txt#1"),
    ("does the warranty cover water damage", "faq.txt#2"),
    ("password reset link expires", "faq.txt#3"),
    ("download invoice pdf", "faq.txt#4"),
]


def hit_rate(max_chars, k=1):
    chunks = chunk_documents([FAQ], max_chars=max_chars)
    E = fake_embed([c.text for c in chunks])
    hits, sent = 0, 0
    for q, gold in QUESTIONS:
        top = np.argsort(-(E @ fake_embed([q])[0]))[:k]
        gold_topic = gold
        # a hit = the retrieved chunk contains the sentence that answers the question
        answer_key = {"faq.txt#0": "30 days", "faq.txt#1": "remote areas", "faq.txt#2": "water damage",
                      "faq.txt#3": "expires", "faq.txt#4": "PDF"}[gold_topic]
        hits += any(answer_key in chunks[i].text for i in top)
        sent += sum(len(chunks[i].text) for i in top)
    return hits / len(QUESTIONS), len(chunks), sent // len(QUESTIONS)


def q5():
    docs = chunk_documents([FAQ], max_chars=140)
    print("sample metadata:", docs[0].metadata)
    print("size  chunks  hit@1  chars sent to LLM per question")
    for size in (60, 140, 300, 1000):
        hr, n, sent = hit_rate(size)
        print(f"{size:>4}  {n:>6}  {hr:.2f}  {sent:>6}")
```

</details>

**Why it works / what to learn**
Size 60 is too small (answers get split: hit@1 drops to 0.80). Size 1000 gets a perfect score only because everything is one chunk - and sends ~5x more text to the LLM. The sweet spot here is ~140-300. Always benchmark on *your* questions: with a real embedder and a larger corpus the curve changes, but the method does not.

**Expected output**
```text
sample metadata: {'source': 'faq.txt', 'chunk': 0, 'start': 0, 'chunk_id': 'faq.txt#0'}
size  chunks  hit@1  chars sent to LLM per question
  60      10  0.80      50
 140       5  1.00     105
 300       3  1.00     194
1000       1  1.00     537
```
