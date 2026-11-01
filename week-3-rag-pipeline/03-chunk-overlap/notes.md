# 03 - Chunk overlap

> Week 3 · Topic 3 of 7 · Time: ~60 min · Pure Python, runs offline

## 🎯 What you will learn

- You will be able to explain **why** adjacent chunks should share some text and what bug overlap prevents.
- You will be able to compute the **cost** of overlap (storage, embedding tokens) with a one-line formula.
- You will be able to implement character overlap and **sentence-aligned overlap**.
- You will be able to **merge overlapping retrieved chunks** so the prompt does not repeat text.
- You will be able to choose an overlap value with a small benchmark instead of copying "20%" from a blog.

## 🧠 The big idea in plain words

Imagine cutting a printed page into strips with scissors. If one important sentence sits exactly where you cut, neither strip contains the whole sentence. Later, when someone asks about it, neither strip matches well and the answer is lost.

**Overlap** means each strip repeats a little of the end of the previous strip. The sentence that was cut in one place is now whole in the next chunk.

Laravel analogy: **pagination with a window**. If you fetch rows 1-100, then 101-200, a record that spans a boundary (say, an order and its line items) is split. Fetching 1-100 then 91-190 guarantees every record that is at most 10 rows long appears whole in at least one page. Overlap is the same idea for text.

## 📚 Concepts, step by step

### 1. The boundary problem

```
text:    ...Warranty covers manufact|uring defects for 12 months...
                                    ^ chunk boundary
chunk A: "...Warranty covers manufact"
chunk B: "uring defects for 12 months..."
```
Question: "How long is the warranty?" Chunk A has *warranty* but no number; chunk B has *12 months* but no subject. Both vectors are weak; retrieval fails or the LLM gets half a fact. Q1 reproduces this exactly.

### 2. How overlap works

```
size = 100, overlap = 30, step = size - overlap = 70

chunk 0: chars   0..100
chunk 1: chars  70..170      <- first 30 chars repeat chunk 0's last 30
chunk 2: chars 140..240
```

**Guarantee:** if every "fact" you care about is at most `overlap` characters long, then every fact lies completely inside at least one chunk. (Proof sketch: consecutive window starts differ by `size - overlap`; a fact of length ≤ overlap that starts in the gap must fit in the next window.) In Q5 the longest fact is 52 characters, and recall hits 1.00 once overlap is ≥ 52.

### 3. What overlap costs

Each character is embedded about `size / (size - overlap)` times.

| overlap / size | multiplier | extra storage & embedding on 1 MB |
|---|---|---|
| 0% | 1.00x | 0 |
| 10% | 1.11x | +11% |
| 15% | 1.18x | +18% |
| 25% | 1.33x | +33% |
| 50% | 2.00x | +100% |

(Q2 computes this table.) Costs: more vectors to store and search, more embedding calls against the free-tier quota, and more near-duplicate chunks competing in the top-k.

### 4. How much overlap is right?

There is no universal number. Rules of thumb:

- Prose/articles: **10-20%** of chunk size.
- FAQ-style or one-fact-per-paragraph content: **0-10%** (paragraph chunks rarely cut facts).
- Dense legal/technical text with long sentences: 20-25%, or better, sentence-aligned.
- More than ~30%: you are mostly paying to store duplicates. Prefer bigger chunks or parent-child retrieval instead.

Real data is lumpy: in Q5 going from 0% to 10% overlap *lowered* recall (0.75 → 0.62 at size 80) because the shifted boundaries cut different facts. **Small character overlaps are a coin-flip; only overlap ≥ your longest fact is a guarantee.** That is why we measure.

### 5. Sentence-aligned overlap (better than characters)

Character overlap starts mid-word (`"ties. Warranty..."`). Instead, repeat the last N **whole sentences**:

```
[0] Refunds ... 30 days. Shipping takes 3 days to metro cities.
[1] Shipping takes 3 days to metro cities. Warranty covers ... 12 months.
[2] Warranty covers ... 12 months. Passwords expire after 90 days.
```
Chunks start on a sentence boundary, read naturally, and no sentence is ever cut. This is what you should use whenever your chunker is sentence-based (Q3).

### 6. The side-effect: duplicate context at query time

Overlapping chunks are often *retrieved together* (they are similar to the same question). If you paste both into the prompt you send the overlapping text twice, wasting tokens and sometimes confusing the model. Fix: **merge retrieved chunks of the same source whose character spans overlap** into one passage (Q4). This needs `source`, `start`, `end` in the metadata, one more reason to record offsets (topic 02).

### 7. Overlap vs its alternatives

| Technique | Solves the boundary problem by | Cost |
|---|---|---|
| Overlap | repeating edges | +10-30% storage |
| Structure-aware chunks (sections/paragraphs) | never cutting inside a unit | none, needs structure |
| Parent-child retrieval (Week 4) | retrieving the neighbours/parent at query time | extra storage for parents |
| Neighbour expansion | at query time, add chunk n-1 and n+1 of each hit | more prompt tokens |

Overlap is the cheapest first fix; the others are better when you outgrow it.

## 💻 Code walkthrough

### Stage 1 - windows with validation

```python
def windows(text, size, overlap):
    if not 0 <= overlap < size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")
    out, i = [], 0
    while i < len(text):
        out.append((i, min(i + size, len(text))))
        if i + size >= len(text): break
        i += size - overlap
    return out
```
Returning `(start, end)` offsets rather than strings lets you slice, merge and test cheaply. The `break` stops after the window that reaches the end; without it the loop would produce extra tiny tail chunks that are entirely inside the previous one.

### Stage 2 - cost formula

```python
mult = sum(b - a for a, b in windows(text, size, overlap)) / len(text)   # ≈ size / (size - overlap)
```

### Stage 3 - sentence overlap

```python
cur = cur[-overlap_sentences:]     # carry the last N sentences into the next chunk
if len(" ".join(cur + [s])) > max_chars: cur = []   # guarantee progress when sentences are long
```
The progress guard matters: if a carried sentence plus the next sentence exceeds the limit, you would otherwise emit the same chunk forever.

### Stage 4 - merging retrieved spans

```python
spans = sorted((start, end, text) ...)            # per source
if s <= cur_e:  cur_t += t[cur_e - s:]            # skip the characters we already have
```
Sorting by `start` then extending the running span is the classic *merge intervals* algorithm.

### Stage 5 - the benchmark

`fact_recall(text, facts, size, overlap)` = fraction of known facts that sit fully inside one window. Sweep sizes and overlap percentages, print recall and cost, and pick the cheapest setting that meets your target. With a real corpus, replace this with hit@k on real questions (topic 02, Q5).

## ⚠️ Common mistakes & debugging

| Symptom | Cause | Fix |
|---|---|---|
| Infinite loop / hang | `overlap >= size`, step ≤ 0 | Validate before chunking |
| Duplicate sentences in the final prompt | Neighbour chunks retrieved together | Merge overlapping spans per source |
| Index 30% bigger than expected | 25-30% overlap multiplies vectors | Lower overlap or use sentence overlap |
| Results full of near-identical chunks | Heavy overlap fills top-k with neighbours | Lower overlap; MMR/diversity (Week 4); merge hits |
| Chunk starts mid-word | Character overlap | Sentence- or word-aligned overlap |
| Recall barely changes with overlap | Chunks already follow paragraph boundaries | Use 0-10% and save cost |
| Last chunk is a tiny duplicate | Loop does not stop after the final window | Break when `start + size >= len(text)` |
| Cannot merge hits | No `start`/`end` metadata | Store offsets on every chunk |

## 🚀 Pro-level insights (what the top 2% know)

- **Overlap is insurance, not a quality knob.** Once chunks follow natural boundaries, extra overlap yields little; measure and cut it.
- **Rule of thumb that is actually provable:** overlap ≥ the length of the longest unit you must keep whole (a sentence, a clause) guarantees no unit is cut.
- **Cost math:** at 15% overlap on 100k pages you embed ~15% more tokens, and your vector DB RAM/disk grows by the same factor. On Gemini's free tier that also means 15% more rate-limited calls.
- **De-duplicate at retrieval:** merge adjacent hits, or drop a hit whose span is contained in a better hit. Do it before building the prompt.
- **Neighbour expansion** (fetch chunk n-1 and n+1 for the top hit using `chunk_id`s) often beats raising overlap, because you pay only for the queries that need it.
- **Version your chunking parameters** (`size=500, overlap=75, v2`) in metadata so you can re-index safely.
- **Evaluate on questions whose answers sit near boundaries** (build them deliberately); random questions hide the benefit.

## 🧪 Mini lab (10-15 min)

1. Take any 3-page text you own. Pick 10 sentences as "facts".
2. Run `fact_recall` for size 300 and overlap 0, 30, 60, 90.
3. Plot or tabulate recall vs cost multiplier. Where is the knee?
4. Chunk with sentence overlap = 1 and compare chunk count with the best character setting.
5. Pretend top-3 hits are chunks 4, 5 and 9; run `merge_hits` and report how many prompt characters you saved.

## ✅ Recap checklist

- [ ] I can draw the boundary problem and how overlap fixes it.
- [ ] I can compute the overlap cost multiplier by hand.
- [ ] My chunker validates `overlap < size` and terminates.
- [ ] I can chunk with sentence-aligned overlap.
- [ ] I can merge overlapping retrieved spans into one passage.
- [ ] I choose overlap using a benchmark, and I know when to use 0.

## 🎤 Interview questions

**1. Why add overlap between chunks?**
So facts that straddle a boundary appear whole in at least one chunk; otherwise neither chunk matches the question well.

**2. What does 20% overlap cost?**
About `1/(1-0.2) = 1.25x` the characters embedded and stored, plus more near-duplicate chunks.

**3. Character vs sentence overlap?**
Sentence overlap starts chunks on natural boundaries and never cuts a sentence; character overlap is simple but can start mid-word.

**4. Overlapping chunks both land in the top-k. What do you do?**
Merge spans of the same source using start/end offsets into one passage so the prompt has no repeated text.

**5. How would you tune overlap?**
Build a benchmark with questions near boundaries, sweep overlap, plot quality vs cost, and choose the cheapest setting meeting the target; consider neighbour expansion or parent-child instead of large overlaps.

## 🔗 Next

**04 - The full RAG pipeline:** load → clean → chunk → embed → store → retrieve → generate, wired together end to end with Gemini.
