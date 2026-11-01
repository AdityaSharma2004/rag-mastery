# Practice - Chunk overlap

Five questions, rising difficulty. Attempt each before opening the hint, and the hint before the solution. Every solution is the exact code in `solutions.py` (`python solutions.py` runs all). **Everything runs offline, no API key.** `Document`, `windows(text, size, overlap)` (returns `(start, end)` offsets) and `POLICY`/`FACTS` (a policy text made of 8 one-sentence facts) are defined at the top of `solutions.py`.

---

### Q1. Warm-up - watch overlap rescue a split fact

**Problem**

Using `windows`, chunk `POLICY` with `size=100` and overlap 0 and then 30. For the warranty sentence, report whether it lies completely inside one chunk, and print the chunks that touch it.

<details><summary><b>Hint</b></summary>

Find the sentence offset with `text.index(fact)`. A chunk `(a, b)` contains it iff `a <= s and s + len(fact) <= b`.

</details>

<details><summary><b>Solution</b></summary>

```python
# --- Q1: overlap rescues a fact that straddles a boundary -------------------
def q1():
    text = POLICY
    fact = "Warranty covers manufacturing defects for 12 months."
    s = text.index(fact)
    for overlap in (0, 30):
        ws = windows(text, size=100, overlap=overlap)
        whole = [w for w in ws if w[0] <= s and s + len(fact) <= w[1]]
        print(f"overlap={overlap:>2}: {len(ws)} chunks; fact fully inside one chunk: {bool(whole)}")
        for a, b in ws:
            if a < s + len(fact) and b > s:
                print("   touching chunk:", repr(text[a:b]))
```

</details>

**Why it works / what to learn**
With overlap 0 the warranty sentence is cut between two chunks (`...manufact` | `uring defects...`). With 30 characters of overlap the second chunk starts early enough to contain the whole sentence.

**Expected output**
```text
overlap= 0: 4 chunks; fact fully inside one chunk: False
   touching chunk: 'Refunds are accepted within 30 days. Shipping takes 3 days to metro cities. Warranty covers manufact'
   touching chunk: 'uring defects for 12 months. Passwords expire after 90 days. Support is available from 9am to 6pm IS'
overlap=30: 5 chunks; fact fully inside one chunk: True
   touching chunk: 'Refunds are accepted within 30 days. Shipping takes 3 days to metro cities. Warranty covers manufact'
   touching chunk: 'ties. Warranty covers manufacturing defects for 12 months. Passwords expire after 90 days. Support i'
```

---

### Q2. Apply - the cost of overlap

**Problem**

For a 1,000,000-character corpus and chunk size 1000, compute for overlaps 0, 100, 150, 250 and 500: number of chunks, total characters embedded, and the multiplier. Compare to the formula `size / (size - overlap)`.

<details><summary><b>Hint</b></summary>

Reuse `windows` on a string of `n` characters and sum `end - start`. Do not hard-code the answer; compute it, then compare with the formula.

</details>

<details><summary><b>Solution</b></summary>

```python
# --- Q2: what does overlap cost? -------------------------------------------
def overlap_cost(n_chars, size, overlap):
    """Return (n_chunks, total_chars_embedded, multiplier)."""
    ws = windows("x" * n_chars, size, overlap)
    total = sum(b - a for a, b in ws)
    return len(ws), total, total / n_chars


def q2():
    n = 1_000_000                                   # a 1 MB corpus
    print("size overlap  pct  chunks  chars_embedded  multiplier  ~formula size/(size-overlap)")
    for size, ov in [(1000, 0), (1000, 100), (1000, 150), (1000, 250), (1000, 500)]:
        c, t, m = overlap_cost(n, size, ov)
        print(f"{size:>4} {ov:>7} {ov / size:>4.0%} {c:>7} {t:>14,} {m:>10.2f}x {size / (size - ov):>10.2f}x")
```

</details>

**Why it works / what to learn**
The simulated multiplier matches the formula. 15% overlap costs ~18% more; 50% doubles everything. Overlap is paid for in vectors, storage and free-tier API calls.

**Expected output**
```text
size overlap  pct  chunks  chars_embedded  multiplier  ~formula size/(size-overlap)
1000       0   0%    1000      1,000,000       1.00x       1.00x
1000     100  10%    1111      1,111,000       1.11x       1.11x
1000     150  15%    1177      1,176,400       1.18x       1.18x
1000     250  25%    1333      1,333,000       1.33x       1.33x
1000     500  50%    1999      1,999,000       2.00x       2.00x
```

---

### Q3. Build - sentence-aligned overlap

**Problem**

Write `chunk_sentences(text, max_chars, overlap_sentences)` that packs whole sentences up to `max_chars`, and starts each new chunk with the last `overlap_sentences` sentences of the previous chunk. It must always make progress even if a carried sentence plus the next one is too long. Run it with 0 and 1 sentences of overlap on `POLICY`.

<details><summary><b>Hint</b></summary>

Keep a list `cur`. When adding the next sentence would exceed `max_chars`, emit `" ".join(cur)`, then set `cur = cur[-n:]`; if that still doesn't fit with the new sentence, set `cur = []`.

</details>

<details><summary><b>Solution</b></summary>

```python
# --- Q3: sentence-aligned overlap -------------------------------------------
def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def chunk_sentences(text, max_chars=120, overlap_sentences=1):
    """Pack whole sentences up to max_chars; each new chunk REPEATS the last N sentences of the previous one."""
    sents = split_sentences(text)
    chunks, cur = [], []
    for s in sents:
        if cur and len(" ".join(cur + [s])) > max_chars:
            chunks.append(" ".join(cur))
            cur = cur[-overlap_sentences:] if overlap_sentences else []
            # if the carried-over sentences + s are still too long, drop the carry (progress guarantee)
            if len(" ".join(cur + [s])) > max_chars:
                cur = []
        cur.append(s)
    if cur and (not chunks or " ".join(cur) != chunks[-1]):
        chunks.append(" ".join(cur))
    return chunks


def q3():
    for ov in (0, 1):
        print(f"--- overlap_sentences={ov}")
        for i, c in enumerate(chunk_sentences(POLICY, max_chars=120, overlap_sentences=ov)):
            print(f"[{i}] {c}")
```

</details>

**Why it works / what to learn**
Every chunk starts at a sentence boundary and no sentence is cut, so there is no mid-word start like in character overlap. The progress guard prevents emitting the same chunk repeatedly.

**Expected output**
```text
--- overlap_sentences=0
[0] Refunds are accepted within 30 days. Shipping takes 3 days to metro cities.
[1] Warranty covers manufacturing defects for 12 months. Passwords expire after 90 days.
[2] Support is available from 9am to 6pm IST. Invoices are emailed after payment. Premium plans include priority support.
[3] Cancellations need 7 days notice.
--- overlap_sentences=1
[0] Refunds are accepted within 30 days. Shipping takes 3 days to metro cities.
[1] Shipping takes 3 days to metro cities. Warranty covers manufacturing defects for 12 months.
[2] Warranty covers manufacturing defects for 12 months. Passwords expire after 90 days.
[3] Passwords expire after 90 days. Support is available from 9am to 6pm IST. Invoices are emailed after payment.
[4] Invoices are emailed after payment. Premium plans include priority support. Cancellations need 7 days notice.
```

---

### Q4. Debug - the prompt repeats itself

**Problem**

Your chunks overlap by 30 characters. A query retrieves chunks 1, 2 and 4 of the same file, and you paste them into the prompt with `"\n---\n".join(...)`. The LLM sees the overlapping text twice. Write `merge_hits(hits)` that merges overlapping or touching spans of the same source into single passages (using `start`/`end` metadata) and show how many prompt characters you save.

<details><summary><b>Hint</b></summary>

This is the classic *merge intervals* algorithm: sort by `start`; if the next `start <= current_end`, extend the current passage by appending only `text[current_end - start:]`.

</details>

<details><summary><b>Solution</b></summary>

```python
# --- Q4: overlapping chunks retrieved together => duplicated prompt text ----
def merge_hits(hits):
    """hits: list of Document with metadata source/start/end. Merge overlapping or touching spans of the
    same source into one passage, so the LLM prompt does not repeat text."""
    out = []
    for source in sorted({h.metadata["source"] for h in hits}):
        spans = sorted(
            ((h.metadata["start"], h.metadata["end"], h.text) for h in hits if h.metadata["source"] == source)
        )
        cur_s, cur_e, cur_t = spans[0]
        for s, e, t in spans[1:]:
            if s <= cur_e:                                   # overlap or touch: extend, skipping repeated chars
                if e > cur_e:
                    cur_t += t[cur_e - s:]
                    cur_e = e
            else:
                out.append(Document(cur_t, {"source": source, "start": cur_s, "end": cur_e}))
                cur_s, cur_e, cur_t = s, e, t
        out.append(Document(cur_t, {"source": source, "start": cur_s, "end": cur_e}))
    return out


def q4():
    ws = windows(POLICY, 100, 30)
    chunks = [Document(POLICY[a:b], {"source": "policy.txt", "start": a, "end": b}) for a, b in ws]
    retrieved = [chunks[1], chunks[2], chunks[4]]            # pretend top-3: two neighbours + one far away
    naive = "\n---\n".join(c.text for c in retrieved)
    merged = merge_hits(retrieved)
    print("naive prompt chars :", len(naive))
    print("merged prompt chars:", sum(len(m.text) for m in merged))
    for m in merged:
        print(m.metadata, repr(m.text))
    assert merged[0].text == POLICY[chunks[1].metadata["start"]:chunks[2].metadata["end"]]
```

</details>

**Why it works / what to learn**
Chunks 1 and 2 overlap, so they merge into one continuous passage; chunk 4 is far away and stays separate. The prompt shrinks and reads naturally. This only works because every chunk stores `source`, `start` and `end`.

**Expected output**
```text
naive prompt chars : 242
merged prompt chars: 202
{'source': 'policy.txt', 'start': 70, 'end': 240} 'ties. Warranty covers manufacturing defects for 12 months. Passwords expire after 90 days. Support is available from 9am to 6pm IST. Invoices are emailed after payment. P'
{'source': 'policy.txt', 'start': 280, 'end': 312} 'ancellations need 7 days notice.'
```

---

### Q5. Production - tune overlap with a benchmark

**Problem**

Write `fact_recall(text, facts, size, overlap)` returning the fraction of facts that lie wholly inside at least one chunk, the number of chunks, and the cost multiplier. Sweep sizes 80 and 120 with overlaps of 0, 10, 20, 30 and 50% of size. What do you notice? Which setting would you ship, and what rule explains the best result?

<details><summary><b>Hint</b></summary>

Reuse `windows`. The longest fact in `FACTS` is 52 characters; compare that with the overlap values that reach recall 1.00.

</details>

<details><summary><b>Solution</b></summary>

```python
# --- Q5: choose overlap with a benchmark (recall of intact facts vs cost) ---
def fact_recall(text, facts, size, overlap):
    ws = windows(text, size, overlap)
    ok = 0
    for f in facts:
        s = text.index(f)
        ok += any(a <= s and s + len(f) <= b for a, b in ws)
    return ok / len(facts), len(ws), sum(b - a for a, b in ws) / len(text)


def q5():
    print("size overlap   fact_recall  chunks  cost")
    best = None
    for size in (80, 120):
        for pct in (0, 10, 20, 30, 50):
            ov = int(size * pct / 100)
            r, n, cost = fact_recall(POLICY, FACTS, size, ov)
            print(f"{size:>4} {ov:>3} ({pct:>2}%)   {r:>9.2f}   {n:>5}  {cost:.2f}x")
            if best is None or (r, -cost) > (best[0], -best[1]):
                best = (r, cost, size, ov)
    print("cheapest setting with the best recall: size=%d overlap=%d (recall %.2f, cost %.2fx)"
          % (best[2], best[3], best[0], best[1]))
```

</details>

**Why it works / what to learn**
Recall is **not monotonic** for small overlaps: 10% overlap at size 80 is *worse* than 0% (0.62 vs 0.75) because shifting boundaries just cuts different facts. Only when overlap ≥ the longest fact (52 chars; here 60) is full recall guaranteed, at a cost of 1.77x. On real data prefer smaller overlap with sentence alignment, or neighbour expansion, and verify with a benchmark of your own.

**Expected output**
```text
size overlap   fact_recall  chunks  cost
  80   0 ( 0%)        0.75       4  1.00x
  80   8 (10%)        0.62       5  1.10x
  80  16 (20%)        0.88       5  1.21x
  80  24 (30%)        0.75       6  1.38x
  80  40 (50%)        0.88       7  1.77x
 120   0 ( 0%)        0.75       3  1.00x
 120  12 (10%)        0.75       3  1.08x
 120  24 (20%)        0.88       3  1.15x
 120  36 (30%)        0.88       4  1.35x
 120  60 (50%)        1.00       5  1.77x
cheapest setting with the best recall: size=120 overlap=60 (recall 1.00, cost 1.77x)
```
