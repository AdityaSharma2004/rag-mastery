# Practice - Handling long documents

Five questions, rising difficulty. Attempt each before opening the hint, and the hint before the solution. Every solution is the exact code in `solutions.py` (`python solutions.py` runs all). **Everything runs offline, no API key:** a toy "summariser" (first sentence) and a hashed bag-of-words embedder stand in for Gemini (set `GEMINI_API_KEY` and `summarize()` uses `gemini-2.5-flash`). `HANDBOOK` (a short employee handbook in Markdown), `fake_embed`, `toy_summarize` and `tokens` are defined at the top of `solutions.py`.

---

### Q1. Warm-up - a map-reduce summary tree

**Problem**

Write `tree_reduce(items, combine, group_size)` that repeatedly combines groups of items until one remains, returning the result, the number of `combine` calls and the number of levels (raise an error if `group_size < 2`). Use it to build `map_reduce_summary(chunks, group_size)` and count the total calls for 40 chunks with group sizes 2, 5 and 10.

<details><summary><b>Hint</b></summary>

Loop `while len(items) > 1`: `items = [combine(items[i:i+g]) for i in range(0, len(items), g)]`. Add the map step's `len(chunks)` calls to the total.

</details>

<details><summary><b>Solution</b></summary>

```python
# --- Q1: map-reduce summarisation of a long document (a tree of summaries) ----
def tree_reduce(items, combine, group_size=4):
    """Repeatedly combine groups of `group_size` items until one remains. Returns (result, llm_calls, levels)."""
    if group_size < 2:
        raise ValueError("group_size must be >= 2 or the tree never shrinks")
    calls, levels = 0, 0
    while len(items) > 1:
        items = [combine(items[i:i + group_size]) for i in range(0, len(items), group_size)]
        calls += len(items)
        levels += 1
    return items[0], calls, levels


def map_reduce_summary(chunks, group_size=4, summarise=toy_summarize):
    partial = [summarise(c) for c in chunks]                            # MAP: one summary per chunk
    result, calls, levels = tree_reduce(partial, lambda g: summarise(" ".join(g)), group_size)   # REDUCE
    return result, len(chunks) + calls, levels


def q1():
    chunks = [f"Section {i} says that rule number {i} applies to team {i}. More detail follows here." for i in range(40)]
    summary, calls, levels = map_reduce_summary(chunks, group_size=5)
    print("final summary:", summary)
    print("LLM calls:", calls, "(40 map + 8 + 2 + 1 reduce)  levels:", levels)
    for g in (2, 5, 10):
        _, c, l = map_reduce_summary(chunks, group_size=g)
        print(f"group_size={g:>2}: calls={c:>3} levels={l}")
```

</details>

**Why it works / what to learn**
40 map calls + 8 + 2 + 1 reduce calls = 51 at group size 5. Small groups mean a deeper tree and more calls; huge groups mean fewer calls but longer reduce prompts that lose detail. The offline summariser just keeps the first sentence; with Gemini you would ask it to preserve numbers and names.

**Expected output**
```text
final summary: Section 0 says that rule number 0 applies to team 0.
LLM calls: 51 (40 map + 8 + 2 + 1 reduce)  levels: 3
group_size= 2: calls= 81 levels=6
group_size= 5: calls= 51 levels=3
group_size=10: calls= 45 levels=2
```

---

### Q2. Apply - split Markdown into sections with breadcrumbs

**Problem**

Write `split_markdown(md)` that returns, for each deepest section, `{"path": "Handbook > Leave > Sick leave", "text": ..., "level": 3}`. Use a stack of `(level, title)` so sibling and parent headings are handled correctly. Print the number of sections, the first four paths, and an example of the text you would embed.

<details><summary><b>Hint</b></summary>

On a heading of level `L`, first flush the buffered text, then pop the stack while `stack[-1][0] >= L`, then push `(L, title)`. The path is the joined titles on the stack.

</details>

<details><summary><b>Solution</b></summary>

```python
# --- Q2: structure-aware splitting with breadcrumb paths ---------------------
def split_markdown(md):
    """Return [{'path': 'Handbook > Leave > Sick leave', 'text': ..., 'level': 3}, ...] for each deepest section."""
    stack, out, buf = [], [], []

    def flush():
        text = " ".join(buf).strip()
        if text and stack:
            out.append({"path": " > ".join(h for _, h in stack), "text": text, "level": stack[-1][0]})
        buf.clear()

    for line in md.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            flush()
            level = len(m.group(1))
            while stack and stack[-1][0] >= level:                  # pop siblings/deeper headings
                stack.pop()
            stack.append((level, m.group(2).strip()))
        elif line.strip():
            buf.append(line.strip())
    flush()
    return out


def q2():
    secs = split_markdown(HANDBOOK)
    print("sections found:", len(secs))
    for s in secs[:4]:
        print(f"  {s['path']:<45} level={s['level']} chars={len(s['text'])}")
    print("embed text example:", repr(secs[1]["path"] + ": " + secs[1]["text"][:50] + "..."))
```

</details>

**Why it works / what to learn**
Popping equal-or-deeper levels is what makes `Travel` replace `Leave` instead of nesting under it. Embedding `path: text` gives each chunk the context ("Leave > Sick leave") that the sentence itself does not state.

**Expected output**
```text
sections found: 10
  Acme Employee Handbook > Leave > Annual leave level=3 chars=148
  Acme Employee Handbook > Leave > Sick leave   level=3 chars=157
  Acme Employee Handbook > Travel > Flights and hotels level=3 chars=164
  Acme Employee Handbook > Travel > Daily allowance level=3 chars=143
embed text example: 'Acme Employee Handbook > Leave > Sick leave: Sick leave is 12 days per year. A medical certific...'
```

---

### Q3. Build - two-level (hierarchical) retrieval

**Problem**

Build an index of **section summaries** (group the subsections by their H2 heading, summarise each), then `two_level_search(question)`: pick the top 2 sections by summary similarity, then rank only the chunks inside them. Compare with `flat_search` on three questions, and print how many vectors each approach scored. Compute the same counts for 100 sections × 100 chunks.

<details><summary><b>Hint</b></summary>

`parents.setdefault(h2_name, []).append(section)`. Stage 1: `S @ qv` over summaries; Stage 2: rank only `cand = chunks of the winning sections`. Scored vectors = `len(names) + len(cand)`.

</details>

<details><summary><b>Solution</b></summary>

```python
# --- Q3: two-level retrieval (summaries first, then chunks inside the winners) -
def build_index(secs):
    parents = {}
    for s in secs:
        parents.setdefault(s["path"].split(" > ")[1], []).append(s)       # group by H2 section
    names = list(parents)
    summaries = [" ".join(toy_summarize(s["text"], 2) for s in parents[n]) + " " + n for n in names]
    return names, parents, fake_embed(summaries)


def two_level_search(question, secs, top_sections=2, k=2):
    names, parents, S = build_index(secs)
    qv = fake_embed([question])[0]
    best = [names[i] for i in np.argsort(-(S @ qv))[:top_sections]]       # stage 1: pick sections
    cand = [s for n in best for s in parents[n]]
    C = fake_embed([s["path"] + ": " + s["text"] for s in cand])
    order = np.argsort(-(C @ qv))[:k]                                     # stage 2: pick chunks inside them
    return [cand[i]["path"] for i in order], len(names) + len(cand)


def flat_search(question, secs, k=2):
    C = fake_embed([s["path"] + ": " + s["text"] for s in secs])
    order = np.argsort(-(C @ fake_embed([question])[0]))[:k]
    return [secs[i]["path"] for i in order], len(secs)


def q3():
    secs = split_markdown(HANDBOOK)
    for q in ["when do I need a medical certificate", "how many characters must a password have", "hotel price limit per night"]:
        h, n_h = two_level_search(q, secs)
        f, n_f = flat_search(q, secs)
        print(f"{q!r}\n   two-level (scored {n_h}): {h[0]}\n   flat      (scored {n_f}): {f[0]}")
    sections, per = 100, 100
    print(f"at scale (100 sections x 100 chunks): flat scores {sections * per:,} vectors; "
          f"two-level scores {sections + 2 * per}")
```

</details>

**Why it works / what to learn**
Both strategies find the right chunk here, but the two-level search scores fewer vectors, and the gap grows with corpus size: 300 vs 10,000 at scale. The risk is that stage 1 must pick the right sections: if a summary omits the detail, the answer is never considered. Mitigate with richer summaries (our toy version keeps two sentences), `top_sections` of 2-3, and a flat fallback.

**Expected output**
```text
'when do I need a medical certificate'
   two-level (scored 9): Acme Employee Handbook > Leave > Sick leave
   flat      (scored 10): Acme Employee Handbook > Leave > Sick leave
'how many characters must a password have'
   two-level (scored 9): Acme Employee Handbook > Security > Passwords
   flat      (scored 10): Acme Employee Handbook > Security > Passwords
'hotel price limit per night'
   two-level (scored 9): Acme Employee Handbook > Travel > Flights and hotels
   flat      (scored 10): Acme Employee Handbook > Travel > Flights and hotels
at scale (100 sections x 100 chunks): flat scores 10,000 vectors; two-level scores 300
```

---

### Q4. Debug - "the bot cannot answer questions about the end of the document"

**Problem**

A teammate builds the context with `doc[: limit_tokens * 4]`. Users ask "Where do I order a standing desk and what is needed?" and the bot says it does not know, although the handbook says it at the very end. Explain the bug and write a fix that selects context by **relevance under a token budget** (use the section list from Q2).

<details><summary><b>Hint</b></summary>

Where does the cut-off fall in a 436-token document with a 120-token limit? Rank sections by similarity to the question and add them while they fit.

</details>

<details><summary><b>Solution</b></summary>

```python
# --- Q4: "the bot can't answer questions about the end of the document" -----
def buggy_context(doc, limit_tokens):
    """BUG: keeps only the first N tokens. Everything after the cut-off silently disappears."""
    return doc[: limit_tokens * 4]


def good_context(secs, question, limit_tokens):
    """FIX: rank sections by relevance, add them while they fit the budget (original order restored)."""
    qv = fake_embed([question])[0]
    scores = fake_embed([s["path"] + ": " + s["text"] for s in secs]) @ qv
    chosen, used = [], 0
    for i in np.argsort(-scores):
        t = tokens(secs[i]["text"]) + 8
        if used + t <= limit_tokens:
            chosen.append(int(i))
            used += t
    return "\n".join(f"[{secs[i]['path']}] {secs[i]['text']}" for i in sorted(chosen))


def q4():
    secs = split_markdown(HANDBOOK)
    question, gold = "Where do I order a standing desk and what is needed?", "doctor's note"
    limit = 120
    print("document tokens :", tokens(HANDBOOK), "| context limit:", limit)
    b = buggy_context(HANDBOOK, limit)
    g = good_context(secs, question, limit)
    print("buggy context has the answer:", gold in b, "| tokens:", tokens(b))
    print("fixed context has the answer:", gold in g, "| tokens:", tokens(g))
```

</details>

**Why it works / what to learn**
Truncating by position silently deletes everything past the limit, and nothing raises an error. Selecting by relevance puts the right section into the prompt whatever its position, and restoring document order keeps it readable. Always log which sections were included, and add a test question about the *last* page of a document.

**Expected output**
```text
document tokens : 436 | context limit: 120
buggy context has the answer: False | tokens: 120
fixed context has the answer: True | tokens: 98
```

---

### Q5. Production - a strategy router with a cost model

**Problem**

Write `route(question, doc_tokens)` choosing between `stuff` (≤ 8k tokens), `retrieve` (lookup in a larger doc), `long_context` (broad question, document fits within half of a 1M-token window) and `map_reduce` (broad question, larger). Write `cost(strategy, doc_tokens)` returning the number of LLM calls and prompt tokens (map-reduce with 400-token chunks and group size 5). Print a table for four documents × two questions.

<details><summary><b>Hint</b></summary>

Detect broad questions with a regex (`summari[sz]e|overview|main points|...`). For map-reduce calls: `n = ceil(tokens/400)` map calls plus `n/5 + n/25 + ...` reduce calls.

</details>

<details><summary><b>Solution</b></summary>

```python
# --- Q5: a strategy router with a cost model ---------------------------------
def route(question, doc_tokens, window=1_000_000, stuff_limit=8_000):
    """Pick a strategy from the document size and the TYPE of question."""
    broad = bool(re.search(r"\b(summari[sz]e|overview|main (points|themes)|key takeaways|what is this (about|document))\b",
                           question.lower()))
    if doc_tokens <= stuff_limit:
        return "stuff"                 # tiny doc: just put it all in the prompt
    if broad:
        return "map_reduce" if doc_tokens > window * 0.5 else "long_context"
    return "retrieve"                  # fact lookup in a big doc: RAG


def cost(strategy, doc_tokens, chunk_tokens=400, k=5, group=5):
    if strategy == "stuff" or strategy == "long_context":
        return {"llm_calls": 1, "prompt_tokens": doc_tokens}
    if strategy == "retrieve":
        return {"llm_calls": 1, "prompt_tokens": k * chunk_tokens}
    n = math.ceil(doc_tokens / chunk_tokens)                                  # map_reduce
    calls, level = n, n
    while level > 1:
        level = math.ceil(level / group)
        calls += level
    return {"llm_calls": calls, "prompt_tokens": doc_tokens + calls * 150}


def q5():
    docs = {"FAQ page": 3_000, "50-page policy": 40_000, "500-page contract set": 600_000, "10,000-page archive": 5_000_000}
    qs = ["What is the notice period?", "Summarise the main points"]
    print(f"{'document':<22} {'question':<28} {'strategy':<13} {'calls':>6} {'prompt tokens':>14}")
    for name, t in docs.items():
        for q in qs:
            st = route(q, t)
            c = cost(st, t)
            print(f"{name:<22} {q:<28} {st:<13} {c['llm_calls']:>6} {c['prompt_tokens']:>14,}")
```

</details>

**Why it works / what to learn**
The same document needs different tools depending on the question: a lookup costs ~2,000 prompt tokens in a 5-million-token archive, while summarising it takes ~15,000 calls. A router prevents both mistakes: using map-reduce for lookups (wasteful) and top-k RAG for global questions (wrong). Replace the keyword test with a small LLM classifier when you need more accuracy, and log the chosen strategy for each request.

**Expected output**
```text
document               question                     strategy       calls  prompt tokens
FAQ page               What is the notice period?   stuff              1          3,000
FAQ page               Summarise the main points    stuff              1          3,000
50-page policy         What is the notice period?   retrieve           1          2,000
50-page policy         Summarise the main points    long_context       1         40,000
500-page contract set  What is the notice period?   retrieve           1          2,000
500-page contract set  Summarise the main points    map_reduce      1876        881,400
10,000-page archive    What is the notice period?   retrieve           1          2,000
10,000-page archive    Summarise the main points    map_reduce     15625      7,343,750
```
