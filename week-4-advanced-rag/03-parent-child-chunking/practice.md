# Practice - Parent-child chunking

Five questions, rising difficulty. Attempt each before opening the hint, and the hint before the solution. Every solution is the exact code in `solutions.py` (`python solutions.py` runs all). **Runs offline, no API key** (a hashed bag-of-words stands in for embeddings; with `GEMINI_API_KEY` set, real `gemini-embedding-001` vectors are used and scores differ). `HANDBOOK` (4 short policy sections), `QUESTIONS` (each with the phrases the context must contain: the fact and what the fact is about), `sentences`, `Child` and `covers` are defined at the top of `solutions.py` or in the solutions below.

---

### Q1. Warm-up - build parents and children

**Problem**

Split each handbook section (parent) into sentence-sized children. Give each child a deterministic id, its `parent_id` and its position. Print the counts, three children and the average sizes.

<details><summary><b>Hint</b></summary>

`pid = f"{name}#p"`; child id `f"{pid}.c{n}"`; keep `parents[pid] = text`.

</details>

<details><summary><b>Solution</b></summary>

```python
# --- Q1: split into parents (sections) and children (sentences) with links -----
def build_parent_child(docs, sentences_per_child=1):
    parents, children = {}, []
    for name, text in docs.items():
        pid = f"{name}#p"
        parents[pid] = text
        sents = sentences(text)
        for i in range(0, len(sents), sentences_per_child):
            children.append(Child(f"{pid}.c{len(children)}", pid, i, " ".join(sents[i:i + sentences_per_child])))
    return parents, children


def q1():
    parents, children = build_parent_child(HANDBOOK)
    print("parents:", len(parents), "| children:", len(children))
    for c in children[:3]:
        print(f"  {c.id:<14} -> parent {c.parent_id:<12} | {c.text}")
    print("average chars  parent:", round(np.mean([len(p) for p in parents.values()])),
          " child:", round(np.mean([len(c.text) for c in children])))
```

</details>

**Why it works / what to learn**
Children are about 4x smaller than parents: good for precise matching. Deterministic ids make re-ingestion safe and let you delete by parent.

**Expected output**
```text
parents: 4 | children: 16
  expenses#p.c0  -> parent expenses#p   | Employees claim travel costs using Form X-12.
  expenses#p.c1  -> parent expenses#p   | The form must be filed within 30 days of the trip.
  expenses#p.c2  -> parent expenses#p   | Late claims need director approval.
average chars  parent: 172  child: 42
```

---

### Q2. Apply - search children, return parents

**Problem**

Build `PCStore` that embeds children, and implement `retrieve_child` and `retrieve_parent` (over-fetch children, collapse to unique parents). For each of the four questions check whether the returned context contains **both** the fact and what it is about.

<details><summary><b>Hint</b></summary>

`child_hits(question, k*3)` then keep the first `k` unique `parent_id`s. `covers(context, needs)` checks all needed phrases appear.

</details>

<details><summary><b>Solution</b></summary>

```python
# --- Q2: search CHILDREN, return PARENTS ---------------------------------------
class PCStore:
    def __init__(self, docs, sentences_per_child=1):
        self.parents, self.children = build_parent_child(docs, sentences_per_child)
        self.matrix = EMBED_DOCS([c.text for c in self.children])

    def child_hits(self, question, k=3, min_score=0.01):
        scores = self.matrix @ EMBED_QUERY(question)
        return [(self.children[i], float(scores[i])) for i in np.argsort(-scores)[:k] if scores[i] > min_score]

    def retrieve_child(self, question, k=1):
        return [c.text for c, _ in self.child_hits(question, k)]

    def retrieve_parent(self, question, k=1):
        seen, out = set(), []
        for c, _ in self.child_hits(question, k * 3):          # over-fetch children, then collapse to unique parents
            if c.parent_id not in seen:
                seen.add(c.parent_id)
                out.append(self.parents[c.parent_id])
            if len(out) == k:
                break
        return out


def covers(context_pieces, needs):
    text = " ".join(context_pieces).lower()
    return all(n in text for n in needs)


def q2():
    store = PCStore(HANDBOOK)
    q = "when must the form be filed after a trip"
    print("child context :", store.retrieve_child(q))
    print("parent context:", store.retrieve_parent(q))
    for name, fn in (("child-only", store.retrieve_child), ("parent", store.retrieve_parent)):
        ok = sum(covers(fn(qq), needs) for qq, needs in QUESTIONS.items())
        print(f"{name:<10} context answers {ok}/{len(QUESTIONS)} questions")
```

</details>

**Why it works / what to learn**
The child "The form must be filed within 30 days of the trip" is the *right hit* but cannot answer by itself (which form?). The parent mentions Form X-12. Child-only answers 0 of 4; parent answers 4 of 4.

**Expected output**
```text
child context : ['The form must be filed within 30 days of the trip.']
parent context: ['Employees claim travel costs using Form X-12. The form must be filed within 30 days of the trip. Late claims need director approval. Receipts above 500 rupees must be attached.']
child-only context answers 0/4 questions
parent     context answers 4/4 questions
```

---

### Q3. Build - sentence-window retrieval

**Problem**

Implement `retrieve_window(store, question, k, window)` returning the matched sentence plus `window` neighbours on each side (siblings of the same parent only). Compare windows 0, 1 and 2 with the full parent for context length and whether the answer is covered.

<details><summary><b>Hint</b></summary>

Sort the siblings of the hit by `pos`, find the hit's index `i`, slice `[max(0, i-w) : i+w+1]`.

</details>

<details><summary><b>Solution</b></summary>

```python
# --- Q3: sentence-window retrieval (child +/- n neighbours, same parent only) --
def retrieve_window(store, question, k=1, window=1):
    out = []
    for c, _ in store.child_hits(question, k):
        sibs = sorted((x for x in store.children if x.parent_id == c.parent_id), key=lambda x: x.pos)
        i = next(j for j, x in enumerate(sibs) if x.id == c.id)
        out.append(" ".join(x.text for x in sibs[max(0, i - window): i + window + 1]))
    return out


def q3():
    store = PCStore(HANDBOOK)
    q = "when must the laptop be returned after leaving"
    for w in (0, 1, 2):
        ctx = retrieve_window(store, q, k=1, window=w)
        print(f"window={w}: {len(ctx[0]):>3} chars | covers: {covers(ctx, QUESTIONS[q])} | {ctx[0][:70]}")
    print("parent   :", len(store.retrieve_parent(q)[0]), "chars")
```

</details>

**Why it works / what to learn**
Window 0 is just the child and fails. Window 1 already includes the sentence that names the subject ("company laptop"), with less text than the parent. Choose the smallest window that keeps answers covered, and never cross a parent boundary.

**Expected output**
```text
window=0:  45 chars | covers: False | It must be returned within 7 days of leaving.
window=1: 131 chars | covers: True | Every new joiner receives a company laptop. It must be returned within
window=2: 171 chars | covers: True | Every new joiner receives a company laptop. It must be returned within
parent   : 171 chars
```

---

### Q4. Debug - "the prompt has the same section three times"

**Problem**

`buggy_parents` returns one parent per child hit. For a question matching several sentences of the same section, the prompt repeats the section and grows. Reproduce it, then write `good_parents` that groups children by parent, ranks parents by their **best child score**, removes duplicates and respects a character budget.

<details><summary><b>Hint</b></summary>

`best[pid] = max(best.get(pid, 0), score)`; iterate parents in descending `best`; add while `used + len(text) <= max_chars`.

</details>

<details><summary><b>Solution</b></summary>

```python
# --- Q4: "the prompt contains the same section three times" ----------------------
def buggy_parents(store, question, k=3):
    """BUG: one parent per child hit -> duplicates, and no limit on context size."""
    return [store.parents[c.parent_id] for c, _ in store.child_hits(question, k)]


def good_parents(store, question, k=3, max_chars=400):
    """FIX: group children by parent, rank parents by their BEST child score, dedupe, respect a size budget."""
    best = {}
    for c, s in store.child_hits(question, k * 3):
        best[c.parent_id] = max(best.get(c.parent_id, 0.0), s)
    out, used = [], 0
    for pid in sorted(best, key=lambda p: -best[p]):
        t = store.parents[pid]
        if used + len(t) <= max_chars:
            out.append(t)
            used += len(t)
    return out


def q4():
    store = PCStore(HANDBOOK)
    q = "when must the form be filed after a trip, and are late claims allowed"
    b, g = buggy_parents(store, q), good_parents(store, q)
    print("buggy: parents returned:", len(b), "| unique:", len(set(b)), "| chars:", sum(map(len, b)))
    print("fixed: parents returned:", len(g), "| unique:", len(set(g)), "| chars:", sum(map(len, g)))
```

</details>

**Why it works / what to learn**
Three child hits from one section produce three copies of it (1 unique in 3). The fix returns each parent once, ordered by relevance, within a budget: less noise, fewer tokens. Parent-child without dedupe and budgeting often makes prompts worse.

**Expected output**
```text
buggy: parents returned: 3 | unique: 1 | chars: 528
fixed: parents returned: 2 | unique: 2 | chars: 347
```

---

### Q5. Production - idempotent ingestion and a mode comparison

**Problem**

Write an `ingest` that stores parents and children by deterministic id and prove that ingesting twice does not duplicate. Then compare three retrieval modes (child, window ±1, parent) on answerable questions and average context size, and decide which you would ship.

<details><summary><b>Hint</b></summary>

Use dict updates keyed by id. For the comparison reuse `covers`, `retrieve_window` and `retrieve_parent`.

</details>

<details><summary><b>Solution</b></summary>

```python
# --- Q5: idempotent ingestion + comparison of child / window / parent ------------
def ingest(store_dict, docs):
    """Deterministic ids -> re-ingesting the same docs overwrites instead of duplicating."""
    parents, children = build_parent_child(docs)
    store_dict["parents"].update(parents)
    for c in children:
        store_dict["children"][c.id] = c
    return len(store_dict["parents"]), len(store_dict["children"])


def q5():
    state = {"parents": {}, "children": {}}
    print("ingest #1 (parents, children):", ingest(state, HANDBOOK))
    print("ingest #2 (parents, children):", ingest(state, HANDBOOK), "<- no duplicates")
    store = PCStore(HANDBOOK)
    modes = {
        "child (1 sentence)": lambda q: store.retrieve_child(q, 1),
        "window +-1": lambda q: retrieve_window(store, q, 1, 1),
        "parent (section)": lambda q: store.retrieve_parent(q, 1),
    }
    print(f"{'mode':<20} {'answerable':>10} {'avg context chars':>18}")
    for name, fn in modes.items():
        ok = sum(covers(fn(q), n) for q, n in QUESTIONS.items())
        chars = np.mean([sum(map(len, fn(q))) for q in QUESTIONS])
        print(f"{name:<20} {ok:>5}/{len(QUESTIONS)} {chars:>18.0f}")
```

</details>

**Why it works / what to learn**
Stable ids keep the store consistent across runs. The table shows the quality/cost trade-off: the child is cheapest but useless alone; the parent answers everything at ~4x the context; the window is a cheaper middle that sometimes misses (3 of 4). On your data, measure and pick the smallest context that keeps answerability near 100%.

**Expected output**
```text
ingest #1 (parents, children): (4, 16)
ingest #2 (parents, children): (4, 16) <- no duplicates
mode                 answerable  avg context chars
child (1 sentence)       0/4                 42
window +-1               3/4                118
parent (section)         4/4                170
```
