"""
Topic 07 - Handling long documents: solutions to practice.md (Q1..Q5).

Run everything:   python solutions.py
Pure Python + numpy, runs OFFLINE: a toy extractive summariser (first sentence) and hashed bag-of-words
embeddings stand in for Gemini. With GEMINI_API_KEY set, `summarize` uses gemini-2.5-flash instead.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))

import hashlib
import math
import os
import re

import numpy as np


_STOP = set("a an the is are of to in on for and or my i me do does how what when where it be at must per".split())


def fake_embed(texts, dim=512):
    """Offline stand-in for embeddings: hashed bag of (stop-word-free, crudely stemmed) words. NOT semantic."""
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in re.findall(r"[a-z0-9]+", t.lower()):
            if w in _STOP:
                continue
            w = w[:-1] if len(w) > 3 and w.endswith("s") else w
            out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1.0
    n = np.linalg.norm(out, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return out / n


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def toy_summarize(text, max_sentences=1):
    """Offline stand-in for an LLM summary: the first sentence(s)."""
    return " ".join(sentences(text)[:max_sentences])


def summarize(text):
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        from gemini_utils import generate
        return generate(f"Summarize in 1-2 sentences, keeping key numbers:\n\n{text}", temperature=0.1)
    return toy_summarize(text, 1)


def tokens(text):
    return max(1, len(text) // 4)


HANDBOOK = """# Acme Employee Handbook

## Leave

### Annual leave
Employees receive 24 days of paid annual leave per calendar year. Leave must be requested in the HR portal. Unused leave up to 10 days carries over.

### Sick leave
Sick leave is 12 days per year. A medical certificate is required after 3 consecutive days of sick leave. Managers are told only that the employee is unwell.

## Travel

### Flights and hotels
Domestic flights must be booked at least 7 days in advance. Hotels are capped at 6,000 rupees per night in metro cities. Book through the approved travel desk only.

### Daily allowance
Staff on tour receive a food allowance of 1,200 rupees per day. Receipts are not needed below 500 rupees. Claims are paid with the next salary.

## Security

### Passwords
Passwords must be at least 14 characters long. Reuse of the last 5 passwords is blocked. Use the company password manager for storage.

### Devices
Laptops must use full disk encryption. Lost devices must be reported to IT within 2 hours. Personal USB drives are not allowed on office machines.

## Payroll

### Salary dates
Salary is credited on the last working day of the month. Payslips are published in the HR portal on the 25th. Bonus is paid in April.

### Tax proofs
Investment proofs for tax saving are due by 31 January. Late proofs are processed in the next financial year. Contact finance for form 12BB help.

## Onboarding

### First week
New joiners receive a laptop on day one and complete security training in the first 3 days. A buddy is assigned for the first 30 days. Probation lasts 6 months.

### Equipment
Monitors and chairs are ordered through the facilities desk. Standing desks need a doctor's note. Equipment must be returned on the last working day.
"""


# region q1
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
# endregion q1


# region q2
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
# endregion q2


# region q3
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
# endregion q3


# region q4
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
# endregion q4


# region q5
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
# endregion q5


if __name__ == "__main__":
    for fn in (q1, q2, q3, q4, q5):
        print(f"\n===== {fn.__name__} =====")
        fn()
