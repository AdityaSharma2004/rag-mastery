"""
Topic 03 - Chunk overlap: solutions to practice.md (Q1..Q5).

Run everything:   python solutions.py
Pure Python, runs OFFLINE, no API key needed (overlap is a text-slicing idea; Gemini is only
needed later, when you embed the chunks).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))

import re
from dataclasses import dataclass, field


@dataclass
class Document:
    text: str
    metadata: dict = field(default_factory=dict)


def windows(text, size, overlap):
    """Character windows: list of (start, end). Validates input, never drops the tail."""
    if not 0 <= overlap < size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")
    out, i = [], 0
    while i < len(text):
        out.append((i, min(i + size, len(text))))
        if i + size >= len(text):
            break
        i += size - overlap
    return out


# A policy text where each sentence is one "fact" the user may ask about.
FACTS = [
    "Refunds are accepted within 30 days.",
    "Shipping takes 3 days to metro cities.",
    "Warranty covers manufacturing defects for 12 months.",
    "Passwords expire after 90 days.",
    "Support is available from 9am to 6pm IST.",
    "Invoices are emailed after payment.",
    "Premium plans include priority support.",
    "Cancellations need 7 days notice.",
]
POLICY = " ".join(FACTS)


# region q1
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
# endregion q1


# region q2
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
# endregion q2


# region q3
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
# endregion q3


# region q4
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
# endregion q4


# region q5
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
# endregion q5


if __name__ == "__main__":
    for fn in (q1, q2, q3, q4, q5):
        print(f"\n===== {fn.__name__} =====")
        fn()
