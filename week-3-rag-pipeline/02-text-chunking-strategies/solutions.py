"""
Topic 02 - Text chunking strategies: solutions to practice.md (Q1..Q5).

Run everything:   python solutions.py
Everything is pure Python and runs OFFLINE. Q3 and Q5 use a tiny local stand-in embedder
(hashed bag-of-words) so you can watch the logic work without an API key.
Swap in the real thing with:  from gemini_utils import embed_texts
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))

import hashlib
import re
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Document:
    text: str
    metadata: dict = field(default_factory=dict)


def fake_embed(texts, dim=128):
    """Offline stand-in for embed_texts: hashed bag-of-words, L2-normalised. NOT semantic - demo only."""
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in re.findall(r"\w+", t.lower()):
            out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1.0
    n = np.linalg.norm(out, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return out / n


SAMPLE = (
    "Refund policy\n\n"
    "Customers may request a refund within 30 days of purchase. Refunds are issued to the original "
    "payment method. Digital goods are refundable only if they have not been downloaded.\n\n"
    "Shipping policy\n\n"
    "Orders ship within 2 business days. Delivery to metro cities takes 3 days. Remote areas may take "
    "up to 10 days. Tracking links are emailed after dispatch."
)


# region q1
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
# endregion q1


# region q2
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
# endregion q2


# region q3
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
# endregion q3


# region q4
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
# endregion q4


# region q5
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
# endregion q5


if __name__ == "__main__":
    for fn in (q1, q2, q3, q4, q5):
        print(f"\n===== {fn.__name__} =====")
        fn()
