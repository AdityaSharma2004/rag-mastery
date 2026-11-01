"""
Week 4 · Topic 03 - Parent-child chunking: solutions to practice.md (Q1..Q5).

Run everything:   python solutions.py
Pure Python + numpy, runs OFFLINE (hashed bag-of-words stands in for embeddings; with GEMINI_API_KEY the same code
uses gemini-embedding-001). Expected outputs in practice.md come from the offline run.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))

import hashlib, os, re
from dataclasses import dataclass

import numpy as np

_STOP = set("a an the is are of to in on for and or my i me do does how what when it be at can with from by up so as that this after".split())


def toy_embed(texts, dim=512):
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in re.findall(r"[a-z0-9\-]+", t.lower()):
            if w in _STOP:
                continue
            w = w[:-1] if len(w) > 3 and w.endswith("s") else w
            w = w[:-2] if len(w) > 5 and w.endswith("ed") else w
            out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1.0
    n = np.linalg.norm(out, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return out / n


if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
    from gemini_utils import embed_texts
    EMBED_DOCS = lambda t: embed_texts(t, task_type="RETRIEVAL_DOCUMENT")
    EMBED_QUERY = lambda q: embed_texts([q], task_type="RETRIEVAL_QUERY")[0]
else:
    EMBED_DOCS = toy_embed
    EMBED_QUERY = lambda q: toy_embed([q])[0]

HANDBOOK = {
    "expenses": "Employees claim travel costs using Form X-12. The form must be filed within 30 days of the trip. "
                "Late claims need director approval. Receipts above 500 rupees must be attached.",
    "laptops": "Every new joiner receives a company laptop. It must be returned within 7 days of leaving. "
               "Damage is reported to IT on the same day. Personal software is not allowed on it.",
    "remote": "Remote work is allowed up to 2 days per week. Managers approve the schedule monthly. "
              "Core hours are 11 to 4 for all staff. The office desk booking tool is used on other days.",
    "training": "All staff complete security training every year. The course takes about 2 hours. "
                "Certificates are uploaded to the HR portal. Missing the deadline blocks system access.",
}

# question -> phrases the context MUST contain for the LLM to answer well (the fact + what the fact is about)
QUESTIONS = {
    "when must the form be filed after a trip": ["30 days", "form x-12"],
    "when must the laptop be returned after leaving": ["7 days", "company laptop"],
    "what happens if I miss the training deadline": ["blocks system access", "security training"],
    "how many hours does the course take": ["2 hours", "security training"],
}


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


@dataclass
class Child:
    id: str
    parent_id: str
    pos: int          # index of the sentence inside its parent
    text: str


# region q1
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
# endregion q1


# region q2
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
# endregion q2


# region q3
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
# endregion q3


# region q4
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
# endregion q4


# region q5
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
# endregion q5


if __name__ == "__main__":
    for fn in (q1, q2, q3, q4, q5):
        print(f"\n===== {fn.__name__} =====")
        fn()
