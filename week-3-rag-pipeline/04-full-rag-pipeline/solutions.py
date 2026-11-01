"""
Topic 04 - The full RAG pipeline: solutions to practice.md (Q1..Q5).

Run everything:   python solutions.py
Runs OFFLINE by default: a toy embedder (stop-word-free bag of words + 3 hand-made synonyms) and a toy
"extractive" LLM stand in for Gemini, so you can watch every moving part without an API key.
Set GEMINI_API_KEY and the SAME code uses gemini-embedding-001 and gemini-2.5-flash instead
(numbers and wording then differ from the expected outputs, which are marked for the offline run).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))

import hashlib, json, os, re, statistics, time
from dataclasses import dataclass, field

import numpy as np

# --- plug points: real Gemini when a key exists, offline toys otherwise -------
STOP = set("a an the is are of to in on for and or my i me can do does how what it be you your after with".split())
SYN = {"money": "refund", "back": "refund", "refunds": "refund", "refunded": "refund"}


def _tokens(text):
    out = []
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if w not in STOP:
            out.append(SYN.get(w, w))
    return out


def toy_embed(texts, dim=256):
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in _tokens(t):
            out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1.0
    n = np.linalg.norm(out, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return out / n


REFUSAL = "I don't know based on the provided documents."

SYSTEM = (
    "You are a support assistant. Answer the question using ONLY the numbered sources provided.\n"
    "After every fact, cite the source number like [1]. Several sources: [1][2].\n"
    f'If the sources do not contain the answer, reply exactly: "{REFUSAL}"\n'
    "Be concise. Never use outside knowledge."
)


def toy_llm(prompt, system=None, temperature=0.1):
    """Offline 'LLM': returns the source sentence sharing the most content words with the question."""
    question = prompt.split("QUESTION:")[-1].replace("ANSWER:", "")
    qset = set(_tokens(question))
    best, best_n, best_rank = None, 0, None
    for m in re.finditer(r"\[(\d+)\] \(source: [^)]*\)\n(.*?)(?=\n\n\[|\n\nQUESTION:)", prompt, re.S):
        for sent in re.split(r"(?<=[.!?])\s+", m.group(2)):
            n = len(qset & set(_tokens(sent)))
            if n > best_n:
                best, best_n, best_rank = sent, n, m.group(1)
    if best_n < 2:
        return REFUSAL
    return f"{best} [{best_rank}]"


if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
    from gemini_utils import embed_texts, embed_query, generate
    EMBED_DOCS = lambda texts: embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")
    EMBED_QUERY = lambda q: embed_query(q)
    LLM = generate
    MODE = "gemini"
else:
    EMBED_DOCS = toy_embed
    EMBED_QUERY = lambda q: toy_embed([q])[0]
    LLM = toy_llm
    MODE = "offline toys"


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    meta: dict = field(default_factory=dict)


@dataclass
class Hit:
    chunk: Chunk
    score: float
    rank: int


def chunk_text(text, max_chars=300):
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, current = [], ""
    for s in sentences:
        if current and len(current) + len(s) + 1 > max_chars:
            chunks.append(current)
            current = s
        else:
            current = f"{current} {s}".strip()
    if current:
        chunks.append(current)
    return chunks


KIRANA_DOCS = {
    "refunds.md": (
        "KiranaBooks refunds a subscription within 14 days of purchase, provided you have not exported "
        "more than 50 invoices. After 14 days payments are non-refundable, but you can cancel at any time "
        "to stop the next renewal. Approved refunds reach your bank account in 5 to 7 working days."
    ),
    "plans.md": (
        "The Starter plan costs 499 rupees per month and includes one user and 100 invoices per month. "
        "The Growth plan costs 1,299 rupees per month and includes five users and unlimited invoices. "
        "Choosing annual billing gives you two months free."
    ),
    "security.md": (
        "You can enable two-factor authentication under Settings > Security. Admins on the Growth plan "
        "can force two-factor authentication for every user. If you lose your phone, use one of the 8 "
        "backup codes that were shown when you first set it up."
    ),
    "export.md": (
        "Owners can export all company data as a ZIP of CSV files from Settings > Data. Exports are ready "
        "within 10 minutes and the download link expires after 24 hours. Backups are kept for 30 days."
    ),
}


# region q1
# --- Q1: build the prompt and verify the citations the model returns --------
CITE = re.compile(r"\[(\d+)\]")


def build_prompt(question, hits):
    blocks = [f"[{h.rank}] (source: {h.chunk.source})\n{h.chunk.text}" for h in hits]
    return "SOURCES:\n" + "\n\n".join(blocks) + f"\n\nQUESTION: {question}\nANSWER:"


def extract_citations(answer, hits):
    """Turn '[1][3]' into real sources; invented numbers like [9] are dropped, not trusted."""
    used = sorted({int(n) for n in CITE.findall(answer)})
    valid = [n for n in used if 1 <= n <= len(hits)]
    return [{"n": n, "source": hits[n - 1].chunk.source, "chunk_id": hits[n - 1].chunk.id} for n in valid], \
           sorted(set(used) - set(valid))


def q1():
    hits = [
        Hit(Chunk("refunds.md#0", "Refunds are possible within 14 days.", "refunds.md"), 0.81, 1),
        Hit(Chunk("plans.md#0", "The Growth plan costs 1,299 rupees per month.", "plans.md"), 0.55, 2),
    ]
    prompt = build_prompt("How long is the refund window?", hits)
    print(prompt)
    print("---")
    good, bad = extract_citations("Refunds are possible for 14 days [1]. Growth is 1,299 rupees [2][9].", hits)
    print("valid citations:", good)
    print("invented citation numbers dropped:", bad)
# endregion q1


# region q2
# --- Q2: ingest -> store -> retrieve on the KiranaBooks docs ----------------
class VectorStore:
    """Smallest useful vector DB: a list of chunks + one matrix. Row i of the matrix = vector of chunks[i]."""
    def __init__(self):
        self.chunks, self.matrix = [], np.zeros((0, 0), dtype=np.float32)

    def add(self, chunks, vectors):
        assert len(chunks) == len(vectors), "one vector per chunk"
        self.matrix = vectors if not self.chunks else np.vstack([self.matrix, vectors])
        self.chunks.extend(chunks)


def ingest(store, docs, max_chars=300):
    chunks = [Chunk(f"{src}#{i}", piece, src, {"pos": i})
              for src, text in docs.items() for i, piece in enumerate(chunk_text(text, max_chars))]
    store.add(chunks, EMBED_DOCS([c.text for c in chunks]))
    return len(chunks)


def retrieve(store, question, k=3, min_score=0.0):
    scores = store.matrix @ EMBED_QUERY(question)            # unit vectors: dot product = cosine
    order = np.argsort(-scores)[:k]
    return [Hit(store.chunks[i], float(scores[i]), r) for r, i in enumerate(order, 1) if scores[i] >= min_score]


def q2():
    store = VectorStore()
    print("mode:", MODE, "| chunks stored:", ingest(store, KIRANA_DOCS))
    for q in ["Can I get my money back after 3 weeks?", "How long does the export download link last?"]:
        print("\nQ:", q)
        for h in retrieve(store, q, k=2):
            print(f"  #{h.rank} {h.score:.2f} {h.chunk.id}")
# endregion q2


# region q3
# --- Q3: the pipeline with a no-answer gate, timings and a trace ------------
@dataclass
class Answer:
    question: str
    text: str
    citations: list
    hits: list
    timings_ms: dict
    refused: bool = False


class RAGPipeline:
    def __init__(self, store, k=3, min_score=0.2, llm=None):
        self.store, self.k, self.min_score = store, k, min_score
        self.llm = llm or LLM
        self.llm_calls = 0

    def ask(self, question, k=None):
        k = k or self.k
        t0 = time.perf_counter()
        hits = retrieve(self.store, question, k=k, min_score=self.min_score)
        t1 = time.perf_counter()
        if not hits:                                           # gate: no relevant chunk -> do NOT call the LLM
            return Answer(question, REFUSAL, [], [], {"retrieve": round((t1 - t0) * 1000, 2)}, refused=True)
        self.llm_calls += 1
        text = self.llm(build_prompt(question, hits), system=SYSTEM, temperature=0.1).strip()
        t2 = time.perf_counter()
        cites, _ = extract_citations(text, hits)
        return Answer(question, text, cites, hits,
                      {"retrieve": round((t1 - t0) * 1000, 2), "generate": round((t2 - t1) * 1000, 2)},
                      refused=REFUSAL in text)


def q3():
    store = VectorStore()
    ingest(store, KIRANA_DOCS)
    pipe = RAGPipeline(store, k=3, min_score=0.2)
    for q in ["How many days do I have to ask for a refund?", "Who won the cricket world cup?"]:
        a = pipe.ask(q)
        print(f"Q: {q}\nA: {a.text}\n   refused={a.refused} cited={[c['source'] for c in a.citations]} "
              f"llm_calls_so_far={pipe.llm_calls}")
# endregion q3


# region q4
# --- Q4: "answers cite the wrong document" - find the bug, add a diagnose() -
def buggy_sort(store):
    """BUG: sorts the chunk list by id (to look tidy) but leaves the matrix rows in the old order."""
    store.chunks.sort(key=lambda c: c.id)


def fixed_sort(store):
    """FIX: permute chunks AND matrix with the same index order (or just never sort)."""
    order = sorted(range(len(store.chunks)), key=lambda i: store.chunks[i].id)
    store.chunks = [store.chunks[i] for i in order]
    store.matrix = store.matrix[order]


def diagnose(pipe, question, gold_phrase):
    hits = retrieve(pipe.store, question, k=pipe.k, min_score=0.0)
    rank = next((h.rank for h in hits if gold_phrase.lower() in h.chunk.text.lower()), None)
    if rank is None:
        return "RETRIEVAL problem: gold chunk not in top-k"
    if gold_phrase.lower() not in pipe.ask(question).text.lower():
        return f"GENERATION problem: gold chunk at rank {rank} but answer lacks it"
    return f"OK: gold chunk at rank {rank}, answer contains it"


def q4():
    q, gold = "How many days do I have to ask for a refund?", "14 days"
    good = VectorStore(); ingest(good, KIRANA_DOCS)
    print("healthy      :", diagnose(RAGPipeline(good, k=1), q, gold))
    broken = VectorStore(); ingest(broken, KIRANA_DOCS)
    buggy_sort(broken)
    print("after bug    :", diagnose(RAGPipeline(broken, k=1), q, gold))
    repaired = VectorStore(); ingest(repaired, KIRANA_DOCS)
    fixed_sort(repaired)
    print("after fix    :", diagnose(RAGPipeline(repaired, k=1), q, gold))
# endregion q4


# region q5
# --- Q5: a mini evaluation harness: hit@k, refusal accuracy, latency --------
EVAL_SET = [
    # (question, gold phrase that must be in the retrieved chunk, should_answer)
    ("How many days do I have to ask for a refund?", "14 days", True),
    ("What does the Growth plan cost per month?", "1,299", True),
    ("How do I turn on two-factor authentication?", "Settings > Security", True),
    ("When does the export download link expire?", "24 hours", True),
    ("How many backup codes do I get?", "8 backup codes", True),
    ("Who won the cricket world cup?", None, False),
    ("What is the capital of France?", None, False),
]


def evaluate(store, k, min_score):
    pipe = RAGPipeline(store, k=k, min_score=min_score)
    hit, answered_ok, refused_ok = 0, 0, 0
    n_ans = sum(1 for *_, s in EVAL_SET if s)
    for q, gold, should in EVAL_SET:
        a = pipe.ask(q, k=k)
        if should:
            hit += any(gold.lower() in h.chunk.text.lower() for h in a.hits)
            answered_ok += (not a.refused)
        else:
            refused_ok += a.refused
    return {"k": k, "min_score": min_score, "hit@k": round(hit / n_ans, 2), "answered": round(answered_ok / n_ans, 2),
            "correct_refusals": f"{refused_ok}/{len(EVAL_SET) - n_ans}", "llm_calls": pipe.llm_calls}


def q5():
    store = VectorStore(); ingest(store, KIRANA_DOCS)
    for k, ms in [(3, 0.0), (3, 0.1), (3, 0.2), (3, 0.3), (1, 0.1)]:
        print(evaluate(store, k, ms))
# endregion q5


if __name__ == "__main__":
    for fn in (q1, q2, q3, q4, q5):
        print(f"\n===== {fn.__name__} =====")
        fn()
