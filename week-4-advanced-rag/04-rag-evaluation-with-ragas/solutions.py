"""
Week 4 · Topic 04 - RAG evaluation (RAGAs-style metrics written by hand): solutions to practice.md (Q1..Q5).

Run everything:   python solutions.py
OFFLINE by default: the "LLM judge" is a transparent word-overlap heuristic (`toy_judge`) so you can see exactly how each
metric is computed. With GEMINI_API_KEY set, `judge_supported` asks gemini-2.5-flash (generate_json) instead.
The metrics are implemented by hand on purpose: they work with Gemini, need no extra library and make every number
explainable. (The notes show how the `ragas` library maps onto them; its API changes between versions.)
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))

import hashlib, json, os, re

import numpy as np

_STOP = set("a an the is are of to in on for and or my i me do does how what when it be at can with from by up so as that this you your if "
            "will would also".split())


def content_words(text):
    out = set()
    for w in re.findall(r"[a-z0-9,\.]+", text.lower()):
        w = w.strip(",.")
        if not w or w in _STOP:
            continue
        w = w[:-1] if len(w) > 3 and w.endswith("s") else w
        out.add(w)
    return out


def toy_embed(texts, dim=512):
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in content_words(t):
            out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1.0
    n = np.linalg.norm(out, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return out / n


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def toy_judge(claim, context):
    """Offline judge: supported if every number in the claim appears in the context AND >= 60% of its content words do.
    (Crude on purpose: it cannot see negation like 'never expires'; real LLM judges are better but also imperfect.)"""
    cw, xw = content_words(claim), content_words(context)
    numbers = {w for w in cw if any(ch.isdigit() for ch in w)}
    return bool(cw) and numbers <= xw and len(cw & xw) / len(cw) >= 0.6


def llm_judge(claim, context):
    from gemini_utils import generate_json
    prompt = ("Context:\n" + context + "\n\nClaim: " + claim + "\n\nDoes the context directly support the claim? "
              'Answer JSON {"supported": true|false}. Use ONLY the context, not outside knowledge.')
    return bool(generate_json(prompt, temperature=0.0)["supported"])


if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
    from gemini_utils import embed_texts
    JUDGE = llm_judge
    EMBED = lambda t: embed_texts(t, task_type="SEMANTIC_SIMILARITY")
    MODE = "gemini judge"
else:
    JUDGE = toy_judge
    EMBED = toy_embed
    MODE = "offline toy judge"

CHUNKS = [
    "KiranaBooks refunds a subscription within 14 days of purchase, provided you have not exported more than 50 invoices.",      # 0
    "After 14 days payments are non-refundable, but you can cancel at any time to stop the next renewal.",                         # 1
    "The Starter plan costs 499 rupees per month and includes one user and 100 invoices per month.",                               # 2
    "The Growth plan costs 1,299 rupees per month and includes five users and unlimited invoices.",                                # 3
    "Owners can export all company data as a ZIP of CSV files from Settings > Data.",                                              # 4
    "Exports are ready within 10 minutes and the download link expires after 24 hours.",                                           # 5
    "If you lose your phone, use one of the 8 backup codes that were shown when you first set up two-factor authentication.",      # 6
    "Admins on the Growth plan can force two-factor authentication for every user.",                                               # 7
]

DATASET = [   # question, ground-truth answer, ids of chunks that are truly relevant
    ("How long do I have to ask for a refund?", "KiranaBooks refunds a subscription within 14 days of purchase.", {0}),
    ("What does the Growth plan cost?", "The Growth plan costs 1,299 rupees per month and includes five users.", {3}),
    ("How long is the export download link valid?", "The download link expires after 24 hours.", {5}),
    ("What if I lose my phone for two-factor login?", "Use one of the 8 backup codes shown when you set up two-factor authentication.", {6}),
]

CHUNK_VECS = EMBED(CHUNKS)


def retrieve(question, k=3):
    scores = CHUNK_VECS @ EMBED([question])[0]
    return [int(i) for i in np.argsort(-scores)[:k] if scores[i] > 0.01]


def generate_answer(question, ids, hallucinate=False):
    """Toy extractive generator: the context sentence closest to the question; optionally adds an invented sentence."""
    if not ids:
        return "I don't know based on the provided documents."
    qv = EMBED([question])[0]
    cands = [s for i in ids for s in sentences(CHUNKS[i])]
    best = max(cands, key=lambda s: float(EMBED([s])[0] @ qv))
    return best + (" Refunds are also available for 90 days after purchase." if hallucinate else "")


# region q1
# --- Q1: retrieval metrics that need NO LLM (hit@k, precision@k, recall@k, MRR) -
def retrieval_metrics(ranked, relevant, k):
    top = ranked[:k]
    hits = [d for d in top if d in relevant]
    rr = next((1 / p for p, d in enumerate(ranked, 1) if d in relevant), 0.0)
    return {"hit@k": float(bool(hits)), "precision@k": len(hits) / k, "recall@k": len(set(hits)) / len(relevant), "mrr": rr}


def q1():
    cases = {
        "relevant at rank 1": ([3, 7, 2], {3}),
        "relevant at rank 3": ([7, 2, 3], {3}),
        "not retrieved": ([7, 2, 4], {3}),
        "two relevant, one found": ([0, 1, 5], {0, 5, 6}),
    }
    for name, (ranked, rel) in cases.items():
        print(f"{name:<24}", {k: round(v, 2) for k, v in retrieval_metrics(ranked, rel, k=3).items()})
# endregion q1


# region q2
# --- Q2: faithfulness = supported claims / all claims (LLM-as-judge) -------------
def faithfulness(answer, contexts, judge=None):
    judge = judge or JUDGE
    claims = sentences(answer)
    if not claims:
        return 1.0, []
    ctx = " ".join(contexts)
    verdicts = [(c, judge(c, ctx)) for c in claims]
    return sum(v for _, v in verdicts) / len(claims), verdicts


def q2():
    ctx = [CHUNKS[0], CHUNKS[1]]
    good = "KiranaBooks refunds a subscription within 14 days of purchase. After 14 days payments are non-refundable."
    bad = good + " Refunds are also available for 90 days after purchase."
    for name, ans in (("faithful answer", good), ("answer with an invented claim", bad)):
        score, verdicts = faithfulness(ans, ctx)
        print(f"{name}: faithfulness = {score:.2f}")
        for claim, ok in verdicts:
            print(f"   {'supported  ' if ok else 'UNSUPPORTED'} | {claim}")
# endregion q2


# region q3
# --- Q3: the four RAGAs-style metrics for one sample -----------------------------
def context_precision(question, ids, relevant):
    """Rank-aware: mean of precision@i taken at each relevant position (relevant chunks near the top score higher)."""
    got, prec = 0, []
    for pos, d in enumerate(ids, 1):
        if d in relevant:
            got += 1
            prec.append(got / pos)
    return float(np.mean(prec)) if prec else 0.0


def context_recall(ground_truth, contexts, judge=None):
    """Share of ground-truth statements that the retrieved context supports."""
    judge = judge or JUDGE
    parts = sentences(ground_truth)
    ctx = " ".join(contexts)
    return sum(judge(p, ctx) for p in parts) / len(parts)


def answer_relevancy(question, answer):
    """Simplified: cosine(question, answer). (RAGAs regenerates questions from the answer, then compares them.)"""
    q, a = EMBED([question, answer])
    return float(q @ a)


def evaluate_sample(question, ground_truth, relevant, k=3, hallucinate=False):
    ids = retrieve(question, k)
    contexts = [CHUNKS[i] for i in ids]
    answer = generate_answer(question, ids, hallucinate)
    return {
        "context_precision": context_precision(question, ids, relevant),
        "context_recall": context_recall(ground_truth, contexts),
        "faithfulness": faithfulness(answer, contexts)[0],
        "answer_relevancy": answer_relevancy(question, answer),
    }


def q3():
    q, gt, rel = DATASET[0]
    ids = retrieve(q, 3)
    print("retrieved ids:", ids, "| answer:", generate_answer(q, ids))
    for name, h in (("normal generator", False), ("hallucinating generator", True)):
        print(f"{name:<24}", {k: round(v, 2) for k, v in evaluate_sample(q, gt, rel, hallucinate=h).items()})
# endregion q3


# region q4
# --- Q4: can you trust the judge? Calibrate it against human labels --------------
HUMAN_LABELS = [   # (claim, context, human verdict)
    ("Refunds are possible within 14 days.", CHUNKS[0], True),
    ("Refunds are available for 90 days.", CHUNKS[0], False),
    ("The Growth plan costs 1,299 rupees per month.", CHUNKS[3], True),
    ("The Growth plan costs 499 rupees per month.", CHUNKS[3], False),
    ("The download link expires after 24 hours.", CHUNKS[5], True),
    ("The download link never expires.", CHUNKS[5], False),
    ("Exports are ready within 10 minutes.", CHUNKS[5], True),
    ("Exports are ready within 10 days.", CHUNKS[5], False),
]


def lenient_judge(claim, context):
    """A sloppy judge that says yes when ANY content word overlaps (typical of vague judge prompts)."""
    return bool(content_words(claim) & content_words(context))


def agreement(judge):
    preds = [judge(c, ctx) for c, ctx, _ in HUMAN_LABELS]
    truth = [h for *_, h in HUMAN_LABELS]
    acc = sum(p == t for p, t in zip(preds, truth)) / len(truth)
    false_pos = sum(p and not t for p, t in zip(preds, truth))
    false_neg = sum((not p) and t for p, t in zip(preds, truth))
    return acc, false_pos, false_neg


def q4():
    for name, j in (("lenient judge", lenient_judge), (f"current judge ({MODE})", JUDGE)):
        acc, fp, fn = agreement(j)
        print(f"{name:<34} agreement={acc:.2f} false_positives={fp} false_negatives={fn}")
    print("lesson: lenient judges accept contradictions; even the better toy judge misses negation ('never') and unit swaps ('10 days' vs '10 minutes'). Calibrate before trusting any judge")
# endregion q4


# region q5
# --- Q5: an evaluation harness with a regression gate (CI for RAG) ---------------
def run_config(name, k=3, hallucinate=False):
    rows = [evaluate_sample(q, gt, rel, k=k, hallucinate=hallucinate) for q, gt, rel in DATASET]
    agg = {m: round(float(np.mean([r[m] for r in rows])), 2) for m in rows[0]}
    return {"config": name, **agg}


def regression_gate(baseline, candidate, max_drop=0.05):
    """Fail the build if any metric drops by more than `max_drop` versus the baseline."""
    failures = {m: (baseline[m], candidate[m]) for m in baseline if m != "config" and baseline[m] - candidate[m] > max_drop}
    return not failures, failures


def q5():
    base = run_config("A: k=3 (baseline)", k=3)
    small = run_config("B: k=1", k=1)
    bad = run_config("C: k=3 + hallucinating generator", k=3, hallucinate=True)
    for r in (base, small, bad):
        print(json.dumps(r))
    for cand in (small, bad):
        ok, fails = regression_gate(base, cand)
        print(f"gate {cand['config'][:2]} ->", "PASS" if ok else f"FAIL {fails}")
# endregion q5


if __name__ == "__main__":
    for fn in (q1, q2, q3, q4, q5):
        print(f"\n===== {fn.__name__} =====")
        fn()
