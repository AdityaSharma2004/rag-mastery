# Practice - RAG evaluation (RAGAs-style metrics)

Five questions, rising difficulty. Attempt each before opening the hint, and the hint before the solution. Every solution is the exact code in `solutions.py` (`python solutions.py` runs all).

**Offline by default.** The "LLM judge" is a transparent heuristic (`toy_judge`: every number in the claim must appear in the context and at least 60% of its content words must) so you can see exactly how each metric is computed. Set `GEMINI_API_KEY` and `judge_supported` / embeddings switch to `gemini-2.5-flash` and `gemini-embedding-001` (numbers then differ). The metrics are written by hand rather than through the `ragas` package because that library's API changes between versions (see the notes). `CHUNKS` (8 support-doc chunks), `DATASET` (4 questions with ground truth and relevant chunk ids), `retrieve`, `generate_answer` (a toy extractive generator that can be told to hallucinate), `sentences` and `content_words` are defined at the top of `solutions.py`.

---

### Q1. Warm-up - retrieval metrics without any LLM

**Problem**

Implement `retrieval_metrics(ranked, relevant, k)` returning hit@k, precision@k, recall@k and MRR. Try four cases: the relevant chunk at rank 1, at rank 3, not retrieved, and two relevant chunks of which only one was found.

<details><summary><b>Hint</b></summary>

`hits = [d for d in ranked[:k] if d in relevant]`. MRR is `1 / position` of the first relevant id (0 if none).

</details>

<details><summary><b>Solution</b></summary>

```python
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
```

</details>

**Why it works / what to learn**
Rank 1 and rank 3 have the same hit@3, recall and precision, but MRR separates them (1.0 vs 0.33): ordering matters when the LLM sees chunks in rank order. Recall is 0.67 in the last case because one of three relevant chunks was never retrieved: no generation trick can fix missing evidence.

**Expected output**
```text
relevant at rank 1       {'hit@k': 1.0, 'precision@k': 0.33, 'recall@k': 1.0, 'mrr': 1.0}
relevant at rank 3       {'hit@k': 1.0, 'precision@k': 0.33, 'recall@k': 1.0, 'mrr': 0.33}
not retrieved            {'hit@k': 0.0, 'precision@k': 0.0, 'recall@k': 0.0, 'mrr': 0.0}
two relevant, one found  {'hit@k': 1.0, 'precision@k': 0.67, 'recall@k': 0.67, 'mrr': 1.0}
```

---

### Q2. Apply - faithfulness from scratch

**Problem**

Write `faithfulness(answer, contexts, judge)`: split the answer into claims (sentences), ask the judge whether each claim is supported by the context, and return supported/total plus the verdicts. Test a faithful answer and the same answer with an invented claim.

<details><summary><b>Hint</b></summary>

`claims = sentences(answer)`; judge each claim against `" ".join(contexts)`. With Gemini the judge is one `generate_json` call returning `{"supported": bool}`.

</details>

<details><summary><b>Solution</b></summary>

```python
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
```

</details>

**Why it works / what to learn**
The added sentence ("Refunds are also available for 90 days") is not in the context, so faithfulness falls to 0.67 and the verdict list shows *which* claim is invented, which is exactly what you want when debugging hallucination.

**Expected output**
```text
faithful answer: faithfulness = 1.00
   supported   | KiranaBooks refunds a subscription within 14 days of purchase.
   supported   | After 14 days payments are non-refundable.
answer with an invented claim: faithfulness = 0.67
   supported   | KiranaBooks refunds a subscription within 14 days of purchase.
   supported   | After 14 days payments are non-refundable.
   UNSUPPORTED | Refunds are also available for 90 days after purchase.
```

---

### Q3. Build - the four metrics for one sample

**Problem**

Implement `context_precision` (rank-aware), `context_recall` (share of ground-truth statements supported by the retrieved context), `answer_relevancy` (embedding similarity between question and answer) and use them with `faithfulness` in `evaluate_sample`. Run one question with a normal generator and with a hallucinating generator.

<details><summary><b>Hint</b></summary>

Context precision = mean of `precision@i` taken at each position `i` where the chunk is relevant. Context recall = judged ground-truth sentences / all ground-truth sentences.

</details>

<details><summary><b>Solution</b></summary>

```python
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
```

</details>

**Why it works / what to learn**
Retrieval metrics (precision 1.0, recall 1.0) are identical for both runs because retrieval did not change, while faithfulness falls from 1.0 to 0.5 for the hallucinating generator: the metrics localise the fault in generation. Our `answer_relevancy` is simplified (cosine of question and answer, so small absolute values are normal with the toy embedder); RAGAs regenerates questions from the answer instead.

**Expected output**
```text
retrieved ids: [0] | answer: KiranaBooks refunds a subscription within 14 days of purchase, provided you have not exported more than 50 invoices.
normal generator         {'context_precision': 1.0, 'context_recall': 1.0, 'faithfulness': 1.0, 'answer_relevancy': 0.26}
hallucinating generator  {'context_precision': 1.0, 'context_recall': 1.0, 'faithfulness': 0.5, 'answer_relevancy': 0.22}
```

---

### Q4. Debug - can you trust the judge?

**Problem**

Your faithfulness numbers look too good. Using eight hand-labelled (claim, context, verdict) pairs, half of them contradictions, measure each judge's agreement with the human labels, false positives and false negatives. Compare a "lenient" judge (any word overlap = supported) with the current judge. What do you learn?

<details><summary><b>Hint</b></summary>

False positives (judge says supported, human says not) inflate faithfulness, which is the dangerous direction for a hallucination metric.

</details>

<details><summary><b>Solution</b></summary>

```python
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
```

</details>

**Why it works / what to learn**
The lenient judge approves every contradiction (4 false positives, 50% agreement). The better judge catches wrong numbers but still misses negation ("never expires") and a unit swap (10 minutes vs 10 days). **Always calibrate a judge on labelled pairs before trusting its scores**, then improve the prompt (or model) and re-measure. With `gemini-2.5-flash` expect better agreement, but not perfection.

**Expected output**
```text
lenient judge                      agreement=0.50 false_positives=4 false_negatives=0
current judge (offline toy judge)  agreement=0.75 false_positives=2 false_negatives=0
lesson: lenient judges accept contradictions; even the better toy judge misses negation ('never') and unit swaps ('10 days' vs '10 minutes'). Calibrate before trusting any judge
```

---

### Q5. Production - a harness with a regression gate

**Problem**

Write `run_config(name, k, hallucinate)` that evaluates a configuration over the whole dataset and returns averaged metrics as one JSON line, and `regression_gate(baseline, candidate, max_drop)` that fails when any metric drops by more than the tolerance. Compare config A (k=3, baseline), B (k=1) and C (k=3 with a hallucinating generator).

<details><summary><b>Hint</b></summary>

Average each metric over the dataset; for the gate compare `baseline[m] - candidate[m] > max_drop`. Return `(passed, failures)` so CI can print the reason.

</details>

<details><summary><b>Solution</b></summary>

```python
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
```

</details>

**Why it works / what to learn**
Config B passes (this tiny corpus does not need k=3), but config C fails on faithfulness (1.0 → 0.5) and relevancy, so the gate blocks the change and says why. Wire this into CI: run the harness on every pull request, store the JSON lines as artifacts, and add cost/latency columns. With only 4 questions the gate is a demonstration; use 30-100 for real decisions.

**Expected output**
```text
{"config": "A: k=3 (baseline)", "context_precision": 1.0, "context_recall": 1.0, "faithfulness": 1.0, "answer_relevancy": 0.41}
{"config": "B: k=1", "context_precision": 1.0, "context_recall": 1.0, "faithfulness": 1.0, "answer_relevancy": 0.41}
{"config": "C: k=3 + hallucinating generator", "context_precision": 1.0, "context_recall": 1.0, "faithfulness": 0.5, "answer_relevancy": 0.34}
gate B: -> PASS
gate C: -> FAIL {'faithfulness': (1.0, 0.5), 'answer_relevancy': (0.41, 0.34)}
```
