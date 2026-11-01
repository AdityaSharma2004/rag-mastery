# 04 - RAG evaluation (RAGAs-style metrics)

> Week 4 · Topic 4 of 7 · Time: ~90 min · Runs offline with a transparent toy judge; real `gemini-2.5-flash` judge when a key is set

## 🎯 What you will learn

- You will be able to separate **retrieval quality** from **generation quality** and measure each.
- You will be able to compute **hit@k, precision@k, recall@k and MRR** without any LLM.
- You will be able to implement **faithfulness, context precision, context recall and answer relevancy** by hand with an LLM judge.
- You will be able to **calibrate a judge** against human labels and spot its blind spots.
- You will be able to build an **evaluation harness with a regression gate** so a bad change fails the build.
- You will be able to map these metrics onto the **`ragas`** library (and know its version caveats).

## 🧠 The big idea in plain words

"It seems to work" is not engineering. Every change in this course (chunk size, overlap, HyDE, parent-child, prompts) *might* help or hurt. Evaluation turns opinions into numbers: you keep a small **golden dataset** of questions with the right answers, run your pipeline, score the outputs, and compare before/after.

A RAG answer can fail in two different places, and you must know which:

```
question ─► RETRIEVAL ─► chunks ─► GENERATION ─► answer
            did we fetch the right facts?     did the model use them faithfully and answer the question?
```

Laravel analogy: **PHPUnit + CI**. Metrics are your test assertions; the golden dataset is your test fixtures; the regression gate is a failing pipeline when someone breaks a feature.

## 📚 Concepts, step by step

### 1. The golden dataset

Each row: `question`, `ground_truth` answer, and ideally the **ids of chunks that truly contain the answer**. Start with 30-50 real questions (support tickets, search logs, domain experts). Include:
- easy and hard questions, paraphrases and typos,
- **unanswerable** questions (the right behaviour is to refuse),
- questions near chunk boundaries.
Version it in git. Do not tune on the same questions you report (keep a small held-out set).

### 2. Retrieval metrics (no LLM needed)

Given the ranked ids and the set of relevant ids:

| Metric | Meaning |
|---|---|
| **hit@k** | was at least one relevant chunk in the top k? |
| **precision@k** | share of the top k that is relevant |
| **recall@k** | share of all relevant chunks that appear in the top k |
| **MRR** | mean of 1 / rank of the first relevant chunk |

They are cheap, deterministic and fast; run them on every change (Q1).

### 3. Generation metrics need a judge

Correctness of free text cannot be checked with `==`. We use an **LLM as judge**: give it the context and a claim and ask for a verdict.

- **Faithfulness** = (claims in the answer supported by the context) / (all claims). Splits the answer into statements and checks each against the retrieved context. Low faithfulness = **hallucination**.
- **Answer relevancy** = does the answer address the question? RAGAs generates questions from the answer and compares them with the original; our simplified version uses embedding similarity between question and answer.
- **Context precision** = are the retrieved chunks relevant, and are relevant ones ranked high? (rank-aware average of precision at each relevant position)
- **Context recall** = does the retrieved context contain what is needed to produce the ground truth? (share of ground-truth statements supported by the context)
- **Answer correctness** = compare answer to ground truth (judge or embedding + facts).

Diagnosis table:

| Faithfulness | Context recall | Meaning | Fix |
|---|---|---|---|
| low | high | model ignores/invents despite good context | prompt/grounding, temperature, model |
| high | low | model is faithful to **incomplete** context | retrieval: chunking, k, query expansion |
| high | high | but answer wrong? check the ground truth / relevancy | dataset or question understanding |
| low | low | everything is broken | start with retrieval |

### 4. Judges are imperfect: calibrate them

LLM judges can be lenient (say "supported" for contradictions), biased toward longer answers, or prefer their own style ("self-preference"). Before trusting a metric:
1. Hand-label 30-100 (claim, context) pairs.
2. Measure judge **agreement**, false positives and false negatives (Q4).
3. Fix the judge prompt ("use ONLY the context", "numbers must match"), use `temperature=0`, ask for a short reason, and re-measure.
4. Spot-check low and high scores manually every release.

### 5. A regression gate (CI for RAG)

Store the baseline metrics. On every change run the harness; fail if any metric drops more than a tolerance (e.g. 0.05) (Q5). Report absolute numbers, the dataset version and the pipeline config, so results are reproducible. Add latency and token cost to the report; quality gains that double cost are a decision, not a free win.

### 6. Variance and sample size

With 30 questions one flipped answer moves a metric by 3 points. Use enough questions, fix random seeds/temperatures, repeat stochastic evaluations 2-3 times and compare ranges, not single numbers. Prefer paired comparisons (same questions, two configs).

### 7. The `ragas` library (version note)

RAGAs packages these metrics (`faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`) and runs them with an LLM and embedding model you supply. Its **API has changed between releases** (dataset classes, metric imports, LLM/embedding wrappers), and it must be configured to use Gemini (a LangChain or LlamaIndex Gemini wrapper) instead of its OpenAI default. Because we cannot guarantee the exact calls for your installed version, **this course implements the metrics by hand with `generate_json`** so they always work with `gemini-2.5-flash`. When you adopt `ragas`, read its current docs for your version, point it at Gemini wrappers, and verify that its numbers broadly agree with these hand-written ones on a few samples.

Rough mapping: `Faithfulness`→our `faithfulness`; `ResponseRelevancy`/`answer_relevancy`→our simplified relevancy; `LLMContextPrecisionWithReference`/`context_precision`→our rank-aware precision; `LLMContextRecall`/`context_recall`→our recall.

### 8. Online evaluation

Offline sets miss real traffic. In production log question, retrieved ids, scores, answer, latency and user feedback (thumbs); sample failures weekly into the golden set; monitor refusal rate, faithfulness on a sample, and "no relevant chunk" rate.

## 💻 Code walkthrough

### Stage 1 - retrieval metrics (Q1)

```python
hits = [d for d in ranked[:k] if d in relevant]
rr = next((1 / p for p, d in enumerate(ranked, 1) if d in relevant), 0.0)
{"hit@k": float(bool(hits)), "precision@k": len(hits)/k, "recall@k": len(set(hits))/len(relevant), "mrr": rr}
```

### Stage 2 - faithfulness (Q2)

```python
claims = sentences(answer)
verdicts = [(c, judge(c, " ".join(contexts))) for c in claims]
score = sum(v for _, v in verdicts) / len(claims)
```
With Gemini the judge is:
```python
generate_json(f"Context:\n{ctx}\n\nClaim: {claim}\n\nDoes the context directly support the claim? "
              'Answer JSON {"supported": true|false}. Use ONLY the context.', temperature=0.0)
```
Batch many claims in one call to save quota (ask for a list of verdicts), and cache by `hash(claim + context)`.

### Stage 3 - the four metrics (Q3)

`context_precision` averages precision@i at each relevant rank; `context_recall` judges each ground-truth sentence against the context; `answer_relevancy` embeds question and answer (task type `SEMANTIC_SIMILARITY`).

### Stage 4 - calibration (Q4)

Eight hand-labelled pairs (supported and contradicted) give accuracy, false positives and false negatives for each judge.

### Stage 5 - harness and gate (Q5)

`run_config` evaluates a configuration over the dataset and prints one JSON line (easy to diff, store, plot). `regression_gate(baseline, candidate)` returns `(passed, failures)`.

## ⚠️ Common mistakes & debugging

| Symptom | Cause | Fix |
|---|---|---|
| Faithfulness 1.0 but answers are wrong | Faithful to wrong/incomplete context | Check context recall and retrieval |
| Scores jump between runs | Judge temperature > 0, tiny dataset | `temperature=0`, more questions, repeat |
| Judge says "supported" for contradictions | Lenient prompt | "Use only the context; numbers/units must match"; calibrate |
| Metrics improve but users unhappy | Dataset unrepresentative | Sample real traffic into the dataset |
| Tuning on the test set | Same questions for tuning and reporting | Hold out a set |
| Eval too slow / hits 429 | One LLM call per claim | Batch claims, cache, throttle, free-tier limits |
| `ragas` import/signature errors | Version changes | Pin the version, read its docs, or keep the hand-written metrics |
| No unanswerable questions | Refusal behaviour untested | Add them and measure correct refusals |
| Only averages reported | Hides failures | Inspect the worst 10 rows every run |

## 🚀 Pro-level insights (what the top 2% know)

- **Evaluate retrieval and generation separately, always.** Most teams discover that 70% of their failures are retrieval failures.
- **Make eval part of the dev loop:** a 40-question run in under two minutes with caching means you actually run it.
- **Use deterministic checks where possible** (regex for numbers, ids, refusal sentence) and the LLM judge only for the fuzzy part.
- **Judge with a different model family if you can** to reduce self-preference; at least use a stronger model/temperature 0.
- **Track cost and latency per query** next to quality; show the trade-off table in every PR.
- **Slice metrics** by question type, tenant, language, document source: averages hide pockets of failure.
- **Synthetic data:** have an LLM generate questions from chunks to bootstrap a dataset, then have humans prune; never trust purely synthetic sets.
- **Golden sets rot:** documents change; revalidate ground truth when the corpus changes.

## 🧪 Mini lab (10-15 min)

1. Build 10 questions from your own chunks (answer, relevant chunk id) plus 2 unanswerable ones.
2. Compute retrieval metrics for k = 1, 3, 5.
3. Hand-label 10 (claim, context) pairs and measure the judge's agreement.
4. Introduce a deliberate regression (k=1 or a worse prompt) and confirm the gate catches it.
5. Write down which metric moved, and what that tells you about where the fault is.

## ✅ Recap checklist

- [ ] I keep a versioned golden dataset including unanswerable questions.
- [ ] I measure retrieval and generation separately.
- [ ] I can compute hit/precision/recall/MRR and explain each.
- [ ] I can define faithfulness, context precision, context recall, answer relevancy.
- [ ] I calibrate my judge against human labels.
- [ ] I have a regression gate and track cost/latency.

## 🎤 Interview questions

**1. How do you evaluate a RAG system?**
Golden dataset; retrieval metrics (hit@k, recall@k, MRR); generation metrics (faithfulness, answer relevancy, correctness) with a calibrated LLM judge; regression gates in CI; online monitoring.

**2. What is faithfulness?**
The fraction of statements in the answer supported by the retrieved context; low faithfulness signals hallucination.

**3. Faithfulness is high but users say answers are wrong. Why?**
The context may be incomplete or wrong (low context recall/precision), so the model is faithful to bad evidence; fix retrieval.

**4. What are the risks of LLM-as-judge?**
Leniency, bias (length, self-preference), non-determinism, cost. Mitigate with calibration against human labels, temperature 0, strict prompts, deterministic checks and spot checks.

**5. How would you prevent quality regressions?**
Store baseline metrics, run the harness on every change, and fail the build when a metric drops beyond a tolerance; slice results and review worst cases.

## 🔗 Next

**05 - Caching embeddings and retrieval:** make the pipeline fast and cheap by caching embeddings, query results and answers (Redis, with an in-memory fallback).
