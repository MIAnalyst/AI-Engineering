# DSPy

> Notes written 2026-11-23. Latest release at the time: **DSPy 3.5.0** (PyPI, 2026-11-10), Python `>=3.10,<3.16`. MIT licensed, maintained by Stanford NLP.
> Sources are the official docs (read from the `stanfordnlp/dspy` repo), the RAG tutorial notebook, and the GEPA / MIPROv2 papers. See [Sources](#sources).

```bash
pip install -U dspy
pip install dspy[numpy]   # needed for dspy.Embedder / dspy.retrievers.Embeddings
```

---

## 1. What is DSPy?

DSPy is a framework for **programming** LLM systems instead of **prompting** them. You declare *what* each step should do (typed inputs/outputs), compose steps in plain Python, and let an **optimizer** find the prompts (or weights) that maximize a metric you define.

The problem it solves: hand-written prompts are brittle. They break when you switch models, when the provider updates a model, or when you change another step in your pipeline, and the work doesn't transfer. In DSPy the prompt is a *compiled artifact*, not source code you maintain.

### Three core concepts

| Concept | What it is | Example |
|---|---|---|
| **Signature** | Typed input -> output spec of a task. Inline string or a class. | `'context, question -> response'` |
| **Module** | A strategy for running a signature. Interchangeable. | `dspy.Predict`, `dspy.ChainOfThought`, `dspy.ReAct` |
| **Optimizer** | An algorithm that tunes the program's prompts and/or weights against a metric and a few examples. | `dspy.MIPROv2`, `dspy.GEPA` |

```python
import dspy

dspy.configure(lm=dspy.LM('openai/gpt-4o-mini'))   # reads OPENAI_API_KEY

# Inline signature
qa = dspy.Predict('question: str -> response: str')

# Same task, different strategy: add step-by-step reasoning
qa = dspy.ChainOfThought('question -> response')

# Class-based signature (the docstring becomes the instruction)
class ExtractEvent(dspy.Signature):
    """Extract event details from an email."""
    email: str = dspy.InputField()
    event_name: str = dspy.OutputField()
    date: str = dspy.OutputField()

extract = dspy.Predict(ExtractEvent)
```

A multi-step program is a `dspy.Module`: declare sub-modules in `__init__`, express control flow in `forward`. Every sub-module with an LM call becomes something the optimizer can tune.

`dspy.inspect_history(n=1)` shows the actual prompt DSPy sent. Use it to see what an optimizer changed.

### DSPy vs. LangChain (for context in this vault)
- LangChain: you assemble components and write the prompts yourself.
- DSPy: you declare the task and a metric, and the framework generates the prompts. It is a good fit when you have an eval set and want to improve measurably. It is less useful if you have no metric and no examples.

---

## 2. Why optimize prompts in RAG?

A RAG pipeline has several LLM-facing prompts (answer generation, and in more advanced designs query rewriting, query decomposition, reranking/filtering, multi-hop query generation). Tuning these by hand is slow and fragile. Prompt optimization replaces that loop:

1. You provide a **program** (the RAG pipeline), a **metric**, and a **small trainset** (the docs say it can be as few as 5 to 10 examples, and inputs only is allowed for some optimizers).
2. The optimizer generates candidate instructions and/or few-shot demos, runs them, scores them, and keeps the best.
3. You save the result as a JSON file and load it at serving time. Inference cost is unchanged apart from longer prompts.

What it **can** improve: answer-generation instructions, few-shot demos (including synthetic reasoning traces), and the prompts of any LM-based step (query generators, rerankers, etc.).

What it **does not** improve (my inference from how the optimizers work, not a statement from the docs): the embedding model, chunking, index, and top-k are not tuned by prompt optimizers. Fix those first (see [[Embedding]], [[Best Practice]], [[RAG Architecture]]) and use prompt optimization on top.

Reported real-world gains (from the official GEPA guide, vendor-reported, not independently verified):
- Shopify moved a single-prompt GPT-5 task to a small Qwen model optimized with GEPA, reported as ~75x cheaper and ~2x more reliable.
- Dropbox used DSPy prompt optimizers to move to a smaller model and double accuracy on a relevance judge.

The same program optimizes differently per model, so when a new model ships you can re-run the optimizer and check whether the swap is worth it.

---

## 3. The optimizers

An optimizer takes **(program, metric, trainset)** and returns a new optimized copy. The original program is not mutated (`.compile()` returns a copy).

Each optimizer tunes one or more of three knobs: **instructions**, **demos** (few-shot examples), **weights**.

| Optimizer | Tunes | Idea | Use when |
|---|---|---|---|
| `LabeledFewShot` | demos | Sample `k` labeled examples as demos. No LM calls. | Honest baseline |
| `BootstrapFewShot` | demos | Run the program (or a `teacher`) on train examples, keep traces that pass the metric as demos. | ~10 examples; safe first try |
| `BootstrapFewShotWithRandomSearch` (`BootstrapRS`) | demos | Repeat bootstrapping with random seeds, keep the best on a valset. | 50+ examples |
| `KNNFewShot` | demos (at inference) | Pick nearest training examples per input. | Inputs vary a lot |
| `COPRO` | instructions | Breadth/depth hill-climbing over instruction candidates. | Demos fine, wording wrong |
| `MIPROv2` | instructions + demos | Bootstrap demos, propose grounded instructions, Bayesian optimization over combinations. | Both need tuning; 200+ examples and 40+ trials recommended |
| `SIMBA` | rules / demos | Mini-batch; finds high-variance failures and writes self-reflective rules. | Failures share a nameable pattern |
| `GEPA` | instructions | Reflective evolution using text feedback and a Pareto frontier. | Strong reflection LM, feedback-rich metric |
| `BootstrapFinetune` | weights | Distill the prompted program into fine-tuned weights. | Prompt-only plateaued and the model is tunable |
| `BetterTogether` | both | Sequence like `"p -> w -> p"`. | Prompt and weight tuning both tried |
| `Ensemble` | n/a | Combine several compiled programs. | Several competent candidates |

Official starting guidance:
- ~10 examples: `BootstrapFewShot`.
- 50+ examples: `BootstrapFewShotWithRandomSearch`.
- Instruction-only (keep the prompt 0-shot): `MIPROv2` configured for 0-shot.
- 200+ examples and willing to run 40+ trials: `MIPROv2`.
- Large LM works and you need an efficient program: `BootstrapFinetune` on a small LM.
- Prompt-only optimizers work with closed APIs. `BootstrapFinetune` needs an LM that exposes fine-tuning.

Practical heuristics from the docs:
- Demo tuning tends to overfit the trainset; instruction tuning tends to generalize better.
- Most teams start prompt-only and fine-tune last. Fine-tuning's marginal lift is usually small and its cost much larger.
- Compile once, save, serve many times. That is how the optimization cost amortizes.
- Cost varies a lot: the docs say a simple run is on the order of $2 and ~10 minutes, from a few cents to tens of dollars (and GEPA/MIPROv2 runs on big setups can cost far more). Check your budget before running on large LMs or datasets.

---

## 4. Worked example: optimize a RAG pipeline with MIPROv2

This follows the official RAG tutorial (RAG-QA Arena "Tech" questions, StackExchange-based).

### 4.1 Setup, data, and splits

```python
import random, orjson, dspy
from dspy.utils import download

dspy.configure(lm=dspy.LM('openai/gpt-4o-mini'))

download("https://huggingface.co/dspy/cache/resolve/main/ragqa_arena_tech_examples.jsonl")
with open("ragqa_arena_tech_examples.jsonl") as f:
    data = [orjson.loads(line) for line in f]

# with_inputs() marks which fields are inputs; the rest are labels/metadata
data = [dspy.Example(**d).with_inputs('question') for d in data]

random.Random(0).shuffle(data)
trainset, devset, testset = data[:200], data[200:500], data[500:1000]
```

Tutorial guidance on splits:
- 30-300 examples each for train and validation is typical.
- For prompt optimizers, it is often better to pass *more* validation than training.
- If you don't pass a `valset`, MIPROv2 splits what you give it **20% train / 80% val**.
- Keep dev (iterate on it) and test (final held-out) separate.

### 4.2 Metric

Answers are long, so exact match is useless. `SemanticF1` is a DSPy module (an LM judge) that scores recall of key ideas from the gold answer and precision of the response.

```python
from dspy.evaluate import SemanticF1
metric = SemanticF1(decompositional=True)

evaluate = dspy.Evaluate(devset=devset, metric=metric, num_threads=24,
                         display_progress=True, display_table=2)
```

### 4.3 Retriever and RAG module

```python
download("https://huggingface.co/dspy/cache/resolve/main/ragqa_arena_tech_corpus.jsonl")
with open("ragqa_arena_tech_corpus.jsonl") as f:
    corpus = [orjson.loads(line)['text'][:6000] for line in f]   # truncate long docs

embedder = dspy.Embedder('openai/text-embedding-3-small', dimensions=512)
search = dspy.retrievers.Embeddings(embedder=embedder, corpus=corpus, k=5)
# needs `pip install faiss-cpu`, or pass brute_force_threshold=30_000

class RAG(dspy.Module):
    def __init__(self):
        self.respond = dspy.ChainOfThought('context, question -> response')

    def forward(self, question):
        context = search(question).passages
        return self.respond(context=context, question=question)
```

Any Python that returns passages works as a retriever (a vector DB client, a LangChain retriever, etc.).

### 4.4 Optimize

```python
tp = dspy.MIPROv2(metric=metric, auto="medium", num_threads=24)
optimized_rag = tp.compile(RAG(), trainset=trainset,
                           max_bootstrapped_demos=2, max_labeled_demos=2)

evaluate(optimized_rag)

optimized_rag.save("optimized_rag.json")      # plain JSON, readable
loaded = RAG(); loaded.load("optimized_rag.json")
```

### 4.5 Reported numbers

The notebook text on GitHub gives approximate figures. The exact numbers below come from a rendered view of the tutorial page, so treat them as one run, not a guarantee:

| Program | SemanticF1 on devset (300) |
|---|---|
| `ChainOfThought` (no retrieval) | ~41.9% ("around 40%" in the tutorial) |
| `RAG` (retrieval, unoptimized) | ~55.5% ("55%") |
| `RAG` + MIPROv2 `medium` | ~61.1% ("approximately 61%") |

The tutorial reports a cost of about $1.5 and 20-30 minutes for the `medium` run. The optimizer docs page quotes "53% to 61%" for the same tutorial (older run). The takeaway: retrieval gave the big jump (+13 points), and prompt optimization added roughly +6 on top.

What MIPROv2 produced in one run: a more detailed instruction ("analyze the information step by step ... explains the concepts involved, highlights key distinctions...") plus two worked few-shot examples with synthetic reasoning. It may differ on a rerun.

The tutorial warns that this is not a magic button: it can **overfit** the trainset, so always validate on held-out data.

### 4.6 How MIPROv2 works

1. **Bootstrap demos.** Run the program on random train examples, keep those where the output is correct. Creates `num_candidates` sets of `max_bootstrapped_demos` bootstrapped + `max_labeled_demos` raw examples.
2. **Propose instructions.** A `prompt_model` writes instruction candidates grounded in (a) a summary of the dataset, (b) a summary of your program code and the specific predictor, (c) the bootstrapped examples, (d) a random "tip" (e.g. "be creative", "be concise") to diversify.
3. **Bayesian optimization.** Over `num_trials` trials, pick combinations of instruction + demo set per predictor, score on a minibatch (`minibatch_size`), and periodically (`minibatch_full_eval_steps`) run the best on the full valset. Return the program that did best on the full valset.

`auto` = `"light"`, `"medium"`, or `"heavy"` sets the budget. Signature: `MIPROv2(metric, prompt_model=None, task_model=None, teacher_settings=None, max_bootstrapped_demos=4, max_labeled_demos=4, auto="light", num_candidates=None, num_threads=None, init_temperature=1.0, track_stats=True)`.

Use a stronger `prompt_model` than `task_model` if you can.

---

## 5. GEPA: reflective prompt evolution (newer, usually the one to try)

**GEPA** (Genetic-Pareto) is from *"GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning"* (Agrawal et al., arXiv:2507.19457, ICLR 2026 oral per a secondary source). It ships as `dspy.GEPA` and as a standalone `gepa` package.

### How it works
1. Keep a pool of candidate programs. Start with the unoptimized one.
2. Pick a candidate (sampled from the **Pareto frontier**: candidates that are best on at least one validation example, sampled in proportion to how many they lead).
3. Run it on a **minibatch** of training examples and collect traces plus metric **feedback text**.
4. Pick one module (round-robin) and ask a **reflection LM** to rewrite that module's instruction based on the traces and feedback.
5. If the minibatch score improves, evaluate on the validation set and add it to the pool.
6. Optionally **merge** modules from different lineages.
7. Stop at the budget; return the best candidate on validation.

The key difference from MIPROv2/COPRO: those see only a scalar score. GEPA reads **natural-language feedback** from your metric, so it learns *why* a rollout failed.

### Results from the paper (read from the arXiv full text)
Models: Qwen3 8B and GPT-4.1 Mini. Benchmarks: HotpotQA, IFBench, HoVer, PUPA, AIME-2025, LiveBench-Math.

| Qwen3 8B | HotpotQA | HoVer | PUPA | Aggregate |
|---|---|---|---|---|
| Baseline | 42.33 | 35.33 | 80.82 | 45.23 |
| GRPO (24,000 rollouts) | 43.33 | 38.67 | 86.66 | 48.91 |
| MIPROv2 | 55.33 | 47.33 | 81.55 | 47.84 |
| **GEPA** | **62.33** | **52.33** | **91.85** | **54.85** |

- GEPA beats GRPO (RL with LoRA) on 5 of 6 tasks, with up to 35x fewer rollouts. GRPO still wins on AIME-2025 with Qwen3 8B (38.00 vs 32.00).
- On GPT-4.1 Mini the aggregate is 65.22 (+12.19) for GEPA vs 58.67 (+5.64) for MIPROv2.
- Pareto sampling beat greedy "always mutate the best" and beam search in the paper's ablation.
- GEPA prompts were up to 9.2x shorter than MIPROv2's.
- Limitations noted: the merge step helps inconsistently (hurt Qwen3 8B on IFBench), and most of GEPA's rollout budget goes to validation.
- Relevance for RAG: HotpotQA and HoVer are **multi-hop retrieval** tasks, which is closest to a multi-step RAG pipeline. These gains are on benchmarks, not your data.

### Using GEPA in DSPy

The metric must accept extra arguments and can return `dspy.Prediction(score=..., feedback=...)`:

```python
def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    ...
    return dspy.Prediction(score=0.0, feedback="Don't reference the input season verbatim.")

optimizer = dspy.GEPA(
    metric=metric,
    reflection_lm=dspy.LM("openai/gpt-5.4"),   # strong model, only called a handful of times
    auto="light",                              # or "medium" / "heavy"
    num_threads=8,
)
optimized = optimizer.compile(program, trainset=train, valset=val)
optimized.save("optimized.json")
```

Constructor (from the docs): `dspy.GEPA(metric, *, auto=None, max_full_evals=None, max_metric_calls=None, reflection_minibatch_size=3, candidate_selection_strategy="pareto", reflection_lm=None, skip_perfect_score=True, instruction_proposer=None, component_selector="round_robin", use_merge=True, max_merge_invocations=5, num_threads=None, track_stats=False, ...)`.

Things to know:
- **Budget:** set exactly one of `auto`, `max_full_evals`, `max_metric_calls`. `auto` maps to 6 / 12 / 18 candidates for light / medium / heavy. For a 2-predictor, 100-example task the docs estimate about 1330 / 1740 / 2045 metric calls.
- **Two LMs:** the program's LM runs evaluations (cheap, frequent). `reflection_lm` rewrites instructions (rare, use a strong one). With a strong reflection LM, reflection can cost more than evaluation.
- **Train/val:** `trainset` feeds reflection; `valset` drives Pareto selection and the final pick. Keep as much as possible in `trainset` and make `valset` the smallest sample that is still representative, because every candidate is scored on the whole valset. If you omit `valset` it reuses `trainset` (allows overfitting). Keep a separate test set.
- **Known limitation:** predictors that share the same signature can receive each other's reflective examples, so don't rely on exact per-predictor feedback in that case.
- `track_stats=True` gives `detailed_results` (all candidates, lineage, per-example scores).
- It optimizes the instructions of **every predictor** in the program, which suits multi-stage RAG.

### Writing RAG feedback for GEPA

The docs' recipe for good feedback: reuse existing artifacts (logs, tests, eval scripts), break scores into components, expose which stage failed, ground it in checks or an LLM judge, and be clear. Their retrieval example: list which documents were correctly retrieved, which were wrong, and which were missed, not just recall/F1.

The sketch below is **my illustration** of that idea, not code from the docs, and I have not run it. It assumes the gold answer is in `gold.response` and reuses `SemanticF1` as the scorer.

```python
semantic_f1 = SemanticF1(decompositional=True)

def rag_feedback_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    score = semantic_f1(gold, pred)            # float in [0, 1]
    if score >= 0.9:
        return dspy.Prediction(score=score, feedback="Good: covers the key points concisely.")
    feedback = (
        f"Semantic F1 = {score:.2f}. "
        "The answer is missing key points or adds unsupported claims. "
        f"Gold answer for reference: {gold.response}. "
        "Answer ONLY from the provided context; if the context lacks the information, say so."
    )
    return dspy.Prediction(score=score, feedback=feedback)

optimizer = dspy.GEPA(metric=rag_feedback_metric,
                      reflection_lm=dspy.LM("openai/gpt-5.4"),
                      auto="light", num_threads=16)
optimized_rag = optimizer.compile(RAG(), trainset=trainset[:100], valset=trainset[100:])
```

Better feedback is stage-specific, e.g. "retrieval returned no relevant passage" vs. "relevant passage was present but ignored". That tells the reflection LM which module's prompt to fix.

### GEPA example from the official guide (not RAG)
A haiku-writing `ReAct` program on `gpt-5.4-nano`, with `gpt-5.4` as the reflection LM, went from 78.1% to 90.1% on the guide's metric, while the unoptimized `gpt-5.4` scored 82.4%. The point is that an optimized small model can beat an unoptimized large one.

### Recent changes (as of this note)
- DSPy 3.5.0: PythonInterpreter improvements, faster GEPA, MCP v2 compatibility.
- DSPy 3.4.1 (2026-10-12, per a secondary source): GEPA's structured `InstructionProposer` became the default, passing tool trajectories, `dspy.History` inputs, and images to the reflection LM as structured content.
- 2026 follow-up papers on GEPA (adaptive feedback allocation, tight evaluation budgets, prompt-level overfitting of optimized agents). I only saw titles in search results, not the papers, so verify before relying on them.

---

## 6. Which optimizer for RAG?

| Situation | Try |
|---|---|
| Just starting, ~10-50 examples | `BootstrapFewShot` -> `BootstrapRS` |
| One generation step, scalar metric (e.g. SemanticF1), 200+ examples | `MIPROv2` (`auto="medium"`) |
| Multi-step pipeline (query rewrite, rerank, multi-hop) and you can write feedback | `GEPA` |
| Instructions OK, want rules you can read | `SIMBA` / `InferRules` |
| Prompt-only plateaued, model is tunable, need cheap inference | `BootstrapFinetune`, or `BetterTogether` |
| Agentic RAG with tool use | `GEPA` or `AvatarOptimizer` |

My suggested workflow:
1. Build the pipeline and get a baseline with `dspy.Evaluate` on a devset.
2. Fix retrieval quality first (embeddings, chunking, k).
3. Run `MIPROv2` or `GEPA`. Save the result.
4. Evaluate on a **test set that neither the optimizer nor you have looked at**.
5. Read the optimized prompt in the saved JSON (or `inspect_history`) and check it is sane.
6. Re-run when you switch models or change the pipeline.

### Pitfalls
- **Overfitting:** optimizers can memorize the trainset. Always evaluate on held-out data.
- **Weak metric = weak result.** The optimizer maximizes whatever you measure. An LLM-judge metric (like SemanticF1) adds cost and noise; refine it against human judgments.
- **Cost:** reflection/proposal LMs plus metric calls (each SemanticF1 call is an LM call) add up. Start with `auto="light"`.
- **Metric shape:** only GEPA uses `Prediction(score, feedback)`; the others treat the metric as a scalar.
- **Rate limits:** tune `num_threads` to your provider.
- **Model temperature issues:** some reasoning models reject a temperature parameter; check your `dspy.LM` settings if calls fail (see the temperature note in [[Langsmith]]).

---

## Sources

- [DSPy home](https://dspy.ai/) (version, concepts, announcements)
- DSPy docs from the repo, read in raw markdown: [Optimizers overview](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/learn/optimization/optimizers.md), [Choosing an optimizer](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/diving-deeper/choosing-an-optimizer.md), [MIPROv2](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/api/optimizers/MIPROv2.md), [GEPA overview](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/api/optimizers/GEPA/overview.md), [GEPA in depth](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/diving-deeper/gepa-in-depth.md), [Prompt optimizing with GEPA](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/getting-started/gepa-optimization.md)
- [DSPy RAG tutorial (notebook)](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/tutorials/rag/index.ipynb)
- [GEPA paper, arXiv:2507.19457](https://arxiv.org/abs/2507.19457) (read via the arXiv HTML; the fetch tool returned only the first ~100k of ~181k characters, so the appendices were not read)
- [MIPROv2 paper, arXiv:2406.11695](https://arxiv.org/abs/2406.11695) (not re-read; described through the DSPy docs)
- [GEPA GitHub](https://github.com/gepa-ai/gepa)
- Secondary (not verified against primary sources): [Decagon on GEPA](https://decagon.ai/blog/optimizing-gepa-for-production), [FutureAGI optimizer overview](https://futureagi.com/blog/dspy-optimizers-explained/), [Morph on GEPA](https://www.morphllm.com/gepa-prompt-optimization)
