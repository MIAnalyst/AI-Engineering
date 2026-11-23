# Multi-hop query generation

> Written 2026-11-23. Primary sources I read: the IRCoT abstract (arXiv:2212.10509), the Self-Ask abstract (arXiv:2210.03350), and one detailed implementation write-up ([Redlinesoft](https://blog.redlinesoft.net/posts/multi-hop-retrieval/)). The rest is explanation and code written by me, which I have not run. See [Sources](#sources).

---

## 1. What is it?

**Multi-hop query generation** is a retrieval technique for questions that no single search can answer. The system retrieves in several rounds ("hops"). After each round an LLM looks at what was found and **writes the next search query** to fill in what is still missing.

Example: *"In which city was the director of the film Inception born?"*

| Hop | Query (written by the LLM) | Retrieved | New fact |
|---|---|---|---|
| 1 | "Who directed the film Inception?" | Film cast and crew passages | The director is Christopher Nolan |
| 2 | "In which city was Christopher Nolan born?" | Biography passages | London |
| answer | n/a | All gathered facts | Final answer, built from the facts above |

The second query could not be written before hop 1 finished. That dependency is what makes it multi-hop.

### Why single-shot RAG fails here
(From the Redlinesoft write-up.)
- **Insufficient context:** the linking fact may share no vocabulary with the question, so vector similarity ranks it low.
- **Lost in the middle:** even if you raise top-k to catch more, models use evidence in the middle of a long context less reliably.
- **No chance to correct course:** retrieval is committed before the model reasons, so a bad query or a distractor goes unnoticed.

### The research behind it
- **Compositionality gap** (Press et al., [Self-Ask](https://arxiv.org/abs/2210.03350)): an LLM can often answer every sub-question correctly and still fail the combined question. This gap did not shrink as GPT-3-family models got bigger. Self-Ask makes the model write and answer its own follow-up questions, and plugging a search engine in to answer those follow-ups raised accuracy further.
- **IRCoT** (Trivedi et al., [arXiv:2212.10509](https://arxiv.org/abs/2212.10509)): interleave retrieval with the steps of a chain of thought. The reasoning so far guides what to retrieve, and what is retrieved informs the next reasoning step. Reported with GPT-3 on HotpotQA, 2WikiMultihopQA, MuSiQue and IIRC: retrieval improved by up to 21 points and downstream QA by up to 15 points. The abstract gives only these maximum gains, not per-dataset numbers.

---

## 2. Use cases

Benchmarks (from the papers above): HotpotQA, 2WikiMultihopQA, MuSiQue, IIRC (multi-hop QA); HoVer (multi-hop claim verification, used in the GEPA paper).

Typical real-world shapes (my illustrations, not from the sources):
- **Bridge-entity questions:** "Who manages the owner of service X?" You find the owner first, then their manager.
- **Cross-document support and troubleshooting:** error code -> known issue -> affected versions -> fix.
- **Legal and contracts:** a clause refers to a definition or an amendment in another document.
- **Research and literature questions:** a paper's claim, the paper it builds on, then that paper's results.
- **Claim verification:** find evidence for each part of a claim, where later evidence depends on earlier findings.
- **Enterprise knowledge bases:** facts spread across wikis, tickets and docs that never mention each other directly.

**Skip it when** (from Redlinesoft): the corpus fits in the context window (roughly under 100k tokens), the questions are single-hop, or latency and cost are tight. Each hop adds a retrieval and an LLM call.

---

## 3. Multi-hop query generation vs. query decomposition

Both break a complex question into retrieval steps. The difference is **when the sub-queries are written** and **whether they depend on each other**.

| | Query decomposition | Multi-hop query generation |
|---|---|---|
| When queries are written | Up front, all at once, from the question alone | One at a time, after seeing what the previous hop retrieved |
| Dependency between queries | Usually independent | Each depends on previous results |
| Execution | Can run in parallel | Sequential |
| Needs retrieved context to plan | No | Yes |
| Typical question | Comparisons, multi-part questions | Chained / bridge questions |
| Cost shape | N parallel retrievals; latency of one hop | Latency grows with each hop |

**Decomposition example:** "Compare the climate of Oslo and Lisbon." Split into "climate of Oslo" and "climate of Lisbon", search both in parallel, combine.

**Multi-hop example:** the Inception question above. Decomposing up front gives "Who directed Inception?" and "Where was [director] born?", and the second contains a placeholder only retrieval can fill.

**They overlap.**
- A decomposer can produce an ordered chain of dependent sub-questions, which is multi-hop in practice. The terms are not used consistently across articles.
- Redlinesoft describes two query strategies: *decomposition* (ordered sub-questions: predictable but brittle if you miss a dependency) and *rewrite / step-back* (rewrite the query each hop: flexible but harder to evaluate). It notes many systems **decompose once, then rewrite within each branch**.
- Rule of thumb: if the sub-questions are independent, use decomposition (cheaper and parallel). If one answer feeds the next query, you need hops.

---

## 4. How developers usually implement it

There is no single standard module in the frameworks I checked. In practice it is a **loop**, and the patterns below are the common variants.

### Pattern A: fixed number of hops (the simplest)
Loop N times (usually 2 or 3). Each iteration: an LLM writes a query from the question plus the context so far, retrieve, append. Then one final LLM call answers. No decision about when to stop. This is the shape of the standard DSPy multi-hop example.

```python
# Sketch (my own, untested). `llm` and `retriever` are placeholders.
def multi_hop(question, hops=2, k=3):
    context = []
    for _ in range(hops):
        query = llm(f"""Question: {question}
Context so far: {context}
Write one search query for the information still missing.""")
        context += retriever.search(query, k=k)
    return llm(f"Answer from the context only.\nContext: {context}\nQuestion: {question}")
```

Pros: predictable cost and latency, easy to evaluate. Cons: wastes calls on easy questions and stops too early on hard ones.

### Pattern B: loop with a sufficiency check (most common in LangGraph tutorials)
Add a decision step: after each retrieval the LLM says whether the evidence is enough. If yes, answer. If not, rewrite the query and hop again, up to a hard `max_hops`.

The Redlinesoft implementation uses this graph:
- **State:** question, current query, retrieved docs, accumulated facts, hop count, hop budget, answer. Facts use a reducer so each node only returns *new* facts.
- **Nodes:** `retrieve` (deterministic retrieval), `reason` (LLM extracts supported facts with source IDs and bumps the hop counter), `rewrite` (LLM writes a query for the missing evidence), `synthesize` (answer from gathered facts only, with citations).
- **Edge:** after `reason`, a conditional edge goes to `rewrite` (continue) or `synthesize` (answer).

```python
# Wiring shown in the Redlinesoft write-up
b.add_conditional_edges("reason", should_continue, {
    "continue": "rewrite",
    "answer": "synthesize",
})
```

Stop conditions: a **hard `max_hops`** (the write-up suggests 2 or 3 for most corpora) plus a **structured yes/no "is evidence sufficient?" judgment** from the LLM, not a substring test.

### Pattern C: reasoning-interleaved (IRCoT / Self-Ask style)
Instead of a separate "write a query" prompt, the model reasons step by step. After each reasoning step the new sentence or follow-up question is used as the search query, the results are added, and reasoning continues.
- **IRCoT:** the CoT sentences guide retrieval.
- **Self-Ask:** the model writes explicit "Follow up: ..." questions that go to a search tool.

Closer to the research papers than to most production code.

### Pattern D: agentic tool-calling loop
Give an LLM agent a `search` tool and let it decide how many times to call it and with what query. No hand-built graph for the hops. This is the "Agentic RAG" style (see [[Agentic Rag]]). It is flexible but has the least predictable cost, and you must still cap the number of tool calls.

LangChain's own LangGraph tutorial ([Build a custom RAG agent with LangGraph](https://docs.langchain.com/oss/python/langgraph/agentic-rag)) is in this family but is aimed at *corrective* RAG: a grader checks the retrieved documents and, if they are irrelevant, rewrites the question and retrieves again. It retries a failed retrieval, which is not the same as chaining facts, but the loop shape is similar.

### Pattern E: decompose first, then rewrite per branch
A planner LLM decomposes the question into sub-questions, then each sub-question runs through a small retrieve -> check -> rewrite loop, and results are merged. This is the hybrid Redlinesoft mentions, and [IBM's tutorial](https://developer.ibm.com/tutorials/awb-build-rag-llm-agents/) combines decomposition with a corrective RAG graph, using the intermediate answers as "multi hop context" (I read only the search summary of that one).

### Pattern F: DSPy
Write the loop as a `dspy.Module` with one `ChainOfThought('context, question -> search_query')` predictor per hop. Each query-writing step is then a prompt an optimizer (GEPA, MIPROv2) can tune. See [[DSPy]].

```python
class MultiHopRAG(dspy.Module):
    def __init__(self, hops=2, k=3):
        self.hops, self.k = hops, k
        self.generate_query = [dspy.ChainOfThought('context, question -> search_query')
                               for _ in range(hops)]
        self.respond = dspy.ChainOfThought('context, question -> response')

    def forward(self, question):
        context = []
        for hop in range(self.hops):
            q = self.generate_query[hop](context=context, question=question).search_query
            context += search(q, k=self.k).passages   # `search` = your retriever
        return self.respond(context=context, question=question)
```

(My sketch, untested.)

### Which one to pick
| Situation | Start with |
|---|---|
| Questions are mostly 2-hop and you want predictable cost | A (fixed hops) |
| Mixed difficulty, need to stop early on easy ones | B (sufficiency check + `max_hops`) |
| You already use LangGraph and want checkpointing / human review | B in LangGraph |
| Open-ended research tasks | D (agentic) with a tool-call cap |
| You have an eval set and want to tune the query prompts | F (DSPy) |

---

## 5. Practical tips and pitfalls

- **More hops is not better.** Each hop adds a retrieval and an LLM call, and errors compound: a bad first query poisons every later hop. Keep `max_hops` low (2 or 3).
- **Store sources with each fact** (document ID, chunk offset). It lets you cite and flag unsupported claims before they reach users.
- **Make the "enough evidence?" check structured** (a boolean field), not free text.
- **Evaluate per hop**, not just the final answer: retrieval recall at each step, correctness of the intermediate facts, final answer correctness, and the hop distribution. If most queries hit `max_hops`, the budget or the rewrite prompt likely needs work.
- **Build an eval harness first.** Without one, a loop can degrade silently after a prompt change.
- **Retrieval quality still matters.** Better chunks, hybrid search and [[Reranking]] make every hop's retrieval more reliable.
- **Watch for query drift.** A rewritten query can wander away from the original question. Passing the original question into every query-writing step helps.

---

## Sources

- IRCoT: [Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions, arXiv:2212.10509](https://arxiv.org/abs/2212.10509) (abstract only)
- Self-Ask: [Measuring and Narrowing the Compositionality Gap in Language Models, arXiv:2210.03350](https://arxiv.org/abs/2210.03350) (abstract only)
- [Multi-Hop Retrieval: Answering Questions That Span Multiple Sources (Redlinesoft)](https://blog.redlinesoft.net/posts/multi-hop-retrieval/): third-party blog, read in full via a fetch tool that summarizes pages
- [Build a custom RAG agent with LangGraph (LangChain docs)](https://docs.langchain.com/oss/python/langgraph/agentic-rag): seen only in search results, not read in full
- [Build a RAG agent with LangGraph, Python, and watsonx.ai (IBM)](https://developer.ibm.com/tutorials/awb-build-rag-llm-agents/): seen only in search results, not read in full
- Related notes: [[DSPy]], [[Agentic Rag]], [[LangGraph]], [[Reranking]]
