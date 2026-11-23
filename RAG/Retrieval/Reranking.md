# Reranking

Part of **Hybrid Search Strategies**. This note covers the second strategy: **re-ranking**. The first strategy, combining dense and sparse retrieval with a hybrid score, is in [Hybrid Search](Hybrid%20Search.md).

## What reranking is

Re-ranking is a **second-stage filtering step** in a retrieval system, especially in RAG pipelines:

1. **Retrieve fast.** A fast retriever (BM25, FAISS, or a hybrid of both) fetches the **top-k** documents quickly.
2. **Re-score slowly but accurately.** A more accurate but slower model, such as a **cross-encoder** or an **LLM**, re-scores the top-k documents and **reorders them by relevance to the query**.

Result: the most relevant documents end up at the top, so the final LLM answer improves.

## Where it fits in the pipeline

**Before (plain RAG):**

```
Vector store ──► retriever ──► top-k docs ──► LLM (+ prompt) ──► output
```

**After (with reranking):** one extra step between retrieval and the LLM.

```
                    Input query
                         │
                    Vector store
              ┌──────────┴──────────┐
        BM25 (exact match)   Embeddings / FAISS (similar match)
          retriever 1             retriever 2
              │      └── Hybrid score ──┘     │
            top-k          top-k            top-k
              └──────────────┼──────────────┘
                             ▼
                         Re-ranker   ◄── input query
                             │  reordered by rank
                             ▼
                            LLM ──► more accurate output
```

The top-k input to the reranker can come from any of the three paths:

- **BM25 only** (sparse, exact match)
- **Semantic search only** (embeddings in FAISS, similar match)
- **Hybrid** (both combined into one score)

## How the reranker reorders documents

Say the retriever returns 5 chunks in order `1, 2, 3, 4, 5`. The reranker is given the query and those chunks, with a prompt along the lines of:

> Here is the context (the top-k documents). Rerank them in order of relevance to the query.

The model assigns each document a rank based on the query, and the order might change to:

| New position | Original document |
|---|---|
| 1 | 5 |
| 2 | 2 |
| 3 | 1 |
| 4 | 4 |
| 5 | 3 |

That is the **reordering**. The LLM (or cross-encoder) decides the rank, **always with the input query in mind**. The reordered chunks then go to the LLM for generation.

## The three stages

1. **Retrieval:** fast retriever returns top-k candidates.
2. **Re-ranking:** a slower, more accurate model reorders them by relevance.
3. **Generation:** the LLM answers using the reordered context.

## Why use a reranker

| Reason | Without reranker | With reranker |
|---|---|---|
| 1. Relevance of context | Top-k docs may be loosely or partially related | Top-k docs are re-scored and reordered for maximum relevance |
| 2. Factual accuracy | LLM may hallucinate from poor context | Irrelevant docs filtered out → grounded, factual answers |
| 3. Handling ambiguity | Retriever lacks deep understanding of query intent | Evaluates the full query–doc pair → better intent alignment |
| 4. Semantic matching | Dense retrievers may miss low-similarity but relevant docs | Uses deeper models (cross-encoders / LLMs) |
| 5. Keyword vs meaning | BM25 may favor exact matches that aren't meaningful | Balances semantic and lexical relevance |
| 6. Prioritization of evidence | All retrieved docs treated equally | Most relevant docs float to the top |
| 7. Long-tail queries | Weak retrievers struggle with uncommon queries | Better at rare but meaningful matches |
| 8. LLM efficiency | Poor context → verbose or incorrect answers | High-quality context → precise, concise answers |
| 9. Noise reduction | Noisy docs (ads, unrelated text) slip in | Noisy docs pushed to the bottom or filtered out |
| 10. Flexible scoring | Fixed retriever scoring | Can factor in metadata, recency, user preferences |

## Does the order matter to the LLM?

### Does the LLM receive the order?

Only implicitly, through **position**. The chunks are concatenated into the prompt as text, so the LLM sees them in that sequence. The rank numbers and scores are **not** passed along unless you add them yourself (e.g. label chunks `[Doc 1]`, `[Doc 2]`, or include a relevance score).

### Selection matters more than order

Most of reranking's value comes from deciding **which chunks make the cut**, not how they are arranged:

- Common pattern: retrieve many (e.g. 50) cheaply → rerank → keep only the **top 3–5** for the LLM.
- A relevant chunk the retriever ranked 30th can get rescued, and noisy chunks get dropped entirely.

### But position still has an effect

**"Lost in the Middle"** (Liu et al., 2023): LLMs use information at the **start and end** of a long context better than information buried in the **middle**. Newer long-context models are better at this, but the effect hasn't fully gone away, especially with long contexts.

### What the order can affect

| Impact | How order matters |
|---|---|
| Answer accuracy | A key fact in the middle of a long context may be overlooked or under-weighted |
| Conflicting information | When chunks disagree (e.g. old vs new policy), the model leans toward the more prominent one |
| Hallucination / noise | Irrelevant chunks near the top can distract the model or pull the answer off-topic |
| Citations | Source numbering follows the order, and top chunks tend to get cited more |
| Cost / latency | Comes from the cut-off rather than the order: fewer, better chunks mean fewer tokens |

### Practical rules of thumb

- Rerank, then **keep only the top few chunks** rather than sending everything.
- Put the **most relevant chunk first**. Some pipelines also put the second-best chunk **last**, to use both "strong" positions (LangChain's `LongContextReorder` does this).
- **Label chunks** (`[Source 1] ...`) if you want citations or want the model to tell the chunks apart.

## Reranking in Azure AI Search (semantic ranker)

In Azure AI Search, reranking is a **built-in step you switch on**, called the **semantic ranker**. You don't build it yourself. It maps directly onto the three stages above:

| Stage | In Azure AI Search |
|---|---|
| 1. Retrieval | Hybrid query: BM25 + vector search, fused with **RRF** |
| 2. Re-ranking | **Semantic ranker** re-scores the top 50 fused results |
| 3. Generation | Your code sends the top N chunks to the LLM |

### The flow

```
                         Query
                           │
            ┌──────────────┴──────────────┐
            ▼                             ▼
      BM25 (keyword)              Vector search (k: 50)
            └──────────────┬──────────────┘
                           ▼
 ┌─────────────────────────────────────────────────────┐
 │ STAGE 1: RETRIEVAL (RRF fusion)                     │
 │ Merges both lists by rank → @search.score           │
 │ Keeps the top 50                                    │
 └─────────────────────────────────────────────────────┘
                           │  50 chunks
                           ▼
 ┌─────────────────────────────────────────────────────┐
 │ STAGE 2: RE-RANKING (semantic ranker)               │
 │ Reads query + chunk text → NEW @search.rerankerScore│
 │ (0–4). @search.score is left unchanged.             │
 │ Re-sorts the 50 by rerankerScore                    │
 └─────────────────────────────────────────────────────┘
                           │  top 10 (top: 10)
                           ▼
 ┌─────────────────────────────────────────────────────┐
 │ STAGE 3: GENERATION                                 │
 │ Your code sends the 10 chunks, in order, to the LLM │
 └─────────────────────────────────────────────────────┘
```

### Does the semantic ranker change the RRF score?

No. It **scores again from scratch** and adds a **separate, new score**. The RRF score is left untouched.

| | `@search.score` (RRF) | `@search.rerankerScore` |
|---|---|---|
| Produced by | Hybrid fusion (stage 1) | Semantic ranker (stage 2) |
| Changed by the ranker? | No, it stays as it was | It's new; only the ranker sets it |
| Based on | Each result's rank in the BM25 and vector lists | The query and the chunk's actual text, read together |
| Decides final order? | No, once the ranker is on | Yes |

The semantic ranker doesn't use the RRF score as an input either. RRF's only job is to **pick which 50 chunks get in**. After that, the ranker judges each chunk afresh from the query and the text in the fields named in your semantic configuration (title and content). That's why a chunk with a low RRF score can end up first: the two scores are independent.

1. **Hybrid retrieval.** BM25 and vector search run in parallel. Their result lists are merged with **Reciprocal Rank Fusion (RRF)**, which gives each result a `@search.score`. This is Azure's version of the hybrid score. RRF works from each result's **rank** in the two lists rather than the weighted α formula in [Hybrid Search](Hybrid%20Search.md).
2. **Semantic ranker.** Azure takes the **top 50** fused results (or fewer, if fewer matched) and re-scores each one against the query. It uses a deep model from Microsoft that reads the query and the chunk together, like a cross-encoder. Each result gets a `@search.rerankerScore` from **0 to 4** (4 = highly relevant), and the results are re-sorted by it.
3. **Your code.** Take the top 5–10 by reranker score, optionally drop anything below a threshold (e.g. `< 2.0`), and send the rest to the LLM **in that order**.

### Example: "I want 10 chunks"

- BM25 and vector search (with `k: 50`) return candidates → RRF merges them into one ranked list.
- The semantic ranker takes the **top 50** of that list and re-scores them.
- With `top: 10`, you get the **10 chunks with the highest `rerankerScore`**, already ordered.

A chunk that RRF placed 35th can still make your final 10 if the semantic ranker judges it highly relevant. That's the "rescue" benefit of reranking.

**Is my top 10 reranked?**
- **Semantic ranker off:** you get the top 10 by RRF score. That's retrieval only, with no reranking.
- **Semantic ranker on:** you get the top 10 by reranker score, chosen from the best 50 hybrid results.

### Turning it on

Define a **semantic configuration** on the index (which fields hold the title and content), then query with it:

```json
POST /indexes/my-index/docs/search?api-version=2024-07-01
{
  "search": "how do I reset my password",
  "vectorQueries": [
    { "kind": "text", "text": "how do I reset my password", "fields": "contentVector", "k": 50 }
  ],
  "queryType": "semantic",
  "semanticConfiguration": "my-semantic-config",
  "top": 10,
  "select": "title, content"
}
```

Each result then carries both scores:

```json
{ "@search.score": 0.0331, "@search.rerankerScore": 3.12, "title": "...", "content": "..." }
```

### Gotchas

- **Set `k: 50` on the vector query.** The 50 isn't automatic for the vector side. With `k: 10`, only 10 vector results enter the fusion, so the ranker's 50 candidates are mostly BM25 results.
- **The two scores aren't comparable.**

  | Score | Scale | Use it for |
  |---|---|---|
  | `@search.score` (RRF) | Small, relative to the other results | Ordering only |
  | `@search.rerankerScore` | Fixed 0–4 | Ordering **and** thresholds |

- **Cost and pricing tier:** the semantic ranker is a separate billed feature that needs a paid tier. Check current pricing.
- **Bonus outputs:** it can also return **captions** (highlighted key passages) and **semantic answers** if you request them.
- **Using your own reranker instead:** skip the semantic ranker, ask Azure for about 50 hybrid results, and rerank them in your own code with a cross-encoder, Cohere Rerank, or an LLM. This gives you more control, but you take on the latency and hosting.

> Parameter names (`queryType`, `semanticConfiguration`, `@search.rerankerScore`) and the top-50 window were written from memory. Azure updates these APIs often, so verify them against the current docs for your API version.

## Key takeaways

- Reranking trades a little latency for much better context: a cheap retriever casts a wide net, and an expensive model sorts the catch.
- It works on top of any retriever: sparse, dense, or hybrid.
- The reranker's job is **ordering by relevance to the query**, not retrieving new documents.

