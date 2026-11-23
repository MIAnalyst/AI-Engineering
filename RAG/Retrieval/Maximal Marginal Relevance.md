# MMR (Maximal Marginal Relevance)

Part of the **Hybrid Search Strategies** module. This note covers strategy #3: **MMR**, a way to pick retrieved documents that are relevant to the query *and* not repeats of each other.

## What is MMR?

MMR is a **diversity-aware retrieval technique** used in information retrieval and RAG pipelines. When selecting documents, it balances **relevance** (does this match the query?) against **novelty** (does this add something new?).

MMR selects documents that are:

1. **Relevant to the query**
2. **Diverse from each other** (non-redundant)

It stops the retriever from returning several near-identical documents that all repeat the same content.

## Why it matters

Plain similarity search returns the top-k closest vectors. If the vector store has near-duplicates, they all come back together.

Example: the query is **"LangChain GenAI"** and the vector store contains:

| # | Document | Note |
|---|---|---|
| 1 | LangChain is good for GenAI | Top match |
| 2 | LangChain for Agentic AI | Related, but a different angle |
| 3 | LangChain for GenAI and applications | Nearly the same as #1 |

- **Similarity search** returns #1 and #3. They are almost the same, so the second result adds little.
- **MMR** returns #1 and #2. Both are still about LangChain, but #2 brings new information.

The LLM gets more varied, more useful context from the same number of documents.

## The formula

$$
\text{MMR}(d) = \lambda \cdot \text{sim}(d, q) \;-\; (1 - \lambda) \cdot \max_{s \in S} \text{sim}(d, s)
$$

| Symbol | Meaning |
|---|---|
| `q` | The query |
| `d` | A candidate document (from the candidate set **D**, not yet selected) |
| `S` | The set of documents already selected |
| `sim(a, b)` | Similarity function, e.g. **cosine similarity** |
| `λ` (lambda) | Tunable weight in **[0, 1]** |

The two terms:

- **`λ · sim(d, q)`** is the **relevance** term: how close the candidate is to the query. It is rewarded.
- **`(1 − λ) · max sim(d, s)`** is the **redundancy** term: how close the candidate is to the *most similar* document already selected. It is subtracted, so redundancy is penalized.

**Tuning λ:**

| λ | Effect |
|---|---|
| High (→ 1) | Favors **relevance to the query**. At λ = 1, MMR is plain similarity search. |
| Low (→ 0) | Favors **diversity**. |
| 0.7 (lecture) | 70% weight on relevance, 30% on diversity. |

## The algorithm

MMR is **greedy and iterative**:

1. **Pick the first document**: the one with the highest `sim(doc, query)`. Nothing is selected yet, so there is nothing to be redundant with.
2. **Pick each next document**: compute the MMR score for every remaining candidate against the documents already selected. Take the highest score.
3. **Repeat** step 2 until you have `k` documents.

## Worked example

**Query:** "How to use LangChain for RAG"

**Documents** (already embedded in the vector store):

| Doc | Content |
|---|---|
| D1 | LangChain enables retrieval with FAISS |
| D2 | LangChain can use Chroma and Pinecone too |
| D3 | LangChain agents can call external tools |

### Step 1: pick the most relevant document

Cosine similarity between each document and the query, `sim(doc, q)`:

| Doc | sim(doc, q) |
|---|---|
| **D1** | **0.95** ✅ |
| D2 | 0.93 |
| D3 | 0.80 |

**D1 is selected** because it has the highest relevance. Now `S = {D1}`.

### Step 2: pick the second document with MMR

Compare the remaining candidates, D2 and D3, on:

1. Relevance to the query
2. Non-redundancy with the selected document (D1)

Document-to-document similarities:

| Pair | Similarity | Interpretation |
|---|---|---|
| sim(D1, D2) | 0.90 | **Redundant**: both are about LangChain + vector stores |
| sim(D1, D3) | 0.30 | **Diverse** |
| sim(D2, D3) | 0.40 | Not used here (D2 isn't selected) |

Only similarities against **selected** documents matter. With `S = {D1}`, the max term is simply the similarity to D1.

With **λ = 0.7**:

$$
\text{MMR}(D2) = 0.7 \times 0.93 - 0.3 \times 0.90 = 0.651 - 0.270 = \mathbf{0.381}
$$

$$
\text{MMR}(D3) = 0.7 \times 0.80 - 0.3 \times 0.30 = 0.560 - 0.090 = \mathbf{0.470}
$$

**D3 wins (0.47 > 0.381).** D2 is more relevant to the query (0.93 vs 0.80), but it is nearly a copy of D1, so the redundancy penalty drops its score.

### Result

| Rank | Document | Why |
|---|---|---|
| 1 | D1 | Highest relevance, `sim(d, q)` |
| 2 | D3 | Best mix of relevance and diversity (highest MMR score) |

Plain similarity search would have returned **D1, D2**. MMR returns **D1, D3**.

> **Note on the board:** the handwritten slides write `+ 0.3 × 0.90` and `+ 0.3 × 0.30`, but the arithmetic subtracts (0.651 − 0.27, 0.56 − 0.09). It is a **minus**, as in the formula.

## When to use MMR (interview points)

"When should you use MMR?" is a common interview question. There are four main use cases.

### 1. In a RAG pipeline: avoid feeding the LLM redundant documents

MMR is most useful in RAG. Every retrieved chunk takes up context space, and duplicates waste it.

Example: searching for **"LangChain"** with a FAISS store and an MMR retriever. Instead of five versions of the same paragraph, you get one chunk on each sub-topic (assuming the data contains them):

| Chunk | Topic |
|---|---|
| 1 | FAISS |
| 2 | Agents |
| 3 | Memory |
| 4 | Prompt chains |
| 5 | Hybrid retrieval |

**Result:** richer, more useful context for the LLM, because each chunk is both **relevant to the query** and **different from the others**.

### 2. Chatbots and search apps

- **FAQ chatbots** (for example, a company FAQ bot)
- **Search apps**
- **Document browsers**

Users see varied results instead of the same answer repeated.

### 3. When the retriever already returns many results

If the retriever already returns many chunks (say 20), MMR adds **diversity** to that set. It removes the near-duplicates and keeps a varied subset.

### 4. Combined with hybrid retrieval (dense + sparse)

Hybrid retrieval = **dense retriever** (embeddings) + **sparse retriever** (BM25/TF-IDF). It returns semantically close results, which are often redundant. MMR on top of it keeps them relevant and makes them diverse.

```
Query → Hybrid retrieval (dense + sparse) → many close, often redundant results
      → MMR → relevant AND diverse chunks → LLM
```

## When NOT to use MMR

MMR is an **extra processing step**, so skip it when it adds nothing:

| Scenario | Why you might skip MMR |
|---|---|
| **Extremely short context window** | Only a few chunks fit, so you want the top-1 (or top few) most relevant, not variety. |
| **You need precision only** | You care about the single most accurate match, not coverage of sub-topics. |
| **Documents are already diverse** | MMR enforces diversity. If the corpus is already diverse, there's nothing to enforce. |
| **You're already reranking with an LLM** | Redundancy is handled by post-filtering during reranking, so MMR repeats work. |

## MMR vs reranking (interview scenario)

**Q:** You're building an FAQ chatbot. Should you apply reranking or MMR?

**A:** **Both.** They solve different problems:

| | Reranking | MMR |
|---|---|---|
| Goal | **Precision**: put the most relevant chunks first | **Diversity**: remove near-duplicates |
| Scores each chunk against | The query only | The query **and** the chunks already picked |
| Fixes | Weak or misordered retrieval | Redundant, repetitive context |

A typical order is: retrieve (hybrid) → rerank → MMR → LLM.

> **Nuance:** the "skip MMR if you rerank with an LLM" row only holds if the reranker actually drops duplicates (for example, the LLM is told to discard repeats). Cross-encoder rerankers and Azure's semantic ranker score each chunk on its own, so duplicates survive and MMR is still useful. See [Reranking](Reranking.md).

## MMR in Azure AI Search

> **Azure AI Search has no built-in MMR.** Its query API has no MMR or diversity option. Hybrid search (BM25 + vector, merged with RRF) and the semantic ranker both score each document against the query independently, so near-duplicate chunks still come back together.

To use MMR with Azure AI Search, run it **client-side** after the search:

```
Query → Azure AI Search (hybrid + optional semantic ranker) → top N candidates
      → MMR in your app → k diverse chunks → LLM
```

- **LangChain:** the `AzureSearch` vector store implements MMR client-side (`search_type="mmr"`).
- **Custom code:** return the vector field in `select` (the field must be *retrievable*), then run the MMR loop over the candidates.
- **Don't plug Azure's scores into the formula.** RRF scores and semantic ranker scores (0–4) are not cosine similarities. Compute `sim(d, q)` from the vectors, or normalize the scores first.
- **Cheaper alternative:** if the duplicates are mostly neighbouring chunks of the same document, capping results per `parent_id` may be enough.

See [Hybrid Search](Hybrid%20Search.md#6-hybrid-search-in-azure-ai-search-my-chatbot).

## Quick recap

- MMR = **relevance − redundancy**, weighted by λ.
- The first document is chosen by pure relevance. Every later one is chosen by MMR score against the already-selected set.
- High λ means more relevance; low λ means more diversity.
- It trades a little relevance for a lot less repetition, so the LLM gets broader context.
- In LangChain: `vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.5})`. `fetch_k` is how many candidates are fetched by similarity first; `k` is how many MMR keeps.

Related: [Hybrid Search](Hybrid%20Search.md) · [Reranking](Reranking.md)
