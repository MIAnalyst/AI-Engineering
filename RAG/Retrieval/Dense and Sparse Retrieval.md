# Hybrid Search Strategies

This module covers 3–4 hybrid search strategies. This note covers the first one: **combining dense and sparse retrieval**.

## Why hybrid search

A RAG app is only as good as the context it retrieves. So far retrieval has been **semantic search only**: the query hits the vector store, the most similar vectors come back as context, and the LLM generates the answer from that context.

Hybrid search improves the context by combining semantic search with exact keyword search. You get the meaning-matching of embeddings and the exact-match precision of keywords.

```
Query ──► Vector store ──► context ──► LLM ──► output
            (semantic + exact search combined)
```

## The two building blocks

### 1. Sparse retrieval (exact keyword search)

- Matches **exact words**.
- Text is converted into a **sparse matrix** (mostly zeros) using classic NLP techniques: **Bag of Words, TF-IDF, BM25**.
- TF-IDF = term frequency × inverse document frequency.
- Because the vectors are word-based, a match means the keywords actually appear in the document.
- Example: "I want to have food" → `[1 0 0 0 1]` (illustrative; one slot per vocabulary word).
- Strength: exact terms such as names, IDs and jargon.

### 2. Dense retrieval (semantic search)

- Captures **semantic meaning** using vector embeddings.
- Stored and searched in vector stores such as **FAISS** and **ChromaDB**.
- The query is embedded and compared with **cosine similarity**. The output is the most similar matching sentences.
- The vectors are a **dense matrix** (every dimension carries a value).
- Strength: matches paraphrases and synonyms that share no keywords.

| | Sparse | Dense |
|---|---|---|
| Matching | Exact keyword | Semantic meaning |
| Representation | Sparse matrix (BoW, TF-IDF, BM25) | Dense embedding vectors |
| Similarity | TF-IDF / BM25 score | Cosine similarity |
| Typical tools | scikit-learn, BM25 | FAISS, Chroma |

**Hybrid = sparse + dense.** Run both searches and merge the scores into a single ranking.

## The hybrid scoring formula

$$
\text{Score}_{hybrid} = \alpha \cdot \text{Score}_{dense} + (1 - \alpha) \cdot \text{Score}_{sparse}
$$

- **Score_dense**: cosine similarity between the query embedding and the document embedding.
- **Score_sparse**: TF-IDF (or BM25) score of the query against the document.
- **α (alpha)**: the weight given to dense vs. sparse, between 0 and 1.
  - `α = 0.5`: equal weight (used in the lecture).
  - `α = 1`: pure dense/semantic search.
  - `α = 0`: pure sparse/keyword search.

The document with the highest hybrid score is retrieved first.

## Worked example

**Documents**

| Doc | Text |
|---|---|
| D1 | "LangChain helps build LLM apps" |
| D2 | "Pinecone is used for vector search" |
| D3 | "Eiffel Tower is in Paris" |

**Query:** "build application using LLM"

- D1 shares "build" and "LLM" with the query, so it scores high on both methods.
- D2 and D3 are barely related.

**Individual scores** (values chosen in the lecture)

| Doc | Dense (cosine) | Sparse (TF-IDF) |
|---|---|---|
| D1 | 0.85 | 0.60 |
| D2 | 0.40 | 0.20 |
| D3 | 0.10 | 0.10 |

**Hybrid scores with α = 0.5**

| Doc | Calculation | Hybrid |
|---|---|---|
| D1 | 0.5 × 0.85 + 0.5 × 0.60 | **0.725** |
| D2 | 0.5 × 0.40 + 0.5 × 0.20 | 0.30 |
| D3 | 0.5 × 0.10 + 0.5 × 0.10 | 0.10 |

**D1 is retrieved first**, since it is the top document under both methods.

> [!warning] Correction to the lecture's arithmetic
> The lecture (and the handwritten page) shows D1 ≈ 0.82, D2 ≈ 0.18, D3 ≈ 0.04. Those numbers don't follow from the formula. The D2 and D3 lines also used the wrong weights and values (e.g. `0.2 × 0.5`, `0.5 × 10`, `0.3 × 0.5`). The correct results are in the table above. The ranking is the same either way. Hybrid scores also don't need to sum to 1.

## Why it works

- Dense search finds relevant text that uses different words.
- Sparse search makes sure exact keywords and terms aren't missed.
- Combining both gives better recall and relevance than either alone, so the LLM gets more accurate chunks as context.
- The highest combined score wins.

## Extra notes (not from the lecture)

- Dense scores (cosine, ~0–1) and sparse scores (TF-IDF/BM25) live on different scales. BM25 in particular is unbounded. In practice, **normalize both** (e.g. min-max) before applying the weighted sum.
- Alternatives to a fixed α: learning-to-rank (mentioned on the slide) and Reciprocal Rank Fusion (RRF), which merges ranks instead of raw scores.

## Benefits Of Combining Dense And Sparse Search

Recap of the flow: the same store is searched two ways in parallel. Exact keyword search finds literal matches and semantic search finds similar meaning. The two results are merged with the α formula, and the top-scoring documents go to the LLM. This section covers **why** that combination is worth it.

> [!note] Transcript slip
> The spoken lecture says semantic search happens "with the help of BM25 / TF-IDF". That's reversed. BM25 and TF-IDF are the **sparse** (keyword) side. Semantic search is the **dense** side.

### Benefits

| Benefit | What it means |
|---|---|
| **Boosts recall** | BM25 catches exact keyword matches that dense retrieval misses. Dense catches meaning even when the keywords differ. You get both. |
| **Handles synonyms and rephrasing** | "create app" vs "build system" share no words, but dense search maps them as similar. BM25 still catches exact terms like "LLM app". |
| **Improves robustness** | Covers users who search by a specific term ("LangChain agents") and users who describe it ("how do I use LangChain to talk to tools"). Either way the relevant context comes back. |
| **Supports lexical importance** | BM25 scores **rare keywords higher**. This matters in technical, legal and medical text, where a rare term like "osteoporosis" should weigh heavily. |
| **Bridges document diversity** | Large corpora (web pages, PDFs, blogs) mix well-structured and loosely written text. Hybrid adapts to both. |
| **Tunable via α** | Shift focus with the weights. `α = 0.7` dense / `0.3` sparse leans semantic. A lower α leans toward keyword search. |
| **Tolerates misspellings and variants** | Dense models handle typos better, which is where BM25 can fail. BM25 covers the exact-term cases where dense is weaker. |

### When to use BM25 + semantic search together

The lecture lists these as the pipelines and domains where hybrid search fits, starting with RAG pipelines.

| Use case | Why hybrid helps |
|---|---|
| **RAG pipelines** | The slide says it "prevents retrieval hallucination by ensuring both exact and fuzzy matches". Exact means keyword search and fuzzy means semantic search. In practice it reduces missed or irrelevant chunks, so the LLM is less likely to answer from bad context. |
| **Technical documentation search** | A developer searches "how to use API" while the doc says "API usage". Semantic search bridges the wording and BM25 anchors on the exact term, so the hit rate improves. |
| **Legal / medical QA** | Questions are phrased in similar but varied ways, and precise terms matter. |
| **E-commerce / product search** | "cheap noise canceling headphones" can match "affordable ANC earbuds". Dense links cheap ↔ affordable and headphones ↔ earbuds, and BM25 confirms the exact term "ANC". |
| **Multilingual / cross-lingual retrieval**, **customer support transcripts** | Listed on the slide but not discussed. Read it yourself. |

### Takeaway

Run the sparse and dense searches separately, then combine them using α to set how much each one counts. You get the exact-match precision of keywords and the meaning-matching of embeddings in one ranking.

