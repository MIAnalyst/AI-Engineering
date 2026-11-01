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

## Next

Practical implementation of dense + sparse retrieval (next video).
