# Hybrid Search

**Hybrid search = sparse (keyword) search + dense (semantic) search, merged into one ranking.** You get the exact-match precision of keywords and the meaning-matching of embeddings, so the LLM receives better context.

## 1. The big picture

A RAG app is only as good as the context it retrieves. Plain RAG uses **semantic search only**: the query is embedded, the most similar vectors come back, and the LLM answers from them. Hybrid search adds exact keyword search next to it.

```
                         ┌──► Sparse search (BM25 / TF-IDF) ──► ranked list A ──┐
Query ──► same corpus ───┤                                                       ├──► MERGE ──► top-k ──► LLM
                         └──► Dense search (embeddings)     ──► ranked list B ──┘
```

Read the note in this order:

1. The two building blocks (sparse and dense)
2. How their results are merged (weighted sum, or RRF)
3. Why it works and when to use it
4. How it looks in Azure AI Search (my chatbot)
5. What comes after retrieval (reranking, MMR)

## 2. The two building blocks

### Sparse retrieval (exact keyword search)

- Matches **exact words**.
- Text becomes a **sparse matrix** (mostly zeros) using classic NLP: **Bag of Words, TF-IDF, BM25**.
- TF-IDF = term frequency × inverse document frequency.
- A match means the keywords really appear in the document.
- Example: "I want to have food" → `[1 0 0 0 1]` (illustrative; one slot per vocabulary word).
- **Strength:** exact terms such as names, IDs, jargon and rare words.
- **Weakness:** misses paraphrases and synonyms, and is fragile with typos.

### Dense retrieval (semantic search)

- Captures **meaning** using vector embeddings, stored and searched in a vector store such as **FAISS** or **ChromaDB**.
- The query is embedded and compared with each chunk using **cosine similarity**. The output is the most similar chunks.
- The vectors are a **dense matrix** (every dimension carries a value).
- **Strength:** matches paraphrases and synonyms that share no keywords, and tolerates typos.
- **Weakness:** can miss an exact rare term (an ID, an acronym like "ANC").

### Side by side

| | Sparse | Dense |
|---|---|---|
| Matching | Exact keyword | Semantic meaning |
| Representation | Sparse matrix (BoW, TF-IDF, BM25) | Dense embedding vectors |
| Similarity | TF-IDF / BM25 score | Cosine similarity |
| Typical tools | scikit-learn, BM25 | FAISS, Chroma |
| Good at | Exact and rare terms | Synonyms, rephrasing, typos |
| Bad at | Synonyms, rephrasing | Exact identifiers |

Each one covers the other's blind spot, which is the whole reason to combine them.

## 3. Merging the two result lists

There are two common ways to merge. Method A is what the lecture teaches. Method B is what Azure AI Search does.

### Method A: weighted sum of scores (α)

$$
\text{Score}_{hybrid} = \alpha \cdot \text{Score}_{dense} + (1 - \alpha) \cdot \text{Score}_{sparse}
$$

- **Score_dense**: cosine similarity between the query embedding and the chunk embedding.
- **Score_sparse**: TF-IDF (or BM25) score of the query against the chunk.
- **α (alpha)**: how much weight dense gets versus sparse, from 0 to 1.
  - `α = 0.5`: equal weight (used in the lecture)
  - `α = 1`: pure dense/semantic search
  - `α = 0`: pure sparse/keyword search
  - `α = 0.7`: leans semantic. A lower α leans toward keywords.

Both scores compare **the query against each chunk** (not chunks against each other). A chunk ranks high only if it is both semantically close **and** keyword-relevant. The highest hybrid score is retrieved first.

#### Worked example

| Doc | Text |
|---|---|
| D1 | "LangChain helps build LLM apps" |
| D2 | "Pinecone is used for vector search" |
| D3 | "Eiffel Tower is in Paris" |

**Query:** "build application using LLM". D1 shares "build" and "LLM" with the query. D2 and D3 are barely related.

| Doc | Dense (cosine) | Sparse (TF-IDF) | Calculation (α = 0.5) | Hybrid |
|---|---|---|---|---|
| D1 | 0.85 | 0.60 | 0.5 × 0.85 + 0.5 × 0.60 | **0.725** |
| D2 | 0.40 | 0.20 | 0.5 × 0.40 + 0.5 × 0.20 | 0.30 |
| D3 | 0.10 | 0.10 | 0.5 × 0.10 + 0.5 × 0.10 | 0.10 |

**D1 is retrieved first**, since it is the top document under both methods.

> [!warning] Correction to the lecture's arithmetic
> The lecture (and the handwritten page) shows D1 ≈ 0.82, D2 ≈ 0.18, D3 ≈ 0.04. Those numbers don't follow from the formula. The D2 and D3 lines also used the wrong weights and values (e.g. `0.2 × 0.5`, `0.5 × 10`, `0.3 × 0.5`). The correct results are in the table above. The ranking is the same either way. Hybrid scores also don't need to sum to 1.

> [!tip] Normalize first
> Dense scores (cosine, ~0–1) and sparse scores (TF-IDF/BM25) live on different scales, and BM25 is unbounded. In practice, **normalize both** (e.g. min-max) before applying the weighted sum. Otherwise the larger scale dominates no matter what α is.

### Method B: Reciprocal Rank Fusion (RRF)

RRF ignores the raw scores and uses only each chunk's **rank** in each list:

$$
\text{RRF}(d) = \sum_{\text{lists}} \frac{1}{k + \text{rank}(d)}
$$

- `k` is a smoothing constant (commonly 60).
- A chunk that ranks high in **both** lists gets the biggest total. A chunk found by only one list still gets a partial score.
- Because it never compares raw scores, there is **no α to tune and no normalization needed**. That's why it's more robust than a fixed weight.

Same example, `k = 60`: D1 is rank 1 in both lists, so 1/61 + 1/61 = 0.0328. D2 is rank 2 in both: 0.0323. D3 is rank 3 in both: 0.0317. The numbers are tiny and mean nothing on their own. Only the **order** matters.

Another alternative to a fixed α is **learning-to-rank**, mentioned on the lecture slide. It trains a model to combine the signals.

## 4. Why it works

| Benefit | What it means |
|---|---|
| **Boosts recall** | BM25 catches exact keyword matches that dense misses. Dense catches meaning when keywords differ. You get both. |
| **Handles synonyms and rephrasing** | "create app" vs "build system" share no words, but dense maps them as similar. BM25 still catches exact terms like "LLM app". |
| **Improves robustness** | Covers users who search by a specific term ("LangChain agents") and users who describe it ("how do I use LangChain to talk to tools"). |
| **Supports lexical importance** | BM25 scores **rare keywords higher**. This matters in technical, legal and medical text, where a rare term like "osteoporosis" should weigh heavily. |
| **Bridges document diversity** | Large corpora (web pages, PDFs, blogs) mix well-structured and loosely written text. Hybrid adapts to both. |
| **Tunable** | Shift focus with α (Method A). |
| **Tolerates misspellings** | Dense models handle typos, where BM25 can fail. BM25 covers exact-term cases where dense is weaker. |

> [!note] Transcript slip
> The spoken lecture says semantic search happens "with the help of BM25 / TF-IDF". That's reversed. BM25 and TF-IDF are the **sparse** (keyword) side. Semantic search is the **dense** side.

## 5. When to use it

| Use case | Why hybrid helps |
|---|---|
| **RAG pipelines** | The slide says it "prevents retrieval hallucination by ensuring both exact and fuzzy matches". Exact means keyword search and fuzzy means semantic search. In practice it reduces missed or irrelevant chunks, so the LLM is less likely to answer from bad context. |
| **Technical documentation search** | A developer searches "how to use API" while the doc says "API usage". Semantic search bridges the wording and BM25 anchors on the exact term. |
| **Legal / medical QA** | Questions are phrased in similar but varied ways, and precise terms matter. |
| **E-commerce / product search** | "cheap noise canceling headphones" can match "affordable ANC earbuds". Dense links cheap ↔ affordable and headphones ↔ earbuds, and BM25 confirms the exact term "ANC". |
| **Multilingual / cross-lingual retrieval**, **customer support transcripts** | Listed on the slide but not discussed in the lecture. |

## 6. Hybrid search in Azure AI Search

### What it is

Azure AI Search runs hybrid search **inside one index and one request**. You don't build two retrievers and merge them yourself. A single `search()` call carries both a keyword query and a vector query, and the service fuses the results with **RRF** (Method B above).

- **Keyword side:** full-text search over the searchable text fields, scored with **BM25**.
- **Vector side:** approximate nearest-neighbour search (**HNSW**) over a vector field, scored with cosine similarity.
- **Fusion:** Reciprocal Rank Fusion with k = 60. Raw BM25 and cosine scores are never compared, so there is no α to tune and no normalization step.
- **Optional third stage:** the **Semantic Ranker (L2)** re-scores the fused top results (see [[#7. After hybrid search]]).

### What the index needs

A hybrid-ready index has both kinds of field on the same document:

| Field | Type | Purpose |
|---|---|---|
| `id` | `Edm.String` (key) | Unique chunk id |
| `content` | `Edm.String`, searchable | Chunk text, used by BM25 |
| `contentVector` | `Collection(Edm.Single)`, with `dimensions` and a vector profile | Chunk embedding, used by vector search |
| `title`, `source`, ... | filterable / retrievable | Metadata for filters and citations |

The vector field must be tied to a **vector search profile** (HNSW algorithm and, optionally, a vectorizer). The embedding dimension must match the model, e.g. 1536 for `text-embedding-3-small` and `text-embedding-ada-002`, or 3072 for `text-embedding-3-large`.

### The flow

**A. Indexing** (once, offline)

```
Documents ──► Chunk ──► Embedding model ──► Upload each chunk to the index
                        (text → vector)     { content, contentVector, metadata }
                                                          │
                                                          ▼
                                       ┌───────────────────────────────────┐
                                       │ Azure AI Search index             │
                                       │  • inverted index  (for BM25)     │
                                       │  • HNSW graph      (for vectors)  │
                                       └───────────────────────────────────┘
```

**B. Query time** (every question)

```
                    User query
                        │
          ┌─────────────┴─────────────┐
          │                           │
    raw query text            embed with the SAME model
          │                           │
          ▼                           ▼
   BM25 keyword search       Vector search (HNSW)
    (sparse side)              (dense side)
          │                           │
          ▼                           ▼
    ranked list A               ranked list B
          │                           │
          └─────────────┬─────────────┘
                        ▼
                   RRF fusion
            (merge by rank, k = 60)
                        │
                        ▼
          Semantic Ranker L2  (optional)
                        │
                        ▼
               top-k chunks ──► LLM
```

Both searches run **in parallel against the same index**, from **one** API call. Only the fused list comes back to you.

Steps:

1. **Create the index** with a text field and a vector field (plus the vector search profile).
2. **Chunk** the documents.
3. **Embed** each chunk with an embedding model (e.g. Azure OpenAI) and **upload** the chunk, its vector and its metadata.
4. At query time, **embed the user's question** with the *same* embedding model. (Alternatively, configure a *vectorizer* on the index and let the service embed the text for you.)
5. Send **one request** with both the raw query text (`search_text`) and the query vector (`vector_queries`).
6. Azure runs the BM25 search and the vector search **in parallel**, then merges the two ranked lists with RRF.
7. Optionally turn on the **semantic ranker** to re-score the fused results.
8. Pass the top-k chunks to the LLM as context.

> [!warning] Same model on both sides
> The query must be embedded with the **same model and dimension** used at indexing time. Otherwise the vectors live in different spaces and the vector side returns noise.

### Python SDK example

Package: `pip install azure-search-documents openai`

```python
import os
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

# 1. Clients ---------------------------------------------------------------
search_client = SearchClient(
    endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],        # https://<name>.search.windows.net
    index_name="my-chatbot-index",
    credential=AzureKeyCredential(os.environ["AZURE_SEARCH_KEY"]),
)

openai_client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_KEY"],
    api_version="2024-06-01",
)

def embed(text: str) -> list[float]:
    # "model" is your Azure *deployment name* of the embedding model
    resp = openai_client.embeddings.create(
        model="text-embedding-3-small", input=text
    )
    return resp.data[0].embedding

# 2. Hybrid query ----------------------------------------------------------
query = "how do I reset my password?"

vector_query = VectorizedQuery(
    vector=embed(query),             # dense side: the query embedding
    k_nearest_neighbors=50,          # how many neighbours the vector search returns
    fields="contentVector",          # the vector field in the index
)

results = search_client.search(
    search_text=query,               # sparse side: BM25 over the text fields
    vector_queries=[vector_query],   # dense side: vector search
    select=["id", "title", "content"],
    top=5,                           # final number of fused results
)

# 3. Read the results ------------------------------------------------------
for r in results:
    print(f"{r['@search.score']:.4f}  {r['title']}")
    print(r["content"][:200], "\n")
```

Passing **both** `search_text` and `vector_queries` is what makes it hybrid:

| Call | Mode |
|---|---|
| `search_text="..."` only | Keyword (BM25) |
| `vector_queries=[...]` only | Pure vector |
| **both** | **Hybrid (RRF)** |

Notes on the output:

- `@search.score` is the **RRF score**. With two lists and k = 60 the maximum is about 2/61 ≈ 0.033, so the values look tiny. Only the **order** matters (same point as the RRF example in section 3).
- `k_nearest_neighbors` is how many candidates the vector side contributes to the fusion, while `top` is how many fused results you get back. Keep `k_nearest_neighbors` at or above `top`.

### Parameter reference

Checked against the `azure-search-documents` **11.6.0** (stable) source. ★ marks the parameters used in the example above.

#### `VectorizedQuery` (dense side, you supply the vector)

```python
VectorizedQuery(vector=..., k_nearest_neighbors=50, fields="contentVector",
                exhaustive=False, oversampling=None, weight=1.0)
```

| Parameter | Type | Meaning |
|---|---|---|
| ★ `vector` | `list[float]` | The query embedding. Its length must equal the `dimensions` of the vector field. |
| ★ `k_nearest_neighbors` | `int` | How many nearest chunks the vector search returns into the fusion. |
| ★ `fields` | `str` | The vector field(s) to search. Several fields are separated by commas, e.g. `"titleVector,contentVector"`. |
| `exhaustive` | `bool` | `True` = brute-force exact KNN over all vectors instead of approximate HNSW. Slower but exact. Default `False`. |
| `oversampling` | `float` (≥ 1) | Fetches extra candidates and re-scores them with the original full-precision vectors. Only valid when the field uses vector compression (quantization). |
| `weight` | `float` (> 0) | Relative importance of this query against the other vector queries and the text query in the **RRF** merge. Default `1.0`. A higher weight pushes this query's matches up the final ranking. |

`VectorizableTextQuery(text=..., k_nearest_neighbors=..., fields=..., exhaustive=..., oversampling=..., weight=...)` takes the **same parameters**, except `text` (a plain string the service embeds with the index's vectorizer) replaces `vector`.

#### `search_client.search(...)` (the full call)

All parameters except `search_text` are keyword-only.

**Core hybrid parameters**

| Parameter | Type | Meaning |
|---|---|---|
| ★ `search_text` | `str` | The keyword query for the BM25 side. Leave it `None` (or `"*"`) for a pure vector search. |
| ★ `vector_queries` | `list[VectorQuery]` | The vector side. One or more `VectorizedQuery` / `VectorizableTextQuery`. Together with `search_text` this makes the request hybrid. |
| ★ `top` | `int` | How many fused results to return. The service default is 50. |
| ★ `select` | `list[str]` | Fields to return. Always list them, so you don't pull the whole vector back in every result. |
| `skip` | `int` | Number of results to skip, for paging (`skip` + `top`). |
| `filter` | `str` | OData filter, e.g. `"source eq 'handbook'"`. Fields must be marked filterable. |
| `vector_filter_mode` | `str` | When `filter` is applied to the vector search: `"preFilter"` (default, filter first then search, best recall), `"postFilter"` (search then filter, can return fewer than k), or `"strictPostFilter"`. |
| `include_total_count` | `bool` | Also return the total number of matches. Read it with `results.get_count()`. |

**Keyword (BM25) side**

| Parameter | Type | Meaning |
|---|---|---|
| `search_fields` | `list[str]` | Restrict the keyword search to these text fields. |
| `search_mode` | `"any"` / `"all"` | `"any"` = a doc matches if any term matches (default). `"all"` = every term must match. |
| `query_type` | `"simple"` / `"full"` / `"semantic"` | `simple` = default parser, `full` = Lucene syntax (fuzzy, wildcard, regex), `semantic` = turn on the Semantic Ranker. |
| `scoring_profile` | `str` | Name of a scoring profile that boosts fields or recency. It affects the **text** side only. |
| `scoring_parameters` | `list[str]` | Values for a scoring profile, e.g. `["location--122.2,44.8"]`. |
| `scoring_statistics` | `"local"` / `"global"` | Compute IDF per shard (local) or across all shards (global, more consistent scores). |
| `minimum_coverage` | `float` | Percent of the index that must be available for the query to succeed. |
| `order_by` | `list[str]` | Sort by a field instead of relevance. Usually not wanted for hybrid, because it overrides the RRF order. |
| `facets` | `list[str]` | Facet expressions, for category counts. |
| `highlight_fields`, `highlight_pre_tag`, `highlight_post_tag` | `str` | Highlight matching terms in the listed fields, wrapped in the given tags. |

**Semantic Ranker (needs `query_type="semantic"`)**

| Parameter | Type | Meaning |
|---|---|---|
| `semantic_configuration_name` | `str` | The semantic configuration defined on the index (which fields to rank on). |
| `semantic_query` | `str` | A separate query text just for the semantic ranker. Defaults to `search_text`. |
| `query_answer` | `"extractive"` | Also return short extractive answers. |
| `query_answer_count` | `int` | Number of answers to return. |
| `query_answer_threshold` | `float` | Minimum confidence score for an answer. |
| `query_caption` | `"extractive"` | Return a caption (the best passage) per result. |
| `query_caption_highlight_enabled` | `bool` | Highlight the caption text. |
| `semantic_error_mode` | `"partial"` / `"fail"` | On a semantic failure, fall back to the unranked results (`partial`) or raise an error (`fail`). |
| `semantic_max_wait_in_milliseconds` | `int` | Time limit for the semantic step. |

**Misc**

| Parameter | Type | Meaning |
|---|---|---|
| `session_id` | `str` | A sticky id that routes repeat queries to the same replicas, giving more consistent scoring. |
| `debug` | `str` | Debug info on the results, e.g. `"vector"` or `"semantic"`. Meant for development. |

> [!note] Newer preview versions
> Later preview releases of the SDK add more hybrid controls that are **not** in 11.6.0: `hybrid_search=HybridSearch(max_text_recall_size=..., count_and_facet_mode=...)` on `search()`, and `threshold`, `filter_override` and `per_document_vector_limit` on the vector queries. Check the SDK version before using them.

> [!tip] The ones that matter most
> For a normal chatbot you only tune five: `k_nearest_neighbors`, `top`, `filter`, `weight` (if you want to favor one side) and `query_type="semantic"`. The rest are for special cases.

### Common additions

```python
from azure.search.documents.models import VectorizableTextQuery

results = search_client.search(
    search_text=query,
    vector_queries=[
        # Let the service embed the query (needs a vectorizer on the index)
        VectorizableTextQuery(
            text=query, k_nearest_neighbors=50, fields="contentVector"
        )
    ],
    filter="source eq 'handbook'",          # metadata filter
    query_type="semantic",                  # turn on the Semantic Ranker (L2)
    semantic_configuration_name="my-semantic-config",
    select=["id", "title", "content"],
    top=5,
)

for r in results:
    print(r["@search.score"], r["@search.reranker_score"], r["title"])
```

- **`VectorizableTextQuery`** passes plain text and the service embeds it, so you can drop the client-side `embed()` call. It requires a vectorizer configured on the index.
- **`filter`** is an OData expression applied to the search (use filterable fields).
- **`query_type="semantic"`** plus a semantic configuration adds `@search.reranker_score` (0 to 4) on top of the RRF score. It needs the Basic tier or higher.
- **Multiple vector queries** (e.g. a title vector and a content vector) can be passed in the `vector_queries` list. They are all fused with the keyword list in the same RRF step.

> [!tip] When to prefer hybrid in Azure
> Start with hybrid as the default for a RAG chatbot. Measure it against pure vector on the eval set. Add the semantic ranker only if the right chunk is in the top 10 but ranked low (see the diagnosis below).

## 7. After hybrid search

Hybrid search is **stage 1: get good candidates**. The usual order is:

```
retrieve (hybrid) ──► rerank ──► MMR (optional) ──► LLM
```

### Reranking (second stage)

Retrieve fast, then re-score slowly but accurately:

1. A fast retriever (BM25, FAISS, or hybrid) fetches the top-k documents.
2. A more accurate but slower model (a cross-encoder or an LLM) re-scores and reorders them.

Full note: [[Reranking]].

**For my chatbot:**

- **Diminishing returns with strong LLMs.** With GPT-4-class models, comfortable context windows and short answers, the LLM's implicit reranking is already good. The lift from a reranker is real but smaller than benchmarks suggest.
- **Diagnose before adding one.** Look at the existing eval set (`Evaluation/`). For each wrong answer, check whether the right chunk was in the top 10 but ranked low.
  - **Yes** → reranking will help.
  - **Not there at all** → fix retrieval first (chunking, embeddings, query rewriting). Don't add a reranker.
  - This diagnostic pays off either way and tells you whether the engineering effort is worth it.
- **Azure's built-in option: Semantic Ranker (L2).** A cross-encoder from Microsoft (similar in spirit to MS MARCO-trained models) that re-scores the top ~50 RRF-fused results using the actual query and chunk text. It adds ~100–300 ms latency and gives a big quality jump for natural-language queries. It needs the **Basic tier or higher** and isn't available on the free tier.

### MMR (diversity)

MMR (Maximal Marginal Relevance) selects documents that are both **relevant to the query** and **diverse from each other**, so the retriever doesn't return near-duplicates.

- **Use it** to avoid feeding the LLM redundant chunks and to get more varied results.
- **Skip it** when the context window is extremely short (you want top-1), when you only need precision, when the documents are already diverse, or when you already rerank with an LLM that drops duplicates.

Full note: [[Maximal Marginal Relevance]]. Azure AI Search has no built-in MMR, so run it client-side after the search.

## 8. Quick recap

- **Sparse** = exact keywords (BM25, TF-IDF). **Dense** = meaning (embeddings, cosine). Each covers the other's weakness.
- **Hybrid** = run both on the same corpus and merge the two ranked lists.
- **Merge with** a weighted sum (`α · dense + (1 − α) · sparse`, normalize first) **or RRF** (rank-based, no tuning). Azure AI Search uses RRF.
- Benefit: better recall and relevance, so the LLM gets better chunks.
- Then optionally **rerank** for precision and **MMR** for diversity.
