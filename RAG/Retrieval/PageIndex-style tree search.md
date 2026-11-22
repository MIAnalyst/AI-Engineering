# PageIndex-Style Tree Search

A **"vectorless," reasoning-based** retrieval approach: build a **tree index** of a document (like a rich table of contents) and let an **LLM search the tree by reasoning** over node titles and summaries. There are no embeddings, no chunk similarity and no vector database.

Named after **[PageIndex](https://github.com/VectifyAI/PageIndex)** (VectifyAI), which describes the tree as *"a table of contents optimized for LLMs and agents"*. Its core claim: **similarity ≠ relevance**, so retrieval should be done by reasoning.

Related: [Retrieval guided by a table of contents](Retrieval%20guided%20by%20a%20table%20of%20contents.md) · [Hierarchical chunking](../Preprocessing/Hierarchical%20chunking.md) · [Reranking](Reranking.md)

## The idea

Vector RAG asks: *"Which chunks **look like** the query?"*
Tree search asks: *"Given what each section is about, **where would the answer be written?**"*

This is how a human expert reads a long report: check the contents, open the likely chapter, skim the subsections, read the right pages, and go back if it was the wrong place.

## Indexing: build the tree

Each document becomes a tree of nodes that follows its natural sections:

```json
{
  "title": "Risk Factors",
  "node_id": "0006",
  "start_index": 21,
  "end_index": 34,
  "summary": "Market, currency, interest-rate and regulatory risks ...",
  "nodes": [
    { "title": "Currency Risk", "node_id": "0007", "start_index": 22, "end_index": 25, "summary": "...", "nodes": [] },
    { "title": "Interest Rate Risk", "node_id": "0008", "start_index": 26, "end_index": 29, "summary": "...", "nodes": [] }
  ]
}
```

| Field | Meaning |
|---|---|
| `node_id` | Stable ID used for navigation and citations |
| `title` | Section title |
| `start_index` / `end_index` | **Page range** of the section |
| `summary` | LLM-written summary of the section, used for reasoning during search |
| `nodes` | Child sections |

- In PageIndex, the **tree structure comes from the document layout** (not from an LLM). The indexing LLM only **writes node summaries and refines** the tree.
- Sections are kept **whole**, so there are no chunking artifacts.
- Index cost (vendor figure): about **$0.001 per page** for local indexing.

## Retrieval: tree search

```
Query: "How did the 2023 FX hedging change affect operating margin?"

[root] ── LLM reads top-level titles + summaries
  ├── 1 Business Overview          ✗
  ├── 2 Risk Factors               ✓ expand
  │     ├── 2.1 Currency Risk      ✓ expand → read pp. 22–25
  │     └── 2.2 Interest Rate Risk ✗
  ├── 3 MD&A                       ✓ expand
  │     └── 3.4 Operating Results  ✓ read pp. 48–52
  └── 4 Financial Statements       ✗

→ LLM answers from pp. 22–25 and 48–52 with page citations
```

1. The LLM sees the query and the **top level** of the tree (titles and summaries only, no full text).
2. It **reasons** about which branches could hold the answer and expands them.
3. It repeats **level by level**, and can **backtrack** if a branch turns out empty. This is like best-first or beam search with the LLM as the scoring function.
4. At the chosen nodes it fetches the **full page text** and answers, citing node IDs and pages.

In PageIndex this runs as an **agent with tools**:
- `get_document_structure()` → the tree, without the text
- `get_page_content(pages)` → the text for a page range

## Usage (PageIndex SDK)

```bash
pip install -U pageindex
```

```python
from pageindex import PageIndexClient

client = PageIndexClient(index="<index-model>", chat="<chat-model>")  # or index="cloud" + PAGEINDEX_API_KEY
doc = client.submit_document("annual_report.pdf")
answer = client.chat("How did the 2023 FX hedging change affect operating margin?")
```

- Supports OpenAI, Anthropic and OpenRouter models (LiteLLM-style model names).
- Ready-made tools for the OpenAI Agents SDK, the Anthropic SDK and the Claude Agent SDK.
- Local indexes are stored in `./.pageindex`.
- **Local mode handles text-based PDFs only.** Scanned PDFs, image understanding, bounding-box citations and the MCP server are **Cloud-only**.
- Check the repo for current model names and API details. They change often.

## Why it's compelling

- **Relevance ≠ similarity.** Multi-step questions often don't resemble the passage that answers them in embedding space. An LLM reading section summaries can still work out where the answer should be.
- **No chunking artifacts**: whole sections are read, with their full context.
- **Explainable and traceable**: you get the **path through the tree** and page citations, not just a cosine score.
- **Strong benchmark results (vendor-reported):** **98.7% on FinanceBench** (financial document QA), compared with about 50% for vector RAG in their chart. *Treat vendor benchmarks with caution.*
- **Cheaper than putting the whole PDF in context (vendor figure):** passing the full PDF cost 2.1× more at 52 pages and 16.6× more at 420 pages, and an 805-page document doesn't fit at all.

## Tradeoffs

| | Vector RAG | PageIndex-style tree search |
|---|---|---|
| **Latency / cost per query** | Low (one ANN lookup + one LLM call) | Higher (**several LLM calls** per query for navigation) |
| **Scale** | Millions of chunks over a huge corpus | Best on **one or a few long documents** |
| **Messy, unstructured text** | Fine | **Weak**: the tree needs real structure |
| **Multi-step "where would this be?" reasoning** | Weak | **Strong** |
| **Indexing cost** | Embeddings (cheap) | LLM summaries for every node (more expensive) |
| **Explainability** | Similarity scores | Navigation path + page citations |
| **Context given to the LLM** | Fragments (chunks) | Whole sections / page ranges |
| **Infrastructure** | Embedding model + vector DB | LLM + tree JSON (no vector DB) |
| **Quality depends on** | Embedding model + chunking | **Chat model's reasoning** ("use the best model you can afford") + summary quality |
| **Updating documents** | Re-embed the changed chunks | Rebuild the tree / summaries for the document |

**Other weak spots:**
- **Navigation errors**: if the LLM drops the right branch early, the answer is lost (backtracking only partly helps).
- **Answers spread over many sections** need many expansions, so cost and latency go up.
- **Many documents**: you need routing across documents first (e.g. BM25/vectors over document summaries) before tree search inside each one.

## When to use

| Use tree search | Use vector / hybrid RAG |
|---|---|
| Long, structured, high-stakes documents: 10-Ks, contracts, regulations, manuals, textbooks | Large collections of short or unstructured texts |
| Accuracy and traceability matter more than latency | Low latency and low cost per query matter |
| Questions need reasoning about *where* information lives | Questions are mostly lookups that keyword or semantic matching handles well |

**Hybrids work well in practice:** route **between** documents with vectors or BM25, use **tree search inside** each long document, then optionally [rerank](Reranking.md) the final candidates.

## How it relates to the other approaches

```
Hierarchical chunking   → keep the structure inside the chunks (still similarity search)
ToC-guided retrieval    → use the structure to narrow the search, then similarity search inside
PageIndex tree search   → the structure IS the index; an LLM reasons its way down
```
