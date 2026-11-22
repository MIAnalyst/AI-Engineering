# Hierarchical Chunking

A chunking strategy that splits a document along its **real structure** (chapter → section → subsection → paragraph) and **keeps that hierarchy** as metadata or as linked chunks of different sizes, rather than cutting text into fixed-size pieces.

Related: [Retrieval guided by a table of contents](../Retrieval/Retrieval%20guided%20by%20a%20table%20of%20contents.md) · [PageIndex-style tree search](../Retrieval/PageIndex-style%20tree%20search.md) · [pdf processing](pdf%20processing.md)

## The problem it solves

Fixed-size chunking (e.g. 500 tokens with 50 overlap) ignores the document's structure:

- A chunk can start in the middle of Section 4.2 and end in 4.3, mixing two topics.
- Once embedded, the chunk **doesn't know where it came from**. "The rate increased to 4%" is meaningless without knowing it sits under *Risk Factors > Interest Rate Risk*.
- **Size dilemma:**
  - **Small chunks** → precise embedding match, but too little context for the LLM.
  - **Big chunks** → plenty of context, but the embedding is a blurry average of many ideas, so retrieval gets worse.

Hierarchical chunking addresses both: **match at a fine level, read at a coarse level**, and every chunk keeps its place in the document.

## Variants

| Variant | How it works | Tools |
|---|---|---|
| **Structure-aware splitting** | Split on headings, not token counts. Each chunk carries its **heading path**, e.g. `Annual Report > Risk Factors > Currency Risk`, which is often added to the start of the chunk text before embedding. | Docling `HierarchicalChunker` / `HybridChunker`, Unstructured `chunk_by_title`, LangChain `MarkdownHeaderTextSplitter` |
| **Parent–child (small-to-big)** | Embed small **child** chunks (sentences or paragraphs) for precise matching, but return the **parent** chunk (the whole section) to the LLM. | LangChain `ParentDocumentRetriever`, LlamaIndex `HierarchicalNodeParser` |
| **Auto-merging** | If several children of the same parent are retrieved (e.g. 3 of 4), replace them with the parent, so the LLM gets one clean section rather than scattered pieces. | LlamaIndex `AutoMergingRetriever` |
| **Multi-level summaries (RAPTOR-style)** | Build a tree **bottom-up**: cluster the leaf chunks, summarise each cluster with an LLM, then cluster and summarise again. Index **every level**, so broad questions match summaries and specific questions match leaves. | RAPTOR, LlamaIndex `RaptorPack` |

## Structure-aware splitting

```
Document
├── 1 Introduction
├── 2 Risk Factors
│   ├── 2.1 Currency Risk      ──► chunk: "Risk Factors > Currency Risk\n\n<text>"
│   └── 2.2 Interest Rate Risk ──► chunk: "Risk Factors > Interest Rate Risk\n\n<text>"
└── 3 Financial Statements
```

- Sections longer than the embedding model's token limit are split further, but each piece **inherits the heading path**.
- Very short sections can be merged with their siblings.
- The heading path also helps **BM25** (the section titles' keywords are now in the chunk) and **reranking**.
- Keep **page numbers** in metadata for citations.

Where the structure comes from:
- PDF bookmarks: `doc.get_toc()` in PyMuPDF (often more reliable than guessing from font sizes)
- Markdown headings after conversion (`#`, `##`, ...)
- Layout models (Docling labels `SECTION_HEADER`, `TITLE`, etc.)

## Parent–child (small-to-big)

```
Parent: §2.1 Currency Risk (full section, ~1500 tokens)   ◄── sent to the LLM
 ├── child 1 (~150 tokens)  ── embedded
 ├── child 2 (~150 tokens)  ── embedded   ◄── matched by the query
 └── child 3 (~150 tokens)  ── embedded
```

The same text is stored **twice**, in two places, at two sizes:

| | **Child chunks** | **Parent chunks** |
|---|---|---|
| Stored in | **Vector store** | **Docstore** (key → value store: dict, Redis, a database table, ...) |
| Embedded? | ✅ Yes, their text is turned into vectors | ❌ No, only looked up by ID |
| Size | Small (~100–200 tokens) | Big (the whole section) |
| Used for | **Finding** (similarity matching) | **Reading** (context sent to the LLM) |
| Link | `parent_id` in the child's metadata | `parent_id` is the docstore key |

### Example

**Source section** (page 22 of an annual report):

```
2.1 Currency Risk
Most of our revenue is earned in EUR and GBP, while our reporting currency is USD.
A 10% stronger USD would reduce reported revenue by about $120M.
In 2023 we began hedging 60% of forecast EUR revenue with 12-month forward contracts.
Hedging gains of $18M were recorded in operating income.
We do not hedge GBP exposure.
```

**Parent chunk → docstore** (not embedded):

```python
docstore["p_2.1"] = {
    "text": "2.1 Currency Risk\nMost of our revenue is earned in EUR and GBP ... "
            "... We do not hedge GBP exposure.",            # the WHOLE section
    "metadata": {"heading_path": "Risk Factors > Currency Risk", "pages": [22, 23]},
}
```

**Child chunks → vector store** (the `text` is embedded; the `metadata` is stored alongside the vector but **not embedded**):

```python
{   "id": "c_2.1_1",
    "text": "Risk Factors > Currency Risk\n"                          # ← heading path prefix (optional, is embedded)
            "Most of our revenue is earned in EUR and GBP, while our reporting "
            "currency is USD. A 10% stronger USD would reduce reported revenue by about $120M.",
    "metadata": {"parent_id": "p_2.1", "page": 22},                  # ← NOT embedded
}
{   "id": "c_2.1_2",
    "text": "Risk Factors > Currency Risk\n"
            "In 2023 we began hedging 60% of forecast EUR revenue with 12-month "
            "forward contracts. Hedging gains of $18M were recorded in operating income.",
    "metadata": {"parent_id": "p_2.1", "page": 22},
}
{   "id": "c_2.1_3",
    "text": "Risk Factors > Currency Risk\nWe do not hedge GBP exposure.",
    "metadata": {"parent_id": "p_2.1", "page": 23},
}
```

### Query flow

```
Query: "How much did FX hedging add to operating income?"
   │
   ▼  1. vector search over CHILD embeddings
c_2.1_2  (score 0.89)   metadata.parent_id = "p_2.1"
   │
   ▼  2. look up docstore["p_2.1"]   (several children with the same parent → fetched once)
   │
   ▼  3. send the PARENT text (the whole §2.1) to the LLM
LLM sees the hedge, the $18M gain AND that GBP is unhedged → full, correct answer
```

**In short:** the **children** are what get embedded and searched; the **parent** is what the LLM reads; `parent_id` in the children's **metadata** is the link between them.

## Auto-merging

Same parent–child tree, plus a rule: if the share of a parent's children that were retrieved is above a threshold (e.g. > 50%), **merge them up** into the parent. This can be applied recursively over several levels (e.g. 2048 → 512 → 128 tokens).

## RAPTOR-style summary tree

```
Level 2:            [summary of the whole doc]
                     /                      \
Level 1:   [summary of cluster A]   [summary of cluster B]
             /    |    \               /      \
Level 0:  chunk chunk chunk         chunk    chunk
```

- Built **bottom-up from semantic clusters**, not from the document's headings, so it also works on documents with no structure.
- Search over all levels together ("collapsed tree") or go down from the top.
- Good for **broad or thematic questions** ("What are the main themes of this report?") that no single chunk answers.
- Cost: LLM summaries are needed at indexing time.

## Tradeoffs

| ✅ Pros | ❌ Cons |
|---|---|
| Chunks are coherent and stay on one topic | Depends on good structure detection; messy PDFs give messy trees |
| Precise matching **and** rich context (parent–child) | More complex indexing and storage (docstore + vector store) |
| Heading paths improve embeddings, BM25 and reranking | Parent sections can be big, which costs more context tokens |
| Free section and page metadata for citations and filtering | Section sizes vary a lot, so extra split and merge rules are needed |
| RAPTOR handles broad, summary-type questions | RAPTOR adds LLM cost at indexing time and must be rebuilt when documents change |

## When to use

- **Long, structured documents**: reports, manuals, contracts, papers, textbooks.
- Answers need **surrounding context** (definitions, conditions, exceptions).
- You need **section or page citations**.
- A simple, good default: **structure-aware splitting with heading paths + parent–child retrieval**.
