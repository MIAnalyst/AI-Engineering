# Retrieval Guided by a Table of Contents

Use the document's **table of contents (ToC)** as a map: first decide **where** to look, then search **only inside** those sections. This is how a person uses a 300-page manual: they don't read every page, they check the contents and jump to the right chapter.

Related: [Hierarchical chunking](../Preprocessing/Hierarchical%20chunking.md) · [PageIndex-style tree search](PageIndex-style%20tree%20search.md) · [Reranking](Reranking.md) · [pdf processing](../Preprocessing/pdf%20processing.md)

## In short

**The idea:** use the table of contents (ToC) as a map. You narrow down **where** to look first, then search inside that area. This is how a person uses a 300-page manual: they don't read every page, they check the contents and jump to the right chapter.

**Typical flow:**

1. **Get the ToC**: from PDF bookmarks (`doc.get_toc()` → `[level, title, page]`), from detected headings, or by having an LLM build one if the document has none.
2. **Route**: give the LLM the query plus the ToC (titles, maybe one-line summaries) and ask which sections are likely relevant. Alternatively, embed the section titles and summaries and match against those.
3. **Scoped retrieval**: run vector or hybrid search only inside the chosen sections' page ranges, or pass those sections whole if they fit in context.
4. **Answer with citations**: page and section metadata comes for free.

**Where it shines:** long, well-structured documents such as 10-Ks, legal contracts, standards, technical manuals and textbooks. In these, the section usually tells you whether a passage is relevant. "Revenue" in *Risk Factors* and "revenue" in *Consolidated Statements* mean very different things to the question.

**Weak spots:** documents with poor or missing structure, questions whose answer is spread over many sections, and questions that span several documents (you need one ToC per document, plus routing between documents).

## The problem it solves

Plain vector search compares the query against **every chunk in the corpus** and ignores where each chunk sits:

- The word "revenue" in *Risk Factors* ("revenue may decline if...") and in *Consolidated Statements* (the real numbers) can look almost the same to an embedding model, but only one of them answers "What was revenue in 2023?".
- In long documents, many chunks look similar, so the correct one can fall out of the top-k.
- Chunks from the ToC itself, the index, or the list of figures are full of keywords but have no content, so they often get retrieved by mistake.

In a well-structured document, **the section tells you a lot about whether a passage is relevant**. ToC-guided retrieval uses that information.

## How it works

```
                Query
                  │
                  ▼
   ┌────────────────────────────┐
   │ 1. ToC (titles + pages,    │
   │    optional summaries)     │
   └─────────────┬──────────────┘
                 ▼
   ┌────────────────────────────┐
   │ 2. Route: which sections   │  ◄── LLM reads the ToC, or embedding match on titles/summaries
   │    are likely relevant?    │
   └─────────────┬──────────────┘
                 ▼
   ┌────────────────────────────┐
   │ 3. Scoped retrieval inside │  ◄── vector/hybrid search filtered by section,
   │    the chosen sections     │      or pass whole sections if they fit
   └─────────────┬──────────────┘
                 ▼
         (optional rerank)
                 ▼
     4. LLM answers with section/page citations
```

### 1. Get the ToC

| Source | Notes |
|---|---|
| **PDF bookmarks** | `doc.get_toc()` (PyMuPDF) → `[level, title, page]`. Best option when present. |
| **Detected headings** | From Markdown after conversion, or from layout labels (Docling `SECTION_HEADER`). |
| **Printed ToC page** | Parse lines like `Introduction ........ 3`, then map printed page numbers to PDF page indexes (they often differ). |
| **LLM-generated** | For documents with no headings: ask an LLM to propose a section structure from page samples. |

Make it richer by adding a **1–2 sentence summary per section**. Titles alone ("Other Matters", "Note 14") are often too vague for routing.

### 2. Route the query to sections

- **LLM routing**: prompt with the query and the ToC, and ask for the IDs of the relevant sections (allow several, and allow "none"). Best quality, but adds one LLM call.
- **Embedding routing**: embed the section titles and summaries, and pick the top-n sections by similarity. Cheap, but weaker.
- **Hybrid**: embedding shortlist → LLM picks from the shortlist.

### 3. Scoped retrieval

- Store `section_id`, `heading_path` and `page_range` as **metadata on every chunk** (this comes from [Hierarchical chunking](../Preprocessing/Hierarchical%20chunking.md)).
- Run vector or hybrid search with a **metadata filter** on the chosen sections.
- Or, if the chosen sections are small enough, **skip chunk search** and pass the full sections to the LLM.

### 4. Answer

Citations come for free: `"... (Section 7.2, p. 84)"`.

## Example prompt for routing

```
You are given a document's table of contents and a question.
Return the IDs of the sections most likely to contain the answer (max 3).
If none look relevant, return [].

Question: {query}

Table of contents:
{toc_with_ids_titles_pages_summaries}
```

## Tradeoffs

| ✅ Pros | ❌ Cons |
|---|---|
| Much smaller search space, so fewer look-alike chunks | **Routing errors are fatal**: if the router picks the wrong section, the answer is never found |
| Uses the author's own organisation of the content | Needs a good ToC; poor or missing structure means poor routing |
| Keeps ToC, index and boilerplate chunks out of the results | Extra LLM call (latency + cost) when routing with an LLM |
| Natural section and page citations, so answers are explainable | Answers **spread over many sections** are hard to route |
| Works well with hybrid search and reranking | Vague titles ("Other", "Note 14") need summaries to be useful |
| | Several documents need **two-stage routing** (pick documents → pick sections) |

**Mitigations:** let the router pick several sections; fall back to global search if scoped search finds nothing good; run scoped and global search in parallel and merge the results.

## When to use

- **Long, well-structured single documents**: 10-K/annual reports, legal contracts, standards, technical manuals, textbooks, policy handbooks.
- Questions where **"where would this be written?"** is easy for a person to answer.
- Not a good fit for: short documents, chat logs, emails, or flat collections of unrelated snippets.

[PageIndex-style tree search](PageIndex-style%20tree%20search.md) takes this idea further: it uses the ToC tree as the **whole index**, and an LLM walks it level by level.
