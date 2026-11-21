# PDF Processing for RAG

> Source: Multimodal RAG lecture — PDF read with **PyMuPDF (`fitz`)** to extract both text and images before embedding.

## Why PDF processing matters

The first step of any RAG pipeline over documents is **reading the PDF and separating what's inside it**: text, images, tables, metadata. A PDF is not a structured document — it's a set of instructions that *draws* characters and shapes at coordinates on a page. There are no real "paragraphs", "headers" or "tables" inside it, only positioned glyphs. So the quality of your extraction (and your cleaning) directly limits the quality of retrieval: **garbage in → garbage chunks → wrong answers.**

---

## Library comparison

From the lecture:

| Library | Text Extraction | Image Extraction | Speed | Memory Usage |
|---|---|---|---|---|
| **PyMuPDF (fitz)** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| PyPDF2 / PyPDF4 | ⭐⭐⭐ | ⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| pdfplumber | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| pdfminer | ⭐⭐⭐⭐ | ⭐ | ⭐⭐ | ⭐⭐ |

Extended with the capabilities that matter for cleaning and RAG:

| Library | Layout / coordinates | Tables | Fonts & sizes (for detecting headers/footnotes) | Render page to image | Scanned PDFs (OCR) | Cleaning tools | License |
|---|---|---|---|---|---|---|---|
| **PyMuPDF (fitz)** | ✅ blocks, lines, spans with bbox; `clip=` rect | ✅ `page.find_tables()` | ✅ via `get_text("dict")` | ✅ `get_pixmap()` | ✅ via Tesseract (`get_textpage_ocr`) | Crop by region, redaction, dehyphenation flag, sort reading order | **AGPL** (or paid commercial) |
| **pypdf** (successor of PyPDF2) | ❌ minimal | ❌ | ❌ | ❌ | ❌ | Split/merge/rotate/encrypt pages only | BSD |
| **pdfplumber** | ✅ every char/line/rect with coordinates | ✅✅ best open-source table extraction | ✅ per character | ✅ (debug images) | ❌ | `crop()`, `filter()` objects, visual debugging | MIT |
| **pdfminer.six** | ✅ layout analysis (LAParams) | ❌ | ✅ | ❌ | ❌ | Low level; you build the cleaning | MIT |

### Notes on each
- **PyMuPDF (`import fitz` / `import pymupdf`)** — built on the C library MuPDF, so it's the fastest and handles both text and images well. That's why the lecture uses it. Main caveat: **AGPL license** — check before using in closed-source commercial products. Companion package **`pymupdf4llm`** outputs Markdown (headings, lists, tables) directly — very convenient for chunking.
- **PyPDF2 / PyPDF4** — PyPDF2 is deprecated and merged back into **`pypdf`**; PyPDF4 is unmaintained. Good for page manipulation (split, merge, rotate, encrypt), weak for extraction: no layout, poor spacing, basically no image support.
- **pdfplumber** — built on pdfminer.six. Slower, but gives you **every character, line and rectangle with coordinates**, which makes it the best choice for **tables** and precise "only take this region" extraction. Has visual debugging (draw what it detected on the page).
- **pdfminer.six** — the low-level engine. Accurate text and layout analysis, but slow, memory-heavy and verbose to use. Use it only if you need fine control and pdfplumber isn't enough.

### Beyond the basic four (higher-level parsers)
| Tool | What it adds |
|---|---|
| **pymupdf4llm** | PDF → clean Markdown for LLMs, fast |
| **Unstructured** | Splits documents into typed elements (`Title`, `NarrativeText`, `Table`, `Header`, `Footer`, `Image`…) → easy to drop unwanted types; `hi_res` strategy uses layout models + OCR |
| **Docling** (IBM) | Layout model + table structure recognition, reading order, exports Markdown/JSON; strong on complex reports |
| **Marker** | PDF → Markdown with layout models, handles equations and tables |
| **LlamaParse**, **Azure Document Intelligence**, **AWS Textract**, **Google Document AI** | Managed/paid parsing with tables, forms, OCR |
| **Camelot**, **tabula-py** | Table-only extraction from text-based PDFs |

**Choosing:** default to **PyMuPDF** (speed + text + images). Add **pdfplumber** when tables matter. Move to **Docling / Unstructured / a cloud parser** when layouts are complex (multi-column reports, scanned forms, nested tables) and quality matters more than speed.

---

## PyMuPDF core capabilities

```python
import fitz  # PyMuPDF

doc = fitz.open("report.pdf")
doc.metadata            # title, author, creation date...
doc.get_toc()           # table of contents / bookmarks → [level, title, page]

for page_num, page in enumerate(doc):
    page.get_text()                  # plain text
    page.get_text("blocks")          # paragraphs-ish blocks with (x0, y0, x1, y1, text, ...)
    page.get_text("words")           # every word with coordinates
    page.get_text("dict")            # blocks → lines → spans with font, size, bold flags, bbox
    page.get_text(sort=True)         # sort top-left → bottom-right (better reading order)
    page.find_tables()               # detect tables → .extract() / .to_pandas()
    page.get_images(full=True)       # embedded raster images (xref ids)
    page.get_drawings()              # vector graphics (lines, rects, curves)
    page.get_pixmap(dpi=150)         # render the whole page (or a clip) to an image
    page.search_for("Revenue")       # find text positions
```

What the lecture did with it (per page): `get_text()` → `Document(type="text")` → split with `RecursiveCharacterTextSplitter` → CLIP text embedding; `get_images()` → `doc.extract_image(xref)` → PIL image → base64 (for the LLM) + CLIP image embedding → `Document(type="image", image_id=...)`.

⚠️ **Gotcha — vector charts:** `get_images()` only returns **embedded raster images**. Many charts (exported from Excel, matplotlib as PDF, PowerPoint) are drawn as **vector graphics**, so they won't appear there at all. Detect them with `page.get_drawings()` and **render the region or page** with `page.get_pixmap(clip=rect, dpi=150)` to get an image you can embed or send to a vision LLM.

---

## pymupdf4llm — PDF → Markdown before chunking

`pymupdf4llm` is a thin layer on top of PyMuPDF that converts a PDF into **Markdown** (headings, lists, bold/italic, tables) instead of a flat stream of text lines. Same speed, local, no models or API calls — so it's a low-risk upgrade over plain `get_text()`.

### Why it adds value
- **Structure-aware chunking (biggest win).** Plain `get_text()` loses headings, so you can only split by character count. With Markdown you can split on `#`/`##`/`###` (`MarkdownHeaderTextSplitter`) → each chunk is one coherent section, and the heading path goes into metadata (better retrieval, citations and filtering).
- **Tables survive** as Markdown tables instead of word soup → rows/columns stay aligned, LLMs read them well, and a table can be kept as one chunk.
- **Better reading order** — multi-column layouts handled, lists kept together.
- **LLM-friendly context** — models are heavily trained on Markdown, so retrieved chunks are easier to follow.
- **Fits the multimodal pipeline** — `page_chunks=True` gives per-page dicts with metadata; `write_images=True` / `embed_images=True` extract images → feed them to the CLIP / vision-LLM step from [[image processing]].

```python
import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

md = pymupdf4llm.to_markdown("report.pdf")
sections = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
).split_text(md)
chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(sections)
# each chunk carries {"h1": ..., "h2": ...} metadata
```

### Limitations
- **Headings are guessed from font size** — if the PDF uses the same size everywhere (or bold body text), `#` levels come out missing/wrong and header-based splitting gets worse.
- **It is not a cleaner** — headers, footers, page numbers, disclaimers, TOC all still end up in the Markdown (see below).
- **Complex tables** (merged cells, multi-row headers, tables spanning pages, borderless tables) can break → Docling or a cloud parser does better.
- **Scanned PDFs** need OCR first — no text layer, nothing to convert.
- **Charts stay images** — still need a vision LLM or CLIP to make them searchable.
- **AGPL** license (same as PyMuPDF).
- Little gain for **CLIP text embeddings** specifically (77-token limit) — the benefit is in splitting and LLM context.

### Does it import TOC, version tables, cover pages…? Yes — everything
pymupdf4llm converts **everything on the pages you give it**, with no judgment about what's useful. It changes the *format*, not the *content*.

| PDF part | What it becomes in Markdown |
|---|---|
| Document version / revision history table | A normal Markdown table |
| Table of contents | Lines like `Introduction ........ 3` (sometimes a table, bold lines or links) |
| Cover page | Title as a heading + company / date / "Confidential" |
| Running headers / footers | Repeated on every page unless cropped |
| Disclaimers / copyright | Plain paragraphs |

If chunked as-is, these become their own chunks. The **TOC is the worst**: it contains every section title, so it keyword-matches many queries and outranks the real content.

**Don't just delete — decide per part:**
| Part | Treatment |
|---|---|
| **Version table** | → **metadata** (latest version + date on every chunk → filter "only v3.2", answer "which version is this?"). If people may ask "what changed in v2.1?", keep it as one small chunk instead |
| **Table of contents** | → **structure**: drop the text, but use `doc.get_toc()` for the section hierarchy (often more reliable than font-size guessing) |
| **Cover, headers/footers, disclaimers** | → **drop** |

**Built-in options that help** (verify against your installed version — the API evolves):
- `pages=[...]` — convert only selected pages (skip cover/version/TOC pages if you know where they are)
- `margins=(left, top, right, bottom)` — ignore top/bottom bands → removes headers/footers
- `page_chunks=True` — one dict per page (`text` + `metadata`) → drop noise pages before joining

```python
import re, fitz, pymupdf4llm

doc = fitz.open("spec.pdf")
pages = pymupdf4llm.to_markdown(doc, page_chunks=True, margins=(0, 50, 0, 50))  # crop header/footer bands

def is_toc(md):
    return len(re.findall(r"\.{4,}\s*\d+\s*$", md, re.M)) > 5 or re.search(r"^#*\s*(table of )?contents", md, re.I | re.M)

def is_version_page(md):
    return re.search(r"(revision|version) history|change log", md, re.I)

doc_meta, kept = {}, []
for p in pages:
    md = p["text"]
    if is_version_page(md):
        rows = [r for r in md.splitlines() if r.startswith("|") and re.search(r"\d", r)]
        if rows:                                       # last row = latest version
            cells = [c.strip() for c in rows[-1].strip("|").split("|")]
            doc_meta["version"], doc_meta["version_date"] = cells[0], cells[1]
        continue
    if is_toc(md) or len(md.strip()) < 50:             # TOC, cover / blank pages
        continue
    kept.append(md)

clean_md = "\n\n".join(kept)
toc = doc.get_toc()   # [[level, title, page], ...] → section structure / metadata
# → MarkdownHeaderTextSplitter on clean_md, then add doc_meta to every chunk's metadata
```

When the TOC / version table share a page with real content, filter **after** header splitting by section name:
```python
DROP = re.compile(r"contents|revision history|version history|document control|disclaimer|copyright|references", re.I)
chunks = [c for c in sections if not DROP.search(" ".join(c.metadata.values()))]
```

### When to use it
| Situation | Choice |
|---|---|
| Reports, manuals, papers with clear headings | **pymupdf4llm** — big win |
| Many complex tables, forms, scanned pages | Docling / Unstructured `hi_res` / LlamaParse |
| Flat text (contracts, emails) with no structure | Plain `get_text()` + cleaning is enough |

**Default pipeline:** clean (crop margins, drop boilerplate pages) → pymupdf4llm → split by Markdown headers → recursive split for long sections → embed. Regexes above are heuristics — look at how *your* TOC/version tables render and adjust; for documents from one template, hard-coding page ranges / section names is often most reliable. A/B test against plain `get_text()` on 10–20 real questions.

---

## Docling (IBM) — core capabilities

Docling is an open-source (**MIT** license) document parser from IBM Research. Unlike PyMuPDF / pymupdf4llm, which read the PDF's text layer and use heuristics such as font size, Docling runs **AI models on the page**: a layout model, a table-structure model and OCR. It turns each page into a structured `DoclingDocument` where every element has a **type label, page number, bounding box and place in the hierarchy**. It is slower and heavier, but much better on complex layouts and tables, and it is the best open-source option for removing noise automatically.

### How the pipeline works
```
PDF / DOCX / PPTX / XLSX / HTML / images
 ├─ 1. Parse the text layer (Docling's own PDF backend)
 ├─ 2. Layout model (trained on DocLayNet) → label every region
 ├─ 3. TableFormer → rebuild table structure (rows, columns, merged cells, headers)
 ├─ 4. OCR (EasyOCR / Tesseract / RapidOCR / macOS OCR) for scanned pages and image text
 ├─ 5. Reading-order model → correct order across columns
 ├─ 6. Optional enrichment: picture classification, picture description (VLM), formulas → LaTeX, code blocks
 └─ 7. DoclingDocument → Markdown / HTML / JSON / DocTags, plus chunking
```

### Key capabilities
| Capability | What it gives you |
|---|---|
| **Many input formats** | PDF, DOCX, PPTX, XLSX, HTML, Markdown, images (PNG/JPG/TIFF), and audio through speech recognition, all into **one unified document model** |
| **Layout labels** | `title`, `section_header`, `text`, `list_item`, `table`, `picture`, `caption`, `footnote`, `formula`, `code`, `page_header`, `page_footer`, `document_index` (TOC)… |
| **Content layers** | Page headers and footers go to a separate **"furniture"** layer and the real content to **"body"**. Markdown export uses the body only, so headers and footers are removed by default |
| **Tables (TableFormer)** | Handles merged cells, multi-row headers and borderless tables. Export to Markdown, HTML or a pandas DataFrame |
| **Reading order** | Correct for multi-column reports and slides |
| **OCR** | Built in; only needed for scanned pages or bitmap text (can be turned off for speed) |
| **Images** | Extracts pictures and charts with page and bbox, and can **classify** them (chart, diagram, logo…) and **describe** them with a VLM. This feeds the multimodal pipeline in [[image processing]] |
| **Formulas and code** | Optional enrichment: formulas → LaTeX, code blocks recognized |
| **VLM pipeline** | Alternative end-to-end mode using a small document VLM (SmolDocling / GraniteDocling) that converts page images straight to DocTags/Markdown |
| **Chunking** | `HierarchicalChunker` (by document structure) and `HybridChunker` (structure plus a token limit for your embedding model). Each chunk carries its **heading path and page numbers** |
| **Integrations** | LangChain (`langchain-docling`, `DoclingLoader`), LlamaIndex, Haystack, CrewAI; CLI (`docling report.pdf --to md`); MCP server |
| **Runs locally** | No API calls; models download on first run; a GPU or Apple MPS speeds it up |

### Basic usage
```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter()
result = converter.convert("report.pdf")        # also accepts a URL
doc = result.document                           # DoclingDocument

md = doc.export_to_markdown()                   # body only, so headers/footers are already gone
doc.save_as_json("report.json")                 # lossless structure for later processing
```

### Storing the output as Markdown
Yes, Docling output can be saved as a `.md` file. There are three ways:
```python
# 1. Built-in save
doc.save_as_markdown("report.md")

# 2. Export the string and write it yourself (lets you post-process first)
from pathlib import Path
Path("report.md").write_text(doc.export_to_markdown(), encoding="utf-8")
```
```bash
# 3. CLI: writes report.md to the output folder
docling report.pdf --to md --output ./parsed/
```
Why store it:
- You **parse once and chunk many times**. Docling is slow, so caching the `.md` saves re-running the models every time you try a new chunk size or splitter.
- The file is **easy to read and correct by hand** before indexing.
- It can **go under version control** with your notes or data.

The `.md` file **loses** the labels, bounding boxes, page numbers and hierarchy. If you need page-level citations or label-based cleaning later, also keep the JSON (`save_as_json`), which can be reloaded as a full `DoclingDocument` with `DoclingDocument.load_from_json(...)`.

### Configuring the pipeline
```python
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

opts = PdfPipelineOptions()
opts.do_ocr = False                              # text-based PDFs: skip OCR (much faster)
opts.do_table_structure = True                   # TableFormer on
opts.table_structure_options.do_cell_matching = True
opts.generate_picture_images = True              # keep image crops for CLIP / VLM
opts.images_scale = 2.0                          # higher resolution crops
# opts.do_picture_description = True             # VLM captions for every picture (slower)

converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
doc = converter.convert("report.pdf").document
```

### Walking elements and cleaning by label
```python
from docling_core.types.doc import DocItemLabel

DROP = {DocItemLabel.DOCUMENT_INDEX,      # table of contents
        DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER,
        DocItemLabel.FOOTNOTE}

for item, level in doc.iterate_items():
    if item.label in DROP:
        continue
    page = item.prov[0].page_no if item.prov else None
    print(item.label, page, getattr(item, "text", "")[:80])
```
Layout labels deal with structural noise. Semantic noise such as disclaimers or version-history tables still needs section-name rules. For example, skip every item under a `section_header` that matches `revision history|disclaimer`.

### Tables and pictures
```python
for table in doc.tables:
    df = table.export_to_dataframe(doc=doc)      # pandas DataFrame (older versions: no doc arg)
    md_table = table.export_to_markdown(doc=doc)

for pic in doc.pictures:
    img = pic.get_image(doc)                     # PIL image (needs generate_picture_images=True)
    caption = pic.caption_text(doc)
    # → CLIP embedding / VLM summary / base64 store, as in [[image processing]]
```

### Chunking for RAG
```python
from docling.chunking import HybridChunker

chunker = HybridChunker(tokenizer="sentence-transformers/all-MiniLM-L6-v2", max_tokens=512)
for chunk in chunker.chunk(dl_doc=doc):
    text_to_embed = chunker.contextualize(chunk)   # prepends heading path → better embeddings
    meta = {"headings": chunk.meta.headings,
            "pages": sorted({p.page_no for it in chunk.meta.doc_items for p in it.prov})}
```

LangChain shortcut:
```python
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType

docs = DoclingLoader(file_path="report.pdf", export_type=ExportType.DOC_CHUNKS).load()  # already chunked + metadata
```

### Trade-offs
- **Slower than PyMuPDF**, by roughly seconds per page against milliseconds, especially with OCR on and no GPU. For large batches, turn off OCR and picture description when you don't need them.
- **Heavy dependencies**: PyTorch plus model downloads, so a bigger install than PyMuPDF.
- **Model errors still happen.** A TOC might be labeled `text`, or a heading missed, so spot-check the output.
- The API changes quickly between versions (for example `export_to_dataframe(doc=...)`). Check the docs for your installed version.

### Docling vs pymupdf4llm vs Unstructured
| | **pymupdf4llm** | **Docling** | **Unstructured** (`hi_res`) |
|---|---|---|---|
| Approach | Heuristics on the text layer | Layout and table AI models | Layout model + OCR |
| Speed | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
| Complex tables | ⭐⭐ | ⭐⭐⭐⭐⭐ (TableFormer) | ⭐⭐⭐ |
| Headers/footers removed | Only by margin cropping | ✅ automatic ("furniture" layer) | ✅ `Header`/`Footer` types |
| TOC detected | ❌ | ✅ `document_index` label | ❌ |
| Scanned PDFs | Needs separate OCR | ✅ built in | ✅ built in |
| Built-in chunker | ❌ (use LangChain splitters) | ✅ Hierarchical / Hybrid | ✅ `chunk_by_title` |
| License | AGPL | **MIT** | Apache 2.0 |
| Best for | Clean, text-based PDFs at scale | Complex reports, tables, mixed formats, quality first | Mixed formats, element-level pipelines |

---

## Cleaning the PDF — remove what doesn't help answer business questions

A lot of what's in a PDF is noise for retrieval: it eats tokens, pollutes chunks, and — worst — repeated boilerplate (same footer on every page) makes many chunks look similar to each other, hurting ranking. Clean **before chunking**.

### What to remove (or separate)
| Noise | Why it hurts | How to detect |
|---|---|---|
| **Headers / footers** (company name, "Confidential", doc title) | Repeated on every page → duplicate signal in every chunk | Fixed vertical position; same line repeated on most pages |
| **Page numbers**, "Page 3 of 40" | Meaningless tokens | Regex; position at top/bottom |
| **Table of contents, lists of figures** | Lots of keywords with no content → retrieved instead of the real section | `doc.get_toc()` page range; lines ending in dot leaders + numbers |
| **Cover pages, blank pages, "This page intentionally left blank"** | Empty or useless chunks | Very short text per page |
| **Legal disclaimers, copyright, forward-looking statements** | Boilerplate that matches many queries loosely | Known phrases; section heading names |
| **Watermarks** ("DRAFT", "CONFIDENTIAL" diagonal) | Injected randomly into text | Rotated spans, huge/light font, repeated on every page |
| **References / bibliography / index / appendices** (often) | Noise for business Q&A | Section heading → drop everything after it |
| **Logos, icons, decorative images, signatures** | Waste of embeddings & vision tokens; same logo on every page | Small size, same image `xref` or hash repeated across pages |
| **Footnotes** (sometimes) | Break the flow of paragraphs | Small font size at page bottom |

### What to fix
- **Hyphenation** across line breaks (`reve-\nnue` → `revenue`)
- **Broken lines** inside paragraphs (join lines that don't end with punctuation)
- **Ligatures & odd Unicode** (`ﬁ` → `fi`, non-breaking spaces) → `unicodedata.normalize("NFKC", text)`
- **Multi-column reading order** — use `sort=True` or `get_text("blocks")` and sort by column/x then y; otherwise left and right columns get interleaved
- **Tables** — extract separately (`find_tables()` / pdfplumber) and store as Markdown/CSV rows, not mashed text
- **Excess whitespace**, repeated blank lines

### What to keep as metadata (not as text)
Page number, section heading, source file, element type (text / table / image), date — useful for **citations, filtering** (e.g. only the 2024 report) and answering "on page 1…" questions, like the lecture's chart query.

### Code recipes (PyMuPDF)

**1. Crop out header/footer bands**
```python
HEADER, FOOTER = 50, 50   # points (72 pt = 1 inch); tune by inspecting your PDFs

def body_text(page):
    r = page.rect
    body = fitz.Rect(r.x0, r.y0 + HEADER, r.x1, r.y1 - FOOTER)
    return page.get_text("text", clip=body, sort=True)
```

**2. Remove lines repeated on most pages (position-independent)**
```python
from collections import Counter

page_lines = [{l.strip() for l in p.get_text().splitlines() if l.strip()} for p in doc]
counts = Counter(l for lines in page_lines for l in lines)
boilerplate = {l for l, c in counts.items() if c >= 0.6 * len(doc)}   # on ≥60% of pages

def remove_boilerplate(text):
    return "\n".join(l for l in text.splitlines() if l.strip() not in boilerplate)
```

**3. Use font size to drop footnotes / small print and detect headings**
```python
def clean_spans(page, min_size=8):
    kept, headings = [], []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                if span["size"] < min_size:          # footnotes, legal small print
                    continue
                if span["size"] >= 14:                # likely a heading → section metadata
                    headings.append(span["text"])
                kept.append(span["text"])
    return " ".join(kept), headings
```
Headings let you **drop whole sections** (e.g. everything under "References", "Disclaimer", "Appendix") and attach the section name to each chunk.

**4. Skip TOC, blank and cover pages**
```python
import re
def is_noise_page(text):
    if len(text.strip()) < 50:                             # blank / cover / divider
        return True
    dot_leaders = len(re.findall(r"\.{4,}\s*\d+\s*$", text, re.M))
    return dot_leaders > 5                                 # table of contents
```

**5. Filter decorative & duplicate images**
```python
import hashlib
seen = set()
for page in doc:
    for img in page.get_images(full=True):
        info = doc.extract_image(img[0])
        if info["width"] < 100 or info["height"] < 100:   # icons, bullets, small logos
            continue
        h = hashlib.md5(info["image"]).hexdigest()
        if h in seen:                                     # logo repeated on every page
            continue
        seen.add(h)
        # → keep: embed / summarize / store base64
```

**6. Fix text artifacts**
```python
import unicodedata, re
def normalize(text):
    text = unicodedata.normalize("NFKC", text)              # ligatures, odd spaces
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)            # de-hyphenate
    text = re.sub(r"(?<![.:!?])\n(?!\n)", " ", text)        # join broken lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return re.sub(r"[ \t]+", " ", text).strip()
```
(PyMuPDF also has `flags=fitz.TEXT_DEHYPHENATE` in `get_text`.)

**7. Redact sensitive data (PII) permanently**
```python
for rect in page.search_for("Account No:"):
    page.add_redact_annot(rect)
page.apply_redactions()     # really removes the content, not just a black box on top
```

**8. Detect scanned pages → send to OCR**
```python
if not page.get_text().strip() and page.get_images():
    tp = page.get_textpage_ocr(dpi=300, full=True)   # needs Tesseract installed
    text = page.get_text(textpage=tp)
```

### Same ideas in pdfplumber
```python
import pdfplumber
with pdfplumber.open("report.pdf") as pdf:
    page = pdf.pages[0]
    body = page.crop((0, 50, page.width, page.height - 50))               # drop header/footer
    body = body.filter(lambda o: o.get("object_type") != "char" or o.get("size", 0) >= 8)  # drop small print
    text = body.extract_text()
    tables = body.extract_tables()                                         # list of rows
```

### Element-based cleaning (Unstructured / Docling)
Higher-level parsers label each element, so cleaning becomes a filter:
```python
from unstructured.partition.pdf import partition_pdf
elements = partition_pdf("report.pdf", strategy="hi_res", infer_table_structure=True)
useful = [e for e in elements if e.category not in {"Header", "Footer", "PageNumber", "PageBreak"}]
```
Note: element types describe **layout, not purpose** — there is no "TOC" or "version history" type in Unstructured (they come out as `Title` / `ListItem` / `Table`), so section-name rules are still needed.

### Libraries & models for cleaning

There is **no single library that knows what is "business-irrelevant"**. Tools fall into two groups:
- **Structural noise** (headers, footers, page numbers, footnotes, sometimes TOC) → **layout-detection models** can remove it automatically.
- **Semantic noise** (disclaimers, version history, boilerplate legal text) → needs **rules** (section names, regex) or an **LLM classifier**.

| Category | Tool / model | What it removes / fixes |
|---|---|---|
| **Layout-aware parsers** | **Docling** (IBM) | Layout model trained on DocLayNet labels regions as `page_header`, `page_footer`, `footnote`, `caption`, `section_header`, `document_index` (TOC)…; headers/footers go to a separate "furniture" layer and are left out of the Markdown export → closest to automatic TOC/header removal |
| | **Unstructured** (`hi_res`) | Typed elements (`Header`, `Footer`, `PageNumber`, …) via YOLOX layout model + OCR; plus text cleaners (below) |
| | **Marker** | PDF → Markdown, removes headers/footers/artifacts, handles tables & equations |
| | **GROBID** | Scientific papers: separates header metadata, body, **references**, footnotes into TEI XML |
| | **LlamaParse** | Paid; accepts natural-language parsing instructions (e.g. "ignore headers, footers and table of contents") |
| **Layout-detection models** (build your own) | **DocLayNet**-trained models (YOLO / DETR), **DocLayout-YOLO**, **LayoutParser** (Detectron2, PubLayNet), **Surya** layout, **LayoutLMv3** (fine-tune for region classification) | Detect region types on the page image → drop the classes you don't want |
| **Text cleaning** | `unstructured.cleaners.core` — `clean_extra_whitespace`, `clean_bullets`, `clean_dashes`, `group_broken_paragraphs`, `replace_unicode_quotes`, `clean_non_ascii_chars` | Ready-made fixes for common extraction artifacts |
| | **ftfy** | Fixes mojibake / broken Unicode (`â€™` → `'`) |
| | `unicodedata` (NFKC), `re` | Ligatures, odd spaces, hyphenation, broken lines |
| **Deduplication** | **datasketch** (MinHash/LSH), exact hashing | Removes near-duplicate chunks (repeated boilerplate across docs/versions) |
| **PII removal** | **Microsoft Presidio**, spaCy NER; PyMuPDF redaction | Detect and mask/redact names, emails, account numbers |
| **LLM / VLM-based cleaning** | Small LLM (GPT-4.1-mini, Gemini Flash, local Llama/Qwen) as a classifier; VLM reading page images → Markdown | "Is this chunk useful content or boilerplate?" / "Convert this page to Markdown, ignore headers, footers, TOC" — most flexible, costs tokens; best for semantic noise |

**Practical approach:** layout model (Docling / Unstructured `hi_res`) or pymupdf4llm + margins for structural noise → section-name / regex rules for known semantic noise → optional cheap LLM classifier on chunks for the rest → text normalization (ftfy, cleaners) → dedup. Always spot-check: every tool misclassifies some regions.

---

## Recommended cleaning pipeline

```
PDF
 ├─ 1. Inspect a few pages (where are headers/footers? columns? scanned? vector charts?)
 ├─ 2. Per page: detect scanned → OCR | skip noise pages (cover, TOC, blank)
 ├─ 3. Text: crop margins → drop repeated boilerplate → drop small print → fix hyphens/lines/unicode
 ├─ 4. Structure: headings → section metadata; drop unwanted sections (references, disclaimers)
 ├─ 5. Tables: extract separately → Markdown rows
 ├─ 6. Images: drop tiny/duplicate → render vector charts → embed / summarize → store base64
 ├─ 7. Attach metadata (page, section, type, source)
 └─ 8. Chunk → embed → vector store
```

Always **spot-check the cleaned output** on a few documents — over-aggressive rules (e.g. a 60% repetition threshold) can delete real content on short documents.

Related: [[image processing]]
