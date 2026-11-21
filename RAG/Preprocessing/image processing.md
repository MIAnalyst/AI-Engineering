# Image Processing for RAG

> Source: Multimodal RAG lecture (PDF with text + images → CLIP embeddings → FAISS → GPT-4.1)

## Why images need their own handling

A normal RAG pipeline only understands text: `data → chunks → embeddings → vector store → retriever → top-k → LLM`. Real documents (reports, slides, manuals) carry a lot of meaning in **charts, diagrams, tables-as-images, scanned pages and photos**. If you only call `page.get_text()`, all of that is silently lost. Multimodal RAG = **text + images** in the same pipeline, so a question like *"What does the chart on page 1 say about revenue?"* can be answered.

There are three main ways to turn an image into something retrievable:

---

## CLIP vs Vision LLM vs OCR

| | **CLIP** (Contrastive Language-Image Pre-training) | **Vision LLM** (VLM / multimodal LLM) | **OCR** (Optical Character Recognition) |
|---|---|---|---|
| **What it does** | Maps images and text into the **same vector space** | *Looks at* the image and **describes / reasons** about it in natural language | **Reads the characters** printed in the image |
| **Output** | A fixed-size vector (e.g. 512-dim) | Free text (caption, summary, answer, JSON, Mermaid…) | Raw text (+ optional bounding boxes) |
| **Understands** | Overall "gist" / visual concepts (a bar chart, a dog, a building) | Content, layout, relationships, trends, numbers, labels | Only the written characters — no meaning |
| **Reads small text?** | Poorly (image resized to 224×224) | Yes, usually well | Yes — that's its whole job |
| **Understands structure (nodes, arrows, axes)?** | Weak | Good | No (just extracts labels) |
| **Cost / speed** | Very cheap, fast, local, open source | Expensive, slower, usually an API call per image | Cheap, local (Tesseract) or paid cloud APIs |
| **Role in RAG** | **Retrieval** — embed images & query directly | **Indexing** (image → description to embed) and/or **generation** (answer from the image) | **Indexing** — turn scanned/text-heavy images into chunkable text |
| **Use when** | Natural images, photos, product catalogues, "find visually similar", fast unified text+image search | Charts, diagrams, infographics, anything where *meaning* and *relations* matter | Scanned PDFs, screenshots of documents, forms, invoices, receipts, tables saved as images |
| **Avoid when** | Text-heavy images, diagrams with fine detail, many look-alike figures | Huge volumes with tight budget/latency | The image's meaning is visual (a chart trend, a photo) |

**Rule of thumb:**
- Image is **mostly text** → OCR.
- Image's **meaning is in its structure/data** (chart, diagram) → Vision LLM to describe it, then embed the description.
- Image is a **natural picture**, or you need a cheap unified index / image-to-image search → CLIP.
- In production these are often **combined**: OCR for the labels + VLM summary for the meaning + (optionally) a CLIP vector, all pointing back to the same original image.

---

## 1. CLIP

### What it is
CLIP (OpenAI, open source, on Hugging Face) is two encoders trained together:
- **Vision Transformer (ViT)** → encodes images
- **Text Transformer** → encodes text

It was trained on ~**400 million image–caption pairs** with a **contrastive** objective: the vector of an image and the vector of *its own caption* are pulled close together, while vectors of mismatched pairs are pushed apart. The result is a **shared embedding space** where "a photo of a bar chart" (text) lands near actual bar chart images. That's why a **single model** can embed both text chunks and images and put them in one vector store — the lecture's "unified embeddings".

The lecture used `openai/clip-vit-base-patch32` → **512-dimensional** vectors, text input capped at **77 tokens**.

### How it's used in the pipeline
```python
from transformers import CLIPProcessor, CLIPModel
import torch

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")  # resizes/normalizes images, tokenizes text

def embed_image(pil_image):
    inputs = clip_processor(images=pil_image, return_tensors="pt")
    with torch.no_grad():
        feats = clip_model.get_image_features(**inputs)
    feats = feats / feats.norm(dim=-1, keepdim=True)   # unit vector → cosine similarity = dot product
    return feats.squeeze().numpy()

def embed_text(text):
    inputs = clip_processor(text=text, return_tensors="pt", padding=True, truncation=True, max_length=77)
    with torch.no_grad():
        feats = clip_model.get_text_features(**inputs)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.squeeze().numpy()
```
Full flow: **PDF → extract text & images (PyMuPDF) → text chunks → `embed_text` / images → `embed_image` + store image as base64 → FAISS (`from_embeddings`) → query embedded with CLIP → `similarity_search_by_vector` → separate text docs and image docs → build multimodal message (text + `image_url` with base64) → GPT-4.1 → multimodal answer.**

Important details from the lab:
- Every document gets metadata `type: "text"` or `type: "image"` (plus `page`, `image_id`) — that's how you know what you retrieved and how to format it for the LLM.
- Text chunks must be short (CLIP truncates at 77 tokens), so chunk size is much smaller than with normal text embedders.

### Does CLIP "retrieve the same image" from the PDF?
CLIP itself never goes back into the PDF. What actually happens is bookkeeping: at indexing time you extract the image, embed it, and store the vector **together with a record that identifies the original** — an `image_id`, the page number, and the **base64 copy of the image** (or a file path / object-store URL). When a query's vector lands closest to that image vector, the vector store returns that **record**, and the record gives you back the exact original image, byte for byte. So yes, you get the *same* image — but because you saved it, not because CLIP found it. Matching is by **semantic similarity**, not exact identity: a text query like "revenue chart by quarter" scores high against a bar chart because CLIP learned that association. If you query with the same or a near-duplicate image (image-to-image search), the cosine score will be close to 1.

### Is the vector used to recreate the image?
No. A CLIP embedding is a **one-way, lossy summary**: 512 numbers that capture *what the image is about*, not the pixels. CLIP has no decoder, so there is nothing that turns the vector back into an image; the colors, exact numbers, labels, layout and fine detail are gone. The vector's only job is to be **a search key** — it decides *which* record ranks highest. What gets sent to the multimodal LLM (GPT-4.1, Gemini 2.5 Flash) is the **stored base64 image**, which is exactly why the lecture saves every image as base64 next to its embedding. (Generative models like DALL·E 2 / unCLIP can *generate a new* image conditioned on a CLIP embedding, but that's a different, loosely similar picture — never a reconstruction of your diagram.)

### What about diagrams with nodes and connections?
This is where CLIP is weakest. It was trained mostly on natural web photos with short captions, so it captures the **global gist** ("this is a flowchart", "this is a network diagram") but not **entity-level structure** — which node is called what, which arrow points where, what connects to what. On top of that the image is downscaled to 224×224, so small labels become unreadable, and the text side only sees 77 tokens, so a detailed query gets cut. In practice CLIP behaves like a "bag of visual concepts": a query such as *"diagram where the auth service calls the user database"* will probably retrieve *a* diagram, but if your document has many similar-looking architecture diagrams it will **not reliably pick the exact one**. There is also a known **modality gap**: text-to-text similarity scores are systematically higher than text-to-image scores, so in a unified index text chunks often outrank the right image even when the image is the better answer.

For diagrams the better pattern is: **describe, then embed as text**. Send each diagram to a Vision LLM at indexing time and ask for a structured description (entities, relationships, flow, labels — even a Mermaid version), optionally add OCR'd labels, embed *that text* with a normal text embedder, and keep a pointer to the original image (the *multi-vector retriever* pattern: search on the summary, return the image). At answer time, pass the original image to the multimodal LLM for the final reasoning. For page-level visual retrieval at scale, document-image retrievers like **ColPali / ColQwen** (late-interaction over page patches) do much better than plain CLIP.

### Libraries & models
- **Models:** `openai/clip-vit-base-patch32` (lecture, 512-dim), `clip-vit-large-patch14` (better, 768-dim), **OpenCLIP** (LAION-trained, larger variants), **SigLIP / SigLIP 2** (Google, generally stronger than CLIP), **Jina CLIP v2**, **Nomic Embed Vision**, Cohere Embed (multimodal), **ColPali / ColQwen** (document page retrieval)
- **Libraries:** Hugging Face `transformers` (`CLIPModel`, `CLIPProcessor`), `open_clip_torch`, `sentence-transformers` (`clip-ViT-B-32`), `Pillow`, `torch`
- **Vector stores:** FAISS (lecture), Chroma, Qdrant, Weaviate, Milvus — any store that accepts precomputed vectors

---

## 2. Vision LLMs (Multimodal LLMs)

### What they are
LLMs trained on **both text and images**: you can pass an image as input and ask questions about it. Normal LLMs (GPT-3.5, text-only GPT-4) only do text generation; multimodal LLMs can read a chart, compare bars, read labels, follow arrows in a diagram, and explain it.

### Two roles in RAG
1. **Generation (lecture's use):** after retrieval, the top-k text chunks and images are formatted into one message and the model answers using both. In the lab GPT-4.1 correctly read that "Q1 had the lowest revenue (blue bar)" and that bars rise left to right.
2. **Indexing (image summarization):** at ingestion, ask the VLM to describe each image ("Summarize this chart: axes, values, trend, key takeaway"). Embed the summary as text. This makes charts and diagrams searchable by meaning and fixes most of CLIP's weaknesses — at the cost of one API call per image.

### Message format
Multimodal APIs need a specific content structure — images are sent as base64 data URLs (or hosted URLs) in an `image_url` block:
```python
from langchain_core.messages import HumanMessage

content = [{"type": "text", "text": f"Question: {query}\n\nContext:\n{text_context}"}]
for img_doc in image_docs:
    b64 = image_store[img_doc.metadata["image_id"]]
    content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

response = llm.invoke([HumanMessage(content=content)])
```
Check each provider's docs — the exact format differs slightly (OpenAI, Gemini, Anthropic).

### Trade-offs
- Best understanding of charts/diagrams/infographics; can output structured text (JSON, tables, Mermaid).
- Costs tokens per image (high-resolution images = more tokens), adds latency, can hallucinate numbers on dense charts — ask it to only report values it can see.

### Libraries & models
- **API models:** OpenAI **GPT-4.1 / GPT-4o** (lecture), Google **Gemini 2.5 Flash / Pro** (lecture), Anthropic **Claude** (Sonnet / Opus)
- **Open source:** **Qwen2.5-VL**, **LLaVA**, **Llama 3.2 Vision**, **InternVL**, **Pixtral**, **Florence-2** (captioning/OCR, small)
- **Libraries:** `langchain` (`init_chat_model`, `HumanMessage`), `openai`, `google-genai`, `anthropic`, `transformers`, `ollama` (local), LangChain `MultiVectorRetriever` for summary→image retrieval

---

## 3. OCR

### What it is
OCR converts pixels of written characters into machine-readable text. It does **not** understand the image — it just reads what's printed.

### When you need it
- **Scanned PDFs**: the page is one big image, so `page.get_text()` returns empty. Quick check: if a page has no text but has images → OCR it.
- Screenshots of documents, forms, invoices, receipts, ID cards.
- **Tables or slides saved as images.**
- Extracting **labels** from diagrams to enrich a VLM description or as keywords for hybrid (BM25) search.

### Trade-offs
- Cheap and fast, very good on clean printed text.
- Struggles with handwriting (classic engines), poor scans, rotated or multi-column layouts, and loses **reading order / table structure** unless you use a layout-aware tool.
- Gives zero information about images whose meaning is visual (a chart's trend, a photo).
- Preprocess for better accuracy: grayscale, binarize, deskew, upscale (render the page at 300 DPI).

### Libraries & models
- **Open source engines:** **Tesseract** (`pytesseract`; also built into PyMuPDF via `page.get_textpage_ocr()`), **EasyOCR**, **PaddleOCR** (strong on layout and multilingual), **docTR**, `ocrmypdf` (adds a text layer to scanned PDFs)
- **Layout-aware / document parsers:** **Docling**, **Unstructured** (`strategy="hi_res"`), **Marker**, **Surya**, **Nougat** (scientific papers)
- **Cloud:** **AWS Textract**, **Google Document AI / Cloud Vision**, **Azure AI Document Intelligence**, **Mistral OCR**
- Note: modern VLMs (GPT-4.1, Gemini, Qwen2.5-VL) are also excellent OCR engines when you need text + understanding together.

---

## Quick decision summary

```
Image in the document
├── Mostly text (scan, form, invoice, screenshot)  → OCR → chunk → text embedding
├── Chart / diagram / infographic                  → Vision LLM summary (+ OCR labels) → text embedding, keep link to original image
└── Natural photo / need cheap unified search      → CLIP image embedding (store base64 alongside)

At answer time → pass the ORIGINAL stored image (base64) + text chunks to a multimodal LLM
```

Related: [[pdf processing]]
