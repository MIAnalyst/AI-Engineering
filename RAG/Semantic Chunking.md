# Semantic Chunking

Part of Enhanced RAG Techniques → Advanced Chunking and Preprocessing. See also [Preprocessing Techniques](Preprocessing%20Techniques.md) for the other chunking strategies.

## Where it fits

Source files (PDF, txt, Word, JSON, database) are converted into `Document` objects, then split, embedded and stored in a vector store. The splitting step so far used a document splitter such as `RecursiveCharacterTextSplitter`. Semantic chunking replaces that splitting step.

## The problem with RecursiveCharacterTextSplitter

It splits only on parameters (size, not meaning):

- `chunk_size = 500`
- `chunk_overlap = 50`

Example document:

> LangChain is a framework to build RAG applications. LangChain can also be used to build Agentic AI applications. Paris is the capital of France.

A size-based split can cut the text at arbitrary points. Sentences 1 and 2 are about the same topic (LangChain) but may land in different chunks, and an unrelated sentence (Paris) can be glued to one of them.

| Chunk | Content |
| --- | --- |
| 1 | LangChain is a framework to build RAG applications. |
| 2 | LangChain can also be used to build Agentic AI applications. |
| 3 | Paris is the capital of France. |

Better: put similar sentences together in one chunk, and put unrelated sentences in a different chunk.

## What is semantic chunking?

Splitting a document into meaningful units (chunks) based on semantic similarity, not just by number of tokens or lines.

Why it matters in RAG:

- Better chunks → better retrieval → better grounding → better answers.
- Chunks should be self-contained, contextually rich and logically separated.

## How it works

1. **Document segmentation**: split the document into sentences or paragraphs (for example with a regex on the full stop).
2. **Sentence embedding**: convert each sentence into a vector. Use OpenAI embedding models, or Hugging Face models (`sentence-transformers`).
3. **Semantic similarity check**: calculate the cosine similarity between *adjacent* sentences.
   - S1 ↔ S2 (for example 0.95, so similar)
   - S2 ↔ S3
   - S3 ↔ S4
4. **Merging of sentences**: merge adjacent sentences if they are semantically similar (similarity reaches the threshold).
5. **Form the chunks**: for example `[S1, S2]` → chunk 1, `[S3]` → chunk 2.

### Threshold

The threshold decides whether two adjacent sentences are merged. Typical values are 0.75, 0.8 or 0.95. There is no universal value. Decide it together with a domain expert and tune it on the data.

- The merge condition is `similarity >= threshold`.
- **Higher threshold** → stricter, fewer merges, smaller chunks.
- **Lower threshold** → looser, more merges, bigger chunks.

## Example

Input sentences:

1. LangChain is a framework for building LLM-powered apps.
2. It integrates with tools like OpenAI and Pinecone.
3. The Eiffel Tower is located in Paris.
4. France is a popular tourist destination.

Output after semantic chunking:

```
["LangChain is a framework...", "It integrates with tools..."]
["The Eiffel Tower is located in Paris."]
["France is a popular tourist destination."]
```

- 1 and 2 are both about LangChain, so they are merged.
- 3 and 4 are related (Paris is in France), but the similarity is not high enough to pass the threshold, so they stay separate.

## Semantic Chunking with Langchain

LangChain provides `SemanticChunker`, which does the 5 steps above for you: split the text into sentences, embed them, compare adjacent sentences with cosine distance, and start a new chunk where the meaning changes.

It is in the `langchain-experimental` package, not in `langchain-text-splitters` (which has no semantic splitter). That package was archived on May 26, 2026 (last release 0.4.2), so it works but gets no more updates. Pin the version in your project.

```bash
pip install langchain-experimental "langchain-huggingface[full]"
```

`langchain-huggingface[full]` includes `sentence-transformers`, which is needed to run the embedding model locally.

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_openai import OpenAIEmbeddings   # alternative, needs OPENAI_API_KEY

text = (
    "LangChain is a framework for building LLM-powered apps. "
    "It integrates with tools like OpenAI and Pinecone. "
    "The Eiffel Tower is located in Paris. "
    "France is a popular tourist destination."
)

# Step 2: the embedding model used to embed each sentence
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# embeddings = OpenAIEmbeddings()

# Steps 1, 3, 4, 5: segment, compare adjacent sentences, merge, form chunks
splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=40,   # 4 sentences = only 3 gaps, so use a low percentile to allow 2 splits
    buffer_size=0,                    # compare the sentences directly (default 1 mixes in the neighbours)
)

chunks = splitter.split_text(text)              # list[str]
docs = splitter.create_documents([text])        # list[Document], ready for a vector store

for i, chunk in enumerate(chunks, 1):
    print(f"Chunk {i}: {chunk}")
```

Output (real run with `langchain-experimental` 0.4.2, `langchain-huggingface` 1.2.3, `sentence-transformers` 5.7.0, Python 3.10):

```
Chunk 1: LangChain is a framework for building LLM-powered apps. It integrates with tools like OpenAI and Pinecone.
Chunk 2: The Eiffel Tower is located in Paris.
Chunk 3: France is a popular tourist destination.
```

The cosine distances between neighbouring sentences were S1→S2 0.55, S2→S3 1.03 and S3→S4 0.62. The smallest gap (sentences 1 and 2, both about LangChain) is merged, and the two bigger gaps are split. This is the same result as the lecture example.

The result depends on the settings, same text and same model:

- Default `breakpoint_threshold_amount` (95): 2 chunks, sentences 1 + 2 and sentences 3 + 4. Only the biggest gap is above the 95th percentile, so Paris and France stay together.
- Amount 40 but the default `buffer_size=1`: 3 chunks, but sentences 1 and 2 are split and 3 + 4 stay together. Each sentence is embedded with its neighbours, which blurs the gaps on a text this short.

So check the chunks on your own data and tune the threshold.

Use `splitter.split_documents(documents)` for documents from a LangChain loader. The metadata is copied to each chunk.

### How `split_text` decides where to split

It splits based on the meaning gap between neighbouring sentences, compared with the other gaps in the same text. It does not use character counts or a fixed similarity value.

1. **Sentences:** split the text with a regex, `(?<=[.?!])\s+` (after `.`, `?` or `!` followed by whitespace).
2. **Embeddings:** embed every sentence. With `buffer_size=0` each sentence is embedded alone. With the default of 1, each sentence is embedded together with its neighbours.
3. **Distances:** calculate the cosine distance (1 − similarity) between each sentence and the next one.
4. **Threshold:** turn `breakpoint_threshold_type` and `breakpoint_threshold_amount` into a distance cut-off. With `percentile` and 40, the cut-off is the 40th percentile of all the distances in this text.
5. **Breakpoints:** wherever a distance is above the cut-off, a chunk ends after that sentence.
6. **Chunks:** the sentences between the breakpoints are joined with spaces, and the leftover sentences at the end form the last chunk.

For the example above, the cut-off was about 0.61:

| Gap | Distance | Above the cut-off? |
| --- | --- | --- |
| S1 → S2 | 0.553 | no, so S1 and S2 stay together |
| S2 → S3 | 1.031 | yes, split after S2 |
| S3 → S4 | 0.622 | yes, split after S3 |

That gives `[S1 S2]`, `[S3]`, `[S4]`.

- It only compares and merges adjacent sentences, never sentences that are far apart.
- Because the cut-off is relative, it always splits at the biggest gaps in a document, even if all the sentences are very similar. The lecture's fixed threshold (for example 0.8) does not behave that way.

### Threshold

The lecture uses a fixed cosine **similarity** (for example 0.8): merge if similarity ≥ threshold. `SemanticChunker` works differently. It calculates the cosine **distance** (1 − similarity) between each pair of adjacent sentences, and starts a new chunk where the distance is unusually big compared to the rest of the document.

`breakpoint_threshold_type` decides what "unusually big" means:

| Type | Split where the distance is... | Default amount |
| --- | --- | --- |
| `percentile` (default) | above the Nth percentile of all distances | 95 |
| `standard_deviation` | above mean + N × standard deviation | 3 |
| `interquartile` | above mean + N × IQR (75th minus 25th percentile) | 1.5 |
| `gradient` | same as percentile, but applied to the gradient (rate of change) of the distances | 95 |

With `percentile`, a **higher** amount means fewer splits and bigger chunks, and a **lower** amount means more splits and smaller chunks. That is the opposite direction of the lecture's similarity threshold, where a higher value means fewer merges.

### Other useful parameters

- `number_of_chunks`: ask for about this many chunks, instead of setting `breakpoint_threshold_amount`.
- `min_chunk_size`: a group shorter than this is carried into the next chunk, so you do not get tiny chunks.
- `buffer_size` (default 1): each sentence is embedded together with its neighbours, which smooths the comparison. On a very short text it blurs the gaps, so the example uses 0 to compare the sentences directly, like step 3 of the lecture.
- `sentence_split_regex`: how the text is split into sentences (default: after `.`, `?` or `!` followed by whitespace).

### Sources

- [langchain-experimental repository (archived)](https://github.com/langchain-ai/langchain-experimental)
- [langchain-experimental on PyPI](https://pypi.org/project/langchain-experimental/)
- [SemanticChunker source code (`text_splitter.py`)](https://raw.githubusercontent.com/langchain-ai/langchain-experimental/main/libs/experimental/langchain_experimental/text_splitter.py)
- [LangChain text splitter integrations](https://docs.langchain.com/oss/python/integrations/splitters)

