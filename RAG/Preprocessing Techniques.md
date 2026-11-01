# Preprocessing Techniques

## The need for data chunking

Chunking is a standard preprocessing method that turns a data set into smaller sections, called chunks. Since vector stores cannot store all the original data, it must be split into chunks.

For instance, the Copilot paragraph could be chunked into two sections:  
Chunk 1:  
We launched Copilot last week, and adoption has been stellar.  
Chunk 2:  
Early metrics show a 32% drop in ticket volume and a 27% jump in self-service resolution within the first 48 hours. Finance estimates an annual savings of $1.4 million if the trend holds.

The problem with this chunking approach is that some chunks might lose their original meaning. For example, if a user asked about Copilot in the RAG system, the search mechanism would only return the first chunk, not the second one, since there is no correlation between the two. The user doesn't get the added context and information about Copilot adoption without the second chunk.

This is where chunking strategies can help chunk data to fit into the vector store and retain its original meaning in a RAG engine.

## Chunking strategies

Without proper strategies, RAG systems can chunk data incorrectly. Different chunking methods are available, including fixed-size chunking, variable-size chunking based on content, rule-based chunking and sliding window chunking.

### 1. Fixed-size chunking

This method involves dividing text into equal-sized segments, such as 400 words or 800 characters per chunk, regardless of the content.  
**2. Variable-size chunking based on content**  
Here, the text is split according to natural boundaries, such as sentence-ending punctuation, line breaks or structural cues, identified by natural language processing ([NLP](https://www.techtarget.com/searchenterpriseai/definition/natural-language-processing-NLP)) tools that analyze document layout and meaning.

### 3. Rule-based chunking

This approach relies on the document's intrinsic structure or linguistic boundaries. Typical methods include chunking by sentences or paragraphs using predefined rules.

### 4. Sliding window chunking

This technique produces overlapping chunks by sliding a fixed-size window through the text with a set size. For example, each chunk might contain 500 words, with the next chunk starting 300 words in to create a 200-word overlap between chunks.

For the Copilot paragraph, a sliding window or rule-based approach would ensure that the paragraph would not lose its meaning since both pieces of content would be available in the same chunk.

[https://www.techtarget.com/searchenterpriseai/tip/How-to-prepare-data-for-your-RAG-pipeline](https://www.techtarget.com/searchenterpriseai/tip/How-to-prepare-data-for-your-RAG-pipeline)

## Data preprocessing and parsing

Clean and standardize the raw data so it's ready for embedding and retrieval. A commonly used tool is [MarkItDown](https://github.com/microsoft/markitdown). It supports PDF, PowerPoint and many other file formats and can convert the content to Markdown format, which makes it easier to structure when chunking.

## Semantic Chunking

Semantic chunking is the process of splitting a document into meaningful units (chunks) based on semantic  
similarity - not just by number of tokens or lines.

This is important in RAG systems because:

- Better chunks - better retrieval - better grounding - better answers.
- Chunks should be self-contained, contextually rich, and logically separated.

How does semantic chunking works, steps are:

1. Document Segmentation: Split the document into sentences and paragraph
2. Sentence Embedding: each sentence is converted into vector representation
3. Semantic similarity check: calculate cosine similarity between adjacent sentences
4. Merging of sentences based on similarity, threshold should be defined
5. Form the chunks
