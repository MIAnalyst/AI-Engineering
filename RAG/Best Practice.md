# Best Practice

To get the most out of RAG, adopt these six best practices.

## 1. Create a data preparation strategy

Before thinking about retrieval, you need a data strategy. Focus first on surfacing your most valuable sources, such as knowledge bases, reports, customer call transcripts and even overlooked internal wikis. Then, build a repeatable [data preparation pipeline](https://web.archive.org/web/20260309084554/https:/www.techtarget.com/searchenterpriseai/tip/How-to-prepare-data-for-your-RAG-pipeline):

### Filter out irrelevant or outdated information.

### Break documents into appropriately sized segments.

Chunking

### Clean and standardize text formats.

- removing noisy content :"noisy content" refers to any irrelevant, redundant, or misleading text within a document that does not contribute to its core meaning and can confuse the retrieval or generation process.
- "Standardizing text formats" means converting the diverse range of source document formats (like PDFs, Word docs, HTML pages) into a uniform, structured, and clean text format that is easy for the chunking and embedding steps to process.
    - Preserves structure: It keeps basic formatting like headings (#), bold, lists, and tables in a plain-text, lightweight way.
    - Aids retrieval: An embedding model can understand that text under a # Heading is a title, which can be useful context. Tools like MarkItDown (mentioned in your first article) or pandoc are often used for this.
    - Improves readability for the LLM: When a chunk of Markdown is passed to an LLM, the formatting hints can help it better understand the structure and relationships within the text (e.g., knowing that a sentence is a list item).

Converting to Markdown is a fantastic modern approach because it:

### Extract and preserve metadata.

This means capturing important information about the document or chunk and storing it alongside the vector and original text in your vector database. This metadata acts as additional fields you can filter on or use to provide more context.

in the vector store example from the first article, you can see this in action:  
`json`  
{￼ "id": "unique-id-123",￼ "values": \[0.81, -0.23, ..., 0.789\], // the vector (embedding)￼ "metadata": { // \<-- THIS IS THE EXTRACTED AND PRESERVED METADATA￼ "text": "We launched Copilot last week and adoption has been stellar….",￼ "source": "Document A",￼ "page": 3,￼ "author": "Marius Sandbu",￼ "date_published": "2025-08-27"￼ }￼}  
By preserving metadata like source, page, author, or date, you can:

- Filter searches: Only retrieve chunks from "Document A" or documents published after a certain date.
- Provide citations: Tell the user exactly where the information came from (e.g., "Source: Document A, Page 3").
- Improve retrieval: Some retrieval mechanisms can use metadata to boost the relevance score of certain documents.

### Use version control for document updates.

This means treating your knowledge base documents with the same rigor that software engineers treat their code. It involves tracking changes to documents over time and managing different versions.  
In a RAG context, this is critical because if a source document is updated (e.g., a product manual is revised), the old information in your vector store becomes outdated. Version control helps manage this by:

- Tracking Change History: Knowing when a document was changed and what was altered.
- Enabling Updates/Re-indexing: When a new version of a document is available, your automated pipeline (mentioned in your provided text) can detect this, fetch the new version, re-process it (clean, chunk, embed), and update or replace the old vectors in the database. This ensures your RAG system always has the latest information.
- Maintaining Auditability: If a question is asked about an older version of a policy, you could theoretically retrieve the correct historical context if the versions are managed.

A data preparation strategy is not a one-time effort but an ongoing process. You also need automated workflows to continuously update knowledge bases as new information becomes available.

## 2. Choose a vector database

[Vector embeddings](https://web.archive.org/web/20260309084554/https:/www.techtarget.com/searchenterpriseai/definition/vector-embeddings) are essential for retrieval. When processing a document, the LLM doesn't just store the raw text for keyword matching. Instead, it passes the text through an embedding model: a [neural network](https://web.archive.org/web/20260309084554/https:/www.techtarget.com/searchenterpriseai/definition/neural-network) trained to translate language into arrays of numbers called high-dimensional vectors.

Think of two sentences that mean similar things but use different words. Vector embeddings map these to nearby points in space. For example, "How can I reset my password?" and "I forgot my login credentials. How do I recover access?" are not similar in their keywords. But they are close together in embedding space because their meanings overlap.

A [vector database](https://web.archive.org/web/20260309084554/https:/www.computerweekly.com/feature/Storage-technology-explained-Vector-databases-at-the-core-of-AI) stores document embeddings and thus enables similarity searches for related meanings -- an essential aspect of RAG. [Popular options include](https://web.archive.org/web/20260309084554/https:/www.techtarget.com/searchdatamanagement/tip/Top-vector-database-options-for-similarity-searches) Pinecone, Weaviate, Milvus and Qdrant, though many organizations also use vector capabilities in their existing database platforms.

When selecting a vector database, consider the following:

- Scalability for when successful projects result in higher usage.
- Query performance for expected load.
- Security features, including encryption and access controls.
- Integration capabilities with existing data ecosystems and architecture.
- Operational requirements and maintenance overhead.

### Azure AI Search: A Search Engine, Not Just a Vector DB

Think of the dedicated vector databases (Pinecone, Weaviate, Milvus, Qdrant) as specialized tools built for one primary purpose: ultra-fast similarity search on vector embeddings. They are laser-focused on that single task.

Azure AI Search, however, is more like a multi-tool. It's a full-featured, cloud-based search service that has added a powerful "vector search" blade to its existing toolkit. Its key strength is that it allows you to store vectors and traditional textual data side-by-side in the same index . This enables hybrid search, where you can combine the semantic understanding of vectors with the precision of keyword matching (using BM25)

## 3. Develop a retrieval strategy

Effective retrieval is crucial for RAG performance, and many projects stumble here. The goal is to retrieve relevant information, but too much data can [create noise](https://web.archive.org/web/20260309084554/https:/www.techtarget.com/searchbusinessanalytics/definition/noisy-data). The information returned should be comprehensive enough to answer user queries yet concise enough for the LLM to process effectively.

Consider the following retrieval best practices:

### Combine keyword and semantic similarity approaches with hybrid search.

### Rerank retrieved documents based on relevance: Not all "similar" documents are useful.

### Filter metadata to narrow search context.

### Use query reformulation techniques to improve search effectiveness.

This involves modifying or expanding a user's original query before searching the vector database to improve the chances of retrieving relevant information. User queries are often short, ambiguous, or not phrased in a way that matches the language in your documents.

The core LLM itself is very capable of reformulating queries if asked to. However, in a standard RAG flow, the user's query is often passed directly to the retrieval step without the LLM's involvement. The LLM only sees the query after the retrieval has happened, when it's time to generate the final answer.

For query reformulation to happen, your orchestration code needs to be configured to call the LLM before the retrieval step, give it a specific instruction (a prompt) to reformulate the query, and then use that new query for the search.

### Create user feedback loops to continuously improve retrieval accuracy.

### With their unpredictable queries, real-world users often find issues in RAG that test teams overlook.

This is a mechanism to capture how users interact with the RAG system's responses and use that data to systematically improve retrieval accuracy. Real-world user behavior is often unpredictable and reveals edge cases that internal testing misses.

Recording the queries and the generated answers is creating the raw data for your feedback loop. However, a "loop" implies that this data is then used to take action and improve the system. Simply storing it is just logging. The loop is closed when you analyze that data and make changes.

## 4. Ensure security and compliance

Security safeguards help ensure that a RAG system doesn't expose information or violate privacy regulations.  
[Security best practices](https://web.archive.org/web/20260309084554/https:/www.techtarget.com/searchenterpriseai/tip/Address-top-AI-privacy-concerns-with-this-checklist) include the following:

- Develop role-based access controls for both retrieval and generation.
- Track data lineage to identify data sources.
- Log all queries and responses for audit purposes.
- Use content filters to prevent exposure of sensitive information.
- Create clear, well-communicated policies for handling personally identifiable information.

## 5. Optimize prompt engineering

Don't treat [prompt engineering](https://web.archive.org/web/20260309084554/https:/www.techtarget.com/searchenterpriseai/tip/Prompt-engineering-tips-and-best-practices) like an end-user issue. Enterprises should define templates, citations and formats in advance. Effective steps include the following:

- Create standardized prompt templates for different use cases.
- Include clear instructions for source citation.
- Specify the expected format and level of detail in responses.
- Provide context about the user's role and access permissions.
- Test prompts systematically and iteratively.
- Well-designed prompts help the LLM properly contextualize retrieved information and generate appropriate responses.

## 6. Govern RAG architecture rigorously

A RAG system's knowledge base can [drift over time](https://web.archive.org/web/20260309084554/https:/www.techtarget.com/searchenterpriseai/tip/How-to-identify-and-manage-AI-model-drift) because information becomes outdated, contradicts new data or is [biased](https://web.archive.org/web/20260309084554/https:/www.techtarget.com/searchenterpriseai/definition/machine-learning-bias-algorithm-bias-or-AI-bias) due to selective usage.

Governance mechanisms help organizations maintain control over their RAG systems and continuously improve performance. An effective governance process includes the following:

- Monitor usage patterns and performance metrics with dashboards.
- Track the retrieved sources and keep an audit log of which sources are used.
- Evaluate user satisfaction and, for advanced teams, implement specific accuracy measures for generative AI.
- Identify and [address potential biases](https://web.archive.org/web/20260309084554/https:/www.techtarget.com/searchenterpriseai/feature/6-ways-to-reduce-different-types-of-bias-in-machine-learning) in responses.
- Manage costs and resources, with alerts for spikes and overruns.

Source: [https://www.removepaywall.com/search?url=https://www.techtarget.com/searchenterpriseai/tip/RAG-best-practices-for-enterprise-AI-teams](https://www.removepaywall.com/search?url=https://www.techtarget.com/searchenterpriseai/tip/RAG-best-practices-for-enterprise-AI-teams)  
[https://learn.microsoft.com/en-us/azure/search/search-indexer-overview](https://learn.microsoft.com/en-us/azure/search/search-indexer-overview)
