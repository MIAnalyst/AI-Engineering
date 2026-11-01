# RAG data pipeline steps

Another consideration in RAG systems is keeping data up to date. For example, if the original data is a Word document before being embedded into a vector store, the data could quickly lose value if not regularly updated.

Also, documents that cannot be directly embedded, such as PDF files, need to be converted into a text-friendly format to push them into the vector store.  
Therefore, it is crucial to [build a data pipeline](https://www.techtarget.com/searchenterpriseai/tip/Tools-and-techniques-for-optimizing-AI-data-pipelines) that can automate the process of continuously embedding the vector store with new data from different sources and file types.

A RAG data pipeline usually consists of the following six steps:

### 1. Corpus selection and ingestion

Choose the most relevant data sources and content depending on the use case.

### 2. Data preprocessing and parsing

Clean and standardize the raw data so it's ready for embedding and retrieval. A commonly used tool is [MarkItDown](https://github.com/microsoft/markitdown). It supports PDF, PowerPoint and many other file formats and can convert the content to Markdown format, which makes it easier to structure when chunking.

### 3. Enrichment

Remove noisy content and add helpful metadata to the data, such as adding the document title on top so that it will be available during the chunking process.

### 4. Filtering

Remove any irrelevant or low-value documents that don't support the use case.

### 5. Chunking

Split the cleaned data into smaller, logical chunks for better retrieval performance. Frameworks such as LangChain have different chunking methods, such as LangChain CharacterTextSplitter.

### 6. Embedding

Convert each chunk into a vector and store it in a vector store.  
Teams can automate much of this process with cloud services and frameworks. For instance, Azure AI Search, a Microsoft cloud-based vector store and search service, provides a schedule feature that automatically pulls data from a storage account, embeds it on a predefined schedule and puts it into Azure AI Search.

[https://www.techtarget.com/searchenterpriseai/tip/How-to-prepare-data-for-your-RAG-pipeline](https://www.techtarget.com/searchenterpriseai/tip/How-to-prepare-data-for-your-RAG-pipeline)
