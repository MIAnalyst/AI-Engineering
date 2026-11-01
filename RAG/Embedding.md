# Embedding

## How RAG makes data searchable

RAG helps generative AI services access external data by using a vector store or database with some search mechanisms.

First, to add unstructured data -- such as a document -- into a vector store, teams need to embed the data through an embedding model.

An embedding converts text into [numerical vectors](https://www.techtarget.com/searchenterpriseai/definition/vector-embeddings) -- a list of decimals that represent text, such as words or sentences, in a way a computer can understand.  
For example, consider the following paragraph from a document:

We launched Copilot last week, and adoption has been stellar. Early metrics show a 32% drop in ticket volume and a 27% jump in self-service resolution within the first 48 hours. Finance estimates an annual savings of $1.4 million if the trend holds.

Embedding this paragraph into vectors that represent the text looks like this:  
\[0.81, -0.23, 0.45, 0.12, -0.67, 0.94, 0.30, -0.55, 0.76, 0.10\]

This is a highly simplified example; for instance, if teams used an embedding model such as OpenAI's text-embedding-3-small, it would output a 1536-dimensional vector. Multiple embedding models are available from different providers. Hugging Face has an embedding [leaderboard](https://huggingface.co/spaces/mteb/leaderboard) that shows the strengths and weaknesses of different embedding models.

These vectors and the original text are stored in a vector store such as Pinecone or Weaviate. The data in the vector store could look something like this:  
{  
"id": "unique-id-123",  
"values": \[0.81, -0.23, ..., 0.789\], // the vector (embedding)  
"metadata": {  
"text": "*We launched Copilot last week and adoption has been stellar…*.",  
"source": "Document A",  
"page": 3  
}  
}

When a user wants to ask a question related to the data, it sends the prompt through an orchestrator -- such as LangGraph, [LangChain or Semantic Kernel](https://www.techtarget.com/searchenterpriseai/tip/Compare-Semantic-Kernel-vs-LangChain-for-AI-development) -- that then sends a request to the vector store using a search method. The search method could return two or more chunks of relevant content and provide this information to the large language model ([LLM](https://www.techtarget.com/whatis/definition/large-language-model-LLM)) that generates an answer to the user.

[https://www.techtarget.com/searchenterpriseai/tip/How-to-prepare-data-for-your-RAG-pipeline](https://www.techtarget.com/searchenterpriseai/tip/How-to-prepare-data-for-your-RAG-pipeline)

Leaderboard: Hugging Face has an embedding leaderboard that shows the strengths and weaknesses of different embedding models.  
[https://huggingface.co/spaces/mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
