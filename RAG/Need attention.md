# Need attention

Data Preparation Strategy, including:

- Clean and standardize text formats.
- Use version control for document updates.
- Create user feedback loops to continuously improve retrieval accuracy. Log all queries and responses for audit purposes.
- Develop role-based access controls for both retrieval and generation.
- Use content filters to prevent exposure of sensitive information.
- Create clear, well-communicated policies for handling personally identifiable information.
- Manage costs and resources, with alerts for spikes and overruns.
- Evaluate user sati sfaction and, for advanced teams, implement specific accuracy measures for generative AI.

Things for documentations:

1. Defination of Rag and what R and A and G stands for
2. What is the data preparation strategy: Things we ay have there: Clean and standardize text formats, Chunking strategy, added metadata to the documents or chunks, is there version control for document updates
3. Indexing Strategy: Chunking strategy, Embedding, Storing, what algorithm used for indexing..etc
4. Retrieval strategy: What is the vector search capabilities we have in this project, hybrid search what is it, what algorithm used for ordering chunks, is retrieved chunks get ranked and what is the process there, how metadata is used to narrow search context, is there query reformulation techniques used, is there feedback loops to continuously improve retrieval accuracy, ..etc
5. prompt engineering: do we have clear instructions for source citation, did we Specify the expected format and level of detail in responses.
6. governance: what happens to outdated documents, audit logs in place, duplication in case of reingesting
7. Ensure security and compliance: are we using content filters to prevent exposure of sensitive information, do we have policies for handling personally identifiable information. ..etc

in case the creating projecting any of these points you can add section for the gaps
