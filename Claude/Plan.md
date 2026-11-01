# Plan

I want to evaluate the generation quality as opposing evaluating the retrieval quality which I can't use because the evaluation of chatbot designed by Microsoft Copilot Studio doesn't provide context (chunks) rather only responses.

I have only reference free questions (without answer) and 69 reference based questions. I have also 10 stress testing questions labeled as out-of-scope questions .

I will use Ragas metrics I want to use:

- End-to-End Metrics because Microsoft Copilot Studio doesn't provide context (chunks) rather only responses.
- LLM-based metrics: which uses LLM for evaluation and these non-deterministic. These will be used for all questions.
- Non-LLM-based metrics: These metrics do not use LLM underneath to do the evaluation. These metrics are deterministic and can be used to evaluate the performance of the AI application without using LLM.
- Single turn metrics: These metrics evaluate the performance of the AI application based on a single turn of interaction between the user and the AI. The two chatbots I build using Rag and copilot studio has no memory
- Let's use discrete metric with classifying pass/fail for questions
    - For in scope questions that starts with "True or false" where the answer either start with true or false
    - For out of scope questions, which include reference free questions and reference based questions, the answer expected to be refusal, request of clarification, or no available information.
- Let's use numeric metrics with range (0 to 1) for questions that reference based, except true and false questions.

Choosing the Right Metrics:

- Ensure Interpretability for the entire team to interpret and reason about.
- Emphasize Objective Over Subjective Metrics: Prioritize metrics with objective criteria, minimizing subjective judgment.
- Few Strong Signals over Many Weak Signals: Avoid a proliferation of metrics that provide weak signals and impede clear decision-making. Instead, select fewer metrics offering strong, reliable signals.

Sources:  
[https://www.evidentlyai.com/llm-guide/rag-evaluation#session-level-evaluation](https://www.evidentlyai.com/llm-guide/rag-evaluation#session-level-evaluation)  
[https://docs.ragas.io/en/latest/concepts/metrics/overview/#types-of-metrics-in-ragas](https://docs.ragas.io/en/latest/concepts/metrics/overview/#types-of-metrics-in-ragas)
