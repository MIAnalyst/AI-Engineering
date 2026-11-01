# Read

[https://claudemagazine.com/claude-system/understanding-claudes-context-window-memory-continuity-and-long-form-intelligence/](https://claudemagazine.com/claude-system/understanding-claudes-context-window-memory-continuity-and-long-form-intelligence/)

Notes:  
When you're chatting with me, I don't actually "remember" what you said earlier the way a person would. Instead, every time you send a new message, the entire conversation so far — your messages and my replies — gets fed back to me as input. I then read through all of it and respond.  
So it *feels* like I remember, but really I'm just re-reading the conversation each time.  
The key implication: there's a limit to how much text I can process at once (called the "context window"). If a conversation gets very long and older messages get trimmed or summarized to fit, I lose access to whatever was cut. I can only work with what's actually in front of me at that moment.  
And between separate chats, I start completely fresh — nothing carries over unless you (or the system) explicitly bring it in.

But long-form intelligence has conditions. When the prompt is messy, the model’s intelligence is spent on navigation. When the prompt is structured, the model’s intelligence is spent on synthesis. That difference is not subtle. It is the difference between a model that sounds smart and a model that stays correct.

[https://docs.ragas.io/en/latest/concepts/metrics/overview/#types-of-metrics-in-ragas](https://docs.ragas.io/en/latest/concepts/metrics/overview/#types-of-metrics-in-ragas)  
**Metrics can be classified into two categories based on the mechanism used underneath the hood**:

- **LLM-based metrics** : These metrics use LLM underneath to do the evaluation. There might be one or more LLM calls that are performed to arrive at the score or result. These metrics can be somewhat non-deterministic as the LLM might not always return the same result for the same input. On the other hand, these metrics has shown to be more accurate and closer to human evaluation.
- **Non-LLM-based metrics** : These metrics do not use LLM underneath to do the evaluation. These metrics are deterministic and can be used to evaluate the performance of the AI application without using LLM. These metrics rely on traditional methods to evaluate the performance of the AI application, such as string similarity, BLEU score, etc. Due to the same, these metrics are known to have a lower correlation with human evaluation.
