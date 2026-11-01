# RLHF

**RLHF (Reinforcement Learning from Human Feedback)** is the training stage that turns a raw next-token predictor into a usable assistant.

**Why it's needed**  
After pretraining, a model can predict text well but isn't helpful. Ask it a question and it might continue with more questions, because that's a plausible continuation of a document. It has no notion that you want an answer. Pretraining optimized for *what's likely*, not *what's useful*.

**How it works**  
Three stages:  
**1. Supervised fine-tuning.** Humans write example conversations showing the desired behavior — question in, good answer out. The model trains on these. This alone gets you something that responds like an assistant.  
**2. Train a reward model.** The model generates several answers to the same prompt. Human raters rank them best to worst. Those rankings train a separate model — the reward model — to predict which answers humans would prefer. The point is scale: you can't have humans rate millions of outputs, but a reward model can.  
**3. Reinforcement learning.** The main model generates answers, the reward model scores them, and the model's weights are adjusted toward higher-scoring behavior. This runs for many iterations.

**What it does and doesn't do**  
It shapes **behavior**: tone, format, following instructions, refusing harmful requests, answering rather than continuing. It does **not** add knowledge — facts come from pretraining. It's why ChatGPT felt different from GPT-3 despite similar underlying capability.  
Known downsides: it can produce **sycophancy** (agreeing with you because agreement is rated highly) and can encourage overconfidence, since raters tend to prefer assured answers over hedged ones. That's a plausible contributor to hallucination.

**What's changed**  
RLHF is now a general label for this whole family rather than one specific method. **DPO** (Direct Preference Optimization) achieves similar results from preference data without training a separate reward model, and is widely used because it's simpler. **RLAIF** and **Constitutional AI** replace much of the human rating with AI feedback guided by written principles, which scales better.

Post-training is the umbrella term for everything done after pretraining — supervised fine-tuning, RLHF, and newer variants like DPO and Constitutional AI.

**Pretraining** — next-token prediction on enormous amounts of text. Produces a model with knowledge and language ability but no assistant behavior. This is the expensive part, months of compute.

**Post-training** — everything after: supervised fine-tuning, reward modeling, reinforcement learning, plus anything else layered on like safety tuning or reasoning training. Comparatively cheap.
