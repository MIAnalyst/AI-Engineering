# Reasoning

What are reasoning LLMs?

Compared to regular LLMs, reasoning LLMs tend to break down a problem into smaller steps (often called reasoning steps or thought processes) before answering a given question.  
!["Regular" LLMs "Reasoning" LLMs Question Question Large…](assets/reasoning-1.jpg)

So what does a “thought process”, “reasoning step”, or “Chain-of-Thought” actually mean?

Although we can philosophize whether LLMs are actually able to think like humans[1](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-reasoning-llms#footnote-1), these reasoning steps break down the process into smaller, structured inferences.  
[![Question I have 10 apples. I gave 2 apples away. I ate 1.…](assets/reasoning-2.png)](https://substackcdn.com/image/fetch/$s_!gy6f!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F302aecd1-eeec-45f6-8f21-97d9791ef67d_1284x960.png)

In other words, instead of having LLMs learn “what” to answer they learn “how” to answer!  
To understand how reasoning LLMs are created, we first explore the paradigm shift from a focus on scaling training (**train**-time compute) to inference (**test**-time compute).

**What is train-time compute?**  
Until half of 2024, to increase the performance of LLMs during **pre-training**, developers often increase the **size** of the:

- Model (# of **parameters** )
- Dataset (# of **tokens** )
- Compute (# of **FLOPs** )

Combined, this is called train-time compute, which refers to the idea that pretraining data is the “fossil fuel of AI”. Essentially, the larger your pretraining budget, the better the resulting model will be.  
![Input Train-time compute The llama is a South American…](assets/reasoning-3.png)

Note: Parameters are the learned weights and biases stored in the model. They are the numbers updated during training. In a Transformer LLM, parameters include:

- Token embedding matrix
- Attention weights
- FFN weights:
- LayerNorm / RMSNorm scales and biases
- Final LM head / unembedding weights,

The # parameters is the total count of all those learned numbers.

Activations / hidden states or contextual representation: token embedding+ attention info+ FFN info. It's not counted as parameters.

Note end

Train-time compute may include both the compute needed during training as well as during fine-tuning.  
![Pre-training Post-training base model fine-tuned The llama…](assets/reasoning-4.jpg)  
Together, they have been a main focus on increasing the performance of LLMs.

**Scaling Laws**  
How a model’s scale (through compute, dataset size, and model size) correlates with a model’s performance is researched through various scaling laws.  
They are so-called “power laws” where an increase in one variable (e.g., compute) results in a proportional change in another variable (e.g., performance).  
[![Normal scale Log-log scale "performance" "performance".…](assets/reasoning-5.png)](https://substackcdn.com/image/fetch/$s_!at_j!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff028438e-4a6c-4398-87c6-5eacb617ad2c_1452x592.png)  
These are typically shown in a log-log scale (which results in a straight line) to showcase the large increase in compute.  
Most well-known are the “Kaplan”[2](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-reasoning-llms#footnote-2) and “Chinchilla”[3](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-reasoning-llms#footnote-3) scaling laws. These laws more and less say that the performance of a model will increase with more compute, tokens, and parameters.  
[![4.2 6 -L= (D/5.4 · 1013)-0.095 5.6 - L= (N/8.8 ·…](assets/reasoning-6.jpg)](https://substackcdn.com/image/fetch/$s_!LUqb!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc2a70247-3a21-4048-adb8-2ed2f9c15b01_1384x508.png)  
Annotated figure of the “[Scaling laws for neural language models](https://arxiv.org/abs/2001.08361)” paper. It shows how performance may increase with different measures of compute (longer training, dataset size, and parameter size).  
They suggest that all three factors must be scaled up in tandem for optimal performance.  
Kaplan’s Scaling Law states that scaling the model’s size is typically more effective than scaling data (given fixed compute). In contrast, the Chinchilla Scaling Law suggests that the model size and data are equally important.  
However, throughout 2024, compute, dataset size and model parameters have steadily grown, yet the gains have shown diminishing returns.  
[![Normal scale "performance" (linear scale) Are we here?…](assets/reasoning-7.png)](https://substackcdn.com/image/fetch/$s_!eJAu!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fdd84d837-0b4f-4e91-9a84-d07976a72cfe_1452x592.png)  
As is with these power laws, there are diminishing returns as you scale up.  
This raises the question:  
*“Have we reached a wall?”*  
**What is test-time Compute?**  
The expensive nature of increasing train-time compute led to an interest in an alternative focus, namely **test-time compute**. --> It means that scaling train-time computehad become prohibitively expensive and was showing diminishing returns, so researchers looked for a different axisto improve performance.  
Instead of continuously increasing pre-training budgets, test-time compute allows modes to “*think longer*” during **inference**.  
[![Question "Reasoning" Large Language Model Test-time compute…](assets/reasoning-8.png)](https://substackcdn.com/image/fetch/$s_!ruJU!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fbdc25691-a91b-4ab3-bd76-554b4233452e_1232x748.png)  
With non-reasoning models, it would normally only output the answer and skip any “reasoning” steps:  
[![Input What is 3 + 2? Tokens What is 3 + 2 ? Large Language…](assets/reasoning-9.png)](https://substackcdn.com/image/fetch/$s_!Qk5L!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F833c7a1e-ce3c-4ff1-8796-97ca5cb67943_1384x776.png)  
Reasoning models, however, would instead use more tokens to derive their answer through a systematic “**thinking**” process:  
[!['Reasoning" Large Language Model 1 token (compute) 6 tokens…](assets/reasoning-10.png)](https://substackcdn.com/image/fetch/$s_!IZYZ!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F6392f3a1-4b9f-4eae-9e33-3b03d9a62dd6_1384x968.png)  
The idea is that the LLM has to spend resources (like VRAM compute) to create an answer. However, if all compute is spent generating the answer, then that is a bit inefficient!  
Instead, by creating more tokens beforehand that contain additional information, relationships, and new thoughts, the model spent more compute generating the final answer.  
More elaboration: An LLM generates one token per forward pass. If it must produce the final answer immediately, it has only a few forward passes to compute it. By generating intermediate reasoning tokens first, the model uses more forward passes — more test-time compute — and writes intermediate results into the context. Later tokens can attend to those results, effectively giving the model more sequential computation and working memory before it outputs the final answer.  
Reasoning adds inference cost. It is not free. It’s extra “test-time compute.” direct-answer LLM generates only the final answer tokens. A reasoning LLM generates:

- intermediate reasoning tokens, then
- the final answer tokens.

So it generates more tokens.

[![Answering # tokens 1 token (compute) 5 More tokens 6 tokens…](assets/reasoning-11.png)](https://substackcdn.com/image/fetch/$s_!XEhG!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F0437cf25-8f58-47d4-b200-92f30829f3ab_1384x744.png)

**Scaling Laws**  
Compared to train-time compute, scaling laws for test-time compute are relatively new. Of note are two interesting sources that relate test-time compute scaling to train-time compute.  
First, is a [post by OpenAI](https://openai.com/index/learning-to-reason-with-llms/) showcasing that test-time compute might actually follow the same trend as scaling train-time compute.  
[![01 AIME accuracy 01 AIME accuracy 100 during training 100…](assets/reasoning-12.jpg)](https://substackcdn.com/image/fetch/$s_!Udr1!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff56b1b36-8e90-48cd-bc50-ab4d10522673_1384x884.png)  
Annotated figure from “[Learning to reason with LLMs](https://openai.com/index/learning-to-reason-with-llms/)”. The red dotted line was added to demonstrate how OpenAI suggests the new paradigm might be test-time compute.  
As such, they claim that there might be a paradigm shift to scaling test-time compute as it is still a new field.  
Second, an interesting paper called “Scaling Scaling Laws with Board Games”[4](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-reasoning-llms#footnote-4), explores **AlphaZero** and trains it to various degrees of compute to play Hex.  
[![AlphaZero Train-time compute · More parameters · Longer…](assets/reasoning-13.jpg)](https://substackcdn.com/image/fetch/$s_!U2qf!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff4eb1eaa-b2da-4b70-ad3d-05d7f9e0d03c_2336x1220.png)  
Annotated figure from “Scaling Scaling Laws with Board Games”. It shows how they constructed varying degrees of **train**-time compute and **test**-time compute.  
Their results show that train-time compute and test-time compute are tightly related. Each dotted line showcases the minimum compute needed for that particular ELO score.  
![Test-time compute log10(test) = - 1.2 . log10(train) +…](assets/reasoning-14.png)

Annotated figure from “[Scaling Scaling Laws with Board Games](https://arxiv.org/pdf/2104.03113)”. It demonstrates the relationship between **train**-time compute and **test**-time compute.  
With test-time compute scaling like train-time compute, a paradigm shift is happening toward “reasoning” models using more test-time compute.  
Through this paradigm shift, instead of focusing purely on train-time compute (pre-training and fine-tuning), these “reasoning” models instead balance training with inference.  
[![Train-time compute Inference "Regular" LLMs Pre-training…](assets/reasoning-15.png)](https://substackcdn.com/image/fetch/$s_!QyVo!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F2fe913b8-2210-4156-9bf6-ac1f489b0cec_1500x436.png)  
Test-time compute could even scale in length:  
[![Question Reasoning" Large Language Model Test-time compute…](assets/reasoning-16.jpg)](https://substackcdn.com/image/fetch/$s_!8AS-!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F574b4416-1903-4d19-8669-875f8ab45f92_1500x624.png)  
Scaling in length is something we will also explore when diving into DeepSeek-R1!  
**Categories of Test-time Compute**  
The incredible success of reasoning models like DeepSeek R-1 and OpenAI o1 showcase that there are more techniques than simply thinking “longer”.  
As we will explore, test-time compute can be many different things, including Chain-of-Thought, revising answers, backtracking, sampling, and more.  
These can be roughly put into two categories[5](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-reasoning-llms#footnote-5):

- Search against Verifiers (sampling generations and selecting the best answer)
- Modifying Proposal Distribution (trained “thinking” process)

[![Search against Modifying Proposal Verifiers Distribution…](assets/reasoning-17.jpg)](https://substackcdn.com/image/fetch/$s_!7bma!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fcbec64b2-53a6-4259-8a6c-6446281b4ee2_1668x1132.png)  
Thus, search against verifiers is *output*-focused whereas modifying the proposal distribution is *input*-focused.  
[![inference (output level) training (input level) RM NVIDIA…](assets/reasoning-18.jpg)](https://substackcdn.com/image/fetch/$s_!5L1V!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa8afbaee-4d2d-4f5f-88dd-20ae58cdf806_1744x560.png)  
There are two types of verifiers that we will explore:

- Outcome Reward Models (ORM)
- Process Reward Models (PRM)

As their names imply, the ORM only judges the outcome and doesn’t care about the underlying process:  
[![Question Large Language Model Thought process 1 Reasoning…](assets/reasoning-19.png)](https://substackcdn.com/image/fetch/$s_!qcFh!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F036ff9da-db9f-4a01-8fc7-022bbc83dbcb_1156x808.png)  
In contrast, the PRM also judges the process that leads to the outcome (the “reasoning”):  
[![Question Large Language Model Thought process 1 PRM 0.2…](assets/reasoning-20.png)](https://substackcdn.com/image/fetch/$s_!8ofg!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fbd3465b7-e01e-4c9f-835f-819f8a30214b_1156x732.png)  
To make these reasoning steps a bit more explicit:  
[![Question I have 10 apples. I gave 2 apples away. I then…](assets/reasoning-21.png)](https://substackcdn.com/image/fetch/$s_!aUd7!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F71760532-53a5-4953-8305-f0182954023c_1440x640.png)  
Note how step 2 is a poor reasoning step and is scored low by the PRM!  
Now that you have a good grasp on ORMs vs. PRMs, let us explore how they can applied in various verification techniques!  
**Search against Verifiers**  
The first major category of test-time compute is to search against verifiers. This generally involves two steps.

- First, multiple samples of reasoning processes and answers are created.
- Second, a verifier (Reward Model) scores the generated output

[![Question Large Language Model thought thought thought…](assets/reasoning-22.png)](https://substackcdn.com/image/fetch/$s_!k50w!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F822bfcc6-a0ab-4b90-b243-4d2f45195edb_1076x884.png)  
The verifier is typically a LLM, fine-tuned for either judging the outcome (ORM) or the process (PRM).  
A major advantage of using verifiers is that there is no need to re-train or fine-tune the LLM that you use for answering the question.  
**Majority Voting**  
The most straightforward method is actually not to use a reward model or verifier but instead, perform a majority vote.  
We let the model generate multiple answers and the answer that is generated most often will be the final answer.  
[![What is 1 + 1? Large Language Model Thoughts Thoughts…](assets/reasoning-23.jpg)](https://substackcdn.com/image/fetch/$s_!oNhu!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F93f66577-0287-421f-b3e8-6bc2e5a47069_1480x744.png)

This method is also called *self-consistency*[6](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-reasoning-llms#footnote-6)as a way to emphasize the need for generating multiple answers and reasoning steps.

**Best-of-N samples**  
The first method that involves a verifier is called Best-of-N samples. This technique generates N samples and then uses a verifier (Outcome Reward Model) to judge each answer:  
First, the LLM (often called the Proposer) generates multiple answers using either a high or varying temperature.  
[![Question Large Language Model Generate n answers (each with…](assets/reasoning-24.jpg)](https://substackcdn.com/image/fetch/$s_!v7Qt!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F6f674bc6-a20f-4e2c-9df1-ef288c169bcd_1776x676.png)  
Second, each answer is put through an Output Reward Model (ORM) and scored on the quality of the answer. The answer with the highest score is selected:  
[![answer 1 answer 2 answer 3 answer n ORM ORM ORM ORM 0.4 0.6…](assets/reasoning-25.jpg)](https://substackcdn.com/image/fetch/$s_!Rs6k!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F5f8cdfb5-7dd8-46d2-9cdd-192ca8e83728_1776x656.png)  
Instead of judging the answer, the reasoning process might also be judged with a **Process Reward Model** (PRM) that judges the quality of each reasoning step. It will pick the candidate with the highest total weight.  
[![Question (1 Large Language Model Generate n answers (each…](assets/reasoning-26.jpg)](https://substackcdn.com/image/fetch/$s_!ziIm!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc3371566-e205-4240-8b6e-f2f0db9bd9a2_1776x960.png)  
With both verifier types, we can also weigh each answer candidate by the RM and pick the one with the highest total weight. This is called Weighted Best-of-N samples:  
[![What is 1 + 1? Large Language Model Thoughts Thoughts…](assets/reasoning-27.jpg)](https://substackcdn.com/image/fetch/$s_!BvXO!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F70c6c06f-5846-4888-b486-3b9e17ce753d_1480x1028.png)

**Beam search with process reward models**  
This process of generating answers and intermediate steps can be further extended with beam search. With beam search, multiple reasoning steps are sampled and each is judged by PRM (similar to Tree of Thought[7](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-reasoning-llms#footnote-7)). The top 3 “**beams**” (best-scoring paths) are tracked throughout the process.  
[![What is 1 + 1? Large Language Model Scored by the PRM…](assets/reasoning-28.png)](https://substackcdn.com/image/fetch/$s_!kcTM!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F5d746daf-1f6f-4e0b-886d-a302ce826fb9_1416x1132.png)  
This method allows for quickly stopping “reasoning” paths that do not end up to be fruitful (scored low by the PRM).  
The resulting answers are then weighted using the Best-of-N approach we explored before.  
**Monte Carlo Tree Search**  
A great technique for making the tree searches more efficient is called Monte Carlo Tree Search. It consists of four steps:

- Selection (select a given leaf based on a pre-determined formulate)
- Expand (create additional nodes)
- Rollouts (randomly create new nodes until you reach the end)
- Backprop (update parent node scores based on output)

The main goal of these steps is to keep expanding the best reasoning steps while also exploring other paths.  
It is therefore a balance between **exploration** and **exploitation**. An example of how nodes are scored and selected is as follows:  
[![Constant to balance exploitation and exploration. selection…](assets/reasoning-29.png)](https://substackcdn.com/image/fetch/$s_!LBzz!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F506b026c-709e-4fcc-b3d6-d2d96f0d1ca6_952x328.png)  
Thus, when we select a new reasoning step to explore, it does not have to be the best-performing path thus far.  
Using this type of formula, we start by **selecting** a node (reasoning step) and **expand** it by generating new reasoning steps. As before, this can be done with reasonably high and varying values of **temperature**:  
[![Selection Expand (select path using the selection score)…](assets/reasoning-30.png)](https://substackcdn.com/image/fetch/$s_!yuY6!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F197ba7fc-e7ff-4d2b-9f3b-6cade4612e26_936x624.png)  
One of the expanded reasoning steps is selected and rolled out multiple times until it reaches several answers.  
These rollouts can be judged based on the reasoning steps (PRM), the rewards (ORM), or a combination of both.  
The scores of the parent nodes are updated (backpropagated) and we can start the process again starting with selection.  
[![Rollouts Backprop (sample from the selected step (score the…](assets/reasoning-31.jpg)](https://substackcdn.com/image/fetch/$s_!QqZ-!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F69f0a6f4-d1a7-46ed-8346-2b466aef68d1_1476x1416.png)

**Modifying Proposal Distribution**  
The second category of making reasoning LLMs is called “Modifying Proposal Distribution”. Instead of searching for the correct reasoning steps with verifiers (*output*-focused), the model is trained to create improved reasoning steps (*input*-focused).  
In other words, the distribution from which completions/thoughts/tokens are sampled is modified.  
Imagine that we have a question and a distribution from which we can sample tokens. A common strategy would be to get the highest-scoring token:  
[![Token probability (proposal distribution) 5 3 What is 3 +…](assets/reasoning-32.jpg)](https://substackcdn.com/image/fetch/$s_!ywel!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F53ff298a-d0d1-4704-92e7-9037e9cd26ba_1832x560.png)  
However, note that some of the tokens are colored red in the image above. Those are the tokens that are more likely to lead to a reasoning process:  
[![answering directly 5 . reasoning before answering What is 3…](assets/reasoning-33.jpg)](https://substackcdn.com/image/fetch/$s_!Oy7N!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F0a17b874-c02e-4e71-83db-285618879928_1804x712.png)  
Although taking the greedy token is not necessarily wrong, selecting a token leading to a reasoning process tends to result in an improved answer.  
When we modify the proposal distribution (the token probability distribution), we are essentially making it so that the model re-ranks the distribution such that “reasoning” tokens are selected more frequently:  
[![Token probability (proposal distribution) Adding If What is…](assets/reasoning-34.jpg)](https://substackcdn.com/image/fetch/$s_!KPpo!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F6c8886f6-da9f-4502-ae8d-180ac0e3cc23_1804x560.png)  
There are various methods for modifying the proposal distribution but they can be generally put in two categories:

- Updating the **prompt** through prompt engineering
- **Training** the model to focus on reasoning tokens/processes

**Prompting**  
With prompt engineering, we are attempting to improve the output by updating the prompt. This process might also nudge the model to showcase some of the reasoning processes we saw before.  
To change the proposal distribution with prompting, we can provide examples to the model (in-context learning) that it has to follow to generate reasoning-like behavior:  
[![reasoning Q: What is 3 + 2? Thoughts: reasoning First, 3…](assets/reasoning-35.png)](https://substackcdn.com/image/fetch/$s_!DJpr!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F8a80be6e-40c2-4d3b-bcd1-4fa6b4c30d3d_1804x680.png)  
This process can be further simplified by simply stating “*Let’s think step-by-step*”[8](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-reasoning-llms#footnote-8). Likewise, this changes the proposal distribution in such a way that the LLM tends to break down the process before answering:  
[![reasoning What is 3 + 2? Let's think step-by-step. LLM](assets/reasoning-36.png)](https://substackcdn.com/image/fetch/$s_!CyBW!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F0b013518-3af0-45e9-b684-79d88e4a44c4_1820x612.png)  
However, the model has not inherently learned to follow this process. Moreover, this is a static and linear process that inhibits self-refinement. If a model starts with an incorrect reasoning process it tends to keep it instead of revising it.  
**STaR**  
Aside from prompting, we can also have the model learn to “reason” by having it train so that it is rewarded for generating these reasoning steps. This typically involves a bunch of reasoning data and reinforcement learning to reward certain behaviors.  
A much-debated technique is called STaR or Self-Taught Reasoner[9](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-reasoning-llms#footnote-9). STaR is a method that uses the LLM to generate its own reasoning data as the input for fine-tuning the model.  
In the first step (1), it generates reasoning steps and an answer. If the answer is correct (2a), then add the reasoning and answer to a triplet training data set (3b). This data is used to perform **supervised fine-tuning** of the model (5):  
[![Generate triplet training data 3a generate training data…](assets/reasoning-37.jpg)](https://substackcdn.com/image/fetch/$s_!MlB0!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc4b352d1-2ed6-4b76-8696-4c7ca8df3c68_1640x1100.png)  
If however, the model provides an incorrect answer (2b), then we provide a “hint” (the correct answer) and ask the model to reason why this answer would be correct (4b). The final reasoning step is to add to the same triplet training data which is used to perform **supervised fine-tuning** of the model (5):  
[![Generate triplet training data 3a generate training…](assets/reasoning-38.jpg)](https://substackcdn.com/image/fetch/$s_!oKX-!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa4546442-c6ad-4119-99be-ee4edbfb0cd3_1640x1100.png)  
A key element here (along with many other techniques for modifying the proposal distribution) is that we explicitly train the model to follow along with the reasoning processes we show it.  
In other words, we decide how the reasoning process should be through **supervised fine-tuning**.  
The full pipeline is quite interesting as it essentially generates **synthetic training** **examples**. Using synthetic training examples (as we will explore in DeepSeek R-1) is an incredible method of also **distilling** this reasoning process in other models.  
**DeepSeek-R1**  
A major release in reasoning models is **DeepSeek-R1**, an open-source model whose weights are available[10](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-reasoning-llms#footnote-10). Directly competing with the OpenAI o1 reasoning model, DeepSeek-R1 has had a major impact on the field.  
DeepSeek has done a remarkable job at elegantly distilling reasoning into its base model (**DeepSeek-V3-Base**) through various techniques.  
Interestingly, no verifiers were involved and instead of using supervised fine-tuning to distill reasoning behavior, a large focus was on reinforcement learning.  
Note: This claim is incorrect. Verifiers were a core component of DeepSeek-R1's training. The model was trained using Reinforcement Learning with Verifiable Rewards (RLVR), where a rule-based verifier checks if the model's final answer is correct (e.g., by matching a ground-truth math answer) and provides the reward signal. While this is a simpler form of verification than a trained neural network ORM or PRM, it is still a verifier.  
Let’s explore how they trained reasoning behavior in their models!  
**Reasoning with DeepSeek-R1 Zero**  
A major breakthrough leading to DeepSeek-R1 was an experimental model called **DeepSeek-R1 Zero**.  
Starting with DeepSeek-V3-Base, instead of using supervised fine-tuning on a bunch of reasoning data, they only used reinforcement learning (RL) to enable reasoning behavior.  
To do so, they start with a very straightforward prompt (similar to a system prompt) to be used in the pipeline:  
[![System prompt A conversation between User and Assistant.…](assets/reasoning-39.png)](https://substackcdn.com/image/fetch/$s_!8JoA!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F1f01d333-2976-41fc-8cc2-2180fb2e9c59_2604x918.png)  
Note how they explicitly mention that the reasoning process should go between **\<think>** tags but they do not specify what the reasoning process should look like.  
During reinforcement learning, two specific rule-based rewards were created:

- Accuracy rewards - Rewards the **answer** by testing it out.
- Format rewards - Rewards using the **\<thinking>** and **\<answer>** tags.

The RL algorithm used in this process is called Group Relative Policy Optimization (GRPO)[11](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-reasoning-llms#footnote-11). The intuition behind this algorithm is that it makes all choices that led to a correct or incorrect answer more or less likely. These choices can be both sets of tokens as well as reasoning steps.  
[![System prompt Use <think> for reasoning and use <answer>…](assets/reasoning-40.jpg)](https://substackcdn.com/image/fetch/$s_!7kW3!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fbf299252-386e-43f3-9875-238dd7d7e6fe_1628x1624.png)  
Interestingly, no examples were given on how the \<think> process should look like. It merely states that it **should use \<think> tags**, and nothing more!  
By providing these indirect rewards related to Chain-of-Thought behavior, the model learned by itself that the longer and more complex the reasoning process, the more likely the answer was correct.  
[![DeepSeek-R1-Zero average length per response during…](assets/reasoning-41.jpg)](https://substackcdn.com/image/fetch/$s_!hAHl!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F746bd683-9e5d-4f9e-9e22-7ab76c5a5402_2616x1724.png)  
Annotated figure from “[DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/pdf/2501.12948?)”. By using indirect RL rewards, the model freely explores the most optimal Chain-of-Thought like behavior through ever increasing reasoning steps.  
This graph is especially important as it reinforces the **paradigm shift** from **train**-time compute to **test**-time compute. As these models generate longer sequences of thoughts, they focus on **test**-time compute.  
Using this training pipeline, they found that the model discovers on its own the most optimal Chain-of-Thought-like behavior, including advanced reasoning capabilities such as self-reflection and self-verification.  
However, it still had a significant drawback. It had poor readability and tended to mix languages. Instead, they explored an alternative, the now well-known **DeepSeek R1**.  
Let’s explore how they stabilized the reasoning process!  
**DeepSeek-R1**  
To create DeepSeek-R1, the authors followed five steps:

1. Cold Start
2. Reasoning-oriented Reinforcement Learning
3. Rejection Sampling
4. Supervised Fine-Tuning
5. Reinforcement Learning for all Scenarios

In **step 1**, DeepSeek-V3-Base was fine-tuned with a small high-quality reasoning dataset (≈5.000 tokens). This was done to prevent the cold start problem resulting in poor readability.  
[![DeepSeek-V3-Base Supervised Reasoning ~5.000 high-quality…](assets/reasoning-42.jpg)](https://substackcdn.com/image/fetch/$s_!VmRV!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ffa0c1e2c-b517-44bf-833a-36ae68aaa551_1628x836.png)  
In **step 2**, the resulting model was trained using a similar RL process as was used to train DeepSeek-R1-Zero. However, another reward measure was added to make sure the target language remained consistent.  
[![DeepSeek-V3-1 <Format> reward Reasoning (is <think> used?)…](assets/reasoning-43.jpg)](https://substackcdn.com/image/fetch/$s_!veCD!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F905ed035-f5bb-4b34-8351-c15f93e9b3c7_1628x1380.png)  
In **step 3**, the resulting RL-trained model was used to generate synthetic reasoning data to be used for supervised fine-tuning in a later stage. Through rejecting sampling (rule-based rewards) and a reward model (DeepSeek-V3-Base), 600,000 high-quality reasoning samples were created.  
Moreover, 200,000 non-reasoning samples were created by using DeepSeek-V3 and part of the data it was trained on.  
[![Reasoning samples Non-reasoning samples Question Question…](assets/reasoning-44.jpg)](https://substackcdn.com/image/fetch/$s_!Z7Q9!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd31892cd-e082-455c-96b0-3c209eca53ec_1712x1828.png)  
In **step 4**, the resulting dataset of 800,000 samples was used to perform supervised fine-tuning of the DeepSeek-V3-Base model.  
[![DeepSeek-V3-Base Supervised Reasoning 800.000 high-quality…](assets/reasoning-45.jpg)](https://substackcdn.com/image/fetch/$s_!a0G6!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F7facfbe9-de6d-4b6e-8372-64d79ef1eb9d_1628x836.png)  
In **step 5**, RL-based training was performed on the resulting model using a similar approach used in DeepSeek-R1-Zero. However, to align with human preferences, additional reward signals were added focused on helpfulness and harmlessness.  
The model was also asked to summarize the reasoning process to prevent readability issues.  
[![DeepSeek-R1 <Format> reward Reasoning (is <think> used?)…](assets/reasoning-46.jpg)](https://substackcdn.com/image/fetch/$s_!ZN7m!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F69d3517a-a3d6-4c56-a269-ac3dab6ba3ad_1628x1620.png)  
And that’s it! This means that DeepSeek-R1 is actually a fine-tune of DeepSeek-V3-Base through supervised fine-tuning and reinforcement learning.  
The bulk of the work is making sure that high-quality samples are generated!  
**Distilling Reasoning with DeepSeek-R1**  
DeepSeek-R1 is a huge model with 671B parameters. Unfortunately, this means running such a model is going to be difficult on consumer hardware.  
Fortunately, the authors explored ways to distill the reasoning quality of DeepSeek-R1 into other models, such as Qwen-32B, which we can run on consumer hardware!  
To do so, they use the DeepSeek-R1 as a **teacher** model and the smaller model as a **student**. Both models are presented with a prompt and have to generate a token probability distribution. During training, the **student** will attempt to closely follow the distribution of the **teacher**.  
[![Token probability (proposal distribution) Adding Teacher If…](assets/reasoning-47.jpg)](https://substackcdn.com/image/fetch/$s_!tMao!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F65aaf771-2030-4e4e-a273-fc8c517aef31_2332x1308.png)  
This process was done using the full 800,000 high-quality samples that we saw before:  
[![Teacher 800.000 high-quality reasoning and non- DeepSeek-R1…](assets/reasoning-48.jpg)](https://substackcdn.com/image/fetch/$s_!Vv2Y!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F7a0c9f75-03b8-493a-a5e3-ffb744153961_1888x820.png)  
The resulting distilled models are quite performant since they not just learned from the 800,000 samples but also the way the teacher (DeepSeek-R1) would answer them!  
**Unsuccessful Attempts**  
Remember when we talked about **Process Reward Models** (PRMs) and **Monte Carlo Tree Search** (MCTS)? It turns out that DeepSeek also tried those techniques to instill reasoning but was not successful in doing so.  
With MCTS, they encountered issues with the large search space and had to limit the node expansions. Moreover, training a fine-grained Reward Model is inherently difficult.  
With PRMs for Best-of-N techniques, they encountered issues with computational overhead to keep re-training the Reward Model to prevent reward-hacking.  
This does not mean these are not valid techniques but it provides interesting insights into the limitations of these techniques!

Summary:

- Reasoning LLMs tend to break down a problem into smaller steps (often called reasoning steps or thought processes) before answering a given question.
- Behind the reasoning LLMs is the paradigm shift from a focus on scaling training (train-time compute) to inference (test-time compute).
- Train-time compute combines models parameter, dataset token, and compute. To increase the performance of LLMs during pre-training, developers often increase the size this combination.
    - Parameters are the learned weights and biases stored in the model.
    - Activations / hidden states or contextual representation: token embedding+ attention info+ FFN info. It's not counted as parameters.
- Train-time compute may include both the compute needed during training as well as during fine-tuning.
- Instead of continuously increasing pre-training budgets, test-time compute allows modes to “think longer” during inference. Reasoning models, however, would instead use more tokens to derive their answer through a systematic “ **thinking** ” process which make it more costly but has better performance.
- More test time compute mean less train time compute, while less-time compute means more train-time compute
- Through this paradigm shift, instead of focusing purely on train-time compute (pre-training and fine-tuning), these “reasoning” models instead balance training with inference.
- Test-time compute can be many different things, including Chain-of-Thought, revising answers, backtracking, sampling, and more. Can be put roughly into two categories:
    - Search against Verifiers (sampling generations and selecting the best answer) (output focus). Using this strategy two models are used one for verifying and one for generating answer.
    - Modifying Proposal Distribution (trained “thinking” process) (input focus)
- There are two types of verifiers (type of LLM fine tuned):
    - Outcome Reward Models (ORM): Judge the outcome and doesn't care about the underlying process
    - Process Reward Models (PRM): Judge the process that lead to outcome (the reasoning)
- Majority voting (usually called majority voting or self-consistency) is a test-time compute method where the model generates multiple answers — often via multiple reasoning paths — and then selects the most frequent final answer among them.
- PRM is the scorer. Beam search is the search algorithm that uses those scores to decide what to do next.
- Monte Carlo Tree Search (MCTS) is a search algorithm that builds a tree of possible reasoning paths by repeatedly selecting promising nodes, expanding them, simulating outcomes, and backpropagating scores from a verifier/reward model to update each node’s value. It balances exploring new paths with exploiting high-scoring ones, making it well-suited for test-time reasoning when paired with ORMs or PRMs or both.
- Modifying Proposal Distribution is the input-focused category where the model’s token sampling distribution is changed to favor reasoning-like tokens, either through prompt engineering or training. Instead of searching over outputs with verifiers, you make the LLM itself more likely to generate good reasoning steps by re-ranking its token probabilities toward reasoning tokens.
    - "reasoning-like tokens" = tokens that tend to initiate or sustain a reasoning process, not a fixed list of words.
- Prompt engineering modifies the proposal distribution only temporarily, by conditioning the model on examples or instructions that encourage step-by-step reasoning. It is a lightweight, static, and linear intervention — it does not change the model’s weights, and it often cannot correct an incorrect reasoning path once started.
- STaR (Self-Taught Reasoner) lets an LLM generate its own reasoning traces and answers, keeps the traces that lead to correct answers, and for incorrect answers uses the correct answer as a hint to produce a rationalization that is also added to the training set. The model is then fine-tuned on this synthetic data, iteratively modifying its proposal distribution toward better reasoning — without using reinforcement learning in the original method.
- DeepSeek-R1 Zero started from DeepSeek-V3-Base and used only reinforcement learning with GRPO plus rule-based accuracy and format rewards, without supervised reasoning examples, to let the model discover longer Chain-of-Thought reasoning, self-reflection, and self-verification on its own. However, it suffered from poor readability and mixed languages, which motivated the development of the stabilized DeepSeek-R1.
- DeepSeek-R1 was built from DeepSeek-V3-Base through a five-step pipeline: a small cold-start reasoning fine-tune, reasoning-oriented RL with a language-consistency reward, rejection sampling to generate 600k reasoning plus 200k non-reasoning samples, supervised fine-tuning on those 800k samples, and a final RL stage for all scenarios with helpfulness/harmlessness rewards and reasoning summarization. Overall, R1 is essentially a fine-tune of DeepSeek-V3-Base via supervised fine-tuning and reinforcement learning, with most of the work going into generating high-quality training samples.
- DeepSeek-R1 (671B parameters) was distilled into smaller consumer-runnable models like Qwen-32B by using R1 as a teacher and training the student to match its token probability distributions on the 800,000 high-quality samples, producing performant distilled models that learn both the data and R1’s answering behavior.

Source  
[https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-reasoning-llms](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-reasoning-llms)
