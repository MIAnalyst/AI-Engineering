# AI Agents vs Agentic AI

Lecture recap. The two terms are often used interchangeably, but they describe different levels: **one agent doing one job** vs **a system of agents working toward a larger goal**.

Related: [Subagent](Subagent.md) · [Swarm Orchestration - Agent Teams](Swarm%20Orchestration%20-%20Agent%20Teams.md) · [Agentic Rag](../RAG/Agentic%20Rag.md) · [LangGraph](../RAG/LangGraph.md)

## Definitions

- **AI agent**: a single software program built to perform a **specific task** with some autonomy, meaning it can do that task without human intervention. It's good for automating focused or repetitive work.
- **Agentic AI**: a **broader framework** in which **multiple AI agents collaborate** and make decisions on their own to reach a **larger goal**. It stresses adaptability and **learning from experience** rather than following predefined rules.

Both run without human intervention. The difference is **scope**: an agent handles one task, while an agentic system handles **complex, multi-step workflows** because several agents share the work.

## Key differences

| Parameter | AI Agents | Agentic AI |
|---|---|---|
| **Functionality** | Handles one specific task, with **limited autonomy**. It won't take on other tasks it wasn't assigned. | Focuses on the **overall system**: it can learn, adapt and make more complex, situation-dependent decisions. |
| **Decision making** | Follows **predefined rules**. | Makes **autonomous decisions** based on **real-time data and circumstances**. |
| **Structure** | A **single entity** with a focused purpose. | A **network of agents** working together toward a goal. |
| **Complexity** | Simple, well-defined processes. | Complex workflows. Agents communicate with each other and learn from the environment. |

## Examples

### AI agent: customer service chatbot

Answers basic questions from a predefined script, such as "When will my order arrive?"

```
User input ──► Chatbot (LLM) ──► Company DB (retriever) ──► Response
```

- It does one thing: customer service lookups.
- An LLM paired with an external database (a vector DB or any other store) acting as a **retriever** is the typical setup.

### AI agent: automated banking bot

Used by many banks for **balance inquiries**, **managing transactions**, and opening current or savings accounts.

- It counts as an AI agent because **the processes are clearly defined**. A single agent carries out each defined task.

### Agentic AI: smart home system

There are several independent agents, one per domain:

- **Lights** agent
- **Thermostat** agent (AC up or down)
- **Appliances** agent

Their **shared goal** is to reduce electricity use. The agents communicate based on the environment. For example, they turn off the lights when nobody is in the room and switch off the thermostat when nobody is home.

Every decision follows a four-step loop:

1. **Perception**: gather data from the surroundings.
2. **Reasoning**: interpret that data to understand what's going on.
3. **Action**: act on the reasoning, e.g. switch off the lights or the thermostat.
4. **Learning**: improve and adapt over time (e.g. reinforcement learning).

```
Perception ──► Reasoning ──► Action ──► Learning
     ▲                                     │
     └─────────────────────────────────────┘
```

### Agentic AI: personalized health assistant

- It analyzes a patient's **medical history**, **real-time health data** and **lifestyle factors**.
- It produces **personalized care recommendations**, which is the shared goal.
- It **adapts** to new medical research and patient feedback, which shows the greater flexibility and complexity of agentic AI.

## Where the course goes from here

- We'll build both **standalone AI agents** and **agentic AI applications** with **LangGraph**.
- The **LLM acts as the "brain"** of each agent. It decides which actions to take and connects to **tools** and **databases**.

## Takeaway

> **AI agent** = one entity, one focused task, rule-driven.
> **Agentic AI** = many agents collaborating, goal-driven, adaptive, and learning from the environment.

*Note:* the lecture's split ("agents follow predefined rules") is a teaching simplification. LLM-based agents also reason about which tool to call, so the line between the two is really about **scope** (one task vs an orchestrated multi-agent goal) and **adaptivity**, not whether an LLM is involved.
