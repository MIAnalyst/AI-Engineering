# Create Agents Using LangChain (v1)

> **LangChain series · Lecture 2 of 7** — Previous: [Langchain](Langchain.md) · Next: [Invoke And Batch Streaming](Invoke%20And%20Batch%20Streaming.md)

> **One-line summary:** An *agent* is an LLM that can **decide on its own** when to call external tools, get context back from them, and then produce the final answer. In LangChain v1, you build one with a single function: `create_agent`.

---

## 1. Setup Recap

| Item | Detail |
|---|---|
| Environment | Virtual environment created with **uv** package manager |
| Dependencies | Installed via uv and recorded in `pyproject.toml` |
| Project folder | `updated langchain/` |
| Notebook | `langchain_intro.ipynb` (kernel = the `.venv` Python environment) |
| Secrets | API keys stored in `.env` and loaded with `python-dotenv` |
| LangChain version used | **1.1.0** (check with `langchain.__version__`) |

```python
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
```

---

## 2. From Plain LLM to Agent (the core idea)

### Stage 1: Plain LLM = a simple GenAI application

```
Input  ──►  LLM  ──►  Output
```

- Example: "Write a 200-word paragraph on AI" → the LLM returns a 200-word paragraph.
- Works with any model (OpenAI, Gemini, Groq, open-source, ...).

### Stage 2: The problem with a plain LLM

- An LLM has a **training cut-off date**. It was trained on past data only.
- So it **cannot answer** questions like *"What is today's AI news?"* or *"What's the weather right now?"*

### Stage 3: LLM + Tool = Agent

To answer such questions the LLM must depend on an **external tool**, such as:

- a third-party API
- Google Search
- any custom function

```
                         ┌──────────────┐
Input ──► LLM (decides) ─┤  Tool / API  │
              ▲          └──────┬───────┘
              │                 │ returns CONTEXT
              └─────────────────┘
              │
              ▼
           Output
```

**Flow:**
1. User gives an input ("What are today's AI news?").
2. The LLM **decides**: "I can't answer this from my training data. I need a tool."
3. It picks the **right tool** and calls it.
4. The tool returns **context** (fresh, real data).
5. The LLM uses that context to **generate the final output**.

> **This is a basic agent.** It is *autonomous* because the model itself decides **when** to call a tool, **which** tool to call, and **how** to solve the task.

### Key terms

| Term | Meaning |
|---|---|
| **LLM** | The "brain" that reasons and writes the answer |
| **Tool** | An external function or API the LLM can call |
| **Context** | Whatever the tool returns, fed back to the LLM |
| **Agent** | LLM + tools + decision-making (autonomous routing) |

---

## 3. Old Way vs. New Way

| | Before (pre-v1) | Now (LangChain v1) |
|---|---|---|
| Steps | Create LLM → create tool separately → manually link them | Pass model + tools to one function |
| Architecture | Had to build a **ReAct** architecture yourself | Handled for you |
| Difficulty | Fairly hard | Very easy |
| API | Several pieces | `create_agent(...)` |

---

## 4. Creating an Agent: Step by Step

### Step 1: Create a basic agent (no tools yet)

```python
from langchain.agents import create_agent

agent = create_agent(
    model="gpt-5",                       # model name (OpenAI here)
    tools=[],                            # no tools yet
    system_prompt="You are a helpful assistant",
)
agent                                    # displays the graph
```

**Parameters:**

| Parameter | Purpose |
|---|---|
| `model` | Which LLM to use. Can be given as a string like `"gpt-5"` |
| `tools` | List of tools the agent may call (empty for now) |
| `system_prompt` | Instruction that defines the agent's behavior |

> **Gotcha:** `verbose=True` caused an *unexpected keyword argument* error, so `create_agent` doesn't support it. **Remove it.**

**Graph diagram at this stage:** `START → model → END`
There is only input → output. With no tools, it's still just a plain LLM.

---

### Step 2: Create a tool (just a Python function)

```python
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny."
```

Points to note:
- Type hints: `city: str` in, `str` out.
- The **docstring is important** (see below).
- In this demo it returns a hard-coded string. In real life it would call a weather API, and that API response would be the **context**.

> **Why the docstring matters:** The LLM reads the function's docstring to understand **what the tool does** and **when to use it**. That's how it knows to call `get_weather` for a weather question.

---

### Step 3: Attach the tool to the agent

```python
agent = create_agent(
    model="gpt-5",
    tools=[get_weather],                 # tool added here
    system_prompt="You are a helpful assistant",
)
agent
```

**Graph diagram now:** `START → model ⇄ tools → END`
The model is now connected to the tools node. For weather questions it will call the tool, get the result, then answer.

---

### Step 4: Run the agent with `invoke`

#### Wrong input (causes an error)

```python
agent.invoke("What is the weather like in New York?")
# Error: expected dictionary
```

`create_agent` agents **require a dictionary** with the key `"messages"`.

#### Correct input

```python
response = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the weather like in New York?"}]}
)
```

#### Get just the final answer

```python
response["messages"][-1].content
# "It's sunny in New York."
```

- `response["messages"]` = the **whole conversation**.
- `[-1]` = the **last message** (the final AI answer).

#### Shorter form

```python
agent.invoke({"messages": "What is the weather in New York?"})
```

You don't have to specify the role. A plain string is **automatically treated as a human message**. It even handled a typo in the question correctly.

---

## 5. What Happens Inside One `invoke` (message trace)

For *"What is the weather like in New York?"* the conversation contains these messages in order:

| # | Message type | What it contains |
|---|---|---|
| 1 | **Human Message** | The user's question |
| 2 | **AI Message** (tool call) | The model decides to call `get_weather` with `city="New York"` |
| 3 | **Tool Message** | The tool's output: "The weather in New York is sunny" |
| 4 | **AI Message** (final) | "It's sunny in New York." |

```
Human ─► AI (tool call) ─► Tool (result = context) ─► AI (final answer)
```

> **How does the LLM know which tool to call?** From the tool's **name, signature and docstring**, which are passed to the model along with the question.

---

## 6. Cheat Sheet

```python
# 1. Import
from langchain.agents import create_agent

# 2. Define tools (functions with docstrings + type hints)
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny."

# 3. Create agent
agent = create_agent(
    model="gpt-5",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

# 4. Run it (input MUST be a dict with a "messages" key)
response = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the weather in New York?"}]}
)

# 5. Read the answer
print(response["messages"][-1].content)
```

---

## 7. Common Errors & Fixes

| Error | Cause | Fix |
|---|---|---|
| `unexpected keyword argument 'verbose'` | `create_agent` has no `verbose` parameter | Remove `verbose=True` |
| `expected dictionary` | Passed a plain string to `invoke` | Pass `{"messages": ...}` |

---

## 8. Key Takeaways

1. A plain LLM is limited by its **training cut-off**. Tools remove that limit.
2. **Agent = LLM + tools + autonomous decision-making.**
3. LangChain v1 reduces agent creation to **one function**: `create_agent`.
4. A tool is just a **Python function**. Its **docstring tells the LLM when to use it**.
5. Input is a **dict with `messages`**, and the answer is `response["messages"][-1].content`.
6. You can add **any number of tools**, and the agent chooses the right one.
7. The agent is *autonomous* because the model decides which tool to call, gets the context, and then answers.

---

## 9. What's Next

- **Model integration:** using different LLM providers with LangChain.
- **Calling a model:** `invoke`, `stream` and `batch`, in [Invoke And Batch Streaming](Invoke%20And%20Batch%20Streaming.md).
- Building agents with **tools**: [Tools In Langchain](Tools%20In%20Langchain.md).
- **Message types:** Human, AI, System and Tool messages in detail, in [Messages Types In Langchain](Messages%20Types%20In%20Langchain.md).
- Controlling an agent's loop: [MiddleWare With Langchain](MiddleWare%20With%20Langchain.md).
