# Tools in LangChain

> **One-line summary:** A tool is a Python function that the LLM can ask to run. In LangChain you turn a function into a tool with the `@tool` decorator. You attach it to a model with `bind_tools`, or pass it to `create_agent`. The model decides when to call it, and your code (or the agent) runs it and sends the result back.

---

## 1. Where We Are in the Series

| Already covered | Now |
|---|---|
| Agents (`create_agent`) | **Tools: how to create and use them** |
| Model integration with `init_chat_model`, plus `ChatGroq`, `ChatOpenAI` and `ChatGoogleGenerativeAI` | |
| `invoke`, `stream`, `batch` | |

Recall the basic agent: **LLM ↔ Tool**. The tool is just some functionality. It can be:
- an API request
- a built-in LangChain tool
- a news-reporting tool
- a Google search tool
- any independent function

---

## 2. What Is a Tool?

From the LangChain docs: *a model can request to call tools that perform tasks such as fetching data from a database, searching the web, or running code.*

A tool is a **pairing of**:

| Part | What it is |
|---|---|
| **Schema** | The tool's **name**, **description** and **argument definitions** |
| **Function (or coroutine)** | The actual code that executes |

The **schema** is what the LLM reads to decide *whether* and *how* to call the tool.

---

## 3. Creating a Tool (the simple way)

### Step 1: Set up the model

The demo uses a Groq-hosted Qwen3 32B model (invoked as in the earlier model-integration section).

### Step 2: Import the `tool` decorator

```python
from langchain.tools import tool
```

`tool` is used as a **decorator**. Placing it above a function turns that function into a LangChain tool.

### Step 3: Define the function with a docstring

```python
@tool
def get_weather(location: str) -> str:
    """Get weather at a location."""
    return f"It's sunny in {location}."
```

| Piece | Why it matters |
|---|---|
| `@tool` | Converts the function into a LangChain tool |
| Function name `get_weather` | Becomes the **tool name** |
| Argument `location: str` | Becomes the tool's **argument schema** (needs type hints) |
| **Docstring** | Becomes the **description**. The LLM reads it to understand **what the tool does** |
| Return value | This is the **context** passed back to the LLM |

> **The docstring is very important.** When the tool is bound to the LLM, the model only knows what the tool does from this description. A vague or missing docstring means the model may not use the tool correctly.

> In this demo the return value is **hard-coded** ("It's sunny..."). In a real tool you would call an **API** or **database** here.

---

## 4. Connecting the Tool to a Model

There are **two ways**.

### Way 1: `bind_tools` (model + tool, no agent)

```python
model_with_tools = model.bind_tools([get_weather])

response = model_with_tools.invoke("What's the weather like in Boston?")
print(response)               # includes reasoning + tool call info
print(response.tool_calls)    # the tool call(s) the model wants to make
```

What you'll see:
- The model reasons: *"The user is asking about the weather in Boston, I need to use the get_weather function."*
- `response.tool_calls` shows:
  - **name:** `get_weather`
  - **args:** `{"location": "Boston"}`

> **Important:** the model only **requests** a tool call. At this point the tool has **not run yet**. Running it is a separate step (see section 5).

### Way 2: `create_agent` (the easiest)

```python
agent = create_agent(
    model="...",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)
```

The agent handles binding and calling for you automatically (covered in the earlier agents lecture).

| | `bind_tools` | `create_agent` |
|---|---|---|
| Who runs the tool? | **You** (manually) | The **agent** (automatically) |
| Control | Full, step by step | Less, but simpler |
| Typical use | Learning internals, custom loops | Most real apps |

---

## 5. The Tool Execution Loop

With `bind_tools`, the model only *asks* to use a tool. **You** must run it and feed the result back. This is the **tool execution loop**.

```python
# 1. Start with the user message
messages = [{"role": "user", "content": "What's the weather in Boston?"}]

# 2. Model decides to call a tool
ai_msg = model_with_tools.invoke(messages)
messages.append(ai_msg)                      # keep the AI's tool-call message

# 3. Run each requested tool and collect the results
for tool_call in ai_msg.tool_calls:
    tool_result = get_weather.invoke(tool_call)   # returns a ToolMessage
    messages.append(tool_result)                  # add the result to the conversation

# 4. Call the model again with the tool results as context
final_response = model_with_tools.invoke(messages)
print(final_response.text)    # "The weather in Boston is sunny"
```

### Flow diagram

```
 User message
     │
     ▼
 Model.invoke ──► AI message with tool_calls
     │                      │
     │                      ▼
     │            get_weather.invoke(tool_call)
     │                      │
     │                      ▼
     │              Tool message (result = CONTEXT)
     │                      │
     ▼                      ▼
 Model.invoke (messages now include tool result)
     │
     ▼
 Final answer: "It's sunny in Boston"
```

### What ends up in `messages`

| # | Message | Content |
|---|---|---|
| 1 | **Human** (`role: user`) | "What's the weather in Boston?" |
| 2 | **AI** | A tool call to `get_weather(location="Boston")` |
| 3 | **Tool** | "It's sunny in Boston." (the context) |
| 4 | **AI** (final) | The natural-language answer |

> `create_agent` runs this loop **for you**. This section shows what happens under the hood.

---

## 6. Cheat Sheet

```python
from langchain.tools import tool

# 1. Define a tool
@tool
def get_weather(location: str) -> str:
    """Get weather at a location."""
    return f"It's sunny in {location}."

# 2a. Bind to a model (manual loop)
model_with_tools = model.bind_tools([get_weather])
ai_msg = model_with_tools.invoke("What's the weather in Boston?")
print(ai_msg.tool_calls)

# 2b. OR let an agent do everything
agent = create_agent(model="...", tools=[get_weather], system_prompt="...")
```

---

## 7. Key Takeaways

1. A **tool** = **schema** (name, description, args) + **function** to execute.
2. Use the **`@tool` decorator** on a function to create a tool.
3. The **docstring is the tool's description**. The LLM relies on it to decide when to call the tool, so write it clearly.
4. The function's **return value is the context** sent back to the LLM. Inside it you can use any logic: APIs, databases, search, or LangChain's built-in tools.
5. Two ways to connect tools: **`model.bind_tools([...])`**, where you run the loop yourself, or **`create_agent(..., tools=[...])`**, which does it all.
6. The model only **requests** tool calls (`response.tool_calls`). The **tool execution loop** is what actually runs them and returns the results.
7. A tool call has a **name** and **args**. The result comes back as a **Tool message**.

---

## 8. What's Next

**Messages:** the different message types, including **System**, **Human** and **AI** messages (and the Tool message seen above).
