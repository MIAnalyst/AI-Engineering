# Middleware in LangChain (Complete Notes)

> **LangChain series · Lecture 7 of 7** — Previous: [LLM Structured Output With Langchain](LLM%20Structured%20Output%20With%20Langchain.md)

> **One-line summary:** Middleware lets you plug extra logic *into* an agent's loop (before/after the model, around tool calls, and so on) to add control, safety, cost savings and reliability, without rewriting the agent. You attach it with `create_agent(..., middleware=[...])`.

**Builds on:** [Create Agents Using Langchain](Create%20Agents%20Using%20Langchain.md) (`create_agent`), [Tools In Langchain](Tools%20In%20Langchain.md) (the tools the examples guard) and [Messages Types In Langchain](Messages%20Types%20In%20Langchain.md) (the message history that gets summarized).

**Sources:** Lecture 1 (Summarization Middleware), Lecture 2 (Human-in-the-Loop Middleware), plus extra built-in middleware the lectures only mentioned or skipped (marked ➕).

---

## Table of Contents
1. [What is middleware?](#1-what-is-middleware)
2. [Hooks: where middleware plugs in](#2-hooks-where-middleware-plugs-in)
3. [Middleware overview table](#3-middleware-overview-table)
4. [Covered in the lectures](#4-covered-in-the-lectures)
   - 4.1 SummarizationMiddleware
   - 4.2 HumanInTheLoopMiddleware
   - 4.3 Mentioned only briefly (limits, fallback, tool selector, to-do list)
5. [➕ Important middleware missing from the lectures](#5--important-middleware-missing-from-the-lectures)
6. [➕ Writing your own middleware](#6--writing-your-own-middleware)
7. [Stacking middleware](#7-stacking-middleware)
8. [Middleware in RAG (without tools)](#8-middleware-in-rag-without-tools)
9. [Key takeaways](#9-key-takeaways)

---

## 1. What is middleware?

**Definition:** Middleware gives you a way to **tightly control what happens inside an agent**.

| Category | Examples |
|---|---|
| Observability | Logging, analytics, debugging |
| Transformation | Modify prompts, select tools, format output |
| Reliability | Retries, fallbacks, early termination |
| Safety / cost | Rate limits, guardrails, PII detection |

### Analogy: airport security
You (the request) pass several checkpoints before reaching your gate (the agent):

```
Passenger ──► Security check ──► Immigration ──► Boarding pass check ──► Gate / Flight
              (Middleware 1)     (Middleware 2)   (Middleware 3)          (Agent)
              luggage, batteries passport, expiry boarding pass valid
```

Each checkpoint does **one specific job** before you move on. Each middleware does one job (check, log, handle errors, summarize...) around the agent.

### What does "Transformation: modify prompts" mean?

The other categories in the table observe or protect the agent. **Transformation** means middleware **changes what goes into the model, or what comes out**, instead of only watching it. The docs overview words it as "transforming prompts, tool selection, and output formatting". The lecture repeated that line but never demonstrated it directly.

| What is transformed | Examples |
|---|---|
| **The prompt** (what the model sees) | Add today's date or the user's name to the system prompt; use a different system prompt by user role; put retrieved documents into the prompt (see [section 8](#8-middleware-in-rag-without-tools)); trim or compress the message history; redact personal data |
| **The tool list** | Show the model only the tools relevant to this query (`LLMToolSelectorMiddleware`) |
| **The output** | Reformat or post-process the answer before it reaches the user |

You already met two built-in examples without the label:
- `SummarizationMiddleware` replaces old messages with a summary, so the model gets a different prompt than the one the conversation built up.
- `PIIMiddleware` redacts the messages before the model sees them.

For custom prompt changes, the custom middleware docs mention a `dynamic_prompt` decorator for generating system prompts. The pages I checked don't show its signature, so there is no example for it in these notes.

**Not the same as DSPy.** Both touch the prompt, but they do different jobs:

| | Middleware: modify prompts | DSPy |
|---|---|---|
| When | At runtime, on every call | Mostly offline, in an optimization step before deployment |
| Who decides the change | You, with fixed rules you wrote ("add the date", "summarize after 10 messages") | An optimizer that searches for better prompts and few-shot examples against your metric and data |
| Goal | Control, safety, cost, reliability | Better answer quality on a task |

They can be combined: DSPy tunes the prompt, and middleware handles the runtime concerns around it. (This DSPy comparison is from my general knowledge, not from the LangChain docs.)

---

## 2. Hooks: where middleware plugs in

**Plain ReAct agent:** `request → model → (tool call?) → tool → model → result`

**With middleware**, the same loop exposes **hooks** (trigger points) where you can inject logic:

```
 before_agent ─┐
               ▼
        ┌─► before_model ─► MODEL ─► after_model ─┐
        │            (wrap_model_call)             ▼
        │                              wrap_tool_call ─► TOOLS
        └──────────────────────────────────────────┘
               ▼
 after_agent ──┘
```

| Hook | Runs | Typical use |
|---|---|---|
| `before_agent` | Once, when the agent starts | Load memory, validate input |
| `before_model` | Before each model call | Summarize history, trim messages |
| `wrap_model_call` | *Around* each model call | Retry, fallback, swap model or prompt |
| `after_model` | After each model response | Guardrails, human approval of tool calls |
| `wrap_tool_call` | *Around* each tool call | Tool retry, error handling, logging |
| `after_agent` | Once, when the agent finishes | Save results, cleanup |

The lectures describe hooks as "trigger points before the agent, before the model, around tool calls, after the model, after the agent".

- **Node-style hooks** (`before_agent`, `before_model`, `after_model`, `after_agent`) run at one point and return state updates or `None`.
- **Wrap-style hooks** (`wrap_model_call`, `wrap_tool_call`) receive a `handler` for the real call, so they can retry it, skip it, or change it.
- **Execution order** with `middleware=[m1, m2, m3]`: `before_*` hooks run m1 → m2 → m3; `wrap_*` hooks nest like function calls (m1 wraps m2, which wraps m3, which wraps the model or tool); `after_*` hooks run m3 → m2 → m1.

---

## 3. Middleware overview table

| # | Middleware | Purpose | Covered? |
|---|---|---|---|
| 1 | **SummarizationMiddleware** | Compress old chat history into a summary when it nears token/message limits | ✅ Lecture 1 (full demo) |
| 2 | **HumanInTheLoopMiddleware** | Pause before sensitive tool calls for a human to approve, edit or reject | ✅ Lecture 2 (full demo) |
| 3 | **ModelCallLimitMiddleware** | Cap the number of model calls to prevent runaway loops and cost | 🟡 Mentioned |
| 4 | **ToolCallLimitMiddleware** | Cap the number of tool executions (all tools, or one named tool) | 🟡 Mentioned |
| 5 | **ModelFallbackMiddleware** | If the primary model or API key fails, fall back to other models in order | 🟡 Mentioned |
| 6 | **LLMToolSelectorMiddleware** | With many tools, let a cheap LLM pick only the relevant ones per query | 🟡 Mentioned |
| 7 | **TodoListMiddleware** | Give the agent a `write_todos` tool to plan and track multi-step tasks | 🟡 Listed only |
| 8 | **ToolRetryMiddleware** | Automatically retry failed tool calls with exponential backoff | 🟡 Listed only |
| 9 | ➕ **PIIMiddleware** | Detect and redact, mask, hash or block personal data | ❌ Missing (only "PII detection" in the intro) |
| 10 | ➕ **ModelRetryMiddleware** | Automatically retry failed model calls with exponential backoff | ❌ Missing |
| 11 | ➕ **ContextEditingMiddleware** | Clear old tool results from context to save tokens | ❌ Missing |
| 12 | ➕ **LLMToolEmulator** | Use an LLM to fake tool outputs, for testing | ❌ Missing |
| 13 | ➕ **ToolErrorMiddleware** | Turn a tool exception into a message the model can react to (needs `langchain>=1.3.14`) | ❌ Missing |
| 14 | ➕ **ModelRefusalMiddleware** | Raise an error when the model returns a refusal (needs `langchain>=1.4.9`) | ❌ Missing |
| 15 | ➕ **ShellToolMiddleware**, **FilesystemFileSearchMiddleware** | Give the agent a shell session, or Glob/Grep file search | ❌ Missing |
| 16 | ➕ **Custom middleware** (`@before_model`, `@wrap_model_call`, `AgentMiddleware`...) | Write your own logic: logging, guardrails, retries | ❌ Missing |

> **Legend:** ✅ explained with code in the lectures · 🟡 only named or described in one line · ❌ not covered · ➕ added by me.
>
> **Checked against the LangChain docs on 2026-11-23** (built-in middleware, custom middleware, human-in-the-loop and guardrails pages): class names, parameters, defaults and the human-in-the-loop resume format. The docs don't say which hook each built-in class uses, so this table has no hook column.
>
> **Newer or niche middleware not detailed here:** `ModelRouterMiddleware` and `AutoModeMiddleware` (beta, `langchain>=1.5.0`), `ProviderToolSearchMiddleware`, provider-specific middleware (Anthropic, AWS, OpenAI), and Deep Agents middleware (`FilesystemMiddleware`, `SubAgentMiddleware`, `RubricMiddleware` from the `deepagents` package). See the [built-in middleware docs](https://docs.langchain.com/oss/python/langchain/middleware/built-in).

---

## 4. Covered in the lectures

### Common setup (used by every example)

```python
from dotenv import load_dotenv
load_dotenv()   # loads OPENAI_API_KEY

from langchain.agents import create_agent
from langchain.agents.middleware import (
    SummarizationMiddleware,
    HumanInTheLoopMiddleware,
)
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver   # memory / checkpointing
from langgraph.types import Command                     # to resume a paused agent
from langchain_core.messages import HumanMessage
```

> The lecturer said "langchain.checkpoint.memory" aloud. The actual class is `InMemorySaver` from **`langgraph.checkpoint.memory`**. The `Command` import from `langgraph.types` fixes the "Command not defined" error hit in Lecture 2.

**Why a checkpointer + `thread_id`?** The checkpointer stores conversation state. The `thread_id` in config identifies one conversation or user. Both middleware below need them: summarization needs persisted history, and human-in-the-loop needs to pause and resume the same run.

---

### 4.1 SummarizationMiddleware (Lecture 1)

**Problem:** In a chatbot the messages list keeps growing. That means more tokens, more cost, slower responses, and the risk of hitting the context limit.

**Solution:** Once the history reaches a threshold (the *trigger*), an LLM summarizes the old messages. The agent keeps only the **summary + the most recent messages**.

```
Before:  [m1][m2][m3][m4][m5][m6][m7][m8][m9][m10]        ← trigger reached (10)
After:   [SUMMARY of m1..m6] [m7][m8][m9][m10]             ← keep recent 4
```

**Useful for:** long-running chatbots, multi-turn dialogues with extensive history, and cases where context matters but full history is too costly.

| Parameter | Meaning |
|---|---|
| `model` | LLM that writes the summary. **Use a cheap model** (e.g. `gpt-4o-mini`) since it runs repeatedly |
| `trigger` | **When** to summarize: messages, tokens or fraction |
| `keep` | **How much recent context** stays un-summarized |

#### Trigger type A: message count
```python
agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[],
    checkpointer=InMemorySaver(),
    middleware=[
        SummarizationMiddleware(
            model="openai:gpt-4o-mini",
            trigger=("messages", 10),    # summarize when history hits 10 messages
            keep=("messages", 4),        # keep the 4 most recent
        ),
    ],
)

config = {"configurable": {"thread_id": "test1"}}   # one unique user/conversation
questions = ["What is 2 + 2?", "What is 10 multiplied by 5?",
             "What is 100 divided by 4?", "What is 15 minus 7?"]   # keep asking past 10 messages

for q in questions:
    response = agent.invoke({"messages": [HumanMessage(content=q)]}, config)
    print(response["messages"][-1].content)
    print("Message count:", len(response["messages"]))
```
**Observed:** count grows 2 → 4 → 6 → 8 → 10. At 10 the middleware fires and the count **drops** (e.g. to ~6). The first message becomes *"Here is a summary of the conversation to date: ..."*.

#### Trigger type B: token count
```python
@tool
def search_hotels(city: str) -> str:
    """Search hotels in a city."""
    return f"Hotels in {city}: [hard-coded list, imagine an API response]"

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[search_hotels],
    checkpointer=InMemorySaver(),
    middleware=[
        SummarizationMiddleware(
            model="openai:gpt-4o-mini",
            trigger=("tokens", 550),     # summarize when history > 550 tokens
            keep=("tokens", 200),        # keep the most recent ~200 tokens
        ),
    ],
)

def count_tokens(messages):              # rule of thumb: 4 characters ≈ 1 token
    return sum(len(m.content) for m in messages) // 4
```
Test with `"Find hotels in {city}"` for Paris, London, Tokyo, New York, Dubai, Singapore.
**Observed:** `149 → 302 → 456 → (over 550) → 396 → ... → 232`. Tokens shrink each time the threshold is crossed, and a new summary appears at the start of the history.

#### Trigger type C: fraction of the model's context window
```python
SummarizationMiddleware(
    model="openai:gpt-4o-mini",
    trigger=("fraction", 0.005),   # 0.5% of the model's context window
    keep=("fraction", 0.002),      # keep 0.2%
)
```
- Relative to the **model's maximum context size**. 0.005 × 128K ≈ 640 tokens. (The lecture said "160K", but 640 / 0.005 = 128K.)
- The tiny value is only so you can *see* it trigger. In production use something like `0.8`.
- Works across models with different context sizes without hard-coding token numbers.

| Trigger | Syntax | Best when |
|---|---|---|
| Messages | `("messages", 10)` | Simple chats with similar-sized messages |
| Tokens | `("tokens", 550)` | You want precise cost and limit control |
| Fraction | `("fraction", 0.8)` | You switch between models with different context sizes |

`keep` accepts the same three unit types. Use exactly one unit per `keep`.

**Beyond the lecture (from the docs):**
- If you omit `trigger`, nothing triggers automatically. The default is `keep=("messages", 20)`.
- `trigger` can combine conditions:
  - a **list** means OR: `[("tokens", 3000), ("messages", 6)]`
  - a **dict** means AND: `{"tokens": 4000, "messages": 10}`
  - a **list of dicts** means OR of ANDs
- Other parameters: `token_counter`, `summary_prompt` (customize how the summary is written), `trim_tokens_to_summarize=4000`.
- `max_tokens_before_summary` and `messages_to_keep` are deprecated; use `trigger` and `keep` as shown above.

---

### 4.2 HumanInTheLoopMiddleware (Lecture 2)

**Problem:** An **autonomous agent** works with little human involvement. That is risky when a mistake is costly. Example: an agent that buys stocks. One bad LLM decision means real financial loss.

**Solution:** Pause the agent right before a sensitive tool runs, and let a human **approve, edit or reject** the call.

```
User request ─► Agent decides to call a tool
                      │
                      ▼
               ⏸  PAUSE (interrupt)
                      │
              Human reviews the tool call
             ┌────────┼─────────┐
          Approve    Edit     Reject
             │         │         │
             ▼         ▼         ▼
        run as-is   run with   don't run;
                    changed    agent is told
                    arguments  it was rejected
```

**Useful for:** database writes, financial transactions, compliance workflows needing human oversight, and long conversations where human feedback guides the agent.

#### The two tools (an email agent)
```python
@tool
def read_email_tool(email_id: str) -> str:
    """Read an email by its ID."""
    return f"Email content for {email_id}: ..."

@tool
def send_email_tool(recipient: str, subject: str, body: str) -> str:
    """Send an email."""
    return f"Email sent to {recipient} with subject '{subject}'"
```
This is a mock. A real version would use SMTP. The point is the approval flow.

#### Create the agent with `interrupt_on`
```python
agent = create_agent(
    model="openai:gpt-4o",
    tools=[read_email_tool, send_email_tool],
    checkpointer=InMemorySaver(),          # required: stores the paused state
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email_tool": {                       # ask the human before sending
                    "allowed_decisions": ["approve", "edit", "reject"],
                },
                "read_email_tool": False,                  # safe, never interrupt
            }
        ),
    ],
)
```
- **Keys** of `interrupt_on` are tool names, which say *where* to interrupt.
- **`allowed_decisions`** lists what the human may choose.
- **`False`** means never interrupt for that tool.
- **`True`** means interrupt and allow all decisions.
- By default the agent pauses **once per model turn** with all pending tool calls. The optional `interrupt_mode="per_call"` pauses once per tool call instead (needs `langchain>=1.4.6`).

#### The flow: invoke, interrupt, resume
```python
config = {"configurable": {"thread_id": "test-approve"}}

result = agent.invoke(
    {"messages": [HumanMessage(
        content="Send an email to john@test.com with subject 'Hello' and body 'How are you'")]},
    config,
)

if "__interrupt__" in result:                       # agent paused, waiting for a human
    result = agent.invoke(
        Command(resume={"decisions": [{"type": "approve"}]}),
        config,                                     # SAME thread_id
    )
```
`Command(resume=...)` tells the workflow to continue from the pause point with the human's decision. The final messages contain a tool message (*"Email sent to ..."*) and the agent's closing reply.

> **Newer docs style:** the current docs invoke with `version="v2"`, which returns a `GraphOutput`. Pending reviews are then in `result.interrupts`; each has `interrupt.value["action_requests"]` (a list of `{name, args, description}`) so you can show the human what is about to run. Resume the same way, with `Command(resume=...)`, the same `config` and `version="v2"`. The lecture's `"__interrupt__" in result` check is the older, default return shape. If your installed version behaves differently, use whichever form your version returns.
> In batched mode (the default), `decisions` is a list whose order must match `action_requests`.

#### The three decisions
```python
# APPROVE: tool runs exactly as the agent proposed
Command(resume={"decisions": [{"type": "approve"}]})

# REJECT: tool is skipped. Use a new thread_id (e.g. "test-reject") for a fresh run.
# The result shows "User rejected the tool call". An optional "message" can say why.
Command(resume={"decisions": [{"type": "reject"}]})
Command(resume={"decisions": [{"type": "reject", "message": "Send to the team address instead."}]})

# EDIT: human fixes the arguments (e.g. a wrong recipient), then the tool runs
Command(resume={"decisions": [{
    "type": "edit",
    "edited_action": {
        "name": "send_email_tool",
        "args": {"recipient": "correct@gmail.com", "subject": "Test", "body": "Hello"},
    },
}]})
```

| Decision | Resume payload | What happens |
|---|---|---|
| Approve | `{"type": "approve"}` | Tool runs with the original args |
| Edit | `{"type": "edit", "edited_action": {...}}` | Tool runs with the human's args |
| Reject | `{"type": "reject"}` | Tool is skipped; agent is told the user rejected it |

**Pattern to remember:**
1. `invoke()`: the agent runs until a guarded tool, then **interrupts**
2. Check `"__interrupt__" in result`
3. `invoke(Command(resume=...), config)` with the **same thread_id**: the agent continues

#### Summarization vs Human-in-the-loop

| | SummarizationMiddleware | HumanInTheLoopMiddleware |
|---|---|---|
| Purpose | Compress long history | Get human approval for tool calls |
| Key config | `model`, `trigger`, `keep` | `interrupt_on` |
| Needs checkpointer? | Yes (history) | **Yes, required** (to pause and resume) |
| Needs `thread_id`? | Yes | **Yes, the same one when resuming** |

---

### 4.3 Mentioned only briefly in the lectures

These were named in the docs table or described in a sentence. The lecturer said "you know the pattern, so apply them the same way".

| Middleware | What the lecture said | Short description |
|---|---|---|
| **ModelCallLimitMiddleware** | "Limit the number of model calls to prevent excessive cost / infinite loops"; has thread-limit and run-limit options | Stops an agent that keeps calling the model. A *thread* limit counts across the whole conversation, a *run* limit counts within one invocation. |
| **ToolCallLimitMiddleware** | "Control tool execution by limiting call counts" | Caps how many times tools (all, or one named tool) can run, e.g. to stop a search tool being hammered. |
| **ModelFallbackMiddleware** | "If the API key or model isn't working, fall back to another model" | Takes an ordered list of backup models and tries them when the primary errors. Improves uptime. |
| **LLMToolSelectorMiddleware** | "Agent with many tools, most not relevant per query. Reduces token usage by filtering" | A cheap LLM first picks the few relevant tools, and only those are sent to the main model. |
| **TodoListMiddleware** | Listed in docs | Gives the agent a to-do list tool so it can plan and track a long multi-step task. |
| **ToolRetryMiddleware** | Listed in docs | Retries a failing tool call (e.g. a flaky API) a set number of times before giving up. |

Usage (parameters checked against the docs):
```python
from langchain.agents.middleware import (
    ModelCallLimitMiddleware, ToolCallLimitMiddleware, ModelFallbackMiddleware,
)

middleware=[
    ModelCallLimitMiddleware(thread_limit=10, run_limit=5, exit_behavior="end"),
    ToolCallLimitMiddleware(tool_name="search_hotels", run_limit=3),
    ModelFallbackMiddleware("openai:gpt-4o-mini", "anthropic:claude-haiku-5-5"),
]
```

**Parameters and defaults (from the docs):**

| Middleware | Key parameters |
|---|---|
| ModelCallLimitMiddleware | `thread_limit` (whole conversation), `run_limit` (one invocation); `None` means no limit. `exit_behavior="end"` (default) or `"error"` |
| ToolCallLimitMiddleware | `tool_name` (`None` = all tools), `thread_limit`, `run_limit` (at least one required). `exit_behavior="continue"` (default), `"error"` or `"end"`; `"end"` only works when limiting a single tool |
| ModelFallbackMiddleware | First model, then any number of extra models: `ModelFallbackMiddleware("model-a", "model-b", ...)` |
| LLMToolSelectorMiddleware | `model` (defaults to the agent's model), `max_tools`, `always_include=[...]` (not counted against `max_tools`), `system_prompt` |
| TodoListMiddleware | Optional `system_prompt` and `tool_description`. Adds a `write_todos` tool |
| ToolRetryMiddleware | `max_retries=2`, `tools=None` (all), `backoff_factor=2.0`, `initial_delay=1.0`, `max_delay=60.0`, `jitter=True`, `retry_on`, `on_failure="continue"` (or `"error"`, or a callable) |

```python
LLMToolSelectorMiddleware(model="openai:gpt-4o-mini", max_tools=3, always_include=["search"])
TodoListMiddleware()
ToolRetryMiddleware(max_retries=3, on_failure="error")
```

---

## 5. ➕ Important middleware missing from the lectures

The lecture intro lists "guardrails, PII detection, retries, fallbacks" as use cases but never shows them. These are the ones I'd add. Details below were checked against the LangChain docs.

### PIIMiddleware: protect personal data
**Purpose:** Detect personally identifiable information in messages and handle it before it reaches the model or your logs. A core compliance and guardrail tool.

- **Built-in types:** `email`, `credit_card`, `ip`, `mac_address`, `url`
- **Custom types:** pass your own name and a regex `detector`
- **Where it applies:** `apply_to_input=True` (default), `apply_to_output=False`, `apply_to_tool_results=False`

| Strategy | Effect |
|---|---|
| `redact` (default) | Replace with `[REDACTED_EMAIL]`-style placeholder |
| `mask` | Partially hide, e.g. `****-****-****-1234` |
| `hash` | Replace with a deterministic hash |
| `block` | Raise an exception when PII is detected |

```python
from langchain.agents.middleware import PIIMiddleware

middleware=[
    PIIMiddleware("email", strategy="redact", apply_to_input=True),
    PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),
    PIIMiddleware("api_key", detector=r"sk-[a-zA-Z0-9]{32}", strategy="block"),   # custom type
]
```

### ModelRetryMiddleware: retry failed model calls
**Purpose:** If an LLM call fails (rate limit, timeout, transient error), retry with **exponential backoff** instead of crashing the agent. It is the model-side twin of `ToolRetryMiddleware`.

Parameters: `max_retries=2`, `backoff_factor=2.0`, `initial_delay=1.0`, `max_delay=60.0`, `jitter=True`, `retry_on`, and `on_failure="continue"` (or `"error"`, or a callable) for what happens once retries run out.

```python
ModelRetryMiddleware(max_retries=3, backoff_factor=2.0, initial_delay=1.0)
```
Combine it with `ModelFallbackMiddleware`: retry the same model first, then move to the next one.

### ToolErrorMiddleware: let the model see tool errors
**Purpose:** When a tool raises an exception, your handler turns it into content for the model (or returns `None` to let the error propagate). Needs `langchain>=1.3.14`.

```python
ToolErrorMiddleware(
    on_error=lambda exc, req: f"{type(exc).__name__}" if isinstance(exc, ValueError) else None
)
```
`tools=[...]` restricts it to specific tools (default: all).

### ModelRefusalMiddleware: detect model refusals
**Purpose:** Raises `ModelRefusalError` (from `langchain_core.exceptions`) when the model returns a refusal content block. The docs pair it with a fallback model:
```python
middleware=[ModelFallbackMiddleware("model-a"), ModelRefusalMiddleware()]
```
Needs `langchain>=1.4.9`.

### ContextEditingMiddleware: trim tool output from context
**Purpose:** Another way to control context size. Instead of summarizing the whole conversation, it **clears old tool results** (often huge: search results, file dumps) and keeps the recent ones. It needs no extra LLM call, so it is cheaper than summarizing. Good for agents that call tools a lot.

```python
from langchain.agents.middleware import ContextEditingMiddleware, ClearToolUsesEdit

ContextEditingMiddleware(edits=[ClearToolUsesEdit(trigger=100000, keep=3)])
```
`ClearToolUsesEdit` parameters: `trigger=100000` (tokens), `keep=3` (recent tool results to keep), `clear_at_least=0`, `clear_tool_inputs=False`, `exclude_tools=()`, `placeholder="[cleared]"`.

### LLMToolEmulator: fake tools for testing
**Purpose:** Replace real tool execution with an LLM-generated plausible output, so you can test agent logic without hitting real APIs, databases or paid services.

```python
LLMToolEmulator(tools=["get_weather"], model="openai:gpt-4o-mini")
```
`tools=None` emulates every tool, and `tools=[]` emulates none.

### ShellToolMiddleware and FilesystemFileSearchMiddleware
**Purpose:** `ShellToolMiddleware` gives the agent a persistent shell session. `FilesystemFileSearchMiddleware(root_path=...)` adds Glob and Grep search tools over a folder.

> ⚠️ **Shell is not compatible with interrupts**, so you **cannot** put human-in-the-loop in front of it. Isolate it with a sandboxed `execution_policy` instead (`HostExecutionPolicy`, `DockerExecutionPolicy` or `CodexSandboxExecutionPolicy`).

> **Which ones matter most?** In my view (this part is opinion, not from the docs): **PIIMiddleware**, **ModelRetryMiddleware**, **ToolRetryMiddleware** and **ModelFallbackMiddleware** for production reliability and safety, and **ModelCallLimit / ToolCallLimit** for cost control.

---

## 6. ➕ Writing your own middleware

The lecture says middleware is for "logging, analytics, debugging, guardrails" but never shows how to write your own. In LangChain v1 you can use **decorated functions** (simple) or an **`AgentMiddleware` subclass** (several hooks, async versions, init-time config).

### Two kinds of hooks
| Kind | Hooks | Signature | Return |
|---|---|---|---|
| **Node-style** (run at one point) | `before_agent`, `before_model`, `after_model`, `after_agent` | `(state, runtime)` | a dict of state updates, or `None` to change nothing |
| **Wrap-style** (wrap a call) | `wrap_model_call`, `wrap_tool_call` | `(request, handler)` | the handler's result, or a different response |

There is also a `dynamic_prompt` decorator for generating system prompts. The docs name it but don't show its signature, so I don't show an example here.

### Node-style example: logging
```python
from langchain.agents.middleware import before_model, after_model

@before_model
def log_before_model(state, runtime):
    print(f"About to call the model with {len(state['messages'])} messages")
    return None          # None = don't change the state

@after_model
def log_after_model(state, runtime):
    print("Model replied:", state["messages"][-1].content)
    return None

agent = create_agent(model="openai:gpt-4o-mini", tools=[],
                     middleware=[log_before_model, log_after_model])
```

### Wrap-style example: retry (from the docs)
`handler(request)` runs the real model call. You decide whether to call it zero times (short-circuit), once, or several times (retry).
```python
from langchain.agents.middleware import wrap_model_call

@wrap_model_call
def retry(request, handler):
    for attempt in range(3):
        try:
            return handler(request)
        except Exception:
            if attempt == 2:
                raise
```
`wrap_tool_call` works the same way, with a tool-call request and a handler that returns a `ToolMessage` or `Command`.

### Class style (several hooks, init-time config)
```python
from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage

class MessageLimit(AgentMiddleware):
    def __init__(self, max_messages: int = 50):
        super().__init__()
        self.max_messages = max_messages

    @hook_config(can_jump_to=["end"])
    def before_model(self, state, runtime):
        if len(state["messages"]) >= self.max_messages:
            return {"messages": [AIMessage("Limit reached.")], "jump_to": "end"}
        return None
```
- **`jump_to`** (in the returned dict) can be `"end"`, `"tools"` or `"model"`. The hook must declare the target in `can_jump_to`.
- Async versions use an `a` prefix, e.g. `abefore_model`.
- Class attributes can also set `state_schema` (custom state fields) and `tools` (tools that ship with the middleware).

> The docs page shows the code but not the import lines. I wrote the imports from memory, so if one fails, check the [custom middleware docs](https://docs.langchain.com/oss/python/langchain/middleware/custom).

### Which hook for which job?
| Goal | Hook |
|---|---|
| Log, validate or trim messages before the LLM sees them | `before_model` |
| Check or block the LLM's output | `after_model` |
| Retry, switch model, or short-circuit around the LLM call | `wrap_model_call` |
| Retry or monitor tool execution | `wrap_tool_call` |
| Setup and teardown for the whole run | `before_agent` / `after_agent` |

---

## 7. Stacking middleware

You can attach any number of middleware as a comma-separated list. With `middleware=[m1, m2, m3]`, the `before_*` hooks run first to last, the `wrap_*` hooks nest like onion layers (m1 is the outermost), and the `after_*` hooks run **last to first**. Put critical middleware first.

```python
agent = create_agent(
    model="openai:gpt-4o",
    tools=[read_email_tool, send_email_tool],
    checkpointer=InMemorySaver(),
    middleware=[
        PIIMiddleware("email", strategy="redact"),          # 1. protect data first
        SummarizationMiddleware(model="openai:gpt-4o-mini",
                                trigger=("tokens", 4000),
                                keep=("messages", 6)),      # 2. keep context small
        ModelCallLimitMiddleware(run_limit=10),             # 3. cap cost
        HumanInTheLoopMiddleware(interrupt_on={             # 4. approve risky actions
            "send_email_tool": {"allowed_decisions": ["approve", "edit", "reject"]},
            "read_email_tool": False,
        }),
    ],
)
```
The order shown is a sensible illustration. Order can matter (e.g. redact PII *before* summarizing so the summary doesn't contain it). In the lecture only a single middleware was used per agent, so treat this combined example as a pattern, not something shown there.

---

## 8. Middleware in RAG (without tools)

**Question:** can you use middleware in a RAG app when no tool is called?

**Answer: yes, with one condition.** Middleware attaches to `create_agent`, not to a plain retriever-then-LLM chain. If your RAG runs through an agent, tools are optional. Lecture 1 already did this: its summarization agent had `tools=[]`.

### Which middleware needs tools?
| Works without tools | Needs tools |
|---|---|
| `SummarizationMiddleware` (long chat history) | `HumanInTheLoopMiddleware` |
| `PIIMiddleware` (user queries) | `ToolCallLimitMiddleware` |
| `ModelRetryMiddleware`, `ModelFallbackMiddleware` | `ToolRetryMiddleware`, `ToolErrorMiddleware` |
| `ModelCallLimitMiddleware` | `LLMToolSelectorMiddleware`, `TodoListMiddleware` |
| Custom logging and guardrail hooks | `LLMToolEmulator`, `ContextEditingMiddleware` (it clears old tool results) |

### RAG styles (from the LangChain retrieval docs)
| Style | How it works | Where middleware fits |
|---|---|---|
| **2-step RAG** | Retrieval always runs before generation. No tool involved | Wraps the generation step: PII, summarization, retry, fallback, call limits |
| **Agentic RAG** | The model decides when to search, using a retriever **tool** | All of the above, plus tool middleware: `ToolCallLimit` (stop endless searching), `ToolRetry`, `ContextEditing` (clear big retrieved chunks from old turns) |
| **Hybrid** | Mix of both, with query enhancement and answer validation | Same as above, plus custom hooks for validation |

### Example: 2-step RAG agent with no tools
Only middleware verified in the docs is used here.
```python
agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[],                                   # no retriever tool
    checkpointer=InMemorySaver(),
    middleware=[
        PIIMiddleware("email", strategy="redact"),                  # protect the user's question
        SummarizationMiddleware(model="openai:gpt-4o-mini",
                                trigger=("tokens", 4000),
                                keep=("messages", 6)),              # keep chat history small
        ModelRetryMiddleware(max_retries=3),                        # survive transient API errors
        ModelFallbackMiddleware("openai:gpt-4o-mini"),              # backup model if the primary fails
    ],
)
```

### Doing the retrieval itself in middleware (unverified)
The RAG and retrieval docs pages I checked do **not** show middleware doing the retrieval. This part is my inference, not documented: a `before_model` hook (it receives `state` and `runtime`) could search your vector store with the last user message and add the results to the prompt.

```python
# UNTESTED SKETCH, not from the docs. `vector_store` is assumed to exist.
from langchain.agents.middleware import before_model
from langchain_core.messages import HumanMessage

@before_model
def add_context(state, runtime):
    question = state["messages"][-1].content
    docs = vector_store.similarity_search(question, k=4)
    context = "\n\n".join(d.page_content for d in docs)
    return {"messages": [HumanMessage(content=f"Context:\n{context}")]}
```
Caveat: returning `messages` from a node-style hook updates state, so this would add the context to the saved history on every turn. A hook that changes only the prompt sent to the model (`wrap_model_call`, or the `dynamic_prompt` decorator) may fit better, but I could not verify those APIs. Check the [custom middleware docs](https://docs.langchain.com/oss/python/langchain/middleware/custom) before building on this.

---

## 9. Key takeaways

1. **Middleware = hooks inside the agent loop** (before/after agent, model, tools). Think airport checkpoints.
2. Attach with `create_agent(..., middleware=[mw1, mw2, ...])`. Learn the pattern once and every built-in works the same way.
3. **SummarizationMiddleware** controls context size and cost. Triggers: `messages`, `tokens`, `fraction`. `keep` sets what stays verbatim. Use a **cheap model** to summarize.
4. **HumanInTheLoopMiddleware** is the safety brake for critical tool calls. Decisions: **approve, edit, reject**. Resume with `Command(resume=...)`.
5. Both need a **checkpointer** (`InMemorySaver`) and a **`thread_id`**, and resuming must reuse the same `thread_id`.
6. Guard only **risky** tools with human-in-the-loop (send, write, pay). Leave read-only tools alone (`False`).
7. For **production**, add the pieces the lectures skipped: **PII protection, retries, fallbacks and call limits**.
8. When no built-in fits, **write custom middleware** with the decorators or `AgentMiddleware`.
9. Verify: watch the message or token count drop after summarization, and look for the interrupt in the result for human-in-the-loop (`__interrupt__` in the lecture, `result.interrupts` in the current docs).
10. **"Transformation"** means middleware changes what the model sees or returns (prompt, tool list, output), using rules you write. It is not prompt *optimization* like DSPy.
11. **RAG without tools** works as long as it runs through `create_agent`. Use the non-tool middleware (PII, summarization, retry, fallback, call limits). Doing the retrieval inside a hook is possible in principle but unverified.
