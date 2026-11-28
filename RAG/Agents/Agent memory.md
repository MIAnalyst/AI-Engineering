# Agent Memory (Checkpointers)

> **Agents series · Lecture 2** — Previous: [Agents And ReACT Agent Architecture](Agents%20And%20ReACT%20Agent%20Architecture.md)

**Goal:** give the ReAct agent a **memory**, so that a follow-up question like "divide *that* by 5" knows what "that" is. In LangGraph, memory comes from a **checkpointer**: it saves the graph's state after every step, under a **thread ID**, and loads it again on the next call with the same thread ID.

This note covers:
- **The problem (§1):** why the agent from the last lecture forgets everything between calls.
- **The concepts (§2–3):** checkpointers, super-steps, checkpoints and threads.
- **The implementation (§4–8):** adding memory, inspecting it, saving it to a database, and the `create_agent` shortcut.

---

## 1. The problem: the agent forgets

We reuse the ReAct graph from [the previous lecture](Agents%20And%20ReACT%20Agent%20Architecture.md) (§10 there): an LLM node with tools (arXiv, Wikipedia, Tavily, `add`, `multiply`, `divide`) and the `tools → LLM` loop.

Ask one question, then a follow-up:

```python
messages = graph.invoke({"messages": [HumanMessage(content="What is 5 plus 8?")]})
messages["messages"][-1].content
# 'The sum of 5 and 8 is 13.'

messages = graph.invoke({"messages": [HumanMessage(content="Divide that by 5")]})
messages["messages"][-1].content
# 'It looks like you want to perform a division, but you have not provided the number
#  to be divided by 5. Could you please specify the number?'
```

The second call has **no idea** that "that" means 13.

**Why:** each `graph.invoke(...)` starts with a **fresh, empty state**. The state, which holds the message list, exists only while one run is executing. When the run ends, the state is thrown away, so the next run sees only "Divide that by 5".

To fix this, the graph needs a place to **keep the state between runs**. That place is a checkpointer.

## 2. The fix: a checkpointer

From the LangGraph docs: LangGraph can use a **checkpointer** to automatically save the graph state after each step. This built-in persistence layer gives the graph memory: it can pick up from its last saved state.

All you have to do is **compile the graph with a checkpointer**:

```python
graph_memory = builder.compile(checkpointer=InMemorySaver())
```

The simplest checkpointer is **`InMemorySaver`**. It keeps everything in your computer's RAM as key-value data (the lecture calls it `MemorySaver`, see §4.1).

## 3. Key concepts: super-step → checkpoint → thread

The lecture builds these up from the bottom.

### 3.1 Super-step: one "tick" of the graph

A **super-step** is one tick of the graph, in which every node scheduled for that tick runs.

- **Sequential nodes** each get their **own** super-step: node 1 runs, then node 2 runs.
- **Parallel nodes** **share** one super-step: they run in the same tick and see the same state.

```mermaid
flowchart LR
    subgraph Sequential["Sequential: one super-step per node"]
        direction LR
        S1([START]) --> A1["node 1<br/>(super-step 1)"] --> B1["node 2<br/>(super-step 2)"] --> E1([END])
    end
```

```mermaid
flowchart LR
    subgraph Parallel["Parallel: nodes 2 and 3 share a super-step"]
        direction LR
        S2([START]) --> A2["node 1<br/>(super-step 1)"]
        A2 --> B2["node 2<br/>(super-step 2)"]
        A2 --> C2["node 3<br/>(super-step 2)"]
        B2 --> E2([END])
        C2 --> E2
    end
```

### 3.2 Checkpoint: a snapshot saved after each super-step

A **checkpoint** is a **snapshot of the state** taken at the end of each super-step. It contains the state plus metadata. Inspecting one returns a `StateSnapshot` with these main fields:

| Field | What it holds |
|---|---|
| `values` | The state itself, e.g. the full `messages` list |
| `next` | Which node(s) will run next. `()` means the run is finished |
| `config` | The `thread_id` and this checkpoint's `checkpoint_id` |
| `metadata` | Details such as `step` (the super-step counter) and what each node wrote |
| `created_at` | When the checkpoint was saved |
| `parent_config` | The previous checkpoint, so the snapshots form a history |

**Example from the docs:** for a sequential graph `START → A → B → END`, one run saves **4 checkpoints**:
1. An empty one, with `START` as the next node.
2. The user input, with `A` next.
3. The state after `A`, with `B` next.
4. The state after `B`, with nothing next (finished).

### 3.3 Thread: the conversation that owns the checkpoints

A **thread** is a collection of checkpoints, identified by a **`thread_id`**. The `thread_id` is the key the checkpointer uses to save and load state:

- **Same `thread_id`** on the next call → the graph loads that thread's latest checkpoint and continues the conversation.
- **New `thread_id`** → the graph starts with an empty state.

In a real app you'd give each conversation its own thread ID. The lecture's example is one thread per logged-in user, so everything that user says is remembered.

```mermaid
flowchart TD
    T["thread_id = '1'<br/>(one conversation)"] --> C1["checkpoint<br/>after super-step 1"]
    T --> C2["checkpoint<br/>after super-step 2"]
    T --> C3["checkpoint …<br/>(latest)"]
    C3 --> V["state (messages) + metadata"]
```

> **Putting it together:** as the graph runs, the checkpointer saves a checkpoint after every super-step, under the current thread. On the next call with the same `thread_id`, the graph starts from the latest checkpoint, so the LLM sees the earlier messages and can answer "divide *that* by 5".

---

## 4. Implementation

### 4.1 What changed since the lecture was recorded

| Lecture used | Now |
|---|---|
| `from langgraph.checkpoint.memory import MemorySaver` | `from langgraph.checkpoint.memory import InMemorySaver`. `MemorySaver` still works, but the code keeps it only "for backwards compatibility" |
| Switched to `ChatOpenAI` because Groq's Qwen model "didn't work with MemorySaver" | Keep any tool-calling model, e.g. `ChatGroq(model="llama-3.3-70b-versatile")`. See §5 |

Nothing extra to install: `InMemorySaver` ships with `langgraph`.

### 4.2 Compile the graph with a checkpointer

Use the same `builder` as in the previous lecture: the same state, LLM node, `ToolNode`, edges and `tools → tool_calling_llm` loop. Only the **compile** line changes:

```python
from langgraph.checkpoint.memory import InMemorySaver

memory = InMemorySaver()
graph_memory = builder.compile(checkpointer=memory)
```

From now on, every super-step of `graph_memory` saves a checkpoint into `memory`.

### 4.3 Choose a thread

Pass the thread ID in a **config** dictionary:

```python
config = {"configurable": {"thread_id": "1"}}
```

### 4.4 Have a conversation

Pass the **same `config`** to every call that belongs to the same conversation:

```python
from langchain_core.messages import HumanMessage

messages = graph_memory.invoke({"messages": [HumanMessage(content="Add 12 and 13")]}, config=config)
messages = graph_memory.invoke({"messages": [HumanMessage(content="Add that number to 25")]}, config=config)
messages = graph_memory.invoke({"messages": [HumanMessage(content="Then multiply that number by 2")]}, config=config)

for m in messages["messages"]:
    m.pretty_print()
```

| Turn | You send | Tool call the LLM makes | Answer | Messages in the thread |
|---|---|---|---|---|
| 1 | "Add 12 and 13" | `add(12, 13)` | 25 | 4 |
| 2 | "Add that number to 25" | `add(25, 25)` | 50 | 8 |
| 3 | "Then multiply that number by 2" | `multiply(50, 2)` | 100 | 12 |

- You send **only the new message** each time. The checkpointer loads the earlier ones, and the `add_messages` reducer appends the new message to them.
- The message count grows with every turn: Human → AI (tool call) → Tool → AI (answer) is 4 messages per turn. The LLM sees all of them, which is how it knows what "that number" means.

### 4.5 A different thread has its own memory

```python
config2 = {"configurable": {"thread_id": "2"}}
graph_memory.invoke({"messages": [HumanMessage(content="Add that number to 25")]}, config=config2)
# → asks which number you mean: thread "2" has never seen 12 + 13
```

Threads are separate conversations. User A's thread never leaks into user B's.

### 4.6 Look inside the memory

```python
snapshot = graph_memory.get_state(config)     # latest checkpoint of thread "1"
snapshot.values["messages"]                   # the whole conversation so far
snapshot.next                                 # () → the last run finished

history = list(graph_memory.get_state_history(config))   # every checkpoint, newest first
len(history)                                  # 15 after the three turns above

{c.config["configurable"]["thread_id"] for c in memory.list(None)}   # all threads: {'1', '2'}
```

`get_state_history` is what makes **time travel** possible: you can go back to an earlier checkpoint and re-run from there. That's covered in a later lecture.


## 5. Production: save memory in a database

`InMemorySaver` keeps checkpoints in **RAM**, so **everything is lost when the program restarts**. Its own docstring says to use it only for debugging or testing. For real apps, use a checkpointer backed by a database. The graph code stays the same; only the checkpointer object changes.

| Database | Package | Class | Use for |
|---|---|---|---|
| SQLite | `langgraph-checkpoint-sqlite` | `SqliteSaver` | Local development, a single file on disk |
| PostgreSQL | `langgraph-checkpoint-postgres` | `PostgresSaver` | Production. LangSmith uses this one itself |
| MongoDB | `langgraph-checkpoint-mongodb` | `MongoDBSaver` | Production |
| Azure Cosmos DB | `langchain-azure-cosmosdb` | `CosmosDBSaver` | Production on Azure |

The LangGraph docs list the four above. Savers for **MySQL** (`langgraph-checkpoint-mysql`, from an independent developer) and **Redis** (`langgraph-checkpoint-redis`, from Redis Inc.) also exist, but other maintainers publish them.

### 5.1 SQLite (survives restarts)

```bash
pip install -U langgraph-checkpoint-sqlite
```

```python
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
graph_memory = builder.compile(checkpointer=SqliteSaver(conn))

config = {"configurable": {"thread_id": "user-42"}}
```

The conversation is now stored in `checkpoints.db`. Stop the program, start it again, and the same `thread_id` continues where it left off.

### 5.2 PostgreSQL (production)

```bash
pip install -U langgraph-checkpoint-postgres "psycopg[binary]"
```

```python
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://user:password@localhost:5432/mydb"

with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()                     # first time only: creates the tables
    graph_memory = builder.compile(checkpointer=checkpointer)
    graph_memory.invoke({"messages": [HumanMessage(content="Add 12 and 13")]}, config=config)
```

- `from_conn_string` is a **context manager**: use it with `with`, as above. Some doc snippets assign it to a variable directly, which doesn't give you a working saver.
- Keep `thread_id` values **under 255 characters**. Postgres stores them in a column of limited length.

### 5.3 Things to watch

- **Every turn re-sends the whole conversation to the LLM.** The LLM node calls `llm_with_tools.invoke(state["messages"])`, and that list keeps growing, so long chats cost more tokens. Long conversations eventually need trimming or summarising.
- **Checkpoints pile up.** Every super-step adds one: 15 after only three turns above. In a database, delete old checkpoints from time to time, for example with a cleanup job that removes those older than N days.

## 6. Shortcut: memory with `create_agent`

The same works with the one-line agent from the previous lecture. Pass a `checkpointer` and use a `thread_id`:

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(model=llm, tools=tools, checkpointer=InMemorySaver())

config = {"configurable": {"thread_id": "1"}}
agent.invoke({"messages": [HumanMessage(content="Add 12 and 13")]}, config)
agent.invoke({"messages": [HumanMessage(content="Add that number to 25")]}, config)   # → 50
```

## 7. Short-term vs. long-term memory

What this lecture built is **short-term memory**: it remembers *one conversation* (one thread). LangGraph also has **long-term memory**, called a **store**, for facts that should survive across conversations.

| | Checkpointer (this lecture) | Store |
|---|---|---|
| Saves | Snapshots of the graph state | Your own key-value data |
| Scope | One thread | Across all threads |
| Memory type | Short-term | Long-term |
| Used for | Continuing a conversation, human-in-the-loop, time travel, recovering from errors | User preferences, facts about the user, shared knowledge |
| How you use it | Pass a `thread_id` in the config | Read and write items from nodes or your own code |

Example: "my name is Sam" said in Monday's chat is in Monday's thread only. A store would let Tuesday's new chat know it too.

### How long-term memory (a store) works

A **store** is a separate, permanent place where the agent keeps facts about a user. It sits **outside** any single conversation, so every thread can read from it and write to it. You compile the graph with **both**: the checkpointer remembers the current conversation, and the store remembers the user.

```mermaid
flowchart LR
    subgraph Mon["Monday: thread 1"]
        M["User: I'm vegetarian"]
    end
    subgraph Tue["Tuesday: thread 2 (a new chat)"]
        T["User: suggest a dinner recipe"]
    end
    S[("Store<br/>namespace: ('user-42', 'memories')<br/>• is vegetarian")]
    M -- "save" --> S
    S -- "recall" --> T
```

**1. How memories are organised.** Each memory is a small JSON document saved under:
- a **namespace**, like a folder, which usually contains the user ID, e.g. `('user-42', 'memories')`, so each user's memories stay separate; and
- a **key**, like a file name, which identifies one memory inside that folder.

So the **thread ID** decides *which conversation* this is, and the **user ID** in the namespace decides *whose memories* to use.

**2. How the agent uses the store (each turn).**
1. **Recall:** a node looks up the user's memories, e.g. the ones most relevant to the latest message, and adds them to the LLM's prompt.
2. **Answer:** the LLM replies using the conversation (from the checkpointer) *and* the memories (from the store).
3. **Save:** when something worth remembering comes up ("I'm vegetarian"), it is written to the store under the user's namespace.

In a brand-new chat next week, step 1 finds "vegetarian" again, because it was never tied to a thread.

**3. Finding the right memories.** You can list everything in a namespace or filter by content. You can also give the store an embedding model and search **by meaning** ("what does the user like to eat?"). That is the same idea as retrieval in RAG.

**4. What to remember.** The LangGraph docs borrow three types from human memory:

| Type | What is stored | Agent example |
|---|---|---|
| **Semantic** | Facts | "User is vegetarian", "User prefers short answers" |
| **Episodic** | Experiences | Past tasks the agent did well, reused as examples in the prompt |
| **Procedural** | Rules and instructions | The agent's own system prompt, improved over time from user feedback |

**5. How to keep facts.** There are two ways:
- **Profile:** one JSON document per user that is rewritten each time. It's easy to hand to the LLM whole, but it gets error-prone as it grows.
- **Collection:** many small memories that keep being added. Less information gets lost, but the agent must also update or delete old memories and search through them.

**6. When to save.** There are two approaches:
- **During the conversation ("hot path"):** the agent decides to save a memory before it replies, often with a "save memory" tool. ChatGPT's memory works this way. New memories are available immediately, but each reply gets slower and the agent has more to juggle.
- **In the background:** a separate job reads conversations later and extracts the memories. Replies stay fast, but a new memory reaches other chats only after the job runs.

**7. Where it's stored.** `InMemoryStore` is for learning and tests, and is lost on restart, just like `InMemorySaver`. For production, use a database-backed store such as `PostgresStore`, `MongoDBStore`, `RedisStore` or `UpstashStore`. When an agent is deployed on LangSmith, a store is provided automatically.

### Code: checkpointer vs. store side by side

A small chatbot that uses **both** kinds of memory. To keep the difference easy to see, it saves to the store only when a message starts with `Remember:`. Anything else is just part of the conversation.

```python
import uuid
from dataclasses import dataclass
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.runtime import Runtime

llm = ChatGroq(model="llama-3.3-70b-versatile")

@dataclass
class Context:
    user_id: str                                   # who is talking (used by the store)

def chatbot(state: MessagesState, runtime: Runtime[Context]):
    namespace = (runtime.context.user_id, "memories")       # this user's "folder" in the store
    last_message = state["messages"][-1].content

    # SAVE: if the user asks to remember something, write it to the store
    if last_message.lower().startswith("remember:"):
        fact = last_message.split(":", 1)[1].strip()
        runtime.store.put(namespace, str(uuid.uuid4()), {"fact": fact})

    # RECALL: read all of this user's facts from the store
    facts = [item.value["fact"] for item in runtime.store.search(namespace)]
    system = SystemMessage(content=f"Known facts about this user: {facts}")

    # state["messages"] = this conversation so far, loaded by the checkpointer
    return {"messages": [llm.invoke([system] + state["messages"])]}

builder = StateGraph(MessagesState, context_schema=Context)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

checkpointer = InMemorySaver()     # short-term: remembers one conversation (thread)
store = InMemoryStore()            # long-term: remembers facts per user, across threads
graph = builder.compile(checkpointer=checkpointer, store=store)
```

Each call needs **two** identifiers: the `thread_id` tells the checkpointer *which conversation*, and the `user_id` tells the store *whose memories*.

```python
def chat(text, thread_id, user_id):
    result = graph.invoke(
        {"messages": [HumanMessage(content=text)]},
        config={"configurable": {"thread_id": thread_id}},   # → checkpointer: which conversation
        context=Context(user_id=user_id),                    # → store: whose memories
    )
    return result["messages"][-1].content
```

Now talk to it:

```python
# Monday: thread 1, user-42
chat("Remember: I am vegetarian", thread_id="1", user_id="user-42")    # saved to the STORE
chat("My favourite number is 7",  thread_id="1", user_id="user-42")    # only in the CONVERSATION
chat("What is my favourite number?", thread_id="1", user_id="user-42") # → "7"  (checkpointer)
chat("Suggest a dinner for me",      thread_id="1", user_id="user-42") # → vegetarian dish (store)

# Tuesday: thread 2 (a new chat), same user
chat("What is my favourite number?", thread_id="2", user_id="user-42") # → doesn't know (new thread)
chat("Suggest a dinner for me",      thread_id="2", user_id="user-42") # → vegetarian dish (store)

# A different user: thread 3, user-7
chat("Suggest a dinner for me",      thread_id="3", user_id="user-7")  # → any dish (no facts for user-7)
```

| What was said in thread 1 | Where it lives | Thread 1 later | Thread 2 (same user) | Thread 3 (other user) |
|---|---|---|---|---|
| "My favourite number is 7" | Checkpointer only (part of thread 1's messages) | ✅ remembered | ❌ forgotten | ❌ |
| "Remember: I am vegetarian" | Store, under `('user-42', 'memories')` | ✅ remembered | ✅ remembered | ❌ |

Look inside each memory:

```python
# Checkpointer: the messages of ONE conversation
len(graph.get_state({"configurable": {"thread_id": "1"}}).values["messages"])   # 8 (4 questions + 4 answers)
len(graph.get_state({"configurable": {"thread_id": "2"}}).values["messages"])   # 4 (a separate conversation)

# Store: the facts about ONE user, shared by all their threads
[item.value for item in store.search(("user-42", "memories"))]   # [{'fact': 'I am vegetarian'}]
store.search(("user-7", "memories"))                             # [] (nothing saved for this user)
```

Key points in the code:
- `compile(checkpointer=..., store=...)`: the graph gets both kinds of memory.
- `context_schema=Context` plus `context=Context(user_id=...)` passes the user ID into the graph. The node reads it, and the store, through the `runtime` parameter, which LangGraph fills in automatically.
- `runtime.store.put(namespace, key, value)` **saves** a memory, and `runtime.store.search(namespace)` **reads** that user's memories back.
- Here the agent saves only on an explicit `Remember:`. A real agent would decide what to save itself, during the conversation or in a background job (point 6 above).

---

## Summary

| Step | What | Code |
|---|---|---|
| 1 | Create a checkpointer | `memory = InMemorySaver()` |
| 2 | Compile the graph with it | `builder.compile(checkpointer=memory)` |
| 3 | Pick a thread for the conversation | `config = {"configurable": {"thread_id": "1"}}` |
| 4 | Pass the same config on every call | `graph_memory.invoke({...}, config=config)` |
| 5 | Inspect the memory | `graph_memory.get_state(config)`, `get_state_history(config)` |
| 6 | Survive restarts | `SqliteSaver` (dev), `PostgresSaver` (production) |

## Key takeaways

- Without a checkpointer, **every `invoke` starts from an empty state**, so the agent can't resolve "that number".
- A **checkpointer** saves a **checkpoint** (state plus metadata) after every **super-step**, grouped by **thread**.
- **Sequential** nodes get one super-step each. **Parallel** nodes share one.
- The **`thread_id`** is the key: the same ID continues a conversation, and a new ID starts a fresh one. Once a checkpointer is set, it is required on every call.
- `InMemorySaver` (formerly `MemorySaver`) is for learning and tests, because it forgets everything on restart. Use **SQLite** locally and **Postgres** (or MongoDB) in production.
- Memory works with any chat model. It just re-sends the saved messages, so long chats cost more tokens over time.
- This is **short-term** memory (one thread). Cross-conversation memory needs a **store**.

**Sources** (read 2026-11-28): LangGraph docs, [Persistence](https://docs.langchain.com/oss/python/langgraph/persistence), [Checkpointers](https://docs.langchain.com/oss/python/langgraph/checkpointers), [Stores](https://docs.langchain.com/oss/python/langgraph/stores) and [Memory overview](https://docs.langchain.com/oss/python/concepts/memory).
