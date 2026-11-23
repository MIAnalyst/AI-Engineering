# Invoke, Streaming & Batch in LangChain

> **One-line summary:** There are three ways to call a model. `invoke` waits for the full answer, `stream` shows the answer piece by piece as it is generated, and `batch` sends many independent requests in parallel.

---

## 1. The Three Calling Modes at a Glance

| Method | What it does | Returns | Best for |
|---|---|---|---|
| `model.invoke(input)` | Sends **one** request and **waits** for the complete response | Full response object | Simple scripts, short answers |
| `model.stream(input)` | Sends **one** request and gives output **chunk by chunk** as it is generated | Iterator (generator) of chunks | Chatbots, long responses, better UX |
| `model.batch([inputs])` | Sends **many independent** requests **in parallel** | List of responses | Processing many inputs at once, better performance and lower cost |

---

## 2. `invoke`: Wait for the Full Response

```python
model.invoke("Write me a 200 words paragraph on artificial intelligence")
```

- The call **blocks** until the LLM has finished generating the entire text.
- Only after that does the output appear.
- For a long answer, the user just waits and sees nothing.

```
Request ──► ... waiting ... waiting ... ──► [ FULL RESPONSE shown at once ]
```

---

## 3. `stream`: Show Output as It Is Generated

### Why streaming?
- Most models can **stream output while it is being generated**.
- Showing output progressively **significantly improves user experience**, especially for **long responses**.
- If you build chatbots for a company, you will use streaming most of the time.

```
Request ──► token ► token ► token ► token ► ...   (each shown immediately)
```

### How it works
`model.stream(...)` returns an **iterator (generator object)** that yields **chunks** of the output.

```python
model.stream("Write me a 200 words paragraph on artificial intelligence")
# <generator object ...>   <- just a generator, nothing is shown yet
```

Simply calling it only creates the generator. You must **loop over it** to read the chunks.

### Displaying the stream

```python
for chunk in model.stream("Write me a 200 words paragraph on artificial intelligence"):
    print(chunk.text)
```

The text appears progressively instead of all at once.

### Making it more readable: `end` and `flush`

```python
for chunk in model.stream("Why do parrots have colorful feathers?"):
    print(chunk.text, end="|", flush=True)
```

| Argument | Purpose |
|---|---|
| `end="|"` | Uses a **delimiter** after each chunk so you can **see where each chunk starts and ends** (the default is a newline) |
| `flush=True` | Forces Python to **print immediately**, so you see chunks appear in real time |

> The `|` delimiter is just for the demo. In a real app you would use `end=""` so the text flows naturally.

### Stream vs. invoke on the same question

| | `invoke` | `stream` |
|---|---|---|
| User sees output | Only after **everything** is generated | **While** it is being generated |
| Waiting experience | Blank wait, then full answer | Text starts appearing straight away |
| Total generation time | About the same | About the same, but **feels much faster** |

---

## 4. `batch`: Many Requests in Parallel

### What is it?
A **batch** is a collection of **independent requests** to a model. Because they are processed **in parallel**, batching can **improve performance and reduce cost**.

### When to use it
When you have **multiple inputs** and want all the answers at once.

```python
responses = model.batch([
    "Why do parrots have colorful feathers?",
    "How do airplanes fly?",
    "What is quantum computing?",
])
```

- Three questions are sent **together**, in parallel.
- You get **three responses at once**, as a list.
- The requests are **independent**, so no question depends on another.

```
                ┌─► Q1 ─► A1 ─┐
List of inputs ─┼─► Q2 ─► A2 ─┼─► List of outputs
                └─► Q3 ─► A3 ─┘     (processed in parallel)
```

### Controlling parallelism: `max_concurrency`

You can limit how many requests run **at the same time** using the `config` parameter:

```python
responses = model.batch(
    [
        "Why do parrots have colorful feathers?",
        "How do airplanes fly?",
        "What is quantum computing?",
    ],
    config={"max_concurrency": 5},
)
```

**Example:** with 10 questions and `max_concurrency=5`, the model processes **5 at a time**. Once the first 5 are done it moves on to the next 5. This helps avoid overloading the API or hitting rate limits.

---

## 5. Which One Should I Use?

```
Need ONE answer, don't care about UX?          → invoke
Need ONE answer, want to show it live (chat)?  → stream
Need answers to MANY independent inputs?       → batch
```

| Scenario | Use |
|---|---|
| Quick script or simple call | `invoke` |
| Chatbot / long answers / better UX | `stream` |
| Many independent prompts, process fast and cheaply | `batch` |
| Batch with API limits | `batch` + `max_concurrency` |

---

## 6. Cheat Sheet

```python
# 1. Invoke: wait for the full response
response = model.invoke("Write a 200 word paragraph on AI")

# 2. Stream: print chunks as they arrive
for chunk in model.stream("Write a 200 word paragraph on AI"):
    print(chunk.text, end="", flush=True)

# 3. Batch: many inputs in parallel
responses = model.batch(
    ["Why do parrots have colorful feathers?",
     "How do airplanes fly?",
     "What is quantum computing?"],
    config={"max_concurrency": 5},   # optional cap on parallel calls
)
```

---

## 7. Key Takeaways

1. `invoke` **waits** for the complete answer, `stream` shows it **progressively**, and `batch` handles **many inputs in parallel**.
2. `stream` returns a **generator**. You must **loop** over it and print `chunk.text`.
3. Use `print(..., end="", flush=True)` to display streamed text smoothly in real time.
4. Streaming greatly improves **user experience**, especially for **long responses**. It is the standard choice for chatbots.
5. `batch` is for **independent** requests. It improves performance and reduces cost through parallel processing.
6. `config={"max_concurrency": N}` limits how many batch calls run **at once**.

---

## 8. What's Next

- **Tools creation:** building custom tools for models and agents.
