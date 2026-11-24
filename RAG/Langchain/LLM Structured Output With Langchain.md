# LLM Structured Output with LangChain

> **LangChain series · Lecture 6 of 7** — Previous: [Messages Types In Langchain](Messages%20Types%20In%20Langchain.md) · Next: [MiddleWare With Langchain](MiddleWare%20With%20Langchain.md)

> **One-line summary:** Structured output makes an LLM return its answer in a fixed schema (title, year, director…) instead of free text, so the result can be parsed and reused by other code. LangChain supports three ways to define the schema: **Pydantic**, **TypedDict**, and **dataclass**.

---

## 1. Comparison at a glance

| | **Pydantic** (`BaseModel`) | **TypedDict** | **dataclass** |
|---|---|---|---|
| Import | `from pydantic import BaseModel, Field` | `from typing_extensions import TypedDict, Annotated` | `from dataclasses import dataclass` |
| Declaration | `class X(BaseModel)` | `class X(TypedDict)` | `@dataclass` above `class X` |
| **Runtime validation** | **Yes** (wrong types raise errors) | **No** (hints only) | **No** (hints only) |
| Field descriptions | `Field(description="...")` | `Annotated[type, ..., "description"]` | Docstring and comments |
| Output you get back | Pydantic object (`result.title`) | Plain `dict` (`result["title"]`) | Dataclass object (`result.title`) |
| Nested structures | Yes | Yes | Yes |
| Optional fields | `Optional[float] = Field(None, ...)` | Via `Annotated` | Via defaults |
| Extra dependency | Pydantic | `typing_extensions` | None (standard library) |
| `include_raw=True` | Supported | Supported | Supported |
| **Pick it when** | You need strict, validated data | You want a lightweight plain dict | You want a simple typed container |

**All three plug into the same two APIs:**

```python
model.with_structured_output(Schema)              # on a chat model
create_agent(model=..., response_format=Schema)   # on an agent
```

---

## 2. Why structured output?

- By default an LLM answers in **free-form text**, which is hard to use programmatically.
- If you ask for an essay, or for movie details, you often want the response to **follow a known structure**.
- If the response matches a schema, it is **easy to parse** and can feed the **next step in a pipeline** (database, API, UI, another chain).
- Earlier lectures covered **messages** (system, human, AI, tool) in [Messages Types In Langchain](Messages%20Types%20In%20Langchain.md). This section builds on them.

---

# Part 1: Pydantic

## 3. Why Pydantic?

Pydantic models give you:

1. **Field validation.** Types are enforced, and a wrong type raises an error.
2. **Field descriptions.** These tell the LLM what each field means.
3. **Nested structures.** Models can contain other models and lists of them.

### 3.1 Step 1: Load the model

```python
import os
from langchain.chat_models import init_chat_model

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

model = init_chat_model("groq:qwen/qwen3-32b")   # Qwen3 32B, a reasoning model hosted on Groq
```

> **Gotcha:** the import is `init_chat_model` (singular), not `init_chat_models`.

### 3.2 Step 2: Define the schema

```python
from pydantic import BaseModel, Field

class Movie(BaseModel):
    """Details about a movie."""
    title: str     = Field(description="The title of the movie")
    year: int      = Field(description="The year the movie was released")
    director: str  = Field(description="The director of the movie")
    rating: float  = Field(description="The movie's rating out of 10")
```

- `BaseModel` is the base class for all Pydantic models. Inheriting from it gives you validation.
- `Field(description=...)` is **important for the LLM**. It tells the model *what to put in which field*.
- `Field` has other parameters too (hover over it in the IDE to see them), such as defaults and constraints.
- The type annotation (`str`, `int`, `float`) is what **Pydantic validates**.

### 3.3 Step 3: Attach the schema to the model

```python
model_with_structure = model.with_structured_output(Movie)
```

- This returns a **new runnable** (a `RunnableBinding`) that wraps the original model, with the schema attached.
- The original `model` is unchanged and has **no schema**.

### 3.4 Step 4: Compare the two outputs

**Without a schema** (`model`):

```python
model.invoke("Provide details of the movie Inception")
```

This returns a long free-text answer (reasoning plus a description of the plot). It is correct but hard to reuse programmatically.

**With a schema** (`model_with_structure`):

```python
response = model_with_structure.invoke("Provide details of the movie Inception")
print(response)
```

```text
Movie(title='Inception', year=2010, director='Christopher Nolan', rating=8.8)
```

- You get a real `Movie` object, so you can use `response.title`, `response.year`, and so on.
- The data comes from the LLM's training knowledge, but it is now **clean and structured**.

### 3.5 Field validation (the core Pydantic benefit)

| Field | Expected type | If the LLM returns something else |
|---|---|---|
| `title` | `str` | Validation error |
| `year` | `int` | Validation error |
| `director` | `str` | Validation error |
| `rating` | `float` (an int is also coerced) | Validation error for non-numeric values |

This guarantees that the data you pass downstream has the types you expect.

### 3.6 Getting the raw message and the parsed result: `include_raw=True`

```python
model_with_structure = model.with_structured_output(Movie, include_raw=True)

result = model_with_structure.invoke("Provide details about the movie Inception")
```

`result` is a **dict** containing:

| Key | Contents |
|---|---|
| `"raw"` | The original `AIMessage` from the model (including tool-call details) |
| `"parsed"` | The structured `Movie` object |
| `"parsing_error"` | `None` if parsing succeeded, otherwise the error |

```python
result["raw"]      # the raw AI message
result["parsed"]   # Movie(title='Inception', ...)
```

**Use it when** you want the structured data and also the raw response, for example to inspect token usage, debug, or log.

### 3.7 Nested structures

Real data is often hierarchical, such as a movie with many actors. Define a smaller model and use it inside the bigger one.

```python
from typing import List, Optional
from pydantic import BaseModel, Field

class Actor(BaseModel):
    name: str = Field(description="Actor's name")
    role: str = Field(description="Character the actor plays")

class MovieDetails(BaseModel):
    """Detailed information about a movie."""
    title: str            = Field(description="The movie title")
    year: int             = Field(description="The year the movie was released")
    cast: List[Actor]     = Field(description="List of the main actors")   # nested model
    genres: List[str]     = Field(description="List of genres")
    budget: Optional[float] = Field(None, description="Budget in millions USD")  # optional, default None
```

```python
model_with_structure = model.with_structured_output(MovieDetails)
response = model_with_structure.invoke("Provide details about the movie Inception")
```

**Example result**

```text
MovieDetails(
  title='Inception',
  year=2010,
  cast=[
    Actor(name='Leonardo DiCaprio',   role='Dom Cobb'),
    Actor(name='Joseph Gordon-Levitt', role='Arthur'),
    ...
  ],
  genres=['Science Fiction', 'Action', ...],
  budget=160.0       # = $160 million
)
```

**What to notice**

- `cast: List[Actor]` is a list of nested `Actor` objects, each with its own validated fields.
- `genres: List[str]` is a list of plain strings.
- `Optional[float] = Field(None, ...)` means the field is **optional** and defaults to `None` if the LLM cannot provide it.
- The `budget` description says "in millions USD", so the LLM returns `160.0` rather than `160000000`. **Descriptions steer the output.**

### 3.8 Workflow at a glance

```text
Define Pydantic schema ──► model.with_structured_output(Schema) ──► .invoke(prompt) ──► validated Schema object
        (BaseModel + Field)            (new runnable)                                    (use .field_name anywhere)
```

---

# Part 2: TypedDict

## 4. What is `TypedDict`?

- It comes from Python's built-in typing system (imported from `typing_extensions`).
- **At runtime it is just a normal dictionary.** The type hints exist for developers and tooling, and Python does not enforce them.
- That is why there is **no runtime validation**. If a field is annotated `str` but the model returns an `int`, nothing raises an error.

**Use `TypedDict` when:**
- You want a simple schema with minimal overhead.
- You don't need strict type enforcement.
- You want the result as a plain dict, for example to pass straight into JSON or other code.

### 4.1 Step 1: Imports

```python
from typing_extensions import TypedDict, Annotated
```

- `TypedDict` is the base class for the schema.
- `Annotated` attaches **metadata (a description)** to a field's type. It plays the role that `Field(description=...)` plays in Pydantic.

### 4.2 Step 2: Define the schema

```python
class MovieDict(TypedDict):
    """A movie with details."""
    title:    Annotated[str,   ..., "The title of the movie"]
    year:     Annotated[int,   ..., "The year the movie was released"]
    director: Annotated[str,   ..., "The director of the movie"]
    rating:   Annotated[float, ..., "The movie's rating out of 10"]
```

**Anatomy of `Annotated[type, default, description]`:**

| Part | Meaning |
|---|---|
| `str` / `int` / `float` | The expected type (a hint only, **not enforced**) |
| `...` (Ellipsis) | Default value slot. `...` means the field is required. This slot is optional and can be left out or left empty. |
| `"The title..."` | The description the LLM reads to understand the field |

> **Compare with Pydantic:** the class inherits from `TypedDict` instead of `BaseModel`. Descriptions go in `Annotated[...]` instead of `Field(...)`.

### 4.3 Step 3: Attach the schema and invoke

```python
model_with_typedict = model.with_structured_output(MovieDict)

response = model_with_typedict.invoke("Provide details of the movie Avengers")
print(response)
```

**Example result (a plain dict, not a class instance):**

```python
{
  "title": "Avengers",
  "year": 2012,
  "director": "Joss Whedon",
  "rating": 8.0
}
```

- Access fields with dictionary syntax: `response["title"]`. Pydantic uses `response.title`.
- The call is the same as with Pydantic: `with_structured_output(Schema)`, then `.invoke(...)`.

### 4.4 Nested structures with `TypedDict`

Nesting works the same way as with Pydantic, but with `TypedDict` classes. Validation is still **not enforced**.

```python
from typing import List

class Actor(TypedDict):
    name: Annotated[str, ..., "Actor's name"]
    role: Annotated[str, ..., "Character played"]

class MovieDetails(TypedDict):
    title:  Annotated[str,         ..., "Movie title"]
    year:   Annotated[int,         ..., "Release year"]
    cast:   Annotated[List[Actor], ..., "List of main actors"]
    genres: Annotated[List[str],   ..., "List of genres"]
    budget: Annotated[float,       ..., "Budget in millions USD"]
```

```python
model_nested = model.with_structured_output(MovieDetails)
model_nested.invoke("Provide details about the movie Inception")
```

The result contains all the details (title, year, cast list, genres, and a budget of about 160 million).

**Key point:** if you accidentally get an `int` where a `str` was expected (for example `director`), it is **accepted silently**. This is the trade-off for being lightweight.

### 4.5 Bonus: the `.profile` property of a model

`.profile` shows **what a model supports**.

```python
model.profile                      # works (the base chat model)
model_with_typedict.profile        # AttributeError
```

- `with_structured_output(...)` returns a `RunnableSequence`, which has **no `profile` attribute**. You get an error like `'RunnableSequence' object has no attribute 'profile'`.
- Call `.profile` on the **original model**.

**What it tells you (for the Qwen3 model used in the course):**

| Capability | Value in the lecture |
|---|---|
| Max input tokens | (listed in profile) |
| Max output tokens | (listed in profile) |
| Image input | `False` |
| Audio input | `False` |
| Video input | `False` |
| Reasoning output | `True` |
| Tool calling | `True` |

It is useful for checking a model's abilities, such as tool calling, reasoning, or multimodal input, before you build on it.

---

# Part 3: Data Classes

## 5. What is a `dataclass`?

- Built into Python since **version 3.7**, in the `dataclasses` module.
- A class mainly meant to **hold data**.
- The `@dataclass` decorator generates boilerplate for you (`__init__`, `__repr__`, and so on).
- It has **no built-in input validation**. Type hints are not enforced at runtime.

**Use a dataclass when** you want a clean, simple data container with no extra dependency (such as Pydantic), and you don't need strict validation.

### 5.1 The scenario: extract contact info

All three examples in this part use the same task. Given a text like:

```text
Extract contact info from: John Doe, john@example.com, (555) 123-4567
```

The agent should return structured fields: **name**, **email**, **phone**.

### 5.2 New approach: `response_format` on an agent

Instead of calling `with_structured_output` on the model, you pass the schema when **creating the agent**:

```python
from langchain.agents import create_agent

agent = create_agent(
    model="openai:gpt-5",            # the lecture used GPT-5 here (OpenAI API key)
    response_format=ContactInfo,     # schema the final answer must follow
)
```

Setup used in the lecture:

```python
import os
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
```

> Parts 1 and 2 used a Groq-hosted Qwen model. This part switched to **GPT-5** to show agent creation. Any chat model works.

- `tools=[...]` is optional. The lecture **removed the tools** because they weren't needed here.
- Whatever the agent does internally, its **final answer follows the schema**.

**Invoking the agent:**

```python
result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"
    }]
})

result["structured_response"]     # the parsed, structured output
```

| Key in `result` | Contents |
|---|---|
| `"messages"` | The conversation: the human message, then the AI message(s) |
| `"structured_response"` | The data in your schema, the part you usually want |

### 5.3 The same schema, three ways

**Pydantic (validated)**

```python
from pydantic import BaseModel

class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str
```

- Every field is **validated**. `name`, `email` and `phone` must be strings.
- `result["structured_response"]` is a `ContactInfo(...)` object:

```text
ContactInfo(name='John Doe', email='john@example.com', phone='(555) 123-4567')
```

**TypedDict (no validation)**

```python
from typing_extensions import TypedDict

class ContactInfo(TypedDict):
    name: str
    email: str
    phone: str
```

- The schema is still declared (strings), but **nothing is enforced**.
- `result["structured_response"]` is a plain **dict**:

```python
{'name': 'John Doe', 'email': 'john@example.com', 'phone': '(555) 123-4567'}
```

**dataclass (no validation, clean class)**

```python
from dataclasses import dataclass

@dataclass
class ContactInfo:
    """Contact information for a person."""
    name: str    # the name of the person
    email: str   # the email address
    phone: str   # the phone number
```

- Use the `@dataclass` decorator and declare fields as `name: type`.
- The output is a **`ContactInfo` dataclass instance**, accessed as `result["structured_response"].name`.

> **Tip:** in a dataclass, put the field meaning in a **docstring or comments**. This helps readability, and the docstring is also passed to the model as part of the schema description.

---

## 6. Quick recap

1. **Structured output** forces the LLM to answer in a schema you define.
2. Define the schema with **Pydantic**, **TypedDict**, or a **dataclass**. They are interchangeable.
3. Only **Pydantic** validates types at runtime.
4. Apply a schema with `model.with_structured_output(Schema)` or, on an agent, `create_agent(..., response_format=Schema)`.
5. Pydantic and dataclass give you **objects** (`.field`). TypedDict gives you a **dict** (`["field"]`).
6. Descriptions (`Field(description=...)`, `Annotated[..., "desc"]`, or docstrings) **guide the LLM** on what goes in each field.
7. `include_raw=True` returns `raw`, `parsed` and `parsing_error`. With an agent, read `result["structured_response"]`.
8. Nested models and `List[...]` fields work in all three.
9. `model.profile` shows model capabilities. It is not available on the structured-output runnable.

---

## 7. Up next

- **Short-term memory** and related agent topics.
- Controlling an agent's loop with [MiddleWare With Langchain](MiddleWare%20With%20Langchain.md).
