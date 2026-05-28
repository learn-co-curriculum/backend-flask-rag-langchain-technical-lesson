# Technical Lesson — Building a Flask RAG Pipeline with LangChain, Chroma, and Ollama

## Introduction

LangChain helps you organize AI application workflows into reusable parts. In a RAG application, that means your backend can separate the route, retriever, prompt template, model call, output parser, response formatting, and verification metadata instead of mixing everything into one long function.

In this lesson, you will build a local Flask API that uses **LangChain**, **Chroma**, and **Ollama** to answer questions from approved reliability runbooks. You will produce a `POST /api/ask` endpoint within a backend engineering scenario using a seeded vector database, LangChain prompt template, local chat model, source-backed response, fallback behavior, and LangChain-specific debug metadata.

---

## Scenario

You are a junior backend developer on an internal platform team. The reliability team has runbooks for incidents such as checkout API error spikes, rollbacks, status page updates, API key exposure, rate limits, and export delays.

Engineers often ask questions like:

> “What should I do if checkout API errors spike right after a release?”

A general model might give generic incident advice, but your company needs an answer grounded in approved runbooks. Your task is to build a Flask endpoint that uses LangChain to connect this workflow:

```text
question
→ retrieve approved runbook chunks from Chroma
→ format context
→ fill a prompt template
→ call a local Ollama chat model
→ parse the answer
→ return answer + sources + LangChain debug metadata
```

---

## What You Will Build

By the end of this technical lesson, your API will return a response shaped like this:

```json
{
  "answer": "Pause additional deployments, prepare a rollback to the previous stable version, and notify the incident channel before rollback begins. After rollback, watch the 5xx rate, checkout latency, and order completion rate before lowering severity.",
  "sources": [
    {
      "source_id": "REL-101",
      "title": "Checkout API Error Spike Runbook",
      "category": "Reliability",
      "section": "Mitigation",
      "chunk_id": "chunk-rel-101-b",
      "distance": 0.1234
    }
  ],
  "langchain": {
    "chain_expression": "ChatPromptTemplate | ChatOllama | StrOutputParser",
    "components": {
      "vector_store": "Chroma",
      "retrieval_method": "similarity_search_with_score",
      "prompt_template": "ChatPromptTemplate",
      "chat_model": "ChatOllama",
      "output_parser": "StrOutputParser"
    },
    "retrieved_count": 3,
    "retrieved_chunk_ids": ["chunk-rel-101-b", "chunk-rel-102-a", "chunk-obs-144-a"],
    "top_k": 3,
    "context_characters": 1420,
    "fallback": false,
    "score_type": "Chroma distance; lower usually means closer in this lesson setup",
    "models": {
      "embedding": "embeddinggemma",
      "generation": "llama3.2"
    }
  }
}
```

The `answer` and `sources` fields are what a React frontend would usually display. The `langchain` field is included here for learning and debugging so you can see the value of the abstraction layer.

---

## What Is `ChatOllama`?

`ChatOllama` is LangChain’s chat model wrapper for Ollama. Instead of writing your own HTTP request to `http://localhost:11434/api/generate`, you create a model object that fits into a LangChain chain.

In this lesson, the chain expression is:

```python
prompt_template | llm | StrOutputParser()
```

That means:

```text
ChatPromptTemplate formats the messages.
ChatOllama sends the formatted messages to the local model.
StrOutputParser converts the model response into a plain string.
```

You are still using a real local model through Ollama. LangChain is organizing the model call so it can work with prompt templates, retrievers, output parsers, and other workflow components.

---

## Tools and Resources

You will use:

- Python 3.10
- Visual Studio Code
- Terminal or integrated terminal
- `pipenv`
- Flask
- LangChain
- LangChain Chroma integration
- LangChain Ollama integration
- Chroma as the local vector database
- Ollama running locally
- An embedding model, such as `embeddinggemma`
- A generation model, such as `llama3.2`
- `curl` for manual endpoint checks
- Optional: `pytest` for smoke tests after implementation

Set up the local Ollama models:

```bash
ollama pull embeddinggemma
ollama pull llama3.2
ollama run llama3.2 "Hello"
```

Install dependencies:

```bash
pipenv install
pipenv shell
```

You can also use `pip` if your environment does not use Pipenv:

```bash
pip install -r requirements.txt
```

---

## Project Structure

```text
m6-langchain-rag-technical-lesson/
├── app.py
├── seed_chroma.py
├── try_langchain_rag.py
├── planning_notes.md
├── Pipfile
├── requirements.txt
├── pytest.ini
├── data/
│   └── runbook_chunks.json
├── lib/
│   ├── __init__.py
│   ├── config.py
│   ├── documents.py
│   ├── langchain_rag_service.py
│   ├── prompt_templates.py
│   ├── response_formatter.py
│   ├── validation.py
│   └── vector_store.py
└── tests/
    ├── test_app.py
    ├── test_langchain_rag_service.py
    ├── test_response_formatter.py
    └── test_validation.py
```

---

# Instructions

Follow the technical process: **Identify → Assemble → Execute → Verify**.

---

## Step 1: Identify the RAG API goal and output contract

To start, we'll define what the backend needs to do before you choose LangChain components.

Open `planning_notes.md` and complete the first section.

Use this completed example as a guide:

```text
User or role:
Internal engineers and support staff who need approved reliability runbook guidance.

Business problem:
During incidents, engineers need fast answers from approved runbooks instead of generic AI advice.

Approved knowledge source:
Reliability runbook chunks stored in data/runbook_chunks.json and seeded into Chroma.

Endpoint route:
POST /api/ask

Manual RAG steps this replaces:
Validate question, retrieve context, build prompt, call model, parse output, return answer and sources.

Expected successful response fields:
answer, sources, langchain

Fallback behavior:
If no approved context is retrieved, return a safe fallback and do not call the model.

LangChain debug fields that would help a developer:
component names, chain expression, retrieved chunk IDs, retrieved count, top_k, context length, fallback status, model names.
```

### Why This Step Matters

LangChain should not hide the workflow from you. Before you build the chain, you should be able to explain what problem the backend solves and what output the frontend needs.

---

## Step 2: Assemble the manual-to-LangChain map

Next, let's connect the RAG steps you already know to LangChain components.

Review this component map:

| Manual RAG responsibility | LangChain-supported component |
|---|---|
| Store searchable chunks | `Document` objects + `Chroma` vector store |
| Create query/document embeddings | `OllamaEmbeddings` |
| Search for relevant chunks | Chroma vector store search or retriever |
| Format the prompt | `ChatPromptTemplate` |
| Call the local model | `ChatOllama` |
| Convert model output to text | `StrOutputParser` |
| Connect prompt → model → parser | Chain expression using `|` |
| Return answer and source metadata | Flask response formatter |

### Explanation

This lesson uses LangChain where it helps organize the workflow, but it keeps important backend responsibilities visible. Flask still validates requests. Chroma still stores and retrieves chunks. The response formatter still controls what the frontend receives.

---

## Step 3: Assemble the LangChain Chroma vector store

Next, let's configure the embedding model and vector store that LangChain uses for retrieval.

Open `lib/vector_store.py`.

Replace `build_embedding_model()` with:

```python
def build_embedding_model():
    """Create the LangChain Ollama embeddings wrapper."""
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
```

Replace `get_vector_store()` with:

```python
def get_vector_store():
    """Return the persistent Chroma vector store wrapped by LangChain."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PATH,
        embedding_function=build_embedding_model(),
    )
```

### Explanation

`OllamaEmbeddings` tells LangChain how to turn text into vectors. `Chroma` stores those vectors and the related runbook metadata. This is the same retrieval idea from earlier modules, but now the vector database is wrapped in a LangChain interface.

### Check Your Work

Your file should still import:

```python
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
```

---

## Step 4: Add retriever and seeding behavior

Now, we will create a retriever helper and seed the Chroma database with a new runbook dataset.

In `lib/vector_store.py`, replace `get_retriever()` with:

```python
def get_retriever(vector_store=None, top_k=DEFAULT_TOP_K):
    """Return a LangChain retriever for the Chroma vector store."""
    store = vector_store or get_vector_store()
    return store.as_retriever(search_kwargs={"k": top_k})
```

Replace `seed_vector_store()` with:

```python
def seed_vector_store(reset=True):
    """Create a fresh Chroma collection and add the lesson runbook documents."""
    if reset:
        reset_chroma_db()

    documents = load_runbook_documents()
    vector_store = get_vector_store()
    ids = [document.metadata["chunk_id"] for document in documents]

    vector_store.add_documents(documents, ids=ids)

    if hasattr(vector_store, "persist"):
        vector_store.persist()

    return len(documents)
```

### Explanation

The retriever helper shows the common LangChain pattern:

```python
vector_store.as_retriever(search_kwargs={"k": top_k})
```

The seed function loads runbook chunks, converts them into LangChain `Document` objects, and stores them in Chroma. Each document includes `page_content` and `metadata`, which matters later when the API returns sources.

### Execute the Seed File

Seed the database:

```bash
python seed_chroma.py
```

Expected output:

```text
Seeded 12 reliability runbook chunks into Chroma.
```

### Common Issue

If seeding fails with an Ollama connection error, make sure Ollama is running and the embedding model is available:

```bash
ollama pull embeddinggemma
ollama list
```

---

## Step 5: Assemble the prompt template

Next, we will replace a manual prompt string with a reusable LangChain prompt template.

Open `lib/prompt_templates.py`.

Replace `build_prompt_template()` with:

```python
def build_prompt_template():
    """Return the LangChain chat prompt template used by the chain."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_INSTRUCTIONS.strip()),
            ("human", HUMAN_TEMPLATE.strip()),
        ]
    )
```

### Explanation

A prompt template keeps the instruction structure reusable. Instead of manually building a long string in a route, the chain can fill `{context}` and `{question}` when it runs.

### Check Your Work

The human template should include both variables:

```text
{context}
{question}
```

If either variable is missing, the chain will not receive the retrieved context or the user’s question correctly.

---

## Step 6: Format sources and LangChain debug metadata

Now, we need to shape the API response so users can inspect sources and developers can inspect LangChain behavior.

Open `lib/response_formatter.py`.

Replace `format_sources()` with:

```python
def format_sources(scored_documents):
    """Return source metadata for retrieved LangChain documents."""
    sources = []
    seen_chunk_ids = set()

    for document, distance in scored_documents:
        metadata = document.metadata or {}
        chunk_id = metadata.get("chunk_id")

        if chunk_id in seen_chunk_ids:
            continue

        seen_chunk_ids.add(chunk_id)

        sources.append(
            {
                "source_id": metadata.get("source_id", "unknown"),
                "title": metadata.get("title", "Untitled"),
                "category": metadata.get("category", "Uncategorized"),
                "section": metadata.get("section", "Unspecified"),
                "chunk_id": chunk_id,
                "distance": round(float(distance), 4),
            }
        )

    return sources
```

Replace `format_langchain_debug()` with:

```python
def format_langchain_debug(
    *,
    scored_documents,
    context,
    top_k=DEFAULT_TOP_K,
    fallback=False,
):
    """Return LangChain-specific metadata for developer inspection."""
    retrieved_chunk_ids = [
        document.metadata.get("chunk_id", "unknown")
        for document, _distance in scored_documents
    ]

    return {
        "chain_expression": "ChatPromptTemplate | ChatOllama | StrOutputParser",
        "components": {
            "vector_store": "Chroma",
            "retrieval_method": "similarity_search_with_score",
            "prompt_template": "ChatPromptTemplate",
            "chat_model": "ChatOllama",
            "output_parser": "StrOutputParser",
        },
        "retrieved_count": len(scored_documents),
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "top_k": top_k,
        "context_characters": len(context or ""),
        "fallback": fallback,
        "score_type": "Chroma distance; lower usually means closer in this lesson setup",
        "models": {
            "embedding": EMBEDDING_MODEL,
            "generation": GENERATION_MODEL,
        },
    }
```

Replace `format_success_response()` with:

```python
def format_success_response(answer, sources, langchain_debug):
    """Return the successful API response body."""
    return {
        "answer": answer.strip(),
        "sources": sources,
        "langchain": langchain_debug,
    }
```

Replace `format_fallback_response()` with:

```python
def format_fallback_response(langchain_debug=None):
    """Return a safe response when there is not enough approved context."""
    return {
        "answer": FALLBACK_MESSAGE,
        "sources": [],
        "langchain": langchain_debug or {"fallback": True},
    }
```

### Explanation

The source formatter keeps full document text out of the API response while still returning enough metadata for review. The `langchain` debug object makes the abstraction visible: learners can see which components ran, how many chunks were retrieved, and whether the system used fallback behavior.

### Production Note

In a production app, you might hide debug metadata behind a developer flag. In this lesson, the metadata is intentionally visible because it helps you inspect the pipeline.

---

## Step 7: Execute the LangChain RAG service

Next, let's compose the prompt, model, parser, retrieval, fallback, and response formatting into one service function.

Open `lib/langchain_rag_service.py`.

Replace `build_chat_model()` with:

```python
def build_chat_model():
    """Create the LangChain Ollama chat model wrapper."""
    return ChatOllama(
        model=GENERATION_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=TEMPERATURE,
    )
```

Replace `build_chain()` with:

```python
def build_chain(prompt_template=None, llm=None):
    """Compose the prompt, model, and output parser into a LangChain runnable chain."""
    prompt_template = prompt_template or build_prompt_template()
    llm = llm or build_chat_model()

    return prompt_template | llm | StrOutputParser()
```

Replace `retrieve_context()` with:

```python
def retrieve_context(question, vector_store=None, top_k=DEFAULT_TOP_K):
    """Retrieve documents and distances from the LangChain Chroma vector store."""
    store = vector_store or get_vector_store()
    return store.similarity_search_with_score(question.strip(), k=top_k)
```

Replace `has_usable_context()` with:

```python
def has_usable_context(scored_documents):
    """Return True when retrieval produced at least one non-empty document."""
    for document, _distance in scored_documents:
        if document.page_content and document.page_content.strip():
            return True

    return False
```

Replace `format_context()` with:

```python
def format_context(scored_documents):
    """Format retrieved LangChain documents into a prompt-ready context block."""
    context_blocks = []

    for index, (document, distance) in enumerate(scored_documents, start=1):
        metadata = document.metadata or {}

        context_blocks.append(
            "\n".join(
                [
                    f"[Context {index}]",
                    f"Source ID: {metadata.get('source_id', 'unknown')}",
                    f"Title: {metadata.get('title', 'Untitled')}",
                    f"Category: {metadata.get('category', 'Uncategorized')}",
                    f"Section: {metadata.get('section', 'Unspecified')}",
                    f"Chunk ID: {metadata.get('chunk_id', 'unknown')}",
                    f"Distance: {float(distance):.4f}",
                    "Text:",
                    document.page_content.strip(),
                ]
            )
        )

    return "\n\n".join(context_blocks)
```

Replace `answer_question()` with:

```python
def answer_question(
    question,
    *,
    vector_store=None,
    chain=None,
    top_k=DEFAULT_TOP_K,
):
    """Run the LangChain-supported RAG workflow for one validated question."""
    cleaned_question = question.strip()

    try:
        scored_documents = retrieve_context(
            cleaned_question,
            vector_store=vector_store,
            top_k=top_k,
        )

        if not has_usable_context(scored_documents):
            debug = format_langchain_debug(
                scored_documents=[],
                context="",
                top_k=top_k,
                fallback=True,
            )
            return format_fallback_response(debug)

        context = format_context(scored_documents)
        rag_chain = chain or build_chain()

        answer = rag_chain.invoke(
            {
                "context": context,
                "question": cleaned_question,
            }
        )

        if not isinstance(answer, str) or not answer.strip():
            raise LangChainServiceError("The LangChain pipeline returned an empty answer.")

        sources = format_sources(scored_documents)
        debug = format_langchain_debug(
            scored_documents=scored_documents,
            context=context,
            top_k=top_k,
            fallback=False,
        )

        return format_success_response(answer, sources, debug)

    except LangChainServiceError:
        raise
    except Exception as exc:
        raise LangChainServiceError(str(exc)) from exc
```

This service is the main LangChain RAG pipeline. Notice the route does not appear here. This function can run inside Flask, but it can also run in a plain Python script.

The chain itself is short:

```python
prompt_template | llm | StrOutputParser()
```

The backend still owns the important application decisions around retrieval, fallback behavior, source formatting, and response shape.

---

## Step 8: Connect the LangChain service to Flask

Now, we'll expose the LangChain RAG service through `POST /api/ask`.

Open `lib/validation.py`.

Replace `validate_question_payload()` with:

```python
def validate_question_payload(payload):
    """Validate the JSON body for POST /api/ask.

    Return:
        (question, None) when valid
        (None, error_dict) when invalid
    """
    if not isinstance(payload, dict):
        return None, {
            "error": "invalid_request",
            "message": "Request body must be a JSON object.",
        }

    if "question" not in payload:
        return None, {
            "error": "missing_question",
            "message": "Request body must include a question field.",
        }

    question = payload["question"]

    if not isinstance(question, str):
        return None, {
            "error": "invalid_question",
            "message": "Question must be a string.",
        }

    question = question.strip()

    if not question:
        return None, {
            "error": "empty_question",
            "message": "Question cannot be blank.",
        }

    if len(question) < MIN_QUESTION_LENGTH:
        return None, {
            "error": "short_question",
            "message": f"Question must be at least {MIN_QUESTION_LENGTH} characters long.",
        }

    return question, None
```

Open `app.py`.

Replace the `/api/ask` route body with:

```python
    @app.post("/api/ask")
    def ask():
        """Accept a question and return a source-backed LangChain RAG response."""
        question, error = validate_question_payload(request.get_json(silent=True))

        if error:
            return jsonify(format_error_response(error["error"], error["message"])), 400

        try:
            response = answer_question(question)
            return jsonify(response), 200
        except LangChainServiceError as exc:
            return (
                jsonify(
                    format_error_response(
                        "langchain_service_error",
                        str(exc),
                    )
                ),
                502,
            )
```

The Flask route has a narrow job:

```text
parse request
validate question
call service
return JSON
handle service errors
```

It does not build prompts, retrieve documents, or call the model directly. That separation is what makes the backend easier to maintain.

---

## Step 9: Execute the app

Now, we can run the full local RAG stack.

Seed Chroma:

```bash
python seed_chroma.py
```

Run the Flask app:

```bash
flask --app app run --debug
```

In a second terminal, test the endpoint:

```bash
curl -i -X POST http://127.0.0.1:5000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What should I do if checkout API errors spike right after a release?"}'
```

### Expected Result

You should receive:

- HTTP `200`
- an `answer`
- at least one source from the reliability runbooks
- a `langchain` object showing the chain expression and retrieved chunk IDs

Your exact wording may differ because the generation model is probabilistic, but the answer should be grounded in the retrieved runbook context.

---

## Step 10: Verify the pipeline without Flask

Let's confirm that LangChain can run in plain Python as well as inside Flask.

### Action

Run:

```bash
python try_langchain_rag.py "When should we publish a status page update?"
```

### Expected Result

The script should print JSON with:

```text
answer
sources
langchain
```

### Explanation

Flask is the API boundary. LangChain is the AI workflow layer. The same service can run from a route, a CLI script, a scheduled job, or another backend component.

---

## Step 11: Verify behavior with smoke tests

Next, we'll use the prebuilt tests to check the core behavior after completing the lesson.

Run:

```bash
pytest
```

The included tests check that:

- invalid questions are rejected,
- source formatting returns metadata without full document text,
- LangChain debug metadata names the expected components,
- the service can run with fake retrieval and fake chain objects,
- fallback behavior does not call the model when no context exists,
- the Flask route returns a structured JSON response.

### Important Note

These tests use fakes where possible. They do not require a live Ollama model. Manual `curl` checks are still important because they verify the real local model and Chroma database.

---

## Step 12: Verify with multiple realistic questions

Try these questions:

```bash
curl -s -X POST http://127.0.0.1:5000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What should I do if checkout errors spike after a deployment?"}' | python -m json.tool
```

```bash
curl -s -X POST http://127.0.0.1:5000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "When do we use a P1 severity?"}' | python -m json.tool
```

```bash
curl -s -X POST http://127.0.0.1:5000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What should I do if an API key was committed to a public repo?"}' | python -m json.tool
```

```bash
curl -s -X POST http://127.0.0.1:5000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the cafeteria lunch menu today?"}' | python -m json.tool
```

For each response, check:

```text
Retrieval relevance:
Do the returned source titles match the question?

Prompt-context alignment:
Did the answer use information from the retrieved context?

Grounding:
Does the answer avoid unsupported claims?

Source attribution:
Can you inspect where the answer came from?

Fallback:
Does the app avoid pretending to know things outside the runbooks?

API response contract:
Would a React frontend be able to display answer, sources, and optional diagnostics?
```

---

## Step 13: Reflect on LangChain’s value and tradeoffs

Open `planning_notes.md` and complete the reflection section.

Use these questions:

```text
What became easier to reuse with LangChain?
What became easier to inspect because of the debug metadata?
What did LangChain hide or make less obvious?
Which parts still require developer judgment?
```

A strong reflection might say:

```text
LangChain made the prompt, model, and output parser easier to compose. The debug metadata made it easier to see the chain expression, component names, and retrieved chunk IDs. The abstraction also hides some data flow, so I still need to inspect retrieved context and source metadata when debugging. Developer judgment is still needed for prompt quality, retrieval quality, fallback rules, and deciding which fields the API should expose.
```

---

# Completed Code Reference

Use this section when you need to check your implementation.

## `lib/vector_store.py`

```python
"""LangChain + Chroma setup helpers for the technical lesson."""

import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from lib.config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
)
from lib.documents import load_runbook_documents


def build_embedding_model():
    """Create the LangChain Ollama embeddings wrapper."""
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


def get_vector_store():
    """Return the persistent Chroma vector store wrapped by LangChain."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PATH,
        embedding_function=build_embedding_model(),
    )


def get_retriever(vector_store=None, top_k=DEFAULT_TOP_K):
    """Return a LangChain retriever for the Chroma vector store."""
    store = vector_store or get_vector_store()
    return store.as_retriever(search_kwargs={"k": top_k})


def reset_chroma_db(path=CHROMA_PATH):
    """Delete the local Chroma directory so the lesson can reseed cleanly."""
    db_path = Path(path)
    if db_path.exists():
        shutil.rmtree(db_path)


def seed_vector_store(reset=True):
    """Create a fresh Chroma collection and add the lesson runbook documents."""
    if reset:
        reset_chroma_db()

    documents = load_runbook_documents()
    vector_store = get_vector_store()
    ids = [document.metadata["chunk_id"] for document in documents]

    vector_store.add_documents(documents, ids=ids)

    if hasattr(vector_store, "persist"):
        vector_store.persist()

    return len(documents)
```

## `lib/prompt_templates.py`

```python
"""Prompt templates for the LangChain RAG pipeline."""

from langchain_core.prompts import ChatPromptTemplate


SYSTEM_INSTRUCTIONS = """
You are a reliability runbook assistant for an internal engineering team.
Use only the approved context provided by the backend.
If the context does not contain enough information, say that you do not have enough approved context.
Do not invent incident procedures, policy requirements, compensation details, security steps, or root causes.
"""


HUMAN_TEMPLATE = """
Approved context:
{context}

Employee question:
{question}

Response requirements:
- Answer in 2-4 concise sentences.
- Base the answer only on the approved context.
- Mention when the answer depends on incident lead approval or verification.
- Do not include raw distance scores in the answer text.
"""


def build_prompt_template():
    """Return the LangChain chat prompt template used by the chain."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_INSTRUCTIONS.strip()),
            ("human", HUMAN_TEMPLATE.strip()),
        ]
    )
```

## `lib/response_formatter.py`

```python
"""Response formatting helpers for source-backed LangChain RAG output."""

from lib.config import (
    DEFAULT_TOP_K,
    EMBEDDING_MODEL,
    FALLBACK_MESSAGE,
    GENERATION_MODEL,
)


def format_sources(scored_documents):
    """Return source metadata for retrieved LangChain documents."""
    sources = []
    seen_chunk_ids = set()

    for document, distance in scored_documents:
        metadata = document.metadata or {}
        chunk_id = metadata.get("chunk_id")

        if chunk_id in seen_chunk_ids:
            continue

        seen_chunk_ids.add(chunk_id)

        sources.append(
            {
                "source_id": metadata.get("source_id", "unknown"),
                "title": metadata.get("title", "Untitled"),
                "category": metadata.get("category", "Uncategorized"),
                "section": metadata.get("section", "Unspecified"),
                "chunk_id": chunk_id,
                "distance": round(float(distance), 4),
            }
        )

    return sources


def format_langchain_debug(
    *,
    scored_documents,
    context,
    top_k=DEFAULT_TOP_K,
    fallback=False,
):
    """Return LangChain-specific metadata for developer inspection."""
    retrieved_chunk_ids = [
        document.metadata.get("chunk_id", "unknown")
        for document, _distance in scored_documents
    ]

    return {
        "chain_expression": "ChatPromptTemplate | ChatOllama | StrOutputParser",
        "components": {
            "vector_store": "Chroma",
            "retrieval_method": "similarity_search_with_score",
            "prompt_template": "ChatPromptTemplate",
            "chat_model": "ChatOllama",
            "output_parser": "StrOutputParser",
        },
        "retrieved_count": len(scored_documents),
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "top_k": top_k,
        "context_characters": len(context or ""),
        "fallback": fallback,
        "score_type": "Chroma distance; lower usually means closer in this lesson setup",
        "models": {
            "embedding": EMBEDDING_MODEL,
            "generation": GENERATION_MODEL,
        },
    }


def format_success_response(answer, sources, langchain_debug):
    """Return the successful API response body."""
    return {
        "answer": answer.strip(),
        "sources": sources,
        "langchain": langchain_debug,
    }


def format_fallback_response(langchain_debug=None):
    """Return a safe response when there is not enough approved context."""
    return {
        "answer": FALLBACK_MESSAGE,
        "sources": [],
        "langchain": langchain_debug or {"fallback": True},
    }


def format_error_response(error, message):
    """Return a standard error response body."""
    return {"error": error, "message": message}
```

## `lib/langchain_rag_service.py`

```python
"""LangChain RAG service for retrieval, prompt execution, and response formatting."""

from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

from lib.config import DEFAULT_TOP_K, GENERATION_MODEL, OLLAMA_BASE_URL, TEMPERATURE
from lib.prompt_templates import build_prompt_template
from lib.response_formatter import (
    format_fallback_response,
    format_langchain_debug,
    format_sources,
    format_success_response,
)
from lib.vector_store import get_vector_store


class LangChainServiceError(RuntimeError):
    """Raised when the LangChain pipeline cannot return a usable answer."""


def build_chat_model():
    """Create the LangChain Ollama chat model wrapper."""
    return ChatOllama(
        model=GENERATION_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=TEMPERATURE,
    )


def build_chain(prompt_template=None, llm=None):
    """Compose the prompt, model, and output parser into a LangChain runnable chain."""
    prompt_template = prompt_template or build_prompt_template()
    llm = llm or build_chat_model()

    return prompt_template | llm | StrOutputParser()


def retrieve_context(question, vector_store=None, top_k=DEFAULT_TOP_K):
    """Retrieve documents and distances from the LangChain Chroma vector store."""
    store = vector_store or get_vector_store()
    return store.similarity_search_with_score(question.strip(), k=top_k)


def has_usable_context(scored_documents):
    """Return True when retrieval produced at least one non-empty document."""
    for document, _distance in scored_documents:
        if document.page_content and document.page_content.strip():
            return True

    return False


def format_context(scored_documents):
    """Format retrieved LangChain documents into a prompt-ready context block."""
    context_blocks = []

    for index, (document, distance) in enumerate(scored_documents, start=1):
        metadata = document.metadata or {}

        context_blocks.append(
            "\n".join(
                [
                    f"[Context {index}]",
                    f"Source ID: {metadata.get('source_id', 'unknown')}",
                    f"Title: {metadata.get('title', 'Untitled')}",
                    f"Category: {metadata.get('category', 'Uncategorized')}",
                    f"Section: {metadata.get('section', 'Unspecified')}",
                    f"Chunk ID: {metadata.get('chunk_id', 'unknown')}",
                    f"Distance: {float(distance):.4f}",
                    "Text:",
                    document.page_content.strip(),
                ]
            )
        )

    return "\n\n".join(context_blocks)


def answer_question(
    question,
    *,
    vector_store=None,
    chain=None,
    top_k=DEFAULT_TOP_K,
):
    """Run the LangChain-supported RAG workflow for one validated question."""
    cleaned_question = question.strip()

    try:
        scored_documents = retrieve_context(
            cleaned_question,
            vector_store=vector_store,
            top_k=top_k,
        )

        if not has_usable_context(scored_documents):
            debug = format_langchain_debug(
                scored_documents=[],
                context="",
                top_k=top_k,
                fallback=True,
            )
            return format_fallback_response(debug)

        context = format_context(scored_documents)
        rag_chain = chain or build_chain()

        answer = rag_chain.invoke(
            {
                "context": context,
                "question": cleaned_question,
            }
        )

        if not isinstance(answer, str) or not answer.strip():
            raise LangChainServiceError("The LangChain pipeline returned an empty answer.")

        sources = format_sources(scored_documents)
        debug = format_langchain_debug(
            scored_documents=scored_documents,
            context=context,
            top_k=top_k,
            fallback=False,
        )

        return format_success_response(answer, sources, debug)

    except LangChainServiceError:
        raise
    except Exception as exc:
        raise LangChainServiceError(str(exc)) from exc
```

## `lib/validation.py`

```python
"""Request validation helpers for POST /api/ask."""

MIN_QUESTION_LENGTH = 3


def validate_question_payload(payload):
    """Validate the JSON body for POST /api/ask.

    Return:
        (question, None) when valid
        (None, error_dict) when invalid
    """
    if not isinstance(payload, dict):
        return None, {
            "error": "invalid_request",
            "message": "Request body must be a JSON object.",
        }

    if "question" not in payload:
        return None, {
            "error": "missing_question",
            "message": "Request body must include a question field.",
        }

    question = payload["question"]

    if not isinstance(question, str):
        return None, {
            "error": "invalid_question",
            "message": "Question must be a string.",
        }

    question = question.strip()

    if not question:
        return None, {
            "error": "empty_question",
            "message": "Question cannot be blank.",
        }

    if len(question) < MIN_QUESTION_LENGTH:
        return None, {
            "error": "short_question",
            "message": f"Question must be at least {MIN_QUESTION_LENGTH} characters long.",
        }

    return question, None
```

## `app.py`

```python
"""Flask API for the LangChain RAG technical lesson."""

from flask import Flask, jsonify, request

from lib.langchain_rag_service import LangChainServiceError, answer_question
from lib.response_formatter import format_error_response
from lib.validation import validate_question_payload


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)

    @app.post("/api/ask")
    def ask():
        """Accept a question and return a source-backed LangChain RAG response."""
        question, error = validate_question_payload(request.get_json(silent=True))

        if error:
            return jsonify(format_error_response(error["error"], error["message"])), 400

        try:
            response = answer_question(question)
            return jsonify(response), 200
        except LangChainServiceError as exc:
            return (
                jsonify(
                    format_error_response(
                        "langchain_service_error",
                        str(exc),
                    )
                ),
                502,
            )

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
```

---

# Common Issues and Fixes

| Issue | What it looks like | How to fix it |
|---|---|---|
| Ollama is not running | Seeding or model calls fail | Start Ollama and run `ollama list` |
| Embedding model is missing | Chroma seeding fails | Run `ollama pull embeddinggemma` |
| Generation model is missing | Endpoint returns a 502 error | Run `ollama pull llama3.2` |
| Chroma has stale data | Sources do not match the current JSON file | Delete `chroma_db/` or rerun `python seed_chroma.py` |
| Prompt variables are wrong | Chain error mentions missing `context` or `question` | Check the template variable names |
| Endpoint returns text but weak sources | Retrieval may be noisy | Inspect `sources`, `distance`, and `retrieved_chunk_ids` |
| Answer includes unsupported details | Model may be overgeneralizing | Strengthen prompt instructions and verify retrieved context |
| Debug output feels too verbose | Production APIs often hide this | Keep it visible for learning; later move it behind a developer flag |

---

# Success Criteria

Your completed technical lesson project should:

- seed a new Chroma collection with reliability runbook chunks,
- use `OllamaEmbeddings` to create embeddings,
- use Chroma through LangChain,
- expose a Flask `POST /api/ask` endpoint,
- validate the incoming question,
- retrieve relevant context,
- format context for the prompt,
- use `ChatPromptTemplate`,
- use `ChatOllama`,
- use `StrOutputParser`,
- return `answer`, `sources`, and `langchain` metadata,
- avoid returning full raw document text in sources,
- use fallback behavior when no approved context is available,
- and support manual verification through `curl`.

---

# Extension Ideas

After you complete the lesson, try one extension:

1. Add an environment variable that hides or shows the `langchain` debug field.
2. Add a `top_k` request field so developers can compare retrieval depth.
3. Add a second route, such as `/api/ask/plain`, that calls the same service without debug metadata.
4. Add a frontend view that displays answer, source cards, and retrieved chunk IDs.
5. Add a retrieval quality note that flags when all returned distances are weak.
6. Compare `similarity_search_with_score()` with `as_retriever().invoke()` and explain what each makes easier or harder to inspect.
