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
    # TODO: Step 7 - return ChatOllama using GENERATION_MODEL, OLLAMA_BASE_URL, and TEMPERATURE.
    raise NotImplementedError("Complete build_chat_model() in Step 7.")


def build_chain(prompt_template=None, llm=None):
    """Compose the prompt, model, and output parser into a LangChain runnable chain."""
    # TODO: Step 7 - return prompt_template | llm | StrOutputParser().
    raise NotImplementedError("Complete build_chain() in Step 7.")


def retrieve_context(question, vector_store=None, top_k=DEFAULT_TOP_K):
    """Retrieve documents and distances from the LangChain Chroma vector store."""
    # TODO: Step 7 - call vector_store.similarity_search_with_score(question, k=top_k).
    raise NotImplementedError("Complete retrieve_context() in Step 7.")


def has_usable_context(scored_documents):
    """Return True when retrieval produced at least one non-empty document."""
    # TODO: Step 7 - check for at least one document with non-empty page_content.
    raise NotImplementedError("Complete has_usable_context() in Step 7.")


def format_context(scored_documents):
    """Format retrieved LangChain documents into a prompt-ready context block."""
    # TODO: Step 7 - include source metadata, distance, and page_content for each retrieved document.
    raise NotImplementedError("Complete format_context() in Step 7.")


def answer_question(
    question,
    *,
    vector_store=None,
    chain=None,
    top_k=DEFAULT_TOP_K,
):
    """Run the LangChain-supported RAG workflow for one validated question."""
    # TODO: Step 7 - retrieve context, fallback if needed, run the chain, and format the response.
    raise NotImplementedError("Complete answer_question() in Step 7.")
