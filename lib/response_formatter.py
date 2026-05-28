"""Response formatting helpers for source-backed LangChain RAG output."""

from lib.config import (
    DEFAULT_TOP_K,
    EMBEDDING_MODEL,
    FALLBACK_MESSAGE,
    GENERATION_MODEL,
)


def format_sources(scored_documents):
    """Return source metadata for retrieved LangChain documents.

    Each item in scored_documents should look like:
        (Document(...), distance)
    """
    # TODO: Step 6 - return unique source dictionaries with source_id, title, category,
    # section, chunk_id, and distance.
    # Do not include full document text in sources.
    raise NotImplementedError("Complete format_sources() in Step 6.")


def format_langchain_debug(
    *,
    scored_documents,
    context,
    top_k=DEFAULT_TOP_K,
    fallback=False,
):
    """Return LangChain-specific metadata for developer inspection."""
    # TODO: Step 6 - return a dictionary that names the LangChain components and trace details.
    raise NotImplementedError("Complete format_langchain_debug() in Step 6.")


def format_success_response(answer, sources, langchain_debug):
    """Return the successful API response body."""
    # TODO: Step 6 - return answer, sources, and langchain debug metadata.
    raise NotImplementedError("Complete format_success_response() in Step 6.")


def format_fallback_response(langchain_debug=None):
    """Return a safe fallback response when there is not enough approved context."""
    # TODO: Step 6 - return FALLBACK_MESSAGE, an empty sources list, and debug metadata.
    raise NotImplementedError("Complete format_fallback_response() in Step 6.")


def format_error_response(error, message):
    """Return a standard error response body."""
    return {"error": error, "message": message}
