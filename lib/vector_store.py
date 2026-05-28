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
    # TODO: Step 3 - return OllamaEmbeddings using EMBEDDING_MODEL and OLLAMA_BASE_URL.
    raise NotImplementedError("Complete build_embedding_model() in Step 3.")


def get_vector_store():
    """Return the persistent Chroma vector store wrapped by LangChain."""
    # TODO: Step 3 - return Chroma with collection_name, persist_directory, and embedding_function.
    raise NotImplementedError("Complete get_vector_store() in Step 3.")


def get_retriever(vector_store=None, top_k=DEFAULT_TOP_K):
    """Return a LangChain retriever for the Chroma vector store."""
    # TODO: Step 4 - convert the vector store into a retriever with search_kwargs={"k": top_k}.
    raise NotImplementedError("Complete get_retriever() in Step 4.")


def reset_chroma_db(path=CHROMA_PATH):
    """Delete the local Chroma directory so the lesson can reseed cleanly."""
    db_path = Path(path)
    if db_path.exists():
        shutil.rmtree(db_path)


def seed_vector_store(reset=True):
    """Create a fresh Chroma collection and add the lesson runbook documents."""
    # TODO: Step 4 - optionally reset Chroma, load documents, add them to the vector store, and return the count.
    raise NotImplementedError("Complete seed_vector_store() in Step 4.")
