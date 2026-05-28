"""Configuration values for the LangChain RAG technical lesson.

The defaults are local-first so the lesson can run with Ollama and Chroma on a
learner machine. Instructors can override these values with environment
variables if the course environment uses different model names or paths.
"""

import os


CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "reliability_runbooks")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embeddinggemma")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "llama3.2")

DEFAULT_TOP_K = int(os.getenv("TOP_K", "3"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))

FALLBACK_MESSAGE = (
    "I do not have enough approved reliability runbook context to answer that question. "
    "Check the approved runbooks or ask an incident lead before acting on this request."
)
