"""Load runbook chunks and convert them into LangChain Document objects."""

import json
from pathlib import Path

from langchain_core.documents import Document


DATA_PATH = Path("data/runbook_chunks.json")


def load_runbook_records(data_path=DATA_PATH):
    """Return runbook chunk dictionaries from the lesson JSON file."""
    with open(data_path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_runbook_documents(data_path=DATA_PATH):
    """Return LangChain Document objects with page content and metadata."""
    records = load_runbook_records(data_path)
    documents = []

    for record in records:
        metadata = {
            "chunk_id": record["chunk_id"],
            "source_id": record["source_id"],
            "title": record["title"],
            "category": record["category"],
            "section": record["section"],
        }

        documents.append(
            Document(
                page_content=record["text"],
                metadata=metadata,
                id=record["chunk_id"],
            )
        )

    return documents
