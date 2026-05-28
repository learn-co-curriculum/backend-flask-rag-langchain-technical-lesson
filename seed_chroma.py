"""Seed the local Chroma vector database for the LangChain RAG lesson."""

from lib.vector_store import seed_vector_store


def main():
    """Create a fresh Chroma collection and store the lesson runbook chunks."""
    count = seed_vector_store(reset=True)
    print(f"Seeded {count} reliability runbook chunks into Chroma.")


if __name__ == "__main__":
    main()
