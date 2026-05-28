from langchain_core.documents import Document

from lib.response_formatter import (
    format_fallback_response,
    format_langchain_debug,
    format_sources,
    format_success_response,
)


def sample_scored_documents():
    return [
        (
            Document(
                page_content="Rollback to the previous stable version when a release causes checkout errors.",
                metadata={
                    "chunk_id": "chunk-rel-102-a",
                    "source_id": "REL-102",
                    "title": "Rollback Procedure",
                    "category": "Reliability",
                    "section": "Decision Criteria",
                },
            ),
            0.12,
        )
    ]


def test_format_sources_returns_source_metadata_without_full_text():
    sources = format_sources(sample_scored_documents())

    assert sources == [
        {
            "source_id": "REL-102",
            "title": "Rollback Procedure",
            "category": "Reliability",
            "section": "Decision Criteria",
            "chunk_id": "chunk-rel-102-a",
            "distance": 0.12,
        }
    ]
    assert "page_content" not in sources[0]
    assert "text" not in sources[0]


def test_format_langchain_debug_names_components_and_trace():
    docs = sample_scored_documents()
    debug = format_langchain_debug(
        scored_documents=docs,
        context="Source ID: REL-102\nText: rollback",
        top_k=3,
        fallback=False,
    )

    assert debug["chain_expression"] == "ChatPromptTemplate | ChatOllama | StrOutputParser"
    assert debug["components"]["vector_store"] == "Chroma"
    assert debug["retrieved_count"] == 1
    assert debug["retrieved_chunk_ids"] == ["chunk-rel-102-a"]
    assert debug["fallback"] is False


def test_format_success_response_includes_answer_sources_and_langchain_metadata():
    response = format_success_response(
        "Prepare a rollback and verify health checks.",
        [{"source_id": "REL-102"}],
        {"fallback": False},
    )

    assert response["answer"] == "Prepare a rollback and verify health checks."
    assert response["sources"] == [{"source_id": "REL-102"}]
    assert response["langchain"] == {"fallback": False}


def test_format_fallback_response_uses_empty_sources():
    response = format_fallback_response({"fallback": True})

    assert "not have enough approved" in response["answer"]
    assert response["sources"] == []
    assert response["langchain"]["fallback"] is True
