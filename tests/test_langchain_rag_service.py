from langchain_core.documents import Document

from lib.langchain_rag_service import answer_question, format_context, has_usable_context


class FakeVectorStore:
    def __init__(self, results):
        self.results = results
        self.query = None
        self.k = None

    def similarity_search_with_score(self, question, k):
        self.query = question
        self.k = k
        return self.results


class FakeChain:
    def __init__(self, answer="Use the rollback procedure and verify health checks."):
        self.answer = answer
        self.inputs = None

    def invoke(self, inputs):
        self.inputs = inputs
        return self.answer


def sample_doc():
    return Document(
        page_content="If a release causes checkout API errors, prepare a rollback.",
        metadata={
            "chunk_id": "chunk-rel-101-b",
            "source_id": "REL-101",
            "title": "Checkout API Error Spike Runbook",
            "category": "Reliability",
            "section": "Mitigation",
        },
    )


def test_has_usable_context_rejects_empty_context():
    assert has_usable_context([]) is False
    assert has_usable_context([(Document(page_content="   ", metadata={}), 0.9)]) is False


def test_has_usable_context_accepts_non_empty_document():
    assert has_usable_context([(sample_doc(), 0.12)]) is True


def test_format_context_includes_metadata_distance_and_text():
    context = format_context([(sample_doc(), 0.12345)])

    assert "REL-101" in context
    assert "Checkout API Error Spike Runbook" in context
    assert "Mitigation" in context
    assert "0.1235" in context
    assert "prepare a rollback" in context


def test_answer_question_runs_chain_with_retrieved_context():
    vector_store = FakeVectorStore([(sample_doc(), 0.12)])
    chain = FakeChain()

    response = answer_question(
        "  What should I do if checkout errors spike after a release?  ",
        vector_store=vector_store,
        chain=chain,
        top_k=2,
    )

    assert vector_store.query == "What should I do if checkout errors spike after a release?"
    assert vector_store.k == 2
    assert chain.inputs["question"] == "What should I do if checkout errors spike after a release?"
    assert "checkout API errors" in chain.inputs["context"]
    assert response["answer"] == "Use the rollback procedure and verify health checks."
    assert response["sources"][0]["source_id"] == "REL-101"
    assert response["langchain"]["retrieved_count"] == 1
    assert response["langchain"]["fallback"] is False


def test_answer_question_returns_fallback_without_calling_chain_when_no_context():
    vector_store = FakeVectorStore([])
    chain = FakeChain(answer="This should not be used.")

    response = answer_question(
        "What is today's cafeteria menu?",
        vector_store=vector_store,
        chain=chain,
        top_k=2,
    )

    assert chain.inputs is None
    assert response["sources"] == []
    assert response["langchain"]["fallback"] is True
