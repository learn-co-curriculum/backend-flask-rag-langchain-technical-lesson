import app as app_module
from lib.langchain_rag_service import LangChainServiceError


def make_client():
    flask_app = app_module.create_app()
    flask_app.config["TESTING"] = True
    return flask_app.test_client()


def test_api_ask_returns_langchain_rag_response(monkeypatch):
    received = {}

    def fake_answer_question(question):
        received["question"] = question
        return {
            "answer": "Prepare a rollback and verify health checks.",
            "sources": [{"source_id": "REL-101"}],
            "langchain": {"fallback": False, "retrieved_count": 1},
        }

    monkeypatch.setattr(app_module, "answer_question", fake_answer_question)

    client = make_client()
    response = client.post(
        "/api/ask",
        json={"question": "  What should I do if checkout errors spike?  "},
    )

    assert response.status_code == 200
    assert received["question"] == "What should I do if checkout errors spike?"

    data = response.get_json()
    assert data["answer"] == "Prepare a rollback and verify health checks."
    assert data["sources"][0]["source_id"] == "REL-101"
    assert data["langchain"]["retrieved_count"] == 1


def test_api_ask_rejects_invalid_input_without_calling_service(monkeypatch):
    called = {"value": False}

    def fake_answer_question(question):
        called["value"] = True

    monkeypatch.setattr(app_module, "answer_question", fake_answer_question)

    client = make_client()
    response = client.post("/api/ask", json={"question": "   "})

    assert response.status_code == 400
    assert called["value"] is False

    data = response.get_json()
    assert data["error"] == "empty_question"


def test_api_ask_returns_502_for_langchain_service_error(monkeypatch):
    def fake_answer_question(question):
        raise LangChainServiceError("Ollama unavailable")

    monkeypatch.setattr(app_module, "answer_question", fake_answer_question)

    client = make_client()
    response = client.post(
        "/api/ask",
        json={"question": "What should I do if checkout errors spike?"},
    )

    assert response.status_code == 502

    data = response.get_json()
    assert data["error"] == "langchain_service_error"
    assert "Ollama unavailable" in data["message"]
