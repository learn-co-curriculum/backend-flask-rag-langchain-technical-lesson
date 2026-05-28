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
        # TODO: Step 8 - validate request JSON, call answer_question(), and return JSON.
        return jsonify({"message": "Complete /api/ask in Step 8."}), 501

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
