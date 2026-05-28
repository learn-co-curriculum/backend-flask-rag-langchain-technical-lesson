"""Run the LangChain RAG service without Flask.

This file shows that LangChain can be used inside a plain Python workflow too.
The Flask route is just the API boundary around the same service.
"""

import json
import sys

from lib.langchain_rag_service import answer_question


DEFAULT_QUESTION = "What should I do if checkout API errors spike right after a release?"


def main():
    question = " ".join(sys.argv[1:]).strip() or DEFAULT_QUESTION
    response = answer_question(question)
    print(json.dumps(response, indent=2))


if __name__ == "__main__":
    main()
