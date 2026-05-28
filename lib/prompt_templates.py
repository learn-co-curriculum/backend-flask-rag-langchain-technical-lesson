"""Prompt templates for the LangChain RAG pipeline."""

from langchain_core.prompts import ChatPromptTemplate


SYSTEM_INSTRUCTIONS = """
You are a reliability runbook assistant for an internal engineering team.
Use only the approved context provided by the backend.
If the context does not contain enough information, say that you do not have enough approved context.
Do not invent incident procedures, policy requirements, compensation details, security steps, or root causes.
"""


HUMAN_TEMPLATE = """
Approved context:
{context}

Employee question:
{question}

Response requirements:
- Answer in 2-4 concise sentences.
- Base the answer only on the approved context.
- Mention when the answer depends on incident lead approval or verification.
- Do not include raw distance scores in the answer text.
"""


def build_prompt_template():
    """Return the LangChain chat prompt template used by the chain."""
    # TODO: Step 5 - return ChatPromptTemplate.from_messages([...]).
    raise NotImplementedError("Complete build_prompt_template() in Step 5.")
