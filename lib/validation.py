"""Request validation helpers for POST /api/ask."""

MIN_QUESTION_LENGTH = 3


def validate_question_payload(payload):
    """Validate the JSON body for POST /api/ask.

    Return:
        (question, None) when valid
        (None, error_dict) when invalid
    """
    # TODO: Require payload to be a dictionary.
    # TODO: Require a question field.
    # TODO: Require question to be a string.
    # TODO: Strip extra whitespace.
    # TODO: Reject blank questions.
    # TODO: Reject questions shorter than MIN_QUESTION_LENGTH.
    raise NotImplementedError("Complete validate_question_payload() in Step 8.")
