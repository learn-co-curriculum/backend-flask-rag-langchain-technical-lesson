import pytest

from lib.validation import validate_question_payload


def test_validate_question_payload_accepts_and_trims_valid_question():
    question, error = validate_question_payload(
        {"question": "  What should I do if checkout API errors spike?  "}
    )

    assert question == "What should I do if checkout API errors spike?"
    assert error is None


@pytest.mark.parametrize(
    "payload, expected_error",
    [
        (None, "invalid_request"),
        ({}, "missing_question"),
        ({"question": 42}, "invalid_question"),
        ({"question": "   "}, "empty_question"),
        ({"question": "hi"}, "short_question"),
    ],
)
def test_validate_question_payload_rejects_invalid_payloads(payload, expected_error):
    question, error = validate_question_payload(payload)

    assert question is None
    assert error["error"] == expected_error
    assert isinstance(error["message"], str)
