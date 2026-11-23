"""A provider's own 5xx or 429 text is an outage, retried in minutes, not a refusal retried in fifteen.

The strings are what the provider SDKs print for the error the job's step re-raises unchanged
(``website_gemini_receipts.retained_gemini_call`` re-raises the SDK's exception; ``_texts`` reads only
``str()`` of each link of the chain): google-genai 3.6.1 ``ServerError(500, body)`` and
``ServerError(504, body)``, and the OpenAI SDK's ``Error code: 500 - ...``.
"""
from __future__ import annotations

import pytest

from blueprint_pipeline.website_preparation_retry import OUTAGE, classify, retry_delay

PROVIDER_TRANSIENT_TEXTS = [
    "500 INTERNAL. {'error': {'code': 500, 'message': 'An internal error has occurred. Please retry or report in "
    "https://developers.generativeai.google/guide/troubleshooting', 'status': 'INTERNAL'}}",
    "504 DEADLINE_EXCEEDED. {'error': {'code': 504, 'message': 'Deadline expired before operation could complete.', "
    "'status': 'DEADLINE_EXCEEDED'}}",
    "Error code: 500 - {'error': {'message': 'The server had an error while processing your request. Sorry about "
    "that!', 'type': 'server_error'}}",
    "Error code: 429 - {'error': {'message': 'Rate limit reached for requests', 'type': 'requests', "
    "'code': 'rate_limit_exceeded'}}",
    "502 Bad Gateway",
]


@pytest.mark.parametrize("text", PROVIDER_TRANSIENT_TEXTS)
def test_provider_transient_error_text_is_an_outage(text: str) -> None:
    assert classify(text) == OUTAGE
    assert retry_delay(classify(text), 1) == 120
