"""The bounded description kept beside a fixed failure code."""

import errno
import io
import json
from urllib.error import HTTPError

import pytest

from blueprint_pipeline import bounded_error
from blueprint_pipeline.bounded_error import (
    bounded_error_text, cause_description, disk_capacity_refusal, error_description, rate_limited,
)


@pytest.mark.parametrize("raw,kept", [
    # A URL keeps its scheme, host and path; a query or fragment can sign it.
    ("GET https://storage.googleapis.com/b/o.mp4?X-Goog-Signature=0f9e&X-Goog-Credential=svc%40p failed",
     "GET https://storage.googleapis.com/b/o.mp4?<redacted> failed"),
    ("see https://h/p#access_token=abc", "see https://h/p?<redacted>"),
    ("Authorization: Bearer ya29.a0AfH6SMBx-secret", "Authorization: <redacted> <redacted>"),
    ("sent Bearer ya29.a0AfH6SMBx-secret", "sent Bearer <redacted>"),
    ("basic dXNlcjpwYXNz", "basic <redacted>"),
    ("client_secret=hunter2 password: hunter3; api-key=hunter4", "client_secret=<redacted> password: <redacted>; "
     "api-key=<redacted>"),
    ("key AIzaSyD3-abcdefghijklmnopqrstuvwxyz012 and sk-live-abcdefgh1234", "key <redacted> and <redacted>"),
    ("jwt eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.sig", "jwt <redacted>"),
    ("-----BEGIN PRIVATE KEY-----\nMIIE\n-----END PRIVATE KEY----- after", "<redacted> after"),
    ("opaque Qx7Lm2Nv9Pr4St6Uw8Yz1Ab3Cd5Ef here", "opaque <redacted> here"),
    # Digests, generations and capture identifiers stay readable.
    ("delivery 277b95d46b3dee94d9af7a0e99fa56153127199db7586c7de9c6946dae4748ed generation 1700000000000001 "
     "walkthrough-req-1", "delivery 277b95d46b3dee94d9af7a0e99fa56153127199db7586c7de9c6946dae4748ed "
     "generation 1700000000000001 walkthrough-req-1"),
    ("[Errno 28] No space left on device: '/var/lib/blueprint/x.pending'",
     "[Errno 28] No space left on device: '/var/lib/blueprint/x.pending'"),
    ("line one\nline two\ttabbed\x00", "line one line two tabbed"),
])
def test_redaction_cases(raw, kept):
    assert bounded_error_text(raw) == kept


@pytest.mark.parametrize("raw,kept", [
    ('{"client_secret": "hunter2", "n": 1}', '{"client_secret": <redacted>, "n": 1}'),
    ("{'api_key': 'hunter2', 'n': 1}", "{'api_key': <redacted>, 'n': 1}"),
    ('{"session":hunter2}', '{"session":<redacted>}'),
    ('{"Authorization": "Bearer abc.def"}', '{"Authorization": <redacted>}'),
    ('{"refresh_token": "x y z", "scene_id": "site-1"}', '{"refresh_token": <redacted>, "scene_id": "site-1"}'),
])
def test_quoted_json_and_printed_dictionary_keys_are_redacted(raw, kept):
    assert bounded_error_text(raw) == kept
    # A body encoded twice, as a message that embeds a JSON string does, escapes its quotes.
    assert "hunter2" not in bounded_error_text(json.dumps({"message": raw}))
    assert "hunter2" not in json.dumps(error_description(RuntimeError(raw)))


def test_text_is_trimmed_after_redaction_and_bounding_is_idempotent():
    long = "token=" + "s" * 500 + " " + "x" * 600
    once = bounded_error_text(long)
    assert len(once) == bounded_error.ERROR_TEXT_LIMIT and once.endswith("...")
    assert once.startswith("token=<redacted> ") and "sss" not in once
    assert bounded_error_text(once) == once
    # Past the scan limit the text is cut at a word boundary, never mid-secret.
    cut = bounded_error_text("a " * 3000 + "token=" + "s" * 50)
    assert "sss" not in cut and len(cut) == bounded_error.ERROR_TEXT_LIMIT
    for raw in ("GET https://h/p?sig=abc", "Bearer abc", '{"api_key": "v"}', "short"):
        assert bounded_error_text(bounded_error_text(raw)) == bounded_error_text(raw)


def test_description_has_exactly_the_qualified_type_and_bounded_message():
    class TransportError(ConnectionError):
        pass

    described = error_description(TransportError("GET https://h/p?sig=abc token=secret"))
    assert set(described) == {"type", "message"}
    assert described["type"].endswith("test_description_has_exactly_the_qualified_type_and_bounded_message"
                                      ".<locals>.TransportError")
    assert described["message"] == "GET https://h/p?<redacted> token=<redacted>"
    assert error_description(KeyError("context_digest")) == {"type": "builtins.KeyError",
                                                              "message": "'context_digest'"}
    assert bounded_error.bounded_error(described) == described

    class Unprintable(Exception):
        def __str__(self):
            raise RuntimeError("no text")
    assert error_description(Unprintable()) == {"type": described["type"].replace("TransportError", "Unprintable"),
                                                "message": ""}


def test_a_repeated_description_is_bounded_again_or_dropped():
    assert bounded_error.bounded_error({"type": "builtins.OSError", "message": "see ?token=abc"}) == {
        "type": "builtins.OSError", "message": "see ?<redacted>"}
    for malformed in (None, "builtins.OSError: boom", {"type": "builtins.OSError"},
                      {"type": "builtins.OSError", "message": "boom", "cause": "x"},
                      {"type": "not a type name", "message": "boom"},
                      {"type": "builtins.OSError", "message": 17}):
        assert bounded_error.bounded_error(malformed) is None


def test_capture_delivery_staging_keeps_its_names_as_re_exports():
    from blueprint_pipeline import capture_delivery_staging as staging

    assert staging.staging_error_description is error_description
    assert staging.bounded_staging_error is bounded_error.bounded_error
    assert staging._bounded_error_text is bounded_error_text
    assert staging._ERROR_TEXT_LIMIT == bounded_error.ERROR_TEXT_LIMIT


def test_cause_and_chain_follow_what_a_traceback_prints():
    try:
        try:
            raise OSError(errno.ENOSPC, "No space left on device")
        except OSError as full:
            raise ValueError("capture_original_birth_refused") from full
    except ValueError as summary:
        assert cause_description(summary) == {"type": "builtins.OSError",
                                              "message": "[Errno 28] No space left on device"}
        assert disk_capacity_refusal(summary)
    assert cause_description(ValueError("typed")) is None
    # An implicit context counts unless it was suppressed with "from None".
    try:
        try:
            raise OSError(errno.EDQUOT, "Disk quota exceeded")
        except OSError:
            raise RuntimeError("write failed")
    except RuntimeError as implicit:
        assert disk_capacity_refusal(implicit)
    try:
        try:
            raise OSError(errno.ENOSPC, "No space left on device")
        except OSError:
            raise RuntimeError("write failed") from None
    except RuntimeError as suppressed:
        assert not disk_capacity_refusal(suppressed)
    assert not disk_capacity_refusal(OSError(errno.EACCES, "Permission denied"))
    # A cycle ends the walk.
    first, second = ValueError("a"), ValueError("b")
    first.__cause__, second.__cause__ = second, first
    assert not disk_capacity_refusal(first) and not rate_limited(first)


def test_rate_limit_is_recognized_by_status_or_typed_code():
    limited = HTTPError("https://h/p?token=x", 429, "Too Many Requests", {}, io.BytesIO(b"{}"))
    assert rate_limited(limited)
    assert rate_limited(ValueError("website_control_task-context_http_429:pipeline_sync_rate_limited"))
    assert rate_limited(ValueError("beta_screening_publication_http_429"))
    try:
        raise ValueError("capture_owner_response_unavailable") from limited
    except ValueError as wrapped:
        assert rate_limited(wrapped)

    class Exhausted(Exception):
        code = 429  # google.api_core's ResourceExhausted and TooManyRequests carry the HTTP code

    assert rate_limited(Exhausted("quota exceeded"))
    for other in (HTTPError("https://h", 503, "Unavailable", {}, io.BytesIO(b"")),
                  ValueError("beta_screening_publication_http_4290"), ValueError("http_42"),
                  type("Flag", (Exception,), {"status": True})("x")):
        assert not rate_limited(other)
