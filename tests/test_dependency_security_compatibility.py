"""Hermetic auth/transport contracts for the October dependency security repair."""
from __future__ import annotations

from io import BytesIO

import anyio
import jwt
import pytest
from urllib3.response import HTTPResponse


def test_unverified_jwt_inspection_does_not_weaken_reused_verification_options() -> None:
    # PYSEC-2026-4146 had no fixed_in annotation on the old package metadata.
    # Exercise the failure itself rather than interpreting an empty annotation
    # as proof of either a patched or an unpatched replacement.
    secret = "local-auth-fixture-key-at-least-32-bytes"
    token = jwt.encode({"sub": "fixture", "aud": "blueprint-test"}, secret, algorithm="HS256")
    options = {"verify_signature": False}
    assert jwt.decode(token, options=options)["sub"] == "fixture"
    assert options == {"verify_signature": False}
    options["verify_signature"] = True
    with pytest.raises(jwt.InvalidAudienceError):
        jwt.decode(token, secret, algorithms=["HS256"], options=options, audience="other")
    with pytest.raises(jwt.InvalidSignatureError):
        jwt.decode(token, "a-different-local-key-at-least-32-bytes", algorithms=["HS256"],
                   audience="blueprint-test")


def test_malformed_signature_is_rejected_instead_of_silently_normalized() -> None:
    secret = "local-auth-fixture-key-at-least-32-bytes"
    token = jwt.encode({"sub": "fixture"}, secret, algorithm="HS256")
    with pytest.raises(jwt.DecodeError):
        jwt.decode(token + "!!!!", secret, algorithms=["HS256"])


def test_urllib3_streaming_preserves_bounded_chunks_and_detects_truncation() -> None:
    payload = b"bounded artifact bytes" * 100
    response = HTTPResponse(body=BytesIO(payload), headers={"Content-Length": str(len(payload))},
                            preload_content=False)
    chunks = list(response.stream(37))
    assert b"".join(chunks) == payload
    assert all(len(chunk) <= 37 for chunk in chunks)
    truncated = HTTPResponse(body=BytesIO(b"short"), headers={"Content-Length": "50"},
                             preload_content=False)
    from urllib3.exceptions import ProtocolError
    with pytest.raises(ProtocolError):
        list(truncated.stream(7))


def test_anyio_timeout_cancels_transport_wait_without_leaking_work() -> None:
    async def check() -> None:
        completed = anyio.Event()

        async def worker() -> None:
            try:
                await anyio.sleep_forever()
            finally:
                completed.set()

        with anyio.move_on_after(.01) as scope:
            async with anyio.create_task_group() as group:
                group.start_soon(worker)
                await anyio.sleep_forever()
        assert scope.cancel_called
        assert completed.is_set()

    anyio.run(check)
