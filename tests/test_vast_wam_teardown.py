from __future__ import annotations

import urllib.error

from blueprint_pipeline.vast_wam_teardown import destroy_vast_instance_with_retry


def _redact(value: object, secrets: list[str]) -> object:
    return {"redacted": True, "secret_count": len(secrets), "value": value}


def test_vast_destroy_success_records_redacted_api_evidence() -> None:
    calls: list[dict[str, object]] = []

    def api_call(**kwargs: object) -> tuple[int, object]:
        calls.append(kwargs)
        return 200, {"api_key": "must-not-survive"}

    continuing_spend, actions = destroy_vast_instance_with_retry(
        instance_id=42,
        api_key="secret",
        api_call=api_call,
        redact_response=_redact,
        sleeper=lambda _seconds: None,
    )

    assert continuing_spend is False
    assert calls[0]["path"] == "/instances/42/"
    assert actions == [
        {
            "instance_id": 42,
            "action": "destroy_instance",
            "attempt": 1,
            "http_status_code": 200,
            "response": {
                "redacted": True,
                "secret_count": 1,
                "value": {"api_key": "must-not-survive"},
            },
            "status": "completed",
        }
    ]


def test_vast_destroy_404_is_already_absent_success() -> None:
    def api_call(**_kwargs: object) -> tuple[int, object]:
        raise urllib.error.HTTPError("https://api.example", 404, "missing", {}, None)

    continuing_spend, actions = destroy_vast_instance_with_retry(
        instance_id=7,
        api_key="secret",
        api_call=api_call,
        redact_response=_redact,
        sleeper=lambda _seconds: None,
    )

    assert continuing_spend is False
    assert actions[0]["reason"] == "instance_already_absent"


def test_vast_destroy_retries_transient_failures_with_bounded_backoff() -> None:
    outcomes: list[object] = [
        urllib.error.HTTPError("https://api.example", 503, "busy", {}, None),
        TimeoutError("offline"),
        (200, {}),
    ]
    waits: list[float] = []

    def api_call(**_kwargs: object) -> tuple[int, object]:
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome  # type: ignore[return-value]

    continuing_spend, actions = destroy_vast_instance_with_retry(
        instance_id=8,
        api_key="secret",
        api_call=api_call,
        redact_response=_redact,
        sleeper=waits.append,
    )

    assert continuing_spend is False
    assert [action["status"] for action in actions] == ["failed", "failed", "completed"]
    assert waits == [3.0, 6.0]


def test_vast_destroy_exhaustion_preserves_continuing_spend_warning() -> None:
    def api_call(**_kwargs: object) -> tuple[int, object]:
        raise RuntimeError("provider unavailable")

    continuing_spend, actions = destroy_vast_instance_with_retry(
        instance_id=9,
        api_key="secret",
        api_call=api_call,
        redact_response=_redact,
        attempts=2,
        backoff_seconds=20,
        sleeper=lambda _seconds: None,
    )

    assert continuing_spend is True
    assert len(actions) == 2
    assert all(action["error_type"] == "RuntimeError" for action in actions)
