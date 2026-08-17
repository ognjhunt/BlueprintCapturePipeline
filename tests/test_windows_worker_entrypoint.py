"""Licence handling contract for the on-instance Windows trainer entry point.

The trainer credential cannot ride in UserData, so the instance fetches it at
run time.  These tests pin the properties that make that safe.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.windows_worker_entrypoint import (
    LICENCE_DELETE_URL_ENV,
    LICENCE_GET_URL_ENV,
    WindowsWorkerEntrypointError,
    build_trainer_environment,
    fetch_licence,
    load_worker_environment,
    redact,
    run,
)

EMAIL = "operator@example.invalid"
PASSWORD = "correct-horse-battery-staple"


class _Response:
    def __init__(self, body: str = "", status: int = 200) -> None:
        self._body = body.encode("utf-8")
        self.status = status

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_args: object) -> None:
        return None


def _opener(*, licence_body: str, delete_status: int = 204, calls: list | None = None):
    def opener(target):
        url = target if isinstance(target, str) else target.full_url
        method = "GET" if isinstance(target, str) else target.get_method()
        if calls is not None:
            calls.append((method, url))
        if method == "DELETE":
            return _Response(status=delete_status)
        return _Response(licence_body)

    return opener


_GOOD = f"POSTSHOT_LOGIN_EMAIL={EMAIL}\nPOSTSHOT_LOGIN_PASSWORD={PASSWORD}\n"


def _env_file(tmp_path: Path, **extra: str) -> Path:
    values = {
        LICENCE_GET_URL_ENV: "https://example.invalid/licence",
        LICENCE_DELETE_URL_ENV: "https://example.invalid/licence?delete",
        "BLUEPRINT_WORKER_IMAGE_DIGEST": "host@sha256:" + "a" * 64,
    }
    values.update(extra)
    path = tmp_path / "worker.env"
    path.write_text(
        "\n".join(f"{k}={v}" for k, v in values.items()) + "\n", encoding="utf-8"
    )
    return path


# --------------------------------------------------------------------------
# Fetch then acknowledge by deleting
# --------------------------------------------------------------------------


def test_licence_object_is_deleted_after_fetch() -> None:
    calls: list = []
    licence = fetch_licence(
        get_url="https://example.invalid/licence",
        delete_url="https://example.invalid/licence?delete",
        opener=_opener(licence_body=_GOOD, calls=calls),
    )
    assert licence["POSTSHOT_LOGIN_EMAIL"] == EMAIL
    assert [method for method, _ in calls] == ["GET", "DELETE"]


def test_unacknowledged_delete_fails_closed() -> None:
    """A licence left behind after a fetch is a standing exposure."""
    with pytest.raises(WindowsWorkerEntrypointError) as excinfo:
        fetch_licence(
            get_url="https://example.invalid/licence",
            delete_url="https://example.invalid/licence?delete",
            opener=_opener(licence_body=_GOOD, delete_status=500),
        )
    assert "delete_not_acknowledged" in str(excinfo.value)


def test_deletion_is_attempted_even_when_the_blob_is_unusable() -> None:
    """A malformed licence must still not be left sitting in the bucket."""
    calls: list = []
    with pytest.raises(WindowsWorkerEntrypointError):
        fetch_licence(
            get_url="https://example.invalid/licence",
            delete_url="https://example.invalid/licence?delete",
            opener=_opener(licence_body="POSTSHOT_LOGIN_EMAIL=only-half\n", calls=calls),
        )
    assert "DELETE" in [method for method, _ in calls]


def test_incomplete_licence_is_refused() -> None:
    with pytest.raises(WindowsWorkerEntrypointError) as excinfo:
        fetch_licence(
            get_url="https://example.invalid/licence",
            delete_url=None,
            opener=_opener(licence_body="POSTSHOT_LOGIN_EMAIL=only-half\n"),
        )
    assert "licence_incomplete" in str(excinfo.value)


def test_licence_blob_cannot_inject_arbitrary_runtime_configuration() -> None:
    """A compromised licence endpoint must not be able to set trainer config."""
    body = _GOOD + "POSTSHOT_CLI_PATH=C:\\evil\\payload.exe\n"
    with pytest.raises(WindowsWorkerEntrypointError) as excinfo:
        fetch_licence(
            get_url="https://example.invalid/licence",
            delete_url="https://example.invalid/licence?delete",
            opener=_opener(licence_body=body),
        )
    assert "unexpected_keys" in str(excinfo.value)
    assert "POSTSHOT_CLI_PATH" in str(excinfo.value)


def test_fetch_failure_does_not_leak_the_url_body() -> None:
    def exploding(_target):
        raise OSError("connection reset")

    with pytest.raises(WindowsWorkerEntrypointError) as excinfo:
        fetch_licence(
            get_url="https://example.invalid/licence",
            delete_url=None,
            opener=exploding,
        )
    assert str(excinfo.value) == "postshot_licence_fetch_failed"


# --------------------------------------------------------------------------
# The credential reaches the trainer and nothing else
# --------------------------------------------------------------------------


def test_trainer_environment_carries_the_credential() -> None:
    merged = build_trainer_environment(
        worker_environment={"BLUEPRINT_WORKER_IMAGE_DIGEST": "host@sha256:" + "a" * 64},
        licence={"POSTSHOT_LOGIN_EMAIL": EMAIL, "POSTSHOT_LOGIN_PASSWORD": PASSWORD},
    )
    assert merged["POSTSHOT_LOGIN_PASSWORD"] == PASSWORD
    assert merged["BLUEPRINT_WORKER_IMAGE_DIGEST"].endswith("a" * 64)


def test_a_staged_credential_cannot_override_the_fetched_one() -> None:
    """Whatever the boot script staged loses to the authoritative licence."""
    merged = build_trainer_environment(
        worker_environment={"POSTSHOT_LOGIN_PASSWORD": "stale-or-planted"},
        licence={"POSTSHOT_LOGIN_EMAIL": EMAIL, "POSTSHOT_LOGIN_PASSWORD": PASSWORD},
    )
    assert merged["POSTSHOT_LOGIN_PASSWORD"] == PASSWORD


def test_redact_removes_every_secret() -> None:
    text = f"login {EMAIL} password {PASSWORD} done"
    cleaned = redact(text, (EMAIL, PASSWORD))
    assert EMAIL not in cleaned and PASSWORD not in cleaned
    assert cleaned.count("***") == 2


def test_worker_environment_parsing_keeps_values_with_equals_signs(
    tmp_path: Path,
) -> None:
    path = tmp_path / "worker.env"
    path.write_text("URL=https://x.invalid/a?b=c&d=e\n", encoding="utf-8")
    assert load_worker_environment(path)["URL"] == "https://x.invalid/a?b=c&d=e"


# --------------------------------------------------------------------------
# End-to-end through run()
# --------------------------------------------------------------------------


def test_run_hands_the_credential_to_the_arm_and_reports_no_disk_write(
    tmp_path: Path,
) -> None:
    seen: dict = {}

    def runner(*, environment):
        seen.update(environment)
        return {"exit_code": 0, "artifact": "postshot-primary.ply"}

    receipt = run(
        worker_env_file=_env_file(tmp_path),
        opener=_opener(licence_body=_GOOD),
        runner=runner,
    )
    assert seen["POSTSHOT_LOGIN_PASSWORD"] == PASSWORD
    assert receipt["licence_object_deleted"] is True
    assert receipt["credential_written_to_disk"] is False
    assert receipt["result"]["exit_code"] == 0


def test_run_redacts_a_credential_that_leaks_into_the_arm_result(
    tmp_path: Path,
) -> None:
    """Defence in depth: even if the arm echoes the secret, the receipt cannot."""

    def leaky_runner(*, environment):
        return {"exit_code": 1, "stderr": f"auth failed for {PASSWORD}"}

    receipt = run(
        worker_env_file=_env_file(tmp_path),
        opener=_opener(licence_body=_GOOD),
        runner=leaky_runner,
    )
    assert PASSWORD not in json.dumps(receipt)
    assert "***" in receipt["result"]["stderr"]


def test_run_without_a_licence_url_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(LICENCE_GET_URL_ENV, raising=False)
    path = tmp_path / "worker.env"
    path.write_text("BLUEPRINT_WORKER_IMAGE_DIGEST=host\n", encoding="utf-8")
    with pytest.raises(WindowsWorkerEntrypointError) as excinfo:
        run(worker_env_file=path, opener=_opener(licence_body=_GOOD), runner=lambda **_: {})
    assert "licence_get_url_missing" in str(excinfo.value)
