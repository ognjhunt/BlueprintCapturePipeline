from __future__ import annotations

import urllib.error
from pathlib import Path

from blueprint_pipeline.runpod_wam_teardown import (
    delete_runpod_pod,
    stop_runpod_pod,
    verify_runpod_pod_inactive,
)


TERMINAL = ("EXITED", "TERMINATED", "STOPPED")


def _status(payload: dict[str, object]) -> str:
    return str(payload.get("status") or "")


def test_verify_runpod_pod_inactive_accepts_provider_terminal_status() -> None:
    result = verify_runpod_pod_inactive(
        pod_id="pod-1",
        api_key="secret",
        generated_at="now",
        request=lambda **_kwargs: (200, {"status": "STOPPED"}),
        pod_status_reader=_status,
        terminal_statuses=TERMINAL,
    )

    assert result["status"] == "completed"
    assert result["spend_released"] is True


def test_verify_runpod_pod_inactive_fails_closed_for_active_or_unreachable() -> None:
    active = verify_runpod_pod_inactive(
        pod_id="pod-2",
        api_key="secret",
        generated_at="now",
        request=lambda **_kwargs: (200, {"status": "RUNNING"}),
        pod_status_reader=_status,
        terminal_statuses=TERMINAL,
    )

    def unavailable(**_kwargs: object) -> tuple[int, dict[str, object]]:
        raise urllib.error.URLError("offline")

    blocked = verify_runpod_pod_inactive(
        pod_id="pod-2",
        api_key="secret",
        generated_at="now",
        request=unavailable,
        pod_status_reader=_status,
        terminal_statuses=TERMINAL,
    )

    assert active["spend_released"] is False
    assert blocked["pod_status"] == "status_probe_error"


def test_delete_runpod_pod_requires_terminal_state_confirmation(tmp_path: Path) -> None:
    manifest = delete_runpod_pod(
        job_dir=tmp_path,
        pod_id="pod-3",
        api_key="secret",
        generated_at="now",
        schema_version="delete.v1",
        request=lambda **_kwargs: (204, {}),
        verify_inactive=lambda **_kwargs: {
            "pod_status": "RUNNING",
            "spend_released": False,
        },
    )

    assert manifest["status"] == "blocked"
    assert manifest["continuing_spend_from_this_run"] is True
    assert manifest["terminal_state_api_confirmed"] is False
    assert "runpod_delete_terminal_state_not_api_confirmed" in manifest["blockers"]
    assert (tmp_path / "runpod_wam_async_delete_manifest.json").is_file()


def test_delete_runpod_pod_404_is_confirmed_absent(tmp_path: Path) -> None:
    def missing(**_kwargs: object) -> tuple[int, dict[str, object]]:
        raise urllib.error.HTTPError("https://api.example", 404, "missing", {}, None)

    manifest = delete_runpod_pod(
        job_dir=tmp_path,
        pod_id="pod-4",
        api_key="secret",
        generated_at="now",
        schema_version="delete.v1",
        request=missing,
        verify_inactive=lambda **_kwargs: {},
    )

    assert manifest["status"] == "completed"
    assert manifest["terminal_state_api_confirmed"] is True
    assert manifest["verified_pod_status"] == "not_found"


def test_stop_runpod_pod_records_warm_candidate_only_on_acknowledged_stop(
    tmp_path: Path,
) -> None:
    manifest = stop_runpod_pod(
        job_dir=tmp_path,
        pod_id="pod-5",
        api_key="secret",
        generated_at="now",
        schema_version="stop.v1",
        request=lambda **_kwargs: (200, {"ok": True}),
        verify_inactive=lambda **_kwargs: {},
        write_stopped_warm_candidate=lambda **_kwargs: {
            "status": "recorded",
            "path": "/tmp/warm.json",
        },
    )

    assert manifest["status"] == "completed"
    assert manifest["stopped_pod_preserved_for_warm_reuse"] is True
    assert manifest["warm_candidate_path"] == "/tmp/warm.json"


def test_stop_runpod_pod_error_can_be_recovered_only_by_terminal_probe(
    tmp_path: Path,
) -> None:
    def stop_error(**_kwargs: object) -> tuple[int, dict[str, object]]:
        raise urllib.error.HTTPError("https://api.example", 500, "failed", {}, None)

    manifest = stop_runpod_pod(
        job_dir=tmp_path,
        pod_id="pod-6",
        api_key="secret",
        generated_at="now",
        schema_version="stop.v1",
        request=stop_error,
        verify_inactive=lambda **_kwargs: {
            "pod_status": "STOPPED",
            "spend_released": True,
            "blockers": [],
        },
        write_stopped_warm_candidate=lambda **_kwargs: {"status": "unexpected"},
    )

    assert manifest["status"] == "completed"
    assert manifest["stop_response_confirmed"] is False
    assert manifest["stopped_pod_preserved_for_warm_reuse"] is False
    assert manifest["warm_candidate"]["reason"] == (
        "runpod_stop_completion_verified_without_reusable_stopped_pod"
    )


def test_delete_acknowledgement_without_terminal_state_keeps_spend_open(
    tmp_path: Path,
) -> None:
    manifest = delete_runpod_pod(
        job_dir=tmp_path,
        pod_id="pod-active",
        api_key="secret",
        generated_at="now",
        schema_version="delete.v1",
        request=lambda **_kwargs: (202, {}),
        verify_inactive=lambda **_kwargs: {
            "status": "blocked",
            "pod_status": "RUNNING",
            "spend_released": False,
            "blockers": ["runpod_stop_error_pod_still_active_after_status_probe"],
        },
    )

    assert manifest["status"] == "blocked"
    assert manifest["continuing_spend_from_this_run"] is True
    assert manifest["terminal_state_api_confirmed"] is False
    assert "runpod_delete_terminal_state_not_api_confirmed" in manifest["blockers"]


def test_delete_transport_error_can_still_prove_provider_absence(tmp_path: Path) -> None:
    def request(**_kwargs: object) -> tuple[int, dict[str, object]]:
        raise urllib.error.URLError("connection reset")

    manifest = delete_runpod_pod(
        job_dir=tmp_path,
        pod_id="pod-gone",
        api_key="secret",
        generated_at="now",
        schema_version="delete.v1",
        request=request,
        verify_inactive=lambda **_kwargs: {
            "status": "completed",
            "pod_status": "not_found",
            "spend_released": True,
            "blockers": [],
        },
    )

    assert manifest["status"] == "completed"
    assert manifest["mutation_error_type"] == "URLError"
    assert manifest["continuing_spend_from_this_run"] is False
    assert manifest["blockers"] == []


def test_stop_transport_error_is_persisted_as_open_spend(tmp_path: Path) -> None:
    def request(**_kwargs: object) -> tuple[int, dict[str, object]]:
        raise TimeoutError("provider timeout")

    manifest = stop_runpod_pod(
        job_dir=tmp_path,
        pod_id="pod-unknown",
        api_key="secret",
        generated_at="now",
        schema_version="stop.v1",
        request=request,
        verify_inactive=lambda **_kwargs: {
            "status": "blocked",
            "pod_status": "status_probe_error",
            "spend_released": False,
            "blockers": ["runpod_stop_error_status_probe_failed"],
        },
        write_stopped_warm_candidate=lambda **_kwargs: {"status": "recorded"},
    )

    assert manifest["status"] == "blocked"
    assert manifest["mutation_error_type"] == "TimeoutError"
    assert manifest["continuing_spend_from_this_run"] is True
    assert manifest["warm_candidate"]["status"] == "not_recorded"
