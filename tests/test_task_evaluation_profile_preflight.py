import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from blueprint_pipeline.task_evaluation_profile_preflight import (
    RELEASE_SCHEMA_VERSION,
    REQUEST_SCHEMA_VERSION,
    READINESS_SCHEMA_VERSION,
    run_task_evaluation_profile_preflight,
)


COMMIT = "a" * 40
NOW = datetime(2026, 8, 10, 12, 0, tzinfo=timezone.utc)


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_digest(value: dict, field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _inputs(tmp_path: Path) -> dict[str, Path]:
    source = tmp_path / "source.json"
    spec = tmp_path / "spec.json"
    readiness = tmp_path / "readiness.json"
    release = tmp_path / "release.json"
    guard = tmp_path / "guard.json"
    request = tmp_path / "request.json"
    _write(source, {"scene": "840313"})
    _write(spec, {"evaluation_run": "frozen"})
    readiness_value = {
        "schema_version": READINESS_SCHEMA_VERSION,
        "status": "blocked",
        "blockers": [
            "exact_adp009d_runtime_adapter_not_on_protected_main",
            "scripted_positive_control_not_passed",
            "allocator_artifact_manifest_not_emitted",
        ],
    }
    readiness_value["receipt_digest"] = _canonical_digest(
        readiness_value, "receipt_digest"
    )
    _write(readiness, readiness_value)
    release_value = {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "status": "passed",
        "source_commit": COMMIT,
        "source_ref": "main",
        "tracked_state": "clean",
    }
    release_value["release_digest"] = _canonical_digest(
        release_value, "release_digest"
    )
    _write(release, release_value)
    _write(
        guard,
        {
            "schema_version": "gpu_spend_guard.v1",
            "generated_at": NOW.isoformat(),
            "status": "blocked",
            "blockers": ["spend_admission:paid_work_admission_locked"],
            "inventory_results": [
                {"provider": provider, "status": "succeeded", "row_count": 0}
                for provider in ("runpod", "vast", "digitalocean")
            ],
            "live_instance_count": 0,
            "total_burn_per_hour_usd": 0.0,
        },
    )
    request_value = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "profile_id": "adp009d-840313-franka-dry-v1",
        "provider": "vast",
        "retry_cap": 0,
        "live_execution_authorized": False,
        "required_provider_zero": ["runpod", "vast", "digitalocean"],
        "max_guard_age_seconds": 300,
        "immutable_inputs": {
            "source_bundle_manifest": {"path": str(source), "digest": _digest(source)},
            "evaluation_run_spec": {"path": str(spec), "digest": _digest(spec)},
            "runtime_readiness": {
                "path": str(readiness),
                "digest": _digest(readiness),
            },
        },
    }
    request_value["request_digest"] = _canonical_digest(
        request_value, "request_digest"
    )
    _write(request, request_value)
    return {
        "request": request,
        "release": release,
        "readiness": readiness,
        "guard": guard,
        "source": source,
    }


def _run(paths: dict[str, Path], *, execute: bool = False) -> dict:
    return run_task_evaluation_profile_preflight(
        request_path=paths["request"],
        release_evidence_path=paths["release"],
        readiness_receipt_path=paths["readiness"],
        provider_guard_path=paths["guard"],
        expected_source_commit=COMMIT,
        observed_source_commit=COMMIT,
        execute=execute,
        now=NOW,
    )


def test_dry_preflight_accepts_typed_live_blockers_without_provider_mutation(
    tmp_path: Path,
) -> None:
    result = _run(_inputs(tmp_path))

    assert result["status"] == "dry_run_ready"
    assert result["provider"] == "vast"
    assert result["provider_zero_verified"] is True
    assert result["provider_mutations_performed"] == 0
    assert result["live_execution_enabled"] is False
    assert "scripted_positive_control_not_passed" in result["live_execution_blockers"]
    assert "spend_admission:paid_work_admission_locked" in result[
        "live_execution_blockers"
    ]


def test_preflight_fails_closed_on_input_change_or_execute(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    paths["source"].write_text('{"scene":"changed"}\n', encoding="utf-8")
    changed = _run(paths)
    assert changed["status"] == "blocked"
    assert (
        "task_evaluation_preflight_immutable_input_invalid:source_bundle_manifest"
        in changed["blockers"]
    )

    paths = _inputs(tmp_path)
    execute = _run(paths, execute=True)
    assert execute["status"] == "blocked"
    assert "task_evaluation_profile_preflight_execute_forbidden" in execute["blockers"]
    assert execute["provider_mutations_performed"] == 0


def test_preflight_requires_fresh_api_confirmed_provider_zero(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    guard = json.loads(paths["guard"].read_text(encoding="utf-8"))
    guard["inventory_results"][1]["row_count"] = 1
    _write(paths["guard"], guard)

    result = _run(paths)

    assert result["status"] == "blocked"
    assert result["provider_zero_verified"] is False
    assert "task_evaluation_preflight_provider_zero_unverified:vast" in result[
        "blockers"
    ]
