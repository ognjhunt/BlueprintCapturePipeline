from __future__ import annotations

import hashlib
import json
import os
import stat
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import scripts.materialize_task_evaluation_standing_launch_authorization as materializer
from blueprint_pipeline.task_evaluation_launch_dispatcher import (
    CANONICAL_ALLOCATOR_ENTRYPOINT,
    EXECUTE_ENV,
    SECRET_PROFILE_ID_ENV,
    canonical_digest,
    process_launch_queue,
)
from scripts.materialize_task_evaluation_standing_launch_authorization import (
    main,
    materialize_standing_launch_authorization,
)


def _file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _profile(tmp_path: Path) -> tuple[Path, dict]:
    source = tmp_path / "source.json"
    spec = tmp_path / "spec.json"
    source.write_text('{"scene":"840920"}\n', encoding="utf-8")
    spec.write_text('{"task":"remove"}\n', encoding="utf-8")
    profile = {
        "schema_version": "task_evaluation_launch_profile.v1",
        "profile_id": "adp-sam31-source-tracks-live-exact-profile",
        "program_id": "arm-decision-proof-v1",
        "source_bundle": {
            "bundle_id": "scene840920-task-a",
            "source_kind": "interiorgs_sage",
            "uri": "gs://blueprint-runs/source.json",
            "digest": "sha256:" + "a" * 64,
        },
        "evaluation_run_spec": {
            "uri": "gs://blueprint-runs/spec.json",
            "digest": "sha256:" + "b" * 64,
        },
        "immutable_inputs": [
            {
                "name": "source_bundle_manifest",
                "path": str(source),
                "digest": _file_digest(source),
            },
            {
                "name": "evaluation_run_spec",
                "path": str(spec),
                "digest": _file_digest(spec),
            },
        ],
        "allocator": {
            "entrypoint": CANONICAL_ALLOCATOR_ENTRYPOINT,
            "subcommand": "gpu-canary",
            "argv": ["--provider", "vast"],
            "max_spend_usd": 1.0,
            "hard_ttl_seconds": 1800,
            "retry_cap": 0,
        },
        "runtime_environment": {},
        "execution_admission": {
            "live_enabled": True,
            "readiness_receipt": {
                "uri": "gs://blueprint-runs/readiness.json",
                "digest": "sha256:" + "c" * 64,
            },
            "blockers": [],
        },
        "reconciliation": {
            "required_providers": ["vast"],
            "max_guard_age_seconds": 300,
        },
        "webapp_sync": {"max_attempts": 20},
        "terminal_contract": {
            "result_path": str(tmp_path / "result.json"),
            "success_statuses": ["completed"],
            "required_values": {"continuing_spend_from_this_run": False},
            "required_path_fields": [
                "teardown_manifest_path",
                "artifact_manifest_path",
            ],
        },
        "required_controls": {
            "canonical_allocator": CANONICAL_ALLOCATOR_ENTRYPOINT,
            "secret_profile_id": "canonical-vast-adp",
            "watchdog_required": True,
            "artifact_storage_required": True,
            "teardown_required": True,
            "provider_zero_required": True,
            "webapp_status_sync_required": True,
            "retry_cap": 0,
        },
        "claim_ceiling": "development_only",
    }
    profile["profile_digest"] = canonical_digest(profile, digest_field="profile_digest")
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile), encoding="utf-8")
    return path, profile


def _times() -> tuple[str, str]:
    now = datetime.now(timezone.utc)
    return now.isoformat(), (now + timedelta(hours=1)).isoformat()


def _request(profile: dict) -> dict:
    now = datetime.now(timezone.utc)
    request = {
        "schema_version": "task_evaluation_launch_request.v1",
        "launch_id": "website-minted-after-standing-authority",
        "run_id": "website-run-minted-after-standing-authority",
        "launch_profile_id": profile["profile_id"],
        "launch_profile_digest": profile["profile_digest"],
        "source_bundle": profile["source_bundle"],
        "evaluation_run_spec": profile["evaluation_run_spec"],
        "authorization": {
            "actor": {"id": "founder-001", "role": "admin"},
            "authorized_at": now.isoformat(),
            "rights": {
                "approved": True,
                "scope": "interiorgs_sage_simulator_evaluation",
                "evidence": {
                    "uri": "firestore://taskEvaluationLaunchAuthorities/rights-001",
                    "digest": "sha256:" + "d" * 64,
                },
            },
            "spend": {
                "approved": True,
                "currency": "USD",
                "max_spend_usd": 1.0,
                "expires_at": (now + timedelta(minutes=30)).isoformat(),
            },
            "execution": {"approved": True},
        },
        "required_controls": profile["required_controls"],
        "claim_ceiling": "development_only",
        "idempotency_key": "website-minted-after-standing-authority",
    }
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    return request


def test_materializer_installs_one_service_readable_exact_profile_authorization(
    tmp_path: Path,
) -> None:
    profile_path, profile = _profile(tmp_path)
    issued_at, expires_at = _times()

    receipt = materialize_standing_launch_authorization(
        profile_path=profile_path,
        output_dir=tmp_path / "standing-authorizations",
        authorized_by="authorized-human",
        authorization_reference="user-authorized-paid-task-a",
        issued_at=issued_at,
        expires_at=expires_at,
        max_launches=1,
        max_total_spend_usd=1.0,
    )

    target = Path(receipt["authorization_path"])
    value = json.loads(target.read_text(encoding="utf-8"))
    assert value["profile_id"] == profile["profile_id"]
    assert value["profile_digest"] == profile["profile_digest"]
    assert value["max_launches"] == 1
    assert value["max_total_spend_usd"] == 1.0
    assert value["provider_mutation_performed"] is False
    assert stat.S_IMODE(target.stat().st_mode) == 0o440
    assert os.access(target, os.R_OK)
    assert receipt["authorization_file_digest"] == _file_digest(target)
    assert receipt["provider_mutation_performed"] is False


def test_materializer_refuses_to_replace_a_different_authorization(tmp_path: Path) -> None:
    profile_path, _ = _profile(tmp_path)
    issued_at, expires_at = _times()
    output_dir = tmp_path / "standing-authorizations"
    common = {
        "profile_path": profile_path,
        "output_dir": output_dir,
        "authorized_by": "authorized-human",
        "authorization_reference": "first-authority",
        "issued_at": issued_at,
        "expires_at": expires_at,
        "max_launches": 1,
        "max_total_spend_usd": 1.0,
    }
    materialize_standing_launch_authorization(**common)

    try:
        materialize_standing_launch_authorization(
            **{**common, "authorization_reference": "different-authority"}
        )
    except ValueError as exc:
        assert "standing_authorization_immutable_conflict" in str(exc)
    else:  # pragma: no cover - assertion aid
        raise AssertionError("different authorization bytes were overwritten")


def test_failed_consumer_readback_never_exposes_an_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile_path, profile = _profile(tmp_path)
    issued_at, expires_at = _times()
    output_dir = tmp_path / "standing-authorizations"
    monkeypatch.setattr(materializer, "_digest_as_account", lambda *args, **kwargs: "")

    with pytest.raises(ValueError, match="consumer_readback_failed"):
        materialize_standing_launch_authorization(
            profile_path=profile_path,
            output_dir=output_dir,
            authorized_by="authorized-human",
            authorization_reference="pre-submit-task-a-authority",
            issued_at=issued_at,
            expires_at=expires_at,
            max_launches=1,
            max_total_spend_usd=1.0,
        )

    assert not (output_dir / f"{profile['profile_id']}.json").exists()


def test_cli_materializes_before_launch_id_exists_and_is_idempotent(
    tmp_path: Path, capsys,
) -> None:
    profile_path, profile = _profile(tmp_path)
    output_dir = tmp_path / "standing-authorizations"
    issued_at, expires_at = _times()
    argv = [
        "--profile",
        str(profile_path),
        "--output-dir",
        str(output_dir),
        "--authorized-by",
        "authorized-human",
        "--authorization-reference",
        "pre-submit-task-a-authority",
        "--issued-at",
        issued_at,
        "--expires-at",
        expires_at,
        "--max-launches",
        "1",
        "--max-total-spend-usd",
        "1.0",
    ]

    assert main(argv) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["created"] is True
    assert "launch_id" not in first
    assert main(argv) == 0
    second = json.loads(capsys.readouterr().out)
    assert second["created"] is False
    assert Path(second["authorization_path"]).name == f"{profile['profile_id']}.json"


def test_pre_submit_authorization_removes_the_launch_id_race(
    tmp_path: Path, monkeypatch,
) -> None:
    """The dispatcher may consume immediately after the website persists the ID."""

    profile_path, profile = _profile(tmp_path)
    state_root = tmp_path / "control-plane" / "launch-runs"
    standing_dir = state_root.parent / "standing-authorizations"
    issued_at, expires_at = _times()
    materialize_standing_launch_authorization(
        profile_path=profile_path,
        output_dir=standing_dir,
        authorized_by="authorized-human",
        authorization_reference="pre-submit-task-a-authority",
        issued_at=issued_at,
        expires_at=expires_at,
        max_launches=1,
        max_total_spend_usd=1.0,
    )

    # Only now does the website mint and persist its unpredictable launch ID.
    request = _request(profile)
    queue_root = tmp_path / "queue"
    pending = queue_root / "pending"
    pending.mkdir(parents=True)
    (pending / f"{request['launch_id']}-request.json").write_text(
        json.dumps(request), encoding="utf-8"
    )
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    (profile_dir / f"{profile['profile_id']}.json").write_text(
        json.dumps(profile), encoding="utf-8"
    )
    monkeypatch.setenv(EXECUTE_ENV, "true")
    monkeypatch.setenv(SECRET_PROFILE_ID_ENV, "canonical-vast-adp")
    monkeypatch.delenv(
        "BLUEPRINT_TASK_EVALUATION_STANDING_AUTHORIZATION_DIR", raising=False
    )
    calls: list[list[str]] = []

    report = process_launch_queue(
        queue_root=queue_root,
        profile_dir=profile_dir,
        state_root=state_root,
        execute=True,
        execute_launch_id=None,
        allocator_runner=lambda argv: calls.append(list(argv)) or 0,
    )

    assert report["processed_count"] == 1
    assert len(calls) == 1
    assert "--execute" in calls[0]
    assert "execute_launch_id_required" not in report["receipts"][0]["blockers"]
    consumed = standing_dir / "consumed" / profile["profile_id"]
    assert [path.name for path in consumed.glob("*.json")] == [
        f"{request['launch_id']}.json"
    ]


def test_cli_refuses_expiry_or_spend_that_cannot_admit_the_profile(
    tmp_path: Path, capsys,
) -> None:
    profile_path, _ = _profile(tmp_path)
    now = datetime.now(timezone.utc)
    common = [
        "--profile",
        str(profile_path),
        "--output-dir",
        str(tmp_path / "standing-authorizations"),
        "--authorized-by",
        "authorized-human",
        "--authorization-reference",
        "bounded-task-a",
        "--issued-at",
        now.isoformat(),
        "--expires-at",
        (now + timedelta(hours=1)).isoformat(),
        "--max-launches",
        "1",
        "--max-total-spend-usd",
        "0.5",
    ]

    assert main(common) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "blocked"
    assert "standing_authorization_spend_ceiling_reached" in result["blockers"][0]
    assert result["provider_mutation_performed"] is False
    assert not (tmp_path / "standing-authorizations").exists()
