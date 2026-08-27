from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_scene_configuration_allocator as lane
from blueprint_pipeline.paid_resource_admission import PaidResourceAdmissionGrant


SOURCE_COMMIT = "a" * 40


def _args(tmp_path: Path, *, execute: bool) -> Namespace:
    authority = tmp_path / "authority.json"
    authority.write_text(
        json.dumps(
            {
                "resource_name": "blueprint-scene-configuration-fixture",
                "authority_digest": "sha256:" + "b" * 64,
                "container_image": "fixture/image@sha256:" + "c" * 64,
                "maximum_hourly_rate_usd": 0.8,
                "hard_attempt_spend_cap_usd": 2.25,
                "maximum_single_resource_ttl_seconds": 3600,
            }
        ),
        encoding="utf-8",
    )
    receipt = tmp_path / "bundle-receipt.json"
    receipt.write_text("{}\n", encoding="utf-8")
    return Namespace(
        scene_configuration_bundle_receipt=str(receipt),
        scene_configuration_attempt_authority=str(authority),
        scene_configuration_job_dir=str(tmp_path / "job"),
        expected_source_commit=SOURCE_COMMIT,
        provider="vast",
        pod_name="blueprint-scene-configuration-fixture",
        admission_out=str(tmp_path / "admission.json"),
        adapter_output=str(tmp_path / "result.json"),
        execute=execute,
        scene_configuration_diagnostic_only=False,
        release_evidence=None,
    )


def _identity() -> tuple[list[str], dict[str, object]]:
    return [], {"orchestrator_source_commit": SOURCE_COMMIT}


def _expected_source(
    expected: str, identity: dict[str, object]
) -> tuple[list[str], str]:
    assert identity["orchestrator_source_commit"] == SOURCE_COMMIT
    return [], expected


def _prepared_bundle() -> dict[str, object]:
    return {
        "run_id": "scene-run-fixture",
        "bundle_sha256": "sha256:" + "d" * 64,
        "portable_construction_envelope_digest": "sha256:" + "e" * 64,
        "toolchain_digest": "sha256:" + "f" * 64,
    }


def test_scene_allocator_preserves_admission_binding_and_queue_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path, execute=True)
    calls: list[dict[str, object]] = []
    monkeypatch.setenv(
        "BLUEPRINT_TASK_EVALUATION_SCENE_CONSTRUCTION_QUEUE_ROOT",
        str(tmp_path / "scene-queue"),
    )
    monkeypatch.setattr(
        lane,
        "resolve_host_resident_bundle_receipt",
        lambda _path: {"blockers": []},
    )
    monkeypatch.setattr(
        lane,
        "load_scene_configuration_provider_bundle_receipt",
        lambda _path, **_kwargs: _prepared_bundle(),
    )
    monkeypatch.setattr(
        lane, "validate_scene_configuration_paid_authority", lambda *_args, **_kwargs: None
    )

    def run(**kwargs):
        calls.append(kwargs)
        assert isinstance(kwargs["paid_resource_admission_grant"], PaidResourceAdmissionGrant)
        return {"status": "completed"}

    monkeypatch.setattr(lane, "run_scene_configuration_vast", run)

    exit_code = lane.run_scene_configuration_allocator_probe(
        args,
        control_plane_identity_probe=_identity,
        expected_source_commit_probe=_expected_source,
    )

    assert exit_code == 0
    assert calls == [
        {
            "job_dir": args.scene_configuration_job_dir,
            "bundle_receipt_path": Path(args.scene_configuration_bundle_receipt),
            "paid_attempt_authority_path": Path(
                args.scene_configuration_attempt_authority
            ),
            "paid_resource_admission_grant": calls[0][
                "paid_resource_admission_grant"
            ],
            "execute": True,
            "diagnostic_only": False,
            "retain_warm_session": False,
            "warm_session_authority_path": None,
            "warm_session_output_root": None,
            "scene_construction_queue_root": str(tmp_path / "scene-queue"),
        }
    ]
    admission = json.loads(Path(args.admission_out).read_text(encoding="utf-8"))
    assert admission["status"] == "admitted"
    assert admission["allocation_binding"] == {
        "program_id": "arm-decision-proof-v1",
        "probe_kind": lane.PROBE_KIND,
        "orchestrator_source_commit": SOURCE_COMMIT,
        "expected_source_commit": SOURCE_COMMIT,
        "run_id": "scene-run-fixture",
        "bundle_sha256": "sha256:" + "d" * 64,
        "portable_construction_envelope_digest": "sha256:" + "e" * 64,
        "toolchain_digest": "sha256:" + "f" * 64,
        "authority_digest": "sha256:" + "b" * 64,
        "resource_name": "blueprint-scene-configuration-fixture",
        "container_image": "fixture/image@sha256:" + "c" * 64,
        "max_hourly_rate_usd": 0.8,
        "hard_cap_usd": 2.25,
        "hard_ttl_seconds": 3600,
        "allowed_active_vast_instance_ids": [],
        "retry_cap": 0,
        "diagnostic_only": False,
        "retain_warm_session": False,
        "warm_session_authority_digest": None,
        "source_diagnostic_checkpoint_digest": None,
        "carried_completed_stage_count": None,
        "diagnostic_bootstrap_mode": None,
        "diagnostic_scientific_binding_digest": None,
        "diagnostic_stage_sequence_ids": None,
    }


def test_diagnostic_allocator_is_separate_nonpublishing_launch_surface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path, execute=True)
    args.scene_configuration_diagnostic_only = True
    args.release_evidence = str(
        tmp_path / "diagnostic-release.json"
    )
    diagnostic_release_digest = "sha256:" + "2" * 64
    monkeypatch.setattr(
        lane,
        "validate_scene_configuration_diagnostic_release_receipt",
        lambda path, **kwargs: {
            "receipt_digest": diagnostic_release_digest,
            "remote_ref": "refs/heads/codex/scene-fix",
            "remote_ref_tip_commit": SOURCE_COMMIT,
        }
        if path == Path(args.release_evidence)
        and kwargs == {
            "expected_source_commit": SOURCE_COMMIT,
            "expected_release_path": lane.ROOT,
        }
        else pytest.fail("diagnostic release validated against wrong identity"),
    )
    checkpoint_digest = "sha256:" + "1" * 64
    scientific_binding_digest = "sha256:" + "3" * 64
    prepared = {
        **_prepared_bundle(),
        "diagnostic_only": True,
        "source_diagnostic_checkpoint_digest": checkpoint_digest,
        "carried_completed_stage_count": 3,
        "diagnostic_bootstrap_mode": "checkpoint_resume",
        "diagnostic_scientific_binding_digest": scientific_binding_digest,
        "diagnostic_stage_sequence_ids": [
            f"stage-{index + 1}" for index in range(6)
        ],
    }
    monkeypatch.setattr(
        lane,
        "resolve_host_resident_bundle_receipt",
        lambda _path: {"blockers": []},
    )
    monkeypatch.setattr(
        lane,
        "load_scene_configuration_provider_bundle_receipt",
        lambda _path, **kwargs: prepared
        if kwargs.get("diagnostic_only") is True
        else pytest.fail("diagnostic bundle opened as production"),
    )
    monkeypatch.setattr(
        lane,
        "validate_scene_configuration_paid_authority",
        lambda *_args, **_kwargs: None,
    )
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        lane,
        "run_scene_configuration_vast",
        lambda **kwargs: calls.append(kwargs)
        or {"status": "completed_diagnostic_only"},
    )

    assert (
        lane.run_scene_configuration_allocator_probe(
            args,
            control_plane_identity_probe=_identity,
            expected_source_commit_probe=_expected_source,
        )
        == 0
    )
    assert calls[0]["diagnostic_only"] is True
    assert calls[0]["scene_construction_queue_root"] is None
    admission = json.loads(Path(args.admission_out).read_text(encoding="utf-8"))
    assert admission["diagnostic_only"] is True
    assert admission["qualification_eligible"] is False
    assert admission["configured_revision_publication_permitted"] is False
    assert admission["offering_publication_permitted"] is False
    assert admission["terminal_e2e_completion_permitted"] is False
    assert admission["allocation_binding"][
        "source_diagnostic_checkpoint_digest"
    ] == checkpoint_digest
    assert admission["allocation_binding"]["carried_completed_stage_count"] == 3
    assert admission["allocation_binding"]["diagnostic_bootstrap_mode"] == (
        "checkpoint_resume"
    )
    assert admission["allocation_binding"][
        "diagnostic_scientific_binding_digest"
    ] == scientific_binding_digest
    assert admission["allocation_binding"][
        "diagnostic_stage_sequence_ids"
    ] == [f"stage-{index + 1}" for index in range(6)]
    assert admission["allocation_binding"][
        "diagnostic_release_receipt_digest"
    ] == diagnostic_release_digest
    assert admission["allocation_binding"]["diagnostic_remote_ref"] == (
        "refs/heads/codex/scene-fix"
    )
    assert admission["allocation_binding"][
        "diagnostic_remote_ref_tip_commit"
    ] == SOURCE_COMMIT


def test_diagnostic_allocator_refuses_arbitrary_detached_checkout_without_release_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path, execute=True)
    args.scene_configuration_diagnostic_only = True
    monkeypatch.setattr(
        lane,
        "resolve_host_resident_bundle_receipt",
        lambda _path: {"blockers": []},
    )
    monkeypatch.setattr(
        lane,
        "load_scene_configuration_provider_bundle_receipt",
        lambda _path, **_kwargs: {
            **_prepared_bundle(),
            "diagnostic_only": True,
        },
    )
    monkeypatch.setattr(
        lane,
        "validate_scene_configuration_paid_authority",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        lane,
        "run_scene_configuration_vast",
        lambda **_kwargs: pytest.fail("provider reached without pushed release receipt"),
    )

    assert (
        lane.run_scene_configuration_allocator_probe(
            args,
            control_plane_identity_probe=_identity,
            expected_source_commit_probe=_expected_source,
        )
        == 2
    )
    admission = json.loads(Path(args.admission_out).read_text(encoding="utf-8"))
    result = json.loads(Path(args.adapter_output).read_text(encoding="utf-8"))
    assert admission["blockers"] == [
        "scene_configuration_diagnostic_release_receipt_missing"
    ]
    assert result["provider_mutations_performed"] == 0
    assert result["diagnostic_only"] is True
    assert result["qualification_eligible"] is False
    assert result["configured_revision_publication_permitted"] is False
    assert result["offering_publication_permitted"] is False
    assert result["terminal_e2e_completion_permitted"] is False


def test_scene_allocator_refuses_before_adapter_when_admission_is_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path, execute=True)
    args.provider = "runpod"
    monkeypatch.setattr(
        lane,
        "resolve_host_resident_bundle_receipt",
        lambda _path: {"blockers": []},
    )
    monkeypatch.setattr(
        lane,
        "load_scene_configuration_provider_bundle_receipt",
        lambda _path, **_kwargs: _prepared_bundle(),
    )
    monkeypatch.setattr(
        lane, "validate_scene_configuration_paid_authority", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        lane,
        "run_scene_configuration_vast",
        lambda **_kwargs: pytest.fail("adapter reached after blocked admission"),
    )

    exit_code = lane.run_scene_configuration_allocator_probe(
        args,
        control_plane_identity_probe=_identity,
        expected_source_commit_probe=_expected_source,
    )

    assert exit_code == 2
    admission = json.loads(Path(args.admission_out).read_text(encoding="utf-8"))
    result = json.loads(Path(args.adapter_output).read_text(encoding="utf-8"))
    assert admission["status"] == "blocked"
    assert admission["blockers"] == ["scene_configuration_provider_must_be_vast"]
    assert result == {
        "status": "blocked",
        "blockers": [
            "paid_resource_admission_has_blockers",
            "paid_resource_admission_not_admitted",
        ],
        "provider_mutations_performed": 0,
        "retry_cap": 0,
    }
