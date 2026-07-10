from __future__ import annotations

import json
import zipfile
from pathlib import Path

from scripts.build_gpu_provider_canary_evidence import (
    MAX_PROVIDER_OUTPUT_MEMBER_SIZE,
    MAX_RAW_JSON_SIZE,
    RAW_RESULT_NAME,
    SAFE_SUMMARY_NAME,
    build_gpu_provider_canary_evidence,
    validate_canary_inputs,
)


SHA = "a" * 40
IMAGE = f"registry.example/blueprint/unitree@sha256:{'b' * 64}"


def _seed_result(job_dir: Path) -> Path:
    run_dir = job_dir / "vast_provider_run"
    run_dir.mkdir(parents=True)
    artifact_names = {
        "vast_provider_adapter_result_path": "vast_provider_adapter_result.json",
        "vast_startup_probe_manifest_path": "vast_startup_probe_manifest.json",
        "vast_gpu_sanity_report_path": "vast_gpu_sanity_report.json",
        "vast_provider_command_result_path": "vast_provider_command_result.json",
        "vast_teardown_manifest_path": "vast_teardown_manifest.json",
        "vast_budget_ledger_path": "vast_budget_ledger.json",
    }
    paths: dict[str, str] = {}
    for field, filename in artifact_names.items():
        path = run_dir / filename
        path.write_text(json.dumps({"status": "completed"}), encoding="utf-8")
        paths[field] = str(path)
    Path(paths["vast_provider_adapter_result_path"]).write_text(
        json.dumps(
            {
                "status": "completed",
                "reason": "https://signed.example/private?token=do-not-retain",
                "private_path": "/Users/runner/private/job",
            }
        ),
        encoding="utf-8",
    )
    output_zip = run_dir / "vast_provider_runtime_output.zip"
    with zipfile.ZipFile(output_zip, "w") as archive:
        archive.writestr(
            "unitree_groot_n17_sonic_policy_provider_output.json",
            json.dumps(
                {
                    "schema_version": "unitree_groot_n17_sonic_policy_provider_output.v1",
                    "status": "completed",
                    "canary_only": True,
                    "canary_marker": "BLUEPRINT_UNITREE_GROOT_N17_SONIC_VAST_IMAGE_CANARY_OK",
                    "blockers": [],
                    "unitree_groot_n17_sonic_model_executed": False,
                    "unitree_groot_n17_sonic_policy_action_command_ran": False,
                    "policy_action_model_command_ran": False,
                    "raw_credentials_written_to_artifacts": False,
                    "secret_hashes_written_to_artifacts": False,
                    "checks": {
                        "python": {"returncode": 0, "duration_seconds": 0.1},
                        "nvidia_smi": {"returncode": 0, "duration_seconds": 0.2},
                    },
                }
            ),
        )
    (job_dir / SAFE_SUMMARY_NAME).write_text(
        json.dumps({"status": "completed", "raw_secret_values_recorded": False}),
        encoding="utf-8",
    )
    result = {
        "schema_version": "unitree_groot_n17_sonic_vast_image_canary.v1",
        "status": "completed",
        "blockers": [],
        "public_image": IMAGE,
        "min_gpu_ram_mb": 48000,
        "heartbeat_completed": True,
        "gpu_sanity_completed": True,
        "provider_bundle_downloaded_and_ran": True,
        "provider_output_upload_ok": True,
        "provider_runtime_output_zip_produced": True,
        "canary_marker_observed": True,
        "continuing_spend_from_this_run": False,
        "vast_instance_ids": [123],
        "estimated_cost_usd": 0.05,
        "actual_live_runtime_seconds": 600.0,
        "selected_hourly_rate_usd": 0.3,
        "selected_offer": {
            "machine_id": 456,
            "gpu_ram_mb": 49152,
            "hourly_rate_usd": 0.3,
            "ask_contract_id": 999,
            "gpu_model_slug": "RTX_A6000",
            "signed_url": "https://signed.example/private?token=do-not-retain",
        },
        "unexpected_secret": "do-not-retain",
        "provider_output_zip_path": str(output_zip),
        "raw_secret_values_recorded": False,
        "claim_boundary": {
            "canary_is_not_policy_inference": True,
            "custom_image_startup_proof_only": True,
            "generated_world_rank_fidelity_result_proven": False,
            "generated_world_policy_evaluation_scope_proven": False,
            "accepted_anchor_manipulation_success_proven": False,
        },
        **paths,
    }
    result_path = job_dir / RAW_RESULT_NAME
    result_path.write_text(json.dumps(result), encoding="utf-8")
    return result_path


def _build(result_path: Path, job_dir: Path) -> dict[str, object]:
    return build_gpu_provider_canary_evidence(
        result_path=result_path,
        job_dir=job_dir,
        evidence_dir=job_dir.parent / "sanitized-evidence",
        image_uri=IMAGE,
        approved_image_uri=IMAGE,
        repository_sha=SHA,
        max_hourly_rate=0.6,
        target_spend_usd=0.2,
        hard_cap_usd=0.5,
        max_live_minutes=20,
        startup_timeout_seconds=900,
    )


def test_gpu_canary_converter_requires_exact_executed_result_and_teardown(
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / "gpu-canary-run"
    result_path = _seed_result(job_dir)

    evidence = _build(result_path, job_dir)

    assert evidence["status"] == "passed"
    assert evidence["executed"] is True
    assert evidence["image_uri"] == IMAGE
    assert evidence["image_digest"] == f"sha256:{'b' * 64}"
    assert evidence["result_contract"]["continuing_spend_from_this_run"] is False
    assert len(evidence["artifact_digests"]) == 9
    assert evidence["source_manifest_digest"].startswith("sha256:")
    assert evidence["source_bundle_digest"].startswith("sha256:")
    assert (job_dir.parent / "sanitized-evidence" / "gpu-canary-source-manifest.json").is_file()
    assert (
        job_dir.parent / "sanitized-evidence" / "gpu-provider-canary-sanitized-evidence.zip"
    ).is_file()
    assert evidence["claim_boundary"]["evid_03_remains_external"] is True
    assert str(tmp_path) not in json.dumps(evidence)
    with zipfile.ZipFile(
        job_dir.parent / "sanitized-evidence" / "gpu-provider-canary-sanitized-evidence.zip"
    ) as archive:
        retained = b"\n".join(archive.read(name) for name in archive.namelist())
    assert b"do-not-retain" not in retained
    assert b"/Users/runner/private/job" not in retained
    assert b"ask_contract_id" not in retained


def test_gpu_canary_inputs_reject_tags_unbounded_spend_and_time() -> None:
    blockers = validate_canary_inputs(
        image_uri="registry.example/blueprint/unitree:latest",
        approved_image_uri=f"registry.example/approved@sha256:{'c' * 64}",
        repository_sha="short",
        max_hourly_rate=1.01,
        target_spend_usd=2.0,
        hard_cap_usd=1.01,
        max_live_minutes=31,
        startup_timeout_seconds=1900,
    )

    assert "gpu_canary_image_not_exact_digest" in blockers
    assert "gpu_canary_image_not_approved" in blockers
    assert "gpu_canary_repository_sha_invalid" in blockers
    assert "gpu_canary_max_hourly_rate_exceeds_contract" in blockers
    assert "gpu_canary_hard_cap_exceeds_contract" in blockers
    assert "gpu_canary_target_spend_exceeds_hard_cap" in blockers
    assert "gpu_canary_max_live_minutes_out_of_bounds" in blockers
    assert "gpu_canary_startup_timeout_out_of_bounds" in blockers

    projected = validate_canary_inputs(
        image_uri=IMAGE,
        approved_image_uri=IMAGE,
        repository_sha=SHA,
        max_hourly_rate=0.6,
        target_spend_usd=0.05,
        hard_cap_usd=0.1,
        max_live_minutes=20,
        startup_timeout_seconds=600,
    )
    assert "gpu_canary_projected_max_cost_exceeds_hard_cap" in projected


def test_gpu_canary_converter_blocks_continuing_spend_and_missing_artifact(
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / "gpu-canary-run"
    result_path = _seed_result(job_dir)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["continuing_spend_from_this_run"] = True
    Path(payload["vast_teardown_manifest_path"]).unlink()
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    evidence = _build(result_path, job_dir)

    assert evidence["status"] == "blocked"
    assert "gpu_canary_teardown_not_proven" in evidence["blockers"]
    assert "gpu_canary_artifact_missing:vast_teardown_manifest.json" in evidence["blockers"]


def test_gpu_canary_converter_rejects_wrong_approved_image(tmp_path: Path) -> None:
    job_dir = tmp_path / "gpu-canary-run"
    result_path = _seed_result(job_dir)

    evidence = build_gpu_provider_canary_evidence(
        result_path=result_path,
        job_dir=job_dir,
        evidence_dir=tmp_path / "sanitized-evidence",
        image_uri=IMAGE,
        approved_image_uri=f"registry.example/other@sha256:{'c' * 64}",
        repository_sha=SHA,
        max_hourly_rate=0.6,
        target_spend_usd=0.2,
        hard_cap_usd=0.5,
        max_live_minutes=20,
        startup_timeout_seconds=900,
    )

    assert evidence["status"] == "blocked"
    assert "gpu_canary_image_not_approved" in evidence["blockers"]


def test_gpu_canary_converter_rejects_oversize_result_before_parsing(
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / "gpu-canary-run"
    result_path = _seed_result(job_dir)
    with result_path.open("wb") as handle:
        handle.truncate(MAX_RAW_JSON_SIZE + 1)

    evidence = _build(result_path, job_dir)

    assert evidence["status"] == "blocked"
    assert "gpu_canary_result_oversize" in evidence["blockers"]
    assert f"gpu_canary_artifact_oversize:{RAW_RESULT_NAME}" in evidence["blockers"]


def test_gpu_canary_converter_rejects_zip_bomb_before_decompression(
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / "gpu-canary-run"
    result_path = _seed_result(job_dir)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    output_zip = Path(payload["provider_output_zip_path"])
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "unitree_groot_n17_sonic_policy_provider_output.json",
            b"0" * (MAX_PROVIDER_OUTPUT_MEMBER_SIZE + 1),
        )

    evidence = _build(result_path, job_dir)

    assert evidence["status"] == "blocked"
    assert "gpu_canary_provider_output_member_oversize" in evidence["blockers"]
    assert "gpu_canary_provider_output_compression_ratio_exceeded" in evidence["blockers"]


def test_gpu_canary_converter_never_reads_or_retains_unvalidated_paths(
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / "gpu-canary-run"
    result_path = _seed_result(job_dir)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    outside_json = tmp_path / "outside-secret.json"
    outside_json.write_text(json.dumps({"status": "short-secret-value"}), encoding="utf-8")
    outside_zip = tmp_path / "outside-secret.zip"
    with zipfile.ZipFile(outside_zip, "w") as archive:
        archive.writestr(
            "unitree_groot_n17_sonic_policy_provider_output.json",
            json.dumps({"status": "zip-secret-value"}),
        )
    outside_zip_link = job_dir / "vast_provider_run" / "outside-secret-link.zip"
    outside_zip_link.symlink_to(outside_zip)
    payload["vast_provider_adapter_result_path"] = str(outside_json)
    payload["provider_output_zip_path"] = str(outside_zip_link)
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    evidence = _build(result_path, job_dir)

    assert evidence["status"] == "blocked"
    assert (
        "gpu_canary_artifact_outside_job:vast_provider_adapter_result.json" in evidence["blockers"]
    )
    assert (
        "gpu_canary_artifact_outside_job:vast_provider_runtime_output.zip" in evidence["blockers"]
    )
    retained = b"\n".join(
        path.read_bytes() for path in (job_dir.parent / "sanitized-evidence").glob("*.json")
    )
    assert b"short-secret-value" not in retained
    assert b"zip-secret-value" not in retained
