from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import subprocess
import time
import zipfile
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from blueprint_pipeline import paid_resource_allocator as allocator
from blueprint_pipeline import single_g1_kitchen_qualification_session as qualification
from blueprint_pipeline import gpu_render_providers
from blueprint_pipeline.groot_oscar_runpod_watchdog import OWNER_TEARDOWN_CANCEL_NAME
from blueprint_pipeline.task_episode_baseline import build_task_episode_baseline
from blueprint_pipeline.single_g1_kitchen_qualification_contract import (
    MODEL_ASSETS_BINDING_SCHEMA_VERSION,
    OFFICIAL_GEAR_SONIC_RUNTIME_COMMAND,
    capture_runtime_attempt_overlay_base,
    embedded_launch_rebind,
)


TEST_SOURCE_COMMIT = "a" * 40
TEST_IMAGE_REF = "registry.example/blueprint-eval@sha256:" + "b" * 64
TEST_IMAGE_DIGEST = "sha256:" + "b" * 64


def _release_evidence(**overrides: object) -> dict:
    release = {
        "schema_version": "groot_oscar_thin_remote_build_result.v1",
        "status": "completed",
        "blockers": [],
        "source_commit": TEST_SOURCE_COMMIT,
        "source_patch_sha256": hashlib.sha256(b"").hexdigest(),
        "resolved_digest_ref": TEST_IMAGE_REF,
        "release_image_ref": TEST_IMAGE_REF,
        "runnable_platform": "linux/amd64",
        "models_embedded": False,
        "required_cuda_version": "12.8",
        "required_cuda_version_source": (
            "image_config_env:BLUEPRINT_GROOT_OSCAR_REQUIRED_CUDA_VERSION"
        ),
        "thin_release_contract_status": "passed",
        "thin_release_contract": {
            "schema_version": "groot_oscar_thin_release_image_contract.v1",
            "status": "passed",
            "blockers": [],
            "release_image_ref": TEST_IMAGE_REF,
            "models_externalized": True,
        },
    }
    release.update(overrides)
    return release


def _minimal_inputs() -> dict:
    return {
        "plan": {
            "sealed_active": True,
            "blockers": [],
            "image_ref": "registry.example/source@sha256:" + "1" * 64,
            "env": {"PYTHONPATH": "/workspace/runtime_overlay/package"},
            "groot_server_command": ["/opt/gr00t/server", "--port", "5550"],
            "gear_sonic_controller_command": ["/opt/wbc/deploy.sh", "sim"],
            "isaac_task_executor_command": [
                "/isaac-sim/python.sh",
                "/workspace/runtime_overlay/run_patched_isaac_executor.py",
            ],
        },
        "attempt": {
            "image_digest": "sha256:" + "1" * 64,
            "source_commit": "f" * 40,
            "source_dirty_patch_sha256": "9" * 64,
            "launch_nonce": "prepared-source-nonce",
            "artifacts": {
                "task_success_contract": {
                    "path": "/source/task_success_contract.json",
                    "sha256": "9" * 64,
                    "size_bytes": 459,
                }
            },
        },
        "route": {"route": []},
        "seed": {"seed": 1001},
        "task_contract": {
            "schema_version": "task_success_contract.v1",
            "task_id": "microwave_door",
            "registered_criteria": [],
        },
        "source_task_contract_sha256": "6" * 64,
        "start_frame": b"exact-frame",
        "bootstrap_script": (
            "/opt/oscar-venv/bin/python -m "
            "blueprint_pipeline.gear_sonic_process_supervisor supervise -- "
            "/opt/wbc/deploy.sh sim > /workspace/gear_sonic_controller.log 2>&1 &\n"
            "cp /workspace/attempt_input_manifest_episode_001.json "
            "/workspace/attempt_input_manifest.json\n"
            "upload_phase inputs_ready\necho exact-episode\n"
        ),
        "runtime_package_overlay_xz_base64": "runtime-overlay",
        "isaac_runtime_backend_overlay_gzip_base64": "backend-overlay",
        "runtime_package_overlay_sha256": "1" * 64,
        "runtime_package_overlay_source_sha256s": {},
        "controller_fk_camera_projection_context": {
            "schema_version": "controller_fk_camera_projection_context.v1"
        },
        "controller_fk_camera_projection_context_bytes": b"{}\n",
        "controller_fk_camera_projection_context_sha256": "5" * 64,
        "bundle_sha256": qualification.BUNDLE_SHA256,
    }


def _external_model_cache_evidence(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": "groot_oscar_external_model_cache_verification.v2",
                "status": "passed",
                "blockers": [],
                "model_manifest_digest": "sha256:" + "d" * 64,
                "cache_root": "/models/blueprint-groot-oscar-v1",
                "checks": {"models_cached_offline": True},
                "raw_secret_values_recorded": False,
            }
        ),
        encoding="utf-8",
    )
    return path


def _embedded_release_evidence(**overrides: object) -> dict:
    foundation_ref = "registry.example/carrier@sha256:" + "c" * 64
    release = _release_evidence(
        models_embedded=True,
        foundation_model_assets="embedded",
        foundation_image_ref=foundation_ref,
        **overrides,
    )
    release["thin_release_contract"].update(
        {
            "foundation_model_assets": "embedded",
            "models_externalized": False,
            "models_embedded_in_foundation": True,
        }
    )
    return release


def _embedded_model_assets_evidence(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": MODEL_ASSETS_BINDING_SCHEMA_VERSION,
                "status": "bound_to_release_foundation",
                "source_commit": TEST_SOURCE_COMMIT,
                "release_image_ref": TEST_IMAGE_REF,
                "foundation_image_ref": (
                    "registry.example/carrier@sha256:" + "c" * 64
                ),
                "external_provider_volume_used": False,
                "runtime_checkpoint_preflight_required": True,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_release_binding_accepts_exact_digest_pinned_clean_release() -> None:
    binding, blockers = qualification._release_binding(
        _release_evidence(), expected_source_commit=TEST_SOURCE_COMMIT
    )

    assert blockers == []
    assert binding["image_ref"] == TEST_IMAGE_REF
    assert binding["image_digest"] == TEST_IMAGE_DIGEST
    assert binding["source_commit"] == TEST_SOURCE_COMMIT
    assert binding["models_externalized"] is True


def test_release_binding_accepts_exact_embedded_foundation() -> None:
    foundation_ref = "registry.example/carrier@sha256:" + "c" * 64
    release = _release_evidence(
        models_embedded=True,
        foundation_model_assets="embedded",
        foundation_image_ref=foundation_ref,
    )
    release["thin_release_contract"].update(
        {
            "foundation_model_assets": "embedded",
            "models_externalized": False,
            "models_embedded_in_foundation": True,
        }
    )

    binding, blockers = qualification._release_binding(
        release, expected_source_commit=TEST_SOURCE_COMMIT
    )

    assert blockers == []
    assert binding["model_assets_mode"] == "embedded"
    assert binding["models_externalized"] is False
    assert binding["models_embedded_in_foundation"] is True
    assert binding["foundation_image_ref"] == foundation_ref


def test_embedded_model_assets_binding_requires_exact_release_and_foundation() -> None:
    foundation_ref = "registry.example/carrier@sha256:" + "c" * 64
    release = _release_evidence(
        models_embedded=True,
        foundation_model_assets="embedded",
        foundation_image_ref=foundation_ref,
    )
    release["thin_release_contract"].update(
        {
            "foundation_model_assets": "embedded",
            "models_externalized": False,
            "models_embedded_in_foundation": True,
        }
    )
    release_binding, release_blockers = qualification._release_binding(
        release, expected_source_commit=TEST_SOURCE_COMMIT
    )
    evidence = {
        "schema_version": MODEL_ASSETS_BINDING_SCHEMA_VERSION,
        "status": "bound_to_release_foundation",
        "source_commit": TEST_SOURCE_COMMIT,
        "release_image_ref": TEST_IMAGE_REF,
        "foundation_image_ref": foundation_ref,
        "external_provider_volume_used": False,
        "runtime_checkpoint_preflight_required": True,
    }

    binding, blockers = qualification.build_model_assets_binding(evidence, release_binding)

    assert release_blockers == []
    assert blockers == []
    assert binding["status"] == "bound"
    evidence["release_image_ref"] = "registry.example/wrong@sha256:" + "d" * 64
    _, mismatch_blockers = qualification.build_model_assets_binding(
        evidence, release_binding
    )
    assert mismatch_blockers == [
        "qualification_embedded_model_assets_release_image_ref_mismatch"
    ]


def test_external_model_cache_is_rejected_without_vast_delivery_binding(
    tmp_path: Path,
) -> None:
    release_binding, release_blockers = qualification._release_binding(
        _release_evidence(), expected_source_commit=TEST_SOURCE_COMMIT
    )
    evidence = json.loads(
        _external_model_cache_evidence(tmp_path / "external-cache.json").read_text()
    )

    binding, blockers = qualification.build_model_assets_binding(
        evidence, release_binding
    )

    assert release_blockers == []
    assert binding["status"] == "blocked"
    assert blockers == ["qualification_external_model_cache_vast_delivery_not_bound"]


def test_allocate_requires_model_assets_evidence() -> None:
    with pytest.raises(
        ValueError,
        match="qualification_allocate_arguments_missing:model_cache_evidence",
    ):
        qualification.run_qualification_session(
            action="allocate",
            session_manifest="session.json",
            episode_bundle="episode.zip",
            provider_bundle_url_file="bundle.txt",
            provider_output_put_url_file="put.txt",
            provider_output_get_url_file="get.txt",
            release_evidence="release.json",
            expected_source_commit=TEST_SOURCE_COMMIT,
            provider_launch_request="request.json",
            preflight_bundle="preflight.json",
            admission_out="admission.json",
            bound_request_out="bound.json",
            adapter_output="result.json",
        )


def test_qualification_release_rebind_preserves_source_and_derives_runtime_inputs() -> None:
    inputs = _minimal_inputs()
    source_ref = inputs["plan"]["image_ref"]
    release_binding, blockers = qualification._release_binding(
        _release_evidence(), expected_source_commit=TEST_SOURCE_COMMIT
    )

    rebound, binding = qualification.bind_inputs_to_release(inputs, release_binding)

    assert blockers == []
    assert inputs["plan"]["image_ref"] == source_ref
    assert inputs["plan"]["gear_sonic_controller_command"] == [
        "/opt/wbc/deploy.sh",
        "sim",
    ]
    assert inputs["attempt"]["image_digest"] == "sha256:" + "1" * 64
    assert rebound["plan"]["image_ref"] == TEST_IMAGE_REF
    assert rebound["plan"]["gear_sonic_controller_command"] == list(
        OFFICIAL_GEAR_SONIC_RUNTIME_COMMAND
    )
    assert rebound["attempt"]["image_digest"] == TEST_IMAGE_DIGEST
    assert rebound["attempt"]["source_commit"] == TEST_SOURCE_COMMIT
    assert rebound["attempt"]["source_dirty_patch_sha256"] == hashlib.sha256(b"").hexdigest()
    assert rebound["attempt"]["source_bundle_attempt_identity"] == {
        "source_commit": "f" * 40,
        "source_dirty_patch_sha256": "9" * 64,
        "image_digest": "sha256:" + "1" * 64,
        "preserved_from_source_bundle": True,
    }
    assert binding["source_image_ref"] == source_ref
    assert binding["release_image_ref"] == TEST_IMAGE_REF
    assert binding["source_bundle_sha256"] == qualification.BUNDLE_SHA256
    assert binding["source_bundle_preserved"] is True
    assert binding["runtime_attempt_manifest_rebound"] is True
    assert binding["gear_sonic_controller_runtime_mode"] == "prebuilt_release_binary"
    assert binding["gear_sonic_controller_command_rebound"] is True
    assert binding["bootstrap_gear_sonic_controller_command_rebound"] is True
    assert binding["source_gear_sonic_controller_command_sha256"] != binding[
        "release_gear_sonic_controller_command_sha256"
    ]
    assert binding[
        "source_bootstrap_gear_sonic_controller_command_sha256"
    ] != binding["release_bootstrap_gear_sonic_controller_command_sha256"]
    assert "/opt/wbc/deploy.sh" not in rebound["bootstrap_script"]
    rebound_controller_shell = (
        "/opt/oscar-venv/bin/python -m "
        "blueprint_pipeline.gear_sonic_process_supervisor supervise -- "
        "bash -lc 'cd /opt/wbc/gear_sonic_deploy && source scripts/setup_env.sh "
        "&& exec ./target/release/g1_deploy_onnx_ref lo "
        "policy/release/model_decoder.onnx reference/example "
        "--obs-config policy/release/observation_config.yaml "
        "--encoder-file policy/release/model_encoder.onnx "
        "--planner-file planner/target_vel/V2/planner_sonic.onnx "
        "--input-type zmq_manager --output-type zmq --zmq-host localhost "
        "--disable-crc-check'"
    )
    assert rebound["bootstrap_script"].count(rebound_controller_shell) == 1
    assert rebound["attempt"]["qualification_source_task_success_contract_sha256"] == (
        "6" * 64
    )
    assert rebound["attempt"]["qualification_runtime_attempt_overlay_base"][
        "task_success_contract_artifact"
    ]["value"]["sha256"] == "9" * 64
    rebound_attempt_bytes = (
        json.dumps(rebound["attempt"], indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    runtime_attempt = json.loads(json.dumps(rebound["attempt"]))
    runtime_attempt.update(
        {
            "prepared_launch_nonce": "prepared-source-nonce",
            "allocation_launch_session_id": "qualification-session-1",
            "launch_nonce": "qualification-session-1:attempt_0001",
            "qualification_attempt_bound": True,
            "qualification_attempt_sequence": 1,
            "qualification_attempt_nonce": "qualification-session-1:attempt_0001",
            "qualification_attempt_nonce_sha256": hashlib.sha256(
                b"qualification-session-1:attempt_0001"
            ).hexdigest(),
        }
    )
    runtime_attempt["artifacts"]["task_success_contract"] = {
        "path": "/workspace/task_success_contract.json",
        "sha256": runtime_attempt[
            "qualification_resolved_task_success_contract_sha256"
        ],
        "size_bytes": 500,
        "derived_from_sha256": runtime_attempt[
            "qualification_source_task_success_contract_sha256"
        ],
        "resolution_artifact_path": (
            "/workspace/closed_loop_out/direct_task_contract_resolution.json"
        ),
    }
    runtime_attempt_bytes = (
        json.dumps(runtime_attempt, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    assert qualification.collected_attempt_derived_artifact_blockers(
        runtime_attempt_bytes, runtime_attempt, binding
    ) == []
    assert qualification.collected_attempt_derived_artifact_blockers(
        runtime_attempt_bytes + b" ", runtime_attempt, binding
    ) == ["qualification_collected_attempt_manifest_digest_mismatch"]
    stale_contract_attempt = json.loads(json.dumps(runtime_attempt))
    stale_contract_attempt.pop("prepared_launch_nonce")
    stale_contract_attempt["artifacts"]["task_success_contract"] = json.loads(
        json.dumps(
            rebound["attempt"]["qualification_runtime_attempt_overlay_base"][
                "task_success_contract_artifact"
            ]["value"]
        )
    )
    stale_contract_bytes = (
        json.dumps(stale_contract_attempt, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    assert "qualification_collected_attempt_runtime_overlay_base_invalid" in (
        qualification.collected_attempt_derived_artifact_blockers(
            stale_contract_bytes, stale_contract_attempt, binding
        )
    )
    source_less_inputs = _minimal_inputs()
    source_less_inputs["attempt"]["artifacts"].pop("task_success_contract")
    source_less_rebound, source_less_binding = qualification.bind_inputs_to_release(
        source_less_inputs, release_binding
    )
    source_less_runtime_attempt = json.loads(
        json.dumps(source_less_rebound["attempt"])
    )
    source_less_runtime_attempt.update(
        {
            key: value
            for key, value in runtime_attempt.items()
            if key
            in {
                "prepared_launch_nonce",
                "allocation_launch_session_id",
                "launch_nonce",
                "qualification_attempt_bound",
                "qualification_attempt_sequence",
                "qualification_attempt_nonce",
                "qualification_attempt_nonce_sha256",
            }
        }
    )
    source_less_runtime_attempt.setdefault("artifacts", {})[
        "task_success_contract"
    ] = json.loads(json.dumps(runtime_attempt["artifacts"]["task_success_contract"]))
    source_less_runtime_bytes = (
        json.dumps(source_less_runtime_attempt, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    assert qualification.collected_attempt_derived_artifact_blockers(
        source_less_runtime_bytes,
        source_less_runtime_attempt,
        source_less_binding,
    ) == []
    wrong_contract_attempt = json.loads(json.dumps(runtime_attempt))
    wrong_contract_attempt["artifacts"]["task_success_contract"]["sha256"] = "8" * 64
    wrong_contract_bytes = (
        json.dumps(wrong_contract_attempt, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    assert "qualification_collected_attempt_runtime_overlay_base_invalid" in (
        qualification.collected_attempt_derived_artifact_blockers(
            wrong_contract_bytes, wrong_contract_attempt, binding
        )
    )
    runtime_attempt["source_commit"] = "3" * 40
    tampered_runtime_bytes = (
        json.dumps(runtime_attempt, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    assert "qualification_collected_attempt_manifest_digest_mismatch" in (
        qualification.collected_attempt_derived_artifact_blockers(
            tampered_runtime_bytes, runtime_attempt, binding
        )
    )
    tampered_attempt = json.loads(json.dumps(rebound["attempt"]))
    tampered_attempt["qualification_derived_launch_plan"]["image_ref"] = (
        "registry.example/tampered@sha256:" + "2" * 64
    )
    assert "qualification_collected_launch_plan_digest_mismatch" in (
        qualification.collected_attempt_derived_artifact_blockers(
            rebound_attempt_bytes, tampered_attempt, binding
        )
    )
    assert "cp /workspace/attempt_input_manifest_episode_001.json" not in rebound[
        "bootstrap_script"
    ]
    assert "Path('/workspace/attempt_input_manifest.json').write_bytes(payload)" in rebound[
        "bootstrap_script"
    ]


def test_qualification_release_rebind_rejects_source_attempt_image_mismatch() -> None:
    inputs = _minimal_inputs()
    inputs["attempt"]["image_digest"] = "sha256:" + "9" * 64
    release_binding, _ = qualification._release_binding(
        _release_evidence(), expected_source_commit=TEST_SOURCE_COMMIT
    )

    with pytest.raises(ValueError, match="qualification_source_attempt_image_mismatch"):
        qualification.bind_inputs_to_release(inputs, release_binding)


@pytest.mark.parametrize(
    ("overrides", "expected_source_commit", "expected_blocker"),
    [
        ({"source_commit": "c" * 40}, TEST_SOURCE_COMMIT,
         "qualification_release_source_commit_mismatch"),
        ({"resolved_digest_ref": "registry.example/blueprint-eval:latest"},
         TEST_SOURCE_COMMIT, "qualification_release_image_not_digest_pinned"),
        ({"source_patch_sha256": "d" * 64}, TEST_SOURCE_COMMIT,
         "qualification_release_source_patch_not_empty"),
        ({"status": "blocked"}, TEST_SOURCE_COMMIT,
         "qualification_release_evidence_not_completed"),
    ],
)
def test_release_binding_rejects_stale_mutable_dirty_or_blocked_release(
    overrides: dict, expected_source_commit: str, expected_blocker: str
) -> None:
    _binding, blockers = qualification._release_binding(
        _release_evidence(**overrides),
        expected_source_commit=expected_source_commit,
    )

    assert expected_blocker in blockers


def test_trained_checkpoint_override_is_exact_and_does_not_claim_qualification() -> None:
    inputs = _minimal_inputs()
    inputs["plan"]["groot_server_command"] = [
        "/opt/gr00t-venv/bin/python",
        "/opt/gr00t/gr00t/eval/run_gr00t_server.py",
        "--model-path",
        "/opt/blueprint/ckpts/sonic",
        "--port",
        "5550",
    ]

    updated = qualification._apply_trained_checkpoint_override(
        inputs,
        qualification.REMOTE_FINAL_CHECKPOINT,
    )

    command = updated["plan"]["groot_server_command"]
    assert command[command.index("--model-path") + 1] == (
        qualification.REMOTE_FINAL_CHECKPOINT
    )
    binding = updated["plan"]["qualification_checkpoint_override"]
    assert binding["same_session_training_required"] is True
    assert binding["open_loop_qualification_required"] is True
    assert binding["isaac_registered_transition_required"] is True
    assert binding["task_compatibility_claimed"] is False

    with pytest.raises(ValueError, match="checkpoint_path_not_fixed"):
        qualification._apply_trained_checkpoint_override(inputs, "/tmp/checkpoint-500")


def _bootstrap_metadata() -> dict:
    return {
        "provider_bootstrap_sha256": "1" * 64,
        "episode_bootstrap_sha256": "2" * 64,
        "control_script_sha256": "3" * 64,
        "refresh_installer_sha256": "4" * 64,
        "component_script_sha256s": {},
        "overlay_revision": 1,
        "control_contract_version": qualification.CONTROL_CONTRACT_VERSION,
    }


def _live_manifest(tmp_path: Path) -> Path:
    prefix = qualification.NAME_PREFIX_ROOT + "0123456789"
    nonce = "single-g1-kitchen-qualification-0123456789"
    release_binding = qualification._release_binding(
        _release_evidence(), expected_source_commit=TEST_SOURCE_COMMIT
    )[0]
    _, launch_rebind = qualification.bind_inputs_to_release(
        _minimal_inputs(), release_binding
    )
    manifest = qualification._manifest_base(
        root=tmp_path,
        resource_name=prefix + "-pod",
        resource_name_prefix=prefix,
        launch_session_id=nonce,
        bootstrap=_bootstrap_metadata(),
        deadline_epoch=time.time() + 3600,
        image_ref=TEST_IMAGE_REF,
        image_digest=TEST_IMAGE_DIGEST,
        source_commit=TEST_SOURCE_COMMIT,
        release_binding=release_binding,
        qualification_launch_rebind=launch_rebind,
    )
    manifest.update(
        {
            "status": "allocated_ready_continuing_spend",
            "instance_id": "12345",
            "continuing_spend": True,
            "watchdog": {
                "provider": "vast",
                "pod_name_prefix": prefix,
                "watchdog_out_dir": str(tmp_path),
                "deadline_epoch": time.time() + 3600,
            },
            "pending_teardown_record": str(tmp_path / "pending.json"),
            "pending_teardown_status": "open",
            "ssh_connection": {
                "instance_id": "12345",
                "ssh_host": "203.0.113.4",
                "ssh_port": 22022,
                "direct_port_ready": True,
            },
            "ssh_host_key": {
                "status": "enrolled",
                "known_hosts_file": str(tmp_path / "vast_ssh_known_hosts"),
                "fingerprint_artifact": str(tmp_path / "vast_ssh_host_key_fingerprint.json"),
                "tofu_pinned": True,
            },
        }
    )
    path = tmp_path / qualification.SESSION_MANIFEST_NAME
    qualification._private_write_json(path, manifest)
    return path


def test_explicit_bound_manifest_requires_complete_release_record(tmp_path: Path) -> None:
    manifest_path = _live_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("release_binding")
    qualification._private_write_json(manifest_path, manifest)

    with pytest.raises(
        ValueError,
        match="qualification_session_manifest_binding_invalid:release_binding_missing",
    ):
        qualification._load_private_manifest(manifest_path)


def _bind_latest_attempt(
    manifest_path: Path,
    *,
    sequence: int = 1,
    remote_process_state: str = "running",
) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    attempt_slug = f"attempt_{sequence:04d}"
    nonce = f"{manifest['launch_session_id']}:{attempt_slug}"
    latest = {
        "schema_version": "single_g1_kitchen_qualification_attempt_binding.v1",
        "attempt_sequence": sequence,
        "attempt_slug": attempt_slug,
        "attempt_nonce": nonce,
        "attempt_nonce_sha256": hashlib.sha256(nonce.encode()).hexdigest(),
        "launch_session_id": manifest["launch_session_id"],
        "episode_bootstrap_sha256": manifest["bootstrap"]["episode_bootstrap_sha256"],
        "bundle_sha256": manifest["bundle_sha256"],
        "overlay_revision": manifest["bootstrap"]["overlay_revision"],
        "qualification_launch_rebind": manifest["qualification_launch_rebind"],
        "dispatched_at": "2026-07-17T00:00:00Z",
        "remote_process_state": remote_process_state,
        "collection_status": "pending",
    }
    manifest["latest_attempt"] = latest
    qualification._private_write_json(manifest_path, manifest)
    return latest


def _qualification_output_zip(
    manifest_path: Path,
    *,
    phase: str,
    sequence: int = 1,
    successful_terminal: bool = False,
    attempt_input_overrides: dict[str, object] | None = None,
) -> bytes:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    attempt_slug = f"attempt_{sequence:04d}"
    nonce = f"{manifest['launch_session_id']}:{attempt_slug}"
    latest_attempt = dict(manifest.get("latest_attempt") or {})
    attempt_binding = (
        latest_attempt
        if latest_attempt.get("attempt_sequence") == sequence
        else {}
    )
    attempt = {
        "schema_version": "single_g1_kitchen_qualification_attempt.v1",
        "attempt_sequence": sequence,
        "attempt_nonce": nonce,
        "attempt_nonce_sha256": hashlib.sha256(nonce.encode()).hexdigest(),
        "launch_session_id": manifest["launch_session_id"],
        "episode_bootstrap_sha256": attempt_binding.get(
            "episode_bootstrap_sha256",
            manifest["bootstrap"]["episode_bootstrap_sha256"],
        ),
        "bundle_sha256": attempt_binding.get(
            "bundle_sha256", manifest["bundle_sha256"]
        ),
        "overlay_revision": attempt_binding.get(
            "overlay_revision", manifest["bootstrap"]["overlay_revision"]
        ),
        "stale_outputs_reused": False,
        "raw_secret_values_recorded": False,
    }
    bootstrap = {
        "schema_version": "groot_oscar_closed_loop_bootstrap.v1",
        "phase": phase,
        "launch_session_id": manifest["launch_session_id"],
        "raw_secret_values_recorded": False,
    }
    files: dict[str, bytes] = {
        "bootstrap.json": json.dumps(bootstrap).encode(),
        "closed_loop_out/qualification_attempt.json": json.dumps(attempt).encode(),
        "initial_policy_frame.png": b"initial-robot-pov",
        "closed_loop_out/isaac_task_state/frames/overview_0000.png": b"initial-overview",
        "closed_loop_out/isaac_task_state/frames/robot_pov_0000.png": b"initial-robot-pov",
        "closed_loop_out/qualification_episode.log": b"attempt-bound log\n",
        "closed_loop_out/qualification_startup_diagnostics.json": json.dumps(
            {
                "schema_version": (
                    "single_g1_kitchen_qualification_startup_diagnostics.v1"
                ),
                "startup_phase": phase,
                "startup_health": "ready_or_terminal",
                "attempt_sequence": sequence,
                "attempt_nonce_sha256": attempt["attempt_nonce_sha256"],
                "raw_secret_values_recorded": False,
            }
        ).encode(),
    }
    if successful_terminal:
        episode = "closed_loop_out/episode_001/"
        state = "closed_loop_out/isaac_task_state/"
        frames = state + "frames/"
        digest = "d" * 64
        derived_plan = {
            "schema_version": "test.qualification_derived_launch_plan.v1",
            "sealed_active": True,
            "image_ref": manifest["image_ref"],
        }
        derived_plan_sha256 = hashlib.sha256(
            json.dumps(
                derived_plan,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest()
        base_attempt_input = {
            "run_id": "single-g1-kitchen-direct",
            "attempt_id": "single-g1-kitchen-attempt-1",
            "launch_nonce": "prepared-source-launch-nonce",
            "source_commit": manifest["source_commit"],
            "source_dirty_patch_sha256": manifest["release_binding"][
                "source_patch_sha256"
            ],
            "image_digest": manifest["image_digest"],
            "qualification_release_rebind": embedded_launch_rebind(
                dict(manifest.get("latest_attempt") or {}).get(
                    "qualification_launch_rebind",
                    manifest["qualification_launch_rebind"],
                )
            ),
            "qualification_derived_launch_plan": derived_plan,
            "qualification_derived_launch_plan_sha256": derived_plan_sha256,
            "qualification_resolved_task_success_contract_sha256": digest,
            "qualification_source_task_success_contract_sha256": digest,
            "selected_task_id": "microwave_door",
            "artifacts": {
                "bundle": {"sha256": manifest["bundle_sha256"]},
                "kitchen_inventory": {"sha256": digest},
                "selection": {"sha256": digest},
                "task_success_contract": {"sha256": digest},
            },
        }
        base_attempt_input["qualification_runtime_attempt_overlay_base"] = (
            capture_runtime_attempt_overlay_base(base_attempt_input)
        )
        base_attempt_input_bytes = (
            json.dumps(base_attempt_input, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        attempt_input = json.loads(base_attempt_input_bytes)
        attempt_input.update(
            {
                "prepared_launch_nonce": base_attempt_input["launch_nonce"],
                "launch_nonce": nonce,
                "allocation_launch_session_id": manifest["launch_session_id"],
                "qualification_attempt_bound": True,
                "qualification_attempt_sequence": sequence,
                "qualification_attempt_nonce": nonce,
                "qualification_attempt_nonce_sha256": hashlib.sha256(
                    nonce.encode()
                ).hexdigest(),
            }
        )
        attempt_input["artifacts"]["task_success_contract"] = {
            "path": "/workspace/task_success_contract.json",
            "sha256": digest,
            "size_bytes": 459,
            "derived_from_sha256": digest,
            "resolution_artifact_path": (
                "/workspace/closed_loop_out/direct_task_contract_resolution.json"
            ),
        }
        attempt_input.update(attempt_input_overrides or {})
        attempt_input_bytes = (
            json.dumps(attempt_input, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        files["closed_loop_out/attempt_input_manifest.json"] = attempt_input_bytes
        latest = dict(manifest.get("latest_attempt") or {})
        latest_rebind = dict(latest.get("qualification_launch_rebind") or {})
        latest_rebind["derived_launch_plan_sha256"] = derived_plan_sha256
        latest_rebind["derived_attempt_manifest_sha256"] = hashlib.sha256(
            base_attempt_input_bytes
        ).hexdigest()
        latest["qualification_launch_rebind"] = latest_rebind
        manifest["latest_attempt"] = latest
        qualification._private_write_json(manifest_path, manifest)
        image_digest = str(manifest["image_digest"]).rsplit("@sha256:", 1)[-1]
        image_digest = image_digest.removeprefix("sha256:")
        identity = {
            "run_id": attempt_input["run_id"],
            "attempt_id": attempt_input["attempt_id"],
            "launch_nonce": nonce,
            "source_commit": attempt_input["source_commit"],
            "source_dirty_patch_sha256": attempt_input[
                "source_dirty_patch_sha256"
            ],
            "image_digest": image_digest,
            "bundle_digest": manifest["bundle_sha256"],
            "kitchen_asset_digest": digest,
            "active_selection_sha256": digest,
            "task_contract_sha256": digest,
            "provider_allocation_id": manifest["instance_id"],
        }
        role_keys = {
            role: Ed25519PrivateKey.generate()
            for role in (
                "startup",
                "task_transition",
                "geometry",
                "policy",
                "controller",
                "scorer",
            )
        }
        public_keys: dict[str, str] = {}
        pinned_roles: dict[str, list[str]] = {}
        for role, key in role_keys.items():
            public = key.public_key().public_bytes(
                serialization.Encoding.Raw,
                serialization.PublicFormat.Raw,
            )
            fingerprint = hashlib.sha256(public).hexdigest()
            public_keys[fingerprint] = base64.b64encode(public).decode()
            pinned_roles[role] = [fingerprint]
        files["runtime_ephemeral_trust.json"] = json.dumps(
            {
                "schema_version": "g1_kitchen_attestation_public_key_pins.v1",
                "algorithm": "ed25519",
                "identity_binding": identity,
                "public_keys": public_keys,
                "roles": pinned_roles,
            }
        ).encode()

        def signed_leaf(name: str, payload: dict, role: str) -> dict:
            observed = {**payload, "identity_binding": identity}
            data = (json.dumps(observed, indent=2, sort_keys=True) + "\n").encode()
            relative = f"{episode}proof_leaves/{name}"
            files[relative] = data
            key = role_keys[role]
            public = key.public_key().public_bytes(
                serialization.Encoding.Raw,
                serialization.PublicFormat.Raw,
            )
            return {
                "path": relative,
                "sha256": hashlib.sha256(data).hexdigest(),
                "size_bytes": len(data),
                "schema_version": observed["schema_version"],
                "attestation": {
                    "algorithm": "ed25519",
                    "role": role,
                    "public_key_fingerprint": hashlib.sha256(public).hexdigest(),
                    "signature_b64": base64.b64encode(key.sign(data)).decode(),
                },
            }

        simulator_session_id = "isaac-session-qualification-1"
        stage_id = "e" * 64
        action_sha256 = hashlib.sha256(b"action-1").hexdigest()
        baseline = build_task_episode_baseline(
            episode_initial_value=0.0,
            attempt_id=identity["attempt_id"],
            launch_nonce=nonce,
            simulator_session_id=simulator_session_id,
            stage_id=stage_id,
            articulation_prim_path="/root/Microwave017/Microwave017_Door",
            task_contract_sha256=digest,
            criterion_id="microwave_door_open_angle",
            unit="rad",
            captured_timestamp="900",
        )
        transition_measurement = {
            "schema_version": "task_transition_measurement.v1",
            "source_step_index": 0,
            "runtime_source_step_index": 1,
            "source_action_sha256": action_sha256,
            "simulator_session_id": simulator_session_id,
            "stage_id": stage_id,
            "before_timestamp": "1000",
            "after_timestamp": "1005",
            "articulation_prim_path": "/root/Microwave017/Microwave017_Door",
            "episode_baseline_digest": baseline["baseline_digest"],
            "episode_initial_value": 0.0,
            "episode_baseline": baseline,
            "episode_baseline_attestation": {"signature_verified": True},
        }
        measurement_leaf = signed_leaf(
            "task_transition_0000.json",
            transition_measurement,
            "task_transition",
        )
        judge = {
            "schema_version": "isaac_manipulation_success_evaluator_results.v1",
            "manipulation_success_proven": True,
            "did_target_manipulation_succeed": True,
        }
        judge_leaf = signed_leaf(
            "manipulation_success_judge.json", judge, "task_transition"
        )
        horizon_leaf = signed_leaf(
            "terminal_horizon.json",
            {
                "schema_version": "g1_kitchen_terminal_horizon.v1",
                "planned_max_steps": 48,
                "executed_step_count": 1,
                "terminal_step_index": 0,
                "termination_reason": (
                    "task_criterion_microwave_door_open_angle_passed_at_step_1"
                ),
                "task_completed": True,
                "scenario_count": 1,
                "source_action_sha256s": [action_sha256],
                "simulator_session_id": simulator_session_id,
                "stage_id": stage_id,
            },
            "task_transition",
        )
        policy_leaf = signed_leaf(
            "policy_action_sequence.json",
            {
                "schema_version": "g1_kitchen_policy_action_sequence.v1",
                "source_action_sha256s": [action_sha256],
                "actions": [{"action": "learned"}],
            },
            "policy",
        )
        controller_leaf = signed_leaf(
            "controller_fk_0000.json",
            {
                "schema_version": "gear_sonic_controller_fk_execution.v1",
                "status": "completed",
                "source_action_sha256": action_sha256,
                "official_controller_action_applied": True,
            },
            "controller",
        )
        consistency_leaf = signed_leaf(
            "strict_action_consistency.json",
            {
                "schema_version": "strict_action_aware_consistency_contract.v1",
                "forward_consistency_proven": False,
                "inverse_consistency_proven": False,
                "source_action_sha256s": [action_sha256],
                "per_step_results": [],
            },
            "scorer",
        )
        stance_leaf = signed_leaf(
            "live_stance_validation.json",
            {
                "schema_version": "g1_kitchen_live_stance_validation.v1",
                "stance_valid": True,
                "reach_valid": True,
                "facing_valid": True,
            },
            "geometry",
        )
        collision_leaf = signed_leaf(
            "live_collision_validation.json",
            {
                "schema_version": "g1_kitchen_live_collision_validation.v1",
                "collision_free": True,
                "clearance_valid": True,
            },
            "geometry",
        )
        startup_leafs = {
            "startup": signed_leaf(
                "startup.json",
                {
                    "schema_version": "groot_oscar_same_allocation_startup_gates.v1",
                    "status": "passed",
                },
                "startup",
            ),
            "fast_canary": signed_leaf(
                "fast_canary.json",
                {
                    "schema_version": "isaac_worker_runtime_preflight.v1",
                    "status": "passed",
                },
                "startup",
            ),
            "review_canary": signed_leaf(
                "review_canary.json",
                {
                    "schema_version": "isaac_review_renderer_canary.v1",
                    "status": "passed",
                },
                "startup",
            ),
            "asset_gate": signed_leaf(
                "asset_gate.json",
                {
                    "schema_version": "kitchen_asset_startup_gate.v1",
                    "status": "completed",
                },
                "startup",
            ),
        }

        def proof_row(*leafs: dict) -> dict:
            return {
                "status": "passed",
                "identity_binding": identity,
                "leaf_artifacts": list(leafs),
                "blockers": [],
            }

        proof_rows = {
            **{name: proof_row(leaf) for name, leaf in startup_leafs.items()},
            "scene_load": proof_row(measurement_leaf),
            "target": proof_row(measurement_leaf),
            "stance": proof_row(stance_leaf),
            "collision": proof_row(collision_leaf),
            "controller_fk": proof_row(policy_leaf, controller_leaf),
            "persistent_simulator_transition": proof_row(
                measurement_leaf, judge_leaf, horizon_leaf
            ),
            "forward_consistency": proof_row(consistency_leaf),
            "inverse_consistency": proof_row(consistency_leaf),
        }

        binding_fields = {
            "source_action_sha256": action_sha256,
            "simulator_session_id": simulator_session_id,
            "stage_id": stage_id,
            "before_timestamp": "1000",
            "after_timestamp": "1005",
            "attempt_id": identity["attempt_id"],
            "launch_nonce": nonce,
            "allocation_launch_session_id": manifest["launch_session_id"],
            "qualification_attempt_bound": True,
            "qualification_attempt_sequence": sequence,
            "qualification_attempt_nonce_sha256": hashlib.sha256(
                nonce.encode()
            ).hexdigest(),
        }
        initial_bindings: dict[str, dict] = {}
        for role in ("overview", "robot_pov"):
            name = f"{role}_0000.png"
            payload = files[frames + name]
            initial_bindings[name] = {
                "camera_role": role,
                "camera_motion_model": (
                    "rigid_head_local_transform"
                    if role == "robot_pov"
                    else "task_framed_third_person_review"
                ),
                "step_index": 0,
                "review_frame_index": 0,
                "control_frame_global_index": 0,
                "initial_frame": True,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "episode_baseline_digest": baseline["baseline_digest"],
                **binding_fields,
            }
        files[frames + "initial_frame_bindings.json"] = json.dumps(
            {
                "schema_version": "isaac_initial_review_frame_bindings.v1",
                "frames": initial_bindings,
            }
        ).encode()
        frame_bindings: dict[str, dict] = {}
        controller_measurements: list[dict] = []
        sampled_indices: list[int] = []
        simulation_time = 2.0
        physics_step_count = 100
        review_index = 0
        for horizon_index in range(40):
            control_index = horizon_index + 1
            before_steps = physics_step_count
            physics_step_count += 1
            before_time = simulation_time
            simulation_time += 0.02
            terminal = control_index == 40
            sampled = control_index % 5 == 0 or terminal
            action_frame_sha = hashlib.sha256(
                f"action-frame-{control_index}".encode()
            ).hexdigest()
            artifacts: list[dict] = []
            sampled_index = None
            if sampled:
                review_index += 1
                sampled_index = review_index
                sampled_indices.append(review_index)
                for role in ("overview", "robot_pov"):
                    name = f"{role}_{review_index:04d}.png"
                    frame_payload = f"png:{role}:{control_index}".encode()
                    files[frames + name] = frame_payload
                    frame_sha = hashlib.sha256(frame_payload).hexdigest()
                    artifacts.append(
                        {
                            "camera_role": role,
                            "frame_index": review_index,
                            "control_frame_global_index": control_index,
                            "sha256": frame_sha,
                        }
                    )
                    frame_bindings[name] = {
                        "camera_role": role,
                        "camera_motion_model": (
                            "rigid_head_local_transform"
                            if role == "robot_pov"
                            else "task_framed_third_person_review"
                        ),
                        "step_index": review_index,
                        "control_frame_global_index": control_index,
                        "physics_step_count_before": before_steps,
                        "physics_step_count_after": physics_step_count,
                        "physics_step_delta": 1,
                        "simulation_time_before_seconds": before_time,
                        "simulation_time_after_seconds": simulation_time,
                        "simulation_time_delta_seconds": 0.02,
                        "outer_source_step_index": 1,
                        "horizon_frame_index": horizon_index,
                        "controller_frame_index": control_index,
                        "source_action_frame_sha256": action_frame_sha,
                        "semantic_terminal_frame": terminal,
                        "sha256": frame_sha,
                        **binding_fields,
                    }
            controller_measurements.append(
                {
                    "control_frame_global_index": control_index,
                    "physics_step_count_before": before_steps,
                    "physics_step_count_after": physics_step_count,
                    "physics_step_delta": 1,
                    "simulation_time_before_seconds": before_time,
                    "simulation_time_after_seconds": simulation_time,
                    "simulation_time_delta_seconds": 0.02,
                    "horizon_frame_index": horizon_index,
                    "controller_frame_index": control_index,
                    "source_action_frame_sha256": action_frame_sha,
                    "registered_transition_passed": terminal,
                    "scheduled_review_frame": control_index % 5 == 0,
                    "sampled_for_review": sampled,
                    "review_frame_index": sampled_index,
                    "review_frame_artifacts": artifacts,
                    "semantic_terminal_frame": terminal,
                }
            )
        files[frames + "frame_step_bindings.json"] = json.dumps(
            {
                "schema_version": "isaac_review_frame_step_bindings.v1",
                "frames": frame_bindings,
            }
        ).encode()
        disk_measurement = {
            "schema_version": "task_transition_measurement.v1",
            "source_step_index": 1,
            "evidence_step_index": 1,
            "episode_baseline_digest": baseline["baseline_digest"],
            "controller_horizon_executed_frame_count": 40,
            "controller_review_frame_count": 8,
            "controller_review_frame_indices": sampled_indices,
            "controller_terminal_review_frame_index": 8,
            "controller_horizon_terminated_on_semantic_success": True,
            "controller_frame_measurements": controller_measurements,
            **binding_fields,
        }
        files[state + "task_measurement_0001.json"] = json.dumps(
            disk_measurement
        ).encode()

        videos = {
            "final_review.mp4": b"paired-final-video",
            "isaac_overview_review.mp4": b"overview-video",
            "isaac_robot_pov_review.mp4": b"robot-pov-video",
            "wam_prediction_review.mp4": b"wam-video",
        }
        for name, payload in videos.items():
            files[episode + name] = payload
        files[episode + "wam-step-1.mp4"] = b"wam-step-one"
        files[episode + "oscar_isaac_closed_loop_trace.jsonl"] = (
            json.dumps(
                {
                    "step_index": 1,
                    "wam_generated_video": (
                        "/workspace/closed_loop_out/episode_001/wam-step-1.mp4"
                    ),
                }
            )
            + "\n"
        ).encode()
        files[episode + "manipulation_success_evaluator_results.json"] = (
            json.dumps(judge).encode()
        )
        wam_review = {
            "schema_version": "groot_oscar_wam_prediction_review_validation.v1",
            "status": "passed",
            "blockers": [],
            "review_source": "oscar_wam_predicted_rollout_clips",
            "path": "/workspace/closed_loop_out/episode_001/wam_prediction_review.mp4",
            "sha256": hashlib.sha256(videos["wam_prediction_review.mp4"]).hexdigest(),
            "trace_step_count": 1,
            "ordered_clip_count": 1,
            "ordered_step_indices": [1],
            "episode_order_verified": True,
            "video_frame_count_mode": "dynamic_from_executed_controller_duration",
            "prediction_review_timeline_mode": "executed_control_prefix_per_decision",
            "executed_prefix_duration_seconds_by_step": [0.16],
            "expected_executed_timeline_duration_seconds": 0.16,
            "duration_seconds": 0.16,
            "full_prediction_horizons_preserved_in_source_clips": True,
            "overlapping_unexecuted_prediction_tails_excluded": True,
        }
        files[episode + "wam_prediction_review_validation.json"] = json.dumps(
            wam_review
        ).encode()
        review = {
            "schema_version": "groot_oscar_episode_review_validation.v1",
            "status": "passed",
            "blockers": [],
            "episode_order_verified": True,
            "review_source": "persistent_same_session_isaac_execution_frames",
            "execution_truth": True,
            "same_session_isaac_frames": True,
            "concat_mode": "primary_same_session_isaac_robot_pov_only",
            "primary_camera_role": "robot_pov",
            "overview_excluded_from_primary_review": True,
            "required_camera_roles": ["overview", "robot_pov"],
            "trace_step_count": 1,
            "ordered_clip_count": 1,
            "ordered_step_indices": [1],
            "width": 640,
            "height": 480,
            "frame_count": 9,
            "ordered_review_frame_count": 9,
            "duration_seconds": 0.9,
            "sha256": hashlib.sha256(videos["final_review.mp4"]).hexdigest(),
            "isaac_frame_evidence": {
                "status": "passed",
                "blockers": [],
                "bound_steps": [{"step_index": 1}],
                "simulator_session_id": simulator_session_id,
                "stage_id": stage_id,
                "attempt_id": identity["attempt_id"],
                "launch_nonce": nonce,
                "ordered_execution_frame_indices": list(range(1, 9)),
                "ordered_review_frame_count": 9,
                "ordered_review_frame_indices": list(range(9)),
                "ordered_review_control_frame_indices": [
                    0,
                    5,
                    10,
                    15,
                    20,
                    25,
                    30,
                    35,
                    40,
                ],
                "terminal_execution_frame_indices": [8],
            },
            "isaac_role_videos": {
                "overview": {
                    "status": "passed",
                    "blockers": [],
                    "path": "/workspace/isaac_overview_review.mp4",
                    "frame_count": 9,
                    "width": 640,
                    "height": 480,
                    "sha256": hashlib.sha256(
                        videos["isaac_overview_review.mp4"]
                    ).hexdigest(),
                },
                "robot_pov": {
                    "status": "passed",
                    "blockers": [],
                    "path": "/workspace/isaac_robot_pov_review.mp4",
                    "frame_count": 9,
                    "width": 640,
                    "height": 480,
                    "sha256": hashlib.sha256(
                        videos["isaac_robot_pov_review.mp4"]
                    ).hexdigest(),
                },
            },
            "wam_prediction_review": wam_review,
        }
        files[episode + "final_review_validation.json"] = json.dumps(review).encode()
        closed_loop = {
            "schema_version": "oscar_isaac_closed_loop_eval.v1",
            "status": "completed",
            "blockers": [],
            "steps_executed": 1,
            "manipulation_success_proven": True,
            "g1_kitchen_proof_rows": proof_rows,
            "success_proof": {
                "manipulation_success_proven": True,
                "did_target_manipulation_succeed": True,
            },
            "proof": {
                "registered_task_completion_transition": {
                    "registered_transition_passed": True,
                    "computed_transition_passed": True,
                    "validation_blockers": [],
                }
            },
            "episode_termination": {
                "reason": (
                    "task_criterion_microwave_door_open_angle_passed_at_step_1"
                ),
                "steps_executed": 1,
                "task_completion_evidence_status": "passed",
                "task_completion_results": [disk_measurement],
            },
        }
        files[episode + "oscar_isaac_closed_loop_manifest.json"] = json.dumps(
            closed_loop
        ).encode()
        runner = {
            "schema_version": "groot_oscar_closed_loop_worker_result.v1",
            "status": "completed",
            "blockers": [],
            "closed_loop_return_code": 0,
            "closed_loop_manifest": closed_loop,
            "raw_secret_values_recorded": False,
        }
        files["isaac_runtime_result.json"] = json.dumps(runner).encode()
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in sorted(files.items()):
            archive.writestr(name, payload)
    return output.getvalue()


def _write_output_get_url(tmp_path: Path) -> None:
    path = tmp_path / "provider_output_get_url.txt"
    path.write_text("https://objects.example/output?signature=secret", encoding="utf-8")
    path.chmod(0o600)


def test_gate_matrix_keeps_historical_attempt_boundaries_explicit() -> None:
    by_id = {row["gate_id"]: row for row in qualification.qualification_gate_matrix()}

    assert by_id["image_bundle_assets"]["attempt_016_status"] == "attempt016_proven"
    assert by_id["groot_checkpoint_server"]["attempt_016_status"] == "attempt016_proven"
    assert by_id["isaac_scene_baseline"]["attempt_016_status"] == "attempt016_proven"
    assert by_id["native_dds_freshness"]["attempt_016_status"] == "attempt016_proven"
    assert by_id["controller_init_done"]["attempt_016_status"] == "pending"
    assert by_id["first_groot_query"]["attempt_016_status"] == "pending"
    assert by_id["first_learned_oscar_transition"]["attempt_016_status"] == "pending"
    assert by_id["validated_final_review_upload"]["attempt_016_status"] == "pending"
    assert by_id["image_bundle_assets"]["attempt_017_status"] == "attempt017_proven"
    assert by_id["groot_checkpoint_server"]["attempt_017_status"] == "attempt017_proven"
    assert by_id["isaac_scene_baseline"]["attempt_017_status"] == "attempt017_proven"
    assert by_id["controller_init_done"]["attempt_017_status"] == "attempt017_proven"
    assert by_id["native_dds_freshness"]["attempt_017_status"] == "attempt017_proven"
    assert by_id["first_groot_query"]["attempt_017_status"] == "attempt017_proven"
    assert by_id["protocol_v4_token_receipt"]["attempt_017_status"] == "attempt017_proven"
    assert (
        by_id["first_official_action"]["attempt_017_status"]
        == "partial_protocol_v4_token_receipt_only"
    )
    assert by_id["isaac_apply_readback"]["attempt_017_status"] == "pending"
    assert by_id["first_learned_oscar_transition"]["attempt_017_status"] == "pending"
    assert by_id["semantic_microwave_transition"]["attempt_017_status"] == "pending"
    assert by_id["ordered_review_render"]["attempt_017_status"] == "pending"
    assert by_id["validated_final_review_upload"]["attempt_017_status"] == "pending"
    assert all(row["current_session_status"] == "pending" for row in by_id.values())

    boundary = qualification._session_claim_boundary()
    prior = boundary["prior_persistent_result"]
    assert prior == {
        "policy_calls": 2,
        "learned_wam_transitions": 1,
        "isaac_kitchen_semantic_success_proven": False,
        "full_episode_video_proven": False,
        "must_not_be_promoted_to_current_goal_completion": True,
    }
    assert boundary["attempt_016_result"]["controller_init_done"] is False
    attempt_017 = boundary["attempt_017_result"]
    assert attempt_017["fresh_action_horizon"] == {
        "frame_count": 40,
        "frame_dimension": 78,
        "selection_mode": "fresh_receding_horizon_first_frame",
        "real_initial_observation": True,
    }
    assert attempt_017["protocol_v4_token_receipt_step"] == 1
    assert attempt_017["matching_g1_debug_fk_output_proven"] is False
    assert attempt_017["isaac_action_apply_readback_proven"] is False
    assert attempt_017["learned_oscar_transition_proven"] is False
    assert attempt_017["semantic_success_proven"] is False
    assert attempt_017["full_episode_video_proven"] is False


def test_qualification_bootstrap_stages_exact_digest_bound_fixed_control() -> None:
    payload, metadata = qualification._qualification_bootstrap_payload(
        _minimal_inputs(),
        "qualification-nonce",
        image_digest=TEST_IMAGE_DIGEST,
        source_commit=TEST_SOURCE_COMMIT,
    )
    syntax = subprocess.run(["bash", "-n"], input=payload, capture_output=True, check=False)

    assert syntax.returncode == 0, syntax.stderr.decode(errors="replace")
    assert metadata["arbitrary_remote_command_allowed"] is False
    assert metadata["fixed_actions"] == [
        "run",
        "status",
        "tail",
        "gpu-status",
        "restart",
        "refresh",
    ]
    assert metadata["fixed_components"] == [
        "bootstrap",
        "episode",
        "gear_sonic_controller",
        "gear_sonic_isaac_dds_bridge",
        "groot_microwave_finetune",
        "groot_server",
        "isaac_task_executor",
    ]
    assert qualification.REMOTE_CONTROL_SCRIPT.encode() in payload
    assert qualification.REMOTE_REFRESH_INSTALLER.encode() in payload
    assert b"/tmp/blueprint-provider-bootstrap.sh" in payload
    assert b"qualification_provider_bootstrap_missing_or_unsafe" in payload
    assert b"run episode" not in payload
    assert metadata["episode_auto_run"] is False

    component_sha = metadata["component_script_sha256s"]
    assert component_sha["episode"] == metadata["episode_bootstrap_sha256"]
    assert component_sha["bootstrap"] == metadata["episode_bootstrap_sha256"]
    assert len(metadata["control_script_sha256"]) == 64
    assert len(metadata["refresh_installer_sha256"]) == 64
    assert metadata["overlay_revision"] == 1

    control = qualification._qualification_control_script(
        launch_session_id="qualification-nonce",
        bundle_sha256=qualification.BUNDLE_SHA256,
        image_digest=TEST_IMAGE_DIGEST,
        source_commit=TEST_SOURCE_COMMIT,
    )
    assert "eval " not in control
    assert 'case "$ACTION" in status|tail|gpu-status|run|restart|stop|refresh)' in control
    stop_case = control[control.index("  stop)\n") : control.index("esac\n", control.index("  stop)\n"))]
    for component in (
        "episode",
        "groot_server",
        "gear_sonic_controller",
        "isaac_task_executor",
        "gear_sonic_isaac_dds_bridge",
        "groot_microwave_finetune",
    ):
        assert f"stop_component {component}" in stop_case
    assert "qualification_gpu_snapshot.v1" in control
    assert "single_g1_kitchen_qualification_startup_diagnostics.v1" in control
    assert "stall_seconds = int(sys.argv[6])" in control
    assert str(qualification.QUALIFICATION_STARTUP_STALL_SECONDS) in control
    assert "startup_health=\" + startup_health" in control
    assert "phase_age_seconds=\" + str(phase_age)" in control
    assert "log_bytes=\" + str(log_bytes)" in control
    assert "root_pid_elapsed_seconds=\" + str(root_pid_elapsed)" in control
    assert "--query-compute-apps=pid,process_name,used_memory" in control
    assert "qualification_action_forbidden" in control
    assert "qualification_component_forbidden" in control
    assert "qualification_overlay_file_sha256_mismatch" in control
    assert "qualification_refresh_requires_all_components_stopped" in control
    assert f"EXPECTED_SOURCE_COMMIT={TEST_SOURCE_COMMIT}" in control
    assert '"source_commit": sys.argv[11]' in control
    assert 'if [ "$process_state" = Z ]; then return 1; fi' in control
    assert "action=stop component=%s" in control
    assert qualification.REMOTE_REFRESH_INSTALLER in control
    assert "single_g1_kitchen_qualification_attempt.v1" in control
    assert '"stale_outputs_reused": False' in control
    assert "/workspace/qualification_attempts" in control
    assert "/workspace/closed_loop_out/qualification_episode.log" in control
    assert "previous_slug=$(printf 'attempt_%04d' \"$previous\")" in control
    assert 'archive="$ATTEMPT_ARCHIVE/$previous_slug"' in control
    assert "attempt_sequence=%s attempt_nonce_sha256=%s" in control
    assert "BLUEPRINT_QUALIFICATION_ATTEMPT_NONCE_SHA256" in control
    assert control.index("prepare_episode_attempt") < control.index('nohup "$selected"')
    attempt_line = next(
        line for line in control.splitlines() if "then prepare_episode_attempt" in line
    )
    assert '"$1" = episode' in attempt_line and '"$1" = bootstrap' in attempt_line
    assert "groot_microwave_finetune" not in attempt_line
    assert "/bin/bash -c" not in control
    control_syntax = subprocess.run(
        ["bash", "-n"], input=control.encode(), capture_output=True, check=False
    )
    assert control_syntax.returncode == 0, control_syntax.stderr.decode(errors="replace")
    python_heredocs = re.findall(r"<<'PY'\n(.*?)\nPY", control, flags=re.DOTALL)
    assert python_heredocs
    for index, source in enumerate(python_heredocs):
        compile(source, f"<qualification-control-heredoc-{index}>", "exec")


def test_fixed_control_startup_diagnostics_keeps_preexecution_phases_stallable(
    tmp_path: Path,
) -> None:
    control = qualification._qualification_control_script(
        launch_session_id="qualification-nonce",
        bundle_sha256=qualification.BUNDLE_SHA256,
        image_digest=TEST_IMAGE_DIGEST,
        source_commit=TEST_SOURCE_COMMIT,
    )
    diagnostic_source = next(
        source
        for source in re.findall(r"<<'PY'\n(.*?)\nPY", control, flags=re.DOTALL)
        if "stall_seconds = int(sys.argv[6])" in source
    )
    diagnostic_path = tmp_path / "qualification_startup_diagnostics.json"
    bootstrap_path = tmp_path / "bootstrap.json"
    log_path = tmp_path / "qualification_episode.log"
    pid_path = tmp_path / "episode.pid"
    log_path.write_bytes(b"")
    pid_path.write_text("99999999\n", encoding="ascii")
    expected_nonce_sha256 = "a" * 64
    diagnostic_path.write_text(
        json.dumps(
            {
                "attempt_sequence": 1,
                "attempt_nonce_sha256": expected_nonce_sha256,
                "dispatched_at_epoch": int(time.time()),
                "unexpected_remote_field": "must-not-survive-allowlist",
            }
        ),
        encoding="utf-8",
    )
    old_timestamp = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 901)
    )
    old_epoch = time.time() - 901
    bootstrap_path.write_text(
        json.dumps(
            {
                "phase": "container_bash_started",
                "updated_at": old_timestamp,
            }
        ),
        encoding="utf-8",
    )
    os.utime(bootstrap_path, (old_epoch, old_epoch))

    stalled = subprocess.run(
        [
            "python",
            "-c",
            diagnostic_source,
            str(diagnostic_path),
            str(bootstrap_path),
            str(log_path),
            str(pid_path),
            "running",
            str(qualification.QUALIFICATION_STARTUP_STALL_SECONDS),
            "1",
            expected_nonce_sha256,
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert stalled.returncode == 0, stalled.stderr
    assert "startup_phase=container_bash_started" in stalled.stdout
    assert "startup_health=stalled" in stalled.stdout
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    assert diagnostic["startup_health"] == "stalled"
    assert diagnostic["startup_phase_age_seconds"] >= 900
    assert diagnostic["log_size_bytes"] == 0
    assert diagnostic["root_pid_state"] == "missing"
    assert diagnostic["diagnostic_attempt_binding"] == "valid"
    assert diagnostic["raw_secret_values_recorded"] is False
    assert "unexpected_remote_field" not in diagnostic

    bootstrap_path.write_text(
        json.dumps(
            {
                "phase": "isaac_task_executor_ready",
                "updated_at": old_timestamp,
            }
        ),
        encoding="utf-8",
    )
    os.utime(bootstrap_path, (old_epoch, old_epoch))
    stalled_preexecution = subprocess.run(
        [
            "python",
            "-c",
            diagnostic_source,
            str(diagnostic_path),
            str(bootstrap_path),
            str(log_path),
            str(pid_path),
            "running",
            str(qualification.QUALIFICATION_STARTUP_STALL_SECONDS),
            "1",
            expected_nonce_sha256,
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert stalled_preexecution.returncode == 0, stalled_preexecution.stderr
    assert "startup_health=stalled" in stalled_preexecution.stdout

    bootstrap_path.write_text(
        json.dumps({"phase": "runner_done", "updated_at": old_timestamp}),
        encoding="utf-8",
    )
    os.utime(bootstrap_path, (old_epoch, old_epoch))
    terminal = subprocess.run(
        [
            "python",
            "-c",
            diagnostic_source,
            str(diagnostic_path),
            str(bootstrap_path),
            str(log_path),
            str(pid_path),
            "running",
            str(qualification.QUALIFICATION_STARTUP_STALL_SECONDS),
            "1",
            expected_nonce_sha256,
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert terminal.returncode == 0, terminal.stderr
    assert "startup_health=ready_or_terminal" in terminal.stdout
    assert "diagnostic_binding=valid" in terminal.stdout

    mismatched = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    mismatched["attempt_nonce_sha256"] = "b" * 64
    diagnostic_path.write_text(json.dumps(mismatched), encoding="utf-8")
    binding_blocked = subprocess.run(
        [
            "python",
            "-c",
            diagnostic_source,
            str(diagnostic_path),
            str(bootstrap_path),
            str(log_path),
            str(pid_path),
            "running",
            str(qualification.QUALIFICATION_STARTUP_STALL_SECONDS),
            "1",
            expected_nonce_sha256,
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert binding_blocked.returncode == 0, binding_blocked.stderr
    assert "startup_health=stalled" in binding_blocked.stdout
    assert "diagnostic_binding=missing_or_mismatch" in binding_blocked.stdout


def test_fixed_control_startup_diagnostics_handles_proc_exit_race(
    tmp_path: Path,
) -> None:
    control = qualification._qualification_control_script(
        launch_session_id="qualification-nonce",
        bundle_sha256=qualification.BUNDLE_SHA256,
        image_digest=TEST_IMAGE_DIGEST,
        source_commit=TEST_SOURCE_COMMIT,
    )
    diagnostic_source = next(
        source
        for source in re.findall(r"<<'PY'\n(.*?)\nPY", control, flags=re.DOTALL)
        if "stall_seconds = int(sys.argv[6])" in source
    )
    parent_pid = os.getpid()
    proc_stat_path = f"/proc/{parent_pid}/stat"
    race_source = diagnostic_source.replace(
        "now = int(time.time())",
        """now = int(time.time())
original_read_text = pathlib.Path.read_text
def race_read_text(path, *args, **kwargs):
    if str(path) == __PROC_STAT_PATH__:
        raise FileNotFoundError(__PROC_STAT_PATH__)
    return original_read_text(path, *args, **kwargs)
pathlib.Path.read_text = race_read_text
""".replace("__PROC_STAT_PATH__", repr(proc_stat_path)),
        1,
    )
    diagnostic_path = tmp_path / "qualification_startup_diagnostics.json"
    bootstrap_path = tmp_path / "bootstrap.json"
    log_path = tmp_path / "qualification_episode.log"
    pid_path = tmp_path / "episode.pid"
    diagnostic_path.write_text(
        json.dumps(
            {
                "attempt_sequence": 1,
                "attempt_nonce_sha256": "a" * 64,
                "dispatched_at_epoch": int(time.time()),
            }
        ),
        encoding="utf-8",
    )
    bootstrap_path.write_text(json.dumps({"phase": "container_bash_started"}))
    log_path.write_bytes(b"")
    pid_path.write_text(f"{parent_pid}\n", encoding="ascii")

    raced = subprocess.run(
        [
            "python",
            "-c",
            race_source,
            str(diagnostic_path),
            str(bootstrap_path),
            str(log_path),
            str(pid_path),
            "stopped",
            str(qualification.QUALIFICATION_STARTUP_STALL_SECONDS),
            "1",
            "a" * 64,
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert raced.returncode == 0, raced.stderr
    assert f"root_pid={parent_pid}" in raced.stdout
    assert "root_pid_state=missing" in raced.stdout
    assert "root_pid_elapsed_seconds=-1" in raced.stdout
    assert "startup_health=stopped" in raced.stdout


def test_dispatch_diagnostics_failure_does_not_wedge_dispatched_attempt(
    tmp_path: Path,
) -> None:
    missing_parent = tmp_path / "removed-output" / "qualification-startup.json"
    source = qualification.dispatch_diagnostics_shell().replace(
        "/workspace/closed_loop_out/qualification_startup_diagnostics.json",
        str(missing_parent),
    )
    completed = subprocess.run(
        [
            "bash",
            "-c",
            "set -e\n"
            "pid=507\n"
            "ATTEMPT_SEQUENCE=1\n"
            f"ATTEMPT_NONCE_SHA256={'a' * 64}\n"
            "EXPECTED_LAUNCH_SESSION_ID=qualification-nonce\n"
            + source
            + "\nprintf 'dispatch_identity_can_still_be_emitted\\n'\n",
            "qualification-dispatch-test",
            "episode",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "dispatch_identity_can_still_be_emitted" in completed.stdout
    assert "qualification_startup_diagnostics_write_failed" in completed.stderr
    assert not missing_parent.exists()


def test_startup_diagnostic_parser_blocks_stall_and_attempt_binding_loss() -> None:
    diagnostics, blockers = qualification.parse_startup_diagnostics(
        "startup_phase=runner_done startup_health=stalled "
        "phase_age_seconds=1 log_bytes=0 log_age_seconds=1 "
        "root_pid=507 root_pid_state=S root_pid_elapsed_seconds=2 "
        "diagnostic_binding=missing_or_mismatch",
        observed_at="2026-07-22T00:00:00Z",
    )

    assert diagnostics is not None
    assert diagnostics["diagnostic_attempt_binding"] == "missing_or_mismatch"
    assert blockers == [
        "qualification_episode_startup_progress_stalled",
        "qualification_episode_startup_diagnostics_binding_invalid",
    ]

    pre_dispatch, pre_dispatch_blockers = qualification.parse_startup_diagnostics(
        "startup_phase=progress_marker_missing startup_health=stopped "
        "phase_age_seconds=0 log_bytes=0 log_age_seconds=-1 "
        "root_pid=0 root_pid_state=missing root_pid_elapsed_seconds=-1 "
        "diagnostic_binding=not_applicable",
        observed_at="2026-07-22T00:00:00Z",
    )
    assert pre_dispatch is not None
    assert pre_dispatch_blockers == []


def test_private_manifest_binds_exact_resource_and_refuses_tampering(tmp_path: Path) -> None:
    path = _live_manifest(tmp_path)
    resolved, manifest = qualification._load_private_manifest(path)

    assert resolved == path
    assert path.stat().st_mode & 0o777 == 0o600
    assert manifest["image_ref"] == TEST_IMAGE_REF
    assert manifest["bundle_sha256"] == qualification.BUNDLE_SHA256
    assert manifest["instance_id"] == "12345"

    manifest["bundle_sha256"] = "0" * 64
    qualification._private_write_json(path, manifest)
    with pytest.raises(ValueError, match="manifest_binding_invalid:bundle"):
        qualification._load_private_manifest(path)


def test_control_revalidates_provider_and_dispatches_only_fixed_component(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = _live_manifest(tmp_path)

    class Provider:
        def inspect(self, instance_id: str) -> dict:
            assert instance_id == "12345"
            return {
                "status": "observed",
                "instance_id": "12345",
                "name": qualification.NAME_PREFIX_ROOT + "0123456789-pod",
                "ssh_host": "203.0.113.4",
                "ssh_port": 22022,
                "image_runtype": "ssh_direct",
                "direct_port_ready": True,
            }

    observed: dict = {}

    def fake_control(connection, **kwargs):
        observed["connection"] = connection
        observed.update(kwargs)
        return {
            "status": "completed",
            "action": kwargs["action"],
            "component": kwargs["component"],
            "stdout": "fixed control completed",
            "blockers": [],
        }

    monkeypatch.setattr(qualification, "get_render_provider", lambda _name: Provider())
    monkeypatch.setattr(gpu_render_providers, "run_vast_ssh_control", fake_control)

    with pytest.raises(ValueError, match="qualification_control_admission_out_missing"):
        qualification.run_qualification_session(
            action="restart-component",
            component="controller",
            session_manifest=manifest_path,
            adapter_output=tmp_path / "blocked-result.json",
            execute=True,
        )
    assert not observed

    admission_out = tmp_path / "control-admission.json"
    result = qualification.run_qualification_session(
        action="restart-component",
        component="controller",
        session_manifest=manifest_path,
        admission_out=admission_out,
        adapter_output=tmp_path / "result.json",
        execute=True,
    )

    assert result["status"] == "component_restarted_continuing_spend"
    assert result["continuing_spend"] is True
    assert observed["action"] == "restart"
    assert observed["component"] == "gear_sonic_controller"
    assert observed["known_hosts_file"] == str(tmp_path / "vast_ssh_known_hosts")
    assert "command" not in observed
    admission = json.loads(admission_out.read_text(encoding="utf-8"))
    assert admission["status"] == "admitted"
    assert admission["fresh_control_admission"] is True
    assert admission["action"] == "restart"
    assert admission["component"] == "gear_sonic_controller"
    persisted = json.loads(manifest_path.read_text())
    assert persisted["continuing_spend"] is True
    assert persisted["last_control"]["component"] == "gear_sonic_controller"
    assert manifest_path.stat().st_mode & 0o777 == 0o600


def test_status_recovers_attach_after_direct_port_precedes_usable_ssh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = _live_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "status": "control_blocked_continuing_spend",
            "ssh_connection": None,
            "ssh_host_key": {
                "status": "blocked",
                "blockers": ["vast_ssh_host_key_scan_failed"],
            },
        }
    )
    qualification._private_write_json(manifest_path, manifest)

    connection = {
        "instance_id": "12345",
        "ssh_host": "203.0.113.4",
        "ssh_port": 22022,
        "image_runtype": "ssh_direct",
        "direct_port_ready": True,
    }

    class Provider:
        def inspect(self, _instance_id: str) -> dict:
            return {
                "status": "observed",
                "instance_id": "12345",
                "name": qualification.NAME_PREFIX_ROOT + "0123456789-pod",
                **connection,
            }

    recovered_host_key = {
        "status": "enrolled",
        "known_hosts_file": str(tmp_path / "vast_ssh_known_hosts"),
        "fingerprint_artifact": str(tmp_path / "vast_ssh_host_key_fingerprint.json"),
        "tofu_pinned": True,
    }
    wait_calls: list[dict] = []

    def fake_wait(_provider, **kwargs):
        wait_calls.append(kwargs)
        return (
            connection,
            [{"host_key_enrollment_status": "enrolled"}],
            recovered_host_key,
            {
                "status": "completed",
                "returncode": 0,
                "action": "status",
                "component": "episode",
                "blockers": [],
                "strict_host_key_checking": True,
            },
        )

    monkeypatch.setattr(qualification, "get_render_provider", lambda _name: Provider())
    monkeypatch.setattr(qualification, "_wait_for_qualification_attach", fake_wait)
    monkeypatch.setattr(
        gpu_render_providers,
        "run_vast_ssh_control",
        lambda _connection, **_kwargs: {
            "status": "completed",
            "returncode": 0,
            "stdout": (
                "action=status component=episode state=stopped pids= "
                f"bootstrap_sha256={manifest['bootstrap']['episode_bootstrap_sha256']} "
                "overlay_revision=1 attempt_sequence= attempt_nonce_sha256="
            ),
            "blockers": [],
        },
    )

    result = qualification.run_qualification_session(
        action="status",
        session_manifest=manifest_path,
        adapter_output=tmp_path / "status.json",
        execute=True,
    )

    assert result["status"] == "status_observed_continuing_spend"
    assert len(wait_calls) == 1
    assert wait_calls[0]["timeout_seconds"] == 180
    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert persisted["ssh_connection"]["ssh_port"] == 22022
    assert persisted["ssh_host_key"]["status"] == "enrolled"
    assert any(
        row.get("action") == "recover-attach-via-status"
        and row.get("status") == "allocated_ready_continuing_spend"
        for row in persisted["history"]
    )


def test_status_recovers_endpoint_change_after_loading_proxy_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = _live_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "status": "allocated_attach_blocked_continuing_spend",
            "ssh_connection": {
                "instance_id": "12345",
                "ssh_host": "ssh6.vast.ai",
                "ssh_port": 14060,
                "image_runtype": "ssh_direct",
                "direct_port_ready": True,
            },
            "ssh_readiness_observations": [{"actual_status": "loading"}],
        }
    )
    qualification._private_write_json(manifest_path, manifest)

    current_connection = {
        "instance_id": "12345",
        "ssh_host": "203.0.113.4",
        "ssh_port": 22022,
        "image_runtype": "ssh_direct",
        "direct_port_ready": True,
    }

    class Provider:
        def inspect(self, _instance_id: str) -> dict:
            return {
                "status": "observed",
                "actual_status": "running",
                "instance_id": "12345",
                "name": qualification.NAME_PREFIX_ROOT + "0123456789-pod",
                **current_connection,
            }

    recovered_host_key = {
        "status": "enrolled",
        "known_hosts_file": str(tmp_path / "vast_ssh_known_hosts"),
        "fingerprint_artifact": str(tmp_path / "vast_ssh_host_key_fingerprint.json"),
        "tofu_pinned": True,
    }
    wait_calls: list[dict] = []

    def fake_wait(_provider, **kwargs):
        wait_calls.append(kwargs)
        return (
            current_connection,
            [{"actual_status": "running", "authenticated_control_status": "completed"}],
            recovered_host_key,
            {
                "status": "completed",
                "returncode": 0,
                "action": "status",
                "component": "episode",
                "blockers": [],
                "strict_host_key_checking": True,
            },
        )

    monkeypatch.setattr(qualification, "get_render_provider", lambda _name: Provider())
    monkeypatch.setattr(qualification, "_wait_for_qualification_attach", fake_wait)
    monkeypatch.setattr(
        gpu_render_providers,
        "run_vast_ssh_control",
        lambda _connection, **_kwargs: {
            "status": "completed",
            "returncode": 0,
            "stdout": (
                "action=status component=episode state=stopped pids= "
                f"bootstrap_sha256={manifest['bootstrap']['episode_bootstrap_sha256']} "
                "overlay_revision=1 attempt_sequence= attempt_nonce_sha256="
            ),
            "blockers": [],
        },
    )

    result = qualification.run_qualification_session(
        action="status",
        session_manifest=manifest_path,
        adapter_output=tmp_path / "status.json",
        execute=True,
    )

    assert result["status"] == "status_observed_continuing_spend"
    assert len(wait_calls) == 1
    assert wait_calls[0]["attempt_dir"] == tmp_path / "ssh_attach_running_endpoint"
    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert persisted["ssh_connection"]["ssh_host"] == "203.0.113.4"
    assert any(
        row.get("action") == "recover-attach-via-status"
        and row.get("status") == "allocated_ready_continuing_spend"
        for row in persisted["history"]
    )


def test_status_recovers_running_endpoint_with_stale_root_host_key_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = _live_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["ssh_host_key"]["blockers"] = ["vast_ssh_existing_host_key_pin_invalid"]
    qualification._private_write_json(manifest_path, manifest)
    pin_path = tmp_path / gpu_render_providers.VAST_SSH_HOST_KEY_FINGERPRINT_NAME
    qualification._private_write_json(
        pin_path,
        {"ssh_host": "ssh6.vast.ai", "ssh_port": 14060},
    )
    connection = {
        "instance_id": "12345",
        "ssh_host": "203.0.113.4",
        "ssh_port": 22022,
        "image_runtype": "ssh_direct",
        "direct_port_ready": True,
    }

    class Provider:
        def inspect(self, _instance_id: str) -> dict:
            return {
                "status": "observed",
                "actual_status": "running",
                "instance_id": "12345",
                "name": qualification.NAME_PREFIX_ROOT + "0123456789-pod",
                **connection,
            }

    recovered_host_key = {
        "status": "enrolled",
        "known_hosts_file": str(tmp_path / "vast_ssh_known_hosts"),
        "fingerprint_artifact": str(pin_path),
        "tofu_pinned": True,
    }
    wait_calls: list[dict] = []

    def fake_wait(_provider, **kwargs):
        wait_calls.append(kwargs)
        return (
            connection,
            [{"actual_status": "running"}],
            recovered_host_key,
            {"status": "completed", "returncode": 0, "blockers": []},
        )

    monkeypatch.setattr(qualification, "get_render_provider", lambda _name: Provider())
    monkeypatch.setattr(qualification, "_wait_for_qualification_attach", fake_wait)
    monkeypatch.setattr(
        gpu_render_providers,
        "run_vast_ssh_control",
        lambda _connection, **_kwargs: {
            "status": "completed",
            "stdout": "action=status component=episode state=stopped pids=",
            "blockers": [],
        },
    )

    result = qualification.run_qualification_session(
        action="status",
        session_manifest=manifest_path,
        adapter_output=tmp_path / "status.json",
        execute=True,
    )

    assert result["status"] == "status_observed_continuing_spend"
    assert wait_calls[0]["attempt_dir"] == tmp_path / "ssh_attach_running_endpoint"


def test_episode_run_records_exact_attempt_and_requires_collection_before_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = _live_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_nonce = f"{manifest['launch_session_id']}:attempt_0001"
    expected_nonce_sha = hashlib.sha256(expected_nonce.encode()).hexdigest()

    class Provider:
        def inspect(self, _instance_id: str) -> dict:
            return {
                "status": "observed",
                "instance_id": "12345",
                "name": qualification.NAME_PREFIX_ROOT + "0123456789-pod",
                "ssh_host": "203.0.113.4",
                "ssh_port": 22022,
                "image_runtype": "ssh_direct",
                "direct_port_ready": True,
            }

    calls = 0

    def fake_control(_connection, **kwargs):
        nonlocal calls
        calls += 1
        return {
            "status": "completed",
            "stdout": (
                "action=run component=episode pid=12 "
                f"bootstrap_sha256={manifest['bootstrap']['episode_bootstrap_sha256']} "
                "overlay_revision=1 attempt_sequence=1 "
                f"attempt_nonce_sha256={expected_nonce_sha}"
            ),
            "blockers": [],
        }

    monkeypatch.setattr(qualification, "get_render_provider", lambda _name: Provider())
    monkeypatch.setattr(gpu_render_providers, "run_vast_ssh_control", fake_control)

    first = qualification.run_qualification_session(
        action="run",
        session_manifest=manifest_path,
        admission_out=tmp_path / "run-admission.json",
        adapter_output=tmp_path / "run.json",
        execute=True,
    )

    assert first["status"] == "episode_dispatched_continuing_spend"
    latest = json.loads(manifest_path.read_text())["latest_attempt"]
    assert latest["attempt_sequence"] == 1
    assert latest["attempt_nonce"] == expected_nonce
    assert latest["attempt_nonce_sha256"] == expected_nonce_sha
    assert latest["episode_bootstrap_sha256"] == manifest["bootstrap"][
        "episode_bootstrap_sha256"
    ]
    assert latest["overlay_revision"] == 1
    assert latest["collection_status"] == "pending"
    with pytest.raises(ValueError, match="collect_required_before_episode_rerun"):
        qualification.run_qualification_session(
            action="run",
            session_manifest=manifest_path,
            adapter_output=tmp_path / "rerun.json",
            execute=True,
        )
    assert calls == 1


def test_episode_status_binds_stopped_state_to_exact_current_attempt_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = _live_manifest(tmp_path)
    latest = _bind_latest_attempt(manifest_path)

    class Provider:
        def inspect(self, _instance_id: str) -> dict:
            return {
                "status": "observed",
                "instance_id": "12345",
                "name": qualification.NAME_PREFIX_ROOT + "0123456789-pod",
                "ssh_host": "203.0.113.4",
                "ssh_port": 22022,
                "image_runtype": "ssh_direct",
                "direct_port_ready": True,
            }

    monkeypatch.setattr(qualification, "get_render_provider", lambda _name: Provider())
    monkeypatch.setattr(
        gpu_render_providers,
        "run_vast_ssh_control",
        lambda _connection, **_kwargs: {
            "status": "completed",
            "stdout": (
                "action=status component=episode state=stopped pids= "
                f"bootstrap_sha256={latest['episode_bootstrap_sha256']} "
                f"overlay_revision={latest['overlay_revision']} "
                f"attempt_sequence={latest['attempt_sequence']} "
                f"attempt_nonce_sha256={latest['attempt_nonce_sha256']} "
                "startup_phase=container_bash_started startup_health=stopped "
                "phase_age_seconds=901 log_bytes=0 log_age_seconds=901 "
                "root_pid=507 root_pid_state=S root_pid_elapsed_seconds=902 "
                "diagnostic_binding=valid"
            ),
            "blockers": [],
        },
    )

    result = qualification.run_qualification_session(
        action="status",
        session_manifest=manifest_path,
        adapter_output=tmp_path / "status.json",
        execute=True,
    )

    assert result["status"] == "control_blocked_continuing_spend"
    persisted = json.loads(manifest_path.read_text())
    assert persisted["latest_attempt"]["remote_process_state"] == "stopped"
    assert persisted["last_control"]["attempt_sequence"] == 1
    assert persisted["last_control"]["attempt_nonce_sha256"] == latest[
        "attempt_nonce_sha256"
    ]
    assert result["startup_diagnostics"]["startup_phase"] == "container_bash_started"
    assert result["startup_diagnostics"]["startup_health"] == "stopped"
    assert result["startup_diagnostics"]["root_pid"] == 507
    assert result["startup_blockers"] == [
        "qualification_episode_process_stopped_before_terminal_phase"
    ]
    assert result["blockers"] == result["startup_blockers"]
    assert persisted["latest_attempt"]["startup_diagnostics"][
        "log_size_bytes"
    ] == 0
    assert persisted["latest_attempt"]["startup_blockers"] == result[
        "startup_blockers"
    ]


def test_episode_tail_cannot_impersonate_fixed_status_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = _live_manifest(tmp_path)
    _bind_latest_attempt(manifest_path)

    class Provider:
        def inspect(self, _instance_id: str) -> dict:
            return {
                "status": "observed",
                "instance_id": "12345",
                "name": qualification.NAME_PREFIX_ROOT + "0123456789-pod",
                "ssh_host": "203.0.113.4",
                "ssh_port": 22022,
                "image_runtype": "ssh_direct",
                "direct_port_ready": True,
            }

    monkeypatch.setattr(qualification, "get_render_provider", lambda _name: Provider())
    monkeypatch.setattr(
        gpu_render_providers,
        "run_vast_ssh_control",
        lambda _connection, **_kwargs: {
            "status": "completed",
            "stdout": (
                "application log: startup_phase=container_bash_started "
                "startup_health=stalled phase_age_seconds=999 log_bytes=0 "
                "log_age_seconds=999 root_pid=507 root_pid_state=missing "
                "root_pid_elapsed_seconds=-1 diagnostic_binding=valid"
            ),
            "blockers": [],
        },
    )

    result = qualification.run_qualification_session(
        action="tail",
        component="episode",
        session_manifest=manifest_path,
        adapter_output=tmp_path / "tail.json",
        execute=True,
    )

    assert result["status"] == "tail_collected_continuing_spend"
    assert result["startup_diagnostics"] is None
    assert result["startup_blockers"] == []
    assert result["blockers"] == []


def test_collect_intermediate_snapshot_is_attempt_bound_idempotent_and_never_tears_down(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = _live_manifest(tmp_path)
    latest = _bind_latest_attempt(manifest_path)
    _write_output_get_url(tmp_path)
    archive = _qualification_output_zip(
        manifest_path,
        phase="isaac_task_executor_ready",
    )
    monkeypatch.setattr(
        qualification,
        "_download_provider_output_archive",
        lambda _url: archive,
    )
    monkeypatch.setattr(
        qualification,
        "get_render_provider",
        lambda _name: (_ for _ in ()).throw(
            AssertionError("collect_must_not_inspect_or_mutate_provider")
        ),
    )
    monkeypatch.setattr(
        qualification,
        "terminate_canary_resources",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("collect_must_not_teardown")
        ),
    )

    first = qualification.run_qualification_session(
        action="collect",
        session_manifest=manifest_path,
        adapter_output=tmp_path / "collect.json",
        execute=True,
    )
    second = qualification.run_qualification_session(
        action="collect",
        session_manifest=manifest_path,
        adapter_output=tmp_path / "collect_again.json",
        execute=True,
    )

    assert first["status"] == "episode_snapshot_collected_continuing_spend"
    assert first["provider_mutations_performed"] == 0
    assert first["attempt_nonce_sha256"] == latest["attempt_nonce_sha256"]
    assert first["already_collected"] is False
    assert second["already_collected"] is True
    assert first["archive_sha256"] == second["archive_sha256"]
    assert Path(first["initial_artifacts"]["overview"]["path"]).read_bytes() == (
        b"initial-overview"
    )
    assert Path(first["initial_artifacts"]["robot_pov"]["path"]).read_bytes() == (
        b"initial-robot-pov"
    )
    assert first["artifact_paths"]["overview_frames"]
    assert first["artifact_paths"]["robot_pov_frames"]
    startup_diagnostics_path = Path(first["artifact_paths"]["startup_diagnostics"])
    assert startup_diagnostics_path.is_file()
    assert json.loads(startup_diagnostics_path.read_text(encoding="utf-8"))[
        "startup_phase"
    ] == "isaac_task_executor_ready"
    persisted = json.loads(manifest_path.read_text())
    assert persisted["latest_attempt"]["collection_status"] == "pending"
    assert len(persisted["collections"]) == 1
    assert "signature=secret" not in manifest_path.read_text()
    assert "signature=secret" not in (tmp_path / "collect.json").read_text()


def test_collect_rejects_stale_attempt_without_publishing_or_provider_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = _live_manifest(tmp_path)
    _bind_latest_attempt(manifest_path, sequence=1)
    _write_output_get_url(tmp_path)
    stale_archive = _qualification_output_zip(
        manifest_path,
        phase="runner_done",
        sequence=2,
    )
    monkeypatch.setattr(
        qualification,
        "_download_provider_output_archive",
        lambda _url: stale_archive,
    )
    monkeypatch.setattr(
        qualification,
        "get_render_provider",
        lambda _name: (_ for _ in ()).throw(AssertionError("provider_touched")),
    )

    with pytest.raises(ValueError, match="collected_output_stale_or_unbound"):
        qualification.run_qualification_session(
            action="collect",
            session_manifest=manifest_path,
            adapter_output=tmp_path / "collect.json",
            execute=True,
        )

    persisted = json.loads(manifest_path.read_text())
    assert persisted["collections"] == []
    snapshots = tmp_path / qualification.COLLECTIONS_DIR_NAME / "attempt_0001" / "snapshots"
    assert not list(snapshots.iterdir())


def test_collect_terminal_success_validates_semantics_and_all_review_media(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = _live_manifest(tmp_path)
    _bind_latest_attempt(manifest_path, remote_process_state="stopped")
    _write_output_get_url(tmp_path)
    archive = _qualification_output_zip(
        manifest_path,
        phase="runner_done",
        successful_terminal=True,
    )
    monkeypatch.setattr(
        qualification,
        "_download_provider_output_archive",
        lambda _url: archive,
    )

    result = qualification.run_qualification_session(
        action="collect",
        session_manifest=manifest_path,
        adapter_output=tmp_path / "collect.json",
        execute=True,
    )

    assert result["status"] == "episode_collected_passed_continuing_spend", result[
        "blockers"
    ]
    assert result["blockers"] == []
    assert result["validation"]["status"] == "passed"
    assert result["validation"]["final_review"]["status"] == "passed"
    assert Path(result["artifact_paths"]["final_review_video"]).is_file()
    assert Path(result["artifact_paths"]["overview_review_video"]).is_file()
    assert Path(result["artifact_paths"]["robot_pov_review_video"]).is_file()
    assert Path(result["artifact_paths"]["wam_prediction_review_video"]).is_file()
    persisted = json.loads(manifest_path.read_text())
    assert persisted["latest_attempt"]["collection_status"] == (
        "collected_terminal_passed"
    )


def test_collect_terminal_uses_immutable_attempt_release_binding_after_refresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = _live_manifest(tmp_path)
    _bind_latest_attempt(
        manifest_path, remote_process_state="stopped"
    )
    _write_output_get_url(tmp_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    refreshed_rebind = dict(manifest["qualification_launch_rebind"])
    refreshed_rebind["source_loaded_plan_sha256"] = "e" * 64
    refreshed_rebind["binding_sha256"] = "f" * 64
    manifest["qualification_launch_rebind"] = refreshed_rebind
    manifest["bootstrap"]["episode_bootstrap_sha256"] = "a" * 64
    manifest["bootstrap"]["overlay_revision"] = 2
    qualification._private_write_json(manifest_path, manifest)

    archive = _qualification_output_zip(
        manifest_path,
        phase="runner_done",
        successful_terminal=True,
    )
    launched = json.loads(manifest_path.read_text(encoding="utf-8"))[
        "latest_attempt"
    ]
    monkeypatch.setattr(
        qualification,
        "_download_provider_output_archive",
        lambda _url: archive,
    )

    result = qualification.run_qualification_session(
        action="collect",
        session_manifest=manifest_path,
        adapter_output=tmp_path / "collect-after-refresh.json",
        execute=True,
    )

    assert result["status"] == "episode_collected_passed_continuing_spend", result[
        "blockers"
    ]
    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert persisted["latest_attempt"]["qualification_launch_rebind"] == launched[
        "qualification_launch_rebind"
    ]
    assert persisted["qualification_launch_rebind"] == refreshed_rebind


def test_collect_terminal_rejects_self_consistent_nonrelease_attempt_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = _live_manifest(tmp_path)
    _bind_latest_attempt(manifest_path, remote_process_state="stopped")
    _write_output_get_url(tmp_path)
    archive = _qualification_output_zip(
        manifest_path,
        phase="runner_done",
        successful_terminal=True,
        attempt_input_overrides={"source_commit": "b" * 40},
    )
    monkeypatch.setattr(
        qualification,
        "_download_provider_output_archive",
        lambda _url: archive,
    )

    result = qualification.run_qualification_session(
        action="collect",
        session_manifest=manifest_path,
        adapter_output=tmp_path / "collect.json",
        execute=True,
    )

    assert result["status"] == "episode_collected_blocked_continuing_spend"
    assert any(
        blocker
        == "qualification_attempt_input_release_binding_mismatch:source_commit"
        for blocker in result["blockers"]
    )


def test_collected_terminal_blocked_attempt_allows_exact_next_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = _live_manifest(tmp_path)
    _bind_latest_attempt(manifest_path, remote_process_state="stopped")
    _write_output_get_url(tmp_path)
    blocked_archive = _qualification_output_zip(
        manifest_path,
        phase="runner_timeout",
    )
    monkeypatch.setattr(
        qualification,
        "_download_provider_output_archive",
        lambda _url: blocked_archive,
    )
    collected = qualification.run_qualification_session(
        action="collect",
        session_manifest=manifest_path,
        adapter_output=tmp_path / "collect.json",
        execute=True,
    )
    assert collected["status"] == "episode_collected_blocked_continuing_spend"
    assert collected["blockers"]

    manifest = json.loads(manifest_path.read_text())
    next_nonce = f"{manifest['launch_session_id']}:attempt_0002"
    next_nonce_sha = hashlib.sha256(next_nonce.encode()).hexdigest()

    class Provider:
        def inspect(self, _instance_id: str) -> dict:
            return {
                "status": "observed",
                "instance_id": "12345",
                "name": qualification.NAME_PREFIX_ROOT + "0123456789-pod",
                "ssh_host": "203.0.113.4",
                "ssh_port": 22022,
                "image_runtype": "ssh_direct",
                "direct_port_ready": True,
            }

    monkeypatch.setattr(qualification, "get_render_provider", lambda _name: Provider())
    monkeypatch.setattr(
        gpu_render_providers,
        "run_vast_ssh_control",
        lambda _connection, **_kwargs: {
            "status": "completed",
            "stdout": (
                "action=run component=episode pid=13 "
                f"bootstrap_sha256={manifest['bootstrap']['episode_bootstrap_sha256']} "
                "overlay_revision=1 attempt_sequence=2 "
                f"attempt_nonce_sha256={next_nonce_sha}"
            ),
            "blockers": [],
        },
    )
    rerun = qualification.run_qualification_session(
        action="run",
        session_manifest=manifest_path,
        admission_out=tmp_path / "rerun-admission.json",
        adapter_output=tmp_path / "rerun.json",
        execute=True,
    )
    assert rerun["status"] == "episode_dispatched_continuing_spend"
    assert json.loads(manifest_path.read_text())["latest_attempt"]["attempt_sequence"] == 2


def test_refresh_payload_changes_only_allowlisted_overlay_files(tmp_path: Path) -> None:
    manifest_path = _live_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text())

    refresh = qualification._materialize_qualification_refresh_payload(
        tmp_path,
        inputs=_minimal_inputs(),
        manifest=manifest,
    )
    payload = json.loads(Path(refresh["path"]).read_text())

    assert payload["schema_version"] == qualification.REFRESH_PAYLOAD_SCHEMA_VERSION
    assert payload["target_revision"] == 2
    assert sorted(payload["files"]) == [
        "qualification_episode_bootstrap.sh",
        "qualification_gear_sonic_controller.sh",
        "qualification_gear_sonic_isaac_dds_bridge.sh",
        "qualification_groot_microwave_finetune.sh",
        "qualification_groot_server.sh",
        "qualification_isaac_task_executor.sh",
    ]
    assert "blueprint_qualification_control.sh" not in payload["files"]
    episode_script = base64.b64decode(
        payload["files"]["qualification_episode_bootstrap.sh"]["base64"],
        validate=True,
    ).decode("utf-8")
    route_export = next(
        line
        for line in episode_script.splitlines()
        if line.startswith("export BLUEPRINT_ROUTE_JSON_B64=")
    )
    encoded_route = route_export.split("=", 1)[1].strip("'")
    assert json.loads(base64.b64decode(encoded_route, validate=True)) == {
        "route": []
    }
    assert (
        payload["immutable_binding"]["control_script_sha256"]
        == manifest["bootstrap"]["control_script_sha256"]
    )
    assert payload["immutable_binding"]["image_digest"] == manifest["image_digest"]
    assert payload["immutable_binding"]["bundle_sha256"] == manifest["bundle_sha256"]
    assert refresh["control_script_unchanged"] is True
    assert refresh["arbitrary_remote_command_allowed"] is False
    assert Path(refresh["path"]).stat().st_mode & 0o777 == 0o600
    compile(
        qualification._qualification_refresh_installer_source(),
        "blueprint_qualification_refresh.py",
        "exec",
    )


def test_v4_control_session_remains_refresh_compatible(tmp_path: Path) -> None:
    manifest_path = _live_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["bootstrap"]["control_contract_version"] = (
        "fixed_qualification_control_script.v4"
    )
    qualification._private_write_json(manifest_path, manifest)

    _, loaded = qualification._load_private_manifest(manifest_path)
    refresh = qualification._materialize_qualification_refresh_payload(
        tmp_path,
        inputs=_minimal_inputs(),
        manifest=loaded,
    )

    assert refresh["from_revision"] == 1
    assert refresh["target_revision"] == 2
    assert (
        refresh["immutable_binding"]["control_contract_version"]
        == "fixed_qualification_control_script.v4"
    )


def test_refresh_bootstrap_is_two_phase_digest_bound_and_audit_chained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = _live_manifest(tmp_path)
    before = json.loads(manifest_path.read_text())
    monkeypatch.setattr(
        qualification, "_load_single_episode_inputs", lambda _path: _minimal_inputs()
    )
    secret_url = tmp_path / "refresh_signed_url.txt"
    signed_url = "https://objects.example/refresh?signature=must-not-persist"
    secret_url.write_text(signed_url, encoding="utf-8")
    secret_url.chmod(0o600)

    class Provider:
        def inspect(self, instance_id: str) -> dict:
            assert instance_id == "12345"
            return {
                "status": "observed",
                "instance_id": "12345",
                "name": qualification.NAME_PREFIX_ROOT + "0123456789-pod",
                "ssh_host": "203.0.113.4",
                "ssh_port": 22022,
                "image_runtype": "ssh_direct",
                "direct_port_ready": True,
            }

    observed: dict = {}

    def fake_control(_connection, **kwargs):
        observed.update(kwargs)
        request = kwargs["refresh_request"]
        payload = json.loads(
            (tmp_path / qualification.QUALIFICATION_REFRESH_PAYLOAD_NAME).read_text()
        )
        episode_sha = payload["files"]["qualification_episode_bootstrap.sh"]["sha256"]
        return {
            "status": "completed",
            "stdout": (
                "action=refresh component=bootstrap "
                f"overlay_revision={request['target_revision']} "
                f"refresh_payload_sha256={request['refresh_payload_sha256']} "
                f"episode_bootstrap_sha256={episode_sha}"
            ),
            "stderr": "",
            "blockers": [],
        }

    monkeypatch.setattr(qualification, "get_render_provider", lambda _name: Provider())
    monkeypatch.setattr(gpu_render_providers, "run_vast_ssh_control", fake_control)
    common = dict(
        action="refresh-bootstrap",
        session_manifest=manifest_path,
        episode_bundle=tmp_path / "episode.zip",
        admission_out=tmp_path / "refresh-admission.json",
        adapter_output=tmp_path / "refresh_result.json",
        execute=True,
    )

    staged = qualification.run_qualification_session(
        **common,
        provider_bootstrap_url_file=None,
    )
    refreshed = qualification.run_qualification_session(
        **common,
        provider_bootstrap_url_file=secret_url,
    )

    assert staged["status"] == "refresh_bootstrap_staging_required_continuing_spend"
    assert staged["provider_mutations_performed"] == 0
    assert refreshed["status"] == "bootstrap_refreshed_continuing_spend"
    assert refreshed["provider_mutations_performed"] == 1
    admission = json.loads((tmp_path / "refresh-admission.json").read_text())
    assert admission["action"] == "refresh"
    assert admission["component"] == "bootstrap"
    assert refreshed["control_script_unchanged"] is True
    assert observed["action"] == "refresh"
    assert observed["component"] == "bootstrap"
    assert observed["refresh_request"]["signed_get_url"] == signed_url
    assert "command" not in observed
    after = json.loads(manifest_path.read_text())
    assert after["bootstrap"]["overlay_revision"] == 2
    assert (
        after["bootstrap"]["control_script_sha256"] == before["bootstrap"]["control_script_sha256"]
    )
    assert (
        after["bootstrap"]["refresh_installer_sha256"]
        == before["bootstrap"]["refresh_installer_sha256"]
    )
    for key in (
        "image_ref",
        "image_digest",
        "bundle_sha256",
        "instance_id",
        "resource_name",
        "resource_name_prefix",
        "launch_session_id",
        "launch_session_nonce_sha256",
    ):
        assert after[key] == before[key]
    assert after["pending_refresh"] is None
    assert len(after["refresh_audit_chain"]) == 1
    audit = after["refresh_audit_chain"][0]
    assert audit["previous_audit_sha256"] == "0" * 64
    assert len(audit["audit_sha256"]) == 64
    assert after["qualification_launch_rebind"]["release_source_commit"] == (
        TEST_SOURCE_COMMIT
    )
    assert audit["qualification_launch_rebind"] == after[
        "qualification_launch_rebind"
    ]
    assert after["last_refresh"]["active_qualification_launch_rebind"] == after[
        "qualification_launch_rebind"
    ]
    assert refreshed["qualification_launch_rebind"] == after[
        "qualification_launch_rebind"
    ]
    assert signed_url not in manifest_path.read_text()
    assert signed_url not in (tmp_path / "refresh_result.json").read_text()


def test_allocate_bad_bundle_writes_blocked_artifacts_without_provider_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Provider:
        def __getattr__(self, name: str):
            raise AssertionError(f"unexpected provider access: {name}")

    monkeypatch.setattr(qualification, "get_render_provider", lambda _name: Provider())
    outputs = {
        "provider_launch_request": tmp_path / "launch.json",
        "preflight_bundle": tmp_path / "preflight.json",
        "admission_out": tmp_path / "admission.json",
        "bound_request_out": tmp_path / "bound.json",
        "adapter_output": tmp_path / "result.json",
    }
    request = {
        "action": "allocate",
        "session_manifest": tmp_path / "session.json",
        "episode_bundle": tmp_path / "missing-episode.zip",
        "provider_bundle_url_file": tmp_path / "missing-bundle-url.txt",
        "provider_output_put_url_file": tmp_path / "missing-put-url.txt",
        "provider_output_get_url_file": tmp_path / "missing-get-url.txt",
        "release_evidence": tmp_path / "missing-release.json",
        "model_cache_evidence": _embedded_model_assets_evidence(
            tmp_path / "model-cache.json"
        ),
        "expected_source_commit": TEST_SOURCE_COMMIT,
        "execute": True,
        **outputs,
    }
    Path(request["release_evidence"]).write_text(
        json.dumps(_embedded_release_evidence(source_commit="c" * 40))
    )
    result = qualification.run_qualification_session(**request)

    assert result["status"] == "blocked"
    assert result["provider_mutations_performed"] == 0
    assert all(path.is_file() for path in outputs.values())
    manifest = json.loads((tmp_path / "session.json").read_text())
    _, reloaded = qualification._load_private_manifest(tmp_path / "session.json")
    assert reloaded["release_binding_status"] == "blocked"
    assert reloaded["source_commit"] == "c" * 40
    assert manifest["bootstrap"]["overlay_revision"] == 0
    assert manifest["bootstrap"]["control_contract_version"] == qualification.CONTROL_CONTRACT_VERSION

    Path(request["release_evidence"]).write_text(
        json.dumps(_embedded_release_evidence())
    )
    retry = qualification.run_qualification_session(**request)
    _, retry_manifest = qualification._load_private_manifest(tmp_path / "session.json")
    assert retry["status"] == "blocked"
    assert retry_manifest["release_binding_status"] == "bound"
    assert retry_manifest["image_ref"] == TEST_IMAGE_REF
    assert retry_manifest["source_commit"] == TEST_SOURCE_COMMIT


def test_changed_refresh_payload_forces_restage_before_remote_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = _live_manifest(tmp_path)
    current_inputs = _minimal_inputs()
    monkeypatch.setattr(
        qualification,
        "_load_single_episode_inputs",
        lambda _path: dict(current_inputs),
    )
    first = qualification.run_qualification_session(
        action="refresh-bootstrap",
        session_manifest=manifest_path,
        episode_bundle=tmp_path / "episode.zip",
        provider_bootstrap_url_file=None,
        adapter_output=tmp_path / "result.json",
        execute=True,
    )
    first_sha = first["refresh_payload"]["refresh_payload_sha256"]
    current_inputs["bootstrap_script"] = (
        "/opt/oscar-venv/bin/python -m "
        "blueprint_pipeline.gear_sonic_process_supervisor supervise -- "
        "/opt/wbc/deploy.sh sim > /workspace/gear_sonic_controller.log 2>&1 &\n"
        "cp /workspace/attempt_input_manifest_episode_001.json "
        "/workspace/attempt_input_manifest.json\n"
        "upload_phase inputs_ready\necho changed-exact-episode\n"
    )
    url_file = tmp_path / "url.txt"
    url_file.write_text("https://objects.example/stale-refresh", encoding="utf-8")
    url_file.chmod(0o600)
    monkeypatch.setattr(
        qualification,
        "get_render_provider",
        lambda _name: (_ for _ in ()).throw(
            AssertionError("provider touched before changed payload was restaged")
        ),
    )

    changed = qualification.run_qualification_session(
        action="refresh-bootstrap",
        session_manifest=manifest_path,
        episode_bundle=tmp_path / "episode.zip",
        provider_bootstrap_url_file=url_file,
        adapter_output=tmp_path / "result.json",
        execute=True,
    )

    assert changed["status"] == "refresh_bootstrap_staging_required_continuing_spend"
    assert changed["provider_mutations_performed"] == 0
    assert changed["refresh_payload"]["refresh_payload_sha256"] != first_sha


def test_attach_retries_keyscan_and_proves_authenticated_fixed_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Provider:
        def inspect(self, instance_id: str) -> dict:
            return {
                "status": "observed",
                "actual_status": "running",
                "instance_id": instance_id,
                "name": "bound-resource",
                "image_runtype": "ssh_direct",
                "ssh_host": "203.0.113.7",
                "ssh_port": 22022,
                "direct_port_ready": True,
            }

    enroll_count = 0

    def fake_enroll(_connection, **_kwargs):
        nonlocal enroll_count
        enroll_count += 1
        if enroll_count == 1:
            return {
                "status": "blocked",
                "blockers": ["vast_ssh_host_key_scan_failed"],
            }
        return {
            "status": "enrolled",
            "known_hosts_file": str(tmp_path / "vast_ssh_known_hosts"),
            "fingerprint_artifact": str(tmp_path / "fingerprint.json"),
            "tofu_pinned": True,
        }

    control_calls: list[dict] = []

    def fake_control(_connection, **kwargs):
        control_calls.append(kwargs)
        return {
            "status": "completed",
            "returncode": 0,
            "action": kwargs["action"],
            "component": kwargs["component"],
            "blockers": [],
            "strict_host_key_checking": True,
        }

    monkeypatch.setattr(gpu_render_providers, "enroll_vast_ssh_host_key", fake_enroll)
    monkeypatch.setattr(gpu_render_providers, "run_vast_ssh_control", fake_control)
    now = [0.0]

    connection, observations, host_key, probe = qualification._wait_for_qualification_attach(
        Provider(),
        instance_id="12345",
        resource_name="bound-resource",
        attempt_dir=tmp_path,
        identity_file=str(tmp_path / "identity"),
        timeout_seconds=10,
        clock=lambda: now[0],
        sleeper=lambda seconds: now.__setitem__(0, now[0] + seconds),
    )

    assert enroll_count == 2
    assert len(observations) == 2
    assert observations[0]["host_key_enrollment_status"] == "blocked"
    assert observations[1]["authenticated_control_status"] == "completed"
    assert connection["image_runtype"] == "ssh_direct"
    assert host_key["status"] == "enrolled"
    assert probe["status"] == "completed"
    assert control_calls == [
        {
            "action": "status",
            "component": "episode",
            "known_hosts_file": str(tmp_path / "vast_ssh_known_hosts"),
            "identity_file": str(tmp_path / "identity"),
            "timeout_seconds": 30.0,
            "tail_lines": 1,
        }
    ]


def test_attach_rejects_non_ssh_direct_provider_row_before_keyscan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Provider:
        calls = 0

        def inspect(self, instance_id: str) -> dict:
            self.calls += 1
            if self.calls > 1:
                return {"status": "absent", "instance_id": instance_id}
            return {
                "status": "observed",
                "actual_status": "running",
                "instance_id": instance_id,
                "name": "bound-resource",
                "image_runtype": "args",
                "ssh_host": "203.0.113.7",
                "ssh_port": 22022,
                "direct_port_ready": True,
            }

    monkeypatch.setattr(
        gpu_render_providers,
        "enroll_vast_ssh_host_key",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("keyscan attempted for non-ssh-direct instance")
        ),
    )
    now = [0.0]

    connection, observations, host_key, probe = qualification._wait_for_qualification_attach(
        Provider(),
        instance_id="12345",
        resource_name="bound-resource",
        attempt_dir=tmp_path,
        identity_file=str(tmp_path / "identity"),
        timeout_seconds=10,
        clock=lambda: now[0],
        sleeper=lambda seconds: now.__setitem__(0, now[0] + seconds),
    )

    assert connection == {}
    assert host_key == {}
    assert probe == {}
    assert observations[0]["ssh_direct_mode_confirmed"] is False


def test_attach_waits_for_running_before_binding_loading_proxy_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Provider:
        calls = 0

        def inspect(self, instance_id: str) -> dict:
            self.calls += 1
            if self.calls == 1:
                return {
                    "status": "observed",
                    "actual_status": "loading",
                    "instance_id": instance_id,
                    "name": "bound-resource",
                    "image_runtype": "ssh_direct",
                    "ssh_host": "ssh6.vast.ai",
                    "ssh_port": 14060,
                    "direct_port_ready": True,
                }
            return {
                "status": "observed",
                "actual_status": "running",
                "instance_id": instance_id,
                "name": "bound-resource",
                "image_runtype": "ssh_direct",
                "ssh_host": "203.0.113.7",
                "ssh_port": 22022,
                "direct_port_ready": True,
            }

    enrolled_connections: list[dict] = []

    def fake_enroll(connection, **_kwargs):
        enrolled_connections.append(connection)
        return {
            "status": "enrolled",
            "known_hosts_file": str(tmp_path / "vast_ssh_known_hosts"),
            "fingerprint_artifact": str(tmp_path / "fingerprint.json"),
            "tofu_pinned": True,
        }

    monkeypatch.setattr(gpu_render_providers, "enroll_vast_ssh_host_key", fake_enroll)
    monkeypatch.setattr(
        gpu_render_providers,
        "run_vast_ssh_control",
        lambda *_args, **_kwargs: {
            "status": "completed",
            "returncode": 0,
            "action": "status",
            "component": "episode",
            "blockers": [],
            "strict_host_key_checking": True,
        },
    )
    now = [0.0]

    connection, observations, _host_key, probe = (
        qualification._wait_for_qualification_attach(
            Provider(),
            instance_id="12345",
            resource_name="bound-resource",
            attempt_dir=tmp_path,
            identity_file=str(tmp_path / "identity"),
            timeout_seconds=10,
            clock=lambda: now[0],
            sleeper=lambda seconds: now.__setitem__(0, now[0] + seconds),
        )
    )

    assert observations[0]["actual_status"] == "loading"
    assert observations[0].get("host_key_enrollment_status") is None
    assert enrolled_connections == [connection]
    assert connection["ssh_host"] == "203.0.113.7"
    assert probe["status"] == "completed"


def test_teardown_closes_only_after_exact_prefix_and_global_absence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = _live_manifest(tmp_path)

    class Provider:
        def billable_inventory(self, *, name_prefix: str) -> dict:
            assert name_prefix == ""
            return {
                "status": "observed",
                "api_confirmed": True,
                "live_resource_count": 0,
                "resources": [],
            }

    monkeypatch.setattr(qualification, "get_render_provider", lambda _name: Provider())
    monkeypatch.setattr(
        qualification,
        "terminate_canary_resources",
        lambda **_kwargs: {
            "status": "provider_terminal",
            "provider_absence_confirmed": True,
            "provider_mutations_performed": 1,
        },
    )
    close_calls: list[tuple[str, dict]] = []

    def fake_close(path, proof):
        close_calls.append((str(path), dict(proof)))
        return {"status": "closed", "path": str(path)}

    monkeypatch.setattr(qualification, "close_pending_teardown", fake_close)

    result = qualification.run_qualification_session(
        action="teardown",
        session_manifest=manifest_path,
        adapter_output=tmp_path / "teardown.json",
        execute=True,
    )

    assert result["status"] == "teardown_completed_provider_zero"
    assert result["continuing_spend"] is False
    assert result["pending_teardown_status"] == "closed"
    assert result["watchdog_cancel_request"]["provider_absence_confirmed"] is True
    cancel_path = tmp_path / OWNER_TEARDOWN_CANCEL_NAME
    assert cancel_path.stat().st_mode & 0o777 == 0o600
    assert json.loads(cancel_path.read_text()) == result["watchdog_cancel_request"]
    assert close_calls[0][1]["exact_id_absence_confirmed"] is True
    assert close_calls[0][1]["name_prefix_absence_confirmed"] is True
    assert close_calls[0][1]["global_inventory_absence_confirmed"] is True
    persisted = json.loads(manifest_path.read_text())
    assert persisted["provider_absence_confirmed"] is True
    assert persisted["continuing_spend"] is False


def test_teardown_keeps_obligation_open_when_global_inventory_is_not_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = _live_manifest(tmp_path)

    class Provider:
        def billable_inventory(self, *, name_prefix: str) -> dict:
            return {
                "status": "observed",
                "api_confirmed": True,
                "live_resource_count": 1,
                "resources": [{"instance_id": "other"}],
            }

    monkeypatch.setattr(qualification, "get_render_provider", lambda _name: Provider())
    monkeypatch.setattr(
        qualification,
        "terminate_canary_resources",
        lambda **_kwargs: {
            "status": "provider_terminal",
            "provider_absence_confirmed": True,
            "provider_mutations_performed": 1,
        },
    )
    monkeypatch.setattr(
        qualification,
        "close_pending_teardown",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("pending teardown closed without global zero")
        ),
    )

    result = qualification.run_qualification_session(
        action="teardown",
        session_manifest=manifest_path,
        adapter_output=tmp_path / "teardown.json",
        execute=True,
    )

    assert result["status"] == "teardown_unverified_continuing_spend_unknown"
    assert result["continuing_spend"] is True
    assert result["blockers"] == ["qualification_teardown_exact_prefix_global_absence_not_proven"]


def test_allocate_dry_run_is_ssh_direct_and_writes_one_private_bound_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict = {}

    class Provider:
        def build_request(self, spec, _root):
            observed["spec"] = spec
            return {
                "provider": "vast",
                "bootstrap_transport": "onstart_plain",
                "create_payload": {"runtype": spec.vast_launch_mode},
            }

        def billable_inventory(self, *, name_prefix: str):
            assert name_prefix == ""
            return {
                "status": "observed",
                "api_confirmed": True,
                "live_resource_count": 0,
                "resources": [],
            }

        def capacity_preflight(self, _request):
            return {
                "status": "available",
                "selection_policy": {},
                "selected_offer": {},
                "viable_gpu_types": [{"on_demand_price_usd_per_hour": 0.5, "gpu_name": "L40S"}],
            }

    monkeypatch.setattr(
        qualification, "_load_single_episode_inputs", lambda _path: _minimal_inputs()
    )
    monkeypatch.setattr(qualification, "get_render_provider", lambda _name: Provider())

    secret_url = tmp_path / "signed_url.txt"
    secret_url.write_text("https://objects.example/artifact?signature=secret")
    secret_url.chmod(0o600)
    release = tmp_path / "release.json"
    release.write_text(json.dumps(_embedded_release_evidence()))
    model_cache = _embedded_model_assets_evidence(tmp_path / "model-cache.json")
    manifest_path = tmp_path / qualification.SESSION_MANIFEST_NAME

    common = dict(
        action="allocate",
        session_manifest=manifest_path,
        provider_name="vast",
        episode_bundle=tmp_path / "episode.zip",
        provider_bundle_url_file=secret_url,
        provider_output_put_url_file=secret_url,
        provider_output_get_url_file=secret_url,
        release_evidence=release,
        model_cache_evidence=model_cache,
        expected_source_commit=TEST_SOURCE_COMMIT,
        provider_launch_request=tmp_path / "provider_request.json",
        preflight_bundle=tmp_path / "preflight.json",
        admission_out=tmp_path / "admission.json",
        bound_request_out=tmp_path / "bound.json",
        adapter_output=tmp_path / "result.json",
        pod_name="ignored-unbound-name",
        execute=False,
    )
    staging = qualification.run_qualification_session(
        **common,
        provider_bootstrap_url_file=None,
    )
    staged_manifest = json.loads(manifest_path.read_text())
    result = qualification.run_qualification_session(
        **common,
        provider_bootstrap_url_file=secret_url,
    )

    assert staging["status"] == "bootstrap_staging_required"
    assert result["status"] == "dry_run_bound"
    assert result["provider_mutations_performed"] == 0
    assert result["pre_spend_preflight"]["status"] == "PASS"
    assert result["pre_spend_preflight"]["spend_admission_lock"]["required"] is False
    assert observed["spec"].vast_launch_mode == "ssh_direct"
    assert observed["spec"].image == TEST_IMAGE_REF
    rebound_plan = json.loads(
        base64.b64decode(
            observed["spec"].env["BLUEPRINT_SEALED_LAUNCH_PLAN_B64"]
        )
    )
    assert rebound_plan["image_ref"] == TEST_IMAGE_REF
    assert rebound_plan["qualification_release_rebind"]["source_bundle_preserved"] is True
    assert observed["spec"].env["BLUEPRINT_SOURCE_COMMIT"] == TEST_SOURCE_COMMIT
    assert observed["spec"].env[qualification.VAST_BOOTSTRAP_URL_ENV].startswith("https://")
    assert len(observed["spec"].env[qualification.VAST_BOOTSTRAP_SHA256_ENV]) == 64
    manifest = json.loads(manifest_path.read_text())
    assert manifest["launch_session_id"] == staged_manifest["launch_session_id"]
    assert manifest["resource_name_prefix"] == staged_manifest["resource_name_prefix"]
    assert (
        manifest["bootstrap"]["provider_bootstrap_sha256"]
        == staged_manifest["bootstrap"]["provider_bootstrap_sha256"]
    )
    assert manifest["image_ref"] == TEST_IMAGE_REF
    assert manifest["source_commit"] == TEST_SOURCE_COMMIT
    assert manifest["qualification_launch_rebind"]["release_image_ref"] == TEST_IMAGE_REF
    assert manifest["bundle_sha256"] == qualification.BUNDLE_SHA256
    assert manifest["continuing_spend"] is False
    assert manifest["bootstrap"]["arbitrary_remote_command_allowed"] is False
    assert manifest_path.stat().st_mode & 0o777 == 0o600

    release.write_text("{bad release evidence")
    bad_retry = qualification.run_qualification_session(
        **common,
        provider_bootstrap_url_file=secret_url,
    )
    preserved_after_bad_retry = json.loads(manifest_path.read_text())
    bad_preflight = json.loads(Path(common["preflight_bundle"]).read_text())
    bad_bound = json.loads(Path(common["bound_request_out"]).read_text())
    assert bad_retry["status"] == "blocked"
    assert preserved_after_bad_retry["release_binding_status"] == "bound"
    assert preserved_after_bad_retry["source_commit"] == TEST_SOURCE_COMMIT
    for artifact in (bad_preflight, bad_bound):
        assert artifact["image_ref"] == TEST_IMAGE_REF
        assert artifact["image_digest"] == TEST_IMAGE_DIGEST
        assert artifact["source_commit"] == TEST_SOURCE_COMMIT
        assert artifact["release_binding"]["source_commit"] == TEST_SOURCE_COMMIT
        assert artifact["release_binding_source"] == "session_manifest"
        assert artifact["release_binding"]["required_cuda_version"] == "12.8"

    changed_source = "c" * 40
    release.write_text(
        json.dumps(_embedded_release_evidence(source_commit=changed_source))
    )
    mismatch = qualification.run_qualification_session(
        **{**common, "expected_source_commit": changed_source},
        provider_bootstrap_url_file=secret_url,
    )
    preserved = json.loads(manifest_path.read_text())
    assert mismatch["status"] == "blocked"
    assert "qualification_existing_manifest_release_binding_mismatch" in mismatch["blockers"]
    assert preserved["image_ref"] == TEST_IMAGE_REF
    assert preserved["source_commit"] == TEST_SOURCE_COMMIT

    legacy_path = tmp_path / "legacy-session.json"
    legacy_manifest = dict(staged_manifest)
    legacy_manifest.pop("release_binding_status")
    legacy_manifest.pop("source_commit")
    qualification._private_write_json(legacy_path, legacy_manifest)
    release.write_text(json.dumps(_embedded_release_evidence()))
    legacy_retry = qualification.run_qualification_session(
        **{
            **common,
            "session_manifest": legacy_path,
            "provider_launch_request": tmp_path / "legacy-provider-request.json",
            "preflight_bundle": tmp_path / "legacy-preflight.json",
            "admission_out": tmp_path / "legacy-admission.json",
            "bound_request_out": tmp_path / "legacy-bound.json",
            "adapter_output": tmp_path / "legacy-result.json",
        },
        provider_bootstrap_url_file=secret_url,
    )
    legacy_after = json.loads(legacy_path.read_text())
    assert legacy_retry["status"] == "dry_run_bound"
    assert legacy_after["release_binding_status"] == "bound"
    assert legacy_after["source_commit"] == TEST_SOURCE_COMMIT
    qualification._load_private_manifest(legacy_path)


def test_qualification_execute_requires_current_paid_spend_lock_before_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launch_called = False

    class Provider:
        def build_request(self, spec, _root):
            return {
                "provider": "vast",
                "bootstrap_transport": "onstart_plain",
                "create_payload": {"runtype": spec.vast_launch_mode},
            }

        def billable_inventory(self, *, name_prefix: str):
            assert name_prefix == ""
            return {
                "status": "observed",
                "api_confirmed": True,
                "live_resource_count": 0,
                "resources": [],
            }

        def capacity_preflight(self, _request):
            selected = {
                "on_demand_price_usd_per_hour": 0.5,
                "gpu_name": "L40S",
                "gpu_ram_mb": 48_000,
            }
            return {
                "status": "available",
                "selected_offer": selected,
                "viable_gpu_types": [selected],
            }

        def launch(self, *_args, **_kwargs):
            nonlocal launch_called
            launch_called = True
            raise AssertionError("provider launch reached without current spend lock")

    monkeypatch.setattr(
        qualification, "_load_single_episode_inputs", lambda _path: _minimal_inputs()
    )
    monkeypatch.setattr(qualification, "get_render_provider", lambda _name: Provider())
    monkeypatch.setattr(
        qualification,
        "_require_signed_output_staging_proof",
        lambda **_kwargs: {
            "status": "passed",
            "raw_signed_urls_recorded": False,
        },
    )
    for name in (
        "BLUEPRINT_LAUNCH_PROOF_MODE",
        "BLUEPRINT_REQUIRE_PAID_SPEND_ADMISSION_LOCK",
        "BLUEPRINT_PAID_SPEND_ADMISSION_LOCK_PATH",
    ):
        monkeypatch.delenv(name, raising=False)

    secret_url = tmp_path / "signed_url.txt"
    secret_url.write_text("https://objects.example/artifact?signature=secret")
    secret_url.chmod(0o600)
    release = tmp_path / "release.json"
    release.write_text(json.dumps(_embedded_release_evidence()))
    model_cache = _embedded_model_assets_evidence(tmp_path / "model-cache.json")
    manifest_path = tmp_path / qualification.SESSION_MANIFEST_NAME
    common = dict(
        action="allocate",
        session_manifest=manifest_path,
        provider_name="vast",
        episode_bundle=tmp_path / "episode.zip",
        provider_bundle_url_file=secret_url,
        provider_output_put_url_file=secret_url,
        provider_output_get_url_file=secret_url,
        release_evidence=release,
        model_cache_evidence=model_cache,
        expected_source_commit=TEST_SOURCE_COMMIT,
        provider_launch_request=tmp_path / "provider_request.json",
        preflight_bundle=tmp_path / "preflight.json",
        admission_out=tmp_path / "admission.json",
        bound_request_out=tmp_path / "bound.json",
        adapter_output=tmp_path / "result.json",
        pod_name="ignored-unbound-name",
    )
    staged = qualification.run_qualification_session(
        **common,
        provider_bootstrap_url_file=None,
        execute=False,
    )
    result = qualification.run_qualification_session(
        **common,
        provider_bootstrap_url_file=secret_url,
        execute=True,
    )

    assert staged["status"] == "bootstrap_staging_required"
    assert result["status"] == "blocked"
    assert result["provider_mutations_performed"] == 0
    assert launch_called is False
    pre_spend = result["pre_spend_preflight"]
    assert pre_spend["status"] == "FAIL"
    assert pre_spend["spend_admission_lock"]["required"] is True
    assert any(
        blocker.startswith("spend_admission:spend_admission_lock_")
        for blocker in pre_spend["blockers"]
    )


def test_qualification_allocate_fails_before_provider_on_missing_output_staging_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        qualification,
        "_load_single_episode_inputs",
        lambda _path: _minimal_inputs(),
    )

    class Provider:
        def __getattr__(self, name: str):
            raise AssertionError(f"provider_must_not_be_used:{name}")

    monkeypatch.setattr(qualification, "get_render_provider", lambda _name: Provider())
    monkeypatch.setattr(
        qualification,
        "_require_signed_output_staging_proof",
        lambda **_kwargs: (_ for _ in ()).throw(
            ValueError("single_episode_signed_output_round_trip_not_proven")
        ),
    )
    secret_url = tmp_path / "signed_url.txt"
    secret_url.write_text(
        "https://objects.example/artifact?signature=must-not-leak",
        encoding="utf-8",
    )
    secret_url.chmod(0o600)
    release = tmp_path / "release.json"
    release.write_text(
        json.dumps(_embedded_release_evidence()),
        encoding="utf-8",
    )
    model_cache = _embedded_model_assets_evidence(tmp_path / "model-cache.json")

    result = qualification.run_qualification_session(
        action="allocate",
        session_manifest=tmp_path / qualification.SESSION_MANIFEST_NAME,
        provider_name="vast",
        episode_bundle=tmp_path / "episode.zip",
        provider_bundle_url_file=secret_url,
        provider_output_put_url_file=secret_url,
        provider_output_get_url_file=secret_url,
        provider_bootstrap_url_file=secret_url,
        release_evidence=release,
        model_cache_evidence=model_cache,
        expected_source_commit=TEST_SOURCE_COMMIT,
        provider_launch_request=tmp_path / "provider_request.json",
        preflight_bundle=tmp_path / "preflight.json",
        admission_out=tmp_path / "admission.json",
        bound_request_out=tmp_path / "bound.json",
        adapter_output=tmp_path / "result.json",
        execute=True,
    )

    assert result["status"] == "blocked"
    assert result["provider_mutations_performed"] == 0
    assert "single_episode_signed_output_round_trip_not_proven" in result["blockers"]
    persisted = (tmp_path / "result.json").read_text(encoding="utf-8")
    assert "must-not-leak" not in persisted


def test_allocator_routes_qualification_status_without_episode_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    observed: dict = {}

    def fake_run(**kwargs):
        observed.update(kwargs)
        return {"status": "status_observed_continuing_spend"}

    monkeypatch.setattr(allocator, "run_qualification_session", fake_run)
    result_path = tmp_path / "result.json"
    exit_code = allocator.main(
        [
            "gpu-canary",
            "--provider-launch-request",
            str(tmp_path / "request.json"),
            "--release-evidence",
            str(tmp_path / "release.json"),
            "--model-cache-evidence",
            str(tmp_path / "cache.json"),
            "--preflight-bundle",
            str(tmp_path / "preflight.json"),
            "--admission-out",
            str(tmp_path / "admission.json"),
            "--bound-request-out",
            str(tmp_path / "bound.json"),
            "--adapter-output",
            str(result_path),
            "--pod-name",
            "unused",
            "--provider",
            "vast",
            "--probe-kind",
            qualification.PROBE_KIND,
            "--qualification-action",
            "status",
            "--qualification-session-manifest",
            str(tmp_path / qualification.SESSION_MANIFEST_NAME),
            "--execute",
        ]
    )

    assert exit_code == 0
    assert observed["action"] == "status"
    assert observed["component"] == "episode"
    assert observed["provider_name"] == "vast"
    assert observed["model_cache_evidence"] == str(tmp_path / "cache.json")
    assert observed["execute"] is True
    assert json.loads(capsys.readouterr().out) == {"success": True}


def test_allocator_routes_collect_and_treats_collected_blocked_as_successful_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    observed: dict = {}

    def fake_run(**kwargs):
        observed.update(kwargs)
        return {"status": "episode_collected_blocked_continuing_spend"}

    monkeypatch.setattr(allocator, "run_qualification_session", fake_run)
    exit_code = allocator.main(
        [
            "gpu-canary",
            "--provider-launch-request",
            str(tmp_path / "request.json"),
            "--release-evidence",
            str(tmp_path / "release.json"),
            "--model-cache-evidence",
            str(tmp_path / "cache.json"),
            "--preflight-bundle",
            str(tmp_path / "preflight.json"),
            "--admission-out",
            str(tmp_path / "admission.json"),
            "--bound-request-out",
            str(tmp_path / "bound.json"),
            "--adapter-output",
            str(tmp_path / "collect.json"),
            "--pod-name",
            "unused",
            "--provider",
            "vast",
            "--probe-kind",
            qualification.PROBE_KIND,
            "--qualification-action",
            "collect",
            "--qualification-session-manifest",
            str(tmp_path / qualification.SESSION_MANIFEST_NAME),
            "--execute",
        ]
    )

    assert exit_code == 0
    assert observed["action"] == "collect"
    assert observed["execute"] is True
    assert json.loads(capsys.readouterr().out) == {"success": True}


def test_allocator_routes_refresh_bootstrap_with_fixed_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    observed: dict = {}

    def fake_run(**kwargs):
        observed.update(kwargs)
        return {"status": "bootstrap_refreshed_continuing_spend"}

    monkeypatch.setattr(allocator, "run_qualification_session", fake_run)
    exit_code = allocator.main(
        [
            "gpu-canary",
            "--provider-launch-request",
            str(tmp_path / "request.json"),
            "--release-evidence",
            str(tmp_path / "release.json"),
            "--model-cache-evidence",
            str(tmp_path / "cache.json"),
            "--preflight-bundle",
            str(tmp_path / "preflight.json"),
            "--admission-out",
            str(tmp_path / "admission.json"),
            "--bound-request-out",
            str(tmp_path / "bound.json"),
            "--adapter-output",
            str(tmp_path / "result.json"),
            "--pod-name",
            "unused",
            "--provider",
            "vast",
            "--probe-kind",
            qualification.PROBE_KIND,
            "--qualification-action",
            "refresh-bootstrap",
            "--qualification-session-manifest",
            str(tmp_path / qualification.SESSION_MANIFEST_NAME),
            "--episode-bundle",
            str(tmp_path / "episode.zip"),
            "--provider-bootstrap-url-file",
            str(tmp_path / "refresh_url.txt"),
            "--execute",
        ]
    )

    assert exit_code == 0
    assert observed["action"] == "refresh-bootstrap"
    assert observed["episode_bundle"] == str(tmp_path / "episode.zip")
    assert observed["provider_bootstrap_url_file"] == str(tmp_path / "refresh_url.txt")
    assert observed["component"] == "episode"
    assert json.loads(capsys.readouterr().out) == {"success": True}
