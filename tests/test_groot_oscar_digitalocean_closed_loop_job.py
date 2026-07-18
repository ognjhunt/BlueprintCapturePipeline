"""Hermetic tests for the sealed GR00T+OSCAR DigitalOcean launcher."""

from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import time
import zipfile
from pathlib import Path

import pytest

from blueprint_pipeline import gpu_render_providers as providers
from blueprint_pipeline import g1_kitchen_pre_allocation_identity as preallocation_identity
from blueprint_pipeline import groot_oscar_digitalocean_closed_loop_job as J
from blueprint_pipeline import groot_oscar_closed_loop_image as gocl
from blueprint_pipeline.g1_kitchen_pre_allocation_identity import (
    REGISTRY_EVIDENCE_SCHEMA_VERSION,
)
from blueprint_pipeline.g1_kitchen_worker_image_evidence import (
    assemble_worker_image_runtime_evidence,
)
from blueprint_pipeline import groot_oscar_digitalocean_job_inputs as J_INPUTS
from blueprint_pipeline.kitchen_attempt_lineage import build_attempt_input_manifest
from blueprint_pipeline.paid_resource_admission import (
    PAID_LANE_ADMISSION_SCHEMA_VERSION,
    build_paid_lane_admission,
    require_paid_resource_admission,
)


IMAGE_HASH = "b" * 64
DIGEST_REF = "docker.io/nijelhunt/blueprint-groot-oscar-eval@sha256:" + IMAGE_HASH
SOURCE_COMMIT = "a" * 40
DIRTY_PATCH = "f" * 64
TASK_PROMPT = (
    "Open the dishwasher door; if the dishwasher is already open, close the dishwasher door."
)
REAL_CONSISTENCY_COMMAND = "python -m owner_runtime.shared_model_inverse_scorer"


def _gpu_render_grant():
    admission = build_paid_lane_admission(resource_class="gpu_render", blockers=[])
    return require_paid_resource_admission(
        admission,
        resource_class="gpu_render",
        expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
    )


@pytest.fixture(autouse=True)
def _configure_real_consistency_scorer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        preallocation_identity,
        "inspect_registry_image",
        lambda image_ref: {
            "schema_version": REGISTRY_EVIDENCE_SCHEMA_VERSION,
            "source": "registry_api",
            "image_ref": image_ref,
            "digest": f"sha256:{IMAGE_HASH}",
        },
    )
    monkeypatch.setattr(
        J,
        "DEFAULT_CONFIGURED_WAM_CONSISTENCY_COMMAND",
        REAL_CONSISTENCY_COMMAND,
    )
    monkeypatch.setattr(
        preallocation_identity,
        "build_source_tree_identity",
        lambda _root: {
            "source_commit": SOURCE_COMMIT,
            "source_dirty_patch_sha256": DIRTY_PATCH,
        },
    )
    monkeypatch.setattr(
        J_INPUTS,
        "build_source_tree_identity",
        lambda _root: {
            "source_commit": SOURCE_COMMIT,
            "source_dirty_patch_sha256": DIRTY_PATCH,
            "dirty": False,
            "untracked_file_count": 0,
            "identity_includes_staged_unstaged_and_untracked": True,
        },
    )


def _completed_closed_loop_manifest(
    *,
    min_coherent_horizon_frames: int = 2,
    min_measured_coherent_horizon_frames: int = 2,
    forward_inverse_consistency_proven: bool = True,
    generated_video_success_label_passed: bool = False,
) -> dict:
    return {
        "status": "completed",
        "steps_executed": 3,
        "forward_inverse_consistency_proven": forward_inverse_consistency_proven,
        "episode_termination": {
            "reason": "task_target_reached_at_step_3",
            "steps_cap": J.DEFAULT_EPISODE_MAX_STEPS,
            "stop_on_task_completion": True,
            "min_steps": J.DEFAULT_MIN_TASK_COMPLETION_STEPS,
            "task_completed_early": True,
        },
        "generated_clip_coherence": {
            "min_coherent_horizon_frames_required": min_coherent_horizon_frames,
            "min_measured_coherent_horizon_frames": min_measured_coherent_horizon_frames,
        },
        "proof": {
            "feed_forward_verified": True,
            "policy_observes_wam_generated_next_observation": True,
            "fresh_learned_policy_requery_steps": 3,
            "external_episode_consistency_scorer_ran_steps": 3,
            "forward_inverse_consistency_proven_steps": 3
            if forward_inverse_consistency_proven
            else 2,
        },
        "success_proof": {
            "manipulation_success_proven": False,
            "simulated_manipulation_success_shown": False,
            "generated_video_success_label_passed": generated_video_success_label_passed,
            "real_world_task_success_proven": False,
            "success_proof_separate_from_structural_loop_proof": True,
        },
    }


def _inputs(
    tmp_path: Path, *, task_prompt: str = TASK_PROMPT, **plan_overrides
) -> tuple[Path, Path]:
    start_frame = tmp_path / "initial_policy_frame.png"
    start_frame.write_bytes(b"fake-png-bytes")
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "schema_version": "kitchen_random_task_selection.v1",
                "selected_task_id": "dishwasher_door",
            }
        ),
        encoding="utf-8",
    )
    selection_sha = hashlib.sha256(selection.read_bytes()).hexdigest()
    route = tmp_path / "route.json"
    route.write_text(
        json.dumps(
            {
                "route_points": [
                    [1.485591, 0.575381, 0.79],
                    [1.785591, 0.575381, 0.79],
                    [2.043728, 0.575381, 0.79],
                    [2.243728, 0.575381, 0.79],
                ],
                "accepted_stance_yaw_rad": 3.141593,
                "source_selection_sha256": selection_sha,
            }
        ),
        encoding="utf-8",
    )
    scenario = tmp_path / "scenario.json"
    scenario.write_text(
        json.dumps({"source_selection_sha256": selection_sha}), encoding="utf-8"
    )
    contract = tmp_path / "task_success_contract.json"
    contract.write_text(
        json.dumps(
            {
                "task_kind": "manipulation",
                "task_id": "dishwasher_door",
                "source_selection_sha256": selection_sha,
                "registered_criteria": [
                    {
                        "criterion_id": "dishwasher_door_open_angle",
                        "observable_transition": "articulation_angle_rad",
                        "comparison": "increase_at_least",
                        "tolerance": 0.35,
                        "unit": "rad",
                        "articulation_prim_path_resolution": {
                            "required_target_root": "Dishwasher054",
                            "required_affordance_terms": ["door"],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with zipfile.ZipFile(tmp_path / "kitchen_assets.zip", "w") as archive:
        archive.writestr("KitchenRoom.usd", "#usda 1.0")
    plan_arguments = {
        "env": {gocl.IMAGE_REF_ENV: DIGEST_REF, gocl.SEALED_CONFIRMED_ENV: "true"},
        "start_frame": "/workspace/initial_policy_frame.png",
        "route_file": "/workspace/route.json",
        "steps": J.DEFAULT_EPISODE_MAX_STEPS,
        "task_prompt": task_prompt,
        "output_dir": "/workspace/closed_loop_out",
        "min_coherent_horizon_frames": J.DEFAULT_MIN_COHERENT_HORIZON_FRAMES,
        "min_task_adaptive_steps": J.DEFAULT_MIN_TASK_COMPLETION_STEPS,
        "wam_consistency_command": REAL_CONSISTENCY_COMMAND,
        "require_generated_video_success_label": False,
        "wam_success_label_command": None,
        "allow_wam_success_labeling": False,
        "wam_success_label_timeout_seconds": J.DEFAULT_WAM_SUCCESS_LABEL_TIMEOUT_SECONDS,
        **plan_overrides,
    }
    plan = gocl.build_sealed_launch_plan(**plan_arguments)
    bundle = J_INPUTS._write_payload_bundle(
        payload_zip=tmp_path / "payload_bundle.zip",
        plan=plan,
        route_payload=json.loads(route.read_text(encoding="utf-8")),
        seed_path=start_frame,
        task_prompt=task_prompt,
        seed_provenance={"source": "unit-test-seed"},
        task_success_contract_path=contract,
        kitchen_asset_archive_path=tmp_path / "kitchen_assets.zip",
    )
    canary = {
        "status": "passed",
        "image_digest": f"sha256:{IMAGE_HASH}",
        "provider_allocation_id": "do-fixture-1",
        "run_id": "test-run-1",
        "attempt_id": "test-attempt-1",
        "launch_nonce": "test-launch-nonce-1",
    }
    evidence = assemble_worker_image_runtime_evidence(
        image_digest=f"sha256:{IMAGE_HASH}",
        source_commit=SOURCE_COMMIT,
        source_dirty_patch_sha256=DIRTY_PATCH,
        build_healthcheck={
            "status": "passed",
            "runtime_metadata": {
                "image_family": "isaac-eval-worker",
                "simulator_family": "isaac_sim",
                "simulator_major_version": 6,
                "blueprint_pipeline_imported": True,
                "configured_g1_asset_binding_valid": True,
                "configured_g1_usd_exists": True,
                "source_commit": SOURCE_COMMIT,
                "source_dirty_patch_sha256": DIRTY_PATCH,
            },
        },
        fast_canary=dict(canary),
        review_canary={**canary, "width": 640, "height": 480},
        teardown={"api_confirmed": True, "terminal_state": "terminated"},
        final_inventory={"api_confirmed": True, "live_resource_count": 0},
    )
    assert evidence["status"] == "passed", evidence["blockers"]
    evidence_path = tmp_path / "worker_image_runtime_evidence.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    manifest = build_attempt_input_manifest(
        run_id="test-run-1",
        attempt_id="test-attempt-1",
        launch_nonce="test-launch-nonce-1",
        provider="digitalocean",
        artifacts={
            "selection": selection,
            "scenario": scenario,
            "route": route,
            "task_success_contract": contract,
            "kitchen_inventory": tmp_path / "kitchen_assets.zip",
            "bundle": bundle,
            "worker_image_runtime_evidence": evidence_path,
        },
        image_digest=f"sha256:{IMAGE_HASH}",
        source_commit=SOURCE_COMMIT,
        source_dirty_patch_sha256=DIRTY_PATCH,
    )
    (tmp_path / "attempt_input_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (tmp_path / "registry_image_evidence.json").write_text(
        json.dumps(
            {
                "schema_version": REGISTRY_EVIDENCE_SCHEMA_VERSION,
                "image_ref": DIGEST_REF,
                "digest": f"sha256:{IMAGE_HASH}",
                "source": "registry_api",
            }
        ),
        encoding="utf-8",
    )
    return start_frame, route


def _requirements_by_id(audit: dict) -> dict[str, dict]:
    return {str(item["id"]): dict(item) for item in audit["requirements"]}


def _active_plan() -> dict:
    return gocl.build_sealed_launch_plan(
        env={gocl.IMAGE_REF_ENV: DIGEST_REF, gocl.SEALED_CONFIRMED_ENV: "true"},
        start_frame="/workspace/initial_policy_frame.png",
        route_file="/workspace/route.json",
        steps=J.DEFAULT_EPISODE_MAX_STEPS,
        task_prompt=TASK_PROMPT,
        output_dir="/workspace/closed_loop_out",
        wam_consistency_command=REAL_CONSISTENCY_COMMAND,
    )


def _available_do_capacity(
    *,
    size: str = "gpu-6000adax1-48gb",
    region: str = "atl1",
    gpu_ram_mb: int = J.DEFAULT_MIN_GPU_RAM_MB,
) -> dict:
    return {
        "status": "available",
        "provider": "digitalocean",
        "blockers": [],
        "viable_size_regions": [
            {
                "size": size,
                "provider_available": True,
                "provider_regions": [region],
                "matching_regions": [region],
                "price_hourly": 1.57,
                "memory_mb": 65536,
                "gpu_ram_mb": gpu_ram_mb,
                "vcpus": 8,
            }
        ],
        "raw_provider_response_recorded": False,
    }


class _PreparedOnlyProvider:
    name = "digitalocean"

    def available(self) -> dict:
        return {"provider": self.name, "available": True, "reason": None}

    def capacity_preflight(self, request=None):
        raise AssertionError("prepared mode must not query DigitalOcean capacity")


def _worker_image_manifest_diagnostic(tmp_path: Path) -> Path:
    """Write a completed registry manifest diagnostic bound to DIGEST_REF."""
    path = tmp_path / "isaac_worker_image_manifest_diagnostic.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "isaac_worker_image_manifest_diagnostic.v2",
                "status": "completed",
                "image_ref": "docker.io/nijelhunt/blueprint-groot-oscar-eval:20260712",
                "resolved_digest": f"sha256:{IMAGE_HASH}",
                "resolved_digest_ref": DIGEST_REF,
                "runnable_platform": "linux/amd64",
                "layer_count": 35,
                "total_compressed_size_bytes": 50_455_509_186,
                "largest_layer_size_bytes": 10_585_790_213,
                "layers_over_1gb": 4,
                "large_image_pull_risk": True,
                "split_layer_layout_suitable": False,
                "recommended_startup_no_runtime_timeout_seconds": 1800,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_paid_launch_fails_closed_without_cli_worker_image_manifest_diagnostic(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A paid digest-pinned launch must fail closed unless the registry
    manifest diagnostic is bound explicitly via the CLI argument; an env-var
    or default-path diagnostic is not evidence for the exact selected image
    (path_source must be cli_argument, same contract as the parity lane)."""
    start_frame, route = _inputs(tmp_path)

    class _NeverConsultedProvider(_PreparedOnlyProvider):
        def capacity_preflight(self, request=None):
            raise AssertionError(
                "diagnostic gate must block before capacity is consulted"
            )

    monkeypatch.setattr(
        J, "get_render_provider", lambda _name: _NeverConsultedProvider()
    )
    monkeypatch.setattr(
        J,
        "stage_bundle",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("diagnostic gate must block before staging")
        ),
    )
    # Even a valid diagnostic bound via env var must not satisfy the paid
    # gate: only the explicit CLI argument binding counts.
    monkeypatch.setenv(
        "BLUEPRINT_ISAAC_WORKER_IMAGE_MANIFEST_DIAGNOSTIC",
        str(_worker_image_manifest_diagnostic(tmp_path)),
    )

    manifest = J.run_groot_oscar_digitalocean_closed_loop_job(
        start_frame=start_frame,
        route_file=route,
        task_prompt=TASK_PROMPT,
        out_dir=tmp_path / "job",
        image_ref=DIGEST_REF,
        allow_paid=True,
        max_spend_usd=10.0,
    )

    assert manifest["status"] == "blocked"
    assert "worker_image_diagnostic_explicit_path_required" in manifest["blockers"]
    validation = manifest["worker_image_diagnostic_validation"]
    assert validation["required"] is True
    assert validation["status"] == "blocked"
    assert validation["digest_pinned_image"] is True
    assert manifest["worker_image_manifest_diagnostic"]["path_source"] == "env"
    assert "provider_capacity_preflight" not in manifest
    assert "staging" not in manifest


def test_paid_launch_accepts_cli_worker_image_manifest_diagnostic(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The CLI-bound diagnostic satisfies the fail-closed gate; the run then
    proceeds to the next paid gate (spend guard here, since no budget is
    given) instead of blocking on the diagnostic."""
    start_frame, route = _inputs(tmp_path)
    monkeypatch.setattr(J, "get_render_provider", lambda _name: _PreparedOnlyProvider())

    manifest = J.run_groot_oscar_digitalocean_closed_loop_job(
        start_frame=start_frame,
        route_file=route,
        task_prompt=TASK_PROMPT,
        out_dir=tmp_path / "job",
        image_ref=DIGEST_REF,
        allow_paid=True,
        worker_image_manifest_diagnostic=_worker_image_manifest_diagnostic(tmp_path),
    )

    assert manifest["status"] == "blocked"
    validation = manifest["worker_image_diagnostic_validation"]
    assert validation["status"] == "passed"
    assert validation["blockers"] == []
    diagnostic = manifest["worker_image_manifest_diagnostic"]
    assert diagnostic["path_source"] == "cli_argument"
    assert diagnostic["metadata_available_for_selected_image"] is True
    assert not any(
        blocker.startswith("worker_image_diagnostic")
        for blocker in manifest["blockers"]
    )
    assert "groot_oscar_closed_loop_prelaunch_spend_guard_not_passed" in manifest[
        "blockers"
    ]
    resume_command = json.loads(
        (tmp_path / "job" / J.PAID_RESUME_COMMAND_FILENAME).read_text(encoding="utf-8")
    )
    assert resume_command["argv"][
        resume_command["argv"].index("--worker-image-manifest-diagnostic") + 1
    ] == str(_worker_image_manifest_diagnostic(tmp_path))


def test_prepared_mode_records_diagnostic_gate_as_not_required(
    tmp_path: Path,
    monkeypatch,
) -> None:
    start_frame, route = _inputs(tmp_path)
    monkeypatch.setattr(J, "get_render_provider", lambda _name: _PreparedOnlyProvider())

    manifest = J.run_groot_oscar_digitalocean_closed_loop_job(
        start_frame=start_frame,
        route_file=route,
        task_prompt=TASK_PROMPT,
        out_dir=tmp_path / "job",
        image_ref=DIGEST_REF,
        seed_provenance={"source": "unit-test-seed"},
    )

    assert manifest["status"] == "prepared"
    assert manifest["worker_image_diagnostic_validation"]["required"] is False
    assert manifest["worker_image_diagnostic_validation"]["status"] == "not_required"


def test_route_without_attempt_bound_yaw_blocks_before_provider_or_staging(
    tmp_path: Path,
    monkeypatch,
) -> None:
    start_frame, route = _inputs(tmp_path)
    route_payload = json.loads(route.read_text(encoding="utf-8"))
    route_payload.pop("accepted_stance_yaw_rad")
    route.write_text(json.dumps(route_payload), encoding="utf-8")

    monkeypatch.setattr(
        J,
        "get_render_provider",
        lambda _name: (_ for _ in ()).throw(
            AssertionError("invalid route must block before provider lookup")
        ),
    )
    monkeypatch.setattr(
        J,
        "stage_bundle",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("invalid route must block before staging")
        ),
    )

    manifest = J.run_groot_oscar_digitalocean_closed_loop_job(
        start_frame=start_frame,
        route_file=route,
        task_prompt=TASK_PROMPT,
        out_dir=tmp_path / "job",
        image_ref=DIGEST_REF,
        allow_paid=True,
        max_spend_usd=10.0,
    )

    assert manifest["status"] == "blocked"
    assert manifest["blockers"] == ["persistent_isaac_route_start_pose_invalid"]
    assert "provider_capacity_preflight" not in manifest
    assert "staging" not in manifest


def test_cli_worker_image_manifest_diagnostic_flag_threads_to_job(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    diagnostic_path = _worker_image_manifest_diagnostic(tmp_path)
    captured: dict[str, object] = {}

    def _capture(**kwargs) -> dict:
        captured.update(kwargs)
        return {"status": "prepared"}

    monkeypatch.setattr(J, "run_groot_oscar_digitalocean_closed_loop_job", _capture)

    exit_code = J.main(
        [
            "--start-frame",
            str(tmp_path / "frame.png"),
            "--route-file",
            str(tmp_path / "route.json"),
            "--task-prompt",
            TASK_PROMPT,
            "--out-dir",
            str(tmp_path / "job"),
            "--worker-image-manifest-diagnostic",
            str(diagnostic_path),
        ]
    )

    assert exit_code == 0
    assert captured["worker_image_manifest_diagnostic"] == str(diagnostic_path)


def test_prepared_mode_writes_bundle_and_manifest_without_capacity_or_staging(
    tmp_path: Path,
    monkeypatch,
) -> None:
    start_frame, route = _inputs(tmp_path)

    monkeypatch.setattr(J, "get_render_provider", lambda _name: _PreparedOnlyProvider())
    monkeypatch.setattr(
        J,
        "stage_bundle",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("prepared mode must not stage")
        ),
    )

    manifest = J.run_groot_oscar_digitalocean_closed_loop_job(
        start_frame=start_frame,
        route_file=route,
        task_prompt=TASK_PROMPT,
        out_dir=tmp_path / "job",
        image_ref=DIGEST_REF,
        seed_provenance={"source": "unit-test-seed"},
        seed_provenance_file=tmp_path / "seed_provenance.json",
    )

    assert manifest["status"] == "prepared"
    assert manifest["steps"] == J.DEFAULT_EPISODE_MAX_STEPS
    assert manifest["min_coherent_horizon_frames"] == 2
    assert manifest["episode_termination"] == "task_completion_or_step_cap"
    assert manifest["episode_length_contract"] == {
        "episode_length_unit": "closed_loop_control_steps",
        "stop_condition": "task_completion_or_step_cap",
        "steps_cap": J.DEFAULT_EPISODE_MAX_STEPS,
        "min_steps_before_task_completion": J.DEFAULT_MIN_TASK_COMPLETION_STEPS,
        "steps_is_safety_cap": True,
        "stop_on_task_completion": True,
        "oscar_num_frames_arg": None,
        "oscar_num_frames_scope": "per_generation_clip_not_episode_limit",
        "episode_not_bound_to_oscar_clip_frames": True,
    }
    assert manifest["sealed_launch_plan"]["sealed_active"] is True
    assert manifest["sealed_launch_plan"]["episode_length_contract"][
        "episode_not_bound_to_oscar_clip_frames"
    ] is True
    assert manifest["sealed_launch_plan"]["episode_length_contract"][
        "min_steps_before_task_completion"
    ] == J.DEFAULT_MIN_TASK_COMPLETION_STEPS
    assert "--stop-on-task-completion" in manifest["sealed_launch_plan"]["closed_loop_command"]
    assert (
        manifest["sealed_launch_plan"]["closed_loop_command"][
            manifest["sealed_launch_plan"]["closed_loop_command"].index("--min-steps") + 1
        ]
        == str(J.DEFAULT_MIN_TASK_COMPLETION_STEPS)
    )
    assert (
        manifest["sealed_launch_plan"]["closed_loop_command"][
            manifest["sealed_launch_plan"]["closed_loop_command"].index(
                "--min-coherent-horizon-frames"
            )
            + 1
        ]
        == "2"
    )
    assert manifest["prelaunch_spend_guard"]["can_launch"] is False
    assert manifest["paid_launch_resume_command"]["will_query_digitalocean"] is True
    assert manifest["paid_launch_resume_command"]["capacity_preflight_before_staging"] is True
    resume_command = json.loads(
        (tmp_path / "job" / J.PAID_RESUME_COMMAND_FILENAME).read_text(encoding="utf-8")
    )
    assert resume_command["budget_placeholder"] == "<MAX_SPEND_USD_REQUIRED>"
    assert resume_command["argv"][:3] == [
        "python",
        "-m",
        "blueprint_pipeline.groot_oscar_digitalocean_closed_loop_job",
    ]
    assert "--allow-paid" in resume_command["argv"]
    assert resume_command["argv"][
        resume_command["argv"].index("--min-coherent-horizon-frames") + 1
    ] == "2"
    assert resume_command["argv"][resume_command["argv"].index("--min-steps") + 1] == str(
        J.DEFAULT_MIN_TASK_COMPLETION_STEPS
    )
    assert resume_command["argv"][resume_command["argv"].index("--image-ref") + 1] == DIGEST_REF
    assert (
        resume_command["argv"][resume_command["argv"].index("--seed-provenance-file") + 1]
        == str(tmp_path / "seed_provenance.json")
    )
    persisted = json.loads(
        (tmp_path / "job" / J.JOB_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert persisted["status"] == "prepared"

    with zipfile.ZipFile(manifest["bundle_zip"]) as zf:
        names = set(zf.namelist())
        bundle_manifest = json.loads(zf.read("bundle_manifest.json"))
        route_payload = json.loads(zf.read("route.json"))
        seed_provenance = json.loads(zf.read("seed_provenance.json"))

    assert {
        "initial_policy_frame.png",
        "route.json",
        "task_prompt.txt",
        "sealed_launch_plan.json",
        "seed_provenance.json",
        "bundle_manifest.json",
    } <= names
    assert bundle_manifest["task_prompt"] == TASK_PROMPT
    assert bundle_manifest["seed_provenance"] == {"source": "unit-test-seed"}
    assert route_payload["route_points"][0] == [1.485591, 0.575381, 0.79]
    assert seed_provenance == {"source": "unit-test-seed"}

    audit = J.audit_prepared_closed_loop_job(tmp_path / "job")
    assert audit["status"] == "PASS"
    assert audit["digitalocean_not_queried_by_audit"] is True
    assert audit["task_adaptive_termination"]["stop_on_task_completion"] is True
    assert audit["oscar_resolution"] == {
        "height": "480",
        "width": "640",
        "native_required": True,
    }
    assert audit["generated_clip_coherence_gate"] == {
        "min_coherent_horizon_frames": 2,
        "required_minimum": 2,
    }
    assert audit["forward_inverse_consistency_gate"] == {
        "required": True,
        "command": REAL_CONSISTENCY_COMMAND,
        "require_flag_present": True,
        "allow_scoring_flag_present": True,
        "command_arg": REAL_CONSISTENCY_COMMAND,
    }
    assert audit["generated_video_success_label_gate"]["required"] is False
    assert audit["digitalocean_gpu_candidate_floor"]["min_gpu_ram_mb"] == (
        J.DEFAULT_MIN_GPU_RAM_MB
    )
    assert "gpu-6000adax1-48gb" in audit["digitalocean_gpu_candidate_floor"][
        "allowed_size_candidates"
    ]
    assert any(
        row["size"] == "gpu-4000adax1-20gb"
        and row["reason"] == "below_min_gpu_ram"
        for row in audit["digitalocean_gpu_candidate_floor"]["rejected_size_candidates"]
    )
    assert audit["episode_length_contract"] == {
        "episode_length_unit": "closed_loop_control_steps",
        "stop_condition": "task_completion_or_step_cap",
        "steps_cap": J.DEFAULT_EPISODE_MAX_STEPS,
        "min_steps_before_task_completion": J.DEFAULT_MIN_TASK_COMPLETION_STEPS,
        "steps_is_safety_cap": True,
        "stop_on_task_completion": True,
        "oscar_num_frames_arg": None,
        "oscar_num_frames_scope": "per_generation_clip_not_episode_limit",
        "episode_not_bound_to_oscar_clip_frames": True,
        "manifest_contract_present": True,
        "sealed_launch_plan_contract_present": True,
        "resume_min_steps_before_task_completion": J.DEFAULT_MIN_TASK_COMPLETION_STEPS,
        "resume_oscar_num_frames_arg": None,
    }
    assert audit["budget_ready"] is False


def test_prepared_mode_preserves_strict_generated_video_success_label_gate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    command = "python -m blueprint_pipeline.wam_generated_video_success_label_openai"
    start_frame, route = _inputs(
        tmp_path,
        require_generated_video_success_label=True,
        wam_success_label_command=command,
        allow_wam_success_labeling=True,
        wam_success_label_timeout_seconds=123.0,
    )

    monkeypatch.setattr(J, "get_render_provider", lambda _name: _PreparedOnlyProvider())

    manifest = J.run_groot_oscar_digitalocean_closed_loop_job(
        start_frame=start_frame,
        route_file=route,
        task_prompt=TASK_PROMPT,
        out_dir=tmp_path / "job",
        image_ref=DIGEST_REF,
        seed_provenance={"source": "unit-test-seed"},
        require_generated_video_success_label=True,
        wam_success_label_command=command,
        allow_wam_success_labeling=True,
        wam_success_label_timeout_seconds=123.0,
    )

    assert manifest["status"] == "prepared"
    plan_cmd = manifest["sealed_launch_plan"]["closed_loop_command"]
    assert "--require-generated-video-success-label" in plan_cmd
    assert plan_cmd[plan_cmd.index("--wam-success-label-command") + 1] == command
    assert "--allow-wam-success-labeling" in plan_cmd
    assert plan_cmd[plan_cmd.index("--wam-success-label-timeout-seconds") + 1] == (
        "123.0"
    )
    assert manifest["sealed_launch_plan"]["quality_gate_contract"][
        "generated_video_success_label_required"
    ] is True
    assert manifest["sealed_launch_plan"]["quality_gate_contract"][
        "generated_video_success_label_command"
    ] == command

    resume_command = json.loads(
        (tmp_path / "job" / J.PAID_RESUME_COMMAND_FILENAME).read_text(encoding="utf-8")
    )
    resume_argv = resume_command["argv"]
    assert "--require-generated-video-success-label" in resume_argv
    assert resume_argv[resume_argv.index("--wam-success-label-command") + 1] == command
    assert "--allow-wam-success-labeling" in resume_argv
    assert resume_argv[resume_argv.index("--wam-success-label-timeout-seconds") + 1] == (
        "123.0"
    )

    audit = J.audit_prepared_closed_loop_job(tmp_path / "job")
    assert audit["status"] == "PASS"
    assert audit["generated_video_success_label_gate"] == {
        "required": True,
        "command": command,
        "require_flag_present": True,
        "allow_labeling_flag_present": True,
        "command_arg": command,
        "resume_require_flag_present": True,
        "resume_allow_labeling_flag_present": True,
        "resume_command_arg": command,
        "claim_boundary": (
            "Generated-video semantic success is a separate review gate. "
            "Prepared readiness does not prove manipulation success."
        ),
    }


def test_objective_readiness_audit_marks_local_ready_and_live_pending(
    tmp_path: Path,
    monkeypatch,
) -> None:
    start_frame, route = _inputs(tmp_path)
    monkeypatch.setattr(J, "get_render_provider", lambda _name: _PreparedOnlyProvider())
    manifest = J.run_groot_oscar_digitalocean_closed_loop_job(
        start_frame=start_frame,
        route_file=route,
        task_prompt=TASK_PROMPT,
        out_dir=tmp_path / "job",
        image_ref=DIGEST_REF,
        seed_provenance={"source": "unit-test-seed"},
    )

    audit = J.audit_kitchen_dishwasher_objective_readiness(tmp_path / "job")
    requirements = _requirements_by_id(audit)

    assert manifest["status"] == "prepared"
    assert audit["objective_status"] == "INCOMPLETE"
    assert audit["local_status"] == "PASS"
    assert audit["digitalocean_not_queried_by_objective_audit"] is True
    assert audit["failed_local_requirements"] == []
    assert set(audit["pending_live_requirements"]) == {
        "digitalocean_capacity_checked",
        "live_digitalocean_droplet_run_completed",
        "live_closed_loop_result_contract_passed",
        "semantic_task_success_evaluated",
        "attempt_bound_g1_kitchen_closure_completed",
    }
    assert requirements["kitchen_dishwasher_open_or_close_task"]["status"] == "PASS"
    assert requirements["provider_is_digitalocean"]["status"] == "PASS"
    assert requirements["sealed_groot_oscar_image_digest_pinned"]["status"] == "PASS"
    assert requirements["native_oscar_resolution"]["status"] == "PASS"
    assert requirements["episode_not_bound_to_81_frame_oscar_clip"]["status"] == "PASS"
    assert requirements["forward_inverse_consistency_gate_configured"]["status"] == "PASS"
    assert requirements["task_adaptive_termination"]["status"] == "PASS"
    assert requirements["gpu_and_disk_floor"]["status"] == "PASS"
    assert requirements["digitalocean_request_scoped_gpu_floor"]["status"] == "PASS"
    assert requirements["digitalocean_request_scoped_gpu_floor"]["evidence"][
        "min_gpu_ram_mb"
    ] == J.DEFAULT_MIN_GPU_RAM_MB
    assert requirements["digitalocean_capacity_checked"]["status"] == "PENDING"
    assert (
        tmp_path / "job" / J.OBJECTIVE_READINESS_AUDIT_FILENAME
    ).is_file()


def test_digitalocean_capacity_probe_artifact_updates_objective_audit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    start_frame, route = _inputs(tmp_path)
    monkeypatch.setattr(J, "get_render_provider", lambda _name: _PreparedOnlyProvider())
    manifest = J.run_groot_oscar_digitalocean_closed_loop_job(
        start_frame=start_frame,
        route_file=route,
        task_prompt=TASK_PROMPT,
        out_dir=tmp_path / "job",
        image_ref=DIGEST_REF,
        seed_provenance={"source": "unit-test-seed"},
    )

    class _BlockedCapacityProbeProvider(_PreparedOnlyProvider):
        def capacity_preflight(self, request=None):
            assert request["min_gpu_ram_mb"] == J.DEFAULT_MIN_GPU_RAM_MB
            assert request["capacity_preflight_before_staging"] is True
            return {
                "status": "blocked",
                "provider": self.name,
                "blockers": ["digitalocean_gpu_size_region_unavailable"],
                "raw_provider_response_recorded": False,
            }

    monkeypatch.setattr(
        J,
        "get_render_provider",
        lambda _name: _BlockedCapacityProbeProvider(),
    )
    probe = J.probe_digitalocean_capacity_for_prepared_dir(tmp_path / "job")
    audit = J.audit_kitchen_dishwasher_objective_readiness(tmp_path / "job")
    requirements = _requirements_by_id(audit)

    assert manifest["status"] == "prepared"
    assert probe["capacity_preflight"]["status"] == "blocked"
    assert (tmp_path / "job" / J.DIGITALOCEAN_CAPACITY_PROBE_FILENAME).is_file()
    assert audit["objective_status"] == "FAILED"
    assert audit["local_status"] == "PASS"
    assert audit["failed_local_requirements"] == []
    assert audit["failed_live_requirements"] == ["digitalocean_capacity_checked"]
    assert set(audit["pending_live_requirements"]) == {
        "live_digitalocean_droplet_run_completed",
        "live_closed_loop_result_contract_passed",
        "semantic_task_success_evaluated",
        "attempt_bound_g1_kitchen_closure_completed",
    }
    capacity_evidence = requirements["digitalocean_capacity_checked"]["evidence"]
    assert requirements["digitalocean_capacity_checked"]["status"] == "FAIL"
    assert capacity_evidence["capacity_source"] == "capacity_probe_artifact"
    assert capacity_evidence["capacity_status"] == "blocked"
    assert capacity_evidence["capacity_blockers"] == [
        "digitalocean_gpu_size_region_unavailable"
    ]


def test_capacity_wait_blocks_without_staging_while_digitalocean_capacity_absent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    start_frame, route = _inputs(tmp_path)
    monkeypatch.setattr(J, "get_render_provider", lambda _name: _PreparedOnlyProvider())
    manifest = J.run_groot_oscar_digitalocean_closed_loop_job(
        start_frame=start_frame,
        route_file=route,
        task_prompt=TASK_PROMPT,
        out_dir=tmp_path / "job",
        image_ref=DIGEST_REF,
        seed_provenance={"source": "unit-test-seed"},
    )

    class _BlockedCapacityWaitProvider(_PreparedOnlyProvider):
        def __init__(self) -> None:
            self.calls = 0

        def capacity_preflight(self, request=None):
            self.calls += 1
            assert request["min_gpu_ram_mb"] == J.DEFAULT_MIN_GPU_RAM_MB
            assert request["capacity_preflight_before_staging"] is True
            return {
                "status": "blocked",
                "provider": self.name,
                "blockers": ["digitalocean_gpu_size_region_unavailable"],
                "raw_provider_response_recorded": False,
            }

    provider = _BlockedCapacityWaitProvider()
    monkeypatch.setattr(J, "get_render_provider", lambda _name: provider)
    monkeypatch.setattr(
        J,
        "stage_bundle",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("capacity wait must not stage while capacity is blocked")
        ),
    )

    wait = J.wait_for_digitalocean_capacity_then_launch_prepared_dir(
        tmp_path / "job",
        max_attempts=2,
        poll_interval_seconds=0,
        launch_when_available=True,
        allow_paid=True,
        max_spend_usd=10.0,
        acknowledge_digitalocean_query_approval=True,
    )

    assert manifest["status"] == "prepared"
    assert provider.calls == 2
    assert wait["status"] == "capacity_blocked"
    assert wait["launch_started"] is False
    assert wait["object_store_staged"] is False
    assert wait["droplet_created"] is False
    assert wait["billable_provider_call"] is False
    assert [row["capacity_status"] for row in wait["attempts"]] == [
        "blocked",
        "blocked",
    ]
    assert (tmp_path / "job" / J.DIGITALOCEAN_CAPACITY_WAIT_FILENAME).is_file()
    assert (tmp_path / "job" / J.DIGITALOCEAN_CAPACITY_PROBE_FILENAME).is_file()
    assert (tmp_path / "job" / J.MATERIALIZED_PAID_COMMAND_FILENAME).is_file()


def test_capacity_wait_launches_after_capacity_appears_with_paid_teardown_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    start_frame, route = _inputs(tmp_path)
    monkeypatch.setattr(J, "get_render_provider", lambda _name: _PreparedOnlyProvider())
    manifest = J.run_groot_oscar_digitalocean_closed_loop_job(
        start_frame=start_frame,
        route_file=route,
        task_prompt=TASK_PROMPT,
        out_dir=tmp_path / "job",
        image_ref=DIGEST_REF,
        worker_image_manifest_diagnostic=_worker_image_manifest_diagnostic(tmp_path),
        seed_provenance={"source": "unit-test-seed"},
    )
    events: list[str] = []

    class _EventuallyAvailableProvider(_PreparedOnlyProvider):
        def __init__(self) -> None:
            self.capacity_calls = 0

        def capacity_preflight(self, request=None):
            self.capacity_calls += 1
            assert request["min_gpu_ram_mb"] == J.DEFAULT_MIN_GPU_RAM_MB
            assert request["capacity_preflight_before_staging"] is True
            events.append(f"capacity:{self.capacity_calls}")
            if self.capacity_calls == 1:
                return {
                    "status": "blocked",
                    "provider": self.name,
                    "blockers": ["digitalocean_gpu_size_region_unavailable"],
                    "raw_provider_response_recorded": False,
                }
            return _available_do_capacity()

        def build_request(self, spec, job_dir):
            events.append("build_request")
            assert spec.image == DIGEST_REF
            return {"name": spec.name, "env": dict(spec.env)}

        def launch(self, job_dir, request, *, cold=False, allow_cold_fallback=True):
            events.append("launch")
            assert cold is True
            assert request["prelaunch_spend_guard"]["can_launch"] is True
            return {"status": "launched", "instance_id": "do-4242"}

    provider = _EventuallyAvailableProvider()

    def fake_stage(bundle_zip, job_dir, *, key_prefix):
        events.append("stage")
        Path(job_dir).mkdir(parents=True)
        (Path(job_dir) / "provider_output_put_url.txt").write_text(
            "https://objects.example/out.zip?sig=put",
            encoding="utf-8",
        )
        (Path(job_dir) / "provider_bundle_url.txt").write_text(
            "https://objects.example/bundle.zip?sig=get",
            encoding="utf-8",
        )
        (Path(job_dir) / "provider_output_get_url.txt").write_text(
            "https://objects.example/out.zip?sig=get",
            encoding="utf-8",
        )
        assert Path(bundle_zip).is_file()
        return {"status": "completed"}

    def fake_watch(job_dir, out_dir, instance_id, *, provider, **kwargs):
        events.append("watch")
        assert instance_id == "do-4242"
        return {
            "status": "completed",
            "runner_result": {
                "status": "completed",
                "closed_loop_manifest": _completed_closed_loop_manifest(),
            },
            "runner_result_source": "isaac_runtime_result.json",
            "last_bootstrap": {"phase": "runner_done"},
            "teardown": {
                "status": "terminated",
                "verification": {"api_confirmed": True, "provider_status": "deleted"},
            },
            "teardown_reason": "runner_done_terminated_no_warm_reuse",
            "runner_done_observed": True,
        }

    monkeypatch.setattr(J, "get_render_provider", lambda _name: provider)
    monkeypatch.setattr(J, "stage_bundle", fake_stage)
    monkeypatch.setattr(
        J,
        "open_pending_teardown",
        lambda **_kwargs: {"path": str(tmp_path / "pending.json")},
    )
    monkeypatch.setattr(
        J,
        "bind_pending_teardown_instance",
        lambda _record_path, instance_id: events.append(f"bind:{instance_id}"),
    )
    monkeypatch.setattr(J, "close_pending_teardown", lambda *_args: {"status": "closed"})
    monkeypatch.setattr(J, "watch_and_collect", fake_watch)

    wait = J.wait_for_digitalocean_capacity_then_launch_prepared_dir(
        tmp_path / "job",
        max_attempts=2,
        poll_interval_seconds=0,
        launch_when_available=True,
        allow_paid=True,
        max_spend_usd=10.0,
        acknowledge_digitalocean_query_approval=True,
    )
    launched_manifest = json.loads(
        (tmp_path / "job" / J.JOB_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )

    assert manifest["status"] == "prepared"
    assert events == [
        "capacity:1",
        "capacity:2",
        "capacity:3",
        "stage",
        "build_request",
        "launch",
        "bind:do-4242",
        "watch",
    ]
    assert wait["status"] == "launch_blocked"
    assert wait["launch_started"] is True
    assert wait["object_store_staged"] is True
    assert wait["droplet_created"] is True
    assert wait["billable_provider_call"] is True
    assert wait["paid_launcher_repeated_capacity_preflight_before_staging"] is True
    assert wait["paid_launcher_uses_pending_teardown_record"] is True
    assert wait["paid_launcher_owns_teardown_proof"] is True
    assert launched_manifest["status"] == "blocked"
    assert launched_manifest["g1_kitchen_attempt_closure"]["status"] == "blocked"
    assert launched_manifest["closed_loop_result_contract"]["status"] == "PASS"


def test_objective_readiness_audit_fails_wrong_task_prompt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    start_frame, route = _inputs(
        tmp_path, task_prompt="Open the refrigerator door."
    )
    monkeypatch.setattr(J, "get_render_provider", lambda _name: _PreparedOnlyProvider())
    manifest = J.run_groot_oscar_digitalocean_closed_loop_job(
        start_frame=start_frame,
        route_file=route,
        task_prompt="Open the refrigerator door.",
        out_dir=tmp_path / "job",
        image_ref=DIGEST_REF,
        seed_provenance={"source": "unit-test-seed"},
    )

    audit = J.audit_kitchen_dishwasher_objective_readiness(tmp_path / "job")
    requirements = _requirements_by_id(audit)

    assert manifest["status"] == "prepared"
    assert audit["objective_status"] == "FAILED"
    assert audit["local_status"] == "FAIL"
    assert audit["failed_local_requirements"] == [
        "kitchen_dishwasher_open_or_close_task"
    ]
    assert requirements["kitchen_dishwasher_open_or_close_task"]["status"] == "FAIL"
    assert requirements["kitchen_dishwasher_open_or_close_task"]["evidence"][
        "task_prompt"
    ] == "Open the refrigerator door."


def test_materialize_paid_resume_command_fills_budget_without_executing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    start_frame, route = _inputs(tmp_path)
    monkeypatch.setattr(J, "get_render_provider", lambda _name: _PreparedOnlyProvider())
    manifest = J.run_groot_oscar_digitalocean_closed_loop_job(
        start_frame=start_frame,
        route_file=route,
        task_prompt=TASK_PROMPT,
        out_dir=tmp_path / "job",
        image_ref=DIGEST_REF,
        seed_provenance={"source": "unit-test-seed"},
    )

    command = J.materialize_paid_resume_command(
        tmp_path / "job",
        max_spend_usd=12.5,
        acknowledge_digitalocean_query_approval=True,
    )

    assert manifest["status"] == "prepared"
    assert command["status"] == "ready"
    assert command["max_spend_usd"] == 12.5
    assert command["will_query_digitalocean_if_executed"] is True
    assert command["executes_now"] is False
    assert command["argv"][command["argv"].index("--max-spend-usd") + 1] == "12.5"
    assert "<MAX_SPEND_USD_REQUIRED>" not in command["shell_command"]
    assert (tmp_path / "job" / J.MATERIALIZED_PAID_COMMAND_FILENAME).is_file()

    audit = J.audit_kitchen_dishwasher_objective_readiness(tmp_path / "job")
    requirements = _requirements_by_id(audit)
    paid_evidence = requirements[
        "paid_resume_requires_explicit_budget_and_do_approval"
    ]["evidence"]
    assert paid_evidence["materialized_command_status"] == "ready"
    assert paid_evidence["materialized_max_spend_usd"] == 12.5


def test_materialize_paid_resume_command_blocks_without_query_ack(
    tmp_path: Path,
    monkeypatch,
) -> None:
    start_frame, route = _inputs(tmp_path)
    monkeypatch.setattr(J, "get_render_provider", lambda _name: _PreparedOnlyProvider())
    manifest = J.run_groot_oscar_digitalocean_closed_loop_job(
        start_frame=start_frame,
        route_file=route,
        task_prompt=TASK_PROMPT,
        out_dir=tmp_path / "job",
        image_ref=DIGEST_REF,
        seed_provenance={"source": "unit-test-seed"},
    )

    command = J.materialize_paid_resume_command(
        tmp_path / "job",
        max_spend_usd=12.5,
        acknowledge_digitalocean_query_approval=False,
    )

    assert manifest["status"] == "prepared"
    assert command["status"] == "blocked"
    assert "digitalocean_query_approval_not_acknowledged" in command["blockers"]
    assert command["will_query_digitalocean_if_executed"] is False
    assert command["executes_now"] is False


def test_prepared_readiness_audit_fails_without_task_adaptive_stop(
    tmp_path: Path,
    monkeypatch,
) -> None:
    start_frame, route = _inputs(tmp_path)
    monkeypatch.setattr(J, "get_render_provider", lambda _name: _PreparedOnlyProvider())
    manifest = J.run_groot_oscar_digitalocean_closed_loop_job(
        start_frame=start_frame,
        route_file=route,
        task_prompt=TASK_PROMPT,
        out_dir=tmp_path / "job",
        image_ref=DIGEST_REF,
        seed_provenance={"source": "unit-test-seed"},
    )
    path = tmp_path / "job" / J.JOB_MANIFEST_FILENAME
    stored = json.loads(path.read_text(encoding="utf-8"))
    stored["sealed_launch_plan"]["closed_loop_command"].remove("--stop-on-task-completion")
    path.write_text(json.dumps(stored), encoding="utf-8")

    audit = J.audit_prepared_closed_loop_job(tmp_path / "job")

    assert manifest["status"] == "prepared"
    assert audit["status"] == "FAIL"
    assert "closed_loop_missing_stop_on_task_completion" in audit["blockers"]


def test_prepared_readiness_audit_fails_without_episode_length_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    start_frame, route = _inputs(tmp_path)
    monkeypatch.setattr(J, "get_render_provider", lambda _name: _PreparedOnlyProvider())
    manifest = J.run_groot_oscar_digitalocean_closed_loop_job(
        start_frame=start_frame,
        route_file=route,
        task_prompt=TASK_PROMPT,
        out_dir=tmp_path / "job",
        image_ref=DIGEST_REF,
        seed_provenance={"source": "unit-test-seed"},
    )
    path = tmp_path / "job" / J.JOB_MANIFEST_FILENAME
    stored = json.loads(path.read_text(encoding="utf-8"))
    del stored["episode_length_contract"]
    del stored["sealed_launch_plan"]["episode_length_contract"]
    path.write_text(json.dumps(stored), encoding="utf-8")

    audit = J.audit_prepared_closed_loop_job(tmp_path / "job")

    assert manifest["status"] == "prepared"
    assert audit["status"] == "FAIL"
    assert "prepared_manifest_missing_episode_length_contract" in audit["blockers"]
    assert "sealed_launch_plan_missing_episode_length_contract" in audit["blockers"]


def test_prepared_readiness_audit_fails_when_min_steps_drops_too_low(
    tmp_path: Path,
    monkeypatch,
) -> None:
    start_frame, route = _inputs(tmp_path)
    monkeypatch.setattr(J, "get_render_provider", lambda _name: _PreparedOnlyProvider())
    manifest = J.run_groot_oscar_digitalocean_closed_loop_job(
        start_frame=start_frame,
        route_file=route,
        task_prompt=TASK_PROMPT,
        out_dir=tmp_path / "job",
        image_ref=DIGEST_REF,
        seed_provenance={"source": "unit-test-seed"},
    )
    path = tmp_path / "job" / J.JOB_MANIFEST_FILENAME
    stored = json.loads(path.read_text(encoding="utf-8"))
    stored["episode_length_contract"]["min_steps_before_task_completion"] = 1
    stored["sealed_launch_plan"]["episode_length_contract"][
        "min_steps_before_task_completion"
    ] = 1
    cmd = stored["sealed_launch_plan"]["closed_loop_command"]
    cmd[cmd.index("--min-steps") + 1] = "1"
    path.write_text(json.dumps(stored), encoding="utf-8")

    resume_path = tmp_path / "job" / J.PAID_RESUME_COMMAND_FILENAME
    resume = json.loads(resume_path.read_text(encoding="utf-8"))
    resume["argv"][resume["argv"].index("--min-steps") + 1] = "1"
    resume_path.write_text(json.dumps(resume), encoding="utf-8")

    audit = J.audit_prepared_closed_loop_job(tmp_path / "job")

    assert manifest["status"] == "prepared"
    assert audit["status"] == "FAIL"
    assert "closed_loop_min_steps_before_task_completion_too_low" in audit["blockers"]
    assert "prepared_manifest_min_steps_before_task_completion_too_low" in audit["blockers"]
    assert "sealed_launch_plan_min_steps_before_task_completion_too_low" in audit["blockers"]
    assert "resume_command_min_steps_before_task_completion_too_low" in audit["blockers"]


def test_prepared_readiness_audit_fails_without_coherence_gate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    start_frame, route = _inputs(tmp_path)
    monkeypatch.setattr(J, "get_render_provider", lambda _name: _PreparedOnlyProvider())
    manifest = J.run_groot_oscar_digitalocean_closed_loop_job(
        start_frame=start_frame,
        route_file=route,
        task_prompt=TASK_PROMPT,
        out_dir=tmp_path / "job",
        image_ref=DIGEST_REF,
        seed_provenance={"source": "unit-test-seed"},
    )
    path = tmp_path / "job" / J.JOB_MANIFEST_FILENAME
    stored = json.loads(path.read_text(encoding="utf-8"))
    cmd = stored["sealed_launch_plan"]["closed_loop_command"]
    idx = cmd.index("--min-coherent-horizon-frames")
    del cmd[idx : idx + 2]
    path.write_text(json.dumps(stored), encoding="utf-8")

    audit = J.audit_prepared_closed_loop_job(tmp_path / "job")

    assert manifest["status"] == "prepared"
    assert audit["status"] == "FAIL"
    assert "closed_loop_missing_generated_clip_coherence_gate" in audit["blockers"]


def test_worker_bootstrap_runs_healthcheck_groot_and_task_adaptive_closed_loop() -> None:
    script = J.build_worker_bootstrap_script(_active_plan())

    assert "upload_phase container_bash_started" in script
    assert "groot_oscar_closed_loop_image_healthcheck.py --require-cuda" in script
    assert "/opt/gr00t-venv/bin/python -m gr00t.eval.run_gr00t_server" in script
    assert "BLUEPRINT_CLOSED_LOOP_RC=\"$RC\" python /workspace/write_result.py" in script
    assert "upload_phase runner_done" in script
    assert "--require-fresh-learned-policy-requery" in script
    assert "--allow-wam-consistency-scoring" in script
    assert "--require-forward-inverse-consistency" in script
    assert REAL_CONSISTENCY_COMMAND in script
    assert "--stop-on-task-completion" in script
    assert "--min-steps 3" in script
    assert "--min-coherent-horizon-frames 2" in script
    assert "--oscar-height 480" in script
    assert "--oscar-width 640" in script
    assert "--oscar-height 240" not in script
    assert ".replace(" not in script
    assert '"Content-Type: application/zip"' in script
    assert 'terminal_phase = phase in {"runner_done", "runner_timeout"}' in script
    assert "max_attempts = 5 if terminal_phase else 1" in script
    assert '"--connect-timeout", "30", "--max-time", "300"' in script
    assert "if uploaded.returncode == 0:" in script
    assert "raise SystemExit(75)" in script
    assert (
        'if [ "$1" = "runner_done" ] || [ "$1" = "runner_timeout" ]; then\n'
        '    BLUEPRINT_BOOTSTRAP_PHASE="$1" python /workspace/upload_progress.py\n'
    ) in script
    assert (
        'BLUEPRINT_BOOTSTRAP_PHASE="$1" python /workspace/upload_progress.py || true'
        in script
    )
    assert "SECRET_ROOT=/workspace/.runtime-secrets" in script
    assert "/run/blueprint-secrets" not in script
    assert "export GROOT_PID GEAR_SONIC_PID ISAAC_TASK_PID" in script
    assert "groot_policy_server_exited_before_ready" in script
    assert "official_gear_sonic_controller_exited_before_ready" in script
    assert "persistent_isaac_task_executor_exited_before_ready" in script


@pytest.mark.parametrize(
    ("return_codes", "expected_calls", "expected_exit"),
    [
        ([7, 0], 2, None),
        ([7, 7, 7, 7, 7], 5, 75),
    ],
)
def test_terminal_output_upload_retries_and_fails_closed_when_exhausted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    return_codes: list[int],
    expected_calls: int,
    expected_exit: int | None,
) -> None:
    script = J.build_worker_bootstrap_script(_active_plan())
    source = script.split("cat > /workspace/upload_progress.py <<'PY'\n", 1)[1].split(
        "\nPY\n\ncat > /workspace/write_result.py",
        1,
    )[0]
    source = source.replace(
        'workspace = Path("/workspace")',
        f"workspace = Path({str(tmp_path)!r})",
    )
    (tmp_path / "out").mkdir()
    calls: list[list[str]] = []

    def fake_run(command, *, check):  # noqa: ANN001
        assert check is False
        calls.append(list(command))
        return subprocess.CompletedProcess(command, return_codes[len(calls) - 1])

    monkeypatch.setenv("BLUEPRINT_BOOTSTRAP_PHASE", "runner_done")
    monkeypatch.setenv(
        "BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL",
        "https://objects.example/output?redacted-signature",
    )
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    if expected_exit is None:
        exec(compile(source, "upload_progress.py", "exec"), {})
    else:
        with pytest.raises(SystemExit) as exc_info:
            exec(compile(source, "upload_progress.py", "exec"), {})
        assert exc_info.value.code == expected_exit

    assert len(calls) == expected_calls
    assert all("--connect-timeout" in command and "--max-time" in command for command in calls)


def test_paid_missing_budget_blocks_before_capacity_preflight_or_staging(
    tmp_path: Path,
    monkeypatch,
) -> None:
    start_frame, route = _inputs(tmp_path)

    monkeypatch.setattr(J, "get_render_provider", lambda _name: _PreparedOnlyProvider())
    monkeypatch.setattr(
        J,
        "stage_bundle",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("spend guard must block before staging")
        ),
    )

    manifest = J.run_groot_oscar_digitalocean_closed_loop_job(
        start_frame=start_frame,
        route_file=route,
        task_prompt=TASK_PROMPT,
        out_dir=tmp_path / "job",
        image_ref=DIGEST_REF,
        allow_paid=True,
        worker_image_manifest_diagnostic=_worker_image_manifest_diagnostic(tmp_path),
    )

    assert manifest["status"] == "blocked"
    assert "groot_oscar_closed_loop_prelaunch_spend_guard_not_passed" in manifest["blockers"]
    assert "groot_oscar_closed_loop_max_spend_usd_missing" in manifest["blockers"]
    assert "provider_capacity_preflight" not in manifest
    assert "staging" not in manifest


def test_paid_capacity_preflight_blocks_before_staging(tmp_path: Path, monkeypatch) -> None:
    start_frame, route = _inputs(tmp_path)

    class _CapacityBlockedProvider(_PreparedOnlyProvider):
        def capacity_preflight(self, request=None):
            assert request["min_gpu_ram_mb"] >= J.DEFAULT_MIN_GPU_RAM_MB
            assert request["capacity_preflight_before_staging"] is True
            return {
                "status": "blocked",
                "provider": self.name,
                "blockers": ["digitalocean_gpu_size_region_unavailable"],
                "raw_provider_response_recorded": False,
            }

    monkeypatch.setattr(J, "get_render_provider", lambda _name: _CapacityBlockedProvider())
    monkeypatch.setattr(
        J,
        "stage_bundle",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("capacity preflight must block before staging")
        ),
    )

    manifest = J.run_groot_oscar_digitalocean_closed_loop_job(
        start_frame=start_frame,
        route_file=route,
        task_prompt=TASK_PROMPT,
        out_dir=tmp_path / "job",
        image_ref=DIGEST_REF,
        allow_paid=True,
        worker_image_manifest_diagnostic=_worker_image_manifest_diagnostic(tmp_path),
        max_spend_usd=10.0,
    )

    assert manifest["status"] == "blocked"
    assert "digitalocean_gpu_size_region_unavailable" in manifest["blockers"]
    assert "provider_capacity_unavailable_before_staging" in manifest["blockers"]
    assert "provider_capacity_preflight_status_blocked" in manifest["blockers"]
    assert "staging" not in manifest


def test_paid_capacity_preflight_unknown_blocks_before_staging(
    tmp_path: Path,
    monkeypatch,
) -> None:
    start_frame, route = _inputs(tmp_path)

    class _CapacityUnknownProvider(_PreparedOnlyProvider):
        def capacity_preflight(self, request=None):
            assert request["min_gpu_ram_mb"] >= J.DEFAULT_MIN_GPU_RAM_MB
            assert request["capacity_preflight_before_staging"] is True
            return {
                "status": "unknown",
                "provider": self.name,
                "blockers": ["digitalocean_token_missing"],
                "raw_provider_response_recorded": False,
            }

    monkeypatch.setattr(J, "get_render_provider", lambda _name: _CapacityUnknownProvider())
    monkeypatch.setattr(
        J,
        "stage_bundle",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unknown capacity must block before staging")
        ),
    )

    manifest = J.run_groot_oscar_digitalocean_closed_loop_job(
        start_frame=start_frame,
        route_file=route,
        task_prompt=TASK_PROMPT,
        out_dir=tmp_path / "job",
        image_ref=DIGEST_REF,
        allow_paid=True,
        worker_image_manifest_diagnostic=_worker_image_manifest_diagnostic(tmp_path),
        max_spend_usd=10.0,
    )

    assert manifest["status"] == "blocked"
    assert "digitalocean_token_missing" in manifest["blockers"]
    assert "provider_capacity_unavailable_before_staging" in manifest["blockers"]
    assert "provider_capacity_preflight_status_unknown" in manifest["blockers"]
    assert "staging" not in manifest


def test_build_launch_spec_carries_image_inputs_and_gpu_sizing(tmp_path: Path) -> None:
    start_frame, route = _inputs(tmp_path)
    job_dir = tmp_path / "object_store"
    job_dir.mkdir()
    (job_dir / "provider_output_put_url.txt").write_text(
        "https://objects.example/out.zip?sig=put",
        encoding="utf-8",
    )
    (job_dir / "provider_bundle_url.txt").write_text(
        "https://objects.example/bundle.zip?sig=get",
        encoding="utf-8",
    )
    route_payload = json.loads(route.read_text(encoding="utf-8"))

    spec = J.build_launch_spec(
        job_dir=job_dir,
        image_ref=DIGEST_REF,
        start_frame=start_frame,
        route_payload=route_payload,
        task_prompt=TASK_PROMPT,
        plan=_active_plan(),
        launch_nonce="test-launch-nonce-1",
        seed_provenance={"source": "unit-test"},
    )

    assert spec.image == DIGEST_REF
    assert spec.container_disk_gb >= 200
    assert spec.volume_gb >= 100
    assert spec.min_gpu_ram_mb >= 48000
    assert spec.env["BLUEPRINT_EVAL_MANIFEST_URI"].endswith("sig=get")
    assert spec.env["BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL"].endswith("sig=put")
    assert spec.env["BLUEPRINT_LAUNCH_SESSION_ID"] == "test-launch-nonce-1"
    bootstrap = spec.bootstrap_argv[-1]
    assert "kitchen_asset_startup_gate" in bootstrap
    assert "isaac_worker_runtime_preflight" in bootstrap
    assert "isaac_review_renderer_canary" in bootstrap
    assert "same_allocation_startup_gates_failed" in bootstrap
    assert base64.b64decode(spec.env["BLUEPRINT_INITIAL_POLICY_FRAME_B64"]) == b"fake-png-bytes"
    decoded_route = json.loads(base64.b64decode(spec.env["BLUEPRINT_ROUTE_JSON_B64"]))
    assert decoded_route == route_payload
    decoded_provenance = json.loads(base64.b64decode(spec.env["BLUEPRINT_SEED_PROVENANCE_B64"]))
    assert decoded_provenance == {"source": "unit-test"}


def test_digitalocean_provider_blocks_20gb_fallback_before_create(
    tmp_path: Path,
    monkeypatch,
) -> None:
    token_path = tmp_path / "do-token.txt"
    token_path.write_text("fake-token", encoding="utf-8")
    monkeypatch.setenv("DIGITALOCEAN_TOKEN_FILE", str(token_path))
    monkeypatch.setenv("BLUEPRINT_DO_GPU_SIZES", "gpu-4000adax1-20gb")

    def fail_do_call(*_args: object, **_kwargs: object) -> tuple[int, dict]:
        raise AssertionError("20 GB candidate must be rejected before droplet create")

    monkeypatch.setattr(providers, "_do_call", fail_do_call)

    provider = providers.DigitalOceanRenderProvider()
    launch = provider.launch(
        tmp_path / "job",
        {
            "name": "blueprint-groot-oscar-closed-loop",
            "size": "gpu-4000adax1-20gb",
            "region": "atl1",
            "image": providers.DO_GPU_BASE_IMAGE,
            "ssh_keys": [12345],
            "min_gpu_ram_mb": J.DEFAULT_MIN_GPU_RAM_MB,
            "max_hourly_rate_usd": J.DEFAULT_MAX_HOURLY_RATE_USD,
            "prelaunch_spend_guard": {
                "required_before_provider_launch": True,
                "can_launch": True,
            },
        },
        cold=True,
        paid_resource_admission_grant=_gpu_render_grant(),
    )

    assert launch["status"] == "blocked"
    assert "digitalocean_gpu_size_below_min_vram" in launch["blockers"]
    assert launch["gpu_ram_policy"]["min_gpu_ram_mb"] == J.DEFAULT_MIN_GPU_RAM_MB
    assert launch["gpu_ram_policy"]["allowed_size_candidates"] == []
    assert launch["gpu_ram_policy"]["rejected_size_candidates"][0]["size"] == (
        "gpu-4000adax1-20gb"
    )
    assert launch["gpu_ram_policy"]["rejected_size_candidates"][0]["reason"] == (
        "below_min_gpu_ram"
    )


def test_paid_launcher_blocks_20gb_capacity_row_before_staging(
    tmp_path: Path,
    monkeypatch,
) -> None:
    start_frame, route = _inputs(tmp_path)

    class _UnderprovisionedCapacityProvider(_PreparedOnlyProvider):
        def capacity_preflight(self, request=None):
            assert request["min_gpu_ram_mb"] >= J.DEFAULT_MIN_GPU_RAM_MB
            assert request["capacity_preflight_before_staging"] is True
            return _available_do_capacity(
                size="gpu-4000adax1-20gb",
                gpu_ram_mb=20000,
            )

    monkeypatch.setattr(
        J,
        "get_render_provider",
        lambda _name: _UnderprovisionedCapacityProvider(),
    )
    monkeypatch.setattr(
        J,
        "stage_bundle",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("under-provisioned lane must block before staging")
        ),
    )

    manifest = J.run_groot_oscar_digitalocean_closed_loop_job(
        start_frame=start_frame,
        route_file=route,
        task_prompt=TASK_PROMPT,
        out_dir=tmp_path / "job",
        image_ref=DIGEST_REF,
        allow_paid=True,
        worker_image_manifest_diagnostic=_worker_image_manifest_diagnostic(tmp_path),
        max_spend_usd=10.0,
    )

    assert manifest["status"] == "blocked"
    assert manifest["selected_digitalocean_capacity"]["size"] == "gpu-4000adax1-20gb"
    assert manifest["lane_hardware_contract"]["status"] == "FAIL"
    assert "gpu_vram_below_lane_floor:20gb_lt_40gb" in manifest[
        "lane_hardware_contract"
    ]["blockers"]
    assert manifest["pre_spend_preflight"]["status"] == "FAIL"
    assert "groot_oscar_closed_loop_pre_spend_preflight_not_passed" in manifest[
        "blockers"
    ]
    assert (
        "hardware_contract_invalid:gpu_vram_below_lane_floor:20gb_lt_40gb"
        in manifest["blockers"]
    )
    assert "staging" not in manifest
    persisted_preflight = json.loads(
        (tmp_path / "job" / "pre_spend_preflight.json").read_text(encoding="utf-8")
    )
    assert persisted_preflight["status"] == "FAIL"
    assert persisted_preflight["hardware_contract"]["gpu_type_id"] == (
        "gpu-4000adax1-20gb"
    )


def test_paid_launch_uses_capacity_staging_and_teardown_proof(
    tmp_path: Path,
    monkeypatch,
) -> None:
    start_frame, route = _inputs(tmp_path)
    events: list[str] = []

    class _LaunchProvider(_PreparedOnlyProvider):
        def capacity_preflight(self, request=None):
            events.append("capacity")
            assert request["min_gpu_ram_mb"] >= J.DEFAULT_MIN_GPU_RAM_MB
            assert request["capacity_preflight_before_staging"] is True
            return _available_do_capacity()

        def build_request(self, spec, job_dir):
            events.append("build_request")
            assert spec.image == DIGEST_REF
            return {"name": spec.name, "env": dict(spec.env)}

        def launch(self, job_dir, request, *, cold=False, allow_cold_fallback=True):
            events.append("launch")
            assert cold is True
            assert request["prelaunch_spend_guard"]["can_launch"] is True
            return {"status": "launched", "instance_id": "do-4242"}

    def fake_stage(bundle_zip, job_dir, *, key_prefix):
        events.append("stage")
        Path(job_dir).mkdir(parents=True)
        (Path(job_dir) / "provider_output_put_url.txt").write_text(
            "https://objects.example/out.zip?sig=put",
            encoding="utf-8",
        )
        (Path(job_dir) / "provider_bundle_url.txt").write_text(
            "https://objects.example/bundle.zip?sig=get",
            encoding="utf-8",
        )
        (Path(job_dir) / "provider_output_get_url.txt").write_text(
            "https://objects.example/out.zip?sig=get",
            encoding="utf-8",
        )
        assert Path(bundle_zip).is_file()
        assert key_prefix == "blueprint/test-groot-oscar"
        return {"status": "completed"}

    def fake_watch(job_dir, out_dir, instance_id, *, provider, **kwargs):
        events.append("watch")
        assert instance_id == "do-4242"
        assert kwargs["progress_stall_phases"] == J.WORKER_PROGRESS_STALL_PHASES
        return {
            "status": "completed",
            "runner_result": {
                "status": "completed",
                "closed_loop_manifest": _completed_closed_loop_manifest(),
            },
            "runner_result_source": "isaac_runtime_result.json",
            "last_bootstrap": {"phase": "runner_done"},
            "teardown": {
                "status": "terminated",
                "verification": {"api_confirmed": True, "provider_status": "deleted"},
            },
            "teardown_reason": "runner_done_terminated_no_warm_reuse",
            "runner_done_observed": True,
        }

    pending_path = tmp_path / "pending.json"
    close_records: list[dict] = []

    monkeypatch.setattr(J, "get_render_provider", lambda _name: _LaunchProvider())
    monkeypatch.setattr(J, "stage_bundle", fake_stage)
    monkeypatch.setattr(
        J,
        "open_pending_teardown",
        lambda **_kwargs: {"path": str(pending_path)},
    )
    monkeypatch.setattr(
        J,
        "bind_pending_teardown_instance",
        lambda record_path, instance_id: events.append(f"bind:{instance_id}"),
    )

    def fake_close(_record_path, teardown_proof):
        close_records.append(dict(teardown_proof))
        return {"status": "closed"}

    monkeypatch.setattr(J, "close_pending_teardown", fake_close)
    monkeypatch.setattr(J, "watch_and_collect", fake_watch)

    manifest = J.run_groot_oscar_digitalocean_closed_loop_job(
        start_frame=start_frame,
        route_file=route,
        task_prompt=TASK_PROMPT,
        out_dir=tmp_path / "job",
        image_ref=DIGEST_REF,
        allow_paid=True,
        worker_image_manifest_diagnostic=_worker_image_manifest_diagnostic(tmp_path),
        max_spend_usd=10.0,
        max_seconds=600,
        key_prefix="blueprint/test-groot-oscar",
        seed_provenance={"source": "unit-test-seed"},
    )

    assert manifest["status"] == "blocked"
    assert events == ["capacity", "stage", "build_request", "launch", "bind:do-4242", "watch"]
    assert manifest["capacity_preflight_request_shape"] == {
        "provider": "digitalocean",
        "min_gpu_ram_mb": J.DEFAULT_MIN_GPU_RAM_MB,
        "max_hourly_rate_usd": 3.5,
        "capacity_preflight_before_staging": True,
    }
    assert manifest["launch_request_shape"]["min_gpu_ram_mb"] >= 48000
    assert manifest["closed_loop_result_contract"]["status"] == "PASS"
    assert manifest["closed_loop_result_contract"]["forward_inverse_consistency"] == {
        "required": True,
        "proven": True,
        "external_episode_consistency_scorer_ran_steps": 3,
        "forward_inverse_consistency_proven_steps": 3,
    }
    assert (
        manifest["closed_loop_result_contract"]["min_steps_before_task_completion"]
        == J.DEFAULT_MIN_TASK_COMPLETION_STEPS
    )
    assert manifest["closed_loop_result_contract"]["task_success_summary"] == {
        "manipulation_success_proven": False,
        "simulated_manipulation_success_shown": False,
        "generated_video_success_label_passed": False,
        "real_world_task_success_proven": False,
        "success_proof_separate_from_structural_loop_proof": True,
    }
    assert manifest["teardown_proof"]["status"] == "PASS"
    assert close_records[0]["status"] == "PASS"
    assert manifest["pending_teardown_close"]["status"] == "closed"
    assert manifest["buyer_readout_projection"]["status"] == "blocked"

    audit = J.audit_kitchen_dishwasher_objective_readiness(tmp_path / "job")
    requirements = _requirements_by_id(audit)
    assert audit["objective_status"] == "FAILED"
    assert audit["local_status"] == "PASS"
    assert audit["failed_local_requirements"] == []
    assert audit["failed_live_requirements"] == [
        "attempt_bound_g1_kitchen_closure_completed"
    ]
    assert audit["pending_live_requirements"] == []
    assert audit["semantic_task_success_passed"] is False
    assert requirements["digitalocean_capacity_checked"]["status"] == "PASS"
    assert requirements["live_digitalocean_droplet_run_completed"]["status"] == "PASS"
    assert requirements["live_closed_loop_result_contract_passed"]["status"] == "PASS"
    assert requirements["semantic_task_success_evaluated"]["status"] == "PASS"
    assert requirements["semantic_task_success_evaluated"]["evidence"][
        "semantic_task_success_passed"
    ] is False


def test_objective_readiness_audit_marks_complete_when_semantic_success_passes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    start_frame, route = _inputs(tmp_path)

    class _LaunchProvider(_PreparedOnlyProvider):
        def capacity_preflight(self, request=None):
            return _available_do_capacity()

        def build_request(self, spec, job_dir):
            return {"name": spec.name, "env": dict(spec.env)}

        def launch(self, job_dir, request, *, cold=False, allow_cold_fallback=True):
            return {"status": "launched", "instance_id": "do-4242"}

    def fake_stage(bundle_zip, job_dir, *, key_prefix):
        Path(job_dir).mkdir(parents=True)
        (Path(job_dir) / "provider_output_put_url.txt").write_text(
            "https://objects.example/out.zip?sig=put",
            encoding="utf-8",
        )
        (Path(job_dir) / "provider_bundle_url.txt").write_text(
            "https://objects.example/bundle.zip?sig=get",
            encoding="utf-8",
        )
        (Path(job_dir) / "provider_output_get_url.txt").write_text(
            "https://objects.example/out.zip?sig=get",
            encoding="utf-8",
        )
        return {"status": "completed"}

    def fake_watch(job_dir, out_dir, instance_id, *, provider, **kwargs):
        return {
            "status": "completed",
            "runner_result": {
                "status": "completed",
                "closed_loop_manifest": _completed_closed_loop_manifest(
                    generated_video_success_label_passed=True
                ),
            },
            "runner_result_source": "isaac_runtime_result.json",
            "last_bootstrap": {"phase": "runner_done"},
            "teardown": {
                "status": "terminated",
                "verification": {"api_confirmed": True, "provider_status": "deleted"},
            },
            "teardown_reason": "runner_done_terminated_no_warm_reuse",
            "runner_done_observed": True,
        }

    monkeypatch.setattr(J, "get_render_provider", lambda _name: _LaunchProvider())
    monkeypatch.setattr(J, "stage_bundle", fake_stage)
    monkeypatch.setattr(
        J,
        "open_pending_teardown",
        lambda **_kwargs: {"path": str(tmp_path / "pending.json")},
    )
    monkeypatch.setattr(J, "bind_pending_teardown_instance", lambda *_args: {})
    monkeypatch.setattr(J, "close_pending_teardown", lambda *_args: {"status": "closed"})
    monkeypatch.setattr(J, "watch_and_collect", fake_watch)

    manifest = J.run_groot_oscar_digitalocean_closed_loop_job(
        start_frame=start_frame,
        route_file=route,
        task_prompt=TASK_PROMPT,
        out_dir=tmp_path / "job",
        image_ref=DIGEST_REF,
        allow_paid=True,
        worker_image_manifest_diagnostic=_worker_image_manifest_diagnostic(tmp_path),
        max_spend_usd=10.0,
        max_seconds=600,
        seed_provenance={"source": "unit-test-seed"},
    )

    audit = J.audit_kitchen_dishwasher_objective_readiness(tmp_path / "job")

    assert manifest["status"] == "blocked"
    assert audit["objective_status"] == "FAILED"
    assert audit["local_status"] == "PASS"
    assert "attempt_bound_g1_kitchen_closure_completed" in audit[
        "failed_live_requirements"
    ]
    assert audit["pending_live_requirements"] == []
    assert audit["semantic_task_success_passed"] is True


def test_paid_launch_blocks_when_collected_closed_loop_contract_regresses(
    tmp_path: Path,
    monkeypatch,
) -> None:
    start_frame, route = _inputs(tmp_path)

    class _LaunchProvider(_PreparedOnlyProvider):
        def capacity_preflight(self, request=None):
            assert request["min_gpu_ram_mb"] >= J.DEFAULT_MIN_GPU_RAM_MB
            return _available_do_capacity()

        def build_request(self, spec, job_dir):
            return {"name": spec.name, "env": dict(spec.env)}

        def launch(self, job_dir, request, *, cold=False, allow_cold_fallback=True):
            return {"status": "launched", "instance_id": "do-4242"}

    def fake_stage(bundle_zip, job_dir, *, key_prefix):
        Path(job_dir).mkdir(parents=True)
        (Path(job_dir) / "provider_output_put_url.txt").write_text(
            "https://objects.example/out.zip?sig=put",
            encoding="utf-8",
        )
        (Path(job_dir) / "provider_bundle_url.txt").write_text(
            "https://objects.example/bundle.zip?sig=get",
            encoding="utf-8",
        )
        (Path(job_dir) / "provider_output_get_url.txt").write_text(
            "https://objects.example/out.zip?sig=get",
            encoding="utf-8",
        )
        return {"status": "completed"}

    def fake_watch(job_dir, out_dir, instance_id, *, provider, **kwargs):
        return {
            "status": "completed",
            "runner_result": {
                "status": "completed",
                "closed_loop_manifest": _completed_closed_loop_manifest(
                    min_measured_coherent_horizon_frames=1
                ),
            },
            "runner_result_source": "isaac_runtime_result.json",
            "last_bootstrap": {"phase": "runner_done"},
            "teardown": {
                "status": "terminated",
                "verification": {"api_confirmed": True, "provider_status": "deleted"},
            },
            "teardown_reason": "runner_done_terminated_no_warm_reuse",
            "runner_done_observed": True,
        }

    monkeypatch.setattr(J, "get_render_provider", lambda _name: _LaunchProvider())
    monkeypatch.setattr(J, "stage_bundle", fake_stage)
    monkeypatch.setattr(
        J,
        "open_pending_teardown",
        lambda **_kwargs: {"path": str(tmp_path / "pending.json")},
    )
    monkeypatch.setattr(J, "bind_pending_teardown_instance", lambda *_args: {})
    monkeypatch.setattr(J, "close_pending_teardown", lambda *_args: {"status": "closed"})
    monkeypatch.setattr(J, "watch_and_collect", fake_watch)

    manifest = J.run_groot_oscar_digitalocean_closed_loop_job(
        start_frame=start_frame,
        route_file=route,
        task_prompt=TASK_PROMPT,
        out_dir=tmp_path / "job",
        image_ref=DIGEST_REF,
        allow_paid=True,
        worker_image_manifest_diagnostic=_worker_image_manifest_diagnostic(tmp_path),
        max_spend_usd=10.0,
        max_seconds=600,
    )

    assert manifest["status"] == "blocked"
    assert "closed_loop_result_contract_failed" in manifest["blockers"]
    assert "closed_loop_result_coherence_below_expected" in manifest["blockers"]
    assert manifest["closed_loop_result_contract"]["status"] == "FAIL"


def test_paid_launch_blocks_without_forward_inverse_consistency_proof(
    tmp_path: Path,
    monkeypatch,
) -> None:
    start_frame, route = _inputs(tmp_path)

    class _LaunchProvider(_PreparedOnlyProvider):
        def capacity_preflight(self, request=None):
            return _available_do_capacity()

        def build_request(self, spec, job_dir):
            return {"name": spec.name, "env": dict(spec.env)}

        def launch(self, job_dir, request, *, cold=False, allow_cold_fallback=True):
            return {"status": "launched", "instance_id": "do-4242"}

    def fake_stage(bundle_zip, job_dir, *, key_prefix):
        Path(job_dir).mkdir(parents=True)
        (Path(job_dir) / "provider_output_put_url.txt").write_text(
            "https://objects.example/out.zip?sig=put",
            encoding="utf-8",
        )
        (Path(job_dir) / "provider_bundle_url.txt").write_text(
            "https://objects.example/bundle.zip?sig=get",
            encoding="utf-8",
        )
        (Path(job_dir) / "provider_output_get_url.txt").write_text(
            "https://objects.example/out.zip?sig=get",
            encoding="utf-8",
        )
        return {"status": "completed"}

    def fake_watch(job_dir, out_dir, instance_id, *, provider, **kwargs):
        return {
            "status": "completed",
            "runner_result": {
                "status": "completed",
                "closed_loop_manifest": _completed_closed_loop_manifest(
                    forward_inverse_consistency_proven=False
                ),
            },
            "runner_result_source": "isaac_runtime_result.json",
            "last_bootstrap": {"phase": "runner_done"},
            "teardown": {
                "status": "terminated",
                "verification": {"api_confirmed": True, "provider_status": "deleted"},
            },
            "teardown_reason": "runner_done_terminated_no_warm_reuse",
            "runner_done_observed": True,
        }

    monkeypatch.setattr(J, "get_render_provider", lambda _name: _LaunchProvider())
    monkeypatch.setattr(J, "stage_bundle", fake_stage)
    monkeypatch.setattr(
        J,
        "open_pending_teardown",
        lambda **_kwargs: {"path": str(tmp_path / "pending.json")},
    )
    monkeypatch.setattr(J, "bind_pending_teardown_instance", lambda *_args: {})
    monkeypatch.setattr(J, "close_pending_teardown", lambda *_args: {"status": "closed"})
    monkeypatch.setattr(J, "watch_and_collect", fake_watch)

    manifest = J.run_groot_oscar_digitalocean_closed_loop_job(
        start_frame=start_frame,
        route_file=route,
        task_prompt=TASK_PROMPT,
        out_dir=tmp_path / "job",
        image_ref=DIGEST_REF,
        allow_paid=True,
        worker_image_manifest_diagnostic=_worker_image_manifest_diagnostic(tmp_path),
        max_spend_usd=10.0,
        max_seconds=600,
    )

    assert manifest["status"] == "blocked"
    assert "closed_loop_result_contract_failed" in manifest["blockers"]
    assert "closed_loop_result_forward_inverse_consistency_not_proven" in manifest[
        "blockers"
    ]


def test_closed_loop_result_contract_blocks_when_strict_success_label_unproven() -> None:
    failing = J._closed_loop_result_contract(
        watch={
            "status": "completed",
            "runner_result": {
                "status": "completed",
                "closed_loop_manifest": _completed_closed_loop_manifest(
                    generated_video_success_label_passed=False
                ),
            },
            "runner_result_source": "isaac_runtime_result.json",
        },
        expected_steps_cap=J.DEFAULT_EPISODE_MAX_STEPS,
        expected_min_task_completion_steps=J.DEFAULT_MIN_TASK_COMPLETION_STEPS,
        expected_min_coherent_horizon_frames=J.DEFAULT_MIN_COHERENT_HORIZON_FRAMES,
        expected_forward_inverse_consistency_required=True,
        expected_generated_video_success_label_required=True,
    )

    assert failing["status"] == "FAIL"
    assert "closed_loop_result_generated_video_success_label_not_proven" in failing[
        "blockers"
    ]
    assert failing["generated_video_success_label"] == {
        "required": True,
        "passed": False,
    }

    passing = J._closed_loop_result_contract(
        watch={
            "status": "completed",
            "runner_result": {
                "status": "completed",
                "closed_loop_manifest": _completed_closed_loop_manifest(
                    generated_video_success_label_passed=True
                ),
            },
            "runner_result_source": "isaac_runtime_result.json",
        },
        expected_steps_cap=J.DEFAULT_EPISODE_MAX_STEPS,
        expected_min_task_completion_steps=J.DEFAULT_MIN_TASK_COMPLETION_STEPS,
        expected_min_coherent_horizon_frames=J.DEFAULT_MIN_COHERENT_HORIZON_FRAMES,
        expected_forward_inverse_consistency_required=True,
        expected_generated_video_success_label_required=True,
    )

    assert passing["status"] == "PASS"
    assert passing["generated_video_success_label"] == {
        "required": True,
        "passed": True,
    }


def test_paid_launch_blocks_on_identity_gate_before_capacity(
    tmp_path: Path, monkeypatch
) -> None:
    start_frame, route = _inputs(tmp_path)
    (tmp_path / "registry_image_evidence.json").unlink()
    monkeypatch.setattr(
        preallocation_identity,
        "inspect_registry_image",
        lambda _image_ref: (_ for _ in ()).throw(RuntimeError("registry unavailable")),
    )

    class _NeverConsultedProvider(_PreparedOnlyProvider):
        def capacity_preflight(self, request=None):
            raise AssertionError(
                "capacity must never be consulted while identity is unproven"
            )

    monkeypatch.setattr(
        J, "get_render_provider", lambda _name: _NeverConsultedProvider()
    )
    manifest = J.run_groot_oscar_digitalocean_closed_loop_job(
        start_frame=start_frame,
        route_file=route,
        task_prompt=TASK_PROMPT,
        out_dir=tmp_path / "job",
        image_ref=DIGEST_REF,
        seed_provenance={"source": "unit-test-seed"},
        allow_paid=True,
        worker_image_manifest_diagnostic=_worker_image_manifest_diagnostic(tmp_path),
        max_spend_usd=10.0,
    )
    assert manifest["status"] == "blocked"
    assert "pre_allocation_identity_gate_not_passed" in manifest["blockers"]
    assert any(
        "registry_image_live_inspection_failed" in item for item in manifest["blockers"]
    )


def test_paid_launch_blocks_on_stale_source_identity(
    tmp_path: Path, monkeypatch
) -> None:
    start_frame, route = _inputs(tmp_path)
    monkeypatch.setattr(
        preallocation_identity,
        "build_source_tree_identity",
        lambda _root: {
            "source_commit": "0" * 40,
            "source_dirty_patch_sha256": DIRTY_PATCH,
        },
    )

    class _NeverConsultedProvider(_PreparedOnlyProvider):
        def capacity_preflight(self, request=None):
            raise AssertionError(
                "capacity must never be consulted while identity is unproven"
            )

    monkeypatch.setattr(
        J, "get_render_provider", lambda _name: _NeverConsultedProvider()
    )
    manifest = J.run_groot_oscar_digitalocean_closed_loop_job(
        start_frame=start_frame,
        route_file=route,
        task_prompt=TASK_PROMPT,
        out_dir=tmp_path / "job",
        image_ref=DIGEST_REF,
        seed_provenance={"source": "unit-test-seed"},
        allow_paid=True,
        worker_image_manifest_diagnostic=_worker_image_manifest_diagnostic(tmp_path),
        max_spend_usd=10.0,
    )
    assert manifest["status"] == "blocked"
    assert "pre_allocation_identity_gate_not_passed" in manifest["blockers"]
    assert any(
        "attempt_source_commit_mismatch" in item for item in manifest["blockers"]
    )
