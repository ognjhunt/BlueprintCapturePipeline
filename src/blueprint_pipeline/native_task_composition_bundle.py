"""Prepare an exact-camera composition diagnostic using the existing GPU preflight gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from blueprint_pipeline.native_task_composition_diagnostic import (
    REQUEST_SCHEMA,
    PASSES,
    file_record,
    seal,
    validate_request,
)


def composition_runtime_sources(*, include_replay=False):
    from blueprint_pipeline.native_task_arena_construction_bundle import (
        construction_runtime_sources,
    )
    from blueprint_pipeline.provider_runtime_import_closure import (
        assert_provider_runtime_import_closure,
    )

    package = Path(__file__).resolve().parent
    names = {path.name: path for path in construction_runtime_sources()}
    # The shared AOV writer lives in the older Isaac runtime module. Ship its
    # complete import closure, including dormant adapters; no policy is invoked.
    for name in (
        "native_task_asset_composition_gate.py",
        "native_task_composition_diagnostic.py",
        "native_task_composition_worker.py",
        "native_task_arena_construction_worker.py",
        "adp009d_isaac_runtime.py",
        "adp009d_approach_capture.py",
        "adp009d_contact_envelope.py",
        "adp009d_hold_trace.py",
        "adp009d_newton_collision_adapter.py",
        "adp009d_newton_gripper_drive.py",
        "adp009d_physics_backend_comparison.py",
        "groot_n17_droid_policy_runtime.py",
        "native_franka_global_seed_search.py",
        "adp009d_groot_worker_identity.py",
        "groot_n17_wire_client.py",
        "policy_request_evidence.py",
    ):
        names[name] = package / name
    if include_replay:
        from blueprint_pipeline.native_task_arena_execution_contract import (
            COMBINED_DIAGNOSTIC_MODULE_NAMES,
        )
        names = {name: package / name for name in COMBINED_DIAGNOSTIC_MODULE_NAMES}
    assert_provider_runtime_import_closure(
        package_source_dir=package,
        shipped_module_names=names,
        code="composition_payload_import_closure",
    )
    return tuple(names.values())


def prepare_composition_bundle(
    *,
    job_dir,
    packet_dir,
    runtime_inputs,
    runtime_source_packet_receipt,
    implementation_commit,
    cell_index=0,
    render_refresh_count=8,
    retained_cell_result=None,
    retained_adapter_reset=None,
    require_composition_gate=False,
):
    from blueprint_pipeline.native_task_arena_bundle import build_native_task_arena_bundle
    from blueprint_pipeline.native_task_arena_policy_canary_worker import _resolved_scene_plan
    from blueprint_pipeline.native_task_isaaclab_launch import NATIVE_TASK_ARENA_IMAGE
    from blueprint_pipeline.native_task_arena_runtime_preflight_bundle import (
        load_verified_native_task_arena_runtime_preflight_bundle,
    )
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest

    job = Path(job_dir)
    if require_composition_gate and (retained_cell_result or retained_adapter_reset):
        raise ValueError("composition_gate_cannot_repeat_retained_command_replay")
    if job.exists() and any(job.iterdir()):
        raise ValueError("composition_job_directory_must_be_fresh")
    packet = Path(packet_dir)
    base_path = packet / "native_task_arena_scene_plan.v1.json"
    receipt = json.loads((packet / "native_task_arena_packet_receipt.v1.json").read_text())
    base = json.loads(base_path.read_text())
    inputs = json.loads(Path(runtime_inputs).read_text())
    if inputs.get("runtime_inputs_digest") != canonical_digest(
        inputs, digest_field="runtime_inputs_digest"
    ):
        raise ValueError("composition_original_runtime_inputs_invalid")
    if type(cell_index) is not int or not 0 <= cell_index < len(inputs["cells"]):
        raise ValueError("composition_selected_cell_invalid")
    cell = inputs["cells"][cell_index]
    plan = _resolved_scene_plan(base, cell, task_success_contract=inputs["task_success_contract"])
    scene_collision = next(
        row for row in plan["objects"] if row["semantic_role"] == "scene_collision"
    )
    if scene_collision["visible"] is not False:
        raise ValueError("composition_requires_original_invisible_scene_collision")
    if not any(
        row["semantic_role"] == "task_support" and row["visible"] is True for row in plan["objects"]
    ):
        raise ValueError("composition_visible_task_support_missing")
    job.mkdir(parents=True, exist_ok=True)
    plan_path = job / "composition_scene_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    request = seal(
        {
            "schema_version": REQUEST_SCHEMA,
            "diagnostic_only": True,
            "source_run_id": inputs["run_id"],
            "packet_receipt_digest": receipt["receipt_digest"],
            "original_scene_plan": file_record(base_path),
            "original_scene_plan_digest": base["plan_digest"],
            "original_runtime_inputs": file_record(runtime_inputs),
            "runtime_inputs_digest": inputs["runtime_inputs_digest"],
            "selected_cell_index": cell_index,
            "cell_id": cell["cell_id"],
            "seed": int(cell["seed"]),
            "resolved_scene_plan": file_record(plan_path),
            "resolved_scene_plan_digest": plan["plan_digest"],
            "camera_role": "external",
            "require_composition_gate": bool(require_composition_gate),
            "camera_source": "actual_native_packet_runtime_camera_builder",
            "target_semantic_class": "task_support",
            "passes": list(PASSES),
            "render_refresh_count": render_refresh_count,
            "policy_queries_permitted": 0,
            "physics_steps_between_passes_permitted": 0,
            "source_asset_mutation_permitted": False,
            "assets": [
                {
                    key: row[key]
                    for key in ("name", "usd_path", "sha256", "size_bytes", "visible", "pose_world")
                }
                for row in plan["objects"]
            ],
            "occlusion_qualified": False,
            "source_removal_authorized": False,
        },
        "request_digest",
    )
    validate_request(request)
    request_path = job / "composition_request.json"
    request_path.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n")
    bound_inputs = {
        "composition_request.json": request_path,
        "composition_scene_plan.json": plan_path,
        "original_policy_runtime_inputs.json": runtime_inputs,
    }
    replay_request = None
    if bool(retained_cell_result) != bool(retained_adapter_reset):
        raise ValueError("combined_diagnostic_requires_source_and_calibration")
    if retained_cell_result:
        if cell_index != 0:
            raise ValueError("combined_diagnostic_composition_requires_cell00")
        from blueprint_pipeline.native_task_retained_command_replay import (
            build_retained_replay_request,
        )
        from blueprint_pipeline.native_task_composition_diagnostic import verify_record

        replay_plan = _resolved_scene_plan(
            base, inputs["cells"][1], task_success_contract=inputs["task_success_contract"]
        )
        replay_request = build_retained_replay_request(
            source_result_path=retained_cell_result,
            resolved_scene_plan=replay_plan,
            expected_cell=inputs["cells"][1],
        )
        source = json.loads(Path(retained_cell_result).read_text())
        calibration_episode = next(
            row for row in source["episodes"] if row["candidate_id"] == "groot_n17_droid"
        )
        verify_record(
            retained_adapter_reset, calibration_episode["evidence_artifacts"]["reset_state"]
        )
        replay_request.update(
            retained_adapter_reset=file_record(retained_adapter_reset),
            calibration_source_candidate_id="groot_n17_droid",
            calibration_used_for_metadata_only=True,
            fresh_gripper_measurement_claimed=False,
        )
        seal(replay_request, "request_digest")
        replay_plan_path = job / "replay_scene_plan.json"
        replay_plan_path.write_text(json.dumps(replay_plan, indent=2, sort_keys=True) + "\n")
        replay_request_path = job / "replay_request.json"
        replay_request_path.write_text(json.dumps(replay_request, indent=2, sort_keys=True) + "\n")
        bound_inputs.update(
            {
                "replay_request.json": replay_request_path,
                "replay_scene_plan.json": replay_plan_path,
                "retained_cell_result.json": retained_cell_result,
                "retained_adapter_reset.json": retained_adapter_reset,
            }
        )
    package = Path(__file__).resolve().parent
    bundle = build_native_task_arena_bundle(
        job_dir=job / "bundle",
        packet_dir=packet,
        worker_source=package
        / (
            "native_task_combined_diagnostic_worker.py"
            if replay_request
            else "native_task_composition_worker.py"
        ),
        runtime_module_sources=composition_runtime_sources(include_replay=bool(replay_request) or require_composition_gate),
        implementation_commit=implementation_commit,
        execution_mode="runtime_preflight",
        runtime_variant=('native_composition_and_retained_command_replay.v1' if replay_request
                         else 'native_asset_composition_gate.v1' if require_composition_gate else None),
        expected_output_filename="native_task_arena_runtime_preflight.v1.json",
        container_image=NATIVE_TASK_ARENA_IMAGE,
        runtime_source_packet_receipt=runtime_source_packet_receipt,
        bound_runtime_inputs=bound_inputs,
    )
    receipt_path = job / "composition_provider_bundle_receipt.json"
    receipt_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
    load_verified_native_task_arena_runtime_preflight_bundle(
        receipt_path,
        expected_implementation_commit=implementation_commit,
        expected_packet_receipt_digest=receipt["receipt_digest"],
    )
    result = seal(
        {
            "schema_version": "native_task_composition_diagnostic_preparation.v1",
            "status": "prepared_no_execution",
            "request_digest": request["request_digest"],
            "bundle_receipt": {"path": str(receipt_path), **file_record(receipt_path)},
            "provider_calls": 0,
            "gpu_execution_performed": False,
            "paid_authorization_granted": False,
            "canonical_allocator": "blueprint_pipeline.paid_resource_allocator gpu-canary",
            "probe_kind": "native-task-arena-runtime-preflight",
            "bundle_argument": "--native-task-arena-bundle-receipt",
            "replay_request_digest": replay_request["request_digest"] if replay_request else None,
            "maximum_replayed_commands": 174 if replay_request else 0,
            "maximum_resource_hard_cap_usd": 0.75 if replay_request else None,
            "maximum_resource_ttl_seconds": 1800 if replay_request else None,
            "maximum_provider_allocations": 1,
            "retry_cap": 0,
            "claim_ceiling": "diagnostic_render_attribution_only",
        }
    )
    (job / "preparation.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--packet-dir", required=True)
    parser.add_argument("--runtime-inputs", required=True)
    parser.add_argument("--runtime-source-packet-receipt", required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--cell-index", type=int, default=0)
    parser.add_argument("--render-refresh-count", type=int, default=8)
    parser.add_argument("--retained-cell-result")
    parser.add_argument("--retained-adapter-reset")
    parser.add_argument("--require-composition-gate", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        required=True,
        help="Build local payload; never allocate or upload",
    )
    args = vars(parser.parse_args(argv))
    args.pop("dry_run")
    result = prepare_composition_bundle(**args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
