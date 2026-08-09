"""Shared static runtime contracts for paid provider bundles."""

from __future__ import annotations

import json
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path


PROVIDER_RUNTIME_BUNDLE_KINDS = (
    "isaac",
    "adp_simready_isaac",
    "wam",
    "evaluator",
    "unitree_unifolm",
    "unitree_groot_n17_sonic",
    "adp_simpler",
    "adp_arena",
    "adp009d_isaac",
    "adp009d_articulated_native",
    "adp009d_ovrtx",
    "adp009d_aura_native",
    "adp_content_agents",
    "adp_joint_agent",
    "adp_aura_smoke",
    "adp_aura_interiorgs",
    "adp_inpaint360_interiorgs",
    "adp_gaussian_excision",
)


def wam_registered_alternative_inputs_present(
    *, bundle_path: Path, zip_entries: Sequence[str]
) -> bool:
    """Return whether a WAM bundle carries one complete registered input layout."""
    entries = set(zip_entries)
    standard_inputs_present = all(
        f"provider_runtime/cosmos3_input/{name}" in entries
        for name in (
            "initial_observation.png",
            "smoke_request_inventory.json",
            "action_streams.json",
        )
    )
    reference_inputs_present = all(
        f"provider_runtime/cosmos3_droid_reference/{name}" in entries
        for name in ("canary_manifest.json", "initial_observation.png", "action_streams.json")
    )
    powered_root = "provider_runtime/cosmos3_powered_droid/"
    powered_packet_name = powered_root + "packet.json"
    powered_inputs_present = False
    if powered_packet_name in entries:
        try:
            with zipfile.ZipFile(bundle_path) as archive:
                payload = json.loads(archive.read(powered_packet_name).decode("utf-8"))
            packet = dict(payload) if isinstance(payload, Mapping) else {}
            rows = packet.get("rows")
            powered_images = (
                {
                    powered_root + str(row.get("initial_observation_relative_path") or "")
                    for row in rows
                    if isinstance(row, Mapping)
                }
                if isinstance(rows, list)
                else set()
            )
            powered_inputs_present = (
                packet.get("schema_version") == "policy_ranking_powered_droid_provider_packet.v1"
                and isinstance(rows, list)
                and len(rows) == 51
                and len(powered_images) == 51
                and powered_images.issubset(entries)
                and all(
                    powered_root + "official_canary/" + name in entries
                    for name in (
                        "canary_manifest.json",
                        "initial_observation.png",
                        "action_streams.json",
                    )
                )
            )
        except (OSError, ValueError, zipfile.BadZipFile, json.JSONDecodeError):
            powered_inputs_present = False
    ctrl_world_inputs_present = all(
        f"provider_runtime/ctrl_world_replay/{name}" in entries
        for name in (
            "canary_manifest.json",
            "annotation.json",
            "view_0.mp4",
            "view_1.mp4",
            "view_2.mp4",
        )
    )
    return (
        standard_inputs_present
        or reference_inputs_present
        or powered_inputs_present
        or ctrl_world_inputs_present
    )


def provider_runtime_contract_blockers(
    *,
    provider_bundle_kind: str,
    entrypoint_text: str,
    runner_text: str,
) -> list[str]:
    """Return stable fail-closed runtime-contract blockers for a provider bundle.

    Bundle-specific admission paths call this helper before paid authority is
    consumed. Provider adapters call it again immediately before any mutation.
    """
    if provider_bundle_kind not in PROVIDER_RUNTIME_BUNDLE_KINDS:
        raise ValueError(f"unsupported_provider_bundle_kind:{provider_bundle_kind}")
    if provider_bundle_kind in {"isaac", "adp_simready_isaac"}:
        entrypoint_valid = (
            "write_missing_result" in entrypoint_text
            and "isaac_runner_process_exited_without_runtime_result" in entrypoint_text
            and "blocked_isaac_process_exited_without_result" in entrypoint_text
        )
        runner_valid = "SimulationApp" in runner_text
        runner_blocker = "provider_runner_missing_isaac_simulation_app_smoke"
    elif provider_bundle_kind == "adp_simpler":
        entrypoint_valid = (
            "adp_simpler_runner_failed_without_runtime_result" in entrypoint_text
            and "blocked_adp_simpler_process_exited_without_result" in entrypoint_text
        )
        runner_valid = (
            "adp_simpler_closed_loop_execution.json" in runner_text
            and "simpler_closed_loop_execution.v2" in runner_text
            and "environment_not_policy" in runner_text
            and "observation_frame_manifest" in runner_text
            and "episode_video" in runner_text
            and "environment_step_info.success" in runner_text
        )
        runner_blocker = "provider_runner_missing_adp_simpler_runtime_contract"
    elif provider_bundle_kind == "adp_arena":
        entrypoint_valid = (
            "adp_arena_runner_failed_without_runtime_result" in entrypoint_text
            and "blocked_adp_arena_process_exited_without_result" in entrypoint_text
        )
        runner_valid = (
            "SimulationApp" in runner_text
            and "adp_arena_native_canary.json" in runner_text
            and "pick_and_place_maple_table" in runner_text
            and "zero_action" in runner_text
            and "record_camera_video" in runner_text
            and "provider_zero_required_after_return" in runner_text
        )
        runner_blocker = "provider_runner_missing_adp_arena_runtime_contract"
    elif provider_bundle_kind == "adp009d_isaac":
        entrypoint_valid = (
            "adp009d_worker_failed_without_runtime_result" in entrypoint_text
            and "adp009d_native_microcheck.json" in entrypoint_text
        )
        runner_valid = all(
            token in runner_text
            for token in (
                "adp009d_native_microcheck.json",
                "ARENA_REVISION",
                "ISAAC_LAB_REVISION",
                "provider_zero_required_after_return",
                "candidate_policy_queried",
            )
        )
        runner_blocker = "provider_runner_missing_adp009d_isaac_runtime_contract"
    elif provider_bundle_kind == "adp009d_articulated_native":
        entrypoint_valid = (
            "articulated_native_runner_failed_without_runtime_result"
            in entrypoint_text
            and "adp009d_native_microcheck.json" in entrypoint_text
            and "write_articulated_native_missing_result" in entrypoint_text
        )
        runner_valid = all(
            token in runner_text
            for token in (
                "adp009d_articulated_native_diagnostic.v1",
                "SimulationApp",
                "candidate_policy_queried",
                "provider_zero_required_after_return",
                "locked_joint_readback_degrees",
                "reset_readback_degrees",
            )
        )
        runner_blocker = (
            "provider_runner_missing_adp009d_articulated_native_runtime_contract"
        )
    elif provider_bundle_kind == "adp009d_ovrtx":
        entrypoint_valid = (
            "adp009d_ovrtx_runner_failed_without_runtime_result" in entrypoint_text
            and "adp009d_ovrtx_live_camera_result.json" in entrypoint_text
            and "ovrtx==0.4.0.346409" in entrypoint_text
            and "ovstage==0.1.0.346039" in entrypoint_text
        )
        runner_valid = all(
            token in runner_text
            for token in (
                "adp009d_ovrtx_live_camera_result.json",
                "DistanceToCameraSD",
                "rtpt_warmup_frames",
                "candidate_policy_queried",
                "provider_zero_required_after_return",
            )
        )
        runner_blocker = "provider_runner_missing_adp009d_ovrtx_runtime_contract"
    elif provider_bundle_kind == "adp009d_aura_native":
        entrypoint_valid = (
            "aura_native_runner_failed_without_result" in entrypoint_text
            and "adp009d_aura_native_live_camera_result.json" in entrypoint_text
            and "torch==2.5.1" in entrypoint_text
            and "--no-build-isolation" in entrypoint_text
            and "submodules/simple-knn" in entrypoint_text
            and "opencv-python-headless==4.11.0.86" in entrypoint_text
        )
        runner_valid = all(
            token in runner_text
            for token in (
                "adp009d_aura_native_live_camera_result.json",
                "surf_depth_expected_camera_z_m",
                "source_modified",
                "candidate_policy_queried",
                "provider_zero_required_after_return",
            )
        )
        runner_blocker = "provider_runner_missing_adp009d_aura_native_runtime_contract"
    elif provider_bundle_kind == "adp_content_agents":
        entrypoint_valid = (
            "adp_content_agents_runner_failed_without_runtime_result" in entrypoint_text
            and "blocked_adp_content_agents_process_exited_without_result" in entrypoint_text
            and "provider_archive.py" in entrypoint_text
            and "BLUEPRINT_PROVIDER_BUNDLE_REHEARSAL" in entrypoint_text
            and "python3 -m zipfile -e" not in entrypoint_text
        )
        runner_valid = all(
            token in runner_text
            for token in (
                "adp_content_agents_vast_result.json",
                "material_agent_executed",
                "texture_agent_executed",
                "physics_agent_executed",
                "validation_agent_executed",
                "joint_agent_plan",
                "runtime_input_binding",
            )
        )
        runner_blocker = "provider_runner_missing_adp_content_agents_runtime_contract"
    elif provider_bundle_kind == "adp_joint_agent":
        entrypoint_valid = (
            "adp_joint_agent_runner_failed_without_runtime_result" in entrypoint_text
            and "blocked_adp_joint_agent_process_exited_without_result" in entrypoint_text
            and "apps/ovrtx_rendering_api" in entrypoint_text
            and "gpu_initialized" in entrypoint_text
            and 'export WU_SO_PACKAGE_DIR=' in entrypoint_text
            and "joint_agent_scene_optimizer_core_missing" in entrypoint_text
            and "ovrtx_daemon_probe.log" in entrypoint_text
            and "joint_agent_ovrtx_daemon_probe_failed" in entrypoint_text
            and "provider_archive.py" in entrypoint_text
            and "BLUEPRINT_PROVIDER_BUNDLE_REHEARSAL" in entrypoint_text
            and "python3 -m zipfile -e" not in entrypoint_text
        )
        runner_valid = all(
            token in runner_text
            for token in (
                "adp_joint_agent_result.json",
                "joint_agent_inference_executed",
                "owned_core_publication_executed",
                "review_joint_agent_articulation",
                "UsdPhysics.Joint",
            )
        )
        runner_blocker = "provider_runner_missing_adp_joint_agent_runtime_contract"
    elif provider_bundle_kind == "adp_aura_smoke":
        entrypoint_valid = (
            "adp_aura_smoke_runner_failed_without_runtime_result" in entrypoint_text
            and "blocked_adp_aura_smoke_process_exited_without_result" in entrypoint_text
            and "--no-build-isolation" in entrypoint_text
            and "setuptools==80.9.0" in entrypoint_text
            and "torch==2.5.1" in entrypoint_text
        )
        runner_valid = all(
            token in runner_text
            for token in (
                "adp_aura_author_smoke_result.json",
                "inpaint_init_executed",
                "author_source_modified",
                "published_expected_output_bound",
                "depth_anything3_used",
            )
        )
        runner_blocker = "provider_runner_missing_adp_aura_smoke_runtime_contract"
    elif provider_bundle_kind == "adp_aura_interiorgs":
        entrypoint_valid = (
            "adp_aura_interiorgs_runner_failed_without_runtime_result" in entrypoint_text
            and "blocked_adp_aura_interiorgs_process_exited_without_result" in entrypoint_text
            and "torch==2.5.1" in entrypoint_text
            and "torch==1.8.0" in entrypoint_text
            and "--no-build-isolation" in entrypoint_text
        )
        runner_valid = all(
            token in runner_text
            for token in (
                "adp_aura_interiorgs_result.json",
                "reference_generation_executed",
                "inpaint_finetune_executed",
                "source_modified",
                "hidden_background_truth_available",
                "visual_candidate_only",
                "openclip_offline_cache_verified",
            )
        )
        runner_blocker = "provider_runner_missing_adp_aura_interiorgs_runtime_contract"
    elif provider_bundle_kind == "adp_inpaint360_interiorgs":
        entrypoint_valid = (
            "adp_inpaint360_runner_failed_without_runtime_result" in entrypoint_text
            and "blocked_adp_inpaint360_process_exited_without_result" in entrypoint_text
            and "torch==2.0.0" in entrypoint_text
            and "torch==1.8.0" in entrypoint_text
            and "--no-build-isolation" in entrypoint_text
        )
        runner_valid = all(
            token in runner_text
            for token in (
                "adp_inpaint360_interiorgs_result.json",
                "mask_association_executed",
                "virtual_masks_materialized",
                "lama_color_executed",
                "lama_depth_executed",
                "inpaint_3d_executed",
                "source_modified",
                "rendered_frames_have_no_hidden_background_truth",
            )
        )
        runner_blocker = "provider_runner_missing_adp_inpaint360_runtime_contract"
    elif provider_bundle_kind == "adp_gaussian_excision":
        entrypoint_valid = (
            "gaussian_excision_runner_failed_without_runtime_result"
            in entrypoint_text
            and "blocked_gaussian_excision_process_exited_without_result"
            in entrypoint_text
            and "torch==2.5.1" in entrypoint_text
            and "--no-build-isolation" in entrypoint_text
            and "provider_archive.py" in entrypoint_text
            and "BLUEPRINT_PROVIDER_BUNDLE_REHEARSAL" in entrypoint_text
            and "python3 -m zipfile -e" not in entrypoint_text
        )
        runner_valid = all(
            token in runner_text
            for token in (
                "adp009b_gaussian_excision_result.json",
                "per_view_class_contribution",
                "released_code_executed",
                "heldout_cameras_accessed_for_classification",
                "provider_zero_required_after_return",
                "depth_anything_3_used",
                "runtime_import_preflight",
                "all_imports_attempted",
                "missing_module_names",
            )
        )
        runner_blocker = "provider_runner_missing_adp_gaussian_excision_contract"
    elif provider_bundle_kind == "unitree_unifolm":
        entrypoint_valid = (
            "unitree_unifolm_provider_runner_failed_without_runtime_result" in entrypoint_text
            and "blocked_unitree_unifolm_process_exited_without_result" in entrypoint_text
        )
        runner_valid = (
            "unitree_unifolm_policy_provider_output.json" in runner_text
            and "unitree_unifolm_model_executed" in runner_text
            and "unitree_unifolm_policy_action_command_ran" in runner_text
        )
        runner_blocker = "provider_runner_missing_unitree_unifolm_runtime_contract"
    elif provider_bundle_kind == "unitree_groot_n17_sonic":
        entrypoint_valid = (
            "unitree_groot_n17_sonic_provider_runner_failed_without_runtime_result"
            in entrypoint_text
            and "blocked_unitree_groot_n17_sonic_process_exited_without_result" in entrypoint_text
        )
        runner_valid = (
            "unitree_groot_n17_sonic_policy_provider_output.json" in runner_text
            and "unitree_groot_n17_sonic_model_executed" in runner_text
            and "unitree_groot_n17_sonic_policy_action_command_ran" in runner_text
        )
        runner_blocker = "provider_runner_missing_unitree_groot_n17_sonic_runtime_contract"
    elif provider_bundle_kind == "evaluator":
        entrypoint_valid = (
            "write_missing_result" in entrypoint_text
            and "evaluator_runner_process_exited_without_runtime_result" in entrypoint_text
            and "blocked_evaluator_process_exited_without_result" in entrypoint_text
        )
        runner_valid = (
            "evaluator_runtime_result.json" in runner_text
            and "Cosmos3-Nano" in runner_text
            and "post_unseal_diagnostic_only" in runner_text
        )
        runner_blocker = "provider_runner_missing_evaluator_runtime_contract"
    else:
        entrypoint_valid = (
            "write_missing_result" in entrypoint_text
            and "wam_runner_process_exited_without_runtime_result" in entrypoint_text
            and "blocked_wam_process_exited_without_result" in entrypoint_text
        )
        runner_valid = (
            "wam_runtime_result.json" in runner_text
            and (
                "OSCAR-2B" in runner_text
                or "Cosmos3-Nano" in runner_text
                or "Ctrl-World" in runner_text
            )
            and "action_conditioned_video_rollout_generated" in runner_text
        )
        runner_blocker = "provider_runner_missing_wam_runtime_contract"
    blockers: list[str] = []
    if not entrypoint_valid:
        blockers.append("provider_entrypoint_missing_runtime_result_crash_fallback")
    if not runner_valid:
        blockers.append(runner_blocker)
    return blockers
