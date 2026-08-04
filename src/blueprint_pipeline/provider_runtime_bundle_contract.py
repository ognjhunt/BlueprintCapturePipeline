"""Shared static runtime contracts for paid provider bundles."""

from __future__ import annotations

import json
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path


PROVIDER_RUNTIME_BUNDLE_KINDS = (
    "isaac",
    "wam",
    "evaluator",
    "unitree_unifolm",
    "unitree_groot_n17_sonic",
    "adp_simpler",
    "adp_arena",
    "adp_content_agents",
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
    if provider_bundle_kind == "isaac":
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
    elif provider_bundle_kind == "adp_content_agents":
        entrypoint_valid = (
            "adp_content_agents_runner_failed_without_runtime_result" in entrypoint_text
            and "blocked_adp_content_agents_process_exited_without_result" in entrypoint_text
        )
        runner_valid = all(
            token in runner_text
            for token in (
                "adp_content_agents_vast_result.json",
                "material_agent_executed",
                "texture_agent_executed",
                "physics_agent_executed",
                "validation_agent_executed",
                "joint_agent_inapplicable_single_rigid_body",
            )
        )
        runner_blocker = "provider_runner_missing_adp_content_agents_runtime_contract"
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
