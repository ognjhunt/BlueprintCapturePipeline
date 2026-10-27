"""Native family qualification and retained evidence in the canary envelope."""
from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .policy_canary_media_integrity import bound_media_artifact as _bound_media_artifact
from .policy_canary_worker_evidence import _write_episode_json_artifact
from .task_family_policy_worker_execution import execute_family_policy_episode, is_task_family_scene_plan


def _digest(value: Any) -> str:
    return canonical_digest({"value": value})


def beta_result_binding(inputs: Mapping[str, Any]) -> dict[str, Any]:
    return {key: inputs[key] for key in ("beta_screening_admission", "scene_revision_digest", "plan_digest",
        "configuration_digest", "pre_policy_site_qualification_required", "family_static_readiness_digest") if key in inputs}


def family_render_backend(plan: Mapping[str, Any]) -> dict[str, Any]:
    from blueprint_pipeline.task_family_native_contracts import validate_task_family_native_runtime_contract
    native = validate_task_family_native_runtime_contract(plan["native_runtime_contract"])
    backend = {"schema_version": "task_family_native_render_binding.v1",
               "backend": "qualified_native_engine_renderer",
               "runtime_identity_digest": native["runtime_identity_digest"],
               "native_runtime_contract_digest": native["contract_digest"],
               "scene_appearance_assets": [row for row in native["assets"]
                   if row.get("semantic_role") == "scene_appearance"],
               "native_renderer_readback_required": True}
    backend["receipt_digest"] = canonical_digest(backend, digest_field="receipt_digest")
    return backend


def qualify_family_policy_scene(*, plan: Mapping[str, Any], specs: Mapping[str, Mapping[str, Any]],
                                packet_root: Path, output_root: Path) -> dict[str, Any]:
    from .task_family_native_controls import qualify_task_family_site
    from .task_evaluation_control_stage_policy import require_controls_enabled

    require_controls_enabled()
    bounds = {(spec["maximum_action_steps"], spec["control_frequency_hz"]) for spec in specs.values()}
    if len(bounds) != 1:
        raise ValueError("task_family_site_controls_candidate_bounds_mismatch")
    steps, hz = next(iter(bounds))
    qualification = qualify_task_family_site(plan["native_runtime_contract"],
        qualified_runtime_selection=plan["task_family_runtime_selection"],
        runtime_catalog_path=Path(os.environ["BLUEPRINT_TASK_FAMILY_RUNTIME_CATALOG_PATH"]),
        asset_root=packet_root, output_dir=output_root / "site_qualification",
        maximum_steps=steps, control_frequency_hz=hz)
    return {"status": qualification["status"],
            "policy_observation_integrity_passed": qualification["status"] == "passed",
            "authority": "native_robot_free_and_independent_controls", "site_qualification": qualification}


def _run_family_canary_episode(
    *, scene_plan: Mapping[str, Any], policy: Mapping[str, Any], context: Mapping[str, Any],
    authority: Mapping[str, Any], inputs: Mapping[str, Any], packet_root: Path,
    output_root: Path, progress: dict[str, Any],
    site_qualification: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize retained family evidence into the existing session envelope."""
    from blueprint_pipeline.native_task_arena_policy_worker import _PolicyQueryTracker

    tracker = _PolicyQueryTracker(policy["client"])
    episode_id = f"{authority['run_id']}--{context['cell_id']}--{context['candidate_id']}"
    episode = execute_family_policy_episode(scene_plan=scene_plan, spec=policy["spec"],
        policy_client=tracker, packet_root=packet_root, output_dir=output_root / "episodes",
        episode_id=episode_id, progress=progress)
    if episode["status"] != "completed":
        _write_episode_json_artifact(output_root, episode_id=episode_id,
            role="native_family_failure_receipt", value=episode)
        raise RuntimeError("task_family_native_episode_invalid:" + str(episode.get("failure")))
    if site_qualification is not None:
        if (site_qualification.get("status") != "passed"
                or site_qualification.get("native_provenance") != episode["native_family_receipt"]["native_provenance"]
                or site_qualification.get("task_success_contract_digest") != episode["score"].get("contract_digest")):
            raise ValueError("task_family_episode_site_qualification_mismatch")
        episode["native_site_qualification"] = dict(site_qualification)
    media_root = output_root / "episodes"
    artifacts = episode["media_artifacts"]
    media = {role: _bound_media_artifact(output_root, media_root=media_root, artifacts=artifacts,
                role=role, role_match=predicate)
             for role, predicate in (
                 ("frame_manifest", lambda name: "frame_manifest" in name),
                 ("review_video", lambda name: "video" in name))}
    if any(value is None for value in media.values()):
        raise ValueError("task_family_completed_episode_media_identity_invalid")
    for artifact in artifacts:
        if _bound_media_artifact(output_root, media_root=media_root, artifacts=[artifact],
                role=str(artifact.get("role") or ""), role_match=lambda _name: True) is None:
            raise ValueError("task_family_completed_episode_media_identity_invalid")
    motion = episode["motion_evidence"]
    states = episode["state_trace"]
    contacts = [{"simulation_time_seconds": state["simulation_time_seconds"],
                 "asset_contacts": {key: {name: value for name, value in asset.items()
                    if "contact" in name or "force" in name}
                    for key, asset in state["asset_states"].items()}}
                for state in states]
    trajectories = [{"simulation_time_seconds": state["simulation_time_seconds"],
                     "asset_states": state["asset_states"]} for state in states]
    episode["contact_force_evidence"] = contacts
    episode["task_object_trajectory"] = trajectories
    evidence_values = {"episode_receipt": episode, "reset_state": states[0],
        "policy_query_receipt": {"candidate_policy_queried": tracker.candidate_policy_queried,
                                 "queries": episode["queries"]},
        "action_sequence": episode["commanded_actions"], "action_delivery_readback": motion,
        "state_trace": states, "contact_force_trace": contacts,
        "task_object_trajectory": trajectories, "score_receipt": episode["score"]}
    if site_qualification is not None:
        evidence_values["native_site_qualification"] = dict(site_qualification)
    evidence = {role: _write_episode_json_artifact(output_root, episode_id=episode_id, role=role, value=value)
                for role, value in evidence_values.items()}
    evidence["episode_receipt"]["media_root_relative"] = "episodes"
    evidence.update(media)
    return {"status": "completed", "candidate_policy_queried": tracker.candidate_policy_queried,
        "actions_reached_robot": motion["actions_reached_robot"], "arm_moved": False,
        "policy_state_changed": motion["policy_state_changed"],
        "checkpoint_digest": policy["checkpoint_digest"],
        "runtime_identity_digest": policy["runtime_identity_digest"],
        "lossless_frame_manifest_digest": media["frame_manifest"]["sha256"],
        "review_video_digest": media["review_video"]["sha256"],
        "returned_action_sequence_digest": _digest(episode["commanded_actions"]),
        "action_delivery_readback_digest": _digest(motion), "state_trace_digest": _digest(states),
        "contact_force_digest": _digest(contacts), "task_object_trajectory_digest": _digest(trajectories),
        "deterministic_score_digest": _digest(episode["score"]),
        "scoring_authority": "deterministic_simulator_state", "episode": episode,
        "episode_environment": episode["episode_environment"],
        "telemetry": episode["native_measurements"],
        "telemetry_digest": _digest(episode["native_measurements"]),
        "code_identity_digest": authority.get("source_commit_digest") or authority["authority_digest"],
        "container_identity_digest": policy["runtime_identity_digest"],
        "scene_revision_digest": inputs.get("scene_revision_digest") or inputs["configuration_digest"],
        "scoring_version_digest": _digest("task_family_scoring.v1"), "evidence_artifacts": evidence}



def _construction_lineage_mode(*, inputs: Mapping[str, Any], base_scene_plan: Mapping[str, Any],
                               construction: Mapping[str, Any]) -> str:
    if is_task_family_scene_plan(base_scene_plan):
        from blueprint_pipeline.task_family_policy_execution import family_policy_execution_binding
        family_policy_execution_binding(base_scene_plan)
        if inputs.get("task_success_contract") != base_scene_plan["task_spec"]["task_success_contract"]:
            raise ValueError("task_family_site_qualification_task_binding_mismatch")
        return "family_native_prepolicy_site_qualification_required"
    from blueprint_pipeline.policy_canary_scene_resolution import construction_lineage_mode

    return construction_lineage_mode(
        inputs=inputs, base_scene_plan=base_scene_plan, construction=construction,
        admitted_run_kinds=("internal_policy_canary", "team_candidate_evaluation", "blueprint_beta_screening"),
    )

