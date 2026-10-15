"""Publish the facts a self-serve robot-team run needs from this Pipeline.

A robot team buys a screening run in Blueprint-WebApp. Before it can pay, the
WebApp prepares an exact Task Evaluation Run request (the "payment setup
record"), and three of its facts are known only here:

- the capture root on the executor's partition (``agent_run_executor`` refuses
  any other root);
- the one scenario the scene's episode specs run;
- how many episodes those specs hold (the executor refuses a quote that
  differs).

This module reads those facts from ``pipeline/simulation_automation/
episode_specs.json`` once it exists and posts them over the existing signed
website control transport. It publishes facts, never authority: the WebApp
still requires a verified buyer account, cleared rights and a funded hold, and
the executor re-checks the root and the count before it claims a run. A capture
whose specs are missing, empty, or span more than one scenario publishes
nothing, so no self-serve run can be bought against it.

Publishing is best-effort and never fails the simulation lane: a missing offer
fails closed on the WebApp side (the plan stays unpayable).

Beta L5: when the scene has a team evaluation profile, the offer instead lists
the team evaluation (``team_evaluation_plan``): 200 hidden cells, listed only
while :func:`team_candidate_quick10_registry.offer_eligibility` holds, with the
reason when it does not. It also carries the public evaluation contract a team
builds against: the instruction, the sealed success contract, the variation
families and their bounds, and the policy interfaces. No scene geometry,
footage or site identity is in it.
Otherwise the ordinary offer is published, so a team listing never outlives
its eligibility.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .website_capture_entry import is_website_entry_source

OFFER_SCHEMA_VERSION = "blueprint.agent_execution_offer.v1"
EPISODE_SPECS_RELATIVE = Path("pipeline") / "simulation_automation" / "episode_specs.json"
PUBLICATION_RELATIVE = Path("pipeline") / "agent_execution_offer_publication.json"


def _read_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def build_agent_execution_offer(capture_root: Path, *, policy_execution_profiles: tuple[str, ...] = ()) -> tuple[dict[str, Any] | None, str | None]:
    """The offer for this capture, or ``(None, reason)`` when it cannot be made."""
    root = Path(capture_root).resolve()
    if (len(set(policy_execution_profiles)) != len(policy_execution_profiles)
            or not set(policy_execution_profiles).issubset({"controlled_observation_v1", "onnx_state_mlp_cpu_v1"})
            or ("onnx_state_mlp_cpu_v1" in policy_execution_profiles and "controlled_observation_v1" not in policy_execution_profiles)):
        return None, "policy_execution_profiles_invalid"
    descriptor = _read_object(root / "capture_descriptor.json")
    if descriptor is None:
        return None, "capture_descriptor_missing"
    scene_id = str(descriptor.get("scene_id") or "").strip()
    capture_id = str(descriptor.get("capture_id") or "").strip()
    if not scene_id or not capture_id or root.name != capture_id or root.parent.parent.name != scene_id:
        return None, "capture_root_identity_mismatch"
    specs_path = root / EPISODE_SPECS_RELATIVE
    if not specs_path.is_file():
        return None, "episode_specs_missing"
    raw = specs_path.read_bytes()
    try:
        specs = json.loads(raw)
    except ValueError:
        return None, "episode_specs_invalid"
    episodes = specs.get("episodes") if isinstance(specs, Mapping) else None
    count = specs.get("episode_count") if isinstance(specs, Mapping) else None
    if not isinstance(episodes, list) or not isinstance(count, int) or isinstance(count, bool):
        return None, "episode_specs_invalid"
    if count <= 0 or count != len(episodes):
        return None, "episode_specs_count_invalid"
    scenarios = {
        str(episode.get("scenario_id") or "").strip()
        for episode in episodes
        if isinstance(episode, Mapping)
    }
    scenarios.discard("")
    if len(scenarios) != 1:
        # The executor admits exactly one scenario per run.
        return None, "episode_specs_scenario_not_single"
    return {
        "schema_version": OFFER_SCHEMA_VERSION,
        "scene_id": scene_id,
        "capture_id": capture_id,
        "capture_root": str(root),
        "scenario_id": scenarios.pop(),
        "episode_count": count,
        "episode_specs_sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
        **({"policy_execution_profiles": list(policy_execution_profiles)} if policy_execution_profiles else {}),
    }, None


def _evaluation_contract(facts: Mapping[str, Any], profile: Mapping[str, Any]) -> dict[str, Any]:
    """What a team is asked to do and how it is judged; nothing about the site's geometry."""
    from .droid_https_policy_client import (
        IMAGE_KEYS,
        INTERFACE_ID,
        STATE_KEYS,
        SUPPORTED_ACTION_CHUNK_ROWS,
    )
    from .team_embodiment import BASE_MODES, CONTROL_HZ, FORMS, MAX_ACTION_CHUNK_ROWS
    from .team_embodiment import ROBOT_ID as TEAM_ROBOT_ID
    from .team_embodiment import SCHEMA as EMBODIMENT_SCHEMA
    from .team_evaluation_plan import EPISODES, SCORED_FAMILY_SHARES, variation_bounds
    from .team_policy_https_client import INTERFACE_ID as TEAM_ROBOT_INTERFACE
    from .team_policy_https_client import METADATA_SCHEMA as TEAM_METADATA_SCHEMA

    scene = facts["scene"]
    task_spec = scene.get("task_spec") if isinstance(scene.get("task_spec"), Mapping) else {}
    bounds = variation_bounds(task_spec, task_kind=str(task_spec.get("task_kind") or ""),
                              nominal_dynamic_friction=profile.get("nominal_dynamic_friction"))
    return {
        "schema_version": "blueprint.team_evaluation_contract.v1",
        "task_kind": task_spec.get("task_kind"), "instruction": facts["prompt"],
        "task_success_contract": scene["task_success_contract"],
        "scored": {"episodes": EPISODES["scored"], "cells_published": False,
                   "families": [name for name, _ in SCORED_FAMILY_SHARES],
                   "parameter_bounds": {name: list(value) for name, value in sorted(bounds.items())}},
        # A team brings a policy for Blueprint's DROID Franka, or for its own robot and cameras.
        "policy_interfaces": [
            {"policy_interface_id": INTERFACE_ID, "embodiment_preset_id": scene["embodiment_preset_id"],
             "observation": {"images": list(IMAGE_KEYS), "image_size": [224, 224],
                             "state": list(STATE_KEYS), "prompt": True},
             "action": {"space": "joint_position", "dimension": 8, "control_hz": 15,
                        "chunk_rows": sorted(SUPPORTED_ACTION_CHUNK_ROWS)},
             "request_timeout_seconds": 30, "authentication": "bearer"},
            {"policy_interface_id": TEAM_ROBOT_INTERFACE, "embodiment_preset_id": TEAM_ROBOT_ID,
             "embodiment_schema": EMBODIMENT_SCHEMA, "forms": list(FORMS), "base_modes": list(BASE_MODES),
             "robot_start": {"fixed": "scene_robot_station_declared_reach_must_cover_task",
                             "free": "closest_clear_floor_stance_facing_task_falling_fails_episode"},
             "observation": {"images": "one_png_per_declared_camera",
                             "state": ["joint_position", "joint_velocity", "gripper_position", "base"],
                             "prompt": True},
             "action": {"space": "joint_targets",
                        "row": "one_value_per_declared_joint_position_or_wheel_velocity_then_gripper_closure_0_to_1",
                        "control_hz": list(CONTROL_HZ), "chunk_rows": [1, MAX_ACTION_CHUNK_ROWS]},
             "metadata_schema": TEAM_METADATA_SCHEMA, "task_kinds": ["rigid_pick_place"],
             "request_timeout_seconds": 30, "authentication": "bearer"},
        ],
    }


def build_team_candidate_offer(capture_root: Path) -> tuple[dict[str, Any] | None, str | None]:
    """The team evaluation offer (200 scored cells), or ``(None, reason)`` when it may not be listed."""
    from .team_candidate_quick10_registry import offer_eligibility, profile_for_capture
    from .team_evaluation_plan import EPISODES, PROFILES, PURPOSES, SCORED

    root = Path(capture_root).resolve()
    descriptor = _read_object(root / "capture_descriptor.json") or {}
    scene_id = str(descriptor.get("scene_id") or "").strip()
    capture_id = str(descriptor.get("capture_id") or "").strip()
    if not scene_id or not capture_id or root.name != capture_id or root.parent.parent.name != scene_id:
        return None, "capture_root_identity_mismatch"
    try:
        profile = profile_for_capture(root)
    except (OSError, ValueError) as exc:
        return None, str(exc) or "team_candidate_registry_invalid"
    listing: dict[str, Any] = {}
    facts = None
    for purpose in PURPOSES:
        purpose_facts, blockers = offer_eligibility(profile, purpose)
        facts = facts or purpose_facts
        listing[purpose] = {"profile": PROFILES[purpose], "episodes": EPISODES[purpose],
                            "available": purpose_facts is not None, "reason": None if purpose_facts else blockers[0]}
    if facts is None:
        return None, listing[SCORED]["reason"]
    return {
        "schema_version": OFFER_SCHEMA_VERSION, "scene_id": scene_id, "capture_id": capture_id,
        "capture_root": str(root), "scenario_id": str(profile["scenario_id"]), "episode_count": EPISODES[SCORED],
        "episode_specs_sha256": facts["episode_specs_sha256"],
        "policy_execution_profiles": [row["profile"] for row in listing.values() if row["available"]],
        "team_evaluation": {**listing, "contract": _evaluation_contract(facts, profile)},
    }, None


def publish_agent_execution_offer(capture_root: Path, *, transport=None,
                                  policy_execution_profiles: tuple[str, ...] = ()) -> dict[str, Any]:
    """Post the offer for a website capture. Never raises; records the outcome."""
    root = Path(capture_root).resolve()
    descriptor = _read_object(root / "capture_descriptor.json") or {}
    metadata = descriptor.get("metadata") if isinstance(descriptor.get("metadata"), Mapping) else {}
    request_id = str(descriptor.get("site_submission_id") or metadata.get("site_submission_id") or "").strip()
    if not is_website_entry_source(metadata.get("capture_entry_source")) or not request_id:
        result: dict[str, Any] = {"status": "skipped", "reason": "not_a_website_capture"}
    else:
        try:
            offer, team_refusal = build_team_candidate_offer(root)
        except Exception as exc:  # noqa: BLE001 - publishing must not fail the lane
            offer, team_refusal = None, "team_candidate_offer_unavailable:" + type(exc).__name__
        reason = None
        if offer is None:
            offer, reason = build_agent_execution_offer(root, policy_execution_profiles=policy_execution_profiles)
        if offer is None:
            result = {"status": "skipped", "reason": reason}
        else:
            if transport is None:
                from .website_task_context import website_webapp_request as transport
            try:
                transport(
                    capture_id=offer["capture_id"],
                    operation="agent-execution-offer",
                    payload={"request_id": request_id, "scene_id": offer["scene_id"], "offer": offer},
                )
                result = {"status": "published", "offer": offer}
            except Exception as exc:  # noqa: BLE001 - publishing must not fail the lane
                result = {"status": "failed", "reason": str(exc)[:500], "offer": offer}
        if team_refusal is not None and team_refusal != "team_candidate_profile_missing":
            result["team_candidate_offer_refused"] = team_refusal
    try:
        (root / PUBLICATION_RELATIVE).parent.mkdir(parents=True, exist_ok=True)
        (root / PUBLICATION_RELATIVE).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    except OSError:
        pass
    return result


def main(argv: list[str] | None = None) -> int:
    """Republish a capture's offer, e.g. after its robot-free probe or Quick-10 inputs land."""
    import argparse

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--capture-root", required=True, type=Path)
    args = parser.parse_args(argv)
    result = publish_agent_execution_offer(args.capture_root)
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("status") == "published" else 2


if __name__ == "__main__":
    raise SystemExit(main())
