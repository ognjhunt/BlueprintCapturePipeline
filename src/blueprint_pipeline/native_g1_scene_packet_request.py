"""Author a G1 packet request from one verified captured-site task packet.

The source packet supplies scene assets and task identity. A team must author
the G1 stance, camera, task parameters, and development scenario explicitly.
It can stage a new native packet, but does not launch Isaac or claim an episode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from copy import deepcopy
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from .adp_task_scoring import validate_rigid_task_success_contract
from .decision_evidence_contracts import canonical_digest
from .native_g1_arena_robot_configuration import (
    G1_USD_SHA256,
    G1_USD_SIZE_BYTES,
    build_pinned_g1_arena_robot_configuration,
)
from .native_g1_navigation_goal import validate_g1_navigation_goal
from .native_task_arena_bundle import verify_native_task_arena_packet
from .native_task_arena_packet import (
    REQUEST_SCHEMA_VERSION,
    _asset_source,
    _validated_scenario_context,
    materialize_native_task_arena_packet,
    validate_native_task_arena_packet_request,
)
from .native_task_robot_contact_topology import _asset_rigid_body_paths
from .native_task_runtime_contract import _camera_rows
from .task_evaluation_g1_catalog import G1_PRESET_ID
from .task_evaluation_packet_planning_setup import (
    SETUP_SCHEMA as PACKET_PLANNING_SETUP_SCHEMA,
    validate_packet_planning_setup,
    validate_packet_policy_pair_choice,
)
from .task_evaluation_policy_pair_choice import validate_policy_pair_choice
from .task_evaluation_policy_canary_setup import validate_policy_canary_setup


SCHEMA = "native_g1_scene_packet_authoring.v1"
FIELDS = frozenset(
    {
        "schema_version",
        "claim_ceiling",
        "source_packet_receipt_digest",
        "pair_choice_digest",
        "base_pose_world",
        "task_hand",
        "cameras",
        "task_spec",
        "scenario",
        "authoring_digest",
    }
)
OPTIONAL_FIELDS = frozenset({"physics_frequency_hz"})
ROBOT_TASK_FIELDS = frozenset(
    {
        "robot_workspace_position_bounds_world_m",
        "release_gripper_width_min_m",
        "action_bounds_m_per_step",
        "control_frequency_hz",
        "maximum_action_steps",
        "maximum_episode_seconds",
        "retreat_clearance_m",
    }
)


def _validated_source_pair(
    *,
    setup: Mapping[str, Any],
    choice: Mapping[str, Any],
    receipt: Mapping[str, Any],
    source: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Accept a published choice or an exact retained-packet planning choice."""

    if setup.get("schema_version") == PACKET_PLANNING_SETUP_SCHEMA:
        published = validate_packet_planning_setup(setup)
        selected = validate_packet_policy_pair_choice(choice, setup=published)
        if (
            published["source_packet_receipt_digest"] != receipt["receipt_digest"]
            or published["source_packet_request_digest"] != receipt["request_digest"]
            or published["source_scene_plan_digest"] != receipt["arena_scene_plan_digest"]
            or published["scene_id"] != source.get("scene_id")
            or published["task_id"] != source.get("task_id")
            or published["source_declared_task_success_contract_digest"]
            != source.get("task_spec", {}).get("task_success_contract_digest")
        ):
            raise ValueError("g1_scene_request_source_task_mismatch")
    else:
        published = validate_policy_canary_setup(setup)
        selected = validate_policy_pair_choice(choice, setup=published)
    if (
        selected["robot_preset_id"] != G1_PRESET_ID
        or source.get("request_digest") != receipt.get("request_digest")
        or source.get("task_id") != published["task_success_contract"]["scope"]["task_id"]
        or source.get("task_spec", {}).get("task_success_contract")
        != published["task_success_contract"]
        or source.get("task_spec", {}).get("task_kind") != "rigid_pick_place"
    ):
        raise ValueError("g1_scene_request_source_task_mismatch")
    return published, selected


def author_g1_scene_packet_request(
    *,
    source_packet_dir: Path,
    evidence_root: Path,
    g1_usd_path: Path,
    setup: Mapping[str, Any],
    choice: Mapping[str, Any],
    authoring: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a new request bound to the exact source packet and shared choice."""

    source_root, receipt, _ = verify_native_task_arena_packet(source_packet_dir)
    source = json.loads((source_root / (REQUEST_SCHEMA_VERSION + ".json")).read_text())
    published, selected = _validated_source_pair(
        setup=setup,
        choice=choice,
        receipt=receipt,
        source=source,
    )
    authored = dict(authoring)
    if (
        not FIELDS <= set(authored) <= FIELDS | OPTIONAL_FIELDS
        or authored.get("schema_version") != SCHEMA
        or authored.get("claim_ceiling") != "development_only"
        or authored.get("source_packet_receipt_digest") != receipt["receipt_digest"]
        or authored.get("pair_choice_digest") != selected["choice_digest"]
        or authored.get("authoring_digest")
        != canonical_digest(authored, digest_field="authoring_digest")
    ):
        raise ValueError("g1_scene_request_authoring_binding_invalid")
    task_spec = authored["task_spec"]
    if not isinstance(task_spec, Mapping):
        raise ValueError("g1_scene_request_task_spec_invalid")
    physics_frequency = authored.get("physics_frequency_hz", source["physics_frequency_hz"])
    control_frequency = task_spec.get("control_frequency_hz")
    if (
        isinstance(physics_frequency, bool)
        or not isinstance(physics_frequency, (int, float))
        or not math.isfinite(physics_frequency)
        or physics_frequency <= 0
    ):
        raise ValueError("g1_scene_request_physics_control_cadence_invalid")
    if (
        isinstance(control_frequency, bool)
        or not isinstance(control_frequency, (int, float))
        or not math.isfinite(control_frequency)
        or control_frequency <= 0
        or physics_frequency / control_frequency < 1
        or not math.isclose(
            physics_frequency / control_frequency,
            round(physics_frequency / control_frequency),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        raise ValueError("g1_scene_request_physics_control_cadence_invalid")
    task_contract = task_spec.get("task_success_contract")
    validate_rigid_task_success_contract(
        task_contract,
        expected_site_id=published["task_success_contract"]["scope"]["site_id"],
        expected_task_id=source["task_id"],
    )
    expected_fields = set(source["task_spec"])
    if selected["objective_id"] == "g1_navigation_goal":
        expected_fields.add("g1_navigation_goal")
    packet_planning = published["schema_version"] == PACKET_PLANNING_SETUP_SCHEMA
    allowed_changed_fields = ROBOT_TASK_FIELDS | (
        {"task_success_contract_digest"} if packet_planning else set()
    )
    if (
        task_spec.get("task_kind") != "rigid_pick_place"
        or task_contract != published["task_success_contract"]
        or task_spec.get("task_success_contract_digest")
        != published["task_success_contract_digest"]
        or set(task_spec) != expected_fields
        or any(
            task_spec[key] != source["task_spec"][key]
            for key in source["task_spec"]
            if key not in allowed_changed_fields
        )
    ):
        raise ValueError("g1_scene_request_task_contract_mismatch")
    if selected["objective_id"] == "g1_navigation_goal":
        validate_g1_navigation_goal(task_spec)
    scenario = _validated_scenario_context(authored["scenario"])
    context = scenario["context_document"]
    if (
        context.get("partition") != "development"
        or context.get("program_id") != "arm-decision-proof-v1"
        or context.get("learned_policy_outcomes_consulted") is not False
    ):
        raise ValueError("g1_scene_request_scenario_not_development")

    evidence = evidence_root.resolve()
    robot = build_pinned_g1_arena_robot_configuration(
        asset=g1_usd_path,
        evidence_root=evidence,
        base_pose_world=authored["base_pose_world"],
        task_hand=authored["task_hand"],
    )
    cameras = authored["cameras"]
    if not isinstance(cameras, list):
        raise ValueError("g1_scene_request_cameras_invalid")
    errors: list[str] = []
    normalized = _camera_rows(cameras, robot_id="unitree_g1", errors=errors)
    if errors or {row["role"] for row in normalized} != {"head", "overview"}:
        raise ValueError("g1_scene_request_cameras_invalid:" + ",".join(errors))
    head = next(row for row in normalized if row["role"] == "head")
    if (
        head["intrinsics"]["width"] != 640
        or head["intrinsics"]["height"] != 480
        or head["parent_prim_path"] not in _asset_rigid_body_paths(g1_usd_path)
    ):
        raise ValueError("g1_scene_request_head_camera_invalid")

    by_role = {row["semantic_role"]: row for row in receipt["source_bindings"]}
    assets = []
    if len(by_role) != len(source["assets"]):
        raise ValueError("g1_scene_request_source_assets_invalid")
    for row in source["assets"]:
        binding = by_role.get(row["semantic_role"])
        if binding is None:
            raise ValueError("g1_scene_request_source_assets_invalid")
        copied = deepcopy(row)
        copied["source"] = {
            "root": "evidence",
            "relative_path": "scene/" + row["filename"],
            "size_bytes": binding["staged_size_bytes"],
            "sha256": binding["staged_sha256"],
        }
        _asset_source(copied, evidence_root=evidence)
        assets.append(copied)
    retained = {
        key: source[key]
        for key in (
            "scene_id",
            "task_id",
            "task_joint_bindings",
            "task_state_binding",
            "appearance_variant",
        )
        if key in source
    }
    request = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        **retained,
        "assets": assets,
        "robot_configuration": robot,
        "robot_base_pose_world": robot["base_pose_world"],
        "robot_joint_reset_positions_rad": robot["joint_reset_positions_rad"],
        "cameras": normalized,
        "task_spec": dict(task_spec),
        "scenario": scenario,
        "physics_frequency_hz": physics_frequency,
        "g1_scene_derivation": {
            "schema_version": "native_g1_scene_packet_derivation.v1",
            "claim_ceiling": "development_only",
            "source_packet_receipt_digest": receipt["receipt_digest"],
            "source_scene_plan_digest": receipt["arena_scene_plan_digest"],
            "setup_digest": published["setup_digest"],
            "pair_choice_digest": selected["choice_digest"],
            "authoring_digest": authored["authoring_digest"],
            "source_physics_frequency_hz": source["physics_frequency_hz"],
            "selected_physics_frequency_hz": physics_frequency,
            **(
                {
                    "source_declared_task_success_contract_digest": published[
                        "source_declared_task_success_contract_digest"
                    ],
                    "task_contract_digest_corrected_from_embedded_confirmed_contract": (
                        published["source_declared_task_success_contract_digest"]
                        != published["task_success_contract_digest"]
                    ),
                }
                if packet_planning
                else {}
            ),
        },
    }
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    return validate_native_task_arena_packet_request(request)


def _copy_preserving_source(source: Path, destination: Path) -> None:
    """Use APFS copy-on-write when available; keep separate inodes everywhere."""

    if sys.platform == "darwin":
        result = subprocess.run(
            ["cp", "-c", str(source), str(destination)],
            check=False,
            capture_output=True,
        )
        if result.returncode == 0:
            return
    shutil.copyfile(source, destination)


def prepare_g1_scene_packet_request(
    *,
    source_packet_dir: Path,
    g1_usd_path: Path,
    setup: Mapping[str, Any],
    choice: Mapping[str, Any],
    authoring: Mapping[str, Any],
    output_dir: Path,
    materialize_packet: bool = False,
) -> dict[str, Any]:
    """Stage verified retained scene bytes beside one G1 packet request."""

    if not isinstance(authoring, Mapping):
        raise ValueError("g1_scene_request_authoring_invalid")
    source_root, receipt, _ = verify_native_task_arena_packet(source_packet_dir)
    source = json.loads((source_root / (REQUEST_SCHEMA_VERSION + ".json")).read_text())
    _, selected = _validated_source_pair(
        setup=setup,
        choice=choice,
        receipt=receipt,
        source=source,
    )
    if (
        selected["robot_preset_id"] != G1_PRESET_ID
        or authoring.get("source_packet_receipt_digest") != receipt["receipt_digest"]
        or authoring.get("pair_choice_digest") != selected["choice_digest"]
    ):
        raise ValueError("g1_scene_request_authoring_binding_invalid")
    if (
        not output_dir.is_absolute()
        or output_dir.exists()
        or output_dir.is_symlink()
        or output_dir.resolve().is_relative_to(source_root)
        or source_root.is_relative_to(output_dir.resolve())
    ):
        raise ValueError("g1_scene_request_output_directory_invalid")
    if (
        g1_usd_path.is_symlink()
        or not g1_usd_path.is_file()
        or g1_usd_path.stat().st_size != G1_USD_SIZE_BYTES
    ):
        raise ValueError("g1_scene_request_robot_asset_invalid")
    digest = hashlib.sha256(g1_usd_path.read_bytes()).hexdigest()
    if digest != G1_USD_SHA256:
        raise ValueError("g1_scene_request_robot_asset_invalid")
    by_role = {row["semantic_role"]: row for row in receipt["source_bindings"]}
    if len(by_role) != len(source["assets"]) or len(source["assets"]) != len(
        {row["semantic_role"] for row in source["assets"]}
    ):
        raise ValueError("g1_scene_request_source_assets_invalid")
    for row in source["assets"]:
        filename = str(row.get("filename") or "")
        if (
            not filename
            or PurePosixPath(filename).name != filename
            or row["semantic_role"] not in by_role
        ):
            raise ValueError("g1_scene_request_source_assets_invalid")
    if len({row["filename"] for row in source["assets"]}) != len(source["assets"]):
        raise ValueError("g1_scene_request_source_assets_invalid")
    output_dir.mkdir(parents=True)
    evidence = output_dir / "evidence"
    scene_dir = evidence / "scene"
    robot_dir = evidence / "robot"
    scene_dir.mkdir(parents=True)
    robot_dir.mkdir()
    for row in source["assets"]:
        binding = by_role[row["semantic_role"]]
        original = source_root / binding["staged_relative_path"]
        staged = scene_dir / row["filename"]
        _copy_preserving_source(original, staged)
    robot_path = robot_dir / g1_usd_path.name
    _copy_preserving_source(g1_usd_path, robot_path)
    request = author_g1_scene_packet_request(
        source_packet_dir=source_root,
        evidence_root=evidence,
        g1_usd_path=robot_path,
        setup=setup,
        choice=choice,
        authoring=authoring,
    )
    (output_dir / (REQUEST_SCHEMA_VERSION + ".json")).write_text(
        json.dumps(request, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / (SCHEMA + ".json")).write_text(
        json.dumps(authoring, indent=2, sort_keys=True) + "\n"
    )
    result = {
        "request_digest": request["request_digest"],
        "request_path": str(output_dir / (REQUEST_SCHEMA_VERSION + ".json")),
        "evidence_root": str(evidence),
        "source_packet_receipt_digest": receipt["receipt_digest"],
        "status": "development_packet_request_staged",
    }
    if materialize_packet:
        packet_dir = output_dir / "packet"
        packet_receipt = materialize_native_task_arena_packet(
            request=request,
            evidence_root=evidence,
            output_dir=packet_dir,
            link_sources_within=output_dir,
        )
        result.update(
            status="development_packet_materialized",
            packet_path=str(packet_dir),
            packet_receipt_digest=packet_receipt["receipt_digest"],
        )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-packet", type=Path, required=True)
    parser.add_argument("--g1-usd", type=Path, required=True)
    parser.add_argument("--setup", type=Path, required=True)
    parser.add_argument("--choice", type=Path, required=True)
    parser.add_argument("--authoring", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--materialize-packet", action="store_true")
    args = parser.parse_args(argv)
    result = prepare_g1_scene_packet_request(
        source_packet_dir=args.source_packet,
        g1_usd_path=args.g1_usd,
        setup=json.loads(args.setup.read_text()),
        choice=json.loads(args.choice.read_text()),
        authoring=json.loads(args.authoring.read_text()),
        output_dir=args.output_dir,
        materialize_packet=args.materialize_packet,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
