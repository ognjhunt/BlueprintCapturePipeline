"""Scene-option research and scenario packet generation for MuJoCo-first G1 evals."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
from defusedxml import ElementTree as ET
from defusedxml.common import DefusedXmlException
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from .common import ensure_dir, utc_now_iso, write_json, write_text
from .local_capture import resolve_local_capture_context
from .scenario_variation_instantiator import (
    SCENARIO_VARIATION_NAMES,
    build_scenario_variation_instances,
)


MUJOCO_SCENE_ASSET_RESEARCH_SCHEMA_VERSION = "mujoco_scene_asset_research.v1"
MUJOCO_SCENE_SCENARIO_PACKET_SCHEMA_VERSION = "mujoco_scene_scenario_packet.v1"
MUJOCO_SCENE_RECORDING_PLAN_SCHEMA_VERSION = "mujoco_scene_recording_plan.v1"
MUJOCO_EXTERNAL_SCENE_CAPTURE_DESCRIPTOR_SCHEMA_VERSION = (
    "external_mujoco_scene_capture_descriptor.v1"
)
DEFAULT_SCENE_ID = "aws_robomaker_small_warehouse_world"
DEFAULT_CAPTURE_ID = "external-scene-packet-v1"
DEFAULT_SCENARIO_COUNT = 500

CLAIM_BOUNDARY: Dict[str, Any] = {
    "artifact_purpose": "external_scene_scenario_packet_for_mujoco_first_eval",
    "external_scene_asset_not_raw_blueprint_capture": True,
    "raw_capture_evidence_authoritative": False,
    "remote_asset_downloads_performed_by_packet_builder": False,
    "conversion_performed": False,
    "simulators_run": False,
    "simulator_execution_proven": False,
    "rank_fidelity_result_proven": False,
    "robot_policy_execution_proven": False,
    "physics_contact_validated": False,
    "non_ranking_operational_claim_validated": False,
    "public_claim_upgrade_allowed": False,
}

REQUIRED_RECORDING_VIEWS = (
    "sim_robot_follow_pov",
    "overview",
    "side",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _string(value: Any) -> str:
    return str(value or "").strip()


def _safe_id(value: Any, *, fallback: str = "item") -> str:
    text = _string(value) or fallback
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return cleaned or fallback


def _sha_payload(payload: Mapping[str, Any]) -> str:
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _scene_option(
    *,
    rank: int,
    asset_id: str,
    name: str,
    source_url: str,
    direct_asset_url: str,
    license_id: str,
    commercial_use_status: str,
    formats: Sequence[str],
    texture_material_quality: str,
    mujoco_conversion_difficulty: str,
    collision_setup: str,
    scene_type: str,
    navigation_and_task_fit: str,
    supports_500_episodes: str,
    risks: Sequence[str],
    recommended_next_action: str,
    immediate_mujoco_candidate: bool = False,
    future_isaac_candidate: bool = False,
    avoid_or_research_only: bool = False,
) -> Dict[str, Any]:
    return {
        "rank": rank,
        "asset_id": asset_id,
        "name": name,
        "source_url": source_url,
        "direct_asset_url": direct_asset_url,
        "license": {
            "id": license_id,
            "commercial_use_status": commercial_use_status,
            "fail_closed_status": "allowed_with_review"
            if "allowed" in commercial_use_status
            else "blocked_or_research_only",
        },
        "formats": list(formats),
        "texture_material_quality": texture_material_quality,
        "mujoco_conversion_difficulty": mujoco_conversion_difficulty,
        "collision_setup": collision_setup,
        "scene_type": scene_type,
        "navigation_and_task_fit": navigation_and_task_fit,
        "supports_500_episodes": supports_500_episodes,
        "risks": list(risks),
        "recommended_next_action": recommended_next_action,
        "immediate_mujoco_candidate": immediate_mujoco_candidate,
        "future_isaac_candidate": future_isaac_candidate,
        "avoid_or_research_only": avoid_or_research_only,
    }


def scene_asset_options() -> List[Dict[str, Any]]:
    """Return the ranked external scene options from the primary-source review."""

    return [
        _scene_option(
            rank=1,
            asset_id="aws_robomaker_small_warehouse_world",
            name="AWS RoboMaker Small Warehouse World",
            source_url="https://github.com/aws-robotics/aws-robomaker-small-warehouse-world",
            direct_asset_url=(
                "git clone https://github.com/aws-robotics/"
                "aws-robomaker-small-warehouse-world.git"
            ),
            license_id="MIT-0",
            commercial_use_status="commercial_use_allowed_no_attribution_required",
            formats=["Gazebo .world", "SDF models", "DAE meshes", "PNG textures"],
            texture_material_quality="textured warehouse shelves, pallets, buckets, floor, lamps",
            mujoco_conversion_difficulty="low_medium",
            collision_setup="SDF collision meshes plus simplified MuJoCo boxes recommended",
            scene_type="warehouse_logistics",
            navigation_and_task_fit=(
                "Aisles, shelves, pallet jack, bins, clutter, loading/staging zones, "
                "blocked-path and inspection tasks."
            ),
            supports_500_episodes=(
                "Yes: combine 10 site-specific scenarios with route seeds, blocked aisles, "
                "lighting, occlusion, label, and approach-angle mutations."
            ),
            risks=[
                "DAE-to-OBJ or DAE-to-MJCF conversion required before textured MuJoCo rendering",
                "Use as first scene, not as the only benchmark environment",
            ],
            recommended_next_action="Use first for MuJoCo scenario packet and conversion spike.",
            immediate_mujoco_candidate=True,
        ),
        _scene_option(
            rank=2,
            asset_id="manchester_nuclear_gazebo_assets",
            name="University of Manchester 3D Simulation Assets for Nuclear Environments",
            source_url=(
                "https://figshare.manchester.ac.uk/articles/code/"
                "3D_Simulation_Assets_for_Nuclear_Environments_Gazebo_Format_/25224974"
            ),
            direct_asset_url=(
                "https://figshare.manchester.ac.uk/ndownloader/articles/25224974/versions/1"
            ),
            license_id="CC-BY-4.0",
            commercial_use_status="commercial_use_allowed_with_attribution",
            formats=["Gazebo .sdf", "meshes", "textures"],
            texture_material_quality="rich industrial objects, clutter, hazards, dials, gauges",
            mujoco_conversion_difficulty="medium",
            collision_setup="derived mesh collider or manual primitive simplification",
            scene_type="industrial_nuclear",
            navigation_and_task_fit=(
                "Inspection routes, clutter navigation, ramps, debris, trenches, barriers, "
                "hazard avoidance, and gauge/dial approach tasks."
            ),
            supports_500_episodes=(
                "Yes: multiple rooms/environments and industrial hazard/task mutations."
            ),
            risks=["2.62 GB download", "attribution required", "subasset audit still required"],
            recommended_next_action="Use second after AWS warehouse conversion path is stable.",
            immediate_mujoco_candidate=True,
        ),
        _scene_option(
            rank=3,
            asset_id="aws_robomaker_bookstore_world",
            name="AWS RoboMaker Bookstore World",
            source_url="https://github.com/aws-robotics/aws-robomaker-bookstore-world",
            direct_asset_url=(
                "git clone https://github.com/aws-robotics/aws-robomaker-bookstore-world.git"
            ),
            license_id="MIT-0",
            commercial_use_status="commercial_use_allowed_no_attribution_required",
            formats=["Gazebo .world", "SDF models", "DAE meshes", "textures"],
            texture_material_quality="retail shelves and tables with practical textures",
            mujoco_conversion_difficulty="low_medium",
            collision_setup="simplified shelf/table/counter/wall colliders",
            scene_type="retail_bookstore",
            navigation_and_task_fit="Shelf approach, aisle traversal, counter approach, label checks.",
            supports_500_episodes="Yes with route, shelf target, obstruction, and lighting seeds.",
            risks=["Smaller and more orderly than industrial scenes"],
            recommended_next_action="Use as the first retail/commercial aisle follow-up.",
            immediate_mujoco_candidate=True,
        ),
        _scene_option(
            rank=4,
            asset_id="replicacad_interactive_and_baked",
            name="ReplicaCAD Interactive and Baked Lighting",
            source_url="https://aihabitat.org/datasets/replica_cad/",
            direct_asset_url=(
                "https://huggingface.co/datasets/ai-habitat/ReplicaCAD_dataset and "
                "https://huggingface.co/datasets/ai-habitat/ReplicaCAD_baked_lighting"
            ),
            license_id="CC-BY-4.0",
            commercial_use_status="commercial_use_allowed_with_attribution",
            formats=["Habitat configs", "GLB/stage assets", "URDF articulated assets", "navmeshes"],
            texture_material_quality="high quality PBR and baked-lighting apartment assets",
            mujoco_conversion_difficulty="medium_high",
            collision_setup="convex collision geometry and navmeshes help, but MJCF extraction needed",
            scene_type="residential_apartment",
            navigation_and_task_fit="Doorway, room, furniture, clutter, and household station tasks.",
            supports_500_episodes="Yes: 84 rearrangements plus object and lighting variants.",
            risks=["Habitat-native conversion path is more involved than Gazebo-to-MJCF"],
            recommended_next_action="Keep for higher-fidelity cross-backend follow-up.",
            future_isaac_candidate=True,
        ),
        _scene_option(
            rank=5,
            asset_id="aws_robomaker_hospital_world",
            name="AWS RoboMaker Hospital World",
            source_url="https://github.com/aws-robotics/aws-robomaker-hospital-world",
            direct_asset_url=(
                "git clone https://github.com/aws-robotics/aws-robomaker-hospital-world.git"
            ),
            license_id="MIT-0 repo plus external Fuel dependencies",
            commercial_use_status="repo_allowed_but_subasset_license_audit_required",
            formats=["Gazebo .world", "SDF models", "Fuel model dependencies"],
            texture_material_quality="hospital corridors, nurse station, beds, carts, equipment",
            mujoco_conversion_difficulty="medium",
            collision_setup="manual simplification for medical equipment and corridors",
            scene_type="hospital",
            navigation_and_task_fit="Hallway routing, equipment avoidance, supply-room tasks.",
            supports_500_episodes="Yes, but only after dependency and license audit.",
            risks=["references many Ignition Fuel models with separate license/provenance review"],
            recommended_next_action="Audit subasset licenses before production use.",
            future_isaac_candidate=True,
        ),
        _scene_option(
            rank=6,
            asset_id="qvpr_cpr_office_extension_gazebo",
            name="QVPR CPR Office Extension Gazebo",
            source_url="https://github.com/QVPR/cpr_office_extension_gazebo",
            direct_asset_url="git clone https://github.com/QVPR/cpr_office_extension_gazebo.git",
            license_id="MIT repo",
            commercial_use_status="repo_allowed_but_inherited_base_asset_audit_required",
            formats=["Gazebo assets", "URDF", "Blender file"],
            texture_material_quality="textured office extension with rooms and objects",
            mujoco_conversion_difficulty="medium",
            collision_setup="manual wall/furniture/counter colliders",
            scene_type="office_corridor",
            navigation_and_task_fit="Hallways, office room entry, desk/counter approaches.",
            supports_500_episodes="Probably, with route and clutter randomization.",
            risks=["Clearpath base asset license chain must be verified"],
            recommended_next_action="Use only after inherited license chain is clear.",
        ),
        _scene_option(
            rank=7,
            asset_id="aws_robomaker_small_house_world",
            name="AWS RoboMaker Small House World",
            source_url="https://github.com/aws-robotics/aws-robomaker-small-house-world",
            direct_asset_url=(
                "git clone https://github.com/aws-robotics/aws-robomaker-small-house-world.git"
            ),
            license_id="MIT-0",
            commercial_use_status="commercial_use_allowed_no_attribution_required",
            formats=["Gazebo .world", "SDF models", "DAE meshes", "PNG textures"],
            texture_material_quality="practical home interior textures",
            mujoco_conversion_difficulty="low_medium",
            collision_setup="simplified wall/furniture/floor colliders",
            scene_type="residential_house",
            navigation_and_task_fit="Doorway, room-to-room, furniture avoidance tasks.",
            supports_500_episodes="Marginal but workable with heavy route and clutter variation.",
            risks=["Small scene can become repetitive"],
            recommended_next_action="Use as regression/smoke scene rather than primary benchmark.",
        ),
        _scene_option(
            rank=8,
            asset_id="sketchfab_lazysakana_supermarket",
            name="Sketchfab Supermarket by LazySakana",
            source_url="https://sketchfab.com/3d-models/supermarket-b80e0c447bc54def82007dda380dda2e",
            direct_asset_url="Sketchfab download, login/API may be required",
            license_id="CC-BY",
            commercial_use_status="commercial_use_allowed_with_attribution_but_marketplace_risk",
            formats=["Sketchfab download formats, likely GLB/FBX/OBJ"],
            texture_material_quality="low-poly supermarket/game asset, visually usable",
            mujoco_conversion_difficulty="low_medium",
            collision_setup="manual shelf/counter/freezer primitive colliders",
            scene_type="supermarket_retail",
            navigation_and_task_fit="Retail aisle, shelf inspection, checkout approach tasks.",
            supports_500_episodes="Probably if aisle count is sufficient; verify after download.",
            risks=["marketplace provenance, attribution, trademark/brand texture risk, login required"],
            recommended_next_action="Prototype only until downloaded contents and provenance pass review.",
        ),
        _scene_option(
            rank=9,
            asset_id="nvidia_openusd_warehouse_industrial_simready",
            name="NVIDIA Omniverse OpenUSD Warehouse/Industrial/SimReady Packs",
            source_url="https://docs.omniverse.nvidia.com/usd/latest/usd_content_samples/downloadable_packs.html",
            direct_asset_url="Download links on NVIDIA Omniverse USD downloadable packs page",
            license_id="NVIDIA asset terms",
            commercial_use_status="usable_in_projects_per_page_but_terms_review_required",
            formats=["OpenUSD", "MDL materials", "SimReady USD assets"],
            texture_material_quality="high quality warehouse, industrial, and factory components",
            mujoco_conversion_difficulty="high",
            collision_setup="USD physics extraction or manual simplification",
            scene_type="warehouse_factory_industrial",
            navigation_and_task_fit="Excellent future warehouse/factory variety.",
            supports_500_episodes="Yes with large asset variety and layout generation.",
            risks=["large downloads", "USD/MDL to MuJoCo conversion complexity", "terms review"],
            recommended_next_action="Prefer for Isaac/Isaac Lab escalation, not first MuJoCo scene.",
            future_isaac_candidate=True,
        ),
        _scene_option(
            rank=10,
            asset_id="hssd",
            name="Habitat Synthetic Scene Dataset",
            source_url="https://huggingface.co/datasets/hssd/hssd-hab",
            direct_asset_url="Hugging Face gated/terms flow",
            license_id="CC-BY-NC-4.0",
            commercial_use_status="non_commercial_research_only",
            formats=["Habitat scenes and object assets"],
            texture_material_quality="high",
            mujoco_conversion_difficulty="high",
            collision_setup="derived/simplified collision required",
            scene_type="indoor_home_embodied_ai",
            navigation_and_task_fit="Technically strong for research indoor navigation.",
            supports_500_episodes="Yes technically; not production-safe commercially.",
            risks=["non-commercial license"],
            recommended_next_action="Avoid for commercial Blueprint evaluation products.",
            avoid_or_research_only=True,
        ),
        _scene_option(
            rank=11,
            asset_id="hm3d_matterport3d",
            name="Habitat-Matterport 3D / HM3D",
            source_url="https://aihabitat.org/datasets/hm3d/",
            direct_asset_url="Matterport academic access flow",
            license_id="academic non-commercial research",
            commercial_use_status="non_commercial_research_only",
            formats=["OBJ", "GLB", "JPG textures", "MTL"],
            texture_material_quality="very high scanned spaces",
            mujoco_conversion_difficulty="high",
            collision_setup="heavy simplification required for scanned geometry",
            scene_type="scanned_indoor_buildings",
            navigation_and_task_fit="Large route variety, poor commercial fit.",
            supports_500_episodes="Yes technically; blocked for commercial use.",
            risks=["non-commercial license", "account/request process"],
            recommended_next_action="Do not use in commercial Blueprint lane.",
            avoid_or_research_only=True,
        ),
        _scene_option(
            rank=12,
            asset_id="igibson_gibson",
            name="iGibson / Gibson Scenes",
            source_url="https://svl.stanford.edu/igibson/1.0/docs/dataset.html",
            direct_asset_url="iGibson script or Gibson license agreement flow",
            license_id="license agreement / software license only",
            commercial_use_status="treat_as_research_only_until_separately_licensed",
            formats=["OBJ", "MTL", "textures", "floor maps", "traversability maps"],
            texture_material_quality="good scanned homes/offices",
            mujoco_conversion_difficulty="high",
            collision_setup="derived/simplified collision required",
            scene_type="home_office_scans",
            navigation_and_task_fit="Good technical navigation fit.",
            supports_500_episodes="Yes technically; licensing blocks production use.",
            risks=["license agreement", "encrypted/copyright-restricted object models"],
            recommended_next_action="Avoid for production unless separately licensed.",
            avoid_or_research_only=True,
        ),
        _scene_option(
            rank=13,
            asset_id="front_3d_future",
            name="3D-FRONT / 3D-FUTURE",
            source_url="https://dlr-rm.github.io/BlenderProc/examples/datasets/front_3d/README.html",
            direct_asset_url="Terms/email access flow via original dataset",
            license_id="dataset terms unclear for Blueprint commercial use",
            commercial_use_status="ambiguous_fail_closed",
            formats=["3D-FRONT JSON", "3D-FUTURE furniture models", "textures"],
            texture_material_quality="strong furnished residential potential",
            mujoco_conversion_difficulty="high",
            collision_setup="manual or derived primitive colliders",
            scene_type="residential_layouts",
            navigation_and_task_fit="Homes/apartments with furniture.",
            supports_500_episodes="Yes technically; license/acquisition blocks immediate use.",
            risks=["terms approval", "unclear commercial permission", "not immediate"],
            recommended_next_action="Do not use until commercial permission is explicit.",
            avoid_or_research_only=True,
        ),
    ]


def build_mujoco_scene_asset_research(
    *,
    output_dir: str | Path | None = None,
    generated_at: str | None = None,
) -> Dict[str, Any]:
    resolved_generated_at = generated_at or utc_now_iso()
    options = scene_asset_options()
    manifest = {
        "schema_version": MUJOCO_SCENE_ASSET_RESEARCH_SCHEMA_VERSION,
        "generated_at": resolved_generated_at,
        "status": "completed_primary_source_review_required_before_production",
        "candidate_count": len(options),
        "options": options,
        "top_3_immediate_mujoco": [
            "aws_robomaker_small_warehouse_world",
            "manchester_nuclear_gazebo_assets",
            "aws_robomaker_bookstore_world",
        ],
        "top_3_future_isaac": [
            "nvidia_openusd_warehouse_industrial_simready",
            "replicacad_interactive_and_baked",
            "aws_robomaker_hospital_world",
        ],
        "avoid_or_research_only": [
            option["asset_id"] for option in options if option.get("avoid_or_research_only")
        ],
        "recommended_first_scene": DEFAULT_SCENE_ID,
        "decision_reason": (
            "AWS Small Warehouse has the best mix of permissive license, direct download, "
            "warehouse task fit, and manageable Gazebo-to-MuJoCo conversion work."
        ),
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    manifest["deterministic_fingerprint"] = _sha_payload(
        {"options": options, "recommended_first_scene": manifest["recommended_first_scene"]}
    )
    if output_dir is not None:
        out_dir = Path(output_dir).resolve()
        write_json(out_dir / "mujoco_scene_asset_research.json", manifest)
    return manifest


def _aws_small_warehouse_definition() -> Dict[str, Any]:
    spawn_zones = [
        {"zone_id": "spawn_loading_west", "label": "west loading lane", "xyz": [-5.2, 8.8, 0.793], "yaw": -1.57},
        {"zone_id": "spawn_loading_east", "label": "east loading lane", "xyz": [5.2, 8.2, 0.793], "yaw": -1.57},
        {"zone_id": "spawn_south_pallet_jack", "label": "pallet jack approach", "xyz": [-0.6, -9.0, 0.793], "yaw": 1.57},
        {"zone_id": "spawn_center_cross_aisle", "label": "center cross aisle", "xyz": [0.2, 2.3, 0.793], "yaw": 0.0},
        {"zone_id": "spawn_north_bucket_row", "label": "bucket row", "xyz": [-1.7, 7.5, 0.793], "yaw": 0.0},
        {"zone_id": "spawn_west_shelf_end", "label": "west shelf end", "xyz": [-5.2, -1.0, 0.793], "yaw": 0.0},
        {"zone_id": "spawn_east_shelf_end", "label": "east shelf end", "xyz": [5.2, -1.1, 0.793], "yaw": 3.14},
        {"zone_id": "spawn_south_cross_aisle", "label": "south cross aisle", "xyz": [0.0, -6.7, 0.793], "yaw": 0.0},
    ]
    target_zones = [
        {"zone_id": "target_shelf_e_upper", "label": "Shelf E bay upper", "xyz": [4.3, 0.6, 0.793], "yaw": 3.14},
        {"zone_id": "target_shelf_e_lower", "label": "Shelf E bay lower", "xyz": [4.2, -4.8, 0.793], "yaw": 3.14},
        {"zone_id": "target_shelf_d_middle", "label": "Shelf D middle bay", "xyz": [4.2, -3.0, 0.793], "yaw": 3.14},
        {"zone_id": "target_shelf_f_west", "label": "Shelf F west bay", "xyz": [-5.2, -1.0, 0.793], "yaw": 0.0},
        {"zone_id": "target_clutter_loading", "label": "loading clutter cluster", "xyz": [4.9, 8.6, 0.793], "yaw": -1.57},
        {"zone_id": "target_bucket_row", "label": "bucket row", "xyz": [0.4, 9.1, 0.793], "yaw": -1.57},
        {"zone_id": "target_pallet_jack", "label": "pallet jack", "xyz": [-0.3, -9.3, 0.793], "yaw": 1.57},
        {"zone_id": "target_trash_can", "label": "trash can corner", "xyz": [-1.5, 7.7, 0.793], "yaw": -0.5},
        {"zone_id": "target_cross_aisle_marker", "label": "cross aisle marker", "xyz": [0.1, 3.8, 0.793], "yaw": 0.0},
        {"zone_id": "target_south_blocked_detour", "label": "south detour endpoint", "xyz": [-1.4, -7.8, 0.793], "yaw": 0.0},
    ]
    tasks = [
        ("inspect_shelf_e_upper", "Inspect Shelf E upper bay and stop with the shelf face centered."),
        ("inspect_shelf_d_middle", "Approach Shelf D middle bay and verify the target bay is reachable."),
        ("verify_pallet_jack_clearance", "Approach the pallet jack zone and keep a passable side buffer."),
        ("check_loading_clutter", "Traverse to the loading clutter cluster and inspect carton placement."),
        ("bucket_row_inventory_check", "Walk to the bucket row and pause for inventory/label review."),
        ("cross_aisle_transfer", "Move from a loading lane to the cross-aisle marker without shelf contact."),
        ("blocked_aisle_detour", "Route around a blocked south aisle to the detour endpoint."),
        ("trash_corner_safety_check", "Approach the trash-can corner while maintaining wall clearance."),
        ("narrow_shelf_approach", "Enter a narrow shelf-end approach and stop inside the target radius."),
        ("wrong_bay_recovery", "Reject the wrong nearby shelf bay and continue to the requested target bay."),
    ]
    scenario_families = [
        {
            "scenario_id": "scenario_warehouse_shelf_e_upper_inspection",
            "task_id": "inspect_shelf_e_upper",
            "spawn_zones": ["spawn_loading_west", "spawn_center_cross_aisle"],
            "target_zones": ["target_shelf_e_upper"],
            "family_label": "Shelf E inspection from loading/cross-aisle starts",
        },
        {
            "scenario_id": "scenario_warehouse_shelf_d_middle_inspection",
            "task_id": "inspect_shelf_d_middle",
            "spawn_zones": ["spawn_loading_east", "spawn_south_cross_aisle"],
            "target_zones": ["target_shelf_d_middle"],
            "family_label": "Shelf D middle bay approach",
        },
        {
            "scenario_id": "scenario_warehouse_pallet_jack_clearance",
            "task_id": "verify_pallet_jack_clearance",
            "spawn_zones": ["spawn_south_cross_aisle", "spawn_east_shelf_end"],
            "target_zones": ["target_pallet_jack"],
            "family_label": "Pallet jack side-clearance approach",
        },
        {
            "scenario_id": "scenario_warehouse_loading_clutter_check",
            "task_id": "check_loading_clutter",
            "spawn_zones": ["spawn_loading_west", "spawn_west_shelf_end"],
            "target_zones": ["target_clutter_loading"],
            "family_label": "Loading clutter carton inspection",
        },
        {
            "scenario_id": "scenario_warehouse_bucket_row_inventory",
            "task_id": "bucket_row_inventory_check",
            "spawn_zones": ["spawn_center_cross_aisle", "spawn_north_bucket_row"],
            "target_zones": ["target_bucket_row"],
            "family_label": "Bucket row inventory route",
        },
        {
            "scenario_id": "scenario_warehouse_cross_aisle_transfer",
            "task_id": "cross_aisle_transfer",
            "spawn_zones": ["spawn_loading_east", "spawn_loading_west"],
            "target_zones": ["target_cross_aisle_marker"],
            "family_label": "Cross-aisle transfer route",
        },
        {
            "scenario_id": "scenario_warehouse_blocked_south_detour",
            "task_id": "blocked_aisle_detour",
            "spawn_zones": ["spawn_south_pallet_jack", "spawn_west_shelf_end"],
            "target_zones": ["target_south_blocked_detour"],
            "family_label": "Blocked south aisle detour",
        },
        {
            "scenario_id": "scenario_warehouse_trash_corner_clearance",
            "task_id": "trash_corner_safety_check",
            "spawn_zones": ["spawn_north_bucket_row", "spawn_center_cross_aisle"],
            "target_zones": ["target_trash_can"],
            "family_label": "Trash corner wall-clearance route",
        },
        {
            "scenario_id": "scenario_warehouse_narrow_shelf_end",
            "task_id": "narrow_shelf_approach",
            "spawn_zones": ["spawn_west_shelf_end", "spawn_east_shelf_end"],
            "target_zones": ["target_shelf_f_west", "target_shelf_e_lower"],
            "family_label": "Narrow shelf-end approach",
        },
        {
            "scenario_id": "scenario_warehouse_wrong_bay_recovery",
            "task_id": "wrong_bay_recovery",
            "spawn_zones": ["spawn_south_cross_aisle", "spawn_loading_east"],
            "target_zones": ["target_shelf_e_lower", "target_shelf_d_middle"],
            "family_label": "Wrong shelf bay recovery",
        },
    ]
    return {
        "asset_id": DEFAULT_SCENE_ID,
        "scene_id": "aws-robomaker-small-warehouse",
        "site_type": "warehouse_logistics",
        "source_url": "https://github.com/aws-robotics/aws-robomaker-small-warehouse-world",
        "license_id": "MIT-0",
        "spawn_zones": spawn_zones,
        "target_zones": target_zones,
        "tasks": [{"task_id": task_id, "task_statement": statement} for task_id, statement in tasks],
        "scenario_families": scenario_families,
    }


def _zone_index(scene: Mapping[str, Any], key: str) -> Dict[str, Dict[str, Any]]:
    return {
        _string(item.get("zone_id")): dict(item)
        for item in scene.get(key, []) or []
        if isinstance(item, Mapping) and _string(item.get("zone_id"))
    }


def _task_cards(scene: Mapping[str, Any]) -> Dict[str, Any]:
    cards = []
    for task in scene.get("tasks", []) or []:
        if not isinstance(task, Mapping):
            continue
        task_id = _string(task.get("task_id"))
        cards.append(
            {
                "task_id": task_id,
                "task_statement": _string(task.get("task_statement")),
                "task_category": "warehouse_navigation_inspection",
                "site_specific": True,
                "robot_profile_id": "unitree_g1_mujoco_humanoid",
                "required_metrics": [
                    "reach_target_radius",
                    "route_distance_m",
                    "timeout",
                    "collision_risk",
                    "forbidden_zone_contact",
                    "camera_target_visibility",
                    "scenario_eval_run_coverage",
                ],
                "recording_views_required": list(REQUIRED_RECORDING_VIEWS),
                "proof_boundary": "task_card_is_eval_contract_not_simulator_execution_proof",
            }
        )
    return {
        "schema_version": "real_site_robot_eval_task_cards.v0.1",
        "source_mode": "external_mujoco_scene_asset",
        "cards": cards,
        "count": len(cards),
    }


def _scenario_cards(scene: Mapping[str, Any]) -> Dict[str, Any]:
    cards = []
    for family in scene.get("scenario_families", []) or []:
        if not isinstance(family, Mapping):
            continue
        cards.append(
            {
                "scenario_id": _string(family.get("scenario_id")),
                "task_id": _string(family.get("task_id")),
                "robot_profile_id": "unitree_g1_mujoco_humanoid",
                "scenario_family": _safe_id(family.get("family_label"), fallback="warehouse"),
                "normal_scenario": {"statement": _string(family.get("family_label"))},
                "start_zone_ids": list(family.get("spawn_zones") or []),
                "target_zone_ids": list(family.get("target_zones") or []),
                "recording_views_required": list(REQUIRED_RECORDING_VIEWS),
                "site_specific": True,
                "variation": {
                    "statement": (
                        "Apply warehouse-specific route, lighting, occlusion, blocked-aisle, "
                        "wrong-bay, and narrow-approach mutations."
                    )
                },
                "observed_vs_inferred_labels": {
                    "layout": "external_asset_source_layout",
                    "capture_truth": "not_raw_blueprint_capture",
                },
                "proof_boundary": "scenario_card_is_eval_contract_not_simulator_execution_proof",
            }
        )
    return {
        "schema_version": "real_site_robot_eval_scenario_cards.v0.1",
        "source_mode": "external_mujoco_scene_asset",
        "cards": cards,
        "count": len(cards),
    }


def _scenario_family_library(scene: Mapping[str, Any], *, generated_at: str) -> Dict[str, Any]:
    families = []
    for family in scene.get("scenario_families", []) or []:
        if not isinstance(family, Mapping):
            continue
        scenario_id = _string(family.get("scenario_id"))
        task_id = _string(family.get("task_id"))
        families.append(
            {
                "family_id": f"family_{_safe_id(scenario_id)}",
                "task_id": task_id,
                "scenario_id": scenario_id,
                "robot_profile_id": "unitree_g1_mujoco_humanoid",
                "scenario_family": _safe_id(family.get("family_label"), fallback=scenario_id),
                "site_specific_context": {
                    "spawn_zone_ids": list(family.get("spawn_zones") or []),
                    "target_zone_ids": list(family.get("target_zones") or []),
                    "recording_views_required": list(REQUIRED_RECORDING_VIEWS),
                    "generated_at": generated_at,
                },
                "variations": [
                    {
                        "variation_id": variation_name,
                        "variation_name": variation_name,
                        "scenario_status": "ready_for_mujoco_engine_adapter",
                        "site_specific_mutation_hint": _warehouse_variation_hint(variation_name),
                    }
                    for variation_name in SCENARIO_VARIATION_NAMES
                ],
            }
        )
    return {
        "schema_version": "scenario_family_library.v1",
        "source_mode": "external_mujoco_scene_asset",
        "family_count": len(families),
        "variation_names_required": list(SCENARIO_VARIATION_NAMES),
        "families": families,
    }


def _warehouse_variation_hint(variation_name: str) -> str:
    hints = {
        "lighting_variation": "dim or brighten overhead warehouse lamps along the route",
        "object_rotation": "rotate carton stack, bucket, or pallet near target bay",
        "cart_shifted": "shift pallet jack/cart into the aisle shoulder",
        "blocked_path": "place cartons or pallet stack across the shortest aisle path",
        "human_crossing": "add a crossing worker actor at a cross aisle",
        "forklift_nearby": "stage a slow or stationary forklift near the target aisle",
        "occlusion": "block target shelf label with carton stack or bucket",
        "glare": "add glare on painted floor marker or shelf label",
        "missing_label": "hide or blank the target shelf/bucket label",
        "wrong_object_nearby": "place a similar wrong bay/object beside the target",
        "narrow_approach_angle": "constrain approach yaw and aisle side clearance",
    }
    return hints.get(variation_name, "warehouse-specific review mutation")


def _recording_plan(scene: Mapping[str, Any], *, generated_at: str) -> Dict[str, Any]:
    cameras = [
        {
            "camera": "sim_robot_follow_pov",
            "required": True,
            "mode": "virtual_free_camera_following_g1_root_not_physical_robot_sensor",
            "purpose": "robot-like route review and target visibility",
        },
        {
            "camera": "overview",
            "required": True,
            "mode": "fixed_overhead_or_high_oblique_scene_camera",
            "purpose": "route, obstacle, and scenario mutation context",
        },
        {
            "camera": "side",
            "required": True,
            "mode": "virtual_side_profile_camera_tracking_g1_root",
            "purpose": "clearance, shelf approach, and blocked-path review",
        },
    ]
    return {
        "schema_version": MUJOCO_SCENE_RECORDING_PLAN_SCHEMA_VERSION,
        "generated_at": generated_at,
        "scene_asset_id": scene["asset_id"],
        "status": "ready_for_mujoco_recording_after_scene_conversion",
        "required_recording_views": list(REQUIRED_RECORDING_VIEWS),
        "cameras": cameras,
        "per_rendered_episode_expectation": {
            "minimum_frame_views": list(REQUIRED_RECORDING_VIEWS),
            "video_outputs_expected_when_ffmpeg_available": [
                "overview.mp4",
                "sim_robot_follow_pov.mp4",
                "side.mp4",
            ],
        },
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _git_commit(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _asset_inventory(local_asset_root: Path | None) -> Dict[str, Any]:
    if local_asset_root is None:
        return {
            "status": "not_inspected_no_local_asset_root",
            "local_asset_root": None,
            "file_count": 0,
            "remote_asset_downloads_performed_by_packet_builder": False,
        }
    root = local_asset_root.resolve()
    if not root.is_dir():
        return {
            "status": "blocked_missing_local_asset_root",
            "local_asset_root": str(root),
            "file_count": 0,
            "remote_asset_downloads_performed_by_packet_builder": False,
        }
    files = sorted(path for path in root.rglob("*") if path.is_file())
    suffix_counts: Dict[str, int] = {}
    for path in files:
        suffix = path.suffix.lower() or "<none>"
        suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
    world_files = [path for path in files if path.suffix == ".world"]
    model_sdfs = [path for path in files if path.name == "model.sdf"]
    visual_meshes = [path for path in files if path.name.lower().endswith("_visual.dae")]
    collision_meshes = [path for path in files if path.name.lower().endswith("_collision.dae")]
    textures = [
        path
        for path in files
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
    ]
    included_models = _world_included_models(world_files[0]) if world_files else []
    return {
        "status": "inspected_local_asset_root",
        "local_asset_root": str(root),
        "git_commit": _git_commit(root),
        "file_count": len(files),
        "suffix_counts": dict(sorted(suffix_counts.items())),
        "world_file_count": len(world_files),
        "model_sdf_count": len(model_sdfs),
        "visual_dae_mesh_count": len(visual_meshes),
        "collision_dae_mesh_count": len(collision_meshes),
        "texture_file_count": len(textures),
        "world_files": [os.path.relpath(path, root).replace("\\", "/") for path in world_files],
        "sample_visual_meshes": [
            os.path.relpath(path, root).replace("\\", "/") for path in visual_meshes[:12]
        ],
        "sample_collision_meshes": [
            os.path.relpath(path, root).replace("\\", "/") for path in collision_meshes[:12]
        ],
        "sample_textures": [os.path.relpath(path, root).replace("\\", "/") for path in textures[:12]],
        "included_model_count": len(included_models),
        "included_models": included_models[:80],
        "remote_asset_downloads_performed_by_packet_builder": False,
    }


def _world_included_models(world_file: Path) -> List[Dict[str, Any]]:
    try:
        root = ET.parse(world_file).getroot()
    except (ET.ParseError, DefusedXmlException):
        return []
    rows: List[Dict[str, Any]] = []
    for model in root.findall(".//model"):
        include = model.find("include")
        if include is None:
            continue
        uri = include.findtext("uri") or ""
        pose_text = model.findtext("pose") or ""
        pose = []
        for value in pose_text.split()[:6]:
            try:
                pose.append(float(value))
            except ValueError:
                pose.append(0.0)
        rows.append(
            {
                "name": _string(model.get("name")),
                "uri": uri,
                "pose": pose,
            }
        )
    return rows


def _preferred_aws_world_file(local_asset_root: Path) -> Path | None:
    candidates = [
        local_asset_root / "worlds" / "no_roof_small_warehouse.world",
        local_asset_root / "worlds" / "small_warehouse.world",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    worlds = sorted((local_asset_root / "worlds").glob("*.world"))
    return worlds[0] if worlds else None


def _parse_pose_values(value: Any) -> List[float]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        numbers = []
        for item in list(value)[:6]:
            try:
                numbers.append(float(item))
            except (TypeError, ValueError):
                numbers.append(0.0)
    else:
        numbers = []
        for item in _string(value).split()[:6]:
            try:
                numbers.append(float(item))
            except ValueError:
                numbers.append(0.0)
    while len(numbers) < 6:
        numbers.append(0.0)
    return numbers[:6]


def _parse_scale_values(value: Any) -> List[float]:
    numbers = []
    for item in _string(value).split()[:3]:
        try:
            numbers.append(float(item))
        except ValueError:
            numbers.append(1.0)
    while len(numbers) < 3:
        numbers.append(1.0)
    return numbers[:3]


def _model_name_from_uri(uri: str) -> str:
    text = _string(uri)
    if text.startswith("model://"):
        text = text[len("model://") :]
    return text.strip("/").split("/")[0]


def _resolve_model_uri(local_asset_root: Path, uri: str) -> Path | None:
    text = _string(uri)
    if text.startswith("model://"):
        parts = text[len("model://") :].strip("/").split("/")
        if not parts or not parts[0]:
            return None
        return local_asset_root / "models" / parts[0] / Path(*parts[1:])
    path = Path(text)
    return path if path.is_absolute() else local_asset_root / path


def _sdf_visual_meshes(local_asset_root: Path, model_name: str) -> List[Dict[str, Any]]:
    model_root = local_asset_root / "models" / model_name
    model_sdf = model_root / "model.sdf"
    if not model_sdf.is_file():
        return []
    try:
        root = ET.parse(model_sdf).getroot()
    except (ET.ParseError, DefusedXmlException):
        return []
    rows: List[Dict[str, Any]] = []
    for link in root.findall(".//link"):
        link_pose = _parse_pose_values(link.findtext("pose"))
        for visual in link.findall("visual"):
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                continue
            uri = _string(mesh.findtext("uri"))
            mesh_path = _resolve_model_uri(local_asset_root, uri)
            if mesh_path is None:
                continue
            rows.append(
                {
                    "visual_name": _string(visual.get("name")) or "visual",
                    "model_name": model_name,
                    "mesh_uri": uri,
                    "mesh_path": mesh_path,
                    "mesh_scale": _parse_scale_values(mesh.findtext("scale")),
                    "link_pose": link_pose,
                    "visual_pose": _parse_pose_values(visual.findtext("pose")),
                }
            )
    return rows


def _collada_unit_scale(loaded_scene: Any) -> float:
    units = _string(getattr(loaded_scene, "units", "")).lower()
    if "0.01" in units and "meter" in units:
        return 0.01
    if "centimeter" in units or units in {"cm", "centimeters"}:
        return 0.01
    if units in {"m", "meter", "meters"}:
        return 1.0
    return 1.0


def _pose_matrix(pose: Sequence[float]) -> Any:
    import trimesh  # type: ignore[import-not-found]

    values = _parse_pose_values(pose)
    matrix = trimesh.transformations.euler_matrix(values[3], values[4], values[5], axes="sxyz")
    matrix[:3, 3] = values[:3]
    return matrix


def _scale_matrix(scale: Sequence[float]) -> Any:
    import numpy as np

    values = list(scale)[:3]
    while len(values) < 3:
        values.append(1.0)
    matrix = np.eye(4)
    matrix[0, 0] = float(values[0])
    matrix[1, 1] = float(values[1])
    matrix[2, 2] = float(values[2])
    return matrix


def _export_scene_glb(scene: Any, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    exported = scene.export(file_type="glb")
    if isinstance(exported, bytes):
        output_path.write_bytes(exported)
    else:
        output_path.write_bytes(bytes(exported))


def _materialize_aws_scene_asset(
    *,
    local_asset_root: Path | None,
    capture_root: Path,
    packet_dir: Path,
    generated_at: str,
) -> Dict[str, Any]:
    manifest_path = packet_dir / "external_scene_materialization_manifest.json"
    base_manifest: Dict[str, Any] = {
        "schema_version": "mujoco_external_scene_materialization.v1",
        "generated_at": generated_at,
        "scene_asset_id": DEFAULT_SCENE_ID,
        "status": "skipped_no_local_asset_root",
        "local_asset_root": str(local_asset_root) if local_asset_root else None,
        "conversion_performed": False,
        "simulators_run": False,
        "simulator_execution_proven": False,
        "robot_policy_execution_proven": False,
        "physics_contact_validated": False,
        "visual_fidelity": {
            "mode": "not_materialized",
            "pbr_texture_parity_proven": False,
            "color_baked_from_source_textures": False,
        },
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    if local_asset_root is None:
        write_json(manifest_path, base_manifest)
        return base_manifest
    root = local_asset_root.resolve()
    base_manifest["local_asset_root"] = str(root)
    if not root.is_dir():
        base_manifest.update(
            {
                "status": "blocked_missing_local_asset_root",
                "blockers": ["local_asset_root_missing"],
            }
        )
        write_json(manifest_path, base_manifest)
        return base_manifest
    world_file = _preferred_aws_world_file(root)
    if world_file is None:
        base_manifest.update(
            {
                "status": "blocked_missing_world_file",
                "blockers": ["worlds_directory_has_no_world_file"],
            }
        )
        write_json(manifest_path, base_manifest)
        return base_manifest
    try:
        import trimesh  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - dependency guard.
        base_manifest.update(
            {
                "status": "blocked_missing_trimesh_runtime",
                "world_file": os.path.relpath(world_file, root).replace("\\", "/"),
                "blockers": [f"trimesh_import_failed:{type(exc).__name__}"],
            }
        )
        write_json(manifest_path, base_manifest)
        return base_manifest

    included_models = _world_included_models(world_file)
    scene = trimesh.Scene()
    blockers: List[str] = []
    materialized_meshes: List[Dict[str, Any]] = []
    total_geometry_count = 0
    total_vertex_count = 0
    total_face_count = 0
    for model_index, model in enumerate(included_models, start=1):
        uri = _string(model.get("uri"))
        model_name = _model_name_from_uri(uri)
        if not model_name:
            blockers.append(f"world_model_{model_index}:missing_model_uri")
            continue
        model_root = root / "models" / model_name
        if not model_root.is_dir():
            blockers.append(f"{model_name}:missing_model_directory")
            continue
        visual_meshes = _sdf_visual_meshes(root, model_name)
        if not visual_meshes:
            blockers.append(f"{model_name}:no_visual_meshes_in_model_sdf")
            continue
        world_pose = _parse_pose_values(model.get("pose"))
        for visual_index, visual in enumerate(visual_meshes, start=1):
            mesh_path = Path(visual["mesh_path"])
            if not mesh_path.is_file():
                rel_mesh_path = os.path.relpath(mesh_path, root).replace("\\", "/")
                blockers.append(
                    f"{model_name}:{visual['visual_name']}:missing_mesh:{rel_mesh_path}"
                )
                continue
            try:
                loaded = trimesh.load(mesh_path, force="scene")
            except Exception as exc:
                blockers.append(
                    f"{model_name}:{visual['visual_name']}:load_failed:{type(exc).__name__}"
                )
                continue
            if not isinstance(loaded, trimesh.Scene):
                loaded = trimesh.Scene(loaded)
            unit_scale = _collada_unit_scale(loaded)
            node_names = list(getattr(loaded.graph, "nodes_geometry", []) or [])
            if not node_names:
                blockers.append(f"{model_name}:{visual['visual_name']}:no_geometry_nodes")
                continue
            instance_geometry_count = 0
            instance_vertex_count = 0
            instance_face_count = 0
            for node_index, node_name in enumerate(node_names, start=1):
                node_transform, geometry_name = loaded.graph[node_name]
                geometry = loaded.geometry[geometry_name].copy()
                to_color = getattr(getattr(geometry, "visual", None), "to_color", None)
                if callable(to_color):
                    geometry.visual = to_color()
                final_transform = (
                    _pose_matrix(world_pose)
                    @ _pose_matrix(visual["link_pose"])
                    @ _pose_matrix(visual["visual_pose"])
                    @ _scale_matrix(visual["mesh_scale"])
                    @ _scale_matrix([unit_scale, unit_scale, unit_scale])
                    @ node_transform
                )
                geometry.apply_transform(final_transform)
                scene.add_geometry(
                    geometry,
                    geom_name=(
                        f"{_safe_id(model.get('name'), fallback=model_name)}_"
                        f"{visual_index:02d}_{node_index:02d}"
                    ),
                )
                instance_geometry_count += 1
                instance_vertex_count += int(len(geometry.vertices))
                instance_face_count += int(len(geometry.faces))
            total_geometry_count += instance_geometry_count
            total_vertex_count += instance_vertex_count
            total_face_count += instance_face_count
            materialized_meshes.append(
                {
                    "instance_name": _string(model.get("name")),
                    "model_name": model_name,
                    "visual_name": visual["visual_name"],
                    "source_mesh": os.path.relpath(mesh_path, root).replace("\\", "/"),
                    "geometry_count": instance_geometry_count,
                    "vertex_count": instance_vertex_count,
                    "face_count": instance_face_count,
                    "collada_unit_scale_applied": unit_scale,
                    "source_texture_colors_baked_to_vertices": True,
                }
            )
    if total_geometry_count == 0:
        manifest = {
            **base_manifest,
            "status": "blocked_no_visual_meshes_materialized",
            "world_file": os.path.relpath(world_file, root).replace("\\", "/"),
            "included_model_count": len(included_models),
            "materialized_mesh_count": 0,
            "blockers": blockers or ["no_visual_meshes_materialized"],
        }
        write_json(manifest_path, manifest)
        return manifest

    asset_dir = capture_root / "pipeline" / "mujoco_external_scene_assets"
    compatibility_dir = capture_root / "pipeline" / "worldlabs_assets"
    glb_path = asset_dir / "aws_small_warehouse_scene.glb"
    compatibility_glb_path = compatibility_dir / "scene.glb"
    _export_scene_glb(scene, glb_path)
    ensure_dir(compatibility_dir)
    shutil.copyfile(glb_path, compatibility_glb_path)
    bounds = scene.bounds.tolist() if scene.bounds is not None else None
    manifest = {
        **base_manifest,
        "status": "completed_with_warnings" if blockers else "completed",
        "world_file": os.path.relpath(world_file, root).replace("\\", "/"),
        "world_file_selection_reason": (
            "prefer_no_roof_variant_for_recordable_overview_when_available"
            if world_file.name.startswith("no_roof")
            else "default_world_file"
        ),
        "included_model_count": len(included_models),
        "materialized_mesh_count": len(materialized_meshes),
        "materialized_geometry_count": total_geometry_count,
        "vertex_count": total_vertex_count,
        "face_count": total_face_count,
        "scene_bounds_m": bounds,
        "scene_glb_path": str(glb_path),
        "compatibility_scene_glb_path": str(compatibility_glb_path),
        "compatibility_scene_glb_contract": (
            "The existing MuJoCo G1 simulator command consumes this GLB through its "
            "capture-root scene discovery path."
        ),
        "blockers": blockers,
        "materialized_meshes": materialized_meshes[:120],
        "conversion_performed": True,
        "visual_fidelity": {
            "mode": "source_texture_sampled_to_vertex_colors_for_mujoco_obj_path",
            "pbr_texture_parity_proven": False,
            "color_baked_from_source_textures": True,
            "white_or_checkerboard_scene_success_allowed": False,
        },
        "claim_boundary": {
            **CLAIM_BOUNDARY,
            "conversion_performed": True,
            "remote_asset_downloads_performed_by_packet_builder": False,
        },
    }
    write_json(manifest_path, manifest)
    return manifest


def _site_card(scene: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": "real_site_robot_eval_site_card.v0.1",
        "scene_id": scene["scene_id"],
        "capture_id": DEFAULT_CAPTURE_ID,
        "site_id": scene["asset_id"],
        "site_type": scene["site_type"],
        "source_mode": "external_mujoco_scene_asset",
        "external_scene_asset_not_raw_capture": True,
        "license_id": scene["license_id"],
        "source_url": scene["source_url"],
    }


def _robot_profile_manifest() -> Dict[str, Any]:
    return {
        "schema_version": "external_scene_robot_profile_manifest.v1",
        "robot_profiles": [
            {
                "robot_profile_id": "unitree_g1_mujoco_humanoid",
                "label": "Unitree G1 MuJoCo humanoid",
                "embodiment": "humanoid",
                "base_type": "biped",
                "sensors": ["virtual_rgb"],
                "source": "google_deepmind_mujoco_menagerie_unitree_g1_expected",
                "proof_boundary": "robot profile declaration only; asset execution requires MuJoCo run",
            }
        ],
    }


def _write_capture_shell(capture_root: Path, scene: Mapping[str, Any], *, generated_at: str) -> None:
    ensure_dir(capture_root / "raw")
    descriptor = {
        "schema_version": MUJOCO_EXTERNAL_SCENE_CAPTURE_DESCRIPTOR_SCHEMA_VERSION,
        "scene_id": scene["scene_id"],
        "capture_id": DEFAULT_CAPTURE_ID,
        "site_type": scene["site_type"],
        "source_mode": "external_mujoco_scene_asset",
        "external_scene_asset_not_raw_capture": True,
        "raw_capture_evidence_available": False,
        "generated_at": generated_at,
        "metadata": {
            "site_identity": {"site_id": scene["asset_id"], "site_type": scene["site_type"]},
            "rights": {
                "license_id": scene["license_id"],
                "source_url": scene["source_url"],
                "commercial_use_review_status": "candidate_allowed_by_license_review_required",
            },
        },
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    raw_manifest = {
        "schema_version": "external_mujoco_scene_raw_manifest.v1",
        "scene_id": scene["scene_id"],
        "capture_id": DEFAULT_CAPTURE_ID,
        "site_type": scene["site_type"],
        "source_mode": "external_mujoco_scene_asset",
        "external_scene_asset": {
            "asset_id": scene["asset_id"],
            "source_url": scene["source_url"],
            "license_id": scene["license_id"],
        },
        "raw_capture_evidence_available": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    task_hypothesis = {
        "schema_version": "external_scene_task_hypothesis.v1",
        "source_mode": "scene_packet_site_specific_tasks",
        "tasks": [
            {
                "task_id": task["task_id"],
                "task_text": task["task_statement"],
                "task_category": "warehouse_navigation_inspection",
            }
            for task in scene.get("tasks", [])
            if isinstance(task, Mapping)
        ],
    }
    write_json(capture_root / "capture_descriptor.json", descriptor)
    write_json(capture_root / "raw" / "manifest.json", raw_manifest)
    write_json(capture_root / "raw" / "task_hypothesis.json", task_hypothesis)


def _pose_with_jitter(zone: Mapping[str, Any], *, seed: int) -> Dict[str, Any]:
    xyz = list(zone.get("xyz") or [0.0, 0.0, 0.793])
    while len(xyz) < 3:
        xyz.append(0.793 if len(xyz) == 2 else 0.0)
    jitter_x = (((seed * 17) % 19) - 9) * 0.025
    jitter_y = (((seed * 31) % 23) - 11) * 0.025
    return {
        "zone_id": zone.get("zone_id"),
        "zone_label": zone.get("label"),
        "xyz": [round(float(xyz[0]) + jitter_x, 4), round(float(xyz[1]) + jitter_y, 4), float(xyz[2])],
        "yaw": round(float(zone.get("yaw") or 0.0), 4),
        "source": "scene_packet_zone_with_deterministic_jitter",
    }


def _runs_for_scene(
    *,
    scene: Mapping[str, Any],
    scenario_count: int,
    variation_manifest: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    spawn_zones = _zone_index(scene, "spawn_zones")
    target_zones = _zone_index(scene, "target_zones")
    scenario_families = [
        dict(item) for item in scene.get("scenario_families", []) or [] if isinstance(item, Mapping)
    ]
    if not scenario_families:
        return []
    instances_by_scenario: Dict[str, List[Dict[str, Any]]] = {}
    for item in variation_manifest.get("instances", []) or []:
        if not isinstance(item, Mapping):
            continue
        scenario_id = _string(item.get("scenario_id"))
        if scenario_id:
            instances_by_scenario.setdefault(scenario_id, []).append(dict(item))
    base_count = max(1, int(scenario_count))
    per_family = base_count // len(scenario_families)
    remainder = base_count % len(scenario_families)
    runs: List[Dict[str, Any]] = []
    for family_index, family in enumerate(scenario_families):
        count = per_family + (1 if family_index < remainder else 0)
        scenario_id = _string(family.get("scenario_id"))
        task_id = _string(family.get("task_id"))
        variations = instances_by_scenario.get(scenario_id) or []
        family_spawn_zones = [spawn_zones[item] for item in family.get("spawn_zones", []) if item in spawn_zones]
        family_target_zones = [
            target_zones[item] for item in family.get("target_zones", []) if item in target_zones
        ]
        if not family_spawn_zones or not family_target_zones:
            continue
        for local_index in range(count):
            ordinal = len(runs) + 1
            variation = variations[local_index % len(variations)] if variations else {}
            seed = int(
                sha256(f"{scenario_id}:{task_id}:{local_index}".encode("utf-8")).hexdigest()[:8],
                16,
            )
            spawn = _pose_with_jitter(
                family_spawn_zones[local_index % len(family_spawn_zones)],
                seed=seed,
            )
            target = _pose_with_jitter(
                family_target_zones[(local_index // max(1, len(family_spawn_zones))) % len(family_target_zones)],
                seed=seed >> 4,
            )
            route_distance = math.sqrt(
                (target["xyz"][0] - spawn["xyz"][0]) ** 2
                + (target["xyz"][1] - spawn["xyz"][1]) ** 2
            )
            variation_name = _string(variation.get("variation_name")) or "base_capture_layout"
            runs.append(
                {
                    "scenario_eval_run_id": f"aws_wh_{ordinal:04d}_{_safe_id(scenario_id)}",
                    "episode_id": f"aws_wh_episode_{ordinal:04d}",
                    "scenario_run_id": f"aws_wh_scenario_run_{ordinal:04d}",
                    "task_id": task_id,
                    "scenario_id": scenario_id,
                    "scenario_variation_instance_id": variation.get("instance_id"),
                    "variation_name": variation_name,
                    "baseline_capture_layout": not bool(variation),
                    "spawn_pose": spawn,
                    "target_pose": target,
                    "start_xyz": spawn["xyz"],
                    "target_xyz": target["xyz"],
                    "episode_seed": seed,
                    "route_distance_m": round(route_distance, 4),
                    "site_specific_task": True,
                    "scene_asset_id": scene["asset_id"],
                    "recording_views_required": list(REQUIRED_RECORDING_VIEWS),
                    "robot_pov_required": True,
                    "overview_required": True,
                    "side_view_required": True,
                    "policy_attempt_required": True,
                    "simulator_rollout_required": True,
                    "review_required": True,
                    "concrete_mutation": variation.get("concrete_mutation") or {},
                    "engine_mutations": variation.get("engine_mutations") or {},
                    "episode_authoring": {
                        "spawn_target_source": "deterministic_scene_packet_zone_jitter",
                        "episode_seed_source": "stable_hash_of_scenario_task_and_local_index",
                        "ai_or_api_proposal_allowed_upstream": True,
                        "ai_or_api_used_for_this_row": False,
                        "runtime_ai_route_selection_allowed": False,
                        "freeze_required_before_eval": True,
                    },
                    "claim_boundary": "scenario_eval_run_contract_not_execution_or_success_proof",
                }
            )
    return runs[:base_count]


def _scenario_eval_matrix(
    *,
    capture_root: Path,
    packet_dir: Path,
    scene: Mapping[str, Any],
    scenario_count: int,
    variation_manifest: Mapping[str, Any],
    generated_at: str,
) -> Dict[str, Any]:
    runs = _runs_for_scene(
        scene=scene,
        scenario_count=scenario_count,
        variation_manifest=variation_manifest,
    )
    manifest = {
        "schema_version": "robot_eval_scenario_eval_matrix.v1",
        "generated_at": generated_at,
        "capture_root": str(capture_root),
        "job_dir": str(packet_dir),
        "status": "completed" if len(runs) == int(scenario_count) else "blocked_generation_incomplete",
        "blockers": [] if len(runs) == int(scenario_count) else ["scenario_eval_matrix_generation_incomplete"],
        "scene_asset_id": scene["asset_id"],
        "source_mode": "external_mujoco_scene_asset",
        "scenario_eval_run_count": len(runs),
        "requested_scenario_count": len(scene.get("scenario_families", []) or []),
        "base_scenario_family_count": len(scene.get("scenario_families", []) or []),
        "target_scenario_eval_run_count": int(scenario_count),
        "required_recording_views": list(REQUIRED_RECORDING_VIEWS),
        "variation_instance_count": int(variation_manifest.get("instance_count") or 0),
        "required_variation_names": list(SCENARIO_VARIATION_NAMES),
        "variation_names_covered": sorted(
            {
                _string(run.get("variation_name"))
                for run in runs
                if _string(run.get("variation_name"))
            }
        ),
        "episode_authoring_contract": {
            "spawn_target_variation_seed_handling": "deterministic_frozen_matrix_rows",
            "ai_or_api_proposal_allowed_upstream": True,
            "ai_or_api_proposal_role": (
                "AI/API may propose candidate spawn, target, route, and variation rows before "
                "this matrix is written; accepted proposals must be validated and frozen here "
                "before evaluation."
            ),
            "ai_or_api_used_for_this_matrix": False,
            "runtime_ai_route_selection_allowed": False,
            "runtime_ai_route_selection_used": False,
            "eval_reproducibility_requirement": (
                "MuJoCo and provider workers must execute this exact matrix and report coverage "
                "against scenario_eval_run_id values; they must not call AI/API services to "
                "change spawn, target, route, variation, or seed choices during evaluation."
            ),
        },
        "runs": runs,
        "source_artifacts": {
            "scenario_family_library": "../robot_eval_dataset/scenario_family_library.json",
            "scenario_variation_instances": "scenario_variation_instances.json",
            "recording_plan": "recording_plan.json",
        },
        "simulator_execution_proven": False,
        "robot_policy_execution_proven": False,
        "rank_fidelity_result_proven": False,
        "public_claim_upgrade_allowed": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    manifest["deterministic_fingerprint"] = _sha_payload(
        {"runs": runs, "scene_asset_id": scene["asset_id"]}
    )
    return manifest


def _conversion_plan(
    scene: Mapping[str, Any],
    inventory: Mapping[str, Any],
    materialization: Mapping[str, Any],
    *,
    generated_at: str,
) -> Dict[str, Any]:
    materialization_status = _string(materialization.get("status")) or "not_attempted"
    converted = materialization_status in {"completed", "completed_with_warnings"}
    return {
        "schema_version": "mujoco_external_scene_conversion_plan.v1",
        "generated_at": generated_at,
        "scene_asset_id": scene["asset_id"],
        "status": "visual_scene_glb_materialized_ready_for_recorded_eval"
        if converted
        else "ready_for_conversion_work_not_converted",
        "source_url": scene["source_url"],
        "license_id": scene["license_id"],
        "local_asset_inventory_status": inventory.get("status"),
        "materialization_status": materialization_status,
        "materialized_scene_glb_path": materialization.get("compatibility_scene_glb_path"),
        "materialized_visual_fidelity": materialization.get("visual_fidelity"),
        "download_steps": [
            "git clone https://github.com/aws-robotics/aws-robomaker-small-warehouse-world.git",
            "pin git rev-parse HEAD into the asset provenance manifest",
            "preserve LICENSE beside converted outputs",
        ],
        "conversion_steps": [
            "inspect worlds/small_warehouse.world and referenced model.sdf files",
            "convert visual DAE meshes to OBJ or GLB with texture paths preserved",
            "write MJCF visual geoms with contype=0 and conaffinity=0",
            "write simplified floor, shelf, pallet, bucket, pallet-jack, wall, and clutter colliders",
            "load the generated MJCF in MuJoCo before any run claim",
            "run blueprint-run-mujoco-g1-simulator-command with the generated scenario matrix",
            "verify nonblank overview, virtual POV, and side-view frames",
        ],
        "expected_mujoco_outputs": [
            "generated_scene.xml",
            "converted_obj_or_glb_visual_meshes",
            "simplified_collision_geoms",
            "mujoco_g1_simulator_output.json",
            "overview.mp4",
            "sim_robot_follow_pov.mp4",
            "side.mp4",
        ],
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _runbook_text(*, capture_root: Path, matrix_path: Path, local_asset_root: str | None) -> str:
    asset_line = local_asset_root or "<path-to-aws-robomaker-small-warehouse-world>"
    return f"""# MuJoCo Scene Scenario Packet

Selected scene: AWS RoboMaker Small Warehouse World.

This packet is an external simulation-scene contract, not raw Blueprint capture evidence.
It provides site-specific warehouse task cards, 10 scenario families, and a 500-row
scenario matrix requiring virtual POV, overview, and side-view recordings.

Source asset root:

```bash
{asset_line}
```

Scenario matrix:

```bash
{matrix_path}
```

After the AWS Gazebo/DAE scene is converted into a MuJoCo-loadable textured scene asset,
run the MuJoCo command against this matrix:

```bash
blueprint-run-mujoco-g1-simulator-command \\
  --capture-root {capture_root} \\
  --scenario-eval-matrix {matrix_path} \\
  --max-rendered-episodes 10
```

Do not claim production, physical robot, contact, safety, or policy-quality readiness from
this packet alone. Those require successful simulator output and accepted owner/runtime proof.
"""


def build_mujoco_scene_scenario_packet(
    *,
    output_dir: str | Path | None = None,
    capture_root: str | Path | None = None,
    scene_asset_id: str = DEFAULT_SCENE_ID,
    scenario_count: int = DEFAULT_SCENARIO_COUNT,
    local_asset_root: str | Path | None = None,
    generated_at: str | None = None,
) -> Dict[str, Any]:
    if scene_asset_id != DEFAULT_SCENE_ID:
        raise ValueError(f"unsupported scene_asset_id for first packet: {scene_asset_id}")
    resolved_generated_at = generated_at or utc_now_iso()
    scene = _aws_small_warehouse_definition()
    if capture_root is None:
        base_dir = (
            Path(output_dir).resolve()
            if output_dir
            else _repo_root() / "output" / "mujoco_scene_scenario_packets" / scene_asset_id
        )
        resolved_capture_root = (
            base_dir
            / "local-blueprint"
            / "scenes"
            / scene["scene_id"]
            / "captures"
            / DEFAULT_CAPTURE_ID
        )
    else:
        resolved_capture_root = Path(capture_root).resolve()
        base_dir = Path(output_dir).resolve() if output_dir else resolved_capture_root / "pipeline"
    packet_dir = resolved_capture_root / "pipeline" / "simulation_automation"
    robot_eval_dir = resolved_capture_root / "pipeline" / "robot_eval_dataset"
    evaluation_prep_dir = resolved_capture_root / "pipeline" / "evaluation_prep"
    ensure_dir(packet_dir)
    ensure_dir(robot_eval_dir)
    ensure_dir(evaluation_prep_dir)

    local_root = Path(local_asset_root).resolve() if local_asset_root else None
    inventory = _asset_inventory(local_root)
    research = build_mujoco_scene_asset_research(
        output_dir=packet_dir,
        generated_at=resolved_generated_at,
    )
    recording_plan = _recording_plan(scene, generated_at=resolved_generated_at)
    materialization = _materialize_aws_scene_asset(
        local_asset_root=local_root,
        capture_root=resolved_capture_root,
        packet_dir=packet_dir,
        generated_at=resolved_generated_at,
    )
    conversion_plan = _conversion_plan(
        scene,
        inventory,
        materialization,
        generated_at=resolved_generated_at,
    )

    _write_capture_shell(resolved_capture_root, scene, generated_at=resolved_generated_at)
    write_json(robot_eval_dir / "site_card.json", _site_card(scene))
    write_json(robot_eval_dir / "task_cards.json", _task_cards(scene))
    write_json(robot_eval_dir / "scenario_cards.json", _scenario_cards(scene))
    write_json(
        robot_eval_dir / "scenario_family_library.json",
        _scenario_family_library(scene, generated_at=resolved_generated_at),
    )
    write_json(evaluation_prep_dir / "site_world_spec.json", _robot_profile_manifest())
    write_json(packet_dir / "recording_plan.json", recording_plan)
    write_json(packet_dir / "external_scene_asset_inventory.json", inventory)
    write_json(packet_dir / "external_scene_conversion_plan.json", conversion_plan)

    variation_manifest = build_scenario_variation_instances(
        capture_root=resolved_capture_root,
        output_dir=packet_dir,
        generated_at=resolved_generated_at,
    )
    matrix = _scenario_eval_matrix(
        capture_root=resolved_capture_root,
        packet_dir=packet_dir,
        scene=scene,
        scenario_count=int(scenario_count),
        variation_manifest=variation_manifest,
        generated_at=resolved_generated_at,
    )
    matrix_path = packet_dir / "scenario_eval_matrix.aws_small_warehouse_500.json"
    write_json(matrix_path, matrix)

    selected_option = next(
        option for option in research["options"] if option["asset_id"] == scene_asset_id
    )
    materialization_status = _string(materialization.get("status")) or "not_attempted"
    materialization_ready = materialization_status in {"completed", "completed_with_warnings"}
    packet = {
        "schema_version": MUJOCO_SCENE_SCENARIO_PACKET_SCHEMA_VERSION,
        "generated_at": resolved_generated_at,
        "status": "ready_for_mujoco_recorded_eval"
        if materialization_ready
        else "ready_for_mujoco_conversion_and_recorded_eval",
        "scene_asset": selected_option,
        "capture_root": str(resolved_capture_root),
        "source_mode": "external_mujoco_scene_asset",
        "external_scene_asset_not_raw_capture": True,
        "local_asset_inventory": inventory,
        "scene_materialization_status": materialization_status,
        "scene_glb_available_for_mujoco_command": materialization_ready,
        "scene_glb_path": materialization.get("compatibility_scene_glb_path"),
        "task_count": len(scene["tasks"]),
        "scenario_family_count": len(scene["scenario_families"]),
        "scenario_eval_run_count": matrix["scenario_eval_run_count"],
        "required_recording_views": list(REQUIRED_RECORDING_VIEWS),
        "artifacts": {
            "research_manifest": "mujoco_scene_asset_research.json",
            "recording_plan": "recording_plan.json",
            "conversion_plan": "external_scene_conversion_plan.json",
            "scene_materialization_manifest": "external_scene_materialization_manifest.json",
            "mujoco_scene_glb": "../worldlabs_assets/scene.glb" if materialization_ready else None,
            "asset_inventory": "external_scene_asset_inventory.json",
            "scenario_variation_instances": "scenario_variation_instances.json",
            "scenario_eval_matrix": matrix_path.name,
            "task_cards": "../robot_eval_dataset/task_cards.json",
            "scenario_cards": "../robot_eval_dataset/scenario_cards.json",
            "scenario_family_library": "../robot_eval_dataset/scenario_family_library.json",
        },
        "next_action": (
            "Convert AWS Gazebo DAE visual/collision meshes into a MuJoCo scene XML, "
            "then run the MuJoCo G1 simulator command against the generated 500-row matrix."
        ),
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    packet["deterministic_fingerprint"] = _sha_payload(
        {
            "scene_asset_id": scene_asset_id,
            "scenario_eval_run_count": packet["scenario_eval_run_count"],
            "task_count": packet["task_count"],
            "scenario_family_count": packet["scenario_family_count"],
        }
    )
    packet_path = packet_dir / "mujoco_scene_scenario_packet.json"
    write_json(packet_path, packet)
    write_text(
        packet_dir / "mujoco_scene_packet_runbook.md",
        _runbook_text(
            capture_root=resolved_capture_root,
            matrix_path=matrix_path,
            local_asset_root=str(local_root) if local_root else None,
        ),
    )

    # Validate the generated folder follows the capture-root contract we expect downstream.
    resolve_local_capture_context(resolved_capture_root)
    return {
        "schema_version": "mujoco_scene_scenario_packet_result.v1",
        "status": packet["status"],
        "capture_root": str(resolved_capture_root),
        "packet_dir": str(packet_dir),
        "packet_path": str(packet_path),
        "scenario_eval_matrix_path": str(matrix_path),
        "scenario_eval_run_count": matrix["scenario_eval_run_count"],
        "task_count": packet["task_count"],
        "scenario_family_count": packet["scenario_family_count"],
        "local_asset_inventory_status": inventory.get("status"),
        "scene_materialization_status": materialization_status,
        "scene_glb_path": materialization.get("compatibility_scene_glb_path"),
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--capture-root", type=Path, default=None)
    parser.add_argument("--scene-asset-id", default=DEFAULT_SCENE_ID)
    parser.add_argument("--scenario-count", type=int, default=DEFAULT_SCENARIO_COUNT)
    parser.add_argument("--local-asset-root", type=Path, default=None)
    args = parser.parse_args(argv)
    result = build_mujoco_scene_scenario_packet(
        output_dir=args.output_dir,
        capture_root=args.capture_root,
        scene_asset_id=args.scene_asset_id,
        scenario_count=args.scenario_count,
        local_asset_root=args.local_asset_root,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "capture_root": result["capture_root"],
                "packet_path": result["packet_path"],
                "scenario_eval_matrix_path": result["scenario_eval_matrix_path"],
                "scenario_eval_run_count": result["scenario_eval_run_count"],
                "local_asset_inventory_status": result["local_asset_inventory_status"],
                "scene_materialization_status": result["scene_materialization_status"],
                "scene_glb_path": result["scene_glb_path"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main().
    raise SystemExit(main())
