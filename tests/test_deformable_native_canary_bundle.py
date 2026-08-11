from __future__ import annotations

import base64
import copy
import hashlib
import json
import struct
import zipfile
import zlib
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from blueprint_pipeline.composed_paired_entity_placement import (
    plan_composed_paired_entity_placement,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.deformable_native_canary_bundle import (
    BUNDLE_FILENAME,
    DYNAMIC_NATIVE_GATE_IDS,
    EXECUTION_REQUEST_SCHEMA_VERSION,
    RECEIPT_FILENAME,
    RETURN_MANIFEST_FILENAME,
    RETURN_MANIFEST_SCHEMA_VERSION,
    WORKER_CONTRACT_FILENAME,
    DeformableNativeCanaryBundleError,
    build_deformable_native_canary_bundle,
    verify_deformable_native_canary_bundle,
    verify_deformable_native_canary_return,
)
from blueprint_pipeline.deformable_native_capability_preflight import (
    build_deformable_native_capability_preflight,
)
from blueprint_pipeline.native_task_entity_asset_authoring_bundle import (
    RECEIPT_FILENAME as AUTHORING_RECEIPT_FILENAME,
)
from blueprint_pipeline.native_task_entity_asset_authoring_bundle import (
    build_native_task_entity_asset_authoring_bundle,
    materialize_native_asset_authoring_runtime_identity,
)
from blueprint_pipeline.native_task_entity_contract import (
    TASK_KIND_DEFORMABLE_TRANSFER,
    materialize_native_task_entity_contract,
)
from blueprint_pipeline.task_entity_asset_candidate import (
    materialize_task_entity_asset_candidate,
)
from blueprint_pipeline.trusted_execution_envelope import (
    TRUSTED_PUBLIC_KEY_SHA256_ENV,
    canonical_trusted_execution_envelope_bytes,
    materialize_trusted_execution_envelope,
    materialize_trusted_execution_payload,
    trusted_execution_signature_message,
)
from tests.test_deformable_native_capability_preflight import (
    _fixture as _preflight_fixture,
)
from tests.test_native_task_entity_asset_authoring_bundle import (
    _candidate as _base_candidate,
)
from tests.test_native_task_entity_asset_authoring_bundle import (
    _contract as _base_contract,
)
from tests.test_native_task_entity_asset_authoring_bundle import _entity as _base_entity
from tests.test_native_task_entity_asset_authoring_bundle import (
    _runtime_identity,
)


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def _solid_rgba_png(red: int, green: int, blue: int) -> bytes:
    pixel = bytes((red, green, blue, 255))
    scanlines = b"".join(b"\x00" + pixel * 2 for _ in range(2))
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 6, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(scanlines))
        + _png_chunk(b"IEND", b"")
    )


# The retained real H.264 fixture below decodes to one 2x2 black RGB sample.
_PNG = _solid_rgba_png(0, 0, 0)
_RED_PNG = _solid_rgba_png(255, 0, 0)


def _sha_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _write_json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _asset_bytes(role: str) -> tuple[str, bytes]:
    if role == "runtime_usd":
        return "usda", b'#usda 1.0\n( metersPerUnit = 1 )\ndef Xform "Asset" {}\n'
    if role == "texture":
        return "png", _PNG
    if role in {"material_definition", "physics_configuration"}:
        return "json", json.dumps(
            {
                "schema_version": "fixture.v1",
                "material" if role.startswith("material") else "solver": {},
            },
            sort_keys=True,
        ).encode()
    return (
        "stl",
        b"solid fixture\nfacet normal 0 0 1\n outer loop\n"
        b"  vertex 0 0 0\n  vertex 1 0 0\n  vertex 0 1 0\n"
        b" endloop\nendfacet\nendsolid fixture\n",
    )


def _real_candidate(
    tmp_path: Path, asset_class: str, *, raw_ply: bool = False
) -> tuple[dict, Path]:
    candidate, root = _base_candidate(tmp_path, asset_class)
    raw = copy.deepcopy(candidate)
    for field in ("status", "claims", "pending_gates", "physically_unresolved", "candidate_digest"):
        raw.pop(field, None)
    rows = []
    for record in raw["files"]:
        role = record["role"]
        suffix, content = _asset_bytes(role)
        if raw_ply and role == "rest_geometry":
            suffix = "stl"
            content = (
                b"ply\nformat ascii 1.0\nproperty float opacity\n"
                b"property float scale_0\nelement vertex 1\nend_header\n0 0\n"
            )
        relative = f"assets/{role}.{suffix}"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        rows.append(
            {
                "role": role,
                "path": relative,
                "sha256": _sha_bytes(content),
                "size_bytes": len(content),
            }
        )
    raw["files"] = rows
    return materialize_task_entity_asset_candidate(raw), root


def _placement() -> dict:
    return plan_composed_paired_entity_placement(
        support_regions=[
            {
                "support_region_id": "work_surface",
                "aabb_min_m": [0.0, 0.0, 0.0],
                "aabb_max_m": [3.0, 2.0, 0.0],
                "supports_entities": True,
                "supports_robot_base": True,
            }
        ],
        obstacle_aabbs=[],
        entity_specs=[
            {"entity_id": "cloth", "footprint_xy_m": [0.2, 0.2], "height_m": 0.05},
            {"entity_id": "basket", "footprint_xy_m": [0.3, 0.3], "height_m": 0.4},
        ],
        canonical_task_centers_m=[[0.0, 0.0, 0.0]],
        robot_spec={
            "base_footprint_xy_m": [0.4, 0.4],
            "base_clearance_height_m": 0.25,
            "reach_annulus_m": [0.4, 1.6],
        },
        minimum_separations_m={
            "canonical_region": 0.4,
            "entity_entity": 0.1,
            "entity_obstacle": 0.05,
            "robot_entity": 0.05,
            "robot_obstacle": 0.05,
            "support_edge": 0.05,
        },
        grid_spacing_m=0.5,
        frozen_seed=2026081001,
    )


def _contract_at_placement(cloth: dict, basket: dict, placement: dict) -> dict:
    base = _base_contract(cloth, basket)
    entities = copy.deepcopy(base["task_entities"])
    entities.extend(
        [
            _base_entity("counter_aux", "support_surface"),
            _base_entity("wall_aux", "obstacle"),
        ]
    )
    selected = {row["subject_id"]: row for row in placement["selection"]["entity_placements"]}
    robot_position = placement["selection"]["robot_base_placement"]["aabb_min_m"]
    for entity in entities:
        entity_id = entity["entity_id"]
        if entity_id in selected:
            entity["initial_state"]["pose_world"]["position_world_m"] = selected[entity_id][
                "center_world_m"
            ]
        elif entity["semantic_role"] == "robot":
            entity["initial_state"]["pose_world"]["position_world_m"] = robot_position
    return materialize_native_task_entity_contract(
        task_kind=TASK_KIND_DEFORMABLE_TRANSFER,
        task_entities=entities,
    )


def _camera(role: str) -> dict:
    return {
        "role": role,
        "policy_input": role in {"external", "wrist"},
        "scoring_input": False,
        "review_only": role == "overview",
        "pose_frame": "robot_body" if role == "wrist" else "world",
        "intrinsics": {
            "fx": 300.0,
            "fy": 300.0,
            "cx": 159.5,
            "cy": 89.5,
            "width": 320,
            "height": 180,
        },
    }


def _scene(contract: dict, placement: dict) -> dict:
    robot_entity = contract["semantic_role_index"]["robot"][0]
    robot_position = placement["selection"]["robot_base_placement"]["aabb_min_m"]
    plan = {
        "schema_version": "native_task_arena_scene_plan.v1",
        "scene_id": "unused-scene-fixture",
        "task_id": "deformable-transfer-fixture",
        "task_kind": TASK_KIND_DEFORMABLE_TRANSFER,
        "task_spec": {
            "deformable_entity_id": "cloth",
            "destination_entity_id": "basket",
            "robot_entity_id": robot_entity,
            "settle_window_samples": 4,
            "maximum_node_speed_mps": 0.02,
            "maximum_principal_strain": 0.25,
            "minimum_robot_clearance_m": 0.1,
            "maximum_receptacle_translation_drift_m": 0.01,
            "maximum_receptacle_rotation_drift_rad": 0.03,
        },
        "scenario": {
            "context_kind": "evaluation_cell",
            "cell_id": "canonical.seed_2026081001",
            "instance_digest": placement["receipt_digest"],
            "seed": 2026081001,
        },
        "robot": {
            "robot_id": "franka_panda",
            "base_pose_world": {
                "position_world_m": robot_position,
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "grasp_frame": {"kind": "body_midpoint", "body_names": ["left", "right"]},
            # Deliberately not the old hard-coded 8-D seam: the canary must project it.
            "action_seam": {
                "kind": "joint_position_with_gripper_and_mode",
                "arm_joint_count": 7,
                "action_dimension": 9,
            },
        },
        "cameras": [_camera(role) for role in ("external", "wrist", "overview")],
        "task_entities": contract["task_entities"],
        "task_entity_role_index": contract["semantic_role_index"],
        "task_entity_contract_digest": contract["contract_digest"],
        "plan_digest": "",
    }
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    return plan


_SCENE_USDA = b'#usda 1.0\n( metersPerUnit = 1 )\ndef Xform "RegisteredScene" {}\n'


def _execution(authoring_digest: str, scene_runtime_reference: dict) -> dict:
    return {
        "schema_version": EXECUTION_REQUEST_SCHEMA_VERSION,
        "run_id": "fixture-native-canary",
        "provider_id": "vast",
        "authority_binding": {
            "authority_receipt_digest": "sha256:" + "a" * 64,
            "authority_reference": "explicit-goal-authority-fixture",
        },
        "scene_runtime_reference": scene_runtime_reference,
        "resource_limits": {
            "maximum_paid_attempts": 1,
            "automatic_retry_count": 0,
            "attempt_spend_cap_usd": 1.0,
            "goal_total_spend_cap_usd": 20.0,
            "goal_spend_before_attempt_usd": 0.0,
            "ttl_seconds": 1200,
            "watchdog_seconds": 300,
            "allowed_active_instance_ids": [],
        },
        "visibility_thresholds": {
            "external": {
                "minimum_deformable_pixel_fraction": 0.01,
                "minimum_destination_pixel_fraction": 0.01,
            },
            "wrist": {
                "minimum_deformable_pixel_fraction": 0.01,
                "minimum_destination_pixel_fraction": 0.005,
            },
            "overview": {"minimum_motion_enclosure_fraction": 0.9},
        },
        "authoring_receipt_digest": authoring_digest,
        "execution_request_digest": "",
    }


def _scene_disclosure(package_sha256: str = "sha256:" + "b" * 64) -> dict:
    value = {
        "schema_version": "deformable_native_canary_scene_disclosure.v1",
        "package_sha256": package_sha256,
        "provider_terms_receipt_digest": "sha256:" + "3" * 64,
        "output_rights_receipt_digest": "sha256:" + "4" * 64,
        "provider_training_permitted": False,
        "retention_mode": "bounded_ephemeral_private_processing",
        "raw_source_digests": ["sha256:" + "6" * 64],
        "members": [
            {
                "path": "scene/registered_scene.usda",
                "content_kind": "derived_scene_usd",
                "size_bytes": len(_SCENE_USDA),
                "sha256": _sha_bytes(_SCENE_USDA),
                "derivation_receipt_digest": "sha256:" + "5" * 64,
            }
        ],
        "receipt_digest": "",
    }
    value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
    return value


def _scene_runtime_fixture(
    root: Path, *, member_path: str = "scene/registered_scene.usda", content: bytes = _SCENE_USDA
) -> tuple[Path, Path, dict]:
    root.mkdir(parents=True, exist_ok=True)
    package_path = root / "derived-scene.zip"
    with zipfile.ZipFile(package_path, "w") as archive:
        archive.writestr(member_path, content)
    package_sha256 = _sha_bytes(package_path.read_bytes())
    disclosure = _scene_disclosure(package_sha256)
    disclosure["members"] = [
        {
            "path": member_path,
            "content_kind": "derived_scene_usd",
            "size_bytes": len(content),
            "sha256": _sha_bytes(content),
            "derivation_receipt_digest": "sha256:" + "5" * 64,
        }
    ]
    disclosure["receipt_digest"] = canonical_digest(disclosure, digest_field="receipt_digest")
    disclosure_path = _write_json(root / "scene-disclosure.json", disclosure)
    reference = {
        "package_id": "derived-scene-package",
        "package_sha256": package_sha256,
        "disclosure_receipt_digest": disclosure["receipt_digest"],
    }
    return package_path, disclosure_path, reference


def _fixture(tmp_path: Path, *, raw_ply: bool = False) -> dict[str, Path | dict]:
    cloth, cloth_root = _real_candidate(tmp_path, "deformable_volume", raw_ply=raw_ply)
    basket, basket_root = _real_candidate(tmp_path, "rigid_receptacle")
    placement = _placement()
    contract = _contract_at_placement(cloth, basket, placement)
    preflight_request, observations = _preflight_fixture(tmp_path / "preflight-runtime")
    identity_raw = copy.deepcopy(_runtime_identity())
    identity_raw.pop("runtime_identity_digest")
    identity_raw["runtime_id"] = preflight_request["simulator_runtime_identity"]["runtime_id"]
    identity_raw["simulator"]["container_image"] = preflight_request["simulator_runtime_identity"][
        "container_image"
    ]
    identity_raw["python"] = {
        "python_tag": preflight_request["runtime_python"]["python_tag"],
        "abi_tag": preflight_request["runtime_python"]["abi_tag"],
        "platform_tag": preflight_request["runtime_python"]["platform_tags"][0],
    }
    authoring_dir = tmp_path / "authoring"
    authoring = build_native_task_entity_asset_authoring_bundle(
        output_dir=authoring_dir,
        task_entity_contract=contract,
        asset_candidates=[cloth, basket],
        asset_source_roots={"cloth": cloth_root, "basket": basket_root},
        runtime_identity=materialize_native_asset_authoring_runtime_identity(identity_raw),
        generated_at="2026-08-10T00:00:00Z",
    )
    matrix = build_deformable_native_capability_preflight(
        request=preflight_request, observations=observations
    )
    inputs = tmp_path / "inputs"
    scene_runtime_package, scene_runtime_disclosure, scene_runtime_reference = (
        _scene_runtime_fixture(tmp_path / "scene-runtime")
    )
    return {
        "authoring_receipt": authoring_dir / AUTHORING_RECEIPT_FILENAME,
        "scene": _write_json(inputs / "scene.json", _scene(contract, placement)),
        "preflight_request": _write_json(inputs / "preflight_request.json", preflight_request),
        "preflight_observations": _write_json(inputs / "preflight_observations.json", observations),
        "preflight_matrix": _write_json(inputs / "preflight_matrix.json", matrix),
        "placement": _write_json(inputs / "placement.json", placement),
        "execution": _write_json(
            inputs / "execution.json",
            _execution(authoring["receipt_digest"], scene_runtime_reference),
        ),
        "scene_runtime_package": scene_runtime_package,
        "scene_runtime_disclosure": scene_runtime_disclosure,
        "contract": contract,
    }


def _build(tmp_path: Path, fixture: dict[str, Path | dict] | None = None) -> tuple[dict, Path]:
    values = fixture or _fixture(tmp_path)
    output = tmp_path / "canary-package"
    receipt = build_deformable_native_canary_bundle(
        output_dir=output,
        authoring_bundle_receipt_path=values["authoring_receipt"],
        scene_plan_path=values["scene"],
        preflight_request_path=values["preflight_request"],
        preflight_observations_path=values["preflight_observations"],
        preflight_matrix_path=values["preflight_matrix"],
        placement_receipt_path=values["placement"],
        execution_request_path=values["execution"],
        scene_runtime_package_path=values["scene_runtime_package"],
        scene_runtime_disclosure_receipt_path=values["scene_runtime_disclosure"],
        generated_at="2026-08-10T01:00:00Z",
    )
    return receipt, output


def _worker(output: Path) -> dict:
    with zipfile.ZipFile(output / BUNDLE_FILENAME) as archive:
        return json.loads(
            archive.read(f"deformable_native_canary_input/{WORKER_CONTRACT_FILENAME}")
        )


def _report(value: dict) -> dict:
    value["report_digest"] = canonical_digest(value, digest_field="report_digest")
    return value


def _receipt(value: dict, digest_field: str) -> dict:
    value[digest_field] = canonical_digest(value, digest_field=digest_field)
    return value


_REAL_H264_MP4_BASE64 = (
    "AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDEAAAMWbW9vdgAAAGxtdmhkAAAAAAAA"
    "AAAAAAAAAAAD6AAAACgAAQAAAQAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAA"
    "AAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgAAAkF0cmFrAAAAXHRraGQAAAAD"
    "AAAAAAAAAAAAAAABAAAAAAAAACgAAAAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAABAAAA"
    "AAAAAAAAAAAAAABAAAAAAAIAAAACAAAAAAAkZWR0cwAAABxlbHN0AAAAAAAAAAEAAAAoAAAAAAAB"
    "AAAAAAG5bWRpYQAAACBtZGhkAAAAAAAAAAAAAAAAAAAyAAAAAgBVxAAAAAAALWhkbHIAAAAAAAAA"
    "AHZpZGUAAAAAAAAAAAAAAABWaWRlb0hhbmRsZXIAAAABZG1pbmYAAAAUdm1oZAAAAAEAAAAAAAAA"
    "AAAAACRkaW5mAAAAHGRyZWYAAAAAAAAAAQAAAAx1cmwgAAAAAQAAASRzdGJsAAAAwHN0c2QAAAAA"
    "AAAAAQAAALBhdmMxAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAAAAAIAAgBIAAAASAAAAAAAAAABFUxh"
    "dmM2MS4xOS4xMDEgbGlieDI2NAAAAAAAAAAAAAAAGP//AAAANmF2Y0MBZAAK/+EAGWdkAAqs2V+I"
    "iMBEAAADAAQAAAMAyDxIllgBAAZo6+PLIsD9+PgAAAAAEHBhc3AAAAABAAAAAQAAABRidHJ0AAAA"
    "AAACKegAAAAAAAAAGHN0dHMAAAAAAAAAAQAAAAEAAAIAAAAAHHN0c2MAAAAAAAAAAQAAAAEAAAAB"
    "AAAAAQAAABRzdHN6AAAAAAAAAsUAAAABAAAAFHN0Y28AAAAAAAAAAQAAA0YAAABhdWR0YQAAAFlt"
    "ZXRhAAAAAAAAACFoZGxyAAAAAAAAAABtZGlyYXBwbAAAAAAAAAAAAAAAACxpbHN0AAAAJKl0b28A"
    "AAAcZGF0YQAAAAEAAAAATGF2ZjYxLjcuMTAwAAAACGZyZWUAAALNbWRhdAAAAq4GBf//qtxF6b3m"
    "2Ui3lizYINkj7u94MjY0IC0gY29yZSAxNjQgcjMxMDggMzFlMTlmOSAtIEguMjY0L01QRUctNCBB"
    "VkMgY29kZWMgLSBDb3B5bGVmdCAyMDAzLTIwMjMgLSBodHRwOi8vd3d3LnZpZGVvbGFuLm9yZy94"
    "MjY0Lmh0bWwgLSBvcHRpb25zOiBjYWJhYz0xIHJlZj0zIGRlYmxvY2s9MTowOjAgYW5hbHlzZT0w"
    "eDM6MHgxMTMgbWU9aGV4IHN1Ym1lPTcgcHN5PTEgcHN5X3JkPTEuMDA6MC4wMiBtaXhlZF9yZWY9"
    "MSBtZV9yYW5nZT0xNiBjaHJvbWFfbWU9MSB0cmVsbGlzPTEgOHg4ZGN0PTEgY3FtPTAgZGVhZHpv"
    "bmU9MjEsMTEgZmFzdF9wc2tpcD0xIGNocm9tYV9xcF9vZmZzZXQ9LTIgdGhyZWFkcz0xIGxvb2th"
    "aGVhZF90aHJlYWRzPTEgc2xpY2VkX3RocmVhZHM9MCBucj0wIGRlY2ltYXRlPTEgaW50ZXJsYWNl"
    "ZD0wIGJsdXJheV9jb21wYXQ9MCBjb25zdHJhaW5lZF9pbnRyYT0wIGJmcmFtZXM9MyBiX3B5cmFt"
    "aWQ9MiBiX2FkYXB0PTEgYl9iaWFzPTAgZGlyZWN0PTEgd2VpZ2h0Yj0xIG9wZW5fZ29wPTAgd2Vp"
    "Z2h0cD0yIGtleWludD0yNTAga2V5aW50X21pbj0yNSBzY2VuZWN1dD00MCBpbnRyYV9yZWZyZXNo"
    "PTAgcmNfbG9va2FoZWFkPTQwIHJjPWNyZiBtYnRyZWU9MSBjcmY9MjMuMCBxY29tcD0wLjYwIHFw"
    "bWluPTAgcXBtYXg9NjkgcXBzdGVwPTQgaXBfcmF0aW89MS40MCBhcT0xOjEuMDAAgAAAAA9liIQA"
    "K//+9nN8CmttsYE="
)


def _mp4() -> bytes:
    return base64.b64decode(_REAL_H264_MP4_BASE64)


def _return_bundle(
    root: Path,
    *,
    package: dict,
    output: Path,
    arm_delta: float = 0.2,
    post_instances: list[dict] | None = None,
    corrupt_frame_path: bool = False,
    frame_content: bytes = _PNG,
) -> Path:
    worker = _worker(output)
    members: dict[str, bytes] = {}
    artifacts: list[dict] = []

    def add(role: str, path: str, content: bytes) -> None:
        members[path] = content
        artifacts.append(
            {"role": role, "path": path, "size_bytes": len(content), "sha256": _sha_bytes(content)}
        )

    camera_results: dict[str, dict] = {}
    for role in ("external", "wrist", "overview"):
        frame_path = f"native_return/media/{role}/frame_000000.png"
        frame_manifest = {
            "schema_version": "deformable_native_canary_frame_manifest.v1",
            "role": role,
            "frames": [
                {
                    "path": frame_path
                    + (".missing" if corrupt_frame_path and role == "wrist" else ""),
                    "sha256": _sha_bytes(frame_content),
                    "size_bytes": len(frame_content),
                    "width": 2,
                    "height": 2,
                    "timestamp_ns": 1_000_000,
                }
            ],
            "manifest_digest": "",
        }
        frame_manifest["manifest_digest"] = canonical_digest(
            frame_manifest, digest_field="manifest_digest"
        )
        video = _mp4()
        ffprobe = {
            "schema_version": "deformable_native_canary_ffprobe_report.v1",
            "video_sha256": _sha_bytes(video),
            "ffprobe_binary_sha256": "sha256:" + "d" * 64,
            "ffprobe_version": "ffprobe fixture 1.0",
            "streams": [{"codec_type": "video", "codec_name": "h264", "width": 2, "height": 2}],
            "report_digest": "",
        }
        ffprobe["report_digest"] = canonical_digest(ffprobe, digest_field="report_digest")
        add(f"camera_frame:{role}:000000", frame_path, frame_content)
        add(
            f"camera_manifest:{role}",
            f"native_return/media/{role}/frames.json",
            json.dumps(frame_manifest, sort_keys=True).encode(),
        )
        add(f"camera_video:{role}", f"native_return/media/{role}/review.mp4", video)
        add(
            f"camera_ffprobe:{role}",
            f"native_return/media/{role}/ffprobe.json",
            json.dumps(ffprobe, sort_keys=True).encode(),
        )
        camera_results[role] = {
            "frame_manifest_digest": frame_manifest["manifest_digest"],
            "video_sha256": _sha_bytes(video),
            "ffprobe_report_digest": ffprobe["report_digest"],
        }

    blank = _report(
        {
            "schema_version": "deformable_native_canary_blank_stage_report.v1",
            "stage_id": "blank_stage_asset_runtime",
            "worker_contract_digest": worker["worker_contract_digest"],
            "started_at": "2026-08-10T01:10:00Z",
            "finished_at": "2026-08-10T01:11:00Z",
            "asset_readbacks": [
                {
                    "entity_id": row["entity_id"],
                    "candidate_digest": row["candidate_digest"],
                    "runtime_asset_sha256": next(
                        item["sha256"]
                        for item in row["staged_files"]
                        if item["role"] == "runtime_usd"
                    ),
                    "applied_schemas": row["operation"].get("required_schemas", []),
                    "observed_prim_type": row["operation"]["expected_prim_type"],
                    "cooked_element_count": 12,
                }
                for row in worker["authoring_operations"]
            ],
            "device_execution": {
                "cuda_device_uuid": "GPU-fixture",
                "physx_gpu_step_count": 10,
                "warp_launch_count": 2,
            },
            "reset_readback": {
                "initial_nodal_state_sha256": "sha256:" + "1" * 64,
                "first_reset_nodal_state_sha256": "sha256:" + "1" * 64,
                "second_reset_nodal_state_sha256": "sha256:" + "1" * 64,
                "post_start_direct_object_state_write_count": 0,
            },
            "report_digest": "",
        }
    )
    task_spec = worker["task_identity"]["task_spec"]
    scene = _report(
        {
            "schema_version": "deformable_native_canary_scene_stage_report.v1",
            "stage_id": "scene_bound_native_execution",
            "worker_contract_digest": worker["worker_contract_digest"],
            "started_at": "2026-08-10T01:11:00Z",
            "finished_at": "2026-08-10T01:12:00Z",
            "worker_reported_success": True,
            "contact_readback": {
                "action_trace": [
                    {"step": 0, "action": [0.0] * 9},
                    {"step": 1, "action": [0.1] * 9},
                ],
                "robot_state_trace": [
                    {
                        "step": 0,
                        "arm_joint_positions_rad": [0.0] * 7,
                        "gripper_joint_positions_rad": [0.0, 0.0],
                    },
                    {
                        "step": 1,
                        "arm_joint_positions_rad": [arm_delta] + [0.0] * 6,
                        "gripper_joint_positions_rad": [0.1, 0.1],
                    },
                ],
                "native_contact_events": [
                    {
                        "step": 1,
                        "body_a": "robot_gripper",
                        "body_b": "cloth",
                        "normal_force_n": 1.0,
                    }
                ],
                "phase_samples": [
                    {"phase": "lift", "deformable_height_delta_m": 0.05},
                    {"phase": "release", "gripper_deformable_contact_count": 0},
                    {"phase": "retreat", "robot_deformable_clearance_m": 0.2},
                ],
                "hidden_attachment_events": [],
                "post_start_direct_object_state_write_events": [],
            },
            "deformable_settle_readback": {
                "settle_window_sample_count": task_spec["settle_window_samples"],
                "maximum_node_speed_mps": task_spec["maximum_node_speed_mps"],
                "maximum_principal_strain": task_spec["maximum_principal_strain"],
                "nan_count": 0,
                "solver_divergence_count": 0,
            },
            "applied_parameter_readback": {
                **{
                    key: worker["input_bindings"][key]
                    for key in (
                        "scene_plan_digest",
                        "placement_receipt_digest",
                        "authoring_receipt_digest",
                        "execution_request_digest",
                    )
                },
                "entity_ids": sorted(
                    row["entity_id"] for row in worker["task_entity_contract"]["task_entities"]
                ),
            },
            "entity_readbacks": [
                {
                    "entity_id": row["entity_id"],
                    "runtime_asset_sha256": row["runtime_asset"]["sha256"],
                    "pose_world": row["initial_state"]["pose_world"],
                }
                for row in worker["task_entity_contract"]["task_entities"]
            ],
            "robot_readback": {
                "robot_id": worker["robot_contract"]["robot_id"],
                "action_seam": worker["robot_contract"]["action_seam"],
                "base_pose_world": worker["robot_contract"]["base_pose_world"],
            },
            "receptacle_readback": {
                "initial_penetration_count": 0,
                "support_contact_event_count": 2,
                "maximum_translation_drift_m": 0.0,
                "maximum_rotation_drift_rad": 0.0,
            },
            "ik_readbacks": [
                {"phase": phase, "solution_count": 1, "collision_count": 0}
                for phase in (
                    "pregrasp",
                    "grasp",
                    "lift",
                    "transport",
                    "deposit",
                    "release",
                    "retreat",
                    "recovery",
                )
            ],
            "policy_adapter_readbacks": [
                {
                    "candidate_id": row["candidate_id"],
                    "adapter_sha256": row["adapter_sha256"],
                    "checkpoint_identity_digest": row["checkpoint_identity_digest"],
                    "load_count": 1,
                    "preprocess_call_count": 1,
                    "action_adapter_call_count": 1,
                    "load_receipt_sha256": "sha256:" + "7" * 64,
                    "preprocessed_observation_sha256": "sha256:" + "8" * 64,
                    "sample_action": [0.0] * 9,
                }
                for row in worker["policy_identities"]
            ],
            "camera_readbacks": [
                {
                    "role": role,
                    **camera_results[role],
                    "intrinsics": next(
                        row["intrinsics"]
                        for row in worker["camera_contracts"]
                        if row["role"] == role
                    ),
                    "pose_frame": next(
                        row["pose_frame"]
                        for row in worker["camera_contracts"]
                        if row["role"] == role
                    ),
                    "camera_calibration_digest": "sha256:" + "e" * 64,
                    "synchronized_frame_count": 1,
                    **(
                        {
                            "minimum_deformable_pixel_fraction": 0.1,
                            "minimum_destination_pixel_fraction": 0.1,
                        }
                        if role != "overview"
                        else {"minimum_motion_enclosure_fraction": 0.95}
                    ),
                }
                for role in ("external", "wrist", "overview")
            ],
            "report_digest": "",
        }
    )
    add(
        "blank_stage_report",
        "native_return/stages/blank.json",
        json.dumps(blank, sort_keys=True).encode(),
    )
    add(
        "scene_stage_report",
        "native_return/stages/scene.json",
        json.dumps(scene, sort_keys=True).encode(),
    )

    limits = worker["paid_allocator_requirements"]["limits"]
    binding = {
        "run_id": worker["run_id"],
        "provider_id": "vast",
        "canary_package_receipt_digest": package["receipt_digest"],
        "canary_package_sha256": package["bundle_sha256"],
        "attempt_spend_cap_usd": limits["attempt_spend_cap_usd"],
        "ttl_seconds": limits["ttl_seconds"],
        "watchdog_seconds": limits["watchdog_seconds"],
        "allowed_active_vast_instance_ids": [],
    }
    admission = {
        "schema_version": "paid_lane_admission.v1",
        "status": "admitted",
        "resource_class": "gpu_canary",
        "blockers": [],
        "provider_mutations_performed": 0,
        "raw_secret_values_recorded": False,
        "allocation_binding": binding,
        "allocation_binding_digest": canonical_digest(binding),
    }
    admission_bytes = json.dumps(admission, sort_keys=True).encode()
    add("provider_admission", "native_return/provider/admission.json", admission_bytes)
    pre_response = json.dumps({"instances": []}, sort_keys=True).encode()
    zero_response = json.dumps(
        {"instances": [] if post_instances is None else post_instances}, sort_keys=True
    ).encode()
    billing_response = json.dumps(
        {"instance_id": 12345, "charged_usd": 0.25}, sort_keys=True
    ).encode()
    with zipfile.ZipFile(output / BUNDLE_FILENAME) as package_archive:
        scene_disclosure_bytes = package_archive.read(
            "deformable_native_canary_input/scene_runtime_disclosure_receipt.json"
        )
    add(
        "prelaunch_inventory_response",
        "native_return/provider/prelaunch_inventory_response.json",
        pre_response,
    )
    add(
        "provider_zero_inventory_response",
        "native_return/provider/provider_zero_inventory_response.json",
        zero_response,
    )
    add(
        "billing_response",
        "native_return/provider/billing_response.json",
        billing_response,
    )
    add(
        "scene_disclosure_receipt",
        "native_return/provider/scene_disclosure_receipt.json",
        scene_disclosure_bytes,
    )
    provider_receipts = {
        "prelaunch_inventory": _receipt(
            {
                "schema_version": "deformable_native_canary_vast_inventory.v1",
                "provider": "vast",
                "observed_at": "2026-08-10T01:09:00Z",
                "response_sha256": _sha_bytes(pre_response),
                "parser_id": (
                    "blueprint_pipeline.deformable_native_canary_bundle:vast_inventory_response:v1"
                ),
                "active_instance_ids": [],
                "inventory_digest": "",
            },
            "inventory_digest",
        ),
        "allocation_receipt": _receipt(
            {
                "schema_version": "deformable_native_canary_allocation.v1",
                "provider": "vast",
                "run_id": worker["run_id"],
                "instance_id": 12345,
                "created_at": "2026-08-10T01:09:20Z",
                "package_receipt_digest": package["receipt_digest"],
                "admission_sha256": _sha_bytes(admission_bytes),
                "attempt_ordinal": 1,
                "retry_count": 0,
                "allocation_digest": "",
            },
            "allocation_digest",
        ),
        "watchdog_receipt": _receipt(
            {
                "schema_version": "deformable_native_canary_watchdog.v1",
                "instance_id": 12345,
                "armed_at": "2026-08-10T01:09:10Z",
                "deadline_at": "2026-08-10T01:29:00Z",
                "watchdog_seconds": 300,
                "watchdog_process_pid": 999,
                "watchdog_command_sha256": "sha256:" + "9" * 64,
                "progress_events": [
                    {"observed_at": value}
                    for value in (
                        "2026-08-10T01:10:00Z",
                        "2026-08-10T01:11:00Z",
                        "2026-08-10T01:12:00Z",
                        "2026-08-10T01:13:00Z",
                    )
                ],
                "watchdog_digest": "",
            },
            "watchdog_digest",
        ),
        "billing_receipt": _receipt(
            {
                "schema_version": "deformable_native_canary_billing.v1",
                "instance_id": 12345,
                "attempt_ordinal": 1,
                "retry_count": 0,
                "response_sha256": _sha_bytes(billing_response),
                "parser_id": (
                    "blueprint_pipeline.deformable_native_canary_bundle:vast_billing_response:v1"
                ),
                "billing_digest": "",
            },
            "billing_digest",
        ),
        "upload_receipt": _receipt(
            {
                "schema_version": "deformable_native_canary_upload.v1",
                "transmitted_artifacts": [
                    {"sha256": package["bundle_sha256"]},
                    {
                        "sha256": next(
                            row
                            for row in worker["stages"]
                            if row["stage_id"] == "scene_bound_native_execution"
                        )["scene_runtime_reference"]["package_sha256"]
                    },
                ],
                "scene_disclosure_receipt_digest": next(
                    row
                    for row in worker["stages"]
                    if row["stage_id"] == "scene_bound_native_execution"
                )["scene_runtime_reference"]["disclosure_receipt_digest"],
                "scene_disclosure_receipt_sha256": _sha_bytes(scene_disclosure_bytes),
                "upload_digest": "",
            },
            "upload_digest",
        ),
        "teardown_receipt": _receipt(
            {
                "schema_version": "deformable_native_canary_teardown.v1",
                "instance_id": 12345,
                "requested_at": "2026-08-10T01:13:00Z",
                "confirmed_at": "2026-08-10T01:13:10Z",
                "teardown_digest": "",
            },
            "teardown_digest",
        ),
        "provider_zero_inventory": _receipt(
            {
                "schema_version": "deformable_native_canary_vast_inventory.v1",
                "provider": "vast",
                "observed_at": "2026-08-10T01:13:20Z",
                "response_sha256": _sha_bytes(zero_response),
                "parser_id": (
                    "blueprint_pipeline.deformable_native_canary_bundle:vast_inventory_response:v1"
                ),
                "active_instance_ids": [
                    row["instance_id"] for row in ([] if post_instances is None else post_instances)
                ],
                "inventory_digest": "",
            },
            "inventory_digest",
        ),
    }
    for role, value in provider_receipts.items():
        add(role, f"native_return/provider/{role}.json", json.dumps(value, sort_keys=True).encode())

    manifest = {
        "schema_version": RETURN_MANIFEST_SCHEMA_VERSION,
        "run_id": worker["run_id"],
        "instance_id": 12345,
        "package_receipt_digest": package["receipt_digest"],
        "package_bundle_sha256": package["bundle_sha256"],
        "worker_contract_digest": worker["worker_contract_digest"],
        "artifacts": sorted(artifacts, key=lambda row: row["role"]),
        "manifest_digest": "",
    }
    manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
    members[RETURN_MANIFEST_FILENAME] = json.dumps(manifest, sort_keys=True).encode()
    root.mkdir(parents=True, exist_ok=True)
    path = root / "native_return.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in sorted(members.items()):
            archive.writestr(name, content)
    return path


def _rewrite_return_artifact(path: Path, role: str, content: bytes) -> None:
    with zipfile.ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(members[RETURN_MANIFEST_FILENAME])
    row = next(item for item in manifest["artifacts"] if item["role"] == role)
    members[row["path"]] = content
    row["size_bytes"] = len(content)
    row["sha256"] = _sha_bytes(content)
    manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
    members[RETURN_MANIFEST_FILENAME] = json.dumps(manifest, sort_keys=True).encode()
    with zipfile.ZipFile(path, "w") as archive:
        for name, value in sorted(members.items()):
            archive.writestr(name, value)


_LIFECYCLE_ROLES = {
    "provider_admission",
    "prelaunch_inventory",
    "prelaunch_inventory_response",
    "allocation_receipt",
    "watchdog_receipt",
    "billing_receipt",
    "billing_response",
    "upload_receipt",
    "teardown_receipt",
    "provider_zero_inventory",
    "provider_zero_inventory_response",
}


def _public_key_bytes(private_key: Ed25519PrivateKey) -> bytes:
    return private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def _lifecycle_digests(returned: Path) -> dict[str, str]:
    with zipfile.ZipFile(returned) as archive:
        manifest = json.loads(archive.read(RETURN_MANIFEST_FILENAME))
    by_role = {row["role"]: row["sha256"] for row in manifest["artifacts"]}
    return {role: by_role[role] for role in sorted(_LIFECYCLE_ROLES)}


def _signed_execution_arguments(
    root: Path,
    *,
    package: dict,
    output: Path,
    returned: Path,
    monkeypatch: pytest.MonkeyPatch,
    signer: Ed25519PrivateKey | None = None,
    configured_signer: Ed25519PrivateKey | None = None,
) -> dict:
    signer = signer or Ed25519PrivateKey.from_private_bytes(b"\x51" * 32)
    configured_signer = configured_signer or signer
    lifecycle = _lifecycle_digests(returned)
    expected = {
        "expected_nonce": "deformable-native-canary-nonce-20260810-0001",
        "expected_worker_entrypoint": (
            "python -m blueprint_pipeline.deformable_native_canary_worker"
        ),
        "expected_worker_source_tree_digest": "sha256:" + "c" * 64,
        "expected_worker_container_digest": "sha256:" + "d" * 64,
        "expected_instance_id": "12345",
        "expected_allocator_lifecycle_artifact_digests": lifecycle,
    }
    returned_bytes = returned.read_bytes()
    payload = materialize_trusted_execution_payload(
        nonce=expected["expected_nonce"],
        run_digest=package["run_digest"],
        package_digest=package["bundle_sha256"],
        execution_request_digest=package["execution_request_digest"],
        worker_entrypoint=expected["expected_worker_entrypoint"],
        worker_source_tree_digest=expected["expected_worker_source_tree_digest"],
        worker_container_digest=expected["expected_worker_container_digest"],
        instance_id=expected["expected_instance_id"],
        return_zip_sha256=_sha_bytes(returned_bytes),
        return_zip_size_bytes=len(returned_bytes),
        started_at="2026-08-10T01:09:00Z",
        ended_at="2026-08-10T01:13:20Z",
        allocator_lifecycle_artifact_digests=lifecycle,
    )
    signature = signer.sign(trusted_execution_signature_message(payload))
    envelope = materialize_trusted_execution_envelope(
        payload=payload,
        public_key_base64=base64.b64encode(_public_key_bytes(signer)).decode("ascii"),
        signature_base64=base64.b64encode(signature).decode("ascii"),
    )
    envelope_path = root / "trusted-execution-envelope.json"
    envelope_path.parent.mkdir(parents=True, exist_ok=True)
    envelope_path.write_bytes(canonical_trusted_execution_envelope_bytes(envelope))
    configured_fingerprint = _sha_bytes(_public_key_bytes(configured_signer))
    monkeypatch.setenv(TRUSTED_PUBLIC_KEY_SHA256_ENV, configured_fingerprint)
    return {"trusted_execution_envelope_path": envelope_path, **expected}


def _verify_signed_return(
    *,
    package: dict,
    output: Path,
    returned: Path,
    monkeypatch: pytest.MonkeyPatch,
    signer: Ed25519PrivateKey | None = None,
    configured_signer: Ed25519PrivateKey | None = None,
    argument_overrides: dict | None = None,
) -> dict:
    arguments = _signed_execution_arguments(
        returned.parent,
        package=package,
        output=output,
        returned=returned,
        monkeypatch=monkeypatch,
        signer=signer,
        configured_signer=configured_signer,
    )
    arguments.update(argument_overrides or {})
    return verify_deformable_native_canary_return(
        package_receipt_path=output / RECEIPT_FILENAME,
        return_bundle_path=returned,
        **arguments,
    )


def test_package_replays_inputs_and_cannot_authorize_paid_execution(tmp_path: Path) -> None:
    package, output = _build(tmp_path)

    assert package["status"] == "packaged_pending_canonical_paid_admission_and_native_return"
    assert package["authorization_capability_issued"] is False
    assert package["native_backend_refreeze_admitted"] is False
    worker = _worker(output)
    assert worker["robot_contract"]["action_seam"]["action_dimension"] == 9
    assert len(worker["task_entity_contract"]["semantic_role_index"]["support_surface"]) == 2
    assert len(worker["task_entity_contract"]["semantic_role_index"]["obstacle"]) == 2
    assert [row["gate_id"] for row in worker["dynamic_gates"]] == list(DYNAMIC_NATIVE_GATE_IDS)
    assert {row["status"] for row in worker["dynamic_gates"]} == {"pending_verified_native_return"}
    assert verify_deformable_native_canary_bundle(output / RECEIPT_FILENAME) == package


def test_signed_hand_authored_return_is_retained_typed_null_without_provider_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package, output = _build(tmp_path)
    returned = _return_bundle(tmp_path / "returned", package=package, output=output)

    result = _verify_signed_return(
        package=package,
        output=output,
        returned=returned,
        monkeypatch=monkeypatch,
    )

    assert result["status"] == "native_canary_return_retained_typed_null"
    assert result["trusted_execution_structural_verified"] is True
    assert result["worker_payload_gate_evidence_satisfied"] is True
    assert result["native_backend_refreeze_admitted"] is False
    assert result["charged_usd"] is None
    assert result["provider_zero_verified_from_empty_api_inventory"] is False
    assert "verifier_owned_provider_lifecycle_and_provider_zero_proof_missing" in result["blockers"]
    assert result["claim_boundary"]["physical_material_equivalence"] is False


def test_rehashed_return_after_runner_signature_is_untrusted_typed_null(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package, output = _build(tmp_path)
    returned = _return_bundle(tmp_path / "returned", package=package, output=output)
    arguments = _signed_execution_arguments(
        returned.parent,
        package=package,
        output=output,
        returned=returned,
        monkeypatch=monkeypatch,
    )
    _rewrite_return_artifact(
        returned,
        "blank_stage_report",
        json.dumps({"attacker": "rehashed"}, sort_keys=True).encode(),
    )

    result = verify_deformable_native_canary_return(
        package_receipt_path=output / RECEIPT_FILENAME,
        return_bundle_path=returned,
        **arguments,
    )

    assert result["native_backend_refreeze_admitted"] is False
    assert result["trusted_execution_structural_verified"] is False
    assert any("return_zip_sha256_mismatch" in blocker for blocker in result["blockers"])


def test_wrong_runner_key_is_untrusted_typed_null(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package, output = _build(tmp_path)
    returned = _return_bundle(tmp_path / "returned", package=package, output=output)
    attacker = Ed25519PrivateKey.from_private_bytes(b"\x61" * 32)
    configured = Ed25519PrivateKey.from_private_bytes(b"\x71" * 32)

    result = _verify_signed_return(
        package=package,
        output=output,
        returned=returned,
        monkeypatch=monkeypatch,
        signer=attacker,
        configured_signer=configured,
    )

    assert result["trusted_execution_structural_verified"] is False
    assert result["native_backend_refreeze_admitted"] is False
    assert any("public_key_not_authorized" in blocker for blocker in result["blockers"])


@pytest.mark.parametrize(
    ("argument_overrides", "expected_blocker"),
    [
        (
            {"expected_instance_id": "67890"},
            "trusted_execution_envelope_instance_id_mismatch",
        ),
        (
            {"expected_worker_container_digest": "sha256:" + "e" * 64},
            "trusted_execution_envelope_worker_container_digest_mismatch",
        ),
    ],
)
def test_wrong_verifier_owned_bindings_are_untrusted_typed_null(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    argument_overrides: dict,
    expected_blocker: str,
) -> None:
    package, output = _build(tmp_path)
    returned = _return_bundle(tmp_path / "returned", package=package, output=output)

    result = _verify_signed_return(
        package=package,
        output=output,
        returned=returned,
        monkeypatch=monkeypatch,
        argument_overrides=argument_overrides,
    )

    assert result["trusted_execution_structural_verified"] is False
    assert result["native_backend_refreeze_admitted"] is False
    assert any(expected_blocker in blocker for blocker in result["blockers"])


def test_recomputed_outer_preflight_digest_cannot_replace_full_replay(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    matrix_path = fixture["preflight_matrix"]
    matrix = json.loads(Path(matrix_path).read_text())
    matrix["static_checks"][0]["evidence"]["selected_robot_id"] = "counterfeit"
    matrix["receipt_digest"] = canonical_digest(matrix, digest_field="receipt_digest")
    _write_json(Path(matrix_path), matrix)

    with pytest.raises(
        DeformableNativeCanaryBundleError,
        match="preflight_full_replay_mismatch",
    ):
        _build(tmp_path, fixture)


def test_recomputed_outer_placement_digest_cannot_replace_planner_replay(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    placement_path = Path(fixture["placement"])
    placement = json.loads(placement_path.read_text())
    placement["selection"]["entity_placements"][0]["center_world_m"][0] += 0.01
    placement["receipt_digest"] = canonical_digest(placement, digest_field="receipt_digest")
    _write_json(placement_path, placement)

    with pytest.raises(
        DeformableNativeCanaryBundleError,
        match="placement_replay_mismatch",
    ):
        _build(tmp_path, fixture)


def test_raw_gaussian_ply_bytes_cannot_hide_under_derived_geometry_role(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, raw_ply=True)

    with pytest.raises(
        DeformableNativeCanaryBundleError,
        match="authoring_file_invalid:rest_geometry",
    ):
        _build(tmp_path, fixture)


@pytest.mark.parametrize(
    "smuggled",
    [
        b"ply\nformat ascii 1.0\nproperty float x\nend_header\n0\n",
        (
            b'#usda 1.0\ndef Xform "Cover" {}\n'
            b"ply\nformat ascii 1.0\nproperty float opacity\n"
            b"property float scale_0\nend_header\n0 0\n"
        ),
    ],
)
def test_scene_runtime_rejects_renamed_or_embedded_raw_ply(tmp_path: Path, smuggled: bytes) -> None:
    fixture = _fixture(tmp_path)
    package_path, disclosure_path, reference = _scene_runtime_fixture(
        tmp_path / "smuggled-scene-runtime",
        content=smuggled,
    )
    fixture["scene_runtime_package"] = package_path
    fixture["scene_runtime_disclosure"] = disclosure_path
    execution_path = Path(fixture["execution"])
    execution = json.loads(execution_path.read_text())
    execution["scene_runtime_reference"] = reference
    execution["execution_request_digest"] = ""
    _write_json(execution_path, execution)

    with pytest.raises(
        DeformableNativeCanaryBundleError,
        match="scene_runtime_member_invalid",
    ):
        _build(tmp_path, fixture)


def test_scene_runtime_package_swap_after_disclosure_fails_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    package_path = Path(fixture["scene_runtime_package"])
    package_path.write_bytes(package_path.read_bytes() + b"swapped")

    with pytest.raises(
        DeformableNativeCanaryBundleError,
        match="scene_runtime_disclosure_invalid",
    ):
        _build(tmp_path, fixture)


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (
            lambda value: value["resource_limits"].__setitem__("attempt_spend_cap_usd", True),
            "execution_resource_limits_invalid",
        ),
        (
            lambda value: value.__setitem__("raw_interiorgs_bytes_included", False),
            "execution_request_fields_unexpected",
        ),
    ],
)
def test_self_attested_transport_flags_and_boolean_caps_are_not_admission(
    tmp_path: Path, mutation, expected: str
) -> None:
    fixture = _fixture(tmp_path)
    execution_path = Path(fixture["execution"])
    execution = json.loads(execution_path.read_text())
    mutation(execution)
    _write_json(execution_path, execution)

    with pytest.raises(DeformableNativeCanaryBundleError, match=expected):
        _build(tmp_path, fixture)


def test_worker_success_boolean_without_arm_motion_is_retained_paid_null(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package, output = _build(tmp_path)
    returned = _return_bundle(tmp_path / "returned", package=package, output=output, arm_delta=0.0)

    result = _verify_signed_return(
        package=package,
        output=output,
        returned=returned,
        monkeypatch=monkeypatch,
    )

    assert result["status"] == "native_canary_return_retained_typed_null"
    assert result["native_backend_refreeze_admitted"] is False
    assert result["paid_null_retained"] is True
    assert "native_gate_blocked:dynamic_genuine_gripper_deformable_contact" in result["blockers"]


def test_media_paths_are_joined_to_real_png_and_h264_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package, output = _build(tmp_path)
    returned = _return_bundle(
        tmp_path / "returned",
        package=package,
        output=output,
        corrupt_frame_path=True,
    )

    with pytest.raises(
        DeformableNativeCanaryBundleError,
        match="camera_frame_invalid:wrist",
    ):
        _verify_signed_return(
            package=package,
            output=output,
            returned=returned,
            monkeypatch=monkeypatch,
        )


def test_valid_but_unrelated_h264_cannot_substitute_for_bound_lossless_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package, output = _build(tmp_path)
    returned = _return_bundle(
        tmp_path / "returned",
        package=package,
        output=output,
        frame_content=_RED_PNG,
    )

    with pytest.raises(
        DeformableNativeCanaryBundleError,
        match="camera_video_derivation_invalid:external",
    ):
        _verify_signed_return(
            package=package,
            output=output,
            returned=returned,
            monkeypatch=monkeypatch,
        )


def test_fake_empty_or_nonempty_worker_inventory_never_establishes_provider_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package, output = _build(tmp_path)
    returned = _return_bundle(
        tmp_path / "returned",
        package=package,
        output=output,
        post_instances=[{"instance_id": 12345}],
    )

    result = _verify_signed_return(
        package=package,
        output=output,
        returned=returned,
        monkeypatch=monkeypatch,
    )

    assert result["provider_zero_verified_from_empty_api_inventory"] is False
    assert result["native_backend_refreeze_admitted"] is False
    assert "verifier_owned_provider_lifecycle_and_provider_zero_proof_missing" in result["blockers"]


def test_normalized_zero_receipt_cannot_hide_nonzero_raw_provider_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package, output = _build(tmp_path)
    returned = _return_bundle(tmp_path / "returned", package=package, output=output)
    _rewrite_return_artifact(
        returned,
        "provider_zero_inventory_response",
        json.dumps({"instances": [{"instance_id": 67890}]}, sort_keys=True).encode(),
    )

    result = _verify_signed_return(
        package=package,
        output=output,
        returned=returned,
        monkeypatch=monkeypatch,
    )

    assert result["provider_zero_verified_from_empty_api_inventory"] is False
    assert result["native_backend_refreeze_admitted"] is False


def test_outer_package_rehash_cannot_hide_nested_disclosure_tamper(tmp_path: Path) -> None:
    package, output = _build(tmp_path)
    bundle = output / BUNDLE_FILENAME
    with zipfile.ZipFile(bundle) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    worker_name = f"deformable_native_canary_input/{WORKER_CONTRACT_FILENAME}"
    worker = json.loads(members[worker_name])
    worker["authorization_capability_issued"] = True
    worker["worker_contract_digest"] = canonical_digest(
        worker, digest_field="worker_contract_digest"
    )
    members[worker_name] = json.dumps(worker, sort_keys=True).encode()
    with zipfile.ZipFile(bundle, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    receipt_path = output / RECEIPT_FILENAME
    receipt = json.loads(receipt_path.read_text())
    receipt["bundle_size_bytes"] = bundle.stat().st_size
    receipt["bundle_sha256"] = _sha_bytes(bundle.read_bytes())
    receipt["worker_contract_digest"] = worker["worker_contract_digest"]
    receipt["receipt_digest"] = canonical_digest(
        {
            key: value
            for key, value in receipt.items()
            if key not in {"bundle_path", "receipt_digest"}
        }
    )
    _write_json(receipt_path, receipt)

    with pytest.raises(
        DeformableNativeCanaryBundleError,
        match="disclosure_payload_identity_mismatch|worker_contract_replay_invalid",
    ):
        verify_deformable_native_canary_bundle(receipt_path)
