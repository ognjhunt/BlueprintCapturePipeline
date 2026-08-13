"""Bind paired-target repaired appearance to native task inputs.

This is a task-neutral admission packet for one to five replacement objects.
It verifies exact bytes and static structure, but deliberately leaves native
Isaac import, camera rendering, reachability, controls, and policy execution
unqualified until their respective runtime receipts exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
from typing import Any, Mapping, Sequence
import zipfile

from .decision_evidence_contracts import canonical_digest
from .dual_task_rehearsal_contract import MAX_REPLACEMENT_OBJECTS
from .nurec_openusd_packaging import validate_safe_usdz_archive
from .replacement_asset_frame_registration import (
    validate_replacement_asset_frame_registration,
)


SCHEMA_VERSION = "paired_target_native_preflight.v1"
APPEARANCE_SCHEMA = "public_scene_artifixer3d_native_appearance_export.v1"
DUAL_INPUT_SCHEMA = "public_scene_artifixer3d_dual_target_inputs.v1"
CAD_RECEIPT_SCHEMA = "simready_graph_asset_receipt.v1"
CAD_STATIC_SCHEMA = "simready_graph_asset_static_qualification.v1"
REGISTERED_ASSET_SCHEMA = "registered_replacement_asset.v1"
SCENARIO_SCHEMA = "third_scene_task_scenario_suite.v1"
FROZEN_CANDIDATES = ("pi05_droid", "groot_n17_droid")


class PairedTargetNativePreflightError(ValueError):
    """A stable, fail-closed native preflight failure."""


def _read(path: str | Path, code: str) -> tuple[Path, dict[str, Any]]:
    candidate = Path(path).expanduser().resolve()
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PairedTargetNativePreflightError(code) from exc
    if candidate.is_symlink() or not isinstance(value, dict):
        raise PairedTargetNativePreflightError(code)
    return candidate, value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise PairedTargetNativePreflightError("native_preflight_file_invalid")
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _digest_valid(value: Mapping[str, Any], field: str) -> bool:
    return value.get(field) == canonical_digest(value, digest_field=field)


def _bound_path(parent: Path, record: Mapping[str, Any], code: str) -> Path:
    relative = str(record.get("relative_path") or "")
    if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise PairedTargetNativePreflightError(code)
    path = (parent / relative).resolve()
    if parent.resolve() not in path.parents or path.is_symlink() or not path.is_file():
        raise PairedTargetNativePreflightError(code)
    if path.stat().st_size != record.get("size_bytes") or _sha256(path) != record.get("sha256"):
        raise PairedTargetNativePreflightError(code)
    return path


def _validate_usdz_contract(path: Path, contract: Any) -> list[dict[str, Any]]:
    if (
        not isinstance(contract, Mapping)
        or contract.get("compression") != "stored"
        or contract.get("payload_alignment_bytes") != 64
        or contract.get("all_payload_offsets_aligned") is not True
        or contract.get("nurec_gzip_mtime_normalized_to_zero") is not True
        or not isinstance(contract.get("members"), list)
    ):
        raise PairedTargetNativePreflightError("native_preflight_usdz_contract_invalid")
    rows: list[dict[str, Any]] = []
    try:
        with path.open("rb") as raw, zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            if [info.filename for info in infos] != [
                "default.usda",
                "repaired_scene.nurec",
                "gauss.usda",
            ]:
                raise ValueError
            for info in infos:
                raw.seek(info.header_offset)
                header = raw.read(30)
                fields = struct.unpack("<IHHHHHIIIHH", header)
                data_offset = info.header_offset + 30 + fields[-2] + fields[-1]
                body = archive.read(info)
                if (
                    info.compress_type != zipfile.ZIP_STORED
                    or data_offset % 64
                    or (
                        info.filename.endswith(".nurec")
                        and (body[:3] != b"\x1f\x8b\x08" or body[4:8] != b"\0" * 4)
                    )
                ):
                    raise ValueError
                rows.append(
                    {
                        "filename": info.filename,
                        "size_bytes": info.file_size,
                        "data_offset_bytes": data_offset,
                        "sha256": "sha256:" + hashlib.sha256(body).hexdigest(),
                    }
                )
    except (OSError, ValueError, zipfile.BadZipFile, struct.error) as exc:
        raise PairedTargetNativePreflightError("native_preflight_usdz_contract_invalid") from exc
    if rows != contract["members"]:
        raise PairedTargetNativePreflightError("native_preflight_usdz_contract_invalid")
    return rows


def materialize_paired_target_native_preflight(
    *,
    scene_id: str,
    task_records: Sequence[Mapping[str, Any]],
    collision_scene_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Validate and seal exact native inputs for one to five tasks."""

    if not 1 <= len(task_records) <= MAX_REPLACEMENT_OBJECTS:
        raise PairedTargetNativePreflightError("native_preflight_task_count_invalid")
    collision = Path(collision_scene_path).expanduser().resolve()
    collision_record = _record(collision)
    tasks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in task_records:
        task_id = str(raw.get("task_id") or "")
        if not task_id or task_id in seen:
            raise PairedTargetNativePreflightError("native_preflight_task_id_invalid")
        seen.add(task_id)
        appearance_path, appearance = _read(
            raw.get("appearance_export_receipt_path", ""),
            "native_preflight_appearance_receipt_invalid",
        )
        if (
            appearance.get("schema_version") != APPEARANCE_SCHEMA
            or not _digest_valid(appearance, "export_digest")
            or appearance.get("native_import_qualified") is not False
            or appearance.get("generated_output_is_capture_or_physical_evidence") is not False
        ):
            raise PairedTargetNativePreflightError("native_preflight_appearance_receipt_invalid")
        usdz_record = appearance.get("isaac_nurec_usdz")
        if not isinstance(usdz_record, Mapping):
            raise PairedTargetNativePreflightError("native_preflight_usdz_invalid")
        usdz = Path(str(usdz_record.get("path") or "")).expanduser().resolve()
        if (
            usdz.is_symlink()
            or not usdz.is_file()
            or usdz.stat().st_size != usdz_record.get("size_bytes")
            or _sha256(usdz) != usdz_record.get("sha256")
        ):
            raise PairedTargetNativePreflightError("native_preflight_usdz_invalid")
        validate_safe_usdz_archive(usdz, task_id)
        usdz_members = _validate_usdz_contract(
            usdz, appearance.get("isaac_nurec_usdz_archive_contract")
        )

        dual_path, dual = _read(
            raw.get("dual_target_inputs_receipt_path", ""),
            "native_preflight_dual_inputs_invalid",
        )
        if (
            dual.get("schema_version") != DUAL_INPUT_SCHEMA
            or not _digest_valid(dual, "receipt_digest")
            or dual.get("publisher_scene_id") != scene_id
            or dual.get("selected_task_ids") != [task_id]
            or len(dual.get("tasks") or []) != 1
        ):
            raise PairedTargetNativePreflightError("native_preflight_dual_inputs_invalid")
        dual_task = dual["tasks"][0]
        if (
            dual_task.get("task_id") != task_id
            or dual_task.get("physical_camera_count") != 8
            or dual_task.get("camera_count") != 8
        ):
            raise PairedTargetNativePreflightError("native_preflight_cameras_invalid")
        scene_directory = Path(dual_task["scene_directory"]).resolve()
        trajectory = _bound_path(
            scene_directory,
            dual_task["review_trajectory"],
            "native_preflight_cameras_invalid",
        )
        camera_index = _bound_path(
            scene_directory,
            dual_task["camera_index"],
            "native_preflight_cameras_invalid",
        )
        camera_rows = json.loads(camera_index.read_text(encoding="utf-8"))
        camera_ids = [str(row.get("camera_id") or "") for row in camera_rows["frames"]]
        if len(camera_ids) != 8 or len(set(camera_ids)) != 8:
            raise PairedTargetNativePreflightError("native_preflight_cameras_invalid")

        cad_path, cad = _read(
            raw.get("simready_asset_receipt_path", ""),
            "native_preflight_cad_receipt_invalid",
        )
        static_path, static = _read(
            raw.get("simready_static_qualification_path", ""),
            "native_preflight_cad_static_invalid",
        )
        if (
            cad.get("schema_version") != CAD_RECEIPT_SCHEMA
            or not _digest_valid(cad, "receipt_digest")
            or cad.get("task_id") != task_id
            or cad.get("status") != "simready_candidate_authored"
            or cad.get("claim_boundary", {}).get("native_simulator_import_qualified") is not False
            or static.get("schema_version") != CAD_STATIC_SCHEMA
            or not _digest_valid(static, "receipt_digest")
            or static.get("task_id") != task_id
            or static.get("asset_id") != cad.get("asset_id")
            or static.get("authored_structure_statically_qualified") is not True
            or static.get("authoring_receipt", {}).get("receipt_digest")
            != cad.get("receipt_digest")
        ):
            raise PairedTargetNativePreflightError("native_preflight_cad_invalid")
        cad_usd = Path(cad["output_usd"]["path"]).expanduser().resolve()
        if _record(cad_usd)["sha256"] != cad["output_usd"]["sha256"]:
            raise PairedTargetNativePreflightError("native_preflight_cad_invalid")

        registered_path, registered = _read(
            raw.get("registered_replacement_asset_receipt_path", ""),
            "native_preflight_registered_asset_invalid",
        )
        if (
            registered.get("schema_version") != REGISTERED_ASSET_SCHEMA
            or registered.get("receipt_digest")
            != canonical_digest(registered, digest_field="receipt_digest")
            or registered.get("task_id") != task_id
            or registered.get("asset_id") != cad.get("asset_id")
            or registered.get("agent_authored_display_colors_preserved") is not True
            or registered.get("neutral_fallback_present") is not False
            or registered.get("deterministic_pose_composition_only") is not True
            or registered.get("geometry_generated_or_modified") is not False
        ):
            raise PairedTargetNativePreflightError("native_preflight_registered_asset_invalid")
        visual_usd = Path(registered["output_usd"]["path"]).expanduser().resolve()
        if _record(visual_usd) != registered["output_usd"]:
            raise PairedTargetNativePreflightError("native_preflight_registered_asset_invalid")
        if (
            static.get("replacement_usd") != registered.get("output_usd")
            or static.get("registered_replacement_asset", {}).get("receipt_digest")
            != registered.get("receipt_digest")
            or static.get("registered_visual_readback", {}).get(
                "asset_frame_registration_digest"
            )
            != registered.get("frame_registration", {}).get("registration_digest")
        ):
            raise PairedTargetNativePreflightError("native_preflight_registered_static_invalid")

        registration_path, registration = _read(
            raw.get("asset_frame_registration_path", ""),
            "native_preflight_asset_frame_registration_invalid",
        )
        try:
            registration = validate_replacement_asset_frame_registration(registration)
        except Exception as exc:
            raise PairedTargetNativePreflightError(
                "native_preflight_asset_frame_registration_invalid"
            ) from exc
        if (
            registration.get("scene_id") != scene_id
            or registration.get("task_id") != task_id
            or registration.get("asset_id") != cad.get("asset_id")
        ):
            raise PairedTargetNativePreflightError(
                "native_preflight_asset_frame_registration_invalid"
            )

        scenario_path, scenario = _read(
            raw.get("scenario_suite_path", ""),
            "native_preflight_scenario_invalid",
        )
        if (
            scenario.get("schema_version") != SCENARIO_SCHEMA
            or scenario.get("scene_id") != scene_id
            or scenario.get("task_id") != task_id
            or tuple(scenario.get("candidate_ids") or ()) != FROZEN_CANDIDATES
            or scenario.get("required_controls") != ["zero_action_negative", "scripted_positive"]
            or scenario.get("initial_execution_order") is None
            or len(scenario["initial_execution_order"]) != 2
            or scenario.get("suite_digest")
            != canonical_digest(scenario, digest_field="suite_digest")
        ):
            raise PairedTargetNativePreflightError("native_preflight_scenario_invalid")

        tasks.append(
            {
                "task_id": task_id,
                "asset_id": cad["asset_id"],
                "appearance_export_receipt": {
                    **_record(appearance_path),
                    "export_digest": appearance["export_digest"],
                },
                "isaac_nurec_usdz": {
                    **_record(usdz),
                    "archive_members": usdz_members,
                },
                "camera_trajectory": _record(trajectory),
                "camera_index": {
                    **_record(camera_index),
                    "camera_index_digest": camera_rows["camera_index_digest"],
                    "camera_ids": camera_ids,
                },
                "simready_asset_receipt": {
                    **_record(cad_path),
                    "receipt_digest": cad["receipt_digest"],
                },
                "simready_static_qualification": {
                    **_record(static_path),
                    "receipt_digest": static["receipt_digest"],
                },
                "simready_usd": _record(cad_usd),
                "registered_replacement_asset_receipt": {
                    **_record(registered_path),
                    "receipt_digest": registered["receipt_digest"],
                },
                "registered_static_qualification": {
                    **_record(static_path),
                    "receipt_digest": static["receipt_digest"],
                },
                "registered_replacement_usd": _record(visual_usd),
                "appearance_contract": {
                    "agent_authored_display_colors_preserved": True,
                    "generated_texture_maps_present": False,
                    "neutral_fallback_permitted": False,
                },
                "asset_frame_registration": {
                    **_record(registration_path),
                    "registration_digest": registration["registration_digest"],
                    "T_observed_world_axes_from_asset_local_axes": registered[
                        "T_observed_world_axes_from_asset_local_axes"
                    ],
                },
                "scenario_suite": {
                    **_record(scenario_path),
                    "suite_digest": scenario["suite_digest"],
                    "initial_execution_order": scenario["initial_execution_order"],
                },
                "native_import_qualified": False,
                "reachability_qualified": False,
                "controls_executed": False,
                "learned_policies_executed": False,
            }
        )

    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "native_inputs_preflighted_pending_native_import_and_controls",
        "program_id": "arm-decision-proof-v1",
        "scene_id": scene_id,
        "replacement_object_count": len(tasks),
        "maximum_replacement_objects": MAX_REPLACEMENT_OBJECTS,
        "collision_scene": collision_record,
        "tasks": tasks,
        "candidate_ids": list(FROZEN_CANDIDATES),
        "required_controls": ["zero_action_negative", "scripted_positive"],
        "native_isaac_import_executed": False,
        "generated_output_is_capture_or_physical_evidence": False,
        "claim_boundary": (
            "byte_and_static_structure_preflight_only_pending_native_isaac_import_"
            "camera_reachability_controls_and_policy_execution"
        ),
    }
    result["receipt_digest"] = canonical_digest(result, digest_field="receipt_digest")
    destination = Path(output_path).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise PairedTargetNativePreflightError("native_preflight_destination_exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    _, request = _read(args.request, "native_preflight_request_invalid")
    result = materialize_paired_target_native_preflight(
        scene_id=request["scene_id"],
        task_records=request["tasks"],
        collision_scene_path=request["collision_scene_path"],
        output_path=args.output,
    )
    print(json.dumps({"receipt_digest": result["receipt_digest"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
