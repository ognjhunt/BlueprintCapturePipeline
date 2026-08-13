"""Isaac worker for one digest-bound 1-5 replacement import probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
from typing import Any


REQUEST_SCHEMA = "paired_target_native_import_request.v1"
RESULT_SCHEMA = "paired_target_native_import_runtime_result.v1"
PROBE_SCHEMA = "simready_replacement_native_import_probe_result.v1"


def _canonical_digest(value: dict[str, Any], *, field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _write(path: Path, value: dict[str, Any], *, digest_field: str) -> None:
    payload = dict(value)
    payload[digest_field] = _canonical_digest(payload, field=digest_field)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _identifier(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_]", "_", value)
    if not text or text[0].isdigit():
        text = "_" + text
    return text


def run(*, request_path: Path, output_root: Path) -> dict[str, Any]:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    replacements = request.get("replacements")
    if (
        request.get("schema_version") != REQUEST_SCHEMA
        or request.get("request_digest") != _canonical_digest(request, field="request_digest")
        or not isinstance(replacements, list)
        or not 1 <= len(replacements) <= 5
        or request.get("replacement_count") != len(replacements)
    ):
        raise ValueError("paired_target_native_import_request_invalid")

    from isaacsim import SimulationApp  # type: ignore

    app = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})
    result: dict[str, Any]
    try:
        from pxr import Usd, UsdGeom, UsdPhysics, UsdShade  # type: ignore

        stage = Usd.Stage.CreateInMemory("paired_target_native_import.usda")
        world = stage.DefinePrim("/World", "Xform")
        stage.SetDefaultPrim(world)
        stage.DefinePrim("/World/Replacements", "Xform")
        rows: list[dict[str, Any]] = []
        for row in replacements:
            asset_id = str(row["asset_id"])
            relative = Path(str(row["relative_path"]))
            asset = (request_path.parent / relative).resolve()
            if (
                request_path.parent.resolve() not in asset.parents
                or asset.is_symlink()
                or not asset.is_file()
                or asset.stat().st_size != row.get("size_bytes")
                or _sha256(asset) != row.get("sha256")
            ):
                raise ValueError(f"paired_target_native_import_asset_invalid:{asset_id}")
            prim_path = f"/World/Replacements/{_identifier(asset_id)}"
            prim = stage.DefinePrim(prim_path, "Xform")
            prim.GetReferences().AddReference(str(asset))
            prim.Load()
            stage.Load()
            for _ in range(8):
                app.update()
            descendants = list(Usd.PrimRange(prim))
            rigid_bodies = sum(child.HasAPI(UsdPhysics.RigidBodyAPI) for child in descendants)
            collisions = sum(child.HasAPI(UsdPhysics.CollisionAPI) for child in descendants)
            joints = sum(child.IsA(UsdPhysics.Joint) for child in descendants)
            visual_meshes = [
                child
                for child in descendants
                if child.IsA(UsdGeom.Mesh)
                and UsdGeom.Imageable(child).ComputeVisibility() != UsdGeom.Tokens.invisible
                and UsdGeom.Imageable(child).ComputePurpose() == UsdGeom.Tokens.default_
            ]
            bound_materials = sum(
                bool(UsdShade.MaterialBindingAPI(child).ComputeBoundMaterial()[0])
                for child in visual_meshes
            )
            authored_colors = sum(
                child.GetCustomDataByKey("blueprint:agentAuthoredDisplayColorRgba") is not None
                for child in visual_meshes
            )
            registration_digest = next(
                (
                    child.GetCustomDataByKey("blueprint:assetFrameRegistrationDigest")
                    for child in descendants
                    if child.GetCustomDataByKey("blueprint:assetFrameRegistrationDigest")
                    is not None
                ),
                None,
            )
            imported = len(descendants) > 1 and (rigid_bodies > 0 or collisions > 0 or joints > 0)
            blockers = []
            if not imported:
                blockers.append("native_import_composed_structure_missing")
            if not visual_meshes or bound_materials != len(visual_meshes):
                blockers.append("native_import_visual_materials_missing")
            if authored_colors != len(visual_meshes):
                blockers.append("native_import_agent_authored_colors_missing")
            if registration_digest != row.get("asset_frame_registration_digest"):
                blockers.append("native_import_asset_frame_registration_missing")
            probe = {
                "schema_version": PROBE_SCHEMA,
                "status": "completed" if imported else "blocked",
                "asset_id": asset_id,
                "replacement_asset_sha256": row["sha256"],
                "registered_static_qualification_digest": row[
                    "registered_static_qualification_digest"
                ],
                "native_isaac_executed": True,
                "native_simulator_import_qualified": imported,
                "physical_equivalence_claimed": False,
                "candidate_policy_queried": False,
                "simulator_import_identity": {
                    "runtime": "isaac_sim",
                    "python_version": platform.python_version(),
                    "headless": True,
                },
                "native_readback": {
                    "asset_imported": imported,
                    "imported_prim_path": prim_path,
                    "composed_prim_count": len(descendants),
                    "rigid_body_prim_count": rigid_bodies,
                    "collision_prim_count": collisions,
                    "joint_prim_count": joints,
                    "render_visible_visual_mesh_count": len(visual_meshes),
                    "bound_material_visual_mesh_count": bound_materials,
                    "agent_authored_color_visual_mesh_count": authored_colors,
                    "asset_frame_registration_digest": registration_digest,
                },
                "joint_physics_behavior_qualified": False,
                "contact_or_support_qualified": False,
                "blockers": blockers,
                "result_digest": "",
            }
            probe_path = (
                output_root / "probes" / f"{int(row['index']):02d}_{_identifier(asset_id)}.json"
            )
            _write(probe_path, probe, digest_field="result_digest")
            sealed_probe = json.loads(probe_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "task_id": row["task_id"],
                    "asset_id": asset_id,
                    "probe_result_path": probe_path.relative_to(output_root).as_posix(),
                    "probe_result_sha256": _sha256(probe_path),
                    "probe_result_digest": sealed_probe["result_digest"],
                    "native_simulator_import_qualified": imported,
                    "blockers": blockers,
                }
            )
        blockers = sorted({code for row in rows for code in row["blockers"]})
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": "completed" if not blockers else "blocked",
            "scene_id": request["scene_id"],
            "request_digest": request["request_digest"],
            "replacement_count": len(rows),
            "replacements": rows,
            "native_isaac_executed": True,
            "all_replacements_import_qualified": not blockers,
            "candidate_policy_queried": False,
            "physical_equivalence_claimed": False,
            "blockers": blockers,
            "claim_boundary": (
                "Native Isaac import and composed-structure readback only; no appearance, "
                "contact, joint behavior, robot task, deployment, or physical claim."
            ),
            "result_digest": "",
        }
        _write(
            output_root / "paired_target_native_import_runtime_result.v1.json",
            result,
            digest_field="result_digest",
        )
        return result
    finally:
        # The result is written before shutdown because SimulationApp.close may
        # terminate the interpreter in some pinned Isaac images.
        app.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run(request_path=args.request.resolve(), output_root=args.output_root.resolve())
    os._exit(0 if result.get("status") == "completed" else 2)


if __name__ == "__main__":
    raise SystemExit(main())
