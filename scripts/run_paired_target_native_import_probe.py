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
CANDIDATE_SCHEMA = "native_task_execution_candidate.v1"


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


def _candidate(request: dict[str, Any], *, request_path: Path) -> dict[str, Any]:
    record = request.get("execution_candidate")
    if not isinstance(record, dict):
        raise ValueError("native_task_execution_candidate_missing")
    relative = Path(str(record.get("relative_path") or ""))
    path = (request_path.parent / relative).resolve()
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or request_path.parent.resolve() not in path.parents
        or path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != record.get("size_bytes")
        or _sha256(path) != record.get("sha256")
    ):
        raise ValueError("native_task_execution_candidate_invalid")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("schema_version") != CANDIDATE_SCHEMA
        or value.get("status") != "prepared_for_exact_runtime_gpu_cook"
        or value.get("candidate_digest") != record.get("candidate_digest")
        or value.get("candidate_digest")
        != _canonical_digest(value, field="candidate_digest")
        or value.get("scene_id") != request.get("scene_id")
        or value.get("asset_count") != request.get("replacement_count")
        or value.get("construction_authorized") is not False
    ):
        raise ValueError("native_task_execution_candidate_invalid")
    return value


def _source_to_imported(source_path: str, *, imported_root: str) -> str:
    parts = Path(source_path).parts
    if not source_path.startswith("/") or len(parts) < 2:
        raise ValueError("native_gpu_collision_intent_prim_path_invalid")
    suffix = "/".join(parts[2:])
    return imported_root + (("/" + suffix) if suffix else "")


def _physics_cook_and_step(
    *,
    app: Any,
    stage: Any,
    output_root: Path,
    imported_roots: dict[str, str],
    candidates: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Attach the exact stage to native PhysX, step it, and read colliders back."""

    from pxr import UsdPhysics  # type: ignore

    physics_scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    physics_scene.CreateGravityMagnitudeAttr(0.0)
    cook_stage_path = output_root / "native_gpu_cook_stage.usda"
    stage.GetRootLayer().Export(str(cook_stage_path))

    import omni.usd  # type: ignore
    from isaacsim.core.api import SimulationContext  # type: ignore

    context = omni.usd.get_context()
    opened = context.open_stage(str(cook_stage_path))
    if opened is False:
        raise RuntimeError("native_gpu_cook_stage_open_failed")
    for _ in range(8):
        app.update()
    cooked_stage = context.get_stage()
    if cooked_stage is None or not cooked_stage.GetPrimAtPath("/World/PhysicsScene").IsValid():
        raise RuntimeError("native_gpu_physics_scene_missing")
    clear_instance = getattr(SimulationContext, "clear_instance", None)
    if callable(clear_instance):
        clear_instance()
    simulation = SimulationContext(
        physics_dt=1.0 / 120.0,
        rendering_dt=1.0 / 120.0,
        stage_units_in_meters=1.0,
    )
    physics_context = simulation.get_physics_context()
    gpu_dynamics = getattr(physics_context, "enable_gpu_dynamics", None)
    gpu_broadphase = getattr(physics_context, "set_broadphase_type", None)
    if not callable(gpu_dynamics) or not callable(gpu_broadphase):
        raise RuntimeError("native_gpu_physics_configuration_unavailable")
    gpu_broadphase("GPU")
    gpu_dynamics(True)
    simulation.initialize_physics()
    simulation.play()
    for _ in range(8):
        try:
            simulation.step(render=False)
        except TypeError:
            simulation.step()
    simulation.stop()

    readbacks: dict[str, dict[str, Any]] = {}
    for asset_id, candidate in candidates.items():
        root = imported_roots[asset_id]
        intent = candidate["collision_intent"]
        rows: list[dict[str, Any]] = []
        blockers: list[str] = []
        declared = [
            *intent.get("dynamic_mesh_colliders", []),
            *intent.get("dynamic_primitive_colliders", []),
        ]
        for row in declared:
            source_path = str(row.get("prim_path") or "")
            imported_path = _source_to_imported(source_path, imported_root=root)
            prim = cooked_stage.GetPrimAtPath(imported_path)
            approximation = None
            if prim.IsValid():
                mesh_collision = UsdPhysics.MeshCollisionAPI(prim)
                if mesh_collision:
                    approximation = str(
                        mesh_collision.GetApproximationAttr().Get() or ""
                    ) or None
            expected = row.get("approximation")
            if not prim.IsValid():
                blockers.append(f"native_gpu_collider_missing:{source_path}")
            elif expected is not None and approximation != expected:
                blockers.append(f"native_gpu_collider_approximation_mismatch:{source_path}")
            rows.append(
                {
                    "source_prim_path": source_path,
                    "imported_prim_path": imported_path,
                    "prim_valid": prim.IsValid(),
                    "expected_approximation": expected,
                    "observed_approximation": approximation,
                }
            )
        if len(rows) != intent.get("dynamic_collision_prim_count"):
            blockers.append("native_gpu_collision_prim_count_mismatch")
        readbacks[asset_id] = {
            "physics_scene_created": True,
            "stage_attached_to_native_physics": True,
            "simulation_steps": 8,
            "gravity_disabled_for_import_probe": True,
            "gpu_dynamics_requested": True,
            "gpu_broadphase_requested": True,
            "dynamic_collision_prim_count": len(rows),
            "colliders": rows,
            "blockers": sorted(set(blockers)),
            "native_gpu_physics_qualified": not blockers,
        }
    return readbacks


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
    execution_candidate = _candidate(request, request_path=request_path)
    candidate_by_asset = {
        str(row["asset_id"]): row for row in execution_candidate["assets"]
    }
    if set(candidate_by_asset) != {
        str(row.get("asset_id") or "") for row in replacements
    }:
        raise ValueError("native_task_execution_candidate_asset_set_mismatch")

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
        imported_roots: dict[str, str] = {}
        for row in replacements:
            asset_id = str(row["asset_id"])
            candidate_asset = candidate_by_asset[asset_id]
            relative = Path(str(row["relative_path"]))
            asset = (request_path.parent / relative).resolve()
            if (
                request_path.parent.resolve() not in asset.parents
                or asset.is_symlink()
                or not asset.is_file()
                or asset.stat().st_size != row.get("size_bytes")
                or _sha256(asset) != row.get("sha256")
                or (candidate_asset.get("registered_asset") or {}).get("sha256")
                != row.get("sha256")
            ):
                raise ValueError(f"paired_target_native_import_asset_invalid:{asset_id}")
            prim_path = f"/World/Replacements/{_identifier(asset_id)}"
            imported_roots[asset_id] = prim_path
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
                "execution_candidate_digest": execution_candidate[
                    "candidate_digest"
                ],
                "collision_intent_digest": candidate_asset[
                    "collision_intent"
                ]["intent_digest"],
                "native_gpu_physics_qualified": False,
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
                    "collision_intent_digest": candidate_asset[
                        "collision_intent"
                    ]["intent_digest"],
                    "native_gpu_physics_qualified": False,
                    "blockers": blockers,
                }
            )
        physics_readbacks = _physics_cook_and_step(
            app=app,
            stage=stage,
            output_root=output_root,
            imported_roots=imported_roots,
            candidates=candidate_by_asset,
        )
        for row in rows:
            asset_id = row["asset_id"]
            probe_path = output_root / row["probe_result_path"]
            probe = json.loads(probe_path.read_text(encoding="utf-8"))
            physics = physics_readbacks[asset_id]
            probe["native_gpu_physics_readback"] = physics
            probe["native_gpu_physics_qualified"] = physics[
                "native_gpu_physics_qualified"
            ]
            probe["blockers"] = sorted(
                set([*probe.get("blockers", []), *physics.get("blockers", [])])
            )
            probe["status"] = "completed" if not probe["blockers"] else "blocked"
            _write(probe_path, probe, digest_field="result_digest")
            sealed_probe = json.loads(probe_path.read_text(encoding="utf-8"))
            row["probe_result_sha256"] = _sha256(probe_path)
            row["probe_result_digest"] = sealed_probe["result_digest"]
            row["native_gpu_physics_qualified"] = probe[
                "native_gpu_physics_qualified"
            ]
            row["blockers"] = probe["blockers"]
        candidate_output = output_root / "native_task_execution_candidate.v1.json"
        _write(candidate_output, execution_candidate, digest_field="candidate_digest")
        blockers = sorted({code for row in rows for code in row["blockers"]})
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": "completed" if not blockers else "blocked",
            "scene_id": request["scene_id"],
            "request_digest": request["request_digest"],
            "replacement_count": len(rows),
            "replacements": rows,
            "execution_candidate_digest": execution_candidate[
                "candidate_digest"
            ],
            "native_isaac_executed": True,
            "all_replacements_import_qualified": not blockers,
            "native_gpu_physics_qualified": all(
                row["native_gpu_physics_qualified"] for row in rows
            ),
            "candidate_policy_queried": False,
            "physical_equivalence_claimed": False,
            "blockers": blockers,
            "claim_boundary": (
                "Native Isaac import, GPU collision cooking, and zero-gravity physics-step "
                "readback only; no contact, joint behavior, robot task, deployment, or "
                "physical claim."
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
