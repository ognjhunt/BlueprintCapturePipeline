"""Catalog static readback feeding the existing native import qualification seam."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from .decision_evidence_contracts import canonical_digest
from .dual_task_rehearsal_contract import validate_task_freeze
from .native_task_gpu_collision_qualification import audit_native_task_gpu_collisions
from .simready_central import SimReadyCentralError, _hash, verify_intake

SCHEMA_VERSION = "simready_central_static_qualification.v1"
STATUS = "catalog_structure_statically_qualified"


def qualify_catalog_asset(*, intake_receipt_path: str | Path, output_path: str | Path,
                          root_usd: str | None = None,
                          task_freeze_receipt_path: str | Path | None = None) -> dict[str, Any]:
    from pxr import Usd, UsdGeom, UsdPhysics

    source = Path(intake_receipt_path).resolve()
    intake = verify_intake(source)
    if root_usd is None and len(intake["root_usds"]) != 1:
        raise SimReadyCentralError("catalog_root_selection_required")
    root_usd = root_usd or intake["root_usds"][0]
    if root_usd not in intake["root_usds"]:
        raise SimReadyCentralError("catalog_root_not_declared")
    usd = source.parent / intake["package_directory"] / root_usd
    stage = Usd.Stage.Open(str(usd))
    if stage is None or not stage.GetDefaultPrim().IsValid():
        raise SimReadyCentralError("catalog_default_prim_invalid")
    prims = list(Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()))
    rigid = [p for p in prims if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    collisions = [p for p in prims if p.HasAPI(UsdPhysics.CollisionAPI)]
    meshes = [p for p in prims if p.IsA(UsdGeom.Mesh)]
    mass = []
    for prim in prims:
        if prim.HasAPI(UsdPhysics.MassAPI):
            value = UsdPhysics.MassAPI(prim).GetMassAttr().Get()
            mass.append({"prim_path": str(prim.GetPath()), "mass_kg": value})
    audit = audit_native_task_gpu_collisions(usd)
    blockers = list(intake["dependency_audit"]["blockers"]) + audit["blockers"]
    if not rigid or not collisions or not meshes:
        blockers.append("catalog_rigid_prop_structure_missing")
    if not mass or any(not isinstance(row["mass_kg"], (int, float))
                       or not math.isfinite(row["mass_kg"]) or row["mass_kg"] <= 0 for row in mass):
        blockers.append("catalog_explicit_positive_mass_required")
    if not stage.HasAuthoredMetadata("metersPerUnit") or UsdGeom.GetStageMetersPerUnit(stage) <= 0:
        blockers.append("catalog_units_unbound")
    task = (validate_task_freeze(json.loads(Path(task_freeze_receipt_path).read_text()))
            if task_freeze_receipt_path else None)
    result = {
        "schema_version": SCHEMA_VERSION, "status": STATUS if not blockers else "blocked",
        "asset_id": task["removal_plan"]["replacement_asset_id"] if task else intake["selection"]["package_path"].replace("/", "_"),
        "task_id": task["task_id"] if task else None,
        "task_freeze_digest": task["task_freeze_digest"] if task else None,
        "development_only": True, "source_kind": "licensed_catalog_package",
        "intake_receipt": {"path": str(source), "sha256": "sha256:" + _hash(source.read_bytes()),
                           "receipt_digest": intake["receipt_digest"]},
        "package_id": intake["package_id"], "license": intake["license"],
        "replacement_usd": {"path": str(usd), "size_bytes": usd.stat().st_size,
                            "sha256": "sha256:" + _hash(usd.read_bytes())},
        "readback": {"default_prim": str(stage.GetDefaultPrim().GetPath()),
                     "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
                     "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
                     "rigid_body_count": len(rigid), "collision_count": len(collisions),
                     "mesh_count": len(meshes), "masses": mass,
                     "root_variants": {name: stage.GetDefaultPrim().GetVariantSet(name).GetVariantSelection()
                                       for name in stage.GetDefaultPrim().GetVariantSets().GetNames()}},
        "gpu_collision_audit": audit, "blockers": sorted(set(blockers)),
        "catalog_structure_statically_qualified": not blockers,
        "native_simulator_import_qualified": False, "physical_equivalence_claimed": False,
        "producer_results_are_task_qualification": False, "receipt_digest": "",
    }
    result["receipt_digest"] = canonical_digest(result, digest_field="receipt_digest")
    output = Path(output_path)
    if output.exists() or output.is_symlink():
        raise SimReadyCentralError("catalog_static_output_exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return result


def valid_static_qualification(value: Mapping[str, Any]) -> bool:
    """Accept either explicit source kind without laundering catalog authorship."""
    expected = {
        "simready_graph_asset_static_qualification.v1": "authored_structure_statically_qualified",
        SCHEMA_VERSION: STATUS,
    }
    if value.get("schema_version") == SCHEMA_VERSION and (
        value.get("catalog_structure_statically_qualified") is not True
        or value.get("source_kind") != "licensed_catalog_package"
        or value.get("native_simulator_import_qualified") is not False
        or value.get("physical_equivalence_claimed") is not False
        or value.get("blockers") != []
    ):
        return False
    return bool(value.get("schema_version") in expected
                and value.get("status") == expected[value["schema_version"]]
                and value.get("receipt_digest") == canonical_digest(value, digest_field="receipt_digest"))


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intake-receipt-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--root-usd")
    parser.add_argument("--task-freeze-receipt-path")
    result = qualify_catalog_asset(**vars(parser.parse_args(argv)))
    print(json.dumps({key: result[key] for key in ("status", "asset_id", "blockers", "receipt_digest")}))
    return 0 if not result["blockers"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
