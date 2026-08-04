"""Compose a deterministic physics-validation stage around a SimReady prop."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "adp009a_simready_validation_stage_receipt.v1"


class PublicSceneSimReadyValidationStageError(ValueError):
    """The validation stage could not be bound to the sealed prop bytes."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _under(path: Path, root: Path) -> Path:
    path = path.expanduser().resolve()
    root = root.expanduser().resolve()
    if path != root and root not in path.parents:
        raise PublicSceneSimReadyValidationStageError(f"path_outside_approved_root:{path}")
    return path


def materialize_validation_stage(
    *,
    source_asset: Path,
    source_receipt: Path,
    repo_root: Path,
    evidence_root: Path,
    output_stage: Path,
    output_receipt: Path,
) -> dict[str, Any]:
    repo_root = repo_root.expanduser().resolve()
    evidence_root = evidence_root.expanduser().resolve()
    source_asset = _under(source_asset, evidence_root)
    source_receipt = _under(source_receipt, repo_root)
    output_stage = _under(output_stage, evidence_root)
    output_receipt = _under(output_receipt, evidence_root)
    if not source_asset.is_file() or not source_receipt.is_file():
        raise PublicSceneSimReadyValidationStageError("source_asset_or_receipt_missing")
    receipt = json.loads(source_receipt.read_text(encoding="utf-8"))
    if receipt.get("receipt_digest") != canonical_digest(receipt, digest_field="receipt_digest"):
        raise PublicSceneSimReadyValidationStageError("source_receipt_digest_mismatch")
    asset_digest = _sha256(source_asset)
    if receipt.get("usd", {}).get("sha256") != asset_digest:
        raise PublicSceneSimReadyValidationStageError("source_asset_digest_mismatch")

    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils
    except ImportError as exc:  # pragma: no cover
        raise PublicSceneSimReadyValidationStageError("openusd_runtime_missing") from exc

    output_stage.parent.mkdir(parents=True, exist_ok=True)
    relative_asset = os.path.relpath(source_asset, output_stage.parent)
    stage = Usd.Stage.CreateNew(str(output_stage))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    scene = UsdPhysics.Scene.Define(stage, "/World/physics_scene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr(9.81)
    prop = UsdGeom.Xform.Define(stage, "/World/control")
    prop.GetPrim().GetReferences().AddReference(relative_asset)
    stage.GetRootLayer().customLayerData = {
        "blueprint:claim_ceiling": "development_only",
        "blueprint:purpose": "validation_context_not_replacement_asset",
        "blueprint:source_asset_sha256": asset_digest,
    }
    stage.GetRootLayer().Save()

    reopened = Usd.Stage.Open(str(output_stage))
    if reopened is None or not reopened.GetPrimAtPath("/World/physics_scene").IsA(UsdPhysics.Scene):
        raise PublicSceneSimReadyValidationStageError("physics_scene_readback_failed")
    if not reopened.GetPrimAtPath("/World/control").HasAPI(UsdPhysics.RigidBodyAPI):
        raise PublicSceneSimReadyValidationStageError("referenced_prop_readback_failed")
    _layers, _assets, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(output_stage)))
    if unresolved:
        raise PublicSceneSimReadyValidationStageError("validation_stage_unresolved_dependencies")

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "program_id": "arm-decision-proof-v1",
        "adp_item": "ADP-009A",
        "source_control_id": receipt["control_id"],
        "source_asset": {
            "relative_path": source_asset.relative_to(evidence_root).as_posix(),
            "size_bytes": source_asset.stat().st_size,
            "sha256": asset_digest,
        },
        "validation_stage": {
            "relative_path": output_stage.relative_to(evidence_root).as_posix(),
            "size_bytes": output_stage.stat().st_size,
            "sha256": _sha256(output_stage),
            "meters_per_unit": 1.0,
            "up_axis": "Z",
            "physics_scene": "/World/physics_scene",
            "referenced_prop": "/World/control",
            "unresolved_dependency_count": 0,
        },
        "claim_boundaries": {
            "validation_stage_is_not_replacement_asset": True,
            "gravity_context_is_not_dynamic_execution": True,
            "physics_values_remain_authoring_priors": True,
            "physical_evidence": False,
        },
        "receipt_digest": "",
    }
    result["receipt_digest"] = canonical_digest(result, digest_field="receipt_digest")
    output_receipt.parent.mkdir(parents=True, exist_ok=True)
    output_receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-asset", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output-stage", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    result = materialize_validation_stage(
        source_asset=args.source_asset,
        source_receipt=args.source_receipt,
        repo_root=args.repo_root,
        evidence_root=args.evidence_root,
        output_stage=args.output_stage,
        output_receipt=args.output_receipt,
    )
    print(json.dumps({"receipt_digest": result["receipt_digest"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
