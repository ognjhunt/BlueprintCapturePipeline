"""Project agent-authored STEP B-Reps into Mesh USD working copies.

This is deterministic format conversion, not CAD authoring. The exact STEP is
the geometry authority. The projection preserves named top-level occurrences
and leaf solids so NVIDIA USD Content Agents can operate on UsdGeom.Mesh prims.
It never creates collision, articulation, mass, or task-scoring authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest


PACKET_SCHEMA_VERSION = "cad_agent_step_mesh_packet.v1"
PROJECTION_SCHEMA_VERSION = "cad_agent_mesh_usd_projection.v1"


class CadAgentMeshProjectionError(ValueError):
    """Stable failure for unsafe or unbound CAD format conversion."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _file_record(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink() or resolved.stat().st_size <= 0:
        raise CadAgentMeshProjectionError("cad_agent_projection_file_invalid")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _record_valid(value: Any, *, verify_files: bool) -> bool:
    if not isinstance(value, Mapping):
        return False
    path = Path(str(value.get("path") or "")).expanduser().resolve()
    digest = str(value.get("sha256") or "")
    if (
        not str(value.get("path") or "")
        or not isinstance(value.get("size_bytes"), int)
        or value.get("size_bytes", 0) <= 0
        or len(digest) != 71
        or not digest.startswith("sha256:")
    ):
        return False
    return not verify_files or (
        path.is_file()
        and not path.is_symlink()
        and path.stat().st_size == value["size_bytes"]
        and _sha256(path) == digest
    )


def _safe_name(value: Any, fallback: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_]", "_", str(value or "").strip())
    text = re.sub(r"_+", "_", text).strip("_") or fallback
    if text[0].isdigit():
        text = "p_" + text
    return text


def _finite_vector(value: Any, length: int) -> list[float] | None:
    if not isinstance(value, list) or len(value) != length:
        return None
    result: list[float] = []
    for item in value:
        if isinstance(item, bool):
            return None
        try:
            number = float(item)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        result.append(number)
    return result


def validate_step_mesh_packet(
    value: Mapping[str, Any], *, verify_files: bool = True
) -> dict[str, Any]:
    packet = json.loads(json.dumps(value))
    errors: list[str] = []
    if packet.get("schema_version") != PACKET_SCHEMA_VERSION:
        errors.append("cad_agent_mesh_packet_schema_invalid")
    if packet.get("geometry_authority") != "exact_agent_authored_step":
        errors.append("cad_agent_mesh_packet_authority_invalid")
    if packet.get("deterministic_geometry_generator_used") is not False:
        errors.append("cad_agent_mesh_packet_generator_claim_invalid")
    if packet.get("conversion_only") is not True:
        errors.append("cad_agent_mesh_packet_conversion_claim_invalid")
    if not _record_valid(packet.get("step"), verify_files=verify_files):
        errors.append("cad_agent_mesh_packet_step_invalid")
    rows = packet.get("meshes")
    if not isinstance(rows, list) or not rows:
        errors.append("cad_agent_mesh_packet_rows_invalid")
        rows = []
    prim_paths: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            errors.append("cad_agent_mesh_packet_row_invalid")
            continue
        prim_path = str(row.get("prim_path") or "")
        points = row.get("points_mm")
        triangles = row.get("triangles")
        if (
            not prim_path.startswith("/Asset/links/")
            or row.get("assembly_transform_applied") is not True
            or not isinstance(points, list)
            or len(points) < 3
            or any(_finite_vector(point, 3) is None for point in points)
            or not isinstance(triangles, list)
            or not triangles
            or any(
                not isinstance(face, list)
                or len(face) != 3
                or any(
                    not isinstance(index, int)
                    or isinstance(index, bool)
                    or index < 0
                    or index >= len(points)
                    for index in face
                )
                for face in triangles or []
            )
        ):
            errors.append("cad_agent_mesh_packet_row_invalid")
        prim_paths.append(prim_path)
    if len(prim_paths) != len(set(prim_paths)):
        errors.append("cad_agent_mesh_packet_duplicate_prim")
    if packet.get("mesh_count") != len(rows):
        errors.append("cad_agent_mesh_packet_count_invalid")
    if packet.get("packet_digest") != canonical_digest(
        packet, digest_field="packet_digest"
    ):
        errors.append("cad_agent_mesh_packet_digest_invalid")
    if errors:
        raise CadAgentMeshProjectionError(";".join(sorted(set(errors))))
    return packet


def extract_step_mesh_packet(
    *,
    step_path: str | Path,
    output_path: str | Path,
    linear_tolerance_mm: float = 0.2,
    angular_tolerance_rad: float = 0.1,
) -> dict[str, Any]:
    """Reopen STEP and tessellate every labeled leaf solid without alteration."""

    try:
        from build123d import import_step
    except ImportError as exc:  # pragma: no cover - CAD-tool environment only
        raise CadAgentMeshProjectionError(
            "cad_agent_mesh_projection_build123d_missing"
        ) from exc
    if linear_tolerance_mm <= 0.0 or angular_tolerance_rad <= 0.0:
        raise CadAgentMeshProjectionError("cad_agent_mesh_tolerance_invalid")
    step = _file_record(step_path)
    try:
        assembly = import_step(step["path"])
    except Exception as exc:  # pragma: no cover - malformed external CAD bytes
        raise CadAgentMeshProjectionError("cad_agent_mesh_step_import_failed") from exc
    top_children = list(getattr(assembly, "children", []) or [])
    if not top_children:
        top_children = [assembly]
    rows: list[dict[str, Any]] = []
    used_paths: set[str] = set()
    for link_index, link in enumerate(top_children):
        link_name = _safe_name(getattr(link, "label", None), f"link_{link_index:03d}")
        leaf_rows: list[tuple[Any, Any]] = []

        def collect_leaves(node: Any, ancestor_location: Any) -> None:
            children = list(getattr(node, "children", []) or [])
            if not children:
                leaf_rows.append((node, ancestor_location))
                return
            next_ancestor = ancestor_location * node.location
            for child in children:
                collect_leaves(child, next_ancestor)

        if getattr(link, "children", None):
            for child in link.children:
                collect_leaves(child, assembly.location * link.location)
        else:
            leaf_rows.append((link, assembly.location))
        for leaf_index, (leaf, ancestor_location) in enumerate(leaf_rows):
            solids = list(leaf.solids())
            if len(solids) != 1:
                raise CadAgentMeshProjectionError(
                    "cad_agent_mesh_leaf_not_single_solid"
                )
            solid = solids[0].moved(ancestor_location)
            solid_name = _safe_name(
                getattr(leaf, "label", None), f"solid_{leaf_index:03d}"
            )
            base_path = f"/Asset/links/{link_name}/geometry/{solid_name}"
            prim_path = base_path
            suffix = 2
            while prim_path in used_paths:
                prim_path = f"{base_path}_{suffix}"
                suffix += 1
            used_paths.add(prim_path)
            try:
                vertices, faces = solid.tessellate(
                    linear_tolerance_mm,
                    angular_tolerance_rad,
                )
            except Exception as exc:  # pragma: no cover - CAD kernel failure
                raise CadAgentMeshProjectionError(
                    "cad_agent_mesh_tessellation_failed"
                ) from exc
            points = [[float(v.X), float(v.Y), float(v.Z)] for v in vertices]
            triangles = [[int(i) for i in face] for face in faces]
            rows.append(
                {
                    "prim_path": prim_path,
                    "link_id": link_name,
                    "solid_id": solid_name,
                    "assembly_transform_applied": True,
                    "points_mm": points,
                    "triangles": triangles,
                }
            )
    payload: dict[str, Any] = {
        "schema_version": PACKET_SCHEMA_VERSION,
        "geometry_authority": "exact_agent_authored_step",
        "deterministic_geometry_generator_used": False,
        "conversion_only": True,
        "step": step,
        "linear_tolerance_mm": float(linear_tolerance_mm),
        "angular_tolerance_rad": float(angular_tolerance_rad),
        "mesh_count": len(rows),
        "meshes": rows,
        "claim_boundary": {
            "cad_authored_by_projection": False,
            "appearance_working_copy_only": True,
            "collision_authority": False,
            "physics_authority": False,
            "simready_qualified": False,
        },
    }
    payload["packet_digest"] = canonical_digest(
        payload, digest_field="packet_digest"
    )
    validated = validate_step_mesh_packet(payload)
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(validated, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return validated


def materialize_mesh_usd_projection(
    *, packet_path: str | Path, output_usd_path: str | Path
) -> dict[str, Any]:
    """Author a Mesh-only USD working copy from an exact tessellation packet."""

    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade
    except ImportError as exc:  # pragma: no cover - environment guard
        raise CadAgentMeshProjectionError("cad_agent_mesh_projection_usd_missing") from exc
    packet_file = _file_record(packet_path)
    try:
        packet = validate_step_mesh_packet(
            json.loads(Path(packet_file["path"]).read_text(encoding="utf-8"))
        )
    except (json.JSONDecodeError, OSError) as exc:
        raise CadAgentMeshProjectionError("cad_agent_mesh_packet_invalid") from exc
    output = Path(output_usd_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Scope.Define(stage, "/Asset/links")
    material = UsdShade.Material.Define(stage, "/Asset/materials/agent_input_neutral")
    shader = UsdShade.Shader.Define(
        stage, "/Asset/materials/agent_input_neutral/PreviewSurface"
    )
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.5, 0.5, 0.5)
    )
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    prim_paths: list[str] = []
    total_points = 0
    total_triangles = 0
    for row in packet["meshes"]:
        mesh = UsdGeom.Mesh.Define(stage, row["prim_path"])
        points = [Gf.Vec3f(*(float(value) / 1000.0 for value in point)) for point in row["points_mm"]]
        faces = row["triangles"]
        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexCountsAttr([3] * len(faces))
        mesh.CreateFaceVertexIndicesAttr([index for face in faces for index in face])
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateOrientationAttr(UsdGeom.Tokens.rightHanded)
        mesh.CreatePurposeAttr(UsdGeom.Tokens.default_)
        mesh.CreateDoubleSidedAttr(False)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        mesh.GetPrim().SetCustomDataByKey(
            "blueprint:geometryAuthority", "exact_agent_authored_step"
        )
        mesh.GetPrim().SetCustomDataByKey(
            "blueprint:contentAgentsWorkingCopy", True
        )
        prim_paths.append(row["prim_path"])
        total_points += len(points)
        total_triangles += len(faces)
    root.GetPrim().SetCustomDataByKey("blueprint:sourceStepSha256", packet["step"]["sha256"])
    root.GetPrim().SetCustomDataByKey("blueprint:meshPacketDigest", packet["packet_digest"])
    root.GetPrim().SetCustomDataByKey("blueprint:deterministicCadGeneratorUsed", False)
    root.GetPrim().SetCustomDataByKey("blueprint:collisionAuthority", False)
    stage.GetRootLayer().Save()
    output_record = _file_record(output)
    receipt: dict[str, Any] = {
        "schema_version": PROJECTION_SCHEMA_VERSION,
        "status": "mesh_working_copy_authored",
        "packet": packet_file,
        "packet_digest": packet["packet_digest"],
        "step": packet["step"],
        "output_usd": output_record,
        "mesh_prim_paths": prim_paths,
        "mesh_count": len(prim_paths),
        "point_count": total_points,
        "triangle_count": total_triangles,
        "default_material_path": str(material.GetPath()),
        "content_agents_input_eligible": True,
        "canonical_simulator_asset": False,
        "claim_boundary": {
            "deterministic_format_conversion_only": True,
            "cad_authored_by_projection": False,
            "collision_authority": False,
            "physics_authority": False,
            "native_simulator_import_qualified": False,
            "physical_equivalence": False,
        },
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    return receipt


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract_parser = subparsers.add_parser("extract-step")
    extract_parser.add_argument("--step", required=True)
    extract_parser.add_argument("--output", required=True)
    extract_parser.add_argument("--linear-tolerance-mm", type=float, default=0.2)
    extract_parser.add_argument("--angular-tolerance-rad", type=float, default=0.1)
    usd_parser = subparsers.add_parser("author-usd")
    usd_parser.add_argument("--packet", required=True)
    usd_parser.add_argument("--output-usd", required=True)
    usd_parser.add_argument("--receipt", required=True)
    args = parser.parse_args(argv)
    if args.command == "extract-step":
        result = extract_step_mesh_packet(
            step_path=args.step,
            output_path=args.output,
            linear_tolerance_mm=args.linear_tolerance_mm,
            angular_tolerance_rad=args.angular_tolerance_rad,
        )
    else:
        result = materialize_mesh_usd_projection(
            packet_path=args.packet,
            output_usd_path=args.output_usd,
        )
        target = Path(args.receipt).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "CadAgentMeshProjectionError",
    "PACKET_SCHEMA_VERSION",
    "PROJECTION_SCHEMA_VERSION",
    "extract_step_mesh_packet",
    "materialize_mesh_usd_projection",
    "validate_step_mesh_packet",
]
