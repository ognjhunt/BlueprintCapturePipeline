"""Materialize one registered SAGE component as an engineered receptacle twin.

The builder replays the complete topology receipt from exact label and collision
bytes before it writes anything.  It uses the registered outer bounds and the
conservative clear-opening prism as the dimensional basis for a five-part,
bottom-centred bin with explicit authored wall and floor thicknesses.  The
source component remains provenance and registration evidence; its material
thickness is never inferred.  Source SAGE bytes are never copied into output.

The result is a development-only candidate.  Native import, support, contacts,
render coverage, and physical-material equivalence remain separate gates.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .engineered_receptacle_visual_basis import (
    EngineeredReceptacleVisualBasisError,
    verify_engineered_receptacle_visual_basis,
)
from .native_task_entity_asset_authoring_bundle import (
    RIGID_EXPECTED_PRIM_TYPE,
    RIGID_RUNTIME_CLASS,
)
from .native_task_runtime_source_packet import ISAACLAB_COMMIT, ISAACLAB_REPOSITORY
from .sage_collision_component_topology import (
    SCHEMA_VERSION as TOPOLOGY_SCHEMA_VERSION,
)
from .sage_collision_component_topology import (
    SageCollisionComponentTopologyError,
    inspect_sage_collision_component_topology,
    read_sage_collision_component_geometry,
)
from .task_entity_asset_candidate import materialize_task_entity_asset_candidate


SCHEMA_VERSION = "registered_static_receptacle_asset.v1"
CANDIDATE_FILENAME = "task_entity_asset_candidate.v1.json"
RECEIPT_FILENAME = "registered_static_receptacle_asset_receipt.v1.json"
VISUAL_BASIS_FILENAME = "engineered_receptacle_visual_design_basis.v1.json"
FROZEN_MINIMUM_COMPONENT_IOU = 0.85
FROZEN_OPENING_GRID_SIZE = 9
FROZEN_OPENING_MARGIN_FRACTION = 0.1
GEOMETRY_QUANTIZATION_RESOLUTION_M = 1.0e-9
RUNTIME_USD_READBACK_TOLERANCE_M = 1.0e-7
AUTHORED_TWIN_WALL_FRACTION = 0.04
AUTHORED_TWIN_FLOOR_FRACTION = 0.05
AUTHORED_TWIN_MINIMUM_THICKNESS_M = 0.003
AUTHORED_TWIN_MAXIMUM_WALL_THICKNESS_M = 0.012
AUTHORED_TWIN_MAXIMUM_FLOOR_THICKNESS_M = 0.008

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_REQUIRED_RIGHTS_FIELDS = frozenset(
    {
        "source_revision",
        "license_id",
        "license_reference",
        "license_sha256",
        "attribution",
        "derived_processing_authority_id",
        "provider_terms_id",
        "output_rights_id",
        "raw_source_private_upload_permitted",
        "derived_asset_private_upload_permitted",
        "raw_redistribution_permitted",
        "provider_retention_permitted",
        "provider_training_permitted",
    }
)
_REQUIRED_PHYSICS_FIELDS = frozenset(
    {
        "static_friction",
        "dynamic_friction",
        "restitution",
        "contact_offset_m",
        "rest_offset_m",
        "diagnostic_display_color_rgb",
    }
)


class RegisteredStaticReceptacleAssetError(ValueError):
    """Stable failures at the registered-component authoring boundary."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _clone(value: Mapping[str, Any], *, error: str) -> dict[str, Any]:
    try:
        cloned = json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise RegisteredStaticReceptacleAssetError([error]) from exc
    if not isinstance(cloned, dict):
        raise RegisteredStaticReceptacleAssetError([error])
    return cloned


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _safe_metadata(
    value: Mapping[str, Any], *, fields: frozenset[str], error: str
) -> dict[str, Any]:
    source = _clone(value, error=error)
    if set(source) != fields:
        raise RegisteredStaticReceptacleAssetError([error])
    for key, item in source.items():
        if isinstance(item, str) and (
            not item.strip() or len(item) > 1024 or any(ord(character) < 32 for character in item)
        ):
            raise RegisteredStaticReceptacleAssetError([f"{error}:{key}"])
    return source


def _number(value: Any, *, field: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RegisteredStaticReceptacleAssetError([f"receptacle_asset_{field}_invalid"])
    result = float(value)
    if not result >= minimum or result == float("inf") or result == float("-inf"):
        raise RegisteredStaticReceptacleAssetError([f"receptacle_asset_{field}_invalid"])
    return result


def _float_text(value: float) -> str:
    text = f"{float(value):.9f}".rstrip("0").rstrip(".")
    return "0" if text in {"", "-0"} else text


def _mesh_obj(vertices: Sequence[Sequence[float]], faces: Sequence[Sequence[int]]) -> str:
    rows = ["# Blueprint registered static component; metres; Z-up"]
    rows.extend("v " + " ".join(_float_text(value) for value in vertex) for vertex in vertices)
    rows.extend("f " + " ".join(str(index + 1) for index in face) for face in faces)
    return "\n".join(rows) + "\n"


def _engineered_open_receptacle_mesh(
    *,
    outer_dimensions_m: Sequence[float],
    wall_thicknesses_m: Mapping[str, float],
    floor_thickness_m: float,
) -> tuple[list[list[float]], list[list[int]]]:
    """Create one deterministic floor-plus-four-walls open bin mesh."""

    outer_x, outer_y, outer_z = (round(float(value), 9) for value in outer_dimensions_m)
    x_min = round(float(wall_thicknesses_m["x_min"]), 9)
    x_max = round(float(wall_thicknesses_m["x_max"]), 9)
    y_min = round(float(wall_thicknesses_m["y_min"]), 9)
    y_max = round(float(wall_thicknesses_m["y_max"]), 9)
    floor = round(float(floor_thickness_m), 9)
    if (
        min(outer_x, outer_y, outer_z, x_min, x_max, y_min, y_max, floor) <= 0.0
        or x_min + x_max >= outer_x
        or y_min + y_max >= outer_y
        or floor >= outer_z
    ):
        raise RegisteredStaticReceptacleAssetError(["receptacle_asset_engineered_geometry_invalid"])

    vertices: list[list[float]] = []
    faces: list[list[int]] = []

    def append_box(minimum: Sequence[float], maximum: Sequence[float]) -> None:
        start = len(vertices)
        x0, y0, z0 = minimum
        x1, y1, z1 = maximum
        vertices.extend(
            [
                [x0, y0, z0],
                [x1, y0, z0],
                [x1, y1, z0],
                [x0, y1, z0],
                [x0, y0, z1],
                [x1, y0, z1],
                [x1, y1, z1],
                [x0, y1, z1],
            ]
        )
        faces.extend(
            [
                [start + value for value in row]
                for row in (
                    (0, 3, 2, 1),
                    (4, 5, 6, 7),
                    (0, 1, 5, 4),
                    (1, 2, 6, 5),
                    (2, 3, 7, 6),
                    (3, 0, 4, 7),
                )
            ]
        )

    half_x = outer_x / 2.0
    half_y = outer_y / 2.0
    append_box((-half_x, -half_y, 0.0), (half_x, half_y, floor))
    append_box((-half_x, -half_y, floor), (-half_x + x_min, half_y, outer_z))
    append_box((half_x - x_max, -half_y, floor), (half_x, half_y, outer_z))
    append_box(
        (-half_x + x_min, -half_y, floor),
        (half_x - x_max, -half_y + y_min, outer_z),
    )
    append_box(
        (-half_x + x_min, half_y - y_max, floor),
        (half_x - x_max, half_y, outer_z),
    )
    return vertices, faces


def _usd_array(rows: Sequence[str], *, indent: str = "        ") -> str:
    return (",\n" + indent).join(rows)


def _runtime_usda(
    *,
    vertices: Sequence[Sequence[float]],
    faces: Sequence[Sequence[int]],
    physics: Mapping[str, Any],
) -> str:
    points = _usd_array(
        ["(" + ", ".join(_float_text(value) for value in row) + ")" for row in vertices]
    )
    counts = ", ".join(str(len(face)) for face in faces)
    indices = ", ".join(str(index) for face in faces for index in face)
    color = ", ".join(_float_text(value) for value in physics["diagnostic_display_color_rgb"])
    return f"""#usda 1.0
(
    defaultPrim = "Asset"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "Asset"
{{
    def Material "PhysicsMaterial" (
        prepend apiSchemas = ["PhysicsMaterialAPI"]
    )
    {{
        float physics:dynamicFriction = {_float_text(physics["dynamic_friction"])}
        float physics:restitution = {_float_text(physics["restitution"])}
        float physics:staticFriction = {_float_text(physics["static_friction"])}
    }}

    def Mesh "Geometry" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
    )
    {{
        rel material:binding:physics = </Asset/PhysicsMaterial>
        color3f[] primvars:displayColor = [({color})] (
            interpolation = "constant"
        )
        int[] faceVertexCounts = [{counts}]
        int[] faceVertexIndices = [{indices}]
        point3f[] points = [
        {points}
        ]
        uniform token physics:approximation = "none"
        uniform token subdivisionScheme = "none"
    }}
}}
"""


def _runtime_usd_point_readback_error(
    path: Path,
    *,
    expected_vertices: Sequence[Sequence[float]],
) -> float:
    try:
        from pxr import Usd, UsdGeom
    except ImportError as exc:
        raise RegisteredStaticReceptacleAssetError(
            ["receptacle_asset_openusd_readback_runtime_missing"]
        ) from exc
    stage = Usd.Stage.Open(str(path), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RegisteredStaticReceptacleAssetError(["receptacle_asset_runtime_usd_readback_failed"])
    points = UsdGeom.Mesh.Get(stage, "/Asset/Geometry").GetPointsAttr().Get() or []
    if len(points) != len(expected_vertices):
        raise RegisteredStaticReceptacleAssetError(["receptacle_asset_runtime_usd_readback_failed"])
    return max(
        abs(float(points[index][axis]) - float(expected_vertices[index][axis]))
        for index in range(len(points))
        for axis in range(3)
    )


def _publish_without_overwrite(*, staging: Path, destination: Path) -> None:
    """Reserve a destination and link files without replacing another writer."""

    try:
        destination.mkdir()
    except FileExistsError as exc:
        raise RegisteredStaticReceptacleAssetError(["receptacle_asset_output_exists"]) from exc
    except OSError as exc:
        raise RegisteredStaticReceptacleAssetError(
            ["receptacle_asset_output_reservation_failed"]
        ) from exc
    incomplete = destination / ".incomplete"
    incomplete.write_text("registered_static_receptacle_asset.v1\n", encoding="ascii")
    children = sorted(
        staging.iterdir(),
        key=lambda path: (path.name == RECEIPT_FILENAME, path.name),
    )
    if not children or children[-1].name != RECEIPT_FILENAME:
        raise RegisteredStaticReceptacleAssetError(["receptacle_asset_receipt_not_published_last"])
    for child in children:
        target = destination / child.name
        try:
            os.link(child, target)
        except FileExistsError as exc:
            raise RegisteredStaticReceptacleAssetError(
                ["receptacle_asset_output_race_detected"]
            ) from exc
        except OSError as exc:
            raise RegisteredStaticReceptacleAssetError(
                ["receptacle_asset_output_publish_failed"]
            ) from exc
    incomplete.unlink()


def _validate_pose(value: Mapping[str, Any]) -> dict[str, Any]:
    pose = _clone(value, error="receptacle_asset_reference_pose_invalid")
    if set(pose) != {"position_world_m", "orientation_xyzw"}:
        raise RegisteredStaticReceptacleAssetError(["receptacle_asset_reference_pose_invalid"])
    for key, length in (("position_world_m", 3), ("orientation_xyzw", 4)):
        row = pose.get(key)
        if (
            isinstance(row, (str, bytes, Mapping))
            or not isinstance(row, list)
            or len(row) != length
        ):
            raise RegisteredStaticReceptacleAssetError(["receptacle_asset_reference_pose_invalid"])
        pose[key] = [_number(item, field="reference_pose", minimum=float("-inf")) for item in row]
    norm = sum(value * value for value in pose["orientation_xyzw"]) ** 0.5
    if abs(norm - 1.0) > 1.0e-6:
        raise RegisteredStaticReceptacleAssetError(["receptacle_asset_reference_pose_invalid"])
    return pose


def _verify_topology(
    *,
    receipt: Mapping[str, Any],
    labels_path: str | Path,
    collision_path: str | Path,
) -> dict[str, Any]:
    expected = _clone(receipt, error="receptacle_asset_topology_receipt_invalid")
    if expected.get("schema_version") != TOPOLOGY_SCHEMA_VERSION:
        raise RegisteredStaticReceptacleAssetError(["receptacle_asset_topology_receipt_invalid"])
    targets = expected.get("targets")
    thresholds = expected.get("thresholds")
    if not isinstance(targets, list) or not targets or not isinstance(thresholds, Mapping):
        raise RegisteredStaticReceptacleAssetError(["receptacle_asset_topology_receipt_invalid"])
    frozen_thresholds = {
        "minimum_component_iou": FROZEN_MINIMUM_COMPONENT_IOU,
        "opening_grid_size": FROZEN_OPENING_GRID_SIZE,
        "opening_margin_fraction": FROZEN_OPENING_MARGIN_FRACTION,
    }
    if dict(thresholds) != frozen_thresholds:
        raise RegisteredStaticReceptacleAssetError(
            ["receptacle_asset_topology_thresholds_not_frozen"]
        )
    target_ids = [
        str(row.get("interiorgs_instance_id") or "") for row in targets if isinstance(row, Mapping)
    ]
    opening_ids = [
        str(row.get("interiorgs_instance_id"))
        for row in targets
        if isinstance(row, Mapping) and row.get("opening_probe") is not None
    ]
    try:
        observed = inspect_sage_collision_component_topology(
            labels_path=labels_path,
            target_instance_ids=target_ids,
            opening_probe_instance_ids=opening_ids,
            sage_collision_usd_path=collision_path,
            minimum_component_iou=FROZEN_MINIMUM_COMPONENT_IOU,
            opening_grid_size=FROZEN_OPENING_GRID_SIZE,
            opening_margin_fraction=FROZEN_OPENING_MARGIN_FRACTION,
        )
    except SageCollisionComponentTopologyError as exc:
        raise RegisteredStaticReceptacleAssetError(
            ["receptacle_asset_topology_source_unreadable"]
        ) from exc
    except (KeyError, TypeError, ValueError) as exc:
        raise RegisteredStaticReceptacleAssetError(
            ["receptacle_asset_topology_receipt_invalid"]
        ) from exc
    if observed != expected:
        raise RegisteredStaticReceptacleAssetError(["receptacle_asset_topology_replay_mismatch"])
    return observed


def _verify_visual_design_basis(
    *,
    visual_review_receipt_path: str | Path,
    render_manifest_path: str | Path,
    frame_root: str | Path,
    semantic_review_attestation_path: str | Path,
    semantic_authority_selection_path: str | Path,
    topology_receipt_digest: str,
    target_instance_id: str,
) -> dict[str, Any]:
    try:
        return verify_engineered_receptacle_visual_basis(
            visual_review_receipt_path=visual_review_receipt_path,
            render_manifest_path=render_manifest_path,
            frame_root=frame_root,
            semantic_review_attestation_path=semantic_review_attestation_path,
            semantic_authority_selection_path=semantic_authority_selection_path,
            current_topology_receipt_digest=topology_receipt_digest,
            source_instance_id=target_instance_id,
        )
    except EngineeredReceptacleVisualBasisError as exc:
        raise RegisteredStaticReceptacleAssetError(
            [
                "receptacle_asset_visual_design_basis_unqualified" + f":{error}"
                for error in exc.errors
            ]
        ) from exc


def build_registered_static_receptacle_asset(
    *,
    labels_path: str | Path,
    sage_collision_usd_path: str | Path,
    topology_receipt: Mapping[str, Any],
    visual_review_receipt_path: str | Path,
    render_manifest_path: str | Path,
    frame_root: str | Path,
    semantic_review_attestation_path: str | Path,
    semantic_authority_selection_path: str | Path,
    target_instance_id: str,
    entity_id: str,
    asset_id: str,
    reference_world_pose: Mapping[str, Any],
    rights: Mapping[str, Any],
    authoring_identity: Mapping[str, Any],
    physics_configuration: Mapping[str, Any],
    simulator_name: str,
    simulator_version: str,
    output_root: str | Path,
) -> dict[str, Any]:
    """Write one portable static-receptacle candidate without native claims."""

    if (
        not isinstance(entity_id, str)
        or not isinstance(asset_id, str)
        or not _IDENTIFIER.fullmatch(entity_id)
        or not _IDENTIFIER.fullmatch(asset_id)
    ):
        raise RegisteredStaticReceptacleAssetError(["receptacle_asset_identifier_invalid"])
    if not isinstance(target_instance_id, str) or not target_instance_id.strip():
        raise RegisteredStaticReceptacleAssetError(["receptacle_asset_target_invalid"])
    target_id = target_instance_id.strip()
    pose = _validate_pose(reference_world_pose)
    normalized_rights = _safe_metadata(
        rights,
        fields=_REQUIRED_RIGHTS_FIELDS,
        error="receptacle_asset_rights_invalid",
    )
    authoring = _clone(
        authoring_identity,
        error="receptacle_asset_authoring_identity_invalid",
    )
    if (
        set(authoring)
        != {
            "source_repository",
            "source_revision",
            "source_tree",
            "package_name",
            "package_version",
        }
        or not _REVISION.fullmatch(str(authoring.get("source_revision") or ""))
        or not _REVISION.fullmatch(str(authoring.get("source_tree") or ""))
    ):
        raise RegisteredStaticReceptacleAssetError(["receptacle_asset_authoring_identity_invalid"])
    for key, value in authoring.items():
        if (
            not isinstance(value, str)
            or not value.strip()
            or len(value) > 512
            or any(ord(character) < 32 for character in value)
        ):
            raise RegisteredStaticReceptacleAssetError(
                [f"receptacle_asset_authoring_identity_invalid:{key}"]
            )
    physics = _safe_metadata(
        physics_configuration,
        fields=_REQUIRED_PHYSICS_FIELDS,
        error="receptacle_asset_physics_invalid",
    )
    numeric_fields = (
        "static_friction",
        "dynamic_friction",
        "restitution",
        "contact_offset_m",
        "rest_offset_m",
    )
    for field in numeric_fields:
        physics[field] = _number(physics[field], field=field)
    if physics["dynamic_friction"] > physics["static_friction"] or physics["restitution"] > 1.0:
        raise RegisteredStaticReceptacleAssetError(["receptacle_asset_physics_invalid"])
    color = physics["diagnostic_display_color_rgb"]
    if isinstance(color, (str, bytes, Mapping)) or not isinstance(color, list) or len(color) != 3:
        raise RegisteredStaticReceptacleAssetError(["receptacle_asset_physics_invalid"])
    physics["diagnostic_display_color_rgb"] = [
        _number(value, field="diagnostic_color") for value in color
    ]
    if any(value > 1.0 for value in physics["diagnostic_display_color_rgb"]):
        raise RegisteredStaticReceptacleAssetError(["receptacle_asset_physics_invalid"])
    if not isinstance(simulator_name, str) or not isinstance(simulator_version, str):
        raise RegisteredStaticReceptacleAssetError(["receptacle_asset_simulator_identity_invalid"])
    if (
        not simulator_name.strip()
        or not simulator_version.strip()
        or len(simulator_name) > 128
        or len(simulator_version) > 128
        or any(ord(character) < 32 for character in simulator_name + simulator_version)
    ):
        raise RegisteredStaticReceptacleAssetError(["receptacle_asset_simulator_identity_invalid"])

    verified = _verify_topology(
        receipt=topology_receipt,
        labels_path=labels_path,
        collision_path=sage_collision_usd_path,
    )
    visual_review = _verify_visual_design_basis(
        visual_review_receipt_path=visual_review_receipt_path,
        render_manifest_path=render_manifest_path,
        frame_root=frame_root,
        semantic_review_attestation_path=semantic_review_attestation_path,
        semantic_authority_selection_path=semantic_authority_selection_path,
        topology_receipt_digest=verified["receipt_digest"],
        target_instance_id=target_id,
    )
    matches = [row for row in verified["targets"] if row["interiorgs_instance_id"] == target_id]
    if len(matches) != 1:
        raise RegisteredStaticReceptacleAssetError(["receptacle_asset_target_not_unique"])
    target = matches[0]
    opening = target.get("opening_probe")
    best = target.get("best_component")
    if (
        target.get("component_collision_identity_passed") is not True
        or not isinstance(best, Mapping)
        or _number(best.get("aabb_iou"), field="component_aabb_iou") < FROZEN_MINIMUM_COMPONENT_IOU
        or _number(
            best.get("target_coverage_fraction"),
            field="component_target_coverage_fraction",
        )
        < FROZEN_MINIMUM_COMPONENT_IOU
        or _number(
            best.get("component_coverage_fraction"),
            field="component_coverage_fraction",
        )
        < FROZEN_MINIMUM_COMPONENT_IOU
        or not isinstance(opening, Mapping)
    ):
        raise RegisteredStaticReceptacleAssetError(
            ["receptacle_asset_registered_outer_bounds_unqualified"]
        )
    source = verified["source_files"]["sage_collision_usd"]
    component = read_sage_collision_component_geometry(
        sage_collision_usd_path=sage_collision_usd_path,
        expected_source_sha256=source["sha256"],
        expected_source_size_bytes=source["size_bytes"],
        prim_path=best["prim_path"],
        component_index=best["component_index"],
        expected_geometry_digest=best["geometry_digest"],
    )

    outer = component["world_aabb_size_m"]
    if len(outer) != 3 or any(_number(value, field="outer_dimension") <= 0.0 for value in outer):
        raise RegisteredStaticReceptacleAssetError(
            ["receptacle_asset_registered_outer_bounds_unqualified"]
        )

    clear_opening = opening.get("conservative_clear_opening")
    source_opening_to_outer_boundary_clearances = (
        dict(clear_opening.get("boundary_clearances_m"))
        if isinstance(clear_opening, Mapping)
        and isinstance(clear_opening.get("boundary_clearances_m"), Mapping)
        else None
    )
    cavity_depth = _number(opening.get("cavity_depth_m"), field="source_cavity_depth")
    source_floor_surface_to_outer_bottom_clearance = (
        None if cavity_depth is None else round(float(outer[2]) - cavity_depth, 9)
    )

    x_wall = min(
        AUTHORED_TWIN_MAXIMUM_WALL_THICKNESS_M,
        max(AUTHORED_TWIN_MINIMUM_THICKNESS_M, float(outer[0]) * AUTHORED_TWIN_WALL_FRACTION),
    )
    y_wall = min(
        AUTHORED_TWIN_MAXIMUM_WALL_THICKNESS_M,
        max(AUTHORED_TWIN_MINIMUM_THICKNESS_M, float(outer[1]) * AUTHORED_TWIN_WALL_FRACTION),
    )
    authored_wall_thicknesses = {
        "x_min": round(x_wall, 9),
        "x_max": round(x_wall, 9),
        "y_min": round(y_wall, 9),
        "y_max": round(y_wall, 9),
    }
    minimum_wall = round(min(authored_wall_thicknesses.values()), 9)
    authored_floor_thickness = round(
        min(
            AUTHORED_TWIN_MAXIMUM_FLOOR_THICKNESS_M,
            max(
                AUTHORED_TWIN_MINIMUM_THICKNESS_M,
                float(outer[2]) * AUTHORED_TWIN_FLOOR_FRACTION,
            ),
        ),
        9,
    )
    interior = [
        round(
            float(outer[0])
            - authored_wall_thicknesses["x_min"]
            - authored_wall_thicknesses["x_max"],
            9,
        ),
        round(
            float(outer[1])
            - authored_wall_thicknesses["y_min"]
            - authored_wall_thicknesses["y_max"],
            9,
        ),
        round(float(outer[2]) - authored_floor_thickness, 9),
    ]
    if min(interior) <= 0.0:
        raise RegisteredStaticReceptacleAssetError(
            ["receptacle_asset_authored_twin_dimensions_invalid"]
        )

    if not isinstance(output_root, (str, os.PathLike)):
        raise RegisteredStaticReceptacleAssetError(["receptacle_asset_output_root_invalid"])
    try:
        raw_destination = Path(output_root).expanduser()
    except (TypeError, ValueError, OSError) as exc:
        raise RegisteredStaticReceptacleAssetError(
            ["receptacle_asset_output_root_invalid"]
        ) from exc
    try:
        if raw_destination.exists() or raw_destination.is_symlink():
            raise RegisteredStaticReceptacleAssetError(["receptacle_asset_output_exists"])
        destination = raw_destination.resolve()
        parent = destination.parent.resolve()
        parent.mkdir(parents=True, exist_ok=True)
    except RegisteredStaticReceptacleAssetError:
        raise
    except OSError as exc:
        raise RegisteredStaticReceptacleAssetError(
            ["receptacle_asset_output_root_invalid"]
        ) from exc
    with tempfile.TemporaryDirectory(prefix="registered-receptacle-", dir=parent) as raw:
        staging = Path(raw) / "asset"
        staging.mkdir()
        vertices, faces = _engineered_open_receptacle_mesh(
            outer_dimensions_m=outer,
            wall_thicknesses_m=authored_wall_thicknesses,
            floor_thickness_m=authored_floor_thickness,
        )
        obj = _mesh_obj(vertices, faces)
        (staging / "visual_geometry.obj").write_text(obj, encoding="utf-8")
        (staging / "collision_geometry.obj").write_text(obj, encoding="utf-8")
        (staging / "runtime_asset.usda").write_text(
            _runtime_usda(vertices=vertices, faces=faces, physics=physics),
            encoding="utf-8",
        )
        runtime_usd_readback_error_m = _runtime_usd_point_readback_error(
            staging / "runtime_asset.usda",
            expected_vertices=vertices,
        )
        if runtime_usd_readback_error_m > RUNTIME_USD_READBACK_TOLERANCE_M:
            raise RegisteredStaticReceptacleAssetError(
                ["receptacle_asset_runtime_usd_readback_tolerance_exceeded"]
            )
        material = {
            "schema_version": "diagnostic_receptacle_material.v1",
            "display_color_rgb": physics["diagnostic_display_color_rgb"],
            "source_appearance_reproduced": False,
            "purpose": "diagnostic_composed_entity_visibility",
        }
        write_json(staging / "material_definition.json", material)
        rgb = [round(value * 255) for value in physics["diagnostic_display_color_rgb"]]
        (staging / "diagnostic_texture.ppm").write_text(
            f"P3\n1 1\n255\n{rgb[0]} {rgb[1]} {rgb[2]}\n",
            encoding="ascii",
        )
        physics_receipt = {
            "schema_version": "static_receptacle_physics_configuration.v1",
            **physics,
            "static_anchored": True,
            "mass_kg": 0.0,
            "native_readback_required": True,
            "material_properties_observed": False,
        }
        write_json(staging / "physics_configuration.json", physics_receipt)
        write_json(staging / VISUAL_BASIS_FILENAME, visual_review)
        visual_basis_file = {
            "path": VISUAL_BASIS_FILENAME,
            "sha256": _sha256(staging / VISUAL_BASIS_FILENAME),
            "size_bytes": (staging / VISUAL_BASIS_FILENAME).stat().st_size,
        }

        file_specs = (
            ("visual_geometry", "visual_geometry.obj"),
            ("collision_geometry", "collision_geometry.obj"),
            ("runtime_usd", "runtime_asset.usda"),
            ("material_definition", "material_definition.json"),
            ("texture", "diagnostic_texture.ppm"),
            ("physics_configuration", "physics_configuration.json"),
        )
        files = [
            {
                "role": role,
                "path": name,
                "sha256": _sha256(staging / name),
                "size_bytes": (staging / name).stat().st_size,
            }
            for role, name in file_specs
        ]
        material_provenance = canonical_digest(material)
        collision_digest = next(
            row["sha256"] for row in files if row["role"] == "collision_geometry"
        )
        candidate = materialize_task_entity_asset_candidate(
            {
                "schema_version": "task_entity_asset_candidate.v1",
                "entity_id": entity_id,
                "asset_id": asset_id,
                "asset_class": "rigid_receptacle",
                "source_observation": {
                    "observation_id": f"sage-component:{target_id}",
                    "source_reference": "SAGE-3D registered collision component",
                    "source_sha256": source["sha256"],
                    "source_size_bytes": source["size_bytes"],
                    "bounds_world": {
                        "minimum_m": component["world_aabb_min_m"],
                        "maximum_m": component["world_aabb_max_m"],
                    },
                    "metric_dimensions_m": outer,
                    "coverage": {
                        "metric_bounds_observed": True,
                        "rest_state_bounded": True,
                        "full_surface_observed": False,
                        "interior_collision_observed": False,
                        "interior_appearance_observed": False,
                        "engineered_interior_not_factual": True,
                        "unobserved_regions": [
                            "interior appearance occluded by source contents",
                            "source collision cavity contains obstructions or unresolved apertures",
                            "material properties unobserved",
                            "publisher collision is not measurement-authoritative surface truth",
                        ],
                    },
                },
                "rights": normalized_rights,
                "authoring": {
                    "method": "released_code_registered_engineered_twin",
                    **authoring,
                    "generated_geometry_used": True,
                    "generated_physics_used": True,
                },
                "files": files,
                "transform": {
                    "authored_origin_m": [0.0, 0.0, 0.0],
                    "pivot_m": [0.0, 0.0, 0.0],
                    "scale_xyz": [1.0, 1.0, 1.0],
                    "world_pose": pose,
                    "meters_per_unit": 1.0,
                    "up_axis": "Z",
                },
                "simulator_import": {
                    "simulator": simulator_name,
                    "simulator_version": simulator_version,
                    "source_repository": ISAACLAB_REPOSITORY,
                    "source_revision": ISAACLAB_COMMIT,
                    "importer_module": RIGID_RUNTIME_CLASS,
                    "expected_prim_type": RIGID_EXPECTED_PRIM_TYPE,
                },
                "receptacle_configuration": {
                    "geometry": {
                        "open_interior": True,
                        "top_cap_present": False,
                        "interior_dimensions_m": interior,
                        "wall_thickness_m": minimum_wall,
                        "wall_clearances_m": authored_wall_thicknesses,
                        "floor_thickness_m": authored_floor_thickness,
                        "engineered_interior": True,
                    },
                    "collision": {
                        "representation": "static_open_triangle_mesh",
                        "collision_sha256": collision_digest,
                        "contact_offset_m": physics["contact_offset_m"],
                        "rest_offset_m": physics["rest_offset_m"],
                    },
                    "material": {
                        "static_friction": physics["static_friction"],
                        "dynamic_friction": physics["dynamic_friction"],
                        "restitution": physics["restitution"],
                        "material_provenance_sha256": material_provenance,
                    },
                    "anchoring": {
                        "static_anchored": True,
                        "mass_kg": 0.0,
                        "inertia_diagonal_kg_m2": [0.0, 0.0, 0.0],
                        "stable_support_readback_required": True,
                        "native_collision_readback_required": True,
                    },
                },
                "retained_diagnostic_requirements": [
                    "native_import",
                    "stable_support_and_no_initial_penetration",
                    "native_contact",
                    "native_reset_readback",
                    "native_render_coverage",
                ],
            }
        )
        write_json(staging / CANDIDATE_FILENAME, candidate)
        candidate_file = {
            "path": CANDIDATE_FILENAME,
            "sha256": _sha256(staging / CANDIDATE_FILENAME),
            "size_bytes": (staging / CANDIDATE_FILENAME).stat().st_size,
        }
        receipt: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "target_instance_id": target_id,
            "topology_receipt_digest": verified["receipt_digest"],
            "visual_design_basis_digest": visual_review["basis_digest"],
            "semantic_review_attestation_digest": visual_review["semantic_authority"][
                "attestation_digest"
            ],
            "semantic_authority_selection_digest": visual_review["semantic_authority"][
                "selection_digest"
            ],
            "semantic_authority": visual_review["semantic_authority"]["authority"],
            "visual_review_digest": visual_review["visual_review_receipt"]["review_digest"],
            "visual_review_collision_topology_receipt_digest": visual_review[
                "current_topology_receipt_digest"
            ],
            "visual_design_basis_file": visual_basis_file,
            "component_geometry_receipt_digest": component["receipt_digest"],
            "component_local_geometry_digest": component["local_geometry_digest"],
            "candidate_digest": candidate["candidate_digest"],
            "candidate_file": candidate_file,
            "derived_geometry": {
                "outer_dimensions_m": outer,
                "conservative_interior_dimensions_m": interior,
                "source_opening_to_outer_boundary_clearances_m": (
                    source_opening_to_outer_boundary_clearances
                ),
                "source_floor_surface_to_outer_bottom_clearance_m": (
                    source_floor_surface_to_outer_bottom_clearance
                ),
                "authored_twin_minimum_wall_thickness_m": minimum_wall,
                "authored_twin_wall_thicknesses_m": authored_wall_thicknesses,
                "authored_twin_floor_thickness_m": authored_floor_thickness,
                "authored_twin_dimension_rule": {
                    "wall_fraction": AUTHORED_TWIN_WALL_FRACTION,
                    "floor_fraction": AUTHORED_TWIN_FLOOR_FRACTION,
                    "minimum_thickness_m": AUTHORED_TWIN_MINIMUM_THICKNESS_M,
                    "maximum_wall_thickness_m": AUTHORED_TWIN_MAXIMUM_WALL_THICKNESS_M,
                    "maximum_floor_thickness_m": AUTHORED_TWIN_MAXIMUM_FLOOR_THICKNESS_M,
                    "source_clearances_used_as_authored_thickness": False,
                },
            },
            "geometry_conversion": {
                "intermediate_and_obj_quantization_resolution_m": (
                    GEOMETRY_QUANTIZATION_RESOLUTION_M
                ),
                "runtime_usd_point_type": "point3f",
                "runtime_usd_readback_tolerance_m": (RUNTIME_USD_READBACK_TOLERANCE_M),
                "runtime_usd_readback_max_abs_error_m": round(runtime_usd_readback_error_m, 12),
                "runtime_usd_readback_within_tolerance": True,
                "native_simulator_readback_required": True,
            },
            "files": sorted(files, key=lambda row: row["role"]),
            "claim_boundary": {
                "source_collision_component_exactly_replayed": False,
                "source_collision_component_deterministically_quantized": True,
                "source_collision_component_used_as_runtime_geometry": False,
                "source_collision_cavity_clear": opening.get("open_collision_cavity_passed")
                is True,
                "source_collision_enclosure_qualified": (
                    isinstance(opening.get("side_projected_coverage"), Mapping)
                    and opening["side_projected_coverage"].get("all_four_sides_passed") is True
                ),
                "source_collision_obstruction_triangle_count": (
                    opening.get("open_prism_obstruction_probe") or {}
                ).get("obstruction_triangle_count"),
                "intermediate_and_obj_quantization_resolution_m": (
                    GEOMETRY_QUANTIZATION_RESOLUTION_M
                ),
                "runtime_usd_point_type": "point3f",
                "runtime_usd_native_readback_qualified": False,
                "source_bytes_copied_to_output": False,
                "engineered_twin_not_source_scene_truth": True,
                "source_wall_thickness_observed": False,
                "source_floor_thickness_observed": False,
                "authored_twin_wall_and_floor_thicknesses_bound": True,
                "authored_twin_dimensions_derived_from_registered_outer_bounds": True,
                "source_opening_clearances_used_as_material_thickness": False,
                "source_interior_appearance_observed": False,
                "visual_semantics_authority_signed": True,
                "signed_visual_semantics_are_native_qualification": False,
                "native_simulator_qualified": False,
                "physical_equivalence_proven": False,
            },
            "receipt_digest": "",
        }
        receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
        write_json(staging / RECEIPT_FILENAME, receipt)
        _publish_without_overwrite(staging=staging, destination=destination)

    return {
        "output_root": str(destination.resolve()),
        "candidate_path": str((destination / CANDIDATE_FILENAME).resolve()),
        "receipt_path": str((destination / RECEIPT_FILENAME).resolve()),
        "candidate": candidate,
        "receipt": receipt,
    }


__all__ = [
    "CANDIDATE_FILENAME",
    "FROZEN_MINIMUM_COMPONENT_IOU",
    "FROZEN_OPENING_GRID_SIZE",
    "FROZEN_OPENING_MARGIN_FRACTION",
    "GEOMETRY_QUANTIZATION_RESOLUTION_M",
    "RUNTIME_USD_READBACK_TOLERANCE_M",
    "RECEIPT_FILENAME",
    "SCHEMA_VERSION",
    "VISUAL_BASIS_FILENAME",
    "RegisteredStaticReceptacleAssetError",
    "build_registered_static_receptacle_asset",
]
