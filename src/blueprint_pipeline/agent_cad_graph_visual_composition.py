"""Compose exact agent-authored CAD meshes with a graph-authored simulator asset.

The graph asset carries joint, mass, and collision candidate data.  The exact
agent-authored STEP mesh projection carries visual geometry.  Neither may
silently stand in for the other: collision geometry remains guide/invisible,
and copied visual meshes never receive a collision API.  The binding is
explicit per agent link, making the seam reusable for one through five
independent replacements without semantic-class or scene-ID assumptions.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .freeze_amendment_carry_forward import (
    FreezeAmendmentCarryForwardError,
    validate_freeze_amendment_carry_forward_content,
)
from .cad_agent_review_media import (
    CadAgentReviewMediaError,
    selected_cad_agent_visual_review,
)
from .cad_agent_mesh_projection import (
    PROJECTION_SCHEMA_VERSION,
    CadAgentMeshProjectionError,
    validate_step_mesh_packet,
)
from .decision_evidence_contracts import canonical_digest, canonical_json
from .dual_task_rehearsal_contract import MAX_REPLACEMENT_OBJECTS, validate_task_freeze
from .simready_cad_agent_contract import (
    MATRIX_SCHEMA_VERSION,
    SimReadyCadAgentContractError,
    validate_cad_agent_matrix,
    validate_cad_agent_output,
)
from .simready_graph_asset import RECEIPT_SCHEMA as GRAPH_ASSET_RECEIPT_SCHEMA


BINDING_SCHEMA_VERSION = "simready_agent_cad_visual_binding.v2"
COMPOSITION_SCHEMA_VERSION = "simready_agent_cad_visual_composition.v2"
COMPOSITION_SET_SCHEMA_VERSION = "scene_agent_cad_visual_composition_set.v1"

_CLAIM_BOUNDARY = {
    "agent_authored_step_visual_geometry": True,
    "deterministic_geometry_generator_used": False,
    "collision_geometry_remains_graph_candidate": True,
    "appearance_materially_qualified": False,
    "native_simulator_import_qualified": False,
    "joint_physics_behavior_qualified": False,
    "physical_equivalence_proven": False,
}

_COMPOSITION_CLAIM_BOUNDARY = {
    **_CLAIM_BOUNDARY,
    "agent_authored_display_colors_preserved": True,
    "generated_texture_maps_present": False,
}


class AgentCadGraphVisualCompositionError(ValueError):
    """Stable failures for unsafe graph/agent-CAD visual composition."""

    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


def _clone(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        clone = json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise AgentCadGraphVisualCompositionError(["agent_cad_visual_payload_not_json"]) from exc
    if not isinstance(clone, dict):
        raise AgentCadGraphVisualCompositionError(["agent_cad_visual_payload_not_mapping"])
    return clone


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _is_digest(value: Any) -> bool:
    text = str(value or "")
    return (
        len(text) == 71
        and text.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in text[7:])
    )


def _file_record(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink() or resolved.stat().st_size <= 0:
        raise AgentCadGraphVisualCompositionError(["agent_cad_visual_file_invalid"])
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _record_valid(value: Any, *, verify_files: bool) -> bool:
    if not isinstance(value, Mapping):
        return False
    path_text = str(value.get("path") or "")
    if (
        not path_text
        or not isinstance(value.get("size_bytes"), int)
        or isinstance(value.get("size_bytes"), bool)
        or int(value["size_bytes"]) <= 0
        or not _is_digest(value.get("sha256"))
    ):
        return False
    if not verify_files:
        return True
    path = Path(path_text).expanduser().resolve()
    return (
        path.is_file()
        and not path.is_symlink()
        and path.stat().st_size == value["size_bytes"]
        and _sha256(path) == value["sha256"]
    )


def _same_file(left: Any, right: Any) -> bool:
    return (
        isinstance(left, Mapping)
        and isinstance(right, Mapping)
        and left.get("size_bytes") == right.get("size_bytes")
        and left.get("sha256") == right.get("sha256")
    )


def _read_json_record(record: Any, code: str, *, verify_files: bool) -> tuple[Path, dict[str, Any]]:
    if not _record_valid(record, verify_files=verify_files):
        raise AgentCadGraphVisualCompositionError([code])
    path = Path(str(record["path"])).expanduser().resolve()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AgentCadGraphVisualCompositionError([code]) from exc
    if not isinstance(value, dict):
        raise AgentCadGraphVisualCompositionError([code])
    return path, value


def _identifier(value: Any) -> str:
    text = str(value or "").strip()
    return text if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", text) else ""


def _safe_prim_token(value: str, fallback: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_]", "_", value).strip("_") or fallback
    return "p_" + result if result[0].isdigit() else result


def _unit_vector(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 3:
        return None
    try:
        vector = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in vector):
        return None
    norm = math.sqrt(sum(item * item for item in vector))
    if norm <= 1.0e-9:
        return None
    return [item / norm for item in vector]


def _normalize_grasp_collision_patches(value: Any) -> list[dict[str, Any]]:
    """Validate exact-mesh patch requests without embedding object labels."""

    if value in (None, []):
        return []
    if not isinstance(value, list):
        raise AgentCadGraphVisualCompositionError(
            ["agent_cad_grasp_collision_patches_invalid"]
        )
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in value:
        if not isinstance(raw, Mapping):
            raise AgentCadGraphVisualCompositionError(
                ["agent_cad_grasp_collision_patches_invalid"]
            )
        patch_id = _identifier(raw.get("patch_id"))
        graph_link_id = _identifier(raw.get("graph_link_id"))
        source_mesh_path = str(raw.get("source_mesh_path") or "")
        role = str(raw.get("grasp_affordance_role") or "")
        axis = raw.get("selection_axis_index")
        minimum = raw.get("minimum_axis_value_m")
        maximum = raw.get("maximum_axis_value_m")
        outward = _unit_vector(raw.get("approach_outward_unit_graph_link"))
        pinch = _unit_vector(raw.get("pinch_axis_graph_link"))
        try:
            minimum_value = float(minimum)
            maximum_value = None if maximum is None else float(maximum)
        except (TypeError, ValueError) as exc:
            raise AgentCadGraphVisualCompositionError(
                ["agent_cad_grasp_collision_patches_invalid"]
            ) from exc
        if (
            not patch_id
            or patch_id in seen
            or not graph_link_id
            or not source_mesh_path.startswith("/Asset/")
            or role != "parallel_jaw_outer_rim_patch"
            or isinstance(axis, bool)
            or not isinstance(axis, int)
            or axis not in {0, 1, 2}
            or not math.isfinite(minimum_value)
            or (
                maximum_value is not None
                and (
                    not math.isfinite(maximum_value)
                    or maximum_value <= minimum_value
                )
            )
            or outward is None
            or pinch is None
            or abs(sum(a * b for a, b in zip(outward, pinch, strict=True)))
            > 1.0e-6
            or raw.get("collision_approximation") != "convexHull"
        ):
            raise AgentCadGraphVisualCompositionError(
                ["agent_cad_grasp_collision_patches_invalid"]
            )
        seen.add(patch_id)
        rows.append(
            {
                "patch_id": patch_id,
                "graph_link_id": graph_link_id,
                "source_mesh_path": source_mesh_path,
                "grasp_affordance_role": role,
                "selection_axis_index": axis,
                "minimum_axis_value_m": minimum_value,
                "maximum_axis_value_m": maximum_value,
                "approach_outward_unit_graph_link": outward,
                "pinch_axis_graph_link": pinch,
                "collision_approximation": "convexHull",
            }
        )
    return sorted(rows, key=lambda row: row["patch_id"])


def _matrix(value: Any) -> list[list[float]] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    result: list[list[float]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != 4:
            return None
        normalized: list[float] = []
        for item in row:
            if isinstance(item, bool):
                return None
            try:
                number = float(item)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(number):
                return None
            normalized.append(number)
        result.append(normalized)
    if any(
        abs(result[3][index] - expected) > 1e-8
        for index, expected in enumerate((0.0, 0.0, 0.0, 1.0))
    ):
        return None
    rotation = [[result[row][column] for column in range(3)] for row in range(3)]
    if any(
        abs(sum(rotation[row][column] * rotation[row][other] for row in range(3)) - expected) > 1e-6
        for column in range(3)
        for other, expected in ((column, 1.0),)
    ):
        return None
    if any(
        abs(sum(rotation[row][column] * rotation[row][other] for row in range(3))) > 1e-6
        for column in range(3)
        for other in range(column + 1, 3)
    ):
        return None
    determinant = (
        rotation[0][0] * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
    )
    return result if abs(determinant - 1.0) <= 1e-6 else None


def _transform_point(matrix: Sequence[Sequence[float]], point: Sequence[float]) -> list[float]:
    vector = [float(point[0]), float(point[1]), float(point[2]), 1.0]
    return [
        sum(float(matrix[row][column]) * vector[column] for column in range(4)) for row in range(3)
    ]


def _quat_rotation_xyzw(value: Any) -> list[list[float]] | None:
    if value is None:
        return None
    try:
        imaginary = value.GetImaginary()
        x, y, z, w = (
            float(imaginary[0]),
            float(imaginary[1]),
            float(imaginary[2]),
            float(value.GetReal()),
        )
    except (AttributeError, IndexError, TypeError, ValueError):
        return None
    if not all(math.isfinite(component) for component in (x, y, z, w)):
        return None
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if abs(norm - 1.0) > 1e-6:
        return None
    return [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ]


def _graph_link_reset_inverse(stage: Any, link_path: str) -> list[list[float]]:
    """Return the asset-root-to-link-local transform at the sealed reset pose."""

    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(link_path)
    operations = UsdGeom.Xformable(prim).GetOrderedXformOps()
    names = [str(operation.GetOpName()) for operation in operations]
    if names != ["xformOp:translate", "xformOp:orient"]:
        raise AgentCadGraphVisualCompositionError(
            ["agent_cad_visual_graph_link_reset_transform_invalid"]
        )
    translation = operations[0].Get()
    rotation = _quat_rotation_xyzw(operations[1].Get())
    try:
        offset = [float(translation[index]) for index in range(3)]
    except (IndexError, TypeError, ValueError) as exc:
        raise AgentCadGraphVisualCompositionError(
            ["agent_cad_visual_graph_link_reset_transform_invalid"]
        ) from exc
    if rotation is None or not all(math.isfinite(value) for value in offset):
        raise AgentCadGraphVisualCompositionError(
            ["agent_cad_visual_graph_link_reset_transform_invalid"]
        )
    transpose = [[rotation[column][row] for column in range(3)] for row in range(3)]
    inverse_translation = [
        -sum(transpose[row][column] * offset[column] for column in range(3)) for row in range(3)
    ]
    return [[*transpose[row], inverse_translation[row]] for row in range(3)] + [
        [0.0, 0.0, 0.0, 1.0]
    ]


def _collision_visual_isolation(stage: Any) -> None:
    from pxr import UsdGeom, UsdPhysics

    violations: list[str] = []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        imageable = UsdGeom.Imageable(prim)
        if (
            imageable.ComputePurpose() != UsdGeom.Tokens.guide
            or imageable.ComputeVisibility() != UsdGeom.Tokens.invisible
            or prim.GetCustomDataByKey("blueprint:collisionGeometryOnly") is not True
        ):
            violations.append(str(prim.GetPath()))
    if violations:
        raise AgentCadGraphVisualCompositionError(["agent_cad_visual_collision_not_isolated"])


def _graph_has_no_renderable_geometry(stage: Any) -> None:
    """Require a graph skeleton to contribute mechanics, never visual pixels."""

    from pxr import UsdGeom

    for prim in stage.Traverse():
        gprim = UsdGeom.Gprim(prim)
        if not gprim:
            continue
        imageable = UsdGeom.Imageable(prim)
        if (
            imageable.ComputePurpose() in {UsdGeom.Tokens.default_, UsdGeom.Tokens.render}
            and imageable.ComputeVisibility() != UsdGeom.Tokens.invisible
        ):
            raise AgentCadGraphVisualCompositionError(
                ["agent_cad_visual_graph_renderable_geometry_present"]
            )


def _load_graph_authoring(record: Any, *, verify_files: bool) -> tuple[dict[str, Any], Any]:
    try:
        from pxr import Usd, UsdGeom
    except ImportError as exc:  # pragma: no cover - environment guard
        raise AgentCadGraphVisualCompositionError(
            ["agent_cad_visual_openusd_runtime_missing"]
        ) from exc
    _, receipt = _read_json_record(
        record, "agent_cad_visual_graph_authoring_receipt_invalid", verify_files=verify_files
    )
    output = receipt.get("output_usd") or {}
    if (
        receipt.get("schema_version") != GRAPH_ASSET_RECEIPT_SCHEMA
        or receipt.get("status") != "simready_candidate_authored"
        or receipt.get("receipt_digest") != canonical_digest(receipt, digest_field="receipt_digest")
        or not _is_digest(receipt.get("task_freeze_digest"))
        or not str(receipt.get("asset_id") or "").strip()
        or not str(receipt.get("task_id") or "").strip()
        or not _record_valid(output, verify_files=verify_files)
    ):
        raise AgentCadGraphVisualCompositionError(
            ["agent_cad_visual_graph_authoring_receipt_invalid"]
        )
    if not verify_files:
        return receipt, None
    stage = Usd.Stage.Open(str(output["path"]), load=Usd.Stage.LoadAll)
    if (
        stage is None
        or str(stage.GetDefaultPrim().GetPath()) != "/Asset"
        or float(UsdGeom.GetStageMetersPerUnit(stage)) != 1.0
        or str(UsdGeom.GetStageUpAxis(stage)).upper() != "Z"
    ):
        raise AgentCadGraphVisualCompositionError(["agent_cad_visual_graph_usd_invalid"])
    root = stage.GetPrimAtPath("/Asset")
    if not root.IsValid() or root.GetCustomDataByKey("blueprint:assetId") != receipt["asset_id"]:
        raise AgentCadGraphVisualCompositionError(["agent_cad_visual_graph_usd_invalid"])
    links = receipt.get("link_paths")
    if not isinstance(links, Mapping) or not links:
        raise AgentCadGraphVisualCompositionError(["agent_cad_visual_graph_link_paths_invalid"])
    seen_paths: set[str] = set()
    for link_id, path in links.items():
        if not _identifier(link_id) or not str(path).startswith("/Asset/links/"):
            raise AgentCadGraphVisualCompositionError(["agent_cad_visual_graph_link_paths_invalid"])
        prim = stage.GetPrimAtPath(str(path))
        if not prim.IsValid() or str(path) in seen_paths:
            raise AgentCadGraphVisualCompositionError(["agent_cad_visual_graph_link_paths_invalid"])
        seen_paths.add(str(path))
    _collision_visual_isolation(stage)
    _graph_has_no_renderable_geometry(stage)
    return receipt, stage


def _load_cad_output(record: Any, *, verify_files: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    _, receipt = _read_json_record(
        record, "agent_cad_visual_cad_output_receipt_invalid", verify_files=verify_files
    )
    try:
        output = validate_cad_agent_output(receipt, verify_files=verify_files)
    except SimReadyCadAgentContractError as exc:
        raise AgentCadGraphVisualCompositionError(
            ["agent_cad_visual_cad_output_receipt_invalid", *exc.codes]
        ) from exc
    if output.get("status") != "candidate_authored":
        raise AgentCadGraphVisualCompositionError(["agent_cad_visual_cad_output_not_authored"])
    task_freeze_record = ((output.get("request") or {}).get("inputs") or {}).get("task_freeze")
    _, task_freeze = _read_json_record(
        task_freeze_record,
        "agent_cad_visual_task_freeze_record_invalid",
        verify_files=verify_files,
    )
    try:
        task_freeze = validate_task_freeze(task_freeze)
    except ValueError as exc:
        raise AgentCadGraphVisualCompositionError(
            ["agent_cad_visual_task_freeze_record_invalid"]
        ) from exc
    return output, task_freeze


def _load_cad_output_from_matrix(
    *,
    record: Any,
    task_id: str,
    asset_id: str,
    backend_id: str,
    verify_files: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _, matrix_value = _read_json_record(
        record, "agent_cad_visual_cad_matrix_invalid", verify_files=verify_files
    )
    try:
        matrix = validate_cad_agent_matrix(matrix_value)
    except SimReadyCadAgentContractError as exc:
        raise AgentCadGraphVisualCompositionError(
            ["agent_cad_visual_cad_matrix_invalid", *exc.codes]
        ) from exc
    if matrix.get("schema_version") != MATRIX_SCHEMA_VERSION:
        raise AgentCadGraphVisualCompositionError(["agent_cad_visual_cad_matrix_invalid"])
    objects = [
        row
        for row in matrix["objects"]
        if row.get("task_id") == task_id and row.get("asset_id") == asset_id
    ]
    if len(objects) != 1:
        raise AgentCadGraphVisualCompositionError(["agent_cad_visual_cad_matrix_identity_invalid"])
    candidates = [
        row
        for row in objects[0]["candidates"]
        if ((row.get("request") or {}).get("backend") or {}).get("backend_id") == backend_id
    ]
    if len(candidates) != 1:
        raise AgentCadGraphVisualCompositionError(["agent_cad_visual_cad_matrix_backend_invalid"])
    output = candidates[0]
    try:
        output = validate_cad_agent_output(output, verify_files=verify_files)
    except SimReadyCadAgentContractError as exc:
        raise AgentCadGraphVisualCompositionError(
            ["agent_cad_visual_cad_matrix_invalid", *exc.codes]
        ) from exc
    task_freeze_record = ((output.get("request") or {}).get("inputs") or {}).get("task_freeze")
    _, task_freeze = _read_json_record(
        task_freeze_record,
        "agent_cad_visual_task_freeze_record_invalid",
        verify_files=verify_files,
    )
    try:
        task_freeze = validate_task_freeze(task_freeze)
    except ValueError as exc:
        raise AgentCadGraphVisualCompositionError(
            ["agent_cad_visual_task_freeze_record_invalid"]
        ) from exc
    return output, task_freeze


def _load_cad_output_source(
    binding: Mapping[str, Any], *, verify_files: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    direct = binding.get("cad_agent_output_receipt")
    matrix = binding.get("cad_agent_matrix")
    if direct is not None and matrix is not None:
        raise AgentCadGraphVisualCompositionError(["agent_cad_visual_cad_source_ambiguous"])
    if direct is not None:
        return _load_cad_output(direct, verify_files=verify_files)
    if matrix is not None:
        return _load_cad_output_from_matrix(
            record=matrix,
            task_id=str(binding.get("task_id") or ""),
            asset_id=str(binding.get("asset_id") or ""),
            backend_id=str(binding.get("cad_agent_backend_id") or ""),
            verify_files=verify_files,
        )
    raise AgentCadGraphVisualCompositionError(["agent_cad_visual_cad_source_missing"])


def _projection_mesh_link_id(path: str) -> str:
    match = re.fullmatch(r"/Asset/links/([^/]+)/geometry/[^/]+", path)
    return match.group(1) if match is not None else ""


def _load_mesh_projection(
    record: Any, *, verify_files: bool
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    try:
        from pxr import Usd, UsdGeom, UsdPhysics
    except ImportError as exc:  # pragma: no cover - environment guard
        raise AgentCadGraphVisualCompositionError(
            ["agent_cad_visual_openusd_runtime_missing"]
        ) from exc
    _, receipt = _read_json_record(
        record, "agent_cad_visual_mesh_projection_receipt_invalid", verify_files=verify_files
    )
    if (
        receipt.get("schema_version") != PROJECTION_SCHEMA_VERSION
        or receipt.get("status") != "mesh_working_copy_authored"
        or receipt.get("receipt_digest") != canonical_digest(receipt, digest_field="receipt_digest")
        or receipt.get("canonical_simulator_asset") is not False
        or not _record_valid(receipt.get("packet"), verify_files=verify_files)
        or not _record_valid(receipt.get("step"), verify_files=verify_files)
        or not _record_valid(receipt.get("output_usd"), verify_files=verify_files)
    ):
        raise AgentCadGraphVisualCompositionError(
            ["agent_cad_visual_mesh_projection_receipt_invalid"]
        )
    _, packet_value = _read_json_record(
        receipt["packet"], "agent_cad_visual_mesh_packet_invalid", verify_files=verify_files
    )
    try:
        packet = validate_step_mesh_packet(packet_value, verify_files=verify_files)
    except CadAgentMeshProjectionError as exc:
        raise AgentCadGraphVisualCompositionError(["agent_cad_visual_mesh_packet_invalid"]) from exc
    if (
        receipt.get("packet_digest") != packet.get("packet_digest")
        or not _same_file(receipt.get("step"), packet.get("step"))
        or receipt.get("mesh_prim_paths") != [row["prim_path"] for row in packet["meshes"]]
        or receipt.get("mesh_count") != packet.get("mesh_count")
    ):
        raise AgentCadGraphVisualCompositionError(["agent_cad_visual_mesh_projection_join_invalid"])
    if not verify_files:
        return receipt, packet, None
    stage = Usd.Stage.Open(str(receipt["output_usd"]["path"]), load=Usd.Stage.LoadAll)
    if (
        stage is None
        or str(stage.GetDefaultPrim().GetPath()) != "/Asset"
        or float(UsdGeom.GetStageMetersPerUnit(stage)) != 1.0
        or str(UsdGeom.GetStageUpAxis(stage)).upper() != "Z"
    ):
        raise AgentCadGraphVisualCompositionError(["agent_cad_visual_mesh_projection_usd_invalid"])
    root = stage.GetPrimAtPath("/Asset")
    if (
        root.GetCustomDataByKey("blueprint:sourceStepSha256") != packet["step"]["sha256"]
        or root.GetCustomDataByKey("blueprint:collisionAuthority") is not False
    ):
        raise AgentCadGraphVisualCompositionError(["agent_cad_visual_mesh_projection_usd_invalid"])
    expected_paths = set(receipt["mesh_prim_paths"])
    actual_paths: set[str] = set()
    for prim in stage.Traverse():
        gprim = UsdGeom.Gprim(prim)
        if not gprim:
            continue
        if not prim.IsA(UsdGeom.Mesh):
            raise AgentCadGraphVisualCompositionError(
                ["agent_cad_visual_mesh_projection_usd_invalid"]
            )
        path = str(prim.GetPath())
        actual_paths.add(path)
        imageable = UsdGeom.Imageable(prim)
        if (
            path not in expected_paths
            or not _projection_mesh_link_id(path)
            or prim.HasAPI(UsdPhysics.CollisionAPI)
            or imageable.ComputePurpose() != UsdGeom.Tokens.default_
            or imageable.ComputeVisibility() == UsdGeom.Tokens.invisible
            or prim.GetCustomDataByKey("blueprint:geometryAuthority") != "exact_agent_authored_step"
        ):
            raise AgentCadGraphVisualCompositionError(
                ["agent_cad_visual_mesh_projection_usd_invalid"]
            )
    if actual_paths != expected_paths:
        raise AgentCadGraphVisualCompositionError(["agent_cad_visual_mesh_projection_usd_invalid"])
    return receipt, packet, stage


def _normalize_link_bindings(
    *,
    value: Any,
    source_link_ids: set[str],
    graph_link_paths: Mapping[str, str],
    graph_stage: Any,
    unmapped_reasons: Any,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    if not isinstance(value, list):
        raise AgentCadGraphVisualCompositionError(["agent_cad_visual_link_bindings_invalid"])
    rows: list[dict[str, Any]] = []
    source_seen: set[str] = set()
    for raw in value:
        if not isinstance(raw, Mapping):
            raise AgentCadGraphVisualCompositionError(["agent_cad_visual_link_bindings_invalid"])
        agent_link_id = _identifier(raw.get("agent_link_id"))
        graph_link_id = _identifier(raw.get("graph_link_id"))
        transform_mode = str(raw.get("transform_mode") or "")
        if graph_link_id not in graph_link_paths:
            transform = None
        elif transform_mode == "explicit_rigid_transform":
            transform = _matrix(raw.get("T_graph_link_from_agent_asset"))
        elif transform_mode == "graph_link_reset_inverse":
            if graph_stage is None:
                raise AgentCadGraphVisualCompositionError(
                    ["agent_cad_visual_graph_link_reset_transform_unavailable"]
                )
            else:
                derived_transform = _graph_link_reset_inverse(
                    graph_stage, str(graph_link_paths.get(graph_link_id) or "")
                )
                supplied_transform = raw.get("T_graph_link_from_agent_asset")
                if supplied_transform in (None, []):
                    transform = derived_transform
                else:
                    normalized_supplied = _matrix(supplied_transform)
                    transform = (
                        derived_transform
                        if normalized_supplied is not None
                        and all(
                            abs(normalized_supplied[row][column] - derived_transform[row][column])
                            <= 1e-8
                            for row in range(4)
                            for column in range(4)
                        )
                        else None
                    )
        else:
            transform = None
        if (
            agent_link_id not in source_link_ids
            or agent_link_id in source_seen
            or transform is None
        ):
            raise AgentCadGraphVisualCompositionError(["agent_cad_visual_link_bindings_invalid"])
        source_seen.add(agent_link_id)
        rows.append(
            {
                "agent_link_id": agent_link_id,
                "graph_link_id": graph_link_id,
                "transform_mode": transform_mode,
                "T_graph_link_from_agent_asset": transform,
            }
        )
    if source_seen != source_link_ids:
        raise AgentCadGraphVisualCompositionError(["agent_cad_visual_link_bindings_incomplete"])
    if not isinstance(unmapped_reasons, Mapping):
        raise AgentCadGraphVisualCompositionError(["agent_cad_visual_unmapped_graph_links_invalid"])
    mapped_graph_ids = {row["graph_link_id"] for row in rows}
    expected_unmapped = set(graph_link_paths) - mapped_graph_ids
    reasons: dict[str, str] = {}
    for link_id, reason in unmapped_reasons.items():
        normalized_id = _identifier(link_id)
        text = str(reason or "").strip()
        if not normalized_id or not text:
            raise AgentCadGraphVisualCompositionError(
                ["agent_cad_visual_unmapped_graph_links_invalid"]
            )
        reasons[normalized_id] = text
    if set(reasons) != expected_unmapped:
        raise AgentCadGraphVisualCompositionError(["agent_cad_visual_unmapped_graph_links_invalid"])
    return (
        sorted(rows, key=lambda row: row["agent_link_id"]),
        {key: reasons[key] for key in sorted(reasons)},
    )


def _validate_binding(
    value: Mapping[str, Any], *, verify_files: bool
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    binding = _clone(value)
    if binding.get("schema_version") != BINDING_SCHEMA_VERSION or binding.get(
        "binding_digest"
    ) != canonical_digest(binding, digest_field="binding_digest"):
        raise AgentCadGraphVisualCompositionError(["agent_cad_visual_binding_digest_invalid"])
    graph, graph_stage = _load_graph_authoring(
        binding.get("graph_authoring_receipt"), verify_files=verify_files
    )
    cad_output, task_freeze = _load_cad_output_source(binding, verify_files=verify_files)
    projection, packet, _ = _load_mesh_projection(
        binding.get("mesh_projection_receipt"), verify_files=verify_files
    )
    request = cad_output["request"]
    try:
        visual_review = selected_cad_agent_visual_review(
            binding.get("cad_agent_visual_review") or {},
            scene_id=str(request.get("scene_id") or ""),
            task_id=str(request.get("task_id") or ""),
            asset_id=str(request.get("asset_id") or ""),
            backend_id=str(((request.get("backend") or {}).get("backend_id")) or ""),
            cad_agent_output_receipt_digest=str(cad_output.get("receipt_digest") or ""),
            verify_files=verify_files,
        )
    except CadAgentReviewMediaError as exc:
        raise AgentCadGraphVisualCompositionError(
            ["agent_cad_visual_reference_review_invalid", str(exc)]
        ) from exc
    if (
        binding.get("scene_id") != request.get("scene_id")
        or binding.get("task_id") != graph.get("task_id")
        or binding.get("task_id") != request.get("task_id")
        or binding.get("asset_id") != graph.get("asset_id")
        or binding.get("asset_id") != request.get("asset_id")
        or binding.get("task_freeze_digest") != graph.get("task_freeze_digest")
        or not _freeze_join_accepted(
            cad_side_digest=str(task_freeze.get("task_freeze_digest") or ""),
            graph_side_digest=str(binding.get("task_freeze_digest") or ""),
            proof=binding.get("freeze_amendment_carry_forward"),
        )
        or binding.get("cad_agent_output_receipt_digest") != cad_output.get("receipt_digest")
        or not _same_file((cad_output.get("artifacts") or {}).get("step"), projection.get("step"))
    ):
        raise AgentCadGraphVisualCompositionError(["agent_cad_visual_binding_identity_mismatch"])
    source_links = {row["link_id"] for row in packet["meshes"]}
    rows, unmapped = _normalize_link_bindings(
        value=binding.get("link_bindings"),
        source_link_ids=source_links,
        graph_link_paths=graph["link_paths"],
        graph_stage=graph_stage,
        unmapped_reasons=binding.get("unmapped_graph_link_reasons"),
    )
    if (
        binding.get("link_bindings") != rows
        or binding.get("unmapped_graph_link_reasons") != unmapped
    ):
        raise AgentCadGraphVisualCompositionError(["agent_cad_visual_binding_not_canonical"])
    context = {
        "graph": graph,
        "cad_output": cad_output,
        "projection": projection,
        "packet": packet,
        "visual_review": visual_review,
    }
    return binding, context


def validate_agent_cad_visual_binding(
    value: Mapping[str, Any], *, verify_files: bool = True
) -> dict[str, Any]:
    """Verify one exact agent-CAD-to-graph transform binding."""

    binding, _ = _validate_binding(value, verify_files=verify_files)
    return binding


def _freeze_join_accepted(
    *,
    cad_side_digest: str,
    graph_side_digest: str,
    proof: Any,
) -> bool:
    """One amendment-shaped divergence, accepted only with its exact proof.

    The CAD receipts were sealed while the superseded freeze was current; the
    graph asset is sealed to the amended one. Those citations are both honest,
    and rewriting either would fake history. What bridges them is a proof that
    the amendment changed nothing the binding consumes -- pinned to exactly
    this pair of content digests, so a proof for one amendment can never
    launder another.
    """

    if cad_side_digest == graph_side_digest:
        return True
    if not isinstance(proof, Mapping):
        return False
    try:
        validate_freeze_amendment_carry_forward_content(
            proof,
            sealed_schema=BINDING_SCHEMA_VERSION,
            superseded_digest=cad_side_digest,
            amended_digest=graph_side_digest,
        )
    except FreezeAmendmentCarryForwardError:
        return False
    return True


def seal_agent_cad_visual_binding(
    *,
    graph_authoring_receipt_path: str | Path,
    cad_agent_output_receipt_path: str | Path | None = None,
    cad_agent_matrix_path: str | Path | None = None,
    cad_agent_backend_id: str | None = None,
    cad_agent_visual_review_path: str | Path,
    mesh_projection_receipt_path: str | Path,
    link_bindings: Sequence[Mapping[str, Any]],
    unmapped_graph_link_reasons: Mapping[str, str],
    output_path: str | Path,
    freeze_carry_forward: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Seal exact visual mapping data before writing a composed USD."""

    graph_record = _file_record(graph_authoring_receipt_path)
    if (cad_agent_output_receipt_path is None) == (cad_agent_matrix_path is None):
        raise AgentCadGraphVisualCompositionError(["agent_cad_visual_cad_source_ambiguous"])
    if cad_agent_output_receipt_path is not None and cad_agent_backend_id is not None:
        raise AgentCadGraphVisualCompositionError(["agent_cad_visual_cad_source_ambiguous"])
    output_record = (
        _file_record(cad_agent_output_receipt_path)
        if cad_agent_output_receipt_path is not None
        else None
    )
    matrix_record = (
        _file_record(cad_agent_matrix_path) if cad_agent_matrix_path is not None else None
    )
    projection_record = _file_record(mesh_projection_receipt_path)
    visual_review_record = _file_record(cad_agent_visual_review_path)
    graph, graph_stage = _load_graph_authoring(graph_record, verify_files=True)
    candidate_source = {
        "task_id": graph["task_id"],
        "asset_id": graph["asset_id"],
        **(
            {"cad_agent_output_receipt": output_record}
            if output_record is not None
            else {
                "cad_agent_matrix": matrix_record,
                "cad_agent_backend_id": str(cad_agent_backend_id or ""),
            }
        ),
    }
    cad_output, task_freeze = _load_cad_output_source(candidate_source, verify_files=True)
    projection, packet, _ = _load_mesh_projection(projection_record, verify_files=True)
    request = cad_output["request"]
    source_links = {row["link_id"] for row in packet["meshes"]}
    rows, unmapped = _normalize_link_bindings(
        value=list(link_bindings),
        source_link_ids=source_links,
        graph_link_paths=graph["link_paths"],
        graph_stage=graph_stage,
        unmapped_reasons=unmapped_graph_link_reasons,
    )
    cad_side_freeze_digest = str(task_freeze.get("task_freeze_digest") or "")
    graph_side_freeze_digest = str(graph.get("task_freeze_digest") or "")
    if (
        request.get("task_id") != graph.get("task_id")
        or request.get("asset_id") != graph.get("asset_id")
        or not _freeze_join_accepted(
            cad_side_digest=cad_side_freeze_digest,
            graph_side_digest=graph_side_freeze_digest,
            proof=freeze_carry_forward,
        )
        or not _same_file((cad_output.get("artifacts") or {}).get("step"), projection.get("step"))
    ):
        raise AgentCadGraphVisualCompositionError(["agent_cad_visual_binding_identity_mismatch"])
    try:
        selected_cad_agent_visual_review(
            visual_review_record,
            scene_id=str(request.get("scene_id") or ""),
            task_id=str(request.get("task_id") or ""),
            asset_id=str(request.get("asset_id") or ""),
            backend_id=str(((request.get("backend") or {}).get("backend_id")) or ""),
            cad_agent_output_receipt_digest=str(cad_output.get("receipt_digest") or ""),
        )
    except CadAgentReviewMediaError as exc:
        raise AgentCadGraphVisualCompositionError(
            ["agent_cad_visual_reference_review_invalid", str(exc)]
        ) from exc
    binding: dict[str, Any] = {
        "schema_version": BINDING_SCHEMA_VERSION,
        "scene_id": request["scene_id"],
        "task_id": graph["task_id"],
        "asset_id": graph["asset_id"],
        "task_freeze_digest": graph["task_freeze_digest"],
        "graph_authoring_receipt": graph_record,
        **(
            {"cad_agent_output_receipt": output_record}
            if output_record is not None
            else {
                "cad_agent_matrix": matrix_record,
                "cad_agent_backend_id": str(cad_agent_backend_id or ""),
            }
        ),
        "cad_agent_output_receipt_digest": cad_output["receipt_digest"],
        "cad_agent_visual_review": visual_review_record,
        "mesh_projection_receipt": projection_record,
        "link_bindings": rows,
        "unmapped_graph_link_reasons": unmapped,
        # A binding that spans the amendment says so, and carries the proof
        # that justifies it: downstream validation re-checks the embedded
        # proof against the digests it observes, with no side channel.
        **(
            {
                "superseded_task_freeze_digest": cad_side_freeze_digest,
                "freeze_amendment_carry_forward": json.loads(
                    json.dumps(freeze_carry_forward)
                ),
            }
            if cad_side_freeze_digest != graph_side_freeze_digest
            else {}
        ),
        "claim_boundary": dict(_CLAIM_BOUNDARY),
        "binding_digest": "",
    }
    binding["binding_digest"] = canonical_digest(binding, digest_field="binding_digest")
    binding = validate_agent_cad_visual_binding(binding)
    target = Path(output_path).expanduser().resolve()
    if target.exists() or target.is_symlink():
        raise AgentCadGraphVisualCompositionError(["agent_cad_visual_binding_destination_exists"])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(canonical_json(binding) + "\n", encoding="utf-8")
    return binding


def _binding_from_path(path_value: str | Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    path = Path(path_value).expanduser().resolve()
    record = _file_record(path)
    _, binding = _read_json_record(
        record, "agent_cad_visual_binding_file_invalid", verify_files=True
    )
    admitted, context = _validate_binding(binding, verify_files=True)
    assert context is not None
    return path, admitted, context


def _copy_visual_mesh(
    *,
    stage: Any,
    source_mesh: Any,
    destination_path: str,
    transform: Sequence[Sequence[float]],
    material: Any,
    display_color_rgba: Sequence[float],
) -> tuple[int, int]:
    from pxr import Gf, UsdGeom, UsdPhysics, UsdShade

    if source_mesh.GetOrderedXformOps():
        raise AgentCadGraphVisualCompositionError(
            ["agent_cad_visual_projection_mesh_xform_unsupported"]
        )
    source_points = source_mesh.GetPointsAttr().Get()
    counts = source_mesh.GetFaceVertexCountsAttr().Get()
    indices = source_mesh.GetFaceVertexIndicesAttr().Get()
    if not source_points or not counts or not indices:
        raise AgentCadGraphVisualCompositionError(
            ["agent_cad_visual_projection_mesh_topology_invalid"]
        )
    mesh = UsdGeom.Mesh.Define(stage, destination_path)
    mesh.CreatePointsAttr(
        [Gf.Vec3f(*_transform_point(transform, point)) for point in source_points]
    )
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateOrientationAttr(source_mesh.GetOrientationAttr().Get() or UsdGeom.Tokens.rightHanded)
    mesh.CreatePurposeAttr(UsdGeom.Tokens.default_)
    mesh.CreateVisibilityAttr(UsdGeom.Tokens.inherited)
    mesh.CreateDoubleSidedAttr(bool(source_mesh.GetDoubleSidedAttr().Get()))
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    if mesh.GetPrim().HasAPI(UsdPhysics.CollisionAPI):
        raise AgentCadGraphVisualCompositionError(
            ["agent_cad_visual_visual_mesh_collision_present"]
        )
    mesh.GetPrim().SetCustomDataByKey("blueprint:geometryAuthority", "exact_agent_authored_step")
    mesh.GetPrim().SetCustomDataByKey("blueprint:collisionGeometryOnly", False)
    mesh.GetPrim().SetCustomDataByKey(
        "blueprint:agentCadSourceMeshPath", str(source_mesh.GetPath())
    )
    mesh.GetPrim().SetCustomDataByKey(
        "blueprint:agentAuthoredDisplayColorRgba",
        Gf.Vec4f(*(float(value) for value in display_color_rgba)),
    )
    return len(source_points), len(counts)


def _copy_grasp_collision_patch(
    *,
    stage: Any,
    source_mesh: Any,
    transform: Sequence[Sequence[float]],
    graph_link_path: str,
    request: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive one invisible convex collision patch from exact STEP vertices."""

    from pxr import Gf, UsdGeom, UsdPhysics, UsdShade

    if source_mesh.GetOrderedXformOps():
        raise AgentCadGraphVisualCompositionError(
            ["agent_cad_visual_projection_mesh_xform_unsupported"]
        )
    source_points = source_mesh.GetPointsAttr().Get()
    counts = source_mesh.GetFaceVertexCountsAttr().Get()
    indices = source_mesh.GetFaceVertexIndicesAttr().Get()
    if not source_points or not counts or not indices:
        raise AgentCadGraphVisualCompositionError(
            ["agent_cad_grasp_collision_source_topology_invalid"]
        )
    points = [_transform_point(transform, point) for point in source_points]
    axis = int(request["selection_axis_index"])
    minimum = float(request["minimum_axis_value_m"])
    maximum = request.get("maximum_axis_value_m")

    selected_faces: list[list[int]] = []
    offset = 0
    for count in counts:
        face = [int(value) for value in indices[offset : offset + int(count)]]
        offset += int(count)
        if len(face) < 3 or any(index < 0 or index >= len(points) for index in face):
            raise AgentCadGraphVisualCompositionError(
                ["agent_cad_grasp_collision_source_topology_invalid"]
            )
        if all(
            points[index][axis] >= minimum - 1.0e-9
            and (maximum is None or points[index][axis] <= float(maximum) + 1.0e-9)
            for index in face
        ):
            selected_faces.append(face)
    used = sorted({index for face in selected_faces for index in face})
    if len(used) < 4 or len(selected_faces) < 4:
        raise AgentCadGraphVisualCompositionError(
            ["agent_cad_grasp_collision_patch_empty_or_degenerate"]
        )
    remap = {old: new for new, old in enumerate(used)}
    patch_points = [points[index] for index in used]
    patch_faces = [[remap[index] for index in face] for face in selected_faces]
    lower = [min(point[index] for point in patch_points) for index in range(3)]
    upper = [max(point[index] for point in patch_points) for index in range(3)]
    if any(upper[index] - lower[index] <= 1.0e-6 for index in range(3)):
        raise AgentCadGraphVisualCompositionError(
            ["agent_cad_grasp_collision_patch_empty_or_degenerate"]
        )
    outward = [float(value) for value in request["approach_outward_unit_graph_link"]]
    pinch = [float(value) for value in request["pinch_axis_graph_link"]]
    pinch_values = [sum(a * b for a, b in zip(point, pinch, strict=True)) for point in patch_points]
    selection_extreme = max(point[axis] for point in patch_points)
    candidates = [
        point for point in patch_points if abs(point[axis] - selection_extreme) <= 1.0e-6
    ]
    outward_extreme = max(
        sum(a * b for a, b in zip(point, outward, strict=True))
        for point in candidates
    )
    contact_face_points = [
        point
        for point in candidates
        if abs(
            sum(a * b for a, b in zip(point, outward, strict=True))
            - outward_extreme
        )
        <= 1.0e-6
    ]
    # The midpoint of the selected outer/front face is a measured surface
    # point for the convex patch.  Picking an arbitrary corner made an
    # otherwise symmetric jaw target depend on source vertex ordering.
    contact = [
        sum(point[index] for point in contact_face_points)
        / len(contact_face_points)
        for index in range(3)
    ]
    patch_path = (
        f"{graph_link_path}/grasp_collision_patches/{request['patch_id']}"
    )
    UsdGeom.Scope.Define(stage, f"{graph_link_path}/grasp_collision_patches")
    mesh = UsdGeom.Mesh.Define(stage, patch_path)
    mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in patch_points])
    mesh.CreateFaceVertexCountsAttr([len(face) for face in patch_faces])
    mesh.CreateFaceVertexIndicesAttr([index for face in patch_faces for index in face])
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateOrientationAttr(source_mesh.GetOrientationAttr().Get() or UsdGeom.Tokens.rightHanded)
    mesh.CreatePurposeAttr(UsdGeom.Tokens.guide)
    mesh.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    mesh.CreateDoubleSidedAttr(bool(source_mesh.GetDoubleSidedAttr().Get()))
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr().Set(
        "convexHull"
    )
    material = stage.GetPrimAtPath(
        f"/Asset/materials/{request['graph_link_id']}"
    )
    if not material.IsValid():
        raise AgentCadGraphVisualCompositionError(
            ["agent_cad_grasp_collision_physics_material_missing"]
        )
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
        UsdShade.Material(material), materialPurpose="physics"
    )
    mesh.GetPrim().SetCustomDataByKey(
        "blueprint:geometryProvenance", "exact_agent_authored_step_convex_patch"
    )
    mesh.GetPrim().SetCustomDataByKey("blueprint:collisionGeometryOnly", True)
    mesh.GetPrim().SetCustomDataByKey(
        "blueprint:agentCadSourceMeshPath", str(source_mesh.GetPath())
    )
    mesh.GetPrim().SetCustomDataByKey(
        "blueprint:graspAffordanceRole", request["grasp_affordance_role"]
    )
    mesh.GetPrim().SetCustomDataByKey(
        "blueprint:graspContactPointLinkM", Gf.Vec3d(*contact)
    )
    mesh.GetPrim().SetCustomDataByKey(
        "blueprint:graspApproachOutwardUnitLink", Gf.Vec3d(*outward)
    )
    mesh.GetPrim().SetCustomDataByKey(
        "blueprint:graspPinchAxisLink", Gf.Vec3d(*pinch)
    )
    pinch_span = max(pinch_values) - min(pinch_values)
    mesh.GetPrim().SetCustomDataByKey("blueprint:graspPinchSpanM", pinch_span)
    return {
        **dict(request),
        "collision_prim_path": patch_path,
        "source_point_count": len(source_points),
        "selected_point_count": len(patch_points),
        "selected_face_count": len(patch_faces),
        "selected_bounds_graph_link_m": {"minimum": lower, "maximum": upper},
        "contact_point_graph_link_m": contact,
        "pinch_span_m": pinch_span,
        "selected_point_digest": canonical_digest({"points": patch_points}),
    }


def _validate_composition_stage(stage: Any, receipt: Mapping[str, Any]) -> None:
    from pxr import UsdGeom, UsdPhysics

    if str(stage.GetDefaultPrim().GetPath()) != "/Asset":
        raise AgentCadGraphVisualCompositionError(["agent_cad_visual_composition_usd_invalid"])
    _collision_visual_isolation(stage)
    expected = receipt.get("visual_meshes")
    if not isinstance(expected, list) or not expected:
        raise AgentCadGraphVisualCompositionError(["agent_cad_visual_composition_usd_invalid"])
    actual_paths: set[str] = set()
    for row in expected:
        if not isinstance(row, Mapping):
            raise AgentCadGraphVisualCompositionError(["agent_cad_visual_composition_usd_invalid"])
        prim = stage.GetPrimAtPath(str(row.get("visual_mesh_path") or ""))
        if not prim.IsA(UsdGeom.Mesh):
            raise AgentCadGraphVisualCompositionError(["agent_cad_visual_composition_usd_invalid"])
        imageable = UsdGeom.Imageable(prim)
        if (
            prim.HasAPI(UsdPhysics.CollisionAPI)
            or imageable.ComputePurpose() != UsdGeom.Tokens.default_
            or imageable.ComputeVisibility() == UsdGeom.Tokens.invisible
            or prim.GetCustomDataByKey("blueprint:geometryAuthority") != "exact_agent_authored_step"
            or prim.GetCustomDataByKey("blueprint:collisionGeometryOnly") is not False
            or prim.GetCustomDataByKey("blueprint:agentCadSourceMeshPath")
            != row.get("source_mesh_path")
        ):
            raise AgentCadGraphVisualCompositionError(["agent_cad_visual_composition_usd_invalid"])
        actual_paths.add(str(prim.GetPath()))
    if actual_paths != {str(row["visual_mesh_path"]) for row in expected}:
        raise AgentCadGraphVisualCompositionError(["agent_cad_visual_composition_usd_invalid"])
    patches = receipt.get("grasp_collision_patches") or []
    actual_patch_paths: set[str] = set()
    for row in patches:
        if not isinstance(row, Mapping):
            raise AgentCadGraphVisualCompositionError(
                ["agent_cad_grasp_collision_patch_receipt_invalid"]
            )
        prim = stage.GetPrimAtPath(str(row.get("collision_prim_path") or ""))
        mesh = UsdGeom.Mesh(prim)
        imageable = UsdGeom.Imageable(prim)
        approximation = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
        if (
            not mesh
            or not prim.HasAPI(UsdPhysics.CollisionAPI)
            or str(approximation) != "convexHull"
            or imageable.ComputePurpose() != UsdGeom.Tokens.guide
            or imageable.ComputeVisibility() != UsdGeom.Tokens.invisible
            or prim.GetCustomDataByKey("blueprint:collisionGeometryOnly") is not True
            or prim.GetCustomDataByKey("blueprint:graspAffordanceRole")
            != row.get("grasp_affordance_role")
            or len(mesh.GetPointsAttr().Get() or []) != row.get("selected_point_count")
            or len(mesh.GetFaceVertexCountsAttr().Get() or [])
            != row.get("selected_face_count")
        ):
            raise AgentCadGraphVisualCompositionError(
                ["agent_cad_grasp_collision_patch_usd_invalid"]
            )
        actual_patch_paths.add(str(prim.GetPath()))
    if actual_patch_paths != {
        str(row["collision_prim_path"]) for row in patches
    }:
        raise AgentCadGraphVisualCompositionError(
            ["agent_cad_grasp_collision_patch_usd_invalid"]
        )


def materialize_agent_cad_visual_composition(
    *,
    binding_path: str | Path,
    destination_usd_path: str | Path,
    receipt_path: str | Path | None = None,
    grasp_collision_patches: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Copy exact agent mesh visuals into an immutable graph-asset USD clone."""

    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade
    except ImportError as exc:  # pragma: no cover - environment guard
        raise AgentCadGraphVisualCompositionError(
            ["agent_cad_visual_openusd_runtime_missing"]
        ) from exc
    binding_file, binding, context = _binding_from_path(binding_path)
    destination = Path(destination_usd_path).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise AgentCadGraphVisualCompositionError(
            ["agent_cad_visual_composition_destination_exists"]
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    graph_path = Path(context["graph"]["output_usd"]["path"])
    graph_stage = Usd.Stage.Open(str(graph_path), load=Usd.Stage.LoadAll)
    if graph_stage is None or not graph_stage.Export(str(destination)):
        raise AgentCadGraphVisualCompositionError(["agent_cad_visual_graph_export_failed"])
    stage = Usd.Stage.Open(str(destination), load=Usd.Stage.LoadAll)
    source_stage = Usd.Stage.Open(
        str(context["projection"]["output_usd"]["path"]), load=Usd.Stage.LoadAll
    )
    if stage is None or source_stage is None:
        raise AgentCadGraphVisualCompositionError(["agent_cad_visual_composition_usd_invalid"])
    patch_requests = _normalize_grasp_collision_patches(
        None if grasp_collision_patches is None else list(grasp_collision_patches)
    )
    materials: dict[tuple[float, float, float, float], Any] = {}

    def material_for(color: Sequence[float]) -> Any:
        key = tuple(float(value) for value in color)
        if len(key) != 4 or any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in key
        ):
            raise AgentCadGraphVisualCompositionError(
                ["agent_cad_visual_agent_authored_color_invalid"]
            )
        if key in materials:
            return materials[key]
        name = f"agent_authored_display_{len(materials):03d}"
        material = UsdShade.Material.Define(stage, f"/Asset/materials/{name}")
        shader = UsdShade.Shader.Define(stage, f"/Asset/materials/{name}/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*key[:3]))
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(key[3])
        shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        materials[key] = material
        return material

    by_agent_link = {row["agent_link_id"]: row for row in binding["link_bindings"]}
    visual_meshes: list[dict[str, Any]] = []
    source_mesh_context: dict[str, tuple[Any, Mapping[str, Any], Mapping[str, Any]]] = {}
    used_paths: set[str] = set()
    for index, row in enumerate(context["packet"]["meshes"]):
        source_path = row["prim_path"]
        source_mesh = UsdGeom.Mesh(source_stage.GetPrimAtPath(source_path))
        if not source_mesh:
            raise AgentCadGraphVisualCompositionError(["agent_cad_visual_projection_mesh_missing"])
        display_color = row.get("agent_authored_display_color_rgba")
        if not isinstance(display_color, list):
            raise AgentCadGraphVisualCompositionError(
                ["agent_cad_visual_agent_authored_color_missing"]
            )
        mapping = by_agent_link[row["link_id"]]
        source_mesh_context[source_path] = (source_mesh, row, mapping)
        graph_link_path = context["graph"]["link_paths"][mapping["graph_link_id"]]
        UsdGeom.Scope.Define(stage, f"{graph_link_path}/visuals")
        leaf = _safe_prim_token(
            f"{mapping['agent_link_id']}__{row['solid_id']}", f"mesh_{index:04d}"
        )
        visual_path = f"{graph_link_path}/visuals/{leaf}"
        suffix = 2
        while visual_path in used_paths or stage.GetPrimAtPath(visual_path).IsValid():
            visual_path = f"{graph_link_path}/visuals/{leaf}_{suffix}"
            suffix += 1
        used_paths.add(visual_path)
        point_count, face_count = _copy_visual_mesh(
            stage=stage,
            source_mesh=source_mesh,
            destination_path=visual_path,
            transform=mapping["T_graph_link_from_agent_asset"],
            material=material_for(display_color),
            display_color_rgba=display_color,
        )
        visual_meshes.append(
            {
                "source_mesh_path": source_path,
                "agent_link_id": mapping["agent_link_id"],
                "graph_link_id": mapping["graph_link_id"],
                "T_graph_link_from_agent_asset": mapping["T_graph_link_from_agent_asset"],
                "visual_mesh_path": visual_path,
                "point_count": point_count,
                "face_count": face_count,
                "agent_authored_display_color_rgba": display_color,
            }
        )
    collision_patches: list[dict[str, Any]] = []
    for request in patch_requests:
        context_row = source_mesh_context.get(request["source_mesh_path"])
        if context_row is None:
            raise AgentCadGraphVisualCompositionError(
                ["agent_cad_grasp_collision_source_mesh_missing"]
            )
        source_mesh, _packet_row, mapping = context_row
        if mapping["graph_link_id"] != request["graph_link_id"]:
            raise AgentCadGraphVisualCompositionError(
                ["agent_cad_grasp_collision_link_binding_mismatch"]
            )
        graph_link_path = context["graph"]["link_paths"][request["graph_link_id"]]
        collision_patches.append(
            _copy_grasp_collision_patch(
                stage=stage,
                source_mesh=source_mesh,
                transform=mapping["T_graph_link_from_agent_asset"],
                graph_link_path=graph_link_path,
                request=request,
            )
        )
    stage.GetRootLayer().documentation = (
        "Graph collision candidate plus exact agent-authored STEP visual geometry; "
        "native behavior and appearance material qualification remain unresolved."
    )
    stage.GetRootLayer().Save()
    output = _file_record(destination)
    receipt: dict[str, Any] = {
        "schema_version": COMPOSITION_SCHEMA_VERSION,
        "status": "agent_cad_visuals_composed",
        "scene_id": binding["scene_id"],
        "task_id": binding["task_id"],
        "asset_id": binding["asset_id"],
        "task_freeze_digest": binding["task_freeze_digest"],
        "binding": {
            **_file_record(binding_file),
            "binding_digest": binding["binding_digest"],
        },
        "output_usd": output,
        "visual_meshes": visual_meshes,
        "visual_mesh_count": len(visual_meshes),
        "collision_visual_isolation_verified": True,
        "agent_authored_display_color_mesh_count": len(visual_meshes),
        "neutral_fallback_mesh_count": 0,
        "generated_texture_map_count": 0,
        **(
            {
                "grasp_collision_patches": collision_patches,
                "grasp_collision_patch_count": len(collision_patches),
            }
            if collision_patches
            else {}
        ),
        "claim_boundary": dict(_COMPOSITION_CLAIM_BOUNDARY),
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    _validate_composition_stage(stage, receipt)
    target = (
        Path(receipt_path).expanduser().resolve()
        if receipt_path is not None
        else destination.with_suffix(".receipt.json")
    )
    if target.exists() or target.is_symlink():
        raise AgentCadGraphVisualCompositionError(
            ["agent_cad_visual_composition_receipt_destination_exists"]
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    return receipt


def validate_agent_cad_visual_composition(
    value: Mapping[str, Any], *, verify_files: bool = True
) -> dict[str, Any]:
    """Verify a composed USD without upgrading its simulator/physics claims."""

    receipt = _clone(value)
    patches = receipt.get("grasp_collision_patches")
    patch_rows = [] if patches is None else patches
    if (
        receipt.get("schema_version") != COMPOSITION_SCHEMA_VERSION
        or receipt.get("status") != "agent_cad_visuals_composed"
        or receipt.get("claim_boundary") != _COMPOSITION_CLAIM_BOUNDARY
        or receipt.get("receipt_digest") != canonical_digest(receipt, digest_field="receipt_digest")
        or not _record_valid(receipt.get("output_usd"), verify_files=verify_files)
    ):
        raise AgentCadGraphVisualCompositionError(["agent_cad_visual_composition_receipt_invalid"])
    _, binding = _read_json_record(
        receipt.get("binding"),
        "agent_cad_visual_composition_binding_invalid",
        verify_files=verify_files,
    )
    admitted_binding = validate_agent_cad_visual_binding(binding, verify_files=verify_files)
    if (
        receipt.get("binding", {}).get("binding_digest") != admitted_binding.get("binding_digest")
        or any(
            receipt.get(field) != admitted_binding.get(field)
            for field in ("scene_id", "task_id", "asset_id", "task_freeze_digest")
        )
        or receipt.get("visual_mesh_count") != len(receipt.get("visual_meshes") or [])
        or receipt.get("agent_authored_display_color_mesh_count")
        != receipt.get("visual_mesh_count")
        or receipt.get("neutral_fallback_mesh_count") != 0
        or receipt.get("generated_texture_map_count") != 0
        or receipt.get("collision_visual_isolation_verified") is not True
        or not isinstance(patch_rows, list)
        or (
            patches is None
            and "grasp_collision_patch_count" in receipt
        )
        or (
            patches is not None
            and (
                not patch_rows
                or receipt.get("grasp_collision_patch_count") != len(patch_rows)
                or _normalize_grasp_collision_patches(patch_rows) != [
                    {
                        key: row[key]
                        for key in (
                            "patch_id",
                            "graph_link_id",
                            "source_mesh_path",
                            "grasp_affordance_role",
                            "selection_axis_index",
                            "minimum_axis_value_m",
                            "maximum_axis_value_m",
                            "approach_outward_unit_graph_link",
                            "pinch_axis_graph_link",
                            "collision_approximation",
                        )
                    }
                    for row in patch_rows
                ]
            )
        )
    ):
        raise AgentCadGraphVisualCompositionError(["agent_cad_visual_composition_binding_mismatch"])
    if verify_files:
        try:
            from pxr import Usd
        except ImportError as exc:  # pragma: no cover - environment guard
            raise AgentCadGraphVisualCompositionError(
                ["agent_cad_visual_openusd_runtime_missing"]
            ) from exc
        stage = Usd.Stage.Open(str(receipt["output_usd"]["path"]), load=Usd.Stage.LoadAll)
        if stage is None:
            raise AgentCadGraphVisualCompositionError(["agent_cad_visual_composition_usd_invalid"])
        _validate_composition_stage(stage, receipt)
    return receipt


def validate_agent_cad_visual_composition_set(
    value: Mapping[str, Any], *, verify_files: bool = True
) -> dict[str, Any]:
    """Validate a co-present 1–5 replacement visual-composition set."""

    payload = _clone(value)
    if (
        payload.get("schema_version") != COMPOSITION_SET_SCHEMA_VERSION
        or payload.get("maximum_replacement_objects") != MAX_REPLACEMENT_OBJECTS
        or payload.get("set_digest") != canonical_digest(payload, digest_field="set_digest")
    ):
        raise AgentCadGraphVisualCompositionError(
            ["agent_cad_visual_composition_set_digest_invalid"]
        )
    rows = payload.get("compositions")
    if not isinstance(rows, list) or not rows or len(rows) > MAX_REPLACEMENT_OBJECTS:
        raise AgentCadGraphVisualCompositionError(
            ["agent_cad_visual_composition_set_count_invalid"]
        )
    identities: set[tuple[str, str]] = set()
    for row in rows:
        _, receipt = _read_json_record(
            row,
            "agent_cad_visual_composition_set_receipt_invalid",
            verify_files=verify_files,
        )
        composition = validate_agent_cad_visual_composition(receipt, verify_files=verify_files)
        if composition.get("scene_id") != payload.get("scene_id"):
            raise AgentCadGraphVisualCompositionError(
                ["agent_cad_visual_composition_set_scene_mismatch"]
            )
        identity = (str(composition["task_id"]), str(composition["asset_id"]))
        if identity in identities:
            raise AgentCadGraphVisualCompositionError(
                ["agent_cad_visual_composition_set_identity_duplicate"]
            )
        identities.add(identity)
    return payload


__all__ = [
    "BINDING_SCHEMA_VERSION",
    "COMPOSITION_SCHEMA_VERSION",
    "COMPOSITION_SET_SCHEMA_VERSION",
    "AgentCadGraphVisualCompositionError",
    "materialize_agent_cad_visual_composition",
    "seal_agent_cad_visual_binding",
    "validate_agent_cad_visual_binding",
    "validate_agent_cad_visual_composition",
    "validate_agent_cad_visual_composition_set",
]
