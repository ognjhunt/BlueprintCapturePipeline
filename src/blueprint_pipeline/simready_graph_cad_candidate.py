"""Materialize prompt-bound parametric CAD candidates from graph asset specs.

This is the reusable, local half of the Text-to-CAD comparison.  A semantic
prompt is retained, but dimensions, link membership, and poses come only from
the digest-bound graph spec.  The STEP output is a generated candidate: it does
not establish observed hidden geometry, articulation behavior, native import,
or physical equivalence.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import re
import shutil
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest, canonical_json
from .simready_graph_asset import (
    SimReadyGraphAssetError,
    validate_simready_graph_asset_spec,
)


REQUEST_SCHEMA_VERSION = "simready_graph_cad_request.v1"
RECEIPT_SCHEMA_VERSION = "simready_graph_cad_candidate.v1"
BINDING_SCHEMA_VERSION = "simready_graph_cad_candidate_binding.v1"
OCP_PACKAGE_VERSION = "7.8.1.1.post1"


class SimReadyGraphCadCandidateError(ValueError):
    """Stable failures for graph-to-CAD candidate authoring."""

    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def normalize_step_header_for_digest(source_text: str) -> str:
    """Remove OpenCascade's process-global metadata from stable STEP geometry."""

    normalized_text, substitutions = re.subn(
        r"(FILE_NAME\('Open CASCADE Shape Model',')[^']+(')",
        r"\g<1>1970-01-01T00:00:00\2",
        source_text,
        count=1,
    )
    if substitutions != 1:
        raise SimReadyGraphCadCandidateError(
            ["graph_cad_step_header_timestamp_not_normalizable"]
        )
    normalized_text = re.sub(
        r"(Open CASCADE STEP translator [0-9.]+ )\d+(\.\d+)?",
        lambda match: f"{match.group(1)}1{match.group(2) or ''}",
        normalized_text,
    )
    occurrence = 0

    def normalize_occurrence(match: re.Match[str]) -> str:
        nonlocal occurrence
        occurrence += 1
        return f"{match.group(1)}{occurrence}{match.group(2)}"

    normalized_text = re.sub(
        r"(NEXT_ASSEMBLY_USAGE_OCCURRENCE\(')\d+(')",
        normalize_occurrence,
        normalized_text,
    )
    return normalized_text


def _digest(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 71 and text.startswith("sha256:") and all(
        character in "0123456789abcdef" for character in text[7:]
    )


def validate_graph_cad_request(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        request = json.loads(json.dumps(value))
    except (TypeError, ValueError) as exc:
        raise SimReadyGraphCadCandidateError(["graph_cad_request_invalid"]) from exc
    errors: list[str] = []
    if request.get("schema_version") != REQUEST_SCHEMA_VERSION:
        errors.append("graph_cad_request_schema_invalid")
    for field in ("request_id", "asset_id", "task_id", "prompt"):
        if not str(request.get(field) or "").strip():
            errors.append(f"graph_cad_request_{field}_missing")
    for field in ("task_freeze_digest", "spec_digest"):
        if not _digest(request.get(field)):
            errors.append(f"graph_cad_request_{field}_invalid")
    if request.get("geometry_authority") != (
        "digest_bound_graph_spec_dimensions_and_poses_prompt_is_semantic_only"
    ):
        errors.append("graph_cad_request_geometry_authority_invalid")
    if request.get("generated_hidden_geometry_is_observed_truth") is not False:
        errors.append("graph_cad_request_hidden_geometry_claim_invalid")
    if request.get("target_format") != "STEP_AP214":
        errors.append("graph_cad_request_target_format_invalid")
    expected = canonical_digest(request, digest_field="request_digest")
    if request.get("request_digest") != expected:
        errors.append("graph_cad_request_digest_invalid")
    if errors:
        raise SimReadyGraphCadCandidateError(errors)
    return request


def seal_graph_cad_request(
    *, spec: Mapping[str, Any], request_id: str, prompt: str
) -> dict[str, Any]:
    try:
        admitted = validate_simready_graph_asset_spec(spec)
    except SimReadyGraphAssetError as exc:
        raise SimReadyGraphCadCandidateError(exc.codes) from exc
    request: dict[str, Any] = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "request_id": request_id,
        "asset_id": admitted["asset_id"],
        "task_id": admitted["task_id"],
        "task_freeze_digest": admitted["task_freeze_digest"],
        "spec_digest": admitted["spec_digest"],
        "prompt": str(prompt).strip(),
        "geometry_authority": (
            "digest_bound_graph_spec_dimensions_and_poses_prompt_is_semantic_only"
        ),
        "generated_hidden_geometry_is_observed_truth": False,
        "target_format": "STEP_AP214",
        "request_digest": "",
    }
    request["request_digest"] = canonical_digest(
        request, digest_field="request_digest"
    )
    return validate_graph_cad_request(request)


def _quat_multiply(left: Sequence[float], right: Sequence[float]) -> list[float]:
    lx, ly, lz, lw = (float(value) for value in left)
    rx, ry, rz, rw = (float(value) for value in right)
    return [
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    ]


def _rotate_vector(quaternion: Sequence[float], vector: Sequence[float]) -> list[float]:
    x, y, z, w = (float(value) for value in quaternion)
    vx, vy, vz = (float(value) for value in vector)
    # q * v * q^-1, expanded for a unit quaternion.
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return [
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    ]


def _compose_pose(
    parent_translation: Sequence[float],
    parent_orientation: Sequence[float],
    local_translation: Sequence[float],
    local_orientation: Sequence[float],
) -> tuple[list[float], list[float]]:
    rotated = _rotate_vector(parent_orientation, local_translation)
    return (
        [
            float(parent_translation[index]) + rotated[index]
            for index in range(3)
        ],
        _quat_multiply(parent_orientation, local_orientation),
    )


def _axis_angle(xyzw: Sequence[float]) -> tuple[list[float], float]:
    x, y, z, w = (float(value) for value in xyzw)
    w = max(-1.0, min(1.0, w))
    angle = 2.0 * math.acos(w)
    scale = math.sqrt(max(0.0, 1.0 - w * w))
    if scale < 1e-12:
        return [1.0, 0.0, 0.0], 0.0
    return [x / scale, y / scale, z / scale], math.degrees(angle)


def _opencascade_export(
    admitted: Mapping[str, Any], destination: Path
) -> dict[str, Any]:
    try:
        from OCP.BRep import BRep_Builder  # noqa: PLC0415
        from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform  # noqa: PLC0415
        from OCP.BRepPrimAPI import (  # noqa: PLC0415
            BRepPrimAPI_MakeBox,
            BRepPrimAPI_MakeCylinder,
        )
        from OCP.IFSelect import IFSelect_RetDone  # noqa: PLC0415
        from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer  # noqa: PLC0415
        from OCP.TopoDS import TopoDS_Compound  # noqa: PLC0415
        from OCP.gp import (  # noqa: PLC0415
            gp_Ax2,
            gp_Dir,
            gp_Pnt,
            gp_Quaternion,
            gp_Trsf,
            gp_Vec,
        )
    except ImportError as exc:
        raise SimReadyGraphCadCandidateError(
            ["graph_cad_opencascade_runtime_missing"]
        ) from exc
    observed_version = importlib.metadata.version("cadquery-ocp")
    if observed_version != OCP_PACKAGE_VERSION:
        raise SimReadyGraphCadCandidateError(
            ["graph_cad_opencascade_version_mismatch"]
        )
    compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(compound)
    part_count = 0
    provenance_counts: dict[str, int] = {}
    for link in admitted["links"]:
        rest = link["rest_pose"]
        for geometry in link["geometry"]:
            if geometry["kind"] == "box":
                sx, sy, sz = (float(value) for value in geometry["size_m"])
                shape = BRepPrimAPI_MakeBox(
                    gp_Pnt(-sx / 2.0, -sy / 2.0, -sz / 2.0), sx, sy, sz
                ).Shape()
            else:
                radius = float(geometry["radius_m"])
                height = float(geometry["height_m"])
                shape = BRepPrimAPI_MakeCylinder(
                    gp_Ax2(
                        gp_Pnt(-height / 2.0, 0.0, 0.0),
                        gp_Dir(1.0, 0.0, 0.0),
                    ),
                    radius,
                    height,
                ).Shape()
            translation, orientation = _compose_pose(
                rest["translation_m"],
                rest["orientation_xyzw"],
                geometry["translation_m"],
                geometry["orientation_xyzw"],
            )
            transform = gp_Trsf()
            transform.SetRotation(gp_Quaternion(*orientation))
            transform.SetTranslationPart(gp_Vec(*translation))
            placed = BRepBuilderAPI_Transform(shape, transform, True).Shape()
            builder.Add(compound, placed)
            part_count += 1
            provenance = str(geometry["provenance"])
            provenance_counts[provenance] = provenance_counts.get(provenance, 0) + 1
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = STEPControl_Writer()
    if writer.Transfer(compound, STEPControl_AsIs) != IFSelect_RetDone:
        raise SimReadyGraphCadCandidateError(["graph_cad_step_transfer_failed"])
    if writer.Write(str(destination)) != IFSelect_RetDone:
        raise SimReadyGraphCadCandidateError(["graph_cad_step_write_failed"])
    source_text = destination.read_text(encoding="utf-8")
    normalized_text = normalize_step_header_for_digest(source_text)
    destination.write_text(normalized_text, encoding="utf-8", newline="\n")
    return {
        "exporter": "direct_opencascade",
        "exporter_version": observed_version,
        "exporter_package": "cadquery-ocp",
        "step_schema": "AP214",
        "deterministic_header_timestamp": "1970-01-01T00:00:00",
        "part_count": part_count,
        "geometry_provenance_counts": dict(sorted(provenance_counts.items())),
    }


def materialize_graph_cad_candidate(
    *,
    request: Mapping[str, Any],
    spec_path: str | Path,
    destination_step: str | Path,
    output_receipt_path: str | Path | None = None,
    exporter: Callable[[Mapping[str, Any], Path], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Export one exact graph spec into a prompt-bound STEP candidate."""

    admitted_request = validate_graph_cad_request(request)
    source = Path(spec_path).expanduser().resolve()
    destination = Path(destination_step).expanduser().resolve()
    if not source.is_file() or source.is_symlink():
        raise SimReadyGraphCadCandidateError(["graph_cad_spec_path_invalid"])
    try:
        raw_spec = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SimReadyGraphCadCandidateError(["graph_cad_spec_invalid"]) from exc
    try:
        admitted = validate_simready_graph_asset_spec(raw_spec)
    except SimReadyGraphAssetError as exc:
        raise SimReadyGraphCadCandidateError(exc.codes) from exc
    if any(
        admitted_request[field] != admitted[field]
        for field in ("asset_id", "task_id", "task_freeze_digest", "spec_digest")
    ):
        raise SimReadyGraphCadCandidateError(["graph_cad_request_spec_mismatch"])
    if destination.exists() or destination.is_symlink():
        raise SimReadyGraphCadCandidateError(["graph_cad_destination_exists"])
    retained_spec = destination.parent / "source_spec.json"
    if retained_spec.exists() or retained_spec.is_symlink():
        raise SimReadyGraphCadCandidateError(["graph_cad_retained_spec_exists"])
    retained_spec.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, retained_spec)
    if _sha256(retained_spec) != _sha256(source):
        raise SimReadyGraphCadCandidateError(["graph_cad_retained_spec_copy_mismatch"])
    export_record = dict((exporter or _opencascade_export)(admitted, destination))
    if (
        not destination.is_file()
        or destination.stat().st_size <= 32
        or b"ISO-10303-21" not in destination.read_bytes()[:256]
    ):
        raise SimReadyGraphCadCandidateError(["graph_cad_step_output_invalid"])
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "generated_parametric_cad_candidate_authored",
        "request": admitted_request,
        "source_spec": {
            "path": str(retained_spec),
            "size_bytes": retained_spec.stat().st_size,
            "sha256": _sha256(retained_spec),
            "spec_digest": admitted["spec_digest"],
            "copied_byte_for_byte_from": str(source),
        },
        "asset_id": admitted["asset_id"],
        "task_id": admitted["task_id"],
        "task_freeze_digest": admitted["task_freeze_digest"],
        "cad_output": {
            "path": str(destination),
            "size_bytes": destination.stat().st_size,
            "sha256": _sha256(destination),
            "format": "STEP_AP214",
        },
        "generator": export_record,
        "claim_boundary": {
            "generated_cad_candidate_only": True,
            "prompt_is_geometry_authority": False,
            "hidden_geometry_observed": False,
            "articulation_behavior_qualified": False,
            "native_simulator_import_qualified": False,
            "physical_equivalence_proven": False,
        },
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    if output_receipt_path is not None:
        output = Path(output_receipt_path).expanduser().resolve()
        if output.exists() or output.is_symlink():
            raise SimReadyGraphCadCandidateError(["graph_cad_receipt_output_exists"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    return receipt


def bind_graph_cad_candidate_receipt(
    *, receipt_path: str | Path, evidence_root: str | Path
) -> dict[str, Any]:
    """Verify an external CAD receipt and emit a portable checked-in binding."""

    root = Path(evidence_root).expanduser().resolve()
    path = Path(receipt_path).expanduser().resolve()
    if path == root or root not in path.parents or path.is_symlink() or not path.is_file():
        raise SimReadyGraphCadCandidateError(["graph_cad_binding_receipt_path_invalid"])
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SimReadyGraphCadCandidateError(["graph_cad_binding_receipt_invalid"]) from exc
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION
        or receipt.get("status") != "generated_parametric_cad_candidate_authored"
        or receipt.get("receipt_digest")
        != canonical_digest(receipt, digest_field="receipt_digest")
    ):
        raise SimReadyGraphCadCandidateError(["graph_cad_binding_receipt_invalid"])
    validate_graph_cad_request(receipt.get("request") or {})
    verified_files: dict[str, dict[str, Any]] = {}
    for role in ("source_spec", "cad_output"):
        row = receipt.get(role)
        candidate = Path(str((row or {}).get("path") or "")).expanduser().resolve()
        if (
            not isinstance(row, Mapping)
            or candidate == root
            or root not in candidate.parents
            or candidate.is_symlink()
            or not candidate.is_file()
            or candidate.stat().st_size != row.get("size_bytes")
            or _sha256(candidate) != row.get("sha256")
        ):
            raise SimReadyGraphCadCandidateError(
                [f"graph_cad_binding_{role}_invalid"]
            )
        verified_files[role] = {
            "relative_path": candidate.relative_to(root).as_posix(),
            "size_bytes": candidate.stat().st_size,
            "sha256": _sha256(candidate),
        }
    binding: dict[str, Any] = {
        "schema_version": BINDING_SCHEMA_VERSION,
        "status": "generated_cad_candidate_bound_not_simulator_qualified",
        "asset_id": receipt["asset_id"],
        "task_id": receipt["task_id"],
        "task_freeze_digest": receipt["task_freeze_digest"],
        "request_digest": receipt["request"]["request_digest"],
        "receipt": {
            "relative_path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "receipt_digest": receipt["receipt_digest"],
        },
        "files": verified_files,
        "claim_boundary": receipt["claim_boundary"],
        "binding_digest": "",
    }
    binding["binding_digest"] = canonical_digest(
        binding, digest_field="binding_digest"
    )
    return validate_graph_cad_candidate_binding(binding)


def validate_graph_cad_candidate_binding(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the portable, repository-safe projection of a CAD receipt."""

    try:
        binding = json.loads(json.dumps(value))
    except (TypeError, ValueError) as exc:
        raise SimReadyGraphCadCandidateError(["graph_cad_binding_invalid"]) from exc
    errors: list[str] = []
    if binding.get("schema_version") != BINDING_SCHEMA_VERSION:
        errors.append("graph_cad_binding_schema_invalid")
    if binding.get("status") != "generated_cad_candidate_bound_not_simulator_qualified":
        errors.append("graph_cad_binding_status_invalid")
    for field in ("asset_id", "task_id"):
        if not str(binding.get(field) or ""):
            errors.append(f"graph_cad_binding_{field}_invalid")
    for field in ("task_freeze_digest", "request_digest"):
        if not _digest(binding.get(field)):
            errors.append(f"graph_cad_binding_{field}_invalid")
    receipt = binding.get("receipt")
    rows = binding.get("files")
    records = {
        "receipt": receipt,
        "source_spec": (rows or {}).get("source_spec") if isinstance(rows, Mapping) else None,
        "cad_output": (rows or {}).get("cad_output") if isinstance(rows, Mapping) else None,
    }
    for role, row in records.items():
        relative = str((row or {}).get("relative_path") or "")
        if (
            not isinstance(row, Mapping)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not isinstance(row.get("size_bytes"), int)
            or row.get("size_bytes", 0) <= 0
            or not _digest(row.get("sha256"))
            or (role == "receipt" and not _digest(row.get("receipt_digest")))
        ):
            errors.append(f"graph_cad_binding_{role}_record_invalid")
    claims = binding.get("claim_boundary")
    if (
        not isinstance(claims, Mapping)
        or claims.get("generated_cad_candidate_only") is not True
        or claims.get("prompt_is_geometry_authority") is not False
        or claims.get("hidden_geometry_observed") is not False
        or claims.get("articulation_behavior_qualified") is not False
        or claims.get("native_simulator_import_qualified") is not False
        or claims.get("physical_equivalence_proven") is not False
    ):
        errors.append("graph_cad_binding_claim_boundary_invalid")
    if binding.get("binding_digest") != canonical_digest(
        binding, digest_field="binding_digest"
    ):
        errors.append("graph_cad_binding_digest_invalid")
    if errors:
        raise SimReadyGraphCadCandidateError(errors)
    return binding


__all__ = [
    "OCP_PACKAGE_VERSION",
    "BINDING_SCHEMA_VERSION",
    "RECEIPT_SCHEMA_VERSION",
    "REQUEST_SCHEMA_VERSION",
    "SimReadyGraphCadCandidateError",
    "bind_graph_cad_candidate_receipt",
    "materialize_graph_cad_candidate",
    "normalize_step_header_for_digest",
    "seal_graph_cad_request",
    "validate_graph_cad_request",
    "validate_graph_cad_candidate_binding",
]
