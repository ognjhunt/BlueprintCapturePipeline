"""Seal a task-neutral CAD-local to observed-world frame registration.

An observed object pose does not define an imported asset's semantic front. A
replacement is therefore inadmissible until its agent-authored local forward
and up axes are explicitly mapped to independently reviewed observed axes. This
contract is reusable for one through five co-present replacement objects and
does not infer an orientation from an axis-aligned bounding box.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "replacement_asset_frame_registration.v1"


class ReplacementAssetFrameRegistrationError(ValueError):
    """Stable fail-closed asset-frame registration failure."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _record(path: str | Path) -> dict[str, Any]:
    value = Path(path).expanduser().resolve()
    if value.is_symlink() or not value.is_file() or value.stat().st_size <= 0:
        raise ReplacementAssetFrameRegistrationError("asset_frame_reference_invalid")
    return {"path": str(value), "size_bytes": value.stat().st_size, "sha256": _sha256(value)}


def _vector(value: Any, code: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise ReplacementAssetFrameRegistrationError(code)
    try:
        result = [float(component) for component in value]
    except (TypeError, ValueError) as exc:
        raise ReplacementAssetFrameRegistrationError(code) from exc
    norm = math.sqrt(sum(component * component for component in result))
    if not all(math.isfinite(component) for component in result) or abs(norm - 1.0) > 1e-6:
        raise ReplacementAssetFrameRegistrationError(code)
    return result


def _frame(forward: Any, up: Any, code: str) -> tuple[list[float], list[float], list[float]]:
    forward_row = _vector(forward, code)
    up_row = _vector(up, code)
    if abs(sum(a * b for a, b in zip(forward_row, up_row, strict=True))) > 1e-6:
        raise ReplacementAssetFrameRegistrationError(code)
    right = [
        forward_row[1] * up_row[2] - forward_row[2] * up_row[1],
        forward_row[2] * up_row[0] - forward_row[0] * up_row[2],
        forward_row[0] * up_row[1] - forward_row[1] * up_row[0],
    ]
    return forward_row, up_row, right


def _registration_matrix(
    asset_forward: Sequence[float],
    asset_up: Sequence[float],
    asset_right: Sequence[float],
    observed_forward: Sequence[float],
    observed_up: Sequence[float],
    observed_right: Sequence[float],
) -> list[list[float]]:
    asset_basis = [list(asset_right), list(asset_forward), list(asset_up)]
    observed_basis = [list(observed_right), list(observed_forward), list(observed_up)]
    rotation = [
        [
            sum(observed_basis[k][row] * asset_basis[k][column] for k in range(3))
            for column in range(3)
        ]
        for row in range(3)
    ]
    return [
        [rotation[0][0], rotation[0][1], rotation[0][2], 0.0],
        [rotation[1][0], rotation[1][1], rotation[1][2], 0.0],
        [rotation[2][0], rotation[2][1], rotation[2][2], 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def validate_replacement_asset_frame_registration(
    value: Mapping[str, Any], *, verify_files: bool = True
) -> dict[str, Any]:
    payload = json.loads(json.dumps(value, allow_nan=False))
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("status") != "asset_frame_registered_pending_multiview_review"
        or payload.get("registration_digest")
        != canonical_digest(payload, digest_field="registration_digest")
        or payload.get("observed_frame_source") != "human_reviewed_multiview_source_evidence"
        or payload.get("identity_assumed_without_review") is not False
        or payload.get("multiview_registration_review") != "accepted"
    ):
        raise ReplacementAssetFrameRegistrationError("asset_frame_registration_invalid")
    asset_forward, asset_up, asset_right = _frame(
        payload.get("asset_local_forward_axis"),
        payload.get("asset_local_up_axis"),
        "asset_frame_local_axes_invalid",
    )
    observed_forward, observed_up, observed_right = _frame(
        payload.get("observed_world_forward_axis"),
        payload.get("observed_world_up_axis"),
        "asset_frame_observed_axes_invalid",
    )
    expected = _registration_matrix(
        asset_forward, asset_up, asset_right, observed_forward, observed_up, observed_right
    )
    if payload.get("T_observed_world_axes_from_asset_local_axes") != expected:
        raise ReplacementAssetFrameRegistrationError("asset_frame_transform_invalid")
    references = payload.get("reference_images")
    if not isinstance(references, list) or len(references) < 2:
        raise ReplacementAssetFrameRegistrationError("asset_frame_references_invalid")
    for record in references:
        if not isinstance(record, Mapping):
            raise ReplacementAssetFrameRegistrationError("asset_frame_references_invalid")
        if verify_files and _record(record.get("path", "")) != dict(record):
            raise ReplacementAssetFrameRegistrationError("asset_frame_references_invalid")
    return payload


def seal_replacement_asset_frame_registration(
    *,
    scene_id: str,
    task_id: str,
    asset_id: str,
    asset_local_forward_axis: Sequence[float],
    asset_local_up_axis: Sequence[float],
    observed_world_forward_axis: Sequence[float],
    observed_world_up_axis: Sequence[float],
    reference_image_paths: Sequence[str | Path],
    reviewed_by: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Seal explicit semantic axes and their deterministic rigid registration."""

    if not all(str(value).strip() for value in (scene_id, task_id, asset_id, reviewed_by)):
        raise ReplacementAssetFrameRegistrationError("asset_frame_identity_invalid")
    asset_forward, asset_up, asset_right = _frame(
        asset_local_forward_axis, asset_local_up_axis, "asset_frame_local_axes_invalid"
    )
    observed_forward, observed_up, observed_right = _frame(
        observed_world_forward_axis, observed_world_up_axis, "asset_frame_observed_axes_invalid"
    )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "asset_frame_registered_pending_multiview_review",
        "scene_id": scene_id,
        "task_id": task_id,
        "asset_id": asset_id,
        "asset_local_forward_axis": asset_forward,
        "asset_local_up_axis": asset_up,
        "observed_world_forward_axis": observed_forward,
        "observed_world_up_axis": observed_up,
        "T_observed_world_axes_from_asset_local_axes": _registration_matrix(
            asset_forward, asset_up, asset_right, observed_forward, observed_up, observed_right
        ),
        "observed_frame_source": "human_reviewed_multiview_source_evidence",
        "reference_images": [_record(path) for path in reference_image_paths],
        "identity_assumed_without_review": False,
        "multiview_registration_review": "accepted",
        "reviewed_by": reviewed_by,
        "generated_geometry": False,
        "physical_equivalence_proven": False,
        "registration_digest": "",
    }
    payload["registration_digest"] = canonical_digest(payload, digest_field="registration_digest")
    validated = validate_replacement_asset_frame_registration(payload)
    destination = Path(output_path).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise ReplacementAssetFrameRegistrationError("asset_frame_destination_exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(validated, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return validated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    result = seal_replacement_asset_frame_registration(**request, output_path=args.output)
    print(json.dumps({"registration_digest": result["registration_digest"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
