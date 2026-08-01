"""Compile independent held-out requests from frozen evaluator-owned evidence."""

from __future__ import annotations

import json
import re
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest, canonical_json
from .processed_observation_dataset import (
    PROCESSED_DATASET_SCHEMA_VERSION,
    PROCESSED_HELDOUT_SCHEMA_VERSION,
    PROCESSED_SPLIT_SCHEMA_VERSION,
)
from .reconstruction_heldout_evaluation import (
    HELDOUT_APPEARANCE_REQUEST_SCHEMA_VERSION,
    build_heldout_appearance_evaluation_request,
    evaluate_heldout_appearance,
)
from .reconstruction_worker_contracts import (
    ReconstructionWorkerContractError,
    build_training_request,
    build_training_result,
)


_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
RENDER_MANIFEST_SCHEMA_VERSION = "candidate_heldout_render_manifest.v1"
_DATASET_SCHEMA_VERSIONS = {
    "reconstruction_dataset_manifest.v1",
    PROCESSED_DATASET_SCHEMA_VERSION,
}
_SPLIT_SCHEMA_VERSIONS = {
    "frozen_reconstruction_split_manifest.v1",
    PROCESSED_SPLIT_SCHEMA_VERSION,
}
_HELDOUT_SCHEMA_VERSIONS = {
    "hidden_heldout_evaluator_manifest.v1",
    PROCESSED_HELDOUT_SCHEMA_VERSION,
}


class HeldoutRequestCompilationError(ValueError):
    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__("; ".join(self.codes))


def _digest(value: Any) -> bool:
    return _DIGEST.fullmatch(str(value or "")) is not None


def _clone(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(canonical_json(dict(value)))


def _safe_relative(value: Any) -> bool:
    path = PurePosixPath(str(value or "").replace("\\", "/"))
    return bool(str(value or "").strip()) and not path.is_absolute() and all(
        part not in {"", ".", ".."} for part in path.parts
    )


def build_candidate_heldout_render_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate renderer output without granting it access to held-out pixels."""

    manifest = _clone(value)
    errors: list[str] = []
    if manifest.get("schema_version") != RENDER_MANIFEST_SCHEMA_VERSION:
        errors.append("heldout_render_manifest_schema_invalid")
    for key in (
        "source_capture_digest",
        "frozen_split_digest",
        "reconstruction_training_result_digest",
        "appearance_asset_digest",
        "renderer_implementation_digest",
    ):
        if not _digest(manifest.get(key)):
            errors.append(f"heldout_render_manifest_{key}_invalid")
    for key in ("candidate_method_id", "candidate_provider_identity", "renderer_identity"):
        if not str(manifest.get(key) or "").strip():
            errors.append(f"heldout_render_manifest_{key}_missing")
    for key in (
        "hidden_pixels_read_by_candidate",
        "heldout_labels_read_by_candidate",
        "candidate_selected_heldout",
        "candidate_self_grading",
    ):
        if manifest.get(key) is not False:
            errors.append(f"heldout_render_manifest_forbidden:{key}")
    if manifest.get("frozen_camera_parameters_only") is not True:
        errors.append("heldout_render_manifest_camera_binding_invalid")
    rows = manifest.get("renders")
    if not isinstance(rows, list) or not rows:
        errors.append("heldout_render_manifest_rows_missing")
    else:
        seen: set[str] = set()
        for index, raw in enumerate(rows):
            if not isinstance(raw, Mapping):
                errors.append(f"heldout_render_manifest_row_invalid:{index}")
                continue
            view_id = str(raw.get("view_id") or "").strip()
            if not view_id or view_id in seen:
                errors.append(f"heldout_render_manifest_view_id_invalid:{index}")
            seen.add(view_id)
            if not _digest(raw.get("real_view_digest")) or not _digest(
                raw.get("candidate_render_digest")
            ):
                errors.append(f"heldout_render_manifest_digest_invalid:{view_id or index}")
            if not _safe_relative(raw.get("candidate_render_relative_path")):
                errors.append(f"heldout_render_manifest_path_invalid:{view_id or index}")
    supplied = manifest.pop("candidate_heldout_render_manifest_digest", None)
    manifest["candidate_heldout_render_manifest_digest"] = canonical_digest(
        manifest, digest_field="candidate_heldout_render_manifest_digest"
    )
    if supplied is not None and supplied != manifest["candidate_heldout_render_manifest_digest"]:
        errors.append("heldout_render_manifest_digest_mismatch")
    if errors:
        raise HeldoutRequestCompilationError(errors)
    return manifest


def compile_heldout_appearance_evaluation_request(
    *,
    training_request: Mapping[str, Any],
    training_result: Mapping[str, Any],
    reconstruction_dataset_manifest: Mapping[str, Any],
    frozen_split_manifest: Mapping[str, Any],
    hidden_heldout_manifest: Mapping[str, Any],
    candidate_render_manifest: Mapping[str, Any],
    evaluator_contract: Mapping[str, Any],
    candidate_root: str,
    evaluator_root: str,
    authority_used: Mapping[str, Any],
    timestamp: str,
) -> dict[str, Any]:
    """Bind successful candidate output to evaluator-owned real observations."""

    errors: list[str] = []
    try:
        request = build_training_request(training_request)
        result = build_training_result(training_result)
    except ReconstructionWorkerContractError:
        request, result = {}, {}
        errors.append("heldout_training_contract_invalid")
    try:
        renders = build_candidate_heldout_render_manifest(candidate_render_manifest)
    except HeldoutRequestCompilationError as exc:
        renders = {}
        errors.extend(exc.codes)
    dataset = _clone(reconstruction_dataset_manifest)
    split = _clone(frozen_split_manifest)
    hidden = _clone(hidden_heldout_manifest)
    evaluator = _clone(evaluator_contract)

    if (
        dataset.get("schema_version") not in _DATASET_SCHEMA_VERSIONS
        or dataset.get("dataset_manifest_digest")
        != canonical_digest(dataset, digest_field="dataset_manifest_digest")
    ):
        errors.append("heldout_dataset_manifest_invalid")
    split_digest = split.get("split_digest")
    if (
        split.get("schema_version") not in _SPLIT_SCHEMA_VERSIONS
        or split.get("frozen") is not True
        or split.get("candidate_can_change_assignments") is not False
        or split.get("hidden_heldout_access") != "independent_evaluator_only"
        or split_digest != canonical_digest(split, digest_field="split_digest")
    ):
        errors.append("heldout_frozen_split_invalid")
    hidden_digest = hidden.get("hidden_heldout_digest")
    if (
        hidden.get("schema_version") not in _HELDOUT_SCHEMA_VERSIONS
        or hidden.get("access_scope") != "independent_evaluator_only"
        or hidden.get("candidate_method_access_allowed") is not False
        or hidden_digest != canonical_digest(hidden, digest_field="hidden_heldout_digest")
    ):
        errors.append("heldout_hidden_manifest_invalid")

    capture_digest = dataset.get("source_capture_digest")
    if any(
        value != capture_digest
        for value in (
            split.get("capture_digest"),
            hidden.get("capture_digest"),
            request.get("source_capture_digest"),
            result.get("source_capture_digest"),
            renders.get("source_capture_digest"),
        )
    ):
        errors.append("heldout_source_capture_binding_mismatch")
    if any(
        value != split_digest
        for value in (
            dataset.get("train_heldout_split_digest"),
            hidden.get("split_digest"),
            request.get("train_heldout_split_digest"),
            result.get("train_heldout_split_digest"),
            renders.get("frozen_split_digest"),
        )
    ):
        errors.append("heldout_split_binding_mismatch")
    if (
        result.get("status") != "succeeded"
        or result.get("reconstruction_training_request_digest")
        != request.get("reconstruction_training_request_digest")
        or renders.get("reconstruction_training_result_digest")
        != result.get("reconstruction_training_result_digest")
    ):
        errors.append("heldout_candidate_training_not_accepted")
    appearance_digests = {
        row.get("digest")
        for row in result.get("output_digests", [])
        if isinstance(row, Mapping) and "appearance" in str(row.get("artifact_id") or "")
    }
    if renders.get("appearance_asset_digest") not in appearance_digests:
        errors.append("heldout_appearance_asset_binding_mismatch")
    hidden_rows = hidden.get("frames")
    split_rows = split.get("assignments")
    if not isinstance(hidden_rows, list) or not hidden_rows:
        errors.append("heldout_hidden_frames_missing")
        hidden_rows = []
    if not isinstance(split_rows, list):
        errors.append("heldout_split_assignments_missing")
        split_rows = []
    expected_hidden = {
        str(row.get("frame_id")): row.get("frame_digest")
        for row in split_rows
        if isinstance(row, Mapping) and row.get("split") == "held_out"
    }
    manifest_hidden = {
        str(row.get("frame_id")): row.get("frame_digest")
        for row in hidden_rows
        if isinstance(row, Mapping)
    }
    if not expected_hidden or manifest_hidden != expected_hidden:
        errors.append("heldout_hidden_frames_do_not_match_frozen_split")
    if set(result.get("registered_observation_ids") or []) & set(expected_hidden):
        errors.append("heldout_training_observation_leakage")
    render_rows = {
        str(row.get("view_id")): row
        for row in renders.get("renders", [])
        if isinstance(row, Mapping)
    }
    if set(render_rows) != set(expected_hidden) or any(
        render_rows[view_id].get("real_view_digest") != digest
        for view_id, digest in expected_hidden.items()
    ):
        errors.append("heldout_render_set_binding_mismatch")

    evaluator_digest = evaluator.get("evaluation_contract_digest")
    if (
        not _digest(evaluator_digest)
        or evaluator_digest
        != canonical_digest(evaluator, digest_field="evaluation_contract_digest")
        or evaluator.get("source_capture_digest") != capture_digest
        or evaluator.get("frozen_split_digest") != split_digest
        or evaluator.get("thresholds_frozen_before_evaluation") is not True
        or evaluator.get("candidate_hidden_pixel_access_permitted") is not False
        or evaluator.get("candidate_self_grading_permitted") is not False
        or evaluator.get("candidate_provider_identity")
        == evaluator.get("evaluator_provider_identity")
    ):
        errors.append("heldout_evaluator_contract_invalid")
    if authority_used.get("local_evaluation_allowed") is not True:
        errors.append("heldout_local_evaluation_authority_missing")
    if errors:
        raise HeldoutRequestCompilationError(errors)

    hidden_by_id = {str(row["frame_id"]): row for row in hidden_rows}
    compiled = {
        "schema_version": HELDOUT_APPEARANCE_REQUEST_SCHEMA_VERSION,
        "stable_run_identity": evaluator["stable_run_identity"],
        "source_capture_identity": dataset["source_capture_identity"],
        "source_capture_digest": capture_digest,
        "reconstruction_dataset_digest": request["reconstruction_dataset_digest"],
        "frozen_split_digest": split_digest,
        "candidate_reconstruction_result_digest": result[
            "reconstruction_training_result_digest"
        ],
        "candidate_method_id": renders["candidate_method_id"],
        "candidate_provider_identity": renders["candidate_provider_identity"],
        "evaluator_identity": evaluator["evaluator_identity"],
        "evaluator_provider_identity": evaluator["evaluator_provider_identity"],
        "evaluator_implementation_digest": evaluator["evaluator_implementation_digest"],
        "source_commit_sha": evaluator["source_commit_sha"],
        "candidate_root": candidate_root,
        "evaluator_root": evaluator_root,
        "coordinate_frame_declaration": request["coordinate_frame_declaration"],
        "split_frozen_before_training": True,
        "candidate_had_hidden_access": False,
        "candidate_selected_heldout": False,
        "candidate_self_grading": False,
        "thresholds_frozen_before_evaluation": True,
        "thresholds": evaluator["thresholds"],
        "pairs": [
            {
                "view_id": view_id,
                "split": "held_out",
                "excluded_from_training": True,
                "projection_form": render_rows[view_id].get(
                    "projection_form", "perspective_rgb"
                ),
                "real_view_relative_path": hidden_by_id[view_id][
                    "evaluator_relative_path"
                ],
                "real_view_digest": expected_hidden[view_id],
                "candidate_render_relative_path": render_rows[view_id][
                    "candidate_render_relative_path"
                ],
                "candidate_render_digest": render_rows[view_id][
                    "candidate_render_digest"
                ],
            }
            for view_id in sorted(expected_hidden)
        ],
        "authority_used": dict(authority_used),
        "timestamp": timestamp,
    }
    return build_heldout_appearance_evaluation_request(compiled)


def compile_heldout_appearance_supervisor_bindings(
    **compiler_arguments: Any,
) -> dict[str, Any]:
    """Prepare trusted context fields for the digest-only evaluator tool."""

    request = compile_heldout_appearance_evaluation_request(**compiler_arguments)
    return {
        "heldout_appearance_evaluation_request": request,
        "heldout_appearance_evaluator": evaluate_heldout_appearance,
    }


__all__ = [
    "HeldoutRequestCompilationError",
    "RENDER_MANIFEST_SCHEMA_VERSION",
    "build_candidate_heldout_render_manifest",
    "compile_heldout_appearance_evaluation_request",
    "compile_heldout_appearance_supervisor_bindings",
]
