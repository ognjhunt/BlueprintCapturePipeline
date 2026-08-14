"""Prepare rights-admitted semantic-teacher image-edit packets.

This module performs no provider call.  It reopens one 1--5-task ArtiFixer
candidate receipt, selects one exact entry from the repository's backend
registry, requires an explicit human rights attestation for that entry, and
stages byte-bound RGB inputs plus an adapter-declared mask encoding whose edit
pixels are exactly the declared repair support. The resulting packet is the
only input a later paid allocator lane may send to the selected editor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .decision_evidence_contracts import canonical_digest, canonical_json
from .image_editor_backend_registry import (
    REGISTRY_SCHEMA_VERSION,
    SEMANTIC_TEACHER_IMAGE_EDIT_CAPABILITY,
    ImageEditorRegistryError,
    load_registry,
)


RIGHTS_SCHEMA_VERSION = "fresh_scene_semantic_teacher_image_edit_rights.v1"
REQUEST_SCHEMA_VERSION = "fresh_scene_semantic_teacher_image_edit_request.v1"
PACKET_SCHEMA_VERSION = "fresh_scene_semantic_teacher_image_edit_packet.v1"
CANDIDATE_SCHEMA_VERSION = "public_scene_artifixer3d_candidate_inputs.v3"
PROMPT_POLICY = "generic_masked_object_absent_background_completion_v2"
PROMPT_POLICIES = {
    PROMPT_POLICY: (
        "Remove the masked task object completely. Reconstruct the realistic empty "
        "background surfaces that continue behind it, matching the existing room, "
        "materials, lighting, perspective, and camera viewpoint. Do not add a "
        "replacement object, silhouette, blank panel, text, watermark, robot, or "
        "new foreground item. Preserve the rest of the image as closely as possible."
    )
}
SUPPORTED_TRANSPORT_KINDS = {"hosted_image_edit", "local_gpu_image_edit"}
SUPPORTED_MASK_ENCODINGS = {
    "rgba_alpha_zero_edit_region_png",
    "binary_white_edit_region_png",
    "binary_black_edit_region_png",
}


class SemanticTeacherImageEditError(ValueError):
    """The immutable semantic-teacher edit packet is invalid."""

    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read(path: Path, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SemanticTeacherImageEditError([code]) from exc
    if path.is_symlink() or not isinstance(value, dict):
        raise SemanticTeacherImageEditError([code])
    return value


def _file(value: Any, *, code: str) -> Path:
    unresolved = Path(str(value or "")).expanduser()
    if unresolved.is_symlink():
        raise SemanticTeacherImageEditError([code])
    path = unresolved.resolve()
    if not path.is_file():
        raise SemanticTeacherImageEditError([code])
    return path


def _record(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    record = {
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if root is None:
        record["path"] = str(path)
    else:
        record["relative_path"] = path.relative_to(root).as_posix()
    return record


def _bound_absolute(value: Any, *, code: str) -> Path:
    if not isinstance(value, Mapping):
        raise SemanticTeacherImageEditError([code])
    path = _file(value.get("path"), code=code)
    if path.stat().st_size != value.get("size_bytes") or _sha256(path) != value.get(
        "sha256"
    ):
        raise SemanticTeacherImageEditError([code])
    return path


def _validated_candidate(path: Path) -> dict[str, Any]:
    value = _read(path, code="semantic_teacher_candidate_receipt_unreadable")
    tasks = value.get("tasks")
    if (
        value.get("schema_version") != CANDIDATE_SCHEMA_VERSION
        or value.get("status") != "candidate_inputs_prepared_no_model_no_execution"
        or value.get("receipt_digest")
        != canonical_digest(value, digest_field="receipt_digest")
        or not isinstance(tasks, list)
        or not 1 <= len(tasks) <= 5
        or value.get("replacement_object_count") != len(tasks)
    ):
        raise SemanticTeacherImageEditError(
            ["semantic_teacher_candidate_receipt_invalid"]
        )
    return value


def _validated_backend(
    path: Path, *, backend_id: str
) -> tuple[dict[str, Any], dict[str, Any], str]:
    payload = _read(path, code="semantic_teacher_backend_registry_unreadable")
    if payload.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        raise SemanticTeacherImageEditError(
            ["semantic_teacher_backend_registry_invalid"]
        )
    try:
        registry = load_registry(path)
    except ImageEditorRegistryError as exc:
        raise SemanticTeacherImageEditError(
            ["semantic_teacher_backend_registry_invalid"]
        ) from exc
    if backend_id not in registry:
        raise SemanticTeacherImageEditError(
            ["semantic_teacher_backend_unregistered"]
        )
    backend = registry[backend_id]
    execution = backend.get("execution")
    if not isinstance(execution, Mapping):
        raise SemanticTeacherImageEditError(
            ["semantic_teacher_backend_execution_contract_invalid"]
        )
    transport_kind = execution.get("transport_kind")
    endpoint = execution.get("endpoint")
    output_formats = execution.get("output_formats")
    default_options = execution.get("default_options")
    if (
        backend.get("commercial_use_permitted") is not True
        or backend.get("capability") != SEMANTIC_TEACHER_IMAGE_EDIT_CAPABILITY
        or transport_kind not in SUPPORTED_TRANSPORT_KINDS
        or not str(execution.get("adapter_id") or "").strip()
        or not str(execution.get("provider_id") or "").strip()
        or not str(execution.get("model_snapshot") or "").strip()
        or execution.get("masked_image_edit_supported") is not True
        or execution.get("high_fidelity_input_supported") is not True
        or not isinstance(output_formats, list)
        or "png" not in output_formats
        or execution.get("mask_encoding") not in SUPPORTED_MASK_ENCODINGS
        or not isinstance(execution.get("external_disclosure_required"), bool)
        or not isinstance(default_options, Mapping)
        or (
            transport_kind == "hosted_image_edit"
            and not str(endpoint or "").startswith("https://")
        )
        or (transport_kind == "local_gpu_image_edit" and endpoint not in (None, ""))
    ):
        raise SemanticTeacherImageEditError(
            ["semantic_teacher_backend_execution_contract_invalid"]
        )
    return backend, dict(execution), canonical_digest(backend)


def _validated_rights(
    path: Path,
    *,
    candidate_digest: str,
    publisher_scene_id: str,
    backend_id: str,
    backend_entry_digest: str,
    execution: Mapping[str, Any],
) -> dict[str, Any]:
    value = _read(path, code="semantic_teacher_rights_attestation_unreadable")
    common_invalid = (
        value.get("schema_version") != RIGHTS_SCHEMA_VERSION
        or value.get("status") != "accepted_for_private_derived_semantic_edit"
        or value.get("attestation_digest")
        != canonical_digest(value, digest_field="attestation_digest")
        or value.get("source_candidate_inputs_receipt_digest") != candidate_digest
        or value.get("publisher_scene_id") != publisher_scene_id
        or value.get("backend_id") != backend_id
        or value.get("backend_entry_digest") != backend_entry_digest
        or value.get("provider_id") != execution.get("provider_id")
        or value.get("model_snapshot") != execution.get("model_snapshot")
        or value.get("raw_nonredistributable_source_bytes_included") is not False
        or value.get("issued_by_agent") is not False
        or not str(value.get("accepted_by") or "").strip()
        or not str(value.get("accepted_on") or "").strip()
    )
    external = execution["external_disclosure_required"]
    disclosure_invalid = (
        external
        and (
            value.get("private_derived_frame_disclosure_authorized") is not True
            or value.get("provider_retention_terms_accepted") is not True
            or value.get("provider_training_terms_accepted") is not True
        )
    ) or (
        not external
        and value.get("local_private_derived_use_authorized") is not True
    )
    if common_invalid or disclosure_invalid:
        raise SemanticTeacherImageEditError(
            ["semantic_teacher_rights_attestation_invalid"]
        )
    return value


def _write_edit_mask(
    *, exact_mask: Path, destination: Path, encoding: str
) -> tuple[int, int]:
    try:
        with Image.open(exact_mask) as image:
            mask = np.asarray(image.convert("L"), dtype=np.uint8)
    except (OSError, ValueError) as exc:
        raise SemanticTeacherImageEditError(
            ["semantic_teacher_exact_mask_unreadable"]
        ) from exc
    if set(mask.tobytes()) - {0, 255} or not np.any(mask):
        raise SemanticTeacherImageEditError(["semantic_teacher_exact_mask_invalid"])
    if encoding == "rgba_alpha_zero_edit_region_png":
        encoded = np.zeros((*mask.shape, 4), dtype=np.uint8)
        encoded[..., :3] = 255
        encoded[..., 3] = np.where(mask > 0, 0, 255).astype(np.uint8)
        mode = "RGBA"
    elif encoding == "binary_white_edit_region_png":
        encoded = mask
        mode = "L"
    elif encoding == "binary_black_edit_region_png":
        encoded = np.where(mask > 0, 0, 255).astype(np.uint8)
        mode = "L"
    else:  # The registry validator should make this unreachable.
        raise SemanticTeacherImageEditError(
            ["semantic_teacher_backend_execution_contract_invalid"]
        )
    Image.fromarray(encoded, mode=mode).save(destination)
    with Image.open(destination) as image:
        if encoding == "rgba_alpha_zero_edit_region_png":
            written_edit = np.asarray(image.convert("RGBA"), dtype=np.uint8)[..., 3] == 0
        elif encoding == "binary_white_edit_region_png":
            written_edit = np.asarray(image.convert("L"), dtype=np.uint8) > 0
        else:
            written_edit = np.asarray(image.convert("L"), dtype=np.uint8) == 0
    if not np.array_equal(written_edit, mask > 0):
        raise SemanticTeacherImageEditError(
            ["semantic_teacher_edit_mask_roundtrip_invalid"]
        )
    return int(np.count_nonzero(mask)), int(mask.size)


def materialize_semantic_teacher_image_edit_packet(
    *, request: Mapping[str, Any], output_root: str | Path
) -> dict[str, Any]:
    """Stage immutable image-edit inputs without network access or spend."""

    value = dict(request)
    selected = value.get("selected_task_ids")
    if (
        value.get("schema_version") != REQUEST_SCHEMA_VERSION
        or value.get("request_digest")
        != canonical_digest(value, digest_field="request_digest")
        or not str(value.get("backend_id") or "").strip()
        or value.get("prompt_policy") not in PROMPT_POLICIES
        or value.get("output_format") != "png"
        or value.get("retry_count") != 0
        or (selected is not None and not isinstance(selected, list))
    ):
        raise SemanticTeacherImageEditError(["semantic_teacher_edit_request_invalid"])
    output = Path(output_root).expanduser().resolve()
    if output.is_symlink() or (output.exists() and any(output.iterdir())):
        raise SemanticTeacherImageEditError(["semantic_teacher_edit_output_not_empty"])
    candidate_path = _file(
        value.get("source_candidate_inputs_receipt_path"),
        code="semantic_teacher_candidate_receipt_missing",
    )
    candidate = _validated_candidate(candidate_path)
    registry_path = _file(
        value.get("backend_registry_path"),
        code="semantic_teacher_backend_registry_missing",
    )
    backend_id = str(value["backend_id"])
    backend, execution, backend_entry_digest = _validated_backend(
        registry_path, backend_id=backend_id
    )
    rights_path = _file(
        value.get("rights_attestation_path"),
        code="semantic_teacher_rights_attestation_missing",
    )
    rights = _validated_rights(
        rights_path,
        candidate_digest=str(candidate["receipt_digest"]),
        publisher_scene_id=str(candidate["publisher_scene_id"]),
        backend_id=backend_id,
        backend_entry_digest=backend_entry_digest,
        execution=execution,
    )
    available = {
        str(task.get("task_id") or ""): task
        for task in candidate["tasks"]
        if isinstance(task, Mapping)
    }
    task_ids = sorted(available) if selected is None else [str(item) for item in selected]
    if (
        not 1 <= len(task_ids) <= 5
        or len(set(task_ids)) != len(task_ids)
        or set(task_ids) - set(available)
    ):
        raise SemanticTeacherImageEditError(["semantic_teacher_task_set_invalid"])

    output.mkdir(parents=True)
    task_rows: list[dict[str, Any]] = []
    request_count = 0
    for task_id in task_ids:
        task = available[task_id]
        frames = task.get("frames")
        if not isinstance(frames, list) or not frames:
            raise SemanticTeacherImageEditError(["semantic_teacher_frame_set_invalid"])
        task_root = output / "tasks" / task_id
        input_root = task_root / "input_frames"
        mask_root = task_root / "edit_masks"
        input_root.mkdir(parents=True)
        mask_root.mkdir()
        frame_rows: list[dict[str, Any]] = []
        for expected_index, frame in enumerate(frames):
            if (
                not isinstance(frame, Mapping)
                or frame.get("frame_index") != expected_index
                or not str(frame.get("camera_id") or "").strip()
            ):
                raise SemanticTeacherImageEditError(
                    ["semantic_teacher_frame_set_invalid"]
                )
            source = _bound_absolute(
                frame.get("input_retained_frame"),
                code="semantic_teacher_source_frame_invalid",
            )
            exact_mask = _bound_absolute(
                frame.get("input_exact_repair_mask"),
                code="semantic_teacher_exact_mask_invalid",
            )
            try:
                with Image.open(source) as source_image:
                    source_rgb = source_image.convert("RGB")
                    source_size = source_rgb.size
                with Image.open(exact_mask) as mask_image:
                    mask_size = mask_image.size
            except (OSError, ValueError) as exc:
                raise SemanticTeacherImageEditError(
                    ["semantic_teacher_source_frame_invalid"]
                ) from exc
            if source_size != mask_size:
                raise SemanticTeacherImageEditError(
                    ["semantic_teacher_frame_shape_mismatch"]
                )
            filename = f"{expected_index:05d}.png"
            staged_source = input_root / filename
            staged_mask = mask_root / filename
            source_rgb.save(staged_source)
            repair_pixels, image_pixels = _write_edit_mask(
                exact_mask=exact_mask,
                destination=staged_mask,
                encoding=str(execution["mask_encoding"]),
            )
            frame_rows.append(
                {
                    "frame_index": expected_index,
                    "camera_id": str(frame["camera_id"]),
                    "source_original": _record(source),
                    "exact_repair_mask": _record(exact_mask),
                    "staged_input_rgb": _record(staged_source, root=output),
                    "staged_edit_mask": _record(staged_mask, root=output),
                    "edit_mask_encoding": execution["mask_encoding"],
                    "width": source_size[0],
                    "height": source_size[1],
                    "repair_pixel_count": repair_pixels,
                    "image_pixel_count": image_pixels,
                    "edit_mask_edit_pixels_equal_exact_repair_support": True,
                }
            )
            request_count += 1
        task_rows.append(
            {
                "task_id": task_id,
                "camera_count": len(frame_rows),
                "frames": frame_rows,
            }
        )
    packet: dict[str, Any] = {
        "schema_version": PACKET_SCHEMA_VERSION,
        "status": "semantic_teacher_image_edit_packet_prepared_no_upload_no_execution",
        "publisher_scene_id": candidate["publisher_scene_id"],
        "source_candidate_inputs_receipt": {
            **_record(candidate_path),
            "receipt_digest": candidate["receipt_digest"],
        },
        "rights_attestation": {
            **_record(rights_path),
            "attestation_digest": rights["attestation_digest"],
        },
        "backend_registry": {
            **_record(registry_path),
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "selected_backend_entry_digest": backend_entry_digest,
        },
        "backend": {
            "registry_entry": backend,
            "backend_entry_digest": backend_entry_digest,
            "execution": execution,
            "prompt_policy": value["prompt_policy"],
            "prompt": PROMPT_POLICIES[str(value["prompt_policy"])],
            "output_format": "png",
        },
        "task_count": len(task_rows),
        "request_count": request_count,
        "retry_count": 0,
        "tasks": task_rows,
        "provider_upload_performed": False,
        "provider_inference_performed": False,
        "provider_mutations_performed": 0,
        "canonical_source_altered": False,
        "raw_nonredistributable_source_bytes_included": False,
        "private_derived_calibrated_frames_included": True,
        "appearance_or_physical_evidence_qualified": False,
        "packet_digest": "",
    }
    packet["packet_digest"] = canonical_digest(packet, digest_field="packet_digest")
    (output / f"{PACKET_SCHEMA_VERSION}.json").write_text(
        canonical_json(packet) + "\n", encoding="utf-8"
    )
    return packet


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    request_path = _file(
        args.request, code="semantic_teacher_edit_request_unreadable"
    )
    request = _read(request_path, code="semantic_teacher_edit_request_unreadable")
    result = materialize_semantic_teacher_image_edit_packet(
        request=request, output_root=args.output_root
    )
    print(canonical_json(result))
    return 0


__all__ = [
    "PACKET_SCHEMA_VERSION",
    "PROMPT_POLICY",
    "PROMPT_POLICIES",
    "REQUEST_SCHEMA_VERSION",
    "RIGHTS_SCHEMA_VERSION",
    "SemanticTeacherImageEditError",
    "main",
    "materialize_semantic_teacher_image_edit_packet",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
