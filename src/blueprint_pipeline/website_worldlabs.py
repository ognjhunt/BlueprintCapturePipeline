"""Submit only prepared website images; retain a single Marble operation."""

from __future__ import annotations

import fcntl
import json
import math
import os
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Mapping

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .local_reconstruction_adapters import _sha256_file
from .paid_resource_admission import PaidResourceAdmissionGrant, require_paid_resource_admission_grant

# Multi-image Marble 1.1 Plus: 100 + 1500 + at most 1500 credits.
# https://docs.worldlabs.ai/api/pricing (verified 2026-09-19); no HQ mesh export.
MAX_GENERATION_COST_USD = 3100 / 1250


def validate_website_prepared_views(*, descriptor: Mapping[str, Any], capture_root: Path) -> tuple[list[Path], dict[str, Any]]:
    """Shared no-network preflight for the allocator and provider adapter."""
    metadata = descriptor.get("metadata") or {}
    clean_plate = metadata.get("clean_plate") or {}
    preparation = clean_plate.get("prepared_views") or {}
    frames = preparation.get("frames") or []
    task_context = metadata.get("site_task_context") or {}
    task_digest = sha256(json.dumps(task_context, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if (clean_plate.get("status") not in {"noop", "objects_removed"}
            or clean_plate.get("privacy_verified") is not True
            or task_context.get("confirmed") is not True
            or preparation.get("status") != "ready" or not 2 <= len(frames) <= 8
            or preparation.get("task_context_sha256") != task_digest):
        raise ValueError("website_prepared_images_required")
    if preparation.get("digest") != canonical_digest(preparation, digest_field="digest"):
        raise ValueError("website_prepared_images_digest_mismatch")
    image_paths = []
    for frame in frames:
        path = Path(frame["image_path"]).resolve()
        if not path.is_relative_to(capture_root.resolve() / "pipeline") or not path.is_file() or _sha256_file(path) != frame.get("image_digest"):
            raise ValueError("website_prepared_image_changed_or_outside_pipeline")
        image_paths.append(path)
    binding = {"prepared_views_digest": preparation["digest"], "model": "marble-1.1-plus",
               "scene_id": descriptor["scene_id"], "capture_id": descriptor["capture_id"]}
    return image_paths, binding


def validate_website_reconstruction_admission(admission: Mapping[str, Any], request_digest: str) -> None:
    budget = admission.get("maximum_cost_usd")
    if (isinstance(budget, bool) or not isinstance(budget, (float, int)) or not math.isfinite(budget)
            or budget < MAX_GENERATION_COST_USD):
        raise ValueError("website_reconstruction_budget_insufficient")
    if admission.get("external_disclosure_allowed") is not True:
        raise ValueError("website_reconstruction_disclosure_not_authorized")
    if admission.get("allocation_binding_digest") != request_digest:
        raise ValueError("website_reconstruction_spend_binding_missing")


def submit_website_prepared_views(*, descriptor: Mapping[str, Any], capture_root: Path,
                                 api_request: Callable[..., Any], upload: Callable[..., Any],
                                 admission_grant: PaidResourceAdmissionGrant | None = None) -> dict[str, Any]:
    image_paths, binding = validate_website_prepared_views(descriptor=descriptor, capture_root=capture_root)
    metadata = descriptor["metadata"]
    frames = metadata["clean_plate"]["prepared_views"]["frames"]
    request_digest = canonical_digest(binding)
    root = capture_root / "pipeline" / "website_reconstruction"
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "submission.json"
    with (root / "submission.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("website_reconstruction_submission_in_progress") from exc
        if state_path.is_file():
            state = json.loads(state_path.read_text())
            if state.get("request_digest") != request_digest:
                raise ValueError("website_reconstruction_already_bound_to_other_inputs")
            if not state.get("operation_id"):
                # A timeout after POST may have purchased a world. Do not buy
                # another while the first request's outcome is unknown.
                raise ValueError("website_reconstruction_submission_requires_reconciliation")
            operation = state["operation"]
            generation = state["generation_request"]
        else:
            admission = metadata.get("website_reconstruction_admission") or {}
            validate_website_reconstruction_admission(admission, request_digest)
            require_paid_resource_admission_grant(admission_grant, resource_class="provider_reconstruction_api",
                                                  allocation_binding_digest=request_digest, require_allocation_binding=True)
            content = []
            for path in image_paths:
                prepared = api_request("/marble/v1/media-assets:prepare_upload", method="POST",
                                       body={"file_name": path.name, "extension": path.suffix.lstrip("."), "kind": "image"})
                asset = prepared.get("media_asset") or {}
                asset_id = asset.get("media_asset_id") or asset.get("id")
                info = prepared.get("upload_info") or {}
                if not asset_id or not info.get("upload_url"):
                    raise ValueError("website_reconstruction_image_upload_invalid")
                upload(info["upload_url"], method=info.get("upload_method") or "PUT", content_type="image/png",
                       data=path.read_bytes(), required_headers=info.get("required_headers") or {})
                # Camera headings are estimates without a measured gravity
                # frame. Marble supports automatic layout when azimuth is absent.
                content.append({"content": {"source": "media_asset", "media_asset_id": asset_id}})
            generation = {"model": "marble-1.1-plus", "permission": {"public": False},
                          "display_name": str(descriptor["capture_id"])[:64],
                          "tags": [f"bp-{request_digest[7:31]}"],
                          "world_prompt": {"type": "multi-image", "multi_image_prompt": content,
                                           "reconstruct_images": True,
                                           "text_prompt": "Reconstruct the same real work area shown in these prepared views. Preserve its layout, supports and remaining objects. Do not add objects into cleared regions."}}
            state = {"request_digest": request_digest, "binding": binding, "status": "submitting",
                     "generation_request": generation}
            # Persist intent before the billable mutation.
            with state_path.open("x") as stream:
                json.dump(state, stream, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            directory_fd = os.open(root, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            operation = api_request("/marble/v1/worlds:generate", method="POST", body=generation)
            operation_id = operation.get("operation_id") or operation.get("id")
            if not operation_id:
                raise ValueError("website_reconstruction_operation_id_missing")
            state.update(status="submitted", operation_id=operation_id, operation=operation)
            temporary = root / "submission.tmp"
            write_json(temporary, state)
            os.replace(temporary, state_path)
        return {"provider_name": "world_labs", "provider_model": "marble-1.1-plus",
                "provider_run_id": state["operation_id"], "worldlabs_operation_id": state["operation_id"],
                "status": "processing", "artifact_uris": {}, "worldlabs_operation": operation,
                "generation_source_type": "prepared_multi_image", "privacy_safe_input": True,
                "worldlabs_request_manifest": {"request_digest": request_digest, "binding": binding,
                                                "generation_request": generation, "prepared_frames": frames},
                "cost_usd": 0.0, "failure_reason": None}
