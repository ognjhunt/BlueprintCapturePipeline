"""ADP-010/day-14: retain owner evidence, with no invented metric/placement truth."""
from __future__ import annotations

import hashlib
import os
import re
import tempfile
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .capture_bridge import CaptureDescriptor
from .common import write_json
from .consent_takedown import read_consent_state
from .decision_evidence_contracts import canonical_digest
from .website_task_context import validate_website_task_context, website_webapp_request


def prepare_website_task_descriptor(*, descriptor: CaptureDescriptor, capture_root: Path,
                                    pipeline_dir: Path, bucket: str,
                                    load_task_context: Callable[..., dict[str, Any]],
                                    load_sponsorship: Callable[..., dict[str, Any]]) -> CaptureDescriptor:
    """Bind current owner evidence before preparation or sponsored processing."""
    consent = read_consent_state(capture_root)
    if consent["state"] == "revoked" or "website_withdrawal/tombstone.json" in (consent.get("source_path") or ""):
        raise ValueError("website_capture_withdrawn")
    context = load_task_context(
        request_id=str(descriptor.site_submission_id or descriptor.metadata.get("site_submission_id") or ""),
        scene_id=descriptor.scene_id, capture_id=descriptor.capture_id,
    )
    write_json(pipeline_dir / "website_task_context.json", context)
    evidence = ingest_website_task_evidence(task_context=context, bucket=bucket,
        output_root=pipeline_dir / "website_task_evidence" / context["context_digest"][7:])
    publish_website_item_evidence(task_context=context, evidence=evidence)
    sponsorship = load_sponsorship(task_context=context, now=time.time())
    write_json(pipeline_dir / "website_scene_sponsorship.json", sponsorship)
    return CaptureDescriptor.from_dict({
        **descriptor.to_dict(),
        "metadata": {**descriptor.metadata, "site_task_context": context,
                     "site_task_evidence": evidence, "website_scene_execution_authority": sponsorship,
                     "capture_rights": dict(context.get("capture_rights") or {}),
                     "task_statement": context["description"]},
    })


def owner_success_criteria(context: Mapping[str, Any]) -> dict[str, Any]:
    targets = context.get("success_criteria")
    return {"status": "not_supplied" if targets is None else "unknown" if targets.get("unknown") is True else "unsupported",
            "targets": targets, "basis": "owner_stated_target", "success_rate_unit": "percent",
            "cycle_time_unit": "seconds", "scorer_translation_verified": False}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _download(bucket: str, image: Mapping[str, Any], target: Path) -> None:
    from google.cloud import storage
    source = image["source"]
    blob = storage.Client().bucket(bucket).blob(image["storage_path"], generation=int(source["generation"]))
    blob.download_to_filename(str(target), if_generation_match=int(source["generation"]), checksum="auto", timeout=30)


def ingest_website_task_evidence(*, task_context: Mapping[str, Any], output_root: Path,
                                 download: Callable[[Mapping[str, Any], Path], None] | None = None,
                                 bucket: str | None = None) -> dict[str, Any]:
    """Only immutable source photos become retained observations. Never buys a model call."""
    context = validate_website_task_context(task_context, request_id=task_context["request_id"],
        scene_id=task_context["scene_id"], capture_id=task_context["capture_id"])
    if context.get("capture_rights", {}).get("derived_scene_generation_allowed") is not True:
        raise ValueError("website_item_consent_not_granted")
    prefix = f"scenes/{context['scene_id']}/items/"
    rows = []
    for item in context.get("task_items") or []:
        images = []
        blocker = "website_item_geometry_and_placement_unverified"
        for image in item.get("images") or []:
            name = image.get("storage_path", "")
            if not isinstance(name, str) or not name.startswith(prefix) or any(p in {"", ".", ".."} for p in name.split("/")):
                raise ValueError("website_item_source_outside_site")
            source = image.get("source")
            if not source:
                blocker = "website_item_original_generation_unknown"
                continue
            if (not re.fullmatch(r"[1-9][0-9]{0,19}", str(source.get("generation", "")))
                    or type(source.get("size_bytes")) is not int or source["size_bytes"] <= 0
                    or not re.fullmatch(r"sha256:[a-f0-9]{64}", str(source.get("sha256", "")))):
                raise ValueError("website_item_source_identity_invalid")
            key = canonical_digest({"path": name, "source": source})[7:]
            suffix = Path(name).suffix.lower()
            if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".heic"}:
                raise ValueError("website_item_image_format_invalid")
            target = output_root / "images" / f"{key}{suffix}"
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_symlink():
                raise ValueError("website_item_source_symlink")
            if not target.exists():
                descriptor, temporary_name = tempfile.mkstemp(prefix=".image-", dir=target.parent)
                os.close(descriptor)
                temporary = Path(temporary_name)
                try:
                    if download is None:
                        if not bucket:
                            raise ValueError("website_item_storage_bucket_missing")
                        _download(bucket, image, temporary)
                    else:
                        download(image, temporary)
                    if temporary.stat().st_size != source["size_bytes"] or _sha(temporary) != source["sha256"]:
                        raise ValueError("website_item_source_changed")
                    temporary.replace(target)
                finally:
                    temporary.unlink(missing_ok=True)
            if target.stat().st_size != source["size_bytes"] or _sha(target) != source["sha256"]:
                raise ValueError("website_item_source_changed")
            images.append({"image_id": image["image_id"], "path": str(target), "sha256": source["sha256"],
                           "source_generation": source["generation"], "storage_path": name,
                           "basis": "owner_supplied_photo", "physical_metrology": False})
        rows.append({"item_id": item["item_id"], "label": item["label"], "basis": item.get("basis"),
                     "images": images, "status": "unsupported", "blocker": blocker,
                     "placement_known": False, "metric_geometry_known": False})
    value = {"schema_version": "website_task_evidence.v1", "request_id": context["request_id"],
             "scene_id": context["scene_id"], "capture_id": context["capture_id"],
             "task_context_digest": context["context_digest"], "items": rows,
             "operator_task_details": context.get("operator_task_details"),
             "success_criteria": owner_success_criteria(context), "physical_measurements_verified": False,
             "claim_ceiling": "development_only", "provider_mutation_performed": False}
    value["digest"] = canonical_digest(value, digest_field="digest")
    write_json(output_root / "task_evidence.json", value)
    return value


def publish_website_item_evidence(*, task_context: Mapping[str, Any], evidence: Mapping[str, Any]) -> None:
    if evidence.get("digest") != canonical_digest(evidence, digest_field="digest"):
        raise ValueError("website_item_evidence_changed")
    if not evidence.get("items"):
        return
    # Paths stay on the executor; the site receives only the bound status/reason.
    value = website_webapp_request(capture_id=task_context["capture_id"], operation="task-item-evidence", payload={
        "request_id": task_context["request_id"], "scene_id": task_context["scene_id"],
        "task_context_digest": task_context["context_digest"], "evidence_digest": evidence["digest"],
        "items": [{"item_id": row["item_id"], "status": row["status"], "blocker": row["blocker"]}
                  for row in evidence["items"]]})
    if value.get("evidence_digest") != evidence["digest"] or value.get("ok") is not True:
        raise ValueError("website_item_evidence_receipt_invalid")
