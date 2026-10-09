"""Pure validation of website task context; no signed reads or execution.

ADP-010/day14 compatibility preserves the exact default digest/confirmation and
explicit preparation projection, including false/null/unknown source facts.
"""
from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

from .decision_evidence_contracts import canonical_digest


def validate_website_task_context(
    value: Mapping[str, Any], *, request_id: str, scene_id: str, capture_id: str,
    purpose: str | None = None,
) -> dict[str, Any]:
    if value.get("schema_version") != "website_site_task_context.v1":
        raise ValueError("website_task_context_schema_invalid")
    if any(value.get(key) != expected for key, expected in (
        ("request_id", request_id), ("scene_id", scene_id), ("capture_id", capture_id),
    )):
        raise ValueError("website_task_context_identity_mismatch")
    if value.get("context_digest") != canonical_digest(value, digest_field="context_digest"):
        raise ValueError("website_task_context_digest_mismatch")
    if purpose not in {None, "scene_preparation"}:
        raise ValueError("website_task_context_purpose_invalid")
    if purpose == "scene_preparation" and value.get("purpose") != purpose and not (
            "purpose" not in value and value.get("confirmed") is True and value.get("confirmed_at")):
        raise ValueError("website_task_context_purpose_mismatch")
    if purpose == "scene_preparation" and (type(value.get("confirmed")) is not bool
            or (value["confirmed"] is False and value.get("confirmed_at") is not None)
            or (value["confirmed"] is True and not value.get("confirmed_at"))):
        raise ValueError("website_task_context_confirmation_invalid")
    if purpose is None and (value.get("confirmed") is not True or not value.get("confirmed_at")):
        raise ValueError("website_task_context_not_confirmed")
    if not isinstance(value.get("description"), str) or not value["description"].strip():
        raise ValueError("website_task_context_description_missing")
    details = value.get("operator_task_details")
    if details is not None and (not isinstance(details, Mapping) or any(key not in {"item_weight", "item_make_model"}
            or not isinstance(item, str) for key, item in details.items())):
        raise ValueError("website_task_context_item_details_invalid")
    criteria = value.get("success_criteria")
    if criteria is not None:
        if not isinstance(criteria, Mapping) or set(criteria) != {"successDefinition", "successRate", "cycleTimeSeconds", "unknown"}:
            raise ValueError("website_task_context_success_criteria_invalid")
        if type(criteria["unknown"]) is not bool or (criteria["successDefinition"] is not None and not isinstance(criteria["successDefinition"], str)):
            raise ValueError("website_task_context_success_criteria_invalid")
        for key in ("successRate", "cycleTimeSeconds"):
            number = criteria[key]
            if number is not None and (isinstance(number, bool) or not isinstance(number, (int, float))
                    or not math.isfinite(number) or number < 0 or (key == "successRate" and number > 100)):
                raise ValueError("website_task_context_success_criteria_invalid")
    items = value.get("task_items")
    if items is not None:
        if not isinstance(items, list) or len(items) > 100:
            raise ValueError("website_task_context_items_invalid")
        identifiers = []
        for item in items:
            if not isinstance(item, Mapping) or not isinstance(item.get("item_id"), str) or not isinstance(item.get("label"), str):
                raise ValueError("website_task_context_items_invalid")
            identifiers.append(item["item_id"])
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,120}", item["item_id"]) or not isinstance(item.get("images"), list):
                raise ValueError("website_task_context_items_invalid")
            for image in item["images"]:
                prefix = f"scenes/{scene_id}/items/{item['item_id']}/"
                name = image.get("storage_path", "") if isinstance(image, Mapping) else ""
                if not isinstance(name, str) or not name.startswith(prefix) or any(part in {"", ".", ".."} for part in name.split("/")):
                    raise ValueError("website_item_source_outside_site")
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("website_task_context_items_invalid")
    continuation = value.get("capture_binding")
    if capture_id.startswith("supplement-") and continuation is None:
        raise ValueError("website_task_context_continuation_invalid")
    if continuation is not None:
        if (not isinstance(continuation, Mapping) or continuation.get("schema_version") != "website_capture_continuation.v1"
                or continuation.get("capture_id") != capture_id or continuation.get("original_capture_id") != f"walkthrough-{request_id}"
                or continuation.get("coordinate_frames_independent") is not True
                or not isinstance(continuation.get("lineage"), list) or not 1 <= len(continuation["lineage"]) <= 8):
            raise ValueError("website_task_context_continuation_invalid")
        expected = capture_id
        seen = set()
        for entry in continuation["lineage"]:
            if (not isinstance(entry, Mapping) or any(not isinstance(entry.get(key), Mapping) for key in ("child", "parent", "supplement"))
                    or entry["child"].get("capture_id") != expected or expected in seen
                    or entry.get("supplement", {}).get("parent_capture_id") != entry.get("parent", {}).get("capture_id")
                    or entry.get("supplement", {}).get("parent_bundle_digest") != entry.get("parent", {}).get("raw_bundle_digest")
                    or entry.get("supplement", {}).get("parent_manifest_uri") != entry.get("parent", {}).get("raw_manifest_uri")):
                raise ValueError("website_task_context_continuation_invalid")
            seen.add(expected)
            expected = entry["parent"]["capture_id"]
        if expected != f"walkthrough-{request_id}":
            raise ValueError("website_task_context_continuation_invalid")
    return dict(value)


