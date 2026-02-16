"""Utilities for reconciling interactive results after retrieval fallback."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping


def find_required_articulation_failures(
    *,
    interactive_results: Mapping[str, Any],
    required_object_ids: Iterable[str],
) -> List[str]:
    required = {str(obj_id) for obj_id in required_object_ids}
    objects = interactive_results.get("objects") if isinstance(interactive_results.get("objects"), list) else []

    failures: List[str] = []
    for entry in objects:
        if not isinstance(entry, Mapping):
            continue
        obj_id = str(entry.get("id") or "")
        if obj_id not in required:
            continue
        if bool(entry.get("is_articulated")):
            continue
        failures.append(obj_id)

    for missing in sorted(required - {str(item.get("id")) for item in objects if isinstance(item, Mapping)}):
        failures.append(missing)

    return sorted(set(failures))


def reconcile_interactive_results(
    *,
    interactive_results: Dict[str, Any],
    resolved_object_ids: Iterable[str],
) -> Dict[str, Any]:
    resolved = {str(obj_id) for obj_id in resolved_object_ids}
    if not resolved:
        return interactive_results

    objects = interactive_results.get("objects") if isinstance(interactive_results.get("objects"), list) else []
    existing_by_id: Dict[str, Dict[str, Any]] = {}
    for entry in objects:
        if not isinstance(entry, dict):
            continue
        obj_id = str(entry.get("id") or "")
        if obj_id:
            existing_by_id[obj_id] = entry

    for obj_id in sorted(resolved):
        entry = existing_by_id.get(obj_id)
        if entry is None:
            objects.append(
                {
                    "id": obj_id,
                    "name": f"obj_{obj_id}",
                    "status": "ok",
                    "backend": "retrieval_fallback",
                    "required_articulation": True,
                    "is_articulated": True,
                    "joint_count": 1,
                    "fallback_reconciled": True,
                }
            )
            continue

        entry["status"] = "ok"
        entry["backend"] = "retrieval_fallback"
        entry["required_articulation"] = True
        entry["is_articulated"] = True
        entry["joint_count"] = max(1, int(entry.get("joint_count") or 0))
        entry["fallback_reconciled"] = True

    interactive_results["objects"] = objects
    interactive_results["total_objects"] = len(objects)
    interactive_results["ok_count"] = sum(1 for item in objects if item.get("status") == "ok")
    interactive_results["error_count"] = sum(1 for item in objects if item.get("status") == "error")
    interactive_results["fallback_count"] = sum(
        1 for item in objects if item.get("status") in {"fallback", "static"}
    )
    interactive_results["articulated_count"] = sum(
        1 for item in objects if bool(item.get("is_articulated"))
    )
    return interactive_results
