"""Field names for ``raw/intake_packet.json``.

Raw Contract V3 (BlueprintCapture ``docs/CAPTURE_RAW_CONTRACT_V3.md``) defines the
intake packet in snake_case — ``workflow_name``, ``task_steps``, ``zone``,
``owner`` — and both the iOS finalizer and Blueprint-WebApp write it that way.
Materialization and preflight were written against camelCase keys
(``workflowName``, ``taskSteps``), so a contract-shaped packet read as empty and
every such capture looked like it had no intake.

``normalize_intake_packet`` returns a copy in which each contract field is also
readable under its camelCase name. When both spellings carry a non-empty value
the camelCase one is kept, which is what these readers saw before. Nothing is
inferred: a field absent in both spellings stays absent.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

_SNAKE_TO_CAMEL = {
    "workflow_name": "workflowName",
    "task_steps": "taskSteps",
    "target_kpi": "targetKPI",
    "facility_template": "facilityTemplate",
    "required_coverage_areas": "requiredCoverageAreas",
    "benchmark_stations": "benchmarkStations",
    "adjacent_systems": "adjacentSystems",
    "privacy_security_limits": "privacySecurityLimits",
    "known_blockers": "knownBlockers",
    "non_routine_modes": "nonRoutineModes",
    "people_traffic_notes": "peopleTrafficNotes",
    "capture_restrictions": "captureRestrictions",
    "lighting_windows": "lightingWindows",
    "shift_traffic_windows": "shiftTrafficWindows",
    "movable_obstacles": "movableObstacles",
    "floor_condition_notes": "floorConditionNotes",
    "reflective_surface_notes": "reflectiveSurfaceNotes",
    "access_rules": "accessRules",
}


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, dict)):
        return len(value) == 0
    return False


def normalize_intake_packet(intake: Mapping[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(intake, Mapping):
        return {}
    normalized: Dict[str, Any] = dict(intake)
    for snake, camel in _SNAKE_TO_CAMEL.items():
        if snake not in intake:
            continue
        if _is_empty(normalized.get(camel)):
            normalized[camel] = intake[snake]
    return normalized
