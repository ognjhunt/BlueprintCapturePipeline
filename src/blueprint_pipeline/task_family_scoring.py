"""Independent native-state scoring for the beta's additional task families.

This module scores a sealed task definition against retained simulator readback.
It never runs a policy, interprets a policy's reward, admits a runtime, or claims
physical validation. Runtime admission and retained observation media remain the
responsibility of the existing execution harness.
"""
from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from typing import Any

try:  # Also usable inside the existing flat provider bundle.
    from decision_evidence_contracts import canonical_digest
except ModuleNotFoundError:
    from .decision_evidence_contracts import canonical_digest

CONTRACT_SCHEMA = "task_family_success_contract.v1"
TELEMETRY_SCHEMA = "task_family_telemetry.v1"
REPORT_SCHEMA = "task_family_scoring.v1"
FAMILY_ALIASES = {
    "pouring": ("pouring", "pour", "liquid_transfer"),
    "wiping": ("wiping", "wipe", "surface_wiping", "surface_cleaning"),
    "insertion": ("insertion", "insert", "peg_in_hole", "rigid_insertion"),
    "stacking": ("stacking", "stack", "object_stacking", "rigid_stacking"),
    "button_press": ("button_press", "button_pressing", "press_button", "buttons"),
    "cloth_folding": ("cloth_folding", "cloth", "folding", "laundry_folding", "fabric_folding"),
    "multi_step": ("multi_step", "multi_step_manipulation", "multistep", "multi-step"),
}
# Criteria are typed quantities, rather than arbitrary evaluation expressions.
FAMILY_CRITERIA = {
    "pouring": ("minimum_transferred_volume_ml", "maximum_spilled_volume_ml", "maximum_volume_balance_error_ml",
                "maximum_final_in_flight_volume_ml"),
    "wiping": ("minimum_cleaned_fraction", "maximum_remaining_dirty_area_m2"),
    "insertion": ("minimum_insertion_depth_m", "maximum_lateral_error_m", "maximum_alignment_error_rad", "maximum_contact_force_n"),
    "stacking": ("minimum_support_margin_m", "maximum_linear_speed_m_s", "maximum_angular_speed_rad_s"),
    "button_press": ("minimum_button_travel_m", "maximum_contact_force_n"),
    "cloth_folding": ("maximum_fold_keypoint_error_m", "minimum_fold_overlap_fraction", "maximum_linear_speed_m_s", "require_release"),
}
FAMILY_OBSERVABLES = {
    "pouring": ("source_volume_ml", "receiver_volume_ml", "in_flight_volume_ml", "spilled_volume_ml"),
    "wiping": ("dirty_area_m2",),
    "insertion": ("insertion_depth_m", "lateral_error_m", "alignment_error_rad", "contact_force_n"),
    "stacking": ("support_contact", "com_support_margin_m", "linear_speed_m_s", "angular_speed_rad_s", "robot_contact_active"),
    "button_press": ("button_travel_m", "button_activated", "contact_force_n"),
    "cloth_folding": ("fold_keypoint_error_m", "fold_overlap_fraction", "linear_speed_m_s", "robot_contact_active"),
}
_BOOL_METRICS = {"support_contact", "button_activated", "robot_contact_active"}
_SHA = re.compile(r"sha256:[0-9a-f]{64}\Z")


class TaskFamilyScoringError(ValueError):
    """A malformed task or unbound trace cannot be scored."""


def canonical_task_family(value: object) -> str | None:
    family = value.strip().lower() if isinstance(value, str) else ""
    return next((key for key, aliases in FAMILY_ALIASES.items() if family in aliases), None)


def _number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _clone(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise TaskFamilyScoringError("task_family_artifact_not_finite_json") from exc


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise TaskFamilyScoringError(code)


def _metric_valid(name: str, value: object) -> bool:
    if name in _BOOL_METRICS:
        return type(value) is bool
    if not _number(value):
        return False
    if name == "com_support_margin_m":
        return True  # A negative margin is a valid, failing native observation.
    return value >= 0 and (name != "fold_overlap_fraction" or value <= 1)


def validate_task_family_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a sealed task, including task-specific reset and thresholds."""
    _require(isinstance(value, Mapping), "task_family_contract_invalid")
    contract = _clone(value)
    _require(contract.get("schema_version") == CONTRACT_SCHEMA, "task_family_contract_schema_invalid")
    _require(contract.get("contract_digest") == canonical_digest(contract, digest_field="contract_digest"),
             "task_family_contract_digest_invalid")
    family = canonical_task_family(contract.get("task_family"))
    _require(family is not None and family == contract.get("task_family"), "task_family_contract_family_invalid")
    scope = contract.get("scope")
    _require(isinstance(scope, dict) and all(isinstance(scope.get(key), str) and scope[key].strip()
             for key in ("scene_id", "task_id")) and bool(_SHA.fullmatch(str(scope.get("task_context_digest", "")))),
             "task_family_contract_scope_invalid")
    _require(bool(_SHA.fullmatch(str(contract.get("runtime_identity_digest", "")))),
             "task_family_contract_runtime_invalid")
    _require(contract.get("claim_ceiling") in {"development_only", "site_scoped_simulation"},
             "task_family_claim_ceiling_invalid")
    _require(_number(contract.get("maximum_sample_gap_seconds")) and contract["maximum_sample_gap_seconds"] > 0,
             "task_family_sample_gap_invalid")
    stages = contract.get("stages")
    _require(isinstance(stages, list) and bool(stages) and all(isinstance(stage, dict) for stage in stages),
             "task_family_stages_invalid")
    ids = [stage.get("stage_id") for stage in stages]
    _require(all(isinstance(key, str) and key.strip() for key in ids) and len(set(ids)) == len(ids),
             "task_family_stage_ids_invalid")
    _require((family == "multi_step" and len(stages) >= 2) or
             (family != "multi_step" and len(stages) == 1 and stages[0].get("task_family") == family),
             "task_family_stage_family_mismatch")
    reset, tolerances = contract.get("reset"), contract.get("reset_tolerances", {})
    _require(isinstance(reset, dict) and set(reset) == set(ids) and isinstance(tolerances, dict)
             and set(tolerances) <= set(ids), "task_family_reset_invalid")
    for stage in stages:
        key, kind, criteria = stage["stage_id"], stage.get("task_family"), stage.get("criteria")
        _require(isinstance(kind, str) and kind in FAMILY_CRITERIA, "task_family_stage_kind_invalid")
        _require(isinstance(criteria, dict) and set(criteria) == set(FAMILY_CRITERIA[kind]),
                 "task_family_criteria_invalid:" + key)
        _require(all((type(v) is bool if name == "require_release" else
                      _number(v) and v >= 0 and (not name.endswith("fraction") or v <= 1))
                     for name, v in criteria.items()), "task_family_threshold_invalid:" + key)
        for name in ("minimum_transferred_volume_ml", "minimum_cleaned_fraction", "minimum_insertion_depth_m",
                     "minimum_button_travel_m", "minimum_fold_overlap_fraction"):
            if name in criteria:
                _require(criteria[name] > 0, "task_family_progress_threshold_invalid:" + key)
        hold = stage.get("minimum_hold_seconds")
        _require(_number(hold) and hold >= 0 and (kind not in {"stacking", "cloth_folding"} or hold > 0),
                 "task_family_hold_invalid:" + key)
        initial = reset[key]
        _require(isinstance(initial, dict) and set(initial) == set(FAMILY_OBSERVABLES[kind])
                 and all(_metric_valid(name, item) for name, item in initial.items()),
                 "task_family_reset_measurements_invalid:" + key)
        tolerance = tolerances.get(key, {})
        _require(isinstance(tolerance, dict) and set(tolerance) <= set(initial)
                 and all(name not in _BOOL_METRICS and _number(item) and item >= 0
                         for name, item in tolerance.items()), "task_family_reset_tolerance_invalid:" + key)
        if kind == "wiping":
            _require(initial["dirty_area_m2"] > 0, "task_family_reset_has_no_work:" + key)
        if kind == "pouring":
            _require(initial["source_volume_ml"] >= criteria["minimum_transferred_volume_ml"],
                     "task_family_reset_has_no_work:" + key)
        _require(not _stage_satisfied(stage, initial, initial), "task_family_reset_already_satisfied:" + key)
    return contract


def seal_task_family_contract(*, task_family: str, scope: Mapping[str, Any], stages: list[Mapping[str, Any]],
                              reset: Mapping[str, Any], runtime_identity_digest: str,
                              maximum_sample_gap_seconds: float = 1.0,
                              reset_tolerances: Mapping[str, Any] | None = None,
                              claim_ceiling: str = "development_only") -> dict[str, Any]:
    """Seal owner-supplied thresholds; defaults only set telemetry sampling policy."""
    value = {"schema_version": CONTRACT_SCHEMA, "task_family": canonical_task_family(task_family),
             "scope": dict(scope), "stages": [dict(stage) for stage in stages], "reset": dict(reset),
             "reset_tolerances": dict(reset_tolerances or {}), "runtime_identity_digest": runtime_identity_digest,
             "maximum_sample_gap_seconds": maximum_sample_gap_seconds, "claim_ceiling": claim_ceiling}
    value["contract_digest"] = canonical_digest(value, digest_field="contract_digest")
    return validate_task_family_contract(value)


def _stage_satisfied(stage: Mapping[str, Any], row: Mapping[str, Any], initial: Mapping[str, Any]) -> bool:
    c, kind = stage["criteria"], stage["task_family"]
    if kind == "pouring":
        gained = row["receiver_volume_ml"] - initial["receiver_volume_ml"]
        spilled = row["spilled_volume_ml"] - initial["spilled_volume_ml"]
        in_flight = row["in_flight_volume_ml"] - initial["in_flight_volume_ml"]
        lost = initial["source_volume_ml"] - row["source_volume_ml"]
        return (gained >= c["minimum_transferred_volume_ml"] and 0 <= spilled <= c["maximum_spilled_volume_ml"]
                and row["in_flight_volume_ml"] <= c["maximum_final_in_flight_volume_ml"]
                and lost >= 0 and abs(lost - gained - spilled - in_flight) <= c["maximum_volume_balance_error_ml"])
    if kind == "wiping":
        return ((initial["dirty_area_m2"] - row["dirty_area_m2"]) / initial["dirty_area_m2"] >= c["minimum_cleaned_fraction"]
                and row["dirty_area_m2"] <= c["maximum_remaining_dirty_area_m2"])
    if kind == "insertion":
        return (row["insertion_depth_m"] >= c["minimum_insertion_depth_m"]
                and row["lateral_error_m"] <= c["maximum_lateral_error_m"]
                and row["alignment_error_rad"] <= c["maximum_alignment_error_rad"]
                and row["contact_force_n"] <= c["maximum_contact_force_n"])
    if kind == "stacking":
        return (row["support_contact"] and not row["robot_contact_active"]
                and row["com_support_margin_m"] >= c["minimum_support_margin_m"]
                and row["linear_speed_m_s"] <= c["maximum_linear_speed_m_s"]
                and row["angular_speed_rad_s"] <= c["maximum_angular_speed_rad_s"])
    if kind == "button_press":
        return (row["button_activated"] and row["button_travel_m"] >= c["minimum_button_travel_m"]
                and row["contact_force_n"] <= c["maximum_contact_force_n"])
    if kind == "cloth_folding":
        return (row["fold_keypoint_error_m"] <= c["maximum_fold_keypoint_error_m"]
                and row["fold_overlap_fraction"] >= c["minimum_fold_overlap_fraction"]
                and row["linear_speed_m_s"] <= c["maximum_linear_speed_m_s"]
                and (not c["require_release"] or not row["robot_contact_active"]))
    return False


def score_task_family_episode(*, contract: Mapping[str, Any], telemetry: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute task success from complete, ordered independent native readback.

    Missing native measurements produce ``undetermined``. A safety violation or
    exceeded force/spill bound anywhere in the episode produces failure, even
    when the terminal state satisfies its target. No policy success field is read.
    """
    contract = validate_task_family_contract(contract)
    _require(isinstance(telemetry, Mapping), "task_family_telemetry_invalid")
    telemetry = _clone(telemetry)
    _require(telemetry.get("schema_version") == TELEMETRY_SCHEMA, "task_family_telemetry_schema_invalid")
    _require(telemetry.get("telemetry_digest") == canonical_digest(telemetry, digest_field="telemetry_digest"),
             "task_family_telemetry_digest_invalid")
    _require(telemetry.get("source") == "native_simulator_state", "task_family_telemetry_source_invalid")
    _require(telemetry.get("scope") == contract["scope"]
             and telemetry.get("contract_digest") == contract["contract_digest"]
             and telemetry.get("runtime_identity_digest") == contract["runtime_identity_digest"],
             "task_family_telemetry_binding_mismatch")
    samples = telemetry.get("samples")
    _require(isinstance(samples, list) and bool(samples) and all(isinstance(row, dict) for row in samples),
             "task_family_samples_invalid")
    times = [row.get("time_seconds") for row in samples]
    _require(all(_number(t) and t >= 0 for t in times) and times[0] == 0
             and all(b > a for a, b in zip(times, times[1:])), "task_family_sample_time_invalid")
    _require(all(b - a <= contract["maximum_sample_gap_seconds"] + 1e-12 for a, b in zip(times, times[1:])),
             "task_family_sample_gap_exceeded")
    gaps: list[str] = []
    failures: set[str] = set()
    states: dict[str, list[bool]] = {}
    for i, sample in enumerate(samples):
        safety = sample.get("safety_violation")
        if type(safety) is not bool:
            gaps.append(f"safety_violation:{i}")
        elif safety:
            failures.add("safety_violation")
    for stage in contract["stages"]:
        key, kind = stage["stage_id"], stage["task_family"]
        initial = contract["reset"][key]
        states[key] = []
        for i, sample in enumerate(samples):
            measurements = sample.get("measurements")
            row = measurements.get(key) if isinstance(measurements, dict) else None
            if not isinstance(row, dict) or any(not _metric_valid(name, row.get(name)) for name in FAMILY_OBSERVABLES[kind]):
                gaps.append(f"native_measurements:{key}:{i}")
                states[key].append(False)
                continue
            if i == 0:
                tolerance = contract.get("reset_tolerances", {}).get(key, {})
                _require(all((row[name] is expected if name in _BOOL_METRICS else
                              abs(row[name] - expected) <= tolerance.get(name, 0))
                             for name, expected in initial.items()), "task_family_reset_mismatch:" + key)
            c = stage["criteria"]
            if "maximum_contact_force_n" in c and row["contact_force_n"] > c["maximum_contact_force_n"]:
                failures.add("contact_force_exceeded:" + key)
            if kind == "pouring":
                if row["spilled_volume_ml"] - initial["spilled_volume_ml"] > c["maximum_spilled_volume_ml"]:
                    failures.add("spill_limit_exceeded:" + key)
                balance = (initial["source_volume_ml"] - row["source_volume_ml"]
                           - row["receiver_volume_ml"] + initial["receiver_volume_ml"]
                           - row["spilled_volume_ml"] + initial["spilled_volume_ml"]
                           - row["in_flight_volume_ml"] + initial["in_flight_volume_ml"])
                if abs(balance) > c["maximum_volume_balance_error_ml"]:
                    failures.add("volume_balance_exceeded:" + key)
            states[key].append(_stage_satisfied(stage, row, initial))
    # Each ordered stage must newly complete after the preceding stage. An
    # early press followed by insertion cannot satisfy insertion-then-press.
    completions: list[dict[str, Any]] = []
    after = 0
    for stage in contract["stages"]:
        values, start, completed = states[stage["stage_id"]], None, None
        waiting_for_transition = values[after]
        for i in range(after + 1, len(samples)):
            if not values[i]:
                start = None
                waiting_for_transition = False
            elif waiting_for_transition:
                continue
            elif start is None:
                start = i
            if start is not None and times[i] - times[start] + 1e-12 >= stage["minimum_hold_seconds"]:
                completed = i
                break
        if completed is None:
            failures.add("stage_incomplete:" + stage["stage_id"])
            break
        completions.append({"stage_id": stage["stage_id"], "sample_index": completed,
                            "time_seconds": times[completed]})
        after = completed
    # The final task state must still be satisfied for its entire terminal hold.
    final = contract["stages"][-1]
    terminal = len(samples) - 1
    while terminal > 0 and states[final["stage_id"]][terminal - 1]:
        terminal -= 1
    if (not states[final["stage_id"]][-1]
            or times[-1] - times[terminal] + 1e-12 < final["minimum_hold_seconds"]):
        failures.add("terminal_state_not_held:" + final["stage_id"])
    status = "undetermined" if gaps else "failed" if failures else "succeeded"
    value = {"schema_version": REPORT_SCHEMA, "status": status, "task_succeeded": status == "succeeded",
             "task_family": contract["task_family"], "scope": contract["scope"],
             "contract_digest": contract["contract_digest"], "telemetry_digest": telemetry["telemetry_digest"],
             "runtime_identity_digest": contract["runtime_identity_digest"], "scoring_source": "deterministic_native_state",
             "failed_criteria": sorted(failures), "readback_gaps": gaps, "stage_completions": completions,
             "duration_seconds": times[-1], "claim_ceiling": contract["claim_ceiling"],
             "physical_validation": False}
    value["report_digest"] = canonical_digest(value, digest_field="report_digest")
    return value
