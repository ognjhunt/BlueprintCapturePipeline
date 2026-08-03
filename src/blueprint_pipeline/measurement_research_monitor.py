"""Deterministic research monitoring for measurement-method candidates.

The monitor consumes validated metadata extracted from primary sources.  It
never stores or executes source prose, follows embedded instructions, changes
catalog eligibility, or approves an admission stage.  It produces immutable
snapshots, candidate/version/access diffs, requalification alerts, benchmark
recommendations, and bounded regression plans suitable for a monthly operator
or automation run.

Network retrieval is an injected boundary.  Third-party responses are untrusted
and reduced to a fixed metadata schema before they can influence monitoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .measurement_adapter_runtime import ADAPTER_RECIPES
from .measurement_method_research_catalog import (
    priority_investigations,
    qualification_benchmark_blueprints,
    research_catalog_snapshot,
    research_intake_catalog,
    validate_research_method_candidate,
)
from .measurement_research_admission import REQUALIFICATION_TRIGGERS


SOURCE_OBSERVATION_SCHEMA_VERSION = "measurement_primary_source_observation.v1"
MONITOR_SNAPSHOT_SCHEMA_VERSION = "measurement_research_monitor_snapshot.v1"
MONITOR_REPORT_SCHEMA_VERSION = "measurement_research_monitor_report.v1"
MONITOR_SCHEDULE_SCHEMA_VERSION = "measurement_research_monitor_schedule.v1"

SOURCE_STATUSES = frozenset(
    {"available", "unavailable", "access_restricted", "not_modified", "error"}
)
CHANGE_TYPES = frozenset(
    {
        "candidate_added",
        "candidate_removed",
        "source_added",
        "source_removed",
        "source_content_changed",
        "version_changed",
        "access_changed",
        "source_became_unavailable",
        "source_became_available",
    }
)


class MeasurementResearchMonitorError(ValueError):
    def __init__(self, *codes: str):
        self.codes = tuple(sorted(set(code for code in codes if code)))
        super().__init__("; ".join(self.codes))


class PrimarySourceFetcher(Protocol):
    def fetch_metadata(self, *, reference: str, source_type: str) -> Mapping[str, Any]: ...


def _clone(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        result = json.loads(json.dumps(dict(value)))
    except (TypeError, ValueError) as exc:
        raise MeasurementResearchMonitorError("research_monitor_artifact_not_json") from exc
    return result


def _digest(value: Mapping[str, Any], field: str) -> str:
    normalized = dict(value)
    normalized.pop(field, None)
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _parse_datetime(value: Any) -> datetime:
    raw = _string(value)
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise MeasurementResearchMonitorError("research_monitor_datetime_invalid") from exc
    if parsed.tzinfo is None:
        raise MeasurementResearchMonitorError("research_monitor_datetime_timezone_missing")
    return parsed.astimezone(timezone.utc)


def build_source_observation(
    candidate_value: Mapping[str, Any],
    *,
    source_reference: str,
    source_type: str,
    observed_at: str,
    fetched_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Reduce an untrusted fetch result to stable metadata.

    Raw page text, HTML, scripts, model instructions, cookies, credentials, and
    response headers outside the allowlist are intentionally discarded.
    """

    candidate = validate_research_method_candidate(candidate_value)
    source_keys = {
        (_string(row.get("source_type")), _string(row.get("reference")))
        for row in candidate["primary_sources"]
    }
    key = (_string(source_type), _string(source_reference))
    if key not in source_keys:
        raise MeasurementResearchMonitorError("research_monitor_source_not_in_dossier")
    metadata = dict(fetched_metadata)
    status = _string(metadata.get("status"))
    if status not in SOURCE_STATUSES:
        raise MeasurementResearchMonitorError("research_monitor_source_status_invalid")
    _parse_datetime(observed_at)
    content_digest = _string(metadata.get("content_digest"))
    if status in {"available", "not_modified"} and not content_digest.startswith("sha256:"):
        raise MeasurementResearchMonitorError("research_monitor_content_digest_invalid")
    access = metadata.get("access")
    access = dict(access) if isinstance(access, Mapping) else {}
    allowed_access = {
        "source_available",
        "api_only",
        "local_offline_supported",
        "commercial_use_status",
        "license_identifier",
        "terms_version",
    }
    sanitized_access = {key: access[key] for key in sorted(allowed_access & set(access))}
    observation = {
        "schema_version": SOURCE_OBSERVATION_SCHEMA_VERSION,
        "candidate_id": candidate["candidate_id"],
        "candidate_digest": candidate["research_candidate_digest"],
        "source_type": source_type,
        "source_reference": source_reference,
        "observed_at": _parse_datetime(observed_at).isoformat(),
        "status": status,
        "content_digest": content_digest or None,
        "observed_version": _string(metadata.get("observed_version")) or None,
        "etag": _string(metadata.get("etag")) or None,
        "last_modified": _string(metadata.get("last_modified")) or None,
        "release_date": _string(metadata.get("release_date")) or None,
        "access": sanitized_access,
        "retrieval": {
            "http_status": metadata.get("http_status")
            if isinstance(metadata.get("http_status"), int)
            else None,
            "final_origin": _string(metadata.get("final_origin")) or None,
            "redirect_count": metadata.get("redirect_count")
            if isinstance(metadata.get("redirect_count"), int)
            else None,
        },
        "raw_content_persisted": False,
        "embedded_instructions_followed": False,
        "credentials_persisted": False,
        "candidate_claims_treated_as_qualification": False,
    }
    observation["source_observation_digest"] = _digest(observation, "source_observation_digest")
    return validate_source_observation(observation)


def validate_source_observation(value: Mapping[str, Any]) -> dict[str, Any]:
    observation = _clone(value)
    errors: list[str] = []
    if observation.get("schema_version") != SOURCE_OBSERVATION_SCHEMA_VERSION:
        errors.append("research_monitor_source_observation_schema_invalid")
    for key in (
        "candidate_id",
        "candidate_digest",
        "source_type",
        "source_reference",
        "observed_at",
        "status",
    ):
        if not _string(observation.get(key)):
            errors.append(f"research_monitor_source_observation_{key}_missing")
    try:
        _parse_datetime(observation.get("observed_at"))
    except MeasurementResearchMonitorError as exc:
        errors.extend(exc.codes)
    if observation.get("status") not in SOURCE_STATUSES:
        errors.append("research_monitor_source_status_invalid")
    if not isinstance(observation.get("access"), Mapping):
        errors.append("research_monitor_source_access_invalid")
    for key in (
        "raw_content_persisted",
        "embedded_instructions_followed",
        "credentials_persisted",
        "candidate_claims_treated_as_qualification",
    ):
        if observation.get(key) is not False:
            errors.append(f"research_monitor_source_observation_{key}_must_be_false")
    expected = _digest(observation, "source_observation_digest")
    supplied = observation.get("source_observation_digest")
    if supplied is not None and supplied != expected:
        errors.append("research_monitor_source_observation_digest_mismatch")
    if errors:
        raise MeasurementResearchMonitorError(*errors)
    observation["source_observation_digest"] = expected
    return observation


def collect_primary_source_observations(
    fetcher: PrimarySourceFetcher,
    *,
    candidate_ids: Sequence[str],
    observed_at: str,
) -> tuple[dict[str, Any], ...]:
    """Fetch and sanitize primary-source metadata through an injected client."""

    candidates = {row["candidate_id"]: row for row in research_intake_catalog()}
    selected = sorted({_string(item) for item in candidate_ids if _string(item)})
    unknown = sorted(set(selected) - set(candidates))
    if unknown:
        raise MeasurementResearchMonitorError(
            "research_monitor_candidate_unknown:" + ",".join(unknown)
        )
    observations: list[dict[str, Any]] = []
    for candidate_id in selected:
        candidate = candidates[candidate_id]
        for source in candidate["primary_sources"]:
            metadata = fetcher.fetch_metadata(
                reference=source["reference"], source_type=source["source_type"]
            )
            if not isinstance(metadata, Mapping):
                raise MeasurementResearchMonitorError("research_monitor_fetch_response_invalid")
            observations.append(
                build_source_observation(
                    candidate,
                    source_reference=source["reference"],
                    source_type=source["source_type"],
                    observed_at=observed_at,
                    fetched_metadata=metadata,
                )
            )
    return tuple(observations)


def build_monitor_snapshot(
    observations: Sequence[Mapping[str, Any]], *, observed_at: str
) -> dict[str, Any]:
    validated = [validate_source_observation(row) for row in observations]
    keys = [(row["candidate_id"], row["source_type"], row["source_reference"]) for row in validated]
    if len(keys) != len(set(keys)):
        raise MeasurementResearchMonitorError("research_monitor_duplicate_source")
    _parse_datetime(observed_at)
    catalog = research_catalog_snapshot()
    snapshot = {
        "schema_version": MONITOR_SNAPSHOT_SCHEMA_VERSION,
        "catalog_version": catalog["catalog_version"],
        "catalog_snapshot_digest": catalog["catalog_snapshot_digest"],
        "observed_at": _parse_datetime(observed_at).isoformat(),
        "observations": sorted(
            validated,
            key=lambda row: (row["candidate_id"], row["source_type"], row["source_reference"]),
        ),
        "production_route_count": 0,
        "qualification_decisions_created": 0,
        "raw_source_content_persisted": False,
    }
    snapshot["monitor_snapshot_digest"] = _digest(snapshot, "monitor_snapshot_digest")
    return validate_monitor_snapshot(snapshot)


def validate_monitor_snapshot(value: Mapping[str, Any]) -> dict[str, Any]:
    snapshot = _clone(value)
    errors: list[str] = []
    if snapshot.get("schema_version") != MONITOR_SNAPSHOT_SCHEMA_VERSION:
        errors.append("research_monitor_snapshot_schema_invalid")
    try:
        _parse_datetime(snapshot.get("observed_at"))
    except MeasurementResearchMonitorError as exc:
        errors.extend(exc.codes)
    observations = snapshot.get("observations")
    if not isinstance(observations, list):
        errors.append("research_monitor_snapshot_observations_invalid")
    else:
        try:
            snapshot["observations"] = [
                validate_source_observation(row) for row in observations if isinstance(row, Mapping)
            ]
        except MeasurementResearchMonitorError as exc:
            errors.extend(exc.codes)
        if len(snapshot["observations"]) != len(observations):
            errors.append("research_monitor_snapshot_observations_invalid")
    for key in ("production_route_count", "qualification_decisions_created"):
        if snapshot.get(key) != 0:
            errors.append(f"research_monitor_snapshot_{key}_must_be_zero")
    if snapshot.get("raw_source_content_persisted") is not False:
        errors.append("research_monitor_snapshot_raw_content_forbidden")
    expected = _digest(snapshot, "monitor_snapshot_digest")
    supplied = snapshot.get("monitor_snapshot_digest")
    if supplied is not None and supplied != expected:
        errors.append("research_monitor_snapshot_digest_mismatch")
    if errors:
        raise MeasurementResearchMonitorError(*errors)
    snapshot["monitor_snapshot_digest"] = expected
    return snapshot


def build_monthly_monitor_schedule(
    *,
    schedule_id: str,
    timezone_name: str,
    day_of_month: int,
    last_completed_at: str | None = None,
) -> dict[str, Any]:
    if not _string(schedule_id) or not _string(timezone_name):
        raise MeasurementResearchMonitorError("research_monitor_schedule_identity_missing")
    if not isinstance(day_of_month, int) or not 1 <= day_of_month <= 28:
        raise MeasurementResearchMonitorError("research_monitor_schedule_day_invalid")
    if last_completed_at is not None:
        _parse_datetime(last_completed_at)
    schedule = {
        "schema_version": MONITOR_SCHEDULE_SCHEMA_VERSION,
        "schedule_id": schedule_id,
        "cadence": "monthly",
        "timezone": timezone_name,
        "day_of_month": day_of_month,
        "last_completed_at": (
            _parse_datetime(last_completed_at).isoformat()
            if last_completed_at is not None
            else None
        ),
        "entrypoint": "python -m blueprint_pipeline.measurement_research_monitor",
        "mode": "metadata_only_fail_closed",
        "network_fetcher_must_be_explicit": True,
        "human_approval_required_for_admission": True,
        "automatic_catalog_promotion": False,
    }
    schedule["monitor_schedule_digest"] = _digest(schedule, "monitor_schedule_digest")
    return schedule


def monitor_is_due(schedule_value: Mapping[str, Any], *, as_of: str) -> bool:
    schedule = _clone(schedule_value)
    if schedule.get("schema_version") != MONITOR_SCHEDULE_SCHEMA_VERSION:
        raise MeasurementResearchMonitorError("research_monitor_schedule_schema_invalid")
    now = _parse_datetime(as_of)
    last = schedule.get("last_completed_at")
    if last is None:
        return True
    return now >= _parse_datetime(last) + timedelta(days=28)


def _observation_key(value: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        _string(value.get("candidate_id")),
        _string(value.get("source_type")),
        _string(value.get("source_reference")),
    )


def _changes(
    previous: Mapping[str, Any] | None, current: Mapping[str, Any]
) -> list[dict[str, Any]]:
    old_rows = {
        _observation_key(row): row
        for row in (previous or {}).get("observations", [])
        if isinstance(row, Mapping)
    }
    new_rows = {
        _observation_key(row): row
        for row in current.get("observations", [])
        if isinstance(row, Mapping)
    }
    changes: list[dict[str, Any]] = []
    for key in sorted(set(old_rows) | set(new_rows)):
        old = old_rows.get(key)
        new = new_rows.get(key)
        if old is None:
            changes.append(
                {"change_type": "source_added", "candidate_id": key[0], "source": key[2]}
            )
            continue
        if new is None:
            changes.append(
                {"change_type": "source_removed", "candidate_id": key[0], "source": key[2]}
            )
            continue
        comparisons = (
            ("observed_version", "version_changed"),
            ("content_digest", "source_content_changed"),
            ("access", "access_changed"),
        )
        for field, change_type in comparisons:
            if old.get(field) != new.get(field):
                changes.append(
                    {
                        "change_type": change_type,
                        "candidate_id": key[0],
                        "source": key[2],
                        "old": old.get(field),
                        "new": new.get(field),
                    }
                )
        if old.get("status") in {"available", "not_modified"} and new.get("status") not in {
            "available",
            "not_modified",
        }:
            changes.append(
                {
                    "change_type": "source_became_unavailable",
                    "candidate_id": key[0],
                    "source": key[2],
                }
            )
        if old.get("status") not in {"available", "not_modified"} and new.get("status") in {
            "available",
            "not_modified",
        }:
            changes.append(
                {
                    "change_type": "source_became_available",
                    "candidate_id": key[0],
                    "source": key[2],
                }
            )
    return changes


def _alert(change: Mapping[str, Any]) -> dict[str, Any]:
    change_type = _string(change.get("change_type"))
    trigger = (
        "license_or_privacy_change"
        if change_type == "access_changed"
        else "engine_solver_api_or_model_update"
        if change_type in {"version_changed", "source_content_changed"}
        else "new_failure_mode_discovered"
        if change_type in {"source_removed", "source_became_unavailable"}
        else "adapter_modification"
    )
    assert trigger in REQUALIFICATION_TRIGGERS
    severity = (
        "critical"
        if change_type in {"access_changed", "source_became_unavailable"}
        else "high"
        if change_type in {"version_changed", "source_removed"}
        else "review"
    )
    return {
        "alert_id": "alert-"
        + hashlib.sha256(json.dumps(change, sort_keys=True).encode()).hexdigest()[:16],
        "candidate_id": change.get("candidate_id"),
        "severity": severity,
        "change_type": change_type,
        "requalification_trigger": trigger,
        "automatic_suspension_recommended": severity in {"critical", "high"},
        "human_review_required": True,
        "automatic_reapproval_forbidden": True,
    }


def _benchmark_recommendations(candidate_ids: set[str]) -> list[dict[str, Any]]:
    recommendations: list[dict[str, Any]] = []
    for blueprint in qualification_benchmark_blueprints():
        matched = sorted(candidate_ids & set(blueprint["methods_compared"]))
        if matched:
            recommendations.append(
                {
                    "benchmark_id": blueprint["benchmark_id"],
                    "candidate_ids": matched,
                    "protocols": blueprint["protocols"],
                    "action": "rerun_development_then_independent_qualification_if_changed",
                }
            )
    return recommendations


def _regression_plan(
    changes: Sequence[Mapping[str, Any]], recommendations: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    candidate_ids = sorted(
        {_string(row.get("candidate_id")) for row in changes if row.get("candidate_id")}
    )
    adapter_ids = sorted(set(candidate_ids) & set(ADAPTER_RECIPES))
    return {
        "schema_version": "measurement_research_regression_plan.v1",
        "candidate_ids": candidate_ids,
        "adapter_probe_candidate_ids": adapter_ids,
        "benchmark_ids": sorted({_string(row.get("benchmark_id")) for row in recommendations}),
        "focused_test_commands": [
            ".venv/bin/python -m pytest -q tests/test_measurement_adapter_runtime.py",
            ".venv/bin/python -m pytest -q tests/test_measurement_qualification_benchmarks.py",
            ".venv/bin/python -m pytest -q tests/test_measurement_research_monitor.py",
        ],
        "paid_execution_authorized": False,
        "provider_execution_authorized": False,
        "physical_execution_authorized": False,
        "human_review_required_before_r4_or_later": True,
    }


def build_monitor_report(
    current_value: Mapping[str, Any],
    previous_value: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    current = validate_monitor_snapshot(current_value)
    previous = validate_monitor_snapshot(previous_value) if previous_value is not None else None
    if previous is not None and _parse_datetime(previous["observed_at"]) >= _parse_datetime(
        current["observed_at"]
    ):
        raise MeasurementResearchMonitorError("research_monitor_snapshot_order_invalid")
    changes = _changes(previous, current)
    if any(row["change_type"] not in CHANGE_TYPES for row in changes):
        raise MeasurementResearchMonitorError("research_monitor_change_type_invalid")
    changed_candidates = {
        _string(row.get("candidate_id")) for row in changes if row.get("candidate_id")
    }
    recommendations = _benchmark_recommendations(changed_candidates)
    priorities = {
        candidate_id: row["priority"]
        for row in priority_investigations()
        for candidate_id in row["candidate_ids"]
    }
    alerts = [_alert(row) for row in changes]
    report = {
        "schema_version": MONITOR_REPORT_SCHEMA_VERSION,
        "previous_snapshot_digest": (
            previous["monitor_snapshot_digest"] if previous is not None else None
        ),
        "current_snapshot_digest": current["monitor_snapshot_digest"],
        "observed_at": current["observed_at"],
        "changes": changes,
        "alerts": alerts,
        "benchmark_recommendations": recommendations,
        "priority_candidates_changed": sorted(
            (
                {"candidate_id": candidate_id, "priority": priorities[candidate_id]}
                for candidate_id in changed_candidates
                if candidate_id in priorities
            ),
            key=lambda row: (row["priority"], row["candidate_id"]),
        ),
        "regression_plan": _regression_plan(changes, recommendations),
        "catalog_mutated": False,
        "admission_advanced": False,
        "qualification_created": False,
        "production_route_created": False,
        "agent_approval_effect": False,
    }
    report["monitor_report_digest"] = _digest(report, "monitor_report_digest")
    return validate_monitor_report(report)


def validate_monitor_report(value: Mapping[str, Any]) -> dict[str, Any]:
    report = _clone(value)
    errors: list[str] = []
    if report.get("schema_version") != MONITOR_REPORT_SCHEMA_VERSION:
        errors.append("research_monitor_report_schema_invalid")
    for key in ("changes", "alerts", "benchmark_recommendations"):
        if not isinstance(report.get(key), list):
            errors.append(f"research_monitor_report_{key}_invalid")
    for key in (
        "catalog_mutated",
        "admission_advanced",
        "qualification_created",
        "production_route_created",
        "agent_approval_effect",
    ):
        if report.get(key) is not False:
            errors.append(f"research_monitor_report_{key}_must_be_false")
    expected = _digest(report, "monitor_report_digest")
    supplied = report.get("monitor_report_digest")
    if supplied is not None and supplied != expected:
        errors.append("research_monitor_report_digest_mismatch")
    if errors:
        raise MeasurementResearchMonitorError(*errors)
    report["monitor_report_digest"] = expected
    return report


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MeasurementResearchMonitorError("research_monitor_input_unreadable") from exc
    if not isinstance(value, Mapping):
        raise MeasurementResearchMonitorError("research_monitor_input_not_object")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a fail-closed measurement research monitor report"
    )
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--previous-snapshot", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--observed-at", default=datetime.now(timezone.utc).isoformat())
    args = parser.parse_args(argv)
    observation_input = _load_json(args.observations)
    rows = observation_input.get("observations")
    if not isinstance(rows, list):
        raise MeasurementResearchMonitorError("research_monitor_observations_missing")
    current = build_monitor_snapshot(
        [row for row in rows if isinstance(row, Mapping)], observed_at=args.observed_at
    )
    previous = _load_json(args.previous_snapshot) if args.previous_snapshot else None
    report = build_monitor_report(current, previous)
    output = {"snapshot": current, "report": report}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CHANGE_TYPES",
    "MONITOR_REPORT_SCHEMA_VERSION",
    "MONITOR_SCHEDULE_SCHEMA_VERSION",
    "MONITOR_SNAPSHOT_SCHEMA_VERSION",
    "MeasurementResearchMonitorError",
    "PrimarySourceFetcher",
    "SOURCE_OBSERVATION_SCHEMA_VERSION",
    "build_monitor_report",
    "build_monitor_snapshot",
    "build_monthly_monitor_schedule",
    "build_source_observation",
    "collect_primary_source_observations",
    "main",
    "monitor_is_due",
    "validate_monitor_report",
    "validate_monitor_snapshot",
    "validate_source_observation",
]
