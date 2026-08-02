"""Automated research/release monitoring for measurement methods.

Implements the longer-term automation lane of the measurement-routing
research: release monitoring, candidate-diff generation, version-change
alerts, R0 intake drafts for newly discovered methods, and requalification
trigger proposals for admitted methods.

Boundary: automation reduces research latency; it never weakens approval.
This module only ever emits *proposals* — it cannot create catalog entries,
advance R0-R8 stages, apply requalification triggers, or approve anything.
Every output artifact carries ``human_action_required=True`` and
``automation_approved_anything=False``, and applying a proposed trigger still
requires ``apply_requalification_trigger`` with a human approval row.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

from .measurement_engine_capability_profiles import engine_capability_profiles
from .measurement_method_research_catalog import (
    RESEARCH_CATALOG_VERSION,
    research_intake_catalog,
)
from .measurement_research_admission import (
    MeasurementAdmissionError,
    validate_research_admission_record,
)


RELEASE_OBSERVATION_SCHEMA_VERSION = "measurement_release_observation.v1"
MONITORING_REPORT_SCHEMA_VERSION = "measurement_research_monitoring_report.v1"

ALERT_KINDS = frozenset(
    {
        "version_changed",
        "new_method_discovered",
        "capability_profile_stale",
        "unchanged",
    }
)


class ResearchMonitoringError(ValueError):
    def __init__(self, *codes: str):
        self.codes = tuple(sorted(set(code for code in codes if code)))
        super().__init__("; ".join(self.codes))


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _digest(value: Mapping[str, Any], field: str) -> str:
    normalized = {key: item for key, item in value.items() if key != field}
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def validate_release_observation(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        observation = json.loads(json.dumps(dict(value)))
    except (TypeError, ValueError) as exc:
        raise ResearchMonitoringError("release_observation_not_json") from exc
    errors: list[str] = []
    if observation.get("schema_version") != RELEASE_OBSERVATION_SCHEMA_VERSION:
        errors.append("release_observation_schema_version_invalid")
    for key in ("method_id", "observed_version", "source_reference", "observed_on"):
        if not _string(observation.get(key)):
            errors.append(f"release_observation_{key}_missing")
    if errors:
        raise ResearchMonitoringError(*errors)
    observation.setdefault("observed_release_date", "")
    observation.setdefault("notes", "")
    observation["release_observation_digest"] = _digest(observation, "release_observation_digest")
    return observation


def build_release_observation(
    *,
    method_id: str,
    observed_version: str,
    source_reference: str,
    observed_on: str,
    observed_release_date: str = "",
    notes: str = "",
) -> dict[str, Any]:
    return validate_release_observation(
        {
            "schema_version": RELEASE_OBSERVATION_SCHEMA_VERSION,
            "method_id": method_id,
            "observed_version": observed_version,
            "observed_release_date": observed_release_date,
            "source_reference": source_reference,
            "observed_on": observed_on,
            "notes": notes,
        }
    )


def _r0_intake_draft(observation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "draft_kind": "r0_intake_draft",
        "candidate_id": f"intake-{observation['method_id']}",
        "method_id": observation["method_id"],
        "stage_data_skeleton": {
            "primary_sources": [observation["source_reference"]],
            "method_identity": {
                "method_id": observation["method_id"],
                "version_observed": observation["observed_version"],
            },
            "claimed_scope": {
                "status": "human_completion_required",
            },
            "access_status": {
                "status": "human_completion_required",
            },
        },
        "requires_human_research_analyst_approval": True,
        "automation_may_create_catalog_entry": False,
    }


def _requalification_proposals(
    method_id: str,
    detail: str,
    admission_records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    proposals: list[dict[str, Any]] = []
    invalid: list[str] = []
    for raw in admission_records:
        try:
            record = validate_research_admission_record(raw)
        except MeasurementAdmissionError as exc:
            invalid.append(f"admission_record_invalid:{';'.join(exc.codes)}")
            continue
        if record["method_id"] != method_id:
            continue
        if record["stage"] not in {"R7", "R8"} or record["suspended"] is True:
            continue
        proposals.append(
            {
                "proposal_kind": "requalification_trigger_proposal",
                "method_id": method_id,
                "admission_record_digest": record["admission_record_digest"],
                "trigger": "engine_solver_api_or_model_update",
                "detail": detail,
                "apply_via": "measurement_research_admission.apply_requalification_trigger",
                "requires_human_approval": True,
                "automation_applied": False,
            }
        )
    return proposals, invalid


def compile_research_monitoring_report(
    observations: Sequence[Mapping[str, Any]],
    *,
    observed_on: str,
    admission_records: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Diff release observations against the catalog and engine profiles."""

    if not _string(observed_on):
        raise ResearchMonitoringError("monitoring_observed_on_missing")
    validated = [validate_release_observation(row) for row in observations]
    catalog = {row["method_id"]: row for row in research_intake_catalog()}
    profiles = {row["method_id"]: row["capabilities"] for row in engine_capability_profiles()}
    alerts: list[dict[str, Any]] = []
    intake_drafts: list[dict[str, Any]] = []
    trigger_proposals: list[dict[str, Any]] = []
    regression_checks: list[str] = []
    input_errors: list[str] = []
    for observation in sorted(validated, key=lambda row: row["method_id"]):
        method_id = observation["method_id"]
        entry = catalog.get(method_id)
        if entry is None:
            alerts.append(
                {
                    "kind": "new_method_discovered",
                    "method_id": method_id,
                    "observed_version": observation["observed_version"],
                    "source_reference": observation["source_reference"],
                }
            )
            intake_drafts.append(_r0_intake_draft(observation))
            continue
        catalog_version = _string(entry.get("version_observed"))
        observed_version = observation["observed_version"]
        version_changed = bool(catalog_version) and catalog_version != observed_version
        profile_capabilities = profiles.get(method_id)
        profile_stale = (
            profile_capabilities is not None
            and _string(profile_capabilities.get("version")) != observed_version
        )
        if version_changed:
            alerts.append(
                {
                    "kind": "version_changed",
                    "method_id": method_id,
                    "catalog_version": catalog_version,
                    "observed_version": observed_version,
                    "source_reference": observation["source_reference"],
                    "actions": [
                        "update_r0_dossier",
                        "r1_source_reverification",
                        "regression_check_required",
                    ],
                }
            )
            regression_checks.append(method_id)
            detail = (
                f"{method_id}: catalog {catalog_version or 'unversioned'} -> "
                f"observed {observed_version} ({observation['source_reference']})"
            )
            proposals, invalid = _requalification_proposals(method_id, detail, admission_records)
            trigger_proposals.extend(proposals)
            input_errors.extend(invalid)
        elif profile_stale:
            alerts.append(
                {
                    "kind": "capability_profile_stale",
                    "method_id": method_id,
                    "profile_version": _string(profile_capabilities.get("version")),
                    "observed_version": observed_version,
                    "actions": ["r1_source_reverification"],
                }
            )
        else:
            alerts.append(
                {
                    "kind": "unchanged",
                    "method_id": method_id,
                    "observed_version": observed_version,
                }
            )
    report = {
        "schema_version": MONITORING_REPORT_SCHEMA_VERSION,
        "observed_on": observed_on,
        "catalog_version": RESEARCH_CATALOG_VERSION,
        "observation_digests": sorted(row["release_observation_digest"] for row in validated),
        "alerts": alerts,
        "r0_intake_drafts": intake_drafts,
        "requalification_trigger_proposals": trigger_proposals,
        "regression_checks_required": sorted(set(regression_checks)),
        "input_errors": sorted(set(input_errors)),
        "human_action_required": True,
        "automation_approved_anything": False,
        "automation_advanced_any_stage": False,
    }
    report["monitoring_report_digest"] = _digest(report, "monitoring_report_digest")
    return report


def github_latest_release_observation(
    *, method_id: str, repository: str, observed_on: str, timeout_seconds: float = 20.0
) -> dict[str, Any]:
    """Fetch the latest GitHub release for ``owner/name`` (network; CLI use).

    Hermetic tests never call this. Failures return a validated observation
    with ``observed_version`` set to ``fetch_failed`` so a monitoring run
    records the attempt instead of silently skipping the method.
    """

    import os
    import urllib.error
    import urllib.request

    url = f"https://api.github.com/repos/{repository}/releases/latest"
    try:
        headers = {"Accept": "application/vnd.github+json"}
        token = _string(os.environ.get("GITHUB_TOKEN"))
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        version = _string(payload.get("tag_name")).lstrip("v") or "unparsed"
        release_date = _string(payload.get("published_at"))[:10]
        notes = ""
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError) as exc:
        version = "fetch_failed"
        release_date = ""
        notes = f"fetch_error:{type(exc).__name__}"
    return build_release_observation(
        method_id=method_id,
        observed_version=version,
        observed_release_date=release_date,
        source_reference=url,
        observed_on=observed_on,
        notes=notes,
    )


__all__ = [
    "ALERT_KINDS",
    "MONITORING_REPORT_SCHEMA_VERSION",
    "RELEASE_OBSERVATION_SCHEMA_VERSION",
    "ResearchMonitoringError",
    "build_release_observation",
    "compile_research_monitoring_report",
    "github_latest_release_observation",
    "validate_release_observation",
]
