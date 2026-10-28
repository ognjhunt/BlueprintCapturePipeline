"""Secret-free projection of the existing root-owned production preflight.

No environments, paths from jobs, command lines, remote responses or customer
content enter this projection. The private report is retained with mode 0600.
This module does not run a preflight or grant filesystem/runtime authority.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import pwd
import re
import stat
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .production_preflight_routes import BETA_RIGID_UNITS

SCHEMA_VERSION = "task_evaluation_production_chain_preflight_summary.v1"
MAX_SUMMARY_BYTES = 128 * 1024
_SHA = re.compile(r"[0-9a-f]{40}")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_UNIT = re.compile(r"blueprint-[a-z0-9-]{1,100}\.(?:service|timer|path)")
_PUBLIC_UNITS = frozenset(BETA_RIGID_UNITS) | frozenset(
    "blueprint-" + name + ".service" for name in (
        "task-evaluation-launch-preparation", "task-evaluation-sam31-preparation-execution",
        "task-evaluation-episode-compilation", "task-evaluation-episode-compilation-remote",
        "task-evaluation-launch-activation", "task-evaluation-configured-controls-progression",
        "task-evaluation-launch-dispatcher", "task-evaluation-launch-reconciler",
        "task-evaluation-launch-supervisor", "task-evaluation-policy-canary-dispatcher",
        "task-evaluation-terminal-resource-release", "gpu-spend-guard",
        "provider-billing-reconciler", "control-plane-storage-gc", "scene-project-spend-refresh",
    )
)
# Only code-owned enum values are published. Unrecognized codes remain in the
# private report and have a generic operational next action in this projection.
_CODES = frozenset("""
activation_intent_missing_for_active_release active_release_link_unresolvable active_release_name_not_commit
closure_reads_home_under_protect_home configured_file_missing configured_file_unreadable_by_service
controls_autoprovision_active_release_unresolved controls_autoprovision_asset_missing
controls_autoprovision_config_incomplete controls_autoprovision_config_unreadable_by_service
controls_autoprovision_config_unset controls_autoprovision_queue_root_disagrees_with_scene_progression
controls_autoprovision_robot_catalog_invalid controls_autoprovision_robot_catalog_unbindable
controls_autoprovision_robot_catalog_unreadable controls_autoprovision_root_missing
controls_autoprovision_root_not_writable_by_service controls_autoprovision_runtime_missing
controls_robot_binding_unresolvable controls_scene_intent_unreadable controls_scene_robot_bindings_resolved
disk_admission_configuration_invalid disk_admission_projection disk_reservations_unreadable disk_usage_unreadable
dispatch_owner_scope_not_persistent_owner_only entry_module_import_failed entry_module_unrecognised
environment_file_missing episode_compilation_auto_config_unusable git_safe_directory_names_other_release
input_installed_under_other_release_prefix installation_receipt_commit_invalid installation_receipt_installed_by_other_release
installation_receipt_unreadable installation_receipt_unreadable_by_service installed_source_binding
installed_source_bindings_invalid_json installed_source_bindings_unset intake_catalog_route_not_ok
intake_catalog_route_unreachable launch_dispatcher_execute_id_unset launch_dispatcher_execution_hold_present
launch_profile_catalog_path_disagrees_with_intake launch_profile_catalog_path_unset
omitted_controls_camera_calibration_invalid omitted_controls_camera_calibration_missing
omitted_controls_camera_calibration_reference_changed omitted_controls_camera_calibration_reference_unreadable
omitted_controls_camera_calibration_unreadable_or_unsafe omitted_controls_robot_asset_digest_missing
owner_scene_intake_root_unreadable_by_service owner_store_config_unreadable_by_service
owner_store_trusted_clients_unset owner_store_unresolvable paid_admission_checkout_identity_refused
paid_admission_identity_probe_failed path_bound_to_other_release policy_canary_notification_email_unset
provider_credit_exhausted provider_credit_guard_disabled provider_credit_key_file_unreadable
provider_credit_key_file_unset provider_credit_unverifiable provider_credit_low provider_credit_available
public_catalog_file_unreadable_by_service publisher_intake_digest_mismatch publisher_intake_unreadable_by_service
release_evidence_grade_development_only release_worktree_head_unreadable release_worktree_identity_drift
remote_cpu_image_drift root_resolver_failed root_resolver_unset root_resolves_under_protected_home
sam31_profile_bound_to_other_release sam31_profile_registry_content_bound sam31_profile_registry_unreadable_by_service
sam31_profile_unbound sam31_profile_unreadable_by_service sam31_provider_profile_bound_to_active_release
sam31_provider_profile_bound_to_other_release sam31_provider_profile_reference_missing sam31_provider_profile_unreadable
sandbox_probe_failed scene_project_spend_config_invalid scene_project_spend_config_unreadable
scene_project_spend_config_unset scene_project_spend_refresh_lock_not_writable scene_project_spend_seed_invalid
scene_project_spend_seed_reference_invalid scene_project_spend_seed_unreadable service_account_missing
ssh_identity_permissions_refused_by_ssh terminal_index_root_unset unit_execution_hold_present
unit_intentionally_held unit_failed_state unit_not_found unit_not_loaded unit_trigger_inactive
unit_trigger_not_required unit_required_failed unit_successful_oneshot webapp_submission_secret_file_unreadable_by_service
webapp_submission_url_unset path_read_only path_permission_denied path_error path_writable_unlink_failed
path_unreadable path_dangling_symlink path_stat_failed static_path_requirement_unconfirmed
path_template_unresolved required_path_access_refused path_access_unproven path_required_missing
entrypoint_coverage_unknown
""".split())


class PreflightSummaryError(ValueError):
    """A sanitized reason; never includes report bytes or filesystem content."""


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _action(code: str) -> tuple[str, str, str]:
    if "held" in code or "hold" in code:
        return "intentional_hold", "task_owner", "await_existing_owner_release"
    if "path" in code or "root" in code or "file" in code:
        return "filesystem", "operator", "inspect_selected_route_access_contract"
    if "trigger" in code or "unit" in code:
        return "service", "operator", "inspect_required_trigger_or_invocation"
    if "source" in code or "release" in code or "digest" in code:
        return "identity", "operator", "reconcile_exact_release_and_source_binding"
    if "credit" in code or "spend" in code or "disk" in code:
        return "capacity", "operator", "inspect_existing_capacity_or_funding_state"
    return "diagnostic", "operator", "inspect_retained_preflight_cause"


def build_safe_summary(report: Mapping[str, Any], *, raw_report_bytes: bytes) -> dict[str, Any]:
    if not isinstance(report, Mapping) or report.get("schema_version") != "task_evaluation_production_chain_preflight.v1":
        raise PreflightSummaryError("preflight_report_schema_invalid")
    if not _SHA.fullmatch(str(report.get("active_sha") or "")):
        raise PreflightSummaryError("preflight_report_release_invalid")
    timestamp = report.get("generated_at")
    try:
        observed = dt.datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc).timestamp()
    except (ValueError, TypeError):
        raise PreflightSummaryError("preflight_report_timestamp_invalid") from None
    try:
        decoded = json.loads(raw_report_bytes)
    except (ValueError, TypeError, UnicodeError):
        raise PreflightSummaryError("preflight_report_bytes_invalid") from None
    if decoded != report:
        raise PreflightSummaryError("preflight_report_bytes_mismatch")
    units = report.get("units")
    if not isinstance(units, Mapping) or len(units) > 128 or not isinstance(report.get("host_findings"), list):
        raise PreflightSummaryError("preflight_report_shape_invalid")
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    counts = {key: 0 for key in ("blocker", "warning", "info")}

    def add(scope: str, row: Any) -> None:
        if not isinstance(row, Mapping) or row.get("severity") not in counts:
            raise PreflightSummaryError("preflight_finding_shape_invalid")
        severity = row["severity"]
        raw_code = row.get("code")
        code = raw_code if isinstance(raw_code, str) and raw_code in _CODES else "unclassified_finding"
        counts[severity] += 1
        category, owner, action = _action(code)
        group = grouped.setdefault((severity, code), {
            "severity": severity, "code": code, "count": 0, "units": [],
            "category": category, "owner": owner, "next_action": action,
        })
        group["count"] += 1
        if scope not in group["units"]:
            group["units"].append(scope)

    for row in report["host_findings"]:
        add("host", row)
    config_digests = {}
    unknown_units = 0
    for name, unit in units.items():
        if not isinstance(unit, Mapping) or not isinstance(unit.get("findings"), list):
            raise PreflightSummaryError("preflight_unit_shape_invalid")
        # No arbitrary unit names or per-job identifiers are exposed.
        scope = name if isinstance(name, str) and name in _PUBLIC_UNITS else "unrecognized_unit"
        for row in unit["findings"]:
            add(scope, row)
        digest = str(unit.get("configuration_sha256") or "")
        if _DIGEST.fullmatch(digest):
            config_digests[scope] = digest
        coverage = unit.get("coverage")
        if not isinstance(coverage, Mapping) or coverage.get("status") != "selected_route_contract":
            unknown_units += 1
    route = report.get("lane") if report.get("lane") in {"diagnostic", "beta_rigid"} else "unknown"
    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": timestamp,
        "observed_at_epoch": observed,
        "active_sha": report["active_sha"],
        "report_sha256": hashlib.sha256(raw_report_bytes).hexdigest(),
        "configuration_sha256": hashlib.sha256(_json_bytes(config_digests)).hexdigest(),
        "lane": route,
        "coverage": {"status": "selected_route_contract" if route == "beta_rigid" and not unknown_units else "unknown_or_partial",
                     "unit_count": len(units), "unknown_units": unknown_units},
        "counts": counts,
        "findings": sorted(grouped.values(), key=lambda row: (row["severity"], row["code"])),
        "proof_boundary": "selected_route_plumbing_only",
    }
    result["summary_sha256"] = hashlib.sha256(_json_bytes(result)).hexdigest()
    if len(_json_bytes(result)) > MAX_SUMMARY_BYTES:
        raise PreflightSummaryError("preflight_summary_too_large")
    return result


def write_safe_summary(path: Path, summary: Mapping[str, Any], *, private_parent: Path | None = None,
                       reader_ids: tuple[int, int] | None = None) -> None:
    """Publish only a schema-allowlisted summary, atomically; never chmod raw data."""
    if summary.get("schema_version") != SCHEMA_VERSION:
        raise PreflightSummaryError("preflight_summary_schema_invalid")
    data = _json_bytes(summary)
    if len(data) > MAX_SUMMARY_BYTES:
        raise PreflightSummaryError("preflight_summary_too_large")
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise PreflightSummaryError("preflight_summary_symlink_refused")
    if reader_ids is None:
        try:
            account = pwd.getpwnam("blueprint")
        except KeyError:
            raise PreflightSummaryError("preflight_summary_reader_identity_unknown") from None
        reader_ids = (account.pw_uid, account.pw_gid)
    if private_parent is not None:
        info = private_parent.stat()
        if private_parent != path.parent.parent or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o022:
            raise PreflightSummaryError("preflight_summary_private_parent_unsafe")
        # Execute only: no directory listing and no change to raw report modes.
        os.chmod(private_parent, 0o711)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
    info = path.parent.stat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o022:
        raise PreflightSummaryError("preflight_summary_parent_unsafe")
    # Only the dedicated public directory becomes listable. Reject any other
    # parent that the intended service account cannot traverse; file mode alone
    # never proves availability through a root-only ancestor.
    os.chmod(path.parent, 0o755)
    uid, gid = reader_ids
    for ancestor in path.parents:
        info = ancestor.stat()
        bit = 0o100 if info.st_uid == uid else (0o010 if info.st_gid == gid else 0o001)
        if not stat.S_IMODE(info.st_mode) & bit:
            raise PreflightSummaryError("preflight_summary_ancestor_not_traversable")
    fd, temporary = tempfile.mkstemp(prefix=".preflight-summary-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            os.fchmod(stream.fileno(), 0o644)
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        Path(temporary).unlink(missing_ok=True)
