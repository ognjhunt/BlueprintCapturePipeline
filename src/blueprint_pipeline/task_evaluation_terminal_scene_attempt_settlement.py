"""Settle every reservation row of a retired scene attempt once nothing can spend it.

A scene attempt reserves up to five rows: its own source row, one
scene-configuration row keyed by the preparation request digest, and three
controls rows. ``reserve_scene_attempt`` counts every non-cancelled row at its
full hold against the owner's spend cap and the paid-attempt count, so an
intent that retries fifteen times holds ~$150 while having spent a few dollars
(InteriorGS 840938, 2026-09-13 06:44Z: 31 rows, $148.38 held, cap exhausted).

The progression already proves an attempt terminal before minting a successor:
a sealed release transition or failure record plus a global ownership
reconciliation (provider zero, no leases, no pending teardowns). This module
turns that proof into a settlement receipt per row, honoured by
``validated_cancellation`` exactly like an unstarted-controls cancellation.
Dependent rows settle only when their launch is terminal or was never queued.

A settlement is not a refund (2026-09-13 audit). A stopped resource proves that
nothing can spend any more, not that nothing was spent: the retired source row
ran its preparation and a terminal scene-configuration launch may have paid for
image edits or a GPU. Each receipt therefore carries ``settled_spend``: the hold
the row keeps against the owner's cap (its full reservation while no billing
reconciliation is bound), and whether it still counts as an executed attempt.
Only rows proven never started -- a launch that was never queued, or a controls
row downstream of a launch that blocked before controls eligibility -- release
in full. ``retained_hold`` derives the same block for receipts sealed before the
field existed.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_retained_controls_evidence import DIRECTORY, _file, _read, validated_cancellation

SCHEMA = "task_evaluation_terminal_scene_attempt_settlement.v1"
STATUS = "settled_after_terminal_attempt"
RETIREMENT_SCHEMAS = {
    "task_evaluation_scene_release_transition.v1",
    "task_evaluation_scene_attempt_failure.v1",
}
OWNERSHIP_SCHEMA = "task_evaluation_scene_attempt_ownership.v1"
TERMINAL_LAUNCH_STATUSES = {"blocked", "completed", "failed"}
CONTROLS_PHASES = ("construction", "controls", "placement")
_ATTEMPT_FIELDS = ("attempt_id", "attempt_digest", "intent_digest", "provider", "maximum_spend_usd", "source_commit")
SETTLED_SPEND_BASES = {
    "retired_source_unreconciled",      # the source row executed its preparation: full hold, counts as an attempt
    "terminal_launch_unreconciled",     # a terminal launch may have paid: full hold until a reconciliation is bound
    "downstream_of_blocked_launch",     # controls rows behind a launch that blocked before controls eligibility
    "launch_never_queued",              # nothing ever started for this row
}


def _require(condition: Any, code: str) -> None:
    if not condition:
        raise ValueError("terminal_settlement_" + code)


def _sealed(path: Path, field: str) -> dict[str, Any]:
    value = _read(path)
    _require(value.get(field) == canonical_digest(value, digest_field=field), "record_seal_invalid")
    return value


def launch_id_for_preparation(preparation_id: str) -> str:
    from .task_evaluation_scene_configuration_activation_automation import _activation_id, _bounded_launch_id

    _require(preparation_id.endswith("-preparation"), "preparation_id_invalid")
    return _bounded_launch_id(_activation_id(preparation_id))


def dependent_row_ids(request_digest: str) -> dict[str, str]:
    _require(re.fullmatch(r"sha256:[0-9a-f]{64}", str(request_digest)) is not None, "request_digest_invalid")
    stem = "controls-" + request_digest[7:47]
    rows = {"scene-configuration-" + request_digest[7:31]: "scene_configuration"}
    rows.update({f"{stem}-{phase}": phase for phase in CONTROLS_PHASES})
    return rows


def _factory_binds_link(reference, retired, link):
    """Website IDs do not embed source attempt IDs; use their sealed factory."""
    if reference is None:
        return False
    from .task_evaluation_scene_configuration_submission_inputs import checked_file, read
    factory = read(checked_file(reference["path"], reference), digest_field="factory_digest")
    if factory.get("schema_version") != "website_scene_attempt_factory.v1":
        return False
    request_ref = factory["submission_request"]
    request = read(checked_file(request_ref["path"], request_ref))
    return (factory.get("attempt_digest") == retired["attempt_digest"]
            and factory.get("intent_digest") == link["intent_digest"]
            and factory.get("source_commit") == link["expected_production_commit"]
            and canonical_digest(request) == link["request_digest"]
            and request.get("preparation_id") == link["preparation_id"])


def validate_terminal_settlement(*, receipt: Mapping[str, Any], attempt: Mapping[str, Any]) -> None:
    """Fail closed unless the settlement still binds live, byte-identical evidence."""
    _require(
        receipt.get("schema_version") == SCHEMA
        and receipt.get("status") == STATUS
        and receipt.get("receipt_digest") == canonical_digest(receipt, digest_field="receipt_digest")
        and all(receipt.get(k) == attempt.get(k) for k in _ATTEMPT_FIELDS)
        and receipt.get("provider_mutation_performed") is False
        and receipt.get("downstream_execution_eligible") is False,
        "receipt_invalid",
    )
    retired = receipt.get("retired_attempt") or {}
    retirement_ref = receipt.get("retirement_record") or {}
    ownership_ref = receipt.get("ownership_reconciliation") or {}
    _require(_file(Path(str(retirement_ref.get("path") or ""))) == retirement_ref, "retirement_record_changed")
    _require(_file(Path(str(ownership_ref.get("path") or ""))) == ownership_ref, "ownership_record_changed")
    retirement = _sealed(Path(retirement_ref["path"]), "failure_digest")
    ownership = _sealed(Path(ownership_ref["path"]), "ownership_digest")
    _require(
        retirement.get("schema_version") in RETIREMENT_SCHEMAS
        and retirement.get("attempt_digest") == retired.get("attempt_digest")
        and ownership.get("schema_version") == OWNERSHIP_SCHEMA
        and ownership.get("status") == "closed_without_resource"
        and ownership.get("attempt_digest") == retired.get("attempt_digest")
        and ownership.get("active_writer_count") == 0
        and ownership.get("unresolved_create_count") == 0,
        "terminal_evidence_invalid",
    )
    if receipt.get("settled_spend") is not None:
        _require(receipt["settled_spend"] == retained_hold(receipt), "settled_spend_invalid")
    if attempt["attempt_id"] == retired.get("attempt_id"):
        _require(receipt.get("dependency") is None and receipt.get("execution_terminal") is None, "source_row_shape_invalid")
        return
    dependency = receipt.get("dependency") or {}
    link_ref = dependency.get("preparation_link") or {}
    _require(_file(Path(str(link_ref.get("path") or ""))) == link_ref, "preparation_link_changed")
    link = _sealed(Path(link_ref["path"]), "link_digest")
    rows = dependent_row_ids(str(link.get("request_digest")))
    _require(
        attempt["attempt_id"] in rows
        and (("-" + str(retired.get("attempt_id")) + "-") in str(link.get("preparation_id"))
             or _factory_binds_link(dependency.get("source_factory"), retired, link))
        and (rows[attempt["attempt_id"]] != "scene_configuration" or attempt.get("input_digest") == link["request_digest"]),
        "dependency_binding_invalid",
    )
    execution = receipt.get("execution_terminal") or {}
    launch_id = launch_id_for_preparation(str(link["preparation_id"]))
    _require(execution.get("launch_id") == launch_id, "execution_launch_id_invalid")
    launch_ref = execution.get("launch_receipt")
    if launch_ref is not None:
        _require(_file(Path(str(launch_ref.get("path") or ""))) == launch_ref, "launch_receipt_changed")
        launch = _read(Path(launch_ref["path"]))
        _require(
            launch.get("launch_id") == launch_id and launch.get("status") in TERMINAL_LAUNCH_STATUSES,
            "launch_not_terminal",
        )
        _require(execution.get("launch_status") in (None, launch.get("status")), "launch_status_changed")
    else:
        _require(execution.get("launch_never_queued") is True, "execution_evidence_missing")


def _derive_settled_spend(*, maximum_spend_usd: Any, dependency: Mapping[str, Any] | None,
                          execution: Mapping[str, Any] | None, phase: str | None,
                          launch_status: str | None) -> dict[str, Any]:
    cap = float(maximum_spend_usd)
    if dependency is None:
        return {"basis": "retired_source_unreconciled", "retained_spend_usd": cap, "counts_as_attempt": True}
    if (execution or {}).get("launch_never_queued") is True:
        return {"basis": "launch_never_queued", "retained_spend_usd": 0.0, "counts_as_attempt": False}
    if phase != "scene_configuration" and launch_status == "blocked":
        return {"basis": "downstream_of_blocked_launch", "retained_spend_usd": 0.0, "counts_as_attempt": False}
    return {"basis": "terminal_launch_unreconciled", "retained_spend_usd": cap, "counts_as_attempt": False}


def retained_hold(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """The hold a settled row keeps against the cap and whether it still counts as an attempt.

    Derived from the receipt's own evidence, so receipts sealed before
    ``settled_spend`` existed are accounted the same conservative way.
    """
    dependency = receipt.get("dependency")
    execution = receipt.get("execution_terminal") or {}
    phase = None
    launch_status = None
    if dependency is not None:
        phase = dependent_row_ids(str(dependency.get("request_digest"))).get(str(receipt.get("attempt_id")))
        launch_ref = execution.get("launch_receipt")
        if launch_ref is not None:
            launch_status = str(_read(Path(str(launch_ref["path"]))).get("status") or "")
    hold = _derive_settled_spend(maximum_spend_usd=receipt.get("maximum_spend_usd"), dependency=dependency,
                                 execution=execution, phase=phase, launch_status=launch_status)
    _require(hold["basis"] in SETTLED_SPEND_BASES and 0 <= hold["retained_spend_usd"] <= float(receipt.get("maximum_spend_usd")),
             "settled_spend_invalid")
    return hold


def budget_retained_hold(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Use bound terminal evidence without rewriting historical settlements.

    A retained CPU output keeps its allowance unless an explicit proof below
    applies. Legacy pre-allocation failures retain the API background allowance.
    Two initial admission refusals prove model work unentered; the native case
    still retains its entire provider allowance, not an estimated final bill.
    Callers must first validate the terminal settlement and ownership evidence.
    """
    hold = retained_hold(receipt)
    if hold["basis"] != "terminal_launch_unreconciled":
        return hold
    try:
        dependency = receipt["dependency"]
        if dependent_row_ids(dependency["request_digest"])[receipt["attempt_id"]] != "scene_configuration":
            return hold
        factory_ref = dependency.get("source_factory")
        if factory_ref is None:
            return hold
        from .task_evaluation_scene_configuration_submission_inputs import checked_file, read
        factory = read(checked_file(factory_ref["path"], factory_ref), digest_field="factory_digest")
        request_ref = factory["submission_request"]
        request = read(checked_file(request_ref["path"], request_ref))
        if (factory.get("schema_version") != "website_scene_attempt_factory.v1"
                or canonical_digest(request) != dependency["request_digest"]
                or request["expected_production_commit"] != receipt["source_commit"]
                or request["spend"]["hard_cap_usd"] != receipt["maximum_spend_usd"]):
            return hold
        launch_ref = receipt["execution_terminal"]["launch_receipt"]
        if _file(Path(launch_ref["path"])) != launch_ref:
            return hold
        launch = _read(Path(launch_ref["path"]))
        if launch.get("status") != "blocked" or launch.get("source_commit") != receipt["source_commit"]:
            return hold

        def terminal_artifact(ref):
            if ref.get("exists") is not True or _file(Path(ref["path"]))["digest"] != ref["digest"]:
                raise ValueError("preallocation_artifact_changed")
            return _read(Path(ref["path"]))

        terminal = launch["terminal_evidence"]
        result = terminal_artifact(terminal["result"])
        teardown = terminal_artifact(terminal["artifacts"]["teardown_manifest_path"])
        from .task_evaluation_unentered_authoring_budget import (
            authoring_never_entered, pretraining_never_entered, prestage_before_first_stage,
        )
        if (result.get("schema_version") == "task_evaluation_scene_configuration_vast_result.v1"
                and result.get("run_id") == request["run_id"]
                and result.get("source_commit") == receipt["source_commit"]
                and result.get("status") == "blocked"
                and result.get("continuing_spend_from_this_run") is False
                and type(result.get("provider_mutations_performed")) is int
                and result.get("provider_mutations_performed") == 1
                and teardown.get("schema_version") == "vast_teardown_manifest.v1"
                and teardown.get("status") == "completed"
                and teardown.get("continuing_spend_from_this_run") is False
                and authoring_never_entered(result, request)):
            # Keep the entire native allowance; partial posted billing or a
            # runtime estimate cannot establish a final provider charge.
            bound = float(request["spend"]["provider_compute_spend_cap_usd"])
            if 0 <= bound <= hold["retained_spend_usd"]:
                return {"basis": "native_allowance_with_unentered_authoring", "retained_spend_usd": bound,
                        "counts_as_attempt": hold["counts_as_attempt"]}
        if (result.get("schema_version") != "task_evaluation_scene_configuration_vast_result.v1"
                or result.get("run_id") != request["run_id"]
                or result.get("source_commit") != receipt["source_commit"]
                or result.get("status") != "blocked"
                or type(result.get("provider_mutations_performed")) is not int
                or result["provider_mutations_performed"] != 0
                or result.get("continuing_spend_from_this_run") is not False
                or teardown.get("schema_version") != "vast_teardown_manifest.v1"
                or teardown.get("status") not in {"not_required_provider_adapter_never_invoked",
                                                  "not_required_prelaunch_inventory_guard_blocked"}
                or teardown.get("vast_instance_ids") != []
                or teardown.get("continuing_spend_from_this_run") is not False):
            return hold
        if result.get("provider_runtime_output_zip_path") is not None:
            if prestage_before_first_stage(result, request):
                return {"basis": "preallocation_unentered_authoring", "retained_spend_usd": 0.0,
                        "counts_as_attempt": hold["counts_as_attempt"]}
            from .task_evaluation_authoring_auth_recovery import initial_authentication_failure
            rejected = initial_authentication_failure(result)
            if rejected and rejected["retained_spend_usd"] <= hold["retained_spend_usd"]:
                return {"basis": "rejected_initial_authoring_request_upper_bound",
                        "retained_spend_usd": rejected["retained_spend_usd"],
                        "counts_as_attempt": hold["counts_as_attempt"]}
            if not authoring_never_entered(result, request):
                return hold
            return {"basis": "preallocation_unentered_authoring", "retained_spend_usd": 0.0,
                    "counts_as_attempt": hold["counts_as_attempt"]}
        caps = request["spend"]["external_service_caps"]["openai"]["stage_max_cost_usd"]
        bound = round(float(caps["artifixer_semantic_teacher"]) + float(caps["artifixer_visual_review"]), 6)
        if (result.get("result_digest") == canonical_digest(result, digest_field="result_digest")
                and pretraining_never_entered(result)):
            bound = 0.0
        if not 0 <= bound <= hold["retained_spend_usd"]:
            return hold
        return {"basis": "preallocation_api_budget_upper_bound", "retained_spend_usd": bound,
                "counts_as_attempt": hold["counts_as_attempt"]}
    except (OSError, ValueError, KeyError, TypeError):
        return hold


def _launch_state(*, launch_id: str, launch_execution_root: Path, launch_queue_root: Path) -> dict[str, Any] | None:
    """Terminal evidence for a dependent row, or None while a launch may still spend."""
    receipt_path = launch_execution_root / launch_id / "launch_receipt.json"
    if receipt_path.is_file():
        launch = _read(receipt_path)
        if launch.get("launch_id") != launch_id or launch.get("status") not in TERMINAL_LAUNCH_STATUSES:
            return None
        return {"launch_id": launch_id, "launch_receipt": _file(receipt_path), "launch_never_queued": False,
                "launch_status": str(launch.get("status"))}
    if (launch_execution_root / launch_id).exists():
        return None  # A launch directory without a terminal receipt is still running or unreconciled.
    for state in ("pending", "processing"):
        if any(path.name.startswith(launch_id) for path in (launch_queue_root / state).glob("*.json")):
            return None
    return {"launch_id": launch_id, "launch_receipt": None, "launch_never_queued": True}


def settle_retired_attempt_rows(*, directory: Path, retired_attempt: Mapping[str, Any],
        retirement_record: Mapping[str, Any], ownership_record: Mapping[str, Any],
        launch_execution_root: Path, launch_queue_root: Path, dry_run: bool = False,
        source_factory: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Write settlement receipts for the retired attempt's rows; idempotent and lock-free by design."""
    from . import task_evaluation_scene_intake as intake

    directory = Path(directory)
    retirement_ref = _file(Path(str(retirement_record["path"])))
    ownership_ref = _file(Path(str(ownership_record["path"])))
    retired = {"attempt_id": retired_attempt["attempt_id"], "attempt_digest": retired_attempt["attempt_digest"]}
    settled, skipped = [], []

    def settle(attempt: Mapping[str, Any], *, dependency: dict[str, Any] | None, execution: dict[str, Any] | None,
               phase: str | None = None) -> None:
        existing = validated_cancellation(directory, attempt)
        if existing is not None:
            settled.append({"attempt_id": attempt["attempt_id"], "status": "already_released", "schema_version": existing.get("schema_version")})
            return
        receipt = {
            "schema_version": SCHEMA, "status": STATUS,
            **{k: attempt[k] for k in _ATTEMPT_FIELDS},
            "retired_attempt": retired, "retirement_record": retirement_ref,
            "ownership_reconciliation": ownership_ref, "dependency": dependency,
            "execution_terminal": execution, "downstream_execution_eligible": False,
            "provider_mutation_performed": False,
            "settled_spend": _derive_settled_spend(
                maximum_spend_usd=attempt["maximum_spend_usd"], dependency=dependency, execution=execution,
                phase=phase, launch_status=(execution or {}).get("launch_status")),
        }
        receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
        validate_terminal_settlement(receipt=receipt, attempt=attempt)
        if not dry_run:
            target = directory / DIRECTORY / (attempt["attempt_id"] + ".json")
            target.parent.mkdir(mode=0o750, exist_ok=True)
            intake.write_exclusive(target, receipt)
        settled.append({"attempt_id": attempt["attempt_id"], "status": "settled"})

    from .task_evaluation_scene_preparation_attempts import preparation_attempt_path, SCHEMA as PREPARATION_SCHEMA
    source_path = preparation_attempt_path(directory, retired["attempt_id"])
    source_row = intake._read(source_path, "attempt_digest")
    _require(source_row["attempt_digest"] == retired["attempt_digest"], "retired_attempt_digest_mismatch")
    if source_row.get("schema_version") == PREPARATION_SCHEMA:
        _require(source_row.get("maximum_spend_usd") == 0 and source_row.get("paid_authority_granted") is False,
                 "preparation_has_paid_authority")
    else:
        settle(source_row, dependency=None, execution=None, phase=None)
    marker = "-" + retired["attempt_id"] + "-"
    for link_path in sorted((directory / "preparations").glob("*.json")):
        if link_path.name.endswith(".activation.json"):
            continue
        link = _read(link_path)
        if link.get("schema_version") != "task_evaluation_scene_preparation_link.v1":
            continue
        factory_bound = _factory_binds_link(source_factory, retired, link)
        if marker not in str(link.get("preparation_id", "")) and not factory_bound:
            continue
        if link.get("link_digest") != canonical_digest(link, digest_field="link_digest"):
            skipped.append({"preparation_link": link_path.name, "reason": "link_seal_invalid"})
            continue
        launch_id = launch_id_for_preparation(str(link["preparation_id"]))
        execution = _launch_state(launch_id=launch_id, launch_execution_root=Path(launch_execution_root),
                                  launch_queue_root=Path(launch_queue_root))
        if execution is None:
            skipped.append({"preparation_link": link_path.name, "reason": "launch_not_terminal", "launch_id": launch_id})
            continue
        dependency = {"preparation_link": _file(link_path), "request_digest": link["request_digest"]}
        if factory_bound:
            dependency["source_factory"] = dict(source_factory)
        for row_id, phase in dependent_row_ids(str(link["request_digest"])).items():
            row_path = directory / "attempts" / (row_id + ".json")
            if not row_path.is_file():
                continue
            settle(intake._read(row_path, "attempt_digest"), dependency=dependency, execution=execution, phase=phase)
    return {"status": "would_settle" if dry_run else "settled", "retired_attempt_id": retired["attempt_id"],
            "rows": settled, "skipped": skipped, "provider_mutation_performed": False}


def sweep_retired_attempts(*, directory: Path, state: Mapping[str, Any], config: Mapping[str, Any],
                           dry_run: bool = False) -> dict[str, Any]:
    """Settle rows for every predecessor the progression already retired; never raises."""
    directory = Path(directory)
    summary: dict[str, Any] = {"settled_rows": 0, "already_released_rows": 0, "skipped": []}
    entries = []
    for lineage in state.get("release_predecessors") or []:
        entries.append((lineage.get("attempt"), lineage.get("reconciliation"), lineage.get("factory"))
                       if isinstance(lineage, Mapping) else (None, None, None))
    for lineage in state.get("recovery_predecessors") or []:
        entries.append((lineage.get("attempt"), lineage.get("evidence"), None)
                       if isinstance(lineage, Mapping) else (None, None, None))
    for attempt_ref, evidence, factory_ref in entries:
        try:
            if not isinstance(attempt_ref, Mapping) or not isinstance(evidence, Mapping):
                raise ValueError("terminal_settlement_lineage_shape_invalid")
            from . import task_evaluation_scene_intake as intake
            retired = intake._read(Path(str(attempt_ref["path"])), "attempt_digest")
            outcome = settle_retired_attempt_rows(
                directory=directory, retired_attempt=retired,
                retirement_record=evidence["failure"], ownership_record=evidence["ownership_reconciliation"],
                launch_execution_root=Path(config["launch_execution_root"]),
                launch_queue_root=Path(config["launch_queue_root"]), dry_run=dry_run, source_factory=factory_ref)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            summary["skipped"].append({"attempt": str((attempt_ref or {}).get("path", ""))[-60:], "reason": str(exc)[:120]})
            continue
        summary["settled_rows"] += sum(1 for row in outcome["rows"] if row["status"] == "settled")
        summary["already_released_rows"] += sum(1 for row in outcome["rows"] if row["status"] == "already_released")
        summary["skipped"].extend(outcome["skipped"])
    summary["skipped"] = sorted({json.dumps(row, sort_keys=True) for row in summary["skipped"]})
    summary["skipped"] = [json.loads(row) for row in summary["skipped"]]
    summary["summary_digest"] = "sha256:" + hashlib.sha256(json.dumps(summary, sort_keys=True).encode()).hexdigest()
    return summary
