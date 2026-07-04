"""Layered, fail-closed success-claim contracts.

Every place the pipeline can claim "success" sits on a different evidence layer:

1. media_validity            — the review media itself is decodable, fresh, and non-degenerate.
2. review_task_success       — a reviewer (VLM/human) judged the review media as task success.
3. task_success_contract     — the task-specific deterministic contract passed in the trace.
4. simulator_execution       — the simulator/runtime actually executed this run's episode.
5. policy_action_execution   — the executed actions came from the claimed policy, not a script.
6. contact_state_change      — measured contact / object state change matching the task metadata.
7. physical_readiness        — real-robot / deployment evidence. Never derivable from 1–6.

Each layer fails closed: missing, stale, or non-boolean evidence produces a FAIL with
blockers, never a silent PASS. A composed ledger reports the highest truthful claim so a
higher layer can never be asserted while a lower layer is unproven.

Requirements are derived generically from task contract metadata (affordance/target ids,
declared success state change), never from task-id string matching.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from .common import utc_now_iso

LEDGER_SCHEMA_VERSION = "success_claim_ledger.v1"
MEDIA_VALIDITY_SCHEMA_VERSION = "media_validity_contract.v1"
REVIEW_TASK_SUCCESS_SCHEMA_VERSION = "review_task_success_contract.v1"
TASK_SUCCESS_CONTRACT_SCHEMA_VERSION = "task_success_contract_result.v1"
SIMULATOR_EXECUTION_SCHEMA_VERSION = "simulator_execution_contract.v1"
POLICY_ACTION_EXECUTION_SCHEMA_VERSION = "policy_action_execution_contract.v1"
CONTACT_STATE_CHANGE_SCHEMA_VERSION = "contact_state_change_proof.v1"
PHYSICAL_READINESS_SCHEMA_VERSION = "physical_readiness_contract.v1"
ARTIFACT_FRESHNESS_SCHEMA_VERSION = "artifact_freshness_evidence.v1"
TASK_PROOF_REQUIREMENTS_SCHEMA_VERSION = "task_proof_requirements.v1"

# Ordered from weakest to strongest truthful claim.
CLAIM_LADDER: tuple[str, ...] = (
    "no_claim",
    "media_valid",
    "review_task_success",
    "simulator_task_success",
    "policy_task_success",
    "contact_state_change_proven",
    "physical_deployment_ready",
)

# Action sources that count as policy execution. Everything else (scripted trajectories,
# kinematic teleports, camera paths) is runtime support, not policy proof.
POLICY_ACTION_SOURCES = frozenset({"learned_policy", "policy_endpoint", "vla_policy"})

_TERMINAL_FAIL_RUNTIME_STATUSES = frozenset(
    {"blocked", "failed", "error", "timeout", "timed_out", "cancelled"}
)


def coerce_strict_success(value: Any) -> bool | None:
    """Return True/False only for real booleans (or 0/1 ints); anything else is None.

    Strings like "1"/"true", None, floats, and empty values must never be read as a
    success verdict — a mistyped JSON field is not evidence.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    return None


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _dedupe_blockers(blockers: Sequence[str]) -> list[str]:
    return sorted({str(b).strip() for b in blockers if str(b).strip()})


def _parse_iso(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _passed(contract: Mapping[str, Any] | None) -> bool:
    if not isinstance(contract, Mapping):
        return False
    return str(contract.get("status") or "").strip().upper() == "PASS"


def _result(
    schema_version: str,
    *,
    passed: bool,
    blockers: Sequence[str],
    claim_boundary: str,
    **fields: Any,
) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "generated_at": utc_now_iso(),
        "status": "PASS" if passed else "FAIL",
        "blockers": _dedupe_blockers(blockers),
        "claim_boundary": claim_boundary,
        **fields,
    }


# ---------------------------------------------------------------------------
# Artifact freshness — stale outputs must never read as current truth.
# ---------------------------------------------------------------------------


def build_artifact_freshness_evidence(
    *,
    artifact_run_id: str | None = None,
    current_run_id: str | None = None,
    artifact_generated_at: str | None = None,
    run_started_at: str | None = None,
    artifact_mtime_epoch: float | None = None,
    run_started_epoch: float | None = None,
) -> dict[str, Any]:
    """Fresh only when at least one positive signal ties the artifact to this run.

    No freshness signal at all is a blocker, not a pass — a pre-existing file from an
    earlier run is indistinguishable from current output without one.
    """
    blockers: list[str] = []
    signals: list[str] = []

    if artifact_run_id and current_run_id:
        if str(artifact_run_id).strip() == str(current_run_id).strip():
            signals.append("run_id_match")
        else:
            blockers.append(
                f"stale_artifact_run_id_mismatch:{artifact_run_id}!={current_run_id}"
            )

    artifact_dt = _parse_iso(artifact_generated_at)
    run_dt = _parse_iso(run_started_at)
    if artifact_dt is not None and run_dt is not None:
        if artifact_dt >= run_dt:
            signals.append("generated_at_after_run_start")
        else:
            blockers.append("stale_artifact_generated_before_run_start")

    if artifact_mtime_epoch is not None and run_started_epoch is not None:
        if float(artifact_mtime_epoch) >= float(run_started_epoch):
            signals.append("mtime_after_run_start")
        else:
            blockers.append("stale_artifact_mtime_before_run_start")

    if not signals and not blockers:
        blockers.append("artifact_freshness_evidence_missing")

    return _result(
        ARTIFACT_FRESHNESS_SCHEMA_VERSION,
        passed=bool(signals) and not blockers,
        blockers=blockers,
        claim_boundary=(
            "Freshness ties an artifact to the current run. It says nothing about the "
            "artifact's content or any success claim."
        ),
        fresh=bool(signals) and not blockers,
        freshness_signals=signals,
    )


# ---------------------------------------------------------------------------
# Layer 1 — media validity.
# ---------------------------------------------------------------------------


def build_media_validity(
    *,
    media_present: bool,
    frame_count: int | None = None,
    min_frames: int = 1,
    decodable: bool | None = None,
    visual_stats: Mapping[str, Any] | None = None,
    freshness: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Layer 1: the review media exists, is decodable, fresh, and non-degenerate.

    This is never task success — a perfectly valid video of a failed task passes.
    """
    blockers: list[str] = []
    if not media_present:
        blockers.append("media_missing")
    if frame_count is not None and int(frame_count) < int(min_frames):
        blockers.append(f"media_frame_count_below_minimum:{frame_count}<{min_frames}")
    if decodable is False:
        blockers.append("media_not_decodable")
    if decodable is None and media_present:
        blockers.append("media_decodability_unverified")

    stats = _mapping(visual_stats)
    if stats:
        degenerate = coerce_strict_success(stats.get("degenerate"))
        if degenerate is True:
            blockers.append("media_visually_degenerate")
        for blocker in _string_list(stats.get("blockers")):
            blockers.append(f"media_visual_stats:{blocker}")

    if freshness is not None:
        if not _passed(freshness):
            blockers.append("media_artifact_not_proven_fresh")
            blockers.extend(_string_list(_mapping(freshness).get("blockers")))
    elif media_present:
        blockers.append("media_freshness_evidence_missing")

    return _result(
        MEDIA_VALIDITY_SCHEMA_VERSION,
        passed=not blockers,
        blockers=blockers,
        claim_boundary=(
            "Media validity proves the review media is real, fresh, and reviewable. It is "
            "not task success, review success, simulator proof, or real-world proof."
        ),
        media_valid=not blockers,
        frame_count=frame_count,
        freshness=_mapping(freshness) if freshness is not None else None,
    )


# ---------------------------------------------------------------------------
# Layer 2 — review task success.
# ---------------------------------------------------------------------------


def build_review_task_success(
    *,
    media_validity: Mapping[str, Any] | None,
    reviewer_verdicts: Sequence[Mapping[str, Any]] | None,
    camera_evidence: Mapping[str, Any] | None = None,
    require_embodied_robot_action: bool = True,
) -> dict[str, Any]:
    """Layer 2: a reviewer judged the (valid) review media as showing task success.

    Camera-only motion and root-follow footage are excluded; each verdict must be a
    strict boolean. Review success on generated or rendered media is never real-world
    or simulator-state proof.
    """
    blockers: list[str] = []
    if not _passed(media_validity):
        blockers.append("media_validity_not_passed")
        blockers.extend(_string_list(_mapping(media_validity).get("blockers")))

    verdicts = [v for v in (reviewer_verdicts or []) if isinstance(v, Mapping)]
    if not verdicts:
        blockers.append("reviewer_verdict_missing")
    successes: list[bool] = []
    for index, verdict in enumerate(verdicts):
        value = coerce_strict_success(verdict.get("success"))
        if value is None:
            blockers.append(f"reviewer_verdict_not_strict_boolean:index_{index}")
        else:
            successes.append(value)
        if not str(verdict.get("reviewer") or verdict.get("source") or "").strip():
            blockers.append(f"reviewer_identity_missing:index_{index}")
    if successes and not all(successes):
        blockers.append("reviewer_judged_task_failure")

    camera = _mapping(camera_evidence)
    camera_mode = str(camera.get("robot_pov_camera_mode") or "").strip()
    if camera_mode == "root_follow":
        blockers.append("camera_motion_is_not_robot_task_evidence:root_follow")
    if require_embodied_robot_action:
        embodied = coerce_strict_success(
            camera.get("visible_embodied_robot_action_evidence")
        )
        if embodied is not True:
            blockers.append("visible_embodied_robot_action_not_proven")

    return _result(
        REVIEW_TASK_SUCCESS_SCHEMA_VERSION,
        passed=not blockers,
        blockers=blockers,
        claim_boundary=(
            "Review task success is a reviewer's judgment of valid review media. It is not "
            "simulator state truth, contact proof, real-world outcome, or deployment "
            "readiness; on generated media it is a judgment of generated pixels only."
        ),
        review_task_success=not blockers,
        reviewer_verdict_count=len(verdicts),
        real_world_proof=False,
    )


# ---------------------------------------------------------------------------
# Layer 3 — task success contract (deterministic trace contract).
# ---------------------------------------------------------------------------


def derive_task_proof_requirements(task_metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """Derive generic proof requirements from task contract metadata.

    - Any declared affordance/target ids ⇒ reach-to-affordance evidence is required.
    - A declared ``success_state_change`` mapping (object + property, e.g. door open)
      ⇒ contact/state-change proof is required before the task itself can be claimed.
    No task-id or task-name matching happens here.
    """
    metadata = _mapping(task_metadata)
    contract_name = (
        str(
            metadata.get("task_success_contract")
            or metadata.get("success_contract")
            or ""
        )
        .strip()
        .lower()
    )
    affordances = _string_list(metadata.get("affordance_object_ids")) or _string_list(
        metadata.get("target_object_ids")
    )
    state_change = _mapping(metadata.get("success_state_change"))
    return {
        "schema_version": TASK_PROOF_REQUIREMENTS_SCHEMA_VERSION,
        "task_success_contract": contract_name or None,
        "requires_reach_to_affordance": bool(affordances),
        "affordance_object_ids": affordances,
        "requires_contact_or_state_change": bool(state_change),
        "success_state_change": state_change or None,
    }


def build_task_success_contract_result(
    *,
    task_metadata: Mapping[str, Any] | None,
    trace_task_success: Any,
    contract_evidence: Mapping[str, Any] | None = None,
    reach_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Layer 3: the task's declared deterministic contract passed for this trace.

    A missing contract name, a non-boolean trace verdict, or missing required reach
    evidence all fail closed.
    """
    requirements = derive_task_proof_requirements(task_metadata)
    blockers: list[str] = []

    if not requirements["task_success_contract"]:
        blockers.append("task_success_contract_missing_from_task_metadata")

    trace_verdict = coerce_strict_success(trace_task_success)
    if trace_verdict is None:
        blockers.append("trace_task_success_not_strict_boolean")
    elif trace_verdict is False:
        blockers.append("trace_task_success_false")

    if contract_evidence is not None and not _passed(contract_evidence):
        blockers.append("task_contract_evidence_not_passed")
        blockers.extend(_string_list(_mapping(contract_evidence).get("blockers")))

    if requirements["requires_reach_to_affordance"]:
        if not _passed(reach_evidence):
            blockers.append("reach_to_affordance_evidence_not_passed")
            if reach_evidence is None:
                blockers.append("visible_arm_presence_is_not_reach_evidence")
            else:
                blockers.extend(_string_list(_mapping(reach_evidence).get("blockers")))

    return _result(
        TASK_SUCCESS_CONTRACT_SCHEMA_VERSION,
        passed=not blockers,
        blockers=blockers,
        claim_boundary=(
            "The task success contract is the task's deterministic trace criterion. It does "
            "not prove contact, object state change, learned-policy quality, real-world "
            "outcome, or deployment readiness."
        ),
        task_success_contract_passed=not blockers,
        proof_requirements=requirements,
        trace_task_success=trace_verdict,
    )


# ---------------------------------------------------------------------------
# Layer 4 — simulator / runtime execution.
# ---------------------------------------------------------------------------


def build_simulator_execution(
    *,
    provider_runtime_status: str | None,
    output_artifacts_present: bool,
    artifact_freshness: Mapping[str, Any] | None = None,
    frames_rendered: int | None = None,
    execution_log_present: bool | None = None,
) -> dict[str, Any]:
    """Layer 4: the simulator/runtime executed *this run's* episode and produced output.

    Provider runtime success (pod ran, zip arrived, exit 0) is recorded separately and is
    never sufficient on its own: fresh output artifacts are required too.
    """
    blockers: list[str] = []
    status = str(provider_runtime_status or "").strip().lower()
    provider_runtime_operational = bool(status) and status not in _TERMINAL_FAIL_RUNTIME_STATUSES
    if not status:
        blockers.append("provider_runtime_status_missing")
    elif status in _TERMINAL_FAIL_RUNTIME_STATUSES:
        blockers.append(f"provider_runtime_status_terminal:{status}")

    if not output_artifacts_present:
        blockers.append("simulator_output_artifacts_missing")
    if artifact_freshness is None:
        blockers.append("simulator_artifact_freshness_evidence_missing")
    elif not _passed(artifact_freshness):
        blockers.append("simulator_output_artifacts_not_proven_fresh")
        blockers.extend(_string_list(_mapping(artifact_freshness).get("blockers")))
    if frames_rendered is not None and int(frames_rendered) <= 0:
        blockers.append("simulator_rendered_zero_frames")
    if execution_log_present is False:
        blockers.append("simulator_execution_log_missing")

    return _result(
        SIMULATOR_EXECUTION_SCHEMA_VERSION,
        passed=not blockers,
        blockers=blockers,
        claim_boundary=(
            "Simulator execution proves this run's episode actually ran and produced fresh "
            "output. Provider runtime success alone is infrastructure health, not task "
            "success, and neither is this layer."
        ),
        simulator_execution_proven=not blockers,
        provider_runtime_operational=provider_runtime_operational,
        provider_runtime_success_is_not_task_success=True,
    )


# ---------------------------------------------------------------------------
# Layer 5 — policy / action execution.
# ---------------------------------------------------------------------------


def build_policy_action_execution(
    *,
    action_source: str | None,
    policy_id: str | None = None,
    action_trace_present: bool = False,
    actions_executed_in_simulator: Any = None,
) -> dict[str, Any]:
    """Layer 5: executed actions came from the claimed policy, applied in the simulator.

    Scripted trajectories, kinematic teleports, and camera paths are recorded as their
    own source and never count as policy execution.
    """
    blockers: list[str] = []
    source = str(action_source or "").strip().lower()
    if not source:
        blockers.append("action_source_missing")
    elif source not in POLICY_ACTION_SOURCES:
        blockers.append(f"action_source_not_policy:{source}")
    if source in POLICY_ACTION_SOURCES and not str(policy_id or "").strip():
        blockers.append("policy_id_missing")
    if not action_trace_present:
        blockers.append("action_trace_missing")
    executed = coerce_strict_success(actions_executed_in_simulator)
    if executed is not True:
        blockers.append("actions_not_proven_executed_in_simulator")

    return _result(
        POLICY_ACTION_EXECUTION_SCHEMA_VERSION,
        passed=not blockers,
        blockers=blockers,
        claim_boundary=(
            "Policy action execution proves the claimed policy's actions were applied in "
            "the simulator. It does not prove task success, contact, state change, or "
            "real-world behavior."
        ),
        policy_action_execution_proven=not blockers,
        action_source=source or None,
        policy_id=str(policy_id or "").strip() or None,
    )


# ---------------------------------------------------------------------------
# Layer 6 — contact / state-change proof.
# ---------------------------------------------------------------------------


def build_contact_state_change_proof(
    *,
    proof_requirements: Mapping[str, Any] | None,
    contact_reports: Sequence[Mapping[str, Any]] | None = None,
    state_change_measurement: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Layer 6: measured contact and/or object state change matching the task metadata.

    Mask overlap, proximity, arm visibility, and reviewer judgments are not accepted
    here — only physics contact reports and before/after state measurements of the
    declared state variable. If the task declares no state change, this layer reports
    not_required and passes vacuously without asserting proof.
    """
    requirements = _mapping(proof_requirements)
    required = bool(requirements.get("requires_contact_or_state_change"))
    declared_change = _mapping(requirements.get("success_state_change"))
    blockers: list[str] = []

    contacts = [c for c in (contact_reports or []) if isinstance(c, Mapping)]
    measured_contact = False
    for report in contacts:
        if coerce_strict_success(report.get("physics_contact_measured")) is True:
            measured_contact = True
        for label in (
            "mask_overlap_only",
            "proximity_only",
            "visual_only",
        ):
            if coerce_strict_success(report.get(label)) is True:
                blockers.append(f"contact_report_rejected_{label}_is_not_contact_proof")

    state_change = _mapping(state_change_measurement)
    measured_state_change = False
    if state_change:
        before = state_change.get("before")
        after = state_change.get("after")
        target_property = str(state_change.get("property") or "").strip()
        if before is None or after is None or not target_property:
            blockers.append("state_change_measurement_incomplete")
        elif before == after:
            blockers.append("state_change_not_observed")
        else:
            measured_state_change = True
            if declared_change and target_property != str(
                declared_change.get("property") or ""
            ).strip():
                blockers.append(
                    f"state_change_property_mismatch:{target_property}!="
                    f"{declared_change.get('property')}"
                )
                measured_state_change = False

    if required and not measured_contact and not measured_state_change:
        blockers.append("contact_or_state_change_proof_missing")
    # When the task declares a state change, that measurement is the success condition;
    # contact alone is only supporting evidence. Without a declared change, either
    # measured contact or a measured state change satisfies the layer.
    if required and declared_change:
        proven = measured_state_change and not blockers
    elif required:
        proven = (measured_contact or measured_state_change) and not blockers
    else:
        proven = False

    passed = (not required and not blockers) or (required and proven)
    return _result(
        CONTACT_STATE_CHANGE_SCHEMA_VERSION,
        passed=passed,
        blockers=blockers,
        claim_boundary=(
            "Contact/state-change proof requires measured physics contact or a measured "
            "before/after change of the task's declared state variable. Simulator state "
            "change is still simulation evidence, not real-world proof."
        ),
        required=required,
        contact_state_change_proven=bool(required and proven),
        measured_physics_contact=measured_contact,
        measured_state_change=measured_state_change,
    )


# ---------------------------------------------------------------------------
# Layer 7 — physical / deployment readiness.
# ---------------------------------------------------------------------------


def build_physical_readiness(
    *,
    real_robot_execution_evidence: Mapping[str, Any] | None = None,
    deployment_approval: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Layer 7: real-robot execution plus explicit deployment approval.

    No combination of simulator, WAM, generated-media, or review evidence can pass this
    layer. Absent real-robot evidence it always fails closed.
    """
    blockers: list[str] = []
    real_evidence = _mapping(real_robot_execution_evidence)
    executed = coerce_strict_success(real_evidence.get("physical_robot_executed"))
    if executed is not True:
        blockers.append("physical_robot_execution_evidence_missing")
    if executed is True and not str(real_evidence.get("run_manifest_uri") or "").strip():
        blockers.append("physical_run_manifest_missing")

    approval = _mapping(deployment_approval)
    approved = coerce_strict_success(approval.get("approved"))
    if approved is not True:
        blockers.append("deployment_approval_missing")
    elif not str(approval.get("approver") or "").strip():
        blockers.append("deployment_approver_missing")

    return _result(
        PHYSICAL_READINESS_SCHEMA_VERSION,
        passed=not blockers,
        blockers=blockers,
        claim_boundary=(
            "Physical/deployment readiness requires real-robot execution evidence and an "
            "explicit named approval. Simulator, WAM, generated-video, and review results "
            "can never satisfy this layer."
        ),
        physical_deployment_ready=not blockers,
        simulation_evidence_cannot_upgrade_this_layer=True,
    )


# ---------------------------------------------------------------------------
# Composed ledger.
# ---------------------------------------------------------------------------


def build_success_claim_ledger(
    *,
    task_metadata: Mapping[str, Any] | None,
    media_validity: Mapping[str, Any] | None = None,
    review_task_success: Mapping[str, Any] | None = None,
    task_success_contract: Mapping[str, Any] | None = None,
    simulator_execution: Mapping[str, Any] | None = None,
    policy_action_execution: Mapping[str, Any] | None = None,
    contact_state_change: Mapping[str, Any] | None = None,
    physical_readiness: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compose the layers into one ledger with the highest truthful claim.

    Missing layers count as failed. Higher claims require every supporting lower layer:
    - media_valid                ⇐ media_validity
    - review_task_success        ⇐ media_validity + review contract
    - simulator_task_success     ⇐ simulator execution + task success contract
    - policy_task_success        ⇐ simulator_task_success + policy action execution
    - contact_state_change_proven⇐ simulator_task_success + contact/state-change proof
    - physical_deployment_ready  ⇐ physical readiness contract (independent, real-world)
    """
    requirements = derive_task_proof_requirements(task_metadata)
    layers = {
        "media_validity": _mapping(media_validity) or None,
        "review_task_success": _mapping(review_task_success) or None,
        "task_success_contract": _mapping(task_success_contract) or None,
        "simulator_execution": _mapping(simulator_execution) or None,
        "policy_action_execution": _mapping(policy_action_execution) or None,
        "contact_state_change": _mapping(contact_state_change) or None,
        "physical_readiness": _mapping(physical_readiness) or None,
    }
    layer_passed = {name: _passed(contract) for name, contract in layers.items()}

    media_valid = layer_passed["media_validity"]
    review_ok = media_valid and layer_passed["review_task_success"]
    simulator_task_ok = (
        layer_passed["simulator_execution"] and layer_passed["task_success_contract"]
    )
    policy_task_ok = simulator_task_ok and layer_passed["policy_action_execution"]
    contact_required = bool(requirements.get("requires_contact_or_state_change"))
    contact_ok = simulator_task_ok and layer_passed["contact_state_change"] and (
        bool(_mapping(layers["contact_state_change"]).get("contact_state_change_proven"))
        or not contact_required
    )
    contact_proven = simulator_task_ok and bool(
        _mapping(layers["contact_state_change"]).get("contact_state_change_proven")
    )
    physical_ok = layer_passed["physical_readiness"]

    highest = "no_claim"
    if media_valid:
        highest = "media_valid"
    if review_ok:
        highest = "review_task_success"
    if simulator_task_ok and (not contact_required or contact_ok):
        highest = "simulator_task_success"
    if policy_task_ok and (not contact_required or contact_ok):
        highest = "policy_task_success"
    if contact_proven and policy_task_ok:
        highest = "contact_state_change_proven"
    if physical_ok:
        highest = "physical_deployment_ready"

    blockers: list[str] = []
    for name, contract in layers.items():
        if contract is None:
            blockers.append(f"{name}_contract_missing")
        elif not layer_passed[name]:
            blockers.extend(f"{name}:{b}" for b in _string_list(contract.get("blockers")))
    if contact_required and not contact_proven:
        blockers.append(
            "task_declares_state_change_but_contact_state_change_not_proven"
        )

    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "highest_truthful_claim": highest,
        "claim_ladder": list(CLAIM_LADDER),
        "proof_requirements": requirements,
        "layers": {
            name: {
                "present": contract is not None,
                "passed": layer_passed[name],
                "blockers": _string_list((contract or {}).get("blockers")),
            }
            for name, contract in layers.items()
        },
        "claims": {
            "media_valid": media_valid,
            "review_task_success": review_ok,
            "simulator_task_success": simulator_task_ok
            and (not contact_required or contact_ok),
            "policy_task_success": policy_task_ok
            and (not contact_required or contact_ok),
            "contact_state_change_proven": contact_proven,
            "physical_deployment_ready": physical_ok,
        },
        "blockers": _dedupe_blockers(blockers),
        "claim_boundary": (
            "Each claim is scoped to its evidence layer. Review success on generated or "
            "rendered media is never simulator, contact, real-world, or deployment proof; "
            "simulator success is never physical readiness."
        ),
    }
