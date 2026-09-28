"""Runtime adapters and immutable publication for exact-workcell matrices."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest, canonical_json
from .exact_workcell_variation_matrix import (
    ISAAC_LAB_PLAN_SCHEMA_VERSION,
    PUBLICATION_SCHEMA_VERSION,
    REQUIRED_CONTROLS,
    SCHEDULE_SCHEMA_VERSION,
    SCHEDULE_REQUEST_SCHEMA_VERSION,
    ExactWorkcellVariationError,
    _IDENTIFIER,
    _is_digest,
    _json_clone,
    _mapping,
    _reject_unknown_fields,
    _require_digest_fields,
    _string,
    build_agent_proposal_brief,
    compile_variation_matrix,
    validate_variation_request,
)


def validate_schedule_request(
    value: Mapping[str, Any], *, matrix_digest: str, matrix_cell_count: int
) -> dict[str, Any]:
    """Validate the later candidate binding without changing the frozen matrix."""

    request = _json_clone(dict(value))
    blockers: list[str] = []
    _reject_unknown_fields(
        request,
        allowed={
            "schema_version",
            "matrix_digest",
            "candidate_set",
            "controls",
            "decision_design",
            "schedule_request_digest",
        },
        label="schedule_request",
        blockers=blockers,
    )
    if request.get("schema_version") != SCHEDULE_REQUEST_SCHEMA_VERSION:
        blockers.append("schedule_request_schema_invalid")
    if request.get("matrix_digest") != matrix_digest:
        blockers.append("schedule_request_matrix_digest_mismatch")
    candidate_set = _mapping(request.get("candidate_set"))
    _reject_unknown_fields(
        candidate_set,
        allowed={
            "candidate_ids",
            "candidate_identity_digests",
            "frozen_before_schedule_generation",
        },
        label="candidate_set",
        blockers=blockers,
    )
    candidate_ids = candidate_set.get("candidate_ids")
    if (
        not isinstance(candidate_ids, list)
        or len(candidate_ids) != 2
        or len(set(map(_string, candidate_ids))) != 2
        or any(not _IDENTIFIER.fullmatch(_string(item)) for item in candidate_ids)
    ):
        blockers.append("candidate_set_exactly_two_distinct_required")
        candidate_ids = []
    if candidate_set.get("frozen_before_schedule_generation") is not True:
        blockers.append("candidate_set_not_frozen")
    if set(map(_string, candidate_ids)) & set(REQUIRED_CONTROLS):
        blockers.append("candidate_id_collides_with_required_control")
    identities = _mapping(candidate_set.get("candidate_identity_digests"))
    if set(identities) != set(map(_string, candidate_ids)) or any(
        not _is_digest(digest) for digest in identities.values()
    ):
        blockers.append("candidate_identity_digests_invalid")

    controls = _mapping(request.get("controls"))
    _reject_unknown_fields(
        controls,
        allowed={"control_ids", "run_on_every_cell", "same_resolved_cell_required"},
        label="controls",
        blockers=blockers,
    )
    if controls.get("control_ids") != list(REQUIRED_CONTROLS):
        blockers.append("required_controls_invalid")
    if controls.get("run_on_every_cell") is not True:
        blockers.append("controls_every_cell_not_required")
    if controls.get("same_resolved_cell_required") is not True:
        blockers.append("controls_same_cell_not_required")

    decision_design = _mapping(request.get("decision_design"))
    _reject_unknown_fields(
        decision_design,
        allowed={
            "preregistered_experiment_digest",
            "power_analysis_digest",
            "minimum_decision_relevant_difference_digest",
            "planned_cells_per_candidate",
            "trial_count_justified_by_preregistered_power_analysis",
            "preregistered_before_policy_outcomes",
        },
        label="decision_design",
        blockers=blockers,
    )
    _require_digest_fields(
        decision_design,
        label="decision_design",
        fields=(
            "preregistered_experiment_digest",
            "power_analysis_digest",
            "minimum_decision_relevant_difference_digest",
        ),
        blockers=blockers,
    )
    if decision_design.get("planned_cells_per_candidate") != matrix_cell_count:
        blockers.append("decision_design_trial_count_matrix_mismatch")
    if (
        decision_design.get("trial_count_justified_by_preregistered_power_analysis")
        is not True
    ):
        blockers.append("decision_design_power_justification_missing")
    if decision_design.get("preregistered_before_policy_outcomes") is not True:
        blockers.append("decision_design_not_preregistered_before_outcomes")

    if request.get("schedule_request_digest") != canonical_digest(
        request, digest_field="schedule_request_digest"
    ):
        blockers.append("schedule_request_digest_mismatch")
    if blockers:
        raise ExactWorkcellVariationError(blockers)
    return request


def compile_evaluation_schedule(
    matrix: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    schedule_request: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind both controls and both policies to the identical matrix cells."""

    validated_request = validate_variation_request(request)
    expected_matrix = compile_variation_matrix(validated_request)
    if dict(matrix) != expected_matrix:
        raise ExactWorkcellVariationError(["matrix_does_not_match_request"])
    validated_schedule_request = validate_schedule_request(
        schedule_request,
        matrix_digest=_string(matrix.get("matrix_digest")),
        matrix_cell_count=int(matrix.get("cell_count", 0)),
    )
    candidate_ids = list(
        _mapping(validated_schedule_request.get("candidate_set"))["candidate_ids"]
    )
    subjects = [
        ("control", REQUIRED_CONTROLS[0]),
        ("control", REQUIRED_CONTROLS[1]),
        ("policy", candidate_ids[0]),
        ("policy", candidate_ids[1]),
    ]
    rows: list[dict[str, Any]] = []
    for cell in matrix["cells"]:
        for subject_type, subject_id in subjects:
            row = {
                "episode_id": f"{cell['cell_id']}.{subject_id}",
                "subject_type": subject_type,
                "subject_id": subject_id,
                "cell_id": cell["cell_id"],
                "cell_digest": cell["cell_digest"],
                "reset_digest": cell["reset_digest"],
                "seed": cell["seed"],
                "execution_order": len(rows),
                "candidate_policy_queried": subject_type == "policy",
                "independent_grader_required": True,
                "complete_planned_duration_required": True,
                "early_success_stop_allowed": False,
            }
            row["episode_binding_digest"] = canonical_digest(
                row, digest_field="episode_binding_digest"
            )
            rows.append(row)
    bindings: dict[str, dict[str, Any]] = {}
    for _subject_type, subject_id in subjects:
        slice_rows = [
            {
                "cell_id": row["cell_id"],
                "cell_digest": row["cell_digest"],
                "reset_digest": row["reset_digest"],
                "seed": row["seed"],
            }
            for row in rows
            if row["subject_id"] == subject_id
        ]
        bindings[subject_id] = {
            "cell_count": len(slice_rows),
            "cell_set_digest": canonical_digest({"cells": slice_rows}),
        }
    if len({binding["cell_set_digest"] for binding in bindings.values()}) != 1:
        raise ExactWorkcellVariationError(["subject_cell_sets_not_identical"])
    schedule = {
        "schema_version": SCHEDULE_SCHEMA_VERSION,
        "matrix_id": matrix.get("matrix_id"),
        "matrix_digest": matrix.get("matrix_digest"),
        "schedule_request_digest": validated_schedule_request.get(
            "schedule_request_digest"
        ),
        "cell_count": matrix.get("cell_count"),
        "episode_count": len(rows),
        "episodes_per_subject": matrix.get("cell_count"),
        "candidate_ids": candidate_ids,
        "control_ids": list(REQUIRED_CONTROLS),
        "decision_design": validated_schedule_request.get("decision_design"),
        "subject_bindings": bindings,
        "all_subjects_receive_identical_cells_resets_and_seeds": True,
        "execution_policy": {
            "controls_before_policies_within_each_cell": True,
            "retry_cap": 0,
            "no_automatic_retries": True,
            "no_early_success_stop": True,
            "terminal_scientific_or_planned_duration_required": True,
        },
        "rows": rows,
        "claim_boundary": {
            "schedule_is_not_execution": True,
            "schedule_is_not_policy_ranking": True,
            "object_cousins_in_primary": False,
        },
    }
    schedule["schedule_digest"] = canonical_digest(
        schedule, digest_field="schedule_digest"
    )
    return schedule


def evaluation_run_task_scenario_pack(
    matrix: Mapping[str, Any], *, request: Mapping[str, Any], matrix_uri: str
) -> dict[str, Any]:
    """Adapt the immutable matrix to the canonical EvaluationRunSpec surface."""

    expected = compile_variation_matrix(request)
    if dict(matrix) != expected:
        raise ExactWorkcellVariationError(["matrix_does_not_match_request"])
    uri = _string(matrix_uri)
    if not uri or not uri.startswith(("gs://", "s3://", "https://", "file://")):
        raise ExactWorkcellVariationError(["matrix_uri_invalid"])
    task_id = _string(_mapping(request.get("task_binding")).get("task_id"))
    return {
        "adapter_id": "exact_workcell_variation_matrix",
        "adapter_version": "1",
        "pack_id": _string(matrix.get("matrix_id")),
        "tasks": [{"task_id": task_id}],
        "scenarios": [
            {
                "scenario_id": row["cell_id"],
                "task_id": task_id,
                "condition_id": row["cell_id"],
                "condition_digest": row["cell_digest"],
                "reset_digest": row["reset_digest"],
                "seed": row["seed"],
                "partition": row["partition"],
                "application_records": row["application_records"],
            }
            for row in matrix["cells"]
        ],
        "matrix_uri": uri,
        "matrix_digest": matrix["matrix_digest"],
        "cell_count": matrix["cell_count"],
        "policy_neutral": True,
        "object_cousins_in_primary": False,
        "required_controls": list(REQUIRED_CONTROLS),
    }


def compile_isaac_lab_event_plan(
    matrix: Mapping[str, Any], *, request: Mapping[str, Any]
) -> dict[str, Any]:
    """Compile cells into manager-targeted Isaac Lab reset/readback terms."""

    expected = compile_variation_matrix(request)
    if dict(matrix) != expected:
        raise ExactWorkcellVariationError(["matrix_does_not_match_request"])
    cells = []
    for row in matrix["cells"]:
        terms = []
        for record in row["application_records"]:
            term = {
                "term_id": f"apply_{record['dimension_id']}",
                "manager_target": record["application_target"],
                "value": record["resolved_value"],
                "unit": record["unit"],
                "authority_digest": record["authority_digest"],
                "readback": {
                    "required": True,
                    "comparison": record["readback_comparison"],
                    "tolerance": record["application_tolerance"],
                    "failure_behavior": "abstain_cell_before_policy_query",
                },
            }
            term["term_digest"] = canonical_digest(term, digest_field="term_digest")
            terms.append(term)
        cell_plan = {
            "cell_id": row["cell_id"],
            "cell_digest": row["cell_digest"],
            "reset_digest": row["reset_digest"],
            "seed": row["seed"],
            "event_mode": "reset",
            "terms": terms,
            "policy_query_allowed_before_all_readbacks_pass": False,
        }
        cell_plan["cell_plan_digest"] = canonical_digest(
            cell_plan, digest_field="cell_plan_digest"
        )
        cells.append(cell_plan)
    plan = {
        "schema_version": ISAAC_LAB_PLAN_SCHEMA_VERSION,
        "matrix_id": matrix.get("matrix_id"),
        "matrix_digest": matrix.get("matrix_digest"),
        "cell_count": len(cells),
        "runtime": "isaac_lab_manager_based",
        "cells": cells,
        "claim_boundary": {
            "plan_is_not_isaac_execution": True,
            "readback_required_before_policy_query": True,
            "object_cousins_in_primary": False,
        },
    }
    plan["event_plan_digest"] = canonical_digest(
        plan, digest_field="event_plan_digest"
    )
    return plan


def validate_matrix_and_schedule(
    *,
    request: Mapping[str, Any],
    schedule_request: Mapping[str, Any],
    matrix: Mapping[str, Any],
    schedule: Mapping[str, Any],
) -> dict[str, Any]:
    expected_matrix = compile_variation_matrix(request)
    if dict(matrix) != expected_matrix:
        raise ExactWorkcellVariationError(["matrix_validation_mismatch"])
    expected_schedule = compile_evaluation_schedule(
        expected_matrix, request=request, schedule_request=schedule_request
    )
    if dict(schedule) != expected_schedule:
        raise ExactWorkcellVariationError(["schedule_validation_mismatch"])
    return {
        "schema_version": "exact_workcell_variation_validation.v1",
        "status": "passed",
        "request_digest": request.get("request_digest"),
        "schedule_request_digest": schedule_request.get("schedule_request_digest"),
        "matrix_digest": matrix.get("matrix_digest"),
        "schedule_digest": schedule.get("schedule_digest"),
        "cell_count": matrix.get("cell_count"),
        "episode_count": schedule.get("episode_count"),
        "exact_workcell_primary": True,
        "object_cousins_in_primary": False,
    }


def apply_runtime_cell(*, matrix: Mapping[str, Any], request: Mapping[str, Any], cell_id: str,
                       runtime_bindings: Mapping[str, Any], reset: Any) -> dict[str, Any]:
    """Apply admitted EventManager terms and read each native value after reset.

    Runtime bindings expose separate ``apply(value)`` and ``read()`` methods.
    Apply return values are deliberately ignored. A missing backend binding
    refuses before any mutation; this adapter never allocates or queries policy.
    """
    plan = compile_isaac_lab_event_plan(matrix, request=request)
    matches = [cell for cell in plan["cells"] if cell["cell_id"] == cell_id]
    if len(matches) != 1:
        raise ExactWorkcellVariationError(["runtime_cell_not_in_frozen_matrix"])
    cell = matches[0]
    targets = [term["manager_target"] for term in cell["terms"]]
    if len(set(targets)) != len(targets) or any(target not in runtime_bindings or
        not callable(getattr(runtime_bindings[target], "apply", None)) or
        not callable(getattr(runtime_bindings[target], "read", None)) for target in targets):
        raise ExactWorkcellVariationError(["runtime_cell_native_bindings_missing_or_ambiguous"])
    for term in cell["terms"]:
        runtime_bindings[term["manager_target"]].apply(term["value"])
    reset(seed=cell["seed"])
    observations = []
    blockers = []
    for term in cell["terms"]:
        observation = runtime_bindings[term["manager_target"]].read()
        if not isinstance(observation, Mapping) or observation.get("unit") != term["unit"] or observation.get("source") != "native_readback":
            blockers.append("runtime_cell_readback_identity_invalid:" + term["term_id"])
            continue
        actual, expected = observation.get("value"), term["value"]
        if term["readback"]["comparison"] == "exact":
            passed = type(actual) is type(expected) and actual == expected
        else:
            passed = (isinstance(actual, (int, float)) and not isinstance(actual, bool)
                and math.isfinite(actual) and abs(actual - expected) <= term["readback"]["tolerance"])
        observations.append({"term_digest": term["term_digest"], "observed": dict(observation), "passed": passed})
        if not passed:
            blockers.append("runtime_cell_native_value_mismatch:" + term["term_id"])
    result = {"schema_version": "exact_workcell_reset_application.v1", "cell_id": cell_id,
        "cell_digest": cell["cell_digest"], "reset_digest": cell["reset_digest"], "seed": cell["seed"],
        "cell_plan_digest": cell["cell_plan_digest"], "observations": observations,
        "status": "blocked" if blockers else "applied_and_readback_verified", "blockers": blockers,
        "policy_query_performed": False, "scope": "parameter_application_not_full_reset_or_qualification"}
    result["application_digest"] = canonical_digest(result, digest_field="application_digest")
    return result


def collect_evaluation_evidence(*, request: Mapping[str, Any], schedule_request: Mapping[str, Any],
                                matrix: Mapping[str, Any], schedule: Mapping[str, Any],
                                episode_records: Sequence[Mapping[str, Any]], evidence_root: Path,
                                output_path: Path | None = None) -> dict[str, Any]:
    """Join real retained episode bytes to the frozen full schedule, never Quick-10.

    The independent episode index validates media and lifecycle. The paired
    diagnostic summary remains separate from a claim of complete simulator
    evidence; missing controls, native reset, wire or sensor evidence block it.
    Resumes supply the union of immutable records; duplicates refuse.
    """
    from .adp_episode_evidence_index import _episode_row, _verify_artifact, EpisodeEvidenceIndexError
    from .policy_scientific_reset import validate_reset_readback, compare_reset_readbacks
    from .policy_paired_summary import paired_summary
    from .adp_task_scoring import score_task_episode_from_spec, OUTCOME_NEVER_MOVED

    validate_matrix_and_schedule(request=request, schedule_request=schedule_request, matrix=matrix, schedule=schedule)
    frozen = {row["episode_id"]: row for row in schedule["rows"]}
    cells = {row["cell_id"]: row for row in matrix["cells"]}
    event_cells = {row["cell_id"]: row for row in compile_isaac_lab_event_plan(matrix, request=request)["cells"]}
    retained = {}
    policies = []
    resets = {}
    blockers = []
    for record in episode_records:
        episode_id = record.get("episode_id")
        if episode_id not in frozen or episode_id in retained:
            raise ExactWorkcellVariationError(["evaluation_episode_duplicate_or_unscheduled"])
        binding = frozen[episode_id]
        if record.get("episode_binding_digest") != binding["episode_binding_digest"]:
            raise ExactWorkcellVariationError(["evaluation_episode_binding_mismatch"])
        artifact = _verify_artifact(evidence_root, record["artifact"], role="evaluation_episode_receipt")
        path = (evidence_root / artifact["relative_path"]).resolve()
        raw = json.loads(path.read_text())
        native_episode_id = raw.get("episode_id") or (raw.get("episode") or {}).get("episode_id")
        if native_episode_id != episode_id:
            raise ExactWorkcellVariationError(["evaluation_episode_id_mismatch"])
        retained[episode_id] = artifact
        index_root = (evidence_root / str(record.get("media_root_relative") or record["artifact"].get("media_root_relative") or ".")).resolve()
        if index_root != evidence_root.resolve() and evidence_root.resolve() not in index_root.parents:
            raise ExactWorkcellVariationError(["evaluation_episode_media_root_outside_evidence"])
        try:
            _episode_row(index_root, path)
        except EpisodeEvidenceIndexError as exc:
            blockers.append("episode_evidence_incomplete:" + episode_id + ":" + str(exc))
            if binding["subject_type"] == "policy":
                cell = cells[binding["cell_id"]]
                policies.append({"candidate_id": binding["subject_id"], "cell_id": binding["cell_id"], "seed": binding["seed"],
                    "family": "canonical_anchor" if cell["cell_id"] == matrix["canonical_anchor_cell_id"] else cell["phase"],
                    "partition": cell["partition"], "status": "blocked", "episode": raw,
                    "candidate_policy_queried": raw.get("candidate_policy_queried") is True,
                    "candidate_policy_query_attempted": raw.get("candidate_policy_query_attempted") is True,
                    "policy_outcome_interpretable": False})
            continue
        application = record.get("reset_application") or {}
        event_cell = event_cells[binding["cell_id"]]
        terms = {term["term_digest"]: term for term in event_cell["terms"]}
        observed_terms = application.get("observations") or []
        application_valid = (application.get("application_digest") == canonical_digest(application, digest_field="application_digest")
            and application.get("cell_plan_digest") == event_cell["cell_plan_digest"]
            and application.get("seed") == binding["seed"]
            and application.get("status") == "applied_and_readback_verified"
            and len(observed_terms) == len(terms) and {row.get("term_digest") for row in observed_terms} == set(terms))
        for measured in observed_terms:
            term = terms.get(measured.get("term_digest"))
            observed = measured.get("observed") or {}
            if term is None or observed.get("unit") != term["unit"] or observed.get("source") != "native_readback":
                application_valid = False
                continue
            actual, expected = observed.get("value"), term["value"]
            if term["readback"]["comparison"] == "exact":
                application_valid = application_valid and type(actual) is type(expected) and actual == expected
            else:
                application_valid = application_valid and isinstance(actual, (int, float)) and not isinstance(actual, bool) and math.isfinite(actual) and abs(actual - expected) <= term["readback"]["tolerance"]
        if not application_valid:
            blockers.append("native_parameter_application_unproven:" + episode_id)
        task_spec = raw.get("task_spec") or {}
        if raw.get("task_spec_digest") != canonical_digest(task_spec) or (task_spec.get("task_success_contract") or {}).get("contract_digest") != request["task_binding"]["success_contract_digest"]:
            blockers.append("task_success_contract_unproven:" + episode_id)
        elif (raw.get("score") or {}).get("status") == "scored":
            state = raw.get("state_trace") or []
            samples = state.get("task_state_samples") if isinstance(state, Mapping) else state
            recomputed = score_task_episode_from_spec(task_spec=task_spec, samples=samples)
            if recomputed != raw.get("score"):
                raise ExactWorkcellVariationError(["evaluation_deterministic_score_mismatch"])
        else:
            blockers.append("deterministic_outcome_unscorable:" + episode_id)
        reset = raw.get("scientific_reset")
        if not isinstance(reset, Mapping):
            blockers.append("scientific_reset_missing:" + episode_id)
        else:
            reset = validate_reset_readback(reset)
            identity = reset["binding"]
            if identity["cell_id"] != binding["cell_id"] or identity["seed"] != binding["seed"] or identity["candidate_id"] != binding["subject_id"]:
                raise ExactWorkcellVariationError(["evaluation_native_reset_binding_mismatch"])
            if identity.get("matrix_cell_digest") != binding["cell_digest"] or identity.get("matrix_reset_digest") != binding["reset_digest"]:
                raise ExactWorkcellVariationError(["evaluation_native_reset_matrix_mismatch"])
            resets[episode_id] = reset
            if not reset["complete"]:
                blockers.append("scientific_reset_incomplete:" + episode_id)
        if binding["subject_type"] == "control":
            expected_success = binding["subject_id"] == "deterministic_scripted_positive"
            score = raw.get("score") or {}
            if (raw.get("control_id") != binding["subject_id"] or raw.get("control_passed") is not True
                    or raw.get("candidate_policy_queried") is not False or raw.get("grader_authority") != "deterministic_simulator_state"
                    or raw.get("phase_execution_blocker") is not None or score.get("status") != "scored"
                    or score.get("task_succeeded") is not expected_success
                    or not expected_success and score.get("outcome") != OUTCOME_NEVER_MOVED):
                blockers.append("control_failed:" + episode_id)
        else:
            if raw.get("schema_version") != "adp009d_policy_episode.v4":
                blockers.append("current_policy_lifecycle_missing:" + episode_id)
            expected_identity = schedule_request["candidate_set"]["candidate_identity_digests"][binding["subject_id"]]
            if record.get("candidate_identity_digest") != expected_identity or raw.get("candidate_id") != binding["subject_id"]:
                raise ExactWorkcellVariationError(["evaluation_frozen_candidate_mismatch"])
            if raw.get("policy_request_evidence_complete") is not True or not raw.get("sensor_freshness") or not all(row.get("verified") is True for row in raw["sensor_freshness"]):
                blockers.append("policy_observation_evidence_incomplete:" + episode_id)
            cell = cells[binding["cell_id"]]
            policies.append({"candidate_id": binding["subject_id"], "cell_id": binding["cell_id"], "seed": binding["seed"],
                "family": "canonical_anchor" if cell["cell_id"] == matrix["canonical_anchor_cell_id"] else cell["phase"],
                "partition": cell["partition"], "status": "completed", "episode": raw,
                "candidate_policy_queried": raw.get("candidate_policy_queried") is True,
                "policy_outcome_interpretable": (raw.get("score") or {}).get("status") == "scored",
                "scientific_reset": reset, "scoring_authority": "deterministic_simulator_state"})
    for cell_id in cells:
        cell_resets = [resets[row["episode_id"]] for row in schedule["rows"] if row["cell_id"] == cell_id and row["episode_id"] in resets]
        for other in cell_resets[1:]:
            if not compare_reset_readbacks(cell_resets[0], other)["comparison_eligible"]:
                blockers.append("evaluation_reset_parity_unproven:" + cell_id)
    missing = sorted(set(frozen) - set(retained))
    if missing:
        blockers.append("evaluation_scheduled_episodes_missing")
    planned = [{"cell_id": cell["cell_id"], "seed": cell["seed"], "partition": cell["partition"],
        "family": "canonical_anchor" if cell["cell_id"] == matrix["canonical_anchor_cell_id"] else cell["phase"]} for cell in matrix["cells"]]
    result = {"schema_version": "exact_workcell_execution_evidence.v1", "matrix_digest": matrix["matrix_digest"],
        "schedule_digest": schedule["schedule_digest"], "status": "blocked" if blockers else "complete_simulator_evidence",
        "retained_episodes": dict(sorted(retained.items())), "missing_episode_ids": missing, "blockers": sorted(set(blockers)),
        "comparison": paired_summary(policies, candidate_ids=schedule["candidate_ids"], planned_cells=planned),
        "qualification_authorized": False, "physical_evidence_claimed": False}
    result["evidence_digest"] = canonical_digest(result, digest_field="evidence_digest")
    if output_path is not None:
        _create_only(output_path, result)
    return result


def _create_only(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    if path.exists():
        raise ExactWorkcellVariationError([f"publication_path_exists:{path.name}"])
    content = (canonical_json(value) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
    observed = path.read_bytes()
    if observed != content:
        raise ExactWorkcellVariationError([f"publication_readback_mismatch:{path.name}"])
    return {
        "relative_path": path.name,
        "size_bytes": len(observed),
        "sha256": "sha256:" + hashlib.sha256(observed).hexdigest(),
        "create_only": True,
        "full_byte_readback_verified": True,
    }


def publish_variation_bundle(
    request: Mapping[str, Any],
    *,
    schedule_request: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Create and read back one immutable request/matrix/schedule bundle."""

    validated_request = validate_variation_request(request)
    matrix = compile_variation_matrix(validated_request)
    isaac_lab_plan = compile_isaac_lab_event_plan(matrix, request=validated_request)
    validated_schedule_request = validate_schedule_request(
        schedule_request,
        matrix_digest=_string(matrix.get("matrix_digest")),
        matrix_cell_count=int(matrix.get("cell_count", 0)),
    )
    schedule = compile_evaluation_schedule(
        matrix,
        request=validated_request,
        schedule_request=validated_schedule_request,
    )
    validation = validate_matrix_and_schedule(
        request=validated_request,
        schedule_request=validated_schedule_request,
        matrix=matrix,
        schedule=schedule,
    )
    requested_root = Path(output_dir).expanduser()
    if requested_root.is_symlink():
        raise ExactWorkcellVariationError(["publication_output_path_invalid"])
    root = requested_root.resolve()
    if root.exists():
        if not root.is_dir():
            raise ExactWorkcellVariationError(["publication_output_path_invalid"])
        try:
            if any(root.iterdir()):
                raise ExactWorkcellVariationError(["publication_output_not_empty"])
        except OSError as exc:
            raise ExactWorkcellVariationError(
                ["publication_output_path_unreadable"]
            ) from exc
    root.mkdir(parents=True, exist_ok=True)
    artifacts = [
        _create_only(root / "exact_workcell_variation_request.v1.json", validated_request),
        _create_only(
            root / "exact_workcell_evaluation_schedule_request.v1.json",
            validated_schedule_request,
        ),
        _create_only(root / "exact_workcell_variation_matrix.v1.json", matrix),
        _create_only(
            root / "exact_workcell_isaac_lab_event_plan.v1.json", isaac_lab_plan
        ),
        _create_only(root / "exact_workcell_evaluation_schedule.v1.json", schedule),
        _create_only(root / "exact_workcell_variation_validation.v1.json", validation),
    ]
    receipt = {
        "schema_version": PUBLICATION_SCHEMA_VERSION,
        "status": "published_create_only_full_byte_readback_verified",
        "matrix_id": matrix.get("matrix_id"),
        "request_digest": validated_request.get("request_digest"),
        "schedule_request_digest": validated_schedule_request.get(
            "schedule_request_digest"
        ),
        "matrix_digest": matrix.get("matrix_digest"),
        "isaac_lab_event_plan_digest": isaac_lab_plan.get("event_plan_digest"),
        "schedule_digest": schedule.get("schedule_digest"),
        "artifacts": artifacts,
    }
    receipt["publication_digest"] = canonical_digest(
        receipt, digest_field="publication_digest"
    )
    _create_only(root / "exact_workcell_variation_publication.v1.json", receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--schedule-request", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--emit-agent-brief", type=Path)
    args = parser.parse_args(argv)
    request = json.loads(args.request.read_text(encoding="utf-8"))
    schedule_request = json.loads(args.schedule_request.read_text(encoding="utf-8"))
    if args.emit_agent_brief:
        brief = build_agent_proposal_brief(request)
        _create_only(args.emit_agent_brief, brief)
    receipt = publish_variation_bundle(
        request, schedule_request=schedule_request, output_dir=args.output_dir
    )
    print(canonical_json(receipt))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
