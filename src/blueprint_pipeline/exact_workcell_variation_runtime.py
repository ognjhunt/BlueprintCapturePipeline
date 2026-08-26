"""Runtime adapters and immutable publication for exact-workcell matrices."""

from __future__ import annotations

import argparse
import hashlib
import json
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
