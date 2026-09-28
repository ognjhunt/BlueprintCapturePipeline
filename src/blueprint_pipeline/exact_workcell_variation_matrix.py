"""Compile a deterministic exact-workcell policy-evaluation matrix.

Agents may propose which admitted dimensions and interactions deserve emphasis.
They cannot create measurements, widen bounds, change the task/object/workcell,
or authorize a cell.  This module is the deterministic authority: it validates
digest-bound scene/task/embodiment inputs, materializes one canonical anchor and
99 bounded variations by default, then gives the identical cells and seeds to
exactly two frozen policies and both required controls.

The primary matrix intentionally excludes object cousins.  A cousin suite is a
separate robustness artifact and must never be folded into this matrix's score.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Mapping, Protocol, Sequence

from .decision_evidence_contracts import canonical_digest, canonical_json


REQUEST_SCHEMA_VERSION = "exact_workcell_variation_request.v1"
MATRIX_SCHEMA_VERSION = "exact_workcell_variation_matrix.v1"
SCHEDULE_SCHEMA_VERSION = "exact_workcell_evaluation_schedule.v1"
SCHEDULE_REQUEST_SCHEMA_VERSION = "exact_workcell_evaluation_schedule_request.v1"
PUBLICATION_SCHEMA_VERSION = "exact_workcell_variation_publication.v1"
ISAAC_LAB_PLAN_SCHEMA_VERSION = "exact_workcell_isaac_lab_event_plan.v1"
AGENT_PROPOSAL_SCHEMA_VERSION = "exact_workcell_variation_agent_proposal.v1"
SCENE_INPUT_SCHEMA_VERSION = "exact_workcell_scene_variation_contract.v1"
TASK_INPUT_SCHEMA_VERSION = "exact_workcell_task_variation_contract.v1"
EMBODIMENT_INPUT_SCHEMA_VERSION = "exact_workcell_embodiment_variation_contract.v1"

DEFAULT_CELL_COUNT = 100
REQUIRED_CONTROLS = (
    "zero_action_negative",
    "deterministic_scripted_positive",
)
ALLOWED_FAMILIES = {
    "reset_state",
    "placement_approach",
    "camera_sensor",
    "illumination",
    "bounded_physics",
    "policy_stochasticity",
}
ALLOWED_APPLICATION_PREFIXES = (
    "EventManager.",
    "ObservationManager.",
    "Scene.",
    "SimulationCfg.",
)
FORBIDDEN_DIMENSION_MARKERS = {
    "asset",
    "cousin",
    "geometry_identity",
    "object_class",
    "object_identity",
    "scene_identity",
    "task_identity",
}
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")


class ExactWorkcellVariationError(ValueError):
    """Stable, fail-closed blockers for matrix compilation or validation."""

    def __init__(self, blockers: Sequence[str]):
        self.blockers = tuple(sorted(set(str(item) for item in blockers if str(item))))
        super().__init__(";".join(self.blockers))


class VariationProposalAgent(Protocol):
    """Replaceable LLM/agent seam; deterministic code still owns admission."""

    model_identity: str

    def propose(self, *, brief: Mapping[str, Any]) -> Mapping[str, Any]: ...


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def _is_digest(value: Any) -> bool:
    return bool(_DIGEST.fullmatch(_string(value)))


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _json_clone(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ExactWorkcellVariationError(["request_not_finite_json"]) from exc


def _require_digest_fields(
    value: Mapping[str, Any], *, label: str, fields: Sequence[str], blockers: list[str]
) -> None:
    for field in fields:
        if not _is_digest(value.get(field)):
            blockers.append(f"{label}_{field}_invalid")


def _reject_unknown_fields(
    value: Mapping[str, Any], *, allowed: set[str], label: str, blockers: list[str]
) -> None:
    blockers.extend(f"{label}_unknown_field:{field}" for field in sorted(set(value) - allowed))


def _dimension_levels(dimension: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    kind = _string(dimension.get("value_type"))
    nominal = dimension.get("nominal")
    if kind in {"continuous", "integer"}:
        minimum = _number(dimension.get("minimum"))
        maximum = _number(dimension.get("maximum"))
        number = _number(nominal)
        if minimum is None or maximum is None or number is None:
            raise ExactWorkcellVariationError(
                [f"dimension_numeric_bounds_invalid:{dimension.get('dimension_id')}"]
            )
        if not minimum < maximum or not minimum <= number <= maximum:
            raise ExactWorkcellVariationError(
                [f"dimension_numeric_bounds_invalid:{dimension.get('dimension_id')}"]
            )
        if kind == "integer":
            if not all(value.is_integer() for value in (minimum, number, maximum)):
                raise ExactWorkcellVariationError(
                    [f"dimension_integer_levels_invalid:{dimension.get('dimension_id')}"]
                )
            low, mid, high = int(minimum), int(number), int(maximum)
            if low == high or not low <= mid <= high:
                raise ExactWorkcellVariationError(
                    [f"dimension_integer_levels_invalid:{dimension.get('dimension_id')}"]
                )
            return low, mid, high
        decimals = dimension.get("decimals", 9)
        if isinstance(decimals, bool) or not isinstance(decimals, int) or not 0 <= decimals <= 12:
            raise ExactWorkcellVariationError(
                [f"dimension_decimals_invalid:{dimension.get('dimension_id')}"]
            )
        low, mid, high = (
            round(minimum, decimals),
            round(number, decimals),
            round(maximum, decimals),
        )
        if not low < high or not low <= mid <= high:
            raise ExactWorkcellVariationError(
                [f"dimension_continuous_levels_collapse:{dimension.get('dimension_id')}"]
            )
        return low, mid, high
    if kind == "categorical":
        values = dimension.get("values")
        if not isinstance(values, list) or len(values) < 2:
            raise ExactWorkcellVariationError(
                [f"dimension_categorical_values_invalid:{dimension.get('dimension_id')}"]
            )
        value_keys = [canonical_json(value) for value in values]
        nominal_key = canonical_json(nominal)
        if any(
            not isinstance(value, (str, int, float, bool))
            or type(value) is not type(nominal)
            for value in values
        ) or (
            nominal_key not in value_keys or len(set(value_keys)) < 2
        ):
            raise ExactWorkcellVariationError(
                [f"dimension_categorical_values_invalid:{dimension.get('dimension_id')}"]
            )
        alternatives = [
            value for value in values if canonical_json(value) != nominal_key
        ]
        return alternatives[0], nominal, alternatives[-1]
    raise ExactWorkcellVariationError(
        [f"dimension_value_type_invalid:{dimension.get('dimension_id')}"]
    )


def _validate_agent_proposal(
    proposal: Mapping[str, Any], *, dimension_ids: set[str]
) -> dict[str, Any]:
    blockers: list[str] = []
    value = _json_clone(dict(proposal))
    _reject_unknown_fields(
        value,
        allowed={
            "schema_version",
            "status",
            "model_identity",
            "prompt_digest",
            "response_digest",
            "raw_proposal",
            "outcome_data_accessed",
            "may_widen_admitted_bounds",
            "may_change_workcell_or_task_identity",
            "dimension_priorities",
            "targeted_interactions",
            "object_cousins",
            "proposal_digest",
        },
        label="agent_proposal",
        blockers=blockers,
    )
    if value.get("schema_version") != AGENT_PROPOSAL_SCHEMA_VERSION:
        blockers.append("agent_proposal_schema_invalid")
    if value.get("status") != "proposal_only":
        blockers.append("agent_proposal_status_invalid")
    if value.get("outcome_data_accessed") is not False:
        blockers.append("agent_proposal_outcome_access_invalid")
    if value.get("may_widen_admitted_bounds") is not False:
        blockers.append("agent_proposal_bound_authority_invalid")
    if value.get("may_change_workcell_or_task_identity") is not False:
        blockers.append("agent_proposal_identity_authority_invalid")
    _require_digest_fields(
        value,
        label="agent_proposal",
        fields=("prompt_digest", "response_digest"),
        blockers=blockers,
    )
    if not _string(value.get("model_identity")):
        blockers.append("agent_proposal_model_identity_missing")
    raw_proposal = _mapping(value.get("raw_proposal"))
    if not raw_proposal:
        blockers.append("agent_proposal_raw_response_missing")
    elif value.get("response_digest") != canonical_digest(
        {"raw_proposal": raw_proposal}
    ):
        blockers.append("agent_proposal_response_digest_mismatch")
    for field in ("dimension_priorities", "targeted_interactions", "object_cousins"):
        if value.get(field) != raw_proposal.get(field):
            blockers.append(f"agent_proposal_raw_response_binding_mismatch:{field}")
    _reject_unknown_fields(
        raw_proposal,
        allowed={"dimension_priorities", "targeted_interactions", "object_cousins"},
        label="agent_proposal_raw_response",
        blockers=blockers,
    )
    priorities = _rows(value.get("dimension_priorities"))
    if not isinstance(value.get("dimension_priorities"), list) or len(priorities) != len(
        value.get("dimension_priorities", [])
    ):
        blockers.append("agent_proposal_dimension_priority_shape_invalid")
    priority_ids = [_string(row.get("dimension_id")) for row in priorities]
    if len(priority_ids) != len(set(priority_ids)) or any(
        dimension_id not in dimension_ids for dimension_id in priority_ids
    ):
        blockers.append("agent_proposal_dimension_priority_invalid")
    for row in priorities:
        weight = _number(row.get("weight"))
        if weight is None or weight <= 0:
            blockers.append(
                f"agent_proposal_weight_invalid:{_string(row.get('dimension_id')) or 'missing'}"
            )
    interactions = _rows(value.get("targeted_interactions"))
    if not isinstance(value.get("targeted_interactions"), list) or len(
        interactions
    ) != len(value.get("targeted_interactions", [])):
        blockers.append("agent_proposal_interaction_shape_invalid")
    for index, row in enumerate(interactions):
        ids = row.get("dimension_ids")
        if (
            not isinstance(ids, list)
            or not 2 <= len(ids) <= 4
            or len(ids) != len(set(ids))
            or any(_string(item) not in dimension_ids for item in ids)
        ):
            blockers.append(f"agent_proposal_interaction_invalid:{index}")
        if not _string(row.get("rationale")):
            blockers.append(f"agent_proposal_interaction_rationale_missing:{index}")
    if value.get("object_cousins") not in (None, []):
        blockers.append("agent_proposal_object_cousins_forbidden_in_primary")
    if value.get("proposal_digest") != canonical_digest(
        value, digest_field="proposal_digest"
    ):
        blockers.append("agent_proposal_digest_mismatch")
    if blockers:
        raise ExactWorkcellVariationError(blockers)
    return value


def validate_variation_request(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the task-neutral authority inputs without performing I/O."""

    request = _json_clone(dict(value))
    blockers: list[str] = []
    _reject_unknown_fields(
        request,
        allowed={
            "schema_version",
            "program_id",
            "matrix_id",
            "matrix_kind",
            "implementation_commit",
            "cell_count",
            "seed_root",
            "scene_binding",
            "task_binding",
            "embodiment_binding",
            "controls",
            "variation_dimensions",
            "agent_proposal",
            "object_cousins",
            "request_digest",
        },
        label="request",
        blockers=blockers,
    )
    if request.get("schema_version") != REQUEST_SCHEMA_VERSION:
        blockers.append("request_schema_invalid")
    if request.get("program_id") != "arm-decision-proof-v1":
        blockers.append("request_program_invalid")
    if request.get("matrix_kind") != "exact_workcell_primary":
        blockers.append("request_matrix_kind_invalid")
    matrix_id = _string(request.get("matrix_id"))
    if not _IDENTIFIER.fullmatch(matrix_id):
        blockers.append("request_matrix_id_invalid")
    if not _COMMIT.fullmatch(_string(request.get("implementation_commit"))):
        blockers.append("request_implementation_commit_invalid")
    count = request.get("cell_count", DEFAULT_CELL_COUNT)
    if isinstance(count, bool) or not isinstance(count, int) or not 10 <= count <= 1000:
        blockers.append("request_cell_count_invalid")
    seed_root = request.get("seed_root")
    if isinstance(seed_root, bool) or not isinstance(seed_root, int) or seed_root < 0:
        blockers.append("request_seed_root_invalid")

    scene = _mapping(request.get("scene_binding"))
    task = _mapping(request.get("task_binding"))
    embodiment = _mapping(request.get("embodiment_binding"))
    for label, binding, id_field, digests in (
        (
            "scene",
            scene,
            "scene_id",
            ("scene_digest", "coordinate_frame_digest", "canonical_object_asset_digest"),
        ),
        (
            "task",
            task,
            "task_id",
            ("task_digest", "reset_contract_digest", "success_contract_digest"),
        ),
        (
            "embodiment",
            embodiment,
            "embodiment_id",
            ("embodiment_digest", "joint_limits_digest", "camera_calibration_digest"),
        ),
    ):
        if not _IDENTIFIER.fullmatch(_string(binding.get(id_field))):
            blockers.append(f"{label}_{id_field}_invalid")
        _require_digest_fields(binding, label=label, fields=digests, blockers=blockers)
    _reject_unknown_fields(
        scene,
        allowed={
            "scene_id",
            "scene_digest",
            "coordinate_frame_digest",
            "canonical_object_asset_id",
            "canonical_object_asset_digest",
        },
        label="scene_binding",
        blockers=blockers,
    )
    _reject_unknown_fields(
        task,
        allowed={
            "task_id",
            "task_digest",
            "reset_contract_digest",
            "success_contract_digest",
        },
        label="task_binding",
        blockers=blockers,
    )
    _reject_unknown_fields(
        embodiment,
        allowed={
            "embodiment_id",
            "embodiment_digest",
            "joint_limits_digest",
            "camera_calibration_digest",
        },
        label="embodiment_binding",
        blockers=blockers,
    )
    if not _string(scene.get("canonical_object_asset_id")):
        blockers.append("scene_canonical_object_asset_id_missing")
    if request.get("object_cousins") not in (None, []):
        blockers.append("object_cousins_forbidden_in_exact_workcell_primary")

    if "candidate_set" in request:
        blockers.append("variation_request_candidate_set_forbidden_policy_neutral_matrix")

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

    raw_dimensions = request.get("variation_dimensions")
    dimensions = _rows(raw_dimensions)
    if (
        not isinstance(raw_dimensions, list)
        or not dimensions
        or len(dimensions) != len(raw_dimensions)
    ):
        blockers.append("variation_dimensions_missing_or_invalid")
    ids = [_string(row.get("dimension_id")) for row in dimensions]
    if len(ids) != len(set(ids)) or any(not _IDENTIFIER.fullmatch(item) for item in ids):
        blockers.append("variation_dimension_ids_invalid")
    for dimension in dimensions:
        dimension_id = _string(dimension.get("dimension_id")) or "missing"
        _reject_unknown_fields(
            dimension,
            allowed={
                "dimension_id",
                "family",
                "value_type",
                "nominal",
                "minimum",
                "maximum",
                "values",
                "decimals",
                "unit",
                "application_target",
                "application_tolerance",
                "parameter_path",
                "source_contract",
                "authority_digest",
                "exact_workcell_invariant",
                "changes_object_or_task_identity",
            },
            label=f"dimension:{dimension_id}",
            blockers=blockers,
        )
        if _string(dimension.get("family")) not in ALLOWED_FAMILIES:
            blockers.append(f"dimension_family_invalid:{dimension_id}")
        if dimension.get("exact_workcell_invariant") is not True:
            blockers.append(f"dimension_exact_workcell_invariant_missing:{dimension_id}")
        if dimension.get("changes_object_or_task_identity") is not False:
            blockers.append(f"dimension_identity_mutation_forbidden:{dimension_id}")
        joined = " ".join(
            _string(dimension.get(field)).lower()
            for field in ("dimension_id", "application_target", "parameter_path")
        )
        if any(marker in joined for marker in FORBIDDEN_DIMENSION_MARKERS):
            blockers.append(f"dimension_identity_surface_forbidden:{dimension_id}")
        if not _string(dimension.get("application_target")):
            blockers.append(f"dimension_application_target_missing:{dimension_id}")
        elif not _string(dimension.get("application_target")).startswith(
            ALLOWED_APPLICATION_PREFIXES
        ):
            blockers.append(f"dimension_application_target_unsupported:{dimension_id}")
        tolerance = _number(dimension.get("application_tolerance"))
        if tolerance is None or tolerance < 0:
            blockers.append(f"dimension_application_tolerance_invalid:{dimension_id}")
        if not _string(dimension.get("unit")):
            blockers.append(f"dimension_unit_missing:{dimension_id}")
        if not _is_digest(dimension.get("authority_digest")):
            blockers.append(f"dimension_authority_digest_invalid:{dimension_id}")
        if dimension.get("source_contract") not in {
            "scene",
            "task",
            "embodiment",
        }:
            blockers.append(f"dimension_source_contract_invalid:{dimension_id}")
        try:
            _dimension_levels(dimension)
        except ExactWorkcellVariationError as exc:
            blockers.extend(exc.blockers)

    if isinstance(count, int) and not isinstance(count, bool):
        minimum_diagnostic_cells = 1 + 2 * len(dimensions)
        if minimum_diagnostic_cells > count:
            blockers.append("cell_budget_insufficient_for_one_factor_diagnosis")

    proposal = request.get("agent_proposal")
    if proposal is not None:
        if not isinstance(proposal, Mapping):
            blockers.append("agent_proposal_invalid")
        else:
            try:
                validated_proposal = _validate_agent_proposal(
                    proposal, dimension_ids=set(ids)
                )
                if validated_proposal.get("prompt_digest") != build_agent_proposal_brief(
                    request
                ).get("brief_digest"):
                    blockers.append("agent_proposal_prompt_digest_mismatch")
            except ExactWorkcellVariationError as exc:
                blockers.extend(exc.blockers)
    if request.get("request_digest") != canonical_digest(
        request, digest_field="request_digest"
    ):
        blockers.append("request_digest_mismatch")
    if blockers:
        raise ExactWorkcellVariationError(blockers)
    return request


def build_agent_proposal_brief(value: Mapping[str, Any]) -> dict[str, Any]:
    """Build the bounded prompt surface an agent may use for proposal work."""

    request = _json_clone(dict(value))
    dimensions = _rows(request.get("variation_dimensions"))
    brief = {
        "schema_version": "exact_workcell_variation_agent_brief.v1",
        "matrix_id": request.get("matrix_id"),
        "task_id": _mapping(request.get("task_binding")).get("task_id"),
        "embodiment_id": _mapping(request.get("embodiment_binding")).get(
            "embodiment_id"
        ),
        "scene_id": _mapping(request.get("scene_binding")).get("scene_id"),
        "cell_count": request.get("cell_count", DEFAULT_CELL_COUNT),
        "allowed_dimensions": [
            {
                "dimension_id": row.get("dimension_id"),
                "family": row.get("family"),
                "nominal": row.get("nominal"),
                "minimum": row.get("minimum"),
                "maximum": row.get("maximum"),
                "values": row.get("values"),
                "unit": row.get("unit"),
                "application_target": row.get("application_target"),
            }
            for row in dimensions
        ],
        "instructions": [
            "Prioritize only listed admitted dimensions.",
            "Propose targeted interactions using only listed dimension IDs.",
            "Do not widen bounds or invent measurements.",
            "Do not change the task, scene, embodiment, or canonical object identity.",
            "Do not include object cousins in the exact-workcell primary matrix.",
            "Do not use policy outcomes or held-out results.",
        ],
        "agent_is_proposal_only": True,
        "deterministic_compiler_is_authority": True,
    }
    brief["brief_digest"] = canonical_digest(brief, digest_field="brief_digest")
    return brief


def seal_agent_proposal(
    raw_proposal: Mapping[str, Any], *, brief: Mapping[str, Any], model_identity: str
) -> dict[str, Any]:
    """Convert untrusted agent output into the proposal-only input contract."""

    raw = _json_clone(dict(raw_proposal))
    permitted = {"dimension_priorities", "targeted_interactions", "object_cousins"}
    unknown = sorted(set(raw) - permitted)
    if unknown:
        raise ExactWorkcellVariationError(
            [f"agent_proposal_unknown_field:{field}" for field in unknown]
        )
    response_digest = canonical_digest({"raw_proposal": raw})
    proposal = {
        "schema_version": AGENT_PROPOSAL_SCHEMA_VERSION,
        "status": "proposal_only",
        "model_identity": _string(model_identity),
        "prompt_digest": brief.get("brief_digest"),
        "response_digest": response_digest,
        "raw_proposal": raw,
        "outcome_data_accessed": False,
        "may_widen_admitted_bounds": False,
        "may_change_workcell_or_task_identity": False,
        "dimension_priorities": raw.get("dimension_priorities") or [],
        "targeted_interactions": raw.get("targeted_interactions") or [],
        "object_cousins": raw.get("object_cousins") or [],
    }
    proposal["proposal_digest"] = canonical_digest(
        proposal, digest_field="proposal_digest"
    )
    dimension_ids = {
        _string(row.get("dimension_id"))
        for row in _rows(brief.get("allowed_dimensions"))
    }
    return _validate_agent_proposal(proposal, dimension_ids=dimension_ids)


def _seed_digest(*parts: Any) -> str:
    return canonical_digest({"parts": list(parts)})


def _seed_int(*parts: Any) -> int:
    return int(_seed_digest(*parts)[7:23], 16) & 0x7FFFFFFF


def _priority_order(
    dimensions: Sequence[Mapping[str, Any]], proposal: Mapping[str, Any] | None
) -> list[dict[str, Any]]:
    weights = {
        _string(row.get("dimension_id")): float(row.get("weight"))
        for row in _rows(_mapping(proposal).get("dimension_priorities"))
    }
    return sorted(
        (dict(row) for row in dimensions),
        key=lambda row: (-weights.get(_string(row.get("dimension_id")), 1.0), _string(row.get("dimension_id"))),
    )


def _binary_pairwise_assignments(ids: Sequence[str], *, seed_root: int) -> list[dict[str, int]]:
    """Greedily construct binary rows covering all four levels for every pair."""

    if len(ids) < 2:
        return []
    uncovered = {
        (left, right, left_level, right_level)
        for left_index, left in enumerate(ids)
        for right in ids[left_index + 1 :]
        for left_level in (0, 1)
        for right_level in (0, 1)
    }
    selected: list[dict[str, int]] = []
    while uncovered:
        round_index = len(selected)
        candidates = [
            {
                dimension_id: _seed_int(
                    seed_root,
                    "pairwise",
                    round_index,
                    ordinal,
                    dimension_id,
                )
                % 2
                for dimension_id in ids
            }
            for ordinal in range(256)
        ]
        fallback_term = sorted(uncovered)[0]
        fallback = {
            dimension_id: _seed_int(
                seed_root, "pairwise_fallback", round_index, dimension_id
            )
            % 2
            for dimension_id in ids
        }
        fallback[fallback_term[0]] = fallback_term[2]
        fallback[fallback_term[1]] = fallback_term[3]
        candidates.append(fallback)
        best: dict[str, int] | None = None
        best_covered: set[tuple[str, str, int, int]] = set()
        for candidate in candidates:
            covered = {
                term
                for term in uncovered
                if candidate[term[0]] == term[2] and candidate[term[1]] == term[3]
            }
            if len(covered) > len(best_covered):
                best, best_covered = candidate, covered
        if best is None or not best_covered:
            raise ExactWorkcellVariationError(["pairwise_covering_array_unsatisfied"])
        selected.append(best)
        uncovered -= best_covered
    return selected


def _sample_value(dimension: Mapping[str, Any], *, seed: int, salt: str) -> Any:
    low, nominal, high = _dimension_levels(dimension)
    kind = _string(dimension.get("value_type"))
    digest = hashlib.sha256(f"{seed}:{salt}".encode()).hexdigest()
    fraction = (int(digest[:15], 16) + 1) / (16**15 + 1)
    if kind == "continuous":
        decimals = int(dimension.get("decimals", 9))
        value = round(float(low) + fraction * (float(high) - float(low)), decimals)
        if value == nominal:
            value = low if fraction < 0.5 else high
        return value
    if kind == "integer":
        values = list(range(int(low), int(high) + 1))
        alternatives = [value for value in values if value != nominal]
        return alternatives[int(digest[:8], 16) % len(alternatives)]
    values = list(dimension.get("values") or [])
    nominal_key = canonical_json(nominal)
    alternatives = [value for value in values if canonical_json(value) != nominal_key]
    return alternatives[int(digest[:8], 16) % len(alternatives)]


def _cell(
    *,
    request: Mapping[str, Any],
    ordinal: int,
    phase: str,
    partition: str,
    values: Mapping[str, Any],
    changed_ids: Sequence[str],
) -> dict[str, Any]:
    scene = _mapping(request.get("scene_binding"))
    task = _mapping(request.get("task_binding"))
    embodiment = _mapping(request.get("embodiment_binding"))
    identity = {
        "scene_id": scene.get("scene_id"),
        "scene_digest": scene.get("scene_digest"),
        "coordinate_frame_digest": scene.get("coordinate_frame_digest"),
        "canonical_object_asset_id": scene.get("canonical_object_asset_id"),
        "canonical_object_asset_digest": scene.get("canonical_object_asset_digest"),
        "task_id": task.get("task_id"),
        "task_digest": task.get("task_digest"),
        "embodiment_id": embodiment.get("embodiment_id"),
        "embodiment_digest": embodiment.get("embodiment_digest"),
    }
    seed = _seed_int(request.get("seed_root"), request.get("matrix_id"), ordinal)
    dimension_by_id = {
        _string(row.get("dimension_id")): row
        for row in _rows(request.get("variation_dimensions"))
    }
    application_records = [
        {
            "dimension_id": dimension_id,
            "family": dimension_by_id[dimension_id].get("family"),
            "application_target": dimension_by_id[dimension_id].get(
                "application_target"
            ),
            "resolved_value": resolved_value,
            "nominal_value": dimension_by_id[dimension_id].get("nominal"),
            "unit": dimension_by_id[dimension_id].get("unit"),
            "application_tolerance": dimension_by_id[dimension_id].get(
                "application_tolerance"
            ),
            "authority_digest": dimension_by_id[dimension_id].get(
                "authority_digest"
            ),
            "source_contract": dimension_by_id[dimension_id].get(
                "source_contract"
            ),
            "independent_readback_required": True,
            "readback_comparison": (
                "exact"
                if dimension_by_id[dimension_id].get("value_type") == "categorical"
                else "absolute_error_lte_tolerance"
            ),
        }
        for dimension_id, resolved_value in sorted(values.items())
    ]
    row = {
        "cell_id": f"{request.get('matrix_id')}.cell_{ordinal:03d}",
        "ordinal": ordinal,
        "phase": phase,
        "partition": partition,
        "seed": seed,
        "seed_digest": _seed_digest(request.get("seed_root"), request.get("matrix_id"), ordinal),
        "exact_workcell_identity": identity,
        "exact_workcell_identity_digest": canonical_digest(identity),
        "resolved_values": dict(sorted(values.items())),
        "application_records": application_records,
        "changed_dimension_ids": sorted(changed_ids),
        "object_cousin": False,
        "policy_neutral": True,
        "required_controls": list(REQUIRED_CONTROLS),
        "reset_digest": canonical_digest(
            {
                "scene_digest": scene.get("scene_digest"),
                "task_reset_contract_digest": task.get("reset_contract_digest"),
                "embodiment_digest": embodiment.get("embodiment_digest"),
                "values": dict(sorted(values.items())),
                "seed": seed,
            }
        ),
    }
    row["cell_digest"] = canonical_digest(row, digest_field="cell_digest")
    return row


def compile_variation_matrix(value: Mapping[str, Any]) -> dict[str, Any]:
    """Materialize the exact number of deterministic, bounded workcell cells."""

    request = validate_variation_request(value)
    count = int(request.get("cell_count", DEFAULT_CELL_COUNT))
    proposal = _mapping(request.get("agent_proposal")) or None
    dimensions = _priority_order(_rows(request.get("variation_dimensions")), proposal)
    by_id = {_string(row.get("dimension_id")): row for row in dimensions}
    nominal = {dimension_id: _dimension_levels(row)[1] for dimension_id, row in by_id.items()}
    planned: list[tuple[str, str, dict[str, Any], list[str]]] = [
        ("canonical_anchor", "qualification", dict(nominal), [])
    ]

    for dimension_id, dimension in by_id.items():
        low, _mid, high = _dimension_levels(dimension)
        for label, resolved in (("low", low), ("high", high)):
            values = dict(nominal)
            values[dimension_id] = resolved
            planned.append(
                (f"one_factor_{label}", "qualification", values, [dimension_id])
            )

    interactions = _rows(_mapping(proposal).get("targeted_interactions"))
    for index, interaction in enumerate(interactions):
        ids = [_string(item) for item in interaction.get("dimension_ids", [])]
        values = dict(nominal)
        for offset, dimension_id in enumerate(ids):
            low, _mid, high = _dimension_levels(by_id[dimension_id])
            values[dimension_id] = high if (index + offset) % 2 == 0 else low
        planned.append(("targeted_interaction", "qualification", values, ids))

    pairwise = _binary_pairwise_assignments(list(by_id), seed_root=int(request["seed_root"]))
    for assignment in pairwise:
        values = dict(nominal)
        for dimension_id, level in assignment.items():
            low, _mid, high = _dimension_levels(by_id[dimension_id])
            values[dimension_id] = high if level else low
        planned.append(("pairwise_covering_array", "qualification", values, list(by_id)))

    unique: set[str] = set()
    deduplicated: list[tuple[str, str, dict[str, Any], list[str]]] = []
    for row in planned:
        key = canonical_json(row[2])
        if key not in unique:
            unique.add(key)
            deduplicated.append(row)
    planned = deduplicated
    if len(planned) > count:
        raise ExactWorkcellVariationError(["cell_budget_insufficient_for_required_coverage"])

    heldout_target = max(10, count // 5)
    qualification_limit = count - heldout_target
    if len(planned) > qualification_limit:
        raise ExactWorkcellVariationError(
            ["cell_budget_insufficient_for_required_coverage_and_heldout_partition"]
        )
    attempts = 0
    while len(planned) < count:
        attempts += 1
        if attempts > count * 100:
            raise ExactWorkcellVariationError(["unique_cell_capacity_exhausted"])
        ordinal = len(planned)
        partition = "qualification" if ordinal < qualification_limit else "held_out"
        phase = "bounded_composed" if partition == "qualification" else "held_out_composed"
        seed = _seed_int(request.get("seed_root"), request.get("matrix_id"), "fill", attempts)
        values = dict(nominal)
        changed: list[str] = []
        for dimension_id, dimension in by_id.items():
            if _seed_int(seed, dimension_id, "include") % 5 != 0:
                values[dimension_id] = _sample_value(
                    dimension, seed=seed, salt=dimension_id
                )
                changed.append(dimension_id)
        if not changed:
            forced_id = list(by_id)[_seed_int(seed, "forced_dimension") % len(by_id)]
            values[forced_id] = _sample_value(
                by_id[forced_id], seed=seed, salt=forced_id
            )
            changed.append(forced_id)
        key = canonical_json(values)
        if key in unique:
            continue
        unique.add(key)
        planned.append((phase, partition, values, changed))

    cells = [
        _cell(
            request=request,
            ordinal=ordinal,
            phase=phase,
            partition=partition,
            values=values,
            changed_ids=changed,
        )
        for ordinal, (phase, partition, values, changed) in enumerate(planned)
    ]
    identity_digests = {row["exact_workcell_identity_digest"] for row in cells}
    if len(identity_digests) != 1:
        raise ExactWorkcellVariationError(["exact_workcell_identity_drift"])
    for dimension_id, dimension in by_id.items():
        low, _mid, high = _dimension_levels(dimension)
        one_factor_values = {
            row["resolved_values"][dimension_id]
            for row in cells
            if row["phase"].startswith("one_factor")
            and row["changed_dimension_ids"] == [dimension_id]
        }
        if one_factor_values != {low, high}:
            raise ExactWorkcellVariationError(
                [f"one_factor_coverage_unsatisfied:{dimension_id}"]
            )
    pairwise_cells = [
        row for row in cells if row["phase"] == "pairwise_covering_array"
    ]
    dimension_ids = list(by_id)
    for left_index, left in enumerate(dimension_ids):
        left_low, _left_mid, left_high = _dimension_levels(by_id[left])
        for right in dimension_ids[left_index + 1 :]:
            right_low, _right_mid, right_high = _dimension_levels(by_id[right])
            observed = {
                (
                    row["resolved_values"][left],
                    row["resolved_values"][right],
                )
                for row in pairwise_cells
            }
            required = {
                (left_low, right_low),
                (left_low, right_high),
                (left_high, right_low),
                (left_high, right_high),
            }
            if not required.issubset(observed):
                raise ExactWorkcellVariationError(
                    [f"pairwise_coverage_unsatisfied:{left}:{right}"]
                )
    matrix = {
        "schema_version": MATRIX_SCHEMA_VERSION,
        "program_id": request.get("program_id"),
        "matrix_id": request.get("matrix_id"),
        "matrix_kind": "exact_workcell_primary",
        "implementation_commit": request.get("implementation_commit"),
        "request_digest": request.get("request_digest"),
        "cell_count": len(cells),
        "canonical_anchor_cell_id": cells[0]["cell_id"],
        "policy_neutral": True,
        "dimension_contracts": dimensions,
        "required_controls": list(REQUIRED_CONTROLS),
        "cells": cells,
        "coverage": {
            "one_factor_cells": sum(row["phase"].startswith("one_factor") for row in cells),
            "pairwise_covering_array_cells": sum(
                row["phase"] == "pairwise_covering_array" for row in cells
            ),
            "targeted_interaction_cells": sum(
                row["phase"] == "targeted_interaction" for row in cells
            ),
            "held_out_composed_cells": sum(row["partition"] == "held_out" for row in cells),
            "families": sorted({_string(row.get("family")) for row in dimensions}),
            "all_dimensions_receive_low_and_high_one_factor_cells": True,
            "all_dimension_pairs_cover_low_low_low_high_high_low_high_high": True,
        },
        "agent_role": {
            "mode": "bounded_proposal" if proposal else "deterministic_fallback_no_agent",
            "proposal_digest": proposal.get("proposal_digest") if proposal else None,
            "agent_may_authorize_cells": False,
            "agent_may_widen_bounds": False,
            "deterministic_compiler_is_authority": True,
        },
        "claim_boundary": {
            "development_only": True,
            "simulator_only_until_executed": True,
            "exact_workcell_primary": True,
            "object_cousins_in_primary": False,
            "multi_site_generalization_proven": False,
            "physical_success_proven": False,
            "policy_ranking_proven": False,
        },
    }
    matrix["matrix_digest"] = canonical_digest(matrix, digest_field="matrix_digest")
    return matrix


__all__ = [
    "DEFAULT_CELL_COUNT",
    "EMBODIMENT_INPUT_SCHEMA_VERSION",
    "ExactWorkcellVariationError",
    "MATRIX_SCHEMA_VERSION",
    "REQUEST_SCHEMA_VERSION",
    "SCENE_INPUT_SCHEMA_VERSION",
    "SCHEDULE_REQUEST_SCHEMA_VERSION",
    "TASK_INPUT_SCHEMA_VERSION",
    "VariationProposalAgent",
    "build_agent_proposal_brief",
    "compile_variation_matrix",
    "seal_agent_proposal",
    "validate_variation_request",
]
