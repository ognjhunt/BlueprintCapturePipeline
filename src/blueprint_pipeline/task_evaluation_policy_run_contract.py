"""Fail-closed configuration and result contracts for paired policy runs.

The Website intentionally exposes only a small set of knobs.  This module
binds those choices to one published setup, expands them into exact shared
cells and seeds, and validates the small terminal projection.  It never
allocates a provider, executes a policy, or treats prepared bytes as evidence.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_json


SETUP_SCHEMA_VERSION = "task_evaluation_policy_run_setup.v1"
SELECTION_SCHEMA_VERSION = "task_evaluation_policy_run_selection.v1"
CONFIGURATION_SCHEMA_VERSION = "task_evaluation_policy_run_configuration.v1"
PLAN_SCHEMA_VERSION = "task_evaluation_policy_run_plan.v1"
RESULT_PROJECTION_SCHEMA_VERSION = (
    "task_evaluation_policy_run_result_projection.v1"
)
CONTROLS_QUALIFICATION_SCHEMA_VERSION = (
    "task_evaluation_policy_controls_qualification.v1"
)
ACTIVATION_MANIFEST_SCHEMA_VERSION = (
    "task_evaluation_policy_campaign_activation.v1"
)
EMBODIMENT_ID = "franka_panda_robotiq_2f85_v1"
FROZEN_CANDIDATE_IDS = ("pi05_droid", "groot_n17_droid")
MATRIX_PROFILE_ID = "franka_rigid_relocation_nested_v1"
SCENARIO_COMPILER_ID = "franka_rigid_relocation_nested_prefix"
SCENARIO_COMPILER_VERSION = "v1"
PRESET_IDS = ("quick_10", "standard_100", "deep_500")
PRESET_COUNTS = (10, 100, 500)
QUICK_FAMILY_COUNTS = {
    "canonical_anchor": 1,
    "placement_approach": 2,
    "illumination": 1,
    "camera_sensor": 1,
    "bounded_physics": 1,
    "pairwise": 2,
    "held_out": 2,
}
REQUIRED_FAMILIES = (
    "canonical_anchor",
    "placement_approach",
    "illumination",
    "camera_sensor",
    "bounded_physics",
    "pairwise",
    "held_out",
)
MAX_TOTAL_EPISODES = 2000
_SCHEMA_ROOT = Path(__file__).resolve().parents[2] / "docs" / "schemas"
SETUP_SCHEMA_PATH = _SCHEMA_ROOT / f"{SETUP_SCHEMA_VERSION}.schema.json"
SELECTION_SCHEMA_PATH = _SCHEMA_ROOT / f"{SELECTION_SCHEMA_VERSION}.schema.json"
CONFIGURATION_SCHEMA_PATH = (
    _SCHEMA_ROOT / f"{CONFIGURATION_SCHEMA_VERSION}.schema.json"
)
RESULT_PROJECTION_SCHEMA_PATH = (
    _SCHEMA_ROOT / f"{RESULT_PROJECTION_SCHEMA_VERSION}.schema.json"
)
CONTROLS_QUALIFICATION_SCHEMA_PATH = (
    _SCHEMA_ROOT / f"{CONTROLS_QUALIFICATION_SCHEMA_VERSION}.schema.json"
)


class TaskEvaluationPolicyRunContractError(ValueError):
    """A policy-run setup, request, plan, or projection is unsafe."""


@lru_cache(maxsize=5)
def _schema(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskEvaluationPolicyRunContractError(
            f"policy_run_schema_unavailable:{path.name}"
        ) from exc
    if not isinstance(value, Mapping):
        raise TaskEvaluationPolicyRunContractError(
            f"policy_run_schema_invalid:{path.name}"
        )
    import jsonschema

    try:
        jsonschema.Draft202012Validator.check_schema(value)
    except jsonschema.SchemaError as exc:
        raise TaskEvaluationPolicyRunContractError(
            f"policy_run_schema_invalid:{path.name}"
        ) from exc
    return dict(value)


def policy_run_setup_schema() -> dict[str, Any]:
    return deepcopy(_schema(SETUP_SCHEMA_PATH))


def policy_run_configuration_schema() -> dict[str, Any]:
    return deepcopy(_schema(CONFIGURATION_SCHEMA_PATH))


def policy_run_selection_schema() -> dict[str, Any]:
    return deepcopy(_schema(SELECTION_SCHEMA_PATH))


def policy_run_result_projection_schema() -> dict[str, Any]:
    return deepcopy(_schema(RESULT_PROJECTION_SCHEMA_PATH))


def policy_controls_qualification_schema() -> dict[str, Any]:
    return deepcopy(_schema(CONTROLS_QUALIFICATION_SCHEMA_PATH))


def _validate_schema(
    value: Mapping[str, Any], *, schema: Mapping[str, Any], code: str
) -> dict[str, Any]:
    import jsonschema

    copied = deepcopy(dict(value))
    validator = jsonschema.Draft202012Validator(
        schema, format_checker=jsonschema.FormatChecker()
    )
    errors = sorted(validator.iter_errors(copied), key=lambda row: list(row.path))
    if errors:
        path = ".".join(str(part) for part in errors[0].path) or "$"
        raise TaskEvaluationPolicyRunContractError(f"{code}:{path}")
    return copied


def _cell_projection(cell: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "cell_id": cell["cell_id"],
        "family": cell["family"],
        "partition": cell["partition"],
        "scored": cell["scored"],
        "cell_spec_digest": cell["cell_spec_digest"],
    }


def _cross_runtime_digest(
    value: Mapping[str, Any], *, digest_field: str | None = None
) -> str:
    normalized = dict(value)
    if digest_field:
        normalized.pop(digest_field, None)
    payload = cross_runtime_canonical_json(normalized).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _nesting_proof(preset: Mapping[str, Any]) -> str:
    return _cross_runtime_digest(
        {
            "preset_id": preset["preset_id"],
            "scenario_set_digest": preset["scenario_set_digest"],
            "parent_preset_id": preset["parent_preset_id"],
            "parent_prefix_count": preset["parent_prefix_count"],
            "selection_rule": "published_ordered_prefix",
        }
    )


def validate_policy_run_setup(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one safe public setup carried by the launch-profile catalog."""

    setup = _validate_schema(
        value,
        schema=policy_run_setup_schema(),
        code="policy_run_setup_invalid",
    )
    presets = setup["presets"]
    if [row["preset_id"] for row in presets] != list(PRESET_IDS) or [
        row["scenario_count_per_policy"] for row in presets
    ] != list(PRESET_COUNTS):
        raise TaskEvaluationPolicyRunContractError(
            "policy_run_setup_preset_order_invalid"
        )
    for index, preset in enumerate(presets):
        expected_parent = None if index == 0 else PRESET_IDS[index - 1]
        expected_prefix = 0 if index == 0 else PRESET_COUNTS[index - 1]
        if (
            preset["parent_preset_id"] != expected_parent
            or preset["parent_prefix_count"] != expected_prefix
            or preset["nesting_proof_digest"] != _nesting_proof(preset)
            or sum(preset["family_counts"].values())
            != preset["scenario_count_per_policy"]
        ):
            raise TaskEvaluationPolicyRunContractError(
                "policy_run_setup_preset_nesting_invalid"
            )
        enabled = preset["availability"] == "enabled"
        cells = preset.get("cells")
        if (
            index == 0
            and (
                not enabled
                or preset["default"] is not True
                or preset["family_counts"] != QUICK_FAMILY_COUNTS
                or not isinstance(cells, list)
                or len(cells) != 10
            )
        ) or (
            index > 0
            and (
                enabled
                or preset["default"] is not False
                or cells is not None
            )
        ):
            raise TaskEvaluationPolicyRunContractError(
                "policy_run_setup_preset_availability_invalid"
            )
        estimate = preset["estimate"]
        if estimate["status"] == "estimated" and (
            estimate["duration_minutes"]["minimum"]
            > estimate["duration_minutes"]["maximum"]
            or estimate["cost_usd"]["minimum"]
            > estimate["cost_usd"]["maximum"]
        ):
            raise TaskEvaluationPolicyRunContractError(
                "policy_run_setup_estimate_range_invalid"
            )
        if cells is None:
            continue
        cell_ids = [cell["cell_id"] for cell in cells]
        family_counts = {
            family: sum(cell["family"] == family for cell in cells)
            for family in REQUIRED_FAMILIES
        }
        if (
            len(cell_ids) != len(set(cell_ids))
            or family_counts != preset["family_counts"]
            or preset["scenario_set_digest"]
            != _cross_runtime_digest({"ordered_cells": cells})
        ):
            raise TaskEvaluationPolicyRunContractError(
                "policy_run_setup_preset_cells_invalid"
            )
        for cell in cells:
            expected_partition = (
                "held_out" if cell["family"] == "held_out" else "qualification"
            )
            if cell["partition"] != expected_partition:
                raise TaskEvaluationPolicyRunContractError(
                    "policy_run_setup_family_partition_invalid"
                )
    template = setup["preparation_template"]
    if template["template_digest"] != _cross_runtime_digest(
        template, digest_field="template_digest"
    ):
        raise TaskEvaluationPolicyRunContractError(
            "policy_run_preparation_template_digest_mismatch"
        )
    if (
        template["scene"].get("mode") != "reuse_configured_revision"
        or template["construction"] != {"mode": "reuse_configured_scene"}
        or template["controller"].get("kind") != "policy_container"
        or template["execution_adapter"].get("kind") != "native_task_arena"
        or template["execution_adapter"].get("version") != "v1"
        or template["spend"].get("retry_cap") != 0
    ):
        raise TaskEvaluationPolicyRunContractError(
            "policy_run_preparation_template_semantics_invalid"
        )
    template_request = deepcopy(template)
    template_request.pop("schema_version")
    template_request.pop("template_digest")
    publication = dict(template_request["publication"])
    publication["input_namespace"] = "template-validation"
    from .task_evaluation_launch_preparation_contract import (
        validate_launch_preparation_request,
    )

    try:
        validate_launch_preparation_request(
            {
                "schema_version": "task_evaluation_launch_preparation_request.v1",
                "run_mode": "episode_evaluation",
                "expected_production_commit": "0" * 40,
                "preparation_id": "template-validation-preparation",
                "team_namespace": "template-validation-team",
                "run_id": "template-validation-run",
                **template_request,
                "publication": publication,
            }
        )
    except ValueError as exc:
        raise TaskEvaluationPolicyRunContractError(
            f"policy_run_preparation_template_invalid:{exc}"
        ) from exc
    if setup["setup_digest"] != _cross_runtime_digest(
        setup, digest_field="setup_digest"
    ):
        raise TaskEvaluationPolicyRunContractError("policy_run_setup_digest_mismatch")
    return setup


def policy_run_setup_digest(value: Mapping[str, Any]) -> str:
    setup = dict(value)
    setup["setup_digest"] = ""
    setup["setup_digest"] = _cross_runtime_digest(
        setup, digest_field="setup_digest"
    )
    return validate_policy_run_setup(setup)["setup_digest"]


def validate_policy_run_selection(value: Mapping[str, Any]) -> dict[str, Any]:
    selection = _validate_schema(
        value,
        schema=policy_run_selection_schema(),
        code="policy_run_selection_invalid",
    )
    return selection


def _seed_for_cell(*, setup_digest: str, run_id: str, preset_id: str, cell_id: str) -> int:
    material = "\0".join((setup_digest, run_id, preset_id, cell_id)).encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big") & 0x7FFFFFFF


def compile_policy_run_configuration(
    value: Mapping[str, Any], *, setup: Mapping[str, Any]
) -> dict[str, Any]:
    """Compile the server-owned preset; callers cannot submit cells or seeds."""

    selection = validate_policy_run_selection(value)
    bound_setup = validate_policy_run_setup(setup)
    if (
        selection["setup_digest"] != bound_setup["setup_digest"]
        or selection["source_launch_id"] != bound_setup["source_launch_id"]
        or selection["offering_digest"] != bound_setup["offering_digest"]
    ):
        raise TaskEvaluationPolicyRunContractError(
            "policy_run_selection_setup_binding_mismatch"
        )
    matches = [
        preset
        for preset in bound_setup["presets"]
        if preset["preset_id"] == selection["preset_id"]
    ]
    if len(matches) != 1 or matches[0]["availability"] != "enabled":
        raise TaskEvaluationPolicyRunContractError(
            "policy_run_selection_preset_not_enabled"
        )
    preset = matches[0]
    cells = [
        {
            **deepcopy(cell),
            "seed": _seed_for_cell(
                setup_digest=bound_setup["setup_digest"],
                run_id=selection["run_id"],
                preset_id=preset["preset_id"],
                cell_id=cell["cell_id"],
            ),
        }
        for cell in preset["cells"]
    ]
    seeds = [cell["seed"] for cell in cells]
    if len(seeds) != len(set(seeds)):
        raise TaskEvaluationPolicyRunContractError(
            "policy_run_configuration_seed_collision"
        )
    scenario_count = preset["scenario_count_per_policy"]
    configuration: dict[str, Any] = {
        "schema_version": CONFIGURATION_SCHEMA_VERSION,
        "run_id": selection["run_id"],
        "source_launch_id": selection["source_launch_id"],
        "offering_digest": selection["offering_digest"],
        "setup_digest": selection["setup_digest"],
        "embodiment_id": EMBODIMENT_ID,
        "candidate_ids": list(FROZEN_CANDIDATE_IDS),
        "preset_id": preset["preset_id"],
        "scenario_count_per_policy": scenario_count,
        "compiler": deepcopy(bound_setup["scenario_compiler"]),
        "matrix": {
            "profile_id": bound_setup["matrix_profile_id"],
            "preregistration_digest": bound_setup["preregistration"]["digest"],
            "scenario_set_digest": preset["scenario_set_digest"],
            "cells": cells,
        },
        "counts": {
            "learned_episode_count": scenario_count * 2,
            "control_episode_count": scenario_count * 2,
            "total_episode_count": scenario_count * 4,
        },
        "execution_guards": {
            "candidate_cells_and_seeds_must_match": True,
            "policy_specific_scenario_changes_prohibited": True,
            "zero_action_negative_every_scored_cell": True,
            "deterministic_scripted_positive_every_scored_cell": True,
            "retry_cap": 0,
        },
        "evidence_requirements": {
            "lossless_policy_input_frames_required": True,
            "digest_bound_frame_manifest_required": True,
            "derived_review_video_required": True,
            "typed_media_gap_before_first_observation_required": True,
            "grader_authority": "deterministic_simulator_state",
            "policy_self_grading_forbidden": True,
        },
        "configuration_digest": "",
    }
    configuration["configuration_digest"] = _cross_runtime_digest(
        configuration, digest_field="configuration_digest"
    )
    return validate_policy_run_configuration(configuration)


def validate_policy_run_configuration(
    value: Mapping[str, Any], *, setup: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Validate exact cells/seeds expanded by the authenticated Website."""

    configuration = _validate_schema(
        value,
        schema=policy_run_configuration_schema(),
        code="policy_run_configuration_invalid",
    )
    if configuration["configuration_digest"] != _cross_runtime_digest(
        configuration, digest_field="configuration_digest"
    ):
        raise TaskEvaluationPolicyRunContractError(
            "policy_run_configuration_digest_mismatch"
        )
    cells = configuration["matrix"]["cells"]
    scenario_count = configuration["scenario_count_per_policy"]
    if (
        len(cells) != scenario_count
        or configuration["counts"]
        != {
            "learned_episode_count": scenario_count * 2,
            "control_episode_count": scenario_count * 2,
            "total_episode_count": scenario_count * 4,
        }
        or scenario_count * 4 > MAX_TOTAL_EPISODES
    ):
        raise TaskEvaluationPolicyRunContractError(
            "policy_run_configuration_episode_counts_invalid"
        )
    if len({cell["seed"] for cell in cells}) != len(cells):
        raise TaskEvaluationPolicyRunContractError(
            "policy_run_configuration_seeds_not_unique"
        )
    if setup is not None:
        expected = compile_policy_run_configuration(
            {
                "schema_version": SELECTION_SCHEMA_VERSION,
                "run_id": configuration["run_id"],
                "source_launch_id": configuration["source_launch_id"],
                "offering_digest": configuration["offering_digest"],
                "setup_digest": configuration["setup_digest"],
                "preset_id": configuration["preset_id"],
            },
            setup=setup,
        )
        if configuration != expected:
            raise TaskEvaluationPolicyRunContractError(
                "policy_run_configuration_not_compiler_output"
            )
    return configuration


def policy_run_configuration_digest(value: Mapping[str, Any]) -> str:
    configuration = dict(value)
    configuration["configuration_digest"] = ""
    configuration["configuration_digest"] = _cross_runtime_digest(
        configuration, digest_field="configuration_digest"
    )
    return validate_policy_run_configuration(configuration)["configuration_digest"]


def build_policy_run_plan(
    value: Mapping[str, Any], *, setup: Mapping[str, Any]
) -> dict[str, Any]:
    """Compile one no-spend, no-execution queue plan from exact setup bytes."""

    configuration = validate_policy_run_configuration(value, setup=setup)
    cells = configuration["matrix"]["cells"]
    plan: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "source_launch_id": configuration["source_launch_id"],
        "offering_digest": configuration["offering_digest"],
        "configuration_digest": configuration["configuration_digest"],
        "setup_digest": configuration["setup_digest"],
        "embodiment_id": EMBODIMENT_ID,
        "candidate_ids": list(FROZEN_CANDIDATE_IDS),
        "matrix_profile_id": MATRIX_PROFILE_ID,
        "preset_id": configuration["preset_id"],
        "cells": deepcopy(cells),
        "counts": {
            "scored_cell_count": len(cells),
            "scenarios_per_policy": configuration["scenario_count_per_policy"],
            "candidate_episode_count": configuration["counts"][
                "learned_episode_count"
            ],
            "control_episode_count": configuration["counts"][
                "control_episode_count"
            ],
            "total_episode_count": configuration["counts"][
                "total_episode_count"
            ],
        },
        "campaign_units": [
            {
                "campaign_unit_id": f"{configuration['run_id']}-{cell['cell_id']}",
                "cell_id": cell["cell_id"],
                "seed": cell["seed"],
                "candidate_ids": list(FROZEN_CANDIDATE_IDS),
                "runtime_contract": "native_task_arena_policy_campaign.v1",
            }
            for cell in cells
        ],
        "execution_guards": deepcopy(configuration["execution_guards"]),
        "evidence_requirements": deepcopy(configuration["evidence_requirements"]),
        "status": "prepared_awaiting_controls_qualified_activation",
        "execution_performed": False,
        "provider_mutation_performed": False,
        "paid_execution_requested": False,
        "spend_usd": 0,
        "blockers": ["controls_qualified_activation_required"],
        "plan_digest": "",
    }
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    return plan


def validate_policy_controls_qualification(
    value: Mapping[str, Any], *, configuration: Mapping[str, Any], plan: Mapping[str, Any]
) -> dict[str, Any]:
    """Require both deterministic controls to pass for every compiled cell."""

    qualification = _validate_schema(
        value,
        schema=policy_controls_qualification_schema(),
        code="policy_controls_qualification_invalid",
    )
    if qualification["qualification_digest"] != canonical_digest(
        qualification, digest_field="qualification_digest"
    ):
        raise TaskEvaluationPolicyRunContractError(
            "policy_controls_qualification_digest_mismatch"
        )
    if (
        qualification["configuration_digest"]
        != configuration["configuration_digest"]
        or qualification["plan_digest"] != plan["plan_digest"]
        or [
            {"cell_id": row["cell_id"], "seed": row["seed"]}
            for row in qualification["cells"]
        ]
        != [
            {"cell_id": row["cell_id"], "seed": row["seed"]}
            for row in configuration["matrix"]["cells"]
        ]
    ):
        raise TaskEvaluationPolicyRunContractError(
            "policy_controls_qualification_matrix_mismatch"
        )
    from .adp009d_control_episode import REQUIRED_CONTROLS

    for row in qualification["cells"]:
        controls = row["controls_result"]
        pair = controls.get("control_pair")
        pair_controls = pair.get("controls") if isinstance(pair, Mapping) else None
        if (
            controls.get("schema_version")
            != "native_task_arena_control_result.v1"
            or controls.get("status") != "completed"
            or controls.get("controls_qualified") is not True
            or controls.get("blockers") != []
            or controls.get("candidate_policy_queried") is not False
            or controls.get("result_digest")
            != canonical_digest(controls, digest_field="result_digest")
            or not isinstance(pair, Mapping)
            or pair.get("schema_version") != "adp_task_control_pair.v1"
            or pair.get("cell_id") != row["cell_id"]
            or pair.get("execution_order") != list(REQUIRED_CONTROLS)
            or pair.get("cell_admitted_for_policy_execution") is not True
            or pair.get("policy_execution_blockers") != []
            or pair.get("candidate_policy_queried") is not False
            or pair.get("pair_digest")
            != canonical_digest(pair, digest_field="pair_digest")
            or not isinstance(pair_controls, list)
            or len(pair_controls) != len(REQUIRED_CONTROLS)
            or any(
                not isinstance(control, Mapping)
                or control.get("control_id") != control_id
                or control.get("control_passed") is not True
                or not isinstance(control.get("receipt_digest"), str)
                or not str(control["receipt_digest"]).startswith("sha256:")
                or len(str(control["receipt_digest"])) != 71
                for control, control_id in zip(
                    pair_controls, REQUIRED_CONTROLS, strict=True
                )
            )
        ):
            raise TaskEvaluationPolicyRunContractError(
                "policy_controls_qualification_cell_result_invalid"
            )
    return qualification


def build_policy_campaign_activation_manifest(
    *,
    configuration: Mapping[str, Any],
    plan: Mapping[str, Any],
    controls_qualification: Mapping[str, Any],
) -> dict[str, Any]:
    """Materialize N exact two-member campaign units without executing them."""

    configuration = validate_policy_run_configuration(configuration)
    if (
        plan.get("schema_version") != PLAN_SCHEMA_VERSION
        or plan.get("configuration_digest") != configuration["configuration_digest"]
        or plan.get("plan_digest")
        != canonical_digest(plan, digest_field="plan_digest")
        or plan.get("campaign_units") is None
    ):
        raise TaskEvaluationPolicyRunContractError(
            "policy_campaign_activation_plan_invalid"
        )
    qualification = validate_policy_controls_qualification(
        controls_qualification, configuration=configuration, plan=plan
    )
    if tuple(configuration["candidate_ids"]) != FROZEN_CANDIDATE_IDS:
        raise TaskEvaluationPolicyRunContractError(
            "policy_campaign_activation_member_pair_invalid"
        )
    controls = {row["cell_id"]: row for row in qualification["cells"]}
    units = []
    for plan_unit in plan["campaign_units"]:
        control = controls[plan_unit["cell_id"]]
        pair_controls = control["controls_result"]["control_pair"]["controls"]
        units.append(
            {
                **deepcopy(plan_unit),
                "controls": {
                    "zero_action_result_digest": pair_controls[0][
                        "receipt_digest"
                    ],
                    "scripted_positive_result_digest": pair_controls[1][
                        "receipt_digest"
                    ],
                    "controls_result_digest": control["controls_result"][
                        "result_digest"
                    ],
                },
                "maximum_automatic_retries": 0,
                "provider_allocation_authorized": False,
            }
        )
    manifest: dict[str, Any] = {
        "schema_version": ACTIVATION_MANIFEST_SCHEMA_VERSION,
        "run_id": configuration["run_id"],
        "configuration_digest": configuration["configuration_digest"],
        "plan_digest": plan["plan_digest"],
        "controls_qualification_digest": qualification["qualification_digest"],
        "candidate_ids": list(FROZEN_CANDIDATE_IDS),
        "campaign_unit_count": len(units),
        "campaign_units": units,
        "status": "paired_campaign_queue_materialized_no_execution",
        "provider_mutation_performed": False,
        "paid_execution_requested": False,
        "activation_digest": "",
    }
    manifest["activation_digest"] = canonical_digest(
        manifest, digest_field="activation_digest"
    )
    return manifest


def expand_policy_run_preparation_request(
    *,
    setup: Mapping[str, Any],
    selection: Mapping[str, Any],
    expected_production_commit: str,
    team_namespace: str,
    run_id: str,
    preparation_id: str,
) -> dict[str, Any]:
    """Expand a compact authenticated Website choice into the existing intake.

    The template is catalog-owned.  Team identity and notification routing are
    not accepted here as user-authored configuration; the authenticated WebApp
    supplies the team argument and retains the notification recipient itself.
    """

    bound_setup = validate_policy_run_setup(setup)
    bound_selection = validate_policy_run_selection(selection)
    if bound_selection["run_id"] != run_id:
        raise TaskEvaluationPolicyRunContractError(
            "policy_run_selection_run_id_mismatch"
        )
    bound_configuration = compile_policy_run_configuration(
        bound_selection, setup=bound_setup
    )
    template = deepcopy(bound_setup["preparation_template"])
    template.pop("schema_version")
    template.pop("template_digest")
    namespace_suffix = hashlib.sha256(
        f"{team_namespace}\0{run_id}".encode("utf-8")
    ).hexdigest()[:16]
    publication = dict(template["publication"])
    publication["input_namespace"] = (
        f"{team_namespace[:64]}-{run_id[:96]}-{namespace_suffix}"
    )
    request = {
        "schema_version": "task_evaluation_launch_preparation_request.v1",
        "run_mode": "episode_evaluation",
        "expected_production_commit": expected_production_commit,
        "preparation_id": preparation_id,
        "team_namespace": team_namespace,
        "run_id": run_id,
        **template,
        "publication": publication,
        "policy_run_setup": bound_setup,
        "policy_run_selection": bound_selection,
        "policy_run_configuration": bound_configuration,
    }
    from .task_evaluation_launch_preparation_contract import (
        validate_launch_preparation_request,
    )

    return validate_launch_preparation_request(request)


def validate_policy_run_result_projection(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the secret-clean paired result sent through terminal sync."""

    result = _validate_schema(
        value,
        schema=policy_run_result_projection_schema(),
        code="policy_run_result_projection_invalid",
    )
    if result["projection_digest"] != policy_run_result_projection_digest(result):
        raise TaskEvaluationPolicyRunContractError(
            "policy_run_result_projection_digest_mismatch"
        )
    candidate_results = result["candidate_results"]
    if [row["candidate_id"] for row in candidate_results] != list(
        FROZEN_CANDIDATE_IDS
    ):
        raise TaskEvaluationPolicyRunContractError(
            "policy_run_result_projection_candidate_order_invalid"
        )
    matrix = result["matrix"]
    expected_total = (
        matrix["candidate_episode_count"] + matrix["control_episode_count"]
    )
    if matrix["expected_episode_count"] != expected_total:
        raise TaskEvaluationPolicyRunContractError(
            "policy_run_result_projection_episode_count_invalid"
        )
    if (
        matrix["control_episode_count"] != matrix["scored_cell_count"] * 2
        or matrix["candidate_episode_count"]
        != matrix["scored_cell_count"] * len(FROZEN_CANDIDATE_IDS)
    ):
        raise TaskEvaluationPolicyRunContractError(
            "policy_run_result_projection_matrix_count_invalid"
        )
    per_candidate = matrix["candidate_episode_count"] // 2
    for candidate in candidate_results:
        if set(candidate["family_metrics"]) != set(REQUIRED_FAMILIES):
            raise TaskEvaluationPolicyRunContractError(
                "policy_run_result_projection_family_coverage_invalid"
            )
        for metric in candidate["family_metrics"].values():
            attempted = metric["attempted"]
            succeeded = metric["succeeded"]
            if succeeded > attempted or (
                attempted > 0
                and abs(metric["success_rate"] - succeeded / attempted) > 1e-9
            ):
                raise TaskEvaluationPolicyRunContractError(
                    "policy_run_result_projection_family_metric_invalid"
                )
        if sum(
            metric["attempted"] for metric in candidate["family_metrics"].values()
        ) != per_candidate:
            raise TaskEvaluationPolicyRunContractError(
                "policy_run_result_projection_family_attempt_count_invalid"
            )
        if candidate["episodes_completed"] > per_candidate:
            raise TaskEvaluationPolicyRunContractError(
                "policy_run_result_projection_candidate_episode_count_invalid"
            )
        evidence = candidate["evidence"]
        if (
            evidence["lossless_frame_manifest_count"]
            > candidate["episodes_completed"]
            or evidence["review_video_count"] > candidate["episodes_completed"]
        ):
            raise TaskEvaluationPolicyRunContractError(
                "policy_run_result_projection_evidence_count_invalid"
            )
    if result["state"] == "decided":
        if (
            result["blockers"]
            or matrix["completed_episode_count"] != matrix["expected_episode_count"]
            or matrix["controls_complete"] is not True
            or any(
                row["episodes_completed"] != per_candidate
                or row["evidence"]["lossless_frame_manifest_count"] != per_candidate
                or row["evidence"]["review_video_count"] != per_candidate
                or row["evidence"]["typed_media_gap_count"] != 0
                for row in candidate_results
            )
            or result["paired_comparison"]["matched_episode_pairs"]
            != per_candidate
        ):
            raise TaskEvaluationPolicyRunContractError(
                "policy_run_result_projection_decision_evidence_incomplete"
            )
    elif not result["blockers"]:
        raise TaskEvaluationPolicyRunContractError(
            "policy_run_result_projection_nondecision_blocker_missing"
        )
    return result


def policy_run_result_projection_digest(value: Mapping[str, Any]) -> str:
    """Digest a Pipeline-to-WebApp projection with shared JSON number semantics."""

    return _cross_runtime_digest(value, digest_field="projection_digest")


__all__ = [
    "CONFIGURATION_SCHEMA_PATH",
    "CONFIGURATION_SCHEMA_VERSION",
    "CONTROLS_QUALIFICATION_SCHEMA_PATH",
    "CONTROLS_QUALIFICATION_SCHEMA_VERSION",
    "EMBODIMENT_ID",
    "FROZEN_CANDIDATE_IDS",
    "MATRIX_PROFILE_ID",
    "MAX_TOTAL_EPISODES",
    "PLAN_SCHEMA_VERSION",
    "PRESET_COUNTS",
    "PRESET_IDS",
    "QUICK_FAMILY_COUNTS",
    "REQUIRED_FAMILIES",
    "RESULT_PROJECTION_SCHEMA_PATH",
    "RESULT_PROJECTION_SCHEMA_VERSION",
    "SETUP_SCHEMA_PATH",
    "SETUP_SCHEMA_VERSION",
    "SELECTION_SCHEMA_PATH",
    "SELECTION_SCHEMA_VERSION",
    "TaskEvaluationPolicyRunContractError",
    "build_policy_run_plan",
    "build_policy_campaign_activation_manifest",
    "compile_policy_run_configuration",
    "expand_policy_run_preparation_request",
    "policy_run_configuration_digest",
    "policy_controls_qualification_schema",
    "policy_run_configuration_schema",
    "policy_run_result_projection_schema",
    "policy_run_result_projection_digest",
    "policy_run_selection_schema",
    "policy_run_setup_digest",
    "policy_run_setup_schema",
    "validate_policy_run_configuration",
    "validate_policy_controls_qualification",
    "validate_policy_run_result_projection",
    "validate_policy_run_selection",
    "validate_policy_run_setup",
]
