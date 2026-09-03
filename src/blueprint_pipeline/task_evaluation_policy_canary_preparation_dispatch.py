"""No-spend diversion from a Website launch into canary preparation."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .adp_task_scoring import (
    confirmed_rigid_task_success_contract_matches_published,
)
from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from .droid_policy_canary_embodiment import DROID_POLICY_CANARY_PRESET_ID
from .task_evaluation_launch_preparation_queue import stage_launch_preparation_request
from .task_evaluation_launch_preparation_contract import (
    validate_launch_preparation_request,
)
from .task_evaluation_launch_webapp_sync import sync_launch_receipt_to_webapp
from .task_evaluation_policy_canary_setup import validate_policy_canary_setup
from .task_evaluation_policy_run_contract import (
    TaskEvaluationPolicyRunContractError,
    expand_policy_run_preparation_request,
    validate_policy_run_setup,
)


PLAN_SCHEMA_VERSION = "task_evaluation_policy_canary_execution_plan.v1"
SELECTION_SCHEMA_VERSION = "task_evaluation_policy_canary_launch_selection.v1"
RUN_KIND = "internal_policy_canary"
CLAIM_CEILING = "diagnostic_policy_execution"
CANDIDATES = ("pi05_droid", "groot_n17_droid")
ROBOT_PRESET = DROID_POLICY_CANARY_PRESET_ID
RECEIPT_SCHEMA_VERSION = "task_evaluation_launch_receipt.v1"


class PolicyCanaryPreparationDispatchError(ValueError):
    pass


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _digest(value: Any) -> bool:
    return bool(re.fullmatch(r"sha256:[0-9a-f]{64}", str(value or "")))


def _reference(value: Any) -> bool:
    row = _mapping(value)
    return bool(
        str(row.get("uri") or "")
        and _digest(row.get("digest"))
        and isinstance(row.get("size_bytes"), int)
        and not isinstance(row.get("size_bytes"), bool)
        and row["size_bytes"] > 0
    )


def _validate_activation_automation(value: Any) -> dict[str, Any]:
    automation = _mapping(value)
    lineage = _mapping(automation.get("lineage"))
    authorization = _mapping(automation.get("authorization_template"))
    requested = _mapping(automation.get("requested_mutations"))
    required_lineage = {
        "kind",
        "prior_authority",
        "prior_result",
        "prior_launch_receipt",
        "prior_webapp_sync",
        "prior_provider_zero",
        "prior_spend_reconciliation",
        "construction_result",
    }
    if (
        set(automation)
        != {
            "mode",
            "release_window_template",
            "lineage",
            "authorization_template",
            "requested_mutations",
        }
        or automation.get("mode") != "automatic_after_no_spend_compilation"
        or not _reference(automation.get("release_window_template"))
        or set(lineage) != required_lineage
        or lineage.get("kind") != "predecessor"
        or any(not _reference(lineage.get(name)) for name in required_lineage - {"kind"})
        or set(authorization)
        != {"reference", "authorized_by", "profile_revision", "valid_for_seconds"}
        or any(
            not str(authorization.get(name) or "")
            for name in ("reference", "authorized_by", "profile_revision")
        )
        or not isinstance(authorization.get("valid_for_seconds"), int)
        or isinstance(authorization.get("valid_for_seconds"), bool)
        or not 300 <= authorization["valid_for_seconds"] <= 86_400
        or requested
        != {
            "profile_publication": False,
            "catalog_synchronization": False,
            "standing_authorization": False,
            "policy_campaign_queue": True,
        }
    ):
        raise PolicyCanaryPreparationDispatchError(
            "policy_canary_activation_automation_invalid"
        )
    return automation


def validate_policy_canary_execution_plan(
    value: Mapping[str, Any], *, public_setup: Mapping[str, Any]
) -> dict[str, Any]:
    plan = json.loads(json.dumps(dict(value), allow_nan=False))
    setup = validate_policy_canary_setup(public_setup)
    try:
        legacy = validate_policy_run_setup(_mapping(plan.get("legacy_policy_run_setup")))
    except TaskEvaluationPolicyRunContractError as exc:
        raise PolicyCanaryPreparationDispatchError(
            "policy_canary_execution_plan_legacy_setup_invalid"
        ) from exc
    resolved = plan.get("resolved_scenarios")
    template = _mapping(plan.get("preparation_template"))
    template_controller = _mapping(template.get("controller"))
    resource = _mapping(plan.get("resource_authority"))
    expected_plan_fields = {
        "schema_version",
        "source_commit",
        "configured_source_launch_id",
        "configured_offering_configuration_run_id",
        "scene_revision_digest",
        "public_setup_digest",
        "task_success_contract",
        "task_success_contract_digest",
        "configured_preparation_request_digest",
        "policy_controller_configuration",
        "model_rights",
        "resolved_scenarios",
        "legacy_policy_run_setup",
        "preparation_template",
        "resource_authority",
        "activation_automation",
        "lineage_aliases",
        "provider_mutation_performed",
        "paid_execution_requested",
        "plan_digest",
    }
    if (
        set(plan) not in (expected_plan_fields, expected_plan_fields | {"scene_id"})
        or plan.get("schema_version") != PLAN_SCHEMA_VERSION
        or not re.fullmatch(r"[0-9a-f]{40}", str(plan.get("source_commit") or ""))
        or plan.get("configured_source_launch_id") != setup["source_launch_id"]
        or (
            plan.get("scene_id") is not None
            and not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", str(plan["scene_id"])
            )
        )
        or not str(plan.get("configured_offering_configuration_run_id") or "")
        or plan.get("scene_revision_digest") != setup["scene_revision_digest"]
        or plan.get("public_setup_digest") != setup["setup_digest"]
        or plan.get("task_success_contract") != setup["task_success_contract"]
        or plan.get("task_success_contract_digest")
        != setup["task_success_contract_digest"]
        or not _digest(plan.get("configured_preparation_request_digest"))
        or not _reference(plan.get("policy_controller_configuration"))
        or not _reference(plan.get("model_rights"))
        or template != legacy["preparation_template"]
        or template_controller.get("kind") != "policy_container"
        or not _reference(template_controller.get("configuration"))
        or template_controller.get("configuration")
        == plan.get("policy_controller_configuration")
        or template_controller.get("model_or_asset_rights")
        != plan.get("model_rights")
        or not isinstance(resolved, list)
        or len(resolved) != 10
        or resource.get("maximum_provider_allocations") != 1
        or resource.get("retry_cap") != 0
        or plan.get("provider_mutation_performed") is not False
        or plan.get("paid_execution_requested") is not False
        or plan.get("plan_digest") != canonical_digest(plan, digest_field="plan_digest")
    ):
        raise PolicyCanaryPreparationDispatchError(
            "policy_canary_execution_plan_invalid"
        )
    _validate_activation_automation(plan.get("activation_automation"))
    quick = legacy["presets"][0]
    if quick.get("preset_id") != "quick_10" or quick.get("availability") != "enabled":
        raise PolicyCanaryPreparationDispatchError(
            "policy_canary_execution_plan_quick10_invalid"
        )
    legacy_cells = quick.get("cells") or []
    public_cells = setup["episode_presets"][0]["matrix"]["cells"]
    for index, (full, legacy_cell, visible) in enumerate(
        zip(resolved, legacy_cells, public_cells, strict=True)
    ):
        if (
            full.get("cell_id") != legacy_cell.get("cell_id")
            or full.get("family") != legacy_cell.get("family")
            or full.get("seed") != legacy_cell.get("seed")
            or full.get("cell_spec_digest") != legacy_cell.get("cell_spec_digest")
            or full.get("resolved_scenario") != legacy_cell.get("resolved_scenario")
            or full.get("cell_id") != visible["cell_id"]
            or full.get("family") != visible["family"]
            or full.get("seed") != visible["seed"]
            or full.get("cell_spec_digest") != visible["cell_digest"]
            or not isinstance(full.get("resolved_scenario"), Mapping)
            or full.get("cell_spec_digest")
            != cross_runtime_canonical_digest(full["resolved_scenario"])
        ):
            raise PolicyCanaryPreparationDispatchError(
                f"policy_canary_execution_plan_cell_invalid:{index}"
            )
    return plan


def policy_canary_execution_plan_blockers(profile: Mapping[str, Any]) -> list[str]:
    setup = profile.get("internal_policy_canary_setup")
    plan = profile.get("internal_policy_canary_execution_plan")
    if setup is None and plan is None:
        return []
    if not isinstance(setup, Mapping) or not isinstance(plan, Mapping):
        return ["launch_profile_policy_canary_execution_plan_missing"]
    try:
        validated = validate_policy_canary_execution_plan(plan, public_setup=setup)
    except (PolicyCanaryPreparationDispatchError, ValueError):
        return ["launch_profile_policy_canary_execution_plan_invalid"]
    if (
        validated["source_commit"] != profile.get("source_commit")
        or validated["configured_source_launch_id"]
        != profile.get("configured_source_launch_id")
    ):
        return ["launch_profile_policy_canary_execution_plan_binding_mismatch"]
    return []


def _validate_selection(
    value: Any, *, setup: Mapping[str, Any], plan: Mapping[str, Any]
) -> dict[str, Any]:
    request = _mapping(value)
    episode_plan = _mapping(request.get("episode_plan"))
    spend = _mapping(_mapping(request.get("authorization")).get("spend"))
    selection = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "run_kind": request.get("run_kind"),
        "claim_ceiling": request.get("claim_ceiling"),
        "public_setup_digest": request.get("setup_digest"),
        "scene_revision_digest": request.get("scene_revision_digest"),
        "robot_preset_id": request.get("robot_preset_id"),
        "candidate_ids": request.get("policy_candidate_ids"),
        "preset_id": request.get("preset_id"),
        "matrix_digest": episode_plan.get("variation_matrix_digest"),
        "cells": [
            {key: row.get(key) for key in ("cell_id", "seed", "cell_digest")}
            for row in episode_plan.get("resolved_cells") or []
            if isinstance(row, Mapping)
        ],
        "capture_session_id": plan.get("configured_source_launch_id"),
        "intake_id": plan.get("configured_offering_configuration_run_id"),
        "team_namespace": request.get("team_namespace"),
        "resource_authority": {
            "hard_cap_usd": spend.get("max_spend_usd"),
            "hard_ttl_seconds": spend.get("hard_ttl_seconds"),
        },
        "task_success_contract": request.get("task_success_contract"),
        "task_success_contract_digest": request.get(
            "task_success_contract_digest"
        ),
        "notification": request.get("notification"),
        "episode_interpretation_authority": request.get(
            "episode_interpretation_authority"
        ),
        "episode_interpretation_source_rights_admission": request.get(
            "episode_interpretation_source_rights_admission"
        ),
    }
    quick = setup["episode_presets"][0]
    matrix = quick["matrix"]
    resource = plan["resource_authority"]
    expected_cells = [
        {key: row[key] for key in ("cell_id", "seed", "cell_digest")}
        for row in matrix["cells"]
    ]
    notification = _mapping(selection.get("notification"))
    selected_resource = _mapping(selection.get("resource_authority"))
    if (
        selection.get("schema_version") != SELECTION_SCHEMA_VERSION
        or selection.get("run_kind") != RUN_KIND
        or selection.get("claim_ceiling") != CLAIM_CEILING
        or selection.get("public_setup_digest") != setup["setup_digest"]
        or selection.get("scene_revision_digest") != setup["scene_revision_digest"]
        or selection.get("robot_preset_id") != ROBOT_PRESET
        or tuple(selection.get("candidate_ids") or ()) != CANDIDATES
        or selection.get("preset_id") != "quick_10"
        or selection.get("matrix_digest") != matrix["matrix_digest"]
        or selection.get("cells") != expected_cells
        or selection.get("capture_session_id") != plan["configured_source_launch_id"]
        or selection.get("intake_id")
        != plan["configured_offering_configuration_run_id"]
        or not str(selection.get("team_namespace") or "")
        or notification.get("notify_on") != ["completed", "blocked", "cancelled"]
        or not str(notification.get("email") or "")
        or selected_resource.get("hard_cap_usd") != resource["hard_cap_usd"]
        or selected_resource.get("hard_ttl_seconds") != resource["hard_ttl_seconds"]
        or selection.get("task_success_contract_digest")
        != _mapping(selection.get("task_success_contract")).get("contract_digest")
        or not confirmed_rigid_task_success_contract_matches_published(
            published=setup["task_success_contract"],
            selected=_mapping(selection.get("task_success_contract")),
        )
    ):
        raise PolicyCanaryPreparationDispatchError(
            "policy_canary_launch_selection_invalid"
        )
    return selection


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(dict(value), sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        if path.read_bytes() != payload:
            raise PolicyCanaryPreparationDispatchError(
                f"policy_canary_launch_immutable_conflict:{path.name}"
            )


def maybe_dispatch_policy_canary_preparation(
    *,
    request: Mapping[str, Any],
    profile: Mapping[str, Any],
    blockers: list[str],
    state_root: str | Path,
    preparation_queue_root: str | Path | None,
) -> dict[str, Any] | None:
    explicit = request.get("run_kind") == RUN_KIND
    configured = profile.get("internal_policy_canary_setup") is not None
    if not explicit and not configured:
        return None
    launch_id = str(request.get("launch_id") or "invalid")
    run_root = Path(state_root).expanduser().resolve() / launch_id
    run_root.mkdir(parents=True, exist_ok=True)
    prior = run_root / "launch_receipt.json"
    if prior.is_file():
        value = json.loads(prior.read_text(encoding="utf-8"))
        if value.get("request_digest") != request.get("request_digest"):
            raise PolicyCanaryPreparationDispatchError(
                "launch_receipt_request_binding_mismatch"
            )
        return value
    _write_immutable(run_root / "launch_request.json", request)
    _write_immutable(run_root / "launch_profile.json", profile)
    queue_receipt: dict[str, Any] | None = None
    local_blockers = list(blockers)
    try:
        if local_blockers:
            raise PolicyCanaryPreparationDispatchError(
                "policy_canary_launch_prevalidation_blocked"
            )
        setup = validate_policy_canary_setup(
            _mapping(profile.get("internal_policy_canary_setup"))
        )
        plan = validate_policy_canary_execution_plan(
            _mapping(profile.get("internal_policy_canary_execution_plan")),
            public_setup=setup,
        )
        selection = _validate_selection(request, setup=setup, plan=plan)
        preparation_queue_root = preparation_queue_root or os.getenv(
            "BLUEPRINT_TASK_EVALUATION_LAUNCH_PREPARATION_QUEUE_ROOT"
        )
        if preparation_queue_root is None:
            raise PolicyCanaryPreparationDispatchError(
                "policy_canary_preparation_queue_root_missing"
            )
        preparation_id = "policy-canary-" + hashlib.sha256(
            f"{launch_id}\0{request.get('request_digest')}".encode()
        ).hexdigest()[:32]
        legacy_setup = plan["legacy_policy_run_setup"]
        legacy_selection = {
            "schema_version": "task_evaluation_policy_run_selection.v1",
            "run_id": request["run_id"],
            "source_launch_id": legacy_setup["source_launch_id"],
            "offering_digest": legacy_setup["offering_digest"],
            "setup_digest": legacy_setup["setup_digest"],
            "preset_id": "quick_10",
            "run_kind": RUN_KIND,
            "claim_ceiling": CLAIM_CEILING,
            "scene_revision_digest": setup["scene_revision_digest"],
            "scene_controls_status_at_submission": "configured_controls_pending",
            "robot_preset_id": ROBOT_PRESET,
            "policy_candidate_ids": list(CANDIDATES),
            "notification": selection["notification"],
            "website_request_digest": request["request_digest"],
            "task_success_contract": selection["task_success_contract"],
            "task_success_contract_digest": selection[
                "task_success_contract_digest"
            ],
        }
        preparation = expand_policy_run_preparation_request(
            setup=legacy_setup,
            selection=legacy_selection,
            expected_production_commit=plan["source_commit"],
            team_namespace=selection["team_namespace"],
            run_id=request["run_id"],
            preparation_id=preparation_id,
        )
        preparation["policy_canary_activation"] = {
            **plan["activation_automation"],
            **(
                {
                    "episode_interpretation_authority": selection[
                        "episode_interpretation_authority"
                    ],
                    "episode_interpretation_source_rights_admission": selection[
                        "episode_interpretation_source_rights_admission"
                    ],
                }
                if isinstance(
                    selection.get("episode_interpretation_authority"), Mapping
                )
                and isinstance(
                    selection.get("episode_interpretation_source_rights_admission"),
                    Mapping,
                )
                else {}
            ),
        }
        preparation = validate_launch_preparation_request(preparation)
        queue_receipt = stage_launch_preparation_request(
            value=preparation,
            queue_root=preparation_queue_root,
            submitted_by=str(_mapping(_mapping(request.get("authorization")).get("actor")).get("id")),
        )
    except (KeyError, TypeError, ValueError) as exc:
        local_blockers.append(str(exc) or type(exc).__name__)
    status = "blocked" if local_blockers else "queued_for_no_spend_preparation"
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": status,
        "launch_id": request.get("launch_id"),
        "run_id": request.get("run_id"),
        "request_digest": request.get("request_digest"),
        "launch_profile_digest": profile.get("profile_digest"),
        "canonical_allocator": "python -m blueprint_pipeline.paid_resource_allocator gpu-canary",
        "allocator_exit_code": None,
        "allocator_invoked": False,
        "execute_requested": False,
        "website_execute_intent_received": True,
        "provider_mutation_attempted": False,
        "provider_mutation_evidence": {"status": "absent_before_paid_admission"},
        "preparation_queue": (
            {
                key: queue_receipt.get(key)
                for key in (
                    "status",
                    "accepted",
                    "already_exists",
                    "preparation_id",
                    "request_digest",
                    "receipt_digest",
                )
            }
            if queue_receipt
            else None
        ),
        "terminal_evidence": {
            "status": "awaiting_no_spend_preparation" if queue_receipt else "blocked",
            "provider_allocation_performed": False,
        },
        "blockers": sorted(set(local_blockers)),
        "raw_secret_values_recorded": False,
        "agent_operator_used": False,
        "claim_ceiling": request.get("claim_ceiling"),
        "receipt_digest_canonicalization": "rfc8785",
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = cross_runtime_canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    _write_immutable(prior, receipt)
    sync_launch_receipt_to_webapp(receipt=receipt)
    return receipt


__all__ = [
    "PLAN_SCHEMA_VERSION",
    "PolicyCanaryPreparationDispatchError",
    "maybe_dispatch_policy_canary_preparation",
    "policy_canary_execution_plan_blockers",
    "validate_policy_canary_execution_plan",
]
