"""Read-only controls cancellation and retained-placement evidence contracts.

The mutation services import these same validators. CPU intake and spend readers
therefore retain all receipt/owner/phase checks without importing execution code.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest

UNSTARTED_SCHEMA = "task_evaluation_unstarted_controls_cancellation.v1"
DIRECTORY = "cancelled-unstarted-controls"
PLACEMENT_SCHEMA = "task_evaluation_completed_placement_adoption.v1"
PLACEMENT_CANCEL_SCHEMA = "task_evaluation_unused_native_plan_cancellation.v1"
VISUAL_SCHEMA = "task_evaluation_visual_review_continuation.v1"
VISUAL_RETIREMENT_SCHEMA = "task_evaluation_unstarted_native_after_visual_review.v1"
TERMINAL_RETIREMENT_SCHEMA = "task_evaluation_unmaterialized_adoption_cancellation.v1"
REVIEW_CAP = .15
NATIVE_CAP = .40
MAX_INPUT_TOKENS = 12000
MAX_OUTPUT_TOKENS = 4096


def _terminal_require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError("controls_autoprovision_" + code)

def _read(path: Path) -> dict[str, Any]:
    if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("unstarted_controls_evidence_unsafe")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("unstarted_controls_evidence_invalid")
    return value


def _file(path: Path) -> dict[str, Any]:
    _read(path)
    return {"path": str(path), "digest": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()}


def _placement_require(condition: Any, code: str) -> None:
    if not condition:
        raise ValueError("completed_placement_adoption_" + code)


def _placement_ref(ref: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(ref["path"])
    _placement_require(_file(path) == dict(ref), "reference_changed")
    return _read(path)


def validate_placement_adoption(
    value: Mapping[str, Any], *, expected_owner_digest: str | None = None
) -> dict[str, Any]:
    from . import task_evaluation_configured_controls_autostart_validation as auto
    from .configured_controls_plan_validation import read_configured_controls_plan as _plan
    from .task_evaluation_robot_placement_agent import validate_robot_placement_receipt

    _placement_require(
        value.get("schema_version") == PLACEMENT_SCHEMA
        and value.get("adoption_digest") == canonical_digest(value, digest_field="adoption_digest"),
        "invalid",
    )
    intent = auto.validate_configured_controls_autostart_intent(_placement_ref(value["source_intent"]))
    result = _placement_ref(value["source_result"])
    auto._validate_result(
        result,
        expected_intent_digest=intent["intent_digest"],
        expected_scene_binding_digest=result["scene_binding_digest"],
        expected_task_binding_digest=result["task_binding_digest"],
        expected_cpu_checkpoint_binding_digest=result["cpu_placement_checkpoint_binding_digest"],
    )
    _placement_ref(value["source_plan"])
    plan = _plan(Path(value["source_plan"]["path"]))
    _placement_require(
        plan["plan_digest"] == result["plan_digest"]
        and plan["expected_production_commit"] == intent["expected_production_commit"]
        and plan["source_launch_id"] == value["source_launch_id"]
        and plan["source_configuration_commit"] == intent["configuration_source_commit"],
        "plan_mismatch",
    )
    checkpoint = _placement_ref(value["source_agent_checkpoint"])
    _placement_require(
        checkpoint.get("checkpoint_digest")
        == canonical_digest(checkpoint, digest_field="checkpoint_digest"),
        "checkpoint_invalid",
    )
    receipt_path = Path(checkpoint["receipt_path"])
    inventory_path = Path(checkpoint["inventory_path"])
    _placement_require(
        _file(receipt_path)["digest"] == checkpoint["receipt_sha256"]
        and _file(inventory_path)["digest"] == checkpoint["inventory_sha256"],
        "checkpoint_changed",
    )
    receipt = validate_robot_placement_receipt(
        _read(receipt_path),
        expected_scene_binding_digest=result["scene_binding_digest"],
        expected_task_binding_digest=result["task_binding_digest"],
    )
    inventory = _read(inventory_path)
    _placement_require(
        receipt["receipt_digest"] == result["placement_agent_receipt_digest"]
        and inventory["checkpoint_digest"]
        == canonical_digest(inventory, digest_field="checkpoint_digest")
        and inventory["candidate_inventory_digest"] == result["candidate_inventory_digest"],
        "placement_mismatch",
    )
    owner = _read(Path(intent["phases"]["construction"]["authorization_path"]))[
        "scene_owner_attempt"
    ]["scene_attempt_binding"]
    _placement_require(
        value["owner_intent_digest"] == owner["intent_digest"]
        and (expected_owner_digest is None or owner["intent_digest"] == expected_owner_digest),
        "owner_mismatch",
    )
    return {
        "intent": intent,
        "result": result,
        "plan": plan,
        "placement": receipt,
        "inventory": inventory,
        "owner": owner,
    }


def validate_placement_cancellation(*, receipt: Mapping[str, Any], attempt: Mapping[str, Any]) -> None:
    source = validate_placement_adoption(
        receipt["completed_placement_adoption"], expected_owner_digest=attempt["intent_digest"]
    )
    _placement_require(
        receipt.get("schema_version") == PLACEMENT_CANCEL_SCHEMA
        and receipt.get("receipt_digest")
        == canonical_digest(receipt, digest_field="receipt_digest")
        and receipt.get("status") == "cancelled_before_native_submission"
        and receipt.get("native_submission_absent") is True
        and receipt.get("model_holds_retained") is True
        and receipt.get("provider_mutation_performed") is False
        and all(
            receipt.get(k) == attempt.get(k)
            for k in (
                "attempt_id",
                "attempt_digest",
                "intent_digest",
                "provider",
                "maximum_spend_usd",
            )
        )
        and attempt["provider"] == "vast"
        and attempt["source_commit"] == source["intent"]["expected_production_commit"],
        "cancellation_invalid",
    )
    owners = [
        _read(Path(p["authorization_path"]))["scene_owner_attempt"]["scene_attempt_binding"]
        for p in source["intent"]["phases"].values()
    ]
    _placement_require(
        any(
            all(
                o[k] == attempt[k]
                for k in (
                    "attempt_id",
                    "intent_id",
                    "intent_digest",
                    "source_commit",
                    "runtime_digest",
                    "input_digest",
                )
            )
            for o in owners
        ),
        "cancellation_owner_mismatch",
    )


def _visual_require(value: Any, code: str) -> None:
    if not value:
        raise ValueError('visual_review_continuation_'+code)


def _visual_ref(value: Mapping[str, Any]) -> dict[str, Any]:
    p = Path(str(value.get('path') or ''))
    _visual_require(_file(p) == dict(value), 'reference_changed')
    return _read(p)


def validate_visual_continuation(value: Mapping[str, Any], *, expected_commit: str | None = None) -> dict[str, Any]:
    from .task_evaluation_configured_controls_autostart_validation import validate_configured_controls_autostart_intent
    _visual_require(value.get('schema_version') == VISUAL_SCHEMA
        and value.get('continuation_digest') == canonical_digest(value,digest_field='continuation_digest')
        and value.get('maximum_reviewer_calls') == 1
        and value.get('review_hard_cap_usd') == REVIEW_CAP
        and value.get('native_phase_hard_cap_usd') == NATIVE_CAP
        and (expected_commit is None or value.get('execution_commit') == expected_commit), 'invalid')
    old = validate_configured_controls_autostart_intent(_visual_ref(value['source_intent']))
    grant = _visual_ref(value['review_authority'])
    _visual_require('visual_review_continuation' not in old and old['expected_production_commit'] != value['execution_commit'], 'recursive_or_same_release')
    receipt = _visual_ref(value['source_placement_receipt'])
    from .task_evaluation_visual_review_authority import validate as validate_authority
    directory = Path(value['review_authority']['path']).parent.parent
    validate_authority(directory=directory,value=grant)
    _visual_require(grant['source_placement_receipt_digest'] == receipt.get('receipt_digest'), 'owner_approved_other_review')
    inventory = _visual_ref(value['source_inventory'])
    _visual_require(receipt.get('receipt_digest') == canonical_digest(receipt,digest_field='receipt_digest')
        and receipt.get('status') == 'blocked' and receipt.get('accepted_pose') is None
        and receipt.get('model') == 'gpt-5.6-sol' and receipt.get('reasoning_effort') == 'high'
        and receipt.get('native_attempt_count') == 0 and receipt.get('model_grades_controls') is False
        and inventory.get('checkpoint_digest') == canonical_digest(inventory,digest_field='checkpoint_digest')
        and inventory.get('candidate_inventory_digest') == receipt.get('candidate_inventory_digest'), 'source_invalid')
    rounds = receipt.get('rounds') or []
    _visual_require(bool(rounds) and all(r.get('geometry_gate',{}).get('status') == 'passed'
        and r.get('visual_review',{}).get('camera_views_are_sufficient') is False
        and r.get('visual_review',{}).get('status') in {'uncertain','rejected'}
        and r.get('native_attempt') is None for r in rounds), 'not_preview_only_failure')
    proposal = rounds[0]['proposal']
    _visual_require(any(c.get('candidate_id') == proposal.get('candidate_id') and c.get('pose') == proposal.get('pose')
        and c.get('support_surface_id') == proposal.get('support_surface_id') for c in inventory.get('candidates',[])), 'proposal_not_inventory_member')
    completion = _visual_ref(value['source_cost_completion'])
    _visual_require(completion.get('completion_receipt_digest') == canonical_digest(completion,digest_field='completion_receipt_digest')
        and completion.get('provider_call_performed') is True
        and completion.get('runtime_result_digest') == receipt['receipt_digest']
        and completion.get('authorization_receipt_digest') == old['intent_digest'], 'source_cost_missing')
    _visual_require(old['configuration_adoption']['source_launch_id'] == value['source_launch_id'], 'source_launch_mismatch')
    return {'intent':old,'receipt':receipt,'inventory':inventory,'proposal':proposal}


def validate_visual_native_retirement(*, receipt: Mapping[str, Any], attempt: Mapping[str, Any]) -> None:
    source=validate_visual_continuation(receipt['visual_review_continuation'])
    old=source['intent']
    _visual_require(receipt.get('schema_version') == VISUAL_RETIREMENT_SCHEMA
        and receipt.get('receipt_digest') == canonical_digest(receipt,digest_field='receipt_digest')
        and receipt.get('status') == 'cancelled_before_native_activation'
        and receipt.get('native_activation_absent') is True and receipt.get('spent_placement_hold_retained') is True
        and receipt.get('provider_mutation_performed') is False
        and all(receipt.get(k)==attempt.get(k) for k in ('attempt_id','attempt_digest','intent_digest','provider','maximum_spend_usd'))
        and attempt['source_commit']==old['expected_production_commit'] and attempt['provider']=='vast', 'retirement_invalid')
    owners=[_read(Path(old['phases'][phase]['authorization_path']))['scene_owner_attempt']['scene_attempt_binding'] for phase in ('construction','controls')]
    _visual_require(any(all(owner[k]==attempt[k] for k in ('attempt_id','intent_id','intent_digest','source_commit','input_digest','runtime_digest')) for owner in owners), 'retirement_owner_mismatch')


def validate_terminal_retirement(*, receipt: Mapping[str, Any], attempt: Mapping[str, Any]) -> None:
    ref = receipt.get('source_adoption_intent') or {}
    intent = _read(Path(str(ref.get('path') or '')))
    _terminal_require(receipt.get('schema_version') == TERMINAL_RETIREMENT_SCHEMA
        and receipt.get('status') == 'cancelled_before_adoption_materialization'
        and receipt.get('receipt_digest') == canonical_digest(receipt, digest_field='receipt_digest')
        and all(receipt.get(k) == attempt.get(k) for k in ('attempt_id', 'attempt_digest', 'intent_digest', 'provider', 'maximum_spend_usd'))
        and _file(Path(ref['path'])) == ref
        and intent.get('intent_digest') == canonical_digest(intent, digest_field='intent_digest')
        and intent.get('expected_production_commit') == attempt.get('source_commit')
        and (intent.get('configuration_adoption') or {}).get('mode') == 'explicit_terminal_adoption'
        and re.fullmatch(r'[0-9a-f]{40}', str(receipt.get('superseding_release')))
        and receipt['superseding_release'] != attempt['source_commit']
        and receipt.get('materialization_started') is False
        and receipt.get('provider_mutation_performed') is False,
        'terminal_adoption_retirement_invalid')
    owner = _read(Path(intent['phases']['construction']['authorization_path']))['scene_owner_attempt']['scene_attempt_binding']
    stem = owner['attempt_id'].removesuffix('-construction')
    _terminal_require(attempt['attempt_id'] in {stem+'-'+p for p in ('construction', 'controls', 'placement')}
        and all(owner[k] == attempt[k] for k in ('intent_id', 'intent_digest', 'source_commit', 'input_digest', 'runtime_digest')),
        'terminal_adoption_retirement_owner_mismatch')


def validated_cancellation(directory: Path, attempt: Mapping[str, Any]) -> dict[str, Any] | None:
    path = directory / DIRECTORY / (str(attempt["attempt_id"]) + ".json")
    if not path.exists() and not path.is_symlink():
        return None
    receipt = _read(path)
    if receipt.get('schema_version') == 'task_evaluation_unused_native_plan_cancellation.v1':
        validate_placement_cancellation(receipt=receipt,attempt=attempt)
        return receipt
    if receipt.get('schema_version') == 'task_evaluation_unstarted_native_after_visual_review.v1':
        validate_visual_native_retirement(receipt=receipt, attempt=attempt)
        return receipt
    if receipt.get('schema_version') == 'task_evaluation_unmaterialized_adoption_cancellation.v1':
        validate_terminal_retirement(receipt=receipt, attempt=attempt)
        return receipt
    original = receipt.get("original_blocked_launch_receipt") or {}
    if (
        receipt.get("schema_version") != UNSTARTED_SCHEMA
        or receipt.get("status") != "cancelled_before_controls_eligibility"
        or receipt.get("receipt_digest") != canonical_digest(receipt, digest_field="receipt_digest")
        or receipt.get("attempt_digest") != attempt.get("attempt_digest")
        or receipt.get("intent_digest") != attempt.get("intent_digest")
        or receipt.get("attempt_id") != attempt.get("attempt_id")
        or receipt.get("maximum_spend_usd") != attempt.get("maximum_spend_usd")
        or receipt.get("provider") != attempt.get("provider")
        or original.get("schema_version") != "task_evaluation_launch_receipt.v1"
        or original.get("status") != "blocked"
        or original.get("receipt_digest") != cross_runtime_canonical_digest(original, digest_field="receipt_digest")
        or original.get("source_commit") != attempt.get("source_commit")
        or receipt.get("downstream_execution_eligible") is not False
        or receipt.get("provider_mutation_performed") is not False
    ):
        raise ValueError("unstarted_controls_cancellation_invalid")
    return receipt
