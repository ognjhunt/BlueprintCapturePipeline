"""One bounded re-review of clearer images for an already proposed placement.

Spent placement holds remain spent. Only native phases with no materialized
plan may move to the repaired release; geometry and the learned proposal are
reopened, never replaced by an authored success or an invented pose.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_unstarted_controls_reservations import _file, _read

SCHEMA = 'task_evaluation_visual_review_continuation.v1'
RETIREMENT_SCHEMA = 'task_evaluation_unstarted_native_after_visual_review.v1'
REVIEW_CAP = .15
NATIVE_CAP = .40
MAX_INPUT_TOKENS = 12000
MAX_OUTPUT_TOKENS = 4096


def require(value: Any, code: str) -> None:
    if not value:
        raise ValueError('visual_review_continuation_'+code)


def reference(value: Mapping[str, Any]) -> dict[str, Any]:
    p = Path(str(value.get('path') or ''))
    require(_file(p) == dict(value), 'reference_changed')
    return _read(p)


def validate(value: Mapping[str, Any], *, expected_commit: str | None = None) -> dict[str, Any]:
    from .task_evaluation_configured_controls_autostart import validate_configured_controls_autostart_intent
    require(value.get('schema_version') == SCHEMA
        and value.get('continuation_digest') == canonical_digest(value,digest_field='continuation_digest')
        and value.get('maximum_reviewer_calls') == 1
        and value.get('review_hard_cap_usd') == REVIEW_CAP
        and value.get('native_phase_hard_cap_usd') == NATIVE_CAP
        and (expected_commit is None or value.get('execution_commit') == expected_commit), 'invalid')
    old = validate_configured_controls_autostart_intent(reference(value['source_intent']))
    grant = reference(value['review_authority'])
    require('visual_review_continuation' not in old and old['expected_production_commit'] != value['execution_commit'], 'recursive_or_same_release')
    receipt = reference(value['source_placement_receipt'])
    from .task_evaluation_visual_review_authority import validate as validate_authority
    directory = Path(value['review_authority']['path']).parent.parent
    validate_authority(directory=directory,value=grant)
    require(grant['source_placement_receipt_digest'] == receipt.get('receipt_digest'), 'owner_approved_other_review')
    inventory = reference(value['source_inventory'])
    require(receipt.get('receipt_digest') == canonical_digest(receipt,digest_field='receipt_digest')
        and receipt.get('status') == 'blocked' and receipt.get('accepted_pose') is None
        and receipt.get('model') == 'gpt-5.6-sol' and receipt.get('reasoning_effort') == 'high'
        and receipt.get('native_attempt_count') == 0 and receipt.get('model_grades_controls') is False
        and inventory.get('checkpoint_digest') == canonical_digest(inventory,digest_field='checkpoint_digest')
        and inventory.get('candidate_inventory_digest') == receipt.get('candidate_inventory_digest'), 'source_invalid')
    rounds = receipt.get('rounds') or []
    require(bool(rounds) and all(r.get('geometry_gate',{}).get('status') == 'passed'
        and r.get('visual_review',{}).get('camera_views_are_sufficient') is False
        and r.get('visual_review',{}).get('status') in {'uncertain','rejected'}
        and r.get('native_attempt') is None for r in rounds), 'not_preview_only_failure')
    proposal = rounds[0]['proposal']
    require(any(c.get('candidate_id') == proposal.get('candidate_id') and c.get('pose') == proposal.get('pose')
        and c.get('support_surface_id') == proposal.get('support_surface_id') for c in inventory.get('candidates',[])), 'proposal_not_inventory_member')
    completion = reference(value['source_cost_completion'])
    require(completion.get('completion_receipt_digest') == canonical_digest(completion,digest_field='completion_receipt_digest')
        and completion.get('provider_call_performed') is True
        and completion.get('runtime_result_digest') == receipt['receipt_digest']
        and completion.get('authorization_receipt_digest') == old['intent_digest'], 'source_cost_missing')
    require(old['configuration_adoption']['source_launch_id'] == value['source_launch_id'], 'source_launch_mismatch')
    return {'intent':old,'receipt':receipt,'inventory':inventory,'proposal':proposal}


def native_plan_absent(*, config: Mapping[str, Any], source_launch_id: str, source_commit: str) -> bool:
    root = Path(config.get('plan_root') or os.getenv('BLUEPRINT_TASK_EVALUATION_CONFIGURED_CONTROLS_PLAN_ROOT')
                or '/etc/blueprint/task-evaluation-configured-controls-plans')
    require(root.is_absolute() and not any(p.is_symlink() for p in (root,*root.parents)), 'plan_store_unsafe')
    for path in root.glob('*.json'):
        plan = _read(path)
        if plan.get('source_launch_id') == source_launch_id and plan.get('expected_production_commit') == source_commit:
            return False
    return True


def discover(*, config: Mapping[str, Any], intent_id: str, source: Mapping[str, Any], expected_commit: str) -> dict[str, Any] | None:
    from .task_evaluation_controls_autoprovision import _sealed
    from .task_evaluation_unstarted_controls_reservations import validated_cancellation
    from . import task_evaluation_scene_intake as intake
    state = Path(config.get('progression_root') or os.getenv('BLUEPRINT_TASK_EVALUATION_CONFIGURED_CONTROLS_STATE_ROOT')
        or str(Path(config['scene_root']).parent/'task-evaluation-configured-controls'))
    binding = state/source['launch_id']/'cpu-robot-binding'
    if not binding.exists():
        return None
    matches=[]
    for path in (Path(config['controls_root'])/'terminal-adoptions'/intent_id).glob('*/terminal_adoption_provisioning.json'):
        provision=_sealed(path,'receipt_digest')
        if provision['execution_source_commit'] == expected_commit:
            continue
        old_path=Path(provision['provisioning']['intent_path'])
        old=_read(old_path)
        if old.get('visual_review_continuation') is not None:
            continue
        auth=_read(Path(old['phases']['construction']['authorization_path']))
        owner=auth['scene_owner_attempt']['scene_attempt_binding']
        directory=Path(config['scene_root'])/intent_id
        attempt=intake._read(directory/'attempts'/(owner['attempt_id']+'.json'),'attempt_digest')
        cancelled=validated_cancellation(directory,attempt)
        if cancelled is not None:
            retained=cancelled.get('visual_review_continuation')
            if retained is not None and retained.get('execution_commit') == expected_commit:
                validate(retained,expected_commit=expected_commit)
                matches.append(dict(retained))
            continue
        require(owner['intent_id'] == intent_id and provision['adoption'] == source['adoption'],'owner_or_source_mismatch')
        token=old['intent_digest'].removeprefix('sha256:')[:16]
        paths=list((binding/f'agent-placement-attempts-{token}').glob('attempt_*/task_evaluation_robot_placement_receipt.v1.json'))
        # Repeated paid attempts or an incomplete attempt cannot be swept into
        # a single supposedly bounded continuation.
        if not paths:
            continue
        require(len(paths)==1,'ambiguous_attempts')
        receipt_path=paths[0]
        attempt_root=receipt_path.parent
        failed=_read(receipt_path)
        if (failed.get('status') != 'blocked' or not failed.get('rounds')
                or any(r.get('visual_review',{}).get('camera_views_are_sufficient') is not False
                       or r.get('geometry_gate',{}).get('status') != 'passed' for r in failed['rounds'])):
            continue
        from .task_evaluation_visual_review_authority import read_authority, DIRECTORY
        placement_id=owner['attempt_id'].removesuffix('-construction')+'-placement'
        approved=read_authority(directory=directory,source_attempt_id=placement_id)
        if approved is None:
            continue
        if not native_plan_absent(config=config,source_launch_id=source['launch_id'],source_commit=old['expected_production_commit']):
            continue
        completion=binding/'agent-official-openai-cost'/attempt_root.parent.name/attempt_root.name/'openai_official_cost_run_completion.v1.json'
        packet={'schema_version':SCHEMA,'execution_commit':expected_commit,'source_launch_id':source['launch_id'],
            'source_intent':_file(old_path),'source_placement_receipt':_file(receipt_path),
            'source_inventory':_file(attempt_root/'task_evaluation_robot_placement_candidate_inventory.v1.json'),
            'review_authority':_file(directory/DIRECTORY/(placement_id+'.json')),
            'source_cost_completion':_file(completion),'maximum_reviewer_calls':1,
            'review_hard_cap_usd':REVIEW_CAP,'native_phase_hard_cap_usd':NATIVE_CAP}
        packet['continuation_digest']=canonical_digest(packet,digest_field='continuation_digest')
        validate(packet,expected_commit=expected_commit)
        matches.append(packet)
    require(len(matches)<=1,'ambiguous_sources')
    return matches[0] if matches else None


def validate_native_retirement(*, receipt: Mapping[str, Any], attempt: Mapping[str, Any]) -> None:
    source=validate(receipt['visual_review_continuation'])
    old=source['intent']
    require(receipt.get('schema_version') == RETIREMENT_SCHEMA
        and receipt.get('receipt_digest') == canonical_digest(receipt,digest_field='receipt_digest')
        and receipt.get('status') == 'cancelled_before_native_activation'
        and receipt.get('native_activation_absent') is True and receipt.get('spent_placement_hold_retained') is True
        and receipt.get('provider_mutation_performed') is False
        and all(receipt.get(k)==attempt.get(k) for k in ('attempt_id','attempt_digest','intent_digest','provider','maximum_spend_usd'))
        and attempt['source_commit']==old['expected_production_commit'] and attempt['provider']=='vast', 'retirement_invalid')
    owners=[_read(Path(old['phases'][phase]['authorization_path']))['scene_owner_attempt']['scene_attempt_binding'] for phase in ('construction','controls')]
    require(any(all(owner[k]==attempt[k] for k in ('attempt_id','intent_id','intent_digest','source_commit','input_digest','runtime_digest')) for owner in owners), 'retirement_owner_mismatch')
