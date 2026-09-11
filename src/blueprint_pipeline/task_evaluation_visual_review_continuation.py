"""One bounded re-review of clearer images for an already proposed placement.

Spent placement holds remain spent. Only native phases with no materialized
plan may move to the repaired release; geometry and the learned proposal are
reopened, never replaced by an authored success or an invented pose.
"""
from __future__ import annotations

from .task_evaluation_retained_controls_evidence import (
    _visual_require as require, validate_visual_continuation as validate,
    _visual_ref as reference,  # noqa: F401 - historical public facade
    validate_visual_native_retirement as validate_native_retirement,  # noqa: F401 - historical public facade
)

import os
from pathlib import Path
from typing import Any, Mapping

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_retained_controls_evidence import _file, _read

SCHEMA = 'task_evaluation_visual_review_continuation.v1'
RETIREMENT_SCHEMA = 'task_evaluation_unstarted_native_after_visual_review.v1'
REVIEW_CAP = .15
NATIVE_CAP = .40
MAX_INPUT_TOKENS = 12000
MAX_OUTPUT_TOKENS = 4096


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
    from .task_evaluation_retained_controls_evidence import validated_cancellation
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
