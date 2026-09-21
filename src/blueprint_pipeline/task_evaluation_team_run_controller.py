"""ADP-009D/day-21: reuse a delivered scene under a team's separate authority."""
from pathlib import Path
import os

from . import task_evaluation_controls_autoprovision as worker
from . import task_evaluation_team_run_authority as authority


def source_for_evaluation(*, config, intent, now=None):
    """Resolve only retained terminal evidence; never reconstruct or alter the source."""
    from .task_evaluation_configured_controls_progression_worker import _validate_source

    selected = intent['request']['task'].get('evaluation_source')
    if selected is None:
        return None
    binding = {**selected, 'scene_intent_digest': intent['intent_digest']}
    authority.authority_scope(intent['request']['submission_id'], binding)
    launch_root = Path(config.get('launch_state_root') or os.getenv('BLUEPRINT_TASK_EVALUATION_LAUNCH_STATE_ROOT')
        or str(Path(config['scene_root']).parent/'task-evaluation-launch-runs'))
    root = launch_root / selected['source_launch_id']
    profile = worker._json(root / 'launch_profile.json')
    try:
        terminal, receipt, zero = _validate_source(root)
    except RuntimeError as exc:
        raise ValueError(str(exc)) from exc
    worker._require(receipt.get('launch_profile_digest') == profile.get('profile_digest')
        and terminal['configured_scene_revision_digest'] == selected['configured_scene_revision_digest'],
        'team_evaluation_source_changed')
    owner = authority.evaluation_owner(source_profile=profile, authority=binding,
        source_launch_id=root.name, configured_scene_revision_digest=terminal['configured_scene_revision_digest'],
        evaluation_run_id=intent['request']['submission_id'], now=now)
    worker._require(owner['intent_id'] == intent['intent_id'], 'team_evaluation_owner_changed')
    sync = worker._json(root / 'webapp_sync_succeeded.json')
    original = None
    for path in Path(config['scene_root']).glob('scene-*/intent.json'):
        retained = worker._scene_intent(path)
        if retained['intent_digest'] == profile['scene_intent_digest']:
            original = retained
            break
    worker._require(original is not None, 'team_evaluation_source_store_mismatch')
    queue = worker.scene_preparation_queue_root(Path(config['preparation_queue_root']), original['intent_id'])
    return {'adoption': {'mode': 'explicit_terminal_adoption', 'source_launch_id': root.name,
        'source_launch_receipt_digest': receipt['receipt_digest'], 'terminal_result_digest': terminal['result_digest'],
        'configured_scene_revision_digest': terminal['configured_scene_revision_digest'],
        'publication_result_digest': terminal['publication_result_digest'],
        'webapp_sync_result_digest': sync['sync_result_digest'],
        'provider_zero_receipt_digest': zero['provider_zero_receipt_digest']},
        'source_commit': receipt['source_commit'], 'launch_id': root.name,
        'source_intent': original, 'preparation_queue_root': queue, 'evaluation_authority': binding}


def materialize_selected_evaluations(*, intent_root, launch_state_root, progression_root, plan_root,
                                     release, materializer=None):
    """Each installed team request gets its own plan, even on the same scene."""
    from .task_evaluation_configured_controls_autostart import (
        materialize_configured_controls_autostart, validate_configured_controls_autostart_intent)
    rows = []
    if intent_root is None:
        return rows
    for path in sorted(Path(intent_root).glob('adoption-*.json')):
        try:
            value = worker._json(path)
            if value.get('evaluation_authority') is None:
                continue
            if release and value.get('expected_production_commit') != release:
                continue
            value = validate_configured_controls_autostart_intent(value)
            binding = value['evaluation_authority']
            profile = worker._json(Path(launch_state_root) / binding['source_launch_id'] / 'launch_profile.json')
            authority.evaluation_owner(source_profile=profile, authority=binding,
                source_launch_id=binding['source_launch_id'],
                configured_scene_revision_digest=binding['configured_scene_revision_digest'],
                evaluation_run_id=value['evaluation_run_id'])
            result = (materializer or materialize_configured_controls_autostart)(
                source_launch_id=binding['source_launch_id'], launch_state_root=Path(launch_state_root),
                progression_root=progression_root, plan_root=plan_root, intent_path_override=path)
            rows.append({'status': result['status'], 'evaluation_run_id': value['evaluation_run_id'],
                'source_launch_id': binding['source_launch_id'], 'plan_digest': result['plan_digest']})
        except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
            rows.append({'status': 'team_evaluation_refused', 'intent': path.name, 'blockers': [str(exc)]})
    return rows


def plan_authority_blocker(value, launch_root):
    """A new evaluation reopens its own authority, not the expired preparation budget."""
    try:
        binding = value['evaluation_authority']
        authority.authority_scope(value.get('evaluation_run_id'), binding)
        worker._require(value['source_launch_id'] == binding['source_launch_id'], 'team_evaluation_source_changed')
        from .task_evaluation_configured_controls_progression_worker import _validate_source
        source_root = Path(launch_root) / value['source_launch_id']
        terminal, receipt, _ = _validate_source(source_root)
        worker._require(receipt['receipt_digest'] == value['source_launch_receipt_digest'],
            'team_evaluation_source_changed')
        profile = worker._json(source_root / 'launch_profile.json')
        authority.evaluation_owner(source_profile=profile, authority=binding,
            source_launch_id=value['source_launch_id'],
            configured_scene_revision_digest=terminal['configured_scene_revision_digest'],
            evaluation_run_id=value['evaluation_run_id'])
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        return str(exc)
    return None
