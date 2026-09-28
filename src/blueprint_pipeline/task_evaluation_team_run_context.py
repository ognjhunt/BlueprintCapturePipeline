"""Read-only website context for selecting an evaluation of a delivered scene.

ADP-009D/day-21: use the retained owner task, never rebuild it from UI prose.
No authority, reservation, assignment, or provider operation is created here.
"""
import os
from pathlib import Path

from . import task_evaluation_scene_intake as intake


def configuration_binding_digest(binding):
    from .decision_evidence_contracts import canonical_digest
    return canonical_digest({k:v for k,v in binding.items() if k not in {
        'expected_production_commit', 'project_spend_reconciliation', 'project_spend_current_path',
        'project_spend_observed_at_epoch'}})


def evaluation_context(*, source_launch_id, owner, config=None):
    from . import task_evaluation_controls_autoprovision as worker
    from .task_evaluation_configured_controls_progression_worker import _validate_source
    from .task_evaluation_scene_policy_capability import supported_policy_candidates

    worker._require(intake._identifier(source_launch_id) and isinstance(owner, dict)
        and set(owner) == {'user_id', 'organization_id'}, 'evaluation_context_request_invalid')
    config = config or worker._json(Path(os.environ[worker.CONFIG_ENV]))
    launch_root = Path(config.get('launch_state_root') or os.getenv('BLUEPRINT_TASK_EVALUATION_LAUNCH_STATE_ROOT')
        or str(Path(config['scene_root']).parent/'task-evaluation-launch-runs'))
    run = launch_root/source_launch_id
    profile = worker._sealed(run/'launch_profile.json', 'profile_digest')
    original = None
    for path in Path(config['scene_root']).glob('scene-*/intent.json'):
        candidate = worker._scene_intent(path)
        if candidate['intent_digest'] != profile.get('scene_intent_digest'):
            continue
        worker._require(candidate['authenticated_issuer'] in config['trusted_clients']
            and candidate['request']['owner'] == owner and not (path.parent/'revoked.json').exists(),
            'evaluation_context_source_access_refused')
        original = intake.validate_request(candidate['request'], now=candidate['accepted_at_epoch'])
        break
    worker._require(original is not None, 'evaluation_context_source_access_refused')
    terminal, receipt, _ = _validate_source(run)
    worker._require(receipt['launch_profile_digest'] == profile['profile_digest'],
        'evaluation_context_source_changed')
    catalog = worker._sealed(Path(config['robot_catalog_path']), 'catalog_digest')
    worker._require(catalog['schema_version'] in {worker.CATALOG_SCHEMA, worker.CONTENT_CATALOG_SCHEMA}
        and isinstance(catalog.get('bindings'), dict), 'evaluation_context_catalog_invalid')
    configurations = []
    for binding_id, binding in sorted(catalog['bindings'].items()):
        worker._require(intake._identifier(binding_id) and isinstance(binding, dict),
            'evaluation_context_catalog_invalid')
        configurations.append({'id':binding_id, 'label':binding.get('label') or binding_id,
            'binding_digest':configuration_binding_digest(binding),
            'policy_candidates':supported_policy_candidates()})
    return intake._seal({'schema_version':'task_evaluation_team_context.v1',
        'owner':owner, 'source_launch_id':source_launch_id, 'source_profile_digest':profile['profile_digest'],
        'configured_scene_revision_digest':terminal['configured_scene_revision_digest'],
        'source':original['source'], 'task':original['task'],
        'rights_reference':original['consent']['rights_reference'],
        'configurations':configurations, 'claim_scope':'development_only',
        'provider_mutation_performed':False}, 'context_digest')
