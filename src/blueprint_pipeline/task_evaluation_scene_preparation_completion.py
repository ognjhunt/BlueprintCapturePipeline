"""Close site-only preparation from published scene evidence, without a robot run."""
import json
from pathlib import Path

from . import task_evaluation_scene_intake as intake
from .decision_evidence_contracts import canonical_digest
from .task_evaluation_scene_scope_restriction import preparation_only
from .task_evaluation_scene_progression_state import safe_path
from .task_evaluation_public_scene_attempt_factory import record


def reconcile_preparation_completion(*, intent, config):
    if not preparation_only(directory=Path(config['intent_root']) / intent['intent_id'], intent=intent):
        return None
    root = safe_path(config['launch_execution_root'])
    for path in sorted(root.glob('*/launch_profile.json'), key=lambda p: p.stat().st_mtime_ns, reverse=True):
        profile = json.loads(safe_path(path).read_text())
        binding = profile.get('scene_attempt_binding') or {}
        if binding.get('intent_id') != intent['intent_id'] or binding.get('intent_digest') != intent['intent_digest']:
            continue
        run = path.parent
        receipt_path = run / 'launch_receipt.json'
        if not receipt_path.is_file():
            continue
        if json.loads(receipt_path.read_text()).get('status') != 'completed':
            continue
        try:
            from .task_evaluation_scene_attempt_binding import require_scene_execution_binding
            from .task_evaluation_configured_controls_progression_worker import _validate_source
            from .task_evaluation_launch_terminal_evidence import _scene_configuration_terminal_projection
            intake._require(profile.get('profile_digest') == canonical_digest(profile, digest_field='profile_digest'),
                            'preparation_completion_profile_invalid')
            require_scene_execution_binding(profile, source_commit=profile['source_commit'])
            attempt = intake._read(safe_path(Path(config['intent_root']) / intent['intent_id'] / 'attempts' /
                                  (binding['attempt_id'] + '.json')), 'attempt_digest')
            intake._require(all(attempt.get(k) == v for k, v in binding.items() if k != 'schema_version'),
                            'preparation_completion_owner_mismatch')
            result, receipt, _zero = _validate_source(run)
            projection, blockers = _scene_configuration_terminal_projection(result)
            scope = profile['task_evaluation_run']
            intake._require(not blockers and projection is not None
                and scope['run_mode'] == 'scene_configuration'
                and scope['task_id'] == intent['request']['task']['task_id']
                and result['run_id'] == scope['configuration_run_id']
                and result['source_commit'] == receipt['source_commit'] == profile['source_commit']
                and result.get('result_digest') == canonical_digest(result, digest_field='result_digest')
                and receipt['launch_id'] == run.name
                and receipt['launch_profile_digest'] == profile['profile_digest']
                and receipt['terminal_evidence']['scene_configuration'] == projection,
                'preparation_completion_publication_mismatch')
            return {'status': 'completed', 'phase': 'scene_prepared', 'blockers': [],
                'result_reference': result['configured_scene_revision_reference'],
                'state': {'scene_preparation_completion': {
                    'launch_receipt': record(receipt_path),
                    'provider_zero': record(run / 'post_teardown_provider_zero_receipt.json'),
                    'website_readback': record(run / 'webapp_sync_succeeded.json'),
                    'robot_evaluation_performed': False}}}
        except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
            return {'status': 'awaiting_execution', 'phase': 'scene_publication',
                    'blockers': [str(exc)[:200]], 'state': {}}
    return None
