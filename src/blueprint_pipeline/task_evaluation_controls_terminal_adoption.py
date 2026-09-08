"""Automatically provision current-release controls from a delivered older scene.

Only the exact holds cancelled before controls eligibility may take this path.
It never reconfigures the scene or allocates a provider; the ordinary controls
worker still owns CPU placement and every subsequent paid admission.
"""
from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import time
from typing import Any, Mapping

from .decision_evidence_contracts import canonical_digest


def terminal_adoption_source(*, config: Mapping[str, Any], intent_id: str,
                             expected_production_commit: str) -> dict[str, Any] | None:
    from . import task_evaluation_scene_intake as intake
    from .task_evaluation_unstarted_controls_reservations import validated_cancellation
    from .task_evaluation_configured_controls_progression_worker import _validate_source
    from .task_evaluation_controls_autoprovision import _json, _require

    directory = Path(config['scene_root']) / intent_id
    retired = []
    for path in (directory/'attempts').glob('*.json'):
        attempt = intake._read(path, 'attempt_digest')
        cancellation = validated_cancellation(directory, attempt)
        if cancellation is not None and cancellation.get('status') == 'cancelled_before_controls_eligibility':
            retired.append((attempt, cancellation))
    if not retired:
        return None
    source_ids = {c['original_blocked_launch_receipt'].get('launch_id') for _, c in retired}
    _require(len(retired) == 3 and len(source_ids) == 1 and None not in source_ids, 'terminal_adoption_retired_scope_invalid')
    launch_id = next(iter(source_ids))
    _require(isinstance(launch_id, str) and Path(launch_id).name == launch_id, 'terminal_adoption_launch_id_invalid')
    launch_root = Path(config.get('launch_state_root') or os.getenv('BLUEPRINT_TASK_EVALUATION_LAUNCH_STATE_ROOT') or str(directory.parent.parent/'task-evaluation-launch-runs'))
    run_root = launch_root/launch_id
    if not (run_root/'launch_receipt.json').is_file() or _json(run_root/'launch_receipt.json').get('status') != 'completed':
        return None
    terminal, receipt, zero = _validate_source(run_root)
    if receipt['source_commit'] == expected_production_commit:
        return None
    _require(all(a['source_commit'] == receipt['source_commit'] for a, _ in retired), 'terminal_adoption_source_commit_mismatch')
    sync = _json(run_root/'webapp_sync_succeeded.json')
    adoption = {'mode': 'explicit_terminal_adoption', 'source_launch_id': launch_id,
        'source_launch_receipt_digest': receipt['receipt_digest'], 'terminal_result_digest': terminal['result_digest'],
        'configured_scene_revision_digest': terminal['configured_scene_revision_digest'],
        'publication_result_digest': terminal['publication_result_digest'],
        'webapp_sync_result_digest': sync['sync_result_digest'], 'provider_zero_receipt_digest': zero['provider_zero_receipt_digest']}
    return {'adoption': adoption, 'source_commit': receipt['source_commit'],
            'retired_attempts': [a for a, _ in retired], 'launch_id': launch_id}


def validate_embedded_intent_replacement(*, run_root: Path, original_path: Path,
                                        replacement_path: Path) -> None:
    from .task_evaluation_configured_controls_autostart import validate_configured_controls_autostart_intent
    from .task_evaluation_controls_autoprovision import _json, _require
    from .task_evaluation_scene_intake import ROOT_ENV

    original = validate_configured_controls_autostart_intent(_json(original_path))
    replacement = validate_configured_controls_autostart_intent(_json(replacement_path))
    authorization = _json(Path(replacement['phases']['construction']['authorization_path']))
    owner = authorization['scene_owner_attempt']['scene_attempt_binding']
    source = terminal_adoption_source(config={'scene_root': os.environ[ROOT_ENV],
        'launch_state_root': str(run_root.parent)}, intent_id=owner['intent_id'],
        expected_production_commit=replacement['expected_production_commit'])
    _require(source is not None and source['launch_id'] == run_root.name
        and source['adoption'] == replacement['configuration_adoption']
        and source['source_commit'] == original['expected_production_commit']
        and original['configuration_adoption'] == {'mode': 'same_commit_automatic'},
        'terminal_adoption_embedded_replacement_invalid')
    original_owner = _json(Path(original['phases']['construction']['authorization_path']))['scene_owner_attempt']['scene_attempt_binding']
    _require(all(original_owner[k] == owner[k] for k in ('intent_id', 'intent_digest'))
        and all(original.get(k) == replacement.get(k) for k in ('team_namespace', 'scene_id', 'task_id')),
        'terminal_adoption_embedded_owner_mismatch')


def provision_terminal_controls_adoption(*, config: Mapping[str, Any], catalog: Mapping[str, Any],
        intent_id: str, expected_production_commit: str, now: float | None = None) -> dict[str, Any] | None:
    from . import task_evaluation_controls_autoprovision as worker
    from . import task_evaluation_scene_intake as intake
    from . import task_evaluation_configured_controls_continuation_provisioning as producer
    from .task_evaluation_scene_execution_authority import bind_scene_attempt
    from .task_evaluation_scene_spend import publish_current_scene_project_spend

    moment = time.time() if now is None else now
    directory = Path(config['scene_root'])/intent_id
    intent = worker._scene_intent(directory/'intent.json')
    worker._require(intent['authenticated_issuer'] in config['trusted_clients'], 'owner_intent_invalid')
    source = terminal_adoption_source(config=config, intent_id=intent_id, expected_production_commit=expected_production_commit)
    if source is None:
        return None
    worker._require(not (directory/'revoked.json').exists(), 'authority_revoked')
    expiry = intake.effective_execution_expiry(directory, intent)
    worker._require(moment < expiry, 'authority_expired')
    from .task_evaluation_terminal_adoption_retirement import retire_unmaterialized_adoptions
    retire_unmaterialized_adoptions(config=config, intent_id=intent_id,
        source=source, expected_production_commit=expected_production_commit)
    request = intake.validate_request(intent['request'], now=intent['accepted_at_epoch'])
    binding = catalog['bindings'].get(request['task'].get('robot_binding_id'))
    worker._require(isinstance(binding, dict) and binding.get('expected_production_commit') == expected_production_commit, 'runtime_release_mismatch')
    robot = worker._asset(binding['robot_asset_usd'])
    cameras = worker._asset(binding['embodiment_camera_template'])
    runtime = Path(binding['runtime_source_payload_dir'])
    worker._require(worker.payload_digest(runtime) == binding['runtime_digest'], 'runtime_digest_mismatch')
    original_caps = {a['attempt_id'].rsplit('-', 1)[-1]: a['maximum_spend_usd'] for a in source['retired_attempts']}
    worker._require(set(original_caps) == {'construction', 'controls', 'placement'}, 'terminal_adoption_phase_caps_invalid')
    cap = min(float(binding.get('phase_hard_cap_usd', producer.DEFAULT_PHASE_HARD_CAP_USD)),
              float(original_caps['construction']), float(original_caps['controls']))
    inference_cap = min(producer.DEFAULT_MAX_PLACEMENT_INFERENCE_COST_USD, float(original_caps['placement']))
    link = worker._configured_scene_preparation_link(intent=intent, preparation_queue_root=Path(config['preparation_queue_root']), expected_production_commit=source['source_commit'])
    worker._require(link is not None, 'terminal_adoption_preparation_missing')
    identity = {'owner_intent_digest': intent['intent_digest'], 'adoption': source['adoption'],
                'execution_source_commit': expected_production_commit, 'catalog_binding_digest': canonical_digest({k:v for k,v in binding.items() if k not in {'project_spend_reconciliation', 'project_spend_observed_at_epoch'}})}
    key = canonical_digest(identity).removeprefix('sha256:')
    root = Path(config['controls_root'])/'terminal-adoptions'/intent_id/key
    worker._require(root.is_absolute() and not any(p.is_symlink() for p in (root, *root.parents)), 'controls_root_unsafe')
    root.mkdir(mode=0o750, parents=True, exist_ok=True)
    with intake._lock(root):
        receipt_path = root/'terminal_adoption_provisioning.json'
        if receipt_path.exists():
            retained_result = worker._sealed(receipt_path, 'receipt_digest')
            worker._require(all(retained_result.get(k) == v for k, v in identity.items()), 'terminal_adoption_identity_changed')
            producer.install_intent_into_registry(intent_path=retained_result['provisioning']['intent_path'],
                intent_root=config['intent_root'], expected_production_commit=expected_production_commit,
                service_group=config.get('service_group', 'blueprint'))
            return retained_result
        phases = {}
        for phase, provider, amount in [('construction','vast',cap), ('controls','vast',cap), ('placement','openai',inference_cap)]:
            attempt = intake.reserve_scene_attempt(queue_root=config['scene_root'], intent_id=intent_id,
                attempt_id='controls-'+key[:40]+'-'+phase, source_commit=expected_production_commit,
                runtime_digest=binding['runtime_digest'], input_digest='sha256:'+key,
                provider=provider, maximum_spend_usd=amount, now=moment)
            if phase != 'placement':
                phases[phase] = bind_scene_attempt(attempt)
        retained_path = root/'terminal_adoption_inputs.json'
        if not retained_path.exists():
            current = worker._json(Path(binding['project_spend_current_path']))
            spend = publish_current_scene_project_spend(scene_root=config['scene_root'],
                seed_reconciliation_path=current['path'], output_root=root/'spend', current_path=root/'spend-current.json', now=moment)
            retained = worker._seal({**identity, 'issued_at_epoch': moment, 'spend_path': spend['pointer']['path']}, 'receipt_digest')
            intake.write_exclusive(retained_path, retained)
        retained = worker._sealed(retained_path, 'receipt_digest')
        worker._require(all(retained.get(k) == v for k, v in identity.items()), 'terminal_adoption_inputs_changed')
        issued = retained['issued_at_epoch']
        authority = 'scene-intent:'+intent['intent_digest']
        result = producer.provision_configured_controls_continuation(expected_production_commit=expected_production_commit,
            configuration_source_commit=source['source_commit'], configuration_adoption=source['adoption'],
            preparation_result_path=Path(config['preparation_queue_root'])/'results'/link['result_filename'],
            preparation_queue_root=config['preparation_queue_root'], robot_asset_usd_path=robot,
            runtime_source_payload_dir=runtime, embodiment_camera_template_path=cameras,
            project_spend_reconciliation_path=retained['spend_path'], controls_root=root/'inputs',
            profile_dir=config['profile_dir'], authorization_reference=authority, authorized_by=request['owner']['user_id'],
            release_reference=authority, openai_project_id=binding['openai_project_id'], openai_api_key_id=binding['openai_api_key_id'],
            phase_hard_cap_usd=cap, phase_ttl_seconds=min(producer.DEFAULT_PHASE_TTL_SECONDS, int(cap*3600/producer.DEFAULT_HOURLY_RATE_USD)),
            max_inference_cost_usd=inference_cap, authority_valid_seconds=int(expiry-issued),
            now=datetime.fromtimestamp(issued, timezone.utc), external_layer_bucket=binding.get('external_layer_bucket'),
            scene_phase_attempts=phases, scene_intake_root=config['scene_root'])
        installed = producer.install_intent_into_registry(intent_path=result['intent_path'], intent_root=config['intent_root'],
            expected_production_commit=expected_production_commit, service_group=config.get('service_group','blueprint'))
        receipt = worker._seal({'status':'installed_terminal_adoption', **identity, 'intent_id':intent_id,
            'provisioning':result, 'installation':installed, 'provider_mutation_performed':False}, 'receipt_digest')
        intake.write_exclusive(receipt_path, receipt)
        return receipt
