"""Release only superseded adoption holds whose materialization never started.

The canonical deploy quiesces workers before switching the active release.
The old release cannot progress afterward. Any CPU binding directory, including
a failed or partial attempt, prevents this path from returning budget or retries.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Mapping

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_unstarted_controls_reservations import DIRECTORY, _file, _read

SCHEMA = 'task_evaluation_unmaterialized_adoption_cancellation.v1'


def validate_retirement(*, receipt: Mapping[str, Any], attempt: Mapping[str, Any]) -> None:
    from .task_evaluation_controls_autoprovision import _require
    ref = receipt.get('source_adoption_intent') or {}
    intent = _read(Path(str(ref.get('path') or '')))
    _require(receipt.get('schema_version') == SCHEMA
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
    _require(attempt['attempt_id'] in {stem+'-'+p for p in ('construction', 'controls', 'placement')}
        and all(owner[k] == attempt[k] for k in ('intent_id', 'intent_digest', 'source_commit', 'input_digest', 'runtime_digest')),
        'terminal_adoption_retirement_owner_mismatch')


def retire_unmaterialized_adoptions(*, config: Mapping[str, Any], intent_id: str,
        source: Mapping[str, Any], expected_production_commit: str, dry_run: bool = False) -> list[dict[str, Any]]:
    from . import task_evaluation_scene_intake as intake
    from .task_evaluation_controls_autoprovision import _require, _sealed
    from .task_evaluation_configured_controls_autostart import validate_configured_controls_autostart_intent
    from .task_evaluation_release_identity import running_release_commit
    from .task_evaluation_unstarted_controls_reservations import validated_cancellation

    base = Path(config['controls_root'])/'terminal-adoptions'/intent_id
    candidates = list(base.glob('*/terminal_adoption_provisioning.json'))
    foreign = []
    for path in candidates:
        provision = _sealed(path, 'receipt_digest')
        if provision['execution_source_commit'] == expected_production_commit:
            continue
        foreign.append(provision)
    if not foreign:
        return []
    _require(dry_run or running_release_commit() == expected_production_commit,
             'terminal_adoption_retirement_requires_running_release')
    state = Path(config.get('progression_root') or os.getenv('BLUEPRINT_TASK_EVALUATION_CONFIGURED_CONTROLS_STATE_ROOT')
        or str(Path(config['scene_root']).parent/'task-evaluation-configured-controls'))
    binding_root = state/source['launch_id']/'cpu-robot-binding'
    _require(state.is_absolute() and not any(p.is_symlink() for p in (binding_root, *binding_root.parents)),
             'terminal_adoption_retirement_state_unsafe')
    results = []
    with intake._lock(Path(config['scene_root'])):
        for provision in foreign:
            intent_path = Path(provision['provisioning']['intent_path'])
            intent = validate_configured_controls_autostart_intent(_read(intent_path))
            _require(provision['adoption'] == source['adoption'] == intent['configuration_adoption']
                and provision['execution_source_commit'] == intent['expected_production_commit'],
                'terminal_adoption_retirement_source_mismatch')
            owner = _read(Path(intent['phases']['construction']['authorization_path']))['scene_owner_attempt']['scene_attempt_binding']
            _require(owner['intent_id'] == intent_id, 'terminal_adoption_retirement_owner_mismatch')
            stem = owner['attempt_id'].removesuffix('-construction')
            directory = Path(config['scene_root'])/intent_id
            for phase in ('construction', 'controls', 'placement'):
                attempt = intake._read(directory/'attempts'/(stem+'-'+phase+'.json'), 'attempt_digest')
                old = validated_cancellation(directory, attempt)
                if old is not None:
                    results.append(old)
                    continue
                _require(not binding_root.exists(), 'terminal_adoption_materialization_already_started')
                receipt = {'schema_version': SCHEMA, 'status': 'cancelled_before_adoption_materialization',
                    **{k: attempt[k] for k in ('attempt_id', 'attempt_digest', 'intent_digest', 'provider', 'maximum_spend_usd')},
                    'source_adoption_intent': _file(intent_path), 'superseding_release': expected_production_commit,
                    'absent_binding_root': str(binding_root), 'materialization_started': False, 'provider_mutation_performed': False}
                receipt['receipt_digest'] = canonical_digest(receipt, digest_field='receipt_digest')
                validate_retirement(receipt=receipt, attempt=attempt)
                if not dry_run:
                    target = directory/DIRECTORY/(attempt['attempt_id']+'.json')
                    target.parent.mkdir(mode=0o750, exist_ok=True)
                    intake.write_exclusive(target, receipt)
                results.append(receipt)
    return results
