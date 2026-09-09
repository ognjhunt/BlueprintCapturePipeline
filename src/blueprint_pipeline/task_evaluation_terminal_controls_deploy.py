"""Supersede unstarted terminal-controls registrations during deploy quiescence.

This command provisions inputs and bounded owner reservations only. It never
materializes placement, calls a model, queues execution, or allocates providers.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from . import task_evaluation_controls_autoprovision as worker
from . import task_evaluation_controls_terminal_adoption as adoption
from . import task_evaluation_scene_intake as intake


def prepare(*, config_path: str | Path, expected_commit: str, now: float | None = None) -> dict[str, Any]:
    config = worker._json(Path(config_path))
    catalog = worker.resolve_robot_catalog(worker._sealed(Path(config['robot_catalog_path']), 'catalog_digest'),
                                          source_commit=expected_commit)
    moment = time.time() if now is None else now
    rows = []
    for path in sorted((Path(config['controls_root'])/'terminal-adoptions').glob('scene-*')):
        intent_id = path.name
        records = [worker._sealed(p, 'receipt_digest') for p in path.glob('*/terminal_adoption_provisioning.json')]
        if not records or all(r['execution_source_commit'] == expected_commit for r in records):
            continue
        directory = Path(config['scene_root'])/intent_id
        if (directory/'revoked.json').exists():
            continue
        intent = worker._scene_intent(directory/'intent.json')
        if intake.effective_execution_expiry(directory, intent) <= moment:
            continue
        source = adoption.terminal_adoption_source(config=config, intent_id=intent_id, expected_production_commit=expected_commit)
        if source is None:
            continue
        state = Path(config.get('progression_root') or os.getenv('BLUEPRINT_TASK_EVALUATION_CONFIGURED_CONTROLS_STATE_ROOT')
            or str(Path(config['scene_root']).parent/'task-evaluation-configured-controls'))
        binding = state/source['launch_id']/'cpu-robot-binding'
        worker._require(not any(p.is_symlink() for p in (path, binding, *binding.parents)), 'terminal_deploy_path_unsafe')
        if binding.exists():
            rows.append({'intent_id': intent_id, 'status': 'retained_started_materialization'})
            continue
        result = adoption.provision_terminal_controls_adoption(config=config, catalog=catalog,
            intent_id=intent_id, expected_production_commit=expected_commit, now=moment)
        rows.append({'intent_id': intent_id, 'status': result['status'], 'receipt_digest': result['receipt_digest']})
    return {'status': 'prepared', 'schema_version': 'task_evaluation_terminal_controls_deploy.v1',
            'source_commit': expected_commit, 'rows': rows, 'provider_mutation_performed': False,
            'model_called': False, 'placement_materialized': False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--expected-commit', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(config_path=args.config, expected_commit=args.expected_commit), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
