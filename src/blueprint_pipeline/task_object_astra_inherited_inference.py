"""Conserve original-run inference journals when adopting partial Astra work."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_supervisor.inference_reservations import InferenceReservationAudit
from .task_object_astra_authoring import AssetAuthoringError

INHERITED_FILE = 'inherited_authoring_inference.json'


def inherited_balance(root: Path, run_id: str) -> dict:
    """Validate copied journals under their original IDs; never treat estimates as bills."""
    path = root / INHERITED_FILE
    if not path.exists():
        return {'cost_usd': 0.0, 'call_count': 0, 'journals': []}
    value = json.loads(path.read_text())
    if (path.is_symlink() or (root / 'inherited').is_symlink() or value.get('schema_version') != 'astra_inherited_inference.v1'
            or value.get('successor_run_id') != run_id
            or value.get('inheritance_digest') != canonical_digest(value, digest_field='inheritance_digest')
            or not isinstance(value.get('journals'), list) or not value['journals']):
        raise AssetAuthoringError('astra_inherited_inference_binding_invalid')
    cost, calls, seen = 0.0, 0, set()
    for record in value['journals']:
        original_run = record.get('run_id')
        relative = Path(record.get('relative_root', ''))
        journal = root / relative
        if (not original_run or original_run == run_id or original_run in seen
                or relative.is_absolute() or '..' in relative.parts
                or relative.parts[:1] != ('inherited',) or journal.is_symlink()
                or not journal.resolve().is_relative_to((root / 'inherited').resolve())):
            raise AssetAuthoringError('astra_inherited_inference_journal_invalid')
        seen.add(original_run)
        for item in journal.rglob('*'):
            if item.is_symlink():
                raise AssetAuthoringError('astra_inherited_inference_symlink')
        manifest = InferenceReservationAudit(run_root=journal, run_id=original_run).manifest()
        if (manifest['in_flight_unknown_count'] or not manifest['reservation_count']
                or record.get('manifest_digest') != manifest['inference_reservation_manifest_digest']):
            raise AssetAuthoringError('astra_inherited_inference_changed_or_unresolved')
        cost += manifest['reserved_max_cost_usd']
        calls += manifest['reservation_count']
    return {'cost_usd': cost, 'call_count': calls, 'journals': value['journals']}


def inherit_completed_inference(*, source: Path, destination: Path,
                               source_run_id: str, successor_run_id: str) -> dict:
    """Copy immutable source journals without changing a reservation or completion."""
    if destination.exists() or source_run_id == successor_run_id:
        raise AssetAuthoringError('astra_inherited_inference_destination_invalid')
    old = inherited_balance(source, source_run_id)
    manifest = InferenceReservationAudit(run_root=source, run_id=source_run_id).manifest()
    if manifest['in_flight_unknown_count']:
        raise AssetAuthoringError('astra_inherited_inference_unresolved')
    journals = []
    sources = [(source / row['relative_root'], row['run_id']) for row in old['journals']]
    if manifest['reservation_count']:
        sources.append((source, source_run_id))
    if not sources or successor_run_id in {run for _, run in sources}:
        raise AssetAuthoringError('astra_inherited_inference_cycle_or_empty')
    for original, original_run in sources:
        original_manifest = InferenceReservationAudit(run_root=original, run_id=original_run).manifest()
        relative = Path('inherited') / canonical_digest({'run_id': original_run}).removeprefix('sha256:')
        target = destination / relative
        # Only validated reservation/completion records are copied, never a rewritten ledger.
        for row in original_manifest['reservations']:
            for field in ('reservation_path', 'completion_path'):
                source_file = original / row[field]
                target_file = target / row[field]
                target_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_file, target_file)
        journals.append({'run_id': original_run, 'relative_root': str(relative),
                         'manifest_digest': original_manifest['inference_reservation_manifest_digest']})
    value = {'schema_version': 'astra_inherited_inference.v1', 'successor_run_id': successor_run_id,
             'journals': journals, 'official_billing_proven': False, 'inheritance_digest': ''}
    value['inheritance_digest'] = canonical_digest(value, digest_field='inheritance_digest')
    (destination / INHERITED_FILE).write_text(json.dumps(value, indent=2) + '\n')
    return inherited_balance(destination, successor_run_id)


def copy_inherited_inference(*, source: Path, destination: Path, run_id: str) -> dict:
    """Carry inherited obligations into another attempt of the same successor."""
    balance = inherited_balance(source, run_id)
    if balance['journals']:
        if (destination / INHERITED_FILE).exists() or (destination / 'inherited').exists():
            raise AssetAuthoringError('astra_inherited_inference_destination_exists')
        shutil.copytree(source / 'inherited', destination / 'inherited')
        shutil.copyfile(source / INHERITED_FILE, destination / INHERITED_FILE)
    return inherited_balance(destination, run_id)


def phase_completion_paths(*, budget_root: Path, request_value: dict, phase: dict) -> list[Path]:
    """Follow a digest-bound adopted phase to its original run's conserved journal."""
    from .task_object_astra_authoring import file_record, validate_request
    current_run = request_value['run_id']
    direct = list((budget_root / 'inference_reservations/completed').glob('*.json'))
    if not phase.get('source_phase'):
        return direct
    balance = inherited_balance(budget_root, current_run)
    if not balance['journals']:
        return direct
    current, seen = phase, set()
    for _ in range(16):
        source = current.get('source_phase')
        if source is None:
            break
        path = Path(source['path'])
        if str(path) in seen or file_record(path) != source:
            raise AssetAuthoringError('astra_inherited_phase_lineage_invalid')
        seen.add(str(path))
        original = json.loads(path.read_text())
        request_path = path.parent / 'request.json'
        if not request_path.is_file():
            request_path = path.parent.parent / 'request.json'
        original_request = json.loads(request_path.read_text())
        validate_request(original_request)
        def relevant(row):
            return {k: v for k, v in row.items()
                    if k not in {'run_id', 'request_digest', 'expected_production_commit'}}
        if (original.get('output') != phase.get('output') or original.get('model') != phase.get('model')
                or original.get('request_digest') != original_request['request_digest']
                or relevant(original_request) != relevant(request_value)):
            raise AssetAuthoringError('astra_inherited_phase_source_inputs_changed')
        current = original
    else:
        raise AssetAuthoringError('astra_inherited_phase_lineage_too_deep')
    original_run = original_request['run_id']
    matches = [row for row in balance['journals'] if row['run_id'] == original_run]
    if len(matches) != 1:
        raise AssetAuthoringError('astra_inherited_phase_journal_missing')
    return direct + list((budget_root / matches[0]['relative_root'] / 'inference_reservations/completed').glob('*.json'))
