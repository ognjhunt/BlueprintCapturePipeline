"""Idempotent control-plane finalization of an already authorized operator canary.

No allocator, policy, model, cleanup, or messaging transport is invoked implicitly.
Callers supply read-only closeout refreshers and the existing Website sync plus
verified download/inbox readback. Native producer files are never edited. Email
status is recorded separately and is not a completion gate (operator waiver).
"""
from __future__ import annotations

from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Mapping

from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from .native_task_arena_policy_canary_session import (
    CANDIDATE_IDS, validate_provider_bundle,
    validate_runtime_input_manifest, validate_session_authority,
)
from .task_evaluation_policy_canary_dispatcher import (
    _adapter_instance_ids, _join_session_closeout, _materialize_official_billing_if_posted,
    _partial_policy_canary_result, _recovered_complete_policy_canary_result,
    _sealed_provider_zero, validate_policy_canary_execution_setup,
)
from .task_evaluation_policy_canary_result_projection import build_policy_canary_result_projection
from .task_evaluation_result_delivery import (
    materialize_policy_canary_result_delivery, materialize_policy_canary_website_delivery,
    strict_identifier,
)
from .vast_official_billing_extractor import validate_vast_official_same_goal_reconciliation

INTENT_SCHEMA = 'operator_policy_canary_terminal_delivery_intent.v1'
RESULT_SCHEMA = 'operator_policy_canary_terminal_delivery.v1'
_RECORD_NAMES = {'registration', 'registration_ack', 'operator_authorization', 'setup',
                 'runtime_inputs', 'session_authority', 'bundle'}
_REQUIRED_PHASES = ('execution', 'official_billing', 'teardown', 'provider_zero',
                    'artifact_package', 'website_publication', 'download_readback', 'owner_inbox')


class OperatorTerminalDeliveryError(ValueError):
    pass


@dataclass(frozen=True)
class OperatorTerminalDeliveryAdapters:
    """Trusted adapters; None leaves that phase pending and performs no request."""
    sync_runner: Callable[..., Mapping[str, Any]] | None = None
    download_readback: Callable[..., Mapping[str, Any]] | None = None
    provider_zero_reader: Callable[..., Mapping[str, Any]] | None = None
    official_billing_refresher: Callable[..., Any] | None = None


def file_record(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise OperatorTerminalDeliveryError('operator_terminal_artifact_missing_or_symlink')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path.resolve()), 'sha256': 'sha256:' + digest, 'size_bytes': path.stat().st_size}


def _read(path: str | Path) -> dict[str, Any]:
    file_record(path)
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise OperatorTerminalDeliveryError('operator_terminal_document_invalid')
    return value


def _record_path(record: Mapping[str, Any], locations: Mapping[str, Any] | None = None) -> Path:
    relocated = (locations or {}).get(str(record.get('path')), record)
    if any(relocated.get(key) != record.get(key) for key in ('sha256', 'size_bytes')):
        raise OperatorTerminalDeliveryError('operator_terminal_relocation_digest_mismatch')
    actual = file_record(relocated['path'])
    if any(actual[key] != record.get(key) for key in ('sha256', 'size_bytes')):
        raise OperatorTerminalDeliveryError('operator_terminal_artifact_digest_mismatch')
    return Path(actual['path'])



def _declared_file_records(value):
    if isinstance(value, Mapping):
        if {'path', 'sha256', 'size_bytes'} <= value.keys():
            yield value
        elif {'path', 'digest', 'size_bytes'} <= value.keys():
            yield {'path':value['path'], 'sha256':value['digest'], 'size_bytes':value['size_bytes']}
        for child in value.values():
            yield from _declared_file_records(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from _declared_file_records(child)


def _without_ephemeral_transport(value):
    """Strip capability-bearing URLs/tokens before any diagnostic or final write."""
    if isinstance(value, Mapping):
        return {key: _without_ephemeral_transport(child) for key, child in value.items()
                if str(key).lower() not in {'ephemeral_downloads', 'authorization', 'headers', 'token', 'api_key', 'secret'}
                and not str(key).lower().endswith(('_url', '_urls', '_token'))}
    if isinstance(value, (list, tuple)):
        return [_without_ephemeral_transport(child) for child in value]
    if isinstance(value, str) and re.search(r'https?://', value):
        return '[redacted-url]'
    return value


def _clean_readback(value):
    clean = _without_ephemeral_transport(value)
    if clean != value:
        clean['source_readback_digest'] = value['readback_digest']
        clean['ephemeral_download_urls_retained'] = False
        clean['readback_digest'] = canonical_digest(clean, digest_field='readback_digest')
    return clean


def _seal(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    """Never replace a completed stage, including one left by a crashed caller."""
    body = dict(value)
    if path.exists():
        if _read(path) != body:
            raise OperatorTerminalDeliveryError('operator_terminal_sealed_stage_conflict:' + path.name)
        return body
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(body, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    descriptor, temporary_name = tempfile.mkstemp(prefix='.' + path.name + '-', dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)  # Publish only fully written bytes; never replace a sealed receipt.
        except FileExistsError:
            if _read(path) != body:
                raise OperatorTerminalDeliveryError('operator_terminal_sealed_stage_conflict:' + path.name)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return body


def materialize_operator_registration_alias(intent: Mapping[str, Any]) -> dict[str, Any]:
    """Retain the verified registration's exact bytes at the artifact-router path.

    Call only after full intent validation. A conflicting alias is never replaced.
    Relocation changes custody, not registration bytes, owner or authorization.
    """
    expected = intent['records']['registration']
    source = _record_path(expected, intent.get('artifact_locations', {}))
    payload = source.read_bytes()
    if ('sha256:' + hashlib.sha256(payload).hexdigest() != expected['sha256']
            or len(payload) != expected['size_bytes']):
        raise OperatorTerminalDeliveryError('operator_terminal_registration_source_changed')
    root = Path(intent['run_root'])
    if not root.is_absolute() or root.is_symlink():
        raise OperatorTerminalDeliveryError('operator_terminal_run_root_invalid')
    root.mkdir(parents=True, exist_ok=True)
    destination = root / 'website-operator-registration.json'
    if not destination.exists() and not destination.is_symlink():
        descriptor, name = tempfile.mkstemp(prefix='.operator-registration-', dir=root)
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, 'wb') as stream:
                os.fchmod(stream.fileno(), source.stat().st_mode & 0o777)
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, destination)
            except FileExistsError:
                pass
            directory = os.open(root, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            temporary.unlink(missing_ok=True)
    actual = file_record(destination)
    if any(actual[key] != expected[key] for key in ('sha256', 'size_bytes')):
        raise OperatorTerminalDeliveryError('operator_terminal_registration_alias_conflict')
    return actual


def _control_plane_path(value: str | Path) -> str:
    """Validate a remote POSIX path without consulting the author's filesystem."""
    path = os.fspath(value)
    if (not isinstance(path, str) or not path.startswith('/') or path.startswith('//')
            or '\x00' in path or '\\' in path
            or any(part in {'.', '..'} for part in path.split('/'))):
        raise OperatorTerminalDeliveryError('operator_terminal_control_plane_path_invalid')
    return path


def operator_terminal_delivery_intent(*, run_id: str, run_root: str | Path,
        records: Mapping[str, Mapping[str, Any]], allocator_result_path: str | Path,
        official_billing_path: str | Path, provider_zero_path: str | Path,
        billing_audit_root: str | Path, native_result_path: str | Path | None = None,
        artifact_locations: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Seal pointers for late closeout files and exact bytes for seven fixed inputs.

    records keys: registration, registration_ack, operator_authorization, setup,
    runtime_inputs, session_authority, bundle. artifact_locations may map an
    original recorded path to relocated identical bytes; it cannot change digests.
    A CP recovery adapter may occupy allocator_result_path, but must reference
    actual teardown/object-cleanup/native records and the same bundle/instance.
    """
    value = {'schema_version': INTENT_SCHEMA, 'run_id': run_id, 'run_root': _control_plane_path(run_root),
             'records': dict(records), 'allocator_result_path': _control_plane_path(allocator_result_path),
             'official_billing_path': _control_plane_path(official_billing_path),
             'provider_zero_path': _control_plane_path(provider_zero_path),
             'billing_audit_root': _control_plane_path(billing_audit_root),
             'native_result_path': _control_plane_path(native_result_path) if native_result_path is not None else None,
             'artifact_locations': dict(artifact_locations or {})}
    value['intent_digest'] = canonical_digest(value, digest_field='intent_digest')
    return value


def _inputs(intent, *, require_relocated=False):
    if (intent.get('schema_version') != INTENT_SCHEMA
            or intent.get('intent_digest') != canonical_digest(intent, digest_field='intent_digest')
            or set(intent.get('records', {})) != _RECORD_NAMES):
        raise OperatorTerminalDeliveryError('operator_terminal_intent_invalid')
    strict_identifier(intent['run_id'], field='run_id', max_length=192)
    locations = intent.get('artifact_locations', {})
    def resolved(record):
        if require_relocated and str(record.get('path')) not in locations:
            raise OperatorTerminalDeliveryError('operator_terminal_nested_relocation_missing')
        return _record_path(record, locations)
    values = {name: _read(resolved(record)) for name, record in intent['records'].items()}
    registration, authorization = values['registration'], values['operator_authorization']
    ack = values['registration_ack'].get('response', values['registration_ack'])
    if (registration.get('schema_version') != 'task_evaluation_operator_policy_canary_registration.v1'
            or registration.get('registration_digest') != cross_runtime_canonical_digest(registration, digest_field='registration_digest')
            or authorization.get('schema_version') != 'task_evaluation_operator_policy_authorization.v1'
            or authorization.get('authorization_digest') != canonical_digest(authorization, digest_field='authorization_digest')
            or registration.get('operator_authorization_digest') != authorization['authorization_digest']
            or authorization.get('run_id') != intent['run_id']
            or ack.get('schema_version') != 'task_evaluation_operator_policy_canary_registration_receipt.v1'
            or ack.get('status') != 'registered' or ack.get('run_id') != intent['run_id']
            or ack.get('registration_digest') != registration['registration_digest']):
        raise OperatorTerminalDeliveryError('operator_terminal_registration_not_authorized')
    setup = values['setup']
    if setup.get('setup_digest') != canonical_digest(setup, digest_field='setup_digest'):
        raise OperatorTerminalDeliveryError('operator_terminal_setup_digest_invalid')
    # Relocation is a validation view only. Original setup bytes/digest remain authoritative.
    view = json.loads(json.dumps(setup))
    for name, record in view['records'].items():
        view['records'][name] = file_record(resolved(record))
    view['setup_digest'] = canonical_digest(view, digest_field='setup_digest')
    validate_policy_canary_execution_setup(view)
    runtime = validate_runtime_input_manifest(values['runtime_inputs'])
    authority = validate_session_authority(values['session_authority'])
    bundle = validate_provider_bundle(values['bundle'], authority=authority)
    if (any(value.get('run_id') != intent['run_id'] for value in (registration, runtime, authority))
            or any(value.get('run_kind') != 'internal_policy_canary' or value.get('claim_ceiling') != 'diagnostic_policy_execution'
                   for value in (registration, authorization, setup, runtime, authority))
            or registration.get('setup_digest') != setup['setup_digest']
            or registration.get('runtime_inputs_digest') != runtime['runtime_inputs_digest']
            or authority['runtime_inputs_digest'] != runtime['runtime_inputs_digest']
            or registration.get('plan_digest') != runtime['plan_digest']
            or registration.get('configuration_digest') != runtime['configuration_digest']
            or any(registration.get(key) != setup.get(key) for key in ('capture_session_id', 'intake_id', 'request_digest'))
            or registration.get('task_success_contract') != runtime['task_success_contract']):
        raise OperatorTerminalDeliveryError('operator_terminal_input_identity_mismatch')
    omission = None
    omitted = [cell['control_diagnostic'].get('omission_authority') for cell in runtime['cells']
               if cell['control_diagnostic'].get('mode') == 'nonblocking_omitted_by_user']
    if omitted:
        omission = omitted[0]
        if (len(omitted) != len(runtime['cells']) or any(row != omission for row in omitted)
                or registration.get('control_omission_authority_digest') != omission['authority_digest']
                or authorization.get('control_omission_authority') != omission):
            raise OperatorTerminalDeliveryError('operator_terminal_control_omission_binding_invalid')
    specs = {}
    for candidate, role in zip(CANDIDATE_IDS, ('pi05_execution_spec', 'groot_execution_spec'), strict=True):
        specs[candidate] = _read(resolved(setup['records'][role]))
    declared = [*intent['records'].values(), *_declared_file_records(values), *_declared_file_records(specs)]
    for record in declared:
        resolved(record)
    paths = sorted({str(record['path']) for record in declared})
    return values | {'setup': setup, 'runtime_inputs': runtime, 'session_authority': authority,
                     'bundle': bundle, 'omission': omission, 'specs': specs,
                     'artifact_relocation': {'declared_file_record_count':len(paths),
                         'unmapped_record_paths':[path for path in paths if path not in locations]}}


def validate_operator_terminal_delivery_inputs(intent: Mapping[str, Any], *,
                                               require_relocated: bool = False) -> dict[str, Any]:
    """Read-only handoff validation of fixed JSON and every nested file reference.

    For cross-host readiness set require_relocated=True. artifact_locations must
    cover every original path, including identity mappings for already-local
    fixed records. Raw producer JSON is never rewritten to change its paths.
    """
    values = _inputs(intent, require_relocated=require_relocated)
    if require_relocated and values['artifact_relocation']['unmapped_record_paths']:
        raise OperatorTerminalDeliveryError('operator_terminal_nested_relocation_missing')
    return values


def _pending(root, intent, phases, blockers):
    value = {'schema_version': RESULT_SCHEMA, 'status': 'pending', 'run_id': intent['run_id'],
             'intent_digest': intent['intent_digest'], 'phases': phases, 'blockers': list(blockers),
             'all_required_phases_done': False, 'allocator_invoked': False,
             'policy_execution_repeated': False, 'native_producer_history_modified': False}
    value['result_digest'] = canonical_digest(value, digest_field='result_digest')
    path = root/'operator_terminal_delivery'/'pending.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')
    temporary.replace(path)
    return value


def _attempt(root, phase, value):
    directory = root/'operator_terminal_delivery'/'attempts'
    directory.mkdir(parents=True, exist_ok=True)
    index = len(list(directory.glob(phase+'-*.json'))) + 1
    path = directory/f'{phase}-{index:04d}.json'
    safe = _without_ephemeral_transport(value)
    if safe != value:
        safe['transport_fields_redacted'] = True
        safe['source_response_digest'] = canonical_digest(value)
    _seal(path, safe)
    return path


def _closure(intent, adapter, values, adapters, root, phases):
    """Verify producer closeout receipts; never manufacture absence/cleanup flags."""
    locations = intent.get('artifact_locations', {})
    closeout = adapter.get('provider_closeout') or {}
    instance_ids = _adapter_instance_ids(adapter)
    if (len(instance_ids) != 1 or adapter.get('bundle_sha256') != values['bundle']['bundle_sha256']
            or adapter.get('retry_cap') != 0):
        raise OperatorTerminalDeliveryError('operator_terminal_adapter_identity_invalid')
    if adapter.get('continuing_spend_from_this_run') is not False:
        return None, ['operator_terminal_allocator_not_closed']
    teardown_record = closeout.get('teardown_manifest')
    if not isinstance(teardown_record, Mapping):
        return None, ['operator_terminal_teardown_missing']
    try:
        teardown_path = _record_path(teardown_record, locations)
    except (OSError, OperatorTerminalDeliveryError):
        return None, ['operator_terminal_teardown_missing_or_unverified']
    teardown = _read(teardown_path)
    if (teardown.get('schema_version') != 'vast_teardown_manifest.v1'
            or teardown.get('status') != 'completed' or teardown.get('runner_gpu_teardown_completed') is not True
            or teardown.get('continuing_spend_from_this_run') is not False
            or teardown.get('vast_instance_ids') != instance_ids):
        return None, ['operator_terminal_teardown_not_completed']
    cleanup_raw = str(adapter.get('object_store_cleanup_path') or '')
    cleanup_location = locations.get(cleanup_raw, {}).get('path', cleanup_raw)
    if not cleanup_raw or not Path(cleanup_location).is_file():
        return None, ['operator_terminal_staged_object_cleanup_missing']
    if cleanup_raw in locations:
        _record_path(locations[cleanup_raw])
    cleanup = _read(cleanup_location)
    objects = cleanup.get('objects')
    if (cleanup.get('schema_version') != 'wam_provider_object_store_cleanup.v1'
            or cleanup.get('status') != 'completed' or cleanup.get('all_objects_absent') is not True
            or cleanup.get('all_ephemeral_objects_absent') is not True or cleanup.get('blockers') != []
            or not isinstance(objects, list) or cleanup.get('exact_object_count') != len(objects)
            or any(row.get('absence', {}).get('absence_confirmed') is not True for row in objects)
            or closeout.get('provider_zero_confirmed') is not True
            or closeout.get('warm_session_retained') is not False or closeout.get('all_staged_objects_absent') is not True):
        return None, ['operator_terminal_staged_object_cleanup_not_verified']
    phases['teardown'] = {'status': 'complete', 'receipt': file_record(teardown_path), 'staged_object_cleanup': file_record(cleanup_location)}
    retained_zero = root/'operator_terminal_delivery'/'provider_zero.json'
    zero_path = retained_zero if retained_zero.exists() else Path(intent['provider_zero_path'])
    zero = _sealed_provider_zero(zero_path)
    if zero is None and adapters.provider_zero_reader is not None:
        try:
            observed = dict(adapters.provider_zero_reader(intent=intent, adapter=adapter))
        except Exception as exc:
            observed = {'status':'failed','reason':type(exc).__name__}
        attempt_path = _attempt(root, 'provider-zero', observed)
        if _sealed_provider_zero(attempt_path) is not None:
            _seal(retained_zero, observed)
            zero_path, zero = retained_zero, observed
    if zero is None:
        return None, ['operator_terminal_provider_zero_pending']
    phases['provider_zero'] = {'status': 'complete', 'receipt': file_record(zero_path)}
    billing_path = Path(intent['official_billing_path'])
    if not billing_path.is_file() and adapters.official_billing_refresher is not None:
        try:
            adapters.official_billing_refresher(intent=intent, adapter=adapter)
        except Exception as exc:
            _attempt(root, 'billing-refresh', {'status':'failed','reason':type(exc).__name__})
    if not billing_path.is_file():
        _materialize_official_billing_if_posted(billing_audit_root=intent['billing_audit_root'],
            adapter_result_path=Path(intent['allocator_result_path']), adapter=adapter,
            launch_label=values['session_authority']['resource_name'], output_path=billing_path)
    if not billing_path.is_file():
        return None, ['operator_terminal_official_billing_pending']
    billing = validate_vast_official_same_goal_reconciliation(billing_path)
    entries = [row for row in billing['entries'] if row.get('provider_instance_id') == instance_ids[0]
               and row.get('launch_label') == values['session_authority']['resource_name']]
    if len(entries) != 1:
        raise OperatorTerminalDeliveryError('operator_terminal_official_billing_identity_mismatch')
    terminal = entries[0].get('terminal_execution_evidence', {}).get('terminal_result', {})
    if any(terminal.get(key) != file_record(intent['allocator_result_path'])[key] for key in ('sha256', 'size_bytes')):
        raise OperatorTerminalDeliveryError('operator_terminal_official_billing_terminal_binding_mismatch')
    phases['official_billing'] = {'status': 'complete', 'official_total_usd': entries[0]['official_charge_usd'],
        'reconciliation_total_usd': billing['official_total_usd'], 'receipt': file_record(billing_path)}
    closure = {'billing': {**file_record(billing_path), 'official_billing_sealed': True},
               'teardown': {**file_record(teardown_path), 'teardown_completed': True},
               'provider_zero': {**file_record(zero_path), 'provider_zero_verified': True}}
    return {'records': closure, 'billing': billing, 'run_official_cost_usd': entries[0]['official_charge_usd'],
            'provider_zero': zero, 'teardown': teardown}, []


def _native_source(intent, adapter):
    source = Path(intent.get('native_result_path') or adapter.get('native_control_result_path') or '')
    if not source.is_file():
        return None
    raw = _read(source)
    if (raw.get('schema_version') != 'native_task_arena_policy_canary_session_result.v1'
            or raw.get('result_digest') != canonical_digest(raw, digest_field='result_digest')
            or raw.get('result_digest') != adapter.get('native_control_result_digest')
            or raw.get('run_kind') != 'internal_policy_canary' or raw.get('claim_ceiling') != 'diagnostic_policy_execution'):
        raise OperatorTerminalDeliveryError('operator_terminal_native_result_invalid')
    return raw, source


def _native_result(intent, values, adapter, root, observed=None):
    observed = observed or _native_source(intent, adapter)
    if observed is None:
        return None
    raw, source = observed
    if raw.get('status') == 'runtime_completed_unqualified_pending_closeout' and len(raw.get('episodes', [])) == 20:
        return raw, source.parent, file_record(source)
    # Helpers may write derived aggregate/gap files. Work on copies of JSON,
    # hardlink only bulk immutable evidence, and never edit the producer tree.
    work = root/'operator_terminal_delivery'/'recovery_attempt'
    evidence = work/'immutable_execution'
    if not evidence.exists():
        staging = work.with_name(work.name+'.staging')
        staging.mkdir(parents=True, exist_ok=True)
        def copy(source_name, destination_name):
            if Path(destination_name).is_symlink():
                raise OperatorTerminalDeliveryError('operator_terminal_recovery_symlink')
            if Path(destination_name).exists() and file_record(destination_name)['sha256'] == file_record(source_name)['sha256']:
                return destination_name
            if Path(source_name).suffix.lower() == '.json':
                return shutil.copy2(source_name, destination_name)
            try:
                os.link(source_name, destination_name)
                return destination_name
            except OSError:
                return shutil.copy2(source_name, destination_name)
        for path in source.parent.rglob('*'):
            if path.is_symlink():
                raise OperatorTerminalDeliveryError('operator_terminal_native_evidence_symlink')
        shutil.copytree(source.parent, staging/'immutable_execution', copy_function=copy, dirs_exist_ok=True)
        command = Path(str(adapter.get('attempt_root') or ''))/'vast_provider_run/vast_provider_command_result.json'
        if command.is_file():
            (staging/'vast_provider_run').mkdir(exist_ok=True)
            shutil.copy2(command, staging/'vast_provider_run/vast_provider_command_result.json')
        staging.rename(work)
    native = evidence/source.name
    relocated_adapter = dict(adapter, attempt_root=str(work))
    recovered = _recovered_complete_policy_canary_result(root=root/'operator_terminal_delivery', native_path=native,
        adapter=relocated_adapter, authority=values['session_authority'], runtime_inputs=values['runtime_inputs'])
    if recovered is not None:
        result, path = recovered
        return result, path.parent, file_record(source)
    partial = _partial_policy_canary_result(native_path=native, fallback=raw,
        runtime_inputs=values['runtime_inputs'], specs=values['specs'])
    if partial is not None:
        result, path = partial
        return result, path.parent, file_record(source)
    return None



def _existing_package(root, values, closure):
    """Adopt a prior sealed dispatcher/manual package without changing its digest."""
    folder = root/'artifacts'/'result_delivery'
    if not (folder/'delivery.json').is_file():
        return None
    delivery, joined, registry = [_read(folder/name) for name in
        ('delivery.json','policy_canary_full_report.json','artifact_registry.json')]
    report = delivery.get('report', {}).get('machine_readable_report', {})
    actual_report = file_record(folder/'policy_canary_full_report.json')
    if (delivery.get('delivery_digest') != cross_runtime_canonical_digest(delivery,digest_field='delivery_digest')
            or delivery.get('run_id') != values['registration']['run_id']
            or delivery.get('claim_ceiling') != 'diagnostic_policy_execution'
            or joined.get('run_id') != values['registration']['run_id']
            or joined.get('result_digest') != canonical_digest(joined,digest_field='result_digest')
            or joined.get('task_success_contract') != values['runtime_inputs']['task_success_contract']
            or report.get('digest') != actual_report['sha256'] or report.get('size_bytes') != actual_report['size_bytes']
            or registry.get('registry_digest') != canonical_digest(registry,digest_field='registry_digest')
            or registry.get('delivery_digest') != delivery['delivery_digest']):
        raise OperatorTerminalDeliveryError('operator_terminal_existing_package_invalid')
    for role, record in closure['records'].items():
        prior = delivery.get('closure', {}).get(role, {})
        if prior.get('digest') != record['sha256'] or prior.get('size_bytes') != record['size_bytes']:
            raise OperatorTerminalDeliveryError('operator_terminal_existing_closure_mismatch')
    omission = values['omission']
    if omission is not None and delivery.get('control_omission', {}).get('authority_digest') != omission['authority_digest']:
        raise OperatorTerminalDeliveryError('operator_terminal_existing_control_omission_mismatch')
    return joined, delivery


def _sync_matches(sync, registration, runtime, delivery, projection):
    expected = {'run_id': registration['run_id'], 'capture_session_id': registration['capture_session_id'],
                'intake_id': registration['intake_id'], 'request_digest': registration['request_digest'],
                'configuration_digest': runtime['configuration_digest'], 'plan_digest': runtime['plan_digest'],
                'operator_registration_digest': registration['registration_digest'],
                'result_delivery_digest': delivery['delivery_digest'], 'policy_canary_projection_digest': projection['projection_digest']}
    response = sync.get('response', {})
    return (sync.get('status') == 'succeeded'
            and response.get('schema_version') in {'capture_task_evaluation_policy_canary_publication_receipt.v1',
                                                  'capture_task_evaluation_run_publication_receipt.v1'}
            and response.get('status') == delivery['result_status']
            and isinstance(response.get('already_exists'), bool)
            and all(response.get(key, sync.get(key)) == value for key, value in expected.items()))


def _readback_matches(readback, registration, delivery, projection):
    expected = {'run_id': registration['run_id'], 'operator_registration_digest': registration['registration_digest'],
                'result_delivery_digest': delivery['delivery_digest'], 'policy_canary_projection_digest': projection['projection_digest']}
    if (readback.get('status') != 'verified' or any(readback.get(k) != v for k,v in expected.items())
            or readback.get('readback_digest') != canonical_digest(readback, digest_field='readback_digest')):
        return False
    artifacts = readback.get('artifacts')
    wanted = {(row['artifact_id'], row['digest'], row['size_bytes']) for row in delivery['artifacts']}
    if (not isinstance(artifacts, list) or len(artifacts) != len(wanted)
            or any(row.get('verified') is not True or row.get('http_status') != 200 for row in artifacts)
            or {(row.get('artifact_id'),row.get('sha256'),row.get('size_bytes')) for row in artifacts} != wanted):
        return False
    inbox = readback.get('inbox', {})
    return (inbox.get('status') == 'verified' and inbox.get('run_id') == registration['run_id']
            and inbox.get('projection_digest') == projection['projection_digest']
            and inbox.get('team_namespace') == registration['team_namespace']
            and inbox.get('source') in {'website_owner_run_index_readback', 'website_authenticated_run_readback'})


def finalize_operator_policy_canary(intent: Mapping[str, Any], *,
        adapters: OperatorTerminalDeliveryAdapters = OperatorTerminalDeliveryAdapters()) -> dict[str, Any]:
    """Resume terminal phases only; `completed` requires Website+download+inbox proof."""
    values = _inputs(intent)
    root = Path(intent['run_root'])
    if root.is_symlink() or not root.is_absolute():
        raise OperatorTerminalDeliveryError('operator_terminal_run_root_invalid')
    root.mkdir(parents=True, exist_ok=True)
    metadata = root/'operator_terminal_delivery'
    metadata.mkdir(exist_ok=True)
    with (metadata/'.lock').open('a+b') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {'schema_version': RESULT_SCHEMA, 'status': 'pending', 'run_id': intent['run_id'],
                    'all_required_phases_done': False, 'blockers': ['operator_terminal_finalization_in_progress']}
        _seal(metadata/'intent.json', dict(intent))
        # The authenticated artifact router admits operator runs through this
        # canonical registration alias. Preserve exact relocated input bytes,
        # including when resuming an already sealed publication or completion.
        materialize_operator_registration_alias(intent)
        final = metadata/'completed.json'
        if final.is_file():
            receipt = _read(final)
            if receipt.get('result_digest') != canonical_digest(receipt, digest_field='result_digest'):
                raise OperatorTerminalDeliveryError('operator_terminal_final_receipt_invalid')
            for record in receipt['sealed_records'].values():
                _record_path(record)
            return receipt
        phases = {name: {'status': 'pending'} for name in _REQUIRED_PHASES}
        phases.update(qualification={'status':'unqualified','claim_ceiling':'diagnostic_policy_execution'},
                      notification={'status':'not_observed','required_for_completion':False})
        allocator_path = Path(intent['allocator_result_path'])
        if not allocator_path.is_file():
            return _pending(root,intent,phases,['operator_terminal_allocator_result_pending'])
        adapter = _read(allocator_path)
        observed_native = _native_source(intent, adapter)
        if observed_native is not None:
            phases['execution'] = {'status':'complete','producer_status':observed_native[0]['status'],
                'native_episode_count':len(observed_native[0].get('episodes', [])), 'delivery_finalized':False}
        closure, blockers = _closure(intent,adapter,values,adapters,root,phases)
        if closure is None:
            return _pending(root,intent,phases,blockers)
        joined_path = metadata/'terminal_result.json'
        package_stage = metadata/'packaged.json'
        if package_stage.is_file():
            staged = _read(package_stage)
            if staged.get('intent_digest') != intent['intent_digest'] or staged.get('stage_digest') != canonical_digest(staged, digest_field='stage_digest'):
                raise OperatorTerminalDeliveryError('operator_terminal_packaged_stage_invalid')
            for record in staged['records'].values():
                _record_path(record)
            joined, delivery, website, projection = [_read(staged['records'][key]['path'])
                for key in ('terminal','delivery','website_delivery','projection')]
        else:
            existing = _existing_package(root,values,closure)
            if existing is not None:
                if observed_native is None:
                    return _pending(root,intent,phases,['operator_terminal_native_provenance_pending'])
                joined, delivery = existing
                original = file_record(observed_native[1])
                _seal(joined_path,joined)
            else:
                native = _native_result(intent,values,adapter,root,observed=observed_native)
                if native is None:
                    return _pending(root,intent,phases,['operator_terminal_native_result_or_recovery_pending'])
                inner,evidence,original = native
                joined = _join_session_closeout(inner=inner,adapter=adapter,provider_zero=closure['provider_zero'])
                joined.update(run_id=intent['run_id'],configuration_digest=values['runtime_inputs']['configuration_digest'],
                    scene_revision_digest=values['setup']['scene_revision_digest'],provider='vast',
                    provider_instance_ids=_adapter_instance_ids(adapter),official_total_usd=closure['run_official_cost_usd'])
                joined['result_digest'] = canonical_digest(joined,digest_field='result_digest')
                _seal(joined_path,joined)
                delivery = materialize_policy_canary_result_delivery(run_root=root,run_id=intent['run_id'],
                    result_status=joined['status'],session_result=joined,evidence_root=evidence,
                    closure_records=closure['records'],control_omission_authority=values['omission'])
            website = materialize_policy_canary_website_delivery(run_root=root,delivery=delivery)
            projection = build_policy_canary_result_projection(setup=values['setup'],result=joined,delivery=website)
            projection_path = metadata/'website_projection.json'
            _seal(projection_path,projection)
            staged = {'intent_digest':intent['intent_digest'],'records':{'original_native_result':original,
                'terminal':file_record(joined_path),'delivery':file_record(root/'artifacts/result_delivery/delivery.json'),
                'website_delivery':file_record(root/'artifacts/result_delivery/website_delivery.json'),
                'projection':file_record(projection_path),
                'artifact_registry':file_record(root/'artifacts/result_delivery/artifact_registry.json')}}
            staged['stage_digest'] = canonical_digest(staged,digest_field='stage_digest')
            _seal(package_stage,staged)
        phases['execution'] = {'status':'complete','result_status':joined['status'],
            'candidate_policy_queried':joined.get('candidate_policy_queried') is True}
        phases['artifact_package'] = {'status':'complete','delivery_digest':website['delivery_digest']}
        publication_path = metadata/'website_publication.json'
        registration,runtime = values['registration'],values['runtime_inputs']
        if publication_path.is_file():
            sync = _read(publication_path)
        elif adapters.sync_runner is None:
            return _pending(root,intent,phases,['operator_terminal_website_sync_adapter_missing'])
        else:
            try:
                sync = dict(adapters.sync_runner(capture_session_id=registration['capture_session_id'],
                intake_id=registration['intake_id'],run_id=intent['run_id'],request_digest=registration['request_digest'],
                configuration_digest=runtime['configuration_digest'],plan_digest=runtime['plan_digest'],
                operator_registration_digest=registration['registration_digest'],result_status=joined['status'],
                result_delivery=website,policy_canary_result=projection))
            except Exception as exc:
                sync = {'status':'failed','reason':type(exc).__name__}
            _attempt(root,'website-sync',sync)
        if not _sync_matches(sync,registration,runtime,website,projection):
            return _pending(root,intent,phases,['operator_terminal_website_publication_unverified'])
        _seal(publication_path,_without_ephemeral_transport(sync))
        phases['website_publication'] = {'status':'complete','receipt':file_record(publication_path)}
        phases['notification'] = {**_without_ephemeral_transport(dict(sync.get('notification_delivery') or {})),'required_for_completion':False,
                                  'email_inbox_delivery_claimed':False}
        readback_path = metadata/'download_inbox_readback.json'
        if readback_path.is_file():
            readback = _read(readback_path)
        elif adapters.download_readback is None:
            return _pending(root,intent,phases,['operator_terminal_download_inbox_readback_pending'])
        else:
            try:
                readback = dict(adapters.download_readback(run_root=root,registration=registration,
                result_delivery=website,policy_canary_result=projection,publication=sync))
            except Exception as exc:
                readback = {'status':'failed','reason':type(exc).__name__}
            _attempt(root,'download-inbox-readback',readback)
        if not _readback_matches(readback,registration,website,projection):
            return _pending(root,intent,phases,['operator_terminal_download_or_owner_inbox_unverified'])
        readback = _clean_readback(readback)
        _seal(readback_path,readback)
        phases['download_readback'] = {'status':'complete','artifact_count':len(readback['artifacts'])}
        phases['owner_inbox'] = dict(readback['inbox'],status='complete')
        receipt = {'schema_version':RESULT_SCHEMA,'status':'completed','run_id':intent['run_id'],
            'intent_digest':intent['intent_digest'],'operator_registration_digest':registration['registration_digest'],
            'plan_digest':runtime['plan_digest'],'execution_status':joined['status'],'phases':phases,
            'all_required_phases_done':True,'allocator_invoked':False,'policy_execution_repeated':False,
            'native_producer_history_modified':False,'email_delivery_required_for_completion':False,
            'sealed_records':{**staged['records'],'publication':file_record(publication_path),
                'download_inbox_readback':file_record(readback_path),**closure['records']},'blockers':[]}
        receipt['result_digest'] = canonical_digest(receipt,digest_field='result_digest')
        _seal(final,receipt)
        return receipt
