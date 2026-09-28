"""Prearmed, bounded SAM result recovery through the shared pinned-SSH transport."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping

from .common import write_json
from .gpu_render_providers import enroll_vast_ssh_host_key, _validated_vast_known_hosts_pin
from .vast_provider_output_recovery import _identity_file, recover_provider_output_before_teardown

MAX_RESULT_BYTES = 64 * 1024**2
PUBLIC_KEY_ENV = 'BLUEPRINT_SAM31_RECOVERY_PUBLIC_KEY'
PIN_PREPARATION_BUDGET_SECONDS = 35.  # Existing provider inspect (30s) plus key scan (5s).


def prepare_sam31_output_recovery(*, root: Path, hard_ttl_seconds: int) -> dict[str, Any]:
    identity = _identity_file()
    if identity is None:
        raise ValueError('sam31_recovery_identity_missing_or_insecure')
    derived = subprocess.run(['ssh-keygen', '-y', '-f', str(identity)], input='',
                             capture_output=True, text=True, timeout=10, check=False)
    fields = derived.stdout.strip().split()
    if (derived.returncode != 0 or len(fields) < 2
            or fields[0] not in {'ssh-ed25519', 'ssh-rsa', 'ecdsa-sha2-nistp256'}
            or not re.fullmatch(r'[A-Za-z0-9+/]+={0,2}', fields[1])):
        raise ValueError('sam31_recovery_identity_not_usable')
    public_key = ' '.join(fields[:2])
    receipt = {'schema_version':'sam31_output_recovery_readiness.v1', 'status':'prepared_before_allocation',
               'ssh_identity_file':str(identity), 'public_key_sha256':'sha256:'+hashlib.sha256(public_key.encode()).hexdigest(),
               'vast_launch_mode':'ssh_direct', 'maximum_result_bytes':MAX_RESULT_BYTES,
               'recovery_reserve_seconds':min(300., max(1., hard_ttl_seconds/5)),
               'host_key_pin_required_before_recovery':True, 'provider_mutations_performed':0,
               'private_key_bytes_retained':False}
    write_json(root/'output_recovery_readiness.json', receipt)
    return {'receipt':receipt, 'public_key':public_key}


def pin_sam31_output_recovery(*, provider: Any, instance_id: str, root: Path,
                              timeout_seconds: float) -> dict[str, Any]:
    observed = provider.inspect(instance_id)
    if (observed.get('api_confirmed') is not True or observed.get('provider') != 'vast'
            or str(observed.get('instance_id')) != str(instance_id)
            or not observed.get('ssh_host') or not observed.get('ssh_port')):
        return {'status':'waiting_for_endpoint', 'blockers':['sam31_recovery_endpoint_unavailable']}
    connection = {key:observed[key] for key in ('ssh_host','ssh_port')}
    enrollment = enroll_vast_ssh_host_key(connection, attempt_dir=root,
                                        timeout_seconds=min(5., max(.1, timeout_seconds)))
    path = enrollment.get('known_hosts_file')
    pin = (_validated_vast_known_hosts_pin(path, host=connection['ssh_host'], port=int(connection['ssh_port']))
           if path and enrollment.get('status') == 'enrolled' else None)
    result = {'status':'pinned' if pin else 'waiting_for_pin', 'instance_id':str(instance_id),
              'connection':connection, 'known_hosts_sha256':pin[1] if pin else None,
              'strict_host_key_checking':True, 'blockers':[] if pin else ['sam31_recovery_pin_unavailable']}
    write_json(root/'output_recovery_endpoint.json', result)
    return result


def recover_sam31_output(*, pinned: Mapping[str, Any], root: Path,
                         timeout_seconds: float) -> dict[str, Any]:
    if pinned.get('status') != 'pinned':
        return {'status':'blocked', 'blockers':['sam31_recovery_pin_unavailable']}
    path = root/'provider_runtime_result.ssh-recovered.json'
    result = recover_provider_output_before_teardown(connection=pinned['connection'],
        provider_bundle_kind='sam31_source_tracks', output_path=path, attempt_dir=root,
        expected_size_bytes=None, maximum_size_bytes=MAX_RESULT_BYTES, timeout_seconds=timeout_seconds)
    if result.get('status') == 'completed':
        try:
            value = json.loads(path.read_text())
            if not isinstance(value, dict):
                raise ValueError('not an object')
        except (ValueError, OSError):
            return {**result, 'status':'blocked', 'result_path':str(path),
                    'terminal_invalid_result':True, 'blockers':['sam31_recovery_json_invalid']}
        return {**result, 'result_path':str(path), 'runtime_result':value}
    return result


def receive_sam31_output(*, root: Path, provider: Any, instance_id: str,
        output_get_url: str, result_fetcher, validator, clock, sleeper,
        deadline: float, recovery_reserve_seconds: float) -> tuple[dict | None, dict]:
    """Give normal execution its budget, then recover before the same hard deadline."""
    pinned: dict[str, Any] = {'status':'waiting_for_endpoint'}
    events = []
    poll_deadline = deadline - recovery_reserve_seconds
    last_error = None
    while (now := float(clock())) < poll_deadline:
        if pinned.get('status') != 'pinned' and poll_deadline-now >= PIN_PREPARATION_BUDGET_SECONDS:
            try:
                pinned = pin_sam31_output_recovery(provider=provider, instance_id=instance_id,
                    root=root, timeout_seconds=min(5., max(1., poll_deadline-now)))
            except (OSError, ValueError, subprocess.SubprocessError):
                pinned = {'status':'waiting_for_pin'}
        try:
            raw = dict(result_fetcher(output_get_url))
            write_json(root/'provider_runtime_result.object-store.json', raw)
            verified = validator(raw)
            return verified, {'status':'verified', 'route':'object_store', 'recovery_attempted':False,
                              'pin_status':pinned.get('status'), 'recovery_events':events}
        except (OSError, ValueError) as exc:
            last_error = type(exc).__name__
        sleeper(min(5., max(0., poll_deadline-float(clock()))))
    while (now := float(clock())) < deadline:
        remaining = deadline-now
        if pinned.get('status') != 'pinned' and remaining >= PIN_PREPARATION_BUDGET_SECONDS:
            try:
                pinned = pin_sam31_output_recovery(provider=provider, instance_id=instance_id,
                    root=root, timeout_seconds=min(5., max(1., remaining)))
            except (OSError, ValueError, subprocess.SubprocessError):
                pinned = {'status':'waiting_for_pin'}
        remaining = deadline-float(clock())
        if remaining <= 0:
            break
        try:
            recovered = recover_sam31_output(pinned=pinned, root=root, timeout_seconds=remaining)
            evidence = {key:value for key,value in recovered.items() if key != 'runtime_result'}
            events.append(evidence)
            if recovered.get('terminal_invalid_result') is True:
                return None, {'status':'failed', 'route':'pinned_ssh', 'recovery_attempted':True,
                              'recovery_events':events, 'blockers':['sam31_recovered_result_invalid']}
            if recovered.get('status') == 'completed':
                try:
                    verified = validator(recovered['runtime_result'])
                except (ValueError, OSError) as exc:
                    return None, {'status':'failed', 'route':'pinned_ssh', 'recovery_attempted':True,
                        'validation_error_type':type(exc).__name__, 'recovery_events':events,
                        'blockers':['sam31_recovered_result_invalid']}
                return verified, {'status':'verified', 'route':'pinned_ssh', 'recovery_attempted':True,
                    'primary_error_type':last_error, 'recovery_events':events}
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            events.append({'status':'blocked', 'error_type':type(exc).__name__})
        sleeper(min(5., max(0., deadline-float(clock()))))
    return None, {'status':'failed', 'route':'unavailable', 'recovery_attempted':True,
                  'primary_error_type':last_error, 'recovery_events':events,
                  'blockers':['sam31_output_delivery_unrecoverable']}
