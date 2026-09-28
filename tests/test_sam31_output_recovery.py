"""Real SAM validation and pinned streaming transport with hermetic provider I/O."""
from __future__ import annotations

import hashlib
import json
import subprocess
from types import SimpleNamespace

import pytest

from blueprint_pipeline import sam31_output_recovery as recovery
from blueprint_pipeline import vast_provider_output_recovery as transport
from blueprint_pipeline.gpu_render_providers import VastRenderProvider
from blueprint_pipeline.vast_args_payload_transport import onstart_mode_script
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_sam31_vast_source_track_canary import (
    _Provider, _bound_request, _preflight, _runtime_result, _grant,
    run_sam31_vast_source_track_canary, _bootstrap_script, INPUT_URL, PUT_URL, GET_URL, TOKEN,
)


def _harness(tmp_path, monkeypatch, failure):
    identity = tmp_path/'identity'
    subprocess.run(['ssh-keygen','-q','-t','ed25519','-N','','-f',str(identity)], check=True, capture_output=True)
    monkeypatch.setenv(transport.VAST_SSH_IDENTITY_FILE_ENV, str(identity))
    public = ' '.join(identity.with_suffix('.pub').read_text().split()[:2])
    value = _runtime_result()
    if failure == 'invalid_result':
        value['checkpoint_digest'] = 'sha256:'+'c'*64
        value['runtime_result_digest'] = canonical_digest(value, digest_field='runtime_result_digest')
    payload = b'incomplete JSON' if failure == 'invalid_json' else json.dumps(value).encode()
    root = tmp_path/'run'
    events = []
    now = [1000.]
    original_run = subprocess.run
    def ssh(command, **kwargs):
        if command[0] == 'ssh-keygen':
            return original_run(command, **kwargs)
        if command[0] == 'ssh-keyscan':
            events.append('pin')
            return SimpleNamespace(returncode=0, stdout=f'[example.invalid]:2222 {public}\n'.encode(), stderr=b'')
        assert command[0] == 'ssh'
        assert 'StrictHostKeyChecking=yes' in command
        assert '/work/sam31_source_track_result.json' in command[-1]
        assert kwargs['timeout'] <= 12.1
        if kwargs.get('text'):
            events.append('metadata')
            if failure == 'missing_result':
                return SimpleNamespace(returncode=1, stdout='')
            size = len(payload) if failure != 'oversize' else recovery.MAX_RESULT_BYTES+1
            return SimpleNamespace(returncode=0, stdout=f'{size} {hashlib.sha256(payload).hexdigest()}\n')
        events.append('stream')
        assert kwargs.get('capture_output') is None
        kwargs['stdout'].write(payload if failure != 'corrupt_stream' else b'x'*len(payload))
        return SimpleNamespace(returncode=0, stderr=b'')
    monkeypatch.setattr(recovery.subprocess, 'run', ssh)
    class Provider(_Provider):
        def build_request(self, spec, job_dir):
            assert spec.vast_launch_mode == 'ssh_direct'
            assert spec.env[recovery.PUBLIC_KEY_ENV] == public
            request = VastRenderProvider().build_request(spec, job_dir)
            assert request['require_direct_port'] is True
            assert request['create_payload']['runtype'] == 'ssh_direct'
            assert request['create_payload']['onstart'] == onstart_mode_script(_bootstrap_script())
            assert 'args_str' not in request['create_payload']
            return request
        def launch(self, job_dir, request, **kwargs):
            assert (root/'output_recovery_readiness.json').is_file()
            assert kwargs['allow_cold_fallback'] is False
            assert request['maximum_create_attempts'] == 1
            events.append('launch')
            return super().launch(job_dir, request, **kwargs)
        def inspect(self, instance_id):
            return {'status':'observed','provider':'vast','instance_id':instance_id,
                    'api_confirmed':True,'ssh_host':'example.invalid','ssh_port':2222}
        def terminate(self, instance_id):
            events.append('terminate')
            delivery = json.loads((root/'output_delivery_receipt.json').read_text())
            if failure in ('put_failed','readback_failed'):
                assert delivery['status'] == 'verified' and delivery['route'] == 'pinned_ssh'
                assert json.loads((root/'provider_runtime_result.json').read_text()) == _runtime_result()
            else:
                assert delivery['status'] == 'failed'
                assert not (root/'provider_runtime_result.json').exists()
            return super().terminate(instance_id)
    provider = Provider()
    def fetch(_url):
        events.append('get')
        if failure == 'readback_failed':
            raise OSError('hermetic readback failure')
        raise FileNotFoundError('worker PUT unavailable')
    def sleep(seconds):
        assert seconds >= 0 and now[0]+seconds <= 1060
        now[0] += seconds
    result = run_sam31_vast_source_track_canary(bound_request=_bound_request(), preflight=_preflight(),
        job_dir=root, input_bundle_get_url=INPUT_URL, output_put_url=PUT_URL, output_get_url=GET_URL,
        hf_token=TOKEN, provider=provider, paid_resource_admission_grant=_grant(), result_fetcher=fetch,
        clock=lambda:now[0], sleeper=sleep, watchdog_validator=lambda *_:True)
    return result, events, provider, root, payload


@pytest.mark.parametrize('failure', ['put_failed','readback_failed','corrupt_stream','invalid_result','missing_result','oversize','invalid_json'])
def test_sam_recovers_or_retains_failure_before_exact_teardown(tmp_path, monkeypatch, failure):
    result, events, provider, root, payload = _harness(tmp_path, monkeypatch, failure)
    assert len(provider.requests) == 1
    assert events[0] == 'launch' and events[-1] == 'terminate'
    assert events.index('pin') < events.index('metadata') < events.index('terminate')
    assert result['duration_seconds'] <= 60
    assert result['provider_zero_verified'] is True and result['provider_mutations_performed'] == 2
    assert result['output_delivery']['recovery_attempted'] is True
    if failure in ('put_failed','readback_failed'):
        assert result['status'] == 'completed'
        assert (root/'provider_runtime_result.ssh-recovered.json').read_bytes() == payload
        event = result['output_delivery']['recovery_events'][-1]
        assert event['strict_host_key_checking'] is True and event['streamed_to_disk'] is True
    else:
        assert result['status'] == 'failed' and result['blockers']
        assert result['source_track_import_result_path'] is None
    if failure == 'oversize':
        assert 'stream' not in events


def test_recovery_preflight_rejects_missing_private_identity_before_allocation(tmp_path, monkeypatch):
    monkeypatch.setenv(transport.VAST_SSH_IDENTITY_FILE_ENV, str(tmp_path/'absent'))
    with pytest.raises(ValueError, match='identity_missing_or_insecure'):
        recovery.prepare_sam31_output_recovery(root=tmp_path, hard_ttl_seconds=1800)
    assert not (tmp_path/'output_recovery_readiness.json').exists()


def test_public_key_bootstrap_and_primary_read_have_bounded_transport(monkeypatch):
    from blueprint_pipeline import sam31_vast_source_track_canary as canary
    bootstrap = _bootstrap_script()
    key_script = bootstrap.split("<<'PYKEY'\n", 1)[1].split("\nPYKEY", 1)[0]
    compile(key_script, 'sam31-public-key-bootstrap', 'exec')
    observed = []
    def read(*args, **kwargs):
        observed.append(kwargs['timeout_seconds'])
        return SimpleNamespace(status=200, body=json.dumps(_runtime_result()).encode())
    monkeypatch.setattr(canary, 'safe_http_request', read)
    assert canary._default_result_fetcher(GET_URL, timeout_seconds=.5) == _runtime_result()
    assert observed == [.5]
