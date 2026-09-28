"""Installed fixed-consent door and timer bridge for the same scene engine.

ADP-009D day-28: verified scene storage retirement. Public callers select one
protected consent by opaque id/raw identity. They cannot supply local paths,
policy, owner scopes or remote locations. No switch is enabled by this module.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import stat
import time
from pathlib import Path

from . import task_evaluation_scene_retirement as engine
from . import task_evaluation_scene_retirement_access as access
from .task_evaluation_scene_retirement_authority import load_authority, load_document
from .object_store_multipart_stream import MultipartStream

CONSENT_ROOT = Path('/var/lib/blueprint/scene-retirement/consents')
GC_SELECTION = Path('/var/lib/blueprint/scene-retirement/gc-selection.json')
ENVIRONMENT_FILE = Path('/etc/blueprint/pipeline-control-plane.env')
_PREFIX = 'blueprint/arm-decision-proof-v1/scene-retirement/'
_CHUNK = 1024*1024
_ARCHIVE_MAX = 48*1024**3
_SETTING = 'BLUEPRINT_CONTROL_PLANE_SCENE_RETIREMENT'
_FILES = dict(access_key='ACCESS_KEY_ID', secret_key='SECRET_ACCESS_KEY', bucket='BUCKET',
              endpoint='ENDPOINT_URL', region='REGION')
_ENV_FILES = {key:'BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_'+name+'_FILE' for key,name in _FILES.items()}
_DEFAULT_FILES = dict(access_key='backblaze_b2_key_id', secret_key='backblaze_b2_application_key',
    bucket='backblaze_b2_bucket', endpoint='backblaze_b2_s3_endpoint_url', region='backblaze_b2_region')


def _require(value, code='scene_retirement_operator_selection_invalid'):
    access._require(value, code)


def _options(action, intent_id, consent_id, expected_sha256, expected_size_bytes, apply):
    _require(action in {'retire','restore'} and type(apply) is bool)
    _require(type(intent_id) is str and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', intent_id))
    _require(type(consent_id) is str and re.fullmatch('[0-9a-f]{32}', consent_id))
    _require(type(expected_sha256) is str and re.fullmatch('sha256:[0-9a-f]{64}', expected_sha256))
    _require(type(expected_size_bytes) is int and 0 < expected_size_bytes <= 512*1024)
    _require(action != 'restore' or apply)


def _kept(code):
    return dict(status='kept', reason=code, mutations=0)


def run_selected_action(action, *, intent_id, consent_id, expected_sha256, expected_size_bytes,
                        apply=False, now=time.time, transport_factory=None):
    """Select only original root-protected bytes, then call the actual engine."""
    try:
        _options(action,intent_id,consent_id,expected_sha256,expected_size_bytes,apply)
        path = CONSENT_ROOT/(consent_id+'.json')
        with access._opened(CONSENT_ROOT,directory=True,protected=True) as (_, info):
            _require(stat.S_IMODE(info.st_mode)==0o700, 'scene_retirement_authority_permissions')
        authority = load_authority(path,action=action,now=now)
        _require(authority['consent_raw_ref']==dict(path=str(path),sha256=expected_sha256,
            size_bytes=expected_size_bytes), 'scene_retirement_raw_reference_changed')
        consent = authority['consent']
        _require(consent['consent_id']==consent_id and consent['intent_id']==intent_id)
        selected = consent['plan_raw_ref' if action=='retire' else 'retired_journal_raw_ref']
        if not apply:
            # A selected KEEP plan is an observation, never current action clearance.
            from .task_evaluation_scene_retirement_authority import selected_document
            plan = selected_document(selected,maximum=16*1024*1024)
            _require(plan.get('schema_version')=='task_evaluation_scene_lifecycle_plan.v1'
                     and plan.get('intent_id')==intent_id)
            return dict(status='planned', intent_id=intent_id, cleanup_authorized=False,
                        plan_sha256=selected['sha256'], mutations=0)
        transport = (transport_factory or installed_transport)()
        try:
            function = engine.retire_scene if action=='retire' else engine.restore_scene
            return function(selected['path'],path,transport=transport,now=now)
        finally:
            transport.close()
    except access.SceneRetirementAccessError as error:
        code=str(error)
        return _kept(code if re.fullmatch('scene_retirement_[a-z_]{1,100}',code)
                     else 'scene_retirement_operator_refused')
    except (OSError,ValueError,KeyError,TypeError,AttributeError,RuntimeError):
        return _kept('scene_retirement_operator_refused')


class SceneArchiveTransport:
    """Native multipart sink, fixed private namespace, single action allowance."""
    def __init__(self, *, client, bucket):
        config = client.meta.config
        _require(config.connect_timeout==5 and config.read_timeout==30
                 and config.retries.get('total_max_attempts')==1 and config.max_pool_connections==4,
                 'scene_retirement_transport_timeouts_unproven')
        _require(type(bucket) is str and re.fullmatch('[a-z0-9][a-z0-9.-]{1,221}',bucket))
        self.client, self.bucket = client, bucket
        self.allowance = None
        self.upload = None
        self.closed = False

    def bind_allowance(self, allowance):
        from .task_evaluation_scene_retirement_preservation import ActionAllowance
        _require(self.allowance is None and isinstance(allowance,ActionAllowance),
                 'scene_retirement_transport_origin_unproven')
        self.allowance = allowance
        allowance.tick()

    def _tick(self):
        _require(not self.closed and self.allowance is not None,'scene_retirement_transport_origin_unproven')
        self.allowance.tick()

    def _call(self, name, **kwargs):
        self._tick()
        try:
            response = getattr(self.client,name)(**kwargs)
        except Exception:
            raise access.SceneRetirementAccessError('scene_retirement_transport_failure') from None
        _require(name=='complete_multipart_upload' or type(response) is dict,
                 'scene_retirement_transport_response_unproven')
        # Retain ONLY the id this invocation created, even if the post-call clock
        # fails. Its bounded abort is cleanup and never action authorization.
        if name=='create_multipart_upload':
            upload = response.get('UploadId')
            _require(type(upload) is str and 0 < len(upload) <= 1024,
                     'scene_retirement_transport_response_unproven')
            self.upload = dict(Bucket=kwargs['Bucket'],Key=kwargs['Key'],UploadId=upload)
        try:
            self._tick()
        except BaseException:
            if name=='get_object' and callable(getattr(response.get('Body'),'close',None)):
                response['Body'].close()
            raise
        return response

    def create_multipart_upload(self, **kwargs):
        return self._call('create_multipart_upload',**kwargs)
    def upload_part(self, **kwargs):
        return self._call('upload_part',**kwargs)
    def complete_multipart_upload(self, **kwargs):
        return self._call('complete_multipart_upload',**kwargs)

    def put_archive(self, name, chunks):
        _require(type(name) is str and re.fullmatch(r'[0-9a-f]{32}\.tar',name))
        self._tick()
        _require(self.upload is None,'scene_retirement_transport_upload_busy')
        key = _PREFIX+name
        sink = None
        try:
            sink = MultipartStream(client=self,bucket=self.bucket,key=key,
                metadata={'ContentType':'application/x-tar'}, expected_digest=None,
                expected_size=min(_ARCHIVE_MAX,self.allowance.limits['archive_bytes']))
            for chunk in chunks:
                self._tick()
                _require(type(chunk) is bytes and 0 < len(chunk) <= _CHUNK,
                         'scene_retirement_transport_chunk_unproven')
                # Engine charges the same stream allowance before yielding. Do
                # not refund or charge it a second time in the SDK adapter.
                sink.write(chunk)
                self._tick()
            self._tick()
            digest, size = 'sha256:'+sink.digest.hexdigest(), sink.size
            _require(size > 0,'scene_retirement_transport_empty_archive')
            # Native sink initially admits only the hard/lower action ceiling.
            # Exact one-shot observed identity is sealed after full exhaustion.
            sink.expected_digest, sink.expected_size = digest, size
            sink.finish()
            self.upload = None
            return dict(uri='s3://'+self.bucket+'/'+key,sha256=digest,size_bytes=size)
        except BaseException:
            if self.upload is not None:
                try:
                    self.client.abort_multipart_upload(**self.upload)
                except Exception:
                    pass  # Preserve original failure; no evidence may be removed.
                self.upload = None
            raise

    def read_archive(self, uri):
        return self._read_archive(uri,charge=False)

    def read_archive_charged(self, uri, allowance):
        _require(allowance is self.allowance,'scene_retirement_transport_origin_unproven')
        return self._read_archive(uri,charge=True)

    def _read_archive(self, uri, *, charge):
        self._tick()
        prefix = 's3://'+self.bucket+'/'+_PREFIX
        _require(type(uri) is str and uri.startswith(prefix)
                 and re.fullmatch(r'[0-9a-f]{32}\.tar',uri[len(prefix):]),
                 'scene_retirement_transport_archive_scope_invalid')
        response = self._call('get_object',Bucket=self.bucket,Key=uri[len('s3://'+self.bucket+'/'):])
        body = response.get('Body')
        try:
            size = response.get('ContentLength')
            _require(type(size) is int and 0 < size <= _ARCHIVE_MAX
                     and size <= self.allowance.limits['remote_bytes']-self.allowance.counts['remote_bytes'],
                     'scene_retirement_byte_limit')
            received = 0
            while received < size:
                self._tick()
                count = min(_CHUNK,size-received)
                _require(count <= self.allowance.limits['remote_bytes']-self.allowance.counts['remote_bytes'],
                         'scene_retirement_byte_limit')
                if charge:
                    # Charge the requested physical read BEFORE the socket can
                    # allocate. Short/faulted reads never refund this origin.
                    self.allowance.charge('remote_bytes',count)
                try:
                    chunk = body.read(count)
                except Exception:
                    raise access.SceneRetirementAccessError('scene_retirement_transport_failure') from None
                self._tick()
                _require(type(chunk) is bytes and 0 < len(chunk) <= count,
                         'scene_retirement_transport_readback_unproven')
                received += len(chunk)
                yield chunk
            self._tick()
            _require(body.read(1)==b'', 'scene_retirement_transport_readback_unproven')
            self._tick()
        finally:
            if callable(getattr(body,'close',None)):
                body.close()

    def close(self):
        if not self.closed:
            self.closed = True
            self.client.close()


def _scalar(path):
    path=access._canonical(str(path))
    with access._opened(path,protected=True) as (fd,before):
        _require(stat.S_IMODE(before.st_mode) in {0o600,0o640} and 0 < before.st_size <= 4096)
        raw=os.read(fd,4097)
        after=os.fstat(fd)
        _require(len(raw)==before.st_size and (access._identity(after),after.st_size,after.st_mtime_ns,after.st_ctime_ns)
                 == (access._identity(before),before.st_size,before.st_mtime_ns,before.st_ctime_ns))
    value=raw.decode().strip()
    _require(value and '\n' not in value and '\r' not in value)
    return value


def installed_transport():
    """Use existing B2 artifact bindings, with explicit action-only socket caps."""
    import boto3
    from botocore.config import Config
    values={key:_scalar(os.environ.get(_ENV_FILES[key]) or
        '/etc/blueprint/provider-secrets/'+_DEFAULT_FILES[key]) for key in _FILES}
    expected=os.environ.get('BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_EXPECTED_BUCKET',
                            'blueprint-task-evaluation-artifacts-prod')
    _require(values['bucket']==expected,'scene_retirement_transport_archive_scope_invalid')
    from urllib.parse import urlsplit
    endpoint=urlsplit(values['endpoint'])
    _require(endpoint.scheme=='https' and endpoint.hostname and not endpoint.username and not endpoint.password
             and not endpoint.query and not endpoint.fragment)
    client=boto3.client('s3',aws_access_key_id=values['access_key'],aws_secret_access_key=values['secret_key'],
        endpoint_url=values['endpoint'],region_name=values['region'],config=Config(signature_version='s3v4',
        connect_timeout=5,read_timeout=30,retries={'total_max_attempts':1},max_pool_connections=4))
    try:
        return SceneArchiveTransport(client=client,bucket=values['bucket'])
    except BaseException:
        client.close()
        raise


def load_installed_environment():
    """Read only this action's literal settings from the protected host file."""
    allowed=set(_ENV_FILES.values()) | {'BLUEPRINT_SCENE_RETIREMENT_POLICY_FILE',
        'BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_EXPECTED_BUCKET'}
    with access._opened(ENVIRONMENT_FILE,protected=True) as (_,info):
        _require(stat.S_IMODE(info.st_mode) in {0o600,0o640}, 'scene_retirement_authority_permissions')
    raw=access._bytes(str(ENVIRONMENT_FILE),protected=True)
    selected={}
    for line in raw.decode().splitlines():
        key,separator,value=line.partition('=')
        if not separator or key not in allowed:
            continue
        _require(key not in selected,'scene_retirement_environment_duplicate_setting')
        if len(value)>=2 and value[0] in "\"'" and value[-1]==value[0]:
            value=value[1:-1]
        _require(value and '\x00' not in value and '\r' not in value)
        if key!='BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_EXPECTED_BUCKET':
            access._canonical(value)
        selected[key]=value
    # Validate the entire selection before any process environment publication.
    for key,value in selected.items():
        os.environ[key]=value


def run_gc_phase(*, apply, now=time.time):
    """One exact protected selection per tick; no store sweep or consent minting."""
    setting=os.environ.get(_SETTING,'0')
    if setting!='1' or not apply:
        return dict(status='disabled' if setting in {'0','1',''} else 'setting_invalid',
                    removed_allocated_bytes=0,mutations=0)
    try:
        selection,_=load_document(GC_SELECTION,maximum=65536,protected=True)
        _require(set(selection)=={'schema_version','retirement'}
                 and selection['schema_version']=='scene_retirement_gc_selection.v1')
        chosen=selection['retirement']
        _require(type(chosen) is dict and set(chosen)=={'intent_id','consent_id','expected_sha256','expected_size_bytes'})
        result=run_selected_action('retire',**chosen,apply=True,now=now)
        return _summary(result)
    except (OSError,ValueError,KeyError,TypeError):
        return dict(status='kept',reason='scene_retirement_gc_selection_unproven',mutations=0,
                    removed_allocated_bytes=0)


def _summary(result):
    # Only proven counters/typed status leave the private action context. Never
    # include private consent, local member paths, credentials or SDK errors.
    keys=('status','reason','mutations','intent_id','logical_bytes','removed_allocated_bytes',
          'planned_unique_allocated_bytes','cleanup_authorized','plan_sha256')
    return {key:result[key] for key in keys if key in result}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('retire','restore'))
    parser.add_argument('--intent-id',required=True)
    parser.add_argument('--consent-id',required=True)
    parser.add_argument('--expected-sha256',required=True)
    parser.add_argument('--expected-size-bytes',required=True,type=int)
    parser.add_argument('--apply',action='store_true')
    args=vars(parser.parse_args(argv))
    action=args.pop('action')
    try:
        load_installed_environment()
        result=run_selected_action(action,**args)
    except (OSError,ValueError,UnicodeError):
        result=_kept('scene_retirement_installed_environment_unproven')
    print(json.dumps(_summary(result),sort_keys=True))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
