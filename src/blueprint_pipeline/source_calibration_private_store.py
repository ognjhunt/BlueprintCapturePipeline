"""Fresh native B2 privacy readback for full-source renderer staging only."""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit
import urllib.request

from .decision_evidence_contracts import canonical_digest, canonical_json
from .task_evaluation_sam31_preparation_profile import _secret_path

SCHEMA = 'source_calibration_private_store_readback.v1'
ENV = {
    'access_key_id_file':'BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_ACCESS_KEY_ID_FILE',
    'secret_access_key_file':'BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_SECRET_ACCESS_KEY_FILE',
    'endpoint_url_file':'BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_ENDPOINT_URL_FILE',
    'bucket_file':'BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_BUCKET_FILE',
    'region_file':'BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_REGION_FILE',
}
FILES = {'access_key_id_file':'backblaze_b2_key_id','secret_access_key_file':'backblaze_b2_application_key',
         'endpoint_url_file':'backblaze_b2_s3_endpoint_url','bucket_file':'backblaze_b2_bucket','region_file':'backblaze_b2_region'}
EXPECTED_BUCKET = 'blueprint-task-evaluation-artifacts-prod'


def _require(value: bool, code: str) -> None:
    if not value:
        raise ValueError('source_calibration_private_store_'+code)


def _transport(method: str, url: str, headers: Mapping[str,str], body: Mapping | None) -> dict:
    _require(url.startswith('https://'),'provider_url_not_https')
    request=urllib.request.Request(url,method=method,headers=dict(headers),
        data=json.dumps(body).encode() if body is not None else None)
    with urllib.request.urlopen(request,timeout=30) as response:  # nosec B310 - https endpoint from the private store binding, checked above
        _require(response.status==200,'provider_read_failed')
        value=json.load(response)
    _require(isinstance(value,dict),'provider_response_invalid')
    return value


def verify_private_source_store(*, output_path: str | Path,
    transport: Callable[[str,str,Mapping[str,str],Mapping | None],dict]=_transport) -> dict[str,Any]:
    paths={key:_secret_path(os.getenv(env) or Path('/etc/blueprint/provider-secrets')/FILES[key],
                           group_read_allowed=True,code='source_store_file_invalid') for key,env in ENV.items()}
    values = {}
    for key, path in paths.items():
        with path.open('rb') as stream:
            raw = stream.read(4097)
        _require(0 < len(raw) <= 4096, 'configuration_file_size_invalid')
        values[key] = raw.decode('utf-8').strip()
        _require(bool(values[key]), 'configuration_file_empty')
    endpoint=values['endpoint_url_file'].rstrip('/')
    bucket=values['bucket_file']
    region=values['region_file']
    parsed=urlsplit(endpoint)
    _require(bucket==EXPECTED_BUCKET and parsed.scheme=='https' and not parsed.username and not parsed.password
             and not parsed.query and not parsed.fragment and parsed.path in ('','/')
             and parsed.hostname==f's3.{region}.backblazeb2.com','configured_identity_invalid')
    credential=base64.b64encode((values['access_key_id_file']+':'+values['secret_access_key_file']).encode()).decode()
    auth=transport('GET','https://api.backblazeb2.com/b2api/v2/b2_authorize_account',{'Authorization':'Basic '+credential},None)
    api=(auth.get('apiInfo') or {}).get('storageApi') or auth
    api_url=str(api.get('apiUrl') or '').rstrip('/')
    native=urlsplit(api_url)
    _require(native.scheme=='https' and bool(native.hostname) and native.hostname.endswith('.backblazeb2.com')
             and not native.username and not native.password and not native.query and not native.fragment
             and native.path in ('','/') and str(api.get('s3ApiUrl') or '').rstrip('/')==endpoint,
             'native_endpoint_mismatch')
    _require(bool(auth.get('authorizationToken')) and bool(auth.get('accountId')),'authorization_response_invalid')
    body=transport('POST',api_url+'/b2api/v2/b2_list_buckets',
                   {'Authorization':auth['authorizationToken'],'Content-Type':'application/json'},
                   {'accountId':auth['accountId'],'bucketName':bucket})
    matches=[row for row in body.get('buckets',[]) if row.get('bucketName')==bucket]
    _require(len(matches)==1 and matches[0].get('bucketType')=='allPrivate'
             and bool(matches[0].get('bucketId')),'bucket_privacy_unproven')
    result={'schema_version':SCHEMA,'status':'verified_private','observed_at':datetime.now(timezone.utc).isoformat(),
            'provider':'backblaze_b2','s3_endpoint':endpoint,'region':region,'bucket':bucket,
            'bucket_id':matches[0]['bucketId'],'bucket_type':matches[0]['bucketType'],
            'native_api_endpoint':api_url,'authenticated_native_readback':True,
            'bucket_response':body,'credential_values_recorded':False,'provider_mutation_performed':False,
            'readback_digest':''}
    result['readback_digest']=canonical_digest(result,digest_field='readback_digest')
    path=Path(output_path)
    with path.open('x',encoding='utf-8') as stream:
        stream.write(canonical_json(result)+'\n')
    return {'staging_kwargs':{key:str(path) for key,path in paths.items()},'readback':result,'readback_path':str(path)}
