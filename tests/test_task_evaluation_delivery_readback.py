import hashlib
import io
import json
from urllib.parse import urlsplit

import pytest

from blueprint_pipeline.task_evaluation_delivery_readback import DeliveryReadbackError, verify_website_delivery


class Response(io.BytesIO):
    status = 200


def fixture(root, count=2):
    bodies = {f'{index:032x}': f'lossless artifact {index}'.encode() for index in range(count)}
    rows = [{'artifact_id': key, 'digest': 'sha256:'+hashlib.sha256(body).hexdigest(), 'size_bytes': len(body)}
            for key, body in bodies.items()]
    registration = {'run_id':'run-1', 'capture_session_id':'capture-1', 'team_namespace':'team-1',
                    'registration_digest':'sha256:'+'1'*64}
    delivery = {'delivery_digest':'sha256:'+'2'*64,'artifacts':rows}
    projection = {'projection_digest':'sha256:'+'3'*64}
    publication = {'status':'succeeded','run_id':'run-1','result_delivery_digest':delivery['delivery_digest'],
                   'policy_canary_projection_digest':projection['projection_digest']}
    state = {'gets':[], 'posts':[], 'corrupt':False, 'wrong_inbox':False, 'foreign_ticket':False}
    def opener(call, *, timeout):
        if call.method == 'POST':
            body=json.loads(call.data)
            state['posts'].append(body)
            assert call.get_header('X-blueprint-pipeline-signature').startswith('sha256=')
            assert len(body['artifact_ids']) <= 12
            tickets = [{'artifact_id':key,'sha256':next(r['digest']for r in rows if r['artifact_id']==key),
                        'size_bytes':len(bodies[key]),
                        'download_url':('https://foreign.example' if state['foreign_ticket'] else '')+
                          '/api/task-evaluation-result-downloads/record/'+key+'?signature=PRIVATE-TICKET'}
                       for key in body['artifact_ids']]
            response={'schema_version':'task_evaluation_delivery_readback.v1','status':'verified',
                **{k:body[k]for k in ('run_id','operator_registration_digest','result_delivery_digest','policy_canary_projection_digest')},
                'inbox':{'status':'verified','run_id':'different' if state['wrong_inbox'] else 'run-1',
                    'projection_digest':projection['projection_digest'],'team_namespace':'team-1',
                    'source':'website_owner_run_index_readback'},'ephemeral_downloads':tickets}
            return Response(json.dumps(response).encode())
        key=urlsplit(call.full_url).path.rsplit('/',1)[-1]
        state['gets'].append(key)
        return Response(b'wrong bytes' if state['corrupt'] else bodies[key])
    return {'run_root':root,'registration':registration,'result_delivery':delivery,'policy_canary_result':projection,
            'publication':publication,'endpoint_url':'https://website.example/api/internal/pipeline/capture-task-evaluation-runs/readback',
            'token':'fixture-token','opener':opener},state


def test_all_downloads_resume_without_repeating_verified_files_or_persisting_tickets(tmp_path):
    args,state=fixture(tmp_path,25)
    first=verify_website_delivery(**args,maximum_batches=1)
    assert first['status']=='pending' and first['verified_artifact_count']==12
    result=verify_website_delivery(**args)
    assert result['status']=='verified' and len(result['artifacts'])==25
    assert len(state['gets'])==25 and len(set(state['gets']))==25
    again=verify_website_delivery(**args)
    assert again==result and len(state['gets'])==25
    assert 'PRIVATE-TICKET' not in json.dumps(result)
    assert all('PRIVATE-TICKET' not in p.read_text() for p in tmp_path.rglob('*.json'))


@pytest.mark.parametrize('mode,reason',[('corrupt','download_digest_mismatch'),('wrong_inbox','owner_inbox_unverified'),
                                       ('foreign_ticket','ticket_origin_invalid')])
def test_bad_bytes_or_wrong_owner_or_redirect_target_prevents_completion(tmp_path,mode,reason):
    args,state=fixture(tmp_path)
    state[mode]=True
    with pytest.raises(DeliveryReadbackError,match=reason):
        verify_website_delivery(**args)
    assert not list(tmp_path.rglob('*.json'))
