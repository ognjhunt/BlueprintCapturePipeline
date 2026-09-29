import hashlib
import io
import json
from functools import partial
from pathlib import Path
from types import SimpleNamespace
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
    state = {'gets':[], 'posts':[], 'corrupt':False, 'wrong_inbox':False, 'foreign_ticket':False, 'wrong_owner':False}
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
                **{k:body[k]for k in ('run_id','operator_registration_digest','request_digest','configuration_digest',
                    'owner_user_id','team_namespace','result_delivery_digest','policy_canary_projection_digest') if k in body},
                'inbox':{'status':'verified','run_id':'different' if state['wrong_inbox'] else 'run-1',
                    **({'owner_user_id':'wrong-owner' if state['wrong_owner'] else body['owner_user_id']} if 'owner_user_id' in body else {}),
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


@pytest.mark.parametrize('wrong_owner', [False, True])
def test_normal_owner_readback_binds_original_owner_request_and_downloads(tmp_path, wrong_owner):
    args, state = fixture(tmp_path)
    args.pop('registration')
    args['owner_execution'] = {'run_id':'run-1','capture_session_id':'capture-1',
        'request_digest':'sha256:'+'4'*64,'configuration_digest':'sha256:'+'5'*64,
        'owner_user_id':'owner-1','team_namespace':'team-1'}
    args['publication'].update({key:args['owner_execution'][key] for key in ('request_digest','configuration_digest')})
    state['wrong_owner'] = wrong_owner
    if wrong_owner:
        with pytest.raises(DeliveryReadbackError, match='owner_inbox_unverified'):
            verify_website_delivery(**args)
        assert not state['gets']
    else:
        result = verify_website_delivery(**args)
        assert result['status'] == 'verified' and len(state['gets']) == 2
        assert state['posts'][0]['schema_version'] == 'task_evaluation_delivery_readback_request.v2'
        assert 'operator_registration_digest' not in state['posts'][0]
        assert result['owner_user_id'] == 'owner-1'


def test_downloads_share_limited_intake_capacity_and_resume_after_busy_response(tmp_path):
    import threading
    import time
    from urllib.error import HTTPError

    args, state = fixture(tmp_path, 5)
    original = args['opener']
    capacity = threading.Lock()
    busy_once = {'value': True}

    class LimitedResponse(Response):
        def __exit__(self, *exc):
            try:
                return super().__exit__(*exc)
            finally:
                capacity.release()

        def read(self, *args):
            time.sleep(0.01)
            return super().read(*args)

    def limited(call, **kwargs):
        if call.method == 'POST':
            return original(call, **kwargs)
        key = urlsplit(call.full_url).path.rsplit('/', 1)[-1]
        if key == f'{2:032x}' and busy_once['value']:
            busy_once['value'] = False
            raise HTTPError(call.full_url, 503, 'busy', {}, None)
        assert capacity.acquire(blocking=False), 'downloads must leave intake capacity for controller requests'
        return LimitedResponse(original(call, **kwargs).getvalue())

    args['opener'] = limited
    pending = verify_website_delivery(**args)
    assert pending['status'] == 'pending'
    assert pending['reason'] == 'website_download_http_503'
    assert pending['verified_artifact_count'] == 2
    result = verify_website_delivery(**args)
    assert result['status'] == 'verified'
    assert len(state['gets']) == len(set(state['gets'])) == 5


def test_readback_of_a_streamed_run_coalesces_archive_member_downloads(tmp_path, monkeypatch):
    """Every download of an archive member used to cost its own B2 range request. The readback
    walks archive members in archive order and a download reads the registered members after it
    in the same range, so a readback of 40 remote frames costs a few span reads."""
    from blueprint_pipeline import task_evaluation_result_artifact_store as store
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    from blueprint_pipeline.task_evaluation_result_delivery import resolve_task_evaluation_result_artifact
    from tests.provider_output_fixtures import serve_member_views, stream_evidence_tree
    from tests.test_task_evaluation_policy_canary_result_delivery import _deliver, _registry, _result

    source = tmp_path / "download" / "immutable_execution"
    source.mkdir(parents=True)
    result = _result(source)
    for index in range(40):
        frame = source / "media" / "frames" / f"{index:06d}.png"
        frame.parent.mkdir(parents=True, exist_ok=True)
        frame.write_bytes(bytes([index]) * (300 + index))
        result["artifact_inventory"].append({
            "role": "retained_lossless_frame", "relative_path": frame.relative_to(source).as_posix(),
            "media_type": "image/png", "size_bytes": frame.stat().st_size,
            "sha256": "sha256:" + hashlib.sha256(frame.read_bytes()).hexdigest()})
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    run_root = tmp_path / "streamed"
    streamed = stream_evidence_tree(source, run_root / "allocator/attempts/attempt_001")
    serve_member_views(monkeypatch, streamed.store)
    delivery = _deliver(run_root, streamed.evidence, result)
    registry = _registry(run_root)
    remote = {row["artifact_id"] for row in registry["artifacts"]
              if not (Path(row["evidence_root"]) / row["relative_path"]).is_file()}
    assert len(remote) == 42  # the 40 frames, the review video and the telemetry
    ledger = tmp_path / "ledger"
    monkeypatch.setenv(store.CACHE_ROOT_ENV, str(tmp_path / "cache"))
    from blueprint_pipeline.control_plane_disk_budget import reserve_control_plane_disk
    monkeypatch.setattr(store, "reserve_control_plane_disk", partial(
        reserve_control_plane_disk, reservation_root=ledger,
        disk_usage=lambda _: SimpleNamespace(total=100 * 1024**3, free=80 * 1024**3)))
    store.clear_archive_member_cache()

    args, state = fixture(tmp_path / "readback")
    args.update(run_root=run_root, result_delivery=delivery,
                publication={**args["publication"], "result_delivery_digest": delivery["delivery_digest"]})
    rows = {row["artifact_id"]: row for row in delivery["artifacts"]}
    downloaded = []

    def opener(call, *, timeout):
        if call.method == "POST":
            body = json.loads(call.data)
            tickets = [{"artifact_id": key, "sha256": rows[key]["digest"], "size_bytes": rows[key]["size_bytes"],
                        "download_url": "/api/task-evaluation-result-downloads/record/" + key} for key in body["artifact_ids"]]
            response = {"schema_version": "task_evaluation_delivery_readback.v1", "status": "verified",
                        **{key: body[key] for key in ("run_id", "operator_registration_digest",
                                                     "result_delivery_digest", "policy_canary_projection_digest")},
                        "inbox": {"status": "verified", "run_id": body["run_id"], "team_namespace": "team-1",
                                  "projection_digest": body["policy_canary_projection_digest"],
                                  "source": "website_owner_run_index_readback"},
                        "ephemeral_downloads": tickets}
            return Response(json.dumps(response).encode())
        key = urlsplit(call.full_url).path.rsplit("/", 1)[-1]
        downloaded.append(key)
        # The Website proxies to the intake service, which resolves exactly as here.
        path, record = resolve_task_evaluation_result_artifact(run_root=run_root, run_id=registry["run_id"],
                                                               artifact_id=key)
        body = path.read_bytes()
        if record.get("_artifact_cleanup"):
            record["_artifact_cleanup"]()
        return Response(body)

    args["opener"] = opener
    args["registration"] = {**args["registration"], "run_id": registry["run_id"]}
    args["publication"]["run_id"] = registry["run_id"]
    verified = verify_website_delivery(**args, maximum_batches=64)

    assert verified["status"] == "verified" and len(verified["artifacts"]) == len(rows)
    # Archive members were fetched first, in archive order...
    offsets = [streamed.rows[rows[key]["relative_path"]]["data_offset"] for key in downloaded[:len(remote)]]
    assert set(downloaded[:len(remote)]) == remote and offsets == sorted(offsets)
    # ...and 42 member downloads cost a handful of B2 range reads, not 42.
    assert len(streamed.data_ranges()) <= 3
    assert not list((tmp_path / "cache").glob("download-*")) and not list(ledger.glob("*.json"))
