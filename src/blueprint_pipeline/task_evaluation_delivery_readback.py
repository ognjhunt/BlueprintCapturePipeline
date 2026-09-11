"""Verify Website inbox membership and every published download without saving tickets."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import uuid
from urllib import error, request
from urllib.parse import urljoin, urlsplit

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_launch_webapp_sync import load_pipeline_sync_token
from .webapp_sync import _pipeline_sync_headers, validated_https_sync_url

ENDPOINT_ENV = 'PIPELINE_TASK_EVALUATION_DELIVERY_READBACK_URL'
MAX_ARTIFACTS = 200_000
BATCH_SIZE = 12


class DeliveryReadbackError(ValueError):
    pass


class _NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _atomic(path: Path, value: dict):
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('x') as stream:
        json.dump(value, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def delivery_readback_matches(value, *, identity, artifacts, team_namespace, owner_user_id=None):
    """Validate actual owner-index membership and every expected byte receipt."""
    if (value.get('status') != 'verified' or any(value.get(k) != v for k, v in identity.items())
            or value.get('readback_digest') != canonical_digest(value, digest_field='readback_digest')):
        return False
    rows = value.get('artifacts')
    wanted = {(row['artifact_id'], row['digest'], row['size_bytes']) for row in artifacts}
    if (not isinstance(rows, list) or len(rows) != len(wanted)
            or any(not isinstance(row, dict) or row.get('verified') is not True
                   or row.get('http_status') != 200 for row in rows)
            or {(row.get('artifact_id'), row.get('sha256'), row.get('size_bytes')) for row in rows} != wanted):
        return False
    inbox = value.get('inbox') or {}
    return (inbox.get('status') == 'verified' and inbox.get('run_id') == identity['run_id']
            and inbox.get('projection_digest') == identity['policy_canary_projection_digest']
            and inbox.get('team_namespace') == team_namespace
            and (owner_user_id is None or inbox.get('owner_user_id') == owner_user_id)
            and inbox.get('source') == 'website_owner_run_index_readback')


def verify_website_delivery(*, run_root, registration=None, owner_execution=None, result_delivery, policy_canary_result,
                            publication, endpoint_url=None, token=None, opener=None,
                            maximum_batches=8, timeout_seconds=90):
    """Resume verified downloads in bounded batches; tickets live only in memory."""
    publication_endpoint = os.environ.get('PIPELINE_TASK_EVALUATION_RUN_WEBAPP_URL', '').rstrip('/')
    endpoint = validated_https_sync_url(endpoint_url or os.environ.get(ENDPOINT_ENV)
        or (publication_endpoint + '/readback' if publication_endpoint else ''))
    secret = load_pipeline_sync_token(token=token)
    http = opener or request.build_opener(_NoRedirect()).open
    if (registration is None) == (owner_execution is None):
        raise DeliveryReadbackError('delivery_readback_authority_binding_invalid')
    authority = registration if registration is not None else owner_execution
    identity = ({'operator_registration_digest': registration['registration_digest']}
        if registration is not None else {key: owner_execution[key] for key in (
            'request_digest', 'configuration_digest', 'owner_user_id', 'team_namespace')})
    expected = {'run_id': authority['run_id'], **identity,
        'result_delivery_digest': result_delivery['delivery_digest'],
        'policy_canary_projection_digest': policy_canary_result['projection_digest']}
    if (publication.get('status') != 'succeeded'
            or any(publication.get(key) != value for key, value in expected.items()
                   if key not in {'operator_registration_digest', 'owner_user_id', 'team_namespace'})):
        raise DeliveryReadbackError('delivery_readback_publication_binding_invalid')
    rows = result_delivery['artifacts']
    wanted = {row['artifact_id']: row for row in rows}
    if not 1 <= len(wanted) == len(rows) <= MAX_ARTIFACTS:
        raise DeliveryReadbackError('delivery_readback_artifact_inventory_invalid')
    context_digest = canonical_digest(expected)
    root = Path(run_root) / 'operator_terminal_delivery' / 'verified_downloads' / context_digest.removeprefix('sha256:')
    if root.is_symlink():
        raise DeliveryReadbackError('delivery_readback_journal_unsafe')
    root.mkdir(parents=True, exist_ok=True)
    finished = {}
    for artifact_id in wanted:
        if not artifact_id.isalnum() or len(artifact_id) > 128:
            raise DeliveryReadbackError('delivery_readback_artifact_id_invalid')
        path = root / (artifact_id + '.json')
        if path.is_file() and not path.is_symlink():
            value = json.loads(path.read_text())
            if (value.get('context_digest') == context_digest
                    and value.get('verification_digest') == canonical_digest(value, digest_field='verification_digest')
                    and value.get('sha256') == wanted[artifact_id]['digest']
                    and value.get('size_bytes') == wanted[artifact_id]['size_bytes']
                    and value.get('verified') is True and value.get('http_status') == 200):
                finished[artifact_id] = value
    inbox = None
    remaining = [key for key in wanted if key not in finished]
    batches = [remaining[index:index+BATCH_SIZE] for index in range(0, len(remaining), BATCH_SIZE)]
    if not batches:
        batches = [list(wanted)[:1]]  # Reopen owner-index membership on every completion attempt.
    for ids in batches[:max(1, min(int(maximum_batches), 64))]:
        body = {'schema_version': ('task_evaluation_delivery_readback_request.v1' if registration is not None
                                   else 'task_evaluation_delivery_readback_request.v2'),
                'capture_session_id': authority['capture_session_id'], **expected, 'artifact_ids': ids}
        encoded = json.dumps(body, separators=(',', ':')).encode()
        call = request.Request(endpoint, data=encoded, headers=_pipeline_sync_headers(secret, encoded), method='POST')
        try:
            with http(call, timeout=timeout_seconds) as response:
                if response.status != 200:
                    raise DeliveryReadbackError('delivery_readback_http_status_invalid')
                raw = response.read(131073)
                if len(raw) > 131072:
                    raise DeliveryReadbackError('delivery_readback_response_too_large')
                result = json.loads(raw)
        except error.HTTPError as exc:
            return {**expected, 'status': 'pending', 'reason': 'website_readback_http_' + str(exc.code),
                    'verified_artifact_count': len(finished), 'required_artifact_count': len(wanted)}
        if (result.get('schema_version') != 'task_evaluation_delivery_readback.v1'
                or result.get('status') != 'verified'
                or any(result.get(key) != value for key, value in expected.items())):
            raise DeliveryReadbackError('delivery_readback_response_binding_invalid')
        inbox = result.get('inbox') or {}
        if (inbox.get('status') != 'verified' or inbox.get('run_id') != expected['run_id']
                or inbox.get('projection_digest') != expected['policy_canary_projection_digest']
                or inbox.get('team_namespace') != authority['team_namespace']
                or (owner_execution is not None and inbox.get('owner_user_id') != authority['owner_user_id'])
                or inbox.get('source') != 'website_owner_run_index_readback'):
            raise DeliveryReadbackError('delivery_readback_owner_inbox_unverified')
        tickets = result.get('ephemeral_downloads') or []
        if len(tickets) != len(ids) or {row.get('artifact_id') for row in tickets} != set(ids):
            raise DeliveryReadbackError('delivery_readback_ticket_inventory_mismatch')
        def download(ticket):
            artifact_id = ticket['artifact_id']
            item = wanted[artifact_id]
            if ticket.get('sha256') != item['digest'] or ticket.get('size_bytes') != item['size_bytes']:
                raise DeliveryReadbackError('delivery_readback_ticket_digest_mismatch')
            target = validated_https_sync_url(urljoin(endpoint, ticket['download_url']))
            if (urlsplit(target).netloc != urlsplit(endpoint).netloc
                    or not urlsplit(target).path.startswith('/api/task-evaluation-result-downloads/')):
                raise DeliveryReadbackError('delivery_readback_ticket_origin_invalid')
            if artifact_id in finished:
                return finished[artifact_id]
            sha, total = hashlib.sha256(), 0
            with http(request.Request(target, method='GET'), timeout=timeout_seconds) as response:
                if response.status != 200:
                    raise DeliveryReadbackError('delivery_readback_download_http_status_invalid')
                while chunk := response.read(1024**2):
                    total += len(chunk)
                    if total > item['size_bytes']:
                        raise DeliveryReadbackError('delivery_readback_download_too_large')
                    sha.update(chunk)
            digest = 'sha256:' + sha.hexdigest()
            if total != item['size_bytes'] or digest != item['digest']:
                raise DeliveryReadbackError('delivery_readback_download_digest_mismatch')
            value = {'artifact_id': artifact_id, 'sha256': digest, 'size_bytes': total,
                     'verified': True, 'http_status': 200, 'context_digest': context_digest}
            value['verification_digest'] = canonical_digest(value, digest_field='verification_digest')
            _atomic(root / (artifact_id + '.json'), value)
            return value
        with ThreadPoolExecutor(max_workers=4) as pool:
            for value in pool.map(download, tickets):
                finished[value['artifact_id']] = value
    if len(finished) != len(wanted):
        return {**expected, 'status': 'pending', 'reason': 'download_batches_remaining',
                'verified_artifact_count': len(finished), 'required_artifact_count': len(wanted)}
    receipt = {**expected, 'status': 'verified', 'inbox': inbox,
               'artifacts': [finished[key] for key in sorted(finished)],
               'ticket_urls_recorded': False, 'every_artifact_downloaded_and_hashed': True}
    receipt['readback_digest'] = canonical_digest(receipt, digest_field='readback_digest')
    return receipt
