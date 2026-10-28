"""Read one server-owned, generation-bound upload using existing host credentials."""
from __future__ import annotations

import os
import re
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import quote, urlsplit


def validate_firebase_transfer(submission):
    transfer = submission['transfer']
    parsed = urlsplit(str(transfer.get('url') or ''))
    bucket = os.getenv('FIREBASE_STORAGE_BUCKET', '').strip() or 'blueprint-8c1ca.appspot.com'
    session = submission['capture_session_id']
    owner = quote(submission['customer_id'], safe='')
    extension = Path(submission['request']['original_file']['original_filename']).suffix.lower()
    expected = f'/operations/capture-uploads/{owner}/{session}/{session}{extension}'
    generation = transfer.get('generation')
    if (parsed.scheme != 'gs' or parsed.netloc != bucket or parsed.path != expected
            or parsed.query or parsed.fragment or parsed.username or parsed.password
            or not isinstance(generation, str) or re.fullmatch(r'[1-9][0-9]{0,19}', generation) is None
            or int(generation) > (1 << 64) - 1
            or transfer.get('authorization') != ''):
        raise ValueError('capture_firebase_source_binding_invalid')
    return {'provider': 'firebase', 'uri': transfer['url'], 'generation': generation}


@contextmanager
def open_firebase_transfer(*, url, generation, timeout_seconds):
    # No owner ID token or service-account credential is sent by WebApp. The
    # authenticated intake packet identifies only this immutable source object.
    from google.cloud import storage

    parsed = urlsplit(url)
    client = storage.Client()
    try:
        blob = client.bucket(parsed.netloc).blob(parsed.path[1:], generation=int(generation))
        with blob.open('rb', timeout=timeout_seconds) as stream:
            yield stream
    finally:
        client.close()
