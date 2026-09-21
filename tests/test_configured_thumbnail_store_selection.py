"""Website thumbnails may be published in either explicitly configured store."""
import hashlib

import pytest

from blueprint_pipeline import task_evaluation_configured_scene_object_store as store
from tests.test_task_evaluation_configured_scene_object_store import _Client


def test_thumbnail_uses_its_published_bucket_with_exact_byte_verification(monkeypatch):
    payload = b'published-task-object-thumbnail'
    digest = 'sha256:' + hashlib.sha256(payload).hexdigest()
    key = store.DEFAULT_KEY_PREFIX+'/scene/thumbnail/sha256/'+digest[7:]+'/thumbnail.png'
    legacy, artifacts = _Client(), _Client()
    artifacts.objects[('artifacts', key)] = payload
    monkeypatch.setattr(store, '_object_store_client', lambda: (legacy, 'legacy'))
    monkeypatch.setattr(store, '_artifact_object_store_client', lambda: (artifacts, 'artifacts'))
    monkeypatch.setenv(store._ARTIFACT_STORE_FILE_ENV['bucket'], '/configured/bucket')
    reference = {'uri':'s3://artifacts/'+key, 'digest':digest, 'size_bytes':len(payload)}
    assert store.read_configured_scene_object(reference=reference) == payload
    with pytest.raises(store.TaskEvaluationConfiguredSceneObjectStoreError, match='read_reference_invalid'):
        store.read_configured_scene_object(reference={**reference,'uri':'s3://unconfigured/'+key})
    artifacts.corrupt_readback = True
    with pytest.raises(store.TaskEvaluationConfiguredSceneObjectStoreError, match='readback_mismatch'):
        store.read_configured_scene_object(reference=reference)
