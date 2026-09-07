"""Tiny streaming faults across the real CAS materialization boundary."""
import io

import pytest

from blueprint_pipeline import task_evaluation_configured_scene_object_store as store
from tests.test_task_evaluation_configured_scene_object_store import _ContentAddressedClient


@pytest.mark.parametrize("fault", ["before_bytes", "midstream", "after_bytes", "wrong_digest", "wrong_size"])
def test_materialization_failure_preserves_source_and_reopens_exact_reference(tmp_path, fault):
    source = tmp_path / "source.zip"
    payload = b"immutable provider output"
    source.write_bytes(payload)
    class Client(_ContentAddressedClient):
        inject = False
        reads = 0
        def get_object(self, **kwargs):
            response = super().get_object(**kwargs)
            if not self.inject or fault in {"wrong_digest", "wrong_size"}:
                return response
            self.reads += 1
            class Interrupted(io.BytesIO):
                step = 0
                def read(self, size=-1):
                    self.step += 1
                    if self.step > 1 or fault == "before_bytes":
                        raise TimeoutError("offline_get_interrupted")
                    return super().read(5 if fault == "midstream" else size)
            return {"Body": Interrupted(payload)}
    client = Client()
    reference = store.publish_configured_scene_artifact(path=source, artifact_kind="diagnostic-checkpoint",
        client=client, bucket="fixture")
    client.inject = True
    altered = dict(reference)
    if fault == "wrong_digest":
        altered["digest"] = "sha256:" + "0" * 64
    elif fault == "wrong_size":
        altered["size_bytes"] += 1
    destination = tmp_path / "delivery" / "output.zip"
    with pytest.raises((store.TaskEvaluationConfiguredSceneObjectStoreError, TimeoutError)):
        store.materialize_configured_scene_artifact(reference=altered, destination=destination,
            maximum_size_bytes=100, client=client, bucket="fixture")
    assert not destination.exists()
    assert not list(destination.parent.glob("*.partial"))
    assert source.read_bytes() == payload
    client.inject = False
    result = store.materialize_configured_scene_artifact(reference=reference, destination=destination,
        maximum_size_bytes=100, client=client, bucket="fixture")
    assert result["status"] == "completed"
    assert destination.read_bytes() == source.read_bytes() == payload
    assert client.upload_count == 1
    assert client.reads == (0 if fault in {"wrong_digest", "wrong_size"} else 1)
