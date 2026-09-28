"""Runtime-source layers are built for the bucket the publisher will honour, and refusals stay per scene.

Scene 840938, 2026-09-12: the catalog binding carried the legacy store bucket, the
publisher used the artifact store, the object store refused the URI mismatch after a
4 GB build, and the uncaught refusal killed the whole controls tick that activation
was waiting on.
"""
from __future__ import annotations

import json

import pytest

from blueprint_pipeline import task_evaluation_configured_controls_continuation_provisioning as provisioning
from blueprint_pipeline import task_evaluation_configured_scene_object_store as store
from tests.test_task_evaluation_configured_controls_continuation_provisioning import (
    COMMIT,
    _Publisher,
    _payload,
    _provision,
)

STORE_ERROR = store.TaskEvaluationConfiguredSceneObjectStoreError


def test_default_layer_bucket_follows_the_artifact_store(monkeypatch) -> None:
    monkeypatch.setattr(store, "_artifact_object_store_client", lambda: ("client", "b2-prod"))
    monkeypatch.setattr(store, "_object_store_client", lambda: ("client", "legacy-spaces"))
    assert provisioning.default_external_layer_bucket() == "b2-prod"
    assert provisioning._live_external_layer_bucket() == "b2-prod"

    def unavailable():
        raise STORE_ERROR("configured_scene_object_store_configuration_missing:X")

    monkeypatch.setattr(store, "_artifact_object_store_client", unavailable)
    assert provisioning._live_external_layer_bucket() is None


def test_catalog_bucket_that_disagrees_with_the_live_store_is_refused_before_building(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(provisioning, "_live_external_layer_bucket", lambda: "b2-prod")
    with pytest.raises(provisioning.ConfiguredControlsProvisioningError,
                       match="^configured_controls_provisioning_external_layer_bucket_mismatch:blueprint:b2-prod$"):
        _provision(tmp_path, external_layer_bucket="blueprint")
    assert not list((tmp_path / "controls").rglob("native_task_runtime_source_adapter_bundle.zip"))
    result, publisher = _provision(tmp_path, external_layer_bucket="b2-prod")
    assert result["status"] == "configured_controls_continuation_provisioned"
    assert publisher.layers and all(row["uri"].startswith("s3://b2-prod/") for row in publisher.layers)


def test_store_refusal_during_publication_is_this_scenes_refusal(tmp_path) -> None:
    def refusing_layers(receipt):
        raise STORE_ERROR("configured_scene_runtime_source_layer_uri_mismatch")

    with pytest.raises(provisioning.ConfiguredControlsProvisioningError,
                       match="^configured_controls_provisioning_runtime_layer_publication_failed:"
                             "configured_scene_runtime_source_layer_uri_mismatch$"):
        _provision(tmp_path, layer_publisher=refusing_layers)

    def refusing_artifact(*, path, artifact_kind):
        raise STORE_ERROR("configured_scene_artifact_existing_identity_mismatch")

    with pytest.raises(provisioning.ConfiguredControlsProvisioningError,
                       match="^configured_controls_provisioning_runtime_source_publication_failed:"):
        _provision(tmp_path / "second", artifact_publisher=refusing_artifact)


def test_retained_wrapper_built_for_another_bucket_is_superseded_and_rebuilt(tmp_path, monkeypatch) -> None:
    publisher = _Publisher()
    controls_root = tmp_path / "controls-root"
    controls_root.mkdir()
    kwargs = dict(payload_dir=_payload(tmp_path), controls_root=controls_root, source_commit=COMMIT,
                  artifact_publisher=publisher.artifact, layer_publisher=publisher.layers_of,
                  external_layer_min_bytes=1024)
    monkeypatch.setattr(provisioning, "_live_external_layer_bucket", lambda: None)  # store unknown here
    provisioning._runtime_source_reference(**kwargs, external_layer_bucket="blueprint")
    receipt_path = controls_root / "native_task_runtime_source_build_receipt.v1.json"
    first = json.loads(receipt_path.read_text())
    assert first["external_layers"][0]["uri"].startswith("s3://blueprint/")
    # Same retained wrapper, live store now known: reused as-is when the bucket agrees.
    monkeypatch.setattr(provisioning, "_live_external_layer_bucket", lambda: "blueprint")
    provisioning._runtime_source_reference(**kwargs, external_layer_bucket="blueprint")
    assert json.loads(receipt_path.read_text()) == first
    assert not list(controls_root.glob("*.superseded-*"))
    # The store moved: the stale wrapper is set aside byte for byte and rebuilt for the live bucket.
    monkeypatch.setattr(provisioning, "_live_external_layer_bucket", lambda: "b2-prod")
    provisioning._runtime_source_reference(**kwargs, external_layer_bucket="b2-prod")
    rebuilt = json.loads(receipt_path.read_text())
    assert rebuilt["external_layers"][0]["uri"].startswith("s3://b2-prod/")
    assert rebuilt["external_layers"][0]["sha256"] == first["external_layers"][0]["sha256"]
    superseded = sorted(controls_root.glob("*.superseded-*"))
    assert {p.name.split(".superseded-")[0] for p in superseded} == {
        "native_task_runtime_source_adapter_bundle.zip", "native_task_runtime_source_build_receipt.v1.json",
        "runtime-source-layers", "native-runtime-source-artifact-reference.json"}
    note = json.loads(next(controls_root.glob("runtime-source-superseded-*.json")).read_text())
    assert note["reason"] == "external_layer_bucket_superseded:blueprint:b2-prod" and note["bytes_deleted"] is False
    old_receipt = next(p for p in superseded if p.name.startswith("native_task_runtime_source_build_receipt"))
    assert json.loads(old_receipt.read_text()) == first
