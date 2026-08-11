from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest

from blueprint_pipeline import adp009d_gated_backbone as backbone


def _git_blob_id(value: bytes) -> str:
    return hashlib.sha1(  # noqa: S324 - test fixture for Git object identity
        f"blob {len(value)}\0".encode() + value
    ).hexdigest()


@pytest.fixture
def small_identity(monkeypatch):
    config = b'{"model":"fixture"}\n'
    weights = b"fixture-weights"
    expected = (
        ("config.json", len(config), "git_blob_sha1", _git_blob_id(config)),
        ("model.safetensors", len(weights), "sha256", hashlib.sha256(weights).hexdigest()),
    )
    monkeypatch.setattr(backbone, "EXPECTED_FILES", expected)
    monkeypatch.setattr(
        backbone,
        "EXPECTED_TOTAL_BYTES",
        len(config) + len(weights),
    )
    return expected, {"config.json": config, "model.safetensors": weights}


def test_worker_verifies_every_object_and_pins_offline_main(
    tmp_path, small_identity
) -> None:
    _expected, values = small_identity
    hub = tmp_path / "hub"
    snapshot = (
        hub
        / "models--nvidia--Cosmos-Reason2-2B"
        / "snapshots"
        / backbone.REVISION
    )
    snapshot.mkdir(parents=True)
    for name, value in values.items():
        (snapshot / name).write_bytes(value)

    receipt = backbone.verify_and_seal_offline_cache(
        cache_dir=hub,
        output_path=tmp_path / "identity.json",
    )

    assert receipt["status"] == "verified"
    assert receipt["file_count"] == 2
    assert all(row["object_id_verified"] for row in receipt["files"])
    assert receipt["credentials_retained_after_materialization"] is False
    assert (
        hub / "models--nvidia--Cosmos-Reason2-2B" / "refs" / "main"
    ).read_text().strip() == backbone.REVISION


def test_worker_rejects_one_changed_object(tmp_path, small_identity) -> None:
    _expected, values = small_identity
    hub = tmp_path / "hub"
    snapshot = (
        hub
        / "models--nvidia--Cosmos-Reason2-2B"
        / "snapshots"
        / backbone.REVISION
    )
    snapshot.mkdir(parents=True)
    for name, value in values.items():
        (snapshot / name).write_bytes(value)
    (snapshot / "model.safetensors").write_bytes(b"changed-weights")

    with pytest.raises(RuntimeError, match="cache_object_mismatch"):
        backbone.verify_and_seal_offline_cache(
            cache_dir=hub,
            output_path=tmp_path / "blocked.json",
        )
    blocked = json.loads((tmp_path / "blocked.json").read_text())
    assert blocked["status"] == "blocked"
    assert blocked["offline_main_ref_written"] is False


def test_runtime_binding_preserves_checkpoint_and_offline_selector_token(
    tmp_path, small_identity
) -> None:
    _expected, values = small_identity
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    for name, value in values.items():
        (snapshot / name).write_bytes(value)

    checkpoint = tmp_path / "sealed-checkpoint"
    checkpoint.mkdir()
    config = json.dumps({"model_name": backbone.MODEL_ID, "model_type": "fixture"})
    (checkpoint / "config.json").write_text(config)
    processor_config = json.dumps({"processor_kwargs": {"fixture": True}})
    (checkpoint / "processor_config.json").write_text(processor_config)
    (checkpoint / "weights.safetensors").write_bytes(b"sealed-policy")
    before = (checkpoint / "config.json").read_bytes()
    processor_before = (checkpoint / "processor_config.json").read_bytes()

    runtime = tmp_path / "runtime-checkpoint"
    aliases = tmp_path / "aliases"
    receipt = backbone.prepare_offline_runtime_binding(
        checkpoint_root=checkpoint,
        backbone_snapshot=snapshot,
        runtime_checkpoint_root=runtime,
        alias_root=aliases,
        output_path=tmp_path / "binding.json",
    )

    runtime_config = json.loads((runtime / "config.json").read_text())
    runtime_processor_config = json.loads(
        (runtime / "processor_config.json").read_text()
    )
    model_name = runtime_config["model_name"]
    assert "nvidia/Cosmos-Reason2-2B" in model_name
    assert Path(model_name).resolve(strict=True) == aliases.resolve()
    assert json.loads((Path(model_name) / "config.json").read_text()) == json.loads(
        values["config.json"]
    )
    assert (checkpoint / "config.json").read_bytes() == before
    assert (checkpoint / "processor_config.json").read_bytes() == processor_before
    assert runtime_processor_config["processor_kwargs"]["model_name"] == model_name
    assert (runtime / "weights.safetensors").is_symlink()
    assert (runtime / "weights.safetensors").resolve() == (
        checkpoint / "weights.safetensors"
    ).resolve()
    assert receipt["status"] == "verified"
    assert receipt["sealed_checkpoint_modified"] is False
    assert receipt["local_backbone_alias_resolves_to_verified_snapshot"] is True


def test_runtime_binding_rejects_changed_publisher_model_name(
    tmp_path, small_identity
) -> None:
    _expected, values = small_identity
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    for name, value in values.items():
        (snapshot / name).write_bytes(value)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text('{"model_name":"changed/model"}')
    (checkpoint / "processor_config.json").write_text('{"processor_kwargs":{}}')

    with pytest.raises(RuntimeError, match="backbone_identity_mismatch"):
        backbone.prepare_offline_runtime_binding(
            checkpoint_root=checkpoint,
            backbone_snapshot=snapshot,
            runtime_checkpoint_root=tmp_path / "runtime",
            alias_root=tmp_path / "aliases",
            output_path=tmp_path / "blocked-binding.json",
        )
    blocked = json.loads((tmp_path / "blocked-binding.json").read_text())
    assert blocked["status"] == "blocked"
    assert blocked["sealed_checkpoint_modified"] is False


def test_local_access_probe_is_revision_and_listing_bound_without_secret_output(
    small_identity,
) -> None:
    expected, _values = small_identity
    siblings = []
    for path, size, kind, object_id in expected:
        siblings.append(
            {
                "rfilename": path,
                "size": size,
                "blobId": object_id if kind == "git_blob_sha1" else "lfs-pointer",
                "lfs": {"sha256": object_id} if kind == "sha256" else None,
            }
        )
    payload = {
        "id": backbone.MODEL_ID,
        "sha": backbone.REVISION,
        "gated": "auto",
        "siblings": siblings,
    }

    def opener(request, *, timeout):
        assert timeout == 40
        assert request.headers["Authorization"] == "Bearer fixture-secret"
        return io.BytesIO(json.dumps(payload).encode())

    receipt = backbone.probe_gated_backbone_access(
        token="fixture-secret",
        opener=opener,
    )

    assert receipt["status"] == "authorized"
    assert receipt["blockers"] == []
    assert "fixture-secret" not in json.dumps(receipt)
    assert receipt["raw_secret_recorded"] is False
    assert receipt["secret_hash_recorded"] is False


def test_local_access_probe_abstains_without_token(monkeypatch) -> None:
    for name in (
        "HF_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)
    receipt = backbone.probe_gated_backbone_access(token="")
    assert receipt["status"] == "blocked"
    assert receipt["blockers"] == ["adp009d_gated_backbone_token_missing"]
