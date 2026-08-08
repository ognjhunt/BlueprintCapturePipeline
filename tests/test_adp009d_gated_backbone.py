from __future__ import annotations

import hashlib
import io
import json

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
