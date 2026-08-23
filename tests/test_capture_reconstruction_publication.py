from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from blueprint_pipeline.capture_reconstruction_publication import (
    CaptureReconstructionPublicationError,
    publish_postshot_output_bundle,
)


KINDS = {
    "standard_3dgs_ply": "scene.ply",
    "compressed_3dgs_spz_v4": "scene.spz",
    "postshot_project": "scene.psht",
    "postshot_coordinate_binding": "coordinate.json",
    "colmap_camera_intrinsics": "cameras.txt",
    "colmap_camera_extrinsics_and_order": "images.txt",
    "colmap_seed_points": "points3D.txt",
    "training_log": "training.log",
}


class _Blob:
    def __init__(self, store, name):
        self.store, self.name = store, name

    def upload_from_filename(self, path, if_generation_match):
        assert if_generation_match == 0
        if self.name in self.store:
            raise type("PreconditionFailed", (Exception,), {})()
        self.store[self.name] = Path(path).read_bytes()

    def download_to_filename(self, path):
        Path(path).write_bytes(self.store[self.name])


class _Bucket:
    def __init__(self, store): self.store = store
    def blob(self, name): return _Blob(self.store, name)


class _Storage:
    def __init__(self): self.store = {}
    def bucket(self, name):
        assert name == "publish-bucket"
        return _Bucket(self.store)


def _bundle(path: Path, *, tamper: bool = False) -> None:
    rows = []
    payloads = {}
    for kind, name in KINDS.items():
        body = f"{kind}\n".encode()
        payloads[name] = body
        rows.append({"kind": kind, "relative_path": name, "digest": "sha256:" + hashlib.sha256(body).hexdigest()})
    if tamper:
        payloads["scene.spz"] = b"mutated"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("canonical_3dgs_vast_output_manifest.json", "{}")
        archive.writestr("results/worker_receipt.json", json.dumps({"artifacts": rows}))
        for name, body in payloads.items():
            archive.writestr("results/" + name, body)


def test_publication_is_create_only_and_full_byte_read_back(tmp_path: Path) -> None:
    bundle = tmp_path / "result.zip"
    _bundle(bundle)
    storage = _Storage()
    first = publish_postshot_output_bundle(
        output_bundle=bundle,
        capture_id="capture-1",
        capture_digest="sha256:" + "a" * 64,
        bucket_name="publish-bucket",
        publication_root=tmp_path / "publication",
        storage_client=storage,
    )
    assert first["artifact_count"] == len(KINDS)
    assert all(row["full_byte_readback_verified"] for row in first["artifacts"])
    second = publish_postshot_output_bundle(
        output_bundle=bundle,
        capture_id="capture-1",
        capture_digest="sha256:" + "a" * 64,
        bucket_name="publish-bucket",
        publication_root=tmp_path / "publication",
        storage_client=storage,
    )
    assert second["artifacts"] == first["artifacts"]


def test_publication_refuses_output_tamper_before_upload(tmp_path: Path) -> None:
    bundle = tmp_path / "result.zip"
    _bundle(bundle, tamper=True)
    storage = _Storage()
    with pytest.raises(CaptureReconstructionPublicationError, match="bundle_invalid"):
        publish_postshot_output_bundle(
            output_bundle=bundle,
            capture_id="capture-1",
            capture_digest="sha256:" + "a" * 64,
            bucket_name="publish-bucket",
            publication_root=tmp_path / "publication",
            storage_client=storage,
        )
    assert storage.store == {}
