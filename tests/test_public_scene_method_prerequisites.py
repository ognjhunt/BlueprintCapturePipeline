from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.public_scene_method_prerequisites import (
    PublicSceneMethodPrerequisiteError,
    materialize_method_prerequisites,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class _Response:
    status_code = 200

    def __init__(
        self,
        payload: dict,
        *,
        headers: dict[str, str] | None = None,
        url: str = "https://example.test/artifact",
    ) -> None:
        self._payload = payload
        self.headers = headers or {}
        self.url = url

    def json(self) -> dict:
        return self._payload


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    repo_root = tmp_path / "repo"
    data_root = tmp_path / "data"
    method_root = tmp_path / "methods"
    authority = method_root / "Entity"
    authority.mkdir(parents=True)
    (authority / "LICENSE").write_text("CC BY-NC 4.0 license text", encoding="utf-8")
    (authority / "README.md").write_text("models are CC BY-NC 4.0", encoding="utf-8")
    checkpoint = data_root / "weights" / "cropformer.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"real-checkpoint-bytes")
    digest = "sha256:" + hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    revision = "1" * 40
    tree = "2" * 40

    def fake_git(_repo: Path, *args: str) -> str:
        return {
            ("rev-parse", "HEAD"): revision,
            ("rev-parse", "HEAD^{tree}"): tree,
            ("status", "--porcelain"): "",
            ("remote", "get-url", "origin"): "https://example.test/Entity",
        }[args]

    monkeypatch.setattr("blueprint_pipeline.public_scene_method_prerequisites._git", fake_git)
    monkeypatch.setattr(
        "blueprint_pipeline.public_scene_method_prerequisites._remote_head",
        lambda _repository: revision,
    )
    remote = {
        "sha": revision,
        "gated": "auto",
        "private": False,
        "siblings": [
            {
                "rfilename": "weights/cropformer.pth",
                "lfs": {
                    "sha256": digest.removeprefix("sha256:"),
                    "size": checkpoint.stat().st_size,
                },
            }
        ],
    }
    monkeypatch.setattr(
        "blueprint_pipeline.public_scene_method_prerequisites.requests.get",
        lambda *_args, **_kwargs: _Response(remote),
    )
    request = {
        "schema_version": "public_scene_method_prerequisite_request.v1",
        "evaluated_on": "2026-08-04",
        "methods": {
            "inpaint360_author_smoke": {
                "rights_authorities": [
                    {
                        "authority_id": "cropformer-rights",
                        "local_path": "Entity",
                        "repository": "https://example.test/Entity",
                        "revision": revision,
                        "tree": tree,
                        "verify_official_remote_head": True,
                        "license_id": "CC-BY-NC-4.0",
                        "evidence_files": [
                            {"path": "LICENSE", "required_text": ["CC BY-NC 4.0"]},
                            {"path": "README.md", "required_text": ["models are CC BY-NC 4.0"]},
                        ],
                    }
                ],
                "artifacts": [
                    {
                        "artifact_id": "cropformer",
                        "local_path": "weights/cropformer.pth",
                        "expected_size_bytes": checkpoint.stat().st_size,
                        "expected_sha256": digest,
                        "publisher": {
                            "service": "huggingface",
                            "repo_type": "dataset",
                            "repository": "example/weights",
                            "revision": revision,
                            "path": "weights/cropformer.pth",
                        },
                        "rights_authority_id": "cropformer-rights",
                    }
                ],
            }
        },
    }
    request_path = repo_root / "request.json"
    _write_json(request_path, request)
    return {
        "repo": repo_root,
        "data": data_root,
        "methods": method_root,
        "request": request_path,
        "checkpoint": checkpoint,
    }


def _run(paths: dict[str, Path]) -> dict:
    return materialize_method_prerequisites(
        request_path=paths["request"],
        repo_root=paths["repo"],
        data_root=paths["data"],
        method_root=paths["methods"],
    )


def test_materializes_remote_and_local_checkpoint_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    receipt = _run(paths)
    method = receipt["methods"]["inpaint360_author_smoke"]
    assert method["checkpoint_rights_established"] is True
    assert method["author_data_rights_established"] is False
    assert method["unchanged_author_smoke_executed"] is False
    assert method["artifacts"][0]["publisher"]["fresh_access_probe_passed"] is True
    assert receipt["raw_secret_values_recorded"] is False


def test_changed_local_checkpoint_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    paths["checkpoint"].write_bytes(b"mutated")
    with pytest.raises(
        PublicSceneMethodPrerequisiteError,
        match="hf_artifact_local_bytes_changed:cropformer",
    ):
        _run(paths)


def test_caller_asserted_status_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    request = json.loads(paths["request"].read_text())
    request["status"] = "admitted"
    _write_json(paths["request"], request)
    with pytest.raises(
        PublicSceneMethodPrerequisiteError,
        match="caller_asserted_prerequisite_status_forbidden",
    ):
        _run(paths)


def test_http_artifact_binds_head_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    request = json.loads(paths["request"].read_text())
    artifact = request["methods"]["inpaint360_author_smoke"]["artifacts"][0]
    artifact["publisher"] = {
        "service": "http",
        "url": "https://example.test/cropformer.pth",
        "filename": "cropformer.pth",
    }
    _write_json(paths["request"], request)
    monkeypatch.setattr(
        "blueprint_pipeline.public_scene_method_prerequisites.requests.head",
        lambda *_args, **_kwargs: _Response(
            {},
            headers={"Content-Length": str(paths["checkpoint"].stat().st_size)},
            url="https://example.test/cropformer.pth",
        ),
    )
    receipt = _run(paths)
    publisher = receipt["methods"]["inpaint360_author_smoke"]["artifacts"][0]["publisher"]
    assert publisher["service"] == "http"
    assert publisher["fresh_access_probe_passed"] is True


def test_http_artifact_rejects_remote_size_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    request = json.loads(paths["request"].read_text())
    artifact = request["methods"]["inpaint360_author_smoke"]["artifacts"][0]
    artifact["publisher"] = {
        "service": "http",
        "url": "https://example.test/cropformer.pth",
        "filename": "cropformer.pth",
    }
    _write_json(paths["request"], request)
    monkeypatch.setattr(
        "blueprint_pipeline.public_scene_method_prerequisites.requests.head",
        lambda *_args, **_kwargs: _Response(
            {}, headers={"Content-Length": "999"}, url="https://example.test/cropformer.pth"
        ),
    )
    with pytest.raises(
        PublicSceneMethodPrerequisiteError,
        match="artifact_remote_size_changed:cropformer",
    ):
        _run(paths)


def test_remote_snapshot_derives_rights_from_publisher_card(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    request = json.loads(paths["request"].read_text())
    method = request["methods"]["inpaint360_author_smoke"]
    method["remote_snapshots"] = [
        {
            "artifact_id": "author-scene",
            "category": "author_data",
            "repo_type": "dataset",
            "repository": "example/author-scene",
            "revision": "1" * 40,
            "path_prefix": "scene/",
            "expected_file_count": 1,
            "expected_total_size_bytes": 12,
            "expected_snapshot_digest": "",
            "expected_license_id": "Apache-2.0",
        }
    ]
    remote_files = [
        {
            "path": "scene/image.png",
            "size_bytes": 12,
            "lfs_sha256": "5" * 64,
            "git_blob_id": "6" * 40,
        }
    ]
    method["remote_snapshots"][0]["expected_snapshot_digest"] = canonical_digest(
        {"files": remote_files}
    )
    _write_json(paths["request"], request)

    checkpoint = paths["checkpoint"]
    checkpoint_digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    responses = {
        "example/weights": {
            "sha": "1" * 40,
            "gated": "auto",
            "private": False,
            "siblings": [
                {
                    "rfilename": "weights/cropformer.pth",
                    "lfs": {
                        "sha256": checkpoint_digest,
                        "size": checkpoint.stat().st_size,
                    },
                }
            ],
        },
        "example/author-scene": {
            "sha": "1" * 40,
            "gated": False,
            "private": False,
            "cardData": {"license": "apache-2.0"},
            "siblings": [
                {
                    "rfilename": "scene/image.png",
                    "size": 12,
                    "blobId": "6" * 40,
                    "lfs": {"sha256": "5" * 64, "size": 12},
                }
            ],
        },
    }

    def fake_get(url: str, **_kwargs: object) -> _Response:
        repository = next(key for key in responses if key in url)
        return _Response(responses[repository])

    monkeypatch.setattr(
        "blueprint_pipeline.public_scene_method_prerequisites.requests.get", fake_get
    )
    receipt = _run(paths)
    snapshot = receipt["methods"]["inpaint360_author_smoke"]["remote_snapshots"][0]
    assert snapshot["rights_established"] is True
    assert snapshot["rights"]["license_id"] == "Apache-2.0"
    assert receipt["methods"]["inpaint360_author_smoke"]["author_data_rights_established"] is True


def test_remote_snapshot_without_publisher_license_stays_unadmitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    request = json.loads(paths["request"].read_text())
    method = request["methods"]["inpaint360_author_smoke"]
    files = [
        {
            "path": "model.bin",
            "size_bytes": 99,
            "lfs_sha256": "7" * 64,
            "git_blob_id": "8" * 40,
        }
    ]
    method["remote_snapshots"] = [
        {
            "artifact_id": "unlicensed-model",
            "category": "checkpoint",
            "repo_type": "model",
            "repository": "example/unlicensed",
            "revision": "1" * 40,
            "path_prefix": "",
            "expected_file_count": 1,
            "expected_total_size_bytes": 99,
            "expected_snapshot_digest": canonical_digest({"files": files}),
            "expected_license_id": "",
        }
    ]
    _write_json(paths["request"], request)

    checkpoint = paths["checkpoint"]
    checkpoint_digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    responses = {
        "example/weights": {
            "sha": "1" * 40,
            "gated": "auto",
            "private": False,
            "siblings": [
                {
                    "rfilename": "weights/cropformer.pth",
                    "lfs": {
                        "sha256": checkpoint_digest,
                        "size": checkpoint.stat().st_size,
                    },
                }
            ],
        },
        "example/unlicensed": {
            "sha": "1" * 40,
            "gated": False,
            "private": False,
            "siblings": [
                {
                    "rfilename": "model.bin",
                    "size": 99,
                    "blobId": "8" * 40,
                    "lfs": {"sha256": "7" * 64, "size": 99},
                }
            ],
        },
    }

    def fake_get(url: str, **_kwargs: object) -> _Response:
        repository = next(key for key in responses if key in url)
        return _Response(responses[repository])

    monkeypatch.setattr(
        "blueprint_pipeline.public_scene_method_prerequisites.requests.get", fake_get
    )
    receipt = _run(paths)
    method_receipt = receipt["methods"]["inpaint360_author_smoke"]
    assert method_receipt["remote_snapshots"][0]["rights_established"] is False
    assert method_receipt["checkpoint_rights_established"] is False
