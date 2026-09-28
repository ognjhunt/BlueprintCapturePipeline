from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import venv

from blueprint_pipeline import adp009d_groot_worker_identity as identity


def _git_blob_sha1(value: bytes) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {len(value)}\0".encode("ascii"))
    digest.update(value)
    return digest.hexdigest()


def test_checkpoint_inventory_binds_paths_sizes_and_bytes_but_not_hf_cache(tmp_path) -> None:
    root = tmp_path / "checkpoint"
    (root / "weights").mkdir(parents=True)
    (root / "weights/model.bin").write_bytes(b"model-bytes")
    (root / "config.json").write_text('{"model":"groot"}\n', encoding="utf-8")
    (root / ".cache/huggingface").mkdir(parents=True)
    (root / ".cache/huggingface/download.lock").write_text("mutable", encoding="utf-8")

    result = identity.checkpoint_inventory(root)
    rows = [
        {
            "path": "config.json",
            "size_bytes": len(b'{"model":"groot"}\n'),
            "sha256": hashlib.sha256(b'{"model":"groot"}\n').hexdigest(),
        },
        {
            "path": "weights/model.bin",
            "size_bytes": len(b"model-bytes"),
            "sha256": hashlib.sha256(b"model-bytes").hexdigest(),
        },
    ]
    expected = hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    assert result == {
        "file_count": 2,
        "total_bytes": sum(row["size_bytes"] for row in rows),
        "checkpoint_files_sha256": expected,
        "checkpoint_files": [
            {
                **row,
                "git_blob_sha1": _git_blob_sha1(
                    (root / row["path"]).read_bytes()
                ),
            }
            for row in rows
        ],
    }


def test_checkpoint_interface_binding_names_exact_loaded_processor_config() -> None:
    assert identity.expected_checkpoint_interface_binding() == {
        "processor_config": {
            "path": "processor_config.json",
            "size_bytes": 2_833,
            "digest_algorithm": "git_blob_sha1",
            "digest": "55b4d74b3565274662ba33eefe9bdb0ca75df3e9",
        },
        "video_delta_indices": [0],
    }


def test_identity_is_verified_only_after_source_bytes_and_environment_are_observed(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "source"
    checkpoint = tmp_path / "checkpoint"
    source.mkdir()
    checkpoint.mkdir()
    (checkpoint / "model.bin").write_bytes(b"frozen")
    monkeypatch.setattr(identity, "EXPECTED_CHECKPOINT_BYTES", len(b"frozen"))
    monkeypatch.setattr(
        identity,
        "EXPECTED_CHECKPOINT_FILE_MANIFEST",
        (("model.bin", 6, "sha256", hashlib.sha256(b"frozen").hexdigest()),),
    )

    def _run(command):
        if command[0] == "git":
            return identity.GROOT_SOURCE_REVISION
        assert command[:2] == ("/venv/bin/python", "-c")
        return json.dumps(
            {
                "schema_version": "python_distribution_inventory.v1",
                "python": {"executable": "/venv/bin/python", "version": "3.12"},
                "distributions": [
                    {"name": "gr00t", "version": "1.0", "direct_url": None},
                    {"name": "pyzmq", "version": "26.0", "direct_url": None},
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    monkeypatch.setattr(identity, "_run_text", _run)
    receipt = identity.build_worker_identity(
        source_root=source, checkpoint_root=checkpoint, python="/venv/bin/python"
    )

    assert receipt["status"] == "verified"
    assert receipt["blockers"] == []
    assert len(receipt["checkpoint_files_sha256"]) == 64
    assert len(receipt["environment_lock_sha256"]) == 64
    assert receipt["environment_lock_distribution_count"] == 2
    assert receipt["environment_lock_observer"] == "stdlib_importlib_metadata"
    assert receipt["publisher_inventory_role"] == (
        "fetch_admission_not_local_content_digest"
    )
    assert receipt["checkpoint_content_manifest_digest"] == (
        identity.expected_checkpoint_content_binding()["file_manifest_digest"]
    )


def test_same_size_checkpoint_tamper_blocks_exact_content_identity(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "source"
    checkpoint = tmp_path / "checkpoint"
    source.mkdir()
    checkpoint.mkdir()
    expected_bytes = b"frozen"
    tampered_bytes = b"tamper"
    assert len(expected_bytes) == len(tampered_bytes)
    (checkpoint / "model.bin").write_bytes(tampered_bytes)
    monkeypatch.setattr(
        identity,
        "EXPECTED_CHECKPOINT_FILE_MANIFEST",
        (
            (
                "model.bin",
                len(expected_bytes),
                "sha256",
                hashlib.sha256(expected_bytes).hexdigest(),
            ),
        ),
    )

    def _run(command):
        if command[0] == "git":
            return identity.GROOT_SOURCE_REVISION
        return json.dumps(
            {
                "schema_version": "python_distribution_inventory.v1",
                "python": {"executable": "/venv/bin/python", "version": "3.12"},
                "distributions": [
                    {"name": "gr00t", "version": "1.0", "direct_url": None}
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    monkeypatch.setattr(identity, "_run_text", _run)
    receipt = identity.build_worker_identity(
        source_root=source, checkpoint_root=checkpoint, python="/venv/bin/python"
    )

    assert receipt["file_count"] == 1
    assert receipt["total_bytes"] == len(expected_bytes)
    assert receipt["status"] == "blocked"
    assert receipt["blockers"] == [
        "groot_worker_checkpoint_file_identity_mismatch:model.bin"
    ]


def test_byte_count_or_source_drift_blocks_identity(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source"
    checkpoint = tmp_path / "checkpoint"
    source.mkdir()
    checkpoint.mkdir()
    (checkpoint / "model.bin").write_bytes(b"wrong-size")

    def _run(command):
        if command[0] == "git":
            return "0" * 40
        return json.dumps(
            {
                "schema_version": "python_distribution_inventory.v1",
                "python": {"executable": "/venv/bin/python", "version": "3.12"},
                "distributions": [
                    {"name": "gr00t", "version": "1.0", "direct_url": None}
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    monkeypatch.setattr(identity, "_run_text", _run)
    receipt = identity.build_worker_identity(
        source_root=source, checkpoint_root=checkpoint, python="/venv/bin/python"
    )

    assert receipt["status"] == "blocked"
    assert "groot_worker_source_revision_mismatch" in receipt["blockers"]
    assert "groot_worker_checkpoint_byte_count_mismatch" in receipt["blockers"]


def test_subprocess_failures_become_typed_blockers(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.bin").write_bytes(b"x")

    def _fail(command):
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(identity, "_run_text", _fail)
    receipt = identity.build_worker_identity(
        source_root=tmp_path / "missing", checkpoint_root=checkpoint, python="python"
    )

    assert receipt["status"] == "blocked"
    assert "groot_worker_source_revision_unobserved" in receipt["blockers"]
    assert "groot_worker_environment_unobserved" in receipt["blockers"]


def test_environment_lock_works_in_uv_style_venv_without_pip(tmp_path) -> None:
    venv_root = tmp_path / "policy-venv"
    venv.EnvBuilder(with_pip=False).create(venv_root)
    python = venv_root / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")

    inventory_text = identity._run_text(
        (str(python), "-c", identity._ENVIRONMENT_INVENTORY_CODE)
    )
    inventory = json.loads(inventory_text)

    assert inventory["schema_version"] == "python_distribution_inventory.v1"
    assert inventory["python"]["executable"] == str(python)
    assert isinstance(inventory["distributions"], list)
    pip_probe = subprocess.run(
        [str(python), "-m", "pip", "--version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert pip_probe.returncode != 0
