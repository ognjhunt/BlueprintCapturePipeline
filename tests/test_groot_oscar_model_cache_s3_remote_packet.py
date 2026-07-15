from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

from blueprint_pipeline.groot_oscar_digitalocean_builder import (
    MODEL_CACHE_TARBALL_NAME,
    model_cache_archive_verifier_script,
    verify_packet_tarball,
)
from blueprint_pipeline.groot_oscar_model_cache_s3_remote_packet import (
    prepare_remote_model_cache_packet,
)


COMMIT = "a" * 40
NONCE = "cacheprep1234"


def _wheelhouse(tmp_path: Path, *, startup_hook: bool = False) -> tuple[Path, Path]:
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir(exist_ok=True)
    wheel = wheelhouse / "boto3-1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("boto3/__init__.py", "__version__ = '1.0'\n")
        archive.writestr("boto3-1.0.dist-info/METADATA", "Name: boto3\nVersion: 1.0\n")
        if startup_hook:
            archive.writestr("evil.pth", "import os\n")
    manifest = tmp_path / "dependency_manifest.json"
    requirements = [{"name": "boto3", "version": "1.0"}]
    closure_digest = hashlib.sha256(
        (json.dumps(requirements, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()
    lock_digest = hashlib.sha256(
        (Path(__file__).resolve().parents[1] / "uv.lock").read_bytes()
    ).hexdigest()
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "blueprint_python_wheelhouse.v1",
                "python_version": "3.12",
                "implementation": "cpython",
                "platform_tags": ["manylinux_2_17_x86_64"],
                "lockfile_sha256": lock_digest,
                "requirements_closure_sha256": closure_digest,
                "requirements": requirements,
                "wheels": [
                    {
                        "distribution": "boto3",
                        "bytes": wheel.stat().st_size,
                        "filename": wheel.name,
                        "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
                        "version": "1.0",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return wheelhouse, manifest


def _packet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    startup_hook: bool = False,
) -> dict:
    monkeypatch.setattr(
        "blueprint_pipeline.groot_oscar_model_cache_s3_remote_packet._source_identity",
        lambda _root: (COMMIT, False),
    )
    wheelhouse, dependency_manifest = _wheelhouse(
        tmp_path, startup_hook=startup_hook
    )
    return prepare_remote_model_cache_packet(
        output_dir=tmp_path / "packet-output",
        repo_root=Path(__file__).resolve().parents[1],
        source_commit=COMMIT,
        source_patch_sha256=hashlib.sha256(b"").hexdigest(),
        source_worktree_dirty=False,
        volume_evidence={
            "schema_version": "groot_oscar_runpod_network_volume_evidence.v1",
            "status": "verified",
            "provider_api_verified": True,
            "id": "volume-1",
            "name": f"blueprint-cache-{NONCE}",
            "data_center_id": "US-WA-1",
            "allocation_nonce": NONCE,
            "allocation_name_verified": True,
            "size_bytes": 50 * 1024**3,
        },
        volume_watchdog_handoff={
            "schema_version": "groot_oscar_model_volume_watchdog_handoff.v1",
            "status": "storage_preparation_watchdog_armed",
            "volume_id": "volume-1",
            "teardown_owner": "independent_model_volume_watchdog",
            "watchdog_deadline_epoch": 9_999_999_999.0,
        },
        allocation_nonce=NONCE,
        data_center_id="US-WA-1",
        dependency_wheelhouse=wheelhouse,
        dependency_manifest_path=dependency_manifest,
    )


def test_packet_round_trip_is_exact_and_preimport_verifiable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet = _packet(tmp_path, monkeypatch)
    assert packet["status"] == "ready"
    assert Path(packet["tarball_path"]).name == MODEL_CACHE_TARBALL_NAME
    verification = verify_packet_tarball(packet)
    assert verification["status"] == "verified"
    script = model_cache_archive_verifier_script(
        packet=packet, tarball_path=Path(packet["tarball_path"])
    )
    assert "extractall" not in script
    assert "stdlib_only_preimport_verification" in script
    remote_root = tmp_path / "remote-blueprint-build"
    remote_root.mkdir()
    shutil.copy2(
        packet["tarball_path"], remote_root / MODEL_CACHE_TARBALL_NAME
    )
    verifier = tmp_path / "verifier.py"
    verifier.write_text(
        script.replace("/root/blueprint-build", str(remote_root)),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, "-S", str(verifier)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    remote_evidence = json.loads(
        (remote_root / "model_cache_archive_verification.json").read_text()
    )
    assert remote_evidence["status"] == "verified"
    assert (
        remote_evidence["expected_member_map_sha256"]
        == packet["archive_member_manifest_sha256"]
    )


def test_packet_output_reuse_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _packet(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="model_cache_packet_output_already_exists"):
        _packet(tmp_path, monkeypatch)


def test_parent_rejects_wheel_startup_hook_before_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet = _packet(tmp_path, monkeypatch, startup_hook=True)
    result = verify_packet_tarball(packet)
    assert result["status"] == "blocked"
    assert "digitalocean_model_cache_wheel_startup_hook_forbidden" in result["blockers"]


def test_parent_rejects_extra_or_nonregular_tar_member(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet = _packet(tmp_path, monkeypatch)
    tarball = Path(packet["tarball_path"])
    replacement = tmp_path / "replacement.tar.gz"
    with tarfile.open(tarball, "r:gz") as source, tarfile.open(replacement, "w:gz") as target:
        for member in source.getmembers():
            stream = source.extractfile(member)
            target.addfile(member, stream)
        info = tarfile.TarInfo("groot_oscar_model_cache_s3_remote/extra.py")
        payload = b"raise RuntimeError('must never execute')\n"
        info.size = len(payload)
        import io

        target.addfile(info, io.BytesIO(payload))
    replacement.replace(tarball)
    packet["tarball_sha256"] = hashlib.sha256(tarball.read_bytes()).hexdigest()
    result = verify_packet_tarball(packet)
    assert result["status"] == "blocked"
    assert "digitalocean_model_cache_archive_inventory_mismatch" in result["blockers"]
