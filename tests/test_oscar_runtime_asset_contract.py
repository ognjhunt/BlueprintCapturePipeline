from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from blueprint_pipeline import oscar_runtime_asset_contract as contract


def _pinned(relative_path: str, contents: bytes) -> contract.PinnedRuntimeFile:
    return contract.PinnedRuntimeFile(
        relative_path=relative_path,
        size_bytes=len(contents),
        sha256=hashlib.sha256(contents).hexdigest(),
    )


def test_contract_pins_exact_oscar_aligned_reason1_and_wan_vae_bytes() -> None:
    assert contract.OSCAR_SOURCE_REVISION == ("4dea2f657e221b0ff24c895fcc8ab4d46d5a9adb")
    assert contract.REASON1_REPO_ID == "nvidia/Cosmos-Reason1-7B"
    assert contract.REASON1_REVISION == ("3210bec0495fdc7a8d3dbb8d58da5711eab4b423")
    assert contract.WAN_VAE_REPO_ID == "Wan-AI/Wan2.1-T2V-1.3B"
    assert contract.WAN_VAE_REVISION == ("37ec512624d61f7aa208f7ea8140a131f93afc9a")
    assert contract.WAN_VAE_FILE.as_dict() == {
        "path": "Wan2.1_VAE.pth",
        "size_bytes": 507_609_880,
        "sha256": "38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981",
    }

    reason1_files = {item.relative_path: item for item in contract.REASON1_SNAPSHOT_FILES}
    assert set(reason1_files) == {
        ".gitattributes",
        "README.md",
        "chat_template.json",
        "config.json",
        "generation_config.json",
        "model-00001-of-00004.safetensors",
        "model-00002-of-00004.safetensors",
        "model-00003-of-00004.safetensors",
        "model-00004-of-00004.safetensors",
        "model.safetensors.index.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    assert reason1_files["model-00001-of-00004.safetensors"].as_dict() == {
        "path": "model-00001-of-00004.safetensors",
        "size_bytes": 4_968_243_304,
        "sha256": "c28404126221997ae8eb70a23b919c96174d42e35ae1d537e0c95093d50b359a",
    }
    assert reason1_files["model-00004-of-00004.safetensors"].as_dict() == {
        "path": "model-00004-of-00004.safetensors",
        "size_bytes": 1_691_924_384,
        "sha256": "91bacbe1ad798e16daa05023b4e4bec70b53c8cd7d757db86c5bc76c4e0bbf15",
    }
    assert contract.REASON1_SNAPSHOT_TOTAL_SIZE_BYTES == 16_591_539_896
    assert contract.RUNTIME_AUXILIARY_ASSET_TOTAL_SIZE_BYTES == 17_099_149_776


def test_contract_pins_exact_oscar_dcp_bytes_and_digest() -> None:
    assert contract.OSCAR_CHECKPOINT_REVISION == ("c9781ffa7dd8556d862d7d9f338a2ea008a58ca6")
    assert [item.as_dict() for item in contract.OSCAR_DCP_FILES] == [
        {
            "path": "model/.metadata",
            "size_bytes": 186_118,
            "sha256": "74ea9359634d4f757ebf1dcf1a6279f6c2980120762f9d5b214d6b95a3cfdac9",
        },
        {
            "path": "model/__0_0.distcp",
            "size_bytes": 4_119_693_122,
            "sha256": "51612eb9d1335b248f3112a206e72c2c3e84c774cee9cf2c459cb3ee62383cd5",
        },
    ]
    payload = contract.asset_contract_payload()
    assert payload["schema_version"] == "oscar_runtime_asset_contract.v1"
    assert payload["network_download_allowed_during_runtime"] is False
    assert payload["contract_digest"].startswith("sha256:")
    assert len(payload["contract_digest"]) == 71


def test_runtime_paths_and_export_contract_are_confined_and_offline(
    tmp_path: Path,
) -> None:
    root = (tmp_path / "runtime_model_cache").resolve()
    paths = contract.runtime_asset_paths(root)
    for path in (
        paths.hf_home,
        paths.reason1_snapshot,
        paths.wan_vae,
        paths.evidence,
        paths.offline_preflight_evidence,
    ):
        assert path.is_relative_to(root)
    assert paths.reason1_snapshot.name == contract.REASON1_REVISION
    assert paths.wan_vae.parent.name == contract.WAN_VAE_REVISION

    environment = contract.offline_runtime_environment(root)
    assert environment == {
        "HF_HOME": str(root / "hf_home"),
        "HF_HUB_CACHE": str(root / "hf_home" / "hub"),
        "HUGGINGFACE_HUB_CACHE": str(root / "hf_home" / "hub"),
        "COSMOS_REASON_PATH": str(paths.reason1_snapshot),
        "WAN_VAE_PATH": str(paths.wan_vae),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    export_script = contract.render_offline_export_script(root)
    assert f"export COSMOS_REASON_PATH={paths.reason1_snapshot}" in export_script
    assert f"export WAN_VAE_PATH={paths.wan_vae}" in export_script
    assert "export HF_HUB_OFFLINE=1" in export_script
    subprocess.run(["bash", "-n"], input=export_script, text=True, check=True)


def test_cache_root_must_be_absolute_and_cannot_be_filesystem_root() -> None:
    with pytest.raises(ValueError, match="cache_root_not_absolute"):
        contract.runtime_asset_paths(Path("relative/cache"))
    with pytest.raises(ValueError, match="cache_root_is_filesystem_root"):
        contract.runtime_asset_paths(Path("/"))


def test_runtime_paths_preserve_hf_wan_snapshot_symlink_to_confined_blob(
    tmp_path: Path,
) -> None:
    root = (tmp_path / "runtime_model_cache").resolve()
    repo_root = (
        root / "hf_home" / "hub" / ("models--" + contract.WAN_VAE_REPO_ID.replace("/", "--"))
    )
    blob_contents = b"confined-wan-vae"
    blob = repo_root / "blobs" / hashlib.sha256(blob_contents).hexdigest()
    blob.parent.mkdir(parents=True)
    blob.write_bytes(blob_contents)
    lexical_snapshot_file = (
        repo_root / "snapshots" / contract.WAN_VAE_REVISION / contract.WAN_VAE_FILENAME
    )
    lexical_snapshot_file.parent.mkdir(parents=True)
    lexical_snapshot_file.symlink_to(Path("../../blobs") / blob.name)

    paths = contract.runtime_asset_paths(root)

    assert paths.wan_vae == lexical_snapshot_file
    assert paths.wan_vae.is_symlink()
    assert paths.wan_vae.name == contract.WAN_VAE_FILENAME
    assert paths.wan_vae.resolve(strict=True) == blob
    verified = contract.verify_pinned_file_set(
        paths.wan_vae.parent,
        (_pinned(contract.WAN_VAE_FILENAME, blob_contents),),
        confinement_root=root,
        strict_inventory=False,
        blocker_prefix="test_wan_vae",
    )
    assert verified["status"] == "passed"
    assert verified["files"][0]["resolved_path"] == str(blob)


def test_runtime_paths_reject_hf_wan_snapshot_symlink_escape(tmp_path: Path) -> None:
    root = (tmp_path / "runtime_model_cache").resolve()
    snapshot_file = (
        root
        / "hf_home"
        / "hub"
        / ("models--" + contract.WAN_VAE_REPO_ID.replace("/", "--"))
        / "snapshots"
        / contract.WAN_VAE_REVISION
        / contract.WAN_VAE_FILENAME
    )
    snapshot_file.parent.mkdir(parents=True)
    outside_blob = tmp_path / "outside-wan-blob"
    outside_blob.write_bytes(b"outside")
    snapshot_file.symlink_to(outside_blob)

    with pytest.raises(ValueError, match="runtime_asset_path_escapes_cache"):
        contract.runtime_asset_paths(root)


def test_file_verifier_rejects_symlink_escape_and_unexpected_inventory(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    snapshot = cache_root / "snapshot"
    snapshot.mkdir(parents=True)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    (snapshot / "model.bin").symlink_to(outside)
    spec = _pinned("model.bin", b"outside")

    escaped = contract.verify_pinned_file_set(
        snapshot,
        (spec,),
        confinement_root=cache_root,
        strict_inventory=True,
        blocker_prefix="test_asset",
    )
    assert escaped["status"] == "blocked"
    assert "test_asset_file_missing_or_unconfined:model.bin" in escaped["blockers"]

    (snapshot / "model.bin").unlink()
    (snapshot / "model.bin").write_bytes(b"outside")
    (snapshot / "unexpected.txt").write_text("extra", encoding="utf-8")
    unexpected = contract.verify_pinned_file_set(
        snapshot,
        (spec,),
        confinement_root=cache_root,
        strict_inventory=True,
        blocker_prefix="test_asset",
    )
    assert unexpected["status"] == "blocked"
    assert "test_asset_snapshot_inventory_mismatch" in unexpected["blockers"]


def _install_fake_huggingface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    reason_files: tuple[contract.PinnedRuntimeFile, ...],
    reason_contents: dict[str, bytes],
    wan_file: contract.PinnedRuntimeFile,
    wan_contents: bytes,
) -> list[str]:
    del tmp_path
    calls: list[str] = []

    def snapshot_download(*, repo_id, revision, cache_dir, token):
        assert repo_id == contract.REASON1_REPO_ID
        assert revision == contract.REASON1_REVISION
        assert token == "test-token"
        calls.append("reason1")
        root = Path(cache_dir) / ("models--" + repo_id.replace("/", "--")) / "snapshots" / revision
        for item in reason_files:
            path = root / item.relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(reason_contents[item.relative_path])
        return str(root)

    def hf_hub_download(*, repo_id, revision, filename, cache_dir, token):
        assert repo_id == contract.WAN_VAE_REPO_ID
        assert revision == contract.WAN_VAE_REVISION
        assert filename == contract.WAN_VAE_FILENAME
        assert token == "test-token"
        calls.append("wan")
        path = (
            Path(cache_dir)
            / ("models--" + repo_id.replace("/", "--"))
            / "snapshots"
            / revision
            / filename
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(wan_contents)
        return str(path)

    fake = ModuleType("huggingface_hub")
    fake.snapshot_download = snapshot_download  # type: ignore[attr-defined]
    fake.hf_hub_download = hf_hub_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    return calls


def test_prepare_once_resumes_verifies_and_reuses_exact_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reason_contents = {
        "config.json": b"reason-config",
        "tokenizer.json": b"reason-tokenizer",
    }
    reason_files = tuple(
        _pinned(relative, contents) for relative, contents in reason_contents.items()
    )
    wan_contents = b"wan-vae"
    wan_file = _pinned(contract.WAN_VAE_FILENAME, wan_contents)
    monkeypatch.setattr(contract, "REASON1_SNAPSHOT_FILES", reason_files)
    monkeypatch.setattr(contract, "WAN_VAE_FILE", wan_file)
    calls = _install_fake_huggingface(
        tmp_path,
        monkeypatch,
        reason_files=reason_files,
        reason_contents=reason_contents,
        wan_file=wan_file,
        wan_contents=wan_contents,
    )
    root = (tmp_path / "runtime_model_cache").resolve()

    first = contract.prepare_runtime_asset_cache(root, token="test-token")
    assert first["status"] == "passed"
    assert first["prepare_action"] == "downloaded_or_resumed_then_verified"
    assert first["verified_file_count"] == 3
    assert calls == ["reason1", "wan"]
    assert "test-token" not in json.dumps(first)

    second = contract.prepare_runtime_asset_cache(root, token="test-token")
    assert second["status"] == "passed"
    assert second["prepare_action"] == "reused_verified_cache"
    assert calls == ["reason1", "wan"]
    assert (
        contract.verify_runtime_asset_cache(root, require_evidence_manifest=True)["status"]
        == "passed"
    )


def test_offline_preflight_checks_environment_processor_and_dcp_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reason_contents = {"config.json": b"reason-config"}
    reason_files = (_pinned("config.json", reason_contents["config.json"]),)
    wan_contents = b"wan-vae"
    wan_file = _pinned(contract.WAN_VAE_FILENAME, wan_contents)
    dcp_contents = {
        "model/.metadata": b"metadata",
        "model/__0_0.distcp": b"checkpoint",
    }
    dcp_files = tuple(_pinned(relative, contents) for relative, contents in dcp_contents.items())
    monkeypatch.setattr(contract, "REASON1_SNAPSHOT_FILES", reason_files)
    monkeypatch.setattr(contract, "WAN_VAE_FILE", wan_file)
    monkeypatch.setattr(contract, "OSCAR_DCP_FILES", dcp_files)
    _install_fake_huggingface(
        tmp_path,
        monkeypatch,
        reason_files=reason_files,
        reason_contents=reason_contents,
        wan_file=wan_file,
        wan_contents=wan_contents,
    )
    root = (tmp_path / "runtime_model_cache").resolve()
    assert contract.prepare_runtime_asset_cache(root, token="test-token")["status"] == "passed"

    processor_calls: list[tuple[str, bool]] = []

    class FakeAutoProcessor:
        @classmethod
        def from_pretrained(cls, path, *, local_files_only):
            processor_calls.append((path, local_files_only))
            return cls()

    transformers = ModuleType("transformers")
    transformers.AutoProcessor = FakeAutoProcessor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    class FakeFileSystemReader:
        def __init__(self, path: str):
            assert path.endswith("/model")

        def read_metadata(self):
            return SimpleNamespace(state_dict_metadata={"net.weight": object()})

    torch = ModuleType("torch")
    distributed = ModuleType("torch.distributed")
    checkpoint = ModuleType("torch.distributed.checkpoint")
    checkpoint.FileSystemReader = FakeFileSystemReader  # type: ignore[attr-defined]
    torch.distributed = distributed  # type: ignore[attr-defined]
    distributed.checkpoint = checkpoint  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", distributed)
    monkeypatch.setitem(sys.modules, "torch.distributed.checkpoint", checkpoint)

    checkpoint_root = (tmp_path / "oscar_checkpoint").resolve()
    for relative, contents in dcp_contents.items():
        path = checkpoint_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
    for name, value in contract.offline_runtime_environment(root).items():
        monkeypatch.setenv(name, value)

    result = contract.offline_preflight(
        root,
        oscar_checkpoint_root=checkpoint_root,
    )
    assert result["status"] == "passed"
    assert result["checks"]["auxiliary_assets_byte_verified"] is True
    assert result["checks"]["reason1_processor_constructible_offline"] is True
    assert result["checks"]["oscar_dcp_bytes_and_metadata_verified"] is True
    assert result["oscar_dcp_verification"]["metadata_state_dict_entry_count"] == 1
    assert processor_calls == [(str(contract.runtime_asset_paths(root).reason1_snapshot), True)]
    assert result["claim_boundary"]["checkpoint_loaded_into_model"] is False
    assert result["claim_boundary"]["generated_video_proven"] is False
    assert contract.runtime_asset_paths(root).offline_preflight_evidence.is_file()


def test_offline_preflight_fails_before_model_probes_when_assets_are_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = (tmp_path / "missing-runtime-cache").resolve()
    for name, value in contract.offline_runtime_environment(root).items():
        monkeypatch.setenv(name, value)
    result = contract.offline_preflight(
        root,
        oscar_checkpoint_root=None,
        processor_probe=False,
    )
    assert result["status"] == "blocked"
    assert "oscar_reason1_root_missing_or_unconfined" in result["blockers"]
    assert "oscar_wan_vae_root_missing_or_unconfined" in result["blockers"]
    assert result["claim_boundary"]["runtime_asset_preflight_passed"] is False


def test_generated_prepare_and_preflight_scripts_are_shell_valid_and_secret_safe(
    tmp_path: Path,
) -> None:
    root = (tmp_path / "runtime_model_cache").resolve()
    checkpoint_root = (tmp_path / "oscar").resolve()
    prepare = contract.render_prepare_once_script(root)
    preflight = contract.render_offline_preflight_script(
        root,
        oscar_checkpoint_root=checkpoint_root,
    )
    for script in (prepare, preflight):
        subprocess.run(["bash", "-n"], input=script, text=True, check=True)
        assert "test-token" not in script
        assert "prepare_runtime_asset_cache" in prepare
        assert "offline_preflight" in preflight
    assert "unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE" in prepare
    assert "export HF_HUB_OFFLINE=1" in preflight
    assert str(root) in prepare
    assert str(checkpoint_root) in preflight
