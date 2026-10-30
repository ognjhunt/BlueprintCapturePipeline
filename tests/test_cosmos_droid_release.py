"""Release bytes, interpreter ABI and money boundaries without model downloads."""

import pytest

from blueprint_pipeline.cosmos_droid_bundle import (
    CANDIDATES,
    CHECKPOINT_BYTES,
    IMAGE_DOWNLOAD_ALLOWANCE_BYTES,
    MAXIMUM_OUTPUT_BYTES,
    runtime_contract,
    stage_runtime,
    transfer_byte_budget,
)
from blueprint_pipeline.cosmos_droid_dependencies import (
    TOTAL_DOWNLOAD_BYTES,
    verified_dependency_manifest,
)
from blueprint_pipeline.cosmos_droid_provisioning import client_wheels, server_environment
from blueprint_pipeline.cosmos_droid_source import (
    SOURCE_ARCHIVE,
    SOURCE_MANIFEST,
    extract_source,
    verify_source_directory,
)


@pytest.mark.parametrize("candidate", CANDIDATES)
def test_release_stages_verified_source_and_counts_every_external_byte(tmp_path, candidate):
    contract = stage_runtime(tmp_path, candidate)
    assert all((tmp_path / name).is_file() for name in contract["required_entries"])
    dependencies = verified_dependency_manifest(tmp_path / "cosmos-droid-dependencies.v1.json")
    external = {
        row["url"]: row["size_bytes"]
        for row in [
            dependencies["interpreter"],
            *dependencies["server_wheels"],
            *dependencies["thin_client_wheels"],
        ]
        if row.get("url")
    }
    assert sum(external.values()) == TOTAL_DOWNLOAD_BYTES == 5_652_022_212
    bundle = {
        "cosmos_runtime": contract,
        "bundle_size_bytes": 1234,
        "runtime_source_packet": {
            "transport": "content_addressed_external_layer.v1",
            "packet_size_bytes": 5678,
        },
    }
    download, upload = transfer_byte_budget(bundle)
    assert (
        download
        == 1234
        + 5678
        + CHECKPOINT_BYTES[candidate]
        + TOTAL_DOWNLOAD_BYTES
        + IMAGE_DOWNLOAD_ALLOWANCE_BYTES
    )
    assert upload == MAXIMUM_OUTPUT_BYTES
    bundle["cosmos_runtime"]["checkpoint_download_bytes"] -= 1
    with pytest.raises(ValueError, match="sealed_runtime_identity"):
        transfer_byte_budget(bundle)


def test_fixed_source_archive_cannot_gain_an_unverified_import(tmp_path):
    stage_runtime(tmp_path, CANDIDATES[1])
    source = extract_source(
        tmp_path / SOURCE_ARCHIVE, tmp_path / SOURCE_MANIFEST, tmp_path / "source"
    )
    assert (source / "cosmos_framework/scripts/action_policy_server_robolab.py").is_file()
    (source / "unexpected_policy_import.py").write_text("raise RuntimeError('unexpected source')\n")
    with pytest.raises(ValueError, match="members_changed"):
        verify_source_directory(source, tmp_path / SOURCE_MANIFEST)


def test_standalone_server_and_thin_native_client_have_distinct_exact_interpreters():
    manifest = verified_dependency_manifest()
    assert manifest["interpreter"]["version"] == "3.13.9"
    names = {row.get("package", row.get("name")) for row in manifest["server_wheels"]}
    assert {"flash-attn", "natten", "torch", "boto3", "webdataset"}.issubset(names)
    assert not {"jupyterlab", "ray", "lerobot", "megatron-core", "torchtitan"}.intersection(names)
    for minor, abi in [("3.11", "cp311"), ("3.12", "cp312")]:
        rows = client_wheels(manifest, minor)
        assert {row["package"] for row in rows} == {
            "openpi-client",
            "websockets",
            "msgpack",
            "typing-extensions",
        }
        assert abi in next(row["filename"] for row in rows if row["package"] == "msgpack")
    with pytest.raises(ValueError, match="interpreter_unsupported"):
        client_wheels(manifest, "3.13")


def test_runtime_rejects_too_small_nano_gpu_after_resealing():
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    from blueprint_pipeline.policy_canary_session_bundle_identity import (
        validate_policy_canary_provider_manifest,
    )

    value = {
        "candidate_ids": [CANDIDATES[1]],
        "cosmos_runtime": runtime_contract(CANDIDATES[1]),
        "minimum_gpu_memory_gb": 48,
    }
    value["input_digest"] = canonical_digest(value, digest_field="input_digest")
    with pytest.raises(ValueError, match="cosmos_gpu_memory"):
        validate_policy_canary_provider_manifest(value)


def test_server_environment_cannot_import_isaac_or_use_ambient_hub_keys(monkeypatch, tmp_path):
    monkeypatch.setenv("PYTHONPATH", "/isaac-sim/site-packages")
    monkeypatch.setenv("PYTHONHOME", "/isaac-sim")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/isaac-sim/kit:/usr/lib")
    monkeypatch.setenv("HF_TOKEN", "dummy-not-a-secret")
    env = server_environment(tmp_path / "runtime", tmp_path / "root")
    assert env["PYTHONPATH"] == str(tmp_path / "runtime")
    assert env["LD_LIBRARY_PATH"] == "/usr/lib"
    assert "PYTHONHOME" not in env and "HF_TOKEN" not in env
    assert env["HF_HUB_OFFLINE"] == env["TRANSFORMERS_OFFLINE"] == "1"


@pytest.mark.parametrize("mutation", ["changed_bytes", "oversized_part", "missing_part"])
def test_source_parts_fit_privileged_leaf_cap_and_reject_changes(tmp_path, monkeypatch, mutation):
    import shutil
    from pathlib import Path
    from blueprint_pipeline import cosmos_droid_source as source
    vendor = Path(source.__file__).parent / "_vendor"
    target_vendor = tmp_path / "package" / "_vendor"
    target_vendor.mkdir(parents=True)
    for name in (*source.SOURCE_PARTS, source.SOURCE_MANIFEST):
        shutil.copyfile(vendor / name, target_vendor / name)
    assert len(source.SOURCE_PARTS) == 5
    assert all((target_vendor / name).stat().st_size <= 1024**2 for name in source.SOURCE_PARTS)
    monkeypatch.setattr(source, "__file__", str(target_vendor.parent / "cosmos_droid_source.py"))
    staged = tmp_path / "staged"
    staged.mkdir()
    source.stage_source(staged)
    assert (staged / source.SOURCE_ARCHIVE).stat().st_size == source.SOURCE_ARCHIVE_BYTES
    part = target_vendor / source.SOURCE_PARTS[0]
    if mutation == "changed_bytes":
        value = bytearray(part.read_bytes())
        value[0] ^= 1
        part.write_bytes(value)
    elif mutation == "oversized_part":
        with part.open("ab") as stream:
            stream.write(b"x")
    else:
        part.unlink()
    with pytest.raises(ValueError, match="source_archive_changed"):
        source.stage_source(staged)
