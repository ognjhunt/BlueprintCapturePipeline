"""Offline budget/closure/bootstrap boundaries, without model weights or GPUs."""

from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace
import time
import zipfile

import pytest

from blueprint_pipeline import trlc_pi05_bundle as bundle
from blueprint_pipeline import trlc_pi05_provisioning as provision
from blueprint_pipeline import native_g1_beta_dependency_profile as dependencies
from blueprint_pipeline import native_task_runtime_source_packet as native_packet
from blueprint_pipeline import trlc_pi05_source as source


def test_native_input_quote_binds_exact_reused_wheels_and_expanded_scene(tmp_path, monkeypatch):
    native, scene = tmp_path / "native.zip", tmp_path / "scene.zip"
    for path, size in ((native, 100_000), (scene, 200_000)):
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("compressible.usda", b"a" * size)
    rows = [
        dict(row)
        for row in dependencies.artifact_manifest()["wheels"]
        if row["source"] == "runtime_source_packet"
    ]
    receipt = {
        "verified_packet_path": str(native),
        "packet_size_bytes": native.stat().st_size,
        "runtime_dependency_wheels": rows,
    }
    monkeypatch.setattr(
        native_packet, "verify_native_task_runtime_source_packet", lambda _: receipt
    )
    result = bundle.validate_native_inputs(
        runtime_source_receipt=tmp_path / "receipt.json", base_bundle_path=scene
    )
    assert result["native_packet_expanded_bytes"] == 100_000
    assert result["base_bundle_expanded_bytes"] == 200_000
    assert 100_000_000_000 < result["bounded_footprint_bytes"] < bundle.PROVIDER_DISK_BYTES
    assert result["provider_disk_gb"] == 200
    monkeypatch.setattr(bundle, "PROVIDER_DISK_BYTES", result["bounded_footprint_bytes"] - 1)
    with pytest.raises(ValueError, match="disk_budget"):
        bundle.validate_native_inputs(
            runtime_source_receipt=tmp_path / "receipt.json", base_bundle_path=scene
        )
    rows[0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="dependency_identity"):
        bundle.validate_native_inputs(
            runtime_source_receipt=tmp_path / "receipt.json", base_bundle_path=scene
        )


def test_bootstrap_downloads_only_pinned_assets_and_installs_offline(tmp_path, monkeypatch):
    runtime, output, root = tmp_path / "runtime", tmp_path / "output", tmp_path / "server"
    runtime.mkdir()
    native_wheel = runtime / "provisioned_runtime_sources/runtime_dependencies/wheels/native.whl"
    native_wheel.parent.mkdir(parents=True)
    native_wheel.write_bytes(b"native")
    external = {
        "source": "external_download",
        "filename": "external.whl",
        "size_bytes": 8,
        "sha256": hashlib.sha256(b"external").hexdigest(),
        "url": "https://files.pythonhosted.org/exact.whl",
    }
    reused = {
        "source": "runtime_source_packet",
        "filename": "native.whl",
        "size_bytes": 6,
        "sha256": hashlib.sha256(b"native").hexdigest(),
        "archive_path": "runtime_dependencies/wheels/native.whl",
    }
    manifest = {"wheels": [reused, external], "core_versions": {"torch": "2.10.0+cu128"}}
    checkpoint = {
        "files": [
            {
                "path": "model.safetensors",
                "size_bytes": 5,
                "sha256": "sha256:" + hashlib.sha256(b"model").hexdigest(),
            }
        ]
    }
    monkeypatch.setenv(provision.DEADLINE_ENV, str(time.time() + 120))
    monkeypatch.setattr(dependencies, "artifact_manifest", lambda *_: manifest)
    monkeypatch.setattr(dependencies, "TOTAL_DOWNLOAD_BYTES", 8)
    monkeypatch.setattr(provision, "checkpoint_identity", lambda: checkpoint)
    monkeypatch.setattr(provision, "validate_execution_spec", lambda value: value)
    events, commands, fetched = [], [], []
    monkeypatch.setattr(source, "extract_source", lambda *args: events.append("source_verified"))
    monkeypatch.setattr(
        source, "verify_source_directory", lambda *args: events.append("source_reverified")
    )
    monkeypatch.setattr(
        provision,
        "verify_tokenizer_directory",
        lambda *args, **kwargs: events.append("tokenizer_verified"),
    )
    monkeypatch.setattr(
        provision,
        "verify_checkpoint_directory",
        lambda *args, **kwargs: events.append("checkpoint_verified"),
    )

    def verify(paths, **kwargs):
        assert set(paths) == {"native.whl", "external.whl"}
        assert paths["native.whl"] == native_wheel
        assert kwargs["deadline_epoch"] > time.time()
        events.append("wheels_verified")
        return {"wheel_count": 2}

    monkeypatch.setattr(dependencies, "verify_wheelhouse", verify)

    def fetch(row, destination, *, deadline, budget):
        budget.reserve(row["size_bytes"])
        fetched.append((deepcopy(row), deadline))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"external" if row.get("filename") else b"model")
        return {"size_bytes": row["size_bytes"]}

    monkeypatch.setattr(provision, "fetch_exact", fetch)
    monkeypatch.setattr(provision, "_run", lambda argv, **kwargs: commands.append((argv, kwargs)))

    def start(**kwargs):
        assert events[-2:] == ["checkpoint_verified", "source_reverified"]
        assert kwargs["python"] == root / "policy-runtime/bin/python"
        return {"status": "ready"}

    monkeypatch.setattr(provision, "_start_server", start)
    result = provision.provision(runtime, output, {"candidate_id": "fixture"}, root=root)
    assert result["maximum_download_bytes"] == result["attempted_download_bytes"] == 13
    assert len(fetched) == 2
    assert (
        fetched[1][0]["url"]
        == f"https://huggingface.co/{provision.MODEL_ID}/resolve/{provision.MODEL_REVISION}/model.safetensors?download=true"
    )
    assert all("apt" not in command for command, _ in commands)
    pip_command = commands[1][0]
    assert pip_command[:3] == ["/isaac-sim/python.sh", "-I", "-B"]
    assert {"--no-index", "--no-deps", "--require-hashes", "--no-compile"} <= set(pip_command)
    assert "--copies" in commands[0][0]
    assert commands[-1][0][-1].find("torch.version.cuda=='12.8'") >= 0
    assert "--hash=sha256:" in (root / "offline-requirements.txt").read_text()
    with pytest.raises(ValueError, match="existing_runtime"):
        provision.provision(runtime, output, {"candidate_id": "fixture"}, root=root)
    assert len(fetched) == 2  # No implicit bootstrap retry.


@pytest.mark.parametrize("fails", [False, True])
def test_server_isolated_environment_and_failed_start_teardown(tmp_path, monkeypatch, fails):
    root, output = tmp_path / "root", tmp_path / "output"
    root.mkdir()
    output.mkdir()
    for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "PYTHONEXE", "HF_TOKEN"):
        monkeypatch.setenv(name, "must-not-reach-server")
    monkeypatch.setattr(provision, "serialized_binding", lambda _: {})
    monkeypatch.setattr(provision, "policy_binding", lambda _: None)
    process = SimpleNamespace(pid=123, terminated=False)
    process.poll = lambda: 1 if process.terminated or fails else None
    process.terminate = lambda: setattr(process, "terminated", True)
    process.wait = lambda **_: 0
    observed = []
    monkeypatch.setattr(
        provision.subprocess,
        "Popen",
        lambda command, **kwargs: observed.append((command, kwargs)) or process,
    )
    resets = []
    monkeypatch.setattr(
        provision,
        "TRLCPi05PolicyClient",
        lambda **_: SimpleNamespace(reset=lambda: resets.append(True), close=lambda: None),
    )
    python = root / "policy-runtime/bin/python"
    kwargs = dict(
        python=python,
        runtime=tmp_path / "runtime",
        output=output,
        root=root,
        spec={"candidate_id": "fixture", "policy_spec": {}},
        deadline=time.time() + 30,
    )
    if fails:
        with pytest.raises(ValueError, match="exited_before_ready"):
            provision._start_server(**kwargs)
        assert not resets
    else:
        result = provision._start_server(**kwargs)
        assert result["status"] == "ready" and result["candidate_policy_queried"] is False
        assert resets == [True]
    command, process_kwargs = observed[0]
    assert command[:3] == ["/isaac-sim/python.sh", "-I", "-B"]
    assert set(process_kwargs["env"]).isdisjoint(
        {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "HF_TOKEN"}
    )
    assert process_kwargs["env"]["PYTHONEXE"] == str(python)
    assert process_kwargs["env"]["HF_HUB_OFFLINE"] == "1"
    receipt = json.loads((output / "adp009d_policy_server_receipt.fixture.json").read_text())
    assert receipt["candidate_inference_performed"] is False


def test_expired_bootstrap_never_creates_runtime_or_downloads(tmp_path, monkeypatch):
    monkeypatch.setattr(provision, "validate_execution_spec", lambda value: value)
    monkeypatch.setenv(provision.DEADLINE_ENV, str(time.time() - 1))
    root = tmp_path / "runtime"
    with pytest.raises(ValueError, match="deadline_exceeded"):
        provision.provision(tmp_path / "staged", tmp_path / "output", {}, root=root)
    assert not root.exists()
