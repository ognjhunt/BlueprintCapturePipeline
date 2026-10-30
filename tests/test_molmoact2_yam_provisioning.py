"""Real bounded stager and lifecycle boundaries with tiny development fixtures."""
import hashlib
import json
from pathlib import Path
import time

import pytest

from blueprint_pipeline import molmoact2_yam_provisioning as provisioning
from blueprint_pipeline import molmoact2_yam_dependencies as dependencies
from tests.test_molmoact2_yam_runtime import prepared_registration


def test_provision_streams_fixed_inputs_then_verifies_before_server(tmp_path, monkeypatch):
    data = b"development-only-exact-input"
    identity = {"size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    interpreter = {**identity, "filename": "python.tgz", "url": "https://github.com/pinned/python.tgz",
                   "maximum_expanded_bytes": 128, "python_relative_path": "python/bin/python3.12", "version": "3.12.12"}
    wheel = {**identity, "filename": "tiny.whl", "url": "https://files.pythonhosted.org/pinned/tiny.whl"}
    files = [{**identity, "path": "configuration.py"}, {**identity, "path": "weights.safetensors"}]
    monkeypatch.setenv(provisioning.DEADLINE_ENV, str(time.time() + 60))
    monkeypatch.setattr(provisioning, "validate_execution_spec", lambda spec: spec)
    monkeypatch.setattr(dependencies, "verified_dependency_manifest", lambda path: {
        "interpreter": interpreter, "server_wheels": [wheel], "total_download_bytes": len(data) * 2})
    monkeypatch.setattr(provisioning, "checkpoint_identity", lambda: {"files": files})
    events = []
    def fetch(url, **kwargs):
        assert kwargs["max_bytes"] == len(data) and kwargs["timeout_seconds"] <= 60
        events.append(("fetch", url))
        kwargs["output_path"].write_bytes(data)
    from blueprint_pipeline.core import security_controls
    monkeypatch.setattr(security_controls, "fetch_bounded_https", fetch)
    monkeypatch.setattr(provisioning, "_extract_interpreter", lambda *_, **__: events.append(("extract", None)))
    monkeypatch.setattr(provisioning, "_command", lambda *_, **__: None)
    monkeypatch.setattr(provisioning, "_install_wheels", lambda *_, **__: events.append(("install", None)))
    def verify(root, *, deadline_epoch):
        assert {path.name for path in root.iterdir()} == {file["path"] for file in files}
        assert all(path.read_bytes() == data for path in root.iterdir())
        assert deadline_epoch > time.time()
        events.append(("verify", None))
    monkeypatch.setattr(provisioning, "verify_checkpoint_directory", verify)
    def start(**kwargs):
        assert events[-1][0] == "verify"
        events.append(("start", None))
        return {"status": "ready"}
    monkeypatch.setattr(provisioning, "_start_server", start)
    result = provisioning.provision(tmp_path / "runtime", tmp_path / "out", {}, root=tmp_path / "root")
    assert result["attempted_download_bytes"] == result["maximum_download_bytes"] == len(data) * 4
    urls = [url for event, url in events if event == "fetch"]
    assert len(urls) == 4 and all(provisioning.MODEL_REVISION in url for url in urls[-2:])
    assert result["peak_gpu_memory_measured"] is False


@pytest.mark.parametrize("fails", [False, True])
def test_local_server_uses_separate_python_offline_cache_identity_reset_and_teardown(tmp_path, monkeypatch, fails):
    _, _, registration = prepared_registration(tmp_path, monkeypatch)
    spec = {"candidate_id": "development-only-yam", "policy_spec": registration["policy_spec"]}
    observed = []
    class Process:
        pid = 12345
        stopped = False
        def poll(self):
            return 1 if self.stopped else None
        def terminate(self):
            self.stopped = True
        def wait(self, timeout):
            return 1
    process = Process()
    def popen(command, **kwargs):
        observed.append((command, kwargs["env"]))
        return process
    class Client:
        def __init__(self, **kwargs):
            assert kwargs["binding"].execution_horizon == 25
        def reset(self):
            if fails:
                raise RuntimeError("identity fixture failure")
        def close(self):
            pass
    monkeypatch.setattr(provisioning.subprocess, "Popen", popen)
    monkeypatch.setattr(provisioning, "MolmoAct2YAMPolicyClient", Client)
    monkeypatch.setenv("HF_TOKEN", "development-only-not-a-credential")
    monkeypatch.setenv("PYTHONHOME", "/isaac-sim/python")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/isaac-sim/lib:/usr/lib")
    output = tmp_path / "out"
    output.mkdir()
    kwargs = dict(python=Path("/separate/python3.12"), runtime=tmp_path, output=output, root=tmp_path,
                  spec=spec, deadline=time.time() + 60)
    if fails:
        with pytest.raises(RuntimeError):
            provisioning._start_server(**kwargs)
        assert process.stopped
    else:
        result = provisioning._start_server(**kwargs)
        assert result["status"] == "ready" and not result["candidate_inference_performed"]
        assert result["server_pid"] == process.pid and not process.stopped
    command, env = observed[0]
    assert command[:3] == ["/separate/python3.12", "-m", "blueprint_pipeline.molmoact2_yam_policy_server"]
    assert "HF_TOKEN" not in env and "PYTHONHOME" not in env
    assert env["LD_LIBRARY_PATH"] == "/usr/lib" and env["PYTHONPATH"] == str(tmp_path)
    assert env["HF_HUB_OFFLINE"] == env["TRANSFORMERS_OFFLINE"] == "1"
    assert Path(env["HF_MODULES_CACHE"]) == tmp_path / "module-cache"
    receipt = json.loads(next(output.glob("adp009d_policy_server_receipt.*.json")).read_text())
    assert receipt["status"] == ("blocked" if fails else "ready")


def test_expired_verification_and_inventory_tamper_refuse_before_code_loading(tmp_path, monkeypatch):
    from blueprint_pipeline import molmoact2_yam_assets as assets
    with pytest.raises(ValueError, match="verification_deadline_exceeded"):
        assets.verify_checkpoint_directory(tmp_path, deadline_epoch=time.time() - 1)
    monkeypatch.setattr(assets, "RELEASE_INVENTORY_SHA256", "sha256:" + "e" * 64)
    with pytest.raises(ValueError, match="release_inventory_changed"):
        assets.release_inventory()


def test_fixed_runtime_contract_counts_required_files_and_refuses_changed_allowance(tmp_path):
    from blueprint_pipeline.molmoact2_yam_bundle import stage_runtime, validate_runtime_contract
    value = stage_runtime(tmp_path)
    assert value["checkpoint_download_bytes"] > 21_768_955_912
    assert value["dependency_download_bytes"] == 4_045_566_007
    assert validate_runtime_contract(value) == value
    value["checkpoint_download_bytes"] -= 1
    with pytest.raises(ValueError, match="sealed_runtime_identity_invalid"):
        validate_runtime_contract(value)
