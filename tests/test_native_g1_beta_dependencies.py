"""Offline server installation reuses native inputs and ignores ambient Python."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import pytest

from blueprint_pipeline import native_g1_beta_dependencies as build
from blueprint_pipeline import native_g1_beta_dependency_profile as profile
from blueprint_pipeline import native_g1_publisher_source_stage as publisher
from blueprint_pipeline.native_g1_policy_runtime_build import PINNED_IMAGE_ENV
from blueprint_pipeline.native_task_isaaclab_launch import NATIVE_TASK_ARENA_IMAGE


def setup(tmp_path, monkeypatch):
    root = tmp_path / "runtime"
    root.mkdir()
    monkeypatch.setenv(PINNED_IMAGE_ENV, NATIVE_TASK_ARENA_IMAGE)
    original = Path.is_file
    monkeypatch.setattr(Path, "is_file", lambda path: True if str(path) == "/isaac-sim/python.sh" else original(path))
    monkeypatch.setattr(publisher, "verify_g1_publisher_source", lambda path: {"source_revision": "reviewed-source"})
    rows, raw = [], {}
    for name, source in (("native", "runtime_source_packet"), ("extra", "external_download")):
        filename = name + "-1.0-py3-none-any.whl"
        path = tmp_path / filename
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(name + "/__init__.py", "VALUE = 1\n")
        raw[filename] = path.read_bytes()
        row = {"filename": filename, "sha256": hashlib.sha256(raw[filename]).hexdigest(), "size_bytes": len(raw[filename]),
               "source": source, "url": "https://files.pythonhosted.org/" + filename,
               "archive_path": "runtime_dependencies/wheels/" + filename}
        rows.append(row)
        if source == "runtime_source_packet":
            retained = root / "provisioned_runtime_sources" / row["archive_path"]
            retained.parent.mkdir(parents=True)
            retained.write_bytes(raw[filename])
    manifest = {"source_revision": "reviewed-source", "manifest_digest": "sha256:" + "a" * 64,
                "wheels": rows, "core_versions": {"torch": "2.10.0+cu128"}}
    monkeypatch.setattr(profile, "artifact_manifest", lambda path=None: manifest)
    fetched = []
    def fetch(row, destination, *, deadline, budget):
        from blueprint_pipeline.local_policy_provisioning import fetch_exact
        fetched.append(row["filename"])
        def stream(url, **kwargs):
            assert kwargs["max_bytes"] == len(raw[row["filename"]])
            Path(kwargs["output_path"]).write_bytes(raw[row["filename"]])
        return fetch_exact(row, destination, deadline=deadline, budget=budget, fetch=stream)
    monkeypatch.setattr(build, "fetch_exact", fetch)
    calls = []
    monkeypatch.setattr(build, "_run", lambda argv, **kwargs: calls.append((argv, kwargs)))
    return root, manifest, fetched, calls


def test_verified_native_wheels_are_reused_and_all_installs_are_offline(tmp_path, monkeypatch):
    root, manifest, fetched, calls = setup(tmp_path, monkeypatch)
    result = build.build_policy_runtime(root=root, output=tmp_path / "output", deadline=time.time() + 60)
    assert fetched == [manifest["wheels"][1]["filename"]]
    assert result["wheelhouse_verification"]["wheel_count"] == 2
    assert result["external_download_bytes"] == manifest["wheels"][1]["size_bytes"]
    assert all(result[key] is False for key in ("cuda_execution_tested", "checkpoint_loaded", "policy_inference_executed"))
    requirements = (tmp_path / "output/offline-requirements.txt").read_text()
    assert "https:" not in requirements and requirements.count("--hash=sha256:") == 2
    install = calls[1][0]
    assert install[:5] == ["/isaac-sim/python.sh", "-I", "-m", "pip", "install"]
    assert {"--no-index", "--no-deps", "--require-hashes", "--no-compile"} <= set(install)
    assert all(call[1]["python"] == tmp_path / "output/policy-runtime/bin/python" for call in calls[1:])
    assert "3,12" in calls[-1][0][-1] and "torch.version.cuda == '12.8'" in calls[-1][0][-1]


@pytest.mark.parametrize("change", ["missing", "tampered", "expired"])
def test_missing_or_changed_native_wheel_never_falls_back_to_download(tmp_path, monkeypatch, change):
    root, manifest, fetched, calls = setup(tmp_path, monkeypatch)
    retained = root / "provisioned_runtime_sources" / manifest["wheels"][0]["archive_path"]
    if change == "missing":
        retained.unlink()
    elif change == "tampered":
        retained.write_bytes(b"not the sealed wheel")
    with pytest.raises((ValueError, TimeoutError)):
        build.build_policy_runtime(root=root, output=tmp_path / "output", deadline=time.time() + (-1 if change == "expired" else 60))
    assert manifest["wheels"][0]["filename"] not in fetched
    assert calls == []


def test_bootstrap_real_cpu_command_cannot_inherit_python_home_or_import_path(tmp_path, monkeypatch):
    poison = tmp_path / "ambient"
    poison.mkdir()
    (poison / "json.py").write_text("raise RuntimeError('poison imported')\n")
    monkeypatch.setenv("PYTHONHOME", "/missing")
    monkeypatch.setenv("PYTHONPATH", str(poison))
    monkeypatch.setenv("PYTHONEXE", "/wrong")
    monkeypatch.setenv("VIRTUAL_ENV", "/wrong")
    log = tmp_path / "probe.log"
    build._run([sys.executable, "-I", "-c", "import json,os; assert 'PYTHONHOME' not in os.environ; print(json.dumps(True))"],
               deadline=time.time() + 10, log=log)
    assert json.loads(log.read_text()) is True


def test_bootstrap_command_deadline_is_bounded(tmp_path):
    with pytest.raises(subprocess.TimeoutExpired):
        build._run([sys.executable, "-I", "-c", "import time; time.sleep(10)"], deadline=time.time() + .1, log=tmp_path / "deadline.log")
