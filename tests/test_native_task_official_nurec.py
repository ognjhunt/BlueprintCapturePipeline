"""Actual retained NVIDIA bytes and extracted-provider closure, no rendering."""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest

from blueprint_pipeline import native_task_official_nurec as official
from blueprint_pipeline.native_task_arena_bundle import build_native_task_arena_bundle
from tests.test_native_task_arena_bundle import _packet


def test_official_helper_imports_no_ml_or_metrics_and_keeps_its_source_license():
    module, receipt = official.verified_helper()
    assert module.load_config()["rendering"]["warmup_steps"] == 800
    assert receipt["license"] == "Apache-2.0" and receipt["metrics_imported"] is False
    assert receipt["source_revision"] == "045ca8b59622b99a408092124377c66346e8d9c2"
    tree = ast.parse((official.ROOT / "rendering_setup.py").read_text())
    imports = [node.name for node in ast.walk(tree) if isinstance(node, ast.alias)]
    assert not any("torch" in name or "metrics" in name for name in imports)


@pytest.mark.parametrize("relative", ["manifest.json", "rendering_setup.py", "config/nurec_config.yaml", "LICENSE"])
def test_changed_official_bytes_are_not_loaded(tmp_path, relative):
    copied = tmp_path / "source"
    shutil.copytree(official.ROOT, copied)
    path = copied / relative
    path.write_bytes(path.read_bytes() + b"\nchanged\n")
    with pytest.raises(ValueError, match="official_nurec_helper"):
        official.verified_helper(copied)


def test_real_provider_zip_contains_executable_helper_and_all_bound_data(tmp_path):
    package = Path(official.__file__).parent
    worker = tmp_path / "development_worker.py"
    worker.write_text("# development-only bundle fixture\n")
    receipt = build_native_task_arena_bundle(job_dir=tmp_path / "job", packet_dir=_packet(tmp_path, scene_id="development-only"),
        worker_source=worker, runtime_module_sources=[package / "native_task_official_nurec.py", package / "native_task_pink_compatibility.py"],
        implementation_commit="a" * 40, generated_at="fixed")
    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        names = set(archive.namelist())
        assert {"provider_runtime/" + name for name in official.runtime_data_files()} <= names
        archive.extractall(extracted)
    runtime = extracted / "provider_runtime"
    environment = {**os.environ, "PYTHONPATH": str(runtime)}
    probe = subprocess.run([sys.executable, "-c",
        'import json; from blueprint_pipeline.native_task_official_nurec import verified_helper; '
        'm,r=verified_helper(); print(json.dumps({"revision":r["source_revision"],"warmup":m.load_config()["rendering"]["warmup_steps"]}))'],
        cwd=tmp_path, env=environment, check=True, capture_output=True, text=True)
    assert json.loads(probe.stdout) == {"revision": "045ca8b59622b99a408092124377c66346e8d9c2", "warmup": 800}
