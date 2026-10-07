"""Staged Claude helpers must import independently of operator/SDK packages."""
from __future__ import annotations

import base64
import json
import lzma
import os
import subprocess
import sys
from pathlib import Path

from blueprint_pipeline import claude_native_transport, inference_reservations
from blueprint_pipeline import single_g1_kitchen_episode_runpod as overlay
from blueprint_pipeline.claude_opus_authoring_invoker import _request_json
from blueprint_pipeline.task_evaluation_supervisor.inference_reservations import (
    InferenceReservationAudit,
)


def test_legacy_interfaces_reexport_the_same_leaf_contracts():
    assert _request_json is claude_native_transport._request_json
    assert InferenceReservationAudit is inference_reservations.InferenceReservationAudit


def test_native_helpers_import_from_staged_archive_with_pre_migration_base(tmp_path):
    root = Path(overlay.__file__).parents[2]
    package = tmp_path / "base/blueprint_pipeline"
    package.mkdir(parents=True)
    # This base deliberately lacks every Agents/authoring/supervisor module.
    # Existing leaf prerequisites are the actual pre-migration Git sources.
    for filename in ("__init__.py", "common.py", "core/__init__.py", "core/common.py",
                     "openai_prompt_cache.py", "openai_successor_models.py"):
        target = package / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(subprocess.check_output(["git", "show", f"HEAD:src/blueprint_pipeline/{filename}"], cwd=root))
    _, _, payload, _ = overlay._runtime_package_overlay_script()
    archive = json.loads(lzma.decompress(base64.b64decode(payload)))
    filenames = ("haiku_vision_judge.py", "claude_native_transport.py", "inference_reservations.py",
                 "decision_evidence_contracts.py", "wam_episode_consistency_label_openai.py",
                 "wam_generated_video_success_label_openai.py", "wam_generated_video_success_label_gemini.py")
    staged = tmp_path / "overlay/blueprint_pipeline"
    staged.mkdir(parents=True)
    for filename in filenames:
        staged.joinpath(filename).write_bytes(base64.b64decode(archive["modules"][filename]["source_base64"]))
    script = '''import importlib, sys, blueprint_pipeline
from pathlib import Path
blueprint_pipeline.__path__.insert(0, sys.argv[1])
for name in sys.argv[2:]:
    module = importlib.import_module("blueprint_pipeline." + name.removesuffix(".py"))
    assert Path(module.__file__).parent == Path(sys.argv[1])
assert not any(name.startswith("blueprint_pipeline.task_evaluation_supervisor") for name in sys.modules)
assert "agents" not in sys.modules
'''
    env = {**os.environ, "PYTHONPATH": str(package.parent), "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run([sys.executable, "-c", script, str(staged), *filenames], env=env,
                            cwd=tmp_path, capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stderr
