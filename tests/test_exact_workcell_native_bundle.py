"""Full native sources seal through the existing provider bundle builder."""

from __future__ import annotations

import os
import subprocess
import sys
import zipfile

import pytest

from blueprint_pipeline.exact_workcell_native_bundle import full_matrix_native_runtime_sources
from blueprint_pipeline.native_task_arena_bundle import build_native_task_arena_bundle
from tests.test_native_task_arena_bundle import _packet


@pytest.mark.slow
def test_full_matrix_bundle_contains_importable_native_entrypoint_and_full_authorities(tmp_path):
    sources = full_matrix_native_runtime_sources()
    names = {path.name for path in sources["runtime_module_sources"]}
    assert {
        "exact_workcell_native_session.py",
        "exact_workcell_native_admission.py",
        "evaluation_run_execution.py",
        "adp009d_scene_policy_readiness.py",
        "adp_episode_evidence_index.py",
        "native_task_arena_controls_admission.py",
    } <= names
    assert "blueprint_pipeline/core/immutable_file.py" in sources["runtime_extra_files"]
    worker = tmp_path / "worker.py"
    worker.write_text(
        "from blueprint_pipeline.exact_workcell_native_session import execute_native_evaluation_schedule\n"
    )
    receipt = build_native_task_arena_bundle(
        job_dir=tmp_path / "job",
        packet_dir=_packet(tmp_path, scene_id="840313"),
        worker_source=worker,
        implementation_commit="a" * 40,
        generated_at="fixed",
        **sources,
    )
    root = tmp_path / "sealed"
    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        archive.extractall(root)
    runtime = root / "provider_runtime"
    # Isolated Python removes repository PYTHONPATH and cwd: these imports
    # must resolve entirely from exact source bytes in the sealed archive.
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "from blueprint_pipeline.exact_workcell_native_session import execute_native_evaluation_schedule, ExactWorkcellNativeEvaluationRunExecutor; "
            "from blueprint_pipeline.exact_workcell_native_admission import derive_native_cell_policy_admission; "
            "from blueprint_pipeline.evaluation_run_execution import EvaluationRunExecutionRegistry; "
            "assert 'exact_workcell_native_droid' in EvaluationRunExecutionRegistry([ExactWorkcellNativeEvaluationRunExecutor()]).manifest(); "
            "assert execute_native_evaluation_schedule.__code__.co_filename.startswith(sys.argv[1])",
            str(runtime),
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
