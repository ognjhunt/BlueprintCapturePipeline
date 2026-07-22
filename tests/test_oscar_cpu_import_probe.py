from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_cpu_import_probe_scopes_cuda_discovery_patch(tmp_path: Path) -> None:
    package = tmp_path / "inference"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "inference_oscar.py").write_text(
        "import torch\n"
        "observed_device = torch.cuda.current_device()\n"
        "assert observed_device == 0\n",
        encoding="utf-8",
    )
    script = (
        Path(__file__).resolve().parents[1]
        / "deploy/docker/robot_eval_worker/groot_oscar_closed_loop/"
        "oscar_cpu_import_probe.py"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path)

    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "BLUEPRINT_OSCAR_CPU_IMPORT_PROBE_PASSED" in completed.stdout
