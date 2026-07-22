from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_cpu_import_probe_scopes_cuda_discovery_patch(tmp_path: Path) -> None:
    torch_package = tmp_path / "torch"
    torch_package.mkdir()
    (torch_package / "__init__.py").write_text(
        "def _current_device():\n"
        "    raise RuntimeError('real CUDA discovery must stay scoped')\n"
        "class _Cuda:\n"
        "    pass\n"
        "cuda = _Cuda()\n"
        "cuda.current_device = _current_device\n",
        encoding="utf-8",
    )
    pytest_dist_info = tmp_path / "pytest-9.1.1.dist-info"
    pytest_dist_info.mkdir()
    (pytest_dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\n"
        "Name: pytest\n"
        "Version: 9.1.1\n",
        encoding="utf-8",
    )
    package = tmp_path / "inference"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "inference_oscar.py").write_text(
        "import torch\n"
        "observed_device = torch.cuda.current_device()\n"
        "assert observed_device == 0\n",
        encoding="utf-8",
    )
    config_package = tmp_path / "worldsim" / "_src" / "configs" / "agibot_control"
    config_package.mkdir(parents=True)
    for parent in (config_package, *config_package.parents[:3]):
        (parent / "__init__.py").write_text("", encoding="utf-8")
    (config_package / "config.py").write_text(
        "import torch\n"
        "observed_device = torch.cuda.current_device()\n"
        "assert observed_device == 0\n"
        "def make_config():\n"
        "    return {'device': observed_device}\n",
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
