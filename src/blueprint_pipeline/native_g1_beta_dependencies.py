"""Build G1 inference offline using exact native wheels and bounded small extras."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

from .decision_evidence_contracts import canonical_digest
from .local_policy_provisioning import TransferBudget, fetch_exact, remaining_seconds


def _run(argv, *, deadline, log, python=None):
    environment = {key: value for key, value in os.environ.items()
                   if key not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "PYTHONEXE"}}
    environment.update(PYTHONNOUSERSITE="1", PIP_NO_INDEX="1", PIP_DISABLE_PIP_VERSION_CHECK="1",
                       HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    if python is not None:
        environment["PYTHONEXE"] = str(python)
    with log.open("ab") as stream:
        subprocess.run(argv, env=environment, stdout=stream, stderr=subprocess.STDOUT,
                       check=True, timeout=remaining_seconds(deadline))


def build_policy_runtime(*, root: Path, output: Path, deadline: float):
    from .native_g1_beta_dependency_profile import (DEPENDENCY_MANIFEST_FILENAME, TOTAL_DOWNLOAD_BYTES,
        artifact_manifest, verify_wheelhouse)
    from .native_g1_policy_runtime_build import PINNED_IMAGE_ENV
    from .native_g1_publisher_source_stage import verify_g1_publisher_source
    from .native_task_isaaclab_launch import NATIVE_TASK_ARENA_IMAGE
    remaining_seconds(deadline)
    if (not root.is_absolute() or root.is_symlink() or not root.is_dir() or not output.is_absolute()
            or output.exists() or output.is_symlink() or os.environ.get(PINNED_IMAGE_ENV) != NATIVE_TASK_ARENA_IMAGE
            or not Path("/isaac-sim/python.sh").is_file()):
        raise ValueError("g1_beta_dependency_environment_invalid")
    publisher = verify_g1_publisher_source(root / "publisher-source")
    manifest = artifact_manifest(root / DEPENDENCY_MANIFEST_FILENAME)
    if manifest["source_revision"] != publisher["source_revision"]:
        raise ValueError("g1_beta_dependency_source_changed")
    output.mkdir(parents=True)
    downloads = output / "wheel-downloads"
    budget = TransferBudget(TOTAL_DOWNLOAD_BYTES)
    wheel_paths = {}
    for row in manifest["wheels"]:
        remaining_seconds(deadline)
        if row["source"] == "runtime_source_packet":
            path = root / "provisioned_runtime_sources" / row["archive_path"]
            if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root / "provisioned_runtime_sources"):
                raise ValueError("g1_beta_required_native_dependency_wheel_missing")
        else:
            path = downloads / row["filename"]
            fetch_exact(row, path, deadline=deadline, budget=budget)
        wheel_paths[row["filename"]] = path
    verified = verify_wheelhouse(wheel_paths, deadline_epoch=deadline)
    environment = output / "policy-runtime"
    python = environment / "bin/python"
    log = output / "build.log"
    _run(["/isaac-sim/python.sh", "-m", "venv", "--copies", str(environment)], deadline=deadline, log=log)
    requirements = output / "offline-requirements.txt"
    requirements.write_text("\n".join(str(wheel_paths[row["filename"]]) + " --hash=sha256:" + row["sha256"]
                                      for row in manifest["wheels"]) + "\n")
    launcher = ["/isaac-sim/python.sh", "-I"]
    _run([*launcher, "-m", "pip", "install", "--no-index", "--no-deps", "--require-hashes", "--no-compile",
          "-r", str(requirements)], deadline=deadline, log=log, python=python)
    _run([*launcher, "-m", "pip", "check"], deadline=deadline, log=log, python=python)
    source = root / "publisher-source/source/lerobot/src"
    probe = (
        "import sys,pathlib,importlib.metadata as m;"
        "assert sys.version_info[:2] == (3,12), 'g1_beta_python_version_changed';"
        f"sys.path.insert(0,{str(source)!r});"
        "import torch,torchvision,numpy,transformers,diffusers,lerobot;"
        "from lerobot.policies.factory import get_policy_class;"
        "from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy;"
        "from lerobot.policies.pi05.modeling_pi05 import PI05Policy;"
        f"assert pathlib.Path(lerobot.__file__).resolve().is_relative_to(pathlib.Path({str(source)!r}));"
        f"expected={manifest['core_versions']!r};"
        "assert all(m.version(name)==version for name,version in expected.items()), 'g1_beta_dependency_version_changed';"
        "assert torch.version.cuda == '12.8', 'g1_beta_torch_cuda_build_changed'"
    )
    _run([*launcher, "-c", probe], deadline=deadline, log=log, python=python)
    remaining_seconds(deadline)
    result = {"schema_version": "native_g1_beta_dependency_build.v1", "status": "offline_import_probe_passed",
        "python_executable": str(python), "source_revision": manifest["source_revision"],
        "dependency_manifest_digest": manifest["manifest_digest"], "wheelhouse_verification": verified,
        "external_download_bytes": budget.attempted_bytes, "cuda_execution_tested": False,
        "checkpoint_loaded": False, "policy_inference_executed": False}
    result["receipt_digest"] = canonical_digest(result, digest_field="receipt_digest")
    (output / "native_g1_beta_dependency_build.v1.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
