"""Hash-pinned FLORA add-ons for an existing vLLM 0.24.0 base image.

ADP-009D / Day-28: CPU rehearsal and admitted workers share this narrow install
path. It never provisions compute, installs model weights, or replaces Torch
or vLLM. The runtime's offline import/preprocessing checks remain mandatory.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
import uuid


VLLM_VERSION = "0.24.0"
ADDONS = {
    "transformers": {"version": "5.8.1", "sha256": "5340fb95962162cdfdae5cc91d7f8fedd92ed75216c1154c5e1f590fcf56dd0e"},
    "qwen-vl-utils": {"version": "0.0.14", "sha256": "5e28657bfd031e56bd447c5901b58ddfc3835285ed100f4c56580e0ade054e96"},
    "decord": {"version": "0.6.0", "sha256": "51997f20be8958e23b7c4061ba45d0efcd86bffd5fe81c695d0befee0d442976"},
    # PyPI av/16.1.0: av-16.1.0-cp312-cp312-manylinux_2_28_x86_64.whl,
    # 41,174,337 bytes; no additional Python dependencies declared.
    "av": {"version": "16.1.0", "sha256": "7ae547f6d5fa31763f73900d43901e8c5fa6367bb9a9840978d57b5a7ae14ed2"},
}
PROTECTED_PACKAGES = ("vllm", "torch", "torchvision", "torchaudio")
_VERSION_PROBE = r'''
import importlib.metadata, json, platform, sys
versions = {}
for name in sys.argv[1:]:
    try:
        versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        versions[name] = None
print(json.dumps({"versions":versions, "python_executable":sys.executable,
    "python_version":platform.python_version(), "platform":sys.platform,
    "machine":platform.machine()}, sort_keys=True))
'''


class FloraDependencyBootstrapError(ValueError):
    """Bootstrap failed; the receipt retains the exact attempt and its logs."""

    def __init__(self, reason: str, *, receipt: dict[str, Any]) -> None:
        super().__init__(reason)
        self.receipt = receipt


def _digest_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                               allow_nan=False).encode()).hexdigest()


def _environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    return environment


def _probe(python: str, *, log_path: Path) -> dict[str, Any]:
    result = subprocess.run([python, "-I", "-B", "-c", _VERSION_PROBE, *PROTECTED_PACKAGES, *ADDONS],
        capture_output=True, text=True, timeout=60, check=False, env=_environment())
    log_path.write_text(result.stdout + result.stderr)
    if result.returncode != 0:
        raise ValueError("flora_dependency_version_probe_failed")
    try:
        value = json.loads(result.stdout)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("flora_dependency_version_probe_invalid") from exc
    versions = value.get("versions") if isinstance(value, dict) else None
    if (not isinstance(versions, dict) or set(versions) != set(PROTECTED_PACKAGES) | set(ADDONS)
            or any(version is not None and (not isinstance(version, str) or not version)
                   for version in versions.values())):
        raise ValueError("flora_dependency_version_probe_invalid")
    return value


def ensure_flora_runtime_dependencies(
    *, output_root: str | Path, python_executable: str | Path | None = None,
) -> dict[str, Any]:
    """Install only missing/wrong approved add-ons, then return a sealed receipt.

    Already-satisfied environments perform no installation or network request.
    Failure raises ``FloraDependencyBootstrapError`` after writing a receipt.
    Callers must still run the shared runtime import/preprocessing preflight.
    """
    python = str(python_executable or sys.executable)
    root = Path(output_root).expanduser().resolve() / ("attempt-" + uuid.uuid4().hex)
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report: dict[str, Any] = {
        "schema_version": "flora_dependency_bootstrap.v1", "status": "blocked",
        "python_executable": python, "vllm_base_version_required": VLLM_VERSION,
        "approved_addons": ADDONS, "dependency_spec_digest": _digest(ADDONS),
        "protected_packages": list(PROTECTED_PACKAGES), "install_attempted": False,
        "network_install_attempted": False, "installed_packages": [], "blockers": [],
        "model_weights_downloaded": False, "model_inference_performed": False,
        "resource_allocation_performed": False, "runtime_import_preflight_required": True,
    }
    try:
        before = _probe(python, log_path=root / "versions_before.log")
        report["before"] = before
        versions = before["versions"]
        if versions["vllm"] != VLLM_VERSION:
            raise ValueError("flora_base_vllm_version_mismatch")
        if versions["torch"] is None:
            raise ValueError("flora_base_torch_missing")
        missing = [name for name, spec in ADDONS.items() if versions[name] != spec["version"]]
        report["requested_packages"] = missing
        if missing:
            requirements = root / "requirements.txt"
            requirements.write_text("".join(
                f"{name}=={ADDONS[name]['version']} --hash=sha256:{ADDONS[name]['sha256']}\n"
                for name in missing))
            command = [python, "-I", "-B", "-m", "pip", "--isolated", "install",
                # PEP 440 equality can admit a local-version variant. Reinstall
                # only these already-filtered mismatches from the exact wheels.
                "--force-reinstall", "--no-deps", "--only-binary=:all:", "--require-hashes", "--no-cache-dir",
                "--disable-pip-version-check", "--no-input", "--retries", "0", "--timeout", "60",
                "--index-url", "https://pypi.org/simple", "--requirement", str(requirements)]
            report.update(install_attempted=True, network_install_attempted=True, install_argv=command,
                          requirements_sha256=_digest_file(requirements))
            with (root / "install.log").open("x") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=600, check=False, env=_environment())
            report["install_exit_code"] = result.returncode
            if result.returncode != 0:
                raise ValueError("flora_dependency_install_failed")
            after = _probe(python, log_path=root / "versions_after.log")
            report["after"] = after
            if any(after["versions"][name] != versions[name] for name in PROTECTED_PACKAGES):
                raise ValueError("flora_protected_dependency_changed")
            if any(after["versions"][name] != spec["version"] for name, spec in ADDONS.items()):
                raise ValueError("flora_dependency_install_version_mismatch")
            report["installed_packages"] = missing
        else:
            report["after"] = before
            (root / "install.log").write_text("No install required; all approved add-ons already match.\n")
        report["status"] = "ready"
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        report["blockers"] = [str(exc) if type(exc) is ValueError else "flora_dependency_bootstrap_failed:" + type(exc).__name__]
    if report["install_attempted"] and "after" not in report:
        # A failed pip process can still have changed an add-on. Retain the
        # post-failure environment without trying another installation.
        try:
            report["after"] = _probe(python, log_path=root / "versions_after.log")
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            report["blockers"].append("flora_dependency_postfailure_probe_failed:" + type(exc).__name__)
    report["wall_seconds"] = time.monotonic() - started
    report["artifacts"] = [{"path": str(path), "sha256": _digest_file(path)}
                           for path in sorted(root.iterdir()) if path.is_file()]
    report["receipt_path"] = str(root / "dependency_bootstrap.json")
    report["receipt_digest"] = _digest(report)
    Path(report["receipt_path"]).write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n")
    if report["status"] != "ready":
        raise FloraDependencyBootstrapError(report["blockers"][0], receipt=report)
    return report
