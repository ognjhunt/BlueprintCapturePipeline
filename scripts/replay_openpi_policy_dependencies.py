"""Replay the retained OpenPI packaging failure and the production repair on CPU.

This networked Linux check installs the frozen policy environment, never a
checkpoint. It consumes the generated exports, venv creation and sync command
so the preflight cannot silently test a different installer. Logs and a digest-bound receipt are
kept separately from the disposable environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
import tomllib
from pathlib import Path
from urllib.request import urlopen

from blueprint_pipeline.adp009d_policy_candidate_admission import EXPECTED_CANDIDATES
from blueprint_pipeline.adp009d_policy_provisioning import (
    ACTIVE_CANDIDATE_DEPENDENCY_LOCKS,
    POLICY_PYTHON_REQUESTS,
    POLICY_PYTHON_INSTALL_DIR,
    POLICY_SOURCE_ROOT,
    UV_ARCHIVE_SHA256,
    UV_ARCHIVE_URL,
    ISAAC_INTERPRETER,
    OPENPI_CLIENT_WEBSOCKETS_WHEEL,
    _isaac_client_commands,
    build_provisioning_script,
    policy_venv_root,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest

CANDIDATE = "pi05_droid"
SCHEMA = "openpi_policy_dependency_replay.v1"


def production_venv_arguments(*, venv: Path, uv: Path) -> list[str]:
    lines = [line for line in build_provisioning_script(CANDIDATE).splitlines()
             if '"$UV" venv ' in line]
    if len(lines) != 1:
        raise ValueError("policy_dependency_replay_venv_command_ambiguous")
    words = shlex.split(lines[0])
    if words != ["$UV", "venv", "--python", POLICY_PYTHON_REQUESTS[CANDIDATE],
                 policy_venv_root(CANDIDATE)]:
        raise ValueError("policy_dependency_replay_venv_binding_invalid")
    return [str(uv), *words[1:-1], str(venv)]


def production_environment_setup(*, source: Path, venv: Path, uv: Path,
                                 python_install_root: Path) -> str:
    """Replay exports, venv creation and frozen sync in one production scope."""
    lines = build_provisioning_script(CANDIDATE).splitlines()
    venv_indexes = [i for i, line in enumerate(lines) if '"$UV" venv ' in line]
    sync_indexes = [i for i, line in enumerate(lines) if '"$UV" sync ' in line]
    if (len(venv_indexes) != 1 or len(sync_indexes) != 1
            or sync_indexes[0] <= venv_indexes[0]):
        raise ValueError("policy_dependency_replay_environment_command_order_invalid")
    settings = {}
    for index, line in enumerate(lines):
        if not line.startswith("export UV_PYTHON_"):
            continue
        if index >= venv_indexes[0]:
            raise ValueError("policy_dependency_replay_python_environment_too_late")
        words = shlex.split(line)
        if len(words) != 2 or "=" not in words[1]:
            raise ValueError("policy_dependency_replay_python_environment_invalid")
        key, value = words[1].split("=", 1)
        if key in settings:
            raise ValueError("policy_dependency_replay_python_environment_duplicated")
        settings[key] = value
    if (set(settings) != {"UV_PYTHON_INSTALL_DIR", "UV_PYTHON_PREFERENCE", "UV_PYTHON_DOWNLOADS"}
            or settings["UV_PYTHON_INSTALL_DIR"] != POLICY_PYTHON_INSTALL_DIR):
        raise ValueError("policy_dependency_replay_python_environment_unbound")
    settings["UV_PYTHON_INSTALL_DIR"] = str(python_install_root)
    return "\n".join([
        "set -euo pipefail",
        *(f"export {key}={shlex.quote(value)}" for key, value in settings.items()),
        shlex.join(production_venv_arguments(venv=venv, uv=uv)),
        f"VIRTUAL_ENV={shlex.quote(str(venv))} "
        + shlex.join(production_sync_arguments(source=source, venv=venv, uv=uv)),
    ])


def production_sync_arguments(*, source: Path, venv: Path, uv: Path) -> list[str]:
    """Rebase only paths; preserve the production installer and every flag."""
    lines = [
        line for line in build_provisioning_script(CANDIDATE).splitlines()
        if '"$UV" sync ' in line
    ]
    if len(lines) != 1:
        raise ValueError("policy_dependency_replay_sync_command_ambiguous")
    words = shlex.split(lines[0])
    expected_env = f"VIRTUAL_ENV={policy_venv_root(CANDIDATE)}"
    if words[:3] != [expected_env, "$UV", "sync"]:
        raise ValueError("policy_dependency_replay_sync_command_unexpected")
    args = [str(uv), *words[2:]]
    for flag, expected, replacement in (
        ("--project", f"{POLICY_SOURCE_ROOT}/{CANDIDATE}", source),
        ("--python", f"{policy_venv_root(CANDIDATE)}/bin/python", venv / "bin/python"),
    ):
        if args.count(flag) != 1 or args[args.index(flag) + 1] != expected:
            raise ValueError(f"policy_dependency_replay_binding_invalid:{flag}")
        args[args.index(flag) + 1] = str(replacement)
    return args


def mujoco_requirement(lock: dict) -> str:
    package = next(p for p in lock["package"] if p["name"] == "mujoco")
    if package["version"] != "2.3.7":
        raise ValueError("policy_dependency_replay_retained_failure_version_changed")
    artifacts = [*package.get("wheels", []), package["sdist"]]
    hashes = sorted({artifact["hash"] for artifact in artifacts})
    return "mujoco==2.3.7 " + " ".join(f"--hash={value}" for value in hashes) + "\n"


def replay(workspace: Path, output: Path, *, source_commit: str) -> dict:
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise ValueError("policy_dependency_replay_requires_linux_x86_64")
    if sys.version_info[:2] != (3, 12):
        raise ValueError("policy_dependency_replay_negative_requires_python312")
    if workspace.exists() or output.exists():
        raise ValueError("policy_dependency_replay_preserve_existing_output")
    if shutil.disk_usage(workspace.parent).free < 12 * 1024**3:
        raise ValueError("policy_dependency_replay_requires_12gib_headroom")
    workspace.mkdir()
    output.mkdir()
    env = {
        "PATH": os.environ["PATH"], "HOME": os.environ["HOME"], "LANG": "C.UTF-8",
        "UV_PYTHON_INSTALL_DIR": str(workspace / "python"),
        "UV_CACHE_DIR": str(workspace / "cache"), "UV_NO_CACHE": "1",
        "UV_CONCURRENT_DOWNLOADS": "4", "UV_CONCURRENT_INSTALLS": "2",
        "GIT_LFS_SKIP_SMUDGE": "1", "JAX_PLATFORMS": "cpu",
    }
    records = []

    def run(label: str, args: list[str], *, expected: int = 0, contains: str = ""):
        print(json.dumps({"phase": label, "status": "started"}), flush=True)
        start = time.monotonic()
        path = output / f"{label}.log"
        with path.open("xb") as stream:
            completed = subprocess.run(
                args, cwd=workspace, env=env, stdout=stream,
                stderr=subprocess.STDOUT, timeout=420,
            )
        data = path.read_bytes()
        completed.stdout = data.decode(errors="replace")
        row = {
            "phase": label, "exit_code": completed.returncode,
            "elapsed_seconds": time.monotonic() - start,
            "path": path.name, "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data), "arguments": args,
        }
        records.append(row)
        print(json.dumps(row), flush=True)
        if completed.returncode != expected or contains not in data.decode(errors="replace"):
            print(data.decode(errors="replace")[-12000:], flush=True)
            raise RuntimeError(f"policy_dependency_replay_phase_failed:{label}")
        return completed

    result = {
        "schema_version": SCHEMA, "status": "running", "source_commit": source_commit,
        "candidate_id": CANDIDATE, "checkpoint_downloaded": False,
        "policy_query_performed": False, "gpu_used": False, "provider_mutation_performed": False,
        "full_environment_installed": False,
        "claim_ceiling": "cpu_policy_dependency_install_only", "records": records,
    }
    try:
        uvroot = workspace / "uv"
        uvroot.mkdir()
        archive = workspace / "uv.tar.gz"
        with urlopen(UV_ARCHIVE_URL, timeout=60) as response:
            archive.write_bytes(response.read())
        if hashlib.sha256(archive.read_bytes()).hexdigest() != UV_ARCHIVE_SHA256:
            raise ValueError("policy_dependency_replay_uv_hash_mismatch")
        with tarfile.open(archive) as bundle:
            bundle.extractall(uvroot, filter="data")
        uv = next(uvroot.glob("*/uv"))
        run("uv-identity", [str(uv), "--version"], contains="uv 0.12.5")
        source = workspace / "openpi"
        frozen = EXPECTED_CANDIDATES[CANDIDATE]
        run("source-clone", ["git", "clone", "--filter=blob:none", "--no-checkout",
                             frozen["source_repository"], str(source)])
        run("source-fetch", ["git", "-C", str(source), "fetch", "--depth", "1", "origin",
                             frozen["source_revision"]])
        run("source-sparse-init", ["git", "-C", str(source), "sparse-checkout", "init", "--cone"])
        run("source-sparse-set", ["git", "-C", str(source), "sparse-checkout", "set", "src", "packages"])
        run("source-checkout", ["git", "-C", str(source), "checkout", "--detach", "FETCH_HEAD"])
        revision = run("source-identity", ["git", "-C", str(source), "rev-parse", "HEAD"])
        if revision.stdout.strip() != frozen["source_revision"]:
            raise ValueError("policy_dependency_replay_source_mismatch")
        lock = source / ACTIVE_CANDIDATE_DEPENDENCY_LOCKS[CANDIDATE]["relative_path"]
        digest = "sha256:" + hashlib.sha256(lock.read_bytes()).hexdigest()
        if digest != ACTIVE_CANDIDATE_DEPENDENCY_LOCKS[CANDIDATE]["sha256"]:
            raise ValueError("policy_dependency_replay_lock_mismatch")
        result.update(upstream_revision=frozen["source_revision"], upstream_lock_sha256=digest)
        negative, positive = workspace / "py312", workspace / "policy"
        run("negative-venv", [str(uv), "venv", "--python", sys.executable, str(negative)])
        requirement = output / "retained-mujoco-requirement.txt"
        requirement.write_text(mujoco_requirement(tomllib.loads(lock.read_text())))
        run("retained-mujoco-refusal", [str(uv), "pip", "install", "--python",
            str(negative / "bin/python"), "--require-hashes", "--no-deps", "-r", str(requirement)],
            expected=1, contains="MUJOCO_PATH environment variable is not set")
        result["retained_failure_reproduced"] = True
        # The earlier replay created the venv under clean discovery defaults.
        # Reproduce the next real refusal before trying the production repair.
        env.update(UV_PYTHON=str(negative / "bin/python"),
                   UV_PYTHON_PREFERENCE="only-system", UV_PYTHON_DOWNLOADS="automatic")
        # Match the image's system-Python search rather than a CI runner's
        # optional hosted-toolcache interpreters.
        original_path = env["PATH"]
        env["PATH"] = "/usr/bin:/bin"
        try:
            run("retained-interpreter-preference-refusal", production_venv_arguments(
                venv=workspace / "blocked-policy", uv=uv), expected=2, contains="only system")
        finally:
            env["PATH"] = original_path
        result["retained_interpreter_preference_failure_reproduced"] = True
        # Downloads=never is a stricter fixture than the observed image refusal.
        # The generated exports must defeat both restrictions, create the real
        # 3.11.9 venv, then sync without the inherited 3.12 selection replacing it.
        env.update(VIRTUAL_ENV=str(positive), UV_PYTHON_DOWNLOADS="never")
        run("full-frozen-install", ["bash", "-c", production_environment_setup(
            source=source, venv=positive, uv=uv, python_install_root=workspace / "python")])
        result["production_venv_setup_replayed"] = True
        result["full_environment_installed"] = True
        run("dependency-consistency", [str(uv), "pip", "check", "--python", str(positive / "bin/python")])
        run("installed-distributions", [str(uv), "pip", "list", "--format", "json",
                                         "--python", str(positive / "bin/python")])
        probe = run("native-imports", [str(positive / "bin/python"), "-c", """
import json, sys, importlib.metadata as metadata
import mujoco, numpy, jax, torch
from openpi.training import config
from openpi.policies import policy_config
assert sys.version_info[:3] == (3, 11, 9), sys.version
assert mujoco.mj_versionString() == '2.3.7'
assert not torch.cuda.is_available()
assert all(device.platform == 'cpu' for device in jax.devices())
print(json.dumps({'python': sys.version, 'mujoco': mujoco.mj_versionString(),
 'numpy': numpy.__version__, 'jax': jax.__version__, 'torch': torch.__version__,
 'openpi': metadata.version('openpi'), 'upstream_policy_imports': True}))
"""])
        result.update(full_environment_installed=True,
                      native_imports=json.loads(probe.stdout.strip().splitlines()[-1]))
        locked_websockets = next(p for p in tomllib.loads(lock.read_text())["package"]
                                 if p["name"] == "websockets")
        if not any(row["url"] == OPENPI_CLIENT_WEBSOCKETS_WHEEL["url"]
                   and row["hash"] == "sha256:" + OPENPI_CLIENT_WEBSOCKETS_WHEEL["sha256"]
                   and row["size"] == OPENPI_CLIENT_WEBSOCKETS_WHEEL["size_bytes"]
                   for row in locked_websockets["wheels"]):
            raise ValueError("policy_dependency_replay_client_wheel_not_in_frozen_lock")
        result["isaac_client_matrix"] = []
        for numpy_version, msgpack_version in (("2.3.1", "1.2.1"), ("2.5.1", "1.1.0")):
            label = "isaac-client-numpy-" + numpy_version
            client = workspace / label
            run(label + "-venv", [str(uv), "venv", "--python", sys.executable, str(client)])
            client_python = client / "bin/python"
            run(label + "-host-dependencies", [str(uv), "pip", "install", "--python", str(client_python),
                "--only-binary", ":all:", f"numpy=={numpy_version}", f"msgpack=={msgpack_version}",
                "typing-extensions==4.13.2"])
            runtime, client_output = workspace / (label + "-runtime"), output / label
            runtime.mkdir()
            client_output.mkdir()
            blueprint_source = Path(__file__).resolve().parents[1] / "src/blueprint_pipeline"
            for name in ("openpi_droid_policy_runtime.py", "droid_policy_bridge.py"):
                shutil.copy2(blueprint_source / name, runtime / name)
            env.update(RUNTIME_DIR=str(runtime), OUT_DIR=str(client_output))
            commands = "set -euo pipefail\n" + "\n".join(_isaac_client_commands(CANDIDATE))
            commands = commands.replace(f"{POLICY_SOURCE_ROOT}/{CANDIDATE}", str(source))
            commands = commands.replace(policy_venv_root(CANDIDATE), str(positive))
            commands = commands.replace(ISAAC_INTERPRETER, str(client_python))
            run(label + "-production-staging", ["bash", "-c", commands])
            client_receipt = client_output / "openpi_client_staging.v1.json"
            staging = json.loads(client_receipt.read_text())
            if staging["isaac_numpy_version"] != numpy_version or staging["isaac_numpy_unchanged"] is not True:
                raise ValueError("policy_dependency_replay_isaac_numpy_changed")
            request, response = workspace / (label + ".request"), workspace / (label + ".response")
            encode = """
import sys, hashlib, json, inspect
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from openpi_droid_policy_runtime import activate_staged_openpi_client_dependencies
activate_staged_openpi_client_dependencies()
import numpy as np
from openpi_client import msgpack_numpy as codec
from websockets.sync.client import connect
inspect.signature(connect).bind('ws://127.0.0.1:1', compression=None, max_size=None,
 additional_headers=None, open_timeout=30, ping_timeout=None, close_timeout=5)
image=np.arange(180*320*3,dtype=np.uint8).reshape(180,320,3)
raw=codec.packb({'exterior':image,'wrist':image.copy(),
 'state':np.arange(8,dtype=np.float64)/128,'scalar':np.float32(.25)})
Path(sys.argv[2]).write_bytes(raw)
print(json.dumps({'synthetic_wire_fixture':True,'numpy':np.__version__,
 'request_sha256':hashlib.sha256(raw).hexdigest(),'model_called':False}))
"""
            run(label + "-encode", [str(client_python), "-c", encode, str(runtime), str(request)])
            decode = """
import sys, json, hashlib, numpy as np
from pathlib import Path
from openpi_client import msgpack_numpy as codec
raw=Path(sys.argv[1]).read_bytes();decoded=codec.unpackb(raw)
assert codec.packb(decoded)==raw
assert decoded['exterior'].shape==(180,320,3)
assert decoded['state'].dtype==np.float64
response=codec.packb({'actions':np.arange(80,dtype=np.float32).reshape(10,8)/128})
Path(sys.argv[2]).write_bytes(response)
print(json.dumps({'server_numpy':np.__version__,'lossless_request_roundtrip':True,
 'request_sha256':hashlib.sha256(raw).hexdigest(),'response_sha256':hashlib.sha256(response).hexdigest()}))
"""
            run(label + "-server-codec", [str(positive / "bin/python"), "-c", decode, str(request), str(response)])
            verify = """
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from openpi_droid_policy_runtime import activate_staged_openpi_client_dependencies
activate_staged_openpi_client_dependencies()
from openpi_client import msgpack_numpy as codec
actions=codec.unpackb(Path(sys.argv[2]).read_bytes())['actions']
assert actions.dtype==np.float32 and actions.shape==(10,8)
assert np.array_equal(actions,np.arange(80,dtype=np.float32).reshape(10,8)/128)
print(json.dumps({'client_numpy':np.__version__,'lossless_action_roundtrip':True,'model_called':False}))
"""
            run(label + "-decode", [str(client_python), "-c", verify, str(runtime), str(response)])
            result["isaac_client_matrix"].append({
                "numpy_version": numpy_version, "msgpack_version": msgpack_version,
                "isaac_numpy_unchanged": True, "wire_roundtrip_passed": True,
                "staging_receipt_path": str(client_receipt.relative_to(output)),
                "staging_receipt_sha256": hashlib.sha256(client_receipt.read_bytes()).hexdigest(),
                "model_called": False,
            })
        result["status"] = "passed"
    except Exception as exc:
        result.update(status="blocked", blocker=f"{type(exc).__name__}:{exc}")
        raise
    finally:
        result["receipt_digest"] = canonical_digest(result, digest_field="receipt_digest")
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if actual != args.source_commit:
        raise ValueError("policy_dependency_replay_implementation_commit_mismatch")
    replay(args.workspace.absolute(), args.output.absolute(), source_commit=actual)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
