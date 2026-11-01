"""Small OS-sandbox hooks for pinned upstream model/test/Blender boundaries.

Upstream still owns construction, simulation, review, repair and serialization.
The controller never receives Python objects pickled by generated code.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
from urllib.parse import urlsplit

from .asset_authoring_sandbox import SandboxedAssetRunner

CONTEXT_ENV = "BLUEPRINT_ARTISAN_SANDBOX_CONTEXT"


def enabled() -> bool:
    return bool(os.environ.get(CONTEXT_ENV))


def context() -> dict:
    value = json.loads(os.environ[CONTEXT_ENV])
    return value


def sandbox(context_value: dict) -> SandboxedAssetRunner:
    return SandboxedAssetRunner(read_roots=[Path(p) for p in context_value["read_roots"]],
        write_root=Path(context_value["write_root"]),
        write_paths=None if context_value.get("write_paths") is None else [Path(p) for p in context_value["write_paths"]],
        # The interpreter's own loader path (Kit's Python finds libpython through it).
        library_environment=context_value.get("library_environment") or {},
        library_executables=[Path(p) for p in context_value.get("library_executables") or []])


def prepare_workspace(root: Path) -> None:
    if not enabled():
        return
    value = context()
    if Path(root).resolve() != Path(value["write_root"]).resolve():
        raise ValueError("artisan_sandbox_workspace_identity_changed")
    for path in value["write_paths"]:
        Path(path).mkdir(parents=True, exist_ok=True)
    sandbox(value).preflight()


def _rpc(operation: str, **payload):
    value = context()
    root = Path(value["write_root"])
    ipc = root / "blueprint_sandbox_ipc"
    ipc.mkdir(exist_ok=True)
    identity = uuid.uuid4().hex
    request, result = ipc / (identity + ".request.json"), ipc / (identity + ".result.json")
    request.write_text(json.dumps({"operation": operation, "payload": payload,
        "context": value, "result": str(result)}))
    runner = sandbox(value)
    runner.preflight()
    completed = runner([str(Path(sys.executable).resolve()), "-S", "-m", "blueprint_pipeline.task_object_artisan_sandbox",
                        str(request)], cwd=root, env={"PYTHONPATH": os.pathsep.join(value["import_roots"])},
                       timeout=min(900, float(payload.get("timeout", 900))), check=False)
    if completed.returncode != 0 or not result.is_file() or result.is_symlink():
        raise RuntimeError("artisan_sandbox_worker_failed:" + str(completed.returncode))
    response = json.loads(result.read_text())
    if not response.get("ok"):
        raise RuntimeError("artisan_sandbox_operation_failed:" + response.get("error_type", "unknown")
                           + ":" + response.get("error_message", "")[:2000])
    return response["value"]


def _decode(value, root: Path):
    # Upstream's explicit SDK dataclass codec, extended only with its own QA types.
    from sdk._core.v0.motion_engine import frozen_build as codec
    from sdk._core.v0._testing.common import TestReport, TestFailure, AllowedOverlap
    original = codec._model_types
    codec._model_types = lambda: {**original(), **{c.__name__: c for c in (TestReport, TestFailure, AllowedOverlap)}}
    try:
        trusted_assets = [Path(path).resolve() for path in context().get("trusted_asset_roots", [])]
        def check_path(value):
            if urlsplit(str(value)).scheme:
                raise ValueError("artisan_generated_external_asset_uri")
            path = (root / value).resolve()
            if not path.is_relative_to(root.resolve()) and not any(path.is_relative_to(p) for p in trusted_assets):
                raise ValueError("artisan_generated_asset_outside_workspace")
        def check(node):
            if isinstance(node, dict):
                if node.get("type") == "Path":
                    check_path(node["value"])
                fields = node.get("fields") or {}
                if fields.get("filename"):
                    check_path(fields["filename"])
                for child in node.values():
                    check(child)
            elif isinstance(node, list):
                for child in node:
                    check(child)
        check(value)
        return codec._decode(value, root)
    finally:
        codec._model_types = original


def load_model_globals(script_path: Path, *, sdk_package="sdk") -> dict:
    source = Path(script_path).resolve()
    value = _rpc("load_model", model_path=str(source), sdk_package=sdk_package)
    model = _decode(value["object_model"], source.parent)
    result = {"object_model": model, "__file__": str(source)}
    if value["has_tests"]:
        result["run_tests"] = lambda: _decode(_rpc("authored_tests", model_path=str(source),
                                                     sdk_package=sdk_package), source.parent)
    return result


def run_qa_tests(model_path: Path, test_code: str, *, sdk_package="sdk", preserve_existing_meshes=False):
    return _rpc("qa", model_path=str(model_path), test_code=test_code,
                sdk_package=sdk_package, preserve_existing_meshes=preserve_existing_meshes)


def run_blender_script(script_path, out_name, workspace, *, visual_budget, timeout):
    return _rpc("blender", script_path=str(script_path), out_name=out_name,
                workspace=str(workspace), visual_budget=visual_budget, timeout=timeout)


def sandbox_subprocess_run(argv, **kwargs):
    if not enabled():
        return subprocess.run(argv, **kwargs)
    if set(kwargs) - {"input", "capture_output", "text", "timeout", "check", "cwd"}:
        raise ValueError("artisan_subprocess_option_not_admitted")
    if kwargs.get("text", True) is not True or kwargs.get("capture_output", True) is not True:
        raise ValueError("artisan_subprocess_protocol_invalid")
    value = _rpc("subprocess", argv=list(map(str, argv)), **kwargs)
    result = subprocess.CompletedProcess(argv, value["returncode"], value["stdout"], value["stderr"])
    if kwargs.get("check"):
        result.check_returncode()
    return result
