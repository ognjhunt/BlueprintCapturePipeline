"""Trusted entrypoint inside the existing kernel sandbox; no credentials/network.

Uses pinned upstream functions and its explicit dataclass codec. Never unpickles
generated data or runs the ARTISAN model controller inside this worker.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


def main() -> int:
    request = json.loads(Path(sys.argv[1]).read_text())
    context = request["context"]
    root = Path(context["write_root"]).resolve()
    result = Path(request["result"])
    if not result.resolve().is_relative_to(root):
        raise ValueError("artisan_sandbox_result_outside_workspace")
    os.environ.pop("BLUEPRINT_ARTISAN_SANDBOX_CONTEXT", None)
    os.environ["ARTISAN_CACHE_DIR"] = str(root / "cache")
    os.environ["ARTISAN_BLENDER_BIN"] = context["blender"]
    operation, payload = request["operation"], request["payload"]
    try:
        if operation in {"load_model", "authored_tests"}:
            from pipeline.compiler import load_model_globals
            from sdk._core.v0.assets import activate_asset_session, asset_session_for_script
            from sdk._core.v0.motion_engine import frozen_build as codec
            from sdk._core.v0._testing.common import TestReport, TestFailure, AllowedOverlap
            original = codec._model_types
            codec._model_types = lambda: {**original(), **{c.__name__: c for c in (TestReport, TestFailure, AllowedOverlap)}}
            source = Path(payload["model_path"])
            if not source.resolve().is_relative_to(root):
                raise ValueError("artisan_model_outside_workspace")
            with activate_asset_session(asset_session_for_script(source)):
                namespace = load_model_globals(source, sdk_package=payload["sdk_package"])
                if operation == "authored_tests":
                    value = codec._encode(namespace["run_tests"]())
                else:
                    value = {"object_model": codec._encode(namespace["object_model"]),
                             "has_tests": callable(namespace.get("run_tests"))}
        elif operation == "qa":
            from sdk.runtime.runtime import run_qa_tests
            value = run_qa_tests(Path(payload.pop("model_path")), **payload)
        elif operation == "blender":
            from sdk.runtime.blender_author import run_script
            value = run_script(**payload)
        elif operation == "subprocess":
            argv = payload.pop("argv")
            directory = Path(payload.pop("cwd", None) or root).resolve()
            if not directory.is_relative_to(root):
                raise ValueError("artisan_subprocess_cwd_outside_workspace")
            completed = subprocess.run(argv, cwd=directory, **payload)
            value = {"returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}
        else:
            raise ValueError("artisan_sandbox_operation_unknown")
        response = {"ok": True, "value": value}
    except Exception as exc:
        response = {"ok": False, "error_type": type(exc).__name__, "error_message": str(exc)[:2000]}
    result.write_text(json.dumps(response, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
