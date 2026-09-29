"""Stage handlers the remote CPU worker tests run inside a real stage child (plan 14 PR 3).

The worker tests ship this file in the fixture release as ``src/remote_cpu_worker_stages.py``, so each
handler runs from the extracted archive in a spawned interpreter, exactly as a registered stage would.
A handler takes the sealed descriptor and the stage roots and returns the stage's sealed result.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path
from typing import Any

from blueprint_pipeline.decision_evidence_contracts import canonical_digest

NEW_BYTES = b'{"adapter": "result"}\n'


def sealed_result(descriptor: dict[str, Any], *, blockers: list[str], **extra: Any) -> dict[str, Any]:
    result = {"schema_version": "task_evaluation_episode_compilation_result.v1",
              "status": "blocked" if blockers else "compiled_for_production_launch",
              "compilation_id": descriptor["outputs"]["output_root"].rsplit("/", 1)[-1],
              "source_commit": descriptor["code"]["source_commit"], "provider_mutation_performed": False,
              "paid_execution_requested": False, "blockers": blockers, **extra, "result_digest": ""}
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    return result


def _identity() -> dict[str, Any]:
    import blueprint_pipeline

    return {"pid": os.getpid(), "ppid": os.getppid(), "argv": sys.argv[1:], "pytest_loaded": "pytest" in sys.modules,
            "package_file": blueprint_pipeline.__file__, "working_directory": os.getcwd(),
            "environment_names": sorted(os.environ),
            "presigned_in_environment": any("X-Amz-" in value for value in os.environ.values())}


def compiled(descriptor: dict[str, Any], roots: Any) -> dict[str, Any]:
    """Write new bytes, a copy of an input and a bundle member under the output root, and a scratch member."""

    output = roots.local(descriptor["outputs"]["output_root"])
    adapter = output / "native-arena-adapter"
    adapter.mkdir(parents=True)
    (adapter / "result.json").write_bytes(NEW_BYTES)
    inputs = {Path(item["materialize_at"]).name: roots.local(item["materialize_at"]) for item in descriptor["inputs"]}
    (output / "robot.json").write_bytes(inputs["robot.json"].read_bytes())
    with zipfile.ZipFile(inputs["runtime.zip"]) as bundle:
        member = bundle.read("runtime/model.bin")
    (adapter / "model.bin").write_bytes(member)
    scratch = roots.local(descriptor["outputs"]["declared_scratch"][0]) / "adapter-members"
    scratch.mkdir(parents=True, exist_ok=True)
    (scratch / "model.bin").write_bytes(member)
    return sealed_result(descriptor, blockers=[], identity=_identity())


def slow_compiled(descriptor: dict[str, Any], roots: Any) -> dict[str, Any]:
    time.sleep(3.5)
    return compiled(descriptor, roots)


def hangs(descriptor: dict[str, Any], roots: Any) -> dict[str, Any]:
    """Record this child and a grandchild in its session, then never return."""

    grandchild = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    marker = Path(os.environ["REMOTE_CPU_TEST_PIDS"])
    marker.write_text(f"{os.getpid()} {grandchild.pid}\n", encoding="utf-8")
    threading.Event().wait()
    raise AssertionError("unreachable")


def reads_success_schema(descriptor: dict[str, Any], roots: Any) -> dict[str, Any]:
    """Read a schema through ``parents[2]``, as the compile does, and turn an unreadable one into a
    deterministic blocked result, as the host's envelope load does (plan 14 Facts)."""

    from blueprint_pipeline.rigid_task_success_contract_schema import rigid_task_success_contract_schema

    try:
        rigid_task_success_contract_schema()
        blocker = "episode_compilation_envelope_invalid"
    except OSError:
        blocker = "rigid_task_success_contract_schema_unavailable"
    return sealed_result(descriptor, blockers=[blocker], automatic_retry_performed=False)
