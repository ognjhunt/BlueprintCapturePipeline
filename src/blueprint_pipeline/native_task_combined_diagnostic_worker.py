"""Run two isolated Isaac children on one already-admitted diagnostic resource."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from blueprint_pipeline.native_task_composition_diagnostic import file_record, seal

CHILDREN = (
    (
        "composition",
        "blueprint_pipeline.native_task_composition_worker",
        "native_task_arena_runtime_preflight.v1.json",
    ),
    (
        "command_replay",
        "blueprint_pipeline.native_task_retained_command_worker",
        "native_task_retained_command_child.json",
    ),
)


def run_diagnostic_children(*, runtime_root, output_root, runner=subprocess.run):
    runtime, output = Path(runtime_root).resolve(), Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((runtime / "adp_arena_provider_manifest.json").read_text())
    result = {
        "schema_version": "native_task_arena_runtime_preflight.v1",
        "status": "blocked",
        "diagnostic_kind": "exact_cell00_composition_and_cell01_retained_commands",
        "preflight_only": True,
        "candidate_policy_queried": False,
        "candidate_outcomes_accessed": True,
        "retrospective_retained_failure_diagnostic": True,
        "new_candidate_outcomes_generated": False,
        "policy_queries": 0,
        "model_calls": 0,
        "task_motion_executed": False,
        "maximum_retained_commands": 174,
        "maximum_child_wall_seconds": 600,
        "provider_allocations_performed_by_worker": 0,
        "provider_zero_required_after_return": True,
        "implementation_commit": manifest["implementation_commit"],
        "container_image": manifest["container_image"],
        "bundle_input_digest": manifest["input_digest"],
        "children": [],
        "blockers": [],
    }
    for name, module, filename in CHILDREN:
        child = output / name
        child.mkdir(exist_ok=True)
        command = [sys.executable, "-m", module, "--runtime-root", str(runtime)]
        environment = {
            **os.environ,
            "BLUEPRINT_ADP_ARENA_OUTPUT_DIR": str(child),
            "PYTHONPATH": str(runtime),
        }
        entry = {"name": name, "module": module, "timeout_seconds": 600, "fresh_process": True}
        try:
            with (child / "process.log").open("xb") as log:
                completed = runner(
                    command,
                    cwd=runtime,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=600,
                    check=False,
                )
            entry["process_exit_code"] = completed.returncode
            entry["process_exited_before_next_child"] = True
            document = json.loads((child / filename).read_text())
            entry["result"] = file_record(child / filename)
            entry["result"]["relative_path"] = name + "/" + filename
            entry["status"] = document.get("status")
            entry["task_motion_executed"] = document.get("task_motion_executed") is True
            result["task_motion_executed"] |= entry["task_motion_executed"]
            if document.get("candidate_policy_queried") is not False:
                raise ValueError("diagnostic_child_policy_query_boundary_invalid")
            if completed.returncode or document.get("status") != "completed":
                result["blockers"].append(name + ":child_not_completed")
        except Exception as exc:
            entry["status"] = "blocked"
            entry["failure_type"] = type(exc).__name__
            result["blockers"].append(name + ":" + type(exc).__name__)
            progress = child / "replay/replay_progress.json"
            if name == "command_replay":
                observed = json.loads(progress.read_text()) if progress.is_file() else {}
                entry["task_motion_executed"] = (
                    True if observed.get("executed_commands", 0) > 0 else None
                )
                result["task_motion_executed"] = entry["task_motion_executed"]
            # A timeout or process-control failure cannot establish safe CUDA
            # close/rebuild. Do not start another child after that boundary.
            result["children"].append(entry)
            break
        result["children"].append(entry)
    if len(result["children"]) == 2 and not result["blockers"]:
        result["status"] = "completed"
    seal(result, "result_digest")
    (output / "native_task_arena_runtime_preflight.v1.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def main():
    runtime = Path(__file__).resolve().parent
    manifest = json.loads((runtime / "adp_arena_provider_manifest.json").read_text())
    if file_record(__file__)["sha256"] != manifest["worker_source_sha256"]:
        raise ValueError("combined_diagnostic_worker_identity_invalid")
    result = run_diagnostic_children(
        runtime_root=runtime,
        output_root=Path(
            os.environ.get("BLUEPRINT_ADP_ARENA_OUTPUT_DIR", runtime / "runtime_output")
        ),
    )
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
