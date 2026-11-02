"""Replay a finished ARTISAN run's export and stage-4 qualification locally, at $0.

A stage-3 bug found on the GPU costs about an hour: a deploy, preparation,
activation, a GPU boot and a 6 GB bundle download all come before ARTISAN.
capture-447ae4f4 (2026-11-02) found three in a row that way. This replays the
glue the paid driver runs after ARTISAN finishes — ``package_upstream_candidate``
(export, USD adaptation, sealed receipt), ``verify_artisan_candidate`` and
stage 4's ``qualify_artisan_asset_static`` — on a real run's retained files,
with no model call, no provider and no spend.

    python -m blueprint_pipeline.task_object_artisan_replay --launch <launch dir name> --out <dir>
    python -m blueprint_pipeline.task_object_artisan_replay --producer <local producer dir> --out <dir>

``--launch`` pulls, read-only through the operator door, only the files the
export reads into ``<out>/producer`` (the launch's
``immutable_execution/stages/stage-3/producer``). The run's absolute GPU paths
are rebased onto that local copy for the replay only; nothing is written back.
Source frames never leave the host: the export does not read them.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable

GPU_PRODUCER_ROOT = ("/workspace/task_evaluation_scene_configuration_provider_bundle/runtime_output/"
                     "stages/stage-3/producer")
HOST_LAUNCH_ROOT = "/var/lib/blueprint/pipeline-control-plane/task-evaluation-launch-runs"
PRODUCER_RELATIVE = "allocator/scene-configuration-job/immutable_execution/stages/stage-3/producer"
TERMINAL = "artisan_runtime/terminal.v1.json"
REQUEST = "artisan_runtime/blueprint_request.v1.json"
#: Read by the export but optional upstream (absent on capture-447's tray).
OPTIONAL = ("assets/blueprint_candidate_annotations.v1.json",)


def rebase_terminal(terminal: dict[str, Any], *, original_root: str, local_root: str) -> dict[str, Any]:
    """The terminal record with every path moved from the GPU root onto a local copy."""

    original_root, local_root = original_root.rstrip("/"), local_root.rstrip("/")
    paths = {}
    for key, value in terminal["paths"].items():
        if not (value == original_root or value.startswith(original_root + "/")):
            raise ValueError(f"artisan_replay_path_outside_producer:{key}")
        paths[key] = local_root + value[len(original_root):]
    return {**terminal, "paths": paths}


def _workspace_relative(terminal: dict[str, Any], key: str) -> str:
    return terminal["paths"][key][len(GPU_PRODUCER_ROOT) + 1:]


def export_inputs(terminal: dict[str, Any], manifest: dict[str, Any]) -> list[str]:
    """Producer-relative files the export reads, from the terminal and its export manifest."""

    exports = _workspace_relative(terminal, "exports")
    files = [exports + "/export_manifest.json"]
    files += [exports + "/" + row["path"] for row in manifest["artifacts"].values()]
    files.append(exports + "/" + manifest["physics_report_path"])
    files += [_workspace_relative(terminal, key) for key in ("candidate_manifest", "model", "blueprint",
                                                             "physics_report")]
    return list(dict.fromkeys(files))


def _door_pull(host_path: str, local_path: Path) -> bool:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    door = Path(__file__).resolve().parents[2] / "scripts/operator_door.py"
    done = subprocess.run([sys.executable, str(door), "pull", host_path, str(local_path)],
                          capture_output=True, text=True, check=False)
    return done.returncode == 0 and local_path.is_file()


def pull_launch(launch: str, producer: Path, *, pull: Callable[[str, Path], bool] = _door_pull) -> list[str]:
    """Copy one launch's stage-3 export inputs into ``producer``; return what was missing."""

    host = f"{HOST_LAUNCH_ROOT}/{launch}/{PRODUCER_RELATIVE}"
    for name in (TERMINAL, REQUEST):
        if not pull(f"{host}/{name}", producer / name):
            raise ValueError(f"artisan_replay_pull_failed:{name}")
    terminal = json.loads((producer / TERMINAL).read_text())
    manifest_name = _workspace_relative(terminal, "exports") + "/export_manifest.json"
    if not pull(f"{host}/{manifest_name}", producer / manifest_name):
        raise ValueError(f"artisan_replay_pull_failed:{manifest_name}")
    manifest = json.loads((producer / manifest_name).read_text())
    missing = [name for name in export_inputs(terminal, manifest)
               if not (producer / name).is_file() and not pull(f"{host}/{name}", producer / name)]
    root = _workspace_relative(terminal, "root")
    for name in OPTIONAL:
        pull(f"{host}/{root}/{name}", producer / root / name)
    return missing


@contextmanager
def _source_inputs_stay_on_host():
    """Check only the request's digest, not its source frames and geometry.

    Those are the site's own images; they stay on the host, and the GPU run
    re-verifies them byte for byte. Everything the export reads is checked.
    """
    from . import task_object_artisan_candidate as candidate
    from .decision_evidence_contracts import canonical_digest
    from .task_object_authoring_contracts import require

    def digest_only(request):
        require(request.get("request_digest") == canonical_digest(request, digest_field="request_digest"),
                "authoring_request_changed")
    original = candidate.verify_construction_request
    candidate.verify_construction_request = digest_only
    try:
        yield
    finally:
        candidate.verify_construction_request = original


def replay(producer: Path, out: Path) -> dict[str, Any]:
    """Run the paid driver's post-ARTISAN glue and stage 4 on a local producer copy."""

    with _source_inputs_stay_on_host():
        report = _replay(producer, out)
    report["source_inputs"] = "request digest checked; frames and source geometry stay on the host"
    (Path(out).resolve() / "replay_report.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    return report


def _replay(producer: Path, out: Path) -> dict[str, Any]:
    from .task_object_artisan_candidate import (
        package_upstream_candidate, qualify_artisan_asset_static, verify_artisan_candidate,
    )
    from .task_object_authoring_contracts import read_json

    producer, out = Path(producer).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    steps: list[dict[str, Any]] = []

    def step(name: str, action: Callable[[], Any]) -> Any:
        try:
            value = action()
        except Exception as exc:  # noqa: BLE001 - the replay reports the first stop, typed
            steps.append({"step": name, "status": "failed", "exception_type": type(exc).__name__,
                          "reason": str(exc)[:2000]})
            raise _Stop() from exc
        steps.append({"step": name, "status": "passed"})
        return value

    try:
        terminal = step("read_terminal", lambda: read_json(producer / TERMINAL))
        request_path = producer / REQUEST
        request = step("read_request", lambda: read_json(request_path))
        local = step("rebase_paths", lambda: rebase_terminal(
            terminal, original_root=GPU_PRODUCER_ROOT, local_root=str(producer)))
        configuration = request["configuration"]
        kind = ("articulated_assembly" if str(configuration.get("schema_version", "")).startswith("articulated_")
                else "rigid_body")
        asset, receipt_path, graph_path = step("package_upstream_candidate", lambda: package_upstream_candidate(
            terminal=local, workspace=producer / "artisan_runtime/workspace", request=request,
            request_path=request_path, candidates=out / "candidates", configuration=configuration,
            object_kind=kind, packaged={"backend": "artisan_v1", "replay": True}))
        receipt, graph = read_json(receipt_path), read_json(graph_path)
        step("verify_artisan_candidate", lambda: verify_artisan_candidate(
            receipt=receipt, graph=graph, asset=asset, configuration=configuration,
            source_candidate_digest=request["source_candidate"]["digest"]))
        qualification = step("stage4_static_qualification", lambda: qualify_artisan_asset_static(
            asset_path=asset, graph_spec=graph, authoring_receipt=receipt,
            replacement_identity=configuration["replacement_identity"],
            output_path=out / "static_qualification_receipt.v1.json"))
        structure = qualification.get("observed_structure") or {}
        report = {"status": "passed", "steps": steps, "asset": str(asset),
                  "dimensions_m": structure.get("collision_dimensions_m") or structure.get("dimensions_m"),
                  "mass_kg": structure.get("mass_kg")}
    except _Stop:
        failed = steps[-1]
        report = {"status": "failed", "steps": steps, "first_failure": failed}
    return report


class _Stop(Exception):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--launch", help="launch directory name under task-evaluation-launch-runs")
    source.add_argument("--producer", type=Path, help="a local copy of the stage-3 producer directory")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    producer = args.producer
    if args.launch:
        producer = args.out / "producer"
        missing = pull_launch(args.launch, producer)
        if missing:
            print(json.dumps({"status": "failed", "missing_inputs": missing}, indent=2))
            return 1
    report = replay(producer, args.out / "replay")
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
