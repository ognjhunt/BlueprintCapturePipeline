"""Seal the provider bundle that records one prepared rigid scene's robot-free probe; rent nothing.

Beta journey step: L5 team policy evaluation. The bundle is the existing
``native_task_arena`` kind in execution mode ``rigid_robot_free_probe``: the
scene's own compiled packet (``team_scene_packet``) byte for byte, the
probe-only worker (``rigid_robot_free_probe_worker``) with exactly its import
closure (``RIGID_ROBOT_FREE_PROBE_RUNTIME_MODULE_NAMES``), and one bound input
that carries:

- the control plane's probe request (``team_robot_free_probe.probe_request``),
  planned and bound against the task specification teams are scored with, the
  scene's under the job's success contract (``scored_task_spec``), which is
  what ``scene_facts(profile)`` returns for a rigid scene; and
- the packet's own scene plan resolved for its nominal cell
  (``team_candidate_quick10_registry.NOMINAL_CELL``), which the worker builds
  exactly as a team session does.

The older ``rigid_robot_free_probe_isaac.prepare_isaac_probe`` binds the
configured revision's specification instead; a receipt from it would refuse
every team with ``task_spec_mismatch``, so it is not used here.

``current_probe_state`` is the one read both the queue and the paid lane make
of a capture's profile: it plans the request and reads the retained receipt,
and spends nothing.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .native_task_arena_execution_contract import (
    NATIVE_METROLOGY_RUNTIME_PACKAGE_FILES,
    RIGID_ROBOT_FREE_PROBE_RUNTIME_MODULE_NAMES,
)
from .rigid_robot_free_probe_worker import (
    EXECUTION_MODE,
    INPUTS_FILENAME,
    INPUTS_SCHEMA,
    RESULT_FILENAME,
    RigidRobotFreeProbeWorkerError,
    validate_probe_inputs,
)

PROBE_KIND = "rigid-robot-free-probe"
BUNDLE_RECEIPT_FILENAME = "native_task_arena_provider_bundle_receipt.v1.json"
PACKET_PLAN_FILENAME = "native_task_arena_scene_plan.v1.json"
PACKET_RECEIPT_FILENAME = "native_task_arena_packet_receipt.v1.json"
WORKER_MODULE = "rigid_robot_free_probe_worker.py"


class RigidRobotFreeProbeBundleError(ValueError):
    """The probe cannot be sealed for these inputs; nothing was rented."""


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RigidRobotFreeProbeBundleError("rigid_robot_free_probe_record_invalid")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def current_probe_state(capture_root: str | Path) -> dict[str, Any]:
    """What the capture's current team profile asks of the probe now; reads only, spends nothing.

    ``status`` is ``probe_required`` (with the planned request), or why no run is
    wanted: no profile, no probe for this task, an unbound revision, or a
    retained receipt that is already decided (never replaced, so a run cannot
    change it).
    """

    from .robot_free_probe_gate import RobotFreeProbeGateError, retained_probe_receipt
    from .robot_free_readiness_retention import DECIDED_STATUSES
    from .team_candidate_quick10_registry import profile_for_capture, scene_facts
    from .team_robot_free_probe import probe_request

    root = Path(capture_root).resolve()
    try:
        profile = profile_for_capture(root)
    except (OSError, ValueError) as exc:
        return {"status": "profile_unavailable", "capture_root": str(root), "reason": str(exc)[:200]}
    if profile is None:
        return {"status": "profile_missing", "capture_root": str(root)}
    try:
        facts = scene_facts(profile)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        return {"status": "scene_facts_unavailable", "capture_root": str(root), "reason": str(exc)[:200]}
    scene = facts["scene"]
    revision, task_spec = scene.get("scene_revision_digest"), scene.get("task_spec")
    if not isinstance(revision, str) or not revision.startswith("sha256:") or len(revision) != 71:
        return {"status": "scene_revision_unbound", "capture_root": str(root)}
    request = probe_request(task_spec=task_spec, scene_revision_digest=revision) if isinstance(
        task_spec, Mapping) else None
    if request is None:
        return {"status": "unavailable_for_task", "capture_root": str(root),
                "task_kind": (task_spec or {}).get("task_kind") if isinstance(task_spec, Mapping) else None}
    probe_root = Path(str(profile.get("robot_free_probe_root") or ""))
    if not probe_root.is_absolute() or not probe_root.is_dir():
        return {"status": "probe_root_unavailable", "capture_root": str(root)}
    try:
        receipt = retained_probe_receipt(probe_root)
    except (OSError, RobotFreeProbeGateError) as exc:
        return {"status": "receipt_unreadable", "capture_root": str(root), "reason": str(exc)[:200]}
    state = {"capture_root": str(root), "robot_free_probe_root": str(probe_root.resolve()), "profile": profile,
             "facts": facts, "request": request, "receipt": receipt,
             "receipt_status": receipt.get("status") if isinstance(receipt, Mapping) else None}
    decided = isinstance(receipt, Mapping) and receipt.get("status") in DECIDED_STATUSES
    return {**state, "status": "receipt_decided" if decided else "probe_required"}


def probe_inputs(*, state: Mapping[str, Any]) -> dict[str, Any]:
    """The sealed input the GPU host executes: the planned request and the scene's nominal plan."""

    from .policy_canary_scene_resolution import resolved_scene_plan
    from .team_candidate_quick10_registry import NOMINAL_CELL
    from .team_robot_free_probe import scored_task_spec

    facts, request = state["facts"], state["request"]
    packet_dir = Path(facts["packet_dir"])
    plan = _read(packet_dir / PACKET_PLAN_FILENAME)
    receipt = _read(packet_dir / PACKET_RECEIPT_FILENAME)
    contract = facts["inputs"]["task_success_contract"]
    # The request must be planned against this packet's own task under the job's
    # contract: the specification a team is scored with, never the revision's.
    if request["binding"]["task_spec_digest"] != canonical_digest(scored_task_spec(plan["task_spec"], contract)):
        raise RigidRobotFreeProbeBundleError("rigid_robot_free_probe_request_not_bound_to_scored_spec")
    nominal = resolved_scene_plan(
        plan, {**NOMINAL_CELL, "control_diagnostic": facts["inputs"]["control_diagnostic"]},
        task_success_contract=contract)
    value = {"schema_version": INPUTS_SCHEMA, "capture_root": str(state["capture_root"]),
             "robot_free_probe_root": str(state["robot_free_probe_root"]),
             "scene_revision_digest": facts["scene"]["scene_revision_digest"], "probe_request": dict(request),
             "scene_plan": nominal, "base_plan_digest": plan.get("plan_digest"),
             "packet_receipt_digest": receipt.get("receipt_digest"),
             "task_success_contract_digest": contract.get("contract_digest"), "inputs_digest": ""}
    value["inputs_digest"] = canonical_digest(value, digest_field="inputs_digest")
    try:
        return validate_probe_inputs(value)
    except RigidRobotFreeProbeWorkerError as exc:
        raise RigidRobotFreeProbeBundleError(str(exc)) from exc


def build_rigid_robot_free_probe_bundle(*, job_dir: str | Path, packet_dir: str | Path,
                                        runtime_source_packet_receipt: str | Path, inputs_path: str | Path,
                                        implementation_commit: str, generated_at: str | None = None
                                        ) -> dict[str, Any]:
    """Package the probe worker, its closure, the packet and the one bound input (no policy, no secret)."""

    from .native_task_arena_bundle import build_native_task_arena_bundle
    from .native_task_isaaclab_launch import NATIVE_TASK_ARENA_IMAGE
    from .provider_runtime_import_closure import assert_provider_runtime_import_closure

    package = Path(__file__).resolve().parent
    assert_provider_runtime_import_closure(
        package_source_dir=package, shipped_module_names=RIGID_ROBOT_FREE_PROBE_RUNTIME_MODULE_NAMES,
        shipped_package_files=NATIVE_METROLOGY_RUNTIME_PACKAGE_FILES,
        code="rigid_robot_free_probe_bundle_import_closure_incomplete")
    inputs = validate_probe_inputs(_read(Path(inputs_path)))
    packet_receipt = _read(Path(packet_dir) / PACKET_RECEIPT_FILENAME)
    if inputs["packet_receipt_digest"] != packet_receipt.get("receipt_digest"):
        raise RigidRobotFreeProbeBundleError("rigid_robot_free_probe_packet_binding_invalid")
    return build_native_task_arena_bundle(
        job_dir=job_dir, packet_dir=packet_dir, worker_source=package / WORKER_MODULE,
        runtime_module_sources=[package / name for name in RIGID_ROBOT_FREE_PROBE_RUNTIME_MODULE_NAMES],
        implementation_commit=implementation_commit, execution_mode=EXECUTION_MODE,
        expected_output_filename=RESULT_FILENAME, container_image=NATIVE_TASK_ARENA_IMAGE,
        runtime_source_packet_receipt=runtime_source_packet_receipt,
        bound_runtime_inputs={INPUTS_FILENAME: inputs_path}, generated_at=generated_at)


def load_verified_rigid_robot_free_probe_bundle(receipt_path: str | Path, *,
                                                expected_implementation_commit: str) -> dict[str, Any]:
    """Reverify a sealed probe bundle's receipt and bytes before the paid lane uses it."""

    from .native_task_isaaclab_launch import NATIVE_TASK_ARENA_IMAGE

    try:
        receipt = _read(Path(receipt_path).expanduser().resolve())
    except (OSError, ValueError) as exc:
        raise RigidRobotFreeProbeBundleError("rigid_robot_free_probe_bundle_receipt_invalid") from exc
    bundle = Path(str(receipt.get("bundle_path") or "")).expanduser().resolve()
    manifest = {key: value for key, value in receipt.items()
                if key not in {"bundle_path", "bundle_size_bytes", "bundle_sha256"}}
    errors: list[str] = []
    if (receipt.get("schema_version") != "native_task_arena_provider_bundle.v1" or receipt.get("status") != "ready"
            or receipt.get("execution_mode") != EXECUTION_MODE
            or receipt.get("expected_output_filename") != RESULT_FILENAME
            or receipt.get("policy_candidate_id") is not None or receipt.get("candidate_policy_queried") is not False
            or receipt.get("program_id") != "blueprint-free-beta-evaluated-pilot"):
        errors.append("rigid_robot_free_probe_bundle_contract_invalid")
    if {Path(str((row or {}).get("relative_path") or "")).name for row in receipt.get("runtime_modules") or []} != set(
            RIGID_ROBOT_FREE_PROBE_RUNTIME_MODULE_NAMES):
        errors.append("rigid_robot_free_probe_bundle_modules_invalid")
    rows = [row for row in receipt.get("bound_runtime_inputs") or [] if isinstance(row, Mapping)]
    if [row.get("relative_path") for row in rows] != [f"runtime_inputs/{INPUTS_FILENAME}"]:
        errors.append("rigid_robot_free_probe_bundle_inputs_invalid")
    if receipt.get("implementation_commit") != expected_implementation_commit:
        errors.append("rigid_robot_free_probe_bundle_commit_mismatch")
    if receipt.get("container_image") != NATIVE_TASK_ARENA_IMAGE:
        errors.append("rigid_robot_free_probe_bundle_image_mismatch")
    if receipt.get("input_digest") != canonical_digest(manifest, digest_field="input_digest"):
        errors.append("rigid_robot_free_probe_bundle_input_digest_invalid")
    if (not bundle.is_file() or receipt.get("bundle_size_bytes") != bundle.stat().st_size
            or receipt.get("bundle_sha256") != _sha256(bundle)):
        errors.append("rigid_robot_free_probe_bundle_identity_invalid")
    if errors:
        raise RigidRobotFreeProbeBundleError(";".join(sorted(set(errors))))
    return receipt


def main(argv: list[str] | None = None) -> int:
    """Rebuild a probe bundle from its retained inputs at an explicit release commit."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("job-dir", "packet-dir", "runtime-source-packet-receipt", "inputs-path"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--generated-at")
    args = parser.parse_args(argv)
    receipt = build_rigid_robot_free_probe_bundle(
        job_dir=args.job_dir, packet_dir=args.packet_dir,
        runtime_source_packet_receipt=args.runtime_source_packet_receipt,
        inputs_path=args.inputs_path, implementation_commit=args.implementation_commit,
        generated_at=args.generated_at)
    print(json.dumps({name: receipt[name] for name in
                      ("bundle_path", "bundle_sha256", "implementation_commit")}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "BUNDLE_RECEIPT_FILENAME",
    "PROBE_KIND",
    "RigidRobotFreeProbeBundleError",
    "build_rigid_robot_free_probe_bundle",
    "current_probe_state",
    "load_verified_rigid_robot_free_probe_bundle",
    "probe_inputs",
]
