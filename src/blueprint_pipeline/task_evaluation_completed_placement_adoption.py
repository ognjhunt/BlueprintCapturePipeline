"""Reuse a completed analytic/visual placement when repairing native phase wiring."""

from __future__ import annotations

from .task_evaluation_retained_controls_evidence import (
    _placement_require as require, _placement_ref as read_ref,
    validate_placement_adoption as validate_adoption,
    validate_placement_cancellation as validate_cancellation,
)

import os
import hashlib
from pathlib import Path
from typing import Any, Mapping

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_retained_controls_evidence import _file, _read

SCHEMA = "task_evaluation_completed_placement_adoption.v1"
CANCELLATION_SCHEMA = "task_evaluation_unused_native_plan_cancellation.v1"


def checkpoint_reference(*, result: Mapping[str, Any], binding: Path, token: str) -> dict:
    """An adopted placement keeps its original checkpoint, not a new model call."""
    inherited = result.get("completed_placement_adoption")
    if inherited is not None:
        validate_adoption(inherited)
        require(result.get("placement_calls_reexecuted") is False, "checkpoint_lineage_invalid")
        reference = dict(inherited["source_agent_checkpoint"])
        read_ref(reference)
        return reference
    return _file(binding / f"agent-placement-checkpoint-{token}.v1.json")


def materialize_legacy_checkpoint_alias(*, intent_path: Path, binding_root: Path) -> dict:
    """Backfill the byte-identical reference expected by pre-lineage readers.

    This compatibility operation never invents a checkpoint or changes a model
    receipt. New discovery follows the original reference directly.
    """
    from . import task_evaluation_configured_controls_autostart as auto

    intent = auto.validate_configured_controls_autostart_intent(_read(intent_path))
    result_path = auto._autostart_result_path(
        root=binding_root, intent_digest=intent["intent_digest"]
    )
    result = _read(result_path)
    auto._validate_result(
        result,
        expected_intent_digest=intent["intent_digest"],
        expected_scene_binding_digest=result["scene_binding_digest"],
        expected_task_binding_digest=result["task_binding_digest"],
        expected_cpu_checkpoint_binding_digest=result["cpu_placement_checkpoint_binding_digest"],
    )
    inherited = result.get("completed_placement_adoption")
    require(
        inherited is not None and inherited == intent.get("completed_placement_adoption"),
        "checkpoint_lineage_invalid",
    )
    require(
        binding_root.name == "cpu-robot-binding"
        and binding_root.parent.name == inherited["source_launch_id"],
        "checkpoint_root_invalid",
    )
    token = intent["intent_digest"].removeprefix("sha256:")[:16]
    reference = checkpoint_reference(result=result, binding=binding_root, token=token)
    target = binding_root / f"agent-placement-checkpoint-{token}.v1.json"
    raw = Path(reference["path"]).read_bytes()
    require(
        "sha256:" + hashlib.sha256(raw).hexdigest() == reference["digest"],
        "checkpoint_source_changed",
    )
    created = False
    try:
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o440)
    except FileExistsError:
        require(_file(target)["digest"] == reference["digest"], "checkpoint_alias_conflict")
    else:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        created = True
    require(_file(target)["digest"] == reference["digest"], "checkpoint_alias_changed")
    return {
        "schema_version": "completed_placement_checkpoint_alias.v1",
        "status": "materialized" if created else "already_present",
        "source": reference,
        "target": _file(target),
        "checkpoint_bytes_changed": False,
        "placement_calls_reexecuted": False,
        "provider_mutation_performed": False,
    }


def native_submission_absent(*, config: Mapping[str, Any], plan: Mapping[str, Any]) -> bool:
    state = Path(
        config.get("progression_root")
        or os.getenv("BLUEPRINT_TASK_EVALUATION_CONFIGURED_CONTROLS_STATE_ROOT")
        or str(Path(config["scene_root"]).parent / "task-evaluation-configured-controls")
    )
    root = (
        state
        / plan["source_launch_id"]
        / f"franka-controls-{plan['expected_production_commit'][:12]}"
    )
    require(not any(p.is_symlink() for p in (root, *root.parents)), "state_unsafe")
    if any(root.rglob("*launch_progression.json")):
        return False
    from .task_evaluation_blocked_native_activation import terminal_blocked_activation

    if any(
        terminal_blocked_activation(marker_path=p, plan=plan, config=config) is None
        for p in root.rglob("*activation_progression.json")
    ):
        return False
    launch_root = Path(
        config.get("launch_state_root")
        or os.getenv("BLUEPRINT_TASK_EVALUATION_LAUNCH_STATE_ROOT")
        or str(Path(config["scene_root"]).parent / "task-evaluation-launch-runs")
    )
    prefixes = [p["expected_activation_id"] for p in plan["future_outputs"].values()]
    return (
        not any(p.name.startswith(tuple(prefixes)) for p in launch_root.iterdir())
        if launch_root.exists()
        else True
    )


def discover(
    *, config: Mapping[str, Any], intent_id: str, source: Mapping[str, Any], expected_commit: str
) -> dict[str, Any] | None:
    from . import task_evaluation_configured_controls_autostart as auto
    from .task_evaluation_controls_autoprovision import _sealed
    from .task_evaluation_retained_controls_evidence import validated_cancellation
    from . import task_evaluation_scene_intake as intake

    state = Path(
        config.get("progression_root")
        or os.getenv("BLUEPRINT_TASK_EVALUATION_CONFIGURED_CONTROLS_STATE_ROOT")
        or str(Path(config["scene_root"]).parent / "task-evaluation-configured-controls")
    )
    binding = state / source["launch_id"] / "cpu-robot-binding"
    matches = []
    for path in (Path(config["controls_root"]) / "terminal-adoptions" / intent_id).glob(
        "*/terminal_adoption_provisioning.json"
    ):
        provision = _sealed(path, "receipt_digest")
        if provision["execution_source_commit"] == expected_commit:
            continue
        old_path = Path(provision["provisioning"]["intent_path"])
        old = _read(old_path)
        owner = _read(Path(old["phases"]["construction"]["authorization_path"]))[
            "scene_owner_attempt"
        ]["scene_attempt_binding"]
        directory = Path(config["scene_root"]) / intent_id
        attempt = intake._read(
            directory / "attempts" / (owner["attempt_id"] + ".json"), "attempt_digest"
        )
        cancelled = validated_cancellation(directory, attempt)
        if cancelled is not None:
            retained = cancelled.get("completed_placement_adoption")
            if retained is not None and retained["execution_commit"] == expected_commit:
                matches.append(retained)
            continue
        result_path = auto._autostart_result_path(root=binding, intent_digest=old["intent_digest"])
        if not result_path.exists():
            continue
        result = _read(result_path)
        token = old["intent_digest"].removeprefix("sha256:")[:16]
        packet = {
            "schema_version": SCHEMA,
            "execution_commit": expected_commit,
            "source_launch_id": source["launch_id"],
            "owner_intent_digest": owner["intent_digest"],
            "source_intent": _file(old_path),
            "source_result": _file(result_path),
            "source_plan": _file(Path(result["plan_path"])),
            "source_agent_checkpoint": checkpoint_reference(
                result=result, binding=binding, token=token
            ),
        }
        packet["adoption_digest"] = canonical_digest(packet, digest_field="adoption_digest")
        verified = validate_adoption(packet)
        if native_submission_absent(config=config, plan=verified["plan"]):
            matches.append(packet)
    require(len(matches) <= 1, "ambiguous_sources")
    return matches[0] if matches else None


def retire_unused_native(
    *, config: Mapping[str, Any], intent_id: str, packet: Mapping[str, Any], dry_run: bool = False
) -> None:
    from . import task_evaluation_scene_intake as intake
    from .task_evaluation_retained_controls_evidence import DIRECTORY, validated_cancellation
    from .task_evaluation_release_identity import running_release_commit

    source = validate_adoption(packet)
    require(
        dry_run or running_release_commit() == packet["execution_commit"],
        "running_release_required",
    )
    require(
        native_submission_absent(config=config, plan=source["plan"]), "native_submission_started"
    )
    directory = Path(config["scene_root"]) / intent_id
    with intake._lock(Path(config["scene_root"])):
        for phase in source["intent"]["phases"].values():
            owner = _read(Path(phase["authorization_path"]))["scene_owner_attempt"][
                "scene_attempt_binding"
            ]
            attempt = intake._read(
                directory / "attempts" / (owner["attempt_id"] + ".json"), "attempt_digest"
            )
            if validated_cancellation(directory, attempt) is not None:
                continue
            receipt = {
                "schema_version": CANCELLATION_SCHEMA,
                "status": "cancelled_before_native_submission",
                **{
                    k: attempt[k]
                    for k in (
                        "attempt_id",
                        "attempt_digest",
                        "intent_digest",
                        "provider",
                        "maximum_spend_usd",
                    )
                },
                "completed_placement_adoption": dict(packet),
                "native_submission_absent": True,
                "model_holds_retained": True,
                "provider_mutation_performed": False,
            }
            receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
            validate_cancellation(receipt=receipt, attempt=attempt)
            if not dry_run:
                target = directory / DIRECTORY / (attempt["attempt_id"] + ".json")
                target.parent.mkdir(mode=0o750, exist_ok=True)
                intake.write_exclusive(target, receipt)


def materialize(
    *,
    intent: Mapping[str, Any],
    root: Path,
    source_launch_id: str,
    launch_root: Path,
    paths: Mapping[str, Any],
    revision: Mapping[str, Any],
    scene_binding: Mapping[str, Any],
    task_binding: Mapping[str, Any],
    trajectory: Mapping[str, Any],
    plan_root: str | Path,
    readiness_materializer: Any,
    plan_materializer: Any,
) -> dict[str, Any]:
    from . import task_evaluation_configured_controls_autostart as auto

    packet = intent["completed_placement_adoption"]
    source = validate_adoption(packet)
    old = source["result"]
    inventory = source["inventory"]
    placement = source["placement"]
    scene_digest = canonical_digest(scene_binding)
    task_digest = canonical_digest(task_binding)
    require(
        old["scene_binding_digest"] == scene_digest
        and old["task_binding_digest"] == task_digest
        and old["trajectory_digest"] == trajectory["trajectory_digest"]
        and old["configured_scene_revision_digest"] == revision["revision_digest"],
        "scientific_binding_changed",
    )
    for name in (
        "robot_asset_usd_path",
        "robot_mount_interface_path",
        "scene_camera_calibration_path",
    ):
        require(
            intent["artifact_inventory"][name]["digest"]
            == source["intent"]["artifact_inventory"][name]["digest"],
            "fixed_input_changed",
        )
    old_universe = _read(Path(old["native_construction_candidate_universe"]["path"]))
    universe_path, universe = auto._materialize_native_feedback_candidate_universe(
        root=root,
        run_id=old_universe["run_id"],
        inventory=inventory,
        trajectory=trajectory,
        camera_template_path=Path(paths["cameras_path"]),
        source_commit=intent["expected_production_commit"],
        maximum_candidates=int(intent["placement"]["candidate_inventory_cap"]),
    )
    universe_ref = {
        "path": str(universe_path),
        "file_sha256": auto._sha256(universe_path),
        "inventory_digest": universe["inventory_digest"],
        "candidate_count": len(universe["candidates"]),
    }
    cameras_path = auto._materialize_placement_aware_cameras(
        root=root,
        camera_template_path=Path(paths["cameras_path"]),
        accepted_pose=placement["accepted_pose"],
        selected_candidate_id=placement["accepted_candidate_id"],
        trajectory=trajectory,
        source_commit=intent["expected_production_commit"],
    )
    require(
        _read(cameras_path)["cameras"] == _read(Path(source["plan"]["cameras_path"]))["cameras"],
        "camera_geometry_changed",
    )
    token = intent["intent_digest"].removeprefix("sha256:")[:16]
    base = root / f"task_evaluation_robot_placement_readiness_candidate-{token}.v1.json"
    if not base.exists():
        readiness_materializer(
            configured_revision=revision,
            scene_binding=scene_binding,
            task_binding=task_binding,
            placement_receipt=placement,
            candidate_inventory=inventory,
            output_path=base,
            native_construction_candidate_universe_reference=universe_ref,
        )
    plan = plan_materializer(
        source_launch_id=source_launch_id,
        launch_state_root=launch_root,
        expected_production_commit=intent["expected_production_commit"],
        bindings={
            "robot_mount_interface_path": paths["robot_mount_interface_path"],
            "scene_camera_calibration_path": paths["scene_camera_calibration_path"],
            "base_pose_candidate_path": str(base),
            "cameras_path": str(cameras_path),
            "runtime_binding_path": paths["runtime_binding_path"],
            "phases": intent["phases"],
        },
        profile_dir=intent["profile_dir"],
        submitted_by=intent["submitted_by"],
        plan_root=plan_root,
    )
    result = {
        **old,
        "intent_digest": intent["intent_digest"],
        "base_pose_candidate_path": str(base),
        "native_construction_candidate_universe": universe_ref,
        "plan_path": plan["plan_path"],
        "plan_digest": plan["plan_digest"],
        "completed_placement_adoption": dict(packet),
        "placement_calls_reexecuted": False,
    }
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    destination = auto._autostart_result_path(root=root, intent_digest=intent["intent_digest"])
    if destination.exists():
        require(_read(destination) == result, "result_changed")
    else:
        from .task_evaluation_configured_controls_progression_worker import _write_immutable

        _write_immutable(destination, result)
    return result
