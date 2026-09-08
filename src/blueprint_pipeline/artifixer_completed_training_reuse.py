"""Reuse exact trained appearance bytes from a terminal, provider-zero scene run.

The old failed review and diagnostic checkpoint remain unchanged. This receipt
only admits reuse of generated inputs; a new independent final review is required.
"""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path
from typing import Any, Mapping

from .decision_evidence_contracts import canonical_digest, canonical_json
from .public_scene_artifixer3d_native_exports import geometry_protection_is_qualified
from .task_evaluation_scene_configuration_artifixer_warm_checkpoint import (
    SCHEMA_VERSION as CHECKPOINT_SCHEMA,
    hydrate_artifixer_post_training_checkpoint,
    validate_artifixer_post_training_checkpoint,
)

SOURCE_ENV = "BLUEPRINT_ARTIFIXER_COMPLETED_TRAINING_SOURCE_LAUNCH_ROOT"
SCHEMA = "task_evaluation_artifixer_completed_training_reuse.v1"
PREFIX = "stages/stage-1/producer/artifixer_post_training_checkpoint/"


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError("artifixer_completed_training_reuse_" + reason)


def _sha(path: Path) -> str:
    _require(path.is_file() and not path.is_symlink(), "file_invalid")
    with path.open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def _read(path: Path) -> dict:
    _sha(path)
    value = json.loads(path.read_text())
    _require(isinstance(value, dict), "document_invalid")
    return value


def _file_identity(value: Mapping[str, Any]) -> dict:
    digest, size = value.get("sha256"), value.get("size_bytes")
    _require(
        isinstance(digest, str)
        and re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is not None
        and type(size) is int
        and size > 0,
        "file_identity_invalid",
    )
    return {"sha256": digest, "size_bytes": size}


def _portable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            k: _portable(v)
            for k, v in value.items()
            if k not in {"path", "relative_path", "materialized_path"}
        }
    if isinstance(value, list):
        return [_portable(v) for v in value]
    return value


def training_identity(
    *,
    candidate: Mapping[str, Any],
    teacher: Mapping[str, Any],
    tuning: Mapping[str, Any],
    configuration_sha256: str,
) -> dict:
    initialization = candidate.get("appearance_initialization")
    _require(
        isinstance(initialization, Mapping)
        and initialization.get("geometry_mode") == "freeze_declared_appearance_initialization",
        "source_preservation_required",
    )
    tasks = candidate.get("tasks")
    _require(
        isinstance(tasks, list) and len(tasks) == 1 and bool(tasks[0].get("frames")), "task_invalid"
    )
    task = tasks[0]
    _require(
        teacher.get("task_id") == task["task_id"]
        and len(teacher.get("frames", [])) == len(task["frames"]),
        "teacher_invalid",
    )
    return {
        "configuration_sha256": configuration_sha256,
        "publisher_scene_id": candidate["publisher_scene_id"],
        "task_id": task["task_id"],
        "shared_initialization": _file_identity(candidate["shared_retained_scene"]),
        "parameter_partition": initialization["parameter_partition"],
        "geometry_mode": initialization["geometry_mode"],
        "transforms": _file_identity(task["transforms"]),
        "camera_index": _file_identity(task["camera_index"]),
        "source_frames": _portable(task["frames"]),
        "teacher_frames": _portable(teacher["frames"]),
        "artifixer_tuning": dict(tuning),
    }


def stage_completed_training(
    *, source_launch_root: Path, prepared: dict, stage_input: dict, tuning: dict, output_root: Path
) -> dict:
    source = Path(source_launch_root).resolve()
    _require(not Path(source_launch_root).is_symlink(), "source_symlink")
    zero = _read(source / "post_teardown_provider_zero_receipt.json")
    _require(
        zero.get("provider_zero_receipt_digest")
        == canonical_digest(zero, digest_field="provider_zero_receipt_digest")
        and zero.get("launch_id") == source.name
        and zero.get("status") == "provider_zero_confirmed"
        and zero.get("provider_zero_verified") is True
        and zero.get("continuing_spend_from_this_run") is False
        and not zero.get("blockers"),
        "source_not_closed",
    )
    launch = _read(source / "launch_receipt.json")
    _require(launch.get("status") in {"blocked", "completed"}, "source_not_terminal")
    job = source / "allocator/scene-configuration-job"
    api_receipt = _read(job / "api_pretraining_receipt.json")
    capsule_path = job / "api_pretraining_capsule.zip"
    _require(
        api_receipt.get("receipt_digest")
        == canonical_digest(api_receipt, digest_field="receipt_digest")
        and api_receipt.get("capsule_sha256") == _sha(capsule_path),
        "source_capsule_changed",
    )
    with zipfile.ZipFile(capsule_path) as archive:
        capsule = json.loads(archive.read("capsule_manifest.json"))
        _require(
            capsule.get("capsule_digest")
            == canonical_digest(capsule, digest_field="capsule_digest")
            and capsule["capsule_digest"] == api_receipt["capsule_digest"],
            "capsule_invalid",
        )
        original_state = json.loads(archive.read(capsule["state_path"]))
        _require(
            original_state.get("state_digest")
            == canonical_digest(original_state, digest_field="state_digest"),
            "source_state_invalid",
        )
        old_prepared = original_state["state"]
        teacher_member = str(
            Path(old_prepared["teacher_receipt_path"]).relative_to(capsule["logical_root"])
        )
        old_teacher = json.loads(archive.read(teacher_member))
        original_stage = json.loads(archive.read("output/stage_production_input.v1.json"))
    provider_archive = job / "vast_provider_run/vast_provider_runtime_output.zip"
    archive_sha = _sha(provider_archive)
    with zipfile.ZipFile(provider_archive) as archive:
        _require(
            len(archive.namelist()) == len(set(archive.namelist())), "duplicate_archive_member"
        )
        checkpoint = json.loads(archive.read(PREFIX + CHECKPOINT_SCHEMA + ".json"))
        expected = training_identity(
            candidate=old_prepared["candidate"],
            teacher=old_teacher,
            tuning=checkpoint["bindings"]["artifixer_tuning"],
            configuration_sha256=original_stage["configuration_sha256"],
        )
        actual = training_identity(
            candidate=prepared["candidate"],
            teacher=_read(Path(prepared["teacher_receipt_path"])),
            tuning=tuning,
            configuration_sha256=stage_input["configuration_sha256"],
        )
        _require(expected == actual, "training_inputs_changed")
        output = Path(output_root)
        _require(not output.exists(), "output_exists")
        output.mkdir(parents=True, mode=0o750)
        checkpoint_root = output / "checkpoint"
        checkpoint_root.mkdir()
        for row in checkpoint["inventory"]:
            relative = Path(row["relative_path"])
            _require(
                not relative.is_absolute() and ".." not in relative.parts, "checkpoint_path_invalid"
            )
            data = archive.read(PREFIX + relative.as_posix())
            _require(
                len(data) == row["size_bytes"]
                and "sha256:" + hashlib.sha256(data).hexdigest() == row["sha256"],
                "checkpoint_file_changed",
            )
            destination = checkpoint_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            destination.chmod(row["mode"])
        (checkpoint_root / (CHECKPOINT_SCHEMA + ".json")).write_text(
            canonical_json(checkpoint) + "\n"
        )
    checked = validate_artifixer_post_training_checkpoint(checkpoint_root=checkpoint_root)
    runtime = _read(checkpoint_root / "runtime/runtime_result.json")
    protection = runtime["tasks"][0]["native_appearance"]["geometry_protection"]
    _require(
        geometry_protection_is_qualified(protection)
        and protection.get("mode") == "freeze_declared_appearance_initialization"
        and runtime.get("artifixer3d_distillation_executed") is True,
        "completed_training_not_preserved",
    )
    receipt = {
        "schema_version": SCHEMA,
        "status": "exact_completed_training_admitted_for_new_review",
        "source_launch_id": source.name,
        "source_commit": original_state["source_commit"],
        "source_provider_output_sha256": archive_sha,
        "source_provider_zero_digest": zero["provider_zero_receipt_digest"],
        "source_checkpoint_digest": checked["checkpoint_digest"],
        "source_capsule_sha256": api_receipt["capsule_sha256"],
        "training_identity": actual,
        "training_identity_digest": canonical_digest(actual),
        "new_training_executed": False,
        "new_independent_review_required": True,
        "appearance_repair_qualified": False,
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    receipt_path = output / "reuse_receipt.json"
    receipt_path.write_text(canonical_json(receipt) + "\n")
    return {
        "checkpoint_root": str(checkpoint_root),
        "receipt_path": str(receipt_path),
        "receipt_digest": receipt["receipt_digest"],
    }


def hydrate_completed_training(
    *,
    reference: dict,
    candidate: dict,
    teacher_receipt_path: Path,
    tuning: dict,
    configuration_sha256: str,
) -> dict:
    receipt = _read(Path(reference["receipt_path"]))
    identity = training_identity(
        candidate=candidate,
        teacher=_read(teacher_receipt_path),
        tuning=tuning,
        configuration_sha256=configuration_sha256,
    )
    _require(
        receipt.get("schema_version") == SCHEMA
        and receipt.get("receipt_digest") == reference["receipt_digest"]
        and receipt["receipt_digest"] == canonical_digest(receipt, digest_field="receipt_digest")
        and receipt.get("new_independent_review_required") is True
        and receipt.get("appearance_repair_qualified") is False
        and receipt.get("training_identity") == identity
        and receipt.get("training_identity_digest") == canonical_digest(identity),
        "receipt_invalid",
    )
    checkpoint = validate_artifixer_post_training_checkpoint(
        checkpoint_root=reference["checkpoint_root"]
    )
    _require(
        checkpoint["checkpoint_digest"] == receipt["source_checkpoint_digest"], "checkpoint_changed"
    )
    result = hydrate_artifixer_post_training_checkpoint(
        checkpoint_root=reference["checkpoint_root"],
        expected_binding_digest=checkpoint["binding_digest"],
    )
    result["reuse_receipt"] = receipt
    return result


REVIEW_ENV = "BLUEPRINT_ARTIFIXER_COMPLETED_TRAINING_REVIEW_ROOT"


def stage_completed_review(*, source_root: Path, output_root: Path) -> dict:
    """Carry a real accepted CPU review; admission on the worker checks its full input."""
    from .task_evaluation_artifixer_ai_visual_review import (
        EXECUTION_SCHEMA_VERSION,
        FINAL_REVIEW_POLICY,
        AI_REVIEW_MODEL,
        _PROMPT,
    )

    original_input = _read(source_root / "review_input.json")
    execution_path = source_root / "review-live" / (EXECUTION_SCHEMA_VERSION + ".json")
    execution = _read(execution_path)
    _require(
        execution.get("execution_digest")
        == canonical_digest(execution, digest_field="execution_digest")
        and original_input.get("receipt_digest")
        == canonical_digest(original_input, digest_field="receipt_digest")
        and execution.get("final_composite_receipt_digest") == original_input["receipt_digest"]
        and execution.get("status") == "completed"
        and execution.get("decision") == "accepted"
        and execution.get("review_phase") == "post_training"
        and original_input.get("review_phase") == "post_training"
        and execution.get("provider_called") is True
        and execution.get("reviewer", {}).get("model") == AI_REVIEW_MODEL
        and execution.get("reviewer", {}).get("runtime") == "openai_agents_sdk"
        and bool(execution.get("usage", {}).get("provider_response_id"))
        and execution.get("review_policy") == FINAL_REVIEW_POLICY
        and execution.get("review_prompt_sha256")
        == "sha256:" + hashlib.sha256(_PROMPT.encode()).hexdigest()
        and execution.get("response_store") is False
        and execution.get("raw_secret_values_recorded") is False,
        "final_review_not_real_accepted_evidence",
    )
    output_root.mkdir(parents=True)
    target = output_root / "original_review_execution.json"
    target.write_bytes(execution_path.read_bytes())
    (output_root / "original_review_input.json").write_text(canonical_json(original_input) + "\n")
    return {"execution_path": str(target), "execution_sha256": _sha(target)}


def reuse_completed_review(
    *,
    reference: dict,
    current_input_path: Path,
    output_root: Path,
    publisher_instance_id: str,
    minimum_frame_count: int,
) -> dict:
    from .task_evaluation_artifixer_ai_visual_review import (
        FINAL_REVIEW_POLICY,
        _PROMPT,
        build_artifixer_ai_visual_review_input,
        seal_artifixer_ai_visual_review,
    )

    source = Path(reference["execution_path"])
    _require(_sha(source) == reference["execution_sha256"], "review_execution_changed")
    old = _read(source)
    current = _read(current_input_path)
    payload, _, inventory, _ = build_artifixer_ai_visual_review_input(
        final_composite_receipt_path=current_input_path
    )
    _require(
        old.get("execution_digest") == canonical_digest(old, digest_field="execution_digest")
        and old.get("review_phase") == current.get("review_phase") == "post_training"
        and old.get("decision") == "accepted"
        and old.get("provider_called") is True
        and old.get("review_policy") == FINAL_REVIEW_POLICY
        and old.get("review_prompt_sha256")
        == "sha256:" + hashlib.sha256(_PROMPT.encode()).hexdigest()
        and old.get("input_digest") == canonical_digest({"input": payload})
        and {r["camera_id"]: r["sha256"] for r in inventory}
        == {r["camera_id"]: r["frame_sha256"] for r in old["frames"]},
        "review_inputs_changed",
    )
    original = output_root / "source_review_execution.json"
    original.write_bytes(source.read_bytes())
    # A derivative binding receipt, explicitly tied to the unchanged actual call.
    execution = {
        **old,
        "final_composite_receipt_digest": current["receipt_digest"],
        "source_execution_digest": old["execution_digest"],
        "source_final_composite_receipt_digest": old["final_composite_receipt_digest"],
        "source_execution_receipt": {"path": str(original), "sha256": _sha(original)},
        "completed_review_execution_reused": True,
        "new_model_call_performed": False,
    }
    execution["execution_digest"] = canonical_digest(execution, digest_field="execution_digest")
    execution_path = output_root / "rebound_review_execution.json"
    execution_path.write_text(canonical_json(execution) + "\n")
    receipt_path = output_root / "accepted_review.json"
    receipt = seal_artifixer_ai_visual_review(
        final_composite_receipt_path=current_input_path,
        review_execution_receipt_path=execution_path,
        publisher_instance_id=publisher_instance_id,
        minimum_review_frames=minimum_frame_count,
        output_path=receipt_path,
    )
    return {
        "review_input": current,
        "review_input_path": current_input_path,
        "review": {
            "decision": "accepted",
            "completed_review_execution_reused": True,
            "new_model_call_performed": False,
            "execution_receipt": {
                "path": str(execution_path),
                "execution_digest": execution["execution_digest"],
            },
            "review_receipt": {
                "path": str(receipt_path),
                "receipt_digest": receipt["receipt_digest"],
            },
        },
    }
