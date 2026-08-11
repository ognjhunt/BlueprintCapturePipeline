"""Bounded, digest-bound prelaunch skills for Task Evaluation launches.

This module is deliberately below the deterministic launch state machine.  A
profile can opt into a small immutable skill plan whose only effects are
retained, local evidence artifacts below the launch run root.  It never
selects a provider, reads a secret, mutates a provider, grants authority, or
invokes the paid allocator.  The dispatcher remains the only caller and still
owns the canonical allocator boundary.

The first adapters are intentionally narrow:

* ``interiorgs_room_survey`` executes the existing deterministic room-wide
  InteriorGS survey against profile-bound structure and labels files.
* ``earthtojake_step_inspection`` executes the existing pinned, STEP-first CAD
  inspection adapter.  It inspects an already admitted STEP artifact; it does
  not generate geometry or promote a CAD candidate to measured truth.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from .public_scene_cad_inspection_capture import (
    PublicSceneCadInspectionCaptureError,
    capture_cad_inspection,
)


PLAN_SCHEMA_VERSION = "task_evaluation_prelaunch_skill_plan.v1"
EXECUTION_SCHEMA_VERSION = "task_evaluation_prelaunch_skill_execution.v1"
CAD_CONFIG_SCHEMA_VERSION = "task_evaluation_earthtojake_cad_config.v1"
_DIGEST_PREFIX = "sha256:"
_ADAPTER_INTERIORGS_ROOM_SURVEY = "interiorgs_room_survey"
_ADAPTER_EARTHTOJAKE_STEP_INSPECTION = "earthtojake_step_inspection"
_SUPPORTED_ADAPTERS = frozenset(
    {_ADAPTER_INTERIORGS_ROOM_SURVEY, _ADAPTER_EARTHTOJAKE_STEP_INSPECTION}
)
_MIN_TIMEOUT_SECONDS = 1
_MAX_TIMEOUT_SECONDS = 900
_MAX_STEPS = 8
_SAFE_ERROR = re.compile(r"^[a-z0-9_]+(?::[a-z0-9_.-]+)?$")


class PrelaunchSkillExecutionError(ValueError):
    """Raised only for a typed, retained prelaunch-skill failure."""


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_digest(value: Mapping[str, Any], *, digest_field: str) -> str:
    payload = dict(value)
    payload.pop(digest_field, None)
    return _DIGEST_PREFIX + hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _file_digest(path: Path) -> str:
    return _DIGEST_PREFIX + hashlib.sha256(path.read_bytes()).hexdigest()


def _is_digest(value: Any) -> bool:
    text = str(value or "")
    return (
        text.startswith(_DIGEST_PREFIX)
        and len(text) == len(_DIGEST_PREFIX) + 64
        and all(character in "0123456789abcdef" for character in text[len(_DIGEST_PREFIX) :])
    )


def _is_identifier(value: Any) -> bool:
    text = str(value or "")
    return (
        bool(text)
        and len(text) <= 192
        and all(character.isalnum() or character in "._-" for character in text)
    )


def _is_nonempty_text(value: Any, *, maximum: int = 256) -> bool:
    text = str(value or "")
    return bool(text.strip()) and len(text) <= maximum and not any(character in text for character in "\r\n\x00")


def _is_absolute_path(value: Any) -> bool:
    return bool(str(value or "")) and Path(str(value)).expanduser().is_absolute()


def _typed_error(value: BaseException) -> str:
    """Return a stable public blocker without retaining paths or arbitrary stderr."""

    if value.args:
        candidate = str(value.args[0]).strip().lower()
        if _SAFE_ERROR.fullmatch(candidate):
            return candidate
    return "adapter_execution_failed"


def _input_index(profile: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in profile.get("immutable_inputs") or []:
        item = _mapping(row)
        name = str(item.get("name") or "")
        if name:
            index[name] = item
    return index


def _plan_descriptor(profile: Mapping[str, Any]) -> dict[str, Any]:
    return _mapping(profile.get("prelaunch_skill_plan"))


def validate_profile_prelaunch_skill_plan(profile: Mapping[str, Any]) -> list[str]:
    """Validate the optional plan pointer without executing a skill.

    A plan is itself an immutable profile input.  This prevents a website
    request, agent recommendation, or operator-provided path from selecting a
    different skill definition at launch time.
    """

    if "prelaunch_skill_plan" not in profile:
        return []
    descriptor = _plan_descriptor(profile)
    blockers: list[str] = []
    if set(descriptor) != {"plan_id", "path", "digest"}:
        blockers.append("launch_profile_prelaunch_skill_plan_fields_invalid")
    if not _is_identifier(descriptor.get("plan_id")):
        blockers.append("launch_profile_prelaunch_skill_plan_id_invalid")
    if not _is_absolute_path(descriptor.get("path")):
        blockers.append("launch_profile_prelaunch_skill_plan_path_invalid")
    if not _is_digest(descriptor.get("digest")):
        blockers.append("launch_profile_prelaunch_skill_plan_digest_invalid")
    input_row = _input_index(profile).get("prelaunch_skill_plan")
    if not input_row:
        blockers.append("launch_profile_prelaunch_skill_plan_not_immutable")
    elif (
        str(input_row.get("path") or "") != str(descriptor.get("path") or "")
        or input_row.get("digest") != descriptor.get("digest")
    ):
        blockers.append("launch_profile_prelaunch_skill_plan_input_binding_mismatch")
    return sorted(set(blockers))


def _validate_step(step_value: Any) -> list[str]:
    step = _mapping(step_value)
    step_id = str(step.get("step_id") or "invalid")
    suffix = step_id if _is_identifier(step_id) else "invalid"
    blockers: list[str] = []
    adapter = step.get("adapter")
    if not _is_identifier(step.get("step_id")):
        blockers.append("prelaunch_skill_step_id_invalid")
    if adapter not in _SUPPORTED_ADAPTERS:
        blockers.append(f"prelaunch_skill_step_adapter_invalid:{suffix}")
        return blockers
    if adapter == _ADAPTER_INTERIORGS_ROOM_SURVEY:
        allowed = {
            "step_id",
            "adapter",
            "structure_input",
            "labels_input",
            "scene_id",
            "timeout_seconds",
            "target_ins_id",
        }
        if set(step) != allowed:
            blockers.append(f"prelaunch_skill_room_survey_fields_invalid:{suffix}")
        for field in ("structure_input", "labels_input"):
            if not _is_identifier(step.get(field)):
                blockers.append(f"prelaunch_skill_room_survey_input_invalid:{suffix}:{field}")
        if not _is_nonempty_text(step.get("scene_id")):
            blockers.append(f"prelaunch_skill_room_survey_scene_invalid:{suffix}")
        target = step.get("target_ins_id")
        if target is not None and not _is_nonempty_text(target):
            blockers.append(f"prelaunch_skill_room_survey_target_invalid:{suffix}")
    elif adapter == _ADAPTER_EARTHTOJAKE_STEP_INSPECTION:
        allowed = {"step_id", "adapter", "step_input", "configuration_input", "timeout_seconds"}
        if set(step) != allowed:
            blockers.append(f"prelaunch_skill_cad_inspection_fields_invalid:{suffix}")
        for field in ("step_input", "configuration_input"):
            if not _is_identifier(step.get(field)):
                blockers.append(f"prelaunch_skill_cad_inspection_input_invalid:{suffix}:{field}")
    timeout = step.get("timeout_seconds")
    if (
        not isinstance(timeout, int)
        or isinstance(timeout, bool)
        or timeout < _MIN_TIMEOUT_SECONDS
        or timeout > _MAX_TIMEOUT_SECONDS
    ):
        blockers.append(f"prelaunch_skill_step_timeout_invalid:{suffix}")
    return sorted(set(blockers))


def validate_prelaunch_skill_plan(value: Mapping[str, Any]) -> list[str]:
    """Fail closed unless a plan contains only supported bounded adapters."""

    plan = _mapping(value)
    blockers: list[str] = []
    expected_fields = {
        "schema_version",
        "program_id",
        "plan_id",
        "source_bundle",
        "steps",
        "plan_digest",
    }
    if set(plan) != expected_fields:
        blockers.append("prelaunch_skill_plan_fields_invalid")
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        blockers.append("prelaunch_skill_plan_schema_version_mismatch")
    if plan.get("program_id") != "arm-decision-proof-v1":
        blockers.append("prelaunch_skill_plan_program_mismatch")
    if not _is_identifier(plan.get("plan_id")):
        blockers.append("prelaunch_skill_plan_id_invalid")
    source = _mapping(plan.get("source_bundle"))
    if set(source) != {"bundle_id", "digest"}:
        blockers.append("prelaunch_skill_plan_source_fields_invalid")
    if not _is_identifier(source.get("bundle_id")):
        blockers.append("prelaunch_skill_plan_source_bundle_id_invalid")
    if not _is_digest(source.get("digest")):
        blockers.append("prelaunch_skill_plan_source_digest_invalid")
    steps = plan.get("steps")
    if not isinstance(steps, list) or not steps or len(steps) > _MAX_STEPS:
        blockers.append("prelaunch_skill_plan_steps_invalid")
    else:
        seen: set[str] = set()
        for step in steps:
            blockers.extend(_validate_step(step))
            step_id = str(_mapping(step).get("step_id") or "")
            if step_id in seen:
                blockers.append("prelaunch_skill_plan_step_id_duplicate")
            seen.add(step_id)
    if plan.get("plan_digest") != canonical_digest(plan, digest_field="plan_digest"):
        blockers.append("prelaunch_skill_plan_digest_mismatch")
    return sorted(set(blockers))


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    payload = (_canonical_json(value) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise PrelaunchSkillExecutionError("prelaunch_skill_immutable_output_conflict")
        return
    with path.open("xb") as stream:
        stream.write(payload)


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.is_file() and not path.is_symlink(),
        "digest": _file_digest(path) if path.is_file() and not path.is_symlink() else None,
        "size_bytes": path.stat().st_size if path.is_file() and not path.is_symlink() else None,
    }


def _load_json_object(path: Path, *, error: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PrelaunchSkillExecutionError(error)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PrelaunchSkillExecutionError(error) from exc
    if not isinstance(value, Mapping):
        raise PrelaunchSkillExecutionError(error)
    return dict(value)


def _profile_input_path(
    inputs: Mapping[str, Mapping[str, Any]], *, name: str, missing_error: str
) -> Path:
    row = _mapping(inputs.get(name))
    raw_path = Path(str(row.get("path") or "")).expanduser()
    if raw_path.is_symlink() or not raw_path.is_file() or _file_digest(raw_path) != row.get("digest"):
        raise PrelaunchSkillExecutionError(missing_error)
    return raw_path.resolve()


def _load_cad_config(path: Path) -> dict[str, Any]:
    config = _load_json_object(path, error="prelaunch_skill_cad_config_invalid")
    expected = {
        "schema_version",
        "repo_root",
        "cad_skill_root",
        "cad_python",
        "expected_commit",
        "expected_tree",
        "config_digest",
    }
    if set(config) != expected:
        raise PrelaunchSkillExecutionError("prelaunch_skill_cad_config_fields_invalid")
    if config.get("schema_version") != CAD_CONFIG_SCHEMA_VERSION:
        raise PrelaunchSkillExecutionError("prelaunch_skill_cad_config_schema_mismatch")
    if not all(_is_absolute_path(config.get(field)) for field in ("repo_root", "cad_skill_root", "cad_python")):
        raise PrelaunchSkillExecutionError("prelaunch_skill_cad_config_path_invalid")
    for field in ("expected_commit", "expected_tree"):
        value = str(config.get(field) or "")
        if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
            raise PrelaunchSkillExecutionError("prelaunch_skill_cad_config_revision_invalid")
    if config.get("config_digest") != canonical_digest(config, digest_field="config_digest"):
        raise PrelaunchSkillExecutionError("prelaunch_skill_cad_config_digest_mismatch")
    return config


def _execute_room_survey(
    *,
    step: Mapping[str, Any],
    inputs: Mapping[str, Mapping[str, Any]],
    output_path: Path,
) -> None:
    structure = _profile_input_path(
        inputs,
        name=str(step["structure_input"]),
        missing_error="prelaunch_skill_room_survey_structure_input_invalid",
    )
    labels = _profile_input_path(
        inputs,
        name=str(step["labels_input"]),
        missing_error="prelaunch_skill_room_survey_labels_input_invalid",
    )
    # Invoke the existing deterministic module through fixed argv so the
    # profile's timeout is enforceable.  There is no shell, arbitrary command,
    # external URL, or agent-selected argument in this adapter.
    command = [
        sys.executable,
        "-m",
        "blueprint_pipeline.public_scene_viewpoint_survey",
        "--structure",
        str(structure),
        "--labels",
        str(labels),
        "--scene-id",
        str(step["scene_id"]),
        "--approved-root",
        str(structure.parent),
        "--approved-root",
        str(labels.parent),
        "--approved-root",
        str(output_path.parent),
        "--out",
        str(output_path),
    ]
    if step.get("target_ins_id") is not None:
        command.extend(["--target-ins-id", str(step["target_ins_id"])])
    try:
        completed = subprocess.run(
            command,
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            timeout=int(step["timeout_seconds"]),
        )
    except subprocess.TimeoutExpired as exc:
        raise PrelaunchSkillExecutionError("prelaunch_skill_room_survey_timeout") from exc
    if completed.returncode != 0:
        raise PrelaunchSkillExecutionError("prelaunch_skill_room_survey_execution_failed")
    survey = _load_json_object(output_path, error="prelaunch_skill_room_survey_output_invalid")
    if survey.get("survey_digest") != canonical_digest(survey, digest_field="survey_digest"):
        raise PrelaunchSkillExecutionError("prelaunch_skill_room_survey_digest_mismatch")


def _execute_cad_inspection(
    *,
    step: Mapping[str, Any],
    inputs: Mapping[str, Mapping[str, Any]],
    output_path: Path,
) -> None:
    step_path = _profile_input_path(
        inputs,
        name=str(step["step_input"]),
        missing_error="prelaunch_skill_cad_step_input_invalid",
    )
    config_path = _profile_input_path(
        inputs,
        name=str(step["configuration_input"]),
        missing_error="prelaunch_skill_cad_config_input_invalid",
    )
    config = _load_cad_config(config_path)
    try:
        capture_cad_inspection(
            repo_root=config["repo_root"],
            evidence_root=output_path.parent,
            cad_skill_root=config["cad_skill_root"],
            cad_python=config["cad_python"],
            expected_commit=config["expected_commit"],
            expected_tree=config["expected_tree"],
            step_path=step_path,
            output_path=output_path,
            timeout_seconds=int(step["timeout_seconds"]),
        )
    except PublicSceneCadInspectionCaptureError as exc:
        raise PrelaunchSkillExecutionError(_typed_error(exc)) from exc
    if output_path.is_symlink() or not output_path.is_file():
        raise PrelaunchSkillExecutionError("prelaunch_skill_cad_output_missing")


def _execution_base(
    *,
    status: str,
    plan: Mapping[str, Any] | None,
    descriptor: Mapping[str, Any],
    steps: list[dict[str, Any]],
    blockers: list[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": EXECUTION_SCHEMA_VERSION,
        "status": status,
        "plan_id": plan.get("plan_id") if plan else descriptor.get("plan_id"),
        "plan_digest": descriptor.get("digest"),
        "provider_mutation_performed": False,
        "allocator_invoked": False,
        "automatic_retry_authorized": False,
        "agent_operator_used": False,
        "steps": steps,
        "blockers": sorted(set(blockers)),
        "claim_boundary": {
            "skills_are_prelaunch_evidence_only": True,
            "skills_do_not_select_provider_or_gpu": True,
            "skills_do_not_grant_rights_or_spend_authority": True,
            "skills_do_not_mutate_providers": True,
            "survey_outputs_are_not_evaluation_method_inputs": True,
            "cad_outputs_are_candidate_inspection_only": True,
        },
    }
    result["execution_digest"] = canonical_digest(result, digest_field="execution_digest")
    return result


def execute_prelaunch_skill_plan(
    *, profile: Mapping[str, Any], run_root: str | Path
) -> dict[str, Any]:
    """Execute one immutable plan and retain a digest-bound terminal receipt.

    The caller must have already validated immutable profile inputs.  This
    function repeats the plan/input checks so it remains fail-closed if called
    outside that expected dispatcher flow.
    """

    if "prelaunch_skill_plan" not in profile:
        return _execution_base(
            status="not_configured", plan=None, descriptor={}, steps=[], blockers=[]
        )

    descriptor = _plan_descriptor(profile)
    root = Path(run_root).expanduser().resolve()
    prelaunch_root = root / "prelaunch_skills"
    receipt_path = prelaunch_root / "execution.json"
    if receipt_path.exists():
        prior = _load_json_object(receipt_path, error="prelaunch_skill_receipt_invalid")
        if (
            prior.get("plan_id") == descriptor.get("plan_id")
            and prior.get("plan_digest") == descriptor.get("digest")
            and prior.get("execution_digest") == canonical_digest(
                prior, digest_field="execution_digest"
            )
        ):
            return prior
        raise PrelaunchSkillExecutionError("prelaunch_skill_receipt_binding_mismatch")

    profile_blockers = validate_profile_prelaunch_skill_plan(profile)
    if profile_blockers:
        result = _execution_base(
            status="blocked",
            plan=None,
            descriptor=descriptor,
            steps=[],
            blockers=profile_blockers,
        )
        _write_immutable(receipt_path, result)
        return result

    plan_path = Path(str(descriptor.get("path") or "")).expanduser()
    plan: dict[str, Any] | None = None
    blockers: list[str] = []
    try:
        if plan_path.is_symlink() or not plan_path.is_file():
            raise PrelaunchSkillExecutionError("prelaunch_skill_plan_input_missing")
        if _file_digest(plan_path) != descriptor.get("digest"):
            raise PrelaunchSkillExecutionError("prelaunch_skill_plan_input_digest_mismatch")
        plan = _load_json_object(plan_path, error="prelaunch_skill_plan_invalid_json")
        blockers.extend(validate_prelaunch_skill_plan(plan))
        if plan.get("plan_id") != descriptor.get("plan_id"):
            blockers.append("prelaunch_skill_plan_descriptor_binding_mismatch")
        profile_source = _mapping(profile.get("source_bundle"))
        plan_source = _mapping(plan.get("source_bundle"))
        if plan_source != {
            "bundle_id": profile_source.get("bundle_id"),
            "digest": profile_source.get("digest"),
        }:
            blockers.append("prelaunch_skill_plan_source_bundle_binding_mismatch")
    except PrelaunchSkillExecutionError as exc:
        blockers.append(_typed_error(exc))

    if blockers or plan is None:
        result = _execution_base(
            status="blocked", plan=plan, descriptor=descriptor, steps=[], blockers=blockers
        )
        _write_immutable(receipt_path, result)
        return result

    inputs = _input_index(profile)
    step_results: list[dict[str, Any]] = []
    for step_value in plan["steps"]:
        step = _mapping(step_value)
        step_id = str(step["step_id"])
        adapter = str(step["adapter"])
        output_path = prelaunch_root / f"{step_id}.json"
        try:
            if output_path.exists():
                raise PrelaunchSkillExecutionError("prelaunch_skill_step_output_already_exists")
            if adapter == _ADAPTER_INTERIORGS_ROOM_SURVEY:
                _execute_room_survey(step=step, inputs=inputs, output_path=output_path)
            elif adapter == _ADAPTER_EARTHTOJAKE_STEP_INSPECTION:
                _execute_cad_inspection(step=step, inputs=inputs, output_path=output_path)
            else:  # guarded by validate_prelaunch_skill_plan; retained for defense in depth.
                raise PrelaunchSkillExecutionError("prelaunch_skill_step_adapter_invalid")
            artifact = _artifact(output_path)
            if not artifact["exists"] or not _is_digest(artifact["digest"]):
                raise PrelaunchSkillExecutionError("prelaunch_skill_step_output_invalid")
            step_results.append(
                {
                    "step_id": step_id,
                    "adapter": adapter,
                    "status": "passed",
                    "output": artifact,
                    "blockers": [],
                }
            )
        except (PrelaunchSkillExecutionError, OSError, subprocess.SubprocessError) as exc:
            blocker = _typed_error(exc)
            blockers.append(f"prelaunch_skill_step_failed:{step_id}:{blocker}")
            step_results.append(
                {
                    "step_id": step_id,
                    "adapter": adapter,
                    "status": "blocked",
                    "output": _artifact(output_path),
                    "blockers": [blocker],
                }
            )
            break

    result = _execution_base(
        status="passed" if not blockers else "blocked",
        plan=plan,
        descriptor=descriptor,
        steps=step_results,
        blockers=blockers,
    )
    _write_immutable(receipt_path, result)
    return result


__all__ = [
    "CAD_CONFIG_SCHEMA_VERSION",
    "EXECUTION_SCHEMA_VERSION",
    "PLAN_SCHEMA_VERSION",
    "PrelaunchSkillExecutionError",
    "canonical_digest",
    "execute_prelaunch_skill_plan",
    "validate_prelaunch_skill_plan",
    "validate_profile_prelaunch_skill_plan",
]
