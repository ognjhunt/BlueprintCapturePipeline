"""Production worker that compiles one configured scene plus team inputs.

The website never supplies a native-Arena packet.  It supplies one configured
scene revision and independently versioned robot, controller or policy, sensor,
and runtime references.  This worker verifies those materialized bytes and
invokes the installed production compiler exactly once to create the internal
episode packet used by the existing Task Evaluation launch path.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_episode_compilation_queue import ENVELOPE_SCHEMA_VERSION
from .task_evaluation_launch_preparation_contract import (
    TaskEvaluationLaunchPreparationContractError,
    validate_launch_preparation_request,
)
from .task_evaluation_launch_preparation_queue import (
    write_launch_preparation_record_exclusive,
)
from .task_evaluation_scene_construction_queue import (
    ensure_scene_construction_queue_root,
)


COMPILER_OUTPUT_SCHEMA_VERSION = "task_evaluation_episode_compiler_output.v1"
RESULT_SCHEMA_VERSION = "task_evaluation_episode_compilation_result.v1"
EpisodeCompiler = Callable[..., Mapping[str, Any]]


class TaskEvaluationEpisodeCompilationWorkerError(RuntimeError):
    """One production episode packet could not be compiled fail-closed."""


def _sha256_and_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return "sha256:" + digest.hexdigest(), size


def _under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _load_envelope(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskEvaluationEpisodeCompilationWorkerError(
            "episode_compilation_envelope_invalid"
        ) from exc
    if (
        path.is_symlink()
        or not isinstance(value, Mapping)
        or value.get("schema_version") != ENVELOPE_SCHEMA_VERSION
        or value.get("envelope_digest")
        != canonical_digest(value, digest_field="envelope_digest")
        or value.get("automatic_progression_required") is not True
        or value.get("robot_specific_episode_packet_compiled_in_production")
        is not True
        or value.get("production_compiler_owns_episode_packet") is not True
        or value.get("customer_supplied_prebuilt_episode_packet") is not False
        or value.get("provider_mutation_performed") is not False
        or value.get("paid_execution_requested") is not False
    ):
        raise TaskEvaluationEpisodeCompilationWorkerError(
            "episode_compilation_envelope_invalid"
        )
    try:
        request = validate_launch_preparation_request(value["request"])
    except (KeyError, TaskEvaluationLaunchPreparationContractError) as exc:
        raise TaskEvaluationEpisodeCompilationWorkerError(
            "episode_compilation_request_invalid"
        ) from exc
    if (
        request["run_mode"] != "episode_evaluation"
        or request["construction"]["mode"] != "reuse_configured_scene"
        or request["task"]["binding_mode"] != "reuse_configured_template"
        or request["task"]["configured_scene_revision_digest"]
        != value.get("configured_scene_revision_digest")
    ):
        raise TaskEvaluationEpisodeCompilationWorkerError(
            "episode_compilation_request_binding_invalid"
        )
    return dict(value)


def _verified_references(
    envelope: Mapping[str, Any], *, input_root: Path
) -> dict[str, dict[str, Any]]:
    rows = envelope.get("materialized_references")
    if not isinstance(rows, list) or not rows:
        raise TaskEvaluationEpisodeCompilationWorkerError(
            "episode_compilation_materialized_references_invalid"
        )
    verified: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise TaskEvaluationEpisodeCompilationWorkerError(
                "episode_compilation_materialized_references_invalid"
            )
        contract_path = str(row.get("contract_path") or "")
        path = Path(str(row.get("materialized_path") or "")).resolve()
        if (
            not contract_path
            or contract_path in verified
            or row.get("full_byte_service_account_readback_passed") is not True
            or not _under(path, input_root)
            or path.is_symlink()
            or not path.is_file()
            or _sha256_and_size(path)
            != (row.get("digest"), row.get("size_bytes"))
        ):
            raise TaskEvaluationEpisodeCompilationWorkerError(
                "episode_compilation_materialized_reference_invalid"
            )
        verified[contract_path] = dict(row)
    bundle = verified.get(
        "scene.configured_revision.configured_scene_bundle"
    )
    expected_bundle = envelope.get("configured_scene_bundle")
    if (
        bundle is None
        or not isinstance(expected_bundle, Mapping)
        or any(
            bundle.get(field) != expected_bundle.get(field)
            for field in ("uri", "digest", "size_bytes")
        )
    ):
        raise TaskEvaluationEpisodeCompilationWorkerError(
            "episode_compilation_configured_scene_bundle_binding_invalid"
        )
    return verified


def _validated_compiler_output(
    value: Mapping[str, Any],
    *,
    envelope: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    output = dict(value)
    artifact = output.get("compiled_episode_packet")
    path = Path(str((artifact or {}).get("path") or "")).resolve()
    if (
        output.get("schema_version") != COMPILER_OUTPUT_SCHEMA_VERSION
        or output.get("status") != "completed"
        or output.get("run_id") != envelope["run_id"]
        or output.get("configured_scene_revision_digest")
        != envelope["configured_scene_revision_digest"]
        or output.get("compiled_by_production") is not True
        or output.get("customer_supplied_prebuilt_episode_packet") is not False
        or output.get("provider_mutation_performed") is not False
        or output.get("paid_execution_requested") is not False
        or output.get("raw_secret_values_recorded") is not False
        or output.get("compiler_output_digest")
        != canonical_digest(output, digest_field="compiler_output_digest")
        or not isinstance(artifact, Mapping)
        or artifact.get("format") != "native_task_arena_bundle_zip"
        or not _under(path, output_root)
        or path.is_symlink()
        or not path.is_file()
        or _sha256_and_size(path)
        != (artifact.get("digest"), artifact.get("size_bytes"))
    ):
        raise TaskEvaluationEpisodeCompilationWorkerError(
            "episode_compiler_output_invalid"
        )
    return output


def process_episode_compilation_queue(
    *,
    queue_root: str | Path,
    input_root: str | Path,
    output_root: str | Path,
    source_commit: str,
    episode_compiler: EpisodeCompiler,
    max_messages: int = 1,
) -> dict[str, Any]:
    """Compile queued Website evaluations without allocating a provider."""

    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise TaskEvaluationEpisodeCompilationWorkerError(
            "episode_compilation_source_commit_unproven"
        )
    if (
        not isinstance(max_messages, int)
        or isinstance(max_messages, bool)
        or not 1 <= max_messages <= 8
    ):
        raise TaskEvaluationEpisodeCompilationWorkerError(
            "episode_compilation_max_messages_invalid"
        )
    queue = ensure_scene_construction_queue_root(queue_root)
    inputs = Path(input_root).resolve(strict=True)
    outputs = Path(output_root)
    outputs.mkdir(parents=True, exist_ok=True, mode=0o750)
    outputs = outputs.resolve(strict=True)
    results_root = queue / "results"
    results_root.mkdir(mode=0o750, exist_ok=True)
    processed: list[dict[str, Any]] = []
    for source in sorted((queue / "pending").glob("*.json"))[:max_messages]:
        claimed = queue / "processing" / source.name
        try:
            descriptor = os.open(
                claimed, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
            )
        except FileExistsError:
            continue
        else:
            os.close(descriptor)
        try:
            os.replace(source, claimed)
        except FileNotFoundError:
            claimed.unlink(missing_ok=True)
            continue
        terminal_state = "completed"
        try:
            envelope = _load_envelope(claimed)
            if envelope["expected_production_commit"] != source_commit:
                raise TaskEvaluationEpisodeCompilationWorkerError(
                    "episode_compilation_source_commit_mismatch"
                )
            references = _verified_references(envelope, input_root=inputs)
            owned_output = outputs / envelope["compilation_id"]
            owned_output.mkdir(mode=0o750, exist_ok=False)
            compiler_output = _validated_compiler_output(
                episode_compiler(
                    envelope=envelope,
                    materialized_references=references,
                    output_root=owned_output,
                ),
                envelope=envelope,
                output_root=owned_output,
            )
            packet = compiler_output["compiled_episode_packet"]
            result: dict[str, Any] = {
                "schema_version": RESULT_SCHEMA_VERSION,
                "status": "compiled_for_production_launch",
                "compilation_id": envelope["compilation_id"],
                "run_id": envelope["run_id"],
                "team_namespace": envelope["team_namespace"],
                "source_commit": source_commit,
                "configured_scene_revision_digest": envelope[
                    "configured_scene_revision_digest"
                ],
                "compiled_episode_packet_digest": packet["digest"],
                "compiled_episode_packet_size_bytes": packet["size_bytes"],
                "compiler_output_digest": compiler_output[
                    "compiler_output_digest"
                ],
                "customer_supplied_prebuilt_episode_packet": False,
                "compiled_by_production": True,
                "provider_mutation_performed": False,
                "paid_execution_requested": False,
                "automatic_progression_required": True,
                "blockers": [],
                "result_digest": "",
            }
        except Exception as exc:
            terminal_state = "blocked"
            result = {
                "schema_version": RESULT_SCHEMA_VERSION,
                "status": "blocked",
                "compilation_id": re.sub(
                    r"-[0-9a-f]{64}\.json$", "", source.name
                ),
                "source_commit": source_commit,
                "provider_mutation_performed": False,
                "paid_execution_requested": False,
                "automatic_retry_performed": False,
                "blockers": [
                    str(exc)
                    if isinstance(
                        exc, TaskEvaluationEpisodeCompilationWorkerError
                    )
                    else f"episode_compilation_failed:{type(exc).__name__}"
                ],
                "result_digest": "",
            }
        result["result_digest"] = canonical_digest(
            result, digest_field="result_digest"
        )
        try:
            write_launch_preparation_record_exclusive(
                results_root / source.name, result
            )
        except FileExistsError:
            terminal_state = "blocked"
        os.replace(claimed, queue / terminal_state / source.name)
        processed.append(result)
    return {
        "schema_version": "task_evaluation_episode_compilation_queue_run.v1",
        "status": "processed" if processed else "idle",
        "processed_count": len(processed),
        "results": processed,
        "provider_mutation_performed": False,
        "paid_execution_requested": False,
        "automatic_retry_performed": False,
    }


__all__ = [
    "COMPILER_OUTPUT_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "TaskEvaluationEpisodeCompilationWorkerError",
    "process_episode_compilation_queue",
]
