"""Production worker that compiles one configured scene plus team inputs.

The website never supplies a native-Arena packet.  It supplies one configured
scene revision and independently versioned robot, controller or policy, sensor,
and runtime references.  This worker verifies those materialized bytes and
invokes the installed production compiler exactly once to create the internal
episode packet used by the existing Task Evaluation launch path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import zipfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .control_plane_disk_budget import (
    ControlPlaneDiskBudgetError,
    DiskReservation,
    reserve_control_plane_disk,
)
from .control_plane_storage_pins import (
    PINS_ROOT_ENV,
    ControlPlaneStoragePinError,
    write_storage_pin,
)
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
from .task_evaluation_native_arena_episode_compiler import (
    compile_native_arena_episode,
)
from .task_evaluation_native_arena_preparation_adapter import (
    MANIFEST_NAME,
    TaskEvaluationNativeArenaAdapterError,
)


COMPILER_OUTPUT_SCHEMA_VERSION = "task_evaluation_episode_compiler_output.v1"
RESULT_SCHEMA_VERSION = "task_evaluation_episode_compilation_result.v1"
EpisodeCompiler = Callable[..., Mapping[str, Any]]
QUEUE_ROOT_ENV = "BLUEPRINT_TASK_EVALUATION_EPISODE_COMPILATION_QUEUE_ROOT"
INPUT_ROOT_ENV = "BLUEPRINT_TASK_EVALUATION_LAUNCH_PREPARATION_INPUT_ROOT"
OUTPUT_ROOT_ENV = "BLUEPRINT_TASK_EVALUATION_EPISODE_COMPILATION_OUTPUT_ROOT"


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
        request["run_mode"] not in {"episode_evaluation", "destination_qualification"}
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
        if contract_path in verified:
            existing = verified[contract_path]
            if (row.get("digest"), row.get("size_bytes")) != (
                existing.get("digest"),
                existing.get("size_bytes"),
            ):
                raise TaskEvaluationEpisodeCompilationWorkerError(
                    "episode_compilation_materialized_reference_invalid"
                )
            continue
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
    adapter_artifact = output.get("adapter_result")
    adapter_path = Path(
        str((adapter_artifact or {}).get("path") or "")
    ).resolve()
    try:
        adapter_result = json.loads(adapter_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        adapter_result = {}
    probe_artifact = output.get("destination_native_probe_request")
    qualification_only = envelope["request"].get("run_mode") == "destination_qualification"
    probe_path = Path(
        str((probe_artifact or {}).get("path") or "")
    ).resolve()
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
        or not isinstance(adapter_artifact, Mapping)
        or not _under(adapter_path, output_root)
        or adapter_path.is_symlink()
        or not adapter_path.is_file()
        or adapter_result.get("status")
        != "native_arena_adapter_materialized"
        or adapter_result.get("result_digest")
        != adapter_artifact.get("digest")
        or adapter_result.get("result_digest")
        != canonical_digest(adapter_result, digest_field="result_digest")
        or adapter_result.get("packet_receipt_digest")
        != adapter_artifact.get("packet_receipt_digest")
        or adapter_result.get("runtime_source_receipt_digest")
        != adapter_artifact.get("runtime_source_receipt_digest")
        or qualification_only != isinstance(probe_artifact, Mapping)
        or (
            qualification_only
            and (
                not _under(probe_path, output_root)
                or probe_path.is_symlink()
                or not probe_path.is_file()
                or _sha256_and_size(probe_path)
                != (probe_artifact.get("digest"), probe_artifact.get("size_bytes"))
            )
        )
    ):
        raise TaskEvaluationEpisodeCompilationWorkerError(
            "episode_compiler_output_invalid"
        )
    return output


# Packet materialization, the built construction archive, and adapter receipts;
# runtime members are counted only when the shared member store lacks them.
COMPILATION_RESERVATION_MARGIN_BYTES = 2 * 1024**3


def _expected_compilation_bytes(
    references: Mapping[str, Mapping[str, Any]], *, content_store_root: Path
) -> int:
    """Bytes this compile will write: runtime members the member store lacks."""

    total = COMPILATION_RESERVATION_MARGIN_BYTES
    row = references.get("execution_adapter.runtime_source_bundle")
    if row is None:
        return total
    fallback = total + int(row.get("size_bytes") or 0)
    try:
        with zipfile.ZipFile(Path(str(row["materialized_path"]))) as archive:
            manifest = json.loads(archive.read(MANIFEST_NAME).decode("utf-8"))
    except (OSError, KeyError, ValueError, zipfile.BadZipFile):
        return fallback
    entries = manifest.get("entries") if isinstance(manifest, Mapping) else None
    if not isinstance(entries, list):
        return fallback
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        digest = str(entry.get("sha256") or "").removeprefix("sha256:")
        size = entry.get("size_bytes")
        if not digest or not isinstance(size, int) or isinstance(size, bool):
            continue
        if not (content_store_root / digest).is_file():
            total += size
    return total


def process_episode_compilation_queue(
    *,
    queue_root: str | Path,
    input_root: str | Path,
    output_root: str | Path,
    source_commit: str,
    episode_compiler: EpisodeCompiler = compile_native_arena_episode,
    max_messages: int = 1,
    disk_reservation_root: str | Path | None = None,
    storage_pins_root: str | Path | None = None,
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
        owned_output: Path | None = None
        disk_reservation: DiskReservation | None = None
        try:
            envelope = _load_envelope(claimed)
            if envelope["expected_production_commit"] != source_commit:
                raise TaskEvaluationEpisodeCompilationWorkerError(
                    "episode_compilation_source_commit_mismatch"
                )
            references = _verified_references(envelope, input_root=inputs)
            if disk_reservation_root is not None:
                try:
                    disk_reservation = reserve_control_plane_disk(
                        "episode_compilation",
                        target_root=outputs,
                        expected_bytes=_expected_compilation_bytes(
                            references,
                            content_store_root=(
                                outputs / "content-addressed" / "adapter-members" / "sha256"
                            ),
                        ),
                        reservation_root=disk_reservation_root,
                    )
                except ControlPlaneDiskBudgetError as exc:
                    raise TaskEvaluationEpisodeCompilationWorkerError(
                        str(exc)
                    ) from exc
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
                "compiled_episode_packet_path": packet["path"],
                "adapter_result_path": compiler_output["adapter_result"]["path"],
                "adapter_result_digest": compiler_output["adapter_result"][
                    "digest"
                ],
                "compiler_output_digest": compiler_output[
                    "compiler_output_digest"
                ],
                **(
                    {
                        "destination_native_probe_request_path": compiler_output[
                            "destination_native_probe_request"
                        ]["path"],
                        "destination_native_probe_request_digest": compiler_output[
                            "destination_native_probe_request"
                        ]["digest"],
                        "destination_native_probe_request_document_digest": compiler_output[
                            "destination_native_probe_request"
                        ]["request_digest"],
                    }
                    if "destination_native_probe_request" in compiler_output
                    else {}
                ),
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
            cleanup_blockers: list[str] = []
            if owned_output is not None and owned_output.exists():
                try:
                    shutil.rmtree(owned_output)
                except OSError as cleanup_exc:
                    cleanup_errno = (
                        cleanup_exc.errno
                        if isinstance(cleanup_exc.errno, int)
                        else "unknown"
                    )
                    cleanup_blockers.append(
                        "episode_compilation_partial_cleanup_failed:"
                        f"errno_{cleanup_errno}"
                    )
            if isinstance(
                exc,
                (
                    TaskEvaluationEpisodeCompilationWorkerError,
                    TaskEvaluationNativeArenaAdapterError,
                ),
            ):
                primary_blocker = str(exc)
            elif isinstance(exc, OSError):
                error_number = exc.errno if isinstance(exc.errno, int) else "unknown"
                primary_blocker = (
                    f"episode_compilation_failed:OSError:errno_{error_number}"
                )
            else:
                primary_blocker = f"episode_compilation_failed:{type(exc).__name__}"
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
                "blockers": [primary_blocker, *cleanup_blockers],
                "result_digest": "",
            }
        if disk_reservation is not None:
            disk_reservation.release()
        if (
            storage_pins_root is not None
            and owned_output is not None
            and result.get("status") == "compiled_for_production_launch"
        ):
            try:
                write_storage_pin(
                    pins_root=storage_pins_root,
                    kind="compilation",
                    owner_id=str(result["compilation_id"]),
                    paths=[owned_output],
                    depends_on=[
                        {"kind": "preparation", "owner_id": str(envelope["preparation_id"])}
                    ],
                )
            except (ControlPlaneStoragePinError, OSError, KeyError):
                pass
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile queued Task Evaluation episodes in production"
    )
    parser.add_argument("--queue-root", default=os.getenv(QUEUE_ROOT_ENV, ""))
    parser.add_argument("--input-root", default=os.getenv(INPUT_ROOT_ENV, ""))
    parser.add_argument("--output-root", default=os.getenv(OUTPUT_ROOT_ENV, ""))
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--max-messages", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.queue_root or not args.input_root or not args.output_root:
        raise SystemExit("episode compilation roots are required")
    result = process_episode_compilation_queue(
        queue_root=args.queue_root,
        input_root=args.input_root,
        output_root=args.output_root,
        source_commit=args.source_commit,
        max_messages=args.max_messages,
        disk_reservation_root=os.getenv(
            "BLUEPRINT_CONTROL_PLANE_DISK_RESERVATION_ROOT"
        ),
        storage_pins_root=os.getenv(PINS_ROOT_ENV),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


__all__ = [
    "COMPILER_OUTPUT_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "TaskEvaluationEpisodeCompilationWorkerError",
    "main",
    "process_episode_compilation_queue",
]


if __name__ == "__main__":
    raise SystemExit(main())
