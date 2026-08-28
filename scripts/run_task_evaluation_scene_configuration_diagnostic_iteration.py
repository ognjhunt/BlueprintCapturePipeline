#!/usr/bin/env python3
"""Stage and run one pushed-branch scene diagnostic retry without a deploy.

The command surface is fixed-purpose: it accepts paths and bounded authority
fields, never an arbitrary command or argv.  It stages source only, reuses the
operator-selected venv, toolchain, checkpoint, and splat-runtime identity by
reference, then invokes the existing bundle, authority, and canonical allocator
entrypoints from the detached release.  ``--execute`` remains the only switch
that can reach paid allocation.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess  # nosec B404 - fixed Python module commands only
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from blueprint_pipeline.core.common import redacted_failure_text
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_scene_configuration_bundle import (
    BUNDLE_SCHEMA_VERSION,
    PROBE_KIND,
)
from blueprint_pipeline.task_evaluation_scene_configuration_diagnostic_release import (
    SceneConfigurationDiagnosticReleaseError,
    stage_scene_configuration_diagnostic_release,
    validate_scene_configuration_diagnostic_release_receipt,
)
from blueprint_pipeline.task_evaluation_scene_configuration_diagnostic_mode import (
    CHECKPOINT_RESUME_DIAGNOSTIC_BOOTSTRAP_MODE,
    FRESH_DIAGNOSTIC_BOOTSTRAP_MODE,
)
from blueprint_pipeline.task_evaluation_scene_configuration_paid_authority import (
    AUTHORITY_SCHEMA_VERSION,
    SCENE_CONFIGURATION_PROVIDER_IMAGE,
)
from blueprint_pipeline.spend_authority_consumption_root import (
    SpendAuthorityRootError,
    prepare_consumption_root,
)
from blueprint_pipeline.task_evaluation_scene_configuration_warm_diagnostic import (
    materialize_scene_configuration_warm_session_authority,
)


PREPARATION_SCHEMA_VERSION = (
    "task_evaluation_scene_configuration_diagnostic_iteration_preparation.v1"
)
CommandRunner = Callable[..., subprocess.CompletedProcess[str]]

_OPENAI_RUNTIME_FILE_ENV_NAMES = (
    "OPENAI_ADMIN_API_KEY_FILE",
    "OPENAI_ARTIFIXER_SEMANTIC_TEACHER_API_KEY_FILE",
    "OPENAI_ARTIFIXER_VISUAL_REVIEW_API_KEY_FILE",
    "OPENAI_CONTENT_AGENTS_API_KEY_FILE",
    "BLUEPRINT_OPENAI_ARTIFIXER_SEMANTIC_TEACHER_COST_SCOPE_ATTESTATION_FILE",
    "BLUEPRINT_OPENAI_ARTIFIXER_VISUAL_REVIEW_COST_SCOPE_ATTESTATION_FILE",
    "BLUEPRINT_OPENAI_CONTENT_AGENTS_COST_SCOPE_ATTESTATION_FILE",
)
_OPENAI_RUNTIME_VALUE_ENV_NAMES = (
    "OPENAI_PROJECT_ID",
    "OPENAI_ARTIFIXER_SEMANTIC_TEACHER_API_KEY_ID",
    "OPENAI_ARTIFIXER_VISUAL_REVIEW_API_KEY_ID",
    "OPENAI_CONTENT_AGENTS_API_KEY_ID",
)
_CHILD_FAILURE_DETAIL_MAX_CHARS = 300


class SceneConfigurationDiagnosticIterationError(ValueError):
    """The fixed diagnostic iteration command could not be prepared safely."""


def _absolute(value: str | Path, *, field: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise SceneConfigurationDiagnosticIterationError(
            f"scene_configuration_diagnostic_iteration_{field}_must_be_absolute"
        )
    return path


def _input_file(value: str | Path, *, field: str) -> Path:
    path = _absolute(value, field=field)
    if path.is_symlink() or not path.is_file():
        raise SceneConfigurationDiagnosticIterationError(
            f"scene_configuration_diagnostic_iteration_{field}_invalid"
        )
    return path.resolve()


def _input_directory(value: str | Path, *, field: str) -> Path:
    path = _absolute(value, field=field)
    if path.is_symlink() or not path.is_dir():
        raise SceneConfigurationDiagnosticIterationError(
            f"scene_configuration_diagnostic_iteration_{field}_invalid"
        )
    return path.resolve()


def _python_executable(value: str | Path) -> Path:
    """Validate a Python entrypoint without discarding its venv identity."""

    path = _absolute(value, field="python_executable")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise SceneConfigurationDiagnosticIterationError(
            "scene_configuration_diagnostic_iteration_python_executable_invalid"
        ) from exc
    if not resolved.is_file() or not os.access(path, os.X_OK):
        raise SceneConfigurationDiagnosticIterationError(
            "scene_configuration_diagnostic_iteration_python_executable_invalid"
        )
    return path


def _output_path(value: str | Path, *, field: str) -> Path:
    path = _absolute(value, field=field)
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise SceneConfigurationDiagnosticIterationError(
            f"scene_configuration_diagnostic_iteration_{field}_invalid"
        )
    if path.parent.exists() and path.parent.is_symlink():
        raise SceneConfigurationDiagnosticIterationError(
            f"scene_configuration_diagnostic_iteration_{field}_parent_invalid"
        )
    return path


def _output_directory(value: str | Path, *, field: str, empty: bool) -> Path:
    path = _absolute(value, field=field)
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        raise SceneConfigurationDiagnosticIterationError(
            f"scene_configuration_diagnostic_iteration_{field}_invalid"
        )
    if empty and path.exists() and any(path.iterdir()):
        raise SceneConfigurationDiagnosticIterationError(
            f"scene_configuration_diagnostic_iteration_{field}_not_empty"
        )
    return path


def _read_json(path: Path, *, schema: str, code: str) -> dict[str, Any]:
    try:
        if path.is_symlink() or not path.is_file():
            raise OSError("unsafe JSON")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SceneConfigurationDiagnosticIterationError(code) from exc
    if not isinstance(value, Mapping) or value.get("schema_version") != schema:
        raise SceneConfigurationDiagnosticIterationError(code)
    return dict(value)


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o440,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short diagnostic iteration receipt write")
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o440)
    finally:
        os.close(descriptor)


def _discard_sealed_bundle_staging_tree(bundle_output: Path) -> bool:
    """Remove only the builder's self-created expanded tree after sealing."""

    staging = bundle_output / "stage"
    if not staging.exists() and not staging.is_symlink():
        return False
    if staging.is_symlink() or not staging.is_dir():
        raise SceneConfigurationDiagnosticIterationError(
            "scene_configuration_diagnostic_iteration_bundle_staging_invalid"
        )
    shutil.rmtree(staging)
    if staging.exists() or staging.is_symlink():
        raise SceneConfigurationDiagnosticIterationError(
            "scene_configuration_diagnostic_iteration_bundle_staging_cleanup_failed"
        )
    return True


def _run_fixed(
    argv: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    runner: CommandRunner,
    code: str,
) -> None:
    try:
        completed = runner(
            list(argv),
            cwd=str(cwd),
            env=dict(environment),
            check=False,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SceneConfigurationDiagnosticIterationError(code) from exc
    if completed.returncode != 0:
        # Keep the typed child refusal so the operator does not need to rerun a
        # paid path just to learn its cause.  Credential-shaped text and signed
        # URL query values are removed before the bounded detail is surfaced.
        detail = redacted_failure_text(completed.stderr or completed.stdout)
        detail = " ".join(detail.split())
        if len(detail) > _CHILD_FAILURE_DETAIL_MAX_CHARS:
            detail = detail[:_CHILD_FAILURE_DETAIL_MAX_CHARS] + "..."
        suffix = f":{detail}" if detail else f":exit_{completed.returncode}"
        raise SceneConfigurationDiagnosticIterationError(code + suffix)


def _preflight_paid_runtime_environment(
    environment: Mapping[str, str],
    *,
    execute: bool,
    openai_max_cost_usd: float,
) -> None:
    """Reject a malformed paid launch before bundle work or provider mutation."""

    if not execute or openai_max_cost_usd <= 0:
        return
    missing = [
        name
        for name in (*_OPENAI_RUNTIME_FILE_ENV_NAMES, *_OPENAI_RUNTIME_VALUE_ENV_NAMES)
        if not str(environment.get(name) or "").strip()
    ]
    if missing:
        raise SceneConfigurationDiagnosticIterationError(
            "scene_configuration_diagnostic_iteration_openai_runtime_environment_missing:"
            + ",".join(missing)
        )
    invalid_files: list[str] = []
    for name in _OPENAI_RUNTIME_FILE_ENV_NAMES:
        path = Path(str(environment[name])).expanduser()
        if (
            not path.is_absolute()
            or path.is_symlink()
            or not path.is_file()
            or not os.access(path, os.R_OK)
        ):
            invalid_files.append(name)
    if invalid_files:
        raise SceneConfigurationDiagnosticIterationError(
            "scene_configuration_diagnostic_iteration_openai_runtime_file_invalid:"
            + ",".join(invalid_files)
        )


def _preflight_paid_service_identity(*, execute: bool) -> None:
    """Reach the canonical single-use-ledger gate before expensive bundle work."""

    if not execute:
        return
    try:
        prepare_consumption_root()
    except SpendAuthorityRootError as exc:
        raise SceneConfigurationDiagnosticIterationError(
            "scene_configuration_diagnostic_iteration_spend_identity_invalid:"
            + str(exc)
        ) from exc


def run_scene_configuration_diagnostic_iteration(
    args: argparse.Namespace,
    *,
    runner: CommandRunner = subprocess.run,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Prepare the source overlay and invoke the fixed diagnostic chain."""

    _preflight_paid_runtime_environment(
        os.environ,
        execute=bool(args.execute),
        openai_max_cost_usd=float(args.openai_max_cost_usd),
    )
    _preflight_paid_service_identity(execute=bool(args.execute))

    source_repo = _input_directory(args.source_repo, field="source_repo")
    release_root = _output_directory(
        args.release_root, field="release_root", empty=False
    )
    state_root = _output_directory(args.state_root, field="state_root", empty=False)
    python = _python_executable(args.python_executable)
    construction_envelope = _input_file(
        args.construction_envelope, field="construction_envelope"
    )
    toolchain_root = _input_directory(args.toolchain_root, field="toolchain_root")
    splat_runtime = _input_directory(
        args.splat_render_runtime_root, field="splat_render_runtime_root"
    )
    fresh_diagnostic_bootstrap = bool(args.fresh_diagnostic_bootstrap)
    diagnostic_bootstrap_mode = (
        FRESH_DIAGNOSTIC_BOOTSTRAP_MODE
        if fresh_diagnostic_bootstrap
        else CHECKPOINT_RESUME_DIAGNOSTIC_BOOTSTRAP_MODE
    )
    if fresh_diagnostic_bootstrap:
        if args.diagnostic_checkpoint_reference:
            raise SceneConfigurationDiagnosticIterationError(
                "scene_configuration_diagnostic_iteration_checkpoint_source_ambiguous"
            )
        checkpoint_reference = None
    else:
        checkpoint_reference = _input_file(
            args.diagnostic_checkpoint_reference,
            field="diagnostic_checkpoint_reference",
        )
    project_spend = _input_file(
        args.project_spend_reconciliation,
        field="project_spend_reconciliation",
    )
    provider_zero = _input_file(
        args.initial_provider_zero, field="initial_provider_zero"
    )
    bundle_output = _output_directory(
        args.bundle_output_root, field="bundle_output_root", empty=True
    )
    authority_output = _output_path(
        args.scene_configuration_attempt_authority,
        field="scene_configuration_attempt_authority",
    )
    job_dir = _output_directory(
        args.scene_configuration_job_dir,
        field="scene_configuration_job_dir",
        empty=False,
    )
    admission_output = _output_path(args.admission_out, field="admission_out")
    adapter_output = _output_path(args.adapter_output, field="adapter_output")
    preparation_output = _output_path(
        args.iteration_preparation_receipt,
        field="iteration_preparation_receipt",
    )
    retain_warm_session = bool(getattr(args, "retain_warm_session", False))
    allowed_machine_values = getattr(args, "allowed_vast_machine_id", ()) or ()
    try:
        if any(isinstance(value, bool) for value in allowed_machine_values):
            raise ValueError("boolean machine id")
        allowed_machine_ids = tuple(
            sorted({int(value) for value in allowed_machine_values})
        )
        if any(value <= 0 for value in allowed_machine_ids):
            raise ValueError("non-positive machine id")
    except (TypeError, ValueError) as exc:
        raise SceneConfigurationDiagnosticIterationError(
            "scene_configuration_diagnostic_iteration_allowed_vast_machine_ids_invalid"
        ) from exc
    if retain_warm_session and not args.execute:
        raise SceneConfigurationDiagnosticIterationError(
            "scene_configuration_diagnostic_iteration_warm_retention_requires_execute"
        )
    warm_authority_output: Path | None = None
    warm_session_output_root: Path | None = None
    if retain_warm_session:
        if not args.warm_session_authority or not args.warm_session_output_root:
            raise SceneConfigurationDiagnosticIterationError(
                "scene_configuration_diagnostic_iteration_warm_outputs_missing"
            )
        warm_authority_output = _output_path(
            args.warm_session_authority,
            field="warm_session_authority",
        )
        warm_session_output_root = _output_directory(
            args.warm_session_output_root,
            field="warm_session_output_root",
            empty=True,
        )
        if warm_session_output_root.exists():
            raise SceneConfigurationDiagnosticIterationError(
                "scene_configuration_diagnostic_iteration_warm_session_output_root_exists"
            )

    preparation_started = clock()
    stage_started = preparation_started
    staged = stage_scene_configuration_diagnostic_release(
        source_repo=source_repo,
        source_commit=args.source_commit,
        remote_branch=args.remote_branch,
        release_root=release_root,
        state_root=state_root,
    )
    stage_elapsed_ms = int((clock() - stage_started) * 1000)
    release_path = Path(staged["release_path"])
    release_receipt = Path(staged["receipt_path"])
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(release_path / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"

    bundle_started = clock()
    bundle_command = [
        str(python),
        "-m",
        "blueprint_pipeline.task_evaluation_scene_configuration_bundle",
        "--construction-envelope",
        str(construction_envelope),
        "--toolchain-root",
        str(toolchain_root),
        "--repository-root",
        str(release_path),
        "--splat-render-runtime-root",
        str(splat_runtime),
        "--output-root",
        str(bundle_output),
        "--expected-source-commit",
        args.source_commit,
    ]
    if fresh_diagnostic_bootstrap:
        bundle_command.append("--fresh-diagnostic-bootstrap")
    else:
        bundle_command.extend(
            ["--diagnostic-checkpoint-reference", str(checkpoint_reference)]
        )
    _run_fixed(
        bundle_command,
        cwd=release_path,
        environment=environment,
        runner=runner,
        code="scene_configuration_diagnostic_iteration_bundle_failed",
    )
    bundle_elapsed_ms = int((clock() - bundle_started) * 1000)
    bundle_receipt_path = (
        bundle_output / f"{BUNDLE_SCHEMA_VERSION}.receipt.json"
    )
    bundle_receipt = _read_json(
        bundle_receipt_path,
        schema=BUNDLE_SCHEMA_VERSION,
        code="scene_configuration_diagnostic_iteration_bundle_receipt_invalid",
    )
    if (
        bundle_receipt.get("source_commit") != args.source_commit
        or bundle_receipt.get("diagnostic_only") is not True
        or bundle_receipt.get("qualification_eligible") is not False
        or bundle_receipt.get("configured_revision_publication_permitted") is not False
        or bundle_receipt.get("offering_publication_permitted") is not False
        or bundle_receipt.get("terminal_e2e_completion_permitted") is not False
        or bundle_receipt.get("diagnostic_bootstrap_mode")
        != diagnostic_bootstrap_mode
        or (
            fresh_diagnostic_bootstrap
            and (
                bundle_receipt.get("source_diagnostic_checkpoint_digest")
                is not None
                or bundle_receipt.get("carried_completed_stage_count") != 0
            )
        )
    ):
        raise SceneConfigurationDiagnosticIterationError(
            "scene_configuration_diagnostic_iteration_bundle_receipt_invalid"
        )
    bundle_staging_tree_removed = _discard_sealed_bundle_staging_tree(bundle_output)
    authority_started = clock()
    authority_command = [
        str(python),
        "-m",
        "blueprint_pipeline.task_evaluation_scene_configuration_paid_authority",
        "--bundle-receipt",
        str(bundle_receipt_path),
        "--project-spend-reconciliation",
        str(project_spend),
        "--initial-provider-zero",
        str(provider_zero),
        "--authorization-reference",
        args.authorization_reference,
        "--authorized-by",
        args.authorized_by,
        "--authorized-on",
        args.authorized_on,
        "--source-commit",
        args.source_commit,
        "--container-image",
        SCENE_CONFIGURATION_PROVIDER_IMAGE,
        "--resource-name",
        args.pod_name,
        "--max-hourly-rate-usd",
        str(args.max_hourly_rate_usd),
        "--hard-cap-usd",
        str(args.hard_cap_usd),
        "--hard-ttl-seconds",
        str(args.hard_ttl_seconds),
        "--provider-compute-spend-cap-usd",
        str(args.provider_compute_spend_cap_usd),
        "--openai-max-cost-usd",
        str(args.openai_max_cost_usd),
        "--openai-max-requests",
        str(args.openai_max_requests),
        "--openai-artifixer-semantic-teacher-max-cost-usd",
        str(args.openai_artifixer_semantic_teacher_max_cost_usd),
        "--openai-artifixer-visual-review-max-cost-usd",
        str(args.openai_artifixer_visual_review_max_cost_usd),
        "--openai-content-agents-max-cost-usd",
        str(args.openai_content_agents_max_cost_usd),
        "--output",
        str(authority_output),
    ]
    _run_fixed(
        authority_command,
        cwd=release_path,
        environment=environment,
        runner=runner,
        code="scene_configuration_diagnostic_iteration_authority_failed",
    )
    authority_finished = clock()
    authority_elapsed_ms = int((authority_finished - authority_started) * 1000)
    total_preparation_elapsed_ms = int(
        (authority_finished - preparation_started) * 1000
    )
    authority = _read_json(
        authority_output,
        schema=AUTHORITY_SCHEMA_VERSION,
        code="scene_configuration_diagnostic_iteration_authority_invalid",
    )
    if (
        authority.get("source_commit") != args.source_commit
        or authority.get("bundle_sha256") != bundle_receipt.get("bundle_sha256")
        or authority.get("diagnostic_only") is not True
        or authority.get("qualification_eligible") is not False
        or authority.get("configured_revision_publication_permitted") is not False
        or authority.get("offering_publication_permitted") is not False
        or authority.get("terminal_e2e_completion_permitted") is not False
        or authority.get("diagnostic_bootstrap_mode")
        != diagnostic_bootstrap_mode
    ):
        raise SceneConfigurationDiagnosticIterationError(
            "scene_configuration_diagnostic_iteration_authority_invalid"
        )
    if retain_warm_session:
        checkpoint_root: Path | None = None
        if not fresh_diagnostic_bootstrap:
            try:
                checkpoint_reference_value = json.loads(
                    checkpoint_reference.read_text(encoding="utf-8")
                )
                checkpoint_root = _input_directory(
                    str(checkpoint_reference_value.get("checkpoint_root") or ""),
                    field="diagnostic_checkpoint_root",
                )
            except (
                AttributeError,
                json.JSONDecodeError,
                OSError,
                UnicodeError,
            ) as exc:
                raise SceneConfigurationDiagnosticIterationError(
                    "scene_configuration_diagnostic_iteration_checkpoint_reference_invalid"
                ) from exc
        materialize_scene_configuration_warm_session_authority(
            bundle_receipt_path=bundle_receipt_path,
            paid_attempt_authority_path=authority_output,
            diagnostic_release_receipt_path=release_receipt,
            checkpoint_root=checkpoint_root,
            maximum_warm_iterations=args.maximum_warm_iterations,
            output_path=warm_authority_output,
        )

    preparation: dict[str, Any] = {
        "schema_version": PREPARATION_SCHEMA_VERSION,
        "status": "ready_for_diagnostic_allocator",
        "program_id": "arm-decision-proof-v1",
        "day_gate": "day-28",
        "probe_kind": PROBE_KIND,
        "source_commit": args.source_commit,
        "remote_ref": staged["remote_ref"],
        "diagnostic_release_receipt": {
            "path": str(release_receipt),
            "receipt_digest": staged["receipt_digest"],
        },
        "release_path": str(release_path),
        "source_only_release": True,
        "source_checkout_reused": staged["reused_existing_checkout"],
        "source_materialization_elapsed_ms": stage_elapsed_ms,
        "source_materialization_target_ms": 5_000,
        "source_materialization_target_met": stage_elapsed_ms < 5_000,
        "total_preparation_elapsed_ms": total_preparation_elapsed_ms,
        "total_preparation_seconds_claimed": False,
        "bundle_build_elapsed_ms": bundle_elapsed_ms,
        "bundle_staging_tree_removed_after_seal": bundle_staging_tree_removed,
        "authority_build_elapsed_ms": authority_elapsed_ms,
        "bundle_receipt_digest": bundle_receipt.get("receipt_digest"),
        "authority_digest": authority.get("authority_digest"),
        "splat_runtime_reused_by_reference": True,
        "splat_runtime_copied": False,
        "remaining_preparation_bottleneck": {
            "diagnostic_bundle_rebuilt_for_exact_source_commit": True,
            "toolchain_tree_copied_and_provider_zip_rebuilt": True,
            "unsafe_hardlink_optimization_used": False,
            "reason": (
                "the existing bundle builder seals modes and byte inventories; "
                "hardlinking its mutable staging tree to shared inputs would let "
                "chmod or later corruption change the shared runtime"
            ),
        },
        "active_release_link_updated": False,
        "systemd_units_reinstalled": False,
        "systemd_services_restarted": False,
        "diagnostic_only": True,
        "development_only": True,
        "qualification_eligible": False,
        "configured_revision_publication_permitted": False,
        "offering_publication_permitted": False,
        "terminal_e2e_completion_permitted": False,
        "paid_execution_requested": bool(args.execute),
        "warm_session_retention_requested": retain_warm_session,
        "allowed_vast_machine_ids": list(allowed_machine_ids),
        "diagnostic_bootstrap_mode": diagnostic_bootstrap_mode,
        "provider_mutation_performed_during_preparation": False,
        "raw_secret_values_recorded": False,
        "preparation_digest": "",
    }
    preparation["preparation_digest"] = canonical_digest(
        preparation, digest_field="preparation_digest"
    )
    _write_exclusive(preparation_output, preparation)

    # This is intentionally the last observation before the canonical
    # allocator subprocess.  A force-push or checkout edit during bundle or
    # authority preparation therefore fails before paid admission.
    validate_scene_configuration_diagnostic_release_receipt(
        release_receipt,
        expected_source_commit=args.source_commit,
        expected_release_path=release_path,
    )
    allocator_command = [
        str(python),
        "-m",
        "blueprint_pipeline.paid_resource_allocator",
        "gpu-canary",
        "--provider",
        "vast",
        "--probe-kind",
        PROBE_KIND,
        "--expected-source-commit",
        args.source_commit,
        "--experimental-branch-diagnostic",
        "--scene-configuration-diagnostic-only",
        "--release-evidence",
        str(release_receipt),
        "--scene-configuration-bundle-receipt",
        str(bundle_receipt_path),
        "--scene-configuration-attempt-authority",
        str(authority_output),
        "--scene-configuration-job-dir",
        str(job_dir),
        "--pod-name",
        args.pod_name,
        "--admission-out",
        str(admission_output),
        "--adapter-output",
        str(adapter_output),
    ]
    if args.execute:
        allocator_command.append("--execute")
    for machine_id in allowed_machine_ids:
        allocator_command.extend(
            ["--scene-configuration-allowed-vast-machine-id", str(machine_id)]
        )
    if retain_warm_session:
        allocator_command.extend(
            [
                "--scene-configuration-retain-warm-session",
                "--scene-configuration-warm-session-authority",
                str(warm_authority_output),
                "--scene-configuration-warm-session-output-root",
                str(warm_session_output_root),
            ]
        )
    _run_fixed(
        allocator_command,
        cwd=release_path,
        environment=environment,
        runner=runner,
        code="scene_configuration_diagnostic_iteration_allocator_failed",
    )
    return preparation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--remote-branch", required=True)
    parser.add_argument("--release-root", required=True)
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--python-executable", required=True)
    parser.add_argument("--construction-envelope", required=True)
    parser.add_argument("--toolchain-root", required=True)
    parser.add_argument("--splat-render-runtime-root", required=True)
    parser.add_argument("--diagnostic-checkpoint-reference")
    parser.add_argument("--fresh-diagnostic-bootstrap", action="store_true")
    parser.add_argument("--bundle-output-root", required=True)
    parser.add_argument("--project-spend-reconciliation", required=True)
    parser.add_argument("--initial-provider-zero", required=True)
    parser.add_argument("--authorization-reference", required=True)
    parser.add_argument("--authorized-by", required=True)
    parser.add_argument("--authorized-on", required=True)
    parser.add_argument("--pod-name", required=True)
    parser.add_argument("--max-hourly-rate-usd", required=True, type=float)
    parser.add_argument("--hard-cap-usd", required=True, type=float)
    parser.add_argument("--hard-ttl-seconds", required=True, type=int)
    parser.add_argument("--provider-compute-spend-cap-usd", required=True, type=float)
    parser.add_argument("--openai-max-cost-usd", type=float, default=0.0)
    parser.add_argument("--openai-max-requests", type=int, default=0)
    parser.add_argument(
        "--openai-artifixer-semantic-teacher-max-cost-usd", type=float, default=0.0
    )
    parser.add_argument(
        "--openai-artifixer-visual-review-max-cost-usd", type=float, default=0.0
    )
    parser.add_argument("--openai-content-agents-max-cost-usd", type=float, default=0.0)
    parser.add_argument("--scene-configuration-attempt-authority", required=True)
    parser.add_argument("--scene-configuration-job-dir", required=True)
    parser.add_argument("--admission-out", required=True)
    parser.add_argument("--adapter-output", required=True)
    parser.add_argument("--iteration-preparation-receipt", required=True)
    parser.add_argument("--retain-warm-session", action="store_true")
    parser.add_argument("--allowed-vast-machine-id", action="append", type=int, default=[])
    parser.add_argument("--warm-session-authority")
    parser.add_argument("--warm-session-output-root")
    parser.add_argument("--maximum-warm-iterations", type=int, default=8)
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run_scene_configuration_diagnostic_iteration(args)
    except (
        OSError,
        SceneConfigurationDiagnosticIterationError,
        SceneConfigurationDiagnosticReleaseError,
    ) as exc:
        print(
            json.dumps(
                {
                    "schema_version": PREPARATION_SCHEMA_VERSION,
                    "status": "blocked",
                    "blockers": [str(exc)],
                    "provider_mutations_performed": 0,
                    "diagnostic_only": True,
                    "qualification_eligible": False,
                    "configured_revision_publication_permitted": False,
                    "offering_publication_permitted": False,
                    "terminal_e2e_completion_permitted": False,
                    "raw_secret_values_recorded": False,
                },
                sort_keys=True,
            )
        )
        return 2
    print(
        json.dumps(
            {
                "schema_version": PREPARATION_SCHEMA_VERSION,
                "status": "allocator_invoked",
                "source_commit": result["source_commit"],
                "source_materialization_elapsed_ms": result[
                    "source_materialization_elapsed_ms"
                ],
                "source_checkout_reused": result["source_checkout_reused"],
                "diagnostic_only": True,
                "qualification_eligible": False,
                "raw_secret_values_recorded": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
