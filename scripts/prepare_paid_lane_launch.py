#!/usr/bin/env python3
"""Prepare one paid lane for a website-triggered launch in a single command.

Every artifact a paid launch consumes is bound to one exact source commit, so
each deploy invalidates the previous set and the whole sequence -- bundle,
immutable manifest, single-use authority, unpaid dry run, live profile,
publication, terminal rehearsal -- has to run again in order.  Reconstructing
that order by hand costs most of an hour per deploy and is where a wrong path
or a stale digest gets in: on 2026-08-16 a rebuild consumed a packet directory
that held the manifest JSON but not the frames it names, and the mistake only
surfaced two steps later.

The order is the contract, so the order lives here as data.  Placeholders are
resolved against one operator-supplied context and every step is validated
before any step runs, so an unsupplied value fails the command instead of
leaving a half-prepared lane behind.  ``--validate-only`` seals that resolved
plan without running a subprocess.  A step that exits non-zero, or that exits
zero without writing the artifact it declares, stops the sequence with its
outputs preserved.

This performs no provider allocation and no paid inference.  The dry run reads
provider inventory and the publication writes one object; neither rents
anything.  Authorization text is never defaulted -- an authority is a human
statement and is required explicitly.
"""

from __future__ import annotations

import argparse
import grp
import hashlib
import json
import os
import pwd
import re
import stat
import string
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


from blueprint_pipeline.core.common import redacted_failure_text

PREPARATION_SCHEMA_VERSION = "paid_lane_launch_preparation.v1"
VALIDATION_SCHEMA_VERSION = "paid_lane_launch_validation.v1"
SCENE_CONFIGURATION_CONTEXT_SCHEMA_VERSION = (
    "task_evaluation_scene_configuration_launch_preparation_context.v1"
)
DEFAULT_SERVICE_ACCOUNT = "blueprint"
DEFAULT_SERVICE_GROUP = "blueprint"


@dataclass(frozen=True)
class LaneStep:
    """One ordered command plus the artifact it must leave behind."""

    step_id: str
    argv: tuple[str, ...]
    produces: str
    exports: tuple[tuple[str, str], ...] = ()
    repeated_argv: tuple[tuple[str, str], ...] = ()
    conditional_argv: tuple[tuple[str, str], ...] = ()
    #: Context names this step exists to consume. When the context supplies
    #: none of them, the step is not part of this lane run at all: it is
    #: skipped by the validator, the plan, and the executor alike, so an
    #: optional input never becomes an empty placeholder in a real argv.
    requires_context: tuple[str, ...] = ()


def _scene_configuration_steps() -> tuple[LaneStep, ...]:
    """Return the no-allocation graph for one Website scene configuration."""

    bundle_receipt = (
        "{set_root}/bundle/"
        "task_evaluation_scene_configuration_provider_bundle.v1.receipt.json"
    )
    authority = (
        "{set_root}/task_evaluation_scene_configuration_paid_authority.v1.json"
    )
    profile = "{set_root}/live_profile-{revision}.v1.json"
    autostart_intent = (
        "{set_root}/task_evaluation_configured_controls_autostart_intent.v1.json"
    )
    return (
        LaneStep(
            step_id="provider_bundle",
            argv=(
                "{python}",
                "-m",
                "blueprint_pipeline.task_evaluation_scene_configuration_bundle",
                "--construction-envelope",
                "{construction_envelope}",
                "--toolchain-root",
                "{toolchain_root}",
                "--repository-root",
                "{repository_root}",
                "--output-root",
                "{set_root}/bundle",
                "--expected-source-commit",
                "{source_commit}",
            ),
            produces=bundle_receipt,
            repeated_argv=(
                ("--partial-astra-successor-selection", "partial_astra_successor_selection"),
                ("--artisan-continuation-selection", "artisan_continuation_selection"),
                ("--scene-owner-attempt", "scene_owner_attempt"),
                ("--expected-toolchain-digest", "expected_toolchain_digest"),
                (
                    "--production-semantic-reuse-checkpoint-root",
                    "production_semantic_reuse_checkpoint_root",
                ),
                (
                    "--production-semantic-reuse-queue-root",
                    "production_semantic_reuse_queue_root",
                ),
                (
                    "--production-semantic-reuse-revision-id",
                    "production_semantic_reuse_revision_id",
                ),
            ),
        ),
        LaneStep(
            step_id="immutable_manifest",
            argv=(
                "{python}",
                "{repository_root}/scripts/"
                "publish_task_evaluation_immutable_manifest.py",
                "--manifest",
                bundle_receipt,
                "--profile-builder",
                "build_task_evaluation_scene_configuration_live_profile.py",
                "--destination-prefix",
                "{destination_prefix}",
                "--output",
                "{set_root}/manifest_publication_receipt.v1.json",
            ),
            produces="{set_root}/manifest_publication_receipt.v1.json",
            exports=(("published_manifest_uri", "published_uri"),),
        ),
        LaneStep(
            step_id="paid_authority",
            argv=(
                "{python}",
                "-m",
                "blueprint_pipeline.task_evaluation_scene_configuration_paid_authority",
                "--bundle-receipt",
                bundle_receipt,
                "--project-spend-reconciliation",
                "{project_spend_reconciliation}",
                "--initial-provider-zero",
                "{initial_provider_zero}",
                "--authorization-reference",
                "{authorization_reference}",
                "--authorized-by",
                "{authorized_by}",
                "--authorized-on",
                "{authorized_on}",
                "--source-commit",
                "{source_commit}",
                "--container-image",
                "{container_image}",
                "--resource-name",
                "{pod_name}",
                "--max-hourly-rate-usd",
                "{maximum_hourly_rate_usd}",
                "--hard-cap-usd",
                "{hard_total_spend_cap_usd}",
                "--provider-compute-spend-cap-usd",
                "{provider_compute_spend_cap_usd}",
                "--openai-max-cost-usd",
                "{openai_max_cost_usd}",
                "--openai-max-requests",
                "{openai_max_requests}",
                "--openai-artifixer-semantic-teacher-max-cost-usd",
                "{openai_artifixer_semantic_teacher_max_cost_usd}",
                "--openai-artifixer-visual-review-max-cost-usd",
                "{openai_artifixer_visual_review_max_cost_usd}",
                "--openai-content-agents-max-cost-usd",
                "{openai_content_agents_max_cost_usd}",
                "--anthropic-max-cost-usd",
                "{anthropic_max_cost_usd}",
                "--anthropic-max-requests",
                "{anthropic_max_requests}",
                "--hard-ttl-seconds",
                "{hard_ttl_seconds}",
                "--output",
                authority,
            ),
            produces=authority,
        ),
        LaneStep(
            step_id="allocator_dry_run",
            argv=(
                "{python}",
                "-m",
                "blueprint_pipeline.paid_resource_allocator",
                "gpu-canary",
                "--admission-out",
                "{set_root}/dry-run-job/admission.json",
                "--bound-request-out",
                "{set_root}/dry-run-job/bound-request.json",
                "--adapter-output",
                "{set_root}/allocator_dry_run.v1.json",
                "--pod-name",
                "{pod_name}",
                "--expected-source-commit",
                "{source_commit}",
                "--provider",
                "vast",
                "--probe-kind",
                "task-evaluation-scene-configuration",
                "--scene-configuration-bundle-receipt",
                bundle_receipt,
                "--scene-configuration-attempt-authority",
                authority,
                "--scene-configuration-job-dir",
                "{set_root}/dry-run-job/scene-configuration-job",
            ),
            produces="{set_root}/allocator_dry_run.v1.json",
            repeated_argv=(
                (
                    "--scene-configuration-queue-root",
                    "production_semantic_reuse_queue_root",
                ),
            ),
        ),
        LaneStep(
            step_id="configured_controls_autostart_intent",
            argv=(
                "{python}",
                "-m",
                "blueprint_pipeline.task_evaluation_configured_controls_autostart",
                "--intent",
                "{configured_controls_autostart_intent_source}",
                "--expected-production-commit",
                "{source_commit}",
                "--team-namespace",
                "{team_namespace}",
                "--scene-id",
                "{scene_id}",
                "--task-id",
                "{task_id}",
                "--output",
                autostart_intent,
            ),
            produces=autostart_intent,
            requires_context=("configured_controls_autostart_intent_source",),
        ),
        LaneStep(
            step_id="live_profile",
            argv=(
                "{python}",
                "{repository_root}/scripts/"
                "build_task_evaluation_scene_configuration_live_profile.py",
                "--bundle-receipt",
                bundle_receipt,
                "--attempt-authority",
                authority,
                "--source-commit",
                "{source_commit}",
                "--raw-manifest-uri",
                "{set_root}/manifest_publication_receipt.v1.json",
                "--revision",
                "{revision}",
                "--max-hourly-rate-usd",
                "{maximum_hourly_rate_usd}",
                "--hard-ttl-seconds",
                "{hard_ttl_seconds}",
                "--max-spend-usd",
                "{hard_total_spend_cap_usd}",
                "--team-namespace",
                "{team_namespace}",
                "--scene-id",
                "{scene_id}",
                "--task-id",
                "{task_id}",
                "--pod-name",
                "{pod_name}",
                "--output",
                profile,
            ),
            produces=profile,
            repeated_argv=(
                (
                    "--configured-controls-autostart-intent",
                    "configured_controls_autostart_intent_artifacts",
                ),
                ("--scene-owner-attempt", "scene_owner_attempt"),
            ),
            exports=(("profile_id", "profile_id"),),
        ),
        LaneStep(
            step_id="terminal_rehearsal",
            argv=(
                "{python}",
                "{repository_root}/scripts/rehearse_lane_terminal_contract.py",
                "--profile",
                profile,
                "--lane-module",
                "task_evaluation_scene_configuration_vast.py",
                "--lane",
                "task_evaluation_scene_configuration",
                "--receipt-out",
                "{set_root}/terminal_rehearsal-{revision}.v1.json",
            ),
            produces="{set_root}/terminal_rehearsal-{revision}.v1.json",
        ),
        LaneStep(
            step_id="profile_publication",
            argv=(
                "{python}",
                "{repository_root}/scripts/"
                "publish_task_evaluation_launch_profiles.py",
                "--profile",
                profile,
                "--profile-dir",
                "{profile_dir}",
                "--webapp-catalog-out",
                "{webapp_catalog_out}",
                "--service-account",
                "{service_account}",
                "--service-group",
                "{service_group}",
                "--receipt-out",
                "{set_root}/profile_publication_receipt.v1.json",
            ),
            produces="{set_root}/profile_publication_receipt.v1.json",
        ),
        LaneStep(
            step_id="standing_authorization",
            argv=(
                "{python}",
                "{repository_root}/scripts/"
                "materialize_task_evaluation_standing_launch_authorization.py",
                "--profile",
                "{profile_dir}/{profile_id}.json",
                "--output-dir",
                "{standing_authorization_dir}",
                "--authorized-by",
                "{authorized_by}",
                "--authorization-reference",
                "{authorization_reference}",
                "--issued-at",
                "{authorized_on}",
                "--expires-at",
                "{standing_authorization_expires_at}",
                "--max-launches",
                "1",
                "--max-total-spend-usd",
                "{hard_total_spend_cap_usd}",
                "--service-account",
                "{service_account}",
            ),
            produces="{standing_authorization_dir}/{profile_id}.json",
        ),
    )


LANES: dict[str, tuple[LaneStep, ...]] = {
    "task_evaluation_scene_configuration": _scene_configuration_steps(),
}


class PaidLaneLaunchPreparationError(RuntimeError):
    """Raised before any step runs, so a bad context prepares nothing."""


def _prepare_set_root_for_service(
    set_root: str | Path, *, service_account: str, service_group: str
) -> Path:
    """Install only this preparation root for the production consumer.

    Paid-lane preparation commonly runs as root because some builders need the
    Docker socket.  The children they create are later opened by the hardened
    service account.  Creating the set root under root's umask left the exact
    bytes present but made their parent untraversable by that consumer.

    This handoff is deliberately non-recursive.  Later writers remain
    responsible for the exact files they create, and a token already present
    below the root keeps its private mode.
    """

    try:
        account_entry = pwd.getpwnam(service_account)
    except KeyError as exc:
        raise PaidLaneLaunchPreparationError(
            f"paid_lane_service_account_missing:{service_account}"
        ) from exc
    try:
        group_entry = grp.getgrnam(service_group)
    except KeyError as exc:
        raise PaidLaneLaunchPreparationError(
            f"paid_lane_service_group_missing:{service_group}"
        ) from exc
    if (
        account_entry.pw_gid != group_entry.gr_gid
        and service_account not in group_entry.gr_mem
    ):
        raise PaidLaneLaunchPreparationError(
            f"paid_lane_service_account_group_mismatch:{service_account}:{service_group}"
        )

    root = Path(set_root).expanduser()
    if root.is_symlink():
        raise PaidLaneLaunchPreparationError("paid_lane_set_root_symlink")
    try:
        root.mkdir(parents=True, exist_ok=True)
        if root.is_symlink() or not root.is_dir():
            raise PaidLaneLaunchPreparationError("paid_lane_set_root_invalid")
        metadata = root.stat()
        expected_mode = stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP
        if metadata.st_gid != group_entry.gr_gid:
            os.chown(root, -1, group_entry.gr_gid)
        if stat.S_IMODE(metadata.st_mode) != expected_mode:
            root.chmod(expected_mode)
    except OSError as exc:
        raise PaidLaneLaunchPreparationError(
            "paid_lane_set_root_permission_install_failed"
        ) from exc
    metadata = root.stat()
    if (
        metadata.st_gid != group_entry.gr_gid
        or stat.S_IMODE(metadata.st_mode) != expected_mode
    ):
        raise PaidLaneLaunchPreparationError(
            "paid_lane_set_root_permission_install_failed"
        )
    return root.resolve()


def _placeholders(template: str) -> set[str]:
    return {
        name
        for _, name, _, _ in string.Formatter().parse(template)
        if name is not None
    }


def step_placeholders(step: LaneStep) -> set[str]:
    names: set[str] = set(_placeholders(step.produces))
    for fragment in step.argv:
        names |= _placeholders(fragment)
    return names


def step_is_active(step: LaneStep, context: Mapping[str, Any]) -> bool:
    """True when this lane run actually supplies what the step consumes.

    A step declaring ``requires_context`` is conditional: it runs only when the
    context names it needs are present and non-empty. Every consumer -- the
    validator, the plan, and the executor -- asks this one question, so a
    skipped step cannot be validated as missing in one place and executed with
    an empty placeholder in another.
    """

    if not step.requires_context:
        return True
    return all(str(context.get(name, "")) != "" for name in step.requires_context)


def validate_lane_context(lane: str, context: Mapping[str, Any]) -> None:
    """Reject an incomplete context before the first command is launched.

    A placeholder resolved to an empty string is how a wrong path reaches a
    real provider, so an unsupplied name is an error rather than a default.
    """

    steps = LANES.get(lane)
    if steps is None:
        raise PaidLaneLaunchPreparationError(f"paid_lane_unknown:{lane}")
    adoption_path = str(context.get("terminal_feedback_adoption") or "")
    adoption_digest = str(
        context.get("terminal_feedback_adoption_digest") or ""
    )
    adoption_binding = (context.get("reference_bindings") or {}).get(
        "terminal_feedback_adoption"
    )
    if bool(adoption_path) != bool(adoption_digest):
        raise PaidLaneLaunchPreparationError(
            "native_task_arena_terminal_feedback_context_incomplete"
        )
    if adoption_path and (
        lane != "native_task_arena_construction"
        or not isinstance(adoption_binding, Mapping)
        or adoption_binding.get("path") != adoption_path
        or adoption_binding.get("checkpoint_digest") != adoption_digest
    ):
        raise PaidLaneLaunchPreparationError(
            "native_task_arena_terminal_feedback_context_invalid"
        )
    available = {str(key) for key, value in context.items() if str(value) != ""}
    missing: list[str] = []
    for step in steps:
        if not step_is_active(step, context):
            continue
        for name in sorted(step_placeholders(step)):
            if name not in available:
                missing.append(f"{step.step_id}:{name}")
        available |= {name for name, _ in step.exports}
    if missing:
        raise PaidLaneLaunchPreparationError(
            "paid_lane_context_incomplete:" + ",".join(missing)
        )


def _render(template: str, context: Mapping[str, Any]) -> str:
    return template.format(**context)


def _repeated_values(value: Any) -> tuple[str, ...]:
    """Normalize optional repeated CLI values without inventing empty argv."""

    if value is None or value == "":
        return ()
    if isinstance(value, (str, int, float)):
        return (str(value),)
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value if str(item) != "")
    raise PaidLaneLaunchPreparationError("paid_lane_repeated_argv_invalid")


def _step_argv(step: LaneStep, context: Mapping[str, Any]) -> list[str]:
    argv = [_render(fragment, context) for fragment in step.argv]
    for flag, context_name in step.repeated_argv:
        for value in _repeated_values(context.get(context_name)):
            argv.extend((flag, value))
    for flag, context_name in step.conditional_argv:
        if _repeated_values(context.get(context_name)) and flag not in argv:
            argv.append(flag)
    return argv


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _default_runner(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # nosec B603 - fixed repository-owned step argv
        list(argv),
        check=False,
        capture_output=True,
        text=True,
    )


_SENSITIVE_CONTEXT_KEY = re.compile(
    r"(?i)(?:api[_-]?key|token|secret|password|credential)"
)
_SECRET_FILE_ENV_SUFFIXES = (
    "_API_KEY_FILE",
    "_TOKEN_FILE",
    "_SECRET_FILE",
    "_CREDENTIAL_FILE",
)


def _read_secret_file_value(value: Any) -> str | None:
    try:
        path = Path(str(value)).expanduser()
        if path.is_symlink() or not path.is_file():
            return None
        secret = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None
    return secret or None


def _step_log_secret_values(context: Mapping[str, Any]) -> tuple[str, ...]:
    """Load only credential values needed to scrub retained step output."""

    values: set[str] = set()
    for key, value in context.items():
        key_text = str(key)
        if not _SENSITIVE_CONTEXT_KEY.search(key_text) or value in (None, ""):
            continue
        if key_text.lower().endswith("_file"):
            secret = _read_secret_file_value(value)
            if secret:
                values.add(secret)
        elif isinstance(value, (str, int, float)):
            values.add(str(value))
    for name, path_value in os.environ.items():
        if name.endswith(_SECRET_FILE_ENV_SUFFIXES):
            secret = _read_secret_file_value(path_value)
            if secret:
                values.add(secret)
    return tuple(sorted(values, key=len, reverse=True))


def _redact_step_output(value: Any, *, secret_values: Sequence[str]) -> str:
    """Preserve diagnostic lines while applying the shared credential scrubber."""

    text = str(value or "")
    for secret in secret_values:
        text = text.replace(secret, "<redacted>")
    return redacted_failure_text(text)


def _write_step_log(path: Path, text: str) -> str:
    """Create or verify one private immutable log and return its digest."""

    encoded = text.encode("utf-8")
    parent = path.parent
    if parent.is_symlink() or not parent.is_dir() or path.is_symlink():
        raise PaidLaneLaunchPreparationError("paid_lane_step_log_path_invalid")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError:
        try:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
                raise PaidLaneLaunchPreparationError(
                    "paid_lane_step_log_immutable_conflict"
                )
        except OSError as exc:
            raise PaidLaneLaunchPreparationError(
                "paid_lane_step_log_readback_failed"
            ) from exc
    except OSError as exc:
        raise PaidLaneLaunchPreparationError(
            "paid_lane_step_log_write_failed"
        ) from exc
    else:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    metadata = path.stat()
    if stat.S_IMODE(metadata.st_mode) != 0o600 or path.read_bytes() != encoded:
        raise PaidLaneLaunchPreparationError(
            "paid_lane_step_log_readback_failed"
        )
    return _sha256_file(path)


def prepare_paid_lane_launch(
    lane: str,
    context: Mapping[str, Any],
    *,
    resume_receipt: Mapping[str, Any] | None = None,
    runner: Callable[
        [Sequence[str]], int | subprocess.CompletedProcess[str]
    ] = _default_runner,
) -> dict[str, Any]:
    """Run one lane's ordered preparation and return a fail-closed receipt."""

    validate_lane_context(lane, context)
    if "set_root" in context:
        account = str(context.get("service_account") or "")
        group = str(context.get("service_group") or "")
        if not account or not group:
            raise PaidLaneLaunchPreparationError(
                "paid_lane_service_identity_required"
            )
        set_root_path = _prepare_set_root_for_service(
            str(context["set_root"]),
            service_account=account,
            service_group=group,
        )
        step_log_root = set_root_path / "step-logs"
        step_log_root.mkdir(mode=0o750, exist_ok=True)
        step_log_root.chmod(0o750)
    else:
        step_log_root = None
    resolved: dict[str, Any] = dict(context)
    completed: list[dict[str, Any]] = []
    step_logs: list[dict[str, Any]] = []
    blockers: list[str] = []
    resumed_rows: list[Mapping[str, Any]] = []
    if resume_receipt is not None:
        if (
            resume_receipt.get("schema_version") != PREPARATION_SCHEMA_VERSION
            or resume_receipt.get("lane") != lane
            or resume_receipt.get("status") != "blocked"
            or resume_receipt.get("source_commit")
            != str(context.get("source_commit") or "")
            or resume_receipt.get("paid_inference_performed") is not False
            or resume_receipt.get("provider_allocation_performed") is not False
            or not isinstance(resume_receipt.get("completed_steps"), list)
            or not resume_receipt.get("blockers")
        ):
            raise PaidLaneLaunchPreparationError(
                "paid_lane_resume_receipt_invalid"
            )
        resumed_rows = list(resume_receipt["completed_steps"])
        retained_logs = resume_receipt.get("step_logs") or []
        if not isinstance(retained_logs, list):
            raise PaidLaneLaunchPreparationError(
                "paid_lane_resume_receipt_invalid"
            )
        step_logs.extend(dict(row) for row in retained_logs if isinstance(row, Mapping))
    resume_index = 0
    secret_values = _step_log_secret_values(context)
    for step in LANES[lane]:
        if not step_is_active(step, resolved):
            continue
        argv = _step_argv(step, resolved)
        produces = Path(_render(step.produces, resolved))
        if resume_index < len(resumed_rows):
            row = resumed_rows[resume_index]
            if (
                not isinstance(row, Mapping)
                or row.get("step_id") != step.step_id
                or Path(str(row.get("artifact_path") or "")).resolve()
                != produces.resolve()
                or produces.is_symlink()
                or not produces.is_file()
                or row.get("artifact_sha256") != _sha256_file(produces)
            ):
                raise PaidLaneLaunchPreparationError(
                    f"paid_lane_resume_step_invalid:{step.step_id}"
                )
            restored = dict(row)
            for name, key in step.exports:
                try:
                    payload = json.loads(produces.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                    raise PaidLaneLaunchPreparationError(
                        f"paid_lane_resume_export_invalid:{step.step_id}:{name}"
                    ) from exc
                value = payload.get(key) if isinstance(payload, Mapping) else None
                if (
                    not isinstance(value, str)
                    or not value
                    or not isinstance(row.get("exports"), Mapping)
                    or row["exports"].get(name) != value
                ):
                    raise PaidLaneLaunchPreparationError(
                        f"paid_lane_resume_export_invalid:{step.step_id}:{name}"
                    )
                resolved[name] = value
            completed.append(restored)
            resume_index += 1
            continue
        execution = runner(argv)
        if isinstance(execution, subprocess.CompletedProcess):
            returncode = int(execution.returncode)
            stdout_text = _redact_step_output(
                execution.stdout, secret_values=secret_values
            )
            stderr_text = _redact_step_output(
                execution.stderr, secret_values=secret_values
            )
        else:
            returncode = int(execution)
            stdout_text = ""
            stderr_text = ""
        if step_log_root is None:
            stdout_path = produces.with_name(
                f"{produces.name}.{step.step_id}.stdout.log"
            )
            stderr_path = produces.with_name(
                f"{produces.name}.{step.step_id}.stderr.log"
            )
        else:
            attempt_token = uuid.uuid4().hex
            stdout_path = (
                step_log_root / f"{step.step_id}-{attempt_token}.stdout.log"
            )
            stderr_path = (
                step_log_root / f"{step.step_id}-{attempt_token}.stderr.log"
            )
        try:
            stdout_sha256 = _write_step_log(stdout_path, stdout_text)
            stderr_sha256 = _write_step_log(stderr_path, stderr_text)
        except PaidLaneLaunchPreparationError as exc:
            blockers.append(f"{step.step_id}:{exc}")
            if returncode != 0:
                blockers.append(f"{step.step_id}:exit_{returncode}")
            break
        step_logs.append(
            {
                "step_id": step.step_id,
                "stdout_path": str(stdout_path),
                "stdout_sha256": stdout_sha256,
                "stderr_path": str(stderr_path),
                "stderr_sha256": stderr_sha256,
                "credential_redaction_applied": True,
            }
        )
        if returncode != 0:
            blockers.append(f"{step.step_id}:exit_{returncode}")
            break
        if produces.is_symlink() or not produces.is_file():
            blockers.append(f"{step.step_id}:declared_artifact_missing")
            break
        record: dict[str, Any] = {
            "step_id": step.step_id,
            "artifact_path": str(produces),
            "artifact_sha256": _sha256_file(produces),
        }
        export_failed = False
        for name, key in step.exports:
            try:
                payload = json.loads(produces.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                payload = None
            value = payload.get(key) if isinstance(payload, Mapping) else None
            if not isinstance(value, str) or not value:
                blockers.append(f"{step.step_id}:export_unavailable:{name}")
                export_failed = True
                break
            resolved[name] = value
            record.setdefault("exports", {})[name] = value
        completed.append(record)
        if export_failed:
            break
    if resume_index != len(resumed_rows):
        raise PaidLaneLaunchPreparationError(
            "paid_lane_resume_completed_steps_not_prefix"
        )
    return {
        "schema_version": PREPARATION_SCHEMA_VERSION,
        "lane": lane,
        "status": "prepared" if not blockers else "blocked",
        "source_commit": str(context.get("source_commit") or ""),
        **(
            {"reference_bindings": dict(context["reference_bindings"])}
            if isinstance(context.get("reference_bindings"), Mapping)
            else {}
        ),
        "step_count": len(LANES[lane]),
        "completed_steps": completed,
        "step_logs": step_logs,
        "blockers": blockers,
        "paid_inference_performed": False,
        "provider_allocation_performed": False,
    }


def validate_paid_lane_launch(
    lane: str,
    context: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one complete launch graph without running a subprocess."""

    validate_lane_context(lane, context)
    resolved: dict[str, Any] = dict(context)
    planned: list[dict[str, Any]] = []
    for step in LANES[lane]:
        if not step_is_active(step, resolved):
            continue
        argv = _step_argv(step, resolved)
        planned.append(
            {
                "step_id": step.step_id,
                "argv": argv,
                "declared_artifact_path": _render(step.produces, resolved),
            }
        )
        for name, key in step.exports:
            resolved[name] = f"<export:{step.step_id}:{key}>"
    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "lane": lane,
        "status": "validated_no_commands_run",
        "source_commit": str(context.get("source_commit") or ""),
        **(
            {"reference_bindings": dict(context["reference_bindings"])}
            if isinstance(context.get("reference_bindings"), Mapping)
            else {}
        ),
        "step_count": len(planned),
        "planned_steps": planned,
        "subprocesses_executed": 0,
        "set_root_permissions_mutated": False,
        "publication_performed": False,
        "standing_authorization_published": False,
        "paid_inference_performed": False,
        "provider_allocation_performed": False,
        "provider_mutation_performed": False,
    }


def _reserve_receipt_output(path: str | Path) -> tuple[Path, int]:
    """Reserve one immutable receipt name before any preparation can run."""

    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o440)
    except FileExistsError as exc:
        raise PaidLaneLaunchPreparationError(
            "paid_lane_receipt_output_exists"
        ) from exc
    return output, descriptor


def _write_reserved_receipt(
    output: Path,
    descriptor: int,
    receipt: Mapping[str, Any],
    *,
    service_account: str | None,
    service_group: str | None,
) -> None:
    account_entry = None
    group_entry = None
    if service_account is not None or service_group is not None:
        if not service_account or not service_group:
            os.close(descriptor)
            raise PaidLaneLaunchPreparationError(
                "paid_lane_receipt_service_identity_invalid"
            )
        try:
            account_entry = pwd.getpwnam(service_account)
            group_entry = grp.getgrnam(service_group)
        except KeyError as exc:
            os.close(descriptor)
            raise PaidLaneLaunchPreparationError(
                "paid_lane_receipt_service_identity_invalid"
            ) from exc
    payload = (json.dumps(dict(receipt), indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    with os.fdopen(descriptor, "wb") as stream:
        if account_entry is not None and group_entry is not None:
            os.fchown(stream.fileno(), account_entry.pw_uid, group_entry.gr_gid)
        os.fchmod(stream.fileno(), 0o440)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    parent_descriptor = os.open(output.parent, os.O_RDONLY)
    try:
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)


def _arg_text(value: Any) -> str:
    return "" if value is None else str(value)


def _context_from_args(args: argparse.Namespace) -> dict[str, Any]:
    context = {
        "python": args.python,
        "repository_root": args.repository_root,
        "set_root": args.set_root,
        "packet": args.packet,
        "source_commit": args.source_commit,
        "destination_prefix": args.destination_prefix,
        "token_file": args.token_file,
        "runtime_image_identity": args.runtime_image_identity,
        "profile_dir": args.profile_dir,
        "webapp_catalog_out": args.webapp_catalog_out,
        "service_account": args.service_account,
        "service_group": args.service_group,
        "pod_name": args.pod_name,
        "revision": args.revision,
        "authorization_reference": args.authorization_reference,
        "authorized_by": args.authorized_by,
        "authorized_on": args.authorized_on,
        "backend_entry_digest": args.backend_entry_digest,
        "task_count": _arg_text(args.task_count),
        "camera_count": _arg_text(args.camera_count),
        "maximum_hourly_rate_usd": _arg_text(args.maximum_hourly_rate_usd),
        "hard_total_spend_cap_usd": _arg_text(args.hard_total_spend_cap_usd),
        "provider_compute_spend_cap_usd": _arg_text(
            args.provider_compute_spend_cap_usd
        ),
        "openai_max_cost_usd": _arg_text(args.openai_max_cost_usd),
        "openai_max_requests": _arg_text(args.openai_max_requests),
        "openai_artifixer_semantic_teacher_max_cost_usd": _arg_text(
            args.openai_artifixer_semantic_teacher_max_cost_usd
        ),
        "openai_artifixer_visual_review_max_cost_usd": _arg_text(
            args.openai_artifixer_visual_review_max_cost_usd
        ),
        "openai_content_agents_max_cost_usd": _arg_text(
            args.openai_content_agents_max_cost_usd
        ),
        "anthropic_max_cost_usd": _arg_text(args.anthropic_max_cost_usd),
        "anthropic_max_requests": _arg_text(args.anthropic_max_requests),
        "hard_ttl_seconds": _arg_text(args.hard_ttl_seconds),
        "aggregate_goal_spend_before_usd": _arg_text(
            args.aggregate_goal_spend_before_usd
        ),
        "aggregate_goal_spend_cap_usd": _arg_text(args.aggregate_goal_spend_cap_usd),
        "prior_spend_reconciliations": (
            [args.prior_spend_reconciliation]
            if args.prior_spend_reconciliation
            else []
        ),
        "excluded_machine_ids": [str(value) for value in args.excluded_machine_id],
    }
    for entry in args.set or []:
        name, _, value = entry.partition("=")
        context[name.strip()] = value
    return context


def _canonical_mapping_digest(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _canonical_artifact_digest(
    value: Mapping[str, Any], *, digest_field: str
) -> str:
    body = dict(value)
    body.pop(digest_field, None)
    return _canonical_mapping_digest(body)


def _load_scene_claim_reference(
    *,
    path: Any,
    expected_digest: Any,
    expected_schema: str,
    expected_status: str,
    digest_field: str,
    scene_id: str,
) -> tuple[Path, dict[str, Any]]:
    unresolved = Path(str(path or "")).expanduser()
    if unresolved.is_symlink():
        raise PaidLaneLaunchPreparationError(
            "native_task_arena_scene_claim_reference_invalid"
        )
    source = unresolved.resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PaidLaneLaunchPreparationError(
            "native_task_arena_scene_claim_reference_unreadable"
        ) from exc
    from blueprint_pipeline.website_native_launch_claims import reference_valid
    if reference_valid(value, expected_schema=expected_schema, scene_id=scene_id):
        if _sha256_file(source) != expected_digest:
            raise PaidLaneLaunchPreparationError("native_task_arena_scene_claim_reference_invalid")
        return source, dict(value)
    embedded_digest = value.get(digest_field) if isinstance(value, Mapping) else None
    if embedded_digest is None:
        observed_digest = "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest()
        digest_valid = observed_digest == expected_digest
    else:
        digest_valid = (
            embedded_digest == expected_digest
            and embedded_digest
            == _canonical_artifact_digest(value, digest_field=digest_field)
        )
    status_aliases = {
        "retained": {"retained", "candidate_source_bytes_retained"},
        "admitted": {"admitted", "admitted_for_internal_development"},
    }
    schema_aliases = {
        "task_evaluation_scene_source_manifest.v1": "task_evaluation_completed_scene_source_manifest.v1",
        "task_evaluation_scene_rights_admission.v1": "task_evaluation_completed_scene_rights_admission.v1",
    }
    completed_schema = schema_aliases.get(expected_schema)
    schema_valid = value.get("schema_version") == expected_schema or (
        value.get("schema_version") == completed_schema
        and completed_schema is not None
        and value.get("source") == "owner_provided_completed_asset"
    )
    observed_scene_id = str(value.get("scene_id") or "")
    scene_id_valid = observed_scene_id == scene_id or scene_id.endswith(
        "-" + observed_scene_id
    )
    if (
        not source.is_file()
        or not isinstance(value, Mapping)
        or not schema_valid
        or value.get("status")
        not in status_aliases.get(expected_status, {expected_status})
        or not scene_id_valid
        or not digest_valid
    ):
        raise PaidLaneLaunchPreparationError(
            "native_task_arena_scene_claim_reference_invalid"
        )
    return source, dict(value)


def _validate_provider_packet_source_rights(
    *,
    packet_receipt: Mapping[str, Any],
    packet_request: Mapping[str, Any] | None = None,
    source_manifest: Mapping[str, Any],
    configured_scene_revision: Mapping[str, Any] | None = None,
    rights_admission: Mapping[str, Any] | None = None,
    destination: Mapping[str, Any] | None = None,
) -> None:
    """Require every provider-staged source binding to be upload-admitted."""

    packet_request = packet_request or {}
    bindings = packet_receipt.get("source_bindings")
    from blueprint_pipeline.website_native_launch_claims import validate_join
    try:
        website_claims = validate_join(source_manifest, rights_admission or {},
            scene_id=((configured_scene_revision or {}).get("scene_identity") or {}).get("id"))
    except ValueError as exc:
        raise PaidLaneLaunchPreparationError(str(exc)) from exc
    artifacts = [] if website_claims else source_manifest.get("artifacts")
    if (
        isinstance(bindings, (str, bytes))
        or not isinstance(bindings, Sequence)
        or not bindings
        or isinstance(artifacts, (str, bytes))
        or not isinstance(artifacts, Sequence)
    ):
        raise PaidLaneLaunchPreparationError(
            "native_task_arena_provider_source_rights_invalid"
        )
    admitted: set[tuple[str, int]] = set()
    appearance_variant = packet_request.get("appearance_variant")
    request_digest_valid = (
        packet_request.get("request_digest")
        == packet_receipt.get("request_digest")
        == _canonical_artifact_digest(packet_request, digest_field="request_digest")
    )
    if configured_scene_revision is not None:
        disclosure = (configured_scene_revision.get("source") or {}).get(
            "provider_disclosure_decision"
        ) or {}
        if (
            not isinstance(rights_admission, Mapping)
            or rights_admission.get("private_provider_processing_allowed") is not True
            or not (website_claims or disclosure.get("human_authority_accepts_provider_terms") is True)
        ):
            raise PaidLaneLaunchPreparationError(
                "native_task_arena_provider_source_rights_invalid"
            )
        for raw in (
            (configured_scene_revision.get("appearance") or {}).get(
                "configured_representation"
            ),
            (configured_scene_revision.get("geometry") or {}).get(
                "configured_collision"
            ),
            (configured_scene_revision.get("replacement") or {}).get("asset"),
        ):
            if isinstance(raw, Mapping) and isinstance(raw.get("size_bytes"), int):
                admitted.add((str(raw.get("digest") or ""), raw["size_bytes"]))
    else:
        for raw in artifacts:
            if not isinstance(raw, Mapping):
                continue
            digest = str(raw.get("sha256") or "")
            size = raw.get("size_bytes")
            if raw.get("provider_upload_allowed") is True and isinstance(size, int):
                admitted.add((digest, size))
    runtime_disclosure = (rights_admission or {}).get("provider_disclosure") or {}
    role_permissions = {
        "scene_appearance": "minimum_digest_bound_derived_appearance_runtime_bytes_may_be_privately_processed",
        "scene_collision": "sage_collision_runtime_bytes_may_be_privately_processed",
        "task_object": "qualified_replacement_usd_may_be_privately_processed",
    }
    # Completed owner assets have one permission covering digest-bound derived
    # runtime files. Their raw source upload remains forbidden. Admit only the
    # exact producer's development contract and the configured asset roles;
    # byte membership and staging lineage are checked below for every binding.
    completed_derived_permission = (
        configured_scene_revision is not None
        and source_manifest.get("schema_version") == "task_evaluation_completed_scene_source_manifest.v1"
        and (rights_admission or {}).get("schema_version") == "task_evaluation_completed_scene_rights_admission.v1"
        and source_manifest.get("source") == (rights_admission or {}).get("source") == "owner_provided_completed_asset"
        and source_manifest.get("scene_id") == (rights_admission or {}).get("scene_id")
        == (configured_scene_revision.get("scene_identity") or {}).get("id")
        and isinstance(source_manifest.get("scene_id"), str) and bool(source_manifest["scene_id"])
        and source_manifest.get("captured_observation_supplied") is False
        and (source_manifest.get("coordinate_system") or {}).get("physical_metrology_claimed") is False
        and (rights_admission or {}).get("status") == "admitted_for_internal_development"
        and (rights_admission or {}).get("source_bytes_redistributable") is False
        and (rights_admission or {}).get("physical_metrology_claimed") is False
        and (rights_admission or {}).get("provider_training_allowed") is False
        and (rights_admission or {}).get("public_redistribution_allowed") is False
        and runtime_disclosure.get("raw_owner_source_bytes_may_be_uploaded") is False
        and runtime_disclosure.get("minimum_digest_bound_derived_runtime_bytes_may_be_privately_processed") is True
    )
    destination_pair = None
    if destination is not None:
        rights_path, destination_rights = _load_unlinked_json(
            destination.get("rights_admission"),
            error="native_task_arena_destination_rights_invalid",
        )
        asset = destination.get("asset") or {}
        if (
            _sha256_file(rights_path) != destination.get("rights_admission_digest")
            or destination_rights.get("schema_version") != "task_evaluation_rigid_destination_rights_admission.v1"
            or destination_rights.get("rights_admission_digest") != _canonical_artifact_digest(destination_rights, digest_field="rights_admission_digest")
            or destination_rights.get("status") != "admitted"
            or destination_rights.get("destination_identity") != destination.get("identity")
            or destination_rights.get("private_provider_processing_allowed") is not True
            or destination_rights.get("provider_training_allowed") is not False
            or destination_rights.get("public_redistribution_allowed") is not False
        ):
            raise PaidLaneLaunchPreparationError("native_task_arena_destination_rights_invalid")
        destination_pair = (asset.get("digest"), asset.get("size_bytes"))
    for raw in bindings:
        if not isinstance(raw, Mapping):
            raise PaidLaneLaunchPreparationError(
                "native_task_arena_provider_source_rights_invalid"
            )
        source = raw.get("source")
        if not isinstance(source, Mapping):
            raise PaidLaneLaunchPreparationError(
                "native_task_arena_provider_source_rights_invalid"
            )
        source_pair = (str(source.get("sha256") or ""), source.get("size_bytes"))
        staged_pair = (
            str(raw.get("staged_sha256") or ""),
            raw.get("staged_size_bytes"),
        )
        adaptation = raw.get("static_scene_collision_adaptation")
        staged_matches_source = source_pair == staged_pair
        staged_is_bound_adaptation = (
            isinstance(adaptation, Mapping)
            and adaptation.get("adaptation") == "static_convex_to_triangle_mesh"
            and adaptation.get("derived_from_sha256") == source_pair[0]
            and isinstance(adaptation.get("converted_prim_paths"), Sequence)
            and bool(adaptation.get("converted_prim_paths"))
            and isinstance(staged_pair[1], int)
            and re.fullmatch(r"sha256:[0-9a-f]{64}", staged_pair[0]) is not None
        )
        configured_appearance = (
            (configured_scene_revision.get("appearance") or {}).get(
                "configured_representation"
            )
            if configured_scene_revision is not None
            else None
        )
        staged_is_bound_particlefield = (
            raw.get("semantic_role") == "scene_appearance"
            and request_digest_valid
            and isinstance(appearance_variant, Mapping)
            and appearance_variant.get("representation")
            == "particlefield_3d_gaussian_splat"
            and appearance_variant.get("representation_conversion_performed") is True
            and appearance_variant.get("exact_learned_arrays_preserved") is True
            and isinstance(configured_appearance, Mapping)
            and appearance_variant.get("source_configured_appearance_digest")
            == configured_appearance.get("digest")
            and source_pair == staged_pair
        )
        source_is_admitted = source_pair in admitted or staged_is_bound_particlefield
        if configured_scene_revision is not None:
            role = raw.get("semantic_role")
            derived_permission = (
                (runtime_disclosure.get(role_permissions.get(role, "")) is True
                 or (completed_derived_permission and role in role_permissions))
                and runtime_disclosure.get("provider_training_allowed") is False
                and runtime_disclosure.get("public_redistribution_allowed") is False
                and bool(runtime_disclosure.get("provider_retention_rule"))
            )
            source_is_admitted = source_is_admitted and (
                website_claims or disclosure.get("rights_admission_permits_upload") is True or derived_permission
            )
            if role == "task_support":
                source_is_admitted = (
                    request_digest_valid
                    and destination_pair is not None
                    and source_pair == destination_pair
                    and raw.get("asset_id") == (destination.get("identity") or {}).get("id")
                )
        if not source_is_admitted or not (
            staged_matches_source
            or staged_is_bound_adaptation
            or staged_is_bound_particlefield
        ):
            raise PaidLaneLaunchPreparationError(
                "native_task_arena_provider_source_rights_invalid"
            )


def _load_unlinked_json(path: Any, *, error: str) -> tuple[Path, dict[str, Any]]:
    unresolved = Path(str(path or "")).expanduser()
    if unresolved.is_symlink():
        raise PaidLaneLaunchPreparationError(error)
    source = unresolved.resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PaidLaneLaunchPreparationError(error) from exc
    if not source.is_file() or not isinstance(value, dict):
        raise PaidLaneLaunchPreparationError(error)
    return source, value


def _load_scene_configuration_context(
    path: str | Path, *, expected_lane: str
) -> dict[str, Any]:
    """Load one production-generated scene configuration preparation context."""

    unresolved = Path(path).expanduser()
    if unresolved.is_symlink():
        raise PaidLaneLaunchPreparationError(
            "scene_configuration_context_invalid"
        )
    source = unresolved.resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PaidLaneLaunchPreparationError(
            "scene_configuration_context_unreadable"
        ) from exc
    if (
        not isinstance(value, Mapping)
        or value.get("schema_version")
        != SCENE_CONFIGURATION_CONTEXT_SCHEMA_VERSION
        or value.get("lane") != expected_lane
        or expected_lane != "task_evaluation_scene_configuration"
        or not str(value.get("team_namespace") or "").strip()
        or not isinstance(value.get("operations"), Mapping)
        or not isinstance(value.get("reference_bindings"), Mapping)
    ):
        raise PaidLaneLaunchPreparationError(
            "scene_configuration_context_invalid"
        )
    operations = dict(value["operations"])
    for key in operations:
        lowered = str(key).lower()
        if any(
            token in lowered
            for token in ("password", "api_key", "secret", "token")
        ):
            raise PaidLaneLaunchPreparationError(
                "scene_configuration_context_secret_value_forbidden"
            )
    if "reference_bindings" in operations or "team_namespace" in operations:
        raise PaidLaneLaunchPreparationError(
            "scene_configuration_context_reference_override_forbidden"
        )
    construction = Path(
        str(operations.get("construction_envelope") or "")
    ).expanduser()
    toolchain = Path(str(operations.get("toolchain_root") or "")).expanduser()
    source_commit = str(operations.get("source_commit") or "")
    semantic_reuse = operations.get(
        "production_semantic_reuse_checkpoint_root"
    )
    semantic_queue = operations.get("production_semantic_reuse_queue_root")
    semantic_revision = operations.get("production_semantic_reuse_revision_id")
    bindings = value["reference_bindings"]
    if (
        construction.is_symlink()
        or toolchain.is_symlink()
        or not construction.resolve().is_file()
        or not toolchain.resolve().is_dir()
        or construction.resolve().stat().st_mode & 0o222
        or toolchain.resolve().stat().st_mode & 0o222
        or bindings.get("construction_envelope_sha256")
        != _sha256_file(construction.resolve())
        or bindings.get("source_commit") != source_commit
        or len(source_commit) != 40
        or any(ch not in "0123456789abcdef" for ch in source_commit)
    ):
        raise PaidLaneLaunchPreparationError(
            "scene_configuration_context_binding_invalid"
        )
    if semantic_reuse is not None:
        checkpoint = Path(str(semantic_reuse)).expanduser()
        queue = Path(str(semantic_queue or "")).expanduser()
        if (
            checkpoint.is_symlink()
            or not checkpoint.resolve().is_dir()
            or queue.is_symlink()
            or not queue.resolve().is_dir()
            or re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}",
                str(semantic_revision or ""),
            )
            is None
        ):
            raise PaidLaneLaunchPreparationError(
                "scene_configuration_context_semantic_reuse_revision_invalid"
            )
        operations["production_semantic_reuse_checkpoint_root"] = str(
            checkpoint.resolve()
        )
        operations["production_semantic_reuse_queue_root"] = str(queue.resolve())
        operations["production_semantic_reuse_revision_id"] = str(
            semantic_revision
        )
    elif semantic_queue is not None or semantic_revision is not None:
        raise PaidLaneLaunchPreparationError(
            "scene_configuration_context_semantic_reuse_revision_invalid"
        )
    operations.update(
        {
            "construction_envelope": str(construction.resolve()),
            "toolchain_root": str(toolchain.resolve()),
            "team_namespace": str(value["team_namespace"]),
            "reference_bindings": dict(bindings),
        }
    )
    return operations


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", required=True, choices=sorted(LANES))
    parser.add_argument("--context-file")
    parser.add_argument("--set-root")
    parser.add_argument("--packet")
    parser.add_argument("--repository-root")
    parser.add_argument("--source-commit")
    parser.add_argument("--token-file")
    parser.add_argument("--destination-prefix")
    parser.add_argument("--runtime-image-identity")
    parser.add_argument("--profile-dir")
    parser.add_argument("--webapp-catalog-out")
    parser.add_argument(
        "--service-account",
        default=DEFAULT_SERVICE_ACCOUNT,
        help="Account that consumes the prepared launch inputs.",
    )
    parser.add_argument(
        "--service-group",
        default=DEFAULT_SERVICE_GROUP,
        help="Canonical group allowed to traverse the set and read published inputs.",
    )
    parser.add_argument("--pod-name")
    parser.add_argument("--revision", default="r1")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--authorization-reference")
    parser.add_argument("--authorized-by")
    parser.add_argument("--authorized-on")
    parser.add_argument("--backend-entry-digest")
    parser.add_argument("--task-count", type=int)
    parser.add_argument("--camera-count", type=int)
    parser.add_argument("--maximum-hourly-rate-usd", type=float)
    parser.add_argument("--hard-total-spend-cap-usd", type=float)
    parser.add_argument("--provider-compute-spend-cap-usd", type=float)
    parser.add_argument("--openai-max-cost-usd", type=float)
    parser.add_argument("--openai-max-requests", type=int)
    parser.add_argument(
        "--openai-artifixer-semantic-teacher-max-cost-usd", type=float
    )
    parser.add_argument(
        "--openai-artifixer-visual-review-max-cost-usd", type=float
    )
    parser.add_argument("--openai-content-agents-max-cost-usd", type=float)
    parser.add_argument("--anthropic-max-cost-usd", type=float, default=0.0)
    parser.add_argument("--anthropic-max-requests", type=int, default=0)
    parser.add_argument("--hard-ttl-seconds", type=int)
    parser.add_argument("--aggregate-goal-spend-before-usd", type=float)
    parser.add_argument("--aggregate-goal-spend-cap-usd", type=float)
    parser.add_argument("--prior-spend-reconciliation")
    parser.add_argument(
        "--excluded-machine-id",
        action="append",
        default=[],
        type=int,
        help="Vast machine ID the immutable live profile must exclude; repeatable.",
    )
    parser.add_argument("--set", action="append", help="Extra name=value context entry")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate and seal the ordered command plan without running a command.",
    )
    parser.add_argument(
        "--resume-from",
        help="Resume after the verified completed-step prefix of one blocked receipt.",
    )
    parser.add_argument("--receipt-out")
    args = parser.parse_args(argv)

    reserved_output: Path | None = None
    reserved_descriptor: int | None = None
    try:
        if args.validate_only and not args.receipt_out:
            raise PaidLaneLaunchPreparationError(
                "paid_lane_validation_receipt_required"
            )
        if args.receipt_out:
            reserved_output, reserved_descriptor = _reserve_receipt_output(
                args.receipt_out
            )
        if args.context_file:
            context = _load_scene_configuration_context(
                args.context_file, expected_lane=args.lane
            )
        else:
            context = _context_from_args(args)
        resume_receipt = None
        if args.resume_from:
            resume_path = Path(args.resume_from).expanduser()
            if resume_path.is_symlink() or not resume_path.resolve().is_file():
                raise PaidLaneLaunchPreparationError(
                    "paid_lane_resume_receipt_invalid"
                )
            try:
                resume_receipt = json.loads(
                    resume_path.resolve().read_text(encoding="utf-8")
                )
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise PaidLaneLaunchPreparationError(
                    "paid_lane_resume_receipt_invalid"
                ) from exc
            if not isinstance(resume_receipt, Mapping):
                raise PaidLaneLaunchPreparationError(
                    "paid_lane_resume_receipt_invalid"
                )
        receipt = (
            validate_paid_lane_launch(args.lane, context)
            if args.validate_only
            else prepare_paid_lane_launch(
                args.lane, context, resume_receipt=resume_receipt
            )
        )
    except PaidLaneLaunchPreparationError as exc:
        if reserved_descriptor is not None:
            os.close(reserved_descriptor)
            reserved_descriptor = None
        if reserved_output is not None:
            reserved_output.unlink(missing_ok=True)
        print(json.dumps({"status": "blocked", "blockers": [str(exc)]}, sort_keys=True))
        return 2
    except BaseException:
        if reserved_descriptor is not None:
            os.close(reserved_descriptor)
        if reserved_output is not None:
            reserved_output.unlink(missing_ok=True)
        raise
    if reserved_output is not None and reserved_descriptor is not None:
        _write_reserved_receipt(
            reserved_output,
            reserved_descriptor,
            receipt,
            service_account=(
                None
                if args.validate_only
                else str(context.get("service_account") or DEFAULT_SERVICE_ACCOUNT)
            ),
            service_group=(
                None
                if args.validate_only
                else str(context.get("service_group") or DEFAULT_SERVICE_GROUP)
            ),
        )
    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt["status"] in {"prepared", "validated_no_commands_run"} else 2


__all__ = [
    "LANES",
    "PREPARATION_SCHEMA_VERSION",
    "VALIDATION_SCHEMA_VERSION",
    "LaneStep",
    "PaidLaneLaunchPreparationError",
    "prepare_paid_lane_launch",
    "validate_paid_lane_launch",
    "step_placeholders",
    "validate_lane_context",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
