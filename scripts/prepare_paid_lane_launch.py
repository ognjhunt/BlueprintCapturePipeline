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
import stat
import string
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import jsonschema

PREPARATION_SCHEMA_VERSION = "paid_lane_launch_preparation.v1"
VALIDATION_SCHEMA_VERSION = "paid_lane_launch_validation.v1"
NATIVE_CONTEXT_SCHEMA_VERSION = "native_task_arena_launch_preparation_context.v2"
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


SEMANTIC_TEACHER_IMAGE_EDIT_STEPS: tuple[LaneStep, ...] = (
    LaneStep(
        step_id="provider_bundle",
        argv=(
            "{python}",
            "-m",
            "blueprint_pipeline.semantic_teacher_image_edit_bundle",
            "--packet",
            "{packet}",
            "--repository-root",
            "{repository_root}",
            "--expected-source-commit",
            "{source_commit}",
            "--output-root",
            "{set_root}/bundle",
        ),
        produces="{set_root}/bundle/semantic_teacher_image_edit_provider_bundle.v1.json",
    ),
    LaneStep(
        step_id="immutable_manifest",
        argv=(
            "{python}",
            "{repository_root}/scripts/publish_task_evaluation_immutable_manifest.py",
            "--manifest",
            "{set_root}/bundle/semantic_teacher_image_edit_provider_bundle.v1.json",
            "--profile-builder",
            "build_semantic_teacher_image_edit_live_profile.py",
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
            "blueprint_pipeline.semantic_teacher_image_edit_paid_authority",
            "--bundle",
            "{set_root}/bundle/semantic_teacher_image_edit_provider_bundle.zip",
            "--bundle-receipt",
            "{set_root}/bundle/semantic_teacher_image_edit_provider_bundle.v1.json",
            "--authorization-reference",
            "{authorization_reference}",
            "--authorized-by",
            "{authorized_by}",
            "--authorized-on",
            "{authorized_on}",
            "--source-commit-sha",
            "{source_commit}",
            "--backend-entry-digest",
            "{backend_entry_digest}",
            "--task-count",
            "{task_count}",
            "--camera-count",
            "{camera_count}",
            "--maximum-hourly-rate-usd",
            "{maximum_hourly_rate_usd}",
            "--hard-total-spend-cap-usd",
            "{hard_total_spend_cap_usd}",
            "--hard-ttl-seconds",
            "{hard_ttl_seconds}",
            "--aggregate-goal-spend-before-usd",
            "{aggregate_goal_spend_before_usd}",
            "--aggregate-goal-spend-cap-usd",
            "{aggregate_goal_spend_cap_usd}",
            "--output",
            "{set_root}/semantic_teacher_image_edit_paid_authority.v1.json",
        ),
        produces="{set_root}/semantic_teacher_image_edit_paid_authority.v1.json",
        repeated_argv=(("--prior-spend-reconciliation", "prior_spend_reconciliations"),),
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
            "{set_root}/dry-run-job/result.json",
            "--pod-name",
            "{pod_name}",
            "--expected-source-commit",
            "{source_commit}",
            "--provider",
            "vast",
            "--probe-kind",
            "semantic-teacher-image-edit",
            "--semantic-teacher-bundle",
            "{set_root}/bundle/semantic_teacher_image_edit_provider_bundle.zip",
            "--semantic-teacher-bundle-receipt",
            "{set_root}/bundle/semantic_teacher_image_edit_provider_bundle.v1.json",
            "--semantic-teacher-attempt-authority",
            "{set_root}/semantic_teacher_image_edit_paid_authority.v1.json",
            "--semantic-teacher-token-file",
            "{token_file}",
            "--semantic-teacher-runtime-image-identity",
            "{runtime_image_identity}",
            "--semantic-teacher-job-dir",
            "{set_root}/dry-run-job/job",
            "--semantic-teacher-dry-run-output",
            "{set_root}/semantic_teacher_image_edit_allocator_dry_run.v1.json",
            "--semantic-teacher-preflight-output",
            "{set_root}/dry-run-job/preflight.json",
        ),
        produces="{set_root}/semantic_teacher_image_edit_allocator_dry_run.v1.json",
        repeated_argv=(("--semantic-teacher-excluded-machine-id", "excluded_machine_ids"),),
    ),
    LaneStep(
        step_id="live_profile",
        argv=(
            "{python}",
            "{repository_root}/scripts/build_semantic_teacher_image_edit_live_profile.py",
            "--bundle-receipt",
            "{set_root}/bundle/semantic_teacher_image_edit_provider_bundle.v1.json",
            "--attempt-authority",
            "{set_root}/semantic_teacher_image_edit_paid_authority.v1.json",
            "--dry-run-receipt",
            "{set_root}/semantic_teacher_image_edit_allocator_dry_run.v1.json",
            "--token-file",
            "{token_file}",
            "--source-commit",
            "{source_commit}",
            "--raw-manifest-uri",
            "{set_root}/manifest_publication_receipt.v1.json",
            "--revision",
            "{revision}",
            "--output",
            "{set_root}/live_profile-{revision}.v1.json",
        ),
        produces="{set_root}/live_profile-{revision}.v1.json",
        repeated_argv=(("--excluded-machine-id", "excluded_machine_ids"),),
    ),
    LaneStep(
        step_id="profile_publication",
        argv=(
            "{python}",
            "{repository_root}/scripts/publish_task_evaluation_launch_profiles.py",
            "--profile",
            "{set_root}/live_profile-{revision}.v1.json",
            "--profile-dir",
            "{profile_dir}",
            "--webapp-catalog-out",
            "{webapp_catalog_out}",
            "--service-account",
            "{service_account}",
            "--service-group",
            "{service_group}",
        ),
        produces="{set_root}/live_profile-{revision}.v1.json",
    ),
    LaneStep(
        step_id="terminal_rehearsal",
        argv=(
            "{python}",
            "{repository_root}/scripts/rehearse_lane_terminal_contract.py",
            "--profile",
            "{set_root}/live_profile-{revision}.v1.json",
            "--lane-module",
            "semantic_teacher_image_edit_vast.py",
            "--receipt-out",
            "{set_root}/terminal_rehearsal-{revision}.v1.json",
        ),
        produces="{set_root}/terminal_rehearsal-{revision}.v1.json",
    ),
)


def _native_task_arena_steps(
    link: str, *, control_selection: str = "control_pair"
) -> tuple[LaneStep, ...]:
    """Return the reusable construction or controls preparation graph."""

    if link not in {"construction", "controls"}:
        raise ValueError(f"native_task_arena_preparation_link_invalid:{link}")
    controls = link == "controls"
    if (
        (not controls and control_selection != "control_pair")
        or control_selection
        not in {
            "control_pair",
            "zero_action_negative",
            "deterministic_scripted_positive",
        }
    ):
        raise ValueError("native_task_arena_control_selection_invalid")
    bundle_module = (
        "blueprint_pipeline.native_task_arena_controls_bundle"
        if controls
        else "blueprint_pipeline.native_task_arena_construction_bundle"
    )
    probe_kind = f"native-task-arena-{link}"
    bundle_argv = [
        "{python}",
        "-m",
        bundle_module,
        "--job-dir",
        "{set_root}/bundle",
        "--packet-dir",
        "{packet_dir}",
        "--runtime-source-packet-receipt",
        "{runtime_source_packet}",
        "--implementation-commit",
        "{source_commit}",
        "--container-image",
        "{container_image}",
    ]
    if controls:
        bundle_argv.extend(
            (
                "--construction-result",
                "{construction_result}",
                "--control-selection",
                control_selection,
            )
        )
        if control_selection == "deterministic_scripted_positive":
            bundle_argv.extend(
                ("--zero-action-result", "{zero_action_result}")
            )
    authority_lineage = (
        (
            "--prior-authority",
            "{prior_authority}",
            "--prior-result",
            "{prior_result}",
            "--prior-provider-zero",
            "{prior_provider_zero}",
            "--prior-spend-reconciliation",
            "{prior_spend_reconciliation}",
        )
        if controls
        else (
            "--project-spend-reconciliation",
            "{project_spend_reconciliation}",
            "--initial-provider-zero",
            "{initial_provider_zero}",
        )
    )
    predecessor_allocator = (
        (
            "--native-task-arena-construction-result",
            "{construction_result}",
        )
        if controls
        else ()
    )
    profile_predecessor = (
        ("--construction-result", "{construction_result}") if controls else ()
    )
    return (
        LaneStep(
            step_id="provider_bundle",
            argv=tuple(bundle_argv),
            produces=(
                "{set_root}/bundle/"
                "native_task_arena_provider_bundle_receipt.v1.json"
            ),
        ),
        LaneStep(
            step_id="immutable_manifest",
            argv=(
                "{python}",
                "{repository_root}/scripts/"
                "publish_task_evaluation_immutable_manifest.py",
                "--manifest",
                "{set_root}/bundle/"
                "native_task_arena_provider_bundle_receipt.v1.json",
                "--profile-builder",
                "build_native_task_arena_live_profile.py",
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
                "{repository_root}/scripts/"
                "issue_native_task_arena_paid_attempt_authority.py",
                "--bundle-receipt",
                "{set_root}/bundle/"
                "native_task_arena_provider_bundle_receipt.v1.json",
                *authority_lineage,
                "--authority-reference",
                "{authorization_reference}",
                "--authorized-by",
                "{authorized_by}",
                "--authorized-on",
                "{authorized_on}",
                "--blueprint-commit",
                "{source_commit}",
                "--max-hourly-rate-usd",
                "{maximum_hourly_rate_usd}",
                "--hard-cap-usd",
                "{hard_total_spend_cap_usd}",
                "--hard-ttl-seconds",
                "{hard_ttl_seconds}",
                "--output",
                "{set_root}/native_task_arena_paid_attempt_authority.v1.json",
            ),
            produces=(
                "{set_root}/native_task_arena_paid_attempt_authority.v1.json"
            ),
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
                probe_kind,
                "--native-task-arena-packet",
                "{packet_dir}",
                "--native-task-arena-runtime-source-packet",
                "{runtime_source_packet}",
                "--native-task-arena-bundle-receipt",
                "{set_root}/bundle/"
                "native_task_arena_provider_bundle_receipt.v1.json",
                "--native-task-arena-attempt-authority",
                "{set_root}/native_task_arena_paid_attempt_authority.v1.json",
                *predecessor_allocator,
                "--adp-job-dir",
                "{set_root}/dry-run-job/job",
                "--adp-max-hourly-rate-usd",
                "{maximum_hourly_rate_usd}",
                "--adp-max-spend-usd",
                "{hard_total_spend_cap_usd}",
                "--adp-hard-ttl-seconds",
                "{hard_ttl_seconds}",
            ),
            produces="{set_root}/allocator_dry_run.v1.json",
            repeated_argv=(
                ("--adp-machine-avoidlist", "machine_avoidlist"),
            ),
        ),
        LaneStep(
            step_id="live_profile",
            argv=(
                "{python}",
                "{repository_root}/scripts/build_native_task_arena_live_profile.py",
                link,
                "--packet-dir",
                "{packet_dir}",
                "--bundle-receipt",
                "{set_root}/bundle/"
                "native_task_arena_provider_bundle_receipt.v1.json",
                "--attempt-authority",
                "{set_root}/native_task_arena_paid_attempt_authority.v1.json",
                "--runtime-source-packet",
                "{runtime_source_packet}",
                "--source-commit",
                "{source_commit}",
                "--scene-id",
                "{scene_id}",
                "--task-id",
                "{task_id}",
                "--raw-manifest-uri",
                "{set_root}/manifest_publication_receipt.v1.json",
                "--revision",
                "{revision}",
                "--max-hourly-rate-usd",
                "{maximum_hourly_rate_usd}",
                "--max-spend-usd",
                "{hard_total_spend_cap_usd}",
                "--hard-ttl-seconds",
                "{hard_ttl_seconds}",
                *profile_predecessor,
                "--output",
                "{set_root}/live_profile-{revision}.v1.json",
            ),
            produces="{set_root}/live_profile-{revision}.v1.json",
            exports=(("profile_id", "profile_id"),),
            repeated_argv=(("--machine-avoidlist", "machine_avoidlist"),),
        ),
        LaneStep(
            step_id="terminal_rehearsal",
            argv=(
                "{python}",
                "{repository_root}/scripts/rehearse_lane_terminal_contract.py",
                "--profile",
                "{set_root}/live_profile-{revision}.v1.json",
                "--lane-module",
                "adp_isaac_lab_arena_vast.py",
                "--lane",
                f"native_task_arena_{link}",
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
                "{set_root}/live_profile-{revision}.v1.json",
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
    "semantic_teacher_image_edit": SEMANTIC_TEACHER_IMAGE_EDIT_STEPS,
    "native_task_arena_construction": _native_task_arena_steps("construction"),
    "native_task_arena_controls": _native_task_arena_steps("controls"),
    "native_task_arena_zero_action": _native_task_arena_steps(
        "controls", control_selection="zero_action_negative"
    ),
    "native_task_arena_scripted_positive": _native_task_arena_steps(
        "controls", control_selection="deterministic_scripted_positive"
    ),
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
        os.chown(root, -1, group_entry.gr_gid)
        root.chmod(stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP)
    except OSError as exc:
        raise PaidLaneLaunchPreparationError(
            "paid_lane_set_root_permission_install_failed"
        ) from exc
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


def validate_lane_context(lane: str, context: Mapping[str, Any]) -> None:
    """Reject an incomplete context before the first command is launched.

    A placeholder resolved to an empty string is how a wrong path reaches a
    real provider, so an unsupplied name is an error rather than a default.
    """

    steps = LANES.get(lane)
    if steps is None:
        raise PaidLaneLaunchPreparationError(f"paid_lane_unknown:{lane}")
    available = {str(key) for key, value in context.items() if str(value) != ""}
    missing: list[str] = []
    for step in steps:
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _default_runner(argv: Sequence[str]) -> int:
    return subprocess.run(list(argv), check=False).returncode


def prepare_paid_lane_launch(
    lane: str,
    context: Mapping[str, Any],
    *,
    runner: Callable[[Sequence[str]], int] = _default_runner,
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
        _prepare_set_root_for_service(
            str(context["set_root"]),
            service_account=account,
            service_group=group,
        )
    resolved: dict[str, Any] = dict(context)
    completed: list[dict[str, Any]] = []
    blockers: list[str] = []
    for step in LANES[lane]:
        argv = [_render(fragment, resolved) for fragment in step.argv]
        for flag, context_name in step.repeated_argv:
            for value in _repeated_values(resolved.get(context_name)):
                argv.extend((flag, value))
        produces = Path(_render(step.produces, resolved))
        returncode = runner(argv)
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
        argv = [_render(fragment, resolved) for fragment in step.argv]
        for flag, context_name in step.repeated_argv:
            for value in _repeated_values(resolved.get(context_name)):
                argv.extend((flag, value))
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
    output: Path, descriptor: int, receipt: Mapping[str, Any]
) -> None:
    payload = (json.dumps(dict(receipt), indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    with os.fdopen(descriptor, "wb") as stream:
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
    if (
        not source.is_file()
        or not isinstance(value, Mapping)
        or value.get("schema_version") != expected_schema
        or value.get("status") != expected_status
        or value.get("scene_id") != scene_id
        or value.get(digest_field) != expected_digest
        or value.get(digest_field)
        != _canonical_artifact_digest(value, digest_field=digest_field)
    ):
        raise PaidLaneLaunchPreparationError(
            "native_task_arena_scene_claim_reference_invalid"
        )
    return source, dict(value)


def _load_rights_evidence(value: Any) -> list[dict[str, Any]]:
    """Reopen the exact terms and human authority used for rights admission."""

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PaidLaneLaunchPreparationError(
            "native_task_arena_rights_evidence_invalid"
        )
    retained: list[dict[str, Any]] = []
    roles: set[str] = set()
    paths: set[Path] = set()
    for raw in value:
        if not isinstance(raw, Mapping):
            raise PaidLaneLaunchPreparationError(
                "native_task_arena_rights_evidence_invalid"
            )
        role = str(raw.get("role") or "")
        expected = str(raw.get("sha256") or "")
        unresolved = Path(str(raw.get("path") or "")).expanduser()
        if unresolved.is_symlink():
            raise PaidLaneLaunchPreparationError(
                "native_task_arena_rights_evidence_invalid"
            )
        path = unresolved.resolve()
        if (
            role not in {
                "publisher_terms",
                "publisher_readme",
                "upstream_license",
                "human_authority_record",
            }
            or path in paths
            or not path.is_file()
            or _sha256_file(path) != expected
        ):
            raise PaidLaneLaunchPreparationError(
                "native_task_arena_rights_evidence_invalid"
            )
        paths.add(path)
        roles.add(role)
        retained.append(
            {
                "role": role,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": expected,
            }
        )
    if not {"publisher_terms", "human_authority_record"}.issubset(roles):
        raise PaidLaneLaunchPreparationError(
            "native_task_arena_rights_evidence_incomplete"
        )
    return retained


def _validate_provider_packet_source_rights(
    *, packet_receipt: Mapping[str, Any], source_manifest: Mapping[str, Any]
) -> None:
    """Require every provider-staged source binding to be upload-admitted."""

    bindings = packet_receipt.get("source_bindings")
    artifacts = source_manifest.get("artifacts")
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
    for raw in artifacts:
        if not isinstance(raw, Mapping):
            continue
        digest = str(raw.get("sha256") or "")
        size = raw.get("size_bytes")
        if raw.get("provider_upload_allowed") is True and isinstance(size, int):
            admitted.add((digest, size))
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
        if source_pair != staged_pair or staged_pair not in admitted:
            raise PaidLaneLaunchPreparationError(
                "native_task_arena_provider_source_rights_invalid"
            )


def _bound_file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


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


def _validate_prior_webapp_lineage(
    *, prior_result_path: Any, launch_receipt_path: Any, webapp_sync_path: Any
) -> dict[str, Any]:
    """Bind a continuing paid lane to the WebApp-synchronized predecessor."""

    prior_path, _prior = _load_unlinked_json(
        prior_result_path, error="native_task_arena_prior_webapp_lineage_invalid"
    )
    receipt_path, receipt = _load_unlinked_json(
        launch_receipt_path,
        error="native_task_arena_prior_webapp_lineage_invalid",
    )
    sync_path, sync = _load_unlinked_json(
        webapp_sync_path, error="native_task_arena_prior_webapp_lineage_invalid"
    )
    terminal = receipt.get("terminal_evidence")
    terminal_result = terminal.get("result") if isinstance(terminal, Mapping) else None
    response = sync.get("response")
    fields = ("launch_id", "run_id", "request_digest", "receipt_digest")
    if (
        receipt.get("schema_version") != "task_evaluation_launch_receipt.v1"
        or receipt.get("receipt_digest")
        != _canonical_artifact_digest(receipt, digest_field="receipt_digest")
        or not isinstance(terminal_result, Mapping)
        or Path(str(terminal_result.get("path") or "")).expanduser().resolve()
        != prior_path
        or terminal_result.get("exists") is not True
        or terminal_result.get("digest") != _sha256_file(prior_path)
        or sync.get("schema_version")
        != "task_evaluation_launch_webapp_sync_result.v1"
        or sync.get("status") != "succeeded"
        or sync.get("provider_mutation_performed") is not False
        or sync.get("sync_result_digest")
        != _canonical_artifact_digest(sync, digest_field="sync_result_digest")
        or not isinstance(sync.get("attempt_number"), int)
        or isinstance(sync.get("attempt_number"), bool)
        or sync["attempt_number"] < 1
        or not str(sync.get("attempted_at") or "").strip()
        or not isinstance(response, Mapping)
        or any(sync.get(field) != receipt.get(field) for field in fields)
        or any(response.get(field) != receipt.get(field) for field in fields)
    ):
        raise PaidLaneLaunchPreparationError(
            "native_task_arena_prior_webapp_lineage_invalid"
        )
    return {
        "launch_id": receipt["launch_id"],
        "run_id": receipt["run_id"],
        "request_digest": receipt["request_digest"],
        "launch_receipt": _bound_file_record(receipt_path),
        "webapp_sync": _bound_file_record(sync_path),
        "terminal_result": _bound_file_record(prior_path),
        "sync_result_digest": sync["sync_result_digest"],
    }


def _validate_zero_action_predecessor(
    *,
    prior_authority_path: Any,
    prior_result_path: Any,
    zero_action_result_path: Any,
    expected_blueprint_commit: str,
    expected_runtime_source_packet_digest: str,
    expected_container_image: str,
    expected_packet_receipt_digest: str,
) -> dict[str, Any]:
    """Prove the scored zero-action bytes came from the paid predecessor."""

    authority_path, authority = _load_unlinked_json(
        prior_authority_path,
        error="native_task_arena_zero_action_predecessor_invalid",
    )
    prior_path, prior = _load_unlinked_json(
        prior_result_path,
        error="native_task_arena_zero_action_predecessor_invalid",
    )
    zero_path, _zero = _load_unlinked_json(
        zero_action_result_path,
        error="native_task_arena_zero_action_predecessor_invalid",
    )
    artifact_path, artifact = _load_unlinked_json(
        prior.get("artifact_manifest_path"),
        error="native_task_arena_zero_action_predecessor_invalid",
    )
    attempt_root = Path(str(prior.get("attempt_root") or "")).expanduser().resolve()
    try:
        zero_relative = zero_path.relative_to(attempt_root).as_posix()
        artifact_path.relative_to(attempt_root)
    except ValueError as exc:
        raise PaidLaneLaunchPreparationError(
            "native_task_arena_zero_action_predecessor_invalid"
        ) from exc
    rows = artifact.get("files")
    matched = [
        row
        for row in rows or []
        if isinstance(row, Mapping)
        and row.get("relative_path") == zero_relative
        and "provider_runtime_evidence" in (row.get("roles") or [])
        and row.get("size_bytes") == zero_path.stat().st_size
        and row.get("sha256") == _sha256_file(zero_path)
    ]
    if (
        authority.get("schema_version")
        != "native_task_arena_paid_attempt_authority.v1"
        or authority.get("authorization_digest")
        != _canonical_artifact_digest(
            authority, digest_field="authorization_digest"
        )
        or authority.get("execution_mode") != "controls"
        or authority.get("blueprint_commit") != expected_blueprint_commit
        or authority.get("runtime_source_packet_receipt_digest")
        != expected_runtime_source_packet_digest
        or authority.get("container_image") != expected_container_image
        or authority.get("packet_receipt_digest")
        != expected_packet_receipt_digest
        or (prior.get("authorization_consumption") or {}).get(
            "authorization_digest"
        )
        != authority.get("authorization_digest")
        or prior.get("bundle_sha256") != authority.get("bundle_sha256")
        or Path(str(prior.get("native_control_result_path") or ""))
        .expanduser()
        .resolve()
        != zero_path
        or artifact.get("schema_version") != "task_evaluation_artifact_manifest.v1"
        or artifact.get("status") != "completed"
        or artifact.get("manifest_digest")
        != _canonical_artifact_digest(artifact, digest_field="manifest_digest")
        or len(matched) != 1
    ):
        raise PaidLaneLaunchPreparationError(
            "native_task_arena_zero_action_predecessor_invalid"
        )
    return {
        "prior_authority": _bound_file_record(authority_path),
        "runtime_identity": {
            "blueprint_commit": expected_blueprint_commit,
            "runtime_source_packet_receipt_digest": (
                expected_runtime_source_packet_digest
            ),
            "container_image": expected_container_image,
            "packet_receipt_digest": expected_packet_receipt_digest,
        },
        "allocator_result": _bound_file_record(prior_path),
        "artifact_manifest": _bound_file_record(artifact_path),
        "zero_action_result": _bound_file_record(zero_path),
        "artifact_manifest_digest": artifact["manifest_digest"],
    }


def _load_native_context(path: str | Path, *, expected_lane: str) -> dict[str, Any]:
    """Load and reopen independent scene/task/robot/runtime references."""

    unresolved_source = Path(path).expanduser()
    if unresolved_source.is_symlink():
        raise PaidLaneLaunchPreparationError("native_task_arena_context_invalid")
    source = unresolved_source.resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PaidLaneLaunchPreparationError(
            "native_task_arena_context_unreadable"
        ) from exc
    if (
        not isinstance(value, Mapping)
        or value.get("schema_version") != NATIVE_CONTEXT_SCHEMA_VERSION
        or value.get("lane") != expected_lane
        or not str(value.get("team_namespace") or "").strip()
    ):
        raise PaidLaneLaunchPreparationError("native_task_arena_context_invalid")
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "docs/schemas/native_task_arena_launch_preparation_context.v2.schema.json"
    )
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.Draft202012Validator(schema).validate(value)
    except (
        OSError,
        json.JSONDecodeError,
        jsonschema.SchemaError,
        jsonschema.ValidationError,
    ) as exc:
        raise PaidLaneLaunchPreparationError(
            "native_task_arena_context_schema_invalid"
        ) from exc
    references = value.get("references")
    operations = value.get("operations")
    if not isinstance(references, Mapping) or not isinstance(operations, Mapping):
        raise PaidLaneLaunchPreparationError("native_task_arena_context_invalid")
    for key in operations:
        lowered = str(key).lower()
        if any(token in lowered for token in ("password", "api_key", "secret", "token")):
            raise PaidLaneLaunchPreparationError(
                "native_task_arena_context_secret_value_forbidden"
            )
    required_reference_keys = {"scene", "task", "robot", "runtime"}
    if set(references) != required_reference_keys:
        raise PaidLaneLaunchPreparationError(
            "native_task_arena_reference_set_invalid"
        )
    scene = references["scene"]
    task = references["task"]
    robot = references["robot"]
    runtime = references["runtime"]
    if not all(isinstance(item, Mapping) for item in (scene, task, robot, runtime)):
        raise PaidLaneLaunchPreparationError("native_task_arena_reference_invalid")
    unresolved_packet_dir = Path(str(scene.get("packet_dir") or "")).expanduser()
    unresolved_runtime_source = Path(
        str(runtime.get("source_packet") or "")
    ).expanduser()
    reference_symlink_present = (
        unresolved_packet_dir.is_symlink() or unresolved_runtime_source.is_symlink()
    )
    packet_dir = unresolved_packet_dir.resolve()
    packet_receipt_path = packet_dir / "native_task_arena_packet_receipt.v1.json"
    runtime_contract_path = packet_dir / "native_task_runtime_contract.v1.json"
    runtime_source_path = unresolved_runtime_source.resolve()
    source_manifest_path, source_manifest = _load_scene_claim_reference(
        path=scene.get("source_manifest"),
        expected_digest=scene.get("source_manifest_digest"),
        expected_schema="task_evaluation_scene_source_manifest.v1",
        expected_status="retained",
        digest_field="source_manifest_digest",
        scene_id=str(scene.get("scene_id") or ""),
    )
    rights_admission_path, rights_admission = _load_scene_claim_reference(
        path=scene.get("rights_admission"),
        expected_digest=scene.get("rights_admission_digest"),
        expected_schema="task_evaluation_scene_rights_admission.v1",
        expected_status="admitted",
        digest_field="rights_admission_digest",
        scene_id=str(scene.get("scene_id") or ""),
    )
    rights_evidence = _load_rights_evidence(scene.get("rights_evidence"))
    try:
        packet_receipt = json.loads(packet_receipt_path.read_text(encoding="utf-8"))
        runtime_contract = json.loads(runtime_contract_path.read_text(encoding="utf-8"))
        runtime_source = json.loads(runtime_source_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PaidLaneLaunchPreparationError(
            "native_task_arena_reference_unreadable"
        ) from exc
    container_image = str(runtime.get("container_image") or "")
    _validate_provider_packet_source_rights(
        packet_receipt=packet_receipt,
        source_manifest=source_manifest,
    )
    if (
        reference_symlink_present
        or packet_receipt.get("scene_id") != scene.get("scene_id")
        or packet_receipt.get("task_id") != task.get("task_id")
        or packet_receipt.get("receipt_digest")
        != scene.get("packet_receipt_digest")
        or runtime_contract.get("task_spec_digest") != task.get("task_spec_digest")
        or (runtime_contract.get("robot") or {}).get("robot_id")
        != robot.get("robot_id")
        or _canonical_mapping_digest(runtime_contract.get("robot") or {})
        != robot.get("configuration_digest")
        or runtime_source.get("receipt_digest")
        != runtime.get("source_packet_receipt_digest")
        or rights_admission.get("private_provider_processing_allowed") is not True
        or rights_admission.get("provider_training_allowed") is not False
        or not isinstance(
            rights_admission.get("public_redistribution_allowed"), bool
        )
        or "@sha256:" not in container_image
        or len(container_image.rsplit("@sha256:", 1)[-1]) != 64
    ):
        raise PaidLaneLaunchPreparationError(
            "native_task_arena_reference_binding_invalid"
        )
    prior_webapp_lineage = None
    if expected_lane != "native_task_arena_construction":
        prior_webapp_lineage = _validate_prior_webapp_lineage(
            prior_result_path=operations.get("prior_result"),
            launch_receipt_path=operations.get("prior_launch_receipt"),
            webapp_sync_path=operations.get("prior_webapp_sync"),
        )
    zero_action_predecessor = None
    if expected_lane == "native_task_arena_scripted_positive":
        zero_action_predecessor = _validate_zero_action_predecessor(
            prior_authority_path=operations.get("prior_authority"),
            prior_result_path=operations.get("prior_result"),
            zero_action_result_path=operations.get("zero_action_result"),
            expected_blueprint_commit=str(operations.get("source_commit") or ""),
            expected_runtime_source_packet_digest=str(
                runtime.get("source_packet_receipt_digest") or ""
            ),
            expected_container_image=container_image,
            expected_packet_receipt_digest=str(
                scene.get("packet_receipt_digest") or ""
            ),
        )
    context = dict(operations)
    forbidden_overrides = {
        "packet_dir",
        "scene_id",
        "task_id",
        "runtime_source_packet",
        "reference_bindings",
    }
    if forbidden_overrides.intersection(context):
        raise PaidLaneLaunchPreparationError(
            "native_task_arena_reference_override_forbidden"
        )
    context.update(
        {
            "packet_dir": str(packet_dir),
            "scene_id": str(scene["scene_id"]),
            "task_id": str(task["task_id"]),
            "runtime_source_packet": str(runtime_source_path),
            "container_image": container_image,
            "reference_bindings": {
                "team_namespace": str(value["team_namespace"]),
                "scene": dict(scene),
                "task": dict(task),
                "robot": dict(robot),
                "runtime": dict(runtime),
                "context_path": str(source),
                "context_sha256": _sha256_file(source),
                "source_manifest_path": str(source_manifest_path),
                "rights_admission_path": str(rights_admission_path),
                "rights_evidence": rights_evidence,
                **(
                    {"prior_webapp_lineage": prior_webapp_lineage}
                    if prior_webapp_lineage is not None
                    else {}
                ),
                **(
                    {"zero_action_predecessor": zero_action_predecessor}
                    if zero_action_predecessor is not None
                    else {}
                ),
            },
            "python": str(operations.get("python") or sys.executable),
            "service_account": str(
                operations.get("service_account") or DEFAULT_SERVICE_ACCOUNT
            ),
            "service_group": str(
                operations.get("service_group") or DEFAULT_SERVICE_GROUP
            ),
            "machine_avoidlist": operations.get("machine_avoidlist") or "",
        }
    )
    source_commit = str(context.get("source_commit") or "")
    if len(source_commit) != 40 or any(ch not in "0123456789abcdef" for ch in source_commit):
        raise PaidLaneLaunchPreparationError(
            "native_task_arena_source_commit_invalid"
        )
    return context


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
            context = _load_native_context(args.context_file, expected_lane=args.lane)
        else:
            context = _context_from_args(args)
        receipt = (
            validate_paid_lane_launch(args.lane, context)
            if args.validate_only
            else prepare_paid_lane_launch(args.lane, context)
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
        _write_reserved_receipt(reserved_output, reserved_descriptor, receipt)
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
