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
leaving a half-prepared lane behind.  A step that exits non-zero, or that exits
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

PREPARATION_SCHEMA_VERSION = "paid_lane_launch_preparation.v1"
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

LANES: dict[str, tuple[LaneStep, ...]] = {
    "semantic_teacher_image_edit": SEMANTIC_TEACHER_IMAGE_EDIT_STEPS,
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
        "step_count": len(LANES[lane]),
        "completed_steps": completed,
        "blockers": blockers,
        "paid_inference_performed": False,
        "provider_allocation_performed": False,
    }


def _context_from_args(args: argparse.Namespace) -> dict[str, str]:
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
        "task_count": str(args.task_count),
        "camera_count": str(args.camera_count),
        "maximum_hourly_rate_usd": str(args.maximum_hourly_rate_usd),
        "hard_total_spend_cap_usd": str(args.hard_total_spend_cap_usd),
        "hard_ttl_seconds": str(args.hard_ttl_seconds),
        "aggregate_goal_spend_before_usd": str(args.aggregate_goal_spend_before_usd),
        "aggregate_goal_spend_cap_usd": str(args.aggregate_goal_spend_cap_usd),
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", required=True, choices=sorted(LANES))
    parser.add_argument("--set-root", required=True)
    parser.add_argument("--packet", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--token-file", required=True)
    parser.add_argument("--destination-prefix", required=True)
    parser.add_argument("--runtime-image-identity", required=True)
    parser.add_argument("--profile-dir", required=True)
    parser.add_argument("--webapp-catalog-out", required=True)
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
    parser.add_argument("--pod-name", required=True)
    parser.add_argument("--revision", default="r1")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--authorization-reference", required=True)
    parser.add_argument("--authorized-by", required=True)
    parser.add_argument("--authorized-on", required=True)
    parser.add_argument("--backend-entry-digest", required=True)
    parser.add_argument("--task-count", type=int, required=True)
    parser.add_argument("--camera-count", type=int, required=True)
    parser.add_argument("--maximum-hourly-rate-usd", type=float, required=True)
    parser.add_argument("--hard-total-spend-cap-usd", type=float, required=True)
    parser.add_argument("--hard-ttl-seconds", type=int, required=True)
    parser.add_argument("--aggregate-goal-spend-before-usd", type=float, required=True)
    parser.add_argument("--aggregate-goal-spend-cap-usd", type=float, required=True)
    parser.add_argument("--prior-spend-reconciliation")
    parser.add_argument(
        "--excluded-machine-id",
        action="append",
        default=[],
        type=int,
        help="Vast machine ID the immutable live profile must exclude; repeatable.",
    )
    parser.add_argument("--set", action="append", help="Extra name=value context entry")
    parser.add_argument("--receipt-out")
    args = parser.parse_args(argv)

    try:
        receipt = prepare_paid_lane_launch(args.lane, _context_from_args(args))
    except PaidLaneLaunchPreparationError as exc:
        print(json.dumps({"status": "blocked", "blockers": [str(exc)]}, sort_keys=True))
        return 2
    if args.receipt_out:
        out = Path(args.receipt_out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt["status"] == "prepared" else 2


__all__ = [
    "LANES",
    "PREPARATION_SCHEMA_VERSION",
    "LaneStep",
    "PaidLaneLaunchPreparationError",
    "prepare_paid_lane_launch",
    "step_placeholders",
    "validate_lane_context",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
