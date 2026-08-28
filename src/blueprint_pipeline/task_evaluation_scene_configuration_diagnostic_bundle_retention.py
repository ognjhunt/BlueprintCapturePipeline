"""Fail-closed reclamation of sealed diagnostic bundles never authorized to run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest, canonical_json
from .task_evaluation_scene_configuration_bundle import (
    BUNDLE_SCHEMA_VERSION,
    load_scene_configuration_provider_bundle_receipt,
)


SCHEMA_VERSION = "task_evaluation_scene_configuration_unlaunched_bundle_retention_plan.v1"
APPLY_SCHEMA_VERSION = "task_evaluation_scene_configuration_unlaunched_bundle_retention_apply.v1"
APPLY_ACKNOWLEDGEMENT = "reap-unlaunched-diagnostic-bundle"
DEFAULT_DIAGNOSTICS_ROOT = Path(
    "/var/lib/blueprint/pipeline-control-plane/scene-configuration-diagnostics"
)
DEFAULT_MINIMUM_AGE_SECONDS = 60 * 60
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_EXECUTION_MARKERS = (
    "admission.json",
    "adapter.json",
    "attempt-authority.json",
    "job",
    "preparation.json",
    "scene_configuration_paid_launch.lock",
    "warm-session-authority.json",
)
BundleValidator = Callable[..., Mapping[str, Any]]


class DiagnosticBundleRetentionError(ValueError):
    """A diagnostic bundle is not proven safe to reclaim."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    info = path.lstat()
    if path.is_symlink() or not path.is_file():
        raise DiagnosticBundleRetentionError(
            "diagnostic_bundle_retention_target_invalid"
        )
    return {
        "path": str(path),
        "device": info.st_dev,
        "inode": info.st_ino,
        "size_bytes": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "sha256": _sha256(path),
    }


def build_unlaunched_bundle_retention_plan(
    *,
    bundle_root: str | Path,
    diagnostics_root: str | Path = DEFAULT_DIAGNOSTICS_ROOT,
    minimum_age_seconds: int = DEFAULT_MINIMUM_AGE_SECONDS,
    now: float | None = None,
    bundle_validator: BundleValidator = load_scene_configuration_provider_bundle_receipt,
) -> dict[str, Any]:
    """Prove one sealed bundle has never crossed paid-attempt authority."""

    if (
        isinstance(minimum_age_seconds, bool)
        or not isinstance(minimum_age_seconds, int)
        or minimum_age_seconds < 60
    ):
        raise DiagnosticBundleRetentionError(
            "diagnostic_bundle_retention_minimum_age_invalid"
        )
    managed = Path(diagnostics_root).expanduser().resolve()
    requested = Path(bundle_root).expanduser().absolute()
    root = requested.resolve()
    try:
        relative = root.relative_to(managed)
    except ValueError as exc:
        raise DiagnosticBundleRetentionError(
            "diagnostic_bundle_retention_root_outside_managed_root"
        ) from exc
    if (
        requested.is_symlink()
        or not root.is_dir()
        or len(relative.parts) != 2
        or not relative.parts[1].startswith("bundle")
    ):
        raise DiagnosticBundleRetentionError(
            "diagnostic_bundle_retention_root_invalid"
        )
    attempt_root = root.parent
    if any((attempt_root / marker).exists() for marker in _EXECUTION_MARKERS):
        raise DiagnosticBundleRetentionError(
            "diagnostic_bundle_retention_execution_evidence_present"
        )
    receipt = root / f"{BUNDLE_SCHEMA_VERSION}.receipt.json"
    bundle = root / "task_evaluation_scene_configuration_provider_bundle.zip"
    bundle_record = _file_record(bundle)
    receipt_record = _file_record(receipt)
    try:
        validated = dict(bundle_validator(receipt, diagnostic_only=True))
    except (OSError, TypeError, ValueError) as exc:
        raise DiagnosticBundleRetentionError(
            "diagnostic_bundle_retention_receipt_invalid"
        ) from exc
    source_commit = str(validated.get("source_commit") or "")
    if (
        _COMMIT.fullmatch(source_commit) is None
        or validated.get("diagnostic_only") is not True
        or validated.get("qualification_eligible") is not False
        or Path(str(validated.get("bundle_path") or "")).resolve() != bundle
        or validated.get("bundle_sha256") != bundle_record["sha256"]
        or validated.get("bundle_size_bytes") != bundle_record["size_bytes"]
    ):
        raise DiagnosticBundleRetentionError(
            "diagnostic_bundle_retention_receipt_invalid"
        )
    observed_now = time.time() if now is None else float(now)
    youngest_mtime = max(
        bundle_record["mtime_ns"] / 1_000_000_000,
        receipt_record["mtime_ns"] / 1_000_000_000,
    )
    age_seconds = int(observed_now - youngest_mtime)
    if age_seconds < minimum_age_seconds:
        raise DiagnosticBundleRetentionError(
            "diagnostic_bundle_retention_target_too_new"
        )
    plan: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "eligible_unlaunched_diagnostic_bundle",
        "diagnostics_root": str(managed),
        "attempt_root": str(attempt_root),
        "bundle_root": str(root),
        "source_commit": source_commit,
        "minimum_age_seconds": minimum_age_seconds,
        "observed_age_seconds": age_seconds,
        "execution_markers_checked": list(_EXECUTION_MARKERS),
        "execution_evidence_present": False,
        "bundle": bundle_record,
        "bundle_receipt": receipt_record,
        "predicted_removed_bytes": bundle_record["size_bytes"]
        + receipt_record["size_bytes"],
        "provider_mutation_performed": False,
        "raw_secret_values_recorded": False,
        "plan_digest": "",
    }
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    return plan


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    payload = (canonical_json(value) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o440,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def write_unlaunched_bundle_retention_plan(
    *, destination: str | Path, **kwargs: Any
) -> dict[str, Any]:
    plan = build_unlaunched_bundle_retention_plan(**kwargs)
    _write_exclusive(Path(destination).expanduser().absolute(), plan)
    return plan


def apply_unlaunched_bundle_retention_plan(
    *,
    plan_path: str | Path,
    receipt_out: str | Path,
    acknowledgement: str,
    now: float | None = None,
    bundle_validator: BundleValidator = load_scene_configuration_provider_bundle_receipt,
) -> dict[str, Any]:
    """Revalidate an unchanged plan, then unlink only its ZIP and receipt."""

    if acknowledgement != APPLY_ACKNOWLEDGEMENT:
        raise DiagnosticBundleRetentionError(
            "diagnostic_bundle_retention_acknowledgement_missing"
        )
    source = Path(plan_path).expanduser().resolve()
    try:
        plan = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DiagnosticBundleRetentionError(
            "diagnostic_bundle_retention_plan_invalid"
        ) from exc
    if (
        source.is_symlink()
        or not isinstance(plan, dict)
        or plan.get("schema_version") != SCHEMA_VERSION
        or plan.get("plan_digest")
        != canonical_digest(plan, digest_field="plan_digest")
    ):
        raise DiagnosticBundleRetentionError(
            "diagnostic_bundle_retention_plan_invalid"
        )
    current = build_unlaunched_bundle_retention_plan(
        bundle_root=str(plan.get("bundle_root") or ""),
        diagnostics_root=str(plan.get("diagnostics_root") or ""),
        minimum_age_seconds=int(plan.get("minimum_age_seconds") or 0),
        now=now,
        bundle_validator=bundle_validator,
    )
    comparable_fields = (
        "diagnostics_root",
        "attempt_root",
        "bundle_root",
        "source_commit",
        "minimum_age_seconds",
        "execution_markers_checked",
        "execution_evidence_present",
        "bundle",
        "bundle_receipt",
        "predicted_removed_bytes",
    )
    if any(current[field] != plan.get(field) for field in comparable_fields):
        raise DiagnosticBundleRetentionError(
            "diagnostic_bundle_retention_plan_changed"
        )
    bundle = Path(str(plan["bundle"]["path"]))
    receipt = Path(str(plan["bundle_receipt"]["path"]))
    bundle.unlink()
    receipt.unlink()
    result: dict[str, Any] = {
        "schema_version": APPLY_SCHEMA_VERSION,
        "status": "unlaunched_diagnostic_bundle_reclaimed",
        "source_plan": _file_record(source),
        "source_plan_digest": plan["plan_digest"],
        "removed_paths": [str(bundle), str(receipt)],
        "removed_bytes": plan["predicted_removed_bytes"],
        "paid_attempt_authority_observed": False,
        "provider_mutation_performed": False,
        "raw_secret_values_recorded": False,
        "receipt_digest": "",
    }
    result["receipt_digest"] = canonical_digest(
        result, digest_field="receipt_digest"
    )
    _write_exclusive(Path(receipt_out).expanduser().absolute(), result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root")
    parser.add_argument("--diagnostics-root", default=str(DEFAULT_DIAGNOSTICS_ROOT))
    parser.add_argument(
        "--minimum-age-seconds", type=int, default=DEFAULT_MINIMUM_AGE_SECONDS
    )
    parser.add_argument("--plan-out")
    parser.add_argument("--apply-plan")
    parser.add_argument("--receipt-out")
    parser.add_argument("--ack")
    args = parser.parse_args(argv)
    try:
        if args.apply_plan:
            if args.bundle_root or args.plan_out or not args.receipt_out:
                raise DiagnosticBundleRetentionError(
                    "diagnostic_bundle_retention_cli_arguments_invalid"
                )
            result = apply_unlaunched_bundle_retention_plan(
                plan_path=args.apply_plan,
                receipt_out=args.receipt_out,
                acknowledgement=str(args.ack or ""),
            )
        else:
            if (
                not args.bundle_root
                or not args.plan_out
                or args.receipt_out
                or args.ack
            ):
                raise DiagnosticBundleRetentionError(
                    "diagnostic_bundle_retention_cli_arguments_invalid"
                )
            result = write_unlaunched_bundle_retention_plan(
                bundle_root=args.bundle_root,
                diagnostics_root=args.diagnostics_root,
                minimum_age_seconds=args.minimum_age_seconds,
                destination=args.plan_out,
            )
    except (OSError, TypeError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "blockers": [str(exc)],
                    "provider_mutation_performed": False,
                },
                sort_keys=True,
            )
        )
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "removed_bytes": result.get("removed_bytes", 0),
                "predicted_removed_bytes": result.get("predicted_removed_bytes", 0),
                "provider_mutation_performed": False,
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "APPLY_ACKNOWLEDGEMENT",
    "DiagnosticBundleRetentionError",
    "apply_unlaunched_bundle_retention_plan",
    "build_unlaunched_bundle_retention_plan",
    "write_unlaunched_bundle_retention_plan",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
