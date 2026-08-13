"""Single-use, file-backed authority for the SAM 3.1 source-track lane."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from .common import ensure_dir, utc_now_iso, write_json
from .decision_evidence_contracts import canonical_digest
from .paid_attempt_authority import normalize_active_instance_allowlist
from .sam31_gpu_admission import REQUEST_SCHEMA_VERSION
from .sam31_source_track_canary_worker import BUNDLE_RECEIPT_SCHEMA_VERSION
from .spend_authority_consumption_root import prepare_consumption_root


AUTHORITY_SCHEMA_VERSION = "semantic_sam31_paid_attempt_authority.v1"
CONSUMPTION_SCHEMA_VERSION = "semantic_sam31_authority_consumption.v1"
MAX_HARD_CAP_USD = 2.0
MAX_TTL_SECONDS = 3_600


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}


def _read(path: Path, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(code) from exc
    if path.is_symlink() or not isinstance(value, dict):
        raise ValueError(code)
    return value


def _finite(value: Any, *, minimum: float = 0.0) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= minimum
    )


def _bound_record(value: Any, code: str) -> Path:
    if not isinstance(value, Mapping):
        raise ValueError(code)
    path = Path(str(value.get("path") or "")).expanduser().resolve()
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != value.get("size_bytes")
        or _sha256(path) != value.get("sha256")
    ):
        raise ValueError(code)
    return path


def materialize_sam31_paid_attempt_authority(
    *,
    request_path: str | Path,
    bundle_path: str | Path,
    bundle_receipt_path: str | Path,
    authorization_reference: str,
    authorized_by: str,
    authorized_on: str,
    blueprint_commit: str,
    max_hourly_rate_usd: float,
    hard_cap_usd: float,
    hard_ttl_seconds: int,
    aggregate_goal_spend_before_attempt_usd: float,
    aggregate_goal_spend_cap_usd: float,
    output_path: str | Path,
    allowed_active_instance_ids: Sequence[int] = (),
    prior_spend_reconciliation_path: str | Path | None = None,
) -> dict[str, Any]:
    """Seal one zero-retry SAM allocation against exact request and bundle bytes."""

    request_file = Path(request_path).expanduser().resolve()
    bundle_file = Path(bundle_path).expanduser().resolve()
    receipt_file = Path(bundle_receipt_path).expanduser().resolve()
    request = _read(request_file, "sam31_paid_authority_request_invalid")
    receipt = _read(receipt_file, "sam31_paid_authority_bundle_receipt_invalid")
    bundle_digest = _sha256(bundle_file) if bundle_file.is_file() else ""
    request_digest = canonical_digest(request, digest_field="request_digest")
    allowed = tuple(sorted({int(value) for value in allowed_active_instance_ids}))
    prior_record = None
    if prior_spend_reconciliation_path is not None:
        prior_path = Path(prior_spend_reconciliation_path).expanduser().resolve()
        prior = _read(prior_path, "sam31_prior_spend_reconciliation_invalid")
        digest_field = "receipt_digest"
        if (
            prior.get("status") != "all_supplemental_spend_terminal_and_provider_zero"
            or prior.get(digest_field) != canonical_digest(prior, digest_field=digest_field)
            or not _finite(prior.get("total_cost_usd"))
            or float(prior["total_cost_usd"]) != float(aggregate_goal_spend_before_attempt_usd)
        ):
            raise ValueError("sam31_prior_spend_reconciliation_invalid")
        prior_record = {**_record(prior_path), "receipt_digest": prior[digest_field]}
    if (
        request.get("schema_version") != REQUEST_SCHEMA_VERSION
        or request.get("request_digest") != request_digest
        or request.get("source_commit_sha") != blueprint_commit
        or receipt.get("schema_version") != BUNDLE_RECEIPT_SCHEMA_VERSION
        or receipt.get("receipt_digest")
        != canonical_digest(receipt, digest_field="receipt_digest")
        or (receipt.get("bundle") or {}).get("sha256") != bundle_digest
        or (receipt.get("bundle") or {}).get("size_bytes") != bundle_file.stat().st_size
        or request.get("input_bundle_digest") != bundle_digest
        or request.get("input_bundle_size_bytes") != bundle_file.stat().st_size
        or request.get("retry_cap") != 0
        or request.get("max_spend_usd") != hard_cap_usd
        or request.get("hard_ttl_seconds") != hard_ttl_seconds
        or not str(request.get("authority_id") or "").strip()
        or not authorization_reference.strip()
        or not authorized_by.strip()
        or not authorized_on.strip()
        or not 0 < max_hourly_rate_usd <= hard_cap_usd <= MAX_HARD_CAP_USD
        or not 1 <= hard_ttl_seconds <= MAX_TTL_SECONDS
        or max_hourly_rate_usd * hard_ttl_seconds / 3600 > hard_cap_usd
        or not _finite(aggregate_goal_spend_before_attempt_usd)
        or not _finite(aggregate_goal_spend_cap_usd, minimum=0.000001)
        or aggregate_goal_spend_before_attempt_usd + hard_cap_usd
        > aggregate_goal_spend_cap_usd
        or (aggregate_goal_spend_before_attempt_usd > 0 and prior_record is None)
        or any(value <= 0 for value in allowed)
    ):
        raise ValueError("sam31_paid_authority_configuration_invalid")
    authority: dict[str, Any] = {
        "schema_version": AUTHORITY_SCHEMA_VERSION,
        "authority_kind": "explicit_user_direction_in_current_goal",
        "authority_reference": authorization_reference.strip(),
        "authorized_by": authorized_by.strip(),
        "authorized_on": authorized_on.strip(),
        "purpose": "one_shot_semantic_sam31_source_tracks",
        "provider": "vast",
        "paid_compute_authorized": True,
        "maximum_paid_attempts": 1,
        "maximum_provider_allocations": 1,
        "maximum_automatic_retries": 0,
        "automatic_paid_retry_authorized": False,
        "zero_retry": True,
        "request": _record(request_file),
        "request_digest": request_digest,
        "request_authority_id": request["authority_id"],
        "bundle": _record(bundle_file),
        "bundle_receipt": _record(receipt_file),
        "bundle_receipt_digest": receipt["receipt_digest"],
        "blueprint_commit": blueprint_commit,
        "worker_image_digest": request.get("worker_image_digest"),
        "hard_attempt_spend_cap_usd": hard_cap_usd,
        "maximum_hourly_rate_usd": max_hourly_rate_usd,
        "maximum_single_resource_ttl_seconds": hard_ttl_seconds,
        "aggregate_goal_spend_before_attempt_usd": aggregate_goal_spend_before_attempt_usd,
        "aggregate_goal_spend_cap_usd": aggregate_goal_spend_cap_usd,
        "prior_spend_reconciliation": prior_record,
        "active_instance_allowlist": {
            "external_provider_owned": list(allowed),
            "same_goal_concurrent": [],
        },
        "raw_nonredistributable_bytes_uploaded": False,
        "canonical_interiorgs_uploaded_or_mutated": False,
        "simulator_output_is_not_physical_evidence": True,
        "authorization_digest": "",
    }
    authority["authorization_digest"] = canonical_digest(
        authority, digest_field="authorization_digest"
    )
    output = Path(output_path).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise ValueError("sam31_paid_authority_output_exists")
    ensure_dir(output.parent)
    write_json(output, authority)
    validate_sam31_paid_attempt_authority(
        authority,
        request=request,
        bundle_path=bundle_file,
        bundle_receipt=receipt,
        blueprint_commit=blueprint_commit,
        max_hourly_rate_usd=max_hourly_rate_usd,
        hard_cap_usd=hard_cap_usd,
        hard_ttl_seconds=hard_ttl_seconds,
        allowed_active_instance_ids=allowed,
    )
    return authority


def validate_sam31_paid_attempt_authority(
    authority: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    bundle_path: str | Path,
    bundle_receipt: Mapping[str, Any],
    blueprint_commit: str,
    max_hourly_rate_usd: float,
    hard_cap_usd: float,
    hard_ttl_seconds: int,
    allowed_active_instance_ids: Sequence[int] = (),
) -> dict[str, Any]:
    value = dict(authority)
    bundle_file = Path(bundle_path).expanduser().resolve()
    errors: list[str] = []
    expected = {
        "schema_version": AUTHORITY_SCHEMA_VERSION,
        "purpose": "one_shot_semantic_sam31_source_tracks",
        "provider": "vast",
        "paid_compute_authorized": True,
        "maximum_paid_attempts": 1,
        "maximum_provider_allocations": 1,
        "maximum_automatic_retries": 0,
        "automatic_paid_retry_authorized": False,
        "zero_retry": True,
        "request_digest": request.get("request_digest"),
        "request_authority_id": request.get("authority_id"),
        "bundle_receipt_digest": bundle_receipt.get("receipt_digest"),
        "blueprint_commit": blueprint_commit,
        "worker_image_digest": request.get("worker_image_digest"),
        "hard_attempt_spend_cap_usd": hard_cap_usd,
        "maximum_hourly_rate_usd": max_hourly_rate_usd,
        "maximum_single_resource_ttl_seconds": hard_ttl_seconds,
        "raw_nonredistributable_bytes_uploaded": False,
        "canonical_interiorgs_uploaded_or_mutated": False,
        "simulator_output_is_not_physical_evidence": True,
    }
    errors.extend(
        f"{key}_mismatch" for key, expected_value in expected.items() if value.get(key) != expected_value
    )
    if value.get("authorization_digest") != canonical_digest(
        value, digest_field="authorization_digest"
    ):
        errors.append("digest_invalid")
    observed = normalize_active_instance_allowlist(value.get("active_instance_allowlist"))
    expected_allowlist = {
        "external_provider_owned": tuple(sorted(set(allowed_active_instance_ids))),
        "same_goal_concurrent": (),
    }
    if observed != expected_allowlist:
        errors.append("active_instance_allowlist_mismatch")
    try:
        if _bound_record(value.get("bundle"), "bundle_unbound") != bundle_file:
            errors.append("bundle_path_mismatch")
        request_path = _bound_record(value.get("request"), "request_unbound")
        receipt_path = _bound_record(value.get("bundle_receipt"), "bundle_receipt_unbound")
        if _read(request_path, "request_invalid") != dict(request):
            errors.append("request_record_mismatch")
        reopened_receipt = _read(receipt_path, "bundle_receipt_invalid")
        if (
            reopened_receipt != dict(bundle_receipt)
            or bundle_receipt.get("receipt_digest")
            != canonical_digest(bundle_receipt, digest_field="receipt_digest")
            or (bundle_receipt.get("bundle") or {}).get("sha256") != _sha256(bundle_file)
            or (bundle_receipt.get("bundle") or {}).get("size_bytes")
            != bundle_file.stat().st_size
        ):
            errors.append("bundle_receipt_mismatch")
        prior = value.get("prior_spend_reconciliation")
        if prior is not None:
            prior_path = _bound_record(prior, "prior_spend_unbound")
            prior_value = _read(prior_path, "prior_spend_invalid")
            if (
                prior.get("receipt_digest") != prior_value.get("receipt_digest")
                or prior_value.get("receipt_digest")
                != canonical_digest(prior_value, digest_field="receipt_digest")
                or prior_value.get("total_cost_usd")
                != value.get("aggregate_goal_spend_before_attempt_usd")
            ):
                errors.append("prior_spend_mismatch")
        elif value.get("aggregate_goal_spend_before_attempt_usd") != 0:
            errors.append("prior_spend_missing")
    except ValueError as exc:
        errors.append(str(exc))
    before = value.get("aggregate_goal_spend_before_attempt_usd")
    cap = value.get("aggregate_goal_spend_cap_usd")
    if not _finite(before) or not _finite(cap, minimum=0.000001) or before + hard_cap_usd > cap:
        errors.append("aggregate_spend_invalid")
    if errors:
        raise ValueError("sam31_paid_authority_invalid:" + ",".join(sorted(set(errors))))
    return value


def consume_sam31_paid_attempt_authority_once(
    authority: Mapping[str, Any], *, blueprint_commit: str
) -> dict[str, Any]:
    """Atomically consume the authority before object-store or provider mutation."""

    digest = str(authority.get("authorization_digest") or "")
    if not digest.startswith("sha256:") or len(digest) != 71:
        return {"status": "blocked", "blockers": ["sam31_paid_authority_identity_invalid"]}
    payload = {
        "schema_version": CONSUMPTION_SCHEMA_VERSION,
        "authorization_digest": digest,
        "request_digest": authority.get("request_digest"),
        "bundle_sha256": (authority.get("bundle") or {}).get("sha256"),
        "blueprint_commit": blueprint_commit,
        "consumed_at": utc_now_iso(),
        "maximum_provider_allocations": 1,
    }
    raw = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    try:
        root = prepare_consumption_root()
    except (OSError, ValueError):
        return {"status": "blocked", "blockers": ["sam31_paid_authority_consumption_failed"]}
    try:
        stat_result = root.stat()
        if root.is_symlink() or stat_result.st_uid != os.getuid() or stat_result.st_mode & 0o077:
            raise OSError("insecure_root")
        identity = digest.removeprefix("sha256:")
        destination = root / f"sam31-{identity}.json"
        temporary = root / f".sam31-{identity}.{os.getpid()}.tmp"
        descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    except FileExistsError:
        return {"status": "blocked", "blockers": ["sam31_paid_authority_consumed"]}
    except OSError:
        return {"status": "blocked", "blockers": ["sam31_paid_authority_consumption_failed"]}
    return {
        "status": "consumed",
        "authorization_digest": digest,
        "consumption_record_sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
        "record_location_disclosed": False,
    }


__all__ = [
    "AUTHORITY_SCHEMA_VERSION",
    "consume_sam31_paid_attempt_authority_once",
    "materialize_sam31_paid_attempt_authority",
    "validate_sam31_paid_attempt_authority",
]


def main(argv: Sequence[str] | None = None) -> int:
    """Materialize a host-resident authority without performing paid mutation."""

    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--bundle-receipt", required=True)
    parser.add_argument("--authorization-reference", required=True)
    parser.add_argument("--authorized-by", required=True)
    parser.add_argument("--authorized-on", required=True)
    parser.add_argument("--blueprint-commit", required=True)
    parser.add_argument("--max-hourly-rate-usd", required=True, type=float)
    parser.add_argument("--hard-cap-usd", required=True, type=float)
    parser.add_argument("--hard-ttl-seconds", required=True, type=int)
    parser.add_argument("--aggregate-spend-before-usd", required=True, type=float)
    parser.add_argument("--aggregate-spend-cap-usd", required=True, type=float)
    parser.add_argument("--prior-spend-reconciliation")
    parser.add_argument("--allowed-active-instance-id", type=int, action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = materialize_sam31_paid_attempt_authority(
        request_path=args.request,
        bundle_path=args.bundle,
        bundle_receipt_path=args.bundle_receipt,
        authorization_reference=args.authorization_reference,
        authorized_by=args.authorized_by,
        authorized_on=args.authorized_on,
        blueprint_commit=args.blueprint_commit,
        max_hourly_rate_usd=args.max_hourly_rate_usd,
        hard_cap_usd=args.hard_cap_usd,
        hard_ttl_seconds=args.hard_ttl_seconds,
        aggregate_goal_spend_before_attempt_usd=args.aggregate_spend_before_usd,
        aggregate_goal_spend_cap_usd=args.aggregate_spend_cap_usd,
        output_path=args.output,
        allowed_active_instance_ids=args.allowed_active_instance_id,
        prior_spend_reconciliation_path=args.prior_spend_reconciliation,
    )
    print(json.dumps({"authorization_digest": result["authorization_digest"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through module CLI
    raise SystemExit(main())
