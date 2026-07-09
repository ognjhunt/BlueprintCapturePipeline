"""Metadata-only GPU provider API key rotation manifest helper.

The helper deliberately never reads or writes raw provider secret values. It
records whether configured secret files exist and whether a local rotation ledger
has fresh metadata for each GPU provider key.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .common import ensure_dir, read_json, utc_now_iso, write_json
from .secret_artifact_policy import (
    redacted_secret_file_status,
    secret_path_disclosure_policy,
)


GPU_PROVIDER_KEY_ROTATION_MANIFEST_SCHEMA_VERSION = (
    "gpu_provider_key_rotation_manifest.v1"
)
GPU_PROVIDER_KEY_ROTATION_LEDGER_SCHEMA_VERSION = "gpu_provider_key_rotation_ledger.v1"
DEFAULT_SECRETS_DIR = "~/.blueprint-secrets"
DEFAULT_LEDGER_FILENAME = "gpu_provider_key_rotation_ledger.json"
DEFAULT_MAX_ROTATION_AGE_DAYS = 90


@dataclass(frozen=True)
class ProviderKeyDescriptor:
    provider: str
    display_name: str
    default_secret_filename: str
    secret_file_env_vars: tuple[str, ...]
    inline_secret_env_vars: tuple[str, ...]
    api_gate_env_vars: tuple[str, ...]
    external_rotation_record_hint: str


GPU_PROVIDER_KEY_DESCRIPTORS: dict[str, ProviderKeyDescriptor] = {
    "runpod": ProviderKeyDescriptor(
        provider="runpod",
        display_name="RunPod",
        default_secret_filename="runpod_api_key",
        secret_file_env_vars=("RUNPOD_API_KEY_FILE",),
        inline_secret_env_vars=("RUNPOD_API_KEY",),
        api_gate_env_vars=("BLUEPRINT_ALLOW_RUNPOD_API_CALLS",),
        external_rotation_record_hint="RunPod API key page, secret-manager version, or ticket URL",
    ),
    "vast": ProviderKeyDescriptor(
        provider="vast",
        display_name="Vast.ai",
        default_secret_filename="vast_api_key",
        secret_file_env_vars=("VAST_API_KEY_FILE",),
        inline_secret_env_vars=("VAST_API_KEY",),
        api_gate_env_vars=("BLUEPRINT_ALLOW_VAST_API_CALLS",),
        external_rotation_record_hint="Vast.ai API key page, secret-manager version, or ticket URL",
    ),
    "lambda": ProviderKeyDescriptor(
        provider="lambda",
        display_name="Lambda Cloud",
        default_secret_filename="lambda_api_key",
        secret_file_env_vars=("LAMBDA_API_KEY_FILE",),
        inline_secret_env_vars=("LAMBDA_API_KEY",),
        api_gate_env_vars=("BLUEPRINT_ALLOW_LAMBDA_API_CALLS",),
        external_rotation_record_hint="Lambda Cloud API key page, secret-manager version, or ticket URL",
    ),
    "digitalocean": ProviderKeyDescriptor(
        provider="digitalocean",
        display_name="DigitalOcean",
        default_secret_filename="digitalocean_api_token",
        secret_file_env_vars=("DIGITALOCEAN_TOKEN_FILE", "DIGITALOCEAN_API_TOKEN_FILE"),
        inline_secret_env_vars=("DIGITALOCEAN_ACCESS_TOKEN", "DIGITALOCEAN_API_TOKEN"),
        api_gate_env_vars=(
            "BLUEPRINT_ALLOW_DIGITALOCEAN_API_CALLS",
            "BLUEPRINT_ALLOW_DIGITALOCEAN_GPU_DROPLET_LAUNCH",
        ),
        external_rotation_record_hint=(
            "DigitalOcean API token page, secret-manager version, or ticket URL"
        ),
    ),
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = value.strip()
    if candidate.endswith("Z"):
        candidate = f"{candidate[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _isoformat_utc(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _read_ledger(ledger_path: Path) -> dict[str, Any]:
    if not ledger_path.is_file():
        return {
            "schema_version": GPU_PROVIDER_KEY_ROTATION_LEDGER_SCHEMA_VERSION,
            "providers": {},
        }
    payload = read_json(ledger_path)
    providers = payload.get("providers")
    if not isinstance(providers, Mapping):
        payload["providers"] = {}
    return payload


def _provider_secret_path(
    descriptor: ProviderKeyDescriptor,
    *,
    secrets_dir: Path,
    env: Mapping[str, str],
) -> tuple[Path, str]:
    for env_name in descriptor.secret_file_env_vars:
        raw_path = env.get(env_name, "").strip()
        if raw_path:
            return Path(raw_path).expanduser(), env_name
    return secrets_dir / descriptor.default_secret_filename, "default_file"


def _file_status(path: Path) -> dict[str, Any]:
    try:
        stat_result = path.stat()
    except OSError as exc:
        status = redacted_secret_file_status(
            path,
            raw_secret_field="secret_value_recorded",
        )
        status.update({"present": False, "error": type(exc).__name__})
        return status
    status = redacted_secret_file_status(
        path,
        raw_secret_field="secret_value_recorded",
    )
    status.update(
        {
            "present": path.is_file(),
            "modified_at": _isoformat_utc(
                datetime.fromtimestamp(stat_result.st_mtime, tz=timezone.utc)
            )
            if path.is_file()
            else None,
            "mode_octal": oct(stat_result.st_mode & 0o777),
        }
    )
    return status


def _present_env_names(names: Iterable[str], env: Mapping[str, str]) -> list[str]:
    return [name for name in names if env.get(name, "").strip()]


def _selected_descriptors(providers: Sequence[str] | None) -> list[ProviderKeyDescriptor]:
    if not providers:
        return list(GPU_PROVIDER_KEY_DESCRIPTORS.values())
    unknown = [name for name in providers if name not in GPU_PROVIDER_KEY_DESCRIPTORS]
    if unknown:
        raise ValueError(f"Unknown GPU provider key descriptor(s): {', '.join(unknown)}")
    return [GPU_PROVIDER_KEY_DESCRIPTORS[name] for name in providers]


def mark_gpu_provider_key_rotated(
    *,
    provider: str,
    ledger_path: Path,
    owner: str,
    rotation_record_uri: str,
    rotated_at: datetime | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Update the local rotation ledger for one provider without handling secrets."""

    if provider not in GPU_PROVIDER_KEY_DESCRIPTORS:
        raise ValueError(f"Unknown GPU provider: {provider}")
    owner = owner.strip()
    rotation_record_uri = rotation_record_uri.strip()
    if not owner:
        raise ValueError("owner is required when marking a provider key rotated")
    if not rotation_record_uri:
        raise ValueError("rotation_record_uri is required when marking a provider key rotated")

    event_time = rotated_at or now or _utc_now()
    updated_at = now or _utc_now()
    ledger = _read_ledger(ledger_path)
    providers = dict(ledger.get("providers") if isinstance(ledger.get("providers"), Mapping) else {})
    providers[provider] = {
        "provider": provider,
        "display_name": GPU_PROVIDER_KEY_DESCRIPTORS[provider].display_name,
        "last_rotated_at": _isoformat_utc(event_time),
        "rotation_owner": owner,
        "rotation_record_uri": rotation_record_uri,
        "updated_at": _isoformat_utc(updated_at),
        "secret_value_recorded": False,
    }
    ledger = {
        "schema_version": GPU_PROVIDER_KEY_ROTATION_LEDGER_SCHEMA_VERSION,
        "updated_at": _isoformat_utc(updated_at),
        "providers": providers,
    }
    write_json(ledger_path, ledger)
    return ledger


def build_gpu_provider_key_rotation_manifest(
    *,
    secrets_dir: Path,
    ledger_path: Path,
    owner: str | None = None,
    providers: Sequence[str] | None = None,
    max_age_days: int = DEFAULT_MAX_ROTATION_AGE_DAYS,
    env: Mapping[str, str] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build a fail-closed manifest for GPU provider key rotation metadata."""

    now_dt = now or _utc_now()
    env_map = dict(env if env is not None else os.environ)
    secrets_root = secrets_dir.expanduser()
    ledger = _read_ledger(ledger_path.expanduser())
    ledger_providers = ledger.get("providers") if isinstance(ledger.get("providers"), Mapping) else {}
    selected = _selected_descriptors(providers)
    provider_payloads: dict[str, Any] = {}
    blockers: list[str] = []

    for descriptor in selected:
        secret_path, secret_path_source = _provider_secret_path(
            descriptor,
            secrets_dir=secrets_root,
            env=env_map,
        )
        file_status = _file_status(secret_path.expanduser())
        inline_env_present = _present_env_names(descriptor.inline_secret_env_vars, env_map)
        api_gate_env_present = _present_env_names(descriptor.api_gate_env_vars, env_map)
        configured_secret_present = bool(file_status.get("present") or inline_env_present)

        provider_ledger_raw = ledger_providers.get(descriptor.provider)
        provider_ledger = provider_ledger_raw if isinstance(provider_ledger_raw, Mapping) else {}
        last_rotated_at_raw = provider_ledger.get("last_rotated_at")
        last_rotated_dt = _parse_iso_datetime(last_rotated_at_raw)
        rotation_owner = (
            provider_ledger.get("rotation_owner")
            if isinstance(provider_ledger.get("rotation_owner"), str)
            else ""
        )
        rotation_record_uri = (
            provider_ledger.get("rotation_record_uri")
            if isinstance(provider_ledger.get("rotation_record_uri"), str)
            else ""
        )

        provider_blockers: list[str] = []
        if not configured_secret_present:
            provider_blockers.append("provider_secret_missing")
        if not provider_ledger:
            provider_blockers.append("rotation_metadata_missing")
        elif last_rotated_dt is None:
            provider_blockers.append("last_rotated_at_missing_or_invalid")
        if provider_ledger and not rotation_owner.strip():
            provider_blockers.append("rotation_owner_missing")
        if provider_ledger and not rotation_record_uri.strip():
            provider_blockers.append("rotation_record_uri_missing")

        days_since_rotation: int | None = None
        if last_rotated_dt is not None:
            age_seconds = max(0.0, (now_dt - last_rotated_dt).total_seconds())
            days_since_rotation = int(age_seconds // 86400)
            if days_since_rotation > max_age_days:
                provider_blockers.append("rotation_stale")

        provider_status = "passed" if not provider_blockers else "blocked"
        for blocker in provider_blockers:
            blockers.append(f"{descriptor.provider}:{blocker}")

        provider_payloads[descriptor.provider] = {
            "provider": descriptor.provider,
            "display_name": descriptor.display_name,
            "status": provider_status,
            "blockers": provider_blockers,
            "default_secret_file_path_redacted": True,
            "secret_file": file_status,
            "secret_file_source": secret_path_source,
            "inline_secret_env_vars_present": inline_env_present,
            "inline_secret_values_recorded": False,
            "api_gate_env_vars_present": api_gate_env_present,
            "configured_secret_present": configured_secret_present,
            "last_rotated_at": _isoformat_utc(last_rotated_dt) if last_rotated_dt else None,
            "days_since_rotation": days_since_rotation,
            "max_age_days": int(max_age_days),
            "rotation_owner": rotation_owner.strip() or None,
            "rotation_record_uri": rotation_record_uri.strip() or None,
            "external_rotation_record_hint": descriptor.external_rotation_record_hint,
            "secret_value_recorded": False,
        }

    manifest = {
        "schema_version": GPU_PROVIDER_KEY_ROTATION_MANIFEST_SCHEMA_VERSION,
        "generated_at": utc_now_iso() if now is None else _isoformat_utc(now_dt),
        "owner": owner.strip() if isinstance(owner, str) and owner.strip() else None,
        "status": "passed" if not blockers else "blocked",
        "blockers": blockers,
        "provider_count": len(provider_payloads),
        "max_age_days": int(max_age_days),
        "secrets_dir_path_redacted": True,
        "ledger_path_redacted": True,
        "ledger_schema_version": ledger.get("schema_version"),
        "providers": provider_payloads,
        "secret_values_recorded": False,
        "secret_artifact_policy": secret_path_disclosure_policy(),
    }
    return manifest


def write_gpu_provider_key_rotation_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    write_json(path.expanduser(), dict(manifest))


def _parse_rotated_at(value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed = _parse_iso_datetime(value)
    if parsed is None:
        raise argparse.ArgumentTypeError(f"Invalid ISO-8601 datetime: {value}")
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit or mark metadata-only GPU provider API key rotation records."
    )
    parser.add_argument(
        "--secrets-dir",
        default=DEFAULT_SECRETS_DIR,
        help="Directory holding local provider key files. Defaults to ~/.blueprint-secrets.",
    )
    parser.add_argument(
        "--ledger-path",
        default=None,
        help=(
            "Rotation ledger JSON path. Defaults to gpu_provider_key_rotation_ledger.json "
            "inside --secrets-dir."
        ),
    )
    parser.add_argument(
        "--output",
        default="gpu_provider_key_rotation_manifest.json",
        help="Manifest output path.",
    )
    parser.add_argument(
        "--owner",
        default=os.getenv("USER") or os.getenv("LOGNAME") or "",
        help="Human or team owner recorded on new rotation ledger entries.",
    )
    parser.add_argument(
        "--provider",
        action="append",
        choices=tuple(GPU_PROVIDER_KEY_DESCRIPTORS),
        help="Provider to audit. Repeat to scope the manifest; defaults to all providers.",
    )
    parser.add_argument(
        "--max-age-days",
        type=int,
        default=DEFAULT_MAX_ROTATION_AGE_DAYS,
        help="Maximum accepted age for rotation metadata.",
    )
    parser.add_argument(
        "--mark-rotated",
        choices=tuple(GPU_PROVIDER_KEY_DESCRIPTORS),
        help="Update the ledger for one provider before writing the manifest.",
    )
    parser.add_argument(
        "--rotation-record-uri",
        default="",
        help="External record proving the provider-side rotation event.",
    )
    parser.add_argument(
        "--rotated-at",
        default=None,
        help="ISO-8601 rotation timestamp. Defaults to now when --mark-rotated is used.",
    )
    parser.add_argument(
        "--fail-on-blocked",
        action="store_true",
        help="Return exit code 1 when the generated manifest status is blocked.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    secrets_dir = Path(args.secrets_dir).expanduser()
    ledger_path = (
        Path(args.ledger_path).expanduser()
        if args.ledger_path
        else secrets_dir / DEFAULT_LEDGER_FILENAME
    )

    if args.mark_rotated:
        try:
            rotated_at = _parse_rotated_at(args.rotated_at)
            mark_gpu_provider_key_rotated(
                provider=args.mark_rotated,
                ledger_path=ledger_path,
                owner=args.owner,
                rotation_record_uri=args.rotation_record_uri,
                rotated_at=rotated_at,
            )
        except (OSError, ValueError, argparse.ArgumentTypeError) as exc:
            parser.error(str(exc))

    manifest = build_gpu_provider_key_rotation_manifest(
        secrets_dir=secrets_dir,
        ledger_path=ledger_path,
        owner=args.owner,
        providers=args.provider,
        max_age_days=args.max_age_days,
    )
    output_path = Path(args.output).expanduser()
    ensure_dir(output_path.parent)
    write_gpu_provider_key_rotation_manifest(output_path, manifest)
    print(json.dumps(manifest, indent=2))
    if args.fail_on_blocked and manifest.get("status") != "passed":
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
