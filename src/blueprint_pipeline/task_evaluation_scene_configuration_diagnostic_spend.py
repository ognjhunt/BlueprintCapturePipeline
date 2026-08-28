"""Seal terminal spend evidence for one direct scene-configuration diagnostic.

This is deliberately separate from the Website launch/billing extractor.  A
direct diagnostic is non-qualifying and has no Website launch identity, but a
paid diagnostic still needs an authoritative, recursively reopenable terminal
record before its charge can enter the project ledger.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_scene_configuration_bundle import (
    load_scene_configuration_provider_bundle_receipt,
)
from .task_evaluation_scene_configuration_paid_authority import (
    AUTHORITY_SCHEMA_VERSION,
    validate_scene_configuration_paid_authority,
)
from .vast_evidence_contracts import (
    SCENE_CONFIGURATION_DIAGNOSTIC_RESULT_SCHEMA_VERSION,
    VAST_PROVIDER_ADAPTER_RESULT_SCHEMA_VERSION,
    VAST_TEARDOWN_SCHEMA_VERSION,
    valid_vast_provider_zero_api_call,
)

DIAGNOSTIC_RESULT_SCHEMA_VERSION = SCENE_CONFIGURATION_DIAGNOSTIC_RESULT_SCHEMA_VERSION


SCHEMA_VERSION = "task_evaluation_scene_configuration_diagnostic_terminal_evidence.v1"
STATUS = "diagnostic_attempt_terminal_and_vast_provider_zero"
_SECRET_FIELDS = frozenset(
    {
        "api_key",
        "api_key_value",
        "authorization_header",
        "bearer_token",
        "credential",
        "credential_value",
        "openai_api_key",
        "password",
        "secret",
        "secret_value",
        "token",
    }
)


class SceneConfigurationDiagnosticSpendError(ValueError):
    """A direct diagnostic cannot be admitted to the spend ledger."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read(path: str | Path, *, code: str) -> tuple[Path, dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SceneConfigurationDiagnosticSpendError(code) from exc
    if source.is_symlink() or not source.is_file() or not isinstance(value, dict):
        raise SceneConfigurationDiagnosticSpendError(code)
    return source, value


def _record(
    path: Path, value: Mapping[str, Any], *, digest_field: str | None
) -> dict[str, Any]:
    record = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "digest_field": digest_field,
        "schema_version": value.get("schema_version"),
    }
    if digest_field is None:
        record["exact_bytes_binding"] = "sha256"
        return record
    digest = value.get(digest_field)
    if digest != canonical_digest(value, digest_field=digest_field):
        raise SceneConfigurationDiagnosticSpendError(
            "scene_configuration_diagnostic_spend_source_digest_invalid"
        )
    record["receipt_digest"] = digest
    return record


def _reopen_record(
    record: Any, *, role: str, digest_field: str | None
) -> tuple[Path, dict[str, Any]]:
    if not isinstance(record, Mapping):
        raise SceneConfigurationDiagnosticSpendError(
            f"scene_configuration_diagnostic_spend_{role}_record_invalid"
        )
    path, value = _read(
        str(record.get("path") or ""),
        code=f"scene_configuration_diagnostic_spend_{role}_invalid",
    )
    if dict(record) != _record(path, value, digest_field=digest_field):
        raise SceneConfigurationDiagnosticSpendError(
            f"scene_configuration_diagnostic_spend_{role}_record_invalid"
        )
    return path, value


def _timestamp(value: Any, *, code: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise SceneConfigurationDiagnosticSpendError(code)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SceneConfigurationDiagnosticSpendError(code) from exc
    if parsed.tzinfo is None:
        raise SceneConfigurationDiagnosticSpendError(code)
    return parsed.astimezone(timezone.utc)


def _contains_secret_material(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key).lower() in _SECRET_FIELDS
            or _contains_secret_material(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_secret_material(child) for child in value)
    return isinstance(value, str) and value.startswith(("sk-", "Bearer "))


def _adapter_secret_proof_valid(adapter: Mapping[str, Any]) -> bool:
    """Accept the legacy v1 omission only with both exact secret proofs.

    Current adapter receipts write ``raw_secret_values_recorded=false``.
    Historical v1 receipts predate that aggregate field, but already prove the
    two underlying properties independently.  An explicitly present aggregate
    field never falls back: null/true/other values remain refusals.
    """

    if "raw_secret_values_recorded" in adapter:
        return adapter.get("raw_secret_values_recorded") is False
    return (
        adapter.get("schema_version") == VAST_PROVIDER_ADAPTER_RESULT_SCHEMA_VERSION
        and adapter.get("status") in {"completed", "blocked"}
        and adapter.get("raw_api_key_stored") is False
        and adapter.get("secret_values_in_artifact") is False
    )


def _exact_instance_ids(value: Any) -> list[int]:
    if (
        not isinstance(value, list)
        or len(value) != 1
        or isinstance(value[0], bool)
        or not isinstance(value[0], int)
        or value[0] <= 0
    ):
        raise SceneConfigurationDiagnosticSpendError(
            "scene_configuration_diagnostic_spend_instance_identity_invalid"
        )
    return value


def _validate_sources(
    *,
    authority: Mapping[str, Any],
    result_path: Path,
    result: Mapping[str, Any],
    adapter_path: Path,
    adapter: Mapping[str, Any],
    teardown_path: Path,
    teardown: Mapping[str, Any],
    provider_zero: Mapping[str, Any],
) -> None:
    bundle_record = authority.get("bundle_receipt")
    if not isinstance(bundle_record, Mapping):
        raise SceneConfigurationDiagnosticSpendError(
            "scene_configuration_diagnostic_spend_authority_invalid"
        )
    bundle_path = Path(str(bundle_record.get("path") or "")).expanduser().resolve()
    try:
        bundle = load_scene_configuration_provider_bundle_receipt(
            bundle_path,
            expected_source_commit=str(authority.get("source_commit") or ""),
            diagnostic_only=True,
        )
        if (
            bundle_path.is_symlink()
            or bundle_record.get("size_bytes") != bundle_path.stat().st_size
            or bundle_record.get("sha256") != _sha256(bundle_path)
        ):
            raise ValueError("bundle record mismatch")
        validate_scene_configuration_paid_authority(
            authority, bundle_receipt=bundle
        )
    except (OSError, TypeError, ValueError) as exc:
        raise SceneConfigurationDiagnosticSpendError(
            "scene_configuration_diagnostic_spend_authority_invalid"
        ) from exc

    authority_digest = authority.get("authority_digest")
    run_id = authority.get("run_id")
    bundle_sha256 = authority.get("bundle_sha256")
    consumption = result.get("authorization_consumption")
    watchdog = result.get("independent_watchdog")
    adapter_ids = _exact_instance_ids(adapter.get("vast_instance_ids"))
    teardown_ids = _exact_instance_ids(teardown.get("vast_instance_ids"))
    result_ids = _exact_instance_ids(watchdog.get("instance_ids") if isinstance(watchdog, Mapping) else None)
    if (
        any(
            _contains_secret_material(source)
            for source in (authority, result, adapter, teardown, provider_zero)
        )
        or
        authority.get("schema_version") != AUTHORITY_SCHEMA_VERSION
        or authority.get("diagnostic_only") is not True
        or authority.get("qualification_eligible") is not False
        or authority.get("configured_revision_publication_permitted") is not False
        or authority.get("offering_publication_permitted") is not False
        or authority.get("terminal_e2e_completion_permitted") is not False
        or authority.get("retry_cap") != 0
        or not isinstance(authority.get("resource_name"), str)
        or not authority.get("resource_name")
        or result.get("schema_version") != DIAGNOSTIC_RESULT_SCHEMA_VERSION
        or result.get("status") not in {"completed_diagnostic_only", "blocked_diagnostic_only"}
        or result.get("diagnostic_only") is not True
        or result.get("qualification_eligible") is not False
        or result.get("configured_revision_publication_permitted") is not False
        or result.get("offering_publication_permitted") is not False
        or result.get("terminal_e2e_completion_permitted") is not False
        or result.get("run_id") != run_id
        or result.get("source_commit") != authority.get("source_commit")
        or result.get("bundle_sha256") != bundle_sha256
        or result.get("authority_digest") != authority_digest
        or result.get("retry_cap") != 0
        or result.get("provider_mutations_performed") != 1
        or result.get("continuing_spend_from_this_run") is not False
        or result.get("raw_secret_values_recorded") is not False
        or not isinstance(consumption, Mapping)
        or consumption.get("status") != "consumed"
        or consumption.get("authorization_digest") != authority_digest
        or not isinstance(watchdog, Mapping)
        or watchdog.get("status") != "provider_terminal"
        or watchdog.get("provider_absence_confirmed") is not True
        or watchdog.get("raw_secret_values_recorded") is not False
        or adapter.get("schema_version") != VAST_PROVIDER_ADAPTER_RESULT_SCHEMA_VERSION
        or adapter.get("status") not in {"completed", "blocked"}
        or adapter.get("provider_bundle_kind") != "task_evaluation_scene_configuration"
        or adapter.get("provider_create_attempted") is not True
        or isinstance(adapter.get("estimated_cost_usd"), bool)
        or not isinstance(adapter.get("estimated_cost_usd"), (int, float))
        or not math.isfinite(float(adapter["estimated_cost_usd"]))
        or float(adapter["estimated_cost_usd"]) < 0
        or adapter.get("continuing_spend_from_this_run") is not False
        or adapter.get("final_validation_status") != "passed"
        or adapter.get("retained_owned") is not False
        or adapter.get("raw_api_key_stored") is not False
        or adapter.get("secret_values_in_artifact") is not False
        or not _adapter_secret_proof_valid(adapter)
        or teardown.get("schema_version") != VAST_TEARDOWN_SCHEMA_VERSION
        or teardown.get("status") != "completed"
        or teardown.get("continuing_spend_from_this_run") is not False
        or teardown.get("runner_gpu_teardown_completed") is not True
        or teardown.get("retention_authorized") is not False
        or teardown.get("raw_secret_values_recorded") is not False
        or adapter_ids != teardown_ids
        or adapter_ids != result_ids
        or result.get("provider_adapter_result_path") != str(adapter_path)
        or result.get("teardown_manifest_path") != str(teardown_path)
        or provider_zero.get("schema_version") != "adp_paid_provider_zero.v1"
        or provider_zero.get("provider") != "vast"
        or provider_zero.get("api_confirmed") is not True
        or provider_zero.get("provider_zero") is not True
        or provider_zero.get("global_live_resource_count") != 0
        or provider_zero.get("inventory") != []
        or not valid_vast_provider_zero_api_call(provider_zero.get("api_command"))
        or provider_zero.get("raw_secret_values_recorded") is not False
        or _timestamp(
            provider_zero.get("observed_at_utc"),
            code="scene_configuration_diagnostic_spend_provider_zero_time_invalid",
        )
        < _timestamp(
            teardown.get("generated_at"),
            code="scene_configuration_diagnostic_spend_teardown_time_invalid",
        )
    ):
        raise SceneConfigurationDiagnosticSpendError(
            "scene_configuration_diagnostic_spend_terminal_binding_invalid"
        )


def validate_scene_configuration_diagnostic_terminal_evidence(
    path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reopen the terminal receipt and every exact diagnostic source byte."""

    source, value = _read(
        path, code="scene_configuration_diagnostic_spend_receipt_invalid"
    )
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("status") != STATUS
        or value.get("goal_id") != "arm-decision-proof-v1"
        or value.get("lane") != "task_evaluation_scene_configuration_diagnostic"
        or value.get("diagnostic_only") is not True
        or value.get("qualification_eligible") is not False
        or value.get("provider") != "vast"
        or value.get("provider_zero_scope") != "global_vast_billable_inventory"
        or value.get("continuing_spend_from_this_run") is not False
        or value.get("raw_secret_values_recorded") is not False
        or value.get("receipt_digest")
        != canonical_digest(value, digest_field="receipt_digest")
    ):
        raise SceneConfigurationDiagnosticSpendError(
            "scene_configuration_diagnostic_spend_receipt_invalid"
        )
    records = value.get("source_receipts")
    if not isinstance(records, Mapping):
        raise SceneConfigurationDiagnosticSpendError(
            "scene_configuration_diagnostic_spend_receipt_invalid"
        )
    _authority_path, authority = _reopen_record(
        records.get("attempt_authority"), role="authority", digest_field="authority_digest"
    )
    result_path, result = _reopen_record(
        records.get("terminal_result"), role="terminal_result", digest_field="result_digest"
    )
    adapter_path, adapter = _reopen_record(
        records.get("provider_adapter_result"), role="adapter", digest_field=None
    )
    teardown_path, teardown = _reopen_record(
        records.get("teardown_manifest"), role="teardown", digest_field=None
    )
    _zero_path, provider_zero = _reopen_record(
        records.get("post_teardown_provider_zero"),
        role="provider_zero",
        digest_field="provider_zero_digest",
    )
    _validate_sources(
        authority=authority,
        result_path=result_path,
        result=result,
        adapter_path=adapter_path,
        adapter=adapter,
        teardown_path=teardown_path,
        teardown=teardown,
        provider_zero=provider_zero,
    )
    if (
        value.get("attempt_id") != authority.get("resource_name")
        or value.get("authority_digest") != authority.get("authority_digest")
        or value.get("run_id") != authority.get("run_id")
        or value.get("bundle_sha256") != authority.get("bundle_sha256")
        or value.get("source_commit") != authority.get("source_commit")
        or value.get("provider_instance_id") != adapter["vast_instance_ids"][0]
        or value.get("estimated_cost_usd") != adapter.get("estimated_cost_usd")
    ):
        raise SceneConfigurationDiagnosticSpendError(
            "scene_configuration_diagnostic_spend_receipt_binding_invalid"
        )
    return value, {
        "path": str(source),
        "size_bytes": source.stat().st_size,
        "sha256": _sha256(source),
        "receipt_digest": value["receipt_digest"],
    }


def materialize_scene_configuration_diagnostic_terminal_evidence(
    *,
    attempt_authority_path: str | Path,
    terminal_result_path: str | Path,
    provider_adapter_result_path: str | Path,
    teardown_manifest_path: str | Path,
    post_teardown_provider_zero_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Seal one direct diagnostic's terminal evidence without provider calls."""

    authority_path, authority = _read(
        attempt_authority_path,
        code="scene_configuration_diagnostic_spend_authority_invalid",
    )
    result_path, result = _read(
        terminal_result_path,
        code="scene_configuration_diagnostic_spend_terminal_result_invalid",
    )
    adapter_path, adapter = _read(
        provider_adapter_result_path,
        code="scene_configuration_diagnostic_spend_adapter_invalid",
    )
    teardown_path, teardown = _read(
        teardown_manifest_path,
        code="scene_configuration_diagnostic_spend_teardown_invalid",
    )
    zero_path, provider_zero = _read(
        post_teardown_provider_zero_path,
        code="scene_configuration_diagnostic_spend_provider_zero_invalid",
    )
    _validate_sources(
        authority=authority,
        result_path=result_path,
        result=result,
        adapter_path=adapter_path,
        adapter=adapter,
        teardown_path=teardown_path,
        teardown=teardown,
        provider_zero=provider_zero,
    )
    evidence: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "goal_id": "arm-decision-proof-v1",
        "lane": "task_evaluation_scene_configuration_diagnostic",
        "attempt_id": authority["resource_name"],
        "run_id": authority["run_id"],
        "source_commit": authority["source_commit"],
        "authority_digest": authority["authority_digest"],
        "bundle_sha256": authority["bundle_sha256"],
        "provider": "vast",
        "provider_instance_id": adapter["vast_instance_ids"][0],
        "estimated_cost_usd": adapter["estimated_cost_usd"],
        "provider_zero_scope": "global_vast_billable_inventory",
        "provider_zero_confirmed": True,
        "diagnostic_only": True,
        "qualification_eligible": False,
        "continuing_spend_from_this_run": False,
        "source_receipts": {
            "attempt_authority": _record(
                authority_path, authority, digest_field="authority_digest"
            ),
            "terminal_result": _record(
                result_path, result, digest_field="result_digest"
            ),
            "provider_adapter_result": _record(
                adapter_path, adapter, digest_field=None
            ),
            "teardown_manifest": _record(
                teardown_path, teardown, digest_field=None
            ),
            "post_teardown_provider_zero": _record(
                zero_path, provider_zero, digest_field="provider_zero_digest"
            ),
        },
        "provider_mutation_performed": False,
        "raw_secret_values_recorded": False,
        "receipt_digest": "",
    }
    evidence["receipt_digest"] = canonical_digest(
        evidence, digest_field="receipt_digest"
    )
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise SceneConfigurationDiagnosticSpendError(
            "scene_configuration_diagnostic_spend_output_exists"
        )
    payload = (json.dumps(evidence, indent=1, sort_keys=True) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o440)
        os.link(temporary, destination)
    except FileExistsError as exc:
        raise SceneConfigurationDiagnosticSpendError(
            "scene_configuration_diagnostic_spend_output_exists"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)
    validate_scene_configuration_diagnostic_terminal_evidence(destination)
    return evidence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-authority", required=True)
    parser.add_argument("--terminal-result", required=True)
    parser.add_argument("--provider-adapter-result", required=True)
    parser.add_argument("--teardown-manifest", required=True)
    parser.add_argument("--post-teardown-provider-zero", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    materialize_scene_configuration_diagnostic_terminal_evidence(
        attempt_authority_path=args.attempt_authority,
        terminal_result_path=args.terminal_result,
        provider_adapter_result_path=args.provider_adapter_result,
        teardown_manifest_path=args.teardown_manifest,
        post_teardown_provider_zero_path=args.post_teardown_provider_zero,
        output_path=args.output,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "SCHEMA_VERSION",
    "SceneConfigurationDiagnosticSpendError",
    "materialize_scene_configuration_diagnostic_terminal_evidence",
    "validate_scene_configuration_diagnostic_terminal_evidence",
]
