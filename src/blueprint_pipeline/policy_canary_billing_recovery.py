"""Retained official-charge recovery without reopening a paid execution."""
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from .vast_official_charge_period import VastOfficialBillingExtractionError


def _candidate_sources(audit: Path, adapter_result_path: Path,
                       adapter: Mapping[str, Any]) -> list[Path]:
    """Skip audits that predate this instance's maximum possible lifetime.

    The canonical audit directories carry their observation time in the name.
    That lets a closeout avoid opening hundreds of thousands of old responses
    when the new instance's official charge has not posted yet. Legacy adapters
    without a TTL retain the original unrestricted lookup.
    """
    if not audit.is_dir() or audit.is_symlink():
        return []
    cutoff = None
    # The adapter writes generated_at before provider creation. Its result does
    # not carry the allocator TTL, so use that creation boundary with a clock
    # skew margin. Missing or invalid timestamps preserve unrestricted lookup.
    generated_at = adapter.get("generated_at")
    if isinstance(generated_at, str):
        try:
            created = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
            if created.tzinfo is not None:
                cutoff = created.timestamp() - 3600.0
        except ValueError:
            pass
    candidates = []
    sources = (audit.glob("*/provider_billing_source_receipt.json")
               if cutoff is not None else audit.rglob("provider_billing_source_receipt.json"))
    for source in sources:
        if cutoff is not None:
            try:
                observed = datetime.strptime(
                    source.parent.name, "%Y%m%dT%H%M%S.%fZ"
                ).replace(tzinfo=timezone.utc).timestamp()
            except ValueError:
                observed = source.stat().st_mtime
            if observed < cutoff:
                continue
        candidates.append(source)
    return sorted(candidates, key=lambda path: path.stat().st_mtime_ns,
                  reverse=True)

def reconcile_posted_billing(
    *,
    billing_audit_root: str | Path,
    adapter_result_path: Path,
    adapter: Mapping[str, Any],
    launch_label: str,
    output_path: Path,
    instance_ids_from_adapter,
    validate_reconciliation,
    materialize_reconciliation,
    record_file,
    write_json,
    dispatch_error,
) -> bool:
    instance_ids = instance_ids_from_adapter(adapter)
    if len(instance_ids) != 1:
        return False
    instance_id = int(instance_ids[0])
    recovery_path = output_path.parent / "official_billing_recovery.json"
    if output_path.is_file():
        value = validate_reconciliation(output_path)
        matches = [entry for entry in value.get("entries", [])
                   if entry.get("provider_instance_id") == instance_id and entry.get("launch_label") == launch_label]
        if len(matches) != 1:
            raise dispatch_error("policy_canary_existing_billing_identity_mismatch")
        terminal = matches[0].get("terminal_execution_evidence", {}).get("terminal_result", {})
        expected = record_file(adapter_result_path)
        if any(terminal.get(key) != expected[key] for key in ("path", "sha256", "size_bytes")):
            raise dispatch_error("policy_canary_existing_billing_identity_mismatch")
        return True
    audit = Path(billing_audit_root).expanduser().resolve()
    candidates = _candidate_sources(audit, adapter_result_path, adapter)
    failures = []
    for source in candidates:
        try:
            materialize_reconciliation(
                provider_billing_source_receipt_path=source,
                expected_instances=[(instance_id, launch_label, adapter_result_path)],
                output_path=output_path,
            )
        except (OSError, VastOfficialBillingExtractionError) as exc:
            failures.append(str(exc) if isinstance(exc, VastOfficialBillingExtractionError) else type(exc).__name__)
            continue
        if recovery_path.exists():
            write_json(recovery_path, {"status": "resolved", "provider_instance_id": instance_id,
                "launch_label": launch_label, "reconciliation": record_file(output_path)})
        return True
    write_json(recovery_path, {
        "schema_version": "policy_canary_official_billing_recovery.v1", "status": "pending",
        "provider_instance_id": instance_id, "launch_label": launch_label,
        "official_charge_usd": None, "blockers": sorted(set(failures)) or ["official_billing_source_missing"],
        "recovery_action": ("refresh_official_billing_with_declared_period"
            if any("period" in failure for failure in failures) else "refresh_exact_instance_official_billing"),
        "provider_mutation_performed": False,
    })
    return False
