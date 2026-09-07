"""Retained official-charge recovery without reopening a paid execution."""
from pathlib import Path
from typing import Any, Mapping
from .vast_official_charge_period import VastOfficialBillingExtractionError

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
    candidates = sorted(
        audit.rglob("provider_billing_source_receipt.json"),
        key=lambda path: path.stat().st_mtime_ns, reverse=True,
    ) if audit.is_dir() and not audit.is_symlink() else []
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
