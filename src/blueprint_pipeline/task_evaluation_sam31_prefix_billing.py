"""Read existing official billing for completed SAM work; never query providers."""
import os
from pathlib import Path

from .task_evaluation_scene_configuration_submission_inputs import require

DEFAULT_AUDIT_ROOT = "/var/lib/blueprint/pipeline-control-plane/gpu_spend_guard/billing-audit"


class Sam31PrefixBillingPending(ValueError):
    """A verified completed tracking run must wait, not be executed again."""


def _charge(execution, source):
    # These are the original tracking-adoption billing predicates, shared by
    # discovery and final immutable readback rather than weakened for lookup.
    from .vast_official_billing_extractor import (
        _validate_source_receipt, _load_vast_responses, extract_vast_official_instance_charge,
    )
    source_path, billing, _ = _validate_source_receipt(source)
    labels = []
    for _, _, _, response in _load_vast_responses(source_receipt_path=source_path, source_receipt=billing):
        labels.extend(row.get("metadata", {}).get("label") for row in response["results"]
                      if row.get("source") == "instance-" + str(execution["instance_id"]))
    require(len(labels) == 1 and isinstance(labels[0], str)
            and labels[0].startswith("blueprint-sam31-source-tracks-")
            and labels[0].endswith(execution["request_digest"].removeprefix("sha256:")[:12]),
            "sam31_adoption_official_billing_instance_invalid")
    charge = extract_vast_official_instance_charge(provider_billing_source_receipt_path=source_path,
        instance_id=int(execution["instance_id"]), launch_label=labels[0])
    require(0 <= charge["official_charge_usd"] <= 1., "sam31_adoption_official_charge_invalid")
    return charge


def tracking_charge(execution, source=None, *, approved_roots=None):
    if source is not None:
        return _charge(execution, source)
    roots = tuple(Path(root).resolve() for root in (approved_roots or ()))
    audit = Path(os.getenv("BLUEPRINT_PROVIDER_BILLING_AUDIT_ROOT") or DEFAULT_AUDIT_ROOT)
    if (not audit.is_absolute() or any(p.is_symlink() for p in (audit, *audit.parents))
            or not any(audit.resolve().is_relative_to(root) for root in roots)):
        raise Sam31PrefixBillingPending("sam31_adoption_official_billing_pending:audit_root_unavailable")
    # Canonical billing reconciliation creates one immutable timestamp directory
    # per observation. Prefer its most recent valid exact-instance observation,
    # as the existing calibration and policy billing readers do.
    try:
        candidates = sorted(audit.glob("*/provider_billing_source_receipt.json"),
                            key=lambda path: path.stat().st_mtime_ns, reverse=True)
    except OSError as exc:
        raise Sam31PrefixBillingPending("sam31_adoption_official_billing_pending:audit_read_unavailable") from exc
    for candidate in candidates:
        try:
            return _charge(execution, candidate)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    raise Sam31PrefixBillingPending("sam31_adoption_official_billing_pending:exact_instance_charge_unavailable")
