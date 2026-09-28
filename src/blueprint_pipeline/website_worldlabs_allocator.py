"""ADP-009B/day-14: canonical allocator entry for prepared website views.

Submit once and return the retained operation for ordinary provider polling.
This creates a development visual reconstruction, not a qualified simulation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .provider_preview import WorldLabsPreviewProvider
from .website_worldlabs import validate_website_prepared_views, validate_website_reconstruction_admission
from .website_capture_entry import is_website_entry_source


def run_website_worldlabs(args: Any, *, load_json: Callable[..., Any],
                         source_checkout_blockers: Callable[..., Any], admission_issuer: Callable[..., Any]) -> dict[str, Any]:
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    blockers = []
    descriptor, capture_root, grant = {}, None, None
    try:
        if not args.descriptor or not args.capture_root:
            raise ValueError("website_reconstruction_inputs_missing")
        descriptor = load_json(Path(args.descriptor).expanduser().resolve())
        capture_root = Path(args.capture_root).expanduser().resolve()
        if not is_website_entry_source((descriptor.get("metadata") or {}).get("capture_entry_source")):
            raise ValueError("website_reconstruction_entry_source_invalid")
        _, binding = validate_website_prepared_views(descriptor=descriptor, capture_root=capture_root)
        admission = descriptor["metadata"].get("website_reconstruction_admission") or {}
        validate_website_reconstruction_admission(admission, canonical_digest(binding))
        source_blockers, _ = source_checkout_blockers(str(admission.get("source_commit") or ""),
            allow_pushed_branch_diagnostic=args.experimental_branch_diagnostic)
        blockers.extend(source_blockers)
        grant = admission_issuer(admission, resource_class="provider_reconstruction_api",
                                 expected_schema_version="paid_lane_admission.v1")
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        blockers.append(str(exc) if isinstance(exc, ValueError) else type(exc).__name__)
    result = {"status": "blocked" if blockers else "dry_run_ready", "blockers": blockers,
              "execute_requested": bool(args.execute), "provider_mutation_attempted": False,
              "claim_ceiling": "development_only"}
    write_json(output / "website_worldlabs_preflight.json", result)
    if blockers or not args.execute:
        return result
    try:
        submitted = WorldLabsPreviewProvider().submit(descriptor=descriptor, capture_root=capture_root,
            provider_adapter_input={"paid_resource_admission_grant": grant})
        result = {**result, "status": "submitted", "provider_mutation_attempted": True, "submission": submitted}
    except Exception as exc:
        # The adapter retains uncertain billable intent; retry cannot purchase twice.
        result = {**result, "status": "failed", "provider_mutation_attempted": True,
                  "blockers": [type(exc).__name__]}
    write_json(output / "website_worldlabs_submission.json", result)
    return result
