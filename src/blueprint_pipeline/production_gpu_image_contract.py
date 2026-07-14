"""Classify an immutable GPU worker image for customer-serving startup modes."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import write_json

SCHEMA_VERSION = "production_gpu_image_serving_contract.v1"
DEFAULT_SCALE_TO_ZERO_TOTAL_COMPRESSED_LIMIT_BYTES = 8 * 1024**3
DEFAULT_SCALE_TO_ZERO_LAYER_COMPRESSED_LIMIT_BYTES = 2 * 1024**3
_DIGEST_REF = re.compile(r"\A[^\s@]+@sha256:[0-9a-f]{64}\Z")


def build_image_serving_contract(
    diagnostic: Mapping[str, Any],
    *,
    expected_image_ref: str,
    models_externalized_with_immutable_manifest: bool,
    total_compressed_limit_bytes: int = DEFAULT_SCALE_TO_ZERO_TOTAL_COMPRESSED_LIMIT_BYTES,
    layer_compressed_limit_bytes: int = DEFAULT_SCALE_TO_ZERO_LAYER_COMPRESSED_LIMIT_BYTES,
) -> dict[str, Any]:
    """Fail closed between active-worker eligibility and scale-to-zero eligibility.

    Size limits are release budgets, not startup-time proof.  A release below
    the limits still needs a measured live cold-start campaign.
    """

    blockers: list[str] = []
    image_ref = str(expected_image_ref or "").strip()
    if not _DIGEST_REF.fullmatch(image_ref):
        blockers.append("expected_worker_image_must_be_digest_pinned")
    observed_ref = str(
        diagnostic.get("resolved_digest_ref") or diagnostic.get("image_ref") or ""
    ).strip()
    exact_release = bool(image_ref and observed_ref == image_ref)
    if not exact_release:
        blockers.append("registry_diagnostic_exact_release_mismatch")
    total = diagnostic.get("total_compressed_size_bytes")
    largest = diagnostic.get("largest_layer_size_bytes")
    measured_sizes = type(total) is int and total > 0 and type(largest) is int and largest > 0
    if not measured_sizes:
        blockers.append("compressed_image_size_evidence_missing")
    measured_total = int(total) if type(total) is int and total > 0 else 0
    measured_largest = int(largest) if type(largest) is int and largest > 0 else 0
    total_within = bool(
        measured_sizes and measured_total <= int(total_compressed_limit_bytes)
    )
    layer_within = bool(
        measured_sizes and measured_largest <= int(layer_compressed_limit_bytes)
    )
    scale_to_zero_checks = {
        "exact_digest_registry_evidence": exact_release,
        "compressed_size_measured": measured_sizes,
        "total_compressed_budget": total_within,
        "largest_layer_budget": layer_within,
        "models_externalized_with_immutable_manifest": (
            models_externalized_with_immutable_manifest is True
        ),
    }
    scale_to_zero_eligible = all(scale_to_zero_checks.values())
    active_worker_checks = {
        "exact_digest_registry_evidence": exact_release,
        "compressed_size_measured": measured_sizes,
        "preload_and_cache_required": True,
        "customer_request_may_cold_start_worker": False,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "scale_to_zero_candidate" if scale_to_zero_eligible else (
            "active_worker_only" if exact_release and measured_sizes else "blocked"
        ),
        "worker_image_ref": image_ref or None,
        "observed_image_ref": observed_ref or None,
        "measured": {
            "total_compressed_size_bytes": total if measured_sizes else None,
            "largest_layer_size_bytes": largest if measured_sizes else None,
        },
        "release_budgets": {
            "scale_to_zero_total_compressed_limit_bytes": int(total_compressed_limit_bytes),
            "scale_to_zero_largest_layer_limit_bytes": int(layer_compressed_limit_bytes),
        },
        "active_worker_checks": active_worker_checks,
        "scale_to_zero_checks": scale_to_zero_checks,
        "scale_to_zero_eligible": scale_to_zero_eligible,
        "customer_serving_mode": (
            "measured_scale_to_zero_candidate"
            if scale_to_zero_eligible
            else "preloaded_active_worker_warm_pool"
        ),
        "blockers": blockers,
        "optimization_actions": [
            "separate_stable_isaac_foundation_from_thin_blueprint_release",
            "externalize_checkpoints_to_digest_verified_provider_cache_or_volume",
            "remove_build_toolchains_and_package_caches_from_runtime_layers",
            "split_dependency_layers_below_release_layer_budget",
            "rebuild_and_measure_cold_start_before_enabling_scale_to_zero",
        ] if not scale_to_zero_eligible else [],
        "claim_boundary": {
            "size_budget_is_not_live_startup_proof": True,
            "active_worker_only_requires_preloaded_cache_before_customer_binding": True,
            "cold_campaign_evidence_is_release_engineering_only": True,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", required=True)
    parser.add_argument("--expected-image-ref", required=True)
    parser.add_argument("--models-externalized", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    diagnostic = json.loads(Path(args.diagnostic).read_text(encoding="utf-8"))
    result = build_image_serving_contract(
        diagnostic,
        expected_image_ref=args.expected_image_ref,
        models_externalized_with_immutable_manifest=args.models_externalized,
    )
    write_json(Path(args.out), result)
    print(json.dumps({"status": result["status"], "scale_to_zero_eligible": result["scale_to_zero_eligible"]}))
    return 0 if result["status"] != "blocked" else 1


if __name__ == "__main__":
    raise SystemExit(main())
