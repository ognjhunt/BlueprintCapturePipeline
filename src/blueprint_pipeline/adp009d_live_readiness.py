"""Compile fail-closed live-launch readiness from one passing controls canary."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .adp009d_contact_envelope import (
    ContactEnvelopeError,
    validate_contact_envelope,
)
from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "task_evaluation_runtime_readiness.v1"
LEGACY_LIVE_PROFILE_BUILDER = "build_adp009d_840313_live_profile.py"
PROFILE_ID = "adp009d-840313-franka-live-v1"
SOURCE_BUNDLE_DIGEST = (
    "sha256:4cbf6781cd43cdf02353e0417aefd9ee4df1a65a99e7dbb2ef69a0a0170f22ba"
)
EVALUATION_RUN_SPEC_DIGEST = (
    "sha256:6e39daf5c5fc8a7e26d7cb34f53c6f9ac92756c1e86a5fc5ec70dd0e4e38b034"
)
RUNTIME_BUNDLE_DIGEST = (
    "sha256:9e09be2082dbc8990032ace5cc84773a32a6b07e8b393da4d1e4f0192ea16c00"
)
INSTANCE_DIGEST = (
    "sha256:243c0e62697da0298081a53c6530cee16cf94cde5a73df08f3773629b52c3001"
)
APPEARANCE_DIGEST = (
    "sha256:4b73dd13e6044b00b59da7737989d79d891ccac157b33411b30ef59542f3e6a2"
)


def _file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _parse_time(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _control_pair_blockers(pair: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    controls = pair.get("controls")
    rows = [dict(row) for row in controls if isinstance(row, Mapping)] if isinstance(controls, list) else []
    by_id = {str(row.get("control_id") or ""): row for row in rows}
    zero = by_id.get("zero_action_negative", {})
    positive = by_id.get("deterministic_scripted_positive", {})
    if (
        pair.get("schema_version") != "adp009d_control_pair.v1"
        or pair.get("pair_digest") != canonical_digest(pair, digest_field="pair_digest")
        or pair.get("instance_digest") != INSTANCE_DIGEST
        or pair.get("candidate_policy_queried") is not False
    ):
        blockers.append("live_readiness_control_pair_invalid")
    if (
        pair.get("cell_admitted_for_policy_execution") is not True
        or pair.get("policy_execution_blockers") != []
        or zero.get("control_passed") is not True
        or zero.get("observed_outcome") != "never_moved"
        or positive.get("control_passed") is not True
        or positive.get("observed_outcome") == "never_moved"
        or set(by_id) != {"zero_action_negative", "deterministic_scripted_positive"}
    ):
        blockers.append("live_readiness_controls_not_passed")
    try:
        contact_envelope = validate_contact_envelope(pair.get("contact_envelope"))
    except ContactEnvelopeError:
        blockers.append("live_readiness_control_contact_envelope_invalid")
    else:
        if any(
            row.get("contact_envelope") != contact_envelope
            for row in rows
        ):
            blockers.append("live_readiness_control_contact_envelope_invalid")
    return blockers


def build_live_readiness(
    *,
    source_commit: str,
    release_evidence: Mapping[str, Any],
    bundle_receipt: Mapping[str, Any],
    allocator_result: Mapping[str, Any],
    control_pair: Mapping[str, Any],
    control_pair_path: str | Path,
    artifact_manifest: Mapping[str, Any],
    teardown_manifest: Mapping[str, Any],
    provider_zero_guard: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a digest-bound readiness receipt; never mutates a provider."""

    blockers = _control_pair_blockers(control_pair)
    if (
        len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
        or release_evidence.get("schema_version")
        != "task_evaluation_pipeline_release_evidence.v1"
        or release_evidence.get("status") != "passed"
        or release_evidence.get("source_commit") != source_commit
        or release_evidence.get("source_ref") != "main"
        or release_evidence.get("tracked_state") != "clean"
        or release_evidence.get("release_digest")
        != canonical_digest(release_evidence, digest_field="release_digest")
    ):
        blockers.append("live_readiness_release_identity_invalid")
    appearance_rows = {
        str(row.get("role") or ""): dict(row)
        for row in bundle_receipt.get("asset_bindings", [])
        if isinstance(row, Mapping)
    }
    try:
        bundle_contact_envelope = validate_contact_envelope(
            bundle_receipt.get("contact_envelope")
        )
        pair_contact_envelope = validate_contact_envelope(
            control_pair.get("contact_envelope")
        )
    except ContactEnvelopeError:
        blockers.append("live_readiness_contact_envelope_invalid")
    else:
        if bundle_contact_envelope != pair_contact_envelope:
            blockers.append("live_readiness_contact_envelope_invalid")
    if (
        bundle_receipt.get("schema_version") != "adp009d_native_microcheck_bundle.v1"
        or bundle_receipt.get("status") != "ready"
        or bundle_receipt.get("implementation_commit") != source_commit
        or bundle_receipt.get("controls_requested") is not True
        or bundle_receipt.get("policy_candidate_id") is not None
        or bundle_receipt.get("scenario_instance_digest") != INSTANCE_DIGEST
        or bundle_receipt.get("retry_cap") != 0
        or appearance_rows.get("aura_appearance", {}).get("sha256") != APPEARANCE_DIGEST
        or appearance_rows.get("aura_appearance", {}).get("visual_only") is not True
        or appearance_rows.get("aura_appearance", {}).get("collision_authority") is not False
    ):
        blockers.append("live_readiness_control_bundle_invalid")
    if (
        allocator_result.get("status") != "completed"
        or allocator_result.get("retry_cap") != 0
        or allocator_result.get("continuing_spend_from_this_run") is not False
        or not allocator_result.get("artifact_manifest_path")
        or not allocator_result.get("teardown_manifest_path")
    ):
        blockers.append("live_readiness_allocator_result_invalid")
    manifest_files = artifact_manifest.get("files")
    file_rows = [dict(row) for row in manifest_files if isinstance(row, Mapping)] if isinstance(manifest_files, list) else []
    pair_path = Path(control_pair_path).expanduser().resolve()
    pair_digest = _file_digest(pair_path) if pair_path.is_file() and not pair_path.is_symlink() else None
    pair_retained = any(
        str(row.get("relative_path") or "").endswith("/adp009d_control_pair.v1.json")
        and row.get("sha256") == pair_digest
        for row in file_rows
    )
    if (
        artifact_manifest.get("schema_version") != "task_evaluation_artifact_manifest.v1"
        or artifact_manifest.get("status") != "completed"
        or artifact_manifest.get("blockers") != []
        or artifact_manifest.get("manifest_digest")
        != canonical_digest(artifact_manifest, digest_field="manifest_digest")
        or not pair_retained
    ):
        blockers.append("live_readiness_artifact_manifest_invalid")
    if (
        teardown_manifest.get("schema_version") != "vast_teardown_manifest.v1"
        or teardown_manifest.get("status") != "completed"
        or teardown_manifest.get("runner_gpu_teardown_completed") is not True
        or teardown_manifest.get("continuing_spend_from_this_run") is not False
    ):
        blockers.append("live_readiness_teardown_invalid")
    inventory = provider_zero_guard.get("inventory_results")
    rows = [dict(row) for row in inventory if isinstance(row, Mapping)] if isinstance(inventory, list) else []
    by_provider = {str(row.get("provider") or ""): row for row in rows}
    teardown_at = _parse_time(teardown_manifest.get("generated_at"))
    guard_at = _parse_time(provider_zero_guard.get("generated_at"))
    if (
        provider_zero_guard.get("schema_version") != "gpu_spend_guard.v1"
        or provider_zero_guard.get("live_instance_count") != 0
        or provider_zero_guard.get("total_burn_per_hour_usd") not in {0, 0.0}
        or any(
            by_provider.get(provider, {}).get("status") != "succeeded"
            or by_provider.get(provider, {}).get("row_count") != 0
            for provider in ("digitalocean", "runpod", "vast")
        )
        or teardown_at is None
        or guard_at is None
        or guard_at < teardown_at
    ):
        blockers.append("live_readiness_provider_zero_invalid")
    blockers = sorted(set(blockers))
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if not blockers else "blocked",
        "program_id": "arm-decision-proof-v1",
        "profile_id": PROFILE_ID,
        "source_commit": source_commit,
        "source_bundle_digest": SOURCE_BUNDLE_DIGEST,
        "evaluation_run_spec_digest": EVALUATION_RUN_SPEC_DIGEST,
        "runtime_bundle_digest": RUNTIME_BUNDLE_DIGEST,
        "live_execution_enabled": not blockers,
        "blockers": blockers,
        "observations": {
            "exact_runtime_adapter_on_protected_main": "live_readiness_release_identity_invalid" not in blockers,
            "scripted_positive_control_passed": not any(
                blocker.startswith("live_readiness_control") for blocker in blockers
            ),
            "allocator_artifact_manifest_emitted": "live_readiness_artifact_manifest_invalid" not in blockers,
            "teardown_completed": "live_readiness_teardown_invalid" not in blockers,
            "provider_zero_after_teardown": "live_readiness_provider_zero_invalid" not in blockers,
            "contact_envelope_bound": "live_readiness_contact_envelope_invalid"
            not in blockers
            and "live_readiness_control_contact_envelope_invalid" not in blockers,
        },
        "evidence_digests": {
            "release": release_evidence.get("release_digest"),
            "control_bundle": bundle_receipt.get("bundle_sha256"),
            "control_pair": control_pair.get("pair_digest"),
            "artifact_manifest": artifact_manifest.get("manifest_digest"),
            "teardown_file": canonical_digest(teardown_manifest),
            "provider_zero_file": canonical_digest(provider_zero_guard),
        },
        "provider_mutation_performed": False,
        "claim_ceiling": "development_only",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    return receipt


__all__ = ["PROFILE_ID", "build_live_readiness"]
