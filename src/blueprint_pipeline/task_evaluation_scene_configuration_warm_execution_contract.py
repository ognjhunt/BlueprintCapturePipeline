"""Cross-bind warm provider output to its exact session and iteration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .task_evaluation_scene_configuration_diagnostic_mode import (
    CHECKPOINT_RESUME_DIAGNOSTIC_BOOTSTRAP_MODE,
)


def warm_execution_binding_blockers(
    *,
    execution: Mapping[str, Any],
    session: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> list[str]:
    chain = execution.get("diagnostic_stage_chain")
    blockers: list[str] = []
    pairs = (
        ("diagnostic_source_commit", authority, "source_commit", "overlay_commit"),
        ("base_bundle_source_commit", session, "source_commit", "base_commit"),
        (
            "diagnostic_source_overlay_manifest_digest",
            authority,
            "source_overlay_manifest_digest",
            "overlay_digest",
        ),
        ("source_checkpoint_digest", authority, "source_checkpoint_digest", "checkpoint"),
        ("diagnostic_toolchain_digest", session, "toolchain_digest", "toolchain"),
        (
            "diagnostic_construction_envelope_digest",
            session,
            "construction_envelope_digest",
            "construction_envelope",
        ),
        ("diagnostic_run_id", session, "run_id", "run_id"),
        ("warm_session_digest", session, "session_digest", "session"),
        (
            "warm_bootstrap_allocation_binding_digest",
            session,
            "bootstrap_allocation_binding_digest",
            "bootstrap_binding",
        ),
    )
    for observed, expected_record, expected, suffix in pairs:
        if execution.get(observed) != expected_record.get(expected):
            blockers.append(f"scene_configuration_warm_execution_{suffix}_mismatch")
    if isinstance(chain, Mapping) and chain.get("run_id") != session.get("run_id"):
        blockers.append("scene_configuration_warm_execution_chain_run_id_mismatch")
    if (
        execution.get("diagnostic_bootstrap_mode")
        != CHECKPOINT_RESUME_DIAGNOSTIC_BOOTSTRAP_MODE
        or not isinstance(chain, Mapping)
        or chain.get("diagnostic_bootstrap_mode")
        != CHECKPOINT_RESUME_DIAGNOSTIC_BOOTSTRAP_MODE
    ):
        blockers.append("scene_configuration_warm_execution_bootstrap_mode_mismatch")
    if (
        execution.get("diagnostic_scientific_binding_digest")
        != session.get("scientific_binding_digest")
        or not isinstance(chain, Mapping)
        or chain.get("scientific_binding_digest")
        != session.get("scientific_binding_digest")
    ):
        blockers.append("scene_configuration_warm_execution_scientific_binding_mismatch")
    if str(execution.get("warm_provider_instance_id") or "") != str(
        session.get("provider_instance_id") or ""
    ):
        blockers.append("scene_configuration_warm_execution_instance_mismatch")
    return blockers


def warm_bootstrap_execution_binding_blockers(
    *,
    execution: Mapping[str, Any],
    bundle_receipt: Mapping[str, Any],
    session_authority: Mapping[str, Any],
    advanced_checkpoint: Mapping[str, Any] | None,
) -> list[str]:
    blockers: list[str] = []
    pairs = (
        ("diagnostic_source_commit", bundle_receipt, "source_commit", "source_commit"),
        ("diagnostic_run_id", bundle_receipt, "run_id", "run_id"),
        (
            "diagnostic_toolchain_digest",
            bundle_receipt,
            "toolchain_digest",
            "toolchain",
        ),
        (
            "diagnostic_construction_envelope_digest",
            bundle_receipt,
            "portable_construction_envelope_digest",
            "construction_envelope",
        ),
        (
            "source_checkpoint_digest",
            session_authority,
            "source_checkpoint_digest",
            "source_checkpoint",
        ),
        (
            "diagnostic_bootstrap_mode",
            session_authority,
            "diagnostic_bootstrap_mode",
            "bootstrap_mode",
        ),
    )
    for observed, expected_record, expected, suffix in pairs:
        if execution.get(observed) != expected_record.get(expected):
            blockers.append(
                f"scene_configuration_warm_bootstrap_execution_{suffix}_mismatch"
            )
    chain = execution.get("diagnostic_stage_chain")
    if (
        execution.get("diagnostic_bootstrap_mode")
        != session_authority.get("diagnostic_bootstrap_mode")
        or execution.get("diagnostic_scientific_binding_digest")
        != session_authority.get("scientific_binding_digest")
        or (
            isinstance(chain, Mapping)
            and (
                chain.get("diagnostic_bootstrap_mode")
                != session_authority.get("diagnostic_bootstrap_mode")
                or chain.get("scientific_binding_digest")
                != session_authority.get("scientific_binding_digest")
            )
        )
    ):
        blockers.append(
            "scene_configuration_warm_bootstrap_execution_scientific_identity_mismatch"
        )
    if not isinstance(advanced_checkpoint, Mapping):
        blockers.append(
            "scene_configuration_warm_bootstrap_advanced_checkpoint_missing"
        )
    else:
        prefix_count = advanced_checkpoint.get("completed_stage_prefix_count")
        advanced_rows = advanced_checkpoint.get("completed_stage_results")
        expected_stage_ids = session_authority.get(
            "diagnostic_stage_sequence_ids"
        )
        if (
            advanced_checkpoint.get("scientific_bindings") or {}
        ).get("binding_digest") != session_authority.get(
            "scientific_binding_digest"
        ):
            blockers.append(
                "scene_configuration_warm_bootstrap_scientific_binding_mismatch"
            )
        if (
            not isinstance(prefix_count, int)
            or isinstance(prefix_count, bool)
            or prefix_count < 3
            or not isinstance(advanced_rows, list)
            or len(advanced_rows) != prefix_count
            or not isinstance(expected_stage_ids, list)
            or len(expected_stage_ids) != 6
            or [str(row.get("stage_id") or "") for row in advanced_rows]
            != [
                str(stage_id) for stage_id in expected_stage_ids[:prefix_count]
            ]
        ):
            blockers.append(
                "scene_configuration_warm_bootstrap_stage_prefix_incomplete"
            )
    return blockers


__all__ = [
    "warm_bootstrap_execution_binding_blockers",
    "warm_execution_binding_blockers",
]
