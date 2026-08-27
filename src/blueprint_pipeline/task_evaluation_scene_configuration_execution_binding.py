"""Cross-check provider execution identities against the shipped bundle."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def provider_execution_binding_blockers(
    execution: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    diagnostic_only: bool,
) -> list[str]:
    blockers: list[str] = []
    chain = execution.get(
        "diagnostic_stage_chain" if diagnostic_only else "stage_chain"
    )
    result_run_id = (
        chain.get("run_id")
        if diagnostic_only and isinstance(chain, Mapping)
        else execution.get("run_id")
    )
    if result_run_id != receipt.get("run_id") or (
        not diagnostic_only
        and (
            not isinstance(chain, Mapping)
            or chain.get("run_id") != receipt.get("run_id")
        )
    ):
        blockers.append("scene_configuration_provider_run_id_mismatch")
    provider_source_commit = execution.get(
        "diagnostic_source_commit" if diagnostic_only else "source_commit"
    )
    if provider_source_commit != receipt.get("source_commit"):
        blockers.append("scene_configuration_provider_source_commit_mismatch")
    if (
        not diagnostic_only
        and execution.get("construction_envelope_digest")
        != receipt.get("portable_construction_envelope_digest")
    ):
        blockers.append("scene_configuration_provider_envelope_mismatch")
    if diagnostic_only and execution.get(
        "source_checkpoint_digest"
    ) != receipt.get("source_diagnostic_checkpoint_digest"):
        blockers.append("scene_configuration_diagnostic_checkpoint_mismatch")
    if diagnostic_only and execution.get(
        "diagnostic_bootstrap_mode"
    ) != receipt.get("diagnostic_bootstrap_mode"):
        blockers.append("scene_configuration_diagnostic_bootstrap_mode_mismatch")
    if diagnostic_only and (
        execution.get("diagnostic_scientific_binding_digest")
        != receipt.get("diagnostic_scientific_binding_digest")
        or not isinstance(chain, Mapping)
        or chain.get("scientific_binding_digest")
        != receipt.get("diagnostic_scientific_binding_digest")
    ):
        blockers.append("scene_configuration_diagnostic_scientific_binding_mismatch")
    return blockers
