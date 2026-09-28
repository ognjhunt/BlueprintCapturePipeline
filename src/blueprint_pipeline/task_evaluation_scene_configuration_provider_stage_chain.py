"""Validate returned scene-configuration stage-chain receipts and their claim boundaries."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .decision_evidence_contracts import canonical_digest


def _completed_stage_chain_valid(
    chain: Any,
    *,
    provider_result: Mapping[str, Any],
    diagnostic_only: bool,
) -> bool:
    if not isinstance(chain, Mapping):
        return False
    rows = chain.get("stage_results")
    if not isinstance(rows, list) or len(rows) != 6:
        return False
    stage_ids: set[str] = set()
    row_digests: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            return False
        stage_id = str(row.get("stage_id") or "")
        digest = str(row.get("stage_result_digest") or "")
        if (
            not stage_id
            or stage_id in stage_ids
            or row.get("schema_version")
            != "task_evaluation_scene_configuration_stage_result.v1"
            or row.get("status") != "completed"
            or row.get("canonical_allocator") is not None
            or row.get("provider_mutations_performed") != 0
            or row.get("paid_execution_requested") is not False
            or row.get("executed_inside_parent_configuration_run") is not True
            or row.get("raw_secret_values_recorded") is not False
            or not isinstance(row.get("output_artifacts"), list)
            or digest != canonical_digest(row, digest_field="stage_result_digest")
            or (
                diagnostic_only
                and (
                    row.get("diagnostic_only") is not True
                    or row.get("qualification_eligible") is not False
                    or row.get("executed_inside_one_parent_provider_run") is not False
                    or row.get("configured_revision_publication_permitted") is not False
                    or row.get("offering_publication_permitted") is not False
                    or row.get("terminal_e2e_completion_permitted") is not False
                )
            )
            or (
                not diagnostic_only
                and (
                    row.get("diagnostic_only") is True
                    or row.get("qualification_eligible") is False
                    or row.get("executed_inside_one_parent_provider_run") is False
                    or row.get("configured_revision_publication_permitted") is False
                    or row.get("offering_publication_permitted") is False
                )
            )
        ):
            return False
        stage_ids.add(stage_id)
        row_digests.append(digest)
    expected_schema = (
        "task_evaluation_scene_configuration_diagnostic_stage_chain.v1"
        if diagnostic_only
        else "task_evaluation_scene_configuration_provider_stage_chain.v1"
    )
    expected_status = (
        "completed_diagnostic_only_not_qualification_eligible"
        if diagnostic_only
        else "completed"
    )
    return bool(
        chain.get("schema_version") == expected_schema
        and chain.get("status") == expected_status
        and str(chain.get("run_id") or "")
        and (
            diagnostic_only
            or chain.get("run_id") == provider_result.get("run_id")
        )
        and chain.get("stage_count") == 6
        and chain.get("stage_result_digests") == row_digests
        and chain.get("executed_inside_one_parent_provider_run")
        is (not diagnostic_only)
        and chain.get("nested_provider_mutations_performed") == 0
        and chain.get("nested_paid_execution_requested") is False
        and chain.get("evaluation_episode_executed") is False
        and chain.get("retry_cap") == 0
        and chain.get("result_digest")
        == canonical_digest(chain, digest_field="result_digest")
    )

