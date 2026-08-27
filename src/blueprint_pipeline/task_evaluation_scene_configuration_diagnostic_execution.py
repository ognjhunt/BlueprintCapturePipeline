"""Close one separately-authorized diagnostic scene retry without publishing.

The caller owns the paid allocator, watchdog, teardown, billing, and provider
zero.  This module validates those receipts and returns a deliberately
non-terminal diagnostic record.  It has no publisher argument by design.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_scene_configuration_diagnostic_checkpoint import (
    validate_scene_configuration_diagnostic_checkpoint,
)
from .task_evaluation_scene_configuration_diagnostic_runtime import (
    RESULT_SCHEMA_VERSION as DIAGNOSTIC_CHAIN_SCHEMA_VERSION,
    STATUS as DIAGNOSTIC_CHAIN_STATUS,
)
from .task_evaluation_scene_configuration_orchestrator import CANONICAL_ALLOCATOR


PARENT_LAUNCH_SCHEMA_VERSION = (
    "task_evaluation_scene_configuration_diagnostic_parent_launch.v1"
)
RESULT_SCHEMA_VERSION = (
    "task_evaluation_scene_configuration_diagnostic_execution.v1"
)
STATUS = "closed_diagnostic_only_not_qualification_eligible"
DiagnosticParentLaunchExecutor = Callable[..., Mapping[str, Any]]
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")


class TaskEvaluationSceneConfigurationDiagnosticExecutionError(RuntimeError):
    """A diagnostic allocation did not close under its separate authority."""


def _positive_finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
    )


def _validate_launch(
    value: Mapping[str, Any], *, checkpoint_digest: str
) -> dict[str, Any]:
    launch = dict(value)
    chain = launch.get("diagnostic_stage_chain")
    if (
        launch.get("schema_version") != PARENT_LAUNCH_SCHEMA_VERSION
        or launch.get("status") != STATUS
        or launch.get("explicit_diagnostic_resume_requested") is not True
        or launch.get("normal_production_lane_used") is not False
        or launch.get("canonical_allocator") != CANONICAL_ALLOCATOR
        or launch.get("provider_mutations_performed") != 1
        or launch.get("paid_execution_requested") is not True
        or launch.get("retry_cap") != 0
        or launch.get("watchdog_armed_before_allocation") is not True
        or launch.get("teardown_completed") is not True
        or launch.get("provider_zero_confirmed") is not True
        or launch.get("source_checkpoint_digest") != checkpoint_digest
        or _COMMIT.fullmatch(str(launch.get("diagnostic_source_commit") or "")) is None
        or _DIGEST.fullmatch(
            str(launch.get("diagnostic_toolchain_digest") or "")
        )
        is None
        or launch.get("diagnostic_only") is not True
        or launch.get("qualification_eligible") is not False
        or launch.get("executed_inside_one_parent_provider_run") is not False
        or launch.get("configured_revision_publication_permitted") is not False
        or launch.get("offering_publication_permitted") is not False
        or launch.get("terminal_e2e_completion_permitted") is not False
        or launch.get("raw_secret_values_recorded") is not False
        or not _positive_finite(launch.get("remaining_spend_hard_cap_usd"))
        or not _positive_finite(launch.get("remaining_runtime_hard_ttl_seconds"))
        or not isinstance(chain, Mapping)
        or chain.get("schema_version") != DIAGNOSTIC_CHAIN_SCHEMA_VERSION
        or chain.get("status") != DIAGNOSTIC_CHAIN_STATUS
        or chain.get("source_checkpoint_digest") != checkpoint_digest
        or chain.get("diagnostic_only") is not True
        or chain.get("qualification_eligible") is not False
        or chain.get("executed_inside_one_parent_provider_run") is not False
        or chain.get("configured_revision_publication_permitted") is not False
        or chain.get("offering_publication_permitted") is not False
        or chain.get("terminal_e2e_completion_permitted") is not False
        or chain.get("result_digest")
        != canonical_digest(chain, digest_field="result_digest")
        or launch.get("launch_digest")
        != canonical_digest(launch, digest_field="launch_digest")
    ):
        raise TaskEvaluationSceneConfigurationDiagnosticExecutionError(
            "scene_configuration_diagnostic_parent_launch_invalid"
        )
    for field in (
        "remaining_spend_authority_digest",
        "billing_reconciliation_digest",
        "watchdog_receipt_digest",
        "teardown_digest",
        "provider_zero_digest",
        "launch_receipt_digest",
    ):
        if _DIGEST.fullmatch(str(launch.get(field) or "")) is None:
            raise TaskEvaluationSceneConfigurationDiagnosticExecutionError(
                f"scene_configuration_diagnostic_governance_invalid:{field}"
            )
    return launch


def execute_scene_configuration_diagnostic_retry(
    *,
    checkpoint_root: str | Path,
    output_root: str | Path,
    diagnostic_parent_launch_executor: DiagnosticParentLaunchExecutor,
) -> dict[str, Any]:
    """Run and close one explicit retry while withholding production claims."""

    checkpoint = validate_scene_configuration_diagnostic_checkpoint(
        checkpoint_root=checkpoint_root
    )
    root = Path(output_root).expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise TaskEvaluationSceneConfigurationDiagnosticExecutionError(
            "scene_configuration_diagnostic_execution_output_root_invalid"
        )
    launch = _validate_launch(
        diagnostic_parent_launch_executor(
            checkpoint=checkpoint,
            checkpoint_root=Path(checkpoint_root).expanduser().resolve(),
            output_root=root,
        ),
        checkpoint_digest=str(checkpoint["checkpoint_digest"]),
    )
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": STATUS,
        "source_checkpoint_digest": checkpoint["checkpoint_digest"],
        "checkpoint_source_commit_provenance": checkpoint.get(
            "source_commit_provenance"
        ),
        "checkpoint_source_toolchain_digest_provenance": checkpoint.get(
            "source_toolchain_digest_provenance"
        ),
        "diagnostic_source_commit": launch["diagnostic_source_commit"],
        "diagnostic_toolchain_digest": launch["diagnostic_toolchain_digest"],
        "diagnostic_stage_chain_digest": launch["diagnostic_stage_chain"][
            "result_digest"
        ],
        "diagnostic_only": True,
        "qualification_eligible": False,
        "executed_inside_one_parent_provider_run": False,
        "configured_revision_publication_permitted": False,
        "offering_publication_permitted": False,
        "terminal_e2e_completion_permitted": False,
        "terminal_e2e_status": None,
        "configured_scene_revision": None,
        "configured_scene_offering": None,
        "canonical_allocator": CANONICAL_ALLOCATOR,
        "provider_mutations_performed": 1,
        "paid_execution_requested": True,
        "retry_cap": 0,
        "remaining_spend_authority_digest": launch[
            "remaining_spend_authority_digest"
        ],
        "remaining_spend_hard_cap_usd": launch["remaining_spend_hard_cap_usd"],
        "remaining_runtime_hard_ttl_seconds": launch[
            "remaining_runtime_hard_ttl_seconds"
        ],
        "watchdog_armed_before_allocation": True,
        "watchdog_receipt_digest": launch["watchdog_receipt_digest"],
        "billing_reconciliation_digest": launch["billing_reconciliation_digest"],
        "teardown_digest": launch["teardown_digest"],
        "provider_zero_digest": launch["provider_zero_digest"],
        "provider_zero_confirmed": True,
        "raw_secret_values_recorded": False,
        "result_digest": "",
    }
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    return result


__all__ = [
    "PARENT_LAUNCH_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "STATUS",
    "TaskEvaluationSceneConfigurationDiagnosticExecutionError",
    "execute_scene_configuration_diagnostic_retry",
]
