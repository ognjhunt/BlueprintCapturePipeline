"""Join one canonical paid configuration launch with immutable publication."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_scene_configuration_orchestrator import (
    CANONICAL_ALLOCATOR,
    PROVIDER_EXECUTION_SCHEMA_VERSION,
)
from .task_evaluation_scene_configuration_publication import (
    Publisher,
    publish_configured_scene_revision,
)
from .task_evaluation_scene_configuration_provider_runtime import (
    RESULT_SCHEMA_VERSION as STAGE_CHAIN_SCHEMA_VERSION,
)


PARENT_LAUNCH_SCHEMA_VERSION = (
    "task_evaluation_scene_configuration_parent_launch.v1"
)
ParentLaunchExecutor = Callable[..., Mapping[str, Any]]


class TaskEvaluationSceneConfigurationExecutionError(RuntimeError):
    """The Website-started configuration launch could not close exactly."""


def _validate_parent_launch(
    value: Mapping[str, Any], *, envelope: Mapping[str, Any]
) -> dict[str, Any]:
    launch = dict(value)
    stage_chain = launch.get("stage_chain")
    if (
        launch.get("schema_version") != PARENT_LAUNCH_SCHEMA_VERSION
        or launch.get("status") != "completed"
        or launch.get("run_id") != envelope["run_id"]
        or launch.get("submitted_via_authenticated_webapp") is not True
        or launch.get("dispatched_by_task_evaluation_dispatcher") is not True
        or launch.get("orchestration_worker_executed") is not True
        or launch.get("canonical_allocator") != CANONICAL_ALLOCATOR
        or launch.get("provider_mutations_performed") != 1
        or launch.get("paid_execution_requested") is not True
        or launch.get("retry_cap") != 0
        or launch.get("evaluation_episode_executed") is not False
        or launch.get("raw_secret_values_recorded") is not False
        or launch.get("launch_digest")
        != canonical_digest(launch, digest_field="launch_digest")
        or not isinstance(stage_chain, Mapping)
        or stage_chain.get("schema_version") != STAGE_CHAIN_SCHEMA_VERSION
        or stage_chain.get("status") != "completed"
        or stage_chain.get("run_id") != envelope["run_id"]
        or stage_chain.get("stage_count") != 6
        or stage_chain.get("executed_inside_one_parent_provider_run") is not True
        or stage_chain.get("nested_provider_mutations_performed") != 0
        or stage_chain.get("evaluation_episode_executed") is not False
        or stage_chain.get("result_digest")
        != canonical_digest(stage_chain, digest_field="result_digest")
    ):
        raise TaskEvaluationSceneConfigurationExecutionError(
            "scene_configuration_parent_launch_invalid"
        )
    for field in (
        "paid_authority_digest",
        "billing_reconciliation_digest",
        "teardown_digest",
        "provider_zero_digest",
        "launch_receipt_digest",
    ):
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(launch.get(field) or "")):
            raise TaskEvaluationSceneConfigurationExecutionError(
                f"scene_configuration_parent_launch_governance_invalid:{field}"
            )
    return launch


def execute_and_publish_scene_configuration(
    *,
    envelope: Mapping[str, Any],
    configurations: Mapping[str, tuple[Mapping[str, Any], Path]],
    output_root: Path,
    parent_launch_executor: ParentLaunchExecutor,
    publisher: Publisher,
) -> dict[str, Any]:
    """Run one canonical parent launch, then publish its robot-neutral result."""

    provider_root = output_root / "provider-run"
    provider_root.mkdir(mode=0o750)
    launch = _validate_parent_launch(
        parent_launch_executor(
            envelope=envelope,
            configurations=configurations,
            output_root=provider_root,
        ),
        envelope=envelope,
    )
    publication_root = output_root / "publication"
    publication_root.mkdir(mode=0o750)
    publication = publish_configured_scene_revision(
        envelope=envelope,
        stage_results=launch["stage_chain"]["stage_results"],
        output_root=publication_root,
        publisher=publisher,
    )
    result: dict[str, Any] = {
        "schema_version": PROVIDER_EXECUTION_SCHEMA_VERSION,
        "status": "completed",
        "canonical_allocator": CANONICAL_ALLOCATOR,
        "provider_mutations_performed": 1,
        "paid_execution_requested": True,
        "retry_cap": 0,
        "evaluation_episode_executed": False,
        "raw_secret_values_recorded": False,
        "paid_authority_digest": launch["paid_authority_digest"],
        "billing_reconciliation_digest": launch[
            "billing_reconciliation_digest"
        ],
        "teardown_digest": launch["teardown_digest"],
        "provider_zero_digest": launch["provider_zero_digest"],
        "launch_receipt_digest": launch["launch_receipt_digest"],
        "stage_results": launch["stage_chain"]["stage_results"],
        "configured_scene_revision": publication[
            "configured_scene_revision"
        ],
        "configured_scene_revision_reference": publication[
            "configured_scene_revision_reference"
        ],
        "configured_scene_revision_digest": publication[
            "configured_scene_revision_digest"
        ],
        "configured_scene_bundle_reference": publication[
            "configured_scene_bundle_reference"
        ],
        "publication_result_digest": publication["result_digest"],
        "full_byte_service_account_readback_passed": True,
        "execution_digest": "",
    }
    result["execution_digest"] = canonical_digest(
        result, digest_field="execution_digest"
    )
    return result


__all__ = [
    "PARENT_LAUNCH_SCHEMA_VERSION",
    "TaskEvaluationSceneConfigurationExecutionError",
    "execute_and_publish_scene_configuration",
]
