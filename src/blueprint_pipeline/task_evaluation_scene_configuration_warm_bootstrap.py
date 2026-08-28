"""Bootstrap a retained scene worker or tear it down fail-closed."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .common import redacted_failure_detail, utc_now_iso, write_json


DIAGNOSTIC_RESULT_SCHEMA_VERSION = (
    "task_evaluation_scene_configuration_diagnostic_vast_result.v1"
)


def validate_warm_bootstrap_request(
    *,
    requested: bool,
    diagnostic_only: bool,
    authority_path: str | Path | None,
    output_root: str | Path | None,
    bundle_receipt: Mapping[str, Any],
    paid_authority: Mapping[str, Any],
    error_factory: Callable[[str], Exception],
) -> dict[str, Any] | None:
    if not requested:
        return None
    if not diagnostic_only or authority_path is None or output_root is None:
        raise error_factory("scene_configuration_warm_session_authority_missing")
    from .task_evaluation_scene_configuration_warm_diagnostic import (  # noqa: PLC0415
        validate_scene_configuration_warm_session_authority,
    )

    authority = validate_scene_configuration_warm_session_authority(authority_path)
    if (
        authority.get("bundle_sha256") != bundle_receipt.get("bundle_sha256")
        or authority.get("paid_attempt_authority_digest")
        != paid_authority.get("authority_digest")
        or authority.get("source_checkpoint_digest")
        != bundle_receipt.get("source_diagnostic_checkpoint_digest")
        or authority.get("diagnostic_bootstrap_mode")
        != bundle_receipt.get("diagnostic_bootstrap_mode")
        or authority.get("scientific_binding_digest")
        != bundle_receipt.get("diagnostic_scientific_binding_digest")
    ):
        raise error_factory("scene_configuration_warm_session_authority_binding_mismatch")
    return authority


def handle_retained_scene_configuration_bootstrap(
    *,
    adapter: Mapping[str, Any],
    runtime_secret_cleanup_blockers: list[str],
    cleanup: Mapping[str, Any],
    output_zip: Path,
    job: Path,
    provider_run: Path,
    expected_upload_bytes: int,
    warm_session_authority_path: str | Path | None,
    warm_session_output_root: str | Path | None,
    paid_resource_admission_grant: Any,
    watchdog: Any,
    watchdog_handoff: Mapping[str, Any],
    receipt: Mapping[str, Any],
    authority: Mapping[str, Any],
    warm_session_authority: Mapping[str, Any] | None,
    diagnostic_claim_boundary: Mapping[str, Any],
    extract_provider_output: Callable[..., tuple[dict[str, Any], list[str]]],
    seal_terminal_result: Callable[[Path, Mapping[str, Any]], dict[str, Any]],
    close_independent_vast_watchdog: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    retained_blockers = list(runtime_secret_cleanup_blockers)
    if cleanup.get("all_objects_absent") is not True:
        retained_blockers.append("object_store_provider_zero_not_proven")
    try:
        if retained_blockers:
            raise ValueError(
                retained_blockers[0]
            )
        from .task_evaluation_scene_configuration_warm_diagnostic import (  # noqa: PLC0415
            materialize_scene_configuration_warm_session,
        )

        bootstrap_execution, bootstrap_extraction_blockers = (
            extract_provider_output(
                output_zip,
                job / "warm_bootstrap_immutable_execution",
                maximum_archive_bytes=expected_upload_bytes,
                diagnostic_only=True,
            )
        )
        retained_blockers.extend(bootstrap_extraction_blockers)
        advanced_checkpoint = bootstrap_execution.pop(
            "_validated_advanced_checkpoint", None
        )
        artifixer_post_training_checkpoint = bootstrap_execution.pop(
            "_validated_artifixer_post_training_checkpoint", None
        )
        from .task_evaluation_scene_configuration_warm_execution_contract import (  # noqa: PLC0415
            warm_bootstrap_execution_binding_blockers,
        )

        retained_blockers.extend(
            warm_bootstrap_execution_binding_blockers(
                execution=bootstrap_execution,
                bundle_receipt=receipt,
                session_authority=warm_session_authority or {},
                advanced_checkpoint=advanced_checkpoint,
                artifixer_post_training_checkpoint=(
                    artifixer_post_training_checkpoint
                ),
            )
        )
        bootstrap_stage_blockers = [
            item
            for item in retained_blockers
            if str(item).startswith("provider_result_blocker:")
        ]
        unsafe_retention_blockers = [
            item
            for item in retained_blockers
            if not str(item).startswith("provider_result_blocker:")
        ]
        checkpoint_kinds = sum(
            isinstance(candidate, Mapping)
            for candidate in (
                advanced_checkpoint,
                artifixer_post_training_checkpoint,
            )
        )
        if unsafe_retention_blockers or checkpoint_kinds != 1:
            raise ValueError(
                unsafe_retention_blockers[0]
                if unsafe_retention_blockers
                else "scene_configuration_warm_checkpoint_kind_invalid"
            )

        warm_session = materialize_scene_configuration_warm_session(
            session_authority_path=warm_session_authority_path,
            adapter_result_path=(
                provider_run / "vast_provider_adapter_result.json"
            ),
            watchdog_handoff_path=(
                job / "vast_independent_watchdog_handoff.json"
            ),
            output_root=warm_session_output_root,
            advanced_checkpoint=advanced_checkpoint,
            artifixer_post_training_checkpoint=(
                artifixer_post_training_checkpoint
            ),
            bootstrap_allocation_binding_digest=str(
                paid_resource_admission_grant.allocation_binding_digest
                if paid_resource_admission_grant is not None
                else ""
            ),
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        # Session sealing failed after the adapter intentionally retained
        # the worker. Immediately destroy and double-prove exact/global
        # absence; the independent watchdog remains armed if any read is
        # uncertain.
        from .gpu_render_providers import VastRenderProvider  # noqa: PLC0415

        retained_provider = VastRenderProvider()
        retained_ids = [
            int(value)
            for value in adapter.get("vast_instance_ids") or []
            if isinstance(value, int) and not isinstance(value, bool)
        ]
        termination: dict[str, Any] = {}
        exact_observations: list[dict[str, Any]] = []
        inventory_observations: list[dict[str, Any]] = []
        if len(retained_ids) == 1:
            try:
                termination = retained_provider.terminate(
                    str(retained_ids[0])
                )
            except Exception as teardown_exc:  # noqa: BLE001
                termination = {
                    "status": "terminate_failed",
                    "error_type": type(teardown_exc).__name__,
                }
            for index in range(2):
                try:
                    exact_observations.append(
                        retained_provider.inspect(str(retained_ids[0]))
                    )
                except Exception as inspect_exc:  # noqa: BLE001
                    exact_observations.append(
                        {
                            "status": "observation_failed",
                            "api_confirmed": False,
                            "error_type": type(inspect_exc).__name__,
                        }
                    )
                try:
                    inventory_observations.append(
                        retained_provider.billable_inventory(name_prefix="")
                    )
                except Exception as inventory_exc:  # noqa: BLE001
                    inventory_observations.append(
                        {
                            "status": "observation_failed",
                            "api_confirmed": False,
                            "error_type": type(inventory_exc).__name__,
                        }
                    )
                if index == 0:
                    time.sleep(2)
        provider_zero = bool(
            len(exact_observations) == 2
            and all(
                row.get("status") == "absent"
                and row.get("provider_absence_confirmed") is True
                and row.get("api_confirmed") is True
                for row in exact_observations
            )
            and len(inventory_observations) == 2
            and all(
                row.get("api_confirmed") is True
                and row.get("live_resource_count") == 0
                and row.get("resources") == []
                for row in inventory_observations
            )
        )
        teardown = {
            "schema_version": "vast_teardown_manifest.v1",
            "generated_at": utc_now_iso(),
            "status": "completed" if provider_zero else "blocked",
            "vast_instance_ids": retained_ids,
            "termination": termination,
            "provider_absence_observations": exact_observations,
            "global_billable_inventory_observations": inventory_observations,
            "continuing_spend_from_this_run": not provider_zero,
            "raw_secret_values_recorded": False,
        }
        write_json(provider_run / "vast_teardown_manifest.json", teardown)
        try:
            watchdog_close = close_independent_vast_watchdog(
                job_dir=job,
                handle=watchdog,
                instance_ids=retained_ids,
                provider_teardown_completed=provider_zero,
                provider_allocation_impossible=False,
            )
        except (OSError, RuntimeError, ValueError) as watchdog_exc:
            watchdog_close = {
                "status": "blocked",
                "watchdog_remains_armed": True,
                "error_type": type(watchdog_exc).__name__,
            }
            provider_zero = False
        blockers = [
            "scene_configuration_warm_session_materialization_failed:"
            + redacted_failure_detail(exc)
        ]
        if not provider_zero:
            blockers.append("provider_zero_not_proven")
        return seal_terminal_result(
            job,
            {
                "schema_version": DIAGNOSTIC_RESULT_SCHEMA_VERSION,
                "status": "blocked_diagnostic_only",
                "run_id": receipt["run_id"],
                "source_commit": receipt["source_commit"],
                "bundle_sha256": receipt["bundle_sha256"],
                "authority_digest": authority["authority_digest"],
                "provider_mutations_performed": len(retained_ids),
                "retry_cap": 0,
                "independent_watchdog": watchdog_close,
                "object_store_cleanup": cleanup,
                "continuing_spend_from_this_run": not provider_zero,
                **diagnostic_claim_boundary,
                "blockers": blockers,
            },
        )
    else:
        return seal_terminal_result(
            job,
            {
                "schema_version": DIAGNOSTIC_RESULT_SCHEMA_VERSION,
                "status": "retained_warm_diagnostic_session",
                "run_id": receipt["run_id"],
                "source_commit": receipt["source_commit"],
                "bundle_sha256": receipt["bundle_sha256"],
                "authority_digest": authority["authority_digest"],
                "warm_session_authority_digest": (
                    warm_session_authority or {}
                ).get("authority_digest"),
                "warm_session_digest": warm_session["session_digest"],
                "warm_session_path": warm_session["session_path"],
                "provider_instance_id": warm_session["provider_instance_id"],
                "provider_mutations_performed": 1,
                "retry_cap": 0,
                "independent_watchdog": watchdog_handoff,
                "object_store_cleanup": cleanup,
                "continuing_spend_from_this_run": True,
                "bootstrap_stage_blockers": bootstrap_stage_blockers,
                "diagnostic_only": True,
                "qualification_eligible": False,
                "executed_inside_one_parent_provider_run": False,
                "configured_revision_publication_permitted": False,
                "offering_publication_permitted": False,
                "terminal_e2e_completion_permitted": False,
                "blockers": [],
            },
        )
