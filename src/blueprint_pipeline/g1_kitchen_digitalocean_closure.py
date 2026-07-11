"""Adapter from DigitalOcean sealed-run artifacts to G1 attempt closure."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .common import utc_now_iso, write_json
from .g1_kitchen_attempt_closure import (
    build_attempt_closure,
    buyer_readout_projection,
)
from .g1_kitchen_proof_row_validation import (
    ATTESTATION_PINS_FILE_ENV,
    load_attestation_pins,
    transition_terminal_horizon,
    transition_step_bindings,
    validate_worker_proof_rows,
)
from .provider_reliability_manifest import (
    TEARDOWN_STATUS_SOURCE_PROVIDER_API,
    build_teardown_proof,
)
from .g1_kitchen_run_index import append_run_index_event
from .isaac_review_media import admit_collected_scenario_episode


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _strict_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _read(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def _collected_media_rows(
    *,
    collected_root: Path,
    identity: Mapping[str, Any],
    expected_frame_count: int,
    expected_scenario_count: int,
    step_bindings: Sequence[Mapping[str, Any]] | None,
    attestation_pins: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    """Admit collected media against the attempt-bound horizon, never the files.

    ``expected_frame_count`` and ``expected_scenario_count`` come from the
    immutable attempt/task/executor request; equally truncated camera streams
    therefore block instead of shrinking the horizon to whatever arrived.
    """
    admissions: list[dict[str, Any]] = []
    for frames_dir in sorted(collected_root.rglob("frames")):
        overview = sorted(frames_dir.glob("overview_[0-9][0-9][0-9][0-9].png"))
        robot_pov = sorted(frames_dir.glob("robot_pov_[0-9][0-9][0-9][0-9].png"))
        if not overview and not robot_pov:
            continue
        admissions.append(
            admit_collected_scenario_episode(
                scenario_dir=frames_dir.parent,
                expected_frame_count=int(expected_frame_count),
                step_bindings=step_bindings,
                attestation_pins=attestation_pins,
                identity_binding=identity,
            )
        )
    passed = (
        bool(admissions)
        and len(admissions) == int(expected_scenario_count)
        and all(row.get("status") == "passed" for row in admissions)
    )
    blockers = sorted(
        {
            str(blocker)
            for admission in admissions
            for blocker in admission.get("blockers", [])
            if str(blocker)
        }
    )
    if not admissions:
        blockers.append("full_ordered_episode_media_not_collected")
    if len(admissions) != int(expected_scenario_count):
        blockers.append(
            f"scenario_count_mismatch:{len(admissions)}!={int(expected_scenario_count)}"
        )
    evidence = {
        "scenario_admissions": admissions,
        "scenario_count": len(admissions),
        "expected_scenario_count": int(expected_scenario_count),
        "expected_frame_count_per_camera": int(expected_frame_count),
    }
    common = {
        "status": "passed" if passed else "blocked",
        "identity_binding": dict(identity),
        "evidence": evidence,
        "blockers": sorted(set(blockers)),
    }
    return {
        "robot_pov": dict(common),
        "semantic_review": dict(common),
    }


def teardown_proof_from_digitalocean_watch(
    *, instance_id: str, watch: Mapping[str, Any]
) -> dict[str, Any]:
    teardown = _mapping(watch.get("teardown"))
    reason = str(watch.get("teardown_reason") or "").strip().lower()
    status = str(teardown.get("status") or "").strip().lower()
    if status in {"stopped", "preserved", "skipped"} or reason in {
        "left_running_by_request",
        "runner_done_preserved_for_warm_reuse",
    }:
        return build_teardown_proof(
            provider="digitalocean",
            allocation_id=instance_id,
            terminate_requested=False,
            provider_terminal_status=None,
            keep_alive_requested=True,
            keep_alive_reason=reason or status or "kept_alive",
        )
    verification = _mapping(teardown.get("verification"))
    observed = str(verification.get("provider_status") or "").strip().lower()
    if verification.get("api_confirmed") is True and observed:
        return build_teardown_proof(
            provider="digitalocean",
            allocation_id=instance_id,
            terminate_requested=True,
            provider_terminal_status=observed,
            verified_at=utc_now_iso(),
            status_source=TEARDOWN_STATUS_SOURCE_PROVIDER_API,
        )
    return build_teardown_proof(
        provider="digitalocean",
        allocation_id=instance_id,
        terminate_requested=True,
        provider_terminal_status="terminated" if status == "terminated" else status or None,
        verified_at=utc_now_iso() if status == "terminated" else None,
    )


def finalize_digitalocean_attempt_closure(
    *,
    provider: Any,
    output_dir: str | Path,
    image_ref: str,
    attempt_input_manifest_file: str | Path,
    task_success_contract_file: str | Path,
    kitchen_asset_archive_file: str | Path,
    launch: Mapping[str, Any],
    watch: Mapping[str, Any],
    teardown_proof: Mapping[str, Any],
    expected_episode_steps: int,
    expected_min_episode_steps: int = 1,
    expected_scenario_count: int = 1,
    attestation_pins_file: str | Path | None = None,
    resource_name_prefix: str = "blueprint-groot-oscar-closed-loop",
) -> dict[str, Any]:
    try:
        inventory = dict(provider.billable_inventory(name_prefix=resource_name_prefix))
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        inventory = {
            "status": "blocked",
            "api_confirmed": False,
            "live_resource_count": None,
            "blockers": [f"provider_final_inventory_failed:{type(exc).__name__}"],
        }
    attempt_input = _read(attempt_input_manifest_file)
    artifacts = _mapping(attempt_input.get("artifacts"))

    def artifact_digest(name: str) -> str | None:
        return str(_mapping(artifacts.get(name)).get("sha256") or "").strip() or None

    identity = {
        "run_id": attempt_input.get("run_id"),
        "attempt_id": attempt_input.get("attempt_id"),
        "launch_nonce": attempt_input.get("launch_nonce"),
        "source_commit": attempt_input.get("source_commit"),
        "source_dirty_patch_sha256": attempt_input.get("source_dirty_patch_sha256"),
        "image_digest": str(image_ref).rsplit("@sha256:", 1)[-1],
        "bundle_digest": artifact_digest("bundle"),
        "kitchen_asset_digest": artifact_digest("kitchen_inventory")
        or hashlib.sha256(Path(kitchen_asset_archive_file).read_bytes()).hexdigest(),
        "active_selection_sha256": artifact_digest("selection"),
        "task_contract_sha256": artifact_digest("task_success_contract")
        or hashlib.sha256(Path(task_success_contract_file).read_bytes()).hexdigest(),
        "provider_allocation_id": launch.get("instance_id"),
    }
    collected_root = Path(output_dir) / "closed_loop_output"
    worker_manifest_path = (
        collected_root / "closed_loop_out" / "oscar_isaac_closed_loop_manifest.json"
    )
    worker_rows: dict[str, Any] = {}
    if worker_manifest_path.is_file():
        try:
            worker_manifest = _read(worker_manifest_path)
        except (json.JSONDecodeError, UnicodeDecodeError):
            worker_manifest = {}
        raw_rows = worker_manifest.get("g1_kitchen_proof_rows")
        if isinstance(raw_rows, Sequence) and not isinstance(raw_rows, (str, bytes)):
            worker_rows = {
                str(row.get("row_id") or ""): dict(row)
                for row in raw_rows
                if isinstance(row, Mapping) and str(row.get("row_id") or "")
            }
        else:
            worker_rows = _mapping(raw_rows)
    pins_source = attestation_pins_file or os.environ.get(ATTESTATION_PINS_FILE_ENV)
    attestation_pins = load_attestation_pins(pins_source) if pins_source else None
    validated = validate_worker_proof_rows(
        worker_rows=worker_rows,
        worker_manifest_path=worker_manifest_path,
        collected_root=collected_root,
        identity=identity,
        attestation_pins=attestation_pins,
    )
    rows: dict[str, Any] = dict(validated["rows"])
    binding = dict(identity)
    step_bindings = transition_step_bindings(rows) or None
    terminal_horizon = transition_terminal_horizon(rows)
    horizon_blocker = None
    if step_bindings is not None and (
        terminal_horizon is None
        or _strict_int(terminal_horizon.get("planned_max_steps"))
        != int(expected_episode_steps)
        or _strict_int(terminal_horizon.get("scenario_count"))
        != int(expected_scenario_count)
        or not int(expected_min_episode_steps)
        <= len(step_bindings)
        <= int(expected_episode_steps)
    ):
        horizon_blocker = (
            "attested_episode_horizon_not_bound_to_immutable_request:"
            f"steps={len(step_bindings)},planned={_mapping(terminal_horizon).get('planned_max_steps')},"
            f"scenario_count={_mapping(terminal_horizon).get('scenario_count')}"
        )
        step_bindings = None
    media_expected = (
        len(step_bindings) if step_bindings else int(expected_episode_steps)
    )
    media_rows = _collected_media_rows(
        collected_root=collected_root,
        identity=identity,
        expected_frame_count=media_expected,
        expected_scenario_count=int(expected_scenario_count),
        step_bindings=step_bindings,
        attestation_pins=attestation_pins,
    )
    if horizon_blocker:
        for row in media_rows.values():
            row["status"] = "blocked"
            row["blockers"] = sorted({*row.get("blockers", []), horizon_blocker})
    rows.update(media_rows)
    rows["allocation"] = {
        "status": "passed" if launch.get("status") == "launched" else "blocked",
        "identity_binding": binding,
        "evidence": {"provider_allocation_id": launch.get("instance_id")},
    }
    rows["teardown"] = {
        "status": "passed" if teardown_proof.get("status") == "PASS" else "blocked",
        "identity_binding": binding,
        "evidence": {
            "api_confirmed": teardown_proof.get("status") == "PASS",
            "terminal_state": teardown_proof.get("provider_terminal_status"),
        },
        "blockers": teardown_proof.get("blockers") or [],
    }
    rows["final_inventory"] = {
        "status": "passed"
        if inventory.get("api_confirmed") is True
        and int(inventory.get("live_resource_count") or 0) == 0
        else "blocked",
        "identity_binding": binding,
        "evidence": {
            "api_confirmed": inventory.get("api_confirmed") is True,
            "live_resource_count": inventory.get("live_resource_count"),
        },
        "blockers": inventory.get("blockers") or [],
    }
    closure = build_attempt_closure(identity=identity, proof_rows=rows)
    output = Path(output_dir) / "g1_kitchen_attempt_closure.json"
    write_json(output, closure)
    index_event = append_run_index_event(
        run_root=output_dir,
        event_type="attempt_terminalized",
        run_id=str(identity.get("run_id") or ""),
        attempt_id=str(identity.get("attempt_id") or ""),
        artifact_paths=[output],
        detail={
            "terminal_reason": closure.get("terminal_reason"),
            "closure_status": closure.get("status"),
        },
    )
    return {
        "closure": closure,
        "closure_path": str(output),
        "final_inventory": inventory,
        "buyer_readout_projection": buyer_readout_projection(closure),
        "run_index_event": index_event,
    }
