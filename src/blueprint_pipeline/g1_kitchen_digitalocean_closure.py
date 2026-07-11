"""Adapter from DigitalOcean sealed-run artifacts to G1 attempt closure."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .common import utc_now_iso, write_json
from .g1_kitchen_attempt_closure import (
    build_attempt_closure,
    buyer_readout_projection,
)
from .provider_reliability_manifest import (
    TEARDOWN_STATUS_SOURCE_PROVIDER_API,
    build_teardown_proof,
)
from .g1_kitchen_run_index import append_run_index_event
from .isaac_review_media import admit_collected_scenario_episode


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _read(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def _startup_rows(
    *, collected_root: Path, identity: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    startup_root = collected_root / "closed_loop_out" / "startup_gates"
    summary_path = startup_root / "supervised_startup_gates.json"
    summary = _read(summary_path) if summary_path.is_file() else {}
    expected_nonce = str(identity.get("launch_nonce") or "")
    expected_image = str(identity.get("image_digest") or "").removeprefix("sha256:")
    observed_image = str(summary.get("image_digest") or "")
    observed_image = observed_image.rsplit("@sha256:", 1)[-1].removeprefix("sha256:")
    identity_matches = bool(
        summary
        and summary.get("launch_session_id") == expected_nonce
        and observed_image == expected_image
    )
    gates = _mapping(summary.get("gates"))
    specs = {
        "fast_canary": "fast_startup_canary",
        "review_canary": "review_renderer_canary",
        "asset_gate": "kitchen_asset_startup_gate",
    }
    rows: dict[str, dict[str, Any]] = {
        "startup": {
            "status": "passed"
            if summary.get("status") == "passed" and identity_matches
            else "blocked",
            "identity_binding": dict(identity),
            "evidence": {
                "summary": summary,
                "attempt_identity_matches": identity_matches,
            },
            "artifact_refs": [str(summary_path)] if summary_path.is_file() else [],
            "blockers": []
            if summary.get("status") == "passed" and identity_matches
            else ["attempt_bound_same_allocation_startup_proof_missing_or_invalid"],
        }
    }
    for row_id, gate_id in specs.items():
        gate = _mapping(gates.get(gate_id))
        passed = gate.get("status") == "passed" and identity_matches
        rows[row_id] = {
            "status": "passed" if passed else "blocked",
            "identity_binding": dict(identity),
            "evidence": {"gate_id": gate_id, "gate": gate},
            "blockers": [] if passed else [f"attempt_bound_{gate_id}_not_passed"],
        }
    return rows


def _collected_media_rows(
    *, collected_root: Path, identity: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    admissions: list[dict[str, Any]] = []
    for frames_dir in sorted(collected_root.rglob("frames")):
        overview = sorted(frames_dir.glob("overview_[0-9][0-9][0-9][0-9].png"))
        robot_pov = sorted(frames_dir.glob("robot_pov_[0-9][0-9][0-9][0-9].png"))
        if not overview and not robot_pov:
            continue
        expected_count = len(overview) if len(overview) == len(robot_pov) else max(
            len(overview), len(robot_pov)
        )
        admissions.append(
            admit_collected_scenario_episode(
                scenario_dir=frames_dir.parent,
                expected_frame_count=expected_count,
            )
        )
    passed = bool(admissions) and all(row.get("status") == "passed" for row in admissions)
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
    evidence = {"scenario_admissions": admissions, "scenario_count": len(admissions)}
    common = {
        "status": "passed" if passed else "blocked",
        "identity_binding": dict(identity),
        "evidence": evidence,
        "blockers": blockers,
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
        "image_digest": str(image_ref).rsplit("@sha256:", 1)[-1],
        "bundle_digest": artifact_digest("bundle"),
        "kitchen_asset_digest": artifact_digest("kitchen_inventory")
        or hashlib.sha256(Path(kitchen_asset_archive_file).read_bytes()).hexdigest(),
        "active_selection_sha256": artifact_digest("selection"),
        "task_contract_sha256": artifact_digest("task_success_contract")
        or hashlib.sha256(Path(task_success_contract_file).read_bytes()).hexdigest(),
        "provider_allocation_id": launch.get("instance_id"),
    }
    worker_manifest = _mapping(
        _mapping(watch.get("runner_result")).get("closed_loop_manifest")
    )
    raw_rows = worker_manifest.get("g1_kitchen_proof_rows")
    if isinstance(raw_rows, Sequence) and not isinstance(raw_rows, (str, bytes)):
        rows = {
            str(row.get("row_id") or ""): dict(row)
            for row in raw_rows
            if isinstance(row, Mapping) and str(row.get("row_id") or "")
        }
    else:
        rows = _mapping(raw_rows)
    binding = dict(identity)
    worker_manifest_sha256 = hashlib.sha256(
        json.dumps(worker_manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    for row in rows.values():
        if not isinstance(row, dict):
            continue
        row["identity_binding"] = binding
        row["evidence"] = {
            **_mapping(row.get("evidence")),
            "worker_manifest_sha256": worker_manifest_sha256,
        }
    collected_root = Path(output_dir) / "closed_loop_output"
    rows.update(_startup_rows(collected_root=collected_root, identity=identity))
    rows.update(_collected_media_rows(collected_root=collected_root, identity=identity))
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
