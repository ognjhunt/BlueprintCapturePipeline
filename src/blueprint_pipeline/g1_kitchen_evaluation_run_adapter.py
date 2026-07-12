"""Compatibility adapter from the legacy G1/kitchen lane to Evaluation Run."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

from .evaluation_run import EVALUATION_RUN_SCHEMA_VERSION


G1_KITCHEN_SCENE_ADAPTER = "openusd_scene_bundle"
G1_KITCHEN_ROBOT_ADAPTER = "isaac_unitree_g1"
G1_KITCHEN_TASK_PACK_ADAPTER = "manifest_task_scenario_pack"
G1_KITCHEN_RUNTIME_ADAPTER = "isaac_provider_runtime"
G1_KITCHEN_PROOF_ADAPTER = "declared_evidence_proof_contract"
G1_KITCHEN_EXECUTION_ADAPTER = "isaac_g1_kitchen_parity_compatibility"


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _safe_id(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._:-]+", "-", value).strip("-._:")
    return normalized[:160] or "evaluation-run"


def _sha256(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def _evidence_uri(value: str | None) -> str:
    """Retain resource identity without persisting signed URL credentials."""

    raw = _string(value)
    if not raw:
        return ""
    parsed = urlsplit(raw)
    if parsed.scheme in {"http", "https"}:
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
    return raw


def _run_id(
    *,
    out_dir: str | Path,
    scenarios: Sequence[Mapping[str, Any]],
    policy_id: str,
) -> str:
    identity = json.dumps(
        {
            "out_dir": str(Path(out_dir).expanduser()),
            "scenario_ids": [
                _string(row.get("scenario_id") or row.get("id")) for row in scenarios
            ],
            "policy_id": policy_id,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    suffix = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    return _safe_id(f"{Path(out_dir).name}-{suffix}")


def _task_pack(scenarios: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    normalized_scenarios: list[dict[str, Any]] = []
    task_ids: list[str] = []
    for index, raw in enumerate(scenarios):
        scenario = dict(raw)
        scenario_id = _string(scenario.get("scenario_id") or scenario.get("id"))
        task_id = _string(scenario.get("task_id")) or "g1_kitchen_parity_task"
        normalized_scenarios.append(
            {
                **scenario,
                "scenario_id": scenario_id or f"scenario-{index + 1}",
                "task_id": task_id,
            }
        )
        if task_id not in task_ids:
            task_ids.append(task_id)
    return {
        "adapter_id": G1_KITCHEN_TASK_PACK_ADAPTER,
        "adapter_version": "1",
        "pack_id": "g1-kitchen-parity",
        "tasks": [{"task_id": task_id} for task_id in task_ids],
        "scenarios": normalized_scenarios,
        "source_contract": "legacy_g1_kitchen_scenarios",
    }


def _policy_adapter(policy_id: str) -> dict[str, Any]:
    learned = policy_id in {
        "groot_sonic",
        "groot",
        "groot_n17_sonic",
        "unitree_groot_n17_sonic_policy",
    }
    return {
        "adapter_id": (
            "unitree_groot_n17_sonic" if learned else "isaac_g1_deterministic_controller"
        ),
        "adapter_version": "1",
        "policy_id": policy_id,
        "observation_schema_ref": "blueprint://schemas/robot_eval_observation.v1",
        "action_schema_ref": "blueprint://schemas/robot_eval_action_trace.v1",
        "execution_kind": "persistent_worker_or_command" if learned else "in_process",
    }


def build_g1_kitchen_evaluation_run_spec(
    *,
    out_dir: str | Path,
    scenarios: Sequence[Mapping[str, Any]],
    kitchen_uri: str | None,
    kitchen_main_usd_relative: str,
    kitchen_asset_inventory: Mapping[str, Any] | None,
    g1_usd: str,
    policy_id: str,
    providers: Sequence[str],
    selected_image: str,
    allow_paid: bool,
    max_spend_usd: float | None,
    image_startup_canary: bool,
    serve: bool,
    requested_render_settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Translate the legacy lane into the stable six-part run interface."""

    inventory = dict(kitchen_asset_inventory or {})
    archive_sha = _string(inventory.get("archive_sha256"))
    content_digest = (
        archive_sha if archive_sha.startswith("sha256:") else f"sha256:{archive_sha}"
    ) if archive_sha else None
    mode = "startup_canary" if image_startup_canary else "serve" if serve else "evaluate"
    scene_uri = _evidence_uri(kitchen_uri)
    identity_status = "verified" if content_digest else "legacy_unverified"
    if mode == "startup_canary" and not scene_uri:
        identity_status = "not_required_for_startup_canary"
    elif not scene_uri:
        # Historical dry-run callers prepared a bundle before choosing or
        # staging a kitchen archive. Preserve that behavior as an explicit,
        # non-proof compatibility reference instead of pretending an asset was
        # resolved.
        scene_uri = "legacy://g1-kitchen-scene-unmaterialized"
    return {
        "schema_version": EVALUATION_RUN_SCHEMA_VERSION,
        "run_id": _run_id(out_dir=out_dir, scenarios=scenarios, policy_id=policy_id),
        "mode": mode,
        "scene_bundle": {
            "adapter_id": G1_KITCHEN_SCENE_ADAPTER,
            "adapter_version": "1",
            "bundle_id": "g1-kitchen-scene",
            "uri": scene_uri or None,
            "entrypoint": kitchen_main_usd_relative if mode != "startup_canary" else None,
            "format": "openusd",
            "content_digest": content_digest,
            "identity_status": identity_status,
            "inventory": {
                "file_count": inventory.get("file_count"),
                "total_bytes": inventory.get("total_bytes"),
            },
        },
        "robot_adapter": {
            "adapter_id": G1_KITCHEN_ROBOT_ADAPTER,
            "adapter_version": "1",
            "robot_profile_id": "unitree_g1",
            "asset_ref": g1_usd if mode != "startup_canary" else g1_usd or None,
            "simulator": "isaac_sim",
        },
        "task_scenario_pack": _task_pack(scenarios),
        "policy_adapter": _policy_adapter(policy_id),
        "runtime_provider_profile": {
            "adapter_id": G1_KITCHEN_RUNTIME_ADAPTER,
            "adapter_version": "1",
            "execution_adapter_id": G1_KITCHEN_EXECUTION_ADAPTER,
            "profile_id": "isaac-gpu-review",
            "providers": list(providers),
            "simulator": "isaac_sim",
            "worker_image_ref": selected_image or None,
            "paid_execution_requested": bool(allow_paid),
            "max_spend_usd": max_spend_usd,
            "render_settings": dict(requested_render_settings),
        },
        "proof_contract": {
            "adapter_id": G1_KITCHEN_PROOF_ADAPTER,
            "adapter_version": "1",
            "contract_id": "g1-kitchen-attempt-closure",
            "contract_schema_version": "g1_kitchen_attempt_closure.v1",
            "required_evidence": [
                "provider_startup",
                "worker_runtime",
                "policy_action_trace",
                "review_media",
                "task_state_change",
                "terminal_provider_teardown",
            ],
            "claim_ceiling": {
                "startup_only": bool(image_startup_canary),
                "simulator_task_success_requires_attempt_closure": True,
                "physical_robot_readiness": False,
                "deployment_readiness": False,
            },
            "prohibited_claims": [
                "physical_robot_readiness",
                "deployment_approval",
                "field_safety",
                "task_success_without_attempt_closure",
            ],
        },
        "metadata": {
            "compatibility_source": "isaac_g1_kitchen_parity_job.v1",
            "legacy_lane_retained_for_artifact_read_compatibility": True,
            "scene_specific_name_is_not_platform_contract": True,
            "source_identity": _sha256(
                f"{kitchen_main_usd_relative}\0{g1_usd}\0{policy_id}"
            ),
        },
    }
