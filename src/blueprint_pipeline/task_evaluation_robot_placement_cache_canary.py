"""Real Sol canary through the production task-aware robot-placement request path."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .openai_prompt_cache_canary import load_secure_api_key_file
from .task_evaluation_openai_inference_usage import (
    build_placement_inference_usage_packet,
    sync_inference_usage_to_webapp,
)
from .task_evaluation_robot_placement_agent import (
    ROBOT_PLACEMENT_AGENT_MODEL,
    run_task_evaluation_robot_placement_agent,
)
from .task_evaluation_supervisor.agents_sdk import (
    OpenAIAgentsSDKConfig,
    OpenAIAgentsSDKInvoker,
)
from .task_evaluation_robot_placement_trajectory import (
    placement_trajectory_from_native_plan,
)


SCHEMA_VERSION = "task_evaluation_robot_placement_cache_canary.v1"
SEQUENCE_COUNT = 4
EXPECTED_CALL_COUNT = 8
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


class RobotPlacementCacheCanaryError(RuntimeError):
    """The production-shape Sol path did not meet its cache or quality contract."""


def _schematic_image() -> tuple[bytes, dict[str, Any]]:
    image = Image.new("RGB", (640, 480), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 340, 600, 410), fill="#b7c3d0", outline="#1f2937", width=4)
    draw.rectangle((150, 210, 250, 340), fill="#5b8def", outline="#1f2937", width=4)
    draw.line((200, 210, 360, 145), fill="#1f2937", width=18)
    draw.ellipse((340, 125, 390, 175), fill="#ef8354", outline="#1f2937", width=4)
    draw.rectangle((470, 285, 540, 340), fill="#55a868", outline="#1f2937", width=4)
    draw.text((125, 420), "SUPPORTED ROBOT BASE", fill="#111827")
    draw.text((450, 255), "TASK TARGET", fill="#111827")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    payload = buffer.getvalue()
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    return payload, {
        "label": "rights_safe_robot_placement_schematic",
        "digest": digest,
        "image_url": "data:image/png;base64," + base64.b64encode(payload).decode("ascii"),
        "detail": "high",
    }


def _trajectory() -> dict[str, Any]:
    plan: dict[str, Any] = {
        "schema_version": "native_rigid_construction_phase_plan.v1",
        "task_kind": "rigid_pick_place",
        "manipulation_strategy": "planar_push",
        "phase_count": 2,
        "execution_parameters": {
            "arrival_tolerance_m": 0.02,
            "arrival_orientation_tolerance_rad": 0.08,
        },
        "phases": [
            {
                "phase_id": "precontact",
                "position_world_m": [0.6, 0.0, 0.8],
                "orientation_world_xyzw": [0.0, 0.70710678, 0.0, 0.70710678],
                "gripper_state": "open",
                "gate_ids": ["precontact_reachability"],
            },
            {
                "phase_id": "push_contact",
                "position_world_m": [0.72, 0.0, 0.8],
                "orientation_world_xyzw": [0.0, 0.70710678, 0.0, 0.70710678],
                "gripper_state": "closed",
                "gate_ids": ["push_contact"],
            },
        ],
        "plan_digest": "",
    }
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    return placement_trajectory_from_native_plan(plan)


def _inventory(trajectory_digest: str) -> dict[str, Any]:
    candidate = {
        "candidate_id": "cache_canary_candidate_0001",
        "pose": {
            "position_world_m": [0.0, 0.0, 0.0],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "support_surface_id": "schematic_floor",
        "geometry_gate_digest": "sha256:" + "1" * 64,
        "trajectory_position_ik_gate_digest": "sha256:" + "2" * 64,
    }
    candidates = [candidate]
    return {
        "deterministic_geometry_passing_candidate_inventory": candidates,
        "deterministic_geometry_passing_candidate_inventory_digest": canonical_digest(
            {"trajectory_digest": trajectory_digest, "candidates": candidates}
        ),
        "deterministic_geometry_passing_candidate_inventory_trajectory_digest": (
            trajectory_digest
        ),
    }


def _geometry_gate(proposal: dict[str, Any]) -> dict[str, Any]:
    gate: dict[str, Any] = {
        "schema_version": "task_evaluation_robot_placement_geometry_gate.v1",
        "status": "passed",
        "candidate_id": proposal["candidate_id"],
        "blockers": [],
        "geometry_gate_digest": "",
    }
    gate["geometry_gate_digest"] = canonical_digest(
        gate, digest_field="geometry_gate_digest"
    )
    return gate


def _calls(receipts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for sequence_index, receipt in enumerate(receipts):
        for round_record in receipt.get("rounds") or []:
            for field, family in (
                ("proposal_usage", "task_aware_robot_placement_proposal"),
                ("visual_review_usage", "robot_placement_visual_review"),
            ):
                usage = round_record.get(field)
                if isinstance(usage, dict):
                    calls.append(
                        {
                            "sequence_index": sequence_index,
                            "family": family,
                            "usage": usage,
                        }
                    )
    return calls


def run_production_shape_canary(
    *,
    output_dir: str | Path,
    api_key_file: str | Path | None,
    source_commit: str,
    max_total_cost_usd: float = 2.0,
    webapp_endpoint: str | None = None,
    require_webapp_sync: bool = False,
    verify_source_commit: bool = True,
) -> dict[str, Any]:
    if _COMMIT.fullmatch(source_commit) is None:
        raise RobotPlacementCacheCanaryError("source_commit_invalid")
    if verify_source_commit:
        repo_root = Path(__file__).resolve().parents[2]
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if completed.stdout.strip() != source_commit:
            raise RobotPlacementCacheCanaryError(
                "source_commit_does_not_match_checkout_head"
            )
    if not 0 < max_total_cost_usd <= 5.0:
        raise RobotPlacementCacheCanaryError("placement_canary_cost_cap_invalid")
    key_path = Path(
        str(api_key_file or os.getenv("OPENAI_API_KEY_FILE") or "")
    ).expanduser().resolve()
    load_secure_api_key_file(key_path)
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True, mode=0o750)
    image_bytes, image_input = _schematic_image()
    image_path = output_root / "rights_safe_robot_placement_schematic.png"
    image_path.write_bytes(image_bytes)
    image_path.chmod(0o440)
    trajectory = _trajectory()
    inventory = _inventory(trajectory["trajectory_digest"])
    scene_binding = {
        "schema_version": "cache_canary_scene_binding.v1",
        "scene_id": "rights_safe_synthetic_cache_canary",
        "revision_digest": "sha256:" + "3" * 64,
    }
    task_binding = {
        "schema_version": "cache_canary_task_binding.v1",
        "task_id": "bounded_planar_push_cache_canary",
        "robot_id": "franka_panda",
    }
    config = OpenAIAgentsSDKConfig(
        model=ROBOT_PLACEMENT_AGENT_MODEL,
        max_turns=1,
        max_output_tokens=8_000,
        allow_live_invocation=True,
        tracing_disabled=True,
        max_inference_cost_usd=max_total_cost_usd,
        input_cost_per_million_tokens_usd=4.0,
        output_cost_per_million_tokens_usd=20.0,
    )
    invoker = OpenAIAgentsSDKInvoker(config)
    previous_key_file = os.environ.get("OPENAI_API_KEY_FILE")
    os.environ["OPENAI_API_KEY_FILE"] = str(key_path)
    os.environ["BLUEPRINT_ALLOW_LIVE_AGENTS_SDK_OPERATORS"] = "true"
    receipts: list[dict[str, Any]] = []
    packets: list[dict[str, Any]] = []
    sync_results: list[dict[str, Any]] = []
    try:
        for sequence_index in range(SEQUENCE_COUNT):
            receipt = run_task_evaluation_robot_placement_agent(
                invoker=invoker,
                run_id=f"placement-cache-canary-{sequence_index + 1}",
                scene_binding=scene_binding,
                task_binding=task_binding,
                scene_context=inventory,
                task_context={
                    "canary_scope": "rights_safe_synthetic_no_robot_or_provider_resource",
                },
                overview_images=[image_input],
                validate_candidate=lambda proposal: _geometry_gate(dict(proposal)),
                render_candidate=lambda _proposal, _round: [image_input],
                execute_candidate=None,
                task_trajectory=trajectory,
                max_rounds=1,
                max_input_tokens=20_000,
                expected_proposal_reuse_count=SEQUENCE_COUNT - 1,
                expected_visual_review_reuse_count=SEQUENCE_COUNT - 1,
                expected_proposal_reuse_probability=1.0,
                expected_visual_review_reuse_probability=1.0,
            )
            receipts.append(receipt)
            receipt_path = output_root / (
                f"task_evaluation_robot_placement_receipt-{sequence_index + 1}.v1.json"
            )
            write_json(receipt_path, receipt)
            packet = build_placement_inference_usage_packet(
                placement_receipt=receipt,
                packet_run_id=f"placement-cache-canary-{sequence_index + 1}",
                launch_id=None,
                source_commit=source_commit,
            )
            packets.append(packet)
            packet_path = output_root / (
                f"openai_inference_usage_packet-{sequence_index + 1}.v1.json"
            )
            write_json(packet_path, packet)
            sync = sync_inference_usage_to_webapp(
                packet=packet,
                endpoint_url=webapp_endpoint,
            )
            sync_results.append(sync)
            write_json(
                output_root
                / f"openai_inference_usage_sync-{sequence_index + 1}.v1.json",
                sync,
            )
    finally:
        if previous_key_file is None:
            os.environ.pop("OPENAI_API_KEY_FILE", None)
        else:
            os.environ["OPENAI_API_KEY_FILE"] = previous_key_file

    calls = _calls(receipts)
    blockers: list[str] = []
    if len(calls) != EXPECTED_CALL_COUNT:
        blockers.append("production_shape_exact_call_count_mismatch")
    for family in (
        "task_aware_robot_placement_proposal",
        "robot_placement_visual_review",
    ):
        family_calls = [call for call in calls if call["family"] == family]
        if len(family_calls) != SEQUENCE_COUNT:
            blockers.append(f"family_call_count_mismatch:{family}")
            continue
        first_usage = family_calls[0]["usage"]
        later_usage = [call["usage"] for call in family_calls[1:]]
        stable_tokens = int(
            (first_usage.get("cache_policy") or {})
            .get("economics", {})
            .get("stable_prefix_tokens", 0)
        )
        if int(first_usage.get("cache_write_tokens") or 0) < 1_024:
            blockers.append(f"first_call_cache_write_missing:{family}")
        if any(
            int(usage.get("cached_tokens") or 0) < 0.7 * stable_tokens
            for usage in later_usage
        ):
            blockers.append(f"subsequent_cache_read_below_target:{family}")
        writes = sum(int(call["usage"].get("cache_write_tokens") or 0) for call in family_calls)
        reads = sum(int(call["usage"].get("cached_tokens") or 0) for call in family_calls)
        if reads <= 0 or writes >= 0.5 * reads:
            blockers.append(f"write_to_read_ratio_target_failed:{family}")
        ordinary_rate = 4.0
        actual_prefix_cost = sum(
            (
                int(call["usage"].get("cache_write_tokens") or 0) * 5.0
                + int(call["usage"].get("cached_tokens") or 0) * 0.4
            )
            / 1_000_000
            for call in family_calls
        )
        baseline_prefix_cost = sum(
            (
                int(call["usage"].get("cache_write_tokens") or 0)
                + int(call["usage"].get("cached_tokens") or 0)
            )
            * ordinary_rate
            / 1_000_000
            for call in family_calls
        )
        if baseline_prefix_cost <= 0 or (
            baseline_prefix_cost - actual_prefix_cost
        ) / baseline_prefix_cost < 0.4:
            blockers.append(f"reusable_prefix_savings_target_failed:{family}")
    if any(
        round_record.get("proposal", {}).get("candidate_id")
        != "cache_canary_candidate_0001"
        for receipt in receipts
        for round_record in receipt.get("rounds") or []
    ):
        blockers.append("exact_candidate_inventory_membership_regressed")
    if require_webapp_sync and any(
        result.get("status") != "succeeded" for result in sync_results
    ):
        blockers.append("production_webapp_usage_sync_failed")
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if not blockers else "failed",
        "model": ROBOT_PLACEMENT_AGENT_MODEL,
        "source_commit": source_commit,
        "source_commit_matches_checkout_head": verify_source_commit,
        "sequence_count": len(receipts),
        "call_count": len(calls),
        "expected_call_count": EXPECTED_CALL_COUNT,
        "max_total_cost_usd": max_total_cost_usd,
        "estimated_total_cost_usd": sum(
            float(call["usage"].get("estimated_total_cost_usd") or 0.0)
            for call in calls
        ),
        "cache_families": sorted({call["family"] for call in calls}),
        "calls": calls,
        "receipt_digests": [receipt["receipt_digest"] for receipt in receipts],
        "packet_digests": [packet["packet_digest"] for packet in packets],
        "webapp_sync": sync_results,
        "candidate_inventory_membership_preserved": not any(
            blocker == "exact_candidate_inventory_membership_regressed"
            for blocker in blockers
        ),
        "gpu_or_vast_resource_used": False,
        "robot_motion_performed": False,
        "raw_prompts_recorded": False,
        "raw_secret_values_recorded": False,
        "blockers": blockers,
        "report_digest": "",
    }
    report["report_digest"] = canonical_digest(report, digest_field="report_digest")
    write_json(output_root / "task_evaluation_robot_placement_cache_canary.v1.json", report)
    if blockers:
        raise RobotPlacementCacheCanaryError(
            "robot_placement_cache_canary_failed:" + ",".join(blockers)
        )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--api-key-file")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--max-total-cost-usd", type=float, default=2.0)
    parser.add_argument("--webapp-endpoint")
    parser.add_argument("--require-webapp-sync", action="store_true")
    args = parser.parse_args(argv)
    run_production_shape_canary(
        output_dir=args.output_dir,
        api_key_file=args.api_key_file,
        source_commit=args.source_commit,
        max_total_cost_usd=args.max_total_cost_usd,
        webapp_endpoint=args.webapp_endpoint,
        require_webapp_sync=args.require_webapp_sync,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "EXPECTED_CALL_COUNT",
    "RobotPlacementCacheCanaryError",
    "SCHEMA_VERSION",
    "SEQUENCE_COUNT",
    "run_production_shape_canary",
]
