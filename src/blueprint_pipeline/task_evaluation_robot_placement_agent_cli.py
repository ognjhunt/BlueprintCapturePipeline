"""Run the bounded production robot-placement agent on exact scene geometry."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .task_evaluation_robot_placement_agent import (
    DETERMINISTIC_ROBOT_PLACEMENT_CLAIM_CEILING,
    DETERMINISTIC_ROBOT_PLACEMENT_MODEL,
    PlacementExecutor,
    _reject_infeasible_orientation_slew,
    robot_placement_agents_sdk_config,
    run_task_evaluation_robot_placement_agent,
)
from .task_evaluation_robot_placement_geometry import (
    build_robot_placement_geometry_index,
    enumerate_robot_placement_geometry_candidates,
    render_robot_placement_geometry_previews,
    summarize_robot_placement_geometry,
    validate_robot_placement_geometry_candidate,
)
from .task_evaluation_robot_placement_inventory import (
    build_candidate_inventory_checkpoint,
    validate_candidate_inventory_checkpoint,
)
from .task_evaluation_robot_placement_trajectory import (
    placement_trajectory_from_native_plan,
    placement_trajectory_from_native_result,
)
from .task_evaluation_robot_placement_warm_executor import (
    native_attempt_from_warm_result,
)
from .task_evaluation_supervisor.agents_sdk import OpenAIAgentsSDKInvoker
from .task_evaluation_supervisor.inference_reservations import (
    InferenceReservationAudit,
)


def _read_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"robot_placement_{label}_invalid") from exc
    if path.is_symlink() or not isinstance(value, Mapping):
        raise ValueError(f"robot_placement_{label}_invalid")
    return dict(value)


def _image_record(path: Path, *, label: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    payload = resolved.read_bytes()
    mime = mimetypes.guess_type(resolved.name)[0] or "image/png"
    if not mime.startswith("image/"):
        raise ValueError("robot_placement_overview_image_type_invalid")
    return {
        "label": label,
        "digest": "sha256:" + hashlib.sha256(payload).hexdigest(),
        "image_url": f"data:{mime};base64," + base64.b64encode(payload).decode("ascii"),
        "detail": "high",
    }


def _persist_images(
    images: Sequence[Mapping[str, Any]], *, output_dir: Path, prefix: str
) -> list[dict[str, Any]]:
    records = []
    for index, image in enumerate(images):
        image_url = str(image.get("image_url") or "")
        header, encoded = image_url.split(",", 1)
        if not header.startswith("data:image/png;base64"):
            raise ValueError("robot_placement_generated_preview_invalid")
        payload = base64.b64decode(encoded, validate=True)
        digest = "sha256:" + hashlib.sha256(payload).hexdigest()
        if digest != image.get("digest"):
            raise ValueError("robot_placement_generated_preview_digest_mismatch")
        path = output_dir / f"{prefix}-{index:02d}-{str(image.get('label') or 'view')}.png"
        path.write_bytes(payload)
        records.append(
            {
                "label": str(image.get("label") or path.stem),
                "digest": digest,
                "size_bytes": len(payload),
                "path": str(path),
                "image_url": image_url,
                "detail": str(image.get("detail") or "high"),
            }
        )
    return records


def run_robot_placement_cli(
    *,
    run_id: str,
    scene_collision_usd: Path,
    robot_asset_usd: Path,
    target_position_world_m: Sequence[float],
    scene_binding: Mapping[str, Any],
    task_binding: Mapping[str, Any],
    overview_image_paths: Sequence[Path],
    output_dir: Path,
    max_rounds: int,
    candidate_inventory_cap: int,
    max_input_tokens: int,
    max_inference_cost_usd: float,
    allow_live_invocation: bool,
    tracing_disabled: bool,
    robot_id: str = "franka_panda",
    execute_candidate: PlacementExecutor | None = None,
    task_trajectory: Mapping[str, Any] | None = None,
    prior_native_attempts: Sequence[Mapping[str, Any]] = (),
    candidate_inventory_checkpoint: Mapping[str, Any] | None = None,
    deterministic_selection: bool = False,
    record_inference_reservations: bool = False,
) -> dict[str, Any]:
    root = output_dir.expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("robot_placement_output_not_empty")
    root.mkdir(parents=True, exist_ok=True)
    preview_root = root / "previews"
    preview_root.mkdir()
    index = build_robot_placement_geometry_index(
        scene_collision_usd_path=scene_collision_usd,
        robot_asset_usd_path=robot_asset_usd,
    )
    summary = summarize_robot_placement_geometry(
        index,
        target_position_world_m=target_position_world_m,
        robot_id=robot_id,
    )
    trajectory_phases = list((task_trajectory or {}).get("phases", []))
    trajectory_waypoints = [
        list(row["position_world_m"])
        for row in trajectory_phases
    ]
    trajectory_phase_ids = [str(row["phase_id"]) for row in trajectory_phases]
    trajectory_orientations = [
        list(row["orientation_world_xyzw"])
        for row in trajectory_phases
    ]
    placement_worker_count = min(8, max(1, os.cpu_count() or 1))
    trajectory_digest = (task_trajectory or {}).get("trajectory_digest")
    geometry_summary_digest = str(summary.get("geometry_summary_digest") or "")
    if candidate_inventory_checkpoint is None:
        candidates = enumerate_robot_placement_geometry_candidates(
            index=index,
            target_position_world_m=target_position_world_m,
            robot_id=robot_id,
            maximum_candidates=candidate_inventory_cap,
            trajectory_waypoints_world_m=trajectory_waypoints,
            trajectory_phase_ids=trajectory_phase_ids,
            trajectory_orientations_world_xyzw=trajectory_orientations,
            geometry_worker_count=placement_worker_count,
            trajectory_worker_count=placement_worker_count,
        )
    else:
        candidates = validate_candidate_inventory_checkpoint(
            checkpoint=candidate_inventory_checkpoint,
            index=index,
            robot_id=robot_id,
            target_position_world_m=target_position_world_m,
            maximum_candidates=candidate_inventory_cap,
            trajectory_digest=trajectory_digest,
            geometry_summary_digest=geometry_summary_digest,
        )
    if not candidates:
        raise ValueError("robot_placement_geometry_candidate_inventory_empty")
    candidate_inventory_digest = canonical_digest(
        {"trajectory_digest": trajectory_digest, "candidates": candidates}
    )
    checkpoint = build_candidate_inventory_checkpoint(
        robot_id=robot_id,
        target_position_world_m=target_position_world_m,
        maximum_candidates=candidate_inventory_cap,
        trajectory_digest=trajectory_digest,
        geometry_summary_digest=geometry_summary_digest,
        candidates=candidates,
    )
    write_json(
        root / "task_evaluation_robot_placement_candidate_inventory.v1.json",
        checkpoint,
    )
    overview_images = [
        _image_record(path, label=f"site_overview_{index_value:02d}")
        for index_value, path in enumerate(overview_image_paths)
    ]
    seed_previews = render_robot_placement_geometry_previews(
        index=index,
        proposal=candidates[0],
        target_position_world_m=target_position_world_m,
        trajectory_waypoints_world_m=trajectory_waypoints,
    )
    overview_images.extend(
        _persist_images(seed_previews, output_dir=preview_root, prefix="geometry-overview")
    )
    artifact_records: list[dict[str, Any]] = [
        {
            key: value
            for key, value in image.items()
            if key in {"label", "digest", "size_bytes", "path"}
        }
        for image in overview_images
        if image.get("path")
    ]

    def validator(proposal: Mapping[str, Any]) -> Mapping[str, Any]:
        return validate_robot_placement_geometry_candidate(
            index=index,
            proposal=proposal,
            target_position_world_m=target_position_world_m,
            robot_id=robot_id,
            trajectory_waypoints_world_m=trajectory_waypoints,
            trajectory_phase_ids=trajectory_phase_ids,
            trajectory_orientations_world_xyzw=trajectory_orientations,
        )

    def renderer(proposal: Mapping[str, Any], round_index: int):
        images = render_robot_placement_geometry_previews(
            index=index,
            proposal=proposal,
            target_position_world_m=target_position_world_m,
            trajectory_waypoints_world_m=trajectory_waypoints,
        )
        persisted = _persist_images(
            images,
            output_dir=preview_root,
            prefix=f"candidate-{round_index:02d}",
        )
        artifact_records.extend(
            {
                key: value
                for key, value in image.items()
                if key in {"label", "digest", "size_bytes", "path"}
            }
            for image in persisted
        )
        return persisted

    if deterministic_selection:
        candidate = candidates[0]
        proposal = {
            "candidate_id": candidate["candidate_id"],
            "pose": candidate["pose"],
            "support_surface_id": candidate["support_surface_id"],
            "rationale": "Top-ranked exact member of the digest-bound CPU inventory.",
            "addressed_blockers": [],
            "uncertainty": "Native orientation, collision/contact, camera, and execution gates remain unresolved.",
        }
        gate = _reject_infeasible_orientation_slew(
            gate=validator(proposal),
            proposal=proposal,
            trajectory=task_trajectory,
            robot_id=robot_id,
            maximum_steps_per_phase=(
                task_trajectory.get("maximum_steps_per_phase")
                if task_trajectory is not None
                else None
            ),
        )
        if gate.get("status") != "passed":
            raise ValueError("robot_placement_deterministic_candidate_rejected")
        previews = renderer(proposal, 0)
        round_record = {
            "round_index": 0,
            "proposal": proposal,
            "proposal_provider": "local_cpu",
            "proposal_model": DETERMINISTIC_ROBOT_PLACEMENT_MODEL,
            "proposal_sdk_version": None,
            "proposal_usage": {"total_tokens": 0},
            "proposal_trace_id": None,
            "geometry_gate": gate,
            "visual_review": None,
            "preview_images": [
                {"label": row["label"], "digest": row["digest"]}
                for row in previews
            ],
            "native_attempt": None,
        }
        receipt = {
            "schema_version": "task_evaluation_robot_placement_receipt.v1",
            "status": "accepted",
            "run_id": run_id,
            "model": DETERMINISTIC_ROBOT_PLACEMENT_MODEL,
            "reasoning_effort": "none",
            "max_rounds": 1,
            "round_count": 1,
            "scene_binding_digest": canonical_digest(scene_binding),
            "task_binding_digest": canonical_digest(task_binding),
            "scene_context_digest": canonical_digest({
                "geometry_summary": summary,
                "deterministic_geometry_passing_candidate_inventory": candidates,
                "deterministic_geometry_passing_candidate_inventory_digest": candidate_inventory_digest,
                "deterministic_geometry_passing_candidate_inventory_trajectory_digest": trajectory_digest,
            }),
            "task_context_digest": canonical_digest({
                "target_position_world_m": [float(value) for value in target_position_world_m],
                "robot_id": robot_id,
                "native_trajectory": task_trajectory,
            }),
            "task_trajectory_digest": trajectory_digest,
            "candidate_inventory_digest": candidate_inventory_digest,
            "candidate_inventory_trajectory_digest": trajectory_digest,
            "overview_images": [
                {"label": row["label"], "digest": row["digest"]}
                for row in overview_images
            ],
            "prior_native_attempts": [],
            "prior_native_attempt_count": 0,
            "rounds": [round_record],
            "accepted_pose": proposal["pose"],
            "accepted_candidate_id": proposal["candidate_id"],
            "accepted_support_surface_id": proposal["support_surface_id"],
            "accepted_geometry_gate_digest": gate["geometry_gate_digest"],
            "native_agent_loop_enabled": False,
            "native_attempt_count": 0,
            "accepted_native_attempt_digest": None,
            "candidate_may_self_authorize": False,
            "physical_execution_authorized": False,
            "native_construction_required": True,
            "model_grades_controls": False,
            "selection_method": "deterministic_inventory_rank",
            "visual_review_completed": False,
            "native_camera_visibility_required": True,
            "claim_ceiling": DETERMINISTIC_ROBOT_PLACEMENT_CLAIM_CEILING,
            "receipt_digest": "",
        }
        receipt["receipt_digest"] = canonical_digest(
            receipt, digest_field="receipt_digest"
        )
        write_json(root / "task_evaluation_robot_placement_receipt.v1.json", receipt)
        write_json(
            root / "task_evaluation_robot_placement_artifact_index.v1.json",
            {
                "schema_version": "task_evaluation_robot_placement_artifact_index.v1",
                "run_id": run_id,
                "receipt_digest": receipt["receipt_digest"],
                "artifacts": artifact_records,
            },
        )
        return receipt

    invoker = OpenAIAgentsSDKInvoker(
        robot_placement_agents_sdk_config(
            max_inference_cost_usd=max_inference_cost_usd,
            allow_live_invocation=allow_live_invocation,
            tracing_disabled=tracing_disabled,
        )
    )
    audit = (
        InferenceReservationAudit(run_root=root, run_id=run_id)
        if record_inference_reservations
        else None
    )
    if audit is not None:
        invoker.configure_reservation_audit(
            record_reservation=audit.record_reservation,
            record_completion=audit.record_completion,
            restored_reserved_cost_usd=0.0,
        )
    try:
        receipt = run_task_evaluation_robot_placement_agent(
            invoker=invoker,
            run_id=run_id,
            scene_binding=scene_binding,
            task_binding=task_binding,
            scene_context={
                "geometry_summary": summary,
                "deterministic_geometry_passing_candidate_inventory": candidates,
                "deterministic_geometry_passing_candidate_inventory_digest": (
                    candidate_inventory_digest
                ),
                "deterministic_geometry_passing_candidate_inventory_trajectory_digest": (
                    trajectory_digest
                ),
                "candidate_inventory_is_advisory": False,
                "model_must_select_exact_inventory_member": True,
                "candidate_pose_or_support_mutation_allowed": False,
            },
            task_context={
                "target_position_world_m": [
                    float(value) for value in target_position_world_m
                ],
                "robot_id": robot_id,
            },
            overview_images=overview_images,
            validate_candidate=validator,
            render_candidate=renderer,
            execute_candidate=execute_candidate,
            task_trajectory=task_trajectory,
            prior_native_attempts=prior_native_attempts,
            max_rounds=max_rounds,
            max_input_tokens=max_input_tokens,
        )
    except Exception:
        if audit is not None:
            audit.write_manifest()
        raise
    if audit is not None:
        audit.write_manifest()
    write_json(root / "task_evaluation_robot_placement_receipt.v1.json", receipt)
    write_json(
        root / "task_evaluation_robot_placement_artifact_index.v1.json",
        {
            "schema_version": "task_evaluation_robot_placement_artifact_index.v1",
            "run_id": run_id,
            "receipt_digest": receipt["receipt_digest"],
            "artifacts": artifact_records,
        },
    )
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--scene-collision-usd", required=True, type=Path)
    parser.add_argument("--robot-asset-usd", required=True, type=Path)
    parser.add_argument("--target-position-world-m", required=True, nargs=3, type=float)
    parser.add_argument("--scene-binding", required=True, type=Path)
    parser.add_argument("--task-binding", required=True, type=Path)
    parser.add_argument("--overview-image", action="append", default=[], type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--robot-id", default="franka_panda")
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--candidate-inventory-cap", type=int, default=12)
    parser.add_argument("--max-input-tokens", type=int, default=120_000)
    parser.add_argument("--max-inference-cost-usd", required=True, type=float)
    parser.add_argument("--allow-live-invocation", action="store_true")
    parser.add_argument("--disable-tracing", action="store_true")
    parser.add_argument("--record-inference-reservations", action="store_true")
    trajectory_source = parser.add_mutually_exclusive_group()
    trajectory_source.add_argument(
        "--native-trajectory-plan",
        type=Path,
        help=(
            "Exact digest-bound native rigid construction phase plan. Required "
            "when --native-loop-config enables iterative warm Isaac feedback."
        ),
    )
    trajectory_source.add_argument(
        "--native-construction-result",
        type=Path,
        help=(
            "Prior digest-bound native construction result whose exact phase plan "
            "becomes the immutable placement trajectory."
        ),
    )
    parser.add_argument(
        "--native-loop-config",
        type=Path,
        help=(
            "File-backed warm Arena executor configuration. Every accepted "
            "candidate is executed through the canonical paid allocator."
        ),
    )
    parser.add_argument(
        "--prior-native-construction-result",
        type=Path,
        help=(
            "Prior rejected native construction result whose exact failure metrics "
            "and selected frames seed the first proposal."
        ),
    )
    parser.add_argument(
        "--prior-native-allocator-result",
        type=Path,
        help=(
            "Canonical allocator result paired with --prior-native-construction-result."
        ),
    )
    parser.add_argument(
        "--candidate-inventory-checkpoint",
        type=Path,
        help=(
            "Prior complete digest-bound CPU candidate inventory. Its exact input "
            "bindings and every retained geometry gate are revalidated before reuse."
        ),
    )
    args = parser.parse_args(argv)
    task_trajectory = None
    if args.native_trajectory_plan:
        task_trajectory = placement_trajectory_from_native_plan(
            _read_mapping(args.native_trajectory_plan, label="native_trajectory")
        )
    elif args.native_construction_result:
        task_trajectory = placement_trajectory_from_native_result(
            _read_mapping(
                args.native_construction_result, label="native_construction_result"
            )
        )
    execute_candidate = None
    if args.native_loop_config:
        if task_trajectory is None:
            parser.error(
                "--native-loop-config requires --native-trajectory-plan or "
                "--native-construction-result"
            )
        from .task_evaluation_robot_placement_warm_executor import (
            WarmNativePlacementExecutor,
        )

        execute_candidate = WarmNativePlacementExecutor(
            config=_read_mapping(args.native_loop_config, label="native_loop_config"),
            task_trajectory=task_trajectory,
            output_root=args.output_dir.expanduser().resolve() / "native-rounds",
        )
    if bool(args.prior_native_construction_result) != bool(
        args.prior_native_allocator_result
    ):
        parser.error(
            "--prior-native-construction-result and "
            "--prior-native-allocator-result are required together"
        )
    prior_native_attempts = []
    if args.prior_native_construction_result:
        prior_native_attempts.append(
            native_attempt_from_warm_result(
                allocator_result=_read_mapping(
                    args.prior_native_allocator_result,
                    label="prior_native_allocator_result",
                ),
                native_result_path=args.prior_native_construction_result,
            )
        )
    receipt = run_robot_placement_cli(
        run_id=args.run_id,
        scene_collision_usd=args.scene_collision_usd,
        robot_asset_usd=args.robot_asset_usd,
        target_position_world_m=args.target_position_world_m,
        scene_binding=_read_mapping(args.scene_binding, label="scene_binding"),
        task_binding=_read_mapping(args.task_binding, label="task_binding"),
        overview_image_paths=args.overview_image,
        output_dir=args.output_dir,
        max_rounds=args.max_rounds,
        candidate_inventory_cap=args.candidate_inventory_cap,
        max_input_tokens=args.max_input_tokens,
        max_inference_cost_usd=args.max_inference_cost_usd,
        allow_live_invocation=args.allow_live_invocation,
        tracing_disabled=args.disable_tracing,
        robot_id=args.robot_id,
        execute_candidate=execute_candidate,
        task_trajectory=task_trajectory,
        prior_native_attempts=prior_native_attempts,
        candidate_inventory_checkpoint=(
            _read_mapping(
                args.candidate_inventory_checkpoint,
                label="candidate_inventory_checkpoint",
            )
            if args.candidate_inventory_checkpoint
            else None
        ),
        record_inference_reservations=args.record_inference_reservations,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt["status"] == "accepted" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main", "run_robot_placement_cli"]
