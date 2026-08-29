"""Run the bounded production robot-placement agent on exact scene geometry."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import write_json
from .task_evaluation_robot_placement_agent import (
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
from .task_evaluation_supervisor.agents_sdk import OpenAIAgentsSDKInvoker


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
    candidates = enumerate_robot_placement_geometry_candidates(
        index=index,
        target_position_world_m=target_position_world_m,
        robot_id=robot_id,
        maximum_candidates=candidate_inventory_cap,
    )
    if not candidates:
        raise ValueError("robot_placement_geometry_candidate_inventory_empty")
    overview_images = [
        _image_record(path, label=f"site_overview_{index_value:02d}")
        for index_value, path in enumerate(overview_image_paths)
    ]
    seed_previews = render_robot_placement_geometry_previews(
        index=index,
        proposal=candidates[0],
        target_position_world_m=target_position_world_m,
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
        )

    def renderer(proposal: Mapping[str, Any], round_index: int):
        images = render_robot_placement_geometry_previews(
            index=index,
            proposal=proposal,
            target_position_world_m=target_position_world_m,
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

    invoker = OpenAIAgentsSDKInvoker(
        robot_placement_agents_sdk_config(
            max_inference_cost_usd=max_inference_cost_usd,
            allow_live_invocation=allow_live_invocation,
            tracing_disabled=tracing_disabled,
        )
    )
    receipt = run_task_evaluation_robot_placement_agent(
        invoker=invoker,
        run_id=run_id,
        scene_binding=scene_binding,
        task_binding=task_binding,
        scene_context={
            "geometry_summary": summary,
            "deterministic_geometry_passing_candidate_inventory": candidates,
            "candidate_inventory_is_advisory": True,
        },
        task_context={
            "target_position_world_m": [float(value) for value in target_position_world_m],
            "robot_id": robot_id,
        },
        overview_images=overview_images,
        validate_candidate=validator,
        render_candidate=renderer,
        max_rounds=max_rounds,
        max_input_tokens=max_input_tokens,
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
    args = parser.parse_args(argv)
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
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt["status"] == "accepted" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main", "run_robot_placement_cli"]
