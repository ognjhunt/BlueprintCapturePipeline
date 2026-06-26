"""Write default owner-GPU smoke-policy artifacts from an owner simulator command."""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .common import read_json, utc_now_iso, write_json
from .owner_gpu_proof_runner import OWNER_SIM_ROBOT_POV_SCHEMA_VERSION


DEFAULT_POLICY_TRACE_SCHEMA_VERSION = "owner_default_policy_execution_trace.v1"
DEFAULT_SMOKE_ARTIFACT_RESULT_SCHEMA_VERSION = "owner_default_smoke_artifact_result.v1"


def _string(value: Any) -> str:
    return str(value or "").strip()


def _path_from(value: str | Path | None, *, label: str) -> Path:
    text = _string(value)
    if not text:
        raise ValueError(f"{label} is required")
    return Path(text).expanduser()


def _read_existing_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return read_json(path)


def _artifact_key(item: Mapping[str, Any]) -> tuple[str, str]:
    return (_string(item.get("kind")), _string(item.get("path") or item.get("uri")))


def _merge_artifacts(
    existing: Sequence[Any],
    additions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in [*existing, *additions]:
        if not isinstance(item, Mapping):
            continue
        normalized = dict(item)
        key = _artifact_key(normalized)
        if key in seen:
            continue
        seen.add(key)
        merged.append(normalized)
    return merged


def write_default_smoke_artifacts(
    *,
    policy_trace_path: str | Path,
    sim_robot_pov_evidence_path: str | Path,
    artifact_manifest_path: str | Path,
    target: str = "walk_to_target_pose",
    simulator: str = "",
    camera: str = "front_rgbd",
    sim_pov_frames: Sequence[str] | None = None,
    sim_pov_videos: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Write policy and simulator POV artifacts without claiming real-robot proof."""

    policy_path = _path_from(policy_trace_path, label="policy_trace_path")
    pov_path = _path_from(sim_robot_pov_evidence_path, label="sim_robot_pov_evidence_path")
    manifest_path = _path_from(artifact_manifest_path, label="artifact_manifest_path")
    frames = [_string(frame) for frame in (sim_pov_frames or []) if _string(frame)]
    videos = [_string(video) for video in (sim_pov_videos or []) if _string(video)]
    if not frames and not videos:
        raise ValueError("at least one simulator POV frame or video path is required")

    generated_at = utc_now_iso()
    policy_target = _string(target) or "walk_to_target_pose"
    simulator_name = _string(simulator) or "owner_gpu_simulator"
    camera_name = _string(camera) or "front_rgbd"

    policy_trace = {
        "schema_version": DEFAULT_POLICY_TRACE_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "completed",
        "simulator": simulator_name,
        "policy_id": "blueprint_default_walk_to_target_smoke_policy",
        "policy_kind": "walk_to_target",
        "target": policy_target,
        "default_policy_executed": True,
        "policy_execution_completed": True,
        "actions": [
            {
                "id": "default_walk_to_target_0001",
                "name": "walk_to_target",
                "target": policy_target,
                "status": "completed",
                "evidence_scope": "simulator_only",
            }
        ],
        "claim_boundary": {
            "default_sim_policy_execution_proven": True,
            "live_robot_policy_execution_proven": False,
            "real_robot_pov_evidence_proven": False,
            "rank_fidelity_result_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    }
    write_json(policy_path, policy_trace)

    sim_pov = {
        "schema_version": OWNER_SIM_ROBOT_POV_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "complete",
        "simulator": simulator_name,
        "sim_robot_pov_captured": True,
        "camera": camera_name,
        "frames": [{"camera": camera_name, "path": frame} for frame in frames],
        "videos": [{"camera": camera_name, "path": video} for video in videos],
        "claim_boundary": {
            "simulator_robot_pov_evidence_proven": True,
            "real_robot_pov_evidence_proven": False,
            "rank_fidelity_result_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    }
    write_json(pov_path, sim_pov)

    existing_manifest = _read_existing_manifest(manifest_path)
    existing_artifacts = existing_manifest.get("artifacts")
    if not isinstance(existing_artifacts, Sequence) or isinstance(existing_artifacts, (str, bytes)):
        existing_artifacts = []
    artifact_manifest = {
        **existing_manifest,
        "schema_version": existing_manifest.get(
            "schema_version",
            "owner_gpu_artifact_manifest.v1",
        ),
        "generated_at": existing_manifest.get("generated_at") or generated_at,
        "updated_at": generated_at,
        "status": "complete",
        "artifact_manifest_complete": True,
        "artifacts": _merge_artifacts(
            existing_artifacts,
            [
                {"kind": "policy_trace", "path": str(policy_path), "required": True},
                {"kind": "sim_robot_pov", "path": str(pov_path), "required": True},
                *[
                    {
                        "kind": "sim_robot_pov_frame",
                        "path": frame,
                        "camera": camera_name,
                        "required": True,
                    }
                    for frame in frames
                ],
                *[
                    {
                        "kind": "sim_robot_pov_video",
                        "path": video,
                        "camera": camera_name,
                        "required": True,
                    }
                    for video in videos
                ],
            ],
        ),
    }
    write_json(manifest_path, artifact_manifest)

    return {
        "schema_version": DEFAULT_SMOKE_ARTIFACT_RESULT_SCHEMA_VERSION,
        "status": "complete",
        "policy_trace_path": str(policy_path),
        "sim_robot_pov_evidence_path": str(pov_path),
        "artifact_manifest_path": str(manifest_path),
        "sim_robot_pov_frame_count": len(frames),
        "sim_robot_pov_video_count": len(videos),
        "claim_boundary": {
            "default_sim_policy_execution_proven": True,
            "simulator_robot_pov_evidence_proven": True,
            "real_robot_pov_evidence_proven": False,
            "rank_fidelity_result_proven": False,
        },
    }


def _env_default(*names: str) -> str:
    for name in names:
        value = _string(os.environ.get(name))
        if value:
            return value
    return ""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Write the default walk-to-target policy trace and simulator robot POV "
            "manifest required by blueprint-run-owner-gpu-proof."
        )
    )
    parser.add_argument(
        "--policy-trace",
        default=_env_default("BLUEPRINT_POLICY_EXECUTION_TRACE", "BLUEPRINT_ACTION_OR_POLICY_TRACE"),
        help="Output JSON path for the default policy execution trace.",
    )
    parser.add_argument(
        "--sim-robot-pov-evidence",
        default=_env_default("BLUEPRINT_SIM_ROBOT_POV_EVIDENCE"),
        help="Output JSON path for simulator robot POV evidence.",
    )
    parser.add_argument(
        "--artifact-manifest",
        default=_env_default("BLUEPRINT_ARTIFACT_MANIFEST"),
        help="Output JSON path for the owner artifact manifest.",
    )
    parser.add_argument(
        "--target",
        default=_env_default("BLUEPRINT_DEFAULT_SMOKE_POLICY_TARGET") or "walk_to_target_pose",
        help="Target label or pose id used by the default walk-to-target smoke policy.",
    )
    parser.add_argument(
        "--simulator",
        default=_env_default("BLUEPRINT_OWNER_SIMULATOR") or "owner_gpu_simulator",
    )
    parser.add_argument("--camera", default="front_rgbd")
    parser.add_argument("--sim-pov-frame", action="append", default=[])
    parser.add_argument("--sim-pov-video", action="append", default=[])
    args = parser.parse_args(argv)

    try:
        result = write_default_smoke_artifacts(
            policy_trace_path=args.policy_trace,
            sim_robot_pov_evidence_path=args.sim_robot_pov_evidence,
            artifact_manifest_path=args.artifact_manifest,
            target=args.target,
            simulator=args.simulator,
            camera=args.camera,
            sim_pov_frames=args.sim_pov_frame,
            sim_pov_videos=args.sim_pov_video,
        )
    except ValueError as exc:
        parser.error(str(exc))

    print(f"[owner-default-smoke-artifacts] status={result['status']}")
    print(f"[owner-default-smoke-artifacts] policy_trace={result['policy_trace_path']}")
    print(
        "[owner-default-smoke-artifacts] sim_robot_pov_evidence="
        f"{result['sim_robot_pov_evidence_path']}"
    )
    print(f"[owner-default-smoke-artifacts] artifact_manifest={result['artifact_manifest_path']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
