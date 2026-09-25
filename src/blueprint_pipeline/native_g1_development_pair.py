"""Run two G1 policies on one sealed task/site as a development comparison.

The existing single-candidate worker still owns Isaac, policy inference,
scoring, media, and teardown. This wrapper binds two requests to one scene and
objective, then preserves each terminal receipt. It cannot qualify a ranking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .native_g1_development_worker import (
    PATH_FIELDS,
    RESULT_FILENAME,
    _request,
    run_g1_development_worker,
)
from .native_g1_navigation_goal import validate_g1_navigation_goal_authority
from .native_g1_shared_scene_episode import G1_BOX_CANDIDATES, G1_NAVIGATION_CANDIDATES


SCHEMA = "native_g1_development_pair.v1"
EPISODE_FILENAME = "native_g1_built_scene_policy_episode.v1.json"
TRACE_FILENAME = "native_g1_shared_scene_episode_trace.v1.json"
PAIR_ORDER = (
    "humanoidarena_dp_g1_dex3_sonic",
    "humanoidarena_pi05_g1_dex3_sonic",
    "humanoidarena_dp_g1_dex3_sonic_vision_navi",
    "humanoidarena_pi05_g1_dex3_sonic_vision_navi",
)
CANDIDATE_FIELDS = frozenset({
    "candidate_id", "rights_review", "request_digest",
})


def _sealed_json(path: Path) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("g1_pair_request_path_invalid")
    value = json.loads(path.read_text(encoding="utf-8"))
    return _request(value)


def validate_g1_development_pair(request_paths: Sequence[Path]) -> dict[str, Any]:
    """Check the shared scene and objective without touching a GPU or output."""

    if len(request_paths) != 2:
        raise ValueError("g1_pair_exactly_two_requests_required")
    requests = [_sealed_json(path) for path in request_paths]
    by_candidate = {request["candidate_id"]: request for request in requests}
    candidate_ids = tuple(candidate for candidate in PAIR_ORDER if candidate in by_candidate)
    if (
        len(by_candidate) != 2
        or len(candidate_ids) != 2
        or not (
            set(candidate_ids) == G1_BOX_CANDIDATES
            or set(candidate_ids) == G1_NAVIGATION_CANDIDATES
        )
    ):
        raise ValueError("g1_pair_objective_or_candidates_mismatch")
    reference = {key: value for key, value in requests[0].items() if key not in CANDIDATE_FIELDS}
    if any(
        {key: value for key, value in request.items() if key not in CANDIDATE_FIELDS} != reference
        for request in requests[1:]
    ):
        raise ValueError("g1_pair_shared_runtime_or_scene_mismatch")
    bundle_root = Path(reference["bundle_root"])
    scene_path = bundle_root / "native_task_arena_scene_plan.v1.json"
    if (
        not bundle_root.is_absolute() or bundle_root.is_symlink()
        or bundle_root.resolve() != bundle_root
        or not bundle_root.is_dir() or scene_path.is_symlink() or not scene_path.is_file()
    ):
        raise ValueError("g1_pair_scene_packet_missing")
    plan = json.loads(scene_path.read_text(encoding="utf-8"))
    if (
        not isinstance(plan, dict)
        or plan.get("plan_digest") != canonical_digest(plan, digest_field="plan_digest")
        or (plan.get("robot") or {}).get("robot_id") != "unitree_g1"
        or plan.get("task_kind") != "rigid_pick_place"
        or not isinstance(plan.get("scene_id"), str) or not plan["scene_id"].strip()
        or not isinstance(plan.get("task_id"), str) or not plan["task_id"].strip()
    ):
        raise ValueError("g1_pair_scene_plan_invalid")
    for request in requests:
        rights = request.get("rights_review")
        if (
            not isinstance(rights, dict)
            or rights.get("candidate_id") != request["candidate_id"]
            or rights.get("scene_plan_digest") != plan["plan_digest"]
            or rights.get("rights_review_digest")
            != canonical_digest(rights, digest_field="rights_review_digest")
        ):
            raise ValueError("g1_pair_rights_review_invalid")
    objective = "g1_navigation_goal" if set(candidate_ids) == G1_NAVIGATION_CANDIDATES else "task_success"
    goal_authority_digest = None
    if objective == "g1_navigation_goal":
        authorities = [
            validate_g1_navigation_goal_authority(
                request.get("navigation_goal_authority"), plan=plan
            )
            for request in requests
        ]
        if authorities[0] != authorities[1]:
            raise ValueError("g1_pair_navigation_authority_mismatch")
        goal_authority_digest = authorities[0]["authority_digest"]
    return {
        "schema_version": SCHEMA,
        "status": "validated_not_executed",
        "scene_id": plan["scene_id"],
        "task_id": plan["task_id"],
        "scene_plan_digest": plan["plan_digest"],
        "objective_id": objective,
        "candidate_ids": list(candidate_ids),
        "request_digests": [by_candidate[candidate]["request_digest"] for candidate in candidate_ids],
        "navigation_goal_authority_digest": goal_authority_digest,
        "ranking_eligible": False,
        "physical_outcome_claimed": False,
    }


def _read_result(
    path: Path, *, candidate_id: str, scene_plan_digest: str, request_digest: str
) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("candidate_id") != candidate_id
        or value.get("request_digest") != request_digest
        or value.get("status") not in {"completed_development_only", "blocked"}
        or value.get("scene_plan_digest") not in {scene_plan_digest, None}
        or (value.get("status") == "completed_development_only"
            and value.get("scene_plan_digest") != scene_plan_digest)
        or value.get("ranking_eligible") is not False
        or value.get("physical_outcome_claimed") is not False
        or value.get("result_digest") != canonical_digest(value, digest_field="result_digest")
    ):
        raise ValueError("g1_pair_worker_receipt_invalid")
    return value


def _score_from_episode(
    path: Path, *, worker: Mapping[str, Any], objective_id: str
) -> dict[str, Any]:
    episode = json.loads(path.read_text(encoding="utf-8"))
    score = episode.get("score")
    if (
        episode.get("result_digest") != canonical_digest(episode, digest_field="result_digest")
        or episode.get("result_digest") != (worker.get("supervised_episode") or {}).get("episode_result_digest")
        or episode.get("candidate_id") != worker.get("candidate_id")
        or episode.get("scene_plan_digest") != worker.get("scene_plan_digest")
        or episode.get("status") != "development_only_scored_episode"
        or episode.get("evaluation_task_kind") != (
            "g1_navigation_goal" if objective_id == "g1_navigation_goal" else "rigid_pick_place"
        )
        or episode.get("ranking_eligible") is not False
        or episode.get("physical_outcome_claimed") is not False
        or not isinstance(score, Mapping)
        or score.get("status") != "scored"
        or not isinstance(score.get("outcome"), str)
        or not score["outcome"]
        or score.get("score_digest") != canonical_digest(score, digest_field="score_digest")
    ):
        raise ValueError("g1_pair_episode_or_score_receipt_invalid")
    return {
        "episode_result_digest": episode["result_digest"],
        "score_digest": score["score_digest"],
        "outcome": score.get("outcome"),
    }


def _verified_review_media(
    episode_path: Path, *, episode: Mapping[str, Any], pair_root: Path
) -> dict[str, Any]:
    """Index exact derived videos for review, without promoting their claim."""

    episode_root = episode_path.parent
    if episode.get("trace_relative_path") != TRACE_FILENAME:
        raise ValueError("g1_pair_trace_path_invalid")
    trace_path = episode_root / TRACE_FILENAME
    if trace_path.is_symlink() or not trace_path.is_file():
        raise ValueError("g1_pair_trace_missing")
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    visual = trace.get("visual_evidence")
    artifacts = trace.get("media_artifacts")
    if (
        trace.get("trace_digest") != episode.get("trace_digest")
        or trace.get("trace_digest") != canonical_digest(trace, digest_field="trace_digest")
        or trace.get("status") != "development_trace_recorded"
        or trace.get("candidate_id") != episode.get("candidate_id")
        or trace.get("scene_plan_digest") != episode.get("scene_plan_digest")
        or trace.get("claim_ceiling") != "simulator_only_unscored"
        or not isinstance(visual, Mapping)
        or visual.get("status") != "complete"
        or set(visual.get("videos") or {}) != {"head", "overview"}
        or set(visual.get("required_camera_ids") or []) != {"head", "overview"}
        or not isinstance(artifacts, list)
    ):
        raise ValueError("g1_pair_trace_or_media_invalid")

    def artifact_file(row: Mapping[str, Any]) -> dict[str, Any]:
        relative = row.get("relative_path")
        if not isinstance(relative, str):
            raise ValueError("g1_pair_media_path_invalid")
        path_part = PurePosixPath(relative)
        if (
            path_part.is_absolute() or ".." in path_part.parts
            or not path_part.parts or path_part.parts[0] != "media"
        ):
            raise ValueError("g1_pair_media_path_invalid")
        path = episode_root.joinpath(*path_part.parts)
        if (
            path.is_symlink() or path.resolve() != path or not path.is_file()
            or isinstance(row.get("size_bytes"), bool)
            or not isinstance(row.get("size_bytes"), int)
            or row["size_bytes"] <= 0
            or path.stat().st_size != row.get("size_bytes")
        ):
            raise ValueError("g1_pair_media_file_invalid")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if row.get("sha256") != "sha256:" + digest.hexdigest():
            raise ValueError("g1_pair_media_digest_mismatch")
        return {
            "relative_path": path.relative_to(pair_root).as_posix(),
            "sha256": row["sha256"],
            "size_bytes": row["size_bytes"],
        }

    manifests = [row for row in artifacts if isinstance(row, Mapping)
                 and row.get("role") == "multicamera_observation_frame_manifest"]
    videos = [row for row in artifacts if isinstance(row, Mapping)
              and row.get("role") == "camera_review_video"]
    if len(manifests) != 1 or len(videos) != 2:
        raise ValueError("g1_pair_media_artifacts_incomplete")
    manifest_ref = artifact_file(manifests[0])
    manifest = json.loads((pair_root / manifest_ref["relative_path"]).read_text(encoding="utf-8"))
    identity = manifest.get("identity")
    if (
        manifest.get("frame_manifest_digest") != visual.get("frame_manifest_digest")
        or manifest.get("frame_manifest_digest")
        != canonical_digest(manifest, digest_field="frame_manifest_digest")
        or not isinstance(identity, Mapping)
        or identity.get("candidate_id") != episode.get("candidate_id")
        or identity.get("scene_plan_digest") != episode.get("scene_plan_digest")
    ):
        raise ValueError("g1_pair_frame_manifest_invalid")
    by_camera = {row.get("camera_id"): row for row in videos}
    if set(by_camera) != {"head", "overview"}:
        raise ValueError("g1_pair_review_cameras_invalid")
    review_videos = {}
    for camera_id in ("head", "overview"):
        row = by_camera[camera_id]
        video = visual["videos"][camera_id]
        if (
            row.get("media_type") != "video/mp4"
            or row.get("relative_path") != video.get("relative_path")
            or row.get("sha256") != video.get("sha256")
            or row.get("size_bytes") != video.get("size_bytes")
            or video.get("derived_from_frame_manifest_digest") != manifest["frame_manifest_digest"]
        ):
            raise ValueError("g1_pair_review_video_binding_invalid")
        review_videos[camera_id] = artifact_file(row)
    return {
        "trace_digest": trace["trace_digest"],
        "frame_manifest_digest": manifest["frame_manifest_digest"],
        "frame_manifest": manifest_ref,
        "review_videos": review_videos,
        "derived_videos_are_human_review_convenience": True,
        "public_redistribution_authorized": False,
    }


def run_g1_development_pair(
    *,
    request_paths: Sequence[Path],
    output_dir: Path,
    mode: str = "local",
    source_receipt_path: Path | None = None,
    source_packet_path: Path | None = None,
    policy_runtime_root: Path | None = None,
    local_runner: Callable[..., dict[str, Any]] = run_g1_development_worker,
) -> dict[str, Any]:
    """Execute in catalog order; stop on an infrastructure block to limit spend."""

    pair = validate_g1_development_pair(request_paths)
    if mode not in {"local", "container"} or (mode == "container" and (
        sys.platform != "linux" or source_receipt_path is None
        or source_packet_path is None or policy_runtime_root is None
    )):
        raise ValueError("g1_pair_execution_mode_invalid")
    output = Path(output_dir).expanduser()
    requests = [_sealed_json(path) for path in request_paths]
    protected = [
        *(Path(path) for path in request_paths),
        *(Path(request[field]).expanduser() for request in requests for field in PATH_FIELDS),
    ]
    if (
        not output.is_absolute() or output.exists() or output.is_symlink()
        or output.resolve() != output
        or any(
            output.resolve().is_relative_to(path.resolve())
            or path.resolve().is_relative_to(output.resolve())
            for path in protected
        )
        or any(char in str(output) for char in ",\n\r")
    ):
        raise ValueError("g1_pair_output_directory_invalid")
    output.mkdir(parents=True)
    by_candidate = {request["candidate_id"]: (path, request)
                    for path, request in zip(request_paths, requests, strict=True)}
    attempts: list[dict[str, Any]] = []
    for candidate_id in pair["candidate_ids"]:
        request_path, request = by_candidate[candidate_id]
        worker_request_digest = request["request_digest"]
        attempt_root = output / candidate_id
        verified: dict[str, Any] | None = None
        result_path: Path | None = None
        score: dict[str, Any] | None = None
        review_media: dict[str, Any] | None = None
        try:
            if mode == "local":
                worker = local_runner(request=request, output_dir=attempt_root)
                result_path = attempt_root / RESULT_FILENAME
                episode_path = attempt_root / "episode" / EPISODE_FILENAME
            else:
                from .native_g1_container_run import prepare_g1_container_run
                from .native_g1_container_host import record_g1_container_host

                plan = prepare_g1_container_run(
                    request_path=request_path,
                    source_receipt_path=source_receipt_path,
                    source_packet_path=source_packet_path,
                    policy_runtime_root=policy_runtime_root,
                    output_dir=attempt_root,
                    repo_root=Path(__file__).resolve().parents[2],
                )
                worker_request_digest = plan["container_request_digest"]
                record_g1_container_host(output_dir=attempt_root)
                with (attempt_root / "container.log").open("x", encoding="utf-8") as stream:
                    process = subprocess.run(
                        plan["command"], stdout=stream, stderr=subprocess.STDOUT,
                        check=False,
                    )
                result_path = attempt_root / "results/episode" / RESULT_FILENAME
                episode_path = attempt_root / "results/episode/episode" / EPISODE_FILENAME
                worker = _read_result(
                    result_path, candidate_id=candidate_id,
                    scene_plan_digest=pair["scene_plan_digest"],
                    request_digest=worker_request_digest,
                )
                if process.returncode != (0 if worker["status"] == "completed_development_only" else 1):
                    raise ValueError("g1_pair_container_exit_or_worker_mismatch")
            verified = _read_result(
                result_path, candidate_id=candidate_id,
                scene_plan_digest=pair["scene_plan_digest"],
                request_digest=worker_request_digest,
            )
            if verified != worker:
                raise ValueError("g1_pair_worker_return_or_receipt_mismatch")
            if (verified["status"] == "completed_development_only"
                and pair["objective_id"] == "g1_navigation_goal") and (
                verified.get("navigation_goal_authority_digest")
                != pair["navigation_goal_authority_digest"]
            ):
                raise ValueError("g1_pair_navigation_authority_receipt_mismatch")
            score = (
                _score_from_episode(
                    episode_path, worker=verified, objective_id=pair["objective_id"]
                )
                if verified.get("status") == "completed_development_only" else None
            )
            review_media = (
                _verified_review_media(
                    episode_path, episode=json.loads(episode_path.read_text(encoding="utf-8")),
                    pair_root=output,
                )
                if score is not None else None
            )
            attempts.append({
                "candidate_id": candidate_id,
                "status": verified["status"],
                "worker_status": verified["status"],
                "worker_result_digest": verified["result_digest"],
                "worker_result_path": str(result_path),
                "score": score,
                "review_media": review_media,
                "blocker": verified.get("blocker"),
            })
        except Exception as exc:
            attempts.append({
                "candidate_id": candidate_id,
                "status": "blocked",
                "worker_result_digest": verified.get("result_digest") if verified else None,
                "worker_result_path": str(result_path) if verified and result_path else None,
                "worker_status": verified.get("status") if verified else None,
                "score": score,
                "review_media": review_media,
                "blocker": {"type": type(exc).__name__, "message": str(exc)},
            })
        if attempts[-1]["status"] != "completed_development_only":
            break
    result = {
        **pair,
        "status": "completed_development_only" if len(attempts) == 2 and all(
            attempt["status"] == "completed_development_only" for attempt in attempts
        ) else "blocked",
        "mode": mode,
        "attempts": attempts,
        "not_attempted_candidate_ids": pair["candidate_ids"][len(attempts):],
    }
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    (output / (SCHEMA + ".json")).write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--mode", choices=("local", "container"), default="local")
    parser.add_argument("--source-receipt", type=Path)
    parser.add_argument("--source-packet", type=Path)
    parser.add_argument("--policy-runtime-root", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(validate_g1_development_pair(args.request), indent=2, sort_keys=True))
        return 0
    if args.output_dir is None:
        parser.error("--execute requires --output-dir")
    result = run_g1_development_pair(
        request_paths=args.request,
        output_dir=args.output_dir,
        mode=args.mode,
        source_receipt_path=args.source_receipt,
        source_packet_path=args.source_packet,
        policy_runtime_root=args.policy_runtime_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "completed_development_only" else 1


if __name__ == "__main__":
    sys.exit(main())
