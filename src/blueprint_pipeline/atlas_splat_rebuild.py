"""Rebuild a ready preserved scene's refined splat under the current input policy, beside the original.

Owner approval 2026-11-08, about 20:45Z, in the library-aisle session: "fix the
input first, then do the rebuild". capture-5b067498's refined splat was bought
from views whose poses and depth disagree (``atlas_splat_input_quality``). A
refinement needs a new job root (``atlas_refinement_requires_new_job``), and
retained data is never deleted. So a rebuild:

- starts only from the operator's existing ``selected-handoff --atlas-retry``.
  ``--atlas-status`` first shows ``job.splat_rebuild`` (why the job qualifies and
  what it should cost), and its digest is the one reviewed. Under the job lock,
  ``atlas_preparation_job_control.retry_preparation_job`` writes the request and
  reopens the job.
- runs on the next controller tick in ``pipeline/atlas/splat-rebuild/<hex>/``,
  inside the job's own Atlas allowance and paid admission: a new selection under
  the current policy (``prepare_video_inputs``), a new source pose, and then the
  refined splat and the mesh (``_preserved_scene(rebuild=True)``). The splat is
  bought only if World Labs accepted pose refinement and enough views agree.
  There is no completion and no new task review.
- publishes its result as the job's ``result.json``. The parent's result stays
  in the rebuild root as ``parent_result.json``, and every earlier pose, splat,
  mesh and bundle stays where it is.
- on a stop (refinement not accepted, too few consistent views, not enough money,
  the draft's fusion refusing the new pose) records why and hands the parent
  result back. The job is then ready again on its original scene, and nothing
  more is bought.

Job control imports this module to decide eligibility, so it reads retained
records only and imports no provider or reconstruction code; the pipeline runs
the rebuild (``website_atlas_pipeline._run_splat_rebuild``).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .common import write_json
from .decision_evidence_contracts import canonical_digest

REQUEST_FILE = "splat_rebuild_request.json"
ROOT = "splat-rebuild"
SCHEMA = "atlas_splat_rebuild_request.v1"
OUTCOME_FILE = "outcome.json"
#: Typed refusals of the new pose's own draft that stop the rebuild rather than the job.
STOP_PREFIXES = ("atlas_depth_splat_", "atlas_splat_rebuild_")


def _read(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("atlas_splat_rebuild_record_missing")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("atlas_splat_rebuild_record_invalid")
    return value


def _inside(job_dir: Path, relative: str) -> Path:
    path = job_dir / relative
    if path.is_symlink() or not path.resolve().is_relative_to(job_dir.resolve()):
        raise ValueError("atlas_splat_rebuild_path_invalid")
    return path


def rebuild_root(job_dir: Path, parent_result_digest: str) -> Path:
    return Path(job_dir) / ROOT / parent_result_digest[7:23]


def rebuild_eligibility(job_dir: Path, job: Mapping[str, Any]) -> dict[str, Any] | None:
    """Why a ready job's refined splat qualifies for one rebuild, and its cost; None when it does not.

    From retained records only. It qualifies when the scene was kept as filmed
    with Atlas's refined splat of the captured views and no generated geometry,
    that splat was bought before this release recorded an input selection, no
    rebuild was requested yet, and its own pose shows the cause: World Labs did not
    accept pose refinement, or a view disagrees with its measured neighbours.
    """
    from .atlas_splat_input_quality import SELECTION_FILE, consistent_frames
    from .website_atlas_preview_store import SCHEMA as JOB_SCHEMA

    job_dir = Path(job_dir)
    if job.get("schema_version") != JOB_SCHEMA or job.get("state") != "ready":
        return None
    try:
        result = _read(job_dir / "result.json")
    except ValueError:
        return None
    appearance = result.get("appearance") or {}
    if (result.get("digest") != job.get("result_digest")
            or result.get("digest") != canonical_digest(result, digest_field="digest")
            or result.get("captured_scene_preserved") is not True
            or appearance.get("basis") != "atlas_refined_splat"
            or result.get("generated_geometry_admitted") is not False
            or result.get("atlas_route") == "geometry_only" or result.get("observed_scene_only")
            or result.get("splat_rebuild") or (job_dir / REQUEST_FILE).exists()
            or (job_dir / SELECTION_FILE).exists()):
        return None
    try:
        pose_root = _inside(job_dir, (result.get("source_pose") or {}).get("pose_relative_path") or "missing")
        pose = _read(pose_root / "atlas_posed_rgbd_result.json")
        draft = _read(job_dir / "splat/atlas_depth_splat_manifest.json")
        policy = _read(job_dir / "configuration.json").get("policy") or {}
        posed = [row["frame_id"] for row in pose["operation_receipt"]["input_usage"]["frames"]]
        selected = _read(job_dir / "pose/atlas_view_selection.json").get("frames") or []
        prices = (float(policy["pose_input_cost_usd"]), float(policy.get("maximum_review_cost_usd", 0)),
                  float(policy["refined_splat_cost_usd"]) + float(policy.get("mesh_export_cost_usd", 0)))
    except (OSError, ValueError, KeyError, TypeError):
        return None  # Not the retained layout this rebuild knows how to rebuild.
    reasons, dropped = [], []
    if pose["operation"]["response"].get("poseRefinementAccepted") is not True:
        reasons.append("pose_refinement_not_accepted")
    if draft.get("pose_result_digest") == pose.get("digest"):
        _, dropped, _ = consistent_frames(posed, (draft.get("cross_view_agreement") or {}).get("pairs") or [])
        if dropped:
            reasons.append("views_disagree_with_measured_neighbours")
    if not reasons:
        return None
    from .atlas_refined_splat_forms import SPLAT_FRAME_TARGET, SPLAT_POSE_INPUT_CAPACITY
    task_views = max(0, len(posed) - len(selected))
    pose_cost, review_cost, after_pose = prices
    return {"parent_result_digest": result["digest"], "reasons": reasons,
            "parent_posed_frame_count": len(posed), "views_dropped_by_consistency": len(dropped),
            "expected_cost_usd": round((SPLAT_FRAME_TARGET + task_views) * pose_cost + after_pose, 6),
            # A scale review (one planner call) runs only if the new pose's references need it.
            "maximum_cost_usd": round((SPLAT_POSE_INPUT_CAPACITY + task_views) * pose_cost + after_pose
                                      + review_cost, 6)}


def request_rebuild(job_dir: Path, job: Mapping[str, Any], *, at_epoch: int) -> dict[str, Any]:
    """Record the operator's reviewed request; called under the job lock by ``retry_preparation_job``."""
    job_dir = Path(job_dir)
    eligibility = rebuild_eligibility(job_dir, job)
    if eligibility is None:
        raise ValueError("atlas_splat_rebuild_not_eligible")
    root = rebuild_root(job_dir, eligibility["parent_result_digest"])
    parent = _read(job_dir / "result.json")
    root.mkdir(parents=True, exist_ok=True)
    retained = root / "parent_result.json"
    if retained.exists() and _read(retained) != parent:
        raise ValueError("atlas_splat_rebuild_parent_changed")
    write_json(retained, parent)
    value = {"schema_version": SCHEMA, **eligibility, "root": str(root.relative_to(job_dir)),
             "requested_at_epoch": at_epoch}
    value["digest"] = canonical_digest(value, digest_field="digest")
    write_json(job_dir / REQUEST_FILE, value)
    return value


def load_request(job_dir: Path) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    request = _read(job_dir / REQUEST_FILE)
    root = _inside(job_dir, str(request.get("root") or "missing"))
    if (request.get("schema_version") != SCHEMA or request.get("digest") != canonical_digest(request, digest_field="digest")
            or root.parent != job_dir / ROOT):
        raise ValueError("atlas_splat_rebuild_request_invalid")
    parent = _read(root / "parent_result.json")
    if parent.get("digest") != request["parent_result_digest"] or parent["digest"] != canonical_digest(
            parent, digest_field="digest"):
        raise ValueError("atlas_splat_rebuild_parent_changed")
    return request, root, parent


def retained_outcome(job_dir: Path, root: Path, parent: Mapping[str, Any]) -> dict[str, Any] | None:
    """The result a finished rebuild left the job with (the rebuilt one, or the parent's), else None."""
    path = root / OUTCOME_FILE
    if not path.is_file():
        return None
    return _read(Path(job_dir) / "result.json") if _read(path).get("status") == "published" else dict(parent)


def record_stop(root: Path, request: Mapping[str, Any], parent: Mapping[str, Any], result: Mapping[str, Any]) -> dict:
    """Record why the rebuild stopped; the job goes on with the parent result, nothing more bought."""
    write_json(root / OUTCOME_FILE, {"schema_version": "atlas_splat_rebuild_outcome.v1", "status": "stopped",
                                     "stage": result.get("stage"), "reason": result.get("reason"),
                                     "request_digest": request["digest"], "parent_result_digest": parent["digest"]})
    return dict(parent)


def publish(job_dir: Path, root: Path, request: Mapping[str, Any], parent: Mapping[str, Any],
            result: Mapping[str, Any]) -> dict[str, Any]:
    """The rebuilt result as the job's ``result.json``, its paths read from the job root."""
    job_dir = Path(job_dir)
    source_pose = dict(result["source_pose"])
    source_pose["pose_relative_path"] = str((root / source_pose["pose_relative_path"]).relative_to(job_dir))
    published = {key: value for key, value in result.items() if key != "digest"}
    published.update(source_pose=source_pose, splat_rebuild={
        "parent_result_digest": parent["digest"], "request_digest": request["digest"], "root": request["root"]})
    published["digest"] = canonical_digest(published, digest_field="digest")
    write_json(root / "published_result.json", published)
    write_json(job_dir / "result.json", published)
    write_json(root / OUTCOME_FILE, {"schema_version": "atlas_splat_rebuild_outcome.v1", "status": "published",
                                     "result_digest": published["digest"], "request_digest": request["digest"],
                                     "parent_result_digest": parent["digest"]})
    return published
