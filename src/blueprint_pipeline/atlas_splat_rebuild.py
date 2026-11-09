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

A rebuild whose new pose World Labs again did not refine may run once more,
with its pose limited in time (owner, 2026-11-09: "yes" to the windowed retry).
The first rebuild re-selected views across the whole clip, so the next pose was
sent the same pan, jump or stationary run that the earlier pose's own cameras
show (``atlas_splat_input_quality.select_splat_frames`` finds them only after a
pose). The windowed retry (``WINDOW_ATTEMPT``):

- qualifies only when the first rebuild stopped at its pose with refinement not
  accepted, and its request was not windowed. Never a third attempt.
- runs in its own root beside the first. The first request is kept in the first
  root as ``request.json`` and named by digest in the new one; the first root's
  pose and outcome stay as they are.
- has the runner (``atlas_splat_rebuild_window``) read the earlier pose's cameras,
  keep the segments ``select_splat_frames`` would keep, and record that window and
  the review it derives once, before any purchase. A pose with no usable window
  (already one moving segment, no moving segment, no timestamps or depth) stops
  the retry by name with nothing bought.
- costs what the first rebuild could: the same expected and maximum cost, the
  same allowance check and paid admission.

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
#: The one retry of a rebuild whose pose World Labs did not refine: its pose limited to the earlier pose's
#: consistent moving segments (``atlas_splat_rebuild_window``).
WINDOW_ATTEMPT = "pose_window"
#: The window the runner recorded in the retry's pose root before its pose (read by ``atlas_source_pose_inputs``).
WINDOW_FILE = "atlas_pose_window.json"
WINDOW_SCHEMA = "atlas_splat_rebuild_pose_window.v1"
#: Where the first rebuild's request is kept once the retry's request takes its place.
RETAINED_REQUEST_FILE = "request.json"
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


def rebuild_root(job_dir: Path, parent_result_digest: str, attempt: str | None = None) -> Path:
    name = parent_result_digest[7:23]
    return Path(job_dir) / ROOT / (name if attempt is None else f"{name}-{attempt.replace('_', '-')}")


def _posed_results(pose_root: Path) -> list[Path]:
    """Every retained pose result under a rebuild's pose root (the full-view source sits a level or two down)."""
    if not pose_root.is_dir() or pose_root.is_symlink():
        return []
    return sorted(path for path in pose_root.rglob("atlas_posed_rgbd_result.json")
                  if not path.is_symlink() and path.resolve().is_relative_to(pose_root.resolve()))


def earlier_pose(pose_root: Path) -> dict[str, Any] | None:
    """The one pose result a stopped rebuild left in its pose root (copies of it agree), else None."""
    found = {}
    for path in _posed_results(pose_root):
        value = _read(path)
        if value.get("digest") != canonical_digest(value, digest_field="digest"):
            return None
        found.setdefault(value["digest"], (path, value))
    if len(found) != 1:
        return None
    path, value = next(iter(found.values()))
    return {**value, "_source_root": str(path.parent)}


def _prices(job_dir: Path, result: Mapping[str, Any]) -> tuple[list[str], list, tuple[float, float, float]]:
    """The parent's posed frame ids, its selected frames and the job's reviewed prices, from retained records."""
    pose_root = _inside(job_dir, (result.get("source_pose") or {}).get("pose_relative_path") or "missing")
    pose = _read(pose_root / "atlas_posed_rgbd_result.json")
    policy = _read(job_dir / "configuration.json").get("policy") or {}
    posed = [row["frame_id"] for row in pose["operation_receipt"]["input_usage"]["frames"]]
    selected = _read(job_dir / "pose/atlas_view_selection.json").get("frames") or []
    prices = (float(policy["pose_input_cost_usd"]), float(policy.get("maximum_review_cost_usd", 0)),
              float(policy["refined_splat_cost_usd"]) + float(policy.get("mesh_export_cost_usd", 0)))
    return posed, selected, prices


def _costs(posed: list[str], selected: list, prices: tuple[float, float, float]) -> dict[str, float]:
    from .atlas_refined_splat_forms import SPLAT_FRAME_TARGET, SPLAT_POSE_INPUT_CAPACITY
    task_views = max(0, len(posed) - len(selected))
    pose_cost, review_cost, after_pose = prices
    return {"expected_cost_usd": round((SPLAT_FRAME_TARGET + task_views) * pose_cost + after_pose, 6),
            # A scale review (one planner call) runs only if the new pose's references need it.
            "maximum_cost_usd": round((SPLAT_POSE_INPUT_CAPACITY + task_views) * pose_cost + after_pose
                                      + review_cost, 6)}


def _window_retry_eligibility(job_dir: Path, result: Mapping[str, Any]) -> dict[str, Any] | None:
    """The one windowed retry of a rebuild that stopped at a pose World Labs did not refine; else None."""
    try:
        request = _read(job_dir / REQUEST_FILE)
        if (request.get("schema_version") != SCHEMA or request.get("attempt") is not None
                or request.get("digest") != canonical_digest(request, digest_field="digest")
                or request.get("parent_result_digest") != result.get("digest")):
            return None
        root = _inside(job_dir, str(request.get("root") or "missing"))
        outcome = _read(root / OUTCOME_FILE)
        if (root.parent != job_dir / ROOT or outcome.get("status") != "stopped"
                or outcome.get("stage") != "source_pose"
                or outcome.get("reason") != "atlas_splat_rebuild_pose_refinement_not_accepted"
                or outcome.get("request_digest") != request["digest"]
                or rebuild_root(job_dir, result["digest"], WINDOW_ATTEMPT).exists()):
            return None
        earlier = earlier_pose(root / "pose")
        if earlier is None:
            return None
        if earlier["operation"]["response"].get("poseRefinementAccepted") is True:
            return None
        earlier_count = len(earlier["operation_receipt"]["input_usage"]["frames"])
        posed, selected, prices = _prices(job_dir, result)
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return {"parent_result_digest": result["digest"], "attempt": WINDOW_ATTEMPT,
            "reasons": ["previous_rebuild_pose_refinement_not_accepted"],
            "previous_request_digest": request["digest"], "previous_root": request["root"],
            "previous_posed_frame_count": earlier_count,
            # The window itself is the runner's to read from the earlier pose's cameras and depth.
            "window_basis": "earlier_pose_consistent_moving_segments",
            "parent_posed_frame_count": len(posed), **_costs(posed, selected, prices)}


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
            or result.get("splat_rebuild") or (job_dir / SELECTION_FILE).exists()):
        return None
    if (job_dir / REQUEST_FILE).exists():
        # Back on its parent after a rebuild: only the one windowed retry of an unrefined pose remains.
        return _window_retry_eligibility(job_dir, result)
    try:
        pose_root = _inside(job_dir, (result.get("source_pose") or {}).get("pose_relative_path") or "missing")
        pose = _read(pose_root / "atlas_posed_rgbd_result.json")
        draft = _read(job_dir / "splat/atlas_depth_splat_manifest.json")
        posed, selected, prices = _prices(job_dir, result)
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
    return {"parent_result_digest": result["digest"], "reasons": reasons,
            "parent_posed_frame_count": len(posed), "views_dropped_by_consistency": len(dropped),
            **_costs(posed, selected, prices)}


def request_rebuild(job_dir: Path, job: Mapping[str, Any], *, at_epoch: int) -> dict[str, Any]:
    """Record the operator's reviewed request; called under the job lock by ``retry_preparation_job``."""
    job_dir = Path(job_dir)
    eligibility = rebuild_eligibility(job_dir, job)
    if eligibility is None:
        raise ValueError("atlas_splat_rebuild_not_eligible")
    attempt = eligibility.get("attempt")
    if attempt is not None:
        # The first request stays, in its own root, before the retry's request takes its place.
        previous = _read(job_dir / REQUEST_FILE)
        kept = _inside(job_dir, previous["root"]) / RETAINED_REQUEST_FILE
        if kept.exists() and _read(kept) != previous:
            raise ValueError("atlas_splat_rebuild_previous_request_changed")
        write_json(kept, previous)
    root = rebuild_root(job_dir, eligibility["parent_result_digest"], attempt)
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
    """Record why the rebuild stopped; the job goes on with the parent result, nothing more bought.

    A stop is final: a windowed retry's stop leaves no further attempt (``rebuild_eligibility``).
    """
    write_json(root / OUTCOME_FILE, {"schema_version": "atlas_splat_rebuild_outcome.v1", "status": "stopped",
                                     "stage": result.get("stage"), "reason": result.get("reason"),
                                     **({"attempt": request["attempt"]} if request.get("attempt") else {}),
                                     **({"flags": list(result["flags"])} if result.get("flags") else {}),
                                     "request_digest": request["digest"], "parent_result_digest": parent["digest"]})
    return dict(parent)


def publish(job_dir: Path, root: Path, request: Mapping[str, Any], parent: Mapping[str, Any],
            result: Mapping[str, Any], window: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The rebuilt result as the job's ``result.json``, its paths read from the job root.

    A windowed retry's result names its window: the intervals posed, the views the
    reviewed coverage lost to it and the review it derived (``atlas_splat_rebuild_window``).
    """
    job_dir = Path(job_dir)
    source_pose = dict(result["source_pose"])
    source_pose["pose_relative_path"] = str((root / source_pose["pose_relative_path"]).relative_to(job_dir))
    published = {key: value for key, value in result.items() if key != "digest"}
    marker = {"parent_result_digest": parent["digest"], "request_digest": request["digest"], "root": request["root"]}
    if window is not None:
        marker.update(attempt=request.get("attempt"), previous_request_digest=request.get("previous_request_digest"),
                      pose_window={key: window[key] for key in (
                          "digest", "intervals_seconds", "excluded_frame_ids", "left_out_required_frame_ids",
                          "anchor_frame_id", "original_anchor_frame_id", "original_review_digest",
                          "review_digest", "flags")})
    published.update(source_pose=source_pose, splat_rebuild=marker)
    published["digest"] = canonical_digest(published, digest_field="digest")
    write_json(root / "published_result.json", published)
    write_json(job_dir / "result.json", published)
    write_json(root / OUTCOME_FILE, {"schema_version": "atlas_splat_rebuild_outcome.v1", "status": "published",
                                     "result_digest": published["digest"], "request_digest": request["digest"],
                                     "parent_result_digest": parent["digest"]})
    return published
