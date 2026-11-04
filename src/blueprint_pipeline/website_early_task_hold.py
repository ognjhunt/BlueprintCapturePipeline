"""Hold a website job whose task Blueprint cannot evaluate right after its Atlas review.

The Atlas scene-planner review is the first purchase of a website job's Atlas lane
and the call that reads the whole task (``website_task_semantics``). Once it is
retained, every input of two decisions exists: the job has no native task profile
(close a door, push, liquid, assembly, navigation), or its family profile (fold,
pour, wipe, insert, stack, press, multi-step) has no runtime installed on this host.
Such a job used to buy its full Atlas scene (pose, generated views, splat, mesh,
background automation) and only then hold at the scene handoff. It now holds here,
before the first World Labs reservation, under the same typed codes, and
``holds_native_task_preparation`` re-derives the hold before the capture stage or
the status reader accepts it.

A hold never discards the job. The review, selection and semantics stay retained;
``website_atlas_preview.retry_failed_job`` reopens it, and the next pass re-derives
the decision from the deployed code and catalog: once the capability ships, it buys
the scene from the retained selection without buying the review again.

Not held here, and why:

- A reading that names a registered profile but leaves part of the job unresolved.
  Its decision needs the operation-coverage clarification, which is bought with the
  scale review after the source pose, or after the mesh
  (``website_operation_coverage_review``). It still holds at the handoff.
- Several task objects, or a soft one (``native_task_subject_gap``). Those read the
  capture's clean-plate object analysis, which runs beside this job rather than
  before it.
- A job that already started buying its scene (``scene_purchase_started``): the
  handoff makes the same decision after it, as before.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .decision_evidence_contracts import canonical_digest

WEBSITE_CONTEXT_SCHEMA = "website_site_task_context.v1"
SCHEMA = "website_atlas_task_preparation_hold.v1"
HOLD_FILE = "task_preparation_hold.json"
#: The preview job's stage while held; ``atlas_preparation_job_control`` keeps it terminal.
STAGE = "task_preparation_hold"
HOLD_POINT = "after_atlas_review"
FAMILY_RUNTIME_GAP = "website_task_family_runtime_preparation_required"
PROFILE_GAP = "website_native_task_profile_preparation_required"
COVERAGE_GAP = "website_task_operation_coverage_unresolved"
#: The holds decided at ``HOLD_POINT``; a subject hold needs the scene's object analysis.
EARLY_CODES = frozenset({PROFILE_GAP, COVERAGE_GAP, FAMILY_RUNTIME_GAP})
#: Stages only the scene purchase creates under a job's Atlas root.
_SCENE_STAGE_DIRS = ("automation", "generated", "splat", "mesh", "preserved_completion", "downloads")
_POSE_PURCHASE_NAMES = frozenset({"atlas_pose", "atlas_pose_attempts", "controller_admission.json",
                                  "atlas_posed_rgbd_result.json"})


def required_task_families(family: str, task_semantics: Mapping[str, Any] | None) -> list[str]:
    """The families one runtime must register: a multi-step job's stages too."""
    from .task_family_scoring import canonical_task_family
    from .task_profile_contracts import task_profile_for
    families = {family}
    if family == "multi_step" and task_semantics is not None:
        clarification = task_semantics.get("operation_coverage_clarification")
        reading = clarification["operation_coverage"] if clarification else task_semantics["semantics"]
        for step in reading["steps"]:
            profile = task_profile_for(step["operation"])
            stage = canonical_task_family(profile.profile_id) if profile is not None else None
            if stage is not None:
                families.add(stage)
    return sorted(families)


def task_family_runtime_gap(task_context: Mapping[str, Any], *, task_semantics: Mapping[str, Any] | None,
                            source_video_digest: str | None,
                            runtimes: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
    """Hold a family task when no runtime on this host registers its families.

    ``runtimes`` is ``task_family_native_runtime.installed_task_family_runtimes()``
    (read when omitted). A catalog that is configured but unreadable holds nothing:
    the job goes on and the later family preparation reports what is missing.
    """
    from .task_family_scoring import canonical_task_family
    from .website_task_semantics import effective_task_profile
    family = canonical_task_family(effective_task_profile(task_context=task_context, task_semantics=task_semantics,
                                                          source_video_digest=source_video_digest))
    if family is None:
        return None
    if runtimes is None:
        from .task_family_native_runtime import installed_task_family_runtimes
        runtimes = installed_task_family_runtimes()
    required = required_task_families(family, task_semantics)
    if runtimes is None or set(required) <= set(runtimes.get("task_families") or []):
        return None
    general = task_context.get("task_family") == "general_task" and task_semantics is not None
    return {
        "code": FAMILY_RUNTIME_GAP, "task_family": task_context["task_family"],
        "task_context_digest": task_context.get("context_digest"), "job_disposition": "held_for_preparation",
        "description": task_context.get("description"),
        "reason": (f"The task is a {family.replace('_', ' ')} job. No simulation runtime for it is installed "
                   "yet; the job is held while one is prepared. This is not a finding that robot teams "
                   "cannot perform the job."),
        "missing": ["qualified_task_family_runtime"], "task_family_profile": family,
        "required_task_families": required,
        "runtime_catalog": {key: runtimes.get(key) for key in ("configured", "catalog_digest", "task_families")},
        **({"task_semantics": dict(task_semantics), "source_video_digest": source_video_digest} if general else {}),
        "claim_boundary": "Native evaluation implementation gap only; no robot-team capability verdict.",
    }


def early_task_preparation_gap(task_context: Mapping[str, Any], *, task_semantics: Mapping[str, Any] | None,
                               source_video_digest: str | None,
                               runtimes: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
    """The hold this website task gets from what its Atlas review already read, or None.

    None when the task can be evaluated, and when the decision still needs an input
    that does not exist yet (the operation-coverage clarification).
    """
    if task_context.get("schema_version") != WEBSITE_CONTEXT_SCHEMA or "task_family" not in task_context:
        return None
    from .website_task_preparation import native_task_preparation_gap
    options = {"task_semantics": task_semantics, "source_video_digest": source_video_digest}
    gap = native_task_preparation_gap(task_context, **options)
    if gap is not None:
        if gap["code"] == COVERAGE_GAP and task_semantics is not None:
            from .website_operation_coverage import clarification_request
            if clarification_request(task_semantics, task_context=task_context,
                                     source_video_digest=source_video_digest) is not None:
                return None
        return gap if gap["code"] in EARLY_CODES else None
    return task_family_runtime_gap(task_context, runtimes=runtimes, **options)


def scene_purchase_started(output_root: Path) -> bool:
    """Whether this Atlas job has started buying its scene; then the handoff holds it, as before."""
    root = Path(output_root)
    if any((root / name).exists() for name in _SCENE_STAGE_DIRS):
        return True
    pose = root / "pose"
    return pose.is_dir() and any(path.name in _POSE_PURCHASE_NAMES for path in pose.rglob("*"))


def hold_after_review(*, output_root: Path, task_context: Mapping[str, Any],
                      source_video_digest: str) -> dict[str, Any] | None:
    """The Atlas pipeline's result for a job held right after its review, or None to go on.

    Called once the review's selection and semantics are retained and before any
    scene stage. Writes the hold record the capture stage reads. Spends nothing.
    """
    if task_context.get("schema_version") != WEBSITE_CONTEXT_SCHEMA:
        return None
    from .website_task_semantics import load_task_semantics
    output_root = Path(output_root)
    try:
        semantics = (load_task_semantics(job_dir=output_root, task_context=task_context,
                                         source_video_digest=source_video_digest)
                     if task_context.get("task_family") == "general_task" else None)
        gap = early_task_preparation_gap(task_context, task_semantics=semantics,
                                         source_video_digest=source_video_digest)
    except (ValueError, TypeError, KeyError):
        # The handoff reaches the same decision, or its own refusal, after the scene.
        return None
    if gap is None or scene_purchase_started(output_root):
        return None
    from .common import write_json
    selection = json.loads((output_root / "selection_binding.json").read_text())
    record = {"schema_version": SCHEMA, "hold_point": HOLD_POINT, "capture_id": task_context.get("capture_id"),
              "scene_id": task_context.get("scene_id"), "task_context_digest": task_context["context_digest"],
              "source_video_digest": source_video_digest,
              "planner_receipt_digest": canonical_digest(selection.get("planner_receipt") or {}),
              "gap": gap, "scene_purchased": False, "provider_mutation_performed": False,
              "resume": "website_atlas_preview.retry_failed_job re-derives this hold from the retained review"}
    record["digest"] = canonical_digest(record, digest_field="digest")
    write_json(output_root / HOLD_FILE, record)
    result = {"schema_version": "website_atlas_task_preparation_hold_result.v1", "status": "needs_input",
              "stage": STAGE, "blockers": [gap["code"]], "source_video_digest": source_video_digest,
              "task_context_digest": task_context["context_digest"],
              "task_preparation_hold": {"path": str(output_root / HOLD_FILE), "digest": record["digest"]},
              "splat_available": False, "mesh_available": False, "simulation_qualified": False,
              "claim_ceiling": "development_only"}
    result["digest"] = canonical_digest(result, digest_field="digest")
    return result


def retain_task_preparation_hold(*, capture_root: Path, task_context: Mapping[str, Any],
                                 source_video: Path | None = None) -> dict[str, Any] | None:
    """The capture stage's handoff for a job its Atlas lane held after the review, or None.

    Only while the capture's Atlas job is itself stopped at that hold, and only when
    the current code and the retained review re-derive the same gap. Reads retained
    records and buys nothing; no frame is processed or sent anywhere.
    """
    from .website_atlas_preview_store import JOB_FILE, read_job
    job_dir = Path(capture_root) / "pipeline" / "atlas"
    path = job_dir / HOLD_FILE
    if (task_context.get("schema_version") != WEBSITE_CONTEXT_SCHEMA or path.is_symlink() or not path.is_file()
            or not (job_dir / JOB_FILE).is_file()):
        return None
    job, record = read_job(job_dir / JOB_FILE), json.loads(path.read_text())
    gap = record.get("gap") if isinstance(record, dict) else None
    if (not isinstance(gap, dict) or record.get("schema_version") != SCHEMA
            or record.get("digest") != canonical_digest(record, digest_field="digest")
            or record.get("task_context_digest") != task_context.get("context_digest")
            or job.get("state") != "needs_input" or job.get("stage") != STAGE or job.get("reason") != gap.get("code")):
        return None
    source = record["source_video_digest"]
    if source_video is not None:
        from .local_reconstruction_adapters import _sha256_file
        if _sha256_file(Path(source_video)) != source:
            raise ValueError("website_task_preparation_hold_source_video_changed")
    from .website_task_semantics import load_task_semantics
    semantics = (load_task_semantics(job_dir=job_dir, task_context=task_context, source_video_digest=source)
                 if task_context.get("task_family") == "general_task" else None)
    # The catalog is the one the Atlas worker read when it declined to buy the scene.
    derived = early_task_preparation_gap(task_context, task_semantics=semantics, source_video_digest=source,
                                         runtimes=gap.get("runtime_catalog") if gap.get("code") == FAMILY_RUNTIME_GAP
                                         else None)
    if derived is None or derived != gap:
        return None
    handoff = {"schema_version": "website_scene_handoff.v1", "status": "needs_input", "blockers": [gap["code"]],
               "claim_ceiling": "development_only", "simulator_ready": False, "visual_reconstruction_ready": False,
               "provider_mutation_performed": False, "capture_id": task_context["capture_id"],
               "scene_id": task_context["scene_id"], "native_task_preparation_gap": derived,
               "task_preparation_hold": {"hold_point": HOLD_POINT, "atlas_hold_digest": record["digest"],
                                         "atlas_scene_purchased": False}}
    handoff["digest"] = canonical_digest(handoff, digest_field="digest")
    from .website_task_context_contracts import holds_native_task_preparation
    if not holds_native_task_preparation(handoff, task_context):
        return None
    from .core.handoff_lease import guarded_handoff_write_json
    guarded_handoff_write_json(Path(capture_root) / "pipeline" / "website_scene_preparation" / "handoff.json",
                               handoff)
    return handoff
