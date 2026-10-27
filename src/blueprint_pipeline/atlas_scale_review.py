"""Repair a source scale estimate without repeating reconstruction or object authoring.

Beta journey: retain the scene, review the actual source dimension endpoints,
then remeasure with named-object masks. A corrected plan has its own receipt;
the earlier assumption and every provider output remain unchanged.
"""
from __future__ import annotations

from pathlib import Path

from .common import write_json
from .decision_evidence_contracts import canonical_digest


def rests_on_height_priors(measurements: dict) -> bool:
    """An estimate no reference stands behind: a guesstimate on height priors alone."""
    from .atlas_scene_measurements import PRIOR_ONLY_BASES

    return measurements.get("status") == "estimated" and (
        measurements.get("reference_agreement") or {}).get("basis") in PRIOR_ONLY_BASES


def needs_scale_review(measurements: dict) -> bool:
    """A review runs unless named references set the scale.

    A guesstimate on height priors alone still asks for references first, as it
    did before those priors existed; if the review finds none, the guesstimate stands.
    """
    return measurements.get("status") != "estimated" or not (
        measurements.get("binding") or {}).get("reference_masks_digest") or rests_on_height_priors(measurements)


def revised_plan(*, automation: dict, proposal: dict, frame_ids: set[str], reference_sources: list) -> dict:
    from .atlas_scene_planner import ScaleProposal

    value = ScaleProposal.model_validate(proposal).model_dump(mode="json", exclude_none=True)
    references = value["scale_references"]
    if value["status"] != "proposed" or not references:
        raise ValueError("atlas_scale_review_abstained")
    if any(row["frame_id"] not in frame_ids or not row.get("segmentation_prompt", "").strip()
           for row in references):
        raise ValueError("atlas_scale_review_reference_invalid")
    return {**automation, "scale_references": references, "reference_sources": reference_sources,
            "scale_review": {"proposal_digest": canonical_digest(value), "summary": value["summary"]}}


def measure_review(*, frames: list[dict], pose_digest: str, automation: dict, proposal: dict,
                   reference_sources: list, reference_masks: list, output_root: Path) -> tuple[dict, Path]:
    """CPU replay shared by the production planner and a retained Codex review.

The caller supplies real mask observations, never an assumed full-image mask.
This function neither invokes a model nor fabricates a provider receipt.
"""
    from .atlas_scene_measurements import estimate_sizes

    plan = revised_plan(automation=automation, proposal=proposal,
                        frame_ids={f["frame_id"] for f in frames}, reference_sources=reference_sources)
    if reference_masks is None:
        raise ValueError("atlas_scale_review_object_masks_required")
    identity = canonical_digest({"pose_digest": pose_digest, "plan": plan, "reference_masks": reference_masks})
    path = output_root / (identity[7:] + ".json")
    output_root.mkdir(parents=True, exist_ok=True)
    result = estimate_sizes(frames=frames, automation=plan, pose_digest=pose_digest,
                            output=path, reference_masks=reference_masks, camera_height=True)
    return result, path


def repair_source_scale(*, pose_root: Path, pose: dict, inputs: dict, automation: dict,
                        previous: dict, task_context: dict, configuration: dict, check,
                        planner_runner) -> tuple[dict, Path]:
    from .atlas_context_fov import full_view_sources
    from .atlas_reference_masks import segment_scale_references
    from .atlas_scene_measurements import load_views

    # Re-review on the complete original grid, even when old coordinates were
    # bound to a provider crop. New normalized endpoints belong to these pixels.
    _, frames = load_views(pose_root, inputs)
    sources = full_view_sources(inputs, pose_root)
    root = pose_root / "scale_review"
    root.mkdir(parents=True, exist_ok=True)
    source_digest = inputs["binding"]["source_video_digest"]
    survey = {"source_video_digest": source_digest, "heldout_pixels_included": False,
              "display_rotation_degrees": 0, "pose_digest": pose["digest"],
              "frames": [{"frame_id": row["frame_id"], "source_image_path": row["image_path"],
                          "source_image_digest": row["image_digest"]} for row in sources]}
    survey["digest"] = canonical_digest(survey, digest_field="digest")
    survey_path = root / (survey["digest"][7:] + ".survey.json")
    write_json(survey_path, survey)
    feedback = {"previews": [], "pose_digest": pose["digest"],
                "previous_references": automation.get("scale_references", []),
                "previous_measurement_digest": previous["digest"],
                "reference_agreement": previous.get("reference_agreement"),
                "reference_gaps": previous.get("reference_gaps", []),
                "named_object_masks_checked": bool((previous.get("binding") or {}).get("reference_masks_digest"))}
    check("scale_review")
    receipt = planner_runner(survey_path=survey_path, task_context=task_context,
                             configuration=configuration, output_root=root / "planner", scaling=feedback)
    proposal = receipt["proposal"]
    plan = revised_plan(automation=automation, proposal=proposal, frame_ids={f["frame_id"] for f in frames},
                        reference_sources=receipt.get("reference_sources", []))
    masks = segment_scale_references(frames=frames, references=plan["scale_references"],
        root=root / "masks" / canonical_digest(plan)[7:], task_context=task_context,
        source_digest=source_digest, checkpoint=check)
    return measure_review(frames=frames, pose_digest=pose["digest"], automation=automation,
        proposal=proposal, reference_sources=receipt.get("reference_sources", []),
        reference_masks=masks, output_root=root / "measurements")
