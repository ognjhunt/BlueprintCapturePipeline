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

    # Operation coverage is an independent source-bound reading, never a scale
    # reference or evidence that the preparation requirements have been met.
    value = ScaleProposal.model_validate({key: item for key, item in proposal.items()
        if key != "operation_coverage"}).model_dump(mode="json", exclude_none=True)
    references = value["scale_references"]
    if value["status"] != "proposed" or not references:
        raise ValueError("atlas_scale_review_abstained")
    if any(row["frame_id"] not in frame_ids or not row.get("segmentation_prompt", "").strip()
           for row in references):
        raise ValueError("atlas_scale_review_reference_invalid")
    retained = {**value, **({"operation_coverage": proposal["operation_coverage"]}
                           if "operation_coverage" in proposal else {})}
    return {**automation, "scale_references": references, "reference_sources": reference_sources,
            "scale_review": {"proposal_digest": canonical_digest(retained), "summary": value["summary"]}}


def _operation_coverage_feedback(root: Path, *, request: dict | None, needed: bool) -> dict:
    """Augment only a fresh scale call; preserve its exact input on every resume.

    Any retained legacy planner record, including an uncertain submission,
    keeps the original scale request. A separate semantic clarification may
    later resolve routing without buying that scale review again.
    """
    from .decision_evidence_contracts import canonical_json
    from .launch_immutable_input_writer import write_exclusive_private_bytes

    path = root / "operation-coverage-request.json"
    retained = path.exists() or path.is_symlink()
    if not retained and (request is None or not needed or any((root / "planner").glob("*.json"))):
        return {}
    if not isinstance(request, dict):
        raise ValueError("atlas_scale_operation_coverage_request_changed")
    payload = (canonical_json(request) + "\n").encode()
    if len(payload) > 65_536:
        raise ValueError("atlas_scale_operation_coverage_request_too_large")
    if retained:
        if (path.is_symlink() or not path.is_file() or path.stat().st_size > 65_536
                or path.read_bytes() != payload):
            raise ValueError("atlas_scale_operation_coverage_request_changed")
    else:
        write_exclusive_private_bytes(path, payload)
    return {"operation_coverage_request": request}


def measure_review(*, frames: list[dict], pose_digest: str, automation: dict, proposal: dict,
                   reference_sources: list, reference_masks: list, output_root: Path,
                   surfaces=(), surface_source: dict | None = None) -> tuple[dict, Path]:
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
                            output=path, reference_masks=reference_masks, camera_height=True,
                            surfaces=surfaces, surface_source=surface_source)
    return result, path


def repair_source_scale(*, pose_root: Path, pose: dict, inputs: dict, automation: dict,
                        previous: dict, task_context: dict, configuration: dict, check,
                        planner_runner, surfaces=(), surface_source: dict | None = None,
                        operation_coverage_request: dict | None = None,
                        operation_coverage_needed: bool = False,
                        operation_coverage_consumer=None) -> tuple[dict, Path]:
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
    feedback.update(_operation_coverage_feedback(root, request=operation_coverage_request,
                                                 needed=operation_coverage_needed))
    check("scale_review")
    receipt = planner_runner(survey_path=survey_path, task_context=task_context,
                             configuration=configuration, output_root=root / "planner", scaling=feedback)
    proposal = receipt["proposal"]
    if "operation_coverage" in proposal and operation_coverage_consumer is not None:
        # A scale abstention must not discard an independently valid reading
        # of the requested operations. The consumer retains and validates it.
        operation_coverage_consumer(receipt)
    plan = revised_plan(automation=automation, proposal=proposal, frame_ids={f["frame_id"] for f in frames},
                        reference_sources=receipt.get("reference_sources", []))
    masks = segment_scale_references(frames=frames, references=plan["scale_references"],
        root=root / "masks" / canonical_digest(plan)[7:], task_context=task_context,
        source_digest=source_digest, checkpoint=check)
    return measure_review(frames=frames, pose_digest=pose["digest"], automation=automation,
        proposal=proposal, reference_sources=receipt.get("reference_sources", []),
        reference_masks=masks, output_root=root / "measurements", surfaces=surfaces, surface_source=surface_source)
