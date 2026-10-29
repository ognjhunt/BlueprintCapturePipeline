"""Original capture evidence must precede a novel-view gap purchase (ADP-040)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from .decision_evidence_contracts import canonical_digest
from .local_reconstruction_adapters import _sha256_file


class GapSourceEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    gap_id: str = Field(min_length=1)
    visibility: str = Field(pattern="^(observed_in_conditioning|observed_elsewhere_in_capture|occluded|not_found_in_reviewed_capture|physical_opening|uncertain)$")
    source_frame_ids: list[str]
    reason: str = Field(min_length=1)


SOURCE_FIRST_INSTRUCTIONS = """
Check the ORIGINAL CAPTURE gallery before proposing any generation. These raw
frames are evidence for this decision only; they are not additional Atlas inputs.
They may still show removed task objects. Compare the fixed surrounding surfaces,
not the task object that was intentionally removed.
For EACH selected gap provide source_evidence with its exact gap_id, visibility,
source_frame_ids and reason. Distinguish: observed_in_conditioning (visible in
the recorded frames whose IDs are in conditioned_frame_ids),
observed_elsewhere_in_capture (a suitable recorded view is not yet conditioned),
occluded, not_found_in_reviewed_capture, physical_opening, or uncertain.
A missing splat or missing/invalid Atlas depth does NOT establish that the video
never saw the surface. A real opening, passage, installation recess, or free space
must not be filled with an invented panel. Black render pixels alone prove neither
a missing wall nor an unseen surface. Inspect complete connected work/support
surfaces, not only the manipulated object's opening.
Cite exact original frame IDs for observed surfaces or visible opening boundaries.
If the surface is already photographed, explain why usable geometry still needs
repair. Prefer recovering its recorded evidence over inventing an appearance.
Use previous_repair_outcomes when supplied: an RGB fill with missing native depth
is different from valid depth rejected by spatial checks. Do not repeat a failed
view without explaining what additional evidence or different angle will help.
If a useful additional source view exists, mark observed_elsewhere_in_capture;
the controller must incorporate it before buying a novel view. Use not_found only
for the reviewed candidate gallery; this never proves absence from every video
frame. Do not claim an occluded installation surface was recorded just because
its covering object was recorded. Small distortions and raster pinholes need no
repair. When the output schema permits sufficient, use it when no material task
surface requires generation.
"""


def combine_source_surveys(*, survey_paths, output):
    """Keep full-video candidates when a dense task-state survey covers less time."""
    rows, surveys, video, rotation = {}, [], None, None
    for path in dict.fromkeys(Path(p) for p in survey_paths):
        value = json.loads(path.read_text())
        if (value.get('schema_version') != 'atlas_video_survey.v1'
                or value.get('digest') != canonical_digest(value, digest_field='digest')
                or value.get('heldout_pixels_included') is not False):
            raise ValueError('atlas_source_gap_candidate_survey_invalid')
        if video is None:
            video, rotation = value['source_video_digest'], value['display_rotation_degrees']
        if value['source_video_digest'] != video or value['display_rotation_degrees'] != rotation:
            raise ValueError('atlas_source_gap_candidate_capture_mismatch')
        surveys.append(value['digest'])
        for row in value['frames']:
            old = rows.get(row['frame_id'])
            if old is not None and any(old[k] != row[k] for k in ('source_image_digest', 'timestamp_seconds')):
                raise ValueError('atlas_source_gap_candidate_frame_changed')
            rows[row['frame_id']] = row
    result = {'schema_version': 'atlas_video_survey.v1', 'source_video_digest': video,
              'display_rotation_degrees': rotation, 'heldout_pixels_included': False,
              'frames': sorted(rows.values(), key=lambda r: r['timestamp_seconds']),
              'source_candidate_survey_digests': surveys,
              'basis': 'full_video_candidates_plus_dense_task_state_candidates',
              'claim_ceiling': 'development_only'}
    result['digest'] = canonical_digest(result, digest_field='digest')
    if output.exists() and json.loads(output.read_text()) != result:
        raise ValueError('atlas_source_gap_full_capture_survey_changed')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    return output


def source_gallery(*, survey_path, source_video_digest, conditioned_frame_ids, root):
    """Bind the complete supplied candidate survey, not just selected RGBD inputs."""
    from .atlas_scene_preservation import _save_image

    if survey_path is None:
        raise ValueError("atlas_source_gap_capture_survey_required")
    survey = json.loads(Path(survey_path).read_text())
    if (survey.get("schema_version") != "atlas_video_survey.v1"
            or survey.get("digest") != canonical_digest(survey, digest_field="digest")
            or survey.get("heldout_pixels_included") is not False
            or survey.get("source_video_digest") != source_video_digest
            or not survey.get("frames")):
        raise ValueError("atlas_source_gap_capture_survey_changed")
    identifiers = [row["frame_id"] for row in survey["frames"]]
    if len(identifiers) != len(set(identifiers)) or not set(conditioned_frame_ids) <= set(identifiers):
        raise ValueError("atlas_source_gap_capture_identity_invalid")
    root.mkdir(parents=True, exist_ok=True)
    previews = []
    for index, row in enumerate(survey["frames"]):
        source = Path(row["source_image_path"])
        if source.is_symlink() or _sha256_file(source) != row["source_image_digest"]:
            raise ValueError("atlas_source_gap_capture_image_changed")
        with Image.open(source) as image:
            pixels = image.convert("RGB").rotate(survey["display_rotation_degrees"], expand=True)
            pixels.thumbnail((1024, 1024))
            target = root / f"{index:05d}.png"
            _save_image(target, np.asarray(pixels))
        previews.append({"path": str(target), "sha256": _sha256_file(target),
                         "source_frame_id": row["frame_id"],
                         "timestamp_seconds": row["timestamp_seconds"],
                         "kind": "original_capture_source_check_only"})
    return {"survey_digest": survey["digest"], "source_video_digest": source_video_digest,
            "reviewed_source_frame_ids": identifiers,
            "conditioned_frame_ids": list(conditioned_frame_ids),
            "original_capture_is_not_generation_input": True,
            "candidate_gallery_does_not_prove_every_video_frame_was_reviewed": True}, previews


def source_actions(entries, gallery):
    """Refuse unsupported generation and route already recorded views first."""
    available = set(gallery["reviewed_source_frame_ids"])
    conditioned = set(gallery["conditioned_frame_ids"])
    additional, openings, unresolved, audits = set(), {}, [], []
    for entry in entries:
        rows = [GapSourceEvidence.model_validate(row).model_dump(mode="json")
                for row in entry.get("source_evidence", [])]
        requested = entry.get("gap_ids", [])
        if (not requested or len(rows) != len(requested) or {r["gap_id"] for r in rows} != set(requested)
                or len({r["gap_id"] for r in rows}) != len(rows)):
            raise ValueError("atlas_source_gap_evidence_required_for_each_gap")
        for row in rows:
            ids = set(row["source_frame_ids"])
            if len(ids) != len(row["source_frame_ids"]) or not ids <= available:
                raise ValueError("atlas_source_gap_evidence_unknown_frame")
            visibility = row["visibility"]
            if visibility in {"observed_in_conditioning", "observed_elsewhere_in_capture", "physical_opening"} and not ids:
                raise ValueError("atlas_source_gap_observation_requires_frame")
            if visibility == "observed_in_conditioning" and not ids <= conditioned:
                raise ValueError("atlas_source_gap_unconditioned_observation")
            if visibility == "observed_elsewhere_in_capture":
                if not ids - conditioned:
                    raise ValueError("atlas_source_gap_additional_view_missing")
                additional.update(ids - conditioned)
            if visibility == "physical_opening":
                openings.setdefault(entry["camera_id"], []).append(row["gap_id"])
            if visibility == "uncertain":
                unresolved.append(row["gap_id"])
            audits.append({"camera_id": entry["camera_id"], **row})
    value = {"gallery": gallery, "gap_evidence": audits,
             "additional_source_frame_ids": sorted(additional),
             "physical_opening_gap_ids": openings, "unresolved_gap_ids": unresolved,
             "source_coverage_is_not_depth_coverage": True}
    value["digest"] = canonical_digest(value, digest_field="digest")
    return value


def repair_depth_outcomes(generated, report):
    """Expose partial per-view depth failures even when another view succeeded."""
    generation = report.get('generation') or {}
    paths = [Path(p) for p in generation.get('preservation_paths', [])
             if Path(p).name == 'rgbd-completion-plan.json']
    if not paths:
        return []  # Older/fake reports have no pixel-bound plan; do not invent counts.
    if len(paths) != 1:
        raise ValueError('atlas_source_gap_depth_plan_ambiguous')
    plan = json.loads(paths[0].read_text())
    if (plan.get('digest') != canonical_digest(plan, digest_field='digest')
            or plan['digest'] != generation.get('mask_plan_digest')):
        raise ValueError('atlas_source_gap_depth_plan_changed')
    outcomes = []
    for row in plan['prepared_views']:
        if not row.get('completion_gap_ids'):
            continue
        matching = [f for f in generated if f['frame_id'].endswith('-' + row['frame_id'])]
        if len(matching) != 1:
            raise ValueError('atlas_source_gap_depth_view_missing')
        mask_path = Path(row['completion_gap_mask_path'])
        if _sha256_file(mask_path) != row['completion_gap_mask_digest']:
            raise ValueError('atlas_source_gap_depth_mask_changed')
        gap = np.asarray(Image.open(mask_path).convert('L')) > 0
        frame = matching[0]
        valid_depth = np.isfinite(frame['depth']) & (frame['depth'] > 0)
        if gap.shape != valid_depth.shape:
            raise ValueError('atlas_source_gap_depth_grid_changed')
        outcomes.append({'camera_id': row['frame_id'], 'gap_ids': row['completion_gap_ids'],
                         'requested_gap_pixels': int(gap.sum()),
                         'native_depth_available_in_gap': int((gap & valid_depth).sum()),
                         'admitted_depth_in_gap': int((gap & valid_depth & (frame['confidence'] > 0)).sum()),
                         'native_depth_is_unmodified': True})
    return outcomes
