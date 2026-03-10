# Evidence Auditor

Use when readiness claims or qualification artifacts need to be verified against actual capture evidence. This is the anti-handwaving skill. Every claim must trace to a specific artifact, field, and measurement. If a claim cannot be traced, it is unsupported.

---

## Trigger

- After intake normalization, before any readiness review skills run.
- After recapture, to verify whether evidence gaps have been closed.
- When any downstream skill references evidence that has not been audited.

---

## Exact inputs

| Artifact | Required | Fields to inspect |
|---|---|---|
| `capture_qa_scorecard.json` | Yes | `overall_grade`, `completeness_score`, `coverage_zones[]`, `qa_flags[]`, `frame_count`, `registration_rate`, `colmap_coverage`, `reconstruction_quality` |
| `geometry_evidence.json` | Yes | `measurements[]`, `measurement_type`, `confidence`, `source_modality`, `calibration_status` |
| `scene_graph.json` | Yes | `objects[]`, `object_type`, `detection_confidence`, `bounding_box`, `relationships[]` |
| `route_graph.json` | Yes | `nodes[]`, `edges[]`, `edge_width_m`, `edge_confidence`, `traffic_type`, `surface_type` |
| `normalized_intake.json` | Yes | `evidence_grade`, `capture_modality` |

---

## Required behavior

### 1. Classify evidence grade per artifact

For each artifact, determine its evidence grade:

| Grade | Definition | Can support |
|---|---|---|
| `metric` | Measurement from calibrated sensor (LiDAR, structured light, calibrated photogrammetry). Confidence > 0.8. | Qualification decisions |
| `estimated` | Measurement derived from uncalibrated reconstruction (splat, uncalibrated SfM). Confidence 0.5-0.8. | Pre-screening only |
| `inferred` | Value derived from object recognition, scene graph relationships, or heuristic. Confidence < 0.5. | Flagging/triage only |
| `absent` | No evidence exists for this claim. | Nothing |

### 2. Audit geometry evidence

For each entry in `geometry_evidence.json`:
- Check `measurement_type` is one of: width, height, depth, clearance, distance, area, volume, slope, flatness.
- Check `source_modality`: LiDAR/structured_light = `metric`; calibrated_photogrammetry = `metric` if `calibration_status` = verified; splat/3DGS = `estimated`; visual_estimate = `inferred`.
- Check `confidence` value exists and is in [0, 1].
- Flag any measurement with `confidence` < 0.5 as `low_confidence_geometry`.
- Flag any measurement derived from splat-only as `splat_only_geometry` — it cannot support qualification decisions alone.

**Specific geometry checks for humanoid qualification:**

| Measurement | Required for | Minimum confidence |
|---|---|---|
| Route segment width | Route access review | 0.7 (metric or calibrated) |
| Overhead clearance | Route access review | 0.7 |
| Doorway dimensions | Route access review | 0.8 |
| Workcell reach distance | Workcell risk review | 0.8 |
| Floor slope/flatness | Route access review | 0.7 |
| Aisle width | Traffic review | 0.7 |
| Choke point width | Route access review | 0.8 |
| Machine interface geometry | Workcell risk review | 0.9 (metric only) |

### 3. Audit capture QA

From `capture_qa_scorecard.json`:
- `registration_rate` < 0.7 (70% frames registered): flag as `poor_registration`. Geometry derived from this capture is unreliable.
- `colmap_coverage` < 0.8: flag as `incomplete_coverage`. Some zones may lack geometric evidence.
- `reconstruction_quality` (PSNR) < 20: flag as `low_reconstruction_quality`.
- If `qa_flags` contains any critical flag: carry forward as evidence limitation.

### 4. Audit scene graph

For each object in `scene_graph.json`:
- Check `detection_confidence`. Objects with confidence < 0.5 are `unverified_objects` — they can be mentioned but not used for capability checks.
- Check for objects that imply traffic (forklift, truck, AGV, pallet_jack) — cross-reference with `site_intake.json` traffic_notes. Detected vehicles with no intake traffic note is a gap.
- Check for objects that imply hazards (conveyor, machine_tool, chemical_container, electrical_panel) — cross-reference with blocker register. Detected hazards with no blocker entry is a gap.

### 5. Audit route graph

For each edge in `route_graph.json`:
- Check `edge_confidence`. Edges with confidence < 0.6 are `low_confidence_routes` — they cannot support route access approval.
- Check `edge_width_m` exists and is numeric. Width from splat-only is `estimated`.
- Check for route segments with no `traffic_type` classification — flag as `unclassified_traffic`.
- Check for disconnected route segments — flag as `route_continuity_gap`.

### 6. Cross-reference evidence against claims

For every capability claim in downstream artifacts (if they exist yet):
- Trace the claim to a specific evidence entry.
- Check the evidence grade meets the minimum for the claim type.
- If a claim references evidence that does not exist or is below minimum grade, flag as `unsupported_claim`.

### 7. Identify hidden zones

A hidden zone is any area that is:
- Referenced in the intake workflow but not covered in capture evidence.
- Visible in the scene graph but has no geometric measurement.
- Part of a route segment that ends at a node with no further edges (dead end in route graph that should not be a dead end per workflow).

Flag all hidden zones with estimated impact on qualification.

---

## Output

Structured evidence audit with:
- Per-artifact evidence grade summary.
- Per-measurement confidence assessment.
- Evidence gaps (specific fields, specific zones, specific measurements missing).
- Unsupported claims list.
- Hidden zone list.
- Cross-reference gaps (scene graph detections not reflected in blockers or traffic notes).
- Overall evidence sufficiency rating: `sufficient_for_qualification`, `sufficient_for_prescreen`, `insufficient`.

---

## Do not

- Clear safety or traffic risk from partial evidence. If evidence is incomplete for a safety-critical check, the check result is `unverifiable`, not `pass`.
- Write final operator-facing language. This skill produces audit findings, not reports.
- Ignore uncertainty or hidden-zone bounds.
- Treat splat/3DGS geometry as equivalent to calibrated metric geometry.
- Treat scene graph object labels as physical facts. A detected "forklift" means the model thinks it saw a forklift. It does not mean forklift traffic is active in that zone.
- Assume absence of evidence is evidence of absence. If the capture did not cover a zone, that zone's evidence state is `absent`, not `safe`.

---

## Fail-closed rules

- If `capture_qa_scorecard.json` is missing: evidence audit cannot proceed. Return `audit_status: blocked` with reason.
- If `geometry_evidence.json` is missing or empty: all geometry claims are `absent`. Set overall sufficiency to `insufficient`.
- If `route_graph.json` is missing or has zero edges: route access review cannot proceed.
- If evidence grade from intake normalization is `pre_screen`: no downstream skill may treat any claim as qualification-grade. Enforce this in the audit output.

---

## Escalation rules

- Any `unsupported_claim` that relates to safety, traffic, or reach must be escalated to `human_actions_required.json`.
- Any hidden zone in a shared-traffic or workcell area must be escalated.
- If overall evidence sufficiency is `insufficient`, the pipeline should route to `recapture_planner` before any readiness review.
