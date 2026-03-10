# Humanoid Route Access Reviewer

Use when route access and mobility constraints need a humanoid-specific assessment. This skill evaluates every route segment for width clearance, overhead clearance, floor condition, surface transitions, and access confidence. It works segment-by-segment and never declares a route "clear" without specific measured evidence.

---

## Trigger

- After evidence_auditor has completed and route_graph.json has been audited.
- When capability_envelope_writer needs route traversal inputs.
- When site_readiness_reviewer needs route-level findings.

---

## Exact inputs

| Artifact | Required | Fields to inspect |
|---|---|---|
| `route_graph.json` | Yes | `nodes[]`, `edges[]` — each edge needs: `edge_id`, `width_m`, `overhead_m`, `surface_type`, `grade_deg`, `confidence`, `traffic_type`, `threshold_mm` |
| `geometry_evidence.json` | Yes | Route-related measurements — widths, clearances, slopes, thresholds |
| `scene_graph.json` | If available | Objects along route segments — obstacles, vehicles, barriers |
| `normalized_intake.json` | Yes | `target_platform`, `evidence_grade` |
| `blocker_register.json` | Yes | Route-related blockers |

---

## Required behavior

### 1. Classify every route segment

For each edge in `route_graph.json`, produce a segment assessment:

```
Segment [edge_id]: [node_a] -> [node_b]
  Width: [measured_value] m (confidence: [value], source: [modality])
  Overhead: [measured_value] m (confidence: [value], source: [modality])
  Surface: [type]
  Grade: [degrees]
  Threshold: [mm] at [location]
  Traffic: [classification]
  Status: [pass / conditional / fail / unverifiable]
```

### 2. Apply clearance checks

**Width clearance per traffic type:**

| Traffic classification | Required width | Basis |
|---|---|---|
| humanoid_only | Platform width + 1.0 m | ISO 3691-4: 0.5 m each side for 2.1 m height |
| shared_pedestrian | Platform width + 0.6 m (pedestrian) + 1.0 m separation | General practice + ISO 3691-4 |
| shared_agv | Platform width + AMR width + 1.0 m | ISO 3691-4 applied to both |
| shared_forklift | Platform width + forklift width + 0.9 m (3 ft) | OSHA/ANSI B56.1 |
| shared_mixed | Widest vehicle + platform width + 1.2 m | Conservative combined |
| unknown | Treat as shared_mixed | Fail-closed |

**Platform width values (shoulder width):**
| Platform | Width |
|---|---|
| Digit | ~0.55 m |
| Figure 02 | ~0.50 m |
| Apollo | ~0.55 m |
| NEO | ~0.45 m |
| Cross-platform conservative | 0.60 m |

**Overhead clearance:**
- Minimum: platform height + 0.30 m
- Platform heights range 1.68-1.75 m, so minimum overhead is 1.98-2.05 m.
- Cross-platform conservative minimum: 2.10 m.

**Floor grade:**
- Maximum slope for bipedal locomotion: 5 degrees (general).
- Maximum slope during manipulation: 3 degrees (tighter due to balance requirements).
- Cross-slope (lateral tilt): maximum 3 degrees.

**Threshold/step:**
- Maximum step height: 50 mm for current bipedal platforms.
- Dock plates: transition must be < 50 mm step AND < 5 degree slope.
- Expansion joints, floor patches, and cable runs across path must be < 25 mm.

### 3. Assess surface transitions

Surface changes along a route segment are bipedal stability concerns:
- Concrete to metal (dock plate): coefficient of friction change.
- Dry to wet: slip hazard.
- Smooth to textured: may affect foot sensor feedback.
- Indoor to outdoor (if applicable): weather exposure, surface change.

Flag all surface transitions with location and type.

### 4. Identify choke points

A choke point is the minimum-width location along any route segment. For each:
- Record the width measurement, confidence, and source.
- Compare against the applicable clearance requirement for that segment's traffic type.
- If the choke point width is < required clearance: `fail`.
- If the choke point width is within 0.2 m of required clearance: `marginal` — flag for verification.

### 5. Assess route continuity

Check that the route graph forms a complete path from task start to task end:
- No disconnected segments.
- No dead ends that should not be dead ends (per workflow).
- All return paths are assessed (the humanoid must also get back).
- Charging station path exists if runtime requires mid-shift charging.

### 6. Assess doorway and passage constraints

For each doorway, gate, or passage point in the route:
| Check | Requirement | Notes |
|---|---|---|
| Width | Platform width + 0.6 m minimum | Standard doors (0.9 m) usually sufficient |
| Height | Platform height + 0.15 m minimum | Standard doors (2.0 m) usually sufficient |
| Door type | Automatic, push, pull, sliding | Push/pull doors require manipulation — check hand DOF |
| Door force | < platform manipulation force | Heavy fire doors may exceed capability |
| Threshold | < 25 mm step | Door thresholds and weather strips |
| Security | Badge access, key, code | Does the humanoid have credentials? |

### 7. Confidence classification

For each segment, classify the overall assessment confidence:
- `verified`: All measurements are metric-grade (confidence >= 0.7).
- `estimated`: Measurements exist but are from splat/estimated sources.
- `unverified`: No measurement exists for this segment.

A route cannot be cleared for qualification if > 30% of segments are `unverified`.

---

## Output

`route_access_review.json` with:
- Per-segment assessment (status, confidence, findings).
- Choke point list with measurements and clearance comparison.
- Surface transition list.
- Doorway/passage assessment.
- Route continuity check result.
- Summary statistics: segments_passed, segments_conditional, segments_failed, segments_unverified.
- Overall route readiness: `clear`, `partially_clear`, `blocked`, `unverifiable`.

---

## Do not

- Claim safe traversal without measured support. "Appears wide enough" is not evidence.
- Ignore low-confidence route edges. They are potential blockers, not "probably fine."
- Convert pre-screen capture into mobility approval.
- Treat splat-derived widths as equivalent to metric widths for clearance checks.
- Assume doors are open or openable without evidence of the mechanism.
- Clear a route segment with `unknown` traffic classification — it must be treated as shared_mixed.
- Round measurements favorably. If the measurement is 1.49 m and the requirement is 1.50 m, it fails.

---

## Fail-closed rules

- If `route_graph.json` is missing or empty: route review cannot proceed. Return error.
- If a route segment has no width measurement: status = `unverifiable`. Cannot pass.
- If > 50% of segments are `unverifiable`: overall route readiness = `unverifiable`.
- If any segment in a shared-forklift zone fails width clearance: `hard_blocker`.
- If evidence grade from intake is `pre_screen`: all route assessments are caveated as pre-screen only.
- If overhead clearance is unmeasured: assume insufficient until proven otherwise.

---

## Escalation rules

- Any `fail` on a segment in a shared-traffic zone: escalate to shared_traffic_reviewer and `human_actions_required.json`.
- Any choke point that is `marginal` (within 0.2 m of requirement): recommend metric re-measurement.
- Route discontinuity that affects the primary workflow path: escalate to recapture_planner.
- Door/passage security constraints (badge access): escalate to site owner for humanoid credential planning.
