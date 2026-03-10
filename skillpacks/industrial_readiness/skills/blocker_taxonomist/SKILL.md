# Blocker Taxonomist

Use when raw qualification issues, evidence gaps, and audit findings need to be normalized into a structured blocker register with industrial humanoid-specific categories, severities, and resolution paths.

---

## Trigger

- After evidence_auditor completes.
- When new blockers are discovered by any review skill.
- When recapture results require blocker re-evaluation.

---

## Exact inputs

| Artifact | Required | Fields to inspect |
|---|---|---|
| `evidence_audit.json` | Yes | `evidence_gaps[]`, `unsupported_claims[]`, `hidden_zones[]`, `cross_reference_gaps[]` |
| `capability_checks.json` | Yes | `checks[]` where `result` is `fail` or `unverifiable` |
| `geometry_evidence.json` | Yes | Measurements with `confidence` < threshold |
| `site_intake.json` | Yes | `known_blockers[]`, `environment_type` |
| `scene_graph.json` | If available | Objects implying hazards or traffic |

---

## Blocker Taxonomy

Every blocker must be assigned exactly one primary category and zero or more secondary categories from this taxonomy. Categories are specific to industrial humanoid site qualification.

### Primary categories

| Category | Description | Examples |
|---|---|---|
| `geometry_clearance` | Route or zone dimensions insufficient or unverified for humanoid passage | Aisle too narrow, overhead obstruction, doorway height, choke point width |
| `geometry_reach` | Manipulation target beyond verified reach envelope | Shelf too high, fixture too deep, machine interface out of range |
| `geometry_floor` | Floor condition creates bipedal stability or safety risk | Slope > 5 degrees, F-number < FF25, standing water, oil, debris, dock plate gap |
| `traffic_shared` | Humanoid shares space with other vehicles without verified traffic management | Forklift aisle, AGV crossing, reach truck zone, dock traffic |
| `traffic_pedestrian` | Humanoid shares space with pedestrian workers without verified separation | Shared pick zone, mixed aisle, break area proximity |
| `safety_estop` | E-stop plan missing, incomplete, or does not address humanoid fall hazard | No e-stop plan, Category 0 only, no fall zone defined |
| `safety_force` | Force/impact concerns for collaborative operation | Platform force output exceeds ISO/TS 15066 limits, no force limiting documented |
| `safety_guarding` | Existing safety guarding incompatible with humanoid access | Light curtains, safety fencing, interlock configuration |
| `safety_fall` | Fall zone, fall energy, or dynamic stability concern | No fall zone calculation, high CG, uneven terrain |
| `environmental` | Environmental conditions outside humanoid operating envelope | Cold storage, heat, humidity, hazardous atmosphere, outdoor exposure |
| `machine_interface` | Machine interaction geometry or force not verified | Button positions unknown, door force unknown, fixture geometry uncaptured |
| `workflow_ambiguity` | Workflow not specific enough to scope qualification | Vague task description, missing subtask decomposition, undefined handoff points |
| `workflow_timing` | Cycle time, takt time, or throughput mismatch | Humanoid slower than line speed, insufficient buffer between tasks |
| `systems_integration` | Connected system interface undefined or unverified | WMS/WES integration, PLC handshake, conveyor coordination, signal protocol |
| `capture_quality` | Capture evidence insufficient for the claim it supports | Low registration, poor coverage, splat-only geometry, low-confidence measurements |
| `capture_coverage` | Zone or feature not captured | Hidden zone, restricted area, occluded equipment, night/shift-variant conditions |
| `privacy_security` | Privacy or security constraint limits evidence collection | Faces in capture, proprietary equipment visible, restricted area |
| `non_routine` | Non-routine operation risk unaddressed | Jam clearing, machine recovery, exception handling, spill response |
| `loto_maintenance` | LOTO or maintenance procedure does not include humanoid | Missing LOTO for humanoid energy sources, no maintenance access plan |
| `infrastructure` | Facility infrastructure insufficient | Electrical capacity, network coverage, charging infrastructure, floor load |
| `platform_limitation` | Target humanoid platform cannot perform the required task | Payload exceeded, DOF insufficient, sensor limitation |

### Secondary categories

A blocker can have secondary categories when it spans concerns. Example: a narrow aisle with forklift traffic is primary `traffic_shared`, secondary `geometry_clearance`.

---

## Severity levels

| Severity | Definition | Pipeline effect |
|---|---|---|
| `hard_blocker` | Cannot proceed to readiness without resolution. | Readiness = `not_ready_yet`. No downstream skill can override. |
| `high` | Significant risk. Requires human review and likely recapture or scope change. | Readiness cannot be better than `risky`. |
| `medium` | Material concern. Should be addressed but may not block pre-screening. | Noted in readiness report. May be acceptable for pre-screen. |
| `low` | Minor concern. Document and track. | Noted. Does not affect readiness state. |
| `informational` | Context note, not a blocker. | Informational only. |

### Severity assignment rules

- **Always hard_blocker:**
  - Missing e-stop plan in shared space.
  - Active forklift traffic in humanoid zone without traffic management protocol.
  - Reach truck zone overlap with humanoid route.
  - Takt time mismatch (humanoid cannot keep up with production line).
  - Platform payload exceeded for required task.
  - Hazardous atmosphere in operating zone.
  - Machine interface force exceeds platform capability.

- **Never lower than high:**
  - Any geometry measurement that is splat-only and safety-critical.
  - Any shared traffic zone without physical separation evidence.
  - Any hidden zone in a workcell or traffic area.
  - Missing LOTO procedure for humanoid maintenance.
  - Missing fall zone calculation.

- **Conservative default:** If evidence is incomplete, severity goes up, not down. An unmeasured aisle in a forklift zone is `hard_blocker`, not `medium`.

---

## Required behavior

### 1. Collect raw blocker sources

Gather all potential blockers from:
- `evidence_audit.json` gaps and unsupported claims.
- `capability_checks.json` failures.
- `site_intake.json` known blockers.
- Scene graph hazard detections.
- Any review skill findings.

### 2. Deduplicate and normalize

- Merge duplicates (same zone + same concern = one blocker).
- Assign primary and secondary categories.
- Assign severity per the rules above.
- Preserve source evidence references (artifact name, field path, measurement ID).

### 3. Assign resolution path

Each blocker must include a `resolution_path`:
- `recapture`: Need better evidence (specific capture modality noted).
- `scope_change`: Need to adjust qualification scope to exclude this area/task.
- `site_modification`: Need physical site change (traffic management, floor repair, guarding).
- `human_review`: Need human expert evaluation (EHS, safety, integrator).
- `platform_change`: Need different humanoid platform or configuration.
- `oem_consultation`: Need OEM/integrator input on platform capability.
- `not_resolvable`: Cannot be resolved within current qualification — stop evaluation for this scope.

### 4. Link to standards

For each blocker, check `curated_standards.json` for matching guidance entries. Link the guidance entry ID if a match exists.

---

## Output

`blocker_register.json` with entries structured as:

```json
{
  "blocker_id": "BLK-001",
  "summary": "Aisle 3B width unverified — active forklift traffic",
  "primary_category": "traffic_shared",
  "secondary_categories": ["geometry_clearance"],
  "severity": "hard_blocker",
  "zone": "Aisle 3B, Racking Zone East",
  "evidence_sources": [
    {"artifact": "geometry_evidence.json", "field": "measurements[12]", "issue": "width confidence 0.4, splat-only"},
    {"artifact": "scene_graph.json", "field": "objects[47]", "issue": "forklift detected, confidence 0.82"}
  ],
  "resolution_path": "recapture",
  "resolution_detail": "Metric measurement of aisle width required. Traffic management protocol must be provided by site owner.",
  "standards_references": ["route-clearance-envelope", "shared-space-risk"],
  "escalation": "traffic_management_review"
}
```

---

## Do not

- Collapse multiple distinct blockers into one generic item. Every blocker gets its own entry.
- Drop evidence provenance. Every blocker must trace to specific artifact fields.
- Reclassify high-risk evidence as informational to make the readiness look better.
- Assign `low` severity to anything in a shared-traffic or safety context without explicit justification.
- Create blockers without resolution paths. Every blocker must tell the human what to do next.
- Invent blockers that are not supported by evidence or audit findings.

---

## Fail-closed rules

- If `evidence_audit.json` is missing: cannot build blocker register. Return error.
- If a blocker has no evidence source: it is either from intake `known_blockers` (valid, mark source as `intake`) or it is invented (invalid, discard).
- If severity cannot be determined due to missing evidence: default to `high`, not `medium`.
- If `environment_type` is missing: cannot apply site-specific blocker categories. Use generic categories and flag for intake re-normalization.

---

## Escalation rules

| Condition | Escalation action |
|---|---|
| Any `hard_blocker` | Entry in `human_actions_required.json` |
| Any `safety_*` category | Route to EHS/safety review |
| Any `traffic_shared` with forklift | Route to traffic management review |
| Any `loto_maintenance` | Route to site safety engineer |
| Any `platform_limitation` | Route to OEM/integrator evaluation |
| 3+ `high` severity in same zone | Escalate zone to `not_ready_yet` regardless of individual resolution paths |
