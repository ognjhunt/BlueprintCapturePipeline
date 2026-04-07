# Recapture Planner

Use when evidence gaps, failed checks, or unverifiable findings need to be converted into a concrete, ordered recapture checklist with specific capture modalities, zones, and priorities. This skill tells the capture operator exactly what to go back and capture, with what equipment, and why.

---

## Trigger

- After evidence_auditor identifies evidence gaps.
- After any review skill produces `unverifiable` findings that could be resolved with better evidence.
- When readiness state is `not_ready_yet` and recapture is the resolution path.
- When blocker register entries have `resolution_path: recapture`.

---

## Exact inputs

| Artifact | Required | Fields to inspect |
|---|---|---|
| `evidence_audit.json` | Yes | `evidence_gaps[]`, `hidden_zones[]`, `low_confidence_measurements[]` |
| `blocker_register.json` | Yes | Entries with `resolution_path` = `recapture` |
| `capture_qa_scorecard.json` | Yes | `qa_flags[]`, `coverage_zones[]`, `registration_rate` |
| `geometry_evidence.json` | Yes | Measurements with low confidence or missing measurements |
| `route_access_review.json` | If available | Segments with status `unverifiable` |
| `workcell_risk_review.json` | If available | Targets with status `unverifiable` |
| `normalized_intake.json` | Yes | `workflow`, `zone`, `environment_type` |

---

## Required behavior

### 1. Collect all recapture needs

From all input artifacts, build a complete list of what evidence is missing or insufficient:
- Evidence gaps from the audit.
- Hidden zones not covered by capture.
- Low-confidence measurements that need metric-grade re-measurement.
- Blocker register entries requiring recapture.
- Unverifiable route segments.
- Unverifiable workcell targets.

### 2. Classify each recapture need by modality

Not all evidence gaps require the same type of recapture. Classify each:

| Required evidence type | Capture modality | Equipment | Notes |
|---|---|---|---|
| Route width (metric) | LiDAR scan or tape measure with photo | LiDAR scanner or laser range finder | Splat/video insufficient for clearance checks |
| Overhead clearance | LiDAR scan or tape measure with photo | LiDAR scanner or laser range finder | Must measure at specific points, not just general ceiling height |
| Floor slope/grade | Digital level + photography | Digital inclinometer, phone level app with calibration | Place at multiple points along route segment |
| Floor flatness (F-number) | Professional floor survey | F-meter or dipstick survey | Hire floor survey contractor if FF-number critical |
| Floor condition | Close-range photography + notes | Phone camera | Document: cracks, oil, water, debris, coating condition |
| Threshold/step height | Tape measure + close-range photo | Tape measure, ruler | Measure at dock plates, door thresholds, floor transitions |
| Workcell reach distances | LiDAR scan from humanoid-standing-position | LiDAR scanner | Must capture from the position where the humanoid would stand |
| Machine interface geometry | Close-range structured light scan + measurements | Structured light scanner, calipers, tape measure | Button positions, handle heights, door clearance, fixture dimensions |
| Aisle width at choke points | LiDAR or tape measure at specific location | LiDAR scanner or tape measure | Measure at the narrowest point, not the average |
| Traffic conditions | Time-lapse video or shift observation | Camera on tripod, 4+ hours | Must capture during active operations, not off-hours |
| Hidden zone coverage | Full capture pass of uncovered area | Same modality as primary capture | May require access permission, escort, or scheduling |
| Doorway dimensions | Tape measure + photo | Tape measure | Width, height, threshold height, door type, handle type |
| Surface type transitions | Close-range photography + location notes | Phone camera | Document each surface change along route |

### 3. Prioritize recapture items

Priority order:

1. **P0 — Hard blocker resolution:** Any recapture that resolves a hard_blocker. Must be done first.
2. **P1 — Safety-critical verification:** Shared traffic zone clearances, e-stop fall zones, floor condition in active work areas.
3. **P2 — Capability-critical verification:** Route segments needed for primary workflow, workcell reach for primary manipulation targets.
4. **P3 — Completeness:** Hidden zone coverage, secondary route segments, non-critical measurements.
5. **P4 — Enhancement:** Better quality evidence for existing adequate evidence, additional angles, supplementary data.

### 4. Group by capture session

Organize recapture items into logical capture sessions:
- Group items by physical zone (minimize travel for capture operator).
- Group items by required equipment (minimize equipment changes).
- Note access requirements (badge, escort, scheduling).
- Note operational state requirements (must capture during active shift vs. can capture during off-hours).
- Estimate time per capture session.

### 5. Write clear capture instructions

Each recapture item must include:

```
Recapture Item [RC-001]
  Zone: [specific location]
  What to capture: [specific measurement or evidence needed]
  Why: [which blocker or gap this resolves, with blocker ID]
  Modality: [specific capture method]
  Equipment: [specific equipment needed]
  Instructions: [step-by-step capture procedure]
  Acceptance criteria: [how to know the capture is sufficient]
  Priority: [P0/P1/P2/P3/P4]
  Access: [any access requirements]
  Timing: [during operations / off-hours / any time]
```

### 6. Include acceptance criteria

Every recapture item must have acceptance criteria so the capture operator knows when they have sufficient evidence:
- "Width measurement at choke point with confidence >= 0.7 from metric source."
- "Close-range photo showing all 4 machine control buttons with ruler for scale."
- "LiDAR scan of workcell from humanoid standing position, covering all manipulation targets."
- "4-hour time-lapse showing all vehicle types that use this aisle during active shift."

---

## Output

`recapture_plan.json` with:
- Ordered list of recapture items with full detail.
- Grouped by capture session.
- Priority distribution (count per priority level).
- Estimated total recapture effort.
- Equipment list (combined across all items).
- Access requirements summary.
- `access_pending`: `true` when any recapture item still needs special access that has not been confirmed, even if other items are open-access.
- Expected impact: which blockers/gaps will be resolved by this recapture.

---

## Do not

- Ask for recapture without a cited reason. Every recapture item must trace to a specific evidence gap, blocker, or unverifiable finding.
- Treat optional cosmetics as blocking evidence. Better-looking splat is not a recapture need unless the visual quality actually blocks a specific check.
- Suggest splat-only remediation for missing intake or QA. Splat cannot fix a missing workflow description.
- Suggest splat-only remediation for metric-critical measurements. If a clearance check needs metric evidence, the recapture must produce metric evidence.
- Request full site recapture when only specific zones/measurements are needed. Be surgical.
- Ignore access requirements. If a zone requires badge access or escort, the capture operator needs to know before going to the site.
- Create recapture items for things that cannot be resolved by capture (e.g., traffic management protocols, e-stop plans, LOTO procedures) — those are `human_review` resolution paths, not recapture.

---

## Fail-closed rules

- If `evidence_audit.json` is missing: recapture plan cannot be generated. Return error.
- If any recapture item requires special access and access is not confirmed: flag the entire recapture plan as `access_pending`, including mixed-access plans that also contain open-access items.
- If the recapture plan has > 10 P0 items: consider whether the capture was fundamentally inadequate and a full re-capture is more efficient than targeted recapture.

---

## Escalation rules

- If recapture requires access to restricted zones: escalate to site owner for access authorization.
- If recapture requires specialized equipment not available to the capture operator: escalate to project manager for equipment sourcing.
- If the volume of recapture suggests the original capture was fundamentally inadequate: recommend reviewing the capture protocol before conducting recapture.
- P0 recapture items that resolve safety-related hard_blockers should be the first items communicated to the project team.
