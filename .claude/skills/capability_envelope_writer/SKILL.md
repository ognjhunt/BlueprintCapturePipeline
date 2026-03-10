# Capability Envelope Writer

Use when capability claims need to be bounded to measured evidence from the qualification pipeline, cross-referenced against humanoid platform specifications. Every capability statement must be traceable to a specific measurement and a specific platform constraint. Unbounded capability claims are not allowed.

---

## Trigger

- After evidence_auditor and blocker_taxonomist have completed.
- When qualification needs to determine whether specific tasks are within humanoid capability.
- When readiness_report_writer needs capability language to include in the report.

---

## Exact inputs

| Artifact | Required | Fields to inspect |
|---|---|---|
| `geometry_evidence.json` | Yes | `measurements[]` — all width, height, depth, distance, clearance, slope measurements |
| `route_graph.json` | Yes | `edges[]` — width, surface, grade; `nodes[]` — zone type |
| `scene_graph.json` | Yes | `objects[]` — manipulation targets, their position, dimensions, articulation type |
| `task_scope_record.json` | Yes | `tasks[]` — task type, targets, cycle time, force requirements |
| `capability_checks.json` | Yes | `checks[]` — result, evidence source, confidence |
| `normalized_intake.json` | Yes | `target_platform`, `evidence_grade` |
| `blocker_register.json` | Yes | All geometry/reach/platform blockers |

---

## Capability domains

Write bounded statements for each of these domains. Every statement must follow the format:

> **[DOMAIN]**: [Bounded claim]. Evidence: [artifact.field, confidence]. Platform constraint: [spec value]. Status: [pass / conditional / fail / unverifiable].

### 1. Route traversal

For each route segment in `route_graph.json`:

| Check | Platform reference | Minimum evidence |
|---|---|---|
| Width clearance | Platform shoulder width + 1.0 m (ISO 3691-4 operating zone) | `edge_width_m` with confidence >= 0.7, metric or calibrated source |
| Overhead clearance | Platform height + 0.3 m | Overhead measurement with confidence >= 0.7 |
| Floor grade | Platform max slope tolerance (typically <= 5 degrees for bipedal) | Slope measurement with confidence >= 0.7 |
| Floor surface | Platform foot type compatibility (rubber, hard floor requirement) | Surface type from capture or intake |
| Step/threshold | Platform max step height (typically <= 50 mm for bipedal) | Threshold measurement with confidence >= 0.8 |
| Turning radius | Platform turning capability at segment junctions | Junction geometry from route graph |

Status mapping:
- All checks pass with metric evidence: `pass`
- All checks pass with estimated evidence: `conditional` (note: "conditional on metric verification")
- Any check fails: `fail`
- Any check cannot be evaluated: `unverifiable`

### 2. Reach envelope

For each manipulation target in `task_scope_record.json`:

| Check | Platform reference | Minimum evidence |
|---|---|---|
| Horizontal reach | Platform max arm reach (Digit: ~0.7 m, Figure 02/Apollo: ~0.8 m estimated) | Target distance measurement with confidence >= 0.8 |
| Vertical reach (floor) | Platform min reach height (varies by platform, ~0.1 m typical) | Target height measurement |
| Vertical reach (overhead) | Platform max reach height (~1.9-2.1 m depending on platform) | Target height measurement |
| Depth reach | Platform reach into confined space | Target depth measurement |

If `target_platform` is specified in intake, use that platform's specs from `humanoid_platform_reference`. If not specified, use the cross-platform envelope (worst case for conservative bounds, best case to indicate any-platform feasibility).

### 3. Manipulation feasibility

For each manipulation target:

| Check | Platform reference | Minimum evidence |
|---|---|---|
| Object weight | Platform payload capacity | Object weight from task scope or estimation |
| Grip geometry | Platform hand DOF and gripper type | Object dimensions and grip requirements |
| Placement precision | Platform demonstrated accuracy | Required tolerance from task scope |
| Force requirement | Platform max sustained force | Force requirement from task scope (e.g., door opening, lever pulling) |
| Articulation complexity | Platform hand DOF | Target articulation type (knob, lever, button, handle, latch) |

**Articulation complexity mapping:**

| Articulation type | Minimum hand DOF | Current platform feasibility |
|---|---|---|
| Simple grasp (tote, box, flat part) | 4 (parallel gripper) | All platforms |
| Button press | 4 | All platforms |
| Lever pull | 6 | Most platforms except simple grippers |
| Knob turn | 12+ | Only dexterous hands (Figure 02, NEO, Optimus Gen 3) |
| Door handle (round) | 10+ | Dexterous hands only |
| Door handle (lever) | 6 | Most platforms |
| Tool use (screwdriver, wrench) | 16+ | Only highest-DOF hands (Figure 02: 16/hand, NEO: 22/hand) |

### 4. Occupancy and choke points

For each zone in the route/scene graph:
- Calculate occupancy: humanoid footprint vs. available floor area.
- Identify choke points: minimum width along any route segment.
- Flag choke points that are < platform width + 0.5 m (cannot pass safely).
- Flag zones where humanoid + worker cannot coexist (width < platform width + pedestrian width + 1.0 m separation).

### 5. Visibility and sensing

For each task and route segment:
- Does the humanoid sensor suite provide adequate coverage? (Check sensor FOV from platform reference.)
- Are there known occlusion sources (pillars, racking, machinery) that block the humanoid's sensors during task execution?
- Is ambient lighting within platform sensor operating range? (Depth cameras typically need > 100 lux; LiDAR works in dark.)

### 6. Duration and shift coverage

For the overall workflow:
- Total estimated task cycle time vs. platform battery life.
- Charging time vs. available window (between shifts, breaks).
- Number of platforms needed for continuous coverage if required.

---

## Output

`capability_envelope.json` with:
- Per-domain capability statements.
- Per-task feasibility assessment.
- Per-route-segment traversal assessment.
- Overall capability summary: `within_envelope`, `partially_within_envelope`, `outside_envelope`, `unverifiable`.
- List of capability-driven blockers (to feed back to blocker_taxonomist).

---

## Do not

- Infer pass/fail on unsupported geometry. If the measurement does not exist, the status is `unverifiable`, never `pass`.
- Invent measurements. No "approximately 2 meters" unless an actual measurement exists.
- Convert bounded evidence into deployment approval. "Within reach envelope based on measured geometry" is not "safe to deploy."
- Use marketing claims from platform specs as evidence of field capability.
- Assume worst-case or best-case without stating which and why.
- Write capability statements that are not traceable to specific evidence + specific platform constraints.

---

## Fail-closed rules

- If `geometry_evidence.json` is missing or empty: all capability checks are `unverifiable`.
- If `task_scope_record.json` is missing: reach and manipulation checks cannot proceed. Return error.
- If `target_platform` is not specified and the task requires platform-specific checks (e.g., hand DOF): output range across platforms with a note that platform selection is required.
- If `evidence_grade` from intake normalization is `pre_screen`: all capability statements must be prefixed with "Pre-screen only — not for qualification decisions."
- Any manipulation target with unknown weight defaults to `unverifiable` for payload check.

---

## Escalation rules

- Any `fail` status on a safety-critical check (route clearance in shared traffic zone, force output in collaborative zone): escalate to `human_actions_required.json`.
- Any `outside_envelope` overall status: readiness cannot be better than `not_ready_yet`.
- Any `unverifiable` status on 3+ domains: recommend recapture before proceeding.
