# Shared Traffic Reviewer

Use when qualification evidence includes zones where humanoid robots would share space with forklifts, reach trucks, pallet jacks, AGVs/AMRs, tow trains, or pedestrian workers. This is the most safety-critical review skill in the pipeline.

---

## Trigger

- `site_intake.json` contains `traffic_notes`, `shared_zones`, or `vehicle_types` fields.
- `route_graph.json` edges overlap with known vehicle traffic zones.
- `blocker_register.json` contains any blocker with category `traffic`, `shared_space`, or `forklift_shared_aisle`.
- The environment type is warehouse, manufacturing, dock, or any site with mixed traffic.

---

## Exact inputs

| Artifact | Required fields |
|---|---|
| `site_intake.json` | `traffic_notes`, `vehicle_types`, `shared_zones`, `zone_map` |
| `route_graph.json` | `edges[].width_m`, `edges[].traffic_type`, `edges[].confidence`, `nodes[].zone_type` |
| `geometry_evidence.json` | `aisle_widths[]`, `intersection_visibility[]`, `floor_markings[]` |
| `blocker_register.json` | All entries with category containing `traffic` or `shared_space` |
| `scene_graph.json` | Object labels related to vehicles, barriers, markings, mirrors |

---

## Required behavior

### 1. Classify every route segment by traffic type

For each edge in `route_graph.json`, determine:
- **humanoid_only**: No vehicle traffic evidence. Pedestrian or robot-only zone.
- **shared_pedestrian**: Humanoid shares with pedestrian workers only.
- **shared_forklift**: Humanoid shares with forklifts or other powered industrial trucks.
- **shared_agv**: Humanoid shares with AGVs/AMRs.
- **shared_mixed**: Multiple vehicle types plus pedestrians.
- **unknown**: Traffic type not evidenced in capture. Treat as shared_mixed for safety.

### 2. Apply clearance checks per traffic type

| Traffic type | Minimum route width | Basis |
|---|---|---|
| humanoid_only | Platform width + 1.0 m | ISO 3691-4 operating zone (0.5 m each side for 2.1 m height) |
| shared_pedestrian | Platform width + pedestrian width (~0.6 m) + 1.0 m separation | General practice |
| shared_forklift | Forklift width + platform width + 3 ft (0.9 m) minimum | OSHA letter of interpretation: 3 ft wider than largest equipment |
| shared_agv | AMR width + platform width + 1.0 m | ISO 3691-4 operating zone applied to both |
| shared_mixed | Widest vehicle + platform width + 1.2 m | Conservative combined clearance |

If aisle width is not measured (only estimated from video), mark clearance check as `unverified` and flag for recapture.

### 3. Check intersection visibility

For every intersection or junction in the route graph:
- Is there evidence of visibility aids (convex mirrors, warning lights, floor markings)?
- Is the approach sightline captured and measurable?
- If intersection visibility is not evidenced, flag as `intersection_visibility_gap`.

### 4. Check speed zone compatibility

- Humanoid walking speed ranges from 1.2-1.4 m/s across platforms.
- ISO 3691-4 critical threshold: **1.2 m/s** — above this, confined-zone-equivalent requirements apply.
- If the humanoid's operating speed in shared zones exceeds 1.2 m/s, flag as `speed_governance_required`.
- Forklift speeds in shared zones: typically 5 mph (8 km/h) general, 3 mph (5 km/h) in pedestrian zones. Forklift speed governance is a human/operational control, not something this pipeline can verify from capture evidence alone — flag as `requires_operational_verification`.

### 5. Flag high-risk traffic patterns

The following patterns are always high severity:
- **Forklift + humanoid in same aisle without physical separation**: hard_blocker unless traffic management protocol evidence exists.
- **Reach truck zone**: Single-occupancy constraint — humanoid and reach truck cannot coexist in narrow aisle. hard_blocker.
- **Dock area with active truck traffic**: hard_blocker unless explicit dock traffic management plan.
- **Tow train route crossing humanoid route**: high severity — tow trains have limited braking.
- **Blind corners without mirrors or sensors**: high severity.

### 6. Assess traffic management protocol evidence

Traffic management cannot be verified from capture geometry alone. Check whether `site_intake.json` or `human_actions_required.json` includes:
- Written traffic management plan or protocol.
- Zone separation (physical barriers, painted zones, light curtains).
- Time-based separation (humanoid operates during off-shift only).
- Signal-based coordination (traffic lights, semaphores, zone locks).

If no traffic management evidence exists and shared traffic is present, the readiness state cannot be better than `not_ready_yet` for those zones.

---

## Output

Structured shared traffic review with:
- Per-route-segment traffic classification.
- Per-segment clearance check result (pass / fail / unverified).
- Intersection visibility gaps.
- Speed governance requirements.
- High-risk pattern flags.
- Traffic management protocol evidence assessment.
- Required human actions for traffic-related blockers.

---

## Do not

- Clear shared forklift zones based on geometry alone — operational traffic management must be verified by humans.
- Assume traffic types from object detection alone (a parked forklift does not mean active forklift traffic; absence of forklifts does not mean no forklift traffic).
- Reduce severity of traffic blockers when evidence is incomplete.
- Treat time-based separation as equivalent to physical separation without human verification.
- Infer traffic patterns from a single capture session — traffic varies by shift, season, and day.

---

## Fail-closed rules

- If `traffic_notes` is missing from `site_intake.json` and the environment is warehouse or manufacturing: output `traffic_classification_unknown` and set readiness to `not_ready_yet` for all shared zones.
- If any route segment is classified as `shared_forklift` or `shared_mixed` and no traffic management protocol evidence exists: hard_blocker.
- If aisle width measurements are video-estimated only (not metric): clearance checks are `unverified` — do not pass them.

---

## Escalation rules

- Any `hard_blocker` traffic finding must appear in `human_actions_required.json` with action type `traffic_management_review`.
- Any `shared_forklift` classification must escalate to EHS/safety review — this pipeline cannot approve shared forklift-humanoid operations.
- Intersection visibility gaps must be flagged for recapture or physical site survey.

---

## Relevant standards

| Standard | Applicability |
|---|---|
| ISO 3691-4:2023 | Zone definitions, 1.2 m/s threshold, 0.5 m clearance for 2.1 m height |
| OSHA 1910.176 | General aisle clearance ("sufficient safe clearance") |
| ANSI/ITSDF B56.1 | Forklift operation in proximity to pedestrians |
| ISO/TS 15066 (via ISO 10218-2:2025) | Speed and separation monitoring formula for collaborative zones |
| ANSI/RIA R15.08 | Industrial mobile robot safety — Type C (mobile manipulator) |
