# Humanoid Workcell Risk Reviewer

Use when workcell-specific risks need to be assessed for humanoid task execution. A workcell is any bounded work area where the humanoid performs manipulation tasks — a pick station, machine tool, assembly fixture, pack station, inspection point, or conveyor interface. This skill checks reach, force, articulation, occlusion, hidden conditions, floor condition, and machine interface compatibility within the workcell.

---

## Trigger

- When `task_scope_record.json` contains any task that requires manipulation (not just locomotion/transport).
- When `geometry_evidence.json` contains measurements for a workcell zone.
- When blocker register contains entries with category `geometry_reach`, `machine_interface`, or `workflow_timing`.

---

## Exact inputs

| Artifact | Required | Fields to inspect |
|---|---|---|
| `task_scope_record.json` | Yes | `tasks[]` — task type, targets, cycle time, force requirements, articulation type |
| `geometry_evidence.json` | Yes | Workcell measurements — target distances, heights, depths, clearances |
| `scene_graph.json` | Yes | Objects in workcell zone — manipulation targets, obstacles, equipment |
| `capability_envelope.json` | Yes | Reach and manipulation feasibility per target |
| `blocker_register.json` | Yes | Workcell-related blockers |
| `normalized_intake.json` | Yes | `target_platform`, `environment_type` |

---

## Required behavior

### 1. Map workcell geometry against platform envelope

For each workcell identified in the task scope:

**Reach assessment:**
- Measure distance from expected humanoid standing position to each manipulation target.
- Compare against platform arm reach (from `humanoid_platform_reference`).
- Check vertical reach: minimum height (floor-level tasks) and maximum height (overhead tasks).
- Check depth reach: how far into a shelf, machine, or fixture the platform must reach.
- Flag targets outside verified reach envelope as `reach_exceeded` or `reach_unverified`.

**Platform-specific reach values:**
| Platform | Max forward reach | Min height | Max height | Notes |
|---|---|---|---|---|
| Digit | ~0.7 m | ~0.1 m | ~1.8 m | 4-DOF arms, limited flexibility |
| Figure 02 | ~0.8 m (est.) | ~0.1 m | ~1.9 m | Demonstrated 5 mm accuracy |
| Apollo | ~0.8 m (est.) | ~0.1 m | ~1.9 m | Specs limited |
| NEO | ~0.7 m (est.) | ~0.1 m | ~1.8 m | Very light (30 kg), stability under load uncertain |

### 2. Assess manipulation feasibility per target

For each manipulation target in the workcell:

| Check | What to verify | Evidence source |
|---|---|---|
| Weight | Target object weight vs. platform payload | `task_scope_record.json` task weight, scene graph object dimensions |
| Grip geometry | Object shape/size vs. gripper/hand compatibility | Scene graph bounding box, task articulation type |
| Articulation | Required manipulation complexity vs. platform hand DOF | Task scope articulation type vs. platform reference |
| Placement precision | Required tolerance vs. platform demonstrated accuracy | Task scope tolerance requirement |
| Force | Required manipulation force vs. platform sustained force | Task scope force requirement |
| Orientation | Required approach angle vs. workcell geometry | Geometry evidence, scene graph spatial relationships |

### 3. Assess workcell visibility and occlusion

- Identify obstacles between the humanoid's expected position and manipulation targets.
- Check whether the humanoid's own body (arms, torso) occludes sensor views during task execution.
- Check for objects above the humanoid that may not be visible (overhead conveyors, utilities, lights).
- Flag any manipulation target that is not fully visible from the humanoid's sensor vantage point as `occlusion_risk`.

### 4. Assess floor condition in workcell

Workcell floors are often worse than general facility floors due to:
- Coolant, oil, or cutting fluid (manufacturing cells).
- Water or condensation (dock areas, cold storage).
- Debris (packaging materials, broken containers).
- Wear (high-traffic zones around machines).

Check `geometry_evidence.json` for floor condition measurements or `site_intake.json` for floor condition notes. Flag:
- `floor_contamination`: Liquid, oil, or debris on floor.
- `floor_damage`: Cracks, holes, spalling, uneven patches.
- `floor_slope`: Slope > 3 degrees in workcell (tighter than route requirement due to balance during manipulation).

### 5. Assess machine interface compatibility (manufacturing only)

If the workcell involves machine tending (CNC, press, injection mold, etc.):

| Check | What to verify | Severity if unverified |
|---|---|---|
| Door/guard operation | How does the machine door open? Force required? Handle type? | hard_blocker if uncaptured |
| Button/control locations | Where are start/stop/e-stop buttons? Height? | hard_blocker if uncaptured |
| Fixture interface | How does the part load into the fixture? Clamp type? Force? | hard_blocker if uncaptured |
| Chuck/collet interface | Part insertion geometry and force | hard_blocker if uncaptured |
| Cycle coordination | How does the humanoid know when the machine cycle is complete? Signal type? | high if undefined |
| Coolant management | Is coolant spray directed at the humanoid position? | high if unverified |
| Chip clearing | Are metal chips in the humanoid foot/sensor path? | medium |

**Machine interface measurements require metric-grade evidence (confidence >= 0.9).** Splat or estimated measurements are NOT sufficient for machine interface verification.

### 6. Assess cycle time feasibility

For tasks with cycle time or takt time requirements:
- Compare humanoid estimated task time against required cycle time.
- Include: approach time + manipulation time + placement time + return time.
- Add contingency for sensor processing and decision latency (~0.5-1.0 s per decision point).
- If humanoid cycle time > 80% of required takt time: flag as `timing_risk`.
- If humanoid cycle time > required takt time: flag as `timing_blocker`.

**Real-world reference:** Figure 02 demonstrated 2-second placement within 37-second load cycle at BMW. This is the current benchmark for manufacturing humanoid cycle times.

### 7. Identify hidden conditions

Hidden conditions in a workcell are states that are not visible in the capture but affect task execution:
- Machine fault states (indicator lights, error codes).
- Part presence/absence sensors.
- Fixture clamping state.
- Coolant level/pressure.
- Tool wear state.

These cannot be determined from capture evidence. Flag each as `hidden_condition` with a note on what information would be needed.

---

## Output

`workcell_risk_review.json` with:
- Per-workcell reach assessment.
- Per-target manipulation feasibility.
- Occlusion risk findings.
- Floor condition findings.
- Machine interface compatibility (if applicable).
- Cycle time feasibility.
- Hidden conditions list.
- Workcell-level risk summary: `low_risk`, `moderate_risk`, `high_risk`, `unverifiable`.

---

## Do not

- Infer safe manipulation from object labels alone. A detected "button" does not mean the humanoid can press it — position, force, and approach geometry must be verified.
- Treat missing geometry as a pass. Missing = unverifiable.
- Rewrite the overall readiness state. This skill produces workcell-level findings for the site readiness reviewer to synthesize.
- Assume machine interface compatibility without metric evidence. Machine tending is the highest-risk humanoid workcell task.
- Clear coolant/oil/debris floor conditions from video appearance. Floor contamination must be verified on-site.
- Accept estimated cycle times without contingency. Real-world humanoid task times include sensor latency, decision time, and error recovery.

---

## Fail-closed rules

- If `task_scope_record.json` has manipulation tasks but `geometry_evidence.json` has no workcell measurements: all reach checks are `unverifiable`. Workcell risk = `high_risk`.
- If machine interface tasks exist but machine interface geometry is not captured at metric grade: `hard_blocker`.
- If floor condition in workcell is unknown: flag as `floor_condition_unknown`, set workcell risk to at least `moderate_risk`.
- If hidden conditions exist that affect safety (machine fault states, pressurized systems): escalate to human review.

---

## Escalation rules

- Any `hard_blocker` in workcell: escalate to `human_actions_required.json` and `blocker_register.json`.
- Machine interface incompatibility: escalate to OEM/integrator for platform-specific evaluation.
- Cycle time blocker on production line: escalate to production engineering — cannot be resolved by the qualification pipeline.
- Hidden conditions in safety-critical machines: escalate to site safety engineer.
