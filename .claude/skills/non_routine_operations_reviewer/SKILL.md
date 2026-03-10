# Non-Routine Operations Reviewer

Use when qualification scope includes exception handling, jam clearing, recovery procedures, mode changes, e-stop behavior, maintenance access, or any non-steady-state operation involving a humanoid. Non-routine operations are where most industrial robot accidents occur (OSHA). This skill must be conservative.

---

## Trigger

- `site_intake.json` contains `non_routine_modes`, `exception_handling`, or `recovery_procedures` fields.
- `task_scope_record.json` includes tasks with `task_type` of: jam_clearing, exception_handling, machine_recovery, mode_change, maintenance_assist, or reset.
- `blocker_register.json` contains blockers with category `workflow_ambiguity` that relate to non-standard operations.
- Any qualification scope includes operations that are not steady-state pick/place/transport.

---

## Exact inputs

| Artifact | Required fields |
|---|---|
| `site_intake.json` | `non_routine_modes`, `exception_handling`, `recovery_procedures`, `estop_plan` |
| `task_scope_record.json` | `tasks[].task_type`, `tasks[].exception_conditions`, `tasks[].recovery_steps` |
| `blocker_register.json` | All entries with category `workflow_ambiguity` or `safety` |
| `capability_checks.json` | `checks[].non_routine_feasibility` if present |
| `human_actions_required.json` | Existing human action items related to non-routine operations |

---

## Required behavior

### 1. Classify non-routine operation types

For each non-routine operation in scope, classify as:

| Type | Description | Humanoid feasibility (current gen) |
|---|---|---|
| `jam_clearing` | Remove stuck part from conveyor, machine, or gripper | Low-Medium. Requires force control, unpredictable geometry. |
| `machine_recovery` | Reset machine after fault, clear error, restart cycle | Low. Requires HMI interaction, specific button sequences. |
| `conveyor_exception` | Handle misrouted, damaged, or jammed items on conveyor | Medium. Similar to pick-and-place but unpredictable. |
| `estop_recovery` | Resume operation after emergency stop | Not humanoid task. Human-only. |
| `mode_change` | Switch between autonomous, supervised, or manual mode | System-level. Not a manipulation task. |
| `maintenance_assist` | Support maintenance worker with part holding, tool pass | Low. Requires collaborative task coordination. |
| `spill_response` | Respond to spilled product, broken container | Not humanoid task. Human-only. |
| `unknown_exception` | Unclassified non-routine event | Treat as human-only until classified. |

### 2. Assess e-stop implications for humanoids

E-stop is uniquely problematic for humanoids because:
- **Category 0 stop (immediate power removal):** The humanoid falls uncontrolled. A 65-72.6 kg robot falling from 1.75 m CG has ~1,245 J impact energy. This is a serious personnel hazard.
- **Category 1 stop (controlled deceleration then power removal):** Preferred for humanoids — allows controlled crouch/kneel before power removal. But requires the robot's functional safety system to be operational during the fault condition that triggered the e-stop.
- **ISO 13850 and IEC 60204-1 limit emergency stop to Categories 0 and 1 only.** Category 2 (stop with power maintained) is not valid for e-stop.

For each zone where the humanoid operates, check:
- Is the e-stop plan documented? If not, flag as `estop_plan_missing`.
- Does the e-stop plan account for humanoid fall hazard? If not, flag as `estop_fall_hazard_unaddressed`.
- Is there a fall zone defined? (Per draft ISO 25785-1, fall zone calculations based on manufacturer specs will be required.) If not, flag as `fall_zone_undefined`.
- Is there a clear post-e-stop recovery procedure that includes humanoid restart? If not, flag as `estop_recovery_undefined`.

### 3. Assess LOTO (Lock-Out / Tag-Out) requirements

OSHA 29 CFR 1910.147 requires LOTO for maintenance and jam clearing. For humanoids, hazardous energy sources include:
- Electrical power (main and battery).
- Pneumatic/hydraulic energy (if applicable).
- Stored mechanical energy (springs, gravity — a powered-down humanoid may still fall).
- Battery energy (lithium-ion fire risk during maintenance).

Check:
- Is there a LOTO procedure that includes the humanoid as an energy source? If not, flag as `loto_procedure_missing`.
- Does the LOTO procedure address battery isolation? If not, flag as `battery_loto_unaddressed`.

### 4. Assess human-robot handoff for exceptions

When a humanoid encounters an exception it cannot handle:
- How does it signal the exception? (Visual indicator, audible alarm, WMS notification, remote operator alert?)
- What does it do while waiting? (Hold position, move to safe zone, power down?)
- Is there a defined escalation path with time limits?
- Is the handoff zone safe for the human who responds?

If any of these are undefined, flag as `exception_handoff_undefined`.

### 5. Check for mode change safety

If the qualification scope includes mode changes (autonomous → supervised → manual):
- Is there a documented mode change procedure?
- Does mode change require authentication (key switch, biometric, software auth)?
- Are speed limits enforced during manual/teach mode? (ISO 10218: < 250 mm/s in teach mode.)
- Is there a three-position enabling device requirement for teach mode?

---

## Output

Structured non-routine operations review with:
- Per-operation-type classification and feasibility assessment.
- E-stop plan assessment.
- Fall zone status.
- LOTO procedure assessment.
- Exception handoff assessment.
- Mode change safety assessment.
- Required human actions.

---

## Do not

- Clear non-routine operations for humanoid execution without explicit task-level evidence of feasibility.
- Assume e-stop Category 1 capability without platform documentation.
- Treat humanoid e-stop the same as traditional robot e-stop — the fall hazard is unique.
- Approve jam clearing or machine recovery tasks for humanoids without demonstrated capability evidence.
- Ignore LOTO requirements because the humanoid is battery-powered — batteries are hazardous energy sources.
- Treat exception handling as an afterthought — it is where accidents happen.

---

## Fail-closed rules

- If non-routine operations are in scope and `non_routine_modes` is missing from intake: `not_ready_yet` for all non-routine tasks.
- If e-stop plan is missing: mandatory blocker. Cannot proceed to readiness without it.
- If LOTO procedure does not address the humanoid: mandatory blocker for any maintenance or jam-clearing scope.
- If exception handoff is undefined: `not_ready_yet` until human review defines the protocol.

---

## Escalation rules

- All non-routine operation findings must escalate to EHS/safety review.
- E-stop and LOTO findings must escalate to the site safety engineer, not just the project team.
- Fall zone calculations must be provided by the platform manufacturer (OEM) — this pipeline cannot compute them.
- Any task classified as `human_only` in the feasibility assessment must be excluded from humanoid qualification scope and documented in `human_actions_required.json`.

---

## Relevant standards

| Standard | Applicability |
|---|---|
| IEC 60204-1 | E-stop categories (0, 1, 2) |
| ISO 13850 | E-stop design requirements |
| ISO 25785-1 (draft) | Dynamically stable robot safety, fall zone calculations |
| OSHA 29 CFR 1910.147 | LOTO for hazardous energy control |
| ISO 10218-1/2:2025 | Teach mode speed limits (< 250 mm/s), mode change requirements |
| ANSI/RIA R15.08 | IMR safety during non-routine operations |
| OSHA General Duty Clause | Non-routine operations are where most robot accidents occur |
