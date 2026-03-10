# OEM Handoff Writer

Use when the qualification result must be packaged for OEM, integrator, or robot-platform review. The handoff package must give a downstream evaluator everything they need to determine whether their platform can serve this site and workflow — without hiding uncertainty, inflating readiness, or making the platform selection decision for them.

---

## Trigger

- When readiness state is `ready` or `risky` and the next step is OEM/integrator evaluation.
- When `target_platform` is specified and platform-specific validation is needed.
- When `opportunity_handoff.json` needs to be populated for downstream review.

---

## Exact inputs

| Artifact | Required | Fields to inspect |
|---|---|---|
| `opportunity_handoff.json` | Yes | Existing handoff contract fields |
| `readiness_decision.json` | Yes | `readiness_state`, `confidence`, `decision_basis` |
| `site_readiness_review.json` | Yes | Full site-level review |
| `capability_envelope.json` | Yes | Per-domain capability assessments |
| `blocker_register.json` | Yes | All entries — especially `platform_limitation` category |
| `standards_notes.json` | Yes | Applicable standards for OEM context |
| `human_actions_required.json` | Yes | Pending actions that may affect OEM evaluation |
| `normalized_intake.json` | Yes | `workflow`, `zone`, `environment_type`, `target_platform`, `evidence_grade` |
| `geometry_evidence.json` | Yes | Key measurements the OEM needs |
| `task_scope_record.json` | Yes | Task details the OEM needs for platform fit assessment |
| `route_access_review.json` | If available | Route constraints relevant to platform mobility |
| `workcell_risk_review.json` | If available | Workcell constraints relevant to platform manipulation |
| `shared_traffic_review.json` | If available | Traffic constraints relevant to platform safety |
| `non_routine_operations_review.json` | If available | Non-routine findings relevant to platform capability |

---

## Required behavior

### 1. Produce a structured handoff package

The handoff must follow this structure:

```json
{
  "handoff_id": "HO-[timestamp]",
  "handoff_date": "[ISO 8601]",
  "qualification_summary": {
    "readiness_state": "[ready/risky/not_ready_yet]",
    "evidence_grade": "[pre_screen/qualification_ready]",
    "confidence": "[value]",
    "summary": "[1-2 sentence readiness summary]"
  },
  "site_profile": {
    "environment_type": "[type]",
    "zone": "[zone description]",
    "workflow": "[workflow description]",
    "shift_info": "[if available]"
  },
  "platform_requirements": {
    "minimum_dimensions": {
      "max_height_m": "[from route overhead clearances]",
      "max_width_m": "[from route choke points]",
      "max_weight_kg": "[from floor load if applicable]"
    },
    "manipulation_requirements": {
      "payload_kg": "[max target weight]",
      "reach_m": "[max target distance]",
      "hand_dof_minimum": "[from articulation complexity mapping]",
      "placement_precision_mm": "[if applicable]",
      "force_requirements_N": "[if applicable]"
    },
    "mobility_requirements": {
      "route_segments": "[count]",
      "max_slope_deg": "[from route grades]",
      "max_step_mm": "[from thresholds]",
      "surface_types": "[list]",
      "operating_speed_ms": "[required/recommended]"
    },
    "runtime_requirements": {
      "min_battery_hours": "[from shift/workflow duration]",
      "charging_infrastructure": "[available/planned/none]"
    },
    "safety_requirements": {
      "shared_space_zones": "[count and description]",
      "force_limiting_required": "[yes/no/unknown]",
      "estop_category": "[if specified]",
      "fall_zone_required": "[yes/no]"
    }
  },
  "open_blockers": [
    {
      "blocker_id": "[ID]",
      "summary": "[description]",
      "category": "[category]",
      "severity": "[severity]",
      "platform_relevance": "[how this affects platform selection/deployment]",
      "resolution_owner": "[who must resolve: site owner / OEM / integrator / both]"
    }
  ],
  "platform_specific_notes": {
    "target_platform": "[if specified]",
    "platform_fit_assessment": "[summary of how the target platform matches requirements]",
    "platform_blockers": "[any platform-specific blockers]",
    "alternative_platforms": "[if target platform has blockers, which other platforms might fit]"
  },
  "evidence_package": {
    "artifacts_included": "[list of artifact files included in handoff]",
    "evidence_grade": "[grade]",
    "evidence_limitations": "[list of known evidence gaps]",
    "recapture_pending": "[yes/no — if yes, include recapture plan]"
  },
  "human_actions_pending": [
    {
      "action": "[description]",
      "owner": "[who]",
      "priority": "[priority]",
      "affects_oem_evaluation": "[yes/no]"
    }
  ],
  "standards_context": [
    {
      "standard": "[standard number]",
      "relevance": "[how it applies to OEM evaluation]"
    }
  ],
  "next_steps_for_oem": [
    "[First thing the OEM should evaluate]",
    "[Second thing]",
    "[etc.]"
  ]
}
```

### 2. Translate qualification findings into platform requirements

The OEM does not need to read every blocker — they need to know:
- What physical dimensions the platform must fit within.
- What manipulation capabilities are required.
- What mobility capabilities are required.
- What safety features are required.
- What is still uncertain and needs their input.

Extract these from the review artifacts and present them as concrete requirements.

### 3. Be explicit about what the OEM must validate

This pipeline can determine whether evidence suggests the platform fits. It cannot:
- Validate the platform's actual field performance.
- Certify the platform's safety systems.
- Verify the platform's manipulation accuracy in this specific workcell.
- Confirm the platform's sensor performance in this specific lighting/environment.

State explicitly what the OEM must verify on their side.

### 4. Separate site-owner-resolvable from OEM-resolvable blockers

Some blockers require the site owner to fix (floor condition, traffic management, guarding changes). Others require the OEM to address (platform capability, force limiting, safety certification). Some require both. Make the ownership clear.

### 5. Include platform comparison data when no target is specified

If `target_platform` is not specified, include a platform requirements summary that any OEM can evaluate against their specs. Reference the cross-platform envelope from `humanoid_platform_reference`:
- Which platforms meet the dimensional requirements.
- Which platforms have sufficient hand DOF.
- Which platforms have sufficient payload.
- Which platforms have deployment track record in this environment type.

Do NOT recommend a specific platform. Present the facts and let the OEM/integrator decide.

---

## Output

Updated `opportunity_handoff.json` with the full handoff package structure above, plus a human-readable `oem_handoff_summary.md` that summarizes the handoff for email/meeting context.

---

## Do not

- Select a robot platform automatically. Platform selection is an OEM/integrator/customer decision.
- Hide evidence gaps from downstream evaluators. The OEM needs to know what is uncertain.
- Rewrite the original handoff contract — preserve existing fields and add to them.
- Inflate readiness to make the opportunity look better. The handoff must be honest.
- Include proprietary site information that the site owner has not authorized for external sharing. Check `privacy_limits` from intake.
- Present `risky` readiness as `ready` in the handoff summary.
- Omit pending human actions that affect OEM evaluation. If a blocker might disappear after recapture, say so.
- Make platform capability claims on behalf of the OEM. "Digit can handle this task" is not something this pipeline can state — it can say "the task requires 16 DOF hands, which is within Digit's specification" but cannot confirm field viability.

---

## Fail-closed rules

- If readiness state is `not_ready_yet`: handoff is premature unless explicitly requested for "early engagement" purposes. If producing anyway, label prominently as "PRE-QUALIFICATION — FOR EARLY ENGAGEMENT ONLY."
- If `opportunity_handoff.json` is missing: create a new one with all required fields.
- If evidence grade is `pre_screen`: label the handoff as pre-screen grade and note that qualification-grade evidence is pending.
- If blocker register has any `safety_*` hard_blockers: do not send to OEM until safety issues are resolved or explicitly noted as site-owner responsibility.

---

## Escalation rules

- If the handoff reveals that no current platform meets the requirements: escalate to project manager with a clear explanation of which requirements are unmet and by how much.
- If platform-specific blockers exist for the specified target platform: recommend the site owner discuss alternatives with the OEM/integrator.
- If privacy/security limits restrict what can be shared with the OEM: escalate to site owner for handoff content approval.
- If the OEM needs on-site access for their own evaluation: note this as a required next step.
