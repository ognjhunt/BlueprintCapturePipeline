# Humanoid Site Readiness Reviewer

Use when a capture package must be evaluated for humanoid-relevant site readiness at the site level. This is the top-level review that synthesizes all lower-level reviews (workcell, route, traffic, non-routine) into a site-wide readiness assessment. It must never approve deployment. It produces a structured readiness assessment that humans use to make the deployment decision.

---

## Trigger

- After evidence_auditor, blocker_taxonomist, capability_envelope_writer, and all applicable review skills have completed.
- When the qualification pipeline needs a site-level readiness determination.

---

## Exact inputs

| Artifact | Required | Fields to inspect |
|---|---|---|
| `readiness_decision.json` | Yes | `readiness_state`, `confidence`, `evidence_grade`, `decision_basis` |
| `blocker_register.json` | Yes | All entries — count by severity and category |
| `capability_envelope.json` | Yes | Overall capability summary, per-domain status |
| `standards_notes.json` | Yes | Per-blocker and per-capability standards references |
| `human_actions_required.json` | Yes | All pending human actions |
| `normalized_intake.json` | Yes | `environment_type`, `workflow`, `zone`, `evidence_grade`, `target_platform` |
| `evidence_audit.json` | Yes | Overall evidence sufficiency, hidden zones |

Also consult (if available):
- `humanoid_workcell_risk_review.json` — workcell-level findings
- `humanoid_route_access_review.json` — route-level findings
- `shared_traffic_review.json` — traffic-level findings
- `non_routine_operations_review.json` — exception/recovery findings

---

## Required behavior

### 1. Determine readiness state

The readiness state is one of three values. It cannot be overridden by any single skill — it is computed from the aggregate of all findings.

| State | Definition | Conditions |
|---|---|---|
| `ready` | Evidence supports proceeding to human review and downstream OEM evaluation. No hard blockers. All critical capability checks pass or are conditional. | Zero `hard_blocker` entries. Zero `high` severity safety blockers. Evidence grade is `qualification_ready`. All critical route segments have metric clearance evidence. |
| `risky` | Some evidence supports readiness, but material concerns exist that require human judgment. | Zero `hard_blocker` entries. One or more `high` severity blockers that are not safety-category. Evidence grade is at least `pre_screen`. Some capability checks are `conditional` or `unverifiable`. |
| `not_ready_yet` | Insufficient evidence, hard blockers, or safety concerns prevent proceeding. Recapture or scope change required. | Any `hard_blocker` exists. OR evidence grade is `insufficient`. OR critical safety blocker exists. OR 3+ `high` severity blockers in the same zone. |

**Critical rule:** `ready` does NOT mean deployment-approved. It means the evidence package is sufficient for a human to review and for OEM/integrator evaluation to begin. The human signoff boundary is always downstream of this assessment.

### 2. Synthesize site-level concerns

Group findings into these site-level concern areas:

**Shared space safety:**
- How many zones have shared traffic (forklift, AGV, pedestrian)?
- What traffic management evidence exists?
- Are there unresolved force/impact concerns?
- Is there a fall zone calculation?

**Route network:**
- What percentage of route segments have verified clearance?
- How many choke points exist and are they cleared?
- Are there disconnected or low-confidence route segments?

**Workcell readiness:**
- How many manipulation targets are within verified reach envelope?
- Are machine interfaces captured with sufficient detail?
- Are force and articulation requirements within platform capability?

**Evidence completeness:**
- What is the overall evidence grade?
- How many hidden zones remain?
- What recapture is needed?

**Non-routine preparedness:**
- Is there an e-stop plan?
- Are LOTO procedures defined?
- Are exception handling protocols documented?

**Infrastructure:**
- Is charging infrastructure planned?
- Is network coverage verified?
- Are any facility modifications required?

### 3. State the human signoff boundary

Every site readiness review must include an explicit statement of what this assessment does and does not represent:

> This assessment evaluates whether the capture evidence is sufficient and whether identifiable blockers exist for humanoid deployment qualification in [zone]. It does NOT constitute generated-world rank-fidelity result, safety certification, or regulatory compliance. Human review by [owner] is required before any deployment decision. EHS/safety review is required for all shared-space and non-routine operation findings. OEM/integrator evaluation is required for platform-specific capability validation.

### 4. Produce actionable summary

The review must include:
- **Top findings** (max 5): The most important things the human reviewer needs to know.
- **Blocker summary**: Count by severity and category. List all hard blockers explicitly.
- **Capability summary**: Which capability domains pass, which fail, which are unverifiable.
- **Evidence summary**: Overall grade, hidden zones, recapture needs.
- **Next actions**: Ordered list of what must happen next (human review, recapture, OEM evaluation).

---

## Output

`site_readiness_review.json` or equivalent structured review with:
- `readiness_state`: ready / risky / not_ready_yet
- `readiness_confidence`: How confident the assessment is (based on evidence completeness)
- `evidence_grade`: From intake normalization
- `blocker_summary`: Count by severity and category
- `hard_blockers[]`: Explicit list
- `capability_summary`: Per-domain status
- `concern_areas[]`: Grouped site-level concerns
- `human_signoff_statement`: The boundary statement
- `next_actions[]`: Ordered action items
- `recapture_needed`: boolean
- `oem_evaluation_ready`: boolean

---

## Do not

- Approve deployment. This assessment is input to a human decision, not the decision itself.
- Ignore mixed pedestrian or vehicle traffic. Any shared traffic zone that is not explicitly cleared is a concern.
- Convert missing evidence into optimistic assumptions. Missing = unknown = conservative.
- Override individual skill findings. If the route reviewer says a segment fails, the site reviewer does not clear it.
- Produce a readiness state of `ready` when evidence grade is `pre_screen`. Pre-screen evidence can support `risky` at best.
- Hide uncertainty behind summary language. If 40% of route segments are unverifiable, say that, do not say "most routes appear clear."
- Use language that implies certainty when evidence is estimated or inferred.

---

## Fail-closed rules

- Any `hard_blocker` in `blocker_register.json`: readiness state = `not_ready_yet`. No exceptions.
- Evidence grade `insufficient`: readiness state = `not_ready_yet`.
- Evidence grade `pre_screen`: readiness state cannot be better than `risky`.
- Missing `evidence_audit.json`: site review cannot proceed.
- Missing `blocker_register.json`: site review cannot proceed.
- If 50%+ of capability domains are `unverifiable`: readiness state = `not_ready_yet`.

---

## Escalation rules

- All `not_ready_yet` assessments must include a specific recapture or scope change recommendation.
- All safety-category blockers must escalate to EHS/safety review in `human_actions_required.json`.
- If `readiness_state` is `ready` but non-routine operations review is incomplete: add a caveat that readiness is for steady-state operations only.
- If `target_platform` is specified but platform-specific blockers exist: escalate to OEM for platform validation.
