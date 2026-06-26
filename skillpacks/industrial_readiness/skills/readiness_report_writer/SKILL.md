# Readiness Report Writer

Use when the final human-readable readiness memo must be drafted from structured qualification and review artifacts. This is the last skill in the pipeline before human review. It transforms structured data into a clear, actionable document that a site owner, project manager, or integrator can read and act on. It must be honest about what the evidence shows and what it does not show.

---

## Trigger

- After humanoid_site_readiness_reviewer produces the site-level readiness assessment.
- When all review skills have completed and findings are available.
- When the qualification pipeline is ready for human review.

---

## Exact inputs

| Artifact | Required | Fields to inspect |
|---|---|---|
| `site_readiness_review.json` | Yes | `readiness_state`, `readiness_confidence`, `blocker_summary`, `concern_areas`, `next_actions` |
| `readiness_decision.json` | Yes | `readiness_state`, `decision_basis`, `confidence` |
| `blocker_register.json` | Yes | All entries — for detailed blocker descriptions |
| `capability_envelope.json` | Yes | Per-domain capability summaries |
| `standards_notes.json` | Yes | Per-blocker standards references |
| `human_actions_required.json` | Yes | All pending human actions |
| `recapture_plan.json` | If available | Recapture recommendations |
| `normalized_intake.json` | Yes | `workflow`, `zone`, `environment_type`, `owner`, `evidence_grade`, `target_platform` |
| `evidence_audit.json` | Yes | Overall evidence sufficiency, hidden zones |

Also include if available:
- `workcell_risk_review.json`
- `route_access_review.json`
- `shared_traffic_review.json`
- `non_routine_operations_review.json`

---

## Required behavior

### 1. Follow the standard report structure

The readiness report must follow this exact structure:

```markdown
# Site Readiness Assessment: [Zone Name]

**Date:** [timestamp]
**Workflow:** [workflow from intake]
**Zone:** [zone from intake]
**Environment:** [environment_type]
**Owner:** [owner]
**Target Platform:** [platform or "Not specified"]
**Evidence Grade:** [pre_screen / qualification_ready]

---

## Readiness State: [READY / RISKY / NOT READY YET]

[1-2 sentence summary of the readiness determination and why.]

---

## Human Signoff Boundary

[Standard boundary statement — what this report is and is not.]

---

## Executive Summary

[3-5 bullet points covering the most important findings.]

---

## Blockers

### Hard Blockers (must resolve before proceeding)
[List each hard blocker with: description, zone, evidence source, resolution path.]

### High Severity (requires human review)
[List each high blocker with: description, zone, evidence source, resolution path.]

### Medium Severity (document and track)
[List each medium blocker with: description, zone, evidence source.]

---

## Capability Assessment

### Route Traversal
[Per-segment summary: pass/conditional/fail/unverifiable. Choke points. Overall route status.]

### Reach and Manipulation
[Per-target summary: within envelope/outside/unverifiable. Articulation feasibility.]

### Occupancy and Timing
[Shift coverage, cycle time feasibility if applicable.]

---

## Evidence Assessment

**Overall grade:** [grade]
**Hidden zones:** [count and description]
**Capture quality:** [summary from QA scorecard]
**Key gaps:** [list specific missing evidence]

---

## Standards References

[Per-blocker applicable standards with standard number, summary, and applicability note.]

---

## Required Human Actions

[Ordered list from human_actions_required.json. Each with: action, owner, reason, priority.]

---

## Recapture Recommendations

[If recapture is needed: ordered list with modality, zone, reason, priority.]

---

## Next Steps

1. [First priority action]
2. [Second priority action]
3. [etc.]
```

### 2. Write in bounded language

Every statement in the report must be bounded:

**Good:** "Route segment A3-B1 width measured at 2.4 m (metric, confidence 0.85). Required clearance for humanoid + forklift: 2.65 m. Status: FAIL (short by 0.25 m)."

**Bad:** "The aisle appears to be wide enough for both the robot and forklifts."

**Good:** "3 of 7 manipulation targets are within verified reach envelope. 2 targets are unverifiable due to missing depth measurements. 2 targets exceed platform payload (estimated target weight 28 kg vs. 25 kg platform limit)."

**Bad:** "Most targets appear reachable."

### 3. Make the readiness state unambiguous

The readiness state must be stated clearly at the top of the document, with the determination logic visible:

- **READY:** "Zero hard blockers. All critical capability checks pass or conditional. Evidence grade: qualification_ready. [N] high-severity concerns require human review before proceeding."
- **RISKY:** "Zero hard blockers. [N] high-severity concerns. Evidence grade: [grade]. Proceed to human review with caution. [List top concerns.]"
- **NOT READY YET:** "[N] hard blockers prevent proceeding. [Describe primary blockers.] [Describe required resolution: recapture / scope change / site modification.]"

### 4. Never omit required human actions

If `human_actions_required.json` has entries, every single one must appear in the report. Do not summarize or skip any.

### 5. Include evidence provenance

For every factual claim in the report, include the source artifact and field in parentheses. The human reviewer must be able to trace any claim back to evidence.

---

## Output

`readiness_report.md` — a human-readable markdown document following the structure above.

---

## Do not

- Invent physical facts. Every measurement, count, and status must come from an artifact.
- State legal or off-scope approval conclusions. "This site is safe for humanoid deployment" is never written.
- Omit required human actions when evidence is incomplete. Incomplete evidence = more required actions, not fewer.
- Use promotional or optimistic language. "Exciting opportunity" or "promising results" have no place in a readiness report.
- Bury bad news. Hard blockers and high-severity findings go at the top, not in an appendix.
- Round or estimate measurements. Use exact values from artifacts.
- Imply that `READY` means deployment-approved. Always include the human signoff boundary statement.
- Omit the evidence grade. The reader must know whether this is a pre-screen or qualification-grade assessment.
- Use hedging language to soften hard findings. "The aisle might be a bit narrow" is not acceptable for a failed clearance check.

---

## Fail-closed rules

- If `site_readiness_review.json` is missing: report cannot be written. Return error.
- If `blocker_register.json` is missing: report cannot be written. Return error.
- If readiness state is `not_ready_yet`: the report must prominently state what must happen before the next review cycle.
- If evidence grade is `pre_screen`: the report header must state "PRE-SCREEN ASSESSMENT ONLY — NOT FOR QUALIFICATION DECISIONS" and this caveat must appear in the executive summary.

---

## Escalation rules

- The report itself is the escalation artifact — it goes to the human reviewer (owner from intake).
- If the report contains any safety-category blockers: recommend EHS review as the first next step.
- If the report recommends recapture: include the recapture plan in the report.
- If the report is for OEM handoff: also produce oem_handoff_writer output (separate artifact).
