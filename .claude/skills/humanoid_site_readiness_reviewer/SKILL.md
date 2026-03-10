# Humanoid Site Readiness Reviewer

Use when a capture must be evaluated for humanoid-relevant site readiness at the site level.

Inputs:
- `readiness_decision.json`
- `blocker_register.json`
- `standards_notes.json`
- `human_actions_required.json`

Required behavior:
- Summarize shared-space, route, access, and hidden-zone concerns for humanoid deployments.
- Keep conclusions bounded to the evidence in the capture package.
- Call out the human signoff boundary.

Do not:
- Approve deployment.
- Ignore mixed pedestrian or vehicle traffic.
- Convert missing evidence into optimistic assumptions.

Output:
- Site-readiness review summary.
