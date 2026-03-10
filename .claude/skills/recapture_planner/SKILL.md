# Recapture Planner

Use when missing evidence must be converted into a concrete field checklist.

Inputs:
- `capture_qa_scorecard.json`
- `geometry_evidence.json`
- `blocker_register.json`

Required behavior:
- Convert evidence gaps into an ordered recapture plan.
- Prefer metric capture for geometry-critical blockers.
- Explain why each recapture step is needed.

Do not:
- Ask for recapture without a cited reason.
- Treat optional cosmetics as blocking evidence.
- Suggest splat-only remediation for missing intake or QA.

Output:
- Ordered recapture plan with justification.
