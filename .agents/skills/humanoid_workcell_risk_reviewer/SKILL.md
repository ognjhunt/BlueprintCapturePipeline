# Humanoid Workcell Risk Reviewer

Use when workcell-specific risks need to be summarized for humanoid task execution.

Inputs:
- `qualification_record.json`
- `geometry_evidence.json`
- `blocker_register.json`

Required behavior:
- Highlight workcell reach, occlusion, articulation, and hidden-condition risks.
- Stay grounded in measured or explicitly stated evidence.
- Keep unresolved conditions visible to the human reviewer.

Do not:
- Infer safe manipulation from object labels alone.
- Treat missing geometry as a pass.
- Rewrite the overall readiness state.

Output:
- Workcell-risk review summary.
