# OEM Handoff Writer

Use when the qualification result must be packaged for OEM, integrator, or robot-platform review.

Inputs:
- `opportunity_handoff.json`
- `readiness_decision.json`
- `human_actions_required.json`

Required behavior:
- Summarize readiness state, evidence boundaries, and next human-owned decisions.
- Keep downstream platform choice explicit when it is still unresolved.
- Preserve the qualification contract fields.

Do not:
- Select a robot platform automatically.
- Hide evidence gaps from downstream evaluators.
- Rewrite the original handoff contract.

Output:
- OEM-facing handoff summary.
