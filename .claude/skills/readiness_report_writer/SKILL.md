# Readiness Report Writer

Use when the final operator memo must be drafted from structured qualification and agent-review artifacts.

Inputs:
- `readiness_decision.json`
- `blocker_register.json`
- `capability_envelope.json`
- `standards_notes.json`
- `human_actions_required.json`
- `recapture_plan.json`

Required behavior:
- Draft a human-readable memo from structured artifacts only.
- State evidence gaps and remediation explicitly.
- Keep the human signoff boundary obvious.

Do not:
- Invent physical facts.
- State legal or safety approval conclusions.
- Omit required human actions when evidence is incomplete.

Output:
- Human-readable readiness memo.
