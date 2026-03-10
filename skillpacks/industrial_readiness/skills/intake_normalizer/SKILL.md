# Intake Normalizer

Use when a qualification run needs structured workflow, zone, owner, and success-criteria normalization before any readiness language is written.

Inputs:
- `site_intake.json`
- `capture_package_manifest.json`

Required behavior:
- Normalize workflow statement, zone, owner, systems, non-routine modes, traffic notes, privacy/security limits, and known blockers.
- Fail closed when workflow, zone, or success criteria are missing.
- Preserve source field names when copying evidence forward.

Do not:
- Make a readiness judgment.
- Invent missing workflow details.
- Treat a splat or geometry artifact as a substitute for intake.

Output:
- Updated structured intake notes only.
