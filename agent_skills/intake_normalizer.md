# Intake Normalizer

Input:
- `site_intake.json`
- `capture_package_manifest.json`

Task:
- Normalize workflow, KPI, owner, zone, systems, non-routine modes, people/traffic notes, privacy/security limits, and blocker text.
- Fail closed when workflow, zone, or success criteria are missing.

Output:
- Updated structured intake notes only. No final readiness judgment.
