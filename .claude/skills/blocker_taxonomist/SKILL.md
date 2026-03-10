# Blocker Taxonomist

Use when raw qualification risks and audit gaps need a normalized blocker register for industrial site review.

Inputs:
- `capability_checks.json`
- `geometry_evidence.json`
- `site_intake.json`
- `evidence_audit.json`

Required behavior:
- Normalize blockers into geometry, safety, privacy, systems, traffic, workflow ambiguity, and capture quality.
- Preserve source evidence references.
- Keep severity conservative when evidence is incomplete.

Do not:
- Collapse multiple blocker types into one generic item.
- Drop evidence provenance.
- Reclassify high-risk evidence as informational.

Output:
- Candidate blocker register entries only.
