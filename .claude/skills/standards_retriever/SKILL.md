# Standards Retriever

Use when blocker categories need curated standards or guidance notes.

Inputs:
- `site_intake.json`
- `blocker_register.json`
- `references/curated_standards.json`

Required behavior:
- Retrieve relevant local curated guidance only.
- Return citations and applicability notes, not legal conclusions.
- Match notes to blocker categories and site context.

Do not:
- Browse the web in v1.
- Present guidance as regulatory approval.
- Return uncited advice.

Output:
- Evidence-linked standards notes only.
