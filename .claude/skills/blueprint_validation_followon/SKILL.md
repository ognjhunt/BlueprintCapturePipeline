# Blueprint Validation Follow-On

Use when a site-world package is being routed into downstream evaluation, adaptation, or world-model validation work.

Primary references:
- `BlueprintValidation/README.md`
- `BlueprintValidation/docs/qualified_opportunity_handoff.md`
- `Blueprint-WebApp/docs/integration-architecture.md`

Required behavior:
- Treat `BlueprintValidation` as a downstream consumer of site-world packages.
- Require a site-world package or equivalent evaluation-prep handoff before recommending deeper evaluation by default.
- Keep downstream evaluation coupled to scoped task, constraints, and evidence links.
- Treat world-model adaptation and RL post-training as valid downstream paths, while keeping stricter validation gates for high-stakes claims.

Do not:
- Treat validation as the default first pass for new sites.
- Replace capture-backed site-world artifacts with downstream experimental outputs.
- Assume every site world should enter validation.

Output:
- Downstream evaluation guidance that preserves the site-world-first operating model.
