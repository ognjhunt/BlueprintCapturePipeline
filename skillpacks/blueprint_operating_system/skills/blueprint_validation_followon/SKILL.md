# Blueprint Validation Follow-On

Use when a qualified opportunity is being routed into downstream evaluation, adaptation, or world-model validation work.

Primary references:
- `BlueprintValidation/README.md`
- `BlueprintValidation/docs/qualified_opportunity_handoff.md`
- `Blueprint-WebApp/docs/integration-architecture.md`

Required behavior:
- Treat `BlueprintValidation` as a post-qualification lane.
- Require a qualified opportunity handoff before recommending deeper evaluation by default.
- Keep downstream evaluation coupled to scoped task, constraints, and evidence links.
- Treat world-model adaptation and RL post-training as valid downstream paths, while keeping stricter validation gates for high-stakes claims.

Do not:
- Treat validation as the default first pass for new sites.
- Replace qualification artifacts with downstream experimental outputs.
- Assume every qualified record should enter validation.

Output:
- Downstream evaluation guidance that preserves the qualification-first operating model.
