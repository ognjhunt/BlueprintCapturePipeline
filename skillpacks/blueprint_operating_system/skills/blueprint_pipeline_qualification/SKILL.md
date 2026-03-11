# Blueprint Pipeline Qualification

Use when work touches qualification artifacts, readiness decisions, handoffs, agent review, or cross-repo routing between evidence and operating state.

Primary references:
- `BlueprintCapturePipeline/PLATFORM_CONTEXT.md`
- `BlueprintCapturePipeline/README.md`
- `Blueprint-WebApp/docs/integration-architecture.md`

Required behavior:
- Treat `BlueprintCapturePipeline` as the qualification engine.
- Keep the default output centered on qualification records and handoff artifacts.
- Preserve fail-closed behavior and human-review boundaries.

Do not:
- Let advanced geometry or simulation lanes replace qualification as the center of gravity.
- Treat downstream validation as the default interpretation of pipeline output.
- Invent readiness claims without reference to actual qualification artifacts.

Output:
- Qualification-grounded analysis, implementation notes, or operating guidance.
