# Blueprint Overview

Use when an agent needs the current business and system framing for Blueprint across the core repos.

Primary references:
- `Blueprint-WebApp/PLATFORM_CONTEXT.md`
- `BlueprintCapture/PLATFORM_CONTEXT.md`
- `BlueprintCapturePipeline/PLATFORM_CONTEXT.md`
- `BlueprintValidation/PLATFORM_CONTEXT.md`
- `Blueprint-WebApp/docs/first-principles-mvp-report.md`
- `Blueprint-WebApp/docs/integration-architecture.md`

Required behavior:
- Treat Blueprint as qualification-first.
- Keep `Blueprint-WebApp`, `BlueprintCapture`, and `BlueprintCapturePipeline` as the default operating path.
- Treat `BlueprintValidation` as a downstream lane, not the default output path.
- Treat world-model RL and downstream adaptation as first-class post-qualification lanes, not as qualification truth.
- Distinguish Notion-first operating docs from repo-authoritative contracts and runtime files.

Do not:
- Reframe Blueprint as a generic marketplace-first business.
- Treat geometry, scene packages, or model adaptation as the default product center.
- Invent new system boundaries without checking the platform context docs first.

Output:
- A grounded summary, recommendation, or plan tied back to the current Blueprint operating model.
