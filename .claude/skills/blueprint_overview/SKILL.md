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
- Treat Blueprint as site-world-first.
- Keep the WebApp browser upload (a person who works at the site films the job on their own phone) and `BlueprintCapturePipeline` as the default operating path for building capture-backed site worlds. The `BlueprintCapture` phone apps are not in use; that repository is kept only for its `extractFrames` upload hand-off.
- Treat `Blueprint-WebApp` as the operating layer around site-world records, runtime state, and downstream consumption.
- Treat `BlueprintValidation` as a downstream consumer of site-world packages, not the default output path.
- Treat legacy qualification/readiness artifacts as compatibility overlays, not source-of-truth product framing.
- Distinguish Notion-first operating docs from repo-authoritative contracts and runtime files.

Do not:
- Reframe Blueprint as a generic marketplace-first business.
- Treat legacy qualification or reporting workflows as the default product center.
- Invent new system boundaries without checking the platform context docs first.

Output:
- A grounded summary, recommendation, or plan tied back to the current Blueprint operating model.
