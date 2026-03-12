# Blueprint WebApp Operations

Use when work touches intake, routing, request state, admin review, exchange surfaces, or monetization flows in the web app.

Primary references:
- `Blueprint-WebApp/PLATFORM_CONTEXT.md`
- `Blueprint-WebApp/docs/first-principles-mvp-report.md`
- `Blueprint-WebApp/docs/integration-architecture.md`

Required behavior:
- Treat the web app as the operating and commercial layer around site-world records, runtime status, and downstream package consumption.
- Keep request state, runtime state, and downstream ingestion aligned with pipeline outputs.
- Preserve the site-world-first product stack in UI, workflows, and business logic.

Do not:
- Reframe the web app as a generic storefront.
- Promote marketplace browsing or CRM abstractions above site-world package truth.
- Assume static content is equivalent to live pipeline-backed state.

Output:
- WebApp recommendations or changes that stay consistent with the site-world-first operating model.
