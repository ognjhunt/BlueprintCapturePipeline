# Blueprint Capture

Use when work touches how site video is captured, the upload contract, or the delivery records the Pipeline starts from.

How capture works today (owner, 2026-11-11; see "How Capture Works" in `PLATFORM_CONTEXT.md`):
- A person who works at the site films the job with their own smartphone camera and uploads it in the browser, through the WebApp's onboarding and site-task flow.
- No capture app is in use. Nobody else films for a site: no outside capturers, no Blueprint filming visits, no capturer marketplace and no capturer payouts.
- The `BlueprintCapture` repository is kept only because its cloud function `extractFrames` still receives every website upload and writes the delivery records the Pipeline starts from. Its phone apps are not in use.

Primary references:
- `Blueprint-WebApp/PLATFORM_CONTEXT.md`
- `BlueprintCapturePipeline/docs/SYSTEM_MAP.md` (the upload row)
- `BlueprintCapture/cloud/extract-frames` (the upload hand-off)

Required behavior:
- Treat the WebApp browser upload as the one capture entry point.
- Preserve the upload layout and the delivery records the Pipeline reads.
- Keep the capture workflow grounded in a scoped site job and its evidence.

Do not:
- Describe the iOS, App Clip or Android apps, capturers or payouts as current.
- Treat raw video alone as the business output.
- Ignore the upload layout, manifest requirements, or delivery records.

Output:
- Guidance or implementation notes grounded in the browser capture path and its contracts.
