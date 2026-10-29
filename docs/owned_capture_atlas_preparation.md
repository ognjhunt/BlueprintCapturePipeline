# Owned-upload Atlas preparation

This development-only adapter closes an ADP-009D day-14 receipt/day-28 replay
dependency. It reuses the existing Atlas controller, bounded worker, private
archive, provider cleanup and explicit replay lifecycle. It does not qualify a
public-scene gate, measured geometry, an embodiment, robot execution or a
physical outcome.

## Owner path

The original upload must first pass normal owned `/process`, scanner,
content-addressing and Pipeline intake admission. The original bytes, complete
task, proposed/confirmed parameter status and source governance remain bound to
the accepted receipt. A source description is not a site-control attestation.

The existing WebApp owner API provides these operations beneath
`/api/capture-uploads/:sessionId/reconstructions/atlas`:

1. `POST /plan`: create an immutable source/task plan, with no queue, grant or
   provider call. Repeating identical input reuses the plan.
2. `POST /authorize`: retain the original finite source allowance only when an
   existing explicit company approval actually matches this owner, source,
   provider terms and accepted processing/retention decision. The automatic
   website-recording preview grant is not consulted. This command establishes
   no robot-execution authority.
3. `POST /:planId/execute`, body
   `{"schema_version":"capture_owned_atlas_execute_command.v1","idempotency_key":"stable-request-1"}`:
   queue the retained plan using a noncreating configuration check. A caller
   cannot supply another owner, source path, provider key or grant. Reconcile
   uncertain acknowledgements with the same request key and owner status.
4. `GET /:planId`: inspect the retained worker result.
5. `GET /:planId/bundle`: download a ready private archive using the existing
   SHA-256/size streaming verifier. An incomplete scene has no ready download.
6. `POST /:planId/retry`, body
   `{"schema_version":"capture_owned_atlas_retry_command.v1","idempotency_key":"explicit-retry-1"}`:
   after a focused correction, resume the same root and completed stages.
   Repeating that retry key does not start another attempt. Source, task, owner,
   original authorization digest and original seven-day deadline must still
   match. This never renews an allowance or changes the task to obtain a pass.

All operations use the existing Firebase-authenticated owner/source checks.
Native execute, retry, inspection and bundle commands are distinct typed POST
bodies signed by the WebApp. Owner, plan and bundle digest are inside the signed
body because the existing intake signature does not authenticate URL parameters.

## Worker and disclosure

The existing `website_atlas_preview tick` timer also discovers explicitly queued
owned jobs under the API's configured work root:
`reconstruction_control_plane/plans/:planId/atlas/owned_job.json`. It uses the
same configured capture store as intake. No `site-*` or `walkthrough-*` source
record is fabricated, and queueing never invokes a provider inline.

Each owned step reopens the immutable plan and current accepted source bytes,
QA, task, governance and lifecycle. Every new source disclosure is fenced,
including Atlas uploads/tasks, planner requests and image-editor HTTP requests.
Atlas client factories additionally compare the actual key's documented
operation account/project metadata with the signed approved project. Missing,
empty, malformed or mismatched project evidence refuses mutation. Project
identity does not verify account training or retention settings.

The existing canonical paid allocator and source-specific WebApp reservations
still gate purchases. Unknown calls retain their full reservation until
reconciled. Reads, settlement and provider cleanup can continue after processing
authority ends. A terminal source withdrawal suppresses/retracts its archive and
cleans retained provider assets. Retry preserves the existing deleted-asset
evidence and rearms cleanup for any new assets.

## Result boundary

`owned_capture_atlas_execution.v1` binds the original plan, task context, source
video, finite authority, revision and private bundle digest. It reports
processing, ready, needs-input, failed or withdrawn. A downloadable result is
an **unscaled provider draft**, even when splat and mesh files exist.

The generic `reconstruction_result.v1`/testbed compiler projection and geometric,
metric, collision, robot/policy and physics admission remain downstream gates.
This preparation receipt cannot silently stand in for them. In particular,
missing navigation inventory is not universal task infeasibility, and a model
assessment or successful reconstruction is not a completed robot evaluation.

Hermetic source, signature, project, lifecycle, replay and media tests exercise
these contracts without calling providers. Their synthetic receipts are never
production permission, provider usage or robot-success evidence.
