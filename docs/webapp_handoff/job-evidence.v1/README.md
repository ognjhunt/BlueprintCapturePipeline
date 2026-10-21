# Jobs and scientific evidence

Owner direction, corrected October 21, 2026: assessment requires an actual
retained job-video receipt and current permission. The existing video analysis
drafts the brief for customer correction; no written description is required.
An options assessment may use zero new evaluation runs when current evidence
is sufficient. Reconstruction, simulation and policy access resolve the actual
job's remaining questions through the existing pipeline. The commercial
workflow owns proposals, selection, agreement and customer-reported outcomes.
Unsupported work keeps its video and brief and reports the exact capability gap.

## Operating flow

Use the existing website **Job workflow** panel. A draft may wait for footage,
and starting requires neither a written description nor a buying budget.
Record an options assessment only after the retained video and current
permissions are verified. Its three dispositions remain options identified,
more information, and no suitable option.

When new evidence is needed, record the exact question in that panel. A question
without a current source-bound scientific request is saved with an information
request and no fabricated capture, testbed or run. For a real attached testbed
and request, Pipeline plans through its existing method catalog. The operator
can run the available read-only local methods. Other methods retain their
existing runtime, access and spending authority.

When site observations are needed, the job links to the existing capture screen
with its job Scene ID prefilled. Actual upload sessions retain their real IDs.
The existing task-approval action records the current confirmed brief, admitted
task items, rights, source capture and approved task; compilation carries that
binding through `site_task_testbed_compilation_submission.v3` into the signed
testbed publication. No extra approval stage is added. Older unbound sources
stay unavailable for current job claims until the existing task action supplies
an authoritative binding.

The same panel opens **Managed policy evaluation** for the job's actual
published scene setup. This uses the existing configured-scene preparation and
allocator; it does not require a technology team to run a policy. The supported
candidate pair, exact configurations and protocol come from that setup.
Unqualified policy canaries keep their diagnostic claim ceiling. An approved
team-hosted controller continues through its existing controlled-observation
integration. Neither an application nor a registry label grants runtime access.

Owners see the accepted scientific summary, exact question, verdicts,
limitations, next action and whether it still applies. Proposals can reference
an exact accepted result. A quote or buying-budget revision does not rewrite
that result. A changed brief/task/configuration can make its applicability
historical. Results never choose a supplier, create an agreement, or authorize
physical work.

Selection, both parties' agreement references, and a permitted customer-reported
outcome remain separate existing actions. Reported outcomes do not call
`physical_outcome_learning.join_physical_outcome`; that path still requires
authoritative exact-ID physical artifacts and permissions. Outcome-recording
permission is currently captured with the agreement; later permission needs an
explicit agreement amendment, not an inferred grant.

## Transport and identities

`job_evidence_request.v1` is a commercial-job binding around the actual
`decision_evidence_request.v1`, not a competing scientific request schema.
`job_evidence_publication.v1` references its real Evidence Plan and Decision
Envelope. The canonical scientific handoff remains
[`decision-evidence-router.v1`](../decision-evidence-router.v1/README.md).

The new projection binds `job_id`, the confirmed scientific `brief_digest`,
wrapper request ID/digest, approved task digest, source intake, testbed,
scientific request, plan and result. Buying requirements and proposal revisions
are separate commercial scope. Historical records without a reliable source
brief/task binding are not silently relabeled as current evidence.

Wrapper/publication digests use RFC 8785. Historical Pipeline scientific
digests continue using the producer's original canonical JSON. Opaque canonical
body strings accompany those artifacts across Node/Python; consumers hash those
exact bytes and verify their parsed content equals the displayed object.
Compiled cards retain their own exact canonical bodies. No digest is recomputed
under a different numeric or key-order convention.

Pipeline routes, using the existing admitted service HMAC:

- `POST /api/live-pipeline/job-evidence-requests`
- `GET /api/live-pipeline/job-evidence-requests/{request_id}`
- `POST /api/live-pipeline/job-evidence-requests/{request_id}/execute`
- `POST /api/live-pipeline/job-evidence-requests/{request_id}/cancel`

Execution accepts the exact request digest and authenticated operator identity.
It permits only the existing read-only local adapter registry. Paid policy
execution remains on the existing configured-scene path. Cancellation also
revokes the underlying local run so the legacy execution URL cannot bypass it.
Stable request IDs/idempotency keys and immutable publications make retry safe.

Signed publication uses `/api/internal/job-evidence-publications` and the
existing Pipeline sync credential. Its receipt must match the exact job,
request and publication digest. `PIPELINE_JOB_EVIDENCE_WEBAPP_URL` may override
the destination; otherwise the existing configured testbed/run sync origin is
reused. The website similarly reuses its existing Pipeline forwarding service
credentials. No owner-link token or provider secret is a service credential.

Request bodies are bounded at 2 MiB; the WebApp stores bounded summaries and
references, with immutable scientific publications outside `job_marketplace`.
Large raw evidence remains in the existing artifact storage. No request on
this projection allocates paid compute, sends outreach or moves hardware.

## Observed verification and release boundary

The development fixture in `tests/fixtures/job_evidence_v1/producer-examples.json`
uses the real Python compiler/request/planner/envelope producers. Node consumes
those exact bytes. Its unmatched qualification correctly produces abstention;
the local method fixtures separately exercise supported and negative decisions.
These are software results, not customer evidence, provider execution,
sim-to-real qualification or physical proof.

### Historical implementation inspection

The following snapshot predates the corrected video-first takeover and is
retained as release history, not the current production or PR inventory.

At that inspection, WebApp PR #1077 was open at `9244e10d530963df3893a6e8c1192c597c8a419c`,
WebApp main was `76f263507ca8ea60c8fc8d40a56eefbc88643e65`, and live `/version.json`
reported `9884867b0b01f5e3a9f2156a26756657e49a2672` built October 20. Pipeline main was
`599264bda025ffb274996037d2ac997ec53f8a1b`; the running intake freshly reported
`b86b78ab83f217fe4fca2bf135e290b7b8aebe56` with `commit_proven=true`.
During implementation #1077 was closed unmerged and superseded by #1078 at
`983ac5b98cca07024999758d34650c85740df1a9`. Those follow-up fixes were merged
into this task's isolated website branch. #1078 then merged as
`bed14fcf7986ea596dd244de0c624edcae061c36`; live `/version.json` independently
reported its build source `983ac5b98cca07024999758d34650c85740df1a9`, built
October 21 at 14:44 UTC. This implementation does not change runtime flags or
deploy either service.

### Current release preparation — October 21, 2026

The owner requested code and release preparation with verified changes held
ready for release. During verification, GitHub externally merged WebApp #1079
as `ec9157917514ad883211a83a722254b56a08c1f1`, Capture #91 as
`7b7fc6e06682d5f1ef5b21ced9bec057b0a4189a`, and Pipeline #2738 as
`2c0ac160735d1d012ab4a66ad53478a19ec6bcb9`. Pipeline main subsequently advanced
through #2739 to `1c36a6bcace391da52766f50bb88877f82c95eb2`. This companion
change preserves its recovery, dependency-security and qualified-method routing
changes, along with matching generated-doctrine and producer-fixture follow-ups.
Pipeline #2737 and Capture #90 are historical. The retained blocked capability
gap wire example is historical; fresh requests use the evidence router and may
answer a qualified bounded question or abstain without native task execution.

The 18:31 UTC readback reports an external WebApp and worker deployment of
`dcfb21e3550821561e0f6958c6c02ee51cc9f638` (the later cancellation-handshake
merge #1084). Video analysis is still disabled and worker Gemini readiness is
false. The last observed Pipeline runtime is
`b86b78ab83f217fe4fca2bf135e290b7b8aebe56`. Repository and deployment receipts
remain separate. This task performed no pull-request merge, production deployment,
video-processing activation, execution-hold release or fresh policy run.

For a later authorized release, deploy the compatible WebApp consumer and
verify its exact `taskEvaluationPolicyRuns` job/time index is READY before
activating new Pipeline publications. Verify approved Gemini bindings and
video-worker readiness separately. Capture's matching doctrine change has no
runtime dependency. Normal website production builds still require the
existing Firebase public configuration; an unconfigured local build remains
compile-only. Missing runtime capabilities and current execution authority
remain explicit blockers rather than new proof.
