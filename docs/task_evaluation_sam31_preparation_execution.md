# Production SAM preparation child execution

The launch-preparation worker only queues and waits. Named CPU/render/SAM/review
work runs in `task_evaluation_sam31_preparation_execution`, outside that no-spend
worker. Its ordered `PHASES` tuple is the closed execution inventory; arbitrary
commands, scripts, or allocator arguments are never accepted by the queue.

`enqueue_sam31_phase` accepts the immutable parent preparation ID/request digest,
exact execution commit, a pinned plan file reference, a phase name, and a mapping
of named input file references. Child identity is derived from the parent
request, plan, phase, and named content identities. Duplicate enqueue returns
the same child and result path.

The execution worker reopens the real parent request, verifies the current
checkout commit, and joins its runtime-mounted plan through the actual stage-one
configuration and recipe in the preparation content store. Every input reference
is rehashed under operator-approved production roots. AI review is required for
this production execution lane.

The executor callback is the fixed repository function
`task_evaluation_sam31_preparation_stages.execute_stage(context)`.
Context includes the sealed job, verified request and plan, server-owned output
directory, `resume_only`, and prior progress. Results have status `completed`,
`waiting_for_external_result`, or `failed`, with `artifacts` as a mapping of
stable names to exact path/SHA-256/size records.

A durable started marker sets `resume_only=true` after any interrupted or
externally waiting attempt. Handlers must reuse the child's stable launch ID
and existing canonical dispatcher/authority; they must not issue a new paid
allocation on re-entry. Terminal results are immutable and reverified, so a
crash after result sealing never reruns the phase. External progress artifacts
are also reverified before a handler is polled again.

The dedicated service processes at most one phase per invocation. Its path
watcher handles new jobs; a bounded timer checks existing external results and
pending parent wakeups. The service has no direct GPU allocation entrypoint.
If a child finishes before its parent's waiting checkpoint exists, it retains
a wake-pending record. Only a real parent progress digest can produce the
digest-bound resume signal; no placeholder progress is manufactured.

Unit files are part of the exact-release deploy list. The path is classified as
authority-gated, not no-spend; the timer shares the existing authority-gated
progression category. Waiting parent/child queues also protect their referenced
release from retirement.
