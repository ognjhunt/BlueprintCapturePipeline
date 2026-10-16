# One GPU per team evaluation and bounded release status

This repairs free-beta journey L5 (an invited team's scored Task Evaluation
Run). It also exercises the runtime/teardown precursor of ADP-009D's day-28
rehearsal; authored fixtures do not qualify that gate or physical performance.

The previous controller made eight separate 25-cell GPU sessions. Two setup
failures and two partial batches made the retained internal diagnostic use
eleven sequential rentals. A machine boundary was being used as a batch
boundary, despite the existing worker already isolating each simulator cell.

New admissions seal all 200 cells into one canonical paid session, with one
cumulative funding cap, bounded TTL, single-use authority and independent
watchdog. The watchdog's absolute deadline is capped by the sealed funding
expiry, including after time spent queued or staging. The existing execution
monitor still stops on founder hold, cancellation, deadline or prolonged loss
of control contact. A lost allocated session cannot authorize a replacement
GPU. Only a refusal proving zero allocation may be restaged.

The worker retains its fresh-interpreter/Isaac process per case. It fsyncs a
digest-bound completed-child checkpoint after each case. After the first 25
cases, more than half failing at the policy interface stops the session and
seals that prefix before normal provider teardown. Ordinary task failures do
not trigger this gate. A missing/timed-out later child preserves the previously
completed prefix and raw child evidence; incomplete measurement stays unknown.
Checkpoints are provider-local until the normal output transfer completes;
they do not claim protection from loss of the provider disk itself. Existing
multi-shard admissions and completed diagnostic evidence remain readable.

The version endpoint was a synchronous FastAPI handler. It competed for the
same finite AnyIO worker pool as long intake handlers. A regression test
occupying that pool reproduces the timeout in the previous implementation.
The endpoint now runs one shared in-flight probe outside that pool, with a
two-second response deadline. Polling cannot multiply hung probes. A slow
probe returns 503 with unknown identity and all disk roles refused, never a
cached success; normal responses retain current source-identity checks.

The killed historical process has no retained Python stack dump. Its exact
blocking intake operation remains unobserved. Worker starvation and unbounded
probe waiting are reproduced source defects addressed here, not a claim that
the historical request responsible for every occupied worker was identified.

Verification protects these claims:

- The real executor, staging, canonical allocator and team transport run all
  200 hidden cells through one provider stand-in, post one terminal, keep the
  total cap, remove the credential and refuse a second execution.
- The real parent worker, validator, sealing and aggregation run 200 isolated
  child stand-ins, exercise both sides of the 25-case gate, and retain a prefix
  after a lost child. The existing lifecycle rehearsal executes real episode,
  client and media seams with fake Isaac; the import-closure suite imports the
  sealed provider bundle in an isolated interpreter.
- Watchdog tests prove the detached process receives the exact absolute cap;
  cancellation, founder hold, stale control contact, ambiguous spawn identity
  and provider-zero closeout regressions remain enforced.
- HTTP tests saturate the shared pool, bound delayed diagnostics, limit twelve
  concurrent polls to one probe, recover afterward and preserve conflicting or
  unavailable source identity and busy-disk failure behavior.

These hermetic tests make no new GPU execution, customer performance, physical
safety or ADP qualification claim. Promotion additionally requires exact-source
CI/signing, reviewed merge, canonical deployment, live version readback and
preserved operational holds.
