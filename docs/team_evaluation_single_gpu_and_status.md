# One GPU per team evaluation and bounded release status

This repairs free-beta journey L5 (an invited team's scored Task Evaluation
Run). It also exercises the runtime/teardown precursor of ADP-009D's day-28
rehearsal; authored fixtures do not qualify that gate or physical performance.

The previous controller made eight separate 25-cell GPU sessions. Two setup
failures and two partial batches made the retained internal diagnostic use
eleven sequential rentals. A machine boundary was being used as a batch
boundary, despite the existing worker already isolating each simulator cell.

New admissions seal all 200 cells into one canonical paid session, with one
cumulative funding cap, a TTL covering the selected control rate, single-use authority and independent
watchdog. The watchdog's absolute deadline is capped by the sealed funding
expiry and the current authenticated execution deadline, including after time
spent queued or staging. The execution deadline can only shorten the sealed
funding expiry. The existing execution
monitor still stops on founder hold, cancellation, deadline or prolonged loss
of control contact. Insufficient funding, execution time or runtime budget refuses before paid
allocation. The same canonical cost-window calculation runs before claiming
the one-use policy credential; supported faster rates still require a profile
whose approved cap covers their full plan. The service timeout covers the maximum supported planner window
instead of silently truncating it to 16 hours. Dispatch also reserves 30 minutes
for metadata, sealing and the initial queue handoff, plus ten minutes for
allocator startup before claiming the token. The runner reserves those ten
minutes before writing a launch intent; canonical authority consumption retains
five minutes for provider-object staging before the full rental TTL.
The current authenticated execution window is checked immediately before
credential claim, with the same hold/cancellation/identity checks as the runner.
Queue delays can still cause a later refusal; they never extend an approval.
A lost allocated session cannot authorize a replacement
GPU. Only a refusal proving zero allocation may be restaged.

The worker retains its fresh-interpreter/Isaac process per case. It fsyncs a
digest-bound completed-child checkpoint after each case. The shared aggregation
validator checks each new child, its scenario/contract/candidate bindings and
its native episode evidence before another case starts. A malformed child is
retained raw while only the validated prefix is sealed. Prefixes containing
only valid pre-policy failures are retained with zero episode rows and no
policy-load claim, including when interruption recovery fails. After the first 25
cases, more than half failing at the policy interface stops the session and
seals that prefix before normal provider teardown. Ordinary task failures do
not trigger this gate. A missing/timed-out later child preserves the previously
completed prefix and raw child evidence; incomplete measurement stays unknown.
Checkpoints are provider-local until the normal output transfer completes;
they do not claim protection from loss of the provider disk itself. The independent transport and sealed-bundle checks accept at most 200 team
cases; the frozen-pair canary remains at ten cells/twenty rollouts. Fallback
SSH progress retains only the latest bounded milestones and uses the sealed
bundle's cell count, including cells 100-199, without widening legacy
canary scopes. Existing
multi-shard admissions and completed diagnostic evidence remain readable.

The version endpoint was a synchronous FastAPI handler. It competed for the
same finite AnyIO worker pool as long intake handlers. A regression test
occupying that pool reproduces the timeout in the previous implementation.
The endpoint now runs one shared in-flight probe outside that pool, with a
two-second response deadline, including the supported Python 3.10 asyncio
timeout type. Polling cannot multiply hung probes. A slow
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

A review concern about the paired stream's 8 GB archive / 640 MiB needed-set
limits was checked against the actual dispatch: team sessions explicitly use
full download; only the frozen pair enters the streaming resolver. A sparse
ZIP regression exceeding 8 GB extracts through the actual download path with
the stream budget reduced to 1 byte. The team transport still quotes its
cell-scaled output allowance (80 GB for 200 cases); no global stream cap is
raised and no missing evidence is treated as present.

The generated provider archiver had a separate 512 MiB terminal-result bound
that also applied to full download. Its allowance now scales from the sealed
cell count, retaining the existing 512 MiB per twenty-rollout budget: at most
200 team cases / 5 GiB. The frozen pair keeps its original bound. A sparse
2,522,303,830-byte capacity stand-in (ten times the retained V25 twenty-row
size) is archived byte-for-byte; one byte above the scaled bound is refused.
That is a CPU retention-capacity test, not measured 200-case GPU output.
