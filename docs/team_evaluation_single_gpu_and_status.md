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

Team terminal rows now contain scoring/motion summaries and digest-bound
references to the complete sealed child files. The parent releases each raw
episode history before starting the next case; raw children and episode
artifacts stay in the output archive. Cycle-time scoring loads and verifies
one state-trace artifact at a time and retains the original simulator-clock
formula. The terminal cap remains 512 MiB rather than being raised to 5 GiB.
Native archive inspection reads a small status sidecar, verifies its own digest
and streams the sealed result bytes to verify their hash. It reports that the
inspection covers status metadata rather than full JSON validation; normal
extraction/scoring validates the compact result. Old archives keep their bounded
JSON fallback. Extraction enforces the original 512 MiB terminal-size bound.
No parser dependency is introduced.

The additive summary contract is explicit:

- `episode_summary_kind=team_terminal_episode_summary.v1` means `episode`
  retains the unchanged score and motion evidence; its complete original body
  remains in the sealed child file referenced by `source_child_result`.
- `source_child_result` records the relative path, byte size, SHA-256 and
  original child result digest. Checkpoints continue to reference that original
  digest, rather than the in-memory summary digest.
- `evidence_artifacts.state_trace` remains the original digest-bound trace.
  Closeout loads it per successful case for the unchanged simulator-clock
  cycle calculation. Missing/invalid evidence keeps cycle time unmeasured.
- `native_task_arena_policy_canary_status.v1.json` is a small status sidecar
  bound to the exact terminal bytes. `runtime_result_parse_scope` declares
  `hash_bound_status_metadata_only`, with `runtime_result_json_validated=false`;
  this read is orchestration metadata, not full result or policy qualification.

Legacy full-body records and the frozen pair retain their existing formats.

The full-download collector streams in at most 1 MiB reads to a temporary file,
checks destination capacity and declared response length, then atomically
publishes only a complete transfer. Interrupted or short downloads remove the
partial file and preserve an existing destination. The existing provider-output
capacity guard remains in place. Remote progress uses a validated monotonic
cell/stage cursor, so rotation of the bounded 64-line tail still advances the
watchdog through cell 199. Reordered or repeated milestones cannot advance it.

Before full archive extraction, the existing shared disk ledger reserves the
sum of all uncompressed member sizes against current free space, other live
writers and the normal floor. The downloaded ZIP is already included in that
filesystem usage. Admission precedes replacement of any earlier extraction;
refusal preserves both it and the ZIP. The hold is released on success or write
failure, and a partial extraction cannot supply a completed runtime result.

Download admission also holds the declared response size in that shared ledger
before the partial file is opened, respecting floors and other live writers.
Responses without a declared length refuse before a body read. An uncollected
paid output retains its remote objects and protected recovery references;
normal cleanup follows a complete local collection or proof of no allocation.
