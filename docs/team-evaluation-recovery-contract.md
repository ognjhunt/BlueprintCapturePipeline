# Team evaluation recovery and evidence contract

Beta journey step L5: safely execute and account for an invited team's own
policy without turning development execution into site-qualified evidence.

## Before allocation

Listing and admission require an actual passed robot-free probe bound to the
scene revision, scored task specification and success contract. Missing,
unavailable and failed probes refuse before paid team allocation. A separate
probe must produce that evidence; a team session cannot obtain its own missing
prerequisite after spend starts.

Funding, team licence-to-run rights, robot model integrity, structure/stance and
initial provider-zero are checked before releasing the single-use bearer lease.
Authenticated policy metadata necessarily requires the released token. The
delivery is therefore persisted only in the private 0700 credential directory,
in 0600 files bound to the exact request and dispatch owner. At most three
prelaunch preflight attempts may reuse that delivery. Failed artifacts are
retained separately, the lease is never released again, and no second paid
attempt is authorized. Exhaustion, cancellation or terminal closeout removes
the private delivery. A response lost before local persistence still requires
reconciliation/new run; this change does not authorize broker redelivery.

Every allocator spawn has a durable launch intent before process creation. A
crash between creation and recording the PID is an unknown allocation, not
permission to spawn again. Re-staging remains restricted to allocator receipts
that prove no provider allocation occurred.

An unresolved spawn intent also prevents terminal settlement and deletion of the
private credential. A momentary global provider-zero snapshot cannot prove that
an already spawned allocator will not allocate later. New shards remain paused
until the original allocator returns its result or its exact process identity is
reconciled. Recorded PID liveness is checked against Linux process-start identity:
a reused PID belongs to another process and is not treated as this allocator or
signalled. A live PID whose identity cannot be established remains unknown.
An incomplete allocator result also leaves that uncertainty unresolved. Secret
cleanup and terminal closeout share the same planned-shard liveness predicate.

## Running work and stop

The runner sends a signed request every 60 seconds:

```text
POST /api/internal/pipeline/agent-runs/:runId/heartbeat
{pipeline_run_id, execution_admission_digest}
```

The WebApp response supplies `ok`, `run_id`, its fixed
`execution_deadline_at_ms` and `last_heartbeat_at_iso`. A definite 409, expired
deadline or cancellation stops new shards. A transient control failure also
defers new allocations; after ten minutes without a valid response the runner
requests teardown, before the WebApp's fifteen-minute stale-heartbeat bound.
The heartbeat does not extend the execution deadline or spending authority.

The founder's existing dispatcher hold at
`/var/lib/blueprint-operator-door/requests/holds/blueprint-agent-run-dispatcher.timer.json`
also stops running team evaluations. It sends SIGTERM only to this run's
recorded allocator with matching Linux process-start identity. Team allocators
cannot inherit the experiment option that ignores SIGTERM. The adapter unwinds
through its existing provider teardown. There is no SIGKILL fallback that could
orphan paid resources. After five minutes of delayed teardown, local processing
returns pending with a retained receipt; independent provider TTL/watchdogs
remain in force. Terminal settlement requires provider-zero evidence.

The internal authored diagnostic has no fabricated WebApp claim. It uses its
immutable owner approval, the earlier of approval expiry and a fixed sixteen-hour
deadline, and the same founder hold. Its real passed probe is now required before
staging or issuing the diagnostic token. See the
[diagnostic runbook](runbooks/internal_team_evaluation_diagnostic.md).

## Outcome accounting

Infrastructure/media failures may be void episodes within the existing five
percent limit. Policy timeouts and invalid policy actions remain failures even
when secondary media sealing also fails. Every planned cell remains in the
reported outcome accounting. Current runs remain development-only.

Cycle metrics use digest-verified simulator-clock task-state samples for every
successful episode. The conservative duration includes terminal settling. An
incomplete clock leaves the measurement null. It is neither provider wall time
nor measured physical latency.

The own-robot HTTPS runtime retains its actual request artifacts and marks
inference exceptions as policy failures, while request-persistence faults remain
infrastructure voids. Both runtimes keep process interruptions out of policy
failure classification. Own-robot task-state clocks include the initial state
and any final partial sampling stride before their trace digests are sealed.

An incomplete or stopped run sends settlement `execution_status: "blocked"`
and `execution_progress`:

```json
{
  "observed_attempted_episode_count": 25,
  "scored_episode_count": 24,
  "succeeded_episode_count": 20,
  "failed_episode_count": 4,
  "void_episode_count": 1,
  "measurement_complete": false
}
```

These are observed counts, potentially lower bounds when measurement is
incomplete. `episodes_run` equals `scored_episode_count`; sponsored charge stays
zero. `blocked_before_any_episode` is true only when measurement is complete
and the observed attempted count is zero. Missing planned shard artifacts,
unknown allocator outcomes and missing/tampered worker output cannot establish
that no episode occurred. Raw partial receipts remain available and cannot
qualify a recommendation.

## Qualification gap

No site-qualified result or live team-policy success is created by this release.
Current admissions keep `evidence_scope: "development_only"`, and result scoring
keeps `claim_ceiling: "diagnostic_policy_execution"`.

The WebApp's future qualified-result contract requires the distinct
`site_task_simulation` claim ceiling and actual registration/fidelity receipt
bindings in `scoring.site_task_qualification` (`capture_id`,
`scene_revision_digest`, `registration_receipt_digest`,
`fidelity_review_receipt_digest`, `evidence_scope: "simulation"`). This Pipeline
does not manufacture those fields. Its website registration currently states
that physical scale/registration are unproven; the canonical 3DGS registration
producer qualifies appearance/frame alignment only. A complete measured
site-task registration/fidelity producer and actual evidence remain prerequisites
to a site-specific recommendation. Owner acceptance alone cannot supply them.

## Release order and verification

The historical beta retention policy now records that original collection terms,
verified legal holds and rights/privacy decisions override internal archive
defaults. It cannot authorize a retention extension or training use. The existing
`website_capture_retention_review.v1` metadata remains unchanged, including unknown
original terms; this is no proof of notice, counsel approval or bucket changes.

Deploy the WebApp heartbeat and partial-blocked-settlement APIs before this
Pipeline runner. Merge first, then use the supported operator door to deploy the
merged immutable commit. At the read-only production baseline on 2026-10-16,
an existing native-policy canary held the paid launch lock and a GPU. Do not
interrupt or race that work; wait for clean lock/resource closeout before release.
No live provider/GPU execution was performed for this change.

Focused hermetic verification:

- Credential/executor tests: 83 passed (loopback transport requires the sandbox's
  network capability).
- Result/admission/episode/session contracts: 84 passed as part of the combined
  167-case set; the same set's one socket-restricted case passed separately when
  loopback access was enabled.
- Complete sharded launch rehearsal: 19 passed in 201.19 seconds. Later focused
  tests cover bounded preflight repair, cancellation, missing shard evidence,
  spawn-before-PID recovery, control outage and delayed stop teardown.
- Lifecycle, team rehearsal and offline media faults: 37 passed.
- Provider import closure: all six cases passed, including the isolated real
  bundle import after installing its declared test dependencies.
- Internal diagnostic: four passed with an explicit hermetic probe fixture;
  production still requires the actual retained probe.
- After independent review, nine crash-window/PID/unknown-result/stop regressions
  passed, and the 200-cell happy run, first-shard failure gate and founder-stop
  rehearsals passed again (three cases). Retention authority validation passed
  nineteen tests and its repository validator command.
- Own-robot/result timing and cause-classification tests: twenty-five passed.
  Lifecycle/import closure plus the DROID transport-failure regression:
  twenty-six passed. Scene/probe tests: seven passed.
- Changed launch/closeout/result/admission/registry tests pass Ruff. Older touched
  files have no new lint findings against their committed baseline.

These tests establish software behavior, not production configuration, a real
team engagement, site fidelity or the first evidence-backed recommendation.
