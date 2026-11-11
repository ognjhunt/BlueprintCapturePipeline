# Release binding of a website job

What a deploy does to a website job in flight, check by check. Mapped from the
code on 2026-11-10 (base `61e55f776`). Line numbers drift; the names do not.

## Two facts that decide everything

1. **The commit is the attestation unit.** The CPU preparation is not a neutral
   artifact. `materialize_website_submission` (`website_native_submission.py`)
   writes `expected_production_commit`, the exact release binding (deploy
   receipt, toolchain and renderer publications) and a namespace that contains
   the commit into the preparation request. That request is the input the paid
   scene construction is attested against. The preparation worker, the launch
   intake for preparations and activations, and the activation worker all accept
   it only when its commit equals the commit they run.
2. **`release["runtime_digest"]` changes on every commit.** It is
   `canonical_digest({"toolchain": ..., "renderer": ...})` over the two
   publication receipts (`task_evaluation_scene_release_binding.py`). Each
   receipt carries `source_commit`, and the toolchain manifest digest includes
   it (`scripts/build_task_evaluation_scene_configuration_toolchain.py`). It
   adds nothing to the attempt identity that `source_commit` does not already
   carry, and it is not stable across a docs-only commit.

So "adopt a completed preparation from release A on release B" (rule A) cannot
be done in scene progression alone: every later stage on B refuses a request
that names A, and those refusals are what make a paid launch run exactly the
code its records name. They are not weakened here.

## The map

Kinds: **paid**: protects that a paid launch runs the attested code its records
name. **evidence**: protects that an artifact names the code that made it.
**none**: protects nothing by itself (the checks after it already refuse); it
only schedules work. **retry**: uses "a new release shipped" as the signal to
retry once.

| # | Where | Compares | Kind | A job in flight when it fires |
|---|---|---|---|---|
| 1 | `task_evaluation_scene_release_binding.resolve_release_binding` | release record vs running checkout | evidence | Tick refuses (`running_release_mismatch`) until the receipt for the new commit exists. Nothing lost. |
| 2 | `task_evaluation_scene_progression._advance_intent`, identity block (`attempt_commit != release.source_commit`) | attempt commit vs live release | none | Calls `_release_successor` (row 3). The attempt id digests intent, binding, commit and `runtime_digest`, so every deploy names a new attempt. |
| 3 | `task_evaluation_scene_progression._release_successor` | (same) | none, except the paid-row checks inside it | Before a paid reservation: retires at once. With a reservation: waits for a terminal preparation queue row and zero ownership, settles the old rows, retires. A new attempt then rebuilds factory, publication, preparation and activation on B. **CPU repeated; the paid reservation is released and taken again (no spend). Since this change, a launch that was submitted, queued or running holds the attempt (row 18).** |
| 4 | `website_scene_dispatch.materialize_website_attempt` (`website_factory_attempt_mismatch`) | attempt commit vs release snapshot | evidence | Only runs for a new attempt. |
| 5 | `_advance_intent` factory receipt (`factory_receipt_invalid`) | factory commit vs attempt commit | evidence | Same attempt only; never fires across a deploy. |
| 6 | `_publish` / submission publication (`expected_source_commit`) | factory commit vs running checkout | evidence | Same attempt only. |
| 7 | `task_evaluation_launch_preparation_worker.materialize_preparation_references` (`launch_preparation_worker_source_commit_mismatch`) | request commit vs worker commit | paid (attested input of the construction) | An A request still pending when B runs is closed `blocked`. `_queue` accepts that result as `rejected_old_release`. **CPU only.** |
| 8 | `live_pipeline_intake_service` preparation and activation handlers (`*_production_commit_mismatch`) | request commit vs intake commit | paid | An A submission arriving after the switch is refused; nothing spent. |
| 9 | `task_evaluation_scene_configuration_activation_automation.advance_scene_configuration_activation` (`preparation_bound_to_superseded_release`) | preparation result commit vs running commit | paid | An A preparation is never activated on B. |
| 10 | `task_evaluation_launch_activation_worker` (`launch_activation_worker_source_commit_mismatch`, configured-controls autostart intent) | activation request commit vs worker commit | paid | A queued A activation is refused on B; nothing spent. The 285-minute door hold exists to keep this stage from being cut. |
| 11 | `task_evaluation_launch_dispatcher` | request commit vs its own profile only (no live-release check) | paid (internal consistency) | A launch already in the queue runs from its own profile on B. This is why row 18 is safe. |
| 12 | `task_evaluation_scene_preparation_completion.reconcile_preparation_completion` | profile, result and receipt commit with each other only | evidence | Commit-agnostic join: a completed A launch closes the job on B (already rule A). |
| 13 | `task_evaluation_scene_terminal_reconciler` (robot-team join) | attempt, request and profile commit with each other only | evidence | Commit-agnostic (its module docstring still says otherwise). |
| 14 | `task_evaluation_controls_autoprovision._registered_terminal_adoption` / `terminal_adoption_source` (`expected_production_commit=release`) | adopted controls result vs live release | not established | Robot-team jobs only. Not explained here; left as is. |
| 15 | `team_scene_packet_worker._new_release_untried`, `_unscorable_half_untried` | status release vs live release | retry | A blocked or unscorable team half is compiled once more per release. **CPU only.** |
| 16 | `rigid_robot_free_probe_runner._new_release_untried` | terminal release vs live release | retry | A `staging_refused` probe is retried once per release. **Paid probe ($1 cap) may run again.** Bounded by one per release. |
| 17 | `beta_policy_screening_launch` (`beta_screening_implementation_commit_changed`) and `beta_policy_screening_coordinator._replan_scope` | frozen implementation commit vs live commit | paid | Running candidates survive. Preparing rows are refused, queued rows are closed `replanned_for_new_implementation`, and a new campaign is planned. **No paid run repeats; queued paid work is replanned.** |
| 18 | `task_evaluation_scene_release_hold.previous_launch_in_flight` (new) | the attempt's launch: queue, run directory, receipt, own submission record | paid | Holds the attempt (`running` / `previous_release`) while its launch is in transit, queued, claimed, running, or without a terminal receipt. |

Headline: 18 rows. By kind: paid 7 (rows 7, 8, 9, 10, 11, 17, 18), evidence 6
(1, 4, 5, 6, 12, 13), none 2 (2, 3), retry 2 (15, 16), not established 1 (14).
Outside this chain, `grep -rln "_commit_mismatch" src/blueprint_pipeline | wc -l`
counts 77 modules.

## What a deploy repeats

| Stage of the job at the deploy | Before this change | Now |
|---|---|---|
| No attempt yet | nothing | nothing |
| Preparation building, queued or done; no paid reservation | rebuilt on B: **CPU** (factory, staging, preparation worker). Large inputs are hard-linked and content-addressed from the earlier attempt, so bytes are not copied again. | the door holds the deploy for it, for 60 min at most (`UNPAID_CAP_SECONDS`); rebuilt only if the deploy goes out first |
| Paid reservation and activation staged, launch not yet submitted | old rows settled, rebuilt on B: **CPU** plus a new activation and a new reservation (no spend) | same |
| Launch submitted to the WebApp, not yet queued | retired and rebuilt: a **second paid launch** was possible while the first was still delivered and dispatched | held; the first launch finishes and is joined |
| Launch queued in `task-evaluation-launches/pending` | same risk: ownership reconciliation reads only `processing` | held |
| Launch claimed or running | ownership reconciliation refused every tick and the job showed `blocked`; with provider zero but a live lease it also wrote a new ownership record each tick | held, shown `running` / `previous_release` |
| Launch ended | completed: joined first (row 12). Failed or blocked: retired and rebuilt under the existing rules | same |

The attempt still reports the existing blocker
`previous_release_attempt_not_terminal`; no new state field is written, so a
job whose state was written by the previous release keeps progressing.

## What is not done, and why

- **Rule A for the CPU preparation.** Not safe in progression alone (fact 1).
  It needs rows 7 to 10 to accept an earlier release, which means running that
  release's own workers.
- **Rule B for the CPU preparation and the activation.** The release tree is
  kept, and the scene-retirement supervisor can already load a worker from a
  bound release source (`task_evaluation_scene_retirement_supervisor.main`).
  The next step is for workers to *skip* (leave pending) a request bound to
  another retained release instead of closing it `blocked`, and for a worker
  loaded from that release's tree to process it. Claims are `O_EXCL`, so two
  releases' workers cannot take one row. Risk: two releases of the activation
  worker run at once against shared queues and the spend guard; it touches
  paid-launch code and needs its own review.
- **Attempt identity.** Dropping `runtime_digest` from the identity, or keying
  it on a runtime-only digest, changes nothing while rows 7 to 10 compare
  whole-repository commits.
- **Row 14** is not explained; it is left as it is.
- **In transit for robot-team jobs.** Their launch is submitted by the
  configured-controls worker, whose record progression does not read. The hold
  covers their queued and running launches, not the WebApp transit window.

## Adopt or rebind: $0 work across a deploy

Rule for this table: a $0 artifact may be adopted by a later release only if no
paid launch is admitted against it. The launch activation worker admits a
paid launch against the preparation request, its result receipt, the adapter,
the construction envelope and the toolchain, and checks that each names its own
commit (`task_evaluation_launch_activation_worker.py`, the checks of
`expected_production_commit` near lines 327, 441, 650, 972 and 1162). So the $0
preparation of a rigid website job is part of the paid admission. It is not
adopted across a commit.

| # | Guards | Adopt or rebind | Why |
|---|---|---|---|
| 1 | which release is live | strict (not an artifact) | Every later binding is read from it. |
| 2 | the $0 attempt id | rebind | The attempt's `source_commit` is copied into the paid reservation row (`_activation_link`), and settlement requires factory, attempt and row to name one commit. Serves money: left strict. |
| 3 | retiring the $0 attempt | rebind | It could adopt only if rows 4 to 10 accepted another commit. |
| 4 | $0 factory (staged bundle, request) | rebind | The request names `expected_production_commit` and the release's toolchain publication, which the activation worker verifies. Serves money: left strict. |
| 5 | $0 factory receipt | n/a | Same attempt only. |
| 6 | $0 bundle publication | rebind | Its namespace and manifest follow the request (row 4). |
| 7 | $0 preparation worker result | rebind | The activation worker checks its receipt's `source_commit` (line 1162). Serves money: left strict. |
| 8 | intake admission | strict | Paid admission. |
| 9 | activation staging | strict | Paid admission. |
| 10 | $0 activation (about an hour of CPU) | strict | Its output is the launch profile and bundle a paid launch runs. |
| 11 | dispatch | strict | Paid. A launch already queued finishes (row 18). |
| 12 | completed scene configuration | adopted today | The join reads the run's own commit. |
| 13 | completed robot-team result | adopted today | Same. |
| 14 | adopted controls result | strict | Not explained here. |
| 15 | $0 team scene half | adopted today; retried once per release | A completed half is kept. Release is used only as the signal that a fix may have shipped. |
| 16 | paid probe retry | strict | Paid. |
| 17 | paid screening | strict | Paid. |
| 18 | a paid launch already left | strict | Paid. |

$0 work that a deploy already does not repeat:

- the website preparation and its registration (`website_scene_dispatch.resolve_website_source` keys on the request digest, not the commit);
- the staged input bytes (hard links from earlier attempts, `earlier_attempt_submissions`);
- fetched inputs (the preparation worker's content-addressed store);
- a public scene's mesh normalization (`tests/test_scene_preparation_restart.py`).

What a deploy still repeats for a job that has not submitted its launch: the
commit-bound wrapper (factory request, publication, preparation worker result)
and the activation. A preparation contract version cannot remove that wrapper
without letting rows 4, 7, 9 and 10 accept a request that names another commit.
That is the side-by-side releases change, which needs validation on the host.

## Adoption across a deploy (dark launch, approved 2026-11-10)

Off by default. The host switch `BLUEPRINT_RELEASE_PREPARATION_ADOPTION=enabled`
in `/etc/blueprint/pipeline-control-plane.env` turns it on
(`release_preparation_adoption.py`). With it off, every check above refuses
exactly as before.

With it on, scene progression keeps the attempt (`task_evaluation_scene_release_adoption.py`)
instead of retiring it when all of these hold:

- the attempt is a $0 preparation with a factory and no paid reservation yet;
- its intent digest and source binding digest are unchanged;
- its earlier release is retained: the attempt's release snapshot, deploy
  receipt, provenance and runtime publications still verify;
- the preparation recorded a **contract identity** when it was made, and it
  equals the running code's identity. A preparation made before this change
  records none and is never adopted.

Progression then writes one sealed record,
`<preparation queue>/release-adoptions/<request digest>-<adopting commit>.json`.
Rows 7, 9 and 10 accept the request that names the earlier commit only when that
record exists, the switch is on, and their own process computes the same
identity. They change nothing else they compare.

**Contract identity.** A digest over the source bytes of the preparation's
producer code: the root modules in `PRODUCER_MODULES`, every module they import
(also inside a function), then the module-level import closure of those (about
340 files). It cannot catch:

- a change to a module reached only through a deeper function-level import;
- a third-party package upgrade;
- non-Python data files;
- the toolchain itself. The toolchain is bound separately: the activation uses
  the earlier release's retained tree (`.../scene-configuration/<commit>`) and
  verifies its manifest against that commit.

**Provenance.** The preparation, its worker result and the paid reservation
keep the commit that made the preparation. The worker result, the activation
staging state, the activation result and the launch state each add
`release_adoption` with `prepared_commit` and `adopting_commit`, plus
`worker_source_commit` (the release that ran that step). No record claims one
commit did both.

**Refusal.** A refusal before any provider call marks the adoption refused,
and the next tick rebuilds once on the live release. The refusal is either the
new worker closing the request `blocked`, or the activation worker refusing
the activation (for example, the retained toolchain is gone). A rebuilt attempt
is bound to the live release.

**Not covered:**
- An activation that was staged before the deploy: it rebinds, as before.
- The robot-team (configured-controls) path.
- A second deploy after the paid reservation was made under adoption: as
  before, the attempt is held or rebuilt by `_release_successor`.

## Policy that could change (owner decision)

The door's activation hold (285 minutes) and the "skip the window while a paid
stage is live" rule protect two stages. After this change, a launch that was
already submitted is no longer interrupted by a deploy, so the hold is needed
only between staging an activation and submitting its launch. The policy text in
`AGENTS.md` and `docs/runbooks/beta-operations.md` is unchanged.

Update, 2026-11-11: the hold was widened instead of narrowed. Rule B is still
not built, so the door now also holds a deploy for work that is not paid yet
and that the new release would refuse (the preparation rows above and a
screening candidate being staged), for 60 minutes at most. The 285 minutes
still protect paid work and are counted from when a paid stage is first live.
Not held for: a preparation waiting on its SAM child job, a preparation waiting
for disk, and the episode compilation and scene object discovery requests.
`AGENTS.md` and the runbook state the new rule.
