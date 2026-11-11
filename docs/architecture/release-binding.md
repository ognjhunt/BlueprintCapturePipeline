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
| Preparation building, queued or done; no paid reservation | rebuilt on B: **CPU** (factory, staging, preparation worker). Large inputs are hard-linked and content-addressed from the earlier attempt, so bytes are not copied again. | same |
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

## Policy that could change (owner decision)

The door's activation hold (285 minutes) and the "skip the window while a paid
stage is live" rule protect two stages. After this change, a launch that was
already submitted is no longer interrupted by a deploy, so the hold is needed
only between staging an activation and submitting its launch. The policy text in
`AGENTS.md` and `docs/runbooks/beta-operations.md` is unchanged.
