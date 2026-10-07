# Pipeline handoff release custody

Source inspected at `dee58dde55e9a36f8b309cac78ad69897152183a` (PR2645), with read-only live metadata observed 2026-10-07 21:30 UTC. This is a release map and deployment-gap record, not a deployment or partner-proof claim. Capture is unchanged.

## Affected consumer

[`blueprint-pubsub-handoff-listener.service`](../../../../deploy/systemd/blueprint-pubsub-handoff-listener.service) runs the `blueprint_pipeline.pubsub_handoff_listener` worker through `task_evaluation_scene_retirement_supervisor`, as user `blueprint`, from the physical checkout selected by `BLUEPRINT_PUBSUB_HANDOFF_REPO` (default `/opt/blueprint/task-evaluation-control-plane`). Its Python default is `/opt/blueprint/BlueprintCapturePipeline/.venv/bin/python`, with the selected checkout's `src` on `PYTHONPATH`. Environment files can override defaults; the source default alone does not prove effective live import bindings.

The timer repeats the oneshot, while respecting its owned hold record. The service defaults to staging ordinary captures into the control plane and skipping `run_e2e`. However, [`pubsub_handoff_scene_operations.py`](../../../../src/blueprint_pipeline/pubsub_handoff_scene_operations.py) explicitly invokes `run_e2e` for website captures even when that skip flag is set, using the qualification lane and completed-stage resume. Both changed modules therefore belong to this host release boundary.

The root [`Dockerfile`](../../../../Dockerfile) instead starts `capture_orchestrator`; source inclusion in its image is not evidence that the listener runs there. CLI/local-bundle consumers can also invoke `run_e2e`. Releasing Web's Render service/worker does not establish this separate systemd consumer's revision. No evidence here assigns Pipeline custody to the Web release owner's identity.

## Observed deployment gap

[`release-live-readback.json`](release-live-readback.json) retains the safe metadata projection. The active release link resolves to `89938271a5c92fc0473b01326733199e0b456444`, the baseline before PR2645. The listener is inactive/dead, last invocation 06:02:17–06:03:00 UTC with status 0; its timer is enabled but inactive with no next trigger reported. Intake is active from 10:07:50 UTC. No new authority to change the listener state was inferred.

A privately retained existing deploy receipt records baseline source checkout and active-release agreement, installed unit hashes, and successful restarted intake identity **at that deployment**. Internal receipt locations, invocation identifiers and proof/authority references are not published in this packet. Its provenance is not promotion eligible. The timer's before/after state is enabled/inactive, `requested_intent=preserve`, and `operator_freeze_preserved=true`. This establishes a preserved freeze in that receipt; it does not identify who initiated it or authorize its release.

Existing authorized access can read the selected metadata. Technical access is not human custody or authorization to activate a frozen consumer. Current listener import bindings, a post-fix invocation, and an accountable Pipeline rollout owner remain unproven.

## Supported release and readback

[`docs/OPERATOR_DOOR.md`](../../../OPERATOR_DOOR.md) and [`door-deploy.sh`](../../../../deploy/operator-door/door-deploy.sh) specify the token-gated HTTPS door, not SSH or an arbitrary host shell. A door deploy admits a commit already on `origin/main`, optionally waits for configured workers to become idle, and runs that commit's deploy tool with `--iteration --preserve-configured-controls-state`. Dirty/unpushed-source checks, paid-launch locks, disk admission, and owned holds remain in force. No deploy, merge, timer action, provider call, or workload was performed in this discovery.

[`deploy_control_plane_commit.py`](../../../../scripts/deploy_control_plane_commit.py) stages a release, aligns source and active-release Git identities, installs units, restarts intake, verifies its revision, and preserves automation intent. Its required restart set is intake only. A oneshot listener must load the new checkout on a subsequent authorized invocation; an intake version alone is not listener execution evidence. The operator must retain the exact merged SHA, deploy outcome/receipt, named surface SHAs, installed unit hashes, listener timer state, and post-release listener invocation identity or equivalent evidence of the effective source binding. A timer left frozen must be reported as frozen, not as a functioning handoff consumer.

Door iteration provenance explicitly has `promotion_eligible=false` and `development_only` evidence grade. The [`Production Release Provenance`](../../../../.github/workflows/production-release-provenance.yml) workflow verifies and retains successful exact-main production-promotion evidence; it does not deploy. An ordinary green PR gate or an iteration receipt must not be called production promotion.

Supported read-only metadata used here: `GET /whoami`, `/units/show`, `/fs/list`, and existing receipt `/fs/read`. Source inspection found that `/status` calls intake `/version`, which calls `disk_headroom`; its ledger preparation/lock opening can create filesystem entries. Neither endpoint was called under this task's prohibition on potentially mutating GETs. Unit metadata omits environment values, and no secret files were read. The receipt's old intake identity was not presented as a fresh runtime probe.

## Rollout and rollback constraints

Before rollout, bind an accountable Pipeline operator, exact merged candidate, and existing frozen timer intent. Preserve that intent unless the owner explicitly authorizes activation. Use the established release mechanism; do not silently point only one checkout at the new SHA or start a listener merely to obtain proof. Any eventual controlled handoff exercise must separately satisfy current capture/consent/generation and spend authority.

Record the previous deployed SHA and receipt before rollout. The deploy failure context restores timer/path intent; it is **not** proof of automatic source/release-link rollback after partial activation. Returning to an earlier allowed main commit must use the same coordinated deploy/readback path, and should preserve the freeze. Deployments do not automatically delete old release trees; release retirement is a separate action. Prior launch profiles/preflights bind their original commit and cannot be assumed valid after a revision change.

Rolling back PR2645 restores its false-completion behavior and does not undo job ledgers, superseded receipts, acknowledgments, or work already run. Do not erase that evidence or replay arbitrary paid stages. The fix itself reopens legacy completion only where retained required-stage snapshots demonstrate failure; absent historical snapshots remain an evidence-backed triage task.

No application tests were run for this documentation-only discovery. The earlier regression/CI evidence remains in the parent packet. This record closes the source-to-runtime mapping gap and leaves actual Pipeline release, effective listener revision, timer activation authority, and hosted handoff proof open.
