# Control-plane startup contract

"Startup" is everything between a merged commit and a paid GPU running valid
inputs: deploy, release package, profile publication, Website submission,
preparation, compile, activation, dispatch. Audited 2026-09-02 against the
transcripts of that day's attempts.

## Failure ledger

Every failure below was discovered on the production host, after the step had
already run, and each cost a replay of the whole chain.

| Failure | Where it surfaced | Structural cause | Fix |
|---|---|---|---|
| Preparation unit failed to start: `ReadWritePaths` named `disk-reservations`, which no deploy created | first Website run after #1537 | unit sandbox paths were provisioned by hand-written per-path installers, one per incident | #1550, then generic provisioning from the unit files (this contract) |
| Same, for `storage-pins` | first Website run after #1559 | same | #1560, then the same generic step |
| Overlay archive and pre-created run directory unreadable by the service account | two paid attempts | inputs staged as root; no access proof before the allocator | #1535 service-access preflight |
| Wrapper layer published under a plural artifact kind | layer publication | the layer URI prefix was a free-form string typed by the operator | prefix derived from the object-store contract; a deviating prefix is refused at build time |
| Release-window template bound to an older commit | activation gate, after preparation and compile | template built for one commit, reused for another; profile validation did not open it | #1561 fetches and validates the template at publication |
| Installer filename typo, file-mode assumption, missing secret-env binding | profile installation | the installer was a script outside the repository, never tested | use `publish_task_evaluation_launch_profiles.py` and the in-repo publishers only |
| Worker lifecycle defects (bundle import, second preflight, lost result, cadence, Replicator graph) | five paid Quick-10 attempts | worker orchestration exercised only on a GPU | #1535 hermetic lifecycle rehearsal |
| Control-plane boundary drifts (progression CLI kwarg, activation lineage) | two no-spend replays | neighbouring stages pinned only by hand-written fixtures | #1547 hermetic control-plane rehearsal |

The pattern is one sentence: a step that runs only in production is validated
only in production.

## Rules

1. **Sandbox paths come from the unit files.** `deploy_control_plane_commit.py`
   reads `ReadWritePaths` and `ReadOnlyPaths` from the staged release's own
   units, creates any missing service-owned directory under
   `/var/lib/blueprint` (never repairing one that exists), and refuses the
   deploy before the release link moves when a path it may not create is
   absent and not marked optional. A new ledger needs a unit line, not an
   installer. `tests/test_deploy_control_plane_commit.py` runs this against the
   real unit files and requires every named path to be a classified storage
   root.
2. **Cross-bindings are derived, never typed.** The layer URI prefix has one
   valid shape per bucket (`external_layer_uri_prefix_for_bucket`); the
   builder takes `--external-layer-bucket`. The same rule applies to any new
   binding between a package artifact and a production contract: compute it
   from the contract's constants and refuse anything else at build time.
3. **Every production-only step has a hermetic rehearsal.** GPU worker (#1535),
   control plane intake-to-dispatch (#1547), deploy provisioning (this
   contract). A new step is not done until its rehearsal exists.
4. **Publish with the repository's tools.** Profiles and catalog through
   `scripts/publish_task_evaluation_launch_profiles.py`; wrapper layers through
   `task_evaluation_native_arena_preparation_adapter publish-runtime-source-layers`;
   large artifacts through `publish_configured_scene_artifact`. A shell script
   in a temporary directory is where the typo lives.
5. **Measure the deploy.** The deploy receipt records `stage_timings_seconds`
   per stage so a slow deploy is a measurement, not a feeling. As of the
   audit, staging the two per-commit runtime trees and validating the
   416-profile catalog dominate; the intake restart itself takes seconds.

## What still costs minutes

- Runtime trees are rebuilt per commit even when their inputs did not change.
  Keying them by input digest and hardlink-cloning an identical predecessor
  would make a deploy a link, not a build.
- 416 published profiles are validated on every deploy and protect 321
  commits from release retirement. Retiring consumed one-shot profiles from
  the catalog is what makes both the deploy and the retirement effective.
