# Task Evaluation scene-configuration release runtime

Website-started scene configuration uses the existing **Task Evaluation Run**.
It is not a scene-specific launch harness. A production release publishes two
reusable exact-SHA inputs before its active symlink moves:

- an InteriorGS render runtime containing pinned Linux Node, Chromium,
  lockfile-resolved Spark/Playwright dependencies, and byte-identical renderer
  sources from the release;
- a scene-configuration toolchain containing the released ArtiFixer, Content
  Agents rigid-replacement, and native SimReady/USD import-qualification
  components.

The deployer performs full-byte readback as the `blueprint` service account and
writes only non-secret exact paths and public object-store prefixes to
`/etc/blueprint/task-evaluation-scene-configuration-release.env`. The
preparation and activation services load that file after the shared credential
environment. Missing, mutable, mismatched, or unreadable runtime bytes block
the deploy before release activation.

## One-time host bootstrap

The two bootstrap commands are platform provisioning, not scene preparation.
They contact public source/package endpoints, allocate no provider resource,
and are idempotent after their immutable outputs pass validation.

```bash
python scripts/bootstrap_task_evaluation_scene_configuration_sources.py \
  --source-root /var/lib/blueprint/task-evaluation-inputs/sources

python scripts/bootstrap_task_evaluation_splat_render_prerequisites.py \
  --repository-root /opt/blueprint/task-evaluation-control-plane \
  --output-root /var/lib/blueprint/task-evaluation-inputs/system-runtime-prerequisites/splat-render-v1 \
  --readback-user blueprint
```

The renderer bootstrap pins Node `22.21.1` to the official archive SHA-256,
uses `npm ci` against the committed lockfile, and lets the pinned Playwright
package select its matching Chromium revision. Every retained executable,
library, package, and browser byte is inventoried in the immutable prerequisite
manifest.

## Release behavior

`scripts/deploy_control_plane_commit.py` uses the governed roots above. For
every exact commit it:

1. stages the detached clean release without activating it;
2. validates the platform prerequisites and pinned public source mirrors;
3. publishes and reads back the exact-release renderer and component toolchain;
4. atomically writes the non-secret service environment;
5. moves the source checkout and active release surfaces together;
6. installs the exact systemd units, restarts intake, and proves the reported
   commit.

After that release operation, robot teams submit versioned scene, task, robot,
camera, runtime, rights, and success-criteria references through the Website.
ArtiFixer, Content Agents replacement, and native import qualification execute
inside the parent production scene-configuration run after submission. Later
episode-evaluation runs reuse the configured-scene revision and do not repeat
scene construction.

