# Replay a real website job offline ($0)

One command pulls what a job's replay needs through the operator door (reads only), replays every remaining stage
of its website path against this tree in a private namespace, and says where each stage stops. It buys nothing,
sends nothing and never changes the pulled evidence. Use it before deploying a fix for a stopped job: the first
stop after your fix is the next problem, found in minutes instead of a deploy and an hour's wait.

```bash
. ~/.blueprint-cloud/env.sh
PYTHONPATH=src "$BLUEPRINT_PIPELINE_PYTHON" -m blueprint_pipeline.job_replay itest-<request id>
PYTHONPATH=src "$BLUEPRINT_PIPELINE_PYTHON" -m blueprint_pipeline.job_replay \
  /var/lib/blueprint/pubsub-handoffs/<bucket>/scenes/site-<id>/captures/walkthrough-<id> --report out.json
# step past a known stop, flagged in the report, to see what comes after it:
... --past website_retained_static_masks_manifest_changed
```

## What it does

1. **Pull.** Every record of the capture (JSON and text, and any file of 256 KiB or less) and its scene's
   withdrawal folder go to `<scratch>/cache/<host path>` (`--scratch`, default `$BLUEPRINT_JOB_REPLAY_SCRATCH` or
   `/tmp/blueprint-job-replay`). Bulk files (frames, splats, meshes, zips) come only when a stage names them: a
   missing input, a path the code looked for and did not find, or a listed folder whose unpulled files are 64 MiB or
   less. Each is pulled once, with its frame set, and the replay runs again (`--rounds`, default 8). A cached file
   whose host size and mtime are unchanged is reused after its digest is checked, so a second replay of the same
   job pulls only what changed. `--pull <glob>` adds files; `--pull-all` takes the whole capture (GBs, about
   1.7 MB/s through the door). `--offline` replays from the last listing and what is already cached. When the
   rounds run out while a stage still names unpulled files, the table says `NOT CONVERGED` (its stops may be the
   pull's, not the job's): run it again (cached) or raise `--rounds`. A first pull of a job at an Atlas stage can
   be 1-2 GB of frame sets, and one Atlas step can run 20+ minutes here.
2. **Replay**, each stage in its own process, in a mount and network namespace (`unshare`) where the pull stands
   at its own host paths: an overlay over `/var/lib/blueprint` whose read-only layer is the wanted cached files
   (hard links, hidden once mounted) and whose upper layer, per run, takes every write (`work_view: overlay`; a
   plain copy where overlayfs is not allowed). A local folder of that name is hidden, never touched:
   - `atlas_preview`: the Atlas preview controller (`atlas_preview_replay.run_steps`);
   - `preparation`: once the preview is ready (or the job has none), `run_e2e.run_end_to_end` with
     `resume_completed_stages=True`, the call the listener's recovery makes once its claim and admission pass;
   - `scene_handoff`: the same run from its call of `website_scene_handoff.prepare_website_scene_handoff`;
   - `company_simulation`: `website_company_simulation.publish_company_simulation`, the package compile.
3. **Report**: one JSON report (`--report`, and `job_replay_report.v1.json` in the run folder, rewritten after every
   round, so a run stopped part way keeps its last complete round) and a table:
   stage, ran / skipped / assumed, stop kind and code, would-buy items with their maximum USD, duration, peak RSS.
   A stage that stops ends the chain; the later stages say `skipped` and why.

Stop kinds: `completed` (the stage's natural end: `ready`, `reached_scene_handoff`, the handoff's status),
`would_buy`, `refusal`, `missing_input`, `external_effect` (a WebApp, artifact store or provider write the replay
never makes; for the company package, the upload after the build: `package` in the report has the zip's digest
and members), `waiting` (an Atlas job that did not move, a package with no source yet) and `error`. A refusal lists
`unpulled_before`: what the code looked for just before it and did not find.

## `--keep-package`

The door does not serve a package that holds an object: `object.usdz` is an archive inside the zip
(`secret_content_refused:nested_compressed_member`). With `--keep-package` the package stage copies the zip it builds
into its run folder as `package.zip` and the report names it (`package.kept`). When its digest is the one the host's
`pipeline/company_simulation/company_simulation.json` records (`package.digest`), these are the bytes the host
delivered, to open and check here.

## `--past <code>`

Steps past a named stop, on the record. Where the step has a retained output the host made, that output stands in
for the recomputation and the stage goes on (`job_replay.SUBSTITUTIONS`):

- `atlas_completion_gap_survey_changed`: the completion's retained `completion.json` (one that did not pass);
- `website_retained_static_masks_manifest_changed`: the kept objects' masks the host completed
  (`<task_masks>/<binding>/task_masks.json`).

Any other code marks its stage `assumed` and the next stage reads the files as the host left them (a stepped-past
preparation does not run the handoff: nothing stands in for its clean plate result). Every assumed step is in
`assumed`, the table prints `ASSUMED` lines and `NOT A CLEAN RUN`, and the report says `clean: false`.

## Boundaries (no flag lifts them)

- No network: the namespace has only `lo` (`boundaries.network_probe`), and every socket connection, name lookup
  and network tool in a stage's process is refused and recorded. Any attempt fails the run (exit 2).
- No credentials: credential-like variables are removed, and the container's credential files (`~/.aws`,
  `~/.config/gcloud`, `~/.ssh`, ...) are masked inside the namespace (`boundaries.masked_paths`).
- Money: every new purchase stops as `would_buy` with its provider, operation and maximum USD; WebApp
  `preparation-spend` is admitted under the snapshot's ceiling exactly as the WebApp admits it, then stops. WebApp
  reads are answered from the job's retained task context, Atlas configuration and scene sponsorship. A
  `preparation-settlement` (it only releases what a bought call held beyond its cost) is recorded under
  `unsent_settlements`, never sent, and answered as no answer, so the code keeps the hold as it does when the
  WebApp does not answer. Every other WebApp write stops as `external_effect` with its call site.
- Evidence: the replay never sees the cache or its links (tmpfs covers both in the namespace, and overlayfs never
  writes a lower layer); its digest is checked before and after every round (`evidence.unchanged`; a change fails
  the run).
- Door: only `/fs/list` and `/fs/read`. Nothing holds, releases, deploys or starts anything.
- Stand-ins, listed in `boundaries.stand_ins`: a pull carries no owner, mode or inode, so the scene retirement
  fence reads the copy's owner as the service account, its generation store gets the host's mode (0700), and a
  copied folder stands for the host folder its generation record pins. Only this capture's records of that shared
  store are pulled.

Without a private namespace (some CI runners) the tool says `not replayed: private_namespace_unavailable` and
exits 5. It cannot show provider behaviour, the host's own memory (peak RSS is this container's), the listener's
WebApp-only preconditions (owner observation, GCS staging, browser admission) or a record the door refuses
(`pull.door_refused`).
