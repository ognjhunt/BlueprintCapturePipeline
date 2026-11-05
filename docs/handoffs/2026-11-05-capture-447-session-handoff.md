# Handoff: capture-447 screening, cleanup and in-flight PRs (written 2026-11-05 ~14:55Z / 8:55 AM CST)

You are picking up from a Claude Code cloud session on another account that ran out of usage. Its container and transcript are gone; this document is everything you need. Read it fully before acting. Work autonomously: the owner wants progress without being asked questions, except for the hard stops below.

## 0. First 15 minutes, in order

1. Bootstrap the cloud session. Repos: `ognjhunt/BlueprintCapturePipeline` (Pipeline) and `ognjhunt/Blueprint-WebApp` (WebApp). From the WebApp checkout run `bash scripts/cloud/bootstrap.sh`, then `bash scripts/cloud/doctor.sh`. Then `. ~/.blueprint-cloud/env.sh` in every shell, which gives the Pipeline venv, the pinned uv and the operator door URL. Read both repos' `CLAUDE.md` and `AGENTS.md`, and the WebApp's `docs/runbooks/cloud-scene-runs.md`.
2. Check the in-flight deploy (section 3.1). It was launched at 14:48Z as door request `20261105T144801Z-deploy-8daa6693`, unit `blueprint-operator-door-deploy-2f688aa98933-8daa6693.service`: run `operator_door.py journal blueprint-operator-door-deploy-2f688aa98933-8daa6693.service -n 100` and `operator_door.py status` to see whether 2f688aa9 is the running release. Also: `python3 scripts/operator_door.py requests` (Pipeline repo) and `python3 scripts/operator_door.py status`. If commit `2f688aa98933b4449b1b98273d92bf539e2d26d2` (or anything newer on main) is not running, deploy it:
   `python3 scripts/operator_door.py deploy 2f688aa98933b4449b1b98273d92bf539e2d26d2 --operation-key deploy-2f688aa9-budget30 --wait --timeout 2400`
   The key is idempotent, so reuse it. If main has moved, deploy the newest attested commit with a new key.
3. Confirm the screening re-plan (section 2.4) and watch the runs.
4. Then the PR queue (section 3).

## 1. Mission, doctrine and owner rules

- **Job:** website job capture-447ae4f4. Task: "Pick up the wooden tray from the dining table and place it at the other end of the same table." Carry it through screening/evals and beyond, and keep cleaning up the harness.
  - Results page: https://tryblueprint.io/app/tasks/capture-447ae4f4-c253-44ae-a897-c999245e0ef7
- **Doctrine (both repos):** video in, site simulation, Blueprint runs the candidates, the site gets a bounded estimate or "held".
  - The only hard stops are **privacy, rights and money**. Everything else is flagged and keeps going.
  - Don't make things up.
  - AI/learned policies only.
  - **Fixes must be general, never scene-specific** (`tests/test_no_scene_specific_code.py`).
  - Don't grow disk.
- **Owner approvals in force** (owner said so explicitly):
  - Every deploy is pre-approved, including WebApp merges, which are Render deploys.
  - GPU runs up to $4 each.
  - Screening budget and retry extension: done ($30 per site, 6 harness retries).
  - "Lane merges": PR #3331, the Cloud Run lane. Merge and deploy it.
  - #3196: approved, already merged.
  - Enable the Cloud Run lane: approved, including the caps $1/attempt, $5/day, $25 total. It needs host steps that a cloud session can't do; see 3.2.
  - Vast top-up: the owner added $10 (balance about $18 at 14:00Z).
  - Render `ttl_seconds` 604800: done by the owner.
  - **The owner does NOT want the Atlas $130 row extended** ("i dont want this run taking another day"). That row is capture-447's website job spending authorization in Render (`website_atlas_authorization.v1`, `maximum_cost_usd` 130). **It expires 2026-11-06T10:16Z (4:16 AM CST).** Every paid screening run must start before then. Plan for the screening to finish today.
  - The owner already started a separate session for the splat viewer bug. Don't touch it.
- **Hard rules:**
  - Host access only through the Pipeline repo's `scripts/operator_door.py`. Never SSH.
  - Never widen money grants or caps without the owner. The caps above are approved; nothing else is.
  - Never work around a classifier or permission denial. Report it instead. Example: a sweep of `/tmp/blueprint-canonical-artifacts-*` was denied; leave it.
  - Don't copy customer frames, meshes, splats, images or videos off the host. Read JSON and text only.
  - Never edit inside a shared block of WebApp `PLATFORM_CONTEXT.md`.
  - No bare `git stash`, `git reset --hard` or force-push. Use merge commits when updating PR branches.
  - No model identifiers in commits or PRs. Use your own session's commit and PR attribution lines.
  - Write to the owner in plain English and CST.

## 2. capture-447 screening: live state and how to read it

### 2.1 Where things are (host paths, read with `operator_door.py cat|ls`)

- **Ledger:** `/var/lib/blueprint/pipeline-control-plane/beta-policy-screening/screening_ledger.v1.json`.
  - `campaigns[<id>]` has `site_id`, `plan_digest`, `supersedes`/`superseded_by`, `candidates[<cid>]` (`status`, `blockers`, `attempt`, `summary`) and `site_budget`.
- **Plans:** `.../beta-policy-screening/plans/<campaign>.json`. Candidates carry `execution_spec.implementation_commit`, `max_cost_usd` and `status` (selected/waitlisted/unavailable).
- **Attempts:** `/mnt/blueprint-work/beta-policy-screening-work/attempts/<attempt_id>/allocator/attempts/attempt_001/vast_provider_run/`.
  - Useful files: `vast_provider_adapter_result.json` (`status`, `reason`, `blockers`, `create_failure_diagnosis`), `vast_runtime_phase_log.jsonl`, `vast_onstart_container.log`, `provider_runtime_diagnostics.v1.json`, `provider_credit_admission.json` (Vast credit) and `vast_offer_selection_manifest.json`.
- **GPU spend guard:** `/var/lib/blueprint/pipeline-control-plane/gpu_spend_guard/latest.json`. Vast `row_count` 0 means nothing is rented.
- **Site id:** `site-capture-447ae4f4-c253-44ae-a897-c999245e0ef7`.
- **Current campaign (as of 14:50Z):** `screening-caeee37a9a03ad4e6467c349d93d41a7`, frozen at commit f950ddfb. Nothing is active and $0 is reserved.

### 2.2 Results so far

| Candidate | Status | Cost |
|---|---|---|
| groot_n17_droid | 10 scored, 0/10 succeeded; published | $1.56 |
| cosmos3_edge_policy_droid | 5 harness failures: VAE, SecurityValidationError, HTTPError on a flaky machine. All fixed in #3389, #3391, #3392 | |
| pi05_droid | 5 harness failures; last run all episodes void before the first observation (`simulator_failed_before_policy`) | |
| molmoact2_droid | 1 harness failure (ffmpeg missing, fixed in #3393). Today's $3.78 attempt never got a machine: first "no qualifying Vast offer", then the offer was taken between search and create (empty HTTP 400). Spent $0 | $0 |
| pi0_droid_jointpos_polaris, pi0_droid_jointpos_100k_polaris, pi0_fast_droid_jointpos_polaris | never run; deferred for site budget | |

- pi05's cause is unknown. #3388 makes the next run record typed episode failure codes (`failure_code`, e.g. `module_not_found:x`). Read them from `provider_runtime_diagnostics.v1.json` and the ledger `summary.episode_outcomes`, then fix generally.
- **Site money:** $11.22 spent of the old $15; the limit is now $30.
- **Vast credit:** about $18. Credit admission needs the candidate cap plus a $1 reserve free, so about $5 per run.

### 2.3 What #3395 (merged at 14:48Z as 2f688aa9) changes, and why the re-plan should now run everything

- Site limit $15 → $30 in Pipeline `SITE_LIMIT`, `SITE_BUDGET_USD` and the launch quote check. WebApp #1216 has been live since 13:45Z, so it accepts `limit_usd` up to 30.
  - A plan's default budget stays $15 per campaign (`CAMPAIGN_BUDGET_USD`); a site reaches $30 across re-plans.
- `HARNESS_RETRY_LIMIT` 4 → 6, so pi05 and cosmos3 (5 failures each) get 2 more tries each.
- **Re-admission of dropped candidates:** after a deploy, `_replan_scope` adds candidates that an earlier plan in the chain listed, whose last row is a harness failure, and whose retries the bound allows (`_dropped_harness_retries`). Simulated on the live ledger, this re-admits exactly `pi05_droid` and `cosmos3_edge_policy_droid`.
- **Offer taken before create:** the adapter emits `vast_selected_offer_gone_before_create` (from `vast_create_failure_diagnosis.offer_gone_codes`). The coordinator treats it like an offer shortage and waits and searches again under a fresh identity, instead of closing the row.
- Native task arena policy bundle: added the missing `team_policy_episode_outcome.py` to `POLICY_EXTRA_RUNTIME_MODULE_NAMES`. Screening's own bundle (`TEAM_RUNTIME_MODULE_NAMES`) already had it.

### 2.4 After the deploy: confirm and watch

- **Expect** a new campaign that supersedes `screening-caeee37a9a…`, listing molmoact2_droid, pi05_droid, cosmos3_edge_policy_droid and the three pi0 rows as pending or selected at the new commit. Check `last_processed_campaign` in the ledger and the new plan file.
- **Runs go one at a time** (`_unproven_retry_in_flight`): a campaign with any harness-failed candidate runs serially until one completes with a scored result. After that, others may run in parallel, within the plan budget ($15 per campaign) and the site limit ($30).
- Each run takes roughly 40–90 min, Isaac start-up plus 10 episodes, and has a TTL of 4 h.
- **If nothing re-plans:** a re-plan happens only when the campaign has no active rows and no reservation, and something is stale (a deploy moved the commit).
  - Check the screening worker journal: `operator_door.py journal <unit> -n 200`. Find the unit with `operator_door.py units` and look for beta-policy-screening.
  - Re-run the simulation with the scratch approach: load the ledger and plan JSON, call `beta_policy_screening_coordinator._replan_scope(ledger, plan, "<new commit>")` with `PYTHONPATH=src`.
- **Results:** the coordinator publishes to the WebApp and needs the WebApp's acknowledgement before launching more (publication-ack gate). If the WebApp rejects a publication (409 or schema), launches stop. Check the publication status.
- **Report to the owner** each candidate's score as it lands ("X of 10 practice runs succeeded"), and the final estimate or "held".

## 3. PR queue (in priority order)

### 3.1 Pipeline main 2f688aa9: deploy (see section 0)

- Contains #3395 and #3196. #3196: website jobs now launch scene configuration from their own progression tick.
- No scene was parked at `awaiting_execution/scene_configuration` at 14:30Z, so #3196 launches nothing right now.
- Deploys wait for idle (about 11 min) and for the CI "Container production contract" attestation on the main push.

### 3.2 Pipeline #3331: Cloud Run lane for team-scene compiles. Merge, deploy, then hand the owner the host steps

- Head `a113a1b0f` (main merged in by a merge commit; conflicts resolved for main's robot-access feature). CI was running at 14:40Z.
- Local run: 2894 passed, 47 failed. All 47 also fail on clean main because of the environment:
  - `test_mount_work_volume_script` (no rsync);
  - `test_production_runtime_catalog_verdict`;
  - the uv version pin test;
  - the root-owned test;
  - the daily research SDK test.
- When CI is green, merge it with a squash. Main has moved, so update the branch first if GitHub requires it, then deploy.
- **Update the PR body** with:
  - The envelope is now the compile's arguments plus `robot_access_offer`.
  - On the remote route only, the host reads the furniture-measurement offer before handing off when the furniture is unmeasured and the scene review names chairs or stools. That hashes the source video and runs ffprobe, which is extra host disk reading.
  - The shipped confirmations now include robot-access approvals.
  - There is a new test file, `tests/test_team_scene_packet_remote_robot_access.py`.
  - The test counts above.
- Merging changes nothing at run time: the lane stays on `host` until the config names the stage.
- **A cloud session cannot enable it.** There are no Google Cloud credentials here, and the door has no operation for writing `/etc/blueprint` or spending authority. Give the owner the steps from the PR body's "Owner approvals needed":
  1. Create the Cloud Run job `blueprint-remote-cpu-team-scene-packet` (4 CPU / 16Gi / 3600 s) through `deploy/scripts/deploy.sh --remote-cpu-only`, and review the scoped plan.
  2. Set `BLUEPRINT_TEAM_SCENE_PACKET_EXECUTION=cloud_run_shadow` in `/etc/blueprint/pipeline-control-plane.env`.
  3. Add a `team_scene_packet` entry to `/etc/blueprint/remote-cpu-workers.json`, reseal it, and check `load_remote_cpu_config(path)[1] == []`.
  4. Re-issue the standing authority with `stages: [episode_compilation, team_scene_packet]` at $1/attempt, $5/day, $25 total.
  5. Run one preflight probe (at most $0.8544).
  6. After a shadow parity `passed`, switch to `cloud_run`.
- Don't build a door operation that writes spending authority; that widens the door's money powers.

### 3.3 Video retention: Pipeline #3396 + WebApp #1218. Fix the review findings, then merge (WebApp first)

- **Branches:**
  - Pipeline `claude/retain-screening-videos`, head 44562d6e3 at 14:50Z.
  - WebApp `claude/task-page-screening-videos`, head 09d8e0d7d.
- A subagent in the old session was applying fixes; it is gone. Check whether the heads moved and what the new commits did. If the heads haven't moved, implement the fixes yourself.
- **What the PRs do:**
  - After GPU teardown, keep up to 80 MP4s per attempt (1 GiB, 64 MiB each) from the promoted output archive, verified against the sealed index.
  - Store them in `<pubsub-handoffs>/<bucket>/scenes/site-<req>/captures/<cap>/pipeline/beta_screening_review_videos/<attempt_id>/` with a sealed `manifest.json`.
  - `summary.review_videos`; an optional publication `review_videos`.
  - A signed Pipeline read route `/api/live-pipeline/beta-screening-review-videos/<sha256>`.
  - WebApp: an owner-only 15-minute view ticket and a byte-range route that re-checks consent and owner on every request.
- **Independent review: must-fix**
  1. `beta_screening_review_videos.py` `mp4_duration_seconds` (~line 200). Return None unless the value is finite and 0 ≤ d ≤ 86400, and check that range in `valid_reference` (~445) and `publication_entry`. Otherwise the WebApp (`max(86_400)`) rejects the whole publication and the ack gate stops the campaign.
  2. WebApp cache `$TMPDIR/blueprint-screening-practice-run-videos` (`server/utils/screeningPracticeRunVideos.ts` ~86–104). Purge the job's cached digests when withdrawal is recorded (`server/utils/websiteCaptureWithdrawal.ts`) and when the ticket or video route refuses for lapsed consent or owner (`server/routes/workspace.ts` ~1648). Add an idle-expiry sweep and a size cap. This is the privacy hard stop.
  3. Pipeline `production_autonomy_acceptance.py` (~50–54, 165 `set(value) != fields`), reused by `production_autonomy_acquisition.py:227`, rejects v2 publications that carry `review_videos`. Allow it as optional and shape-checked, and add a test with the shared fixture.
- **Should-fix**
  - a. WebApp memory on Render's 512 MB instance: stream fetches to file while hashing, at most 2 concurrent Pipeline fetches (`screeningPracticeRunVideos.ts` ~123).
  - c. Retention races withdrawal: take the lock withdrawal deletion takes (it takes the Atlas `preview.lock`), re-check withdrawal and capture-root identity before each video and before the manifest, and never create missing parents (`provider_output_range_ingestion.py:210` `parents=True`).
  - d. `beta_screening_review_videos.py:487` `sorted(base.iterdir())[:64]`: look videos up by manifest digest, without truncating.
  - e. The streamed path never re-checks the disk floor: `provider_output_range_ingestion.py:512` passes `lambda needed: None`. Pass the real capacity check.
  - f. Map a retired or missing video to a "no longer kept" state on the page.
  - Skip the env-flag idea. Enforce the deploy order instead: **WebApp #1218 live first** (check `https://tryblueprint.io/version.json` `git_sha`), then Pipeline #3396.
- **Then backfill groot's videos.** The owner asked to watch them.
  - Attempt `screen-cea30f50e4d7a2b61e0eca0a92cfc58f`; its archive was promoted to the private object store before settlement (promotion receipt status `promoted`, key `sha256:d4035cde…`; read the full key from the attempt's `provider_output_promotion.v1.json`).
  - 60 MP4s, 77 MB.
  - Build a general backfill for settled attempts whose archive is still promoted: an existing unit or a door operation, through a PR. Run it, then tell the owner the videos are on the task page.
  - Don't download the videos into the session.

### 3.4 Pipeline #3397: withdrawal deletion now covers screening work. Review, merge, deploy

- Branch `claude/withdrawal-deletes-screening-work`, head ee00915b0. 134 tests passed.
- **What it does:**
  - New helper `website_capture_withdrawal_screening.py`.
  - Deletes the attempt directories, plus every version and delete marker of the promoted output and witness archives, through `B2ObjectVersions`.
  - Binds attempts to the site through the ledger and plans.
  - Defers attempts that are active, have an open reservation, or are pending settlement.
  - Holds the coordinator's `.ledger.lock` and never writes the ledger.
  - Fails closed on unreadable or unsealed records, symlinks, or shared archives.
- An independent review was running in the old session and is gone. Redo it, focusing on:
  - over-deletion, i.e. another site's data (the shared-archive check skips other sites' unreadable records: fail-open?);
  - how long `.ledger.lock` is held during network deletes (could it starve the coordinator?);
  - whether the "money final" classification covers every reservation state;
  - staged copies left behind.
- **Independent review finished (14:58Z). Not ready to merge.** File:line refer to `src/blueprint_pipeline/website_capture_withdrawal_screening.py` at ee00915b0 unless noted. A probe test reproducing must-fix 1 and 3 and should-fix 5 was in the old session's scratchpad and is lost; rewrite it as real tests.
- **Must-fix**
  1. **An unmounted work volume counts as "already deleted"** (`:212` `if directory.exists()`, `:180` silently skips a missing attempts root).
     - With `/mnt/blueprint-work` unmounted, apply runs clean, the archives stay in B2, and the receipt says `covered_copies_deleted: True`.
     - Fix: count an absent directory as deleted only when its parent `attempts` directory exists and is a real directory, not a symlink. Otherwise add `screening_attempt_root_unavailable` to `unresolved`.
  2. **Deleting a held attempt's directory destroys the only record of its staged copies** (`:210`, `:264`, `:295`, and `website_capture_withdrawal_deletion.py:679`).
     - The staged keys live only in `object_store_staging/wam_provider_object_store_staging_manifest.json` inside the attempt directory.
     - Fix: accept a valid `staged_object_absence_proof.v1.json` (`load_staged_object_absence_proof`). Otherwise defer with `screening_staged_objects_unproven` ("run provider-output-resume first").
     - Set `pipeline_controlled_deletion_complete` to false while any staged copy is unproven, and record the staged keys in `remaining`.
  3. **Attempt IDs are not pattern-checked; directories are not confined** (`:196` uses `coordinator._attempt_path`).
     - Reproduced: a ledger row with `attempt_id: "../plans"` deleted `<queue>/plans`. `".."` would delete the whole queue root.
     - Fix: require `re.fullmatch(r"screen-[0-9a-f]{32}", attempt_id)` and `directory.parent in {queue_root/"attempts", work_root/"attempts"}`, with neither root a symlink. Otherwise add the attempt to `unresolved`.
- **Should-fix**
  4. **`.ledger.lock` is held for the whole apply** (`website_capture_withdrawal_deletion.py:617-619`), which starves the coordinator for every site during network deletes. There's no deadlock and no overspend.
     - Fix: re-plan and check the digest without the lock. Then lock, re-run `plan_screening` and require it to equal `plan["screening"]`. Do only the screening deletes under the lock, then release.
  5. **The shared-archive check fails open** (`:136-149`): one unsealed `...superseded-x` file drops a whole foreign directory's names.
     - Fix: collect names per record, send an unreadable foreign record to `unresolved`, and protect archives that this site's deferred attempts name.
  6. **The plan seal is not verified** (`:85-100`, `_plan_record` trusts `plan_digest`).
     - Fix: use `canonical_digest(plan, digest_field="plan_digest")` or `validate_screening_plan`.
     - Also flag `screening_campaign_names_another_site` when a campaign binds by capture or intent but its ledger `site_id` is another site.
  7. **Object keys are confined only to the bucket.** Require the kind to be `policy-canary-provider-output` or `policy-canary-paired-witness`, and the key prefix `{LARGE_ARTIFACT_KEY_PREFIX}/{kind}/sha256/<hex>/`.
  8. **Receipt honesty:**
     - Report a named queue root that doesn't exist (`:174`).
     - Report allocated attempts whose directory is absent.
     - Report receipts whose `paired_witness.state` is `pending`.
     - List `selections/`, holds and the `attempts/<id>.attempt_dir.json` pointers among retained records.
  9. **Unlocked writers:** take each staging directory's `.provider_output_promotion.lock` without blocking before deleting it, which guards against `provider-output-resume` re-promoting. Expect `backfill_runtime_diagnostics` races to fail closed.
- Checked and fine: the "money final" classification, deleting a final attempt's directory (nothing later reads it), crash safety, and symlinks inside attempts.
- Then: wait for green CI, merge, deploy. Prefer a quiet moment: no screening run mid-launch.

### 3.5 Backlog (lower priority, from the old session's task list)

- Beta proof: after capture-447 publishes, run a second scene/task hands-off and log every manual intervention.
- Local-first iteration: pre-deploy replay of a job's next progression step against copied production records, and an ARTISAN stage-3 dry run on recorded responses (no OpenAI spend).
- Chairs option: WebApp #1212 and Pipeline #3357, stacked on #3355. Held until capture-447 screening ends.
- Permanent collider OOM fix: binary colliders at source.
- Early robot-fit #3336 needs review; window #3337 needs a CI fix.
- A privacy backfill follow-up from #3397: archives no local record names.

## 4. Engineering notes and pitfalls

- **Tests:**
  - Use the pinned tools: `/opt/blueprint-cloud/pipeline-venv/bin/python` and `/opt/blueprint-cloud/pipeline-venv/bin/ruff` (0.15.21; the global ruff is stricter than CI).
  - Impacted selection: `PYTHONPATH=src python -m blueprint_pipeline.impacted_test_selection --plan-only --no-worktree --changed-file <f> …`. Without `--changed-file` it reads the main checkout. Run `wide_impacted_tests` from its JSON.
  - **Pytest deselects slow and gpu tests by default** (`pyproject` addopts `-m "not slow and not gpu"`). Use `-m ''` to match CI. Missing this hid a bundle import failure until CI caught it.
  - Use `-n 8` (xdist), and `--basetemp=<scratch dir>` when two pytest runs share the machine.
  - WebApp: `npm run check` needs `NODE_OPTIONS=--max-old-space-size=6144`. Symlink `node_modules` from the main checkout into worktrees and unlink it before committing.
- **These tests fail on clean main in a cloud container** (environment, not code): `test_task_evaluation_launch_activation_claim_recovery` (root), `test_completed_scene_consumer_rehearsal`, `test_scene_preparation_installation`, `test_scene_retirement_connected_acceptance`, `test_mount_work_volume_script` (rsync), `test_production_runtime_catalog_verdict`, the uv pin test, and Blender/torch/`launch_profile_input_not_host_resident` cases. Always compare against a clean main worktree before blaming a change.
- **Governance:** `docs/source_governance_policy.json` caps module lines. `vast_provider_adapter.py` sits exactly at its limit (10420), so edit in place or move code to helpers. Coordinator, plan and screening modules have no cap.
- **Bundles:** a provider bundle ships an explicit module list.
  - Screening canary session: `TEAM_RUNTIME_MODULE_NAMES` in `native_task_arena_policy_canary_bundle.py`.
  - Native task arena policy: `POLICY_EXTRA_RUNTIME_MODULE_NAMES` in `native_task_arena_execution_contract.py`.
  - A new import in shipped code must go into both. `test_real_policy_bundles_pass_preflight_and_import_cleanly` (slow) catches misses.
- **Money mechanics:**
  - Coordinator `SITE_LIMIT` must be ≤ WebApp `SCREENING_SITE_LIMIT_USD` (`server/utils/betaPolicyScreeningContract.ts`). Change the WebApp first.
  - Settlement brings committed spend down to the run's ledger estimate.
  - Vast credit admission needs the cap plus a $1 reserve.
  - The job ceiling (WebApp `websiteJobCeiling.ts`) still bounds the job.
- **Vast:** an empty HTTP 400 on create is never reselected within a run; it is ambiguous by design. The coordinator-level retry relies on its no-allocation proof and the next run's pre-launch inventory guard.
- **Disk in the cloud container is tight.** Remove finished worktrees with `git worktree remove` (it refuses dirty ones). Don't sweep shared `/tmp`.
- **Deploying while a candidate runs:** the deploy waits for idle. A re-plan happens only once the campaign has no active rows. Every deploy moves the commit, so a harness-failed candidate can retry once per deploy, bounded by `HARNESS_RETRY_LIMIT` 6.

## 5. Key PRs from this session, for context

- **Merged and live before 14:00Z:**
  - #3388 typed failure codes
  - #3389 cosmos offline VAE
  - #3390 stale spend-lock retry
  - #3391 ranged downloads
  - #3392 zero-byte retries
  - #3393 ffmpeg bootstrap
  - #3394 site-fit quote
  - #3369 drawer prior made general
  - #3315 with WebApp #1204 (settled holds count in the job ceiling)
- **WebApp #1216:** site limit $30, live since 13:45Z.
- **Merged at 14:30–14:48Z, deploying:** #3196 and #3395.
- **Open:** #3331, #3396, WebApp #1218, #3397.

## 6. Owner to-dos you may need to remind them of

- Cloud Run host steps (3.2).
- Watch Vast credit: about $18; each run needs about $5 free.
- Nothing else is pending from the owner. The Atlas row is intentionally expiring.
