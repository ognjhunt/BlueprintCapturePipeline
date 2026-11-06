# Bandit triage re-review packet

For: @ognjhunt · Prepared 2026-11-06 · Repo `BlueprintCapturePipeline` @ `3d1cd4dd`
Scope: all 96 entries in `docs/bandit_triage.json` (36 expired 2026-11-05, 60 expiring 2026-11-12). Nothing in the repo was edited.

`bp/` = `src/blueprint_pipeline/`. `[n]` = index in the `entries` array of `docs/bandit_triage.json`. "(was N)" = the `line_number` stored in the entry, where the code has since moved.

---

## 1. How the gate works (`scripts/verify_bandit_policy.py`)

- **Matching.** Each finding gets a SHA-256 fingerprint built from: the relative filename, the test id, the narrowest enclosing class or function (AST scope), the column offset, the issue text, the canonical code snippet, and the normalized source **±6 lines** around the finding (`fingerprint_policy = filename_test_id_ast_scope_column_issue_text_canonical_code_and_source_anchor`). If only the line number moves, the fingerprint still matches. Any edit within 6 lines of the finding, a renamed function, a moved file or a rule change produces a new fingerprint, so the site needs a new review.
- **What blocks:**
  - any HIGH finding, which can never be baselined;
  - any MEDIUM finding with no matching entry (`untriaged_medium`);
  - an entry whose `expires_on` is before today (`triage_expired`). An entry is good *through* its `expires_on` date;
  - an entry whose fingerprint matches no live MEDIUM finding (`orphaned_triage_entry`);
  - an entry whose `test_id` or `filename` disagrees with the finding;
  - a wrong `schema_version` (`blueprint.bandit_triage.v2`), `scanner_version` (`1.8.6`) or `fingerprint_policy`.

  "Expired or orphaned" is these two blockers. The `policy.expired_or_orphaned_entries_block: true` flag in the JSON only describes this. The script hard-codes the behavior.
- **Rules for an entry to be valid:**
  - `severity` is MEDIUM;
  - `disposition` is one of `accepted_risk`, `false_positive` or `mitigated_by_control`;
  - `owner` is at least 3 characters;
  - `reason` is at least 20 characters;
  - `reviewed_on` is not later than today;
  - `expires_on` is on or after `reviewed_on`;
  - **`expires_on` is no more than 90 days after `reviewed_on`.**
- **Longest extension allowed:** signed 2026-11-06, the latest `expires_on` is **2027-02-04**.
- **What the gate does not check:**
  - `line_number`;
  - whether `owner` is a real person (any string of 3 or more characters passes);
  - whether the `reason` is true;
  - LOW findings (1,785 today);
  - inline `# nosec` suppressions.

## 2. Gate result today (run like CI)

`bandit -r src/blueprint_pipeline scripts --format json --quiet` uses Bandit 1.8.6 from the pinned pipeline venv. The exit code was 1, which CI allows because it only means findings were present. I called the venv binary directly instead of `uv run` so that uv could not re-sync or re-lock the repo. The flags are the same as CI's. I then ran `verify_bandit_policy.py` on that report.

| | |
|---|---|
| Status | **blocked** |
| HIGH | 0 |
| MEDIUM | 96 (every one matches an entry) |
| Blockers | **36 × `triage_expired`** (all the 2026-11-05 entries), nothing else |
| Orphaned entries | **0** |
| New or unbaselined MEDIUM/HIGH | **0** |
| Metadata mismatches | 0 |
| Next cliff | The 60 entries dated 2026-11-12 become blockers on **2026-11-13** |

The raw outputs (`bandit.json`, `gate.json`, and `joined.json`, which maps each entry to its live finding) were kept in the reviewing session; rerun the two commands above to reproduce them.

## 3. My assessment

| Verdict | Meaning | Expired (36) | Nov-12 (60) | Total |
|---|---|---|---|---|
| **KEEP** | keep accepted, low risk | 30 | 47 | **77** |
| **TIGHTEN** | keep accepted, but make a specific hardening change | 6 | 12 | **18** |
| **FIX** | fix the code now | 0 | 1 | **1** |
| **DELETE** | orphaned, delete the entry | 0 | 0 | **0** |

No customer-supplied data reaches `exec`, a subprocess, pickle or `torch.load`, or tar or zip extraction at any of these 96 sites. The places where data from outside Blueprint does reach `urlopen` or a file write come from third-party providers (DeepInfra, Palatial, World Labs, RunPod pods) or from a bearer-authenticated internal runner request. They are the B310 TIGHTEN and FIX rows in §5.

---

## 4. Recommended owner action

1. **Unblock CI today.** All 36 expired entries can be extended. 30 are KEEP. The other 6 are TIGHTEN, so give them a short window:
   - [75] `capacity_controller.py:871`
   - [93] and [82] `storage_gc.py:1715` and `:1720`
   - [59] the sandbox bridge, `:121`
   - [70] the sandbox manager, `:374`
   - [71] `remote_cpu_worker.py:76`
2. **Before 2026-11-13,** re-review the 60. Extend 47 KEEP. Extend 12 TIGHTEN for a short window. Do not extend **[51] `bp/wam_compute_providers.py:351`**. Fix it (§5) and delete its entry in the same PR.
3. **Suggested windows** (the policy maximum is 90 days):
   - KEEP: `reviewed_on 2026-11-06`, `expires_on 2027-02-04`.
   - TIGHTEN: `expires_on` around `2026-12-06`, linked to a ticket for each fix.
   - FIX: at most 7 days, only if CI needs it while the fix lands.
   - Optionally stagger the KEEP dates so that 96 entries do not expire on the same day again.
4. **Correct the record when you sign.** Some entries do not say who reviewed them or say something that is false.
   - **38 entries are not owned by a person.** [58]–[93] (36 entries) have the owner "Codex /root/independent_review", and [94] and [95] have "codex/real-video-e2e-20261028".
   - **7 reasons say the review is not a human approval.** [58], [59], [68], [70] and [83] say "No accepted risk, policy waiver or human approval". [94] and [95] say it is "not a claim of … human risk acceptance". Once you sign, those clauses are false.
   - **33 B310 entries share a template reason.** It says "...after confirmed unsafe endpoints were migrated to bounded validators..." ([0–5, 16–18, 20, 21, 29–31, 33–38, 40, 41, 43–48, 50, 51, 55–57]). Most of those sites have **no** validator: they pass an environment or CLI URL straight to `urlopen` with no scheme check. Replace it with the wording for each group in §5.
5. **When a fix lands,** the fingerprint changes, so the old entry becomes orphaned and blocks CI. Delete it in the same PR. If Bandit still flags the new code (it does for any `urlopen` on a variable), add a new entry with the new fingerprint, computed with `finding_fingerprint()` from the gate script.
6. **Outside this packet.** 229 findings are suppressed inline with `# nosec` (mostly B310, B603, B404 and B615). The expiring gate never sees them. Some worth a separate look:
   - `bp/task_object_blender_runtime.py:85`, `nosec B102`: it runs candidate code with `exec`.
   - `bp/measured_articulation_derivation.py:309`, `nosec B604`: it passes `shell=`.
   - `scripts/operator_door.py:380`, `nosec B202`: it calls `tarfile.extractall`.

   I did not review them.

**Optional hardening across all URL sites (no deadline).** Add one shared `require_https_url()` helper, http allowed only for loopback, and call it for every URL that comes from env or CLI (the 17 rows in §5, group B310-b). This would let those reasons truthfully say a validator exists.

---

## 5. Expired 2026-11-05 (36)

All 36 were reviewed 2026-10-06. The owner is "Codex /root/independent_review", which is an agent label, not a person.

### B102 exec: loaders for protected installed source (4)
What the code does: `exec(compile(bytes))` on source bytes read through checks that refuse symlinks, require root ownership and refuse group- or world-writable parents. In effect it is a custom importer.
Input: host files under root-owned install roots (`/usr/lib/blueprint/...`, `/opt/blueprint/operator-door`). No text from a caller or customer.

| Site | Verdict · why |
|---|---|
| [86] `scripts/scene_retirement_continuous_bootstrap.py:248` | KEEP · runs a fixed 7-file cohort; requires `-I -S`, euid 0, mode 0644, a single link |
| [76] `bp/task_evaluation_scene_retirement_supervisor.py:949` | KEEP · runs only bytes that root has already verified; never uses stale .pyc |
| [87] `bp/control_plane_lane_experiment_retirement.py:73` | KEEP · runs the fixed `/opt/blueprint/operator-door/operator_door/config.py` via `files.read(protected=True)` |
| [73] `bp/control_plane_lane_owner_consents.py:466` | KEEP · the same loader as [87], copied. Optionally share one helper so there is one entry instead of two |

### B103 chmod: control-plane state directories and locks (11)
What the code does: sets directories to 0755/0750/02770 and files to 0660 so root and the service account can coordinate.
Input: fixed host paths under the protected state directory. The capacity and GC report roots can be overridden by operator env or CLI.

| Site | Verdict · why |
|---|---|
| [90] `bp/control_plane_break_glass.py:200` | KEEP · the new notes root is 0755 so the door can list it. Notes are 0644, so privacy depends on the STATE_DIR parent being 0750 |
| [75] `bp/control_plane_capacity_controller.py:871` (was 755) | **TIGHTEN** · the directory is made 0755, but `latest.json` and `history.jsonl` get whatever the umask allows. Outside the UMask=0077 unit, or with `BLUEPRINT_CAPACITY_REPORT_ROOT` set, the full report is world-readable. Create both with `os.open(..., 0o600)` |
| [93] `bp/control_plane_storage_gc.py:1715` (was 1698) | **TIGHTEN** · `_write_report` publishes the *full* GC report as 0644 in a 0755 directory, not only `summary.json`. The entry's own reason says full paths are not always secret-free. Write the full report 0600 or 0640 |
| [82] `bp/control_plane_storage_gc.py:1720` (was 1703) | **TIGHTEN** · the fallback branch (`fchown`, then 0755) of the same function. Same fix |
| [81] `bp/control_plane_disk_footprints.py:106` | KEEP · the new history directory takes the ledger's group and 02770, set through a descriptor opened without following links |
| [78] `bp/control_plane_disk_footprints.py:122` | KEEP · history file 0660, changed on the held descriptor only by its owner, regular files only |
| [92] `bp/control_plane_disk_ledger.py:106` (was 103) | KEEP · ledger root 02770; a best-effort change made only by the owner |
| [66] `bp/control_plane_registered_checkpoint_cache.py:1262` | KEEP · stage directory set to 0750 root:blueprint after the identity is checked again |
| [83] `bp/control_plane_workspace_lock.py:128` | KEEP · lock directory 0770, owner checked, identities checked again |
| [68] `bp/control_plane_workspace_lock.py:151` | KEEP · lock file 0660, created with O_EXCL, shared owner verified |
| [61] `bp/task_evaluation_result_artifact_store.py:214` | KEEP · read-lease lock 0660 through an O_NOFOLLOW descriptor, changed only while this process owns it |

### B104 listening on 0.0.0.0: company policy sandbox (2)
What the code does: runs an HTTPS server on all interfaces with TLS 1.2 or later and a bearer token compared in constant time.
Input: operator CLI or settings. **The default is 0.0.0.0**, and the reasons say no firewall or allowlist is known to be deployed.

| Site | Verdict · why |
|---|---|
| [59] `scripts/serve_company_policy_sandbox_bridge.py:121` | **TIGHTEN** · make `--bind-host` required instead of defaulting to 0.0.0.0, and write down the ingress rule |
| [70] `bp/company_policy_sandbox_manager.py:374` | **TIGHTEN** · same: `bind_host` defaults to 0.0.0.0. Same fix |

### B108 "/tmp" that names a private tmpfs in a sandbox command (9)
What the code does: the literal is bwrap's `--tmpfs /tmp`, docker's `--tmpfs /tmp:…`, or HOME/XDG set *inside* that namespace.
Input: constants. No file is created in the host's /tmp.

| Site | Verdict · why |
|---|---|
| [62] `bp/asset_authoring_sandbox.py:174` (was 169) | KEEP · bwrap `--tmpfs /tmp` |
| [88] `bp/asset_authoring_sandbox.py:181` (was 176) | KEEP · "/tmp" is in a set of parent paths to skip; no file operation |
| [74] `bp/company_policy_sandbox_executor.py:398` | KEEP · the expected docker plan: `/tmp:noexec,nosuid,nodev,size=…` under runsc with a read-only root |
| [65] `bp/company_policy_sandbox_v2.py:238` | KEEP · the same plan, on the side that builds it |
| [60] `bp/native_g1_team_artifact_runtime.py:154` | KEEP · bwrap `--unshare-all --tmpfs /tmp`, uid 65534 |
| [63] `bp/native_g1_team_artifact_runtime.py:167` | KEEP · `--setenv HOME /tmp` inside the namespace, after `--clearenv` |
| [79] `bp/native_g1_team_artifact_runtime.py:168` | KEEP · `--setenv XDG_CACHE_HOME /tmp` inside the namespace |
| [84] `bp/native_g1_team_container_runtime.py:53` | KEEP · docker `--tmpfs /tmp` and a HOME env var inside the team's container |
| [69] `bp/native_g1_team_vm_system_installation.py:105` | KEEP · bwrap probe with `--tmpfs /tmp` |

### B108 HOME=/tmp in the environment of the host bwrap launcher (3)
What the code does: the env dict passed to `subprocess` for the `bwrap` launcher. bwrap clears it before the sandboxed program starts.
Input: constants.

| Site | Verdict · why |
|---|---|
| [91] `bp/native_g1_team_archive_gpu.py:198` | KEEP · the CUDA probe launcher; bwrap does not use HOME |
| [77] `bp/native_g1_team_artifact_runtime.py:291` | KEEP · the same, for the team-policy `Popen` |
| [80] `bp/native_g1_team_artifact_runtime.py:292` | KEEP · `XDG_CACHE_HOME`, same |

### B108 real scratch use in the host's /tmp (6)
| Site | Verdict · why |
|---|---|
| [85] `bp/native_g1_team_vm_host.py:229` | KEEP · `TemporaryDirectory(dir="/tmp")` makes a unique 0700 directory for the relay socket |
| [89] `bp/remote_cpu_worker.py:75` | KEEP · the fixed `/tmp/blueprint-release` on a Cloud Run job with one task; it refuses an existing release root; tar members are filtered and extracted with `filter="data"` |
| [71] `bp/remote_cpu_worker.py:76` | **TIGHTEN** · `/tmp/blueprint-pycache` is passed to the stage child as `PYTHONPYCACHEPREFIX`. Python still *reads* .pyc files from there when `PYTHONDONTWRITEBYTECODE=1`, so this is only safe because each Cloud Run instance has its own filesystem. Move it inside the 0700 release or scratch directory |
| [67] `bp/remote_cpu_worker.py:698` (was 697) | KEEP · creates the runtime-local /tmp parent |
| [72] `bp/remote_cpu_worker.py:699` (was 698) | KEEP · `mkdtemp` (0700) under that parent |
| [58] `bp/vast_provider_transfer_upload.py:13` | KEEP · "/tmp" is only the parent for `umask 077; mktemp -d` in the Vast upload shell function |

### B608 SQL built from a string (1)
| Site | Verdict · why |
|---|---|
| [64] `bp/agent_execution/journal.py:205` | KEEP · the f-string inserts one of two constant WHERE clauses; LIMIT is bound as `?` after a 1–1000 check. Optionally, two literal queries would remove the finding |

---

## 6. Expiring 2026-11-12 (60)

[0]–[57] were reviewed 2026-08-14 by @ognjhunt, exactly at the 90-day limit. [94] and [95] were reviewed 2026-10-28 by "codex/real-video-e2e-20261028". All are valid today and become blockers on 2026-11-13.

### B310 a: URL open to a constant host (5)
Input: a constant URL, plus an API key from env.

| Site | Verdict · why |
|---|---|
| [3] `bp/capture_enrichment_llm.py:338` (was 306) | KEEP · `https://api.anthropic.com/v1/messages` |
| [18] `bp/live_pipeline_setup.py:454` | KEEP · `https://api.digitalocean.com/v2/droplets`; the droplet name is URL-encoded |
| [34] `bp/provider_preview.py:52` (was 50) | KEEP · `https://api.worldlabs.ai` plus a path |
| [40] `bp/runpod_live_execution_proof.py:190` | KEEP · the constant `RUNPOD_REST_API_BASE` |
| [2] `scripts/run_sim_only_beta_local_gate.py:215` (was 211) | KEEP · `http://127.0.0.1:{free port}/health` |

### B310 b: URL configured by the operator (env, CLI or a job-state file our tooling wrote) (17)
What the code does: calls `urlopen` on a URL taken from configuration. Most of these sites have **no scheme check**, so a mistaken `file://` value would be read. Customers cannot reach them. The template reason ("bounded validators") is not accurate here.

| Site | Verdict · why |
|---|---|
| [0] `scripts/run_beta_intake_soak_test.py:57` | KEEP · POSTs a probe to `--target-url` |
| [1] `scripts/run_sim_only_beta_deployment_parity_proof.py:112` (was 102) | KEEP · URL from CLI or the route-proof JSON. A host is required but any scheme is accepted, and signed intake headers go to that URL |
| [4] `bp/city_launch_autonomy_harness.py:905` | KEEP · origin of `PIPELINE_SYNC_WEBAPP_URL`; the city slug is URL-encoded |
| [5] `bp/consent_takedown.py:622` (was 606) | KEEP · `PIPELINE_SYNC_WEBAPP_URL` with a signed token. This is the consent-revocation path; on failure it reports failed or queued and does not claim success |
| [20] `bp/mujoco_g1_wam_vla_policy_endpoint_eval.py:936` | KEEP · GETs `/health` of the endpoint URL from env |
| [21] `bp/mujoco_g1_wam_vla_policy_endpoint_eval.py:6823` | KEEP · POSTs an observation with the token-file bearer to the env endpoint. It would be the first to benefit from https-only |
| [29] `bp/palatial_physready.py:729` | KEEP · `PALATIAL_GENERATE_URL`, or the https default |
| [31] `bp/privacy_processing.py:152` (was 145) | KEEP · runner URL from env; http(s) and a host are checked; Cloud Run ID token |
| [37] `bp/provider_worker_policy_command_adapter.py:180` | KEEP · `*_POLICY_WORKER_URL` from env or CLI, with a bearer token |
| [38] `bp/robot_eval_worker.py:974` | KEEP · a signed PUT URL from env, used to upload the runtime manifest |
| [41] `bp/site_world_runtime_service_client.py:67` | KEEP · `SITE_WORLD_RUNTIME_SERVICE_URL` with a bearer token. `quote(id)` keeps "/", so use `safe=""` |
| [43] `bp/unitree_unifolm_runpod_server.py:665` | KEEP · server URL from CLI or the job's `state.json` |
| [44] `bp/unitree_unifolm_vla_server_bridge.py:213` | KEEP · from env or CLI, default `http://127.0.0.1:8777/act`. The json_numpy decoder uses `frombuffer`, not pickle |
| [48] `bp/video_to_world_client.py:76` | KEEP · `VIDEO_TO_WORLD_URL` from env with an ID token. No scheme check here, although `privacy_processing` has one that could be reused |
| [50] `bp/wam_compute_providers.py:338` (was 333) | KEEP · DeepInfra base URL from env, or the https default |
| [55] `bp/webapp_sync.py:662` (was 645) | KEEP · `PIPELINE_BUYER_ACCESS_CHECK_URL` from env, signed |
| [56] `bp/webapp_sync.py:898` (was 881) | KEEP · `PIPELINE_SYNC_WEBAPP_URL` from env, signed |

### B310 c: signed provider staging URLs (10)
Input: presigned https URLs that our staging wrote into 0600 secret files, or URLs that passed `_provider_url_public_blocker` (http(s) only, and IP literals must not be private). Content is checked against a size and SHA-256.

| Site | Verdict · why |
|---|---|
| [6] `bp/g1_microwave_finetune_provider_job.py:420` | KEEP · range probe of each checkpoint part. URLs come from `_read_secret_url` (0600, https) |
| [7] `bp/g1_microwave_finetune_provider_job.py:468` | KEEP · full download, checked against each part's SHA and the total size |
| [8] `bp/g1_microwave_finetune_provider_job.py:751` | KEEP · streams the parts to Vast over ssh, checking each part's SHA |
| [9] `bp/g1_microwave_finetune_provider_job.py:818` | KEEP · the output ZIP comes from the Vast host. Size is capped and `_safe_extract_output` rejects path traversal and links and caps member count and total size |
| [13] `bp/single_g1_kitchen_episode_runpod.py:587` | KEEP · 0600 https URL; requires status 206 and an unchanged `geturl()` |
| [14] `bp/single_g1_kitchen_episode_runpod.py:632` | KEEP · range reads of an exact byte count |
| [15] `bp/single_g1_kitchen_episode_runpod.py:647` | KEEP · full SHA pass against the pinned `BUNDLE_SHA256`. Two small gaps: the `geturl()` check rejects a redirected *result* after the request was sent, and the object could change between the hash pass and the range reads (zip CRCs catch most of that). Acceptable |
| [45] `bp/vast_provider_adapter.py:3621` (was 2111) | KEEP · HEAD on the bundle URL after the public-host check |
| [46] `bp/vast_provider_adapter.py:3649` (was 2139) | KEEP · fallback range GET on the same URL |
| [47] `bp/vast_provider_adapter.py:3762` (was 2252) | KEEP · 22-byte probe PUT to the output URL, after the same check |

### B310 d: URLs or content supplied by a third-party provider (8)
These are where data from outside Blueprint really meets `urlopen`.

| Site | Verdict · why |
|---|---|
| [51] `bp/wam_compute_providers.py:351` (was 346) | **FIX** · the download URL comes from DeepInfra's JSON response. Any URL that has a scheme is used as is, so `urlopen` would accept `file://`; there is no host check and no size cap. The file is written into the job directory; the later mp4 check flags a bad file, but the bytes stay there. Allow only https on DeepInfra hosts (plus `data:` if their API returns that), cap the size, then replace this entry. Its accepted-risk reason claims a validator that does not exist |
| [16] `bp/isaac_particlefield_render_job.py:551` | **TIGHTEN** · the GET URL is read from `job_dir/provider_output_get_url.txt` without the https or 0600 checks the finetune job uses. `.read()` loads the whole response into memory, and the ZIP from the RunPod pod is checked for traversal and links but has **no cap on member count or uncompressed size**. Reuse `_read_secret_url`, read with a byte cap, and copy the finetune job's extraction caps |
| [17] `bp/isaac_particlefield_render_job.py:621` | **TIGHTEN** · the final fetch in the same function. Same fix |
| [30] `bp/palatial_physready.py:829` | **TIGHTEN** · the reference comes from Palatial's API JSON. Any http(s) host is fetched, internal ones included, as long as the path ends in an export suffix. The branch for non-URLs copies **any absolute local path** with a .usd/.urdf/… suffix into the artifacts. Allow only https on Palatial and CDN hosts, and copy local files only from the capture and pipeline roots |
| [33] `bp/privacy_service_runtime.py:178` (was 173) | **TIGHTEN** · the bearer-authenticated Cloud Run runner downloads any http(s) URI in the request body and reads the whole response into memory. It writes the result to `working_dir / relative_path`, where `relative_path` comes from the depth or confidence manifest and is **not checked to stay inside `working_dir`**: an absolute path or `../` escapes. Today the pipeline sends gs:// URIs it builds itself, so customers cannot reach this. Still: allow only gs:// or an allowlist, stream with a cap, and reject absolute and `..` paths |
| [35] `bp/provider_preview.py:78` (was 76) | **TIGHTEN** · PUTs the capture video, with the audio removed, to an upload URL returned by World Labs, and plain http is allowed. This is customer video, so require https |
| [36] `bp/provider_preview.py:107` (was 105) | **TIGHTEN** · reads the video URI chosen from the descriptor (gs://, http(s) or **any local path**) and sends the bytes to World Labs. The pipeline writes those descriptor fields; the WebApp only reads them. Allow only gs:// in the capture's bucket and prefix, so a bad descriptor cannot send a host file to a third party |
| [57] `bp/worldlabs_asset_materialization.py:202` | **TIGHTEN** · asset URLs come from the World Labs manifest. Only http(s) is allowed but any host, and the size cap is optional (`max_bytes=None` is allowed). Require https on World Labs and CDN hosts, and always cap the size |

### B104 listening on 0.0.0.0: privacy and video-to-world runners on Cloud Run (2)
| Site | Verdict · why |
|---|---|
| [32] `bp/privacy_runner_service.py:297` | KEEP · listens on 0.0.0.0 only when `K_SERVICE` is set, otherwise 127.0.0.1; refuses to start without a token; bearer compared with `compare_digest` |
| [49] `bp/video_to_world_runner_service.py:270` | KEEP · same |

### B103 chmod: production preflight summary (2)
| Site | Verdict · why |
|---|---|
| [94] `bp/production_preflight_summary.py:250` (was 247) | KEEP · sets the private parent to 0711 only after checking uid == euid and no group or other write |
| [95] `bp/production_preflight_summary.py:254` (was 251) | KEEP · publication directory 0755, holding only the allowlisted summary; the raw report stays 0600 |

### B108 /tmp (5)
| Site | Verdict · why |
|---|---|
| [19] `bp/mujoco_g1_simulator_command.py:690` (was 683) | **TIGHTEN** · when fetching is turned on (`--allow-fetch-g1-assets`), it clones into the fixed `/tmp/blueprint_mujoco_menagerie`. An existing `g1.xml` there is used without checking its owner or that HEAD equals `menagerie_ref`. Default to a per-user cache directory and check owner and ref first |
| [39] `bp/robot_eval_worker.py:1572` (was 1570) | **TIGHTEN** · without `BLUEPRINT_WORKER_DIR`, the worker directory defaults to `/tmp/blueprint-worker`, and the worker manifest is written and read there. Use `mkdtemp` when the variable is unset, or require it |
| [23] `bp/object_index_stage.py:641` (was 635) | KEEP · "/tmp/input.json" is only substituted so the template can be split with shlex for a `which()` preflight. It is never run or opened |
| [24] `bp/object_index_stage.py:642` (was 636) | KEEP · same |
| [25] `bp/object_index_stage.py:643` (was 637) | KEEP · same |

### B614 torch.jit.load: Unitree G1 policy (3)
What the code does: loads `motion.pt` with `torch.jit.load`, with the path taken from `g1.yaml` (read with `yaml.safe_load`).
Input: a local unitree_rl_gym checkout the operator chose. It is not an upload path, but **nothing pins the file's hash** before it is loaded.

| Site | Verdict · why |
|---|---|
| [22] `bp/mujoco_g1_wam_vla_policy_endpoint_eval.py:7428` | **TIGHTEN** · check the file against a pinned SHA-256 before loading |
| [26] `bp/official_g1_policy_handoff.py:1945` (was 1946) | **TIGHTEN** · the SHA is only recorded *after* loading. The check that the path is inside the root is `str.startswith`, which `../` passes. Use `resolve()` with `is_relative_to`, and check the pinned SHA first |
| [42] `bp/unitree_g1_policy_execution.py:104` | **TIGHTEN** · add the same hash pin |

### B615 Hugging Face downloads (8)
| Site | Verdict · why |
|---|---|
| [27] `bp/openvla_policy_command_adapter.py:617` | KEEP · local checkpoint; `local_files_only=True`, `trust_remote_code=False` |
| [28] `bp/openvla_policy_command_adapter.py:631` | KEEP · same, for the model |
| [10] `bp/oscar_runtime_asset_contract.py:636` | KEEP · `snapshot_download` pinned to the 40-hex `REASON1_REVISION`; the resolved path is compared |
| [11] `bp/oscar_runtime_asset_contract.py:646` | KEEP · `hf_hub_download` pinned to `WAN_VAE_REVISION`; the path is compared |
| [12] `bp/oscar_runtime_asset_contract.py:794` | KEEP · the local snapshot that was already verified; `local_files_only` |
| [52] `bp/wam_real_provider_validation_probe.py:536` | KEEP · refuses unless the revision equals the pinned SHA; `trust_remote_code=False` |
| [53] `bp/wam_real_provider_validation_probe.py:537` | KEEP · same, for the processor |
| [54] `bp/wam_real_provider_validation_probe.py:913` | KEEP · `snapshot_download` pinned to the DA3 revision; any other revision is refused |

---

## 7. What signing off changes

These are the fields to edit in `docs/bandit_triage.json` for each entry you keep. I have not made these changes.

**Change on every kept entry:**
- `reviewed_on` becomes the date you actually sign, e.g. `"2026-11-06"`. It must not be later than the CI runner's date in UTC.
- `expires_on` becomes at most `reviewed_on` + 90 days, which is `"2027-02-04"` if signed 2026-11-06. Suggested values: `2027-02-04` for KEEP, about `2026-12-06` for TIGHTEN, and none for [51], which you would fix and delete.

**Also change on these entries:**
- `owner` becomes `"@ognjhunt"` on [58]–[93] (36 entries, currently "Codex /root/independent_review") and on [94] and [95] (currently "codex/real-video-e2e-20261028"). [0]–[57] already list @ognjhunt.
- `reason` must change on [58], [59], [68], [70], [83], [94] and [95]. Remove the "not a human approval / risk acceptance / waiver" clause and state what you are accepting.
- `reason` should change on the 33 B310 entries that share the template ([0–5, 16–18, 20, 21, 29–31, 33–38, 40, 41, 43–48, 50, 51, 55–57]). Use the wording for each group in §6, e.g. "operator-configured env URL; no scheme validation; not customer-reachable".
- `line_number` can optionally be updated to the live line in the "(was N)" rows. The gate ignores it.
- `disposition` can optionally be reclassified. [2], [3], [18], [34], [40] (constant URLs) and [10], [11] (pinned revisions) are really `false_positive`, not `accepted_risk` or `mitigated_by_control`.

**Do not change:**
- `fingerprint`, `filename`, `test_id`, `severity`;
- the top-level `schema_version`, `scanner_version`, `fingerprint_policy`;
- `generated_from_audit_date` is not checked; update it if you like.

Keep the JSON edit and code changes in separate PRs. Any code change within 6 lines of a kept site changes its fingerprint, and that entry would then be reported as orphaned.

**Example for one entry, [59]:**
```diff
-      "expires_on": "2026-11-05",
+      "expires_on": "2026-12-06",
-      "owner": "Codex /root/independent_review",
+      "owner": "@ognjhunt",
-      "reason": "Intentional listener secured by TLS>=1.2 ... No accepted risk, policy waiver or human approval. Source mitigations do not prove CI or live runtime deployment.",
+      "reason": "<owner's basis: TLS>=1.2 + constant-time bearer; 0.0.0.0 default to be made explicit (ticket ...)>",
-      "reviewed_on": "2026-10-06",
+      "reviewed_on": "2026-11-06",
```

## Sign-off (2026-11-06)

The owner replied "Bandit OK as recommended" at about 16:05Z. It was applied on top of #3420's renewal of all 96 entries, which had reviewed them on 11-06 to expire on 2027-02-04:

- **TIGHTEN, 18 entries:** they expire on 2026-12-06, and their reasons say hardening is due before then.
- **KEEP, 77 entries:** they expire on 2027-02-04. Template B310 reasons now state the URL's actual source (groups a–d). Entries that said "no human approval" now record the owner's acceptance, and every entry's owner is `@ognjhunt`.
- **[51]:** fixed in `wam_compute_providers.py`:
  - the DeepInfra result URL opens only over https on `deepinfra.com` or `*.deepinfra.com`;
  - the download is deleted and refused past 2 GiB;
  - a `data:` URI is refused too, which stops that job and is not risky.

  Bandit still flags the `urlopen`. Its old entry is gone, and a fresh `mitigated_by_control` entry (line 353) was computed with the gate's own `finding_fingerprint`.

The gate passes on the branch: high 0, medium 96, no blockers.
