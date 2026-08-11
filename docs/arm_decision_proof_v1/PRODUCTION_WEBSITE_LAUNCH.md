# Production Website Task Evaluation Launch

This is the deployment bridge for ADP-009D's day-28 public-scene rehearsal and
the later Raw V3.2 partner-scene path. It does not redesign the scientific
harness. It makes the WebApp call the same immutable Pipeline profile and
canonical paid-resource allocator used by the maintained CLI path.

## Ownership

1. The WebApp records an admin/ops rights, spend, and execution authority
   envelope and sends one digest-bound `task_evaluation_launch_request.v1`.
2. `task_evaluation_launch_dispatcher` owns the durable pending -> processing ->
   completed/blocked state machine. Terminal or processing replays never invoke
   paid work again; the paid retry cap is zero.
3. `paid_resource_allocator gpu-canary` is the only provider-mutation boundary.
   The WebApp cannot provide its arguments, local paths, provider choice, or
   secrets.
4. The independent GPU spend guard and launch reconciler own liveness,
   provider inventory, orphan recovery, teardown closeout, and provider-zero.
5. The optional OpenAI Agents SDK supervisor has no tools. It can explain
   blockers, recommend only a deterministically admissible Pipeline profile, or
   request one human decision. The run remains safe and operable when it is off.

## Publish an immutable profile

Build the scene bundle and `EvaluationRunSpec` first. Their immutable URIs and
SHA-256 digests, locally readable immutable manifest/spec files and their
digests, the allocator input files, safe non-secret ADP runtime environment,
terminal result path, required Vast inventory, spend ceiling, TTL, secret
profile ID, execution-admission receipt, and control requirements must be
frozen in one `task_evaluation_launch_profile.v1` JSON file. Every allocator
output path must use the validated `{launch_run_root}` placeholder so separate
website launches cannot overwrite one another's evidence.

For the first frozen 840313 InteriorGS/SAGE dry route, build the exact profile
only from the clean protected-main production checkout. The builder rehashes all
five materialized scene files and emits immutable preflight and release inputs:

```bash
python scripts/build_adp009d_840313_launch_profile.py \
  --source-commit "$(git rev-parse HEAD)" \
  --repo-root /opt/blueprint/BlueprintCapturePipeline \
  --production-input-root /var/lib/blueprint/task-evaluation-inputs/adp009d-840313-interiorgs-sage-v1 \
  --provider-guard-path /var/lib/blueprint/pipeline-control-plane/gpu_spend_guard/latest.json \
  --output-dir /var/lib/blueprint/pipeline-control-plane/task-evaluation-profile-releases/"$(git rev-parse HEAD)"
```

Then validate and publish it with:

```bash
python scripts/publish_task_evaluation_launch_profiles.py \
  --profile /var/lib/blueprint/pipeline-control-plane/task-evaluation-profile-releases/<commit>/adp009d-840313-franka-dry-<commit>.json \
  --profile-dir /etc/blueprint/task-evaluation-launch-profiles \
  --webapp-catalog-out /var/lib/blueprint/pipeline-control-plane/task-evaluation-launch-profile-catalog.json
```

The command hashes every `immutable_inputs` file, fails on
profile/digest/control errors, and refuses to overwrite a
different profile with the same ID. The generated WebApp catalog deliberately
omits allocator arguments and runtime environment, contains no secret values,
and exposes only the safe execution-admission state and its typed blockers. The
intake service validates that projection again and serves it from
`GET /api/live-pipeline/task-evaluation-launch-profiles`; the WebApp discovers
that endpoint from its canonical Pipeline forwarding URL. An inline
`TASK_EVALUATION_LAUNCH_PROFILES_JSON` remains an optional emergency deployment
override, not the normal production source.

The dry profile is immutable and remains bound to the deployment commit that
created it. Do not overwrite it to enable spend. After the protected-main
controls canary passes, build a separate
`adp009d-840313-franka-live-v1` profile. The live builder verifies all five
InteriorGS/SAGE source bytes, the retained Aura construction and task-volume
exclusion lineage, the exact NuRec appearance byte, the SimReady can, SAGE
collision, harness/scenario inputs, the passing zero/positive control pair, the
allocator-owned artifact manifest, teardown, and an API-confirmed provider-zero
snapshot taken after teardown. It refuses to enable execution if any link is
missing or drifted:

```bash
python scripts/build_adp009d_840313_live_profile.py \
  --source-commit "$(git rev-parse HEAD)" \
  --repo-root /opt/blueprint/BlueprintCapturePipeline \
  --source-input-root /var/lib/blueprint/task-evaluation-inputs/adp009d-840313-interiorgs-sage-v1 \
  --runtime-input-root /var/lib/blueprint/task-evaluation-inputs/adp009d-840313-franka-runtime-v1 \
  --provider-guard-path /var/lib/blueprint/pipeline-control-plane/gpu_spend_guard/latest.json \
  --release-evidence-path <controls-canary-release-evidence.json> \
  --control-bundle-receipt-path <controls-canary-bundle-receipt.json> \
  --allocator-result-path <controls-canary-allocator-result.json> \
  --control-pair-path <controls-canary-adp009d_control_pair.v1.json> \
  --artifact-manifest-path <controls-canary-artifact_manifest.json> \
  --teardown-manifest-path <controls-canary-vast_teardown_manifest.json> \
  --provider-zero-guard-path <post-canary-provider-zero.json> \
  --readiness-uri s3://<immutable-artifact-store>/<receipt-digest>/readiness.json \
  --output-dir /var/lib/blueprint/pipeline-control-plane/task-evaluation-profile-releases/"$(git rev-parse HEAD)"/live
```

Upload the readiness byte to the exact immutable URI before publishing the
profile. The profile contains no credential: it names the canonical secret
profile and the allocator resolves Vast and gated-model credentials only from
the production secret integration. The dispatcher, not the profile, appends
`--execute` after validating a current signed spend envelope and the production
execute gate.

## Required production configuration

Pipeline host:

- install `deploy/systemd/blueprint-task-evaluation-launch-dispatcher.{path,service}`;
- enable the dispatcher path, GPU spend-guard timer, launch reconciler timer,
  and optional supervisor timer with `scripts/install_live_pipeline_control_plane.sh`;
- configure canonical intake HMAC client secrets, provider secret files,
  the read-only provider billing reconciler, artifact storage,
  `PIPELINE_SYNC_TOKEN`, and
  `PIPELINE_TASK_EVALUATION_LAUNCH_WEBAPP_URL`;
- configure `PIPELINE_TASK_EVALUATION_LAUNCH_SUPERVISION_WEBAPP_URL` when the
  optional supervisor should publish recommendations and human-decision prompts
  into the same WebApp control room;
- set `BLUEPRINT_TASK_EVALUATION_SECRET_PROFILE_ID` to the non-secret identity
  named in the immutable profile;
- set `BLUEPRINT_TASK_EVALUATION_LAUNCH_PUBLIC_CATALOG_PATH` to the publisher's
  generated catalog path;
- set `BLUEPRINT_ALLOW_TASK_EVALUATION_LAUNCH_TRIGGER=true` to accept and
  dispatch signed dry routes. Keep `BLUEPRINT_TASK_EVALUATION_LAUNCH_EXECUTE`
  unset for dry proof;
- set `BLUEPRINT_TASK_EVALUATION_LAUNCH_EXECUTE=true` only after selecting a
  profile whose `execution_admission.live_enabled` is true and separately
  confirming current rights, execution, and spend authority.

WebApp:

- configure the canonical `ROBOT_EVAL_JOB_REQUEST_FORWARD_URL` and
  `ROBOT_EVAL_JOB_REQUEST_FORWARD_TOKEN`; the Task Evaluation URL, public
  catalog URL, and HMAC secret are derived from that existing integration;
- configure the matching `PIPELINE_SYNC_TOKEN` for terminal callbacks;
- use `/ops/task-evaluation-launches` as the authenticated admin/ops control
  surface.

The first InteriorGS/SAGE run must keep `claim_ceiling=development_only`. A Raw
V3.2 capture or Scaniverse-derived import uses the same request/state/allocator
contracts with a separately frozen profile and its truthful source-kind and
rights evidence. Scaniverse-derived evidence does not silently become native
Raw V3.2 capture evidence.

## Promotion proof

Before the first paid website trigger, retain:

- deployed WebApp and Pipeline commit identities;
- exact profile, source bundle, and `EvaluationRunSpec` digests;
- signed WebApp queue receipt and Pipeline launch binding;
- allocator admission and spend authority;
- watchdog heartbeat, lossless policy media, artifact manifest, terminal
  receipt, teardown manifest, provider-zero report, and WebApp sync receipt;
- explicit confirmation that no provider mutation occurred inside either HTTP
  request path and no automatic paid retry occurred.

A queued request, an Agents SDK recommendation, a simulator startup, or a GPU
allocation alone is not a completed Task Evaluation Run.
