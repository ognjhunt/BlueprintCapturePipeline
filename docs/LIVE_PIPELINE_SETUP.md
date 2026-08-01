# Live Pipeline Setup

This repo can automate the deterministic Arena/package pipeline locally. Live
external execution is a separate gate and must remain explicit.

## Setup Audit

Run:

```bash
blueprint-audit-live-pipeline-setup \
  --capture-root /path/to/capture-root \
  --package-dir /path/to/capture-root/pipeline/robot_eval_jobs/<job_id> \
  --digitalocean-droplet-name paperclip-prod-01 \
  --digitalocean-droplet-ip 206.81.11.69
```

The command writes:

```text
pipeline/live_pipeline_setup/live_pipeline_setup_manifest.json
```

It checks:

- local env files, with secret values redacted from output
- simulator, vision-labeling, and delivery command hooks
- owner-supplied Arena result directories ready for ingest
- OpenAI Agents SDK and Codex SDK module availability
- WebApp upstream IDs required for production proof
- Arena package proof-boundary audit status
- optional DigitalOcean control-plane metadata

`--arena-results-dir` or `BLUEPRINT_ARENA_RESULTS_DIR` is accepted as an
owner-supplied result-ingest path. A directory with JSON result artifacts can
make the Arena section `ready_for_result_ingest` without opening
`BLUEPRINT_ALLOW_SIMULATOR_EXECUTION`. That does not prove simulator execution,
robot policy execution, contact, safety, or generated-world rank fidelity; it only means the
pipeline has result artifacts it can ingest and audit.
The live closure audit also checks
`simulation_automation/simulator_engine_plugin_registry.json`; every supported
engine must be present with a ready adapter contract and managed execution
support before the simulator-plugin gate can pass.

## Always-On Control Plane

The droplet can run one safe control-plane pass on a timer:

```bash
blueprint-run-live-pipeline-control-plane
```

The command loads repo/cwd/capture env files, writes
`pipeline/live_pipeline_control_plane/live_pipeline_control_plane_manifest.json`,
runs the setup audit, and consumes `BLUEPRINT_ROBOT_EVAL_JOB_REQUEST_INBOX`
through the existing `robot_eval_job_request.v1` orchestrator. In fleet mode,
each accepted request supplies `site_package.capture_root`; no single global
`BLUEPRINT_PIPELINE_CAPTURE_ROOT` is required.

The control-plane command itself exits 0 even when blocked so a timer does not
restart-loop. The production systemd unit runs a post-check that reads the
manifest and either sends `BLUEPRINT_OPERATOR_ALERT_WEBHOOK_URL` or fails the
unit when `BLUEPRINT_OPERATOR_ALERT_REQUIRE_WEBHOOK=true` and a blocked pass has
no configured webhook. Missing capture roots, inboxes, simulator commands,
vision commands, delivery commands, or proof inputs are recorded as manifest
blockers. It also writes:

```text
pipeline/live_pipeline_control_plane/live_pipeline_external_input_packet.json
pipeline/live_pipeline_control_plane/live_pipeline_external_input_packet.md
pipeline/live_pipeline_control_plane/live_pipeline_proof_boundary_audit.json
pipeline/live_pipeline_control_plane/live_pipeline_manifest_alert.json
pipeline/live_pipeline_control_plane/live_pipeline_input_intake_audit.json
pipeline/live_pipeline_control_plane/live_pipeline_staged_inputs.json
```

### GPU Spend Watchdog

Install and enable the companion spend guard timer alongside the control plane:

```bash
sudo cp deploy/systemd/blueprint-gpu-spend-guard.service /etc/systemd/system/
sudo cp deploy/systemd/blueprint-gpu-spend-guard.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now blueprint-gpu-spend-guard.timer
```

The timer runs `scripts/gpu_spend_guard.py --reap` every few minutes and writes:

```text
/var/lib/blueprint/pipeline-control-plane/gpu_spend_guard/latest.json
```

That `gpu_spend_guard.v1` snapshot is spend/allocation evidence only. It records
typed provider-inventory outcomes, live allocations, expiring warm-worker
ownership leases, reap candidates, verified reap results, optional current
billing-export reconciliation, and the aggregate `gpu_fleet_budget_guard.v1`
ceiling. Missing credentials, inventory API failures, unverified DigitalOcean
deletion, stale warm markers, and failed reap attempts keep the command red.
Launch gates should pass that file as both
`--spend-guard-pre-snapshot` and `--spend-guard-post-snapshot` around paid
canaries, or use two copied snapshots if preserving before/after state:

```bash
python scripts/run_external_alpha_launch_gate.py \
  --spend-guard-pre-snapshot /var/lib/blueprint/pipeline-control-plane/gpu_spend_guard/pre-canary.json \
  --spend-guard-post-snapshot /var/lib/blueprint/pipeline-control-plane/gpu_spend_guard/post-canary.json
```

The gate fails when snapshots are missing, stale, not generated with `--reap`,
still contain reap candidates, or carry a blocked fleet budget. This is a cost
and teardown guard, not provider runtime, artifact quality, or task-success proof.

Render workers add pod-side containment on top of that host watchdog:
`BLUEPRINT_RENDER_POD_HARD_TTL_SECONDS` defaults to 7200 seconds and terminates
the worker command if the launcher/control plane dies, while
`BLUEPRINT_RENDER_POD_IDLE_TTL_SECONDS` defaults to 1800 seconds after
`runner_done`. Provider API teardown is still required for billing proof; these
limits only prevent an in-container render process from running forever.

Customer robot-eval failover uses the generated
`gpu_provider_race_handoff.json` plus:

```bash
BLUEPRINT_ALLOW_GPU_PROVIDER_RACE_LAUNCH=true \
blueprint-run-robot-eval-provider-race \
  --job-dir /path/to/capture-root/pipeline/robot_eval_jobs/<job_id> \
  --allow-live-provider-race
```

The launcher runs provider adapter commands serially until one succeeds. It is a
runtime failover path, not proof that the remote worker completed the simulator
or produced valid task evidence.

Lambda is not a live fallback candidate: its adapter is retained only for dry-run and
read-only inventory compatibility, and its mutating CLI modes are hard-disabled.
Any pre-existing Lambda allocation requires separately authorized provider-console/API
cleanup plus a fresh read-only zero-inventory check; guest shutdown is not spend
closure.

### Sim-Only Beta Profile

Use this profile when the beta surface is intentionally simulator-only and every
accepted upload or WebApp job request should progress into task-eval/simulation
automation without waiting for IRL evidence:

```bash
BLUEPRINT_SIM_ONLY_BETA_DEFAULT_TASK_EVAL=true
BLUEPRINT_SIM_ONLY_BETA_AUTONOMY=true
BLUEPRINT_ALLOW_SIMULATOR_EXECUTION=true
BLUEPRINT_MUJOCO_G1_MODEL_ROOT=/path/to/mujoco_menagerie/unitree_g1
```

`BLUEPRINT_SIM_ONLY_BETA_DEFAULT_TASK_EVAL=true` routes captures with no
explicit requested outputs into `qualification`, `evaluation_prep`, and
`simulation_automation`. `BLUEPRINT_SIM_ONLY_BETA_AUTONOMY=true` makes
auto-staged job requests and the live control plane prefer MuJoCo instead of
the fixture simulator. `BLUEPRINT_ALLOW_SIMULATOR_EXECUTION=true` is still
required before a non-fixture simulator command can execute. The packaged
command is configured only when `ROBOT_EVAL_JOB_DEFAULT_SIMULATOR_COMMAND` is
set, `BLUEPRINT_MUJOCO_G1_MODEL_ROOT` points at local Unitree G1 MuJoCo assets,
or `BLUEPRINT_MUJOCO_ALLOW_FETCH_G1_ASSETS=true` allows asset fetch.

Optional beta knobs:

```bash
BLUEPRINT_MUJOCO_BETA_STEPS=32
BLUEPRINT_MUJOCO_BETA_SKIP_RENDER_FRAMES=false
```

Do not enable `BLUEPRINT_MUJOCO_BETA_SKIP_RENDER_FRAMES=true` for customer beta
closure evidence; sim-only beta core closure still requires visual media
coverage, trace package coverage, attempt metrics, and scenario-run coverage.
This profile does not prove generated-world rank fidelity, live
customer delivery, or external robot-team closure. WAM/substrate artifacts add
only evaluator-bounded policy comparison unless paired real-world validation
anchors are accepted separately.

That packet is the machine-readable handoff for the remaining external inputs:

- real WebApp upstream IDs: `site_submission_id`, `request_id`,
  `buyer_request_id`, and `capture_job_id`
- accepted WebApp sources: `capture_descriptor.json`, `raw/manifest.json`,
  `pipeline/opportunity_handoff.json`, and queued
  `robot_eval_job_request.v1` files for scheduling when their source identifies
  Blueprint-WebApp and their `site_package.capture_root` matches the capture
  root under audit
- owner-system Isaac Lab-Arena result artifacts under
  `BLUEPRINT_ARENA_RESULTS_DIR` or a gated simulator command path
- robot-team policy package references for one supported execution or trace
  modality
- real-world validation follow-up draft queues generated by completed jobs, with
  a safe `blueprint-run-robot-eval-job --capture-root ... --job-request-inbox ...`
  command for exact rerun requests
- gated command hooks for rollout vision labeling and package delivery
- gated Agents SDK and Codex SDK/Codex CLI operator credentials

The example `robot_eval_job_request.v1` inside the packet uses placeholders
only. The packet is a request/contract artifact and does not prove simulator
execution, robot policy execution, contact, safety, or generated-world rank fidelity.
Follow-up queues are also request-contract artifacts: processing them creates a
new deterministic rerun job, but real-world validation still requires fresh
owner-supplied actuals and closure evidence.

`live_pipeline_setup` supports two upstream modes. In single-capture audit mode,
provide `--capture-root` or `BLUEPRINT_PIPELINE_CAPTURE_ROOT`; queued WebApp
requests count as upstream truth only when the request includes all four WebApp
IDs and `site_package.capture_root` resolves to that configured capture root.
In fleet mode, configure `--job-request-inbox` or
`BLUEPRINT_ROBOT_EVAL_JOB_REQUEST_INBOX`; the setup audit reports
`ready_for_per_request_capture_roots` when the inbox exists. That is intake
readiness only: the control plane still resolves and validates
`site_package.capture_root` per request, and requests for another capture are
reported in `webapp_inbox_truth` rather than silently accepted.
When a queued WebApp request carries the public library path
`/synced-artifacts/sites/<slug>`, configure
`ROBOT_EVAL_JOB_REQUEST_FORWARD_CAPTURE_ROOT_BY_SITE_JSON` to map that slug to
the local Pipeline capture root. `ROBOT_EVAL_JOB_REQUEST_FORWARD_CAPTURE_ROOT`
is a narrower single-root fallback for one-site rehearsals. Without one of
those overrides, the inbox runner quarantines the request instead of treating
the public WebApp path as local capture truth.

Before a multi-site beta, validate that every beta site slug is covered on both
the WebApp preflight and Pipeline maps:

```bash
python scripts/validate_capture_root_by_site_coverage.py \
  --expected-site-roots-json '{"site-one":"/captures/site-one","site-two":"/captures/site-two"}' \
  --pipeline-site-roots-json "$ROBOT_EVAL_JOB_REQUEST_FORWARD_CAPTURE_ROOT_BY_SITE_JSON" \
  --webapp-forwarding-preflight /path/to/Blueprint-WebApp/output/pipeline/robot_eval_job_requests/forwarding_preflight.json \
  --require-paths-exist \
  --output output/beta_capacity/capture_root_by_site_coverage.json
```

`capture_root_by_site_beta_coverage.v1` is coverage evidence only. It does not
prove live forwarding, pipeline processing, simulator execution, or buyer
delivery; it only proves the beta site slug map is complete before requests are
allowed to fan out across multiple capture roots.

The proof-boundary audit can also be run directly:

```bash
blueprint-audit-live-pipeline-proof-boundary \
  --manifest-path /var/lib/blueprint/pipeline-control-plane/live_pipeline_control_plane_manifest.json
```

It verifies manifest/packet/setup consistency, checks that `secrets_leaked`
remains false, rejects forbidden proof-boolean upgrades, and separates external
blockers from internal audit failures. A healthy waiting state exits zero unless
`--require-live-ready` is provided. When `live_pipeline_staged_inputs.json`
exists, the audit also validates its schema and proof boundary; malformed or
blocked staged pointers fail internally.

Candidate external inputs can be audited before the timer sees them:

```bash
blueprint-intake-live-pipeline-inputs \
  --manifest-path /var/lib/blueprint/pipeline-control-plane/live_pipeline_control_plane_manifest.json \
  --webapp-job-request /path/to/robot_eval_job_request.json \
  --arena-results-dir /path/to/owner-system/isaac-lab-arena-results \
  --policy-package /path/to/robot_team_policy_package.json \
  --stage-webapp-request \
  --stage-arena-results \
  --stage-policy-package
```

The intake command checks that a WebApp request is a direct
`robot_eval_job_request.v1` or queue envelope, contains all four WebApp IDs, and
either points at the configured single-capture root or carries a real
`site_package.capture_root` for per-request mode. With `--stage-webapp-request`,
it copies that validated request into the configured inbox. Arena result
directories are only marked `ready_for_ingest`; intake does not run Arena, set
env files, process the job, or upgrade proof claims. With `--stage-arena-results`, intake writes
`live_pipeline_staged_inputs.json`; the next control-plane pass can consume the
validated owner-results pointer when no `BLUEPRINT_ARENA_RESULTS_DIR` or
`--arena-results-dir` override is set.

`--policy-package` accepts `robot_team_policy_package.v1` or a direct
policy-package body for one supported robot-team modality: API endpoint, Docker
container, recorded action trace, high-level skill trace, teleop demo, or sim
controller plugin. With `--stage-policy-package`, intake writes the validated
handoff to
`pipeline/robot_eval_inputs/<job_id>/policy_package.json`. This is an execution
input only; policy proof still requires the job-level policy execution bundle to
produce attempts. The closure audit revalidates selected modality status and
modality-specific required fields before the policy-interface gate can pass.

`--real-robot-pov` is optional and accepts `real_robot_pov_manifest.v1` with exact
`scenario_eval_run_id` and `scenario_variation_instance_id` keys, robot camera
video refs, action log refs, timestamp alignment, and owner evidence or
operator attestation. With `--stage-real-robot-pov`, intake copies the manifest
to `pipeline/robot_eval_inputs/real_robot_pov_manifest.json`. Generated robot
POV support artifacts do not satisfy real-POV proof, but missing real POV
evidence is a diagnostic/proof-boundary state and not a sim-only control-plane
blocker.

For live WebApp-to-droplet handoff, run the authenticated intake service:

```bash
BLUEPRINT_LIVE_PIPELINE_INTAKE_TOKEN=<redacted> \
blueprint-live-pipeline-intake-service --host 127.0.0.1 --port 8765
```

The intake token is intentionally not a default. It is the shared HMAC secret
used to sign WebApp forwarding requests and Pipeline intake-audit probes, so it
must live in local/deployment secrets instead of source control. Generate a
local env file with matching WebApp and Pipeline variables:

```bash
python -m blueprint_pipeline.live_pipeline_forwarding_secret_setup \
  --env-file "$HOME/.blueprint-secrets/live_pipeline_forwarding.env" \
  --forward-url "https://paperclip.tryblueprint.io/api/live-pipeline/job-requests" \
  --capture-root "$CAPTURE_ROOT" \
  --site-slug "$WEBAPP_SITE_SLUG"
```

Source that file on the Pipeline intake host before starting the service, and
pass the same file to the WebApp read-only preflight:

```bash
set -a
source "$HOME/.blueprint-secrets/live_pipeline_forwarding.env"
set +a
blueprint-live-pipeline-intake-service --host 127.0.0.1 --port 8765

npm run pipeline:forwarding:preflight -- \
  --require-forwarding \
  --probe-intake-audit \
  --forwarding-env-file "$HOME/.blueprint-secrets/live_pipeline_forwarding.env"
```

For production deployment, copy the same generated token into the WebApp secret
store as `ROBOT_EVAL_JOB_REQUEST_FORWARD_TOKEN` and the Pipeline intake service
secret store as `BLUEPRINT_LIVE_PIPELINE_INTAKE_TOKEN`. WebApp sends
`X-Blueprint-Pipeline-Timestamp`, `X-Blueprint-Pipeline-Nonce`, and
`X-Blueprint-Pipeline-Signature: sha256=<hmac>` over `timestamp.nonce.body`; it
does not send the shared value as a bearer credential.

The service exposes:

- `GET /health`
- `GET /api/live-pipeline/version`
- `POST /api/live-pipeline/capture-upload-intakes`
- `POST /api/live-pipeline/capture-upload-intakes/{capture_session_id}/{intake_id}/lifecycle`
- `GET /api/live-pipeline/capture-upload-intakes/{capture_session_id}/{intake_id}/lifecycle`
- `POST /api/live-pipeline/capture-upload-intakes/{capture_session_id}/{intake_id}/provider-deletion-evidence`
- `POST /api/live-pipeline/capture-upload-intakes/{capture_session_id}/{intake_id}/external-revocation-evidence`
- `POST /api/live-pipeline/reconstructions/plan`
- `POST /api/live-pipeline/reconstructions/{plan_id}/authorize`
- `POST /api/live-pipeline/reconstructions/{plan_id}/execute`
- `GET /api/live-pipeline/reconstructions/{plan_id}`
- `POST /api/live-pipeline/testbeds/compile`
- `POST /api/live-pipeline/task-evaluation-runs/plan`
- `POST /api/live-pipeline/task-evaluation-runs/{run_id}/authorize`
- `POST /api/live-pipeline/task-evaluation-runs/{run_id}/execute`
- `GET /api/live-pipeline/task-evaluation-runs/{run_id}`

`GET /health` includes a non-secret `task_evaluation_supervisor` object. Check
`configuration_status`, `zero_spend_lifecycle_ready`,
`live_inference_configured`, `live_operator_gate_enabled`,
`live_inference_ready`, and `execution_profile_digest` before enabling capture
traffic. The health response never exposes the configured budget, credentials,
or raw environment values, and always reports that proof/recovery authority is
not granted by the inference profile.
- `POST /api/live-pipeline/job-requests`
- `POST /api/live-pipeline/policy-packages`
- `POST /api/live-pipeline/real-robot-pov`
- `POST /api/live-pipeline/deployment-outcomes`
- `POST /api/live-pipeline/live-closure-evidence`
- `GET /api/live-pipeline/intake-audit`

`GET /api/live-pipeline/version` is the deployment-identity authority. It
returns HTTP 503 unless `BLUEPRINT_SOURCE_COMMIT` is an exact 40-hex commit
bound by the deployment process. A successful response uses schema
`blueprint_pipeline_deployment_identity.v1`, sets `commit_proven=true`, and
keeps the claim ceiling at `deployed_service_identity_only`. An operator-typed
commit, a mutable image tag, or a healthy process is not a substitute for this
response.

Generate an environment-bound deployment proof only from clean checkouts that
match `origin/main`:

```bash
python scripts/run_sim_only_beta_deployment_parity_proof.py \
  --capture-root <rights-cleared-capture-root> \
  --deployment-environment staging \
  --webapp-url https://<staging-webapp-host> \
  --pipeline-intake-url https://<staging-pipeline-host>/api/live-pipeline/job-requests \
  --webapp-repo <clean-webapp-main-checkout> \
  --pipeline-repo <clean-pipeline-main-checkout> \
  --capture-repo <clean-capture-main-checkout> \
  --allow-local-git-parity-only
```

The proof verifies WebApp `/version.json`, Pipeline
`/api/live-pipeline/version`, health/intake readiness, and exact clean git
parity. `--allow-local-git-parity-only` disables only the redundant
operator-supplied commit cross-check; it never disables either live identity
probe. A verified `staging` proof sets `staging_deployment_proven=true` and
cannot satisfy the production release gate. Generate a separate `production`
proof only after the production deployment and rollback gates pass.

### Isolated Pipeline staging intake

The persistent host may run a second, localhost-only intake process for staging
without sharing the production checkout, state tree, port, or HMAC secret. Use
the checked-in staging unit and installer:

```bash
STAGING_REPO=/opt/blueprint/BlueprintCapturePipeline-staging \
  scripts/install_live_pipeline_staging.sh
```

The staging checkout must be clean with `HEAD == origin/main`. Fill
`/etc/blueprint/pipeline-intake-staging.env` with that exact 40-character commit
and a newly generated staging-only HMAC secret, then enable the service:

```bash
systemctl enable --now blueprint-pipeline-intake-staging.service
curl --fail http://127.0.0.1:8766/health
curl --fail http://127.0.0.1:8766/api/live-pipeline/version
```

The unit uses `/var/lib/blueprint-staging`, port `8766`, and the production
fail-closed runtime posture. It has no control-plane trigger, Pub/Sub listener,
provider credential, simulator gate, paid-compute gate, or physical-action gate.
Health/version success proves only the isolated intake process and exact source
identity. Capture-transfer readiness additionally requires an allowlisted signed
download host and absolute malware-scanner command, and the cross-repository
staging proof still requires a real staging WebApp identity and health endpoint.

`site_task_testbed_compilation_submission.v2` must name the exact accepted
session/intake, approved-task digest, completed reconstruction plan/result, new
testbed ID/version, an owner-attested robot binding, and optional provider-neutral
Decision/Evidence Request constraints. It must not contain capture/QA/
reconstruction artifacts, SimReady or placement conclusions, evaluator/reset
references, supported-condition claims, or a predecessor manifest. Pipeline
loads or derives those scientific artifacts itself and currently emits an
explicit placement abstention until qualified candidate evidence exists.

The capture-upload intake endpoint is a separate Task Evaluation Run product
seam; it does not use the legacy robot-evaluation capture-handoff converter or a
caller-selected local capture root. Configure it with:

```bash
PIPELINE_CAPTURE_INTAKE_STORE_ROOT=/var/lib/blueprint/capture-intakes
PIPELINE_CAPTURE_TRANSFER_ALLOWED_HOSTS=f005.backblazeb2.com
PIPELINE_CAPTURE_MALWARE_SCANNER_ARGV_JSON='["/usr/bin/clamdscan","--no-summary"]'
```

The host list must contain the exact HTTPS download host returned by the current
B2 account authorization. The scanner argv must name an absolute installed
executable; no shell expansion is used. The endpoint consumes an HMAC-authenticated,
short-lived object-prefix grant, rejects redirects outside the same allowlist,
streams into quarantine, verifies exact size and media shape, requires a clean
scanner result, computes SHA-256, writes an immutable content-addressed Capture
Intake receipt, and runs deterministic Capture QA against the same verified
bytes. It returns the intake receipt and a separately digest-validated Capture QA
publication; it never persists or echoes the URL or grant. Capture QA may accept
the input or request exact recapture, but neither artifact establishes
reconstruction, task success, physical success, deployment readiness, safety
certification, or policy-ranking support.

Reconstruction planning and execution require the same configured capture
store. The service resolves capture bytes by the accepted session/intake
receipt and never accepts a caller-selected local path. Planning alone is not
authorization. The authorize endpoint must name an exact planned local adapter;
the execute endpoint cannot enable live providers, paid compute, or physical
robot work. Testbed compilation accepts
`site_task_testbed_compilation_submission.v2`, which references the exact
Pipeline-owned reconstruction plan and execution-result digest. Version 2
rejects caller-supplied intake, QA, reconstruction-plan, or reconstruction-result
objects.

Completed-capture lifecycle actions are destructive and fail closed. They
require the exact capture and envelope digests and never accept a caller path.
Consent revocation and operator deletion require the intake's revocation policy;
retention expiry is computed from the immutable receipt time plus `max_days`;
legal hold prevents deletion. A marker blocks use before deletion begins. The
final tombstone contains digests and deletion counts, not the raw capture or
customer identifiers. Shared content-addressed objects are preserved until the
last active reference is removed. Provider deletion and WebApp/signed-download
revocation remain separate, inspectable obligations rather than being inferred
from local file deletion.

Signed intake headers are required by default. Temporary legacy bearer support
exists only when `BLUEPRINT_LIVE_PIPELINE_INTAKE_ALLOW_LEGACY_BEARER=true` is
set on the Pipeline service. The POST body can be either a direct
`robot_eval_job_request.v1` or a `robot_eval_job_request_inbox.v1` queue
envelope. A staged request is still only handoff proof; the control plane must
process it and the proof audit must remain clean. Keep the service bound to
localhost unless a reverse proxy/TLS layer owns public exposure. To trigger the
one-shot control plane after successful staging, set
`BLUEPRINT_ALLOW_LIVE_PIPELINE_INTAKE_TRIGGER=true` and
`BLUEPRINT_LIVE_PIPELINE_INTAKE_TRIGGER_COMMAND`, for example
`/bin/systemctl start blueprint-pipeline-control-plane.service`.

For capture-bridge Pub/Sub handoffs, the deployed
`blueprint-pubsub-handoff-listener.timer` repeatedly drains
`BLUEPRINT_PUBSUB_HANDOFF_SUBSCRIPTION`. In production the listener should keep:

```bash
BLUEPRINT_PUBSUB_HANDOFF_STAGE_CONTROL_PLANE=true
BLUEPRINT_PUBSUB_HANDOFF_SKIP_RUN_E2E=true
```

With those flags, the listener claims a revisioned owner/token lease, extends
the Pub/Sub ack deadline while work is active, downloads the completed capture bundle, enriches
the handoff from staged raw sidecars, writes a `robot_eval_job_request.v1`
envelope into `BLUEPRINT_ROBOT_EVAL_JOB_REQUEST_INBOX`, and records the staged
request path in `pipeline_job_ledger.json`. Terminal output is committed through
`pipeline_job_output_commit.json`; retryable/blocked outcomes are nacked and
permanent-invalid inputs are acknowledged with typed failure evidence. It does not execute simulator or
provider work; the next control-plane pass consumes the inbox and resolves
`site_package.capture_root` per request.

Install templates live under:

```text
deploy/systemd/blueprint-pipeline-control-plane.service
deploy/systemd/blueprint-pipeline-control-plane.timer
deploy/systemd/blueprint-pipeline-intake.service
deploy/systemd/pipeline-control-plane.env.example
scripts/install_live_pipeline_control_plane.sh
```

The env file should provide paths and optional gates, not secrets in unit files.
`scripts/install_live_pipeline_control_plane.sh` creates the default
`/var/lib/blueprint/pipeline-control-plane` state tree, including
`robot-eval-job-requests` and `incoming_webapp_job_requests`. Leave live action
gates unset until the exact command hook is ready.

## OpenAI Auth Boundary

Repo subprocesses cannot read ChatGPT Pro or Codex host OAuth tokens. When a
pipeline step runs inside this Python CLI, use one of:

- `OPENAI_API_KEY` for OpenAI API/SDK calls
- `BLUEPRINT_ALLOW_CODEX_CLI_HOST_OAUTH=true` plus
  `BLUEPRINT_ALLOW_LIVE_CODEX_SDK_OPERATORS=true` when the installed `codex`
  CLI is already authenticated by the host/user profile
- a configured command hook that owns its own OAuth flow
- a host-triggered tool outside this subprocess, with its output returned as a
  deterministic artifact

ChatGPT Pro/Codex OAuth is useful through the installed Codex CLI or where the
host application triggers a tool directly. It is not exported as a secret token
and must not be copied into pipeline manifests.

## Vision Model Boundary

The pipeline is not hardwired to Gemini. Rollout vision labeling is behind the
replaceable `BLUEPRINT_ROLLOUT_VISION_LABELING_COMMAND` contract. The command
must write `rollout_vision_labels.command.json` in the Arena package output
directory, and ingest will force command labels to remain review-required.

Supported setup paths:

- OpenAI: `blueprint-label-rollout-vision-openai`, using `OPENAI_API_KEY`
- Gemini/Google GenAI: a wrapper using `GEMINI_API_KEY` or
  `GOOGLE_GENAI_API_KEY`
- another reviewed local/HTTP command that writes the same command-label JSON

OpenAI setup example:

```bash
BLUEPRINT_ALLOW_ROLLOUT_VISION_LABELING=true
BLUEPRINT_ROLLOUT_VISION_LABELING_COMMAND="blueprint-label-rollout-vision-openai --output-dir ."
```

Model-derived labels are support evidence only. They do not prove contact,
safety, policy execution, or generated-world rank fidelity until accepted through review or
owner-system proof.

## Delivery Command Boundary

The default Arena package path already writes a local `delivery_bundle/`.
Gated delivery commands are optional and must write `delivery_upload.command.json`
or `signed_access.command.json` in the Arena package output directory.

Built-in local delivery:

```bash
BLUEPRINT_ALLOW_PACKAGE_DELIVERY_UPLOAD=true
BLUEPRINT_LOCAL_DELIVERY_ROOT=/var/lib/blueprint/pipeline-control-plane/deliveries
BLUEPRINT_PACKAGE_DELIVERY_UPLOAD_COMMAND="blueprint-deliver-arena-package-local --output-dir ."
```

`blueprint-deliver-arena-package-local` copies the delivery bundle into the
local delivery root and returns local access paths.

Built-in GCS delivery source for WebApp signed-URL handoff:

```bash
BLUEPRINT_ALLOW_PACKAGE_DELIVERY_UPLOAD=true
BLUEPRINT_PACKAGE_DELIVERY_GCS_PREFIX=gs://blueprint-buyer-artifacts/post-training-packages
BLUEPRINT_PACKAGE_DELIVERY_ENTITLEMENT_ID=<marketplaceEntitlements doc id>
BLUEPRINT_PACKAGE_DELIVERY_UPLOAD_COMMAND="blueprint-deliver-arena-package-local --output-dir ."
```

With `BLUEPRINT_PACKAGE_DELIVERY_GCS_PREFIX`, the command uploads the
`delivery_bundle/` plus package archive to GCS, writes
`artifact_uri` / `post_training_data_package_uri`, and prepares a
`marketplace_entitlement_patch` payload for the WebApp entitlement record. Set
`BLUEPRINT_PACKAGE_DELIVERY_SIGNED_URLS=true` only for an owner-reviewed proof
run that should mint short-lived signed URLs locally. Entitlement authorization
and buyer access remain WebApp responsibilities; a cloud object URI is delivery
source proof, not buyer authorization or deployment approval.

## DigitalOcean Droplet Boundary

`paperclip-prod-01` at `206.81.11.69` can be used as an always-on control plane
for scheduling, manifest hosting, repo sync, and watchdogs.

DigitalOcean API reads are optional advisory inventory checks. Leaving the read
gate unset does not block the pipeline; setting the gate without a token or
matching droplet still fails closed in the setup manifest.

It is not by itself:

- Isaac Lab-Arena execution proof
- GPU provisioning proof
- robot policy execution proof
- physics/contact validation
- off-scope validation
- generated-world rank fidelity proof

Those claims require owner-system simulator logs, accepted artifacts, and the
normal proof-boundary audit. Job-level closure is recorded in
`pipeline/robot_eval_jobs/<job_id>/live_eval_closure_manifest.json`; the only
ready state for the full live loop is `live_end_to_end_verified`. Anything else
is a local/package-ready or externally blocked state, even if deterministic
package artifacts are complete.

Optional GPU/provider closure can be audited separately with
`blueprint-audit-provider-closure --job-dir
pipeline/robot_eval_jobs/<job_id>`. The command is read-only: it checks local
watchdog, spend-ledger, artifact-output finalizer/upload, and teardown evidence
and writes `provider_closure_audit_report.json`. Missing credentials or missing
provider artifacts are blocked optional provider closure, not a local sim-only
beta blocker and not proof of rank fidelity, physical readiness, safety, or
field success.

Robot-team policy packages should be staged per job before claiming policy
execution input readiness:

```bash
blueprint-intake-live-pipeline-inputs \
  --manifest-path /var/lib/blueprint/pipeline-control-plane/live_pipeline_control_plane_manifest.json \
  --policy-package /path/to/robot_team_policy_package.json \
  --stage-policy-package
```

The same path is available over HTTP:

```text
POST /api/live-pipeline/policy-packages
```

The body must carry a safe `job_id` and one supported modality. The service
stages the package under `pipeline/robot_eval_inputs/<job_id>/policy_package.json`
and does not run policy execution or set proof booleans.

Predicted-vs-actual deployment records can be staged separately:

```bash
blueprint-intake-live-pipeline-inputs \
  --manifest-path /var/lib/blueprint/pipeline-control-plane/live_pipeline_control_plane_manifest.json \
  --deployment-outcomes /path/to/deployment_outcome_manifest.json \
  --stage-deployment-outcomes
```

The intake command copies validated records to
`pipeline/robot_eval_inputs/<job_id>/deployment_outcomes/inbox/`. This is a
real-world validation diagnostic input only; the robot-eval job still has to
pair it with predictions before a calibration score appears. Exact
`scenario_eval_run_id` or `scenario_variation_instance_id` keys and owner
evidence are needed only before claiming calibration or real-world outcome
proof. Missing keys, unmatched run-level actuals, and missing owner evidence are
recorded as diagnostics without adding required inputs for sim-only work.

Closure evidence for review acceptance, signed delivery/access, rights/privacy,
and safety/contact/physics readiness should also be staged per job with:

```bash
blueprint-intake-live-pipeline-inputs \
  --manifest-path /var/lib/blueprint/pipeline-control-plane/live_pipeline_control_plane_manifest.json \
  --live-closure-evidence /path/to/live_eval_closure_evidence.json \
  --stage-live-closure-evidence
```

The intake command copies validated evidence to
`pipeline/robot_eval_inputs/<job_id>/live_eval_closure_evidence.json`. This is a
closure-audit input only; it does not set `rank_fidelity_result_proven`,
`non_ranking_operational_claim_validated`, or `public_claim_upgrade_allowed`.

When the authenticated intake service is running, the same handoffs can be
submitted without shell access through:

```http
POST /api/live-pipeline/deployment-outcomes
POST /api/live-pipeline/real-robot-pov
POST /api/live-pipeline/live-closure-evidence
```

Deployment-outcome bodies must include a safe `job_id`, task/scenario IDs, and
an actual result signal. Real-robot-POV bodies must include exact run/variation
keys plus camera/action evidence refs. Closure-evidence bodies must be
`live_robot_eval_closure_evidence.v1` JSON objects with a safe `job_id` and the
required review, delivery, and safety/contact/physics sections.

Do not commit DigitalOcean tokens. If a token was pasted into chat or a terminal
transcript, rotate it after use.
