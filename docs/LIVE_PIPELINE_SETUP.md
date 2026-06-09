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
robot policy execution, contact, safety, or robot readiness; it only means the
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
through the existing `robot_eval_job_request.v1` orchestrator when both inbox
and capture root are configured.

The control plane exits 0 even when blocked so a systemd timer does not
restart-loop. Missing capture roots, inboxes, simulator commands, vision
commands, delivery commands, or proof inputs are recorded as manifest blockers.
It also writes:

```text
pipeline/live_pipeline_control_plane/live_pipeline_external_input_packet.json
pipeline/live_pipeline_control_plane/live_pipeline_external_input_packet.md
pipeline/live_pipeline_control_plane/live_pipeline_proof_boundary_audit.json
pipeline/live_pipeline_control_plane/live_pipeline_input_intake_audit.json
pipeline/live_pipeline_control_plane/live_pipeline_staged_inputs.json
```

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
- gated command hooks for rollout vision labeling and package delivery
- gated Agents SDK and Codex SDK/Codex CLI operator credentials

The example `robot_eval_job_request.v1` inside the packet uses placeholders
only. The packet is a request/contract artifact and does not prove simulator
execution, robot policy execution, contact, safety, or robot readiness.

Queued WebApp requests count as upstream truth only when the request includes
all four WebApp IDs and `site_package.capture_root` resolves to the same
capture root configured for the control plane. Requests for another capture are
reported in `webapp_inbox_truth` but remain blocked.

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
points at the configured capture root. With `--stage-webapp-request`, it copies
that validated request into the configured inbox. Arena result directories are
only marked `ready_for_ingest`; intake does not run Arena, set env files, process
the job, or upgrade proof claims. With `--stage-arena-results`, intake writes
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

For live WebApp-to-droplet handoff, run the authenticated intake service:

```bash
BLUEPRINT_LIVE_PIPELINE_INTAKE_TOKEN=<redacted> \
blueprint-live-pipeline-intake-service --host 127.0.0.1 --port 8765
```

The service exposes:

- `GET /health`
- `POST /api/live-pipeline/job-requests`
- `POST /api/live-pipeline/policy-packages`
- `POST /api/live-pipeline/deployment-outcomes`
- `POST /api/live-pipeline/live-closure-evidence`
- `GET /api/live-pipeline/intake-audit`

Send the token with `Authorization: Bearer ...` or
`X-Blueprint-Intake-Token`. The POST body can be either a direct
`robot_eval_job_request.v1` or a `robot_eval_job_request_inbox.v1` queue
envelope. A staged request is still only handoff proof; the control plane must
process it and the proof audit must remain clean. Keep the service bound to
localhost unless a reverse proxy/TLS layer owns public exposure. To trigger the
one-shot control plane after successful staging, set
`BLUEPRINT_ALLOW_LIVE_PIPELINE_INTAKE_TRIGGER=true` and
`BLUEPRINT_LIVE_PIPELINE_INTAKE_TRIGGER_COMMAND`, for example
`/bin/systemctl start blueprint-pipeline-control-plane.service`.

Install templates live under:

```text
deploy/systemd/blueprint-pipeline-control-plane.service
deploy/systemd/blueprint-pipeline-control-plane.timer
deploy/systemd/blueprint-pipeline-intake.service
deploy/systemd/pipeline-control-plane.env.example
scripts/install_live_pipeline_control_plane.sh
```

The env file should provide paths and optional gates, not secrets in unit files.
Leave live action gates unset until the exact command hook is ready.

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
safety, policy execution, or robot readiness until accepted through review or
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
local delivery root and returns local access paths. It does not create signed
URLs, upload to cloud storage, verify entitlement, or upgrade proof claims.
Cloud signed-access delivery still requires a provider-specific command that
returns signed URLs and explicit owner review.

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
- safety validation
- robot readiness proof

Those claims require owner-system simulator logs, accepted artifacts, and the
normal proof-boundary audit. Job-level closure is recorded in
`pipeline/robot_eval_jobs/<job_id>/live_eval_closure_manifest.json`; the only
ready state for the full live loop is `live_end_to_end_verified`. Anything else
is a local/package-ready or externally blocked state, even if deterministic
package artifacts are complete.

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
real-world validation input only; the robot-eval job still has to pair it with
predictions before a calibration score appears. For predicted-vs-actual
calibration, every staged actual record must include `scenario_eval_run_id` or
`scenario_variation_instance_id`; task/scenario-only records remain real-world
validation inputs but keep `predicted_vs_actual_exact_match_keys` open. If an
actual record includes a `scenario_eval_run_id`, the prediction match must be
for that same run; unmatched run-level actuals remain predicted-vs-actual
blockers. `real_world_outcome_proven` requires owner evidence on every actual
outcome record, such as `evidence_refs`, an owner proof URI, or an
operator/owner attestation. If records have task, scenario, actual-result, and
exact-match-key fields but lack owner evidence, the control plane accepts them
for calibration and keeps `real_world_deployment_outcome_owner_evidence` as the
remaining proof blocker.

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
closure-audit input only; it does not set `robot_readiness_proven`,
`safety_validated`, or `public_claim_upgrade_allowed`.

When the authenticated intake service is running, the same handoffs can be
submitted without shell access through:

```http
POST /api/live-pipeline/deployment-outcomes
POST /api/live-pipeline/live-closure-evidence
```

Deployment-outcome bodies must include a safe `job_id`, task/scenario IDs, and
an actual result signal. Closure-evidence bodies must be
`live_robot_eval_closure_evidence.v1` JSON objects with a safe `job_id` and the
required review, delivery, and safety/contact/physics sections.

Do not commit DigitalOcean tokens. If a token was pasted into chat or a terminal
transcript, rotate it after use.
