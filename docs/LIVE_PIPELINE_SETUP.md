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
- OpenAI Agents SDK and Codex SDK module availability
- WebApp upstream IDs required for production proof
- Arena package proof-boundary audit status
- optional DigitalOcean control-plane metadata

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

Install templates live under:

```text
deploy/systemd/blueprint-pipeline-control-plane.service
deploy/systemd/blueprint-pipeline-control-plane.timer
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

It is not by itself:

- Isaac Lab-Arena execution proof
- GPU provisioning proof
- robot policy execution proof
- physics/contact validation
- safety validation
- robot readiness proof

Those claims require owner-system simulator logs, accepted artifacts, and the
normal proof-boundary audit.

Do not commit DigitalOcean tokens. If a token was pasted into chat or a terminal
transcript, rotate it after use.
