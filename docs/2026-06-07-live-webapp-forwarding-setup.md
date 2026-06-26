# Live WebApp Forwarding Setup - 2026-06-07

This note records the proof-bounded live setup for WebApp-to-Pipeline
`robot_eval_job_request.v1` forwarding.

## Configured Paths

- Public TLS endpoint: `https://paperclip.tryblueprint.io/api/live-pipeline/job-requests`
- Droplet: `paperclip-prod-01`, `206.81.11.69`
- Reverse proxy: Caddy route for `/api/live-pipeline/*` to `127.0.0.1:8765`
- Pipeline service: `blueprint-pipeline-intake.service`
- Control-plane timer: `blueprint-pipeline-control-plane.timer`
- WebApp forward env keys:
  - `ROBOT_EVAL_JOB_REQUEST_FORWARD_URL`
  - `ROBOT_EVAL_JOB_REQUEST_FORWARD_TOKEN`
  - `ROBOT_EVAL_JOB_REQUEST_FORWARD_REQUIRED=true`
  - `ROBOT_EVAL_JOB_REQUEST_FORWARD_TIMEOUT_MS=10000`

Secret values are intentionally omitted. The intake token is stored only in
runtime environment configuration.

## Live Verification

- `https://paperclip.tryblueprint.io/` remained healthy after the Caddy route
  change.
- Unauthenticated POST to
  `https://paperclip.tryblueprint.io/api/live-pipeline/job-requests` returned
  `401`.
- Authenticated malformed POST through the public endpoint reached the intake
  service and returned `422` with `accepted=false`.
- Render WebApp deploy `dep-d8igd0urnols73bmqba0` went live on commit
  `87537a0c4e3236bef87d81634a4f8cac81bafc8c`.
- A live WebApp route smoke against
  `https://tryblueprint.io/api/robot-eval/job-requests` returned
  `pipeline_forward.status=failed`, `endpoint_configured=true`,
  `required=true`, and `pipeline_forward.http_status=422` because the smoke
  request intentionally used a capture root that does not match the active
  control plane.
- After the smoke, the live intake audit was reset to `waiting_for_inputs` and
  the proof-boundary audit returned `passed_external_inputs_blocked` with no
  internal blockers.

## Proof Boundary

This setup proves live network reachability and fail-closed handoff behavior. It
does not prove:

- real WebApp upstream truth for the active capture
- Isaac Lab-Arena owner-system execution
- robot policy execution
- contact, safety, or generated-world rank fidelity
- cloud storage delivery entitlement

The remaining external blockers are still `webapp_upstream_truth` and
`isaac_lab_arena_owner_evidence`.
