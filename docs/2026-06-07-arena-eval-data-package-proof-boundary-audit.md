# Arena Eval Data Package Proof-Boundary Audit

Date: 2026-06-07

Scope: `docs/goals/2026-06-07-arena-eval-data-package-live-operators.md`

## Result

Status: local deterministic implementation complete; live external execution remains gated.

The repo now has a proof-bounded Isaac Lab-Arena package lane that can take an
Arena-style fixture/result directory, schedule a 500-scenario batch, ingest
rollout results, normalize attempts, label failures, manifest clips, write
review-resolution artifacts, package JSONL/checksum/archive outputs, generate a
customer handoff report, prepare delivery artifacts, build a rerun plan, and log
operator decisions.

The lane intentionally does not claim:

- simulator execution proof from this process
- robot policy execution proof
- physics/contact validation
- safety validation
- robot readiness
- public claim upgrade eligibility

## Artifact Assertions

`blueprint-audit-arena-package` is the final local proof-boundary auditor. It
checks:

- required Arena result/package artifacts exist
- `arena_eval_schedule.json` matches the expected 500-scenario schedule
- normalized attempts, labels, clips, report, delivery, package, archive, and
  operator ledgers are present
- forbidden proof booleans remain false unless deterministic accepted evidence
  exists

The auditor writes `arena_package_proof_boundary_audit.json`.

`blueprint-smoke-arena-package-local --output-dir <dir>` is the one-command
local smoke for this lane. It creates synthetic capture/results fixtures, runs
the real ingest CLI path with a 500-scenario schedule, exercises
review-required vision labels, local delivery, fake local operators, and the
package audit, then writes `arena_fixture_smoke_manifest.json`. It is local
package-pipeline proof only; it does not prove WebApp upstream truth or
owner-system Isaac Lab-Arena execution.

## Live Gates

Real external actions remain blocked unless explicit owner gates are supplied:

- real simulator/provider execution:
  `BLUEPRINT_ALLOW_SIMULATOR_EXECUTION=true`,
  `--allow-simulator-execution`, `--allow-simulator <framework>`, and an
  explicit `--simulator-command`
- rollout vision labeling:
  `BLUEPRINT_ALLOW_ROLLOUT_VISION_LABELING=true`,
  `--allow-rollout-vision-labeling`, and `--vision-labeling-command`
  (`blueprint-label-rollout-vision-openai --output-dir .` is the built-in
  OpenAI command hook)
- package upload/signed access:
  `BLUEPRINT_ALLOW_PACKAGE_DELIVERY_UPLOAD=true`,
  `--allow-delivery-upload`, and `--delivery-command`
  (`blueprint-deliver-arena-package-local --output-dir .` is the built-in local
  filesystem delivery hook; cloud signed URLs still require a provider command)
- live Agents SDK operator:
  `BLUEPRINT_ALLOW_LIVE_AGENTS_SDK_OPERATORS=true`, credentials/dependency
  availability, and the CLI allow flag
- live Codex SDK operator:
  `BLUEPRINT_ALLOW_LIVE_CODEX_SDK_OPERATORS=true`, SDK availability, and the CLI
  allow flag
- local fake operator:
  `BLUEPRINT_ALLOW_FAKE_LIVE_OPERATORS=true` and `--operator-mode fake`
- live setup preflight:
  `blueprint-audit-live-pipeline-setup` checks env gates, command hooks,
  package audit status, WebApp upstream IDs, SDK availability, and optional
  control-plane droplet metadata without printing secret values or running
  provider jobs
- always-on control plane:
  `blueprint-run-live-pipeline-control-plane` performs the same setup audit and
  optionally drains a `robot_eval_job_request.v1` inbox on the DigitalOcean
  droplet. It exits cleanly when blocked and records missing live commands,
  capture roots, inboxes, or owner proof as manifest blockers rather than
  treating the droplet itself as simulator proof.

ChatGPT Pro/Codex OAuth may be used by the host application when a host-managed
tool is triggered. The repo can also use an installed authenticated `codex` CLI
when `BLUEPRINT_ALLOW_CODEX_CLI_HOST_OAUTH=true` and the matching live Codex
operator gate are set. Repo subprocesses that call OpenAI SDKs directly still
require `OPENAI_API_KEY` or a configured command hook that owns its own OAuth
flow.

Rollout vision labeling is provider-replaceable behind
`BLUEPRINT_ROLLOUT_VISION_LABELING_COMMAND`. The repo includes
`blueprint-label-rollout-vision-openai`, which uses `OPENAI_API_KEY`, extracted
keyframes, and the OpenAI Responses API to write
`rollout_vision_labels.command.json`. Gemini/Google GenAI remains possible
through a wrapper using `GEMINI_API_KEY` or `GOOGLE_GENAI_API_KEY`. The model
backend is not a proof source and labels remain review-required until accepted.

Package delivery is command-backed. `blueprint-deliver-arena-package-local`
copies `delivery_bundle/` into a local delivery root and writes
`delivery_upload.command.json` with local access paths. It does not perform
cloud upload, create signed URLs, verify entitlement, or upgrade proof claims.

## External Blockers

No repo-local blocker remains for the deterministic package workflow.

External/live proof still requires owner input:

- an owner Arena/GPU result directory or owner-system execution proof if the
  user wants simulator execution proof rather than local result ingest
- live vision-labeling command and env gate if model-derived rollout labels are
  required
- live storage/upload command and entitlement context if signed delivery URLs are
  required
- capture-root and WebApp job-request inbox paths if the always-on control plane
  should process production requests instead of only writing blocked/noop
  readiness manifests
- OpenAI SDK dependencies/credentials and explicit env/CLI gates if real
  Agents SDK or Codex SDK execution is required

Until those inputs exist, the correct state is local package proof complete and
live external proof blocked by policy.
