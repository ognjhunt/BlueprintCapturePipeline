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

## Live Gates

Real external actions remain blocked unless explicit owner gates are supplied:

- real simulator/provider execution:
  `BLUEPRINT_ALLOW_SIMULATOR_EXECUTION=true`,
  `--allow-simulator-execution`, `--allow-simulator <framework>`, and an
  explicit `--simulator-command`
- rollout vision labeling:
  `BLUEPRINT_ALLOW_ROLLOUT_VISION_LABELING=true`,
  `--allow-rollout-vision-labeling`, and `--vision-labeling-command`
- package upload/signed access:
  `BLUEPRINT_ALLOW_PACKAGE_DELIVERY_UPLOAD=true`,
  `--allow-delivery-upload`, and `--delivery-command`
- live Agents SDK operator:
  `BLUEPRINT_ALLOW_LIVE_AGENTS_SDK_OPERATORS=true`, credentials/dependency
  availability, and the CLI allow flag
- live Codex SDK operator:
  `BLUEPRINT_ALLOW_LIVE_CODEX_SDK_OPERATORS=true`, SDK availability, and the CLI
  allow flag
- local fake operator:
  `BLUEPRINT_ALLOW_FAKE_LIVE_OPERATORS=true` and `--operator-mode fake`

## External Blockers

No repo-local blocker remains for the deterministic package workflow.

External/live proof still requires owner input:

- an owner Arena/GPU result directory or owner-system execution proof if the
  user wants simulator execution proof rather than local result ingest
- live vision-labeling command and env gate if model-derived rollout labels are
  required
- live storage/upload command and entitlement context if signed delivery URLs are
  required
- OpenAI SDK dependencies/credentials and explicit env/CLI gates if real
  Agents SDK or Codex SDK execution is required

Until those inputs exist, the correct state is local package proof complete and
live external proof blocked by policy.
