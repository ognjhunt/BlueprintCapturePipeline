# CI Required Checks

Status: required merge and deploy contract for the Pipeline repo.

Branch protection for `main` requires one stable aggregate merge check:

| Check | Workflow | Why it is required |
| --- | --- | --- |
| `Impacted test and sentinel gate` | `.github/workflows/ci.yml` | Runs dependency-mapped tests plus always-on success-claim, runtime-security, paid-resource-admission, and release-policy sentinels. The ordinary lane is capped at ten minutes. If deterministic impact analysis classifies the change as cross-cutting, this aggregate waits for the reusable full lane and fails unless it passes. |

Other CI, CodeQL, compatibility, container, supply-chain, and simulator jobs are
hosted diagnostics or claim-specific release evidence. They remain visible and
must be green when their claim or promotion scope requires them, but they are
not blanket merge gates for every ordinary PR.

`python -m blueprint_pipeline.impacted_test_selection` is the shared selector
and runner. Its local build-loop default is a hard 120-second wall-time limit.
CI runs the same plan
with a 480-second test-process limit inside a ten-minute job limit, leaving time
for frozen dependency setup and artifact upload. Every plan records changed
files, selected tests, escalation reasons, and the protected risks. Unknown
executable surfaces, dependency changes, or an impact set exceeding the bounded
file/test budgets request the full lane rather than silently reducing coverage.

Canonical Pipeline launch evidence uses Python `3.12`. The source of truth is
`docs/CI_PYTHON_INTERPRETER_MATRIX.json`, documented in
`docs/CI_PYTHON_INTERPRETER_MATRIX.md`, and enforced by
`scripts/validate_python_interpreter_matrix.py`. The required compatibility
workflow installs the frozen lock and runs the declared bounded suite on Python
`3.10`, `3.11`, and `3.12`. Compatibility output from `3.10` and `3.11` is not
canonical launch/deploy proof. Python `3.13` output is non-canonical and must be
rerun under Python `3.12` before it can support launch claims.

A lock-only change triggers `CI`, `Sim-Only Local Gate`, and `Python
Compatibility`. The impact selector classifies `uv.lock` as cross-cutting, so
the CI aggregate invokes and waits for the reusable Full Test Lane. Every install
in those workflows uses `uv sync --frozen`.
`pyproject.toml` plus tracked `uv.lock` are the authoritative dependency graph;
`requirements.txt` and `requirements-geometry.txt` are hash-pinned compatibility
exports generated and checked by `scripts/verify_dependency_exports.py`. They
must not be edited as independent dependency declarations.

The resolver/build tool is also frozen: `pyproject.toml` requires uv `0.10.7`,
and every workflow installs that exact version through the commit-pinned
`setup-uv` action. A moving uv release is not allowed to rewrite or reinterpret
the release lock implicitly.

Critical capability lanes are defined in
`docs/critical_capability_lanes.json`. CPU and container evidence is mandatory
for every release scope. Native LeRobot round-trip evidence is mandatory for
the legacy `PTDP` evidence-export scope and SC3/paid/live scopes, the pinned GPU provider canary is mandatory for
SC3/paid/live scopes, and Pub/Sub emulator integration is mandatory for
paid/live scopes. A missing, skipped, wrong-SHA, or non-passing critical lane is
a blocker only for scopes that depend on it. The sim-only lane does not invent
a physical-robot requirement.

The self-hosted critical-capability workflow is intentionally not evidence by
its existence. Its Pub/Sub job must complete a loopback-emulator publish/pull/
ack round trip, its LeRobot job must use the installed native loader against a
real export, and its GPU lane must validate exact-SHA provider-canary evidence.
Queued, skipped, fixture-only, or policy-only results cannot support those
claims. Scope evaluation reopens bounded retained sources: the Pub/Sub round-
trip transcript, native LeRobot relative-file manifest, sanitized GPU source
bundle, container build/smoke sources, and CPU collection/JUnit files. Missing,
oversize, stale, digest-mismatched, or envelope-only evidence fails closed.

The canonical CPU full lane installs the frozen `groot-libero-cpu` extra so the
Torch OSCAR shim checks and native LeRobot load check execute rather than skip.
It emits `cpu_full.json`, binding the exact repository SHA, planned/executed
node-ID sequence and digest, per-test JUnit node-ID properties/outcomes, and
source-artifact digests. Any failure, error, skip, missing node-ID property,
duplicate node ID, or same-count testcase substitution makes the lane fail.

The GPU critical-lane dispatch runs the Unitree GR00T/SONIC Vast startup canary
itself. The requested exact `@sha256` image must equal the repository variable
`BLUEPRINT_GPU_CANARY_APPROVED_IMAGE_URI`; preflight caps hourly rate and total
spend at 1 USD and live duration at 30 minutes. Passing evidence additionally
requires bounded observed rate/runtime/cost, GPU and upload checks, the canary
marker, source-artifact digests, and zero continuing spend. This is exact-image
startup evidence only. It is not WAM/policy execution or SC3 rank-fidelity
evidence and does not close `EVID-03`.

Deploying a Pipeline commit requires `Full Test Lane / Full pytest lane on CPU
runner` to have passed in a fresh `workflow_dispatch` run with reason
`production_deployment_promotion` on that exact protected-main commit SHA.
The lane assigns whole test files deterministically across four independent
serial shards. Each shard retains its own collection, JUnit, and verification
receipt; the canonical job proves their disjoint union has no duplicates,
omissions, failures, errors, or skips before rebuilding the existing canonical
JUnit and `cpu_full.json` artifact. The deploy verifier revalidates every shard
and the aggregate rather than trusting job status alone. No shard uses xdist or
reduces the collected test surface. `deploy/scripts/deploy.sh` enforces this with:

```bash
FULL_TEST_LANE_COMMIT="$(git rev-parse HEAD)" \
FULL_TEST_LANE_EVIDENCE_URI="https://github.com/.../actions/runs/..." \
  deploy/scripts/deploy.sh
```

The nightly scheduled run is supplementary health evidence only; it is not deploy
proof for a different commit. Emergency rollback remains available through
`deploy/scripts/deploy.sh --rollback --rollback-image-tag <tag>` and has its own
rollback verification/health checks.

Release build tags are SHA-derived and deployed images are digest pinned. `deploy/scripts/deploy.sh`
defaults `IMAGE_TAG` to the current git SHA prefix, refuses `latest`/`dev`/`test`
/`local`, resolves pushed images to registry digests when GCR returns them, and
writes `output/deployments/pipeline-deployment-manifest.json` with:

- `release_id`, `git_sha`, and `image_tag`;
- pipeline, SAM3, VIP, DeepPrivacy2, and video-to-world image refs;
- the exact Full Test Lane commit/evidence URI used for the deploy gate;
- a rollback command template for that release tag.

Terraform image variables require immutable `@sha256` digests. The deploy script
resolves every exact-SHA build tag through the registry and rejects supplied
digest mismatches before Terraform can plan.

The deploy gate also rejects dirty or untracked checkout state and requires
exact `origin/main` parity. It queries the canonical GitHub run, binds the
repository/workflow/main-push/SHA/conclusion/job/steps, downloads the exact
unexpired full-lane artifact, and verifies planned versus executed node IDs and
green zero-skip JUnit outcomes plus the recomputed `cpu_full.json` envelope. A
URL or operator assertion alone is never deploy evidence;
there is no text-only bypass.

GitHub artifacts are 90-day transport copies, not the durable audit archive.
For a release, `Release Evidence Retention / Immutable BASE release evidence
archive` must archive the exact-SHA evidence bundle to versioned S3 Object Lock
`COMPLIANCE` storage, retain it for at least 2,555 days, and verify version,
checksum, size, metadata, retention, and a full version-specific GET digest
readback. External artifact-signature and immutable-retention envelopes are
required even for `BASE`; the immutable receipt is release evidence, while the
workflow YAML and an unexecuted bundle are not. See
`docs/RELEASE_SUPPLY_CHAIN_AND_RETENTION.md`.

Dependabot configuration and pinned CodeQL workflow files are committed, but
repository vulnerability alerts, branch protection, successful CodeQL uploads,
keyless attestations, container signatures, immutable archive receipts, and
deployed-digest signature/SBOM readback remain external until verified on the
actual repository/release.

Release packets and deploy evidence must also follow
`docs/SECRET_ARTIFACT_DISCLOSURE_POLICY.md`: raw secret values, secret hashes,
and absolute local credential-file paths are not publishable evidence. Readiness
artifacts may record credential presence, mode, source, and redaction metadata
only.
