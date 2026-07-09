# CI Required Checks

Status: required merge and deploy contract for the Pipeline repo.

Branch protection for `main` must require these Pipeline checks before merge:

| Check | Workflow | Why it is required |
| --- | --- | --- |
| `CI / test` | `.github/workflows/ci.yml` | Fast unit/contract lane. |
| `Full Test Lane / Full pytest lane on CPU runner` | `.github/workflows/full-test-lane.yml` | Slow, integration, subprocess, render, module-entrypoint, and provider-adapter coverage via `scripts/pytest_full.sh`. |

Canonical Pipeline launch evidence uses Python `3.12`. The source of truth is
`docs/CI_PYTHON_INTERPRETER_MATRIX.json`, documented in
`docs/CI_PYTHON_INTERPRETER_MATRIX.md`, and enforced by
`scripts/validate_python_interpreter_matrix.py`. Python `3.10` and `3.11`
remain package-supported compatibility interpreters, but their local output is
not canonical launch/deploy proof. Python `3.13` output is non-canonical and
must be rerun under Python `3.12` before it can support launch claims.

Deploying a Pipeline commit requires `Full Test Lane / Full pytest lane on CPU
runner` to have passed for that exact commit SHA, or a fresh manual
`workflow_dispatch` full-lane run on that SHA before deploy. `deploy/scripts/deploy.sh`
enforces this with:

```bash
FULL_TEST_LANE_COMMIT="$(git rev-parse HEAD)" \
FULL_TEST_LANE_EVIDENCE_URI="https://github.com/.../actions/runs/..." \
  deploy/scripts/deploy.sh
```

The weekly scheduled run is supplementary health evidence only; it is not deploy
proof for a different commit. Emergency rollback remains available through
`deploy/scripts/deploy.sh --rollback --rollback-image-tag <tag>` and has its own
rollback verification/health checks.

Release images must be versioned or digest pinned. `deploy/scripts/deploy.sh`
defaults `IMAGE_TAG` to the current git SHA prefix, refuses `latest`/`dev`/`test`
/`local`, resolves pushed images to registry digests when GCR returns them, and
writes `output/deployments/pipeline-deployment-manifest.json` with:

- `release_id`, `git_sha`, and `image_tag`;
- pipeline, SAM3, VIP, DeepPrivacy2, and video-to-world image refs;
- the exact Full Test Lane commit/evidence URI used for the deploy gate;
- a rollback command template for that release tag.

Terraform image variables are required inputs and reject `:latest`. Use the
manifest digests when available; otherwise use the git-SHA release tag from the
same deploy.

Release packets and deploy evidence must also follow
`docs/SECRET_ARTIFACT_DISCLOSURE_POLICY.md`: raw secret values, secret hashes,
and absolute local credential-file paths are not publishable evidence. Readiness
artifacts may record credential presence, mode, source, and redaction metadata
only.
