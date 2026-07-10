# Deployment provenance, state, and secret boundary

`deploy/scripts/deploy.sh` is the only supported production entrypoint and
Terraform is the sole topology owner. A production deployment is rejected
unless all of these checks pass:

- the checkout has no tracked or untracked changes and `HEAD` exactly equals
  freshly fetched `origin/main`;
- the supplied GitHub URL identifies this repository's canonical Full Test Lane
  main-push run at that exact SHA, with a successful canonical job and required
  collection/run/verification steps;
- the downloaded unexpired artifact has identical planned, executed, and JUnit
  node IDs in the same order, no duplicates or substitutions, and zero failures,
  errors, or skips; the `cpu_full.json` envelope and source hashes are recomputed;
- every release image resolves to an immutable registry digest, and any supplied
  digest exactly matches the tag readback;
- the remote GCS state bucket is US-hosted, uniform-access-only,
  public-access-prevention enforced, versioned, retained for at least 30 days,
  and protected by the configured US Cloud KMS key.

The GCS backend is mandatory and supplies remote state locking. Local state,
local `terraform.tfvars`, plan files, and overrides are ignored and are not an
approved production workflow.

Secret payloads are provisioned outside Terraform in Secret Manager. Deployment
configuration contains only these secret names:

- `PRIVACY_RUNNER_TOKEN_SECRET_NAME`
- `VIDEO_TO_WORLD_RUNNER_TOKEN_SECRET_NAME`
- `PIPELINE_SYNC_TOKEN_SECRET_NAME`
- `WORLDLABS_API_KEY_SECRET_NAME`
- optional `HUGGINGFACE_TOKEN_SECRET_NAME`

Terraform verifies that the named secrets exist, grants accessor rights only to
the service accounts that consume each secret, and configures Cloud Run
`secret_key_ref` entries. Raw token or API-key values must never be placed in
Terraform variables, tfvars, process arguments, deployment manifests, or state.

These controls prove configuration and provenance behavior. A successful live
apply, authenticated canary, Cloud KMS/Secret Manager audit readback, and remote
state-lock contention test remain release-bound external evidence.
