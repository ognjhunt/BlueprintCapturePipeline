Arm Decision Proof v1, ADP-009D / day 28: deploy the episode-compilation CPU
worker without restoring or adopting unrelated legacy infrastructure.

The existing project has broad Editor and TokenCreator grants. Editors can
delete isolation tags, and Google's deny-permission list does not support tag
key/value administration. The worker therefore lives in the Terraform-owned,
parentless `blueprint-remote-cpu-8c1ca` project, on the existing billing account.
Its authoritative project policy permits only the founder Owner and the managed
Cloud Run service agent. The dispatcher and worker have resource-scoped custom
roles. The managed agent cannot impersonate the dispatcher. The only grant
added in the original project is repository Reader for the new service agent
on the verified `us/gcr.io` Artifact Registry repository.

Use `deploy/scripts/deploy.sh --remote-cpu-only` from clean, promoted main.
The normal exact-commit Full Test Lane gate remains mandatory. Build and push
the pipeline image for that exact commit first. Supply the approved GCS state
bucket, canonical `capture-pipeline` prefix, CMEK key, and billing account through
the existing deployment settings. Authenticate as `ohstnhunt@gmail.com`;
project creation and billing-link permissions and quota are required.

This command uses the canonical `main.tf` and state, with a fixed target list.
The saved JSON plan must contain only the reviewed worker resources, the
bounded 4 CPU / 16 GiB / 1800 second / zero-retry job, the $25 budget, and the
specific IAM policies. Destructive or unrelated changes fail before apply.
The project's postcondition blocks dependents if an organization or folder
parent appears. A scoped provider refresh must have zero changes. The retained
state marker prevents a later full deployment until legacy topology adoption
is separately reviewed; scoped evidence never claims full-project adoption.

Infrastructure deployment alone does not enable execution. Before publishing
the host credential or transport descriptors, retain real checks proving:

- The new project has exactly the expected principals and no inherited parent.
- The service agent can use the worker identity and cannot impersonate the dispatcher.
- Worker/dispatcher transport access and job permissions match their custom roles.
- Non-owner identities cannot strip the identity tags or mint dispatcher keys.
- The source image and host config use the exact deployed release and immutable digest.

Install the root-owned dispatcher credential only after those checks pass.
Retain the standing $25 authority, B2 lifecycle readback, paid preflight result,
per-class shadow parity and terminal teardown evidence. Existing captures and
these infrastructure checks remain `development_only`.

Primary references: [deny permission support](https://docs.cloud.google.com/iam/docs/deny-permissions-support),
[service account tags](https://docs.cloud.google.com/iam/docs/service-accounts-tags),
[project creation](https://docs.cloud.google.com/resource-manager/docs/creating-managing-projects),
and [Cloud Run job image access](https://docs.cloud.google.com/run/docs/create-jobs).
