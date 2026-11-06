# ADP-009D / day 28: parentless Cloud Run bootstrap

Observed blocker: live apply rejects project-level `roles/iam.denyAdmin` with
400; this account has no organization. One worker project cannot protect a
co-located dispatcher from its managed Run service agent's token permission.

Reviewed architecture: retain the parentless worker project, and create the
parentless `blueprint-cpu-dispatch-8c1ca` project for host dispatch credentials
and transport. Worker project IAM is exactly Founder Owner and Run service
agent. Dispatch project IAM is exactly Founder Owner, with no Run API. Worker
remains in the runtime project; it receives only transport objects.get on the
new bucket. New Dispatcher receives the minimal custom job role on the stage
job and create/get/delete on transport. No project-wide token grants or
cross-project service-identity exemption are needed.

The managed runtime agent can impersonate Worker and therefore read inputs.
It is part of the trusted runtime boundary. It must not mutate transport,
impersonate the new Dispatcher, or execute jobs, directly or as Worker.

Preserve the keyless old Dispatcher through a Terraform moved block, its old
bucket, custom roles and tags. Revoke its bucket bindings, grant it no job
permissions, and verify it has no user keys. Tags describe retained resources;
they do not enforce this design. No resource destroy or data deletion is in
scope. Both parentless projects' whole costs share the single $25 budget;
remove the label filter so unlabeled transport/API charges are included.

Completion artifacts: exact merged release's canonical Full Test Lane;
immutable image revision; canonical-backend strict two-project plan and
zero-change scoped refresh; actual direct and impersonation-chain IAM refusal
receipts with temporary probe grants restored; root-protected host credential
and config readback; bounded paid preflight; three actual shadow parity passes
per eligible closure class and provider-zero settlement. Infrastructure apply
alone does not close the worker activation or concurrency proof.

Validation: three regression tests first failed on the single-project design.
The updated configuration validates with Terraform 1.14 / Google 5.45.2; a
read-only provider-refreshed plan preserves all existing resources and passes
scope validation. Focused contracts cover custom-role escalation, wrong
project, inherited parents, extra project/job/bucket authority, immutable image
revision, compute/spend limits and resource destroy refusal. SPEC and separate
QUALITY must review the exact code head before protected merge.
