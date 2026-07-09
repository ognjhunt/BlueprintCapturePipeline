# Beta Data Retention Policy

Schema: `blueprint.beta_data_retention_policy.v1`
Status: `declared_validator_enforced_operator_signoff_required`

This is the machine-checked beta retention policy for the 100-external-user
launch model. The canonical JSON artifact is
`docs/beta_data_retention_policy_2026-07-09.json`; the validator is
`python scripts/validate_beta_capacity_storage.py`.

## Policy Classes

| Data class | Scope | Retention |
| --- | --- | --- |
| Raw capture truth | `scenes/`, `targets/` | Nearline after 30 days, Coldline after 90 days, delete after 180 days unless legal hold or a contract-specific hold overrides. |
| Temporary processing | `tmp/`, `staging/`, `debug/` | Delete after 14 days unless legal hold overrides. |
| Buyer/eval/hosted artifacts | `buyer_delivery/`, `marketplace/`, `hosted_sessions/`, `robot_eval_jobs/` | Delete after 365 days unless contract-specific retention hold overrides. |
| Local output snapshots | `output/` | Canonical launch evidence 365 days, CI/capacity artifacts 90 days, provider/runtime or paid-run snapshots 30 days, local preflight/dry-run snapshots 14 days. |
| Local `robot_eval_jobs/` cache | repo-root `robot_eval_jobs/` | Delete after 30 days by default, review at 25 GiB, hard stop at 50 GiB. |

## Support Operations

Support tickets for retention-sensitive beta issues must identify the
`capture_id` or `capture_job_id`, affected data class, artifact URI or manifest
path, requested action, privacy or rights sensitivity, legal-hold status, and
operator decision owner. The default support evidence window is 90 days.

Deletion requests route through rights/privacy ops review before a storage
action. Legal hold overrides normal retention, and raw capture truth deletion
requires provenance and rights review.

## Claim Boundary

This artifact is not signed DPA or access-audit proof, not live bucket apply proof,
not user-deletion execution proof, not backup/restore drill proof, and not live
support-ticket SLA proof. The manual legal/privacy evidence id remains
`operator_dpa_data_processing_terms`.
