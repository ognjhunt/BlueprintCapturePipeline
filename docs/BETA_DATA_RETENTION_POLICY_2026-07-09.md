# Beta Data Retention Policy

Schema: `blueprint.beta_data_retention_policy.v1`
Status: `declared_validator_enforced_operator_signoff_required`

This is the machine-checked beta retention policy for the 100-external-user
launch model. The canonical JSON artifact is
`docs/beta_data_retention_policy_2026-07-09.json`; the validator is
`python scripts/validate_beta_capacity_storage.py`.

## Collection-Terms Clarification — October 16, 2026

This additive clarification governs the historical storage defaults below for
the beta recording/consent journey (L2). The defaults describe storage classes
and maintenance, not permission to retain every recording indefinitely, extend
its original terms, or train a model on it. Original collection terms and
verified legal holds or rights/privacy decisions supersede internal archive
defaults. A later account acceptance does not extend existing recordings.

Before changing a recording's retention period, verify its collection-time
terms and obtain the lawful review and applicable notice/authorization. The
previous website policy's 180 days after job closure remains applicable to
recordings collected under it, subject to verified legal holds. Neither a new
policy's publication/effective date nor storage configuration supplies that
authority. Unknown collection terms remain unknown until verified; no longer
period is authorized by this artifact. Training needs separate authorization.

Preserve `website_capture_retention_review.v1` metadata unmodified with the
original manifest and rights record, including first receipt, unknown collection
policy, review requirements and the absence of extension authority. Projecting
or archiving a capture must not replace those fields with the internal default.
This clarification changes no bucket rule, deletes no recording, and proves
neither counsel approval nor notice delivery or deletion execution. Existing
rights/privacy and legal workflows continue to govern those actions. The
machine-readable counterpart is `retention_authority` in the JSON policy.

## Policy Classes

| Data class | Scope | Retention |
| --- | --- | --- |
| Raw capture truth | `scenes/`, `targets/` | Nearline after 30 days, Coldline after 90 days, Archive after 365 days. Never lifecycle-deleted; deletion happens only through rights/privacy or legal workflows. |
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
