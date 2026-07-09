# Output Artifact Retention

Status: local operator policy for `BlueprintCapturePipeline/output/`.

`output/` contains point-in-time local proof snapshots, provider/runtime run
data, bundles, preflights, and reusable asset caches. It is not a standing
source of truth. Before handing evidence to an operator, select canonical
artifacts and prune old local run data.

Use the dry-run inventory first:

```bash
python scripts/manage_output_artifact_retention.py
```

This writes `output/output_artifact_retention_manifest.json` with:

- top-level size inventory;
- canonical launch/CI/paid-gate artifact selection;
- superseded canonical artifacts;
- prune candidates by retention class.

Deletion is opt-in:

```bash
python scripts/manage_output_artifact_retention.py \
  --execute \
  --acknowledge-delete-output-artifacts delete-output-artifacts
```

Retention defaults:

| Class | Default | Notes |
| --- | ---: | --- |
| canonical launch evidence | 365 days | Current handoff artifacts are selected by canonical key and protected from pruning. |
| external asset cache | no automatic delete | Reusable local meshes/models may be expensive to reacquire. |
| provider/runtime or paid run | 30 days | Bundles, object-store staging, and provider output are local snapshots, not current proof. |
| local preflight or dry run | 14 days | Dry renders, bootstraps, canaries, and no-spend smoke artifacts are reproducible. |
| CI/capacity artifact | 90 days | Keep enough runway for launch review and audits. |
| uncategorized output | 30 days | Review before delete; classification should be improved for repeated paths. |

The script does not automate legal deletion, raw-capture truth retention, or
live bucket lifecycle policy. If an artifact is under legal hold, move or copy it
to the legal-hold location before pruning local `output/`.

The beta-wide retention policy is the checksum-backed artifact
`docs/beta_data_retention_policy_2026-07-09.json`
(`blueprint.beta_data_retention_policy.v1`). It is verified by
`scripts/validate_beta_capacity_storage.py` and included in the launch readiness
packet as `beta_data_retention_policy_json`. That policy is still not signed DPA,
live bucket apply proof, or user-deletion execution proof.

The same dry-run-first tool is used for repo-root robot eval job cache:

```bash
python scripts/manage_output_artifact_retention.py \
  --output-root robot_eval_jobs \
  --manifest-path output/robot_eval_jobs_retention_manifest.json
```

For 100 beta testers, `docs/beta_capacity_cost_storage_model_2026-07-08.json`
models 75 robot-eval jobs per month, a 25 GiB local review threshold, and a 50
GiB local hard stop. Repo-root `robot_eval_jobs/` entries are not launch proof
unless selected into the current launch readiness packet or copied into a
current operator evidence bundle.
