# Policy candidate registry

ADP-050 (integrate the two frozen candidates). A run still compares exactly two
frozen candidates. What changes is where the two come from: any two runnable
entries in `src/blueprint_pipeline/task_evaluation_policy_candidate_registry.py`,
not one hardcoded pair.

## What the registry holds

Each entry pins a checkpoint (repository, revision, digest, size), its license
and whether that license allows a paid hosted service, the adapter and
observation/action schemas, the minimum GPU memory, and a status:

| Status | Meaning | Shown to a team |
| --- | --- | --- |
| `runnable` | Served by the canary runtime, commercial license, checkpoint verified from downloaded bytes, no open blockers | Selectable |
| `integration_pending` | License allows it; runtime work or the integration canary is outstanding | Listed, not selectable: "Coming soon" |
| `license_review_pending` | License does not clearly allow a paid service | Listed, not selectable: "pending license review" |

As of 2026-09-23:

| Candidate | License | Status | Why |
| --- | --- | --- | --- |
| `pi05_droid` | Apache-2.0 + Gemma terms | runnable | Admitted pair |
| `groot_n17_droid` | NVIDIA Open Model License | runnable | Admitted pair |
| `cosmos3_nano_policy_droid` | OpenMDW-1.1 | integration_pending | No serving path, three cameras, absolute joint action mapping, GPU memory, bytes not yet verified, no integration canary receipt |
| `molmoact2_droid` | Apache-2.0 with research-intent language | license_review_pending | Commercial use under review |
| `flux3_action_droid` | FLUX Kommunity License 1.0 | license_review_pending | Non-commercial except a revenue-capped outputs grant |

Pending checkpoints are pinned from the Hugging Face tree listing
(`hub_tree_listing`), not downloaded bytes. That is enough to name the exact
revision; it is not enough to run it.

## Gates

`registry_violations()` fails if any entry overstates what can run: a runnable
entry without the canary runtime, a commercial license, a checkpoint verified
from downloaded bytes, or with open blockers; an unavailable entry without a
reason; a license-blocked entry not marked as such; fewer than two runnable
entries. `validate_selected_pair()` admits two distinct runnable entries and
returns them in registry order, so the same two policies make the same run
whichever was picked first.

The public setup lists every entry. Unavailable ones carry
`readiness.status = "unavailable"`, a null receipt and the reason. Only
`verified_runnable` rows can be selected, bound to an owner pair, or handed off.

## Promoting an entry

An entry becomes `runnable` only by removing its blockers in code: runtime
serving path and adapter with hermetic tests, checkpoint bytes downloaded and
inventory-verified, and a passing paid integration canary receipt
(`policy_integration_canary`). The paid canary needs explicit approval before it
runs. A license change needs a written legal decision recorded in the entry.
