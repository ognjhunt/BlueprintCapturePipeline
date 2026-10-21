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

## Hosted API policies

`HOSTED_REGISTRY` lists policies that run behind a provider's API. Nothing is
downloaded, so an entry records the provider's model ID, whether the provider
promises that ID never changes (`version_frozen`), the credential variable it
needs, and the provider's terms and data-use status. Hosted entries are never
part of the two-candidate pair or the scene setup's checkpoint list. A hosted
entry may become `runnable` only with a hosted runtime adapter, reviewed
terms, a configured credential and no open blockers
(`hosted_registry_violations`). Sending site images to a provider also needs
the job's existing external-processing permission.

| Candidate | Provider | Status | Why |
| --- | --- | --- | --- |
| `dreamlabs_dream1` | Dream Labs | integration_pending | No API key yet (`DREAM_LABS_API_KEY`); terms and data use not published; request/response schema unverified; no hosted runtime adapter; no runnable two-arm, three-camera robot setup; no integration canary |

Robot setups a policy can run on are listed in
`task_evaluation_embodiment_registry.py`. Only the DROID Franka + Robotiq 2F-85
setup is `runnable`. G1 and Atlas are unavailable. The I2RT YAM and two-arm
Franka Research 3 declare Dream-1's interface but are not integrated. UR5e,
UR10e, FANUC CRX-10iA, ABB GoFa, KUKA LBR iiwa 14, Kinova Gen3 and xArm 7 are
listed as not integrated, each with its missing asset, controller, camera-rig,
reset and canary work. `job_capability_routing` matches policies to setups by
their declared camera and action interfaces. It reports which setup would
unlock which unavailable policy.

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
