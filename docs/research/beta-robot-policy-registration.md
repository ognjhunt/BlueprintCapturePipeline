# Registering a prepared robot and policy for automatic screening

The default catalog considers 84 named candidates: 74 deployment robot–policy
route combinations across 16 robot configurations, plus 10 existing policy
candidates. The local pi0.5 DROID, GR00T N1.7 DROID and Cosmos Edge DROID
paths are enabled; actual scene compatibility and budget still determine which
ones execute. Cosmos requires the third, real `exterior_right` policy camera.
Missing cameras produce an individual reason while other candidates continue.

Robot source preparation and imported USD are different states. A reusable
registration needs the imported model and its actual manifest, an exact served
policy identity, scoped run rights, private credential delivery and bounded
compute/API charges. A URDF download alone does not satisfy those inputs.

The deployment catalog's nine route definitions describe planning possibilities.
Each robot–route pair retains its own result or reason; a shared robot does not
collapse different policy routes into one result. The public source preparation
layer has 18 runtime profiles, covering 14 of the 16 configurations. The four
additional profiles are UR10e, FANUC CRX-10iA (without the /L suffix), single
Kinova Gen3 and single xArm7. Standard Bots RO1/Core still lack a complete,
licensed source closure. These counts do not establish imported models,
compatible trained policies or measured task results.

## Import public robot assets on the production worker

The `public-embodiment-preparation` operator-door request prepares the configured
public bodies on the existing Isaac GPU infrastructure. It accepts a sealed,
explicit batch grant with ordered embodiment IDs, the installed release commit,
one stable authorization ID and a total spend ceiling of at most $4. The existing
paid allocator prices compute, setup, transfers and storage before allocation.
See [the operator-door contract](../OPERATOR_DOOR.md) for the request fields.

```bash
python3 scripts/operator_door.py public-embodiment-preparation --request-file grant.json --wait
python3 scripts/operator_door.py journal blueprint-embodiment-preparation.service -n 80
python3 scripts/operator_door.py cat /var/lib/blueprint/pipeline-control-plane/embodiment-preparation/latest.json
```

The first command confirms that the grant was queued. After execution and
verification, the batch receipt is
`/var/lib/blueprint/robot-models/batches/<authorization_id>.json`. Preparation or
allocation failures remain in `latest.json`. Each requested body has a retained
import or its own failure, source, time, output or allocation reason. Bulk
preparation uses `/mnt/blueprint-work/embodiment-preparation` with
disk reservations; verified models are published under the operator model store.
Keep the same authorization ID when recovering a lost response. A restarted
controller reconciles retained execution state without allocating another GPU.

For a retained body, the receipt supplies `binding_inputs_path`, its byte digest
and the prepared asset directory. Load those exact bytes into the registration
helper below and provide an actual matching policy identity and rights record.
The internal model URL identifies the verified operator-owned asset; the
screening bundle stages its dependency closure directly.

An asset import establishes simulator model preparation. A separate native
controller diagnostic records bounded reset and arm/gripper motion with gravity
and self-collisions disabled. Its failure or timeout retains the verified import
and a diagnostic reason. A policy registration adds a compatible executable
policy. A scored simulation adds measured task results. Preserve these states
separately in agent answers and candidate results.

## One registration, every compatible task

`deployment_embodiment_runtime_configuration.bind_prepared_configuration()`
verifies the source configuration, imported USD and dependency closure and
returns `embodiment_runtime_binding_inputs.v1`. Connect that result to screening:

```python
from blueprint_pipeline.beta_policy_screening_team import registration_from_prepared_inputs

registration = registration_from_prepared_inputs(
    prepared=prepared_binding_inputs,
    model_path=imported_usd_path,
    destination=operator_manifest_directory,
    candidate_id=candidate_id,
    team_id=team_id,
    checkpoint_id=checkpoint_id,
    policy_spec=served_policy_spec,
    rights=actual_scoped_rights_record,
    display_name=display_name,
)
```

The rights record uses the existing `blueprint.team_policy_candidate_rights.v1`
contract. Its sealed digest, candidate, team and checkpoint scope must match;
`license_to_run_granted` must be true. The helper writes a nonsensitive manifest
and returns artifact references. It does not grant rights or make provider calls.

Install the returned registration in the existing **operator-owned** team
registry under `screening_candidates[candidate_id]`, either on one profile or in
`derived_profiles.defaults` to apply to subsequently compiled task scenes:

```python
screening_candidates[candidate_id] = {
    "robot_policy": registration,
    "credential_file": "/protected/operator/policy-bearer",
    "credential_origin": "https://your-policy-service.example",
    "provider_hard_cap_usd": "4.00",
    "hosted_inference_payment": "caller_paid",
    "hosted_inference_maximum_usd": "0.00",
    "hosted_max_cost_per_request_usd": "0",
}
```

Those billing values describe an account whose caller has separately authorized
the API charges. For Blueprint-paid inference, set `hosted_inference_payment`
to `blueprint`, supply a positive maximum price per HTTP request and an inclusive
API maximum. The planner verifies the worst-case request count and reserves
compute plus inference within the unchanged **$15 per-site allowance**. Unknown
prices do not produce a runnable quote. The default compute envelope remains
four hours/$4; operator overrides must still cover hourly rate × hard TTL and
the allocator's transfer/storage/teardown admission.

Credential contents belong in the existing private delivery lane, never in
registry JSON, catalog JSON, argv, model files or bundles. Files must meet the
existing ownership and private-mode checks. `credential_origin` is exact. The
provider bootstrap delivers the token under `BLUEPRINT_TEAM_POLICY_BEARER_FILE`.

`robot_policy` contains these reusable fields:

| Field | Meaning |
|---|---|
| `manifest`, `model` | `{path, sha256, size_bytes}` references to actual imported files |
| `team_id`, `checkpoint_id` | Scope of the supplied rights record |
| `rights_binding` | Full sealed existing team-policy rights record |
| `policy_spec` | Existing `TeamPolicySpec` or `DreamLabsPolicySpec` fields |
| `dependency_closure`, `source_import_provenance` | Import receipts copied from prepared output |
| `display_name` | Name presented for this configuration |
| `policy_revision` | Optional precise served revision; otherwise exact model/policy ID |
| `minimum_gpu_memory_gb` | Optional deployment requirement; default 46 GB; canonical floor is never lowered |

For `blueprint_policy_https.v1`, the policy spec names the exact endpoint and
metadata URLs, allowed origin, model identity digest, embodiment ID, ordered
camera names, action chunk size and execution horizon. The service implements
`TeamPolicySpec.expected_metadata()` on GET and the existing named camera/state
request and joint/gripper action protocol on POST. Hosting ACT/OpenPI/GR00T behind
this protocol still requires a checkpoint trained for that robot's observations
and actions; it does not make a DROID checkpoint compatible with another arm.

The worker derives the effective catalog from valid registrations before naming
the campaign. New model/rights/registration revisions, credential readiness or
credential-path changes create a new plan identity. Prior spending remains in
the site's cumulative ledger. It then builds the scene-specific robot placement,
camera rig, instruction, action/query limits and sealed execution spec itself.
No per-task execution JSON needs to be hand-authored.

## Dream-specific inputs

Use `dreamlabs_dream1.v1` with the exact official inference and metadata URLs in
`dreamlabs_policy_client.py`. Add the exact account model record as
`provider_model_metadata` and the real scoped `provider_disclosure` receipt to
the registration helper. Discovery binds provider metadata, not unseen weights.
The adapter requires the supported two-arm/two-gripper/three-camera contract,
60 Hz native control and 20 Hz pose targets. It disables extra SDK recording.

The existing disclosure contract includes the actual scene revision, candidate,
team, checkpoint, exact model ID, accepted retention/training terms and authorized
team-account billing. Reuse the applicable authorized receipt for each scene;
do not manufacture one from a catalog entry. This path currently accepts
caller-paid Dream usage, not a new Blueprint-paid Dream billing arrangement.
Saving `DREAM_API_KEY` in Codex's vault must also result in its availability in
the execution environment and the production private delivery lane.

Dream's published body demonstrations map to dual YAM, PiPER, UR5e, FR3, Gen3 and
xArm7. Axol and TRLC retain independent controller candidates; the catalog does
not claim Dream publisher compatibility for them. The public dual UR5e profile
now composes two actuated Robotiq 2F-85 tools from pinned sources; its generic
simulation assembly does not establish Dream's exact tools, calibration or
action contract. Bare single-arm profiles still need task-appropriate tooling.
Private GMR-AI/LUNA/other vendor policies remain explicit
unavailable candidates until access and a matching adapter exist.

## Verification and result semantics

Hermetic tests exercise 16 compatible registrations: 15 are selected under a
$15 fixture budget and the 16th retains its budget reason. One malformed model
does not suppress the others. Generic robot and Dream registrations both reach
actual sealed provider bundles with their verified model bytes and no secrets.
These are integration tests, not measured policy success or production Isaac
execution. Every candidate retains a result or reason, and simulation outcomes
remain bounded estimates.
