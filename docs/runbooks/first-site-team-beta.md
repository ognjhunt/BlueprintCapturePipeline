# First site and robot team: supervised beta acceptance

This is the operator route for beta intake, preparation, team setup, evaluation,
results and recommendation. The free beta and Arm Decision Proof v1 are both
active. Inviting a site/team does not require completed ADP physical validation;
a site-specific recommendation still requires its own qualified simulation
evidence and a passing completed team run. Existing development captures and
diagnostics cannot supply that evidence.

## Select the first task and its evidence

Use the normal website configuration route for one rigid object's
`pick_and_place` between regions of the same support surface. The downstream
team packet also reads `planar_push`, but the website submission route currently
admits only `pick_and_place`. A separate destination body, articulated object,
cloth or insertion is not silently converted into this task. Broad robot-form
applications and illustrations do not establish task/runtime compatibility.

Retain the site's approved instruction, start/goal/reset distribution, outcome
criteria and relevant timing requirement. Bind these to the capture, scene
revision, task specification and success contract. A compatibility-default
success contract is not evidence that the site approved its thresholds.

Follow [site-task qualification](../beta-site-task-simulation-qualification.md)
for independent and simulator observations of scale, registration, collision,
task state, policy observations and task dynamics. Freeze task-specific sample
counts and max/RMS tolerances before review. For the first task, identify the
measurement method for task-critical dimensions/landmarks, support/contact
geometry, camera calibration, and relevant object mass/contact response. A
numerically valid receipt does not make an unsuitable measurement method valid.
The [fidelity-profile templates](../beta-site-task-simulation-qualification.md#site-measurements-and-fidelity-profile-templates)
name each check's site and simulator reading. Site readings cover dimensions,
support heights, clearances and mass or pull force; they do not calibrate a
camera. The reviewer must be independent of both measurement producers.

The robot-free probe must pass for the exact scene revision before allocation.
It checks scorer/state discrimination with a virtual carrier; it does not
establish that a robot can grasp the object. Before relying on a customer score,
inspect a successful robot-actuated baseline and intentional failure examples
on the exact task, with state traces and media that agree with the independent
outcome check. Preserve strict-control requirements when a contract asks for
them; the ordinary team packet must not waive them to become eligible.

## Compile the scene's team half

No operator command is needed per scene. When a website scene's preparation
completes and its robot-free readiness receipt is retained, scene progression
queues one request under
`/var/lib/blueprint/pipeline-control-plane/team-scene-packets/pending`, and
`blueprint-team-scene-packet.path` starts the no-spend worker
(`python -m blueprint_pipeline.team_scene_packet_worker`). It compiles the
scene half on the CPU, writes the capture's derived row with scenario
`nominal`, and republishes the job's offer. An operator row for the capture,
or a derived row holding a frozen internal pair, always wins and nothing is
compiled.

The worker stays off, with the typed reason
`team_scene_packet_inputs_not_configured`, until the protected registry's
`derived_profiles` names the inputs only the operator fixes. Refresh the
registry digest after adding them (the paths are illustrations):

```json
{
  "packet_inputs": {
    "droid_profile_path": "/etc/blueprint/team-scene/droid_profile.json",
    "droid_profile_digest": "sha256:<64 lowercase hexadecimal characters>",
    "runtime_source_receipt_path": "/etc/blueprint/team-scene/runtime_source_packet.json",
    "output_root": "/var/lib/blueprint/task-evaluation-inputs/team-scene-packets"
  }
}
```

Both files must be operator-owned and not group- or world-writable. Each
request ends in `team-scene-packets/status/<key>.json`: `completed`, `skipped`
with a reason, or `blocked` with typed blockers. After fixing the cause, run
`python -m blueprint_pipeline.team_scene_packet_worker --queue-root
/var/lib/blueprint/pipeline-control-plane/team-scene-packets --requeue-unfinished`
as the service account; the path unit then runs those requests again. A
retry reuses a scene half already recorded at the same revision. Scenes that
completed before this worker was deployed queued nothing.

## Connect the team's own robot and policy

The current qualified route uses the team's declared, verified robot model.
The default DROID provider asset lacks retained model-byte proof here and stays
diagnostic. Applications may declare any form; admission checks the actual
fixed/free base, joints, cameras, model, scene and task.

Supply one self-contained flattened USD of at most 512 MiB, its SHA-256, declared
joint order/limits/reset/actuators, tool/contact links and one to four RGB cameras.
The supported control rates are 15, 30, 60 or 120 Hz. Use the existing
`blueprint_policy_https.v1` authenticated observation/action endpoint and the
credential flow; do not place tokens in robot manifests, model URLs, logs or
bundles. Registration of other delivery modes is not proof that their runtime
is admitted. Fixed-base reach screening does not prove a collision-free grasp.

A public HTTPS model URL remains supported. Proprietary models can instead use
an already approved private object in existing company-controlled GCS storage.
Keep the manifest's `model.url` a stable, credential-free locator:
`https://storage.googleapis.com/approved-bucket/team-models/robot.usd`.
This URL does not need public access when the operator pins the mapping below.
Do not publish a proprietary model or add signed-query credentials to make the
anonymous download path work.

In the existing protected `BLUEPRINT_TEAM_CANDIDATE_QUICK10_REGISTRY`, set
`private_robot_models` on the capture's explicit profile or operator
`derived_profiles.defaults`. Scene-derived rows cannot supply this authority.
The example is a schema illustration, not an admitted artifact:

```json
{
  "private_robot_models": {
    "canonical_uri_prefixes": ["gs://approved-bucket/team-models/"],
    "artifacts": [{
      "team_id": "exact-team-id",
      "checkpoint_id": "exact-checkpoint-id",
      "model_url": "https://storage.googleapis.com/approved-bucket/team-models/robot.usd",
      "sha256": "sha256:<64 lowercase hexadecimal characters>",
      "uri": "gs://approved-bucket/team-models/robot.usd",
      "generation": "123456789",
      "size_bytes": 123456
    }]
  }
}
```

Use the actual existing object generation, byte size and SHA; refresh the
registry's existing canonical digest. The team/checkpoint and URL/SHA must match
the request exactly. The control plane uses its existing ADC identity only for
that selected generation, checks remote size/generation, bounds the actual
download, then verifies SHA and the existing USD structure/stance contract.
Private-selection errors never fall back to an anonymous fetch. No customer
field can choose GCS access, request headers or a credential.

This code grants no storage rights and performs no upload or bucket mutation.
If the approved object or current read authorization is missing, preserve the
model locally and report that exact gap. Use the existing approved transfer and
rights process; do not create a bucket, widen IAM or transfer private bytes to
an unapproved destination. The team licence-to-run and provider-use rights
remain separate prerequisites.

The job retains `team_robot_model_delivery.json` with the selected canonical
URI, generation, size and hash, plus the verified model bytes and embodiment
binding. These are portable JSON/USD records in the existing job evidence
location. Recover with authorized generation-specific GCS access and recheck the
hash; a stored URI alone is not proof that remote recovery succeeded. No key or
token belongs in this receipt.

## Complete and inspect the result

Produce/publish the actual qualification receipt and freeze its complete
capture/scene/task/model/interface identity in WebApp admission. Admission and
closeout must revalidate the receipt, original bytes and expiry. Then execute
the team's own compatible policy under the existing internal spending,
deadline, hold, first-25 interface-failure and provider teardown controls. Free
site intake does not require a customer budget or per-run spending approval.

Retain all 200 planned-cell outcomes or honest partial progress and typed
refusal. Check media/state completeness, result delivery to the right accounts,
the reviewed recommendation, booking and direct site/team handoff. A successful
HTTP request, deployment or diagnostic run cannot replace this journey.

Reported cycle time is simulator task-state duration, including settling, for
successful episodes with complete clocks. Policy requests are awaited before
simulator stepping, so policy-service/network latency and rendering/persistence
wall time are excluded. Retain policy latency separately; simulator seconds
alone cannot establish physical throughput or a real-time controller.

Measure observed preparation/run wall time, storage peaks, retained provider
cost receipts, retries and operator intervention before increasing cohort size.
Reservation TTLs are safety bounds, not customer ETAs.

## Recover without changing the original evidence

Use [team execution recovery](../team-evaluation-recovery-contract.md) for
preflight repair, lost allocator receipts, cancellation, partial progress and
provider-zero. Observe a real stopped/partial outcome and its WebApp delivery
before claiming the recovery journey is proven.

For missing capture information, preserve the original capture and failure
reason, and give the site specific views/measurements to re-record. Supplemental
uploads are not an automatic continuation: the current owner observer admits
walkthroughs only, and the preparation watch alerts on supplemental uploads.
Do not relabel a supplement as the original walkthrough or edit finalized
source membership to force admission.

The supported supervised recovery route is a **new job/request through the
existing site intake**, followed by its new walkthrough. Each job's capture
token resolves its fixed `walkthrough-<requestId>` capture inside
`site-<requestId>`; there is no same-job replacement selector or supported
relinking operation. Preserve the original job and capture bytes. Support may
reference both job IDs in owner notes for context, but must not rewrite source
membership, copy old evidence joins or imply that the original job was resumed.

The new job requires fresh consent/finalization, ordinary owner observation,
and its own task packet, robot-free probe, measured qualification and execution
admission. An old receipt or recommendation cannot qualify the new capture or
scene revision. This route must be observed on the actual site account before
calling capture recovery complete.

## Reconstruction providers

Atlas is currently a pose/generation payload handoff, not a completed production
submit/poll/download adapter. Preserve its current refusal when final assets or
native geometry binding are missing. A new provider needs verified capability
data and exact returned image/depth/camera/asset provenance before admission;
do not invent an image limit or turn generated hidden surfaces into measured
site truth. Recruiting a compatible first task/team can proceed independently.
