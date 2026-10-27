# Capture-to-screening audit, 2026-10-27

Scope: beta capture preparation through automatic policy screening; ADP-009D's
day-28 rehearsal is the related scientific precursor. The owner's request to
support more sites, tasks, policies and embodiments authorizes implementation;
it does not establish their performance or change ADP's frozen proof protocol.

## Observations before repair

Production's operator-door snapshot at 18:20:36 UTC and current main both
identified `868df8929bf7003f59a36c0598e697ffde544dba`.

| Reported claim | Verified finding |
| --- | --- |
| Disk blocks every paid launch | Root had 9,979,891,712 free bytes and remained under pressure. The work volume had 52,509,503,488 free bytes. Intake's role-specific projection admitted its configured roles. Capacity incorrectly projected those same roles onto root as well. |
| Only two task types are accepted | Intake retains an open `task_family`. Native scoring and the probe gate currently support rigid pick/place and articulated open/close. Other accepted jobs retain a preparation gap. |
| Rigid tasks cannot reach screening | The recorder and scorer existed, but the capped allocator lane was unmerged and its compilation handoff was missing. |
| Every robot is a floor-mounted DROID Franka | The default enabled policy catalog contains two DROID candidates. `team_embodiment_runtime` separately supports admitted fixed or free bases, position/velocity joints, cameras, and typed arm controls. Stance construction can author elevated mounts. These are execution capabilities, not qualification for every policy or robot. |
| A top-down precheck limits rigid tasks | Confirmed in `website_rigid_feasibility`. This repair does not claim side-grasp or arbitrary-task qualification. |
| Containers, weak captures and unfamiliar sites work universally | Unproven. Destination relations still require their exact robot-free prerequisite; missing metric scale, coverage or task evidence cannot be inferred away. |
| Failed handoffs keep retrying | The source subscription had a five-delivery dead-letter policy, but the Pub/Sub service agent lacked both forwarding grants and the destination topic had no subscription retaining forwarded messages. |
| Atlas failures always resume | Reconstruction resume explicitly handles typed processing waits. It is not permission to retry an unknown provider outcome or to overwrite decided evidence. |
| No capture has ever completed without intervention | Not established by a repository or status snapshot. No new uninterrupted capture-to-screening acceptance run is claimed here. |

## Repairs

- Capacity alerts now resolve configured role targets by filesystem device,
  including bind mounts, while retaining root-utilization warnings and atomic
  admission guards.
- Deliveries beyond the configured limit retain a diagnostic without reopening
  provider processing or acknowledging away the broker's payload.
- Reused the existing capped rigid-probe implementation, repaired its current
  runtime import closure, and wired both new scene compilation and previously
  prepared packets to its queue. The deployed service defaults to automatic
  execution under the capture's current website grant, with an operator opt-out.
  A five-minute timer revisits deferrals; each pass starts at most one attempt
  per capture. Each attempt remains capped at $1, $0.80/hour and 3,600 seconds;
  the existing per-scene funded-attempt limit, independent watchdog, immutable
  bindings, decided-receipt preservation and provider-zero ingestion gate remain.

The following existing infrastructure declarations were applied directly and
read back, without a broad Terraform apply or a provider/GPU allocation:

1. Created `pipeline-trigger-dlq-retained` on `pipeline-trigger-dlq` with a
   600-second acknowledgement deadline, seven-day retention and no inactivity
   expiry, **before** enabling forwarding.
2. Granted the existing Pub/Sub service agent
   `service-744608654760@gcp-sa-pubsub.iam.gserviceaccount.com`
   `roles/pubsub.publisher` on that topic and `roles/pubsub.subscriber` on
   `blueprint-pipeline-handoff-listener`.

These match `deploy/terraform/main.tf` and the existing deployment script.
Configuration readback proves their installation; actual forwarding remains a
separate observation. No synthetic message was injected into a customer queue.

## Remaining acceptance

Source tests and deployment do not prove a real GPU run. A fresh, eligible
capture must still retain the probe trace, independently scored screening
episodes, complete policy-input media, result delivery and teardown. A supported
scored task, exact admitted policy/robot interface, adequate capture evidence,
current consent and bounded spending authority remain necessary. Expired grants
are not silently renewed, and founder holds are not released by these repairs.

Arbitrary embodiments, deformables, multistep tasks and every site are not
validated by this change. Each new family requires its own runnable task
construction, reset/scoring contract, robot-free controls and execution evidence.
