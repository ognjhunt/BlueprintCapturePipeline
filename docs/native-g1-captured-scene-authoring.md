# G1 packet request from a captured task

The G1 development worker consumes the same native task packet format as the
Franka worker. The shared Task Evaluation Run configurator exports a
`task_evaluation_policy_pair_choice.v1.json` for the unavailable G1 pair.
`native_g1_scene_packet_request` connects that choice to one verified retained
scene packet. It does not enable the public G1 run profile.

The selected 841757 packet is newer than its published canary setup. For this
case, the packet itself can produce a planning-only setup and pair choice with
the same G1 catalog. This route carries the verified packet receipt identity;
it does not invent a configured offering or claim the pair is runnable:

```bash
PYTHONPATH=src python -m blueprint_pipeline.task_evaluation_packet_planning_setup \
  --source-packet /path/to/verified/packet-direct-policy-camera-v12 \
  --objective task_success \
  --output-dir /path/to/new/packet-planning
```

Use `--objective g1_navigation_goal` for the movement pair. Both resulting
files can be passed as `--setup` and `--choice` to the G1 packet builder below.
The source packet's declared task-success digest differs from its embedded,
owner-confirmed contract. The planning setup records both digests. G1 authoring
must use the embedded contract and its actual digest; the derived request
records the correction. Object, destination, scoring criteria, and other task
facts cannot change through this correction. It remains a development packet,
not a published Website setup or executable policy profile.

For the first book relocation rehearsal, the retained source is the
`packet-direct-policy-camera-v12` packet for scene `interiorgs-841757` and task
`scene-841757-book-to-marked-area`. The packet receipt digest is
`sha256:9b3e8ba34242eea5977c1fe709bf8351f62efae8da1a2fac5d2577c27182464a`.
Use the packet's own path on the machine running this command; the original
source files need not still exist.

The team supplies an authoring JSON with these fields:

- `schema_version`: `native_g1_scene_packet_authoring.v1`
- `claim_ceiling`: `development_only`
- `source_packet_receipt_digest`: the verified receipt digest
- `pair_choice_digest`: from the exported choice
- `physics_frequency_hz`: an explicit G1 physics cadence when the source
  cadence cannot evenly decimate to the policy control rate; for 841757, use
  200 Hz physics with 50 Hz G1 control (four physics steps per action). The
  request records both source and selected rates.
- `base_pose_world`: a reviewed G1 stance in the source scene's metric frame
- `task_hand`: `left` or `right`
- `cameras`: a 640 by 480 policy `head` camera parented to a rigid body in the
  official G1 USD, plus a world-frame review `overview` camera
- `task_spec`: the source task specification with G1 robot workspace, release,
  action, cadence, and episode bounds reviewed for the new embodiment. The
  object, target, destination, scoring contract, and other task facts must match
  the source packet exactly.
- `scenario`: an `evaluation_cell` context with `partition: development` and
  its sealed instance digest
- `authoring_digest`: canonical digest of the JSON excluding that field

For the first 841757 book manipulation rehearsal, the
[`g1_841757_book_manipulation_authoring.v1.json`](arm_decision_proof_v1/manifests/g1_841757_book_manipulation_authoring.v1.json)
file binds the retained packet and the two manipulation candidates. Its base
pose is a floor-clear **candidate** sampled by the static reach probe. Its head
and overview camera extrinsics are authored estimates; a live simulator frame
must confirm object visibility before this can be treated as a viable episode
setup. It uses 200 Hz physics and 50 Hz control. The task success contract and
book/target facts are copied from the retained packet, with only the recorded
declared-versus-embedded contract digest correction.

For a movement pair, the authored task specification also needs a
`g1_navigation_goal` in the same scene. The later pair-selection step requires
the human-confirmed navigation authority bound to the sealed G1 scene plan.

The stance, camera extrinsics, task parameters, and development scenario must
be authored for G1. Reusing the Franka wrist camera, robot workspace, or
qualification scenario would misdescribe the episode. The builder checks
binding, integral physics/control decimation, and camera shape but native collision, reach, visibility, and task
performance still require a G1 simulator run.

Before authoring a stance, run the static reach diagnostic against the retained
packet. State the scene's floor height explicitly:

```bash
PYTHONPATH=src python -m blueprint_pipeline.native_g1_scene_reach_probe \
  --source-packet /path/to/verified/packet-direct-policy-camera-v12 \
  --floor-z 0 \
  --output /path/to/new/g1-static-reach.json
```

For the verified 841757 v12 packet (receipt `9b3e8ba…`), a 5 cm grid over a
1.25 m radius found 346 floor-clear nominal standing poses. Its book and mark
are at 0.286 m, while the G1 profile's nominal shoulder is 1.08 m and arm span
is 0.45 m. Even with horizontal alignment, the shoulder would have to lower at
least 0.344 m to put the object within that nominal span. The nearest sampled
shoulder-to-subject-center and shoulder-to-mark horizontal distances were
0.718 m and 0.717 m respectively. A fixed nominal standing pose is therefore
not a credible starting assumption for this task. The policy must be allowed to
approach, lean, or crouch, and the resulting motion still needs an actual
simulator reach, contact, and task test. This diagnostic samples static boxes;
it is not an inverse-kinematics or policy-success result.

```bash
python -m blueprint_pipeline.native_g1_scene_packet_request \
  --source-packet /path/to/verified/packet-direct-policy-camera-v12 \
  --g1-usd /path/to/g1_29dof_with_hand_rev_1_0.usd \
  --setup /path/to/published/task_evaluation_policy_canary_setup.v1.json \
  --choice /path/to/task_evaluation_policy_pair_choice.v1.json \
  --authoring /path/to/native_g1_scene_packet_authoring.v1.json \
  --output-dir /path/to/new/g1-request \
  --materialize-packet
```

The output contains an evidence directory, a digest-bound
`native_task_arena_packet_request.v1.json`, and, with `--materialize-packet`, a
sealed G1 packet. On macOS, retained assets are cloned with separate inodes to
reduce disk use; other hosts copy them. The packet materializer can hard-link
within this single derived output directory. The pair selection step then binds that packet to
the same setup, choice, checkpoint inventory, candidate rights reviews, and
optional navigation authority. A successful request or packet build proves
neither checkpoint inference nor an executed episode.
