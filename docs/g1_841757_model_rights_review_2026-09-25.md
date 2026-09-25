# Scene 841757 G1 model-rights decision packet

**Status:** approved by Nijel Hunt for internal development simulation on
2026-09-25. The decision is recorded in [issue #2245](https://github.com/ognjhunt/BlueprintCapturePipeline/issues/2245#issuecomment-5830761422).
The four candidate-specific, digest-bound rights receipts are in
`docs/arm_decision_proof_v1/manifests/g1_841757_*_development_rights_review.v1.json`.
This decision is not GPU-spend authorization or permission to publish episode media.

## Blocker Title

Review the four pinned HumanoidArena G1 checkpoints and GEAR-SONIC assets for
internal development simulation on the scene 841757 book task.

## Blocker Id

`human-blocker:g1-841757-model-rights-20260925`

## Decision Context

ADP-050 Day 28 compatibility work verified the 841757 packet handoff. The G1
worker requires a human-reviewed rights receipt for **each** candidate before
a checkpoint can run. The same worker uses SONIC controller assets.
The repository validator checks the receipt's identity and digest; it cannot
decide whether the reviewer has authority or whether the publisher's and
upstream model terms cover Blueprint's use.

Nijel Hunt confirmed review and approval of the pinned checkpoint, inherited
π0.5 base-model, source, and SONIC terms for the four named candidates below.
No candidate was held. Each receipt binds that decision to its exact candidate,
inventory file, source revision, and manipulation or movement scene-plan digest.

## Original Review Guidance

An authorized reviewer should inspect the exact sources below and, **only if**
the HumanoidArena checkpoint terms, any inherited base-model terms, and the
SONIC terms permit this use, approve internal development simulation for the
four named candidates. If the π0.5 base-model provenance or terms cannot be
resolved from the released artifacts, hold those two candidates and ask the
publisher or counsel for clarification. Keep publication and redistribution
as separate decisions after real episode media exists.

## Alternatives

- Approve only the two diffusion candidates after review and hold both π0.5
  candidates. This permits single-candidate development checks, but not either
  selected two-policy comparison.
- Hold all model use pending publisher or counsel clarification.

## Downside / Risk

The ModelScope repository-level Apache 2.0 label may not fully describe rights
in every checkpoint or an inherited π0.5 base model. The released π0.5 config
has `license: null` and points to a publisher-local `pi05_base` path; that is
provenance evidence, not a grant of rights. SONIC weights have separate NVIDIA
Open Model License terms from its Apache 2.0 source code. A development-use
approval would not itself establish permission to publish G1 videos.

## Original Response Requested

Reply with **approve** or **hold** for development-only download and simulation
of these four candidates and the pinned SONIC encoder/decoder, naming the
authorized reviewer and whether the π0.5 inherited terms were reviewed:

1. `humanoidarena_dp_g1_dex3_sonic`
2. `humanoidarena_pi05_g1_dex3_sonic`
3. `humanoidarena_dp_g1_dex3_sonic_vision_navi`
4. `humanoidarena_pi05_g1_dex3_sonic_vision_navi`

An approval can name a subset. A reply that does not address inherited π0.5
terms leaves the π0.5 candidates on hold. No receipt may mark
`checkpoint_terms_reviewed` or `source_and_sonic_terms_reviewed` true until
that review actually occurs.

## Execution Owner After Reply

`pipeline-codex`, with the authorized rights reviewer owning the rights
decision. The durable review result belongs in this packet or an owning
issue/run artifact before worker staging.

## Immediate Next Action After Reply

For each approved candidate, record a
`native_g1_development_rights_review.v1` receipt bound to the reviewer,
candidate, source revision
`68479287a784a69be9ce6ad739311d2f11f75ef9`, inventory file SHA-256
`6be81cc07bd9c3a4c924c96c57bc23b1a8e518f7884cc30dcccfa1868aebe259`,
and exact derived scene-plan digest. Validate it with the no-spend G1 staging
path. Checkpoint download and a paid Linux GPU run remain separately gated by
storage, host, spend, and launch preflight.

## Deadline / Checkpoint

The reviewer decision is recorded above. Before starting the first G1 policy
episode, complete storage, host, spend, and runtime preflight.

## Evidence

- The [pinned HumanoidArena source license](https://github.com/William-wAng618/HumanoidArena/blob/68479287a784a69be9ce6ad739311d2f11f75ef9/LICENSE)
  is MIT (downloaded file SHA-256
  `be658ddc2a384d79c822a83f2b280feafcbaccd371e28094b9656cfc3b586ae0`).
  Its [README at that revision](https://github.com/William-wAng618/HumanoidArena/blob/68479287a784a69be9ce6ad739311d2f11f75ef9/README.md)
  directs readers to review upstream licenses and model/data artifact terms.
- The [publisher's ModelScope model repository](https://modelscope.cn/models/Twang2026/HumanoidArena_models)
  currently reports `License: Apache License 2.0` in its model API metadata.
  The repository is served from mutable `master`; the four exact file lists,
  sizes, and hashes are pinned in
  `configs/g1_humanoidarena_checkpoint_inventory.v1.json`. Its model card
  provides no detailed terms for the individual checkpoints.
- The released [π0.5 book-task config](https://modelscope.cn/models/Twang2026/HumanoidArena_models/resolve/master/pi/HOI_pp_box/pi05_sonic_ppbox_0529/100000/pretrained_model/config.json)
  lists `license: null` and a publisher-local `pi05_base` pretrained path. The
  diffusion [book-task config](https://modelscope.cn/models/Twang2026/HumanoidArena_models/resolve/master/small/HOI_pp_box/diffusion_sonic_ppbox_0529/pretrained_model/config.json)
  also lists `license: null`. These config fields do not override the repository
  metadata; they explain why inherited terms need explicit review.
- The [pinned GEAR-SONIC license](https://huggingface.co/nvidia/GEAR-SONIC/blob/6733128a3d8a523b1418b06bca3cdf61c8b0987f/LICENSE)
  separates Apache 2.0 source code from NVIDIA Open Model License weights
  (downloaded license SHA-256
  `24ab66be50d1aca4fc5e029ef76ce4ceaac6557ea21665caf4b140695a76ffee`).
  The two ONNX files and their hashes are pinned in
  `configs/g1_sonic_default_asset_inventory.v1.json`.
- The exact 841757 book-task handoff verified as `verified_not_executed` with
  packet receipt
  `sha256:8549bf3fb41d30399773e100feb2b9a620baf1cb1df077595aeb87bed3a7fcc1`.
  It did not review rights or execute a policy.

## Channel Target

Durable review route: `ohstnhunt@gmail.com` with the blocker id in the subject.
This file does not send email or assert that a reply watcher is active. The
same decision may be provided in the current Codex thread and then recorded
in the owning artifact.

## Non-Scope

This decision does not authorize public video publication, redistribution of
model weights, physical robot operation, a qualified policy ranking, a
production profile, payment, or allocation of a GPU host. The movement goal
for scene 841757 is a separate owner task-truth decision.
