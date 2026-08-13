# Fresh-scene paired-target preparation

This is the production sequence for turning one calibrated, rights-admitted
scene with one to five preregistered task objects into paired-target
ArtiFixer3D inputs. It replaces the earlier practice of hand-staging masks and
intermediate receipts. The coordinator is
`blueprint_pipeline.fresh_scene_paired_target_preparation`.

## Responsibility split

Agents may propose task-object labels, review SAM tracks, author candidate CAD,
and inspect visual results. They do not choose Gaussian indices, rewrite source
PLY bytes, alter camera calibration, grant provider rights, grade a policy, or
upgrade generated pixels to observed evidence.

Deterministic Pipeline code owns camera joins, RLE decoding, binary masks,
Gaussian contribution accounting, derived PLY byte accounting, digests,
rights/spend admission, immutable receipts, outside-support checks, and
provider-zero closure.

## Required stages

1. `calibrated_scene_views`
   - Implementation: `public_scene_inpainting_inputs` using the exact,
     purpose-bound calibrated standard-3DGS renderer.
   - Bind the immutable standard-3DGS derivative, registered frame, camera
     intrinsics/extrinsics, and lossless source renders for every task. This
     stage does not use an appearance-repair model.
2. `sam31_task_inputs`
   - Implementation: `public_scene_sam31_task_inputs`.
   - Deterministically retain each task's exact ordered calibrated PNG set, a
     lossless FFV1 sequence, fixed JPEG analysis derivatives, camera/frame
     joins, prompts, and the pinned SAM provider profile. This producer creates
     the full SAM frame registry and removes all hand-authored packet staging.
3. `sam31_source_tracks`
   - Backend: pinned Meta SAM 3.1 Object Multiplex.
   - Production lane: `semantic-sam31-source-tracks` through
     `paid_resource_allocator gpu-canary`.
   - Input: ordered, hash-bound calibrated source-frame derivatives, explicit
     object prompts, checkpoint/license/privacy/trade/execution authority.
   - Run one task-local packet for each preregistered object so every object can
     use its own calibrated target-visible camera set. Output is compact
     persistent RLE mask tracks only. No geometry or identity qualification.
4. `calibrated_object_masks`
   - Implementation: `public_scene_calibrated_object_masks`.
   - An operator or review agent explicitly selects the track ID or track-ID
     union corresponding to each preregistered task object.
   - The materializer verifies exact source-frame and camera-record digests,
     requires an explicit one-to-one camera-to-SAM-frame map, decodes the RLE,
     and writes one undilated binary PNG per calibrated camera. It never assumes
     that a camera ID happens to equal a source-frame ID.
5. `excision_freezes` and `segment_sweep_freezes`
   - Implementations: `public_scene_gaussian_excision_audit` and
     `public_scene_segment_contribution_cutout`.
   - Bind the immutable standard-3DGS derivative, registered collision object,
     calibrated views, source images, reviewed object masks, and 1--5 task
     freezes before any contribution execution.
6. `gaussian_contribution_evidence`
   - Backend: FlashSplat commit
     `3e3b14786333bf0163ba1b8541e86a3765112d7d`, rasterizer commit
     `189c483ffa33dd6d5661343ce496df0c6eb80a0c`.
   - Production lane: `adp-gaussian-excision` through the canonical paid
     allocator, with two deterministic repetitions, one immutable bundle,
     zero retry, hard cap/TTL, independent watchdog, cleanup, and API-zero.
7. `segment_cutout_set`
   - Implementation: `public_scene_segment_contribution_cutout`.
   - Union every Gaussian with renderer-detectable contribution to the exact
     task-object segments across all calibrated views and repetitions.
   - Write only a derived, byte-accounted, digest-bound retained PLY; never
     mutate canonical InteriorGS.
8. `segment_repair_preflight` and `artifixer_candidate_inputs`
   - Bind exact repair support, rights, cameras, original frames, task freezes,
     and the derived retained PLY. A missing upstream input must remain an
     upstream `fresh_scene_*_missing` blocker, not surface later as a generic
     ArtiFixer preflight failure.
9. `semantic_teacher_receipts`
   - Preferred editor: rights-admitted `gpt-image-2` full-frame empty-scene
     candidates. A pinned local editor may be used when disclosure rights or
     production credentials do not admit the hosted route.
   - Do not hard-paste generated pixels into a washer/laptop-shaped composite
     before ArtiFixer3D. Preserve the whole teacher image and separately retain
     the original anchor plus the excluded repair-region loss mask.
10. `dual_target_artifixer_inputs` and `artifixer3d_result`
   - Interleave two records at every camera pose: the whole semantic teacher,
     and the original source frame whose repair area is excluded from its loss.
   - Run paired-target ArtiFixer3D first. Render and retain calibrated review
     frames. Run 3D+ only after raw 3D review passes and never let 3D+ become the
     primary removal painter.

## Droplet capability contract

The droplet needs repository code at the deployed protected-main commit, the
published task-evaluation profiles, allocator/provider credentials in the
canonical secret integration, the pinned SAM/FlashSplat/ArtiFixer source and
checkpoint bindings declared by their bundle builders, and rights-admitted
scene inputs staged as immutable profile inputs. It must not depend on a laptop
path, a manually copied mask, an unrecorded Codex image, or an agent worktree.

Run the mutation-free coordinator after each retained stage. Its
`first_blocker` and `next_required_stage` are the authoritative next action.
Production dispatchers should publish the next paid profile only after all
prior deterministic stages validate and a new file-backed authority has been
materialized. Automatic paid retries remain forbidden.

For `sam31_source_tracks`, build that profile with
`scripts/build_sam31_source_tracks_live_profile.py`. The builder binds the
exact request, input bundle, bundle receipt, single-use authority, deployed
commit, host-resident secret-file path, hard cap/TTL, and zero retry. The
profile deliberately does not publish a capacity snapshot: after the
independent watchdog is armed, the allocator collects a fresh Vast capacity
and provider-zero preflight at execution time, writes it beneath the launch
root, and fails closed before authority consumption or provider mutation when
that live check does not pass. Every terminal execute result also seals the
shared Task Evaluation artifact manifest and teardown record required by the
website reconciler. A successful terminal result must additionally name the
normalized `semantic_source_track_import_result.v1` artifact. That compact,
digest-bound track file is the direct input to the reviewed calibrated-mask
request; the production handoff never returns or redistributes source-frame
bytes.

## Agents SDK orchestration

The repository already pins `openai-agents` and the production control plane
already exposes the Task Evaluation Supervisor. Use that supervisor as the
coordinator, not as the evidence writer. Its agent may inspect the fresh-scene
status, propose task prompts and SAM track IDs, request human review for
ambiguous masks, and invoke the next registered producer. Each producer must
remain a typed repository tool with immutable input digests and a scoped output
directory. Paid producers are never generic SDK tools: they materialize a
file-backed authority and dispatch only through `paid_resource_allocator`.

The required agent-facing tools are:

- `inspect_fresh_scene_preparation`: read the status ledger and return the first
  blocker and next legal producer.
- `materialize_sam31_task_inputs`: produce the calibrated, task-local SAM input
  packet without uploading bytes or starting paid work.
- `materialize_calibrated_object_masks`: invoke the deterministic task-local
  camera/frame bridge after explicit track selections are present.

`fresh_scene_supervisor_bindings` is the production bridge. It accepts only a
host-resident, digest-bound status plus the exact available tool request, and
then runs the existing reconstruction continuation with those registered
tools. The agent can recommend visual review or the next paid stage in prose,
but review approval, paid authority, allocation, and retry remain separate
typed human/control-plane actions rather than SDK tools.

Until those bindings are present in a deployed task-evaluation profile, the
same modules remain usable through their typed CLIs, but the flow is not yet a
website-triggered agent workflow and must not be called production-complete.

## Claim boundary

SAM masks, GPT frames, ArtiFixer outputs, SimReady assets, and simulator results
are derived support. Visual qualification, native import, reachability,
zero-action/scripted controls, and the two frozen policy candidates are later
gates. None of these outputs is physical evidence.
