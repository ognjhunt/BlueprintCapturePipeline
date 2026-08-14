# Live lane reachability

This is the current product-path inventory for paid and provider-backed Task
Evaluation probes. The allocator's `probe_kind` dispatch table is the
denominator; a transport module is not a lane and one profile builder may emit
more than one ordered probe kind.

<!-- reachability-inventory:start -->
Current executable inventory: **32 dispatched, 17 website-reachable, 15 named
non-reachable, 0 awaiting-builder.**
<!-- reachability-inventory:end -->

`tests/test_website_reachable_probe_kinds.py` derives these sets from the
allocator and every `build_*_live_profile.py` module. It also verifies the
inventory sentence above, so code and this operator ledger fail together when
the denominator changes.

## What reachable means

A probe kind is website-reachable only when a live profile builder emits it.
The profile is the immutable bridge across the website boundary: it binds the
probe kind, source commit, TTL band, allocator arguments, immutable inputs,
secret-profile identity, terminal contract, and claim ceiling.

Reachability alone does **not** prove that a profile is published on the live
host, that the deployed control plane is at the same commit, or that a paid
attempt completed. Production proof additionally requires:

1. all mutable and immutable control-plane surfaces deployed to one clean Git
   commit by `scripts/deploy_control_plane_commit.py`;
2. the exact live profile published in the host catalog;
3. a signed website launch record bound to the Pipeline intake receipt;
4. allocator admission and attempt authority where the lane requires it;
5. retained terminal artifacts, teardown, billing reconciliation, and a fresh
   provider-zero receipt; and
6. no automatic paid retry.

## Website-reachable probe kinds

| Probe kind | Live profile builder |
| --- | --- |
| `adp-artifixer3d-exact-support` | `build_artifixer3d_live_profile.py` |
| `adp-gaussian-excision` | `build_gaussian_excision_live_profile.py` |
| `adp-isaac-lab-arena-native-control` | `build_arena_native_control_live_profile.py` |
| `adp-paired-target-native-import` | `build_paired_target_native_import_live_profile.py` |
| `adp-retained-scene-gpu-render` | `build_retained_scene_render_live_profile.py` |
| `adp-usd-content-agents` | `build_content_agents_live_profile.py` |
| `adp-usd-joint-agent` | `build_joint_agent_live_profile.py` |
| `adp009b-exact-simready-isaac` | `build_simready_isaac_live_profile.py` |
| `adp009d-franka-native-microcheck` | `build_adp009d_840313_live_profile.py` |
| `native-task-arena-construction` | `build_native_task_arena_live_profile.py` |
| `native-task-arena-controls` | `build_native_task_arena_live_profile.py` |
| `native-task-arena-policy` | `build_native_task_arena_live_profile.py` |
| `new-site-diagnostic-canary` | `build_new_site_diagnostic_canary_live_profile.py` |
| `new-site-native-camera` | `build_new_site_native_camera_live_profile.py` |
| `reconstruction-worker-smoke` | `build_reconstruction_worker_smoke_live_profile.py` |
| `semantic-sam31-source-tracks` | `build_sam31_source_tracks_live_profile.py` |
| `semantic-teacher-image-edit` | `build_semantic_teacher_image_edit_live_profile.py` |

The three `native-task-arena-*` rows are ordered stages of one chain, not
three independent campaigns. Likewise, the appearance path is ordered:

`adp-artifixer3d-exact-support` -> `adp-paired-target-native-import`

The import authority validates the predecessor terminal result, cleanup, and
provider-zero receipt and carries forward aggregate spend against the shared
campaign cap. The import cannot be authorized first.

## Named non-reachable probe kinds

These fifteen allocator branches are deliberate decisions, not builder debt.

### Retired appearance/reference approaches (7)

- `adp-aurafusion360-author-smoke`
- `adp-aurafusion360-exact-residual`
- `adp-aurafusion360-interiorgs`
- `adp-inpaint360-interiorgs`
- `adp-simpler-public-reference`
- `adp009d-aura-native-live-camera`
- `adp009d-aura-ovrtx-live-camera`

Their historical receipts remain immutable spend and provenance anchors. No
new profile should make them reachable unless the active-program decision is
explicitly changed.

### Frozen programs (7)

- `openpi-policy-ranking`
- `persistent-policy-wam-loop`
- `policy-ranking-cosmos-reasoner`
- `policy-ranking-successor-cosmos`
- `single-kitchen-episode`
- `single-kitchen-finetune`
- `single-kitchen-qualification`

Arm Decision Proof v1 is the sole active program. Policy-ranking,
world-model, and post-training work remains frozen.

### Internal allocator preflight (1)

- `task-evaluation-profile-preflight`

This is not a website lane. The allocator runs it against a profile before any
provider mutation.

## Terminal and production rules

Every live profile must rehearse `would_pass` through
`scripts/rehearse_lane_terminal_contract.py` before a paid attempt. The shared
terminal contract requires an allocator-owned artifact manifest, teardown
manifest, run-owned post-teardown provider-zero receipt, retained guard
snapshot, and website sync receipt. The new-site camera and diagnostic lanes
are included in the same AST-discovered terminal-artifact contract as every
other provider transport.

The canonical deployment sequence is documented in
`PRODUCTION_WEBSITE_LAUNCH.md`. Deployments must use the repo's atomic deploy
tool rather than moving the mutable checkout and release symlink separately.
The deploy tool holds every paid-launch slot while activating, rejects a dirty
surface, restarts the intake service, and verifies the public runtime reports
the requested commit.

Do not report a lane as "working in production" from builder reachability, a
CPU test, simulator startup, or provider allocation alone. That claim requires
the exact deployed commit plus a terminal launch record satisfying the six
production-proof conditions above.
