# Live lane reachability

Which paid lanes can be launched from the website, what each one still needs,
and which are deliberately out of scope.

A lane needs three things to be reachable. Two are code and one is a command:

1. **A live profile builder.** A launch profile is the only thing that carries a
   lane across the website boundary. The skeleton lives once in
   `src/blueprint_pipeline/task_evaluation_live_profile.py`; a lane brings a
   `LaneLiveProfileSpec` — its probe kind, TTL band, allocator arguments, and
   the receipts it pins.
2. **An attempt-authority issuer**, but only for the lanes whose allocator
   branch demands one. Six of the fourteen do not.
3. **Host-resident staging**, which is `scripts/stage_paid_lane_bundle.py` — one
   command per lane, no new code.

## Status

| Lane | Profile builder | Attempt authority | Reachable |
| --- | --- | --- | --- |
| `adp_retained_scene_render_vast` | yes | yes | **proven** — completed 2026-08-13 |
| `adp_content_agents_vast` | yes | yes | **proven** — completed 2026-08-13 |
| `adp009d_franka_vast` | yes | n/a | yes |
| `public_scene_simready_isaac_vast` | yes | yes | yes — never launched |
| `adp_gaussian_excision_vast` | needed | yes | no |
| `adp_joint_agent_vast` | needed | not required | no |
| `native_task_arena_vast` | needed | not required | no |
| `adp009d_ovrtx_vast` | needed | not required | no |
| `adp009d_aura_native_vast` | needed | not required | no |
| `adp009d_native_microcheck` | needed | not required | no |

Bundle receipts already read `status: ready, blockers: []` for every lane in the
table, and the canonical allocator has always had a branch for each. Reaching
them is launch-path plumbing, not new capability.

## Deliberately out of scope

These are **not** blocked on work. They are scoped out, and their code, bundles,
and allocator branches remain in the repository.

### AuraFusion360 (4 lanes) and Inpaint360GS

`adp_aura_author_smoke_vast`, `adp_aura_interiorgs_vast`,
`adp009d_aura_native_vast` (as an Aura appearance method),
`public_scene_aura_exact_residual_vast`, and
`adp_inpaint360_interiorgs_vast`.

**Decision, 2026-08-13:** a real run of artifixer3d with `gpt-image-2` produced
materially better results than either method, so neither is needed as a quality
challenger. No profile builder will be written for them and no aura
attempt-authority issuer is required.

What would have been needed had they stayed in scope, recorded so nobody
rediscovers it:

- **Aura.** `adp009b_aura_runtime_prerequisite_receipt.v1.json` reads
  `author_data_rights_established: false`. Its only snapshot is
  `aurafusion360_openclip_vit_h_14` (MIT, rights established); the two artifacts
  the lane requires — `aurafusion360_sunflower_author_scene` and
  `aurafusion360_sunflower_expected_output` — are absent. A rights decision, not
  an engineering one.
- **Inpaint360GS.** `big-lama.zip` is already host-resident and its sha256 and
  size match the pinned identity exactly. Missing is a prerequisite receipt
  carrying an `inpaint360_author_smoke` method with `rights_established: true`;
  `public_scene_method_prerequisites.py` is request-driven and has no reference
  to that method, so a request document would have to be authored first.

Note that `CLAUDE.md` and `AGENTS.md` still name Inpaint360GS author smoke and
AuraFusion360-as-challenger among ADP-009's completion requirements. That text
was not changed alongside this decision. **If the artifixer3d result is meant to
replace those requirements rather than sit beside them, the doctrine has to move
too** — otherwise the repository's binding guide and this file disagree about
what ADP-009 requires.

### SIMPLER (`simpler_public_vast`)

The SIMPLER policy-ranking reference. `CLAUDE.md` freezes five-policy and
general-ranking work, so building it a launch path would be outside the active
program. Frozen, not blocked.
