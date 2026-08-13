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

Every one of the fifteen paid lanes below has an established status: it has
completed through the live path, or it names the exact thing stopping it. None
is left merely unattempted.

| Lane | Builder | Status |
| --- | --- | --- |
| `adp_retained_scene_render_vast` | yes | **completed** — all controls passed, provider-zero verified |
| `adp_content_agents_vast` | yes | **completed** — all controls passed, provider-zero verified |
| `public_scene_simready_isaac_vast` | yes | **completed** — all controls passed, provider-zero verified |
| `adp009d_franka_vast` | yes | reachable, never fired. Its three published profiles are `task-evaluation-profile-preflight` dry-run profiles, not lane launches; a launchable profile has not been published. |
| `adp_gaussian_excision_vast` | yes | **blocked: no bundle CLI.** `build_gaussian_excision_vast_bundle` exists but the module has no `main()`, so the bundle cannot be rebuilt at the deployed commit — and the allocator refuses a bundle from any other commit. Its inputs (cameras, execution authority, scene PLY, dependency wheelhouse) survive **only inside the existing bundle zip**, so a rebuild also needs them extracted. |
| `adp_joint_agent_vast` | yes | **blocked: scattered inputs.** The module has a CLI needing eight inputs; `execution_authority.json`, `joint_agent_packet.json`, and the review contract survive inside the bundle zip, but the freeze and scope-amendment documents were not located on disk. |
| `native_task_arena_vast` | no | needs a builder. Three probe kinds (construction/controls/policy) and up to five input packets; all inputs located under `second_scene_840796_e2e`. |
| `adp_isaac_lab_arena_vast` | no | needs a builder. Uses the shared artifact manifest directly. |
| `adp009d_ovrtx_vast` | no | needs a builder. Appearance/camera transport; see the retirement note below before building it. |
| `adp009d_aura_native_vast` | no | **retired** with the Aura appearance method. |
| `adp_aura_author_smoke_vast` | no | **retired** — artifixer3D+ with `gpt-image-2`. |
| `adp_aura_interiorgs_vast` | no | **retired** — artifixer3D+ with `gpt-image-2`. |
| `public_scene_aura_exact_residual_vast` | no | **retired** — artifixer3D+ with `gpt-image-2`. |
| `adp_inpaint360_interiorgs_vast` | no | **retired** — artifixer3D+ with `gpt-image-2`. |
| `simpler_public_vast` | no | **frozen by doctrine** — SIMPLER policy-ranking reference; five-policy work is frozen. |

Six lanes retired or frozen, three completed, six outstanding — of which two are
blocked on recoverable inputs and three need a builder.

## What every completed run still cannot claim

All three return `website_trigger_proven: false` with
`webapp_launch_record_missing`. The runs are real — signed HMAC intake, canonical
allocator, digest-bound immutable inputs, retained artifacts, teardown receipt,
provider-zero verified — but nothing binds a run to a website record. Until that
is closed, the honest phrasing is "triggered through the intake API", not
"triggered by the website".

## Rehearse before firing

`scripts/rehearse_lane_terminal_contract.py` asks the launch's own terminal
question against a lane's real sealing path for **$0**, in about a second. All
23 lane profiles published on the control plane currently rehearse
`would_pass`. Two of the defects that cost paid GPU runs on 2026-08-13 were path
bugs this would have caught first, so run it on every profile before firing.

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

`CLAUDE.md` and `AGENTS.md` were updated alongside this decision, so the binding
guides and this file agree: artifixer3D+ with `gpt-image-2` is the appearance
path, and neither retired method is an outstanding requirement or an open rights
question. Nothing here is pending anyone's decision.

### SIMPLER (`simpler_public_vast`)

The SIMPLER policy-ranking reference. `CLAUDE.md` freezes five-policy and
general-ranking work, so building it a launch path would be outside the active
program. Frozen, not blocked.
