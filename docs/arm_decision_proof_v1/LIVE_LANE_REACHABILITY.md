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
| `simpler_public_vast` | n/a | **retired** 2026-08-13 — not needed. Was already frozen by doctrine as a five-policy/general-ranking reference; now retired outright. |

Six lanes retired or frozen, three completed, six outstanding — of which two are
blocked on recoverable inputs and three need a builder.

## What every completed run still cannot claim

All three return `website_trigger_proven: false` with
`webapp_launch_record_missing`. The runs are real — signed HMAC intake, canonical
allocator, digest-bound immutable inputs, retained artifacts, teardown receipt,
provider-zero verified — but nothing binds a run to a website record. Until that
is closed, the honest phrasing is "triggered through the intake API", not
"triggered by the website".

## Defect-class sweep, 2026-08-13

Each named shape swept across the tree rather than fixed where first seen.
Three came back clean because they had already been fixed *as classes*
earlier, which is the point of the rule.

| Shape | Result | Rediscovery contract |
| --- | --- | --- |
| Artifact recording the authoring machine's absolute paths | fixed in 5 places (#464, #484, #487, #488, #492) | `launch_profile_residency_blockers`; the developer-home credential contract |
| A lane not emitting the evidence its terminal contract requires | fixed in 8 lanes (#481), plus the seal-*root* variant in 1 of 9 (#501) | `test_paid_lane_terminal_artifact_contract.py` |
| A lane whose bundle cannot be rebuilt from a command line | **found 4** (#512) | `test_paid_lane_bundle_cli_contract.py` |
| Secret resolved only from a developer home under `ProtectHome=true` | fixed in 3 modules (#492) | `test_no_module_resolves_a_credential_only_from_a_developer_home` |
| Bytes verified as the transfer user, not the consuming account | fixed for wire transfer (#485) and local install (#493) | `test_install_paid_lane_evidence_for_consumer.py` |
| A gate reading a channel that can fail independently | **clean** — `watching_a_live_second_channel` (#459, #477) | in `vast_provider_adapter` |
| A frozen pin encoding one machine's identity | **clean** — every lane carries its own `instance_label_prefix` (#473); no frozen instance-id allowlist remains | prelaunch inventory guard |
| Two components disagreeing about a directory's mode or schema | **clean** — one instance (`0o755` consumption ledger, #479) | — |

The false lead worth recording so nobody re-chases it: the four `mode_is_0600`
gates look like they disagree with the `0640 root:blueprint` provider secrets
and do not. They cover staging URL files and handoff capabilities the code
chmods itself; provider secrets are read through `_read_secret`, which does not
check mode. No change was made.

## The appearance chain is ordered, not parallel

`public_scene_artifixer3d_vast` and `paired_target_native_import_vast` are not
two independent lanes. The import gate's attempt authority validates a
`prior_terminal_artifixer` chain -- the predecessor's authority, terminal
result, object store cleanup, and provider-zero receipt -- and carries its
`aggregate_goal_spend_before_attempt_usd` forward against a **$12 campaign
cap** shared by both.

    public_scene_artifixer3d_vast  --terminal spend chain-->  paired_target_native_import_vast
      cap $10, TTL 7200..21600                                  TTL 1800..7200

So firing the import gate first is not slower, it is impossible: there is
nothing to authorize it against. ArtiFixer3D runs first, and its spend reduces
what remains for the gate.

Both now have live profile builders. The paired-target bundle is already built
at a deployed commit and staged host-resident, waiting only on its predecessor.

Both links can now be authorized from a command line via
`scripts/issue_appearance_chain_paid_attempt_authority.py`, which was the third
scope of the missing-entry-point class after #512 (lanes) and #520 (bundle
modules): modules that mint an authority rather than seal a bundle.

**Do not delete the retired AuraFusion360 receipts.** A prior Aura authority and
terminal result are what the ArtiFixer3D authority anchors its campaign spend
on. Retiring the lane means no new launch profile and no new attempt; it does
not mean the historical artifacts are disposable. Deleting them would strand the
appearance chain with no anchor.

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

**Retired 2026-08-13.** Not needed.

Worth keeping straight for anyone reading later: SIMPLER is a *policy-ranking*
benchmark reference, not an appearance method, so it is not literally replaced
by artifixer3D+ with `gpt-image-2` the way AuraFusion360 and Inpaint360GS are.
It was already frozen by doctrine as five-policy/general-ranking work; the
decision here retires it outright rather than leaving it frozen-but-pending.

Its bundle CLI landed in #512 anyway. Retired means we do not run it, not that
it should stay a landmine for whoever unfreezes it.

## The denominator is 30, not 14

Counting `*_vast.py` lane modules understated the gap. A `*_vast.py` is
transport, and several probe kinds have no lane module of their own -- the two
oldest builders emit kinds (`adp009d-franka-native-microcheck`,
`adp-retained-scene-gpu-render`) that appear in no lane module at all.

The allocator dispatches on **probe kind**, so that is the unit. Read from its
own `if args.probe_kind == ...` branches: **30 probe kinds are executable, and
8 are reachable from a live profile builder.** `tests/test_website_reachable_probe_kinds.py`
rediscovers that set from the allocator on every run, so a new branch there
cannot become the next unreachable lane without either a builder or a named
reason.

The 22 unreachable kinds are not one problem:

| Reason | Count | Meaning |
| --- | --- | --- |
| `retired_appearance_approach` | 7 | Superseded by the GPT-teacher/ArtiFixer3D path. Not to be relaunched; receipts retained as spend anchors. |
| `frozen_program` | 7 | Frozen by the active-program contract in `CLAUDE.md`. |
| `not_a_website_lane` | 1 | The profile preflight, which the allocator runs itself. |
| `awaiting_builder` | 7 | Real debt: executable, not retired, not frozen, unreachable. |

Only the last row is work. It is the Arena family (construction, controls,
policy, and the Isaac Lab Arena native control), the two fresh-site probes, and
the reconstruction worker smoke. The contract caps that row at its current size,
so it can shrink but not silently grow.
