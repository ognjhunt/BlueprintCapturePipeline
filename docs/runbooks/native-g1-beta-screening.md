# Native G1 beta screening

The four HumanoidArena DP/π0.5 manipulation/navigation candidates are individual
rows in the beta catalog. Their execution backend is `native_g1.v1`, using the
existing G1/Dex3/SONIC worker and exact Arena USD. They do not pass through the
fixed-arm importer, and selecting DP does not require Pi checkpoint or tokenizer
assets. Planning capability is distinct from a successful current-task run.

The operator profile binds existing immutable file records under
`screening_candidates.<candidate_id>.native_g1`:

```json
{
  "scene_packet": {"path": "/absolute/g1/native_task_arena_packet_receipt.v1.json", "sha256": "sha256:<exact file hash>", "size_bytes": 123},
  "publisher_source": {"path": "/absolute/source/native_g1_publisher_source_stage.v1.json", "sha256": "sha256:<exact file hash>", "size_bytes": 123},
  "runtime_source": {"path": "/absolute/runtime/source-packet-receipt.json", "sha256": "sha256:<exact file hash>", "size_bytes": 123}
}
```

The values above illustrate field names, not usable receipts. The scene packet
must derive from this site's current retained packet: source assets, task,
success contract and authored geometry stay bound. G1 needs its actual body
pose, controller, contact topology and 640×480 head camera. Navigation candidates
add `navigation_authority`, an immutable record for the current task's authored
goal. A missing navigation goal does not block manipulation or other robots.
The runtime source packet must verify as `unitree_g1`. Correcting a registration
changes the campaign catalog identity while preserving the existing site spend.

`beta_policy_screening_g1_prepare.prepare_native_g1_beta_packet` supplies the
initial G1 packet from the trusted base packet, sidecar task-success contract,
current source identity and exact pinned Arena USD. It uses the existing floor
and footprint search, keeping long collision fixtures in the obstacle set;
the G1 pelvis starts 0.79 m above the inferred floor. The fixed development head
camera is mounted on the actual `head_link`, while the site's overview camera
is retained unchanged. Those choices, the right task hand and the 50 Hz control /
200 Hz physics rates are explicit simulation assumptions, not measured camera
or robot calibration. The action-step cap is resampled without increasing the
source episode duration. No navigation goal, scene geometry, task instruction,
human decision or ADP context is invented.

Preparation writes under the protected site root at
`native-g1/<base-receipt-digest>/<current-source-digest>/`. A changed current
context gets its own immutable packet. The
`native_g1_beta_scene_preparation.v1.json` receipt records these assumptions and
retains `current_consent_revalidated=false`: technical preparation does not
replace the live current-source checks in planning and launch. The initial CPU
validation compiled a real synthetic task fixture with the exact 38,195,671-byte
public Arena USD and read all 43 joint limits; it did not start Isaac or run a
policy. `OmniPBR.mdl` remains the declared Kit runtime dependency and is not
claimed resolved by this host-side test.

The planner composes a machine authority from the unchanged exact model-term
review and current trusted capture/task authority. Its basis is the owner's
2026-11-06 decision that G1 is a screening option for every site and task. It records `new_human_review_claimed=false`; it never
changes the old 841757 scene approval. Current consent and task identity are
reopened at planning and immediately before launch. No additional public-media
approval is needed for a private task-bound status/count summary. Public media,
model redistribution, ranking, qualification and physical claims remain false.

All candidates share the persistent $30 site allowance (`SITE_LIMIT` in
`beta_policy_screening_coordinator.py`) and the canonical allocator.
G1's conservative existing 48 GB memory floor is not a measured requirement for
every checkpoint. A G1 session runs the plan's first four cells
(`CELLS_PER_SESSION`, owner decision 2026-11-06: about four 50-minute cells fit
the 4 h session). They retain the shared cell IDs and seeds, but G1 uses frozen authored geometry and no unimplemented camera/object
perturbations. This does not establish robustness across scene variations.

The exact source is HumanoidArena commit
`68479287a784a69be9ce6ad739311d2f11f75ef9`; all four publisher configurations
return 20 actions from state64/head RGB into semantic40 commands, then SONIC
maps those commands to 43 named joints at 50 Hz. Publisher preprocessing and
normalization run once. The checkpoint and SONIC inventories remain
`configs/g1_humanoidarena_checkpoint_inventory.v1.json` and
`configs/g1_sonic_default_asset_inventory.v1.json`.

Source inspection found two startup details that matter to result identity:

- DP constructs pretrained torchvision ResNet18 before loading the policy.
  Its required `https://download.pytorch.org/models/resnet18-f37072fd.pth` is
  46,830,571 bytes, SHA-256
  `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.
  The beta inventory stages and accounts for this exact initialization asset;
  the authority retains its provenance for internal use without inferring
  redistribution permission.
- Pi catches missing/failed weight loads and can return an initialized model.
  Beta requires its exact `All keys loaded successfully!` marker from the owned
  unbuffered server process and rejects missing/unexpected-key or caught-load
  warnings before reset. Readiness alone is insufficient. The legacy worker's
  default behavior remains unchanged.

Every retained beta query binds its returned action array to that load
observation, source revision, candidate inventory and scene. A partial episode
with returned actions remains learned execution with a void task score.
Server startup, reset, timeout without returned actions, or imported robot assets
do not count as a learned rollout. A scored result additionally verifies the
worker/episode/trace/scorer chain and controller-witnessed scoped teardown.
Only the existing `learned_policy` private summary is published; raw observations,
diagnostic videos, prompts and local paths stay outside it.

Historical evidence is limited: the retained DP trial in
[issue 2245](https://github.com/ognjhunt/BlueprintCapturePipeline/issues/2245#issuecomment-5861467027)
returned 150 queries and failed the task with collision/containment failure.
Pi returned no actions before timeout. Those results do not transfer to another
scene, and this change does not restart that stopped campaign. Current validation
uses CPU contract/source tests, not new checkpoint inference or a GPU run.

## Automatic registration (the default)

The owner's standing approval (2026-11-06, beta launch tracker, G1 section,
answer 2) lets the service register each site by itself, with no per-site
person. Each pass of the screening worker (`beta_policy_screening_worker`)
calls `native_g1_site_registration.ensure_site_registration` for a website
site (`site-…`) before it reads the site's profile, for the newest completed
request of each scene intent:

- Only a `rigid_pick_place` task is registered. Any other kind records
  `g1_beta_native_rigid_task_required`.
- The site's base packet and task contract resolve exactly as in the door's
  builder: `native_g1_registration_contract.site_inputs` reads the team
  registry and the completed source queue.
- The shared inputs come from the controller's pinned
  `config/native_g1_shared_inputs.v1.json`, not from the historical ADP
  registry. Each is measured when it is used:
  - the publisher source stage receipt, against its pinned `receipt_digest`;
  - the runtime source receipt, against its pinned sha256 and size, which are
    the bytes `restore-g1-runtime-source` puts back;
  - the Arena robot USD, found as the old manipulation scene plan's
    `robot.usd_path`, against its pinned sha256 and size.
- It reads the site's consent live, as planning does, before it prepares a
  packet and again before it writes any row, also when the packet already
  exists. With no packet for the current source yet, it runs
  `prepare_native_g1_beta_packet` with the builder's own arguments. The packet
  lands under the site's `native-g1/<base>/<source>/`.
- A refused consent (a `beta_screening_*` code, `team_scene_capture_withdrawn`
  or `native_g1_install_retained_task_consent_invalid`) prepares nothing,
  writes no row and removes the site's existing row. Its status file then
  holds only the reason, with no source or candidates. An unanswered check
  writes nothing new and is tried again on the next pass. Registration never
  deletes a packet: the operator's withdrawal deletion removes the packets, the
  row and its status (`website_capture_withdrawal_native_g1`).
- It writes the service's own row to
  `/var/lib/blueprint/pipeline-control-plane/native-g1-site-registrations/<site_id>.json`
  (`BLUEPRINT_NATIVE_G1_SITE_REGISTRATION_ROOT`). The row has exactly the
  shape of `site_screening_candidates.<site_id>`, is replaced atomically and is
  mode 0640. `<site_id>.status.json` beside it records `registered_by:
  service_automatic`, the standing `authorization_reference`, the time, and
  `registered` or `unavailable` with a reason for each candidate.
- Navigation rows are registered only when the base task has a navigation goal
  and the site holds exactly one goal authority for the G1 scene. Until then
  they keep `g1_current_navigation_goal_authority_missing`.

`team_candidate_registry_reader._site_screening_profile` uses the root
registry's row while it names the current source. Otherwise it reads the
service row with the same checks. That row must also be a regular,
non-symlink file directly under the root, owned by root or the service user
and writable by no one else; anything else is ignored. A new row changes the
catalog revision (`beta_policy_screening_g1_catalog`), so the site plans again
by itself. An unchanged source with files that still verify does nothing.

Planning stays the verifier (`candidate_execution_spec` and
`compose_beta_authority`), and registration grants no spend. A failure is a
fixed reason in the status file and never stops the site or its other
candidates. Shared-input reasons, and a preparation already running, are tried
again on the next pass. Every other reason waits for a new source, pin or
release.

`BLUEPRINT_NATIVE_G1_AUTO_REGISTRATION=0` turns it off in the process that has
it set: that process neither registers nor reads service rows. The screening
unit reads `/etc/blueprint/pipeline-control-plane.env`.

The door's `install-native-g1-policies` below stays for manual or override use.
Its root registry row wins while it names the current source.

### Spawn check

The 2026-09-27 DP trial began with the G1 inside the room's geometry and blew
up at spawn. The packet recipe now measures the whole body on CPU before any
spend (`native_g1_spawn_clearance`). It poses the Arena USD's colliders at the
packet's reset joints, with the pelvis at the stance. It then measures them
against the scene collision with the arm check's rest-overlap measure
(`scene_physics_preflight`). A sole up to 2 cm into an upward-facing surface
is standing. Anything else inside the room is overlap. A body with no collider
makes the reading `unmeasured`, never `clear`. If the searched stance
overlaps, preparation takes the first other searched stance that clears; with
none it keeps the overlap. The report, `native_g1_spawn_clearance.v1.json`,
sits beside the packet, and the preparation receipt carries a
`spawn_clearance` line. Planning gives a G1 row `g1_spawn_overlap` or
`g1_spawn_clearance_unmeasured` and funds nothing for it unless the report
reads `clear` for that packet's plan. Registration writes the report for a
packet prepared before the check. To read one by hand (CPU only):
`python -m blueprint_pipeline.native_g1_spawn_clearance --packet <packet dir>`
(add `--write` to keep it).

## Activate a current site's G1 candidates

First make sure the shared runtime source is on the host. On 2026-11-06 the
directory the shared registry's `runtime_source_receipt_path` names
(`/mnt/blueprint-work/g1-cuda-runtime-source-7c1f528-20260926/`) was gone, and
every candidate ended `g1_shared_source_verification_failed`. The door's
`restore-g1-runtime-source` puts the packet and its receipt back byte for byte
from the object-store copy a 2026-09-27 run retained, so the registry binding
verifies unchanged:

1. The door must run a `main` commit that has the operation
   (`python3 scripts/operator_door.py upgrade-door <sha on main> --wait`), and the
   active release must contain `blueprint_pipeline.native_g1_runtime_source_restore`
   (a deploy of such a commit). Otherwise the request refuses with
   `kind_unknown` or `g1_runtime_source_release_lacks_module`.
2. Plan it. The plan only reads and writes nothing but its result:
   `python3 scripts/operator_door.py restore-g1-runtime-source --wait`. It
   should end `planned`, with the build record verified, the retained object
   present at its exact size and digest, enough disk (about 4.9 GB with the
   margin) and both targets `absent`, or `matches` once restored.
3. Restore it:
   `python3 scripts/operator_door.py restore-g1-runtime-source --apply --wait`.
   The outcome is `restored`, or `already_restored` if a run before it finished.
   Any other bytes at either target refuse with
   `g1_runtime_source_target_conflict` and stay untouched.
4. Use the result's `runtime_source` (`receipt_sha256` and
   `receipt_size_bytes` in the outcome; the full result is
   `results/<id>.g1-runtime-source.json`) as the install request's
   `runtime_source` reference below. The receipt is the build record's exact
   bytes, `sha256:b303f3017e799a36a2a467de60f235109b1d810a1ef3a87d8d92feb02e6ca5d2`
   and 44,943 bytes. The pinned `config/native_g1_shared_inputs.v1.json`
   names the same bytes, so after the restore the service registers each
   pick-and-place site on its next screening pass. The install request below
   is needed only for manual or override use.

It starts no compute and spends nothing beyond one object-store download. A
refusal names one fixed `g1_runtime_source_*` code; see the beta launch
tracker's G1 section.

Use the authenticated operator door; activation does not require editing the
protected registry or logging into the controller:

```sh
python scripts/operator_door.py install-native-g1-policies \
  --request-file /private/current-site-g1-request.json --wait
```

The sealed `native_g1_policy_install_request.v1` contains only
`installation_id`, exact `release_commit`, `site_id`, the existing six-field
beta `source`, `expected_registry_digest`, `shared_registry_digest`,
`authorization_reference`, `publisher_source`, `runtime_source`, `candidates`,
and `request_digest` (plus `schema_version`). The two shared references contain
only `sha256` and `size_bytes` for the retained receipt files. Each candidate
row contains one fixed built-in `candidate_id`, a `scene_packet` receipt
reference or `null`, and a `navigation_authority` receipt reference or `null`.
The digest is SHA-256 of sorted, compact UTF-8 JSON excluding `request_digest`.
No source paths, commands, endpoint URLs, credentials or rights assertions are
accepted in this request. It has no budget field and grants no provider spend.

A null scene reference asks the service user to prepare the current site's
packet using the reviewed recipe. It resolves the current base packet and task
contract from the protected team registry and the existing completed source
queue. On this door path, shared publisher/runtime locators come exclusively from
`/var/lib/blueprint/task-evaluation-inputs/g1-team-campaign-registry.json`.
The historical binding supplies shared source and exact robot model bytes;
its old task, scene approval and navigation authority are not transferred.
No policy checkpoints need to be downloaded to the controller.

Preparation retains the exact current scene assets and task contract. It uses
an explicitly labeled simulated floor stance, the fixed G1 head camera,
right-hand contacts, 50 Hz control and 200 Hz physics. The original episode
wall-time limit is preserved when action steps are resampled. The result lives
under the current site's existing scene directory at
`native-g1/<base-receipt-digest>/<current-source-digest>/`. A new task/source
identity gets a new directory; historical packets are never overwritten.
Missing navigation semantics remain an individual navigation-candidate reason.

The installed root process uses Python's standard library only and independently
reopens the current context, packet, derivation, source inventory and runtime
archive. It never imports the active pipeline checkout or runs retained Git
configuration/hooks. Only exact tracked publisher source files enter the beta
runtime bundle; ignored historical bytecode is not shipped. The installer
retains a durable request, proposal comparison, before/after snapshots, intent
and receipt, using the same lock and compare-and-swap discipline as existing
public-policy installers. Retrying the same installation is idempotent.

Registrations are scoped to `site_screening_candidates.<site_id>` and the exact
six source fields. Global defaults, other sites, existing policies and derived
scene refresh remain intact. Every requested candidate receives `registered`
or `unavailable` with a reason. Registration means its source receipts were
installed: it does not claim live consent was queried, grant execution authority,
prove checkpoint inference or score the task. The planner and launcher recheck
current consent and task identity and compose the existing reviewed model terms
before budgeted execution. This operation does not release the founder dispatcher
hold, start a simulator, allocate a provider or authorize public media.

## Dedicated provider runtime

Each selected beta candidate gets one sealed provider bundle and one consumption
receipt. The receipt fixes its absolute deadline before transport, including
image startup, downloads, installation, simulation and collection. An uncertain
allocation never receives a new deadline or automatic retry. The exact resource
name uses the existing native-policy-canary family, and independent watchdog,
fleet ownership, provider absence and staged-object cleanup are retained.

Only that candidate's checkpoint and the reviewed SONIC pair are fetched. DP
also requires the exact torchvision ResNet18 initialization file; Pi alone
stages the reviewed PaliGemma tokenizer. Every download is streamed and checked
against its size and SHA, within the sealed transfer quote and deadline. The
96-wheel policy environment reuses 4,132,469,228 bytes of existing native packet
wheels and downloads 261,777,389 bytes of exact extra wheels. Installation uses a
fresh isolated Python 3.12 environment, offline pip, fixed hashes and no online
resolver or apt. Expanded native wheels, per-cell scene copies, policy install,
retained output and the cold-image allowance determine the requested disk.

The beta server ignores ambient Python import paths. Its pinned source tree is
copied from tracked files with sanitized Git configuration and a freshly built
exact-commit index. DP and Pi must retain the publisher's complete checkpoint
load observation; known partial-load or random-initialization warnings refuse
the loaded-checkpoint claim. That observation is bound into returned action
checkpoints. Readiness alone is not inference evidence.

Cells repeat the exact frozen scene with explicit reset seeds. The session runs
what fits its admitted lifetime and retains a result or bounded reason for every
cell. The compact provider result references retained worker, episode, trace and
query JSON instead of duplicating arrays. After download, the controller reopens
those exact bytes, independently reconstructs each derived scene and authority,
and recomputes the task score. Partial returned actions survive a timeout as
queried but unscored evidence. Provider transport and scientific result digests
are distinct. Only scoped controller closeout authorizes result projection.

Git must be present in the pinned image before source verification. Its absence
produces a per-cell bootstrap reason; the lane never installs it with apt.

Pi readiness uses an explicit tokenizer directory when supplied, then the
existing `BLUEPRINT_G1_PI_TOKENIZER_DIR` override, then the retained model-only
cache at
`/var/lib/blueprint/task-evaluation-inputs/g1-841757-development/pi-tokenizer-f1eaa13`.
Planning and bundle preparation recheck the exact reviewed six-file inventory;
the directory name alone establishes neither current bytes nor task authority.
An absent or inaccessible directory yields
`g1_beta_selected_pi_tokenizer_unavailable`; changed files or inventory yield
`g1_beta_selected_pi_tokenizer_invalid`. An invalid configured override never
silently falls back. DP does not inspect these assets. This path performs no
credential read, gated download, environment edit or allocation.

On the provider, source verification recomputes the actual clean checkout and
all publisher file identities. Only the `source_root` location and resulting
receipt digest may differ from the sealed controller receipt; the original
receipt stays retained unchanged and every other field must match exactly.
