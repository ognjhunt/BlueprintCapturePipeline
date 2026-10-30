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
review and current trusted capture/task authority, within the owner's requested
internal simulation scope. It records `new_human_review_claimed=false`; it never
changes the old 841757 scene approval. Current consent and task identity are
reopened at planning and immediately before launch. No additional public-media
approval is needed for a private task-bound status/count summary. Public media,
model redistribution, ranking, qualification and physical claims remain false.

All candidates share the persistent $15 site allowance and canonical allocator.
G1's conservative existing 48 GB memory floor is not a measured requirement for
every checkpoint. Ten diagnostic cells retain the shared cell IDs and seeds,
but G1 uses frozen authored geometry and no unimplemented camera/object
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

## Activate a current site's G1 candidates

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
queue. Shared publisher/runtime locators come exclusively from
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
