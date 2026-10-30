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
