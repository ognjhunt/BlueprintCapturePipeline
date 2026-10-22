# Admitted Atlas pose operation

Beta: concurrent scene preparation. Captures without a scene-preparation
sponsorship wait at `needs_input`
(`website_source_camera_depth_registration_required`) for source camera and
depth registration. World Labs Atlas `images2PosedRGBD` might supply that
registration once access arrives. This page describes the admitted producer that
calls it. The producer exists so the first Atlas run can happen under the same
spend and evidence controls as every other paid call. It does not make Atlas a
default or change any gate.

Sources: the Atlas beta docs, read 2026-10-19:
[images to posed RGBD](https://atlas-beta.worldlabs.ai/docs/images-to-posed-images),
[API reference](https://atlas-beta.worldlabs.ai/docs/api-reference),
[assets](https://atlas-beta.worldlabs.ai/docs/assets),
[operations](https://atlas-beta.worldlabs.ai/docs/operations) and
[billing](https://atlas-beta.worldlabs.ai/docs/billing).

## Who calls Atlas

Only `paid_resource_allocator.run_sponsored_website_atlas_pose`, the canonical
allocator entry, calls Atlas. `scripts/verify_paid_resource_allocator.py` pins
that: any other caller of `run_admitted_atlas_pose` or
`submit_atlas_pose_operation` fails the verifier. The adapter,
`website_atlas_pose.py`, is registered as a grant-gated surface in
`docs/architecture/paid-resource-mutation-surfaces.json`. Its HTTP goes through
the audited boundary, `safe_outbound_http`:

- API calls are pinned to the API host.
- Provider-issued upload and read URLs are pinned to their own host and never
  receive the API key.
- Redirects are refused.

Scene preparation never buys Atlas itself:

- The default geometry route (`run_website_scene_geometry`) still only writes
  the pending `atlas_pose_inputs.json`, with or without a task context.
- The scene handoff reads the capture's finished Atlas job
  (`docs/website-atlas-scene-source.md`). Without one, it waits at
  `needs_input`.

Website captures no longer buy Marble worlds (owner decision 2026-10-22):
`WorldLabsPreviewProvider.submit` refuses a website capture
(`website_marble_reconstruction_retired`), and the Pipeline no longer requests a
`world_labs` reservation. No Marble-shaped request can name Atlas either:

- The WebApp refuses every model name that starts with `atlas`.
- The legacy Marble request builder refuses any model that is not a Marble
  model (`worldlabs_model_not_a_world_model`).

## Spend identity and consent

The reservation uses the provider identity `world_labs_atlas`; the Pipeline no
longer reserves Marble's `world_labs` for website captures. Atlas has its own terms, and the producer uploads unedited source
frames: the 518-pixel model inputs of the source-frame preparation, with people
not removed. A site's consent to the Marble terms covers people-removed views
sent to Marble, not this. The WebApp's `preparation-spend` route does not accept
`world_labs_atlas` yet. **So until a sponsorship records the site's consent to
Atlas's terms, every reservation is refused, before any upload.** That WebApp
change, which needs the founder's decision on Atlas terms, is a follow-up.

## Configuration

Nothing is defaulted. A missing value refuses before any reservation or upload.

| Variable | Meaning |
|---|---|
| `WORLDLABS_ATLAS_API_KEY` | Project API key with `tasks.create`, `operations.read`, `assets.create` and `assets.read` scopes, plus `assets.delete` for the automatic website preview worker, which deletes every asset a job made once its bundle is archived. It may not contain whitespace or control characters. |
| `WORLDLABS_ATLAS_API_BASE_URL` | The documented `$WLT_API_BASE_URL`: `https://<host>.worldlabs.ai/api/v2`. The docs give no production host, so the value is copied from the platform. |
| `BLUEPRINT_ATLAS_POSE_PRICE_REVIEW_JSON` | The reviewed price (below). Without it: `website_atlas_reviewed_price_required` |
| `BLUEPRINT_ATLAS_PROVIDER_RETENTION` | The owner's decision on where World Labs keeps uploaded frames and outputs. One of `account_does_not_retain` (the account is set not to keep uploads and outputs) or `project_library_retention_approved`. Anything else: `website_atlas_provider_retention_decision_required` |

### The Atlas spend line

Atlas pricing is unpublished. The billing page points to the platform's rate
card, warns that it may change during the beta, and gives a beta conversion of
1,000 credits per US dollar (Marble used 1,250, which does not apply here).
The reservation ceiling is whatever
an owner records after reading the rate card:

```json
{
  "schema_version": "website_atlas_pose_price_review.v1",
  "endpoint": "/api/v2/tasks:images2PosedRGBD",
  "request_options": {"refinePoses": true, "base64Outputs": false},
  "target_resolutions": [[1280, 720], [720, 1280]],
  "maximum_cost_usd": "<worst case for one operation, from the rate card>",
  "max_input_images": "<largest frame count that price covers>",
  "rate_card_reference": "<where the price was read>",
  "reviewed_by": "<owner>",
  "reviewed_on": "YYYY-MM-DD"
}
```

The placeholders are strings, so this example fails validation if pasted
unchanged. The spend line refuses in each of these cases:

- **A field is wrong.** The error names it: `website_atlas_reviewed_price_invalid:<field>`.
- **The review is out of date.** It must be dated today or within the last
  30 days, as `YYYY-MM-DD`, because the beta rate card may change.
- **The ceiling is too high.** The WebApp's route refuses more than $1,000.
- **The options don't match.** The review must cover the request's exact
  options and grid (`website_atlas_reviewed_price_options_mismatch`), with real
  booleans and integers.
- **There are too many frames.** The request may not have more frames than the
  review covers (`website_atlas_reviewed_price_input_count_exceeded`).

How the price was typed does not change the binding: `1` and `1.0` are one
price.

## Request

One operation per admitted binding. The binding covers:

- the pending inputs' digest;
- each frame's identity and image digest;
- the options and the grid;
- the reviewed price;
- the provider identity;
- the retention decision;
- the task context;
- the attempt number.

The body:

- `frames`: the pending frames, in order, each as `{"imageAsset": {"assetId": …}}`
  after a single-object upload (create, `:createUploadUrl`, upload, `:completeUpload`).
  These are the 518-pixel model inputs of the source-frame preparation. Held-out
  pixels are never included.
- `targetResolution`: the documented 1280×720 grid, in whichever orientation
  keeps the most of every view. Atlas scales each view to cover the grid and
  then center-crops it. If any view would keep less than 95% of its pixels, the
  request is refused (`website_atlas_pose_view_crop_too_large`). This applies to
  4:3 or square video, and to mixed orientations.
- `refinePoses: true`, because mask lifting compares views, and
  `base64Outputs: false`.
- An `Idempotency-Key` header derived from the binding.

The capture whose task context is charged must be the capture named by the
caller (`website_atlas_pose_task_capture_mismatch`).

## Lifecycle and files

The state check, the reservation and the submission run under one lock per
capture. Everything lives beside `atlas_pose_inputs.json`:

| File | Written |
|---|---|
| `atlas_pose/controller_admission.json` | After the WebApp grants the reservation |
| `atlas_pose/submission.json` | Before the first upload, and at each later step: when each asset is created (`created_assets`, so no asset holding site pixels is ever untracked), after each upload, after each `READY` check, before each POST (`submitting`, with the exact request bytes, the key and an unanswered entry in `sends`), with each raw answer, and after success (`submitted`, with the operation ID and the API root it belongs to) |
| `atlas_pose/outputs/<frame>/{image,depth,confidence,mask}.*` | Downloaded through fresh `:createReadUrl` links, size-checked against the asset's `sizeBytes` (a bounded HTTP read can end early without an error), fsynced |
| `atlas_posed_rgbd_result.json` | `website_atlas_pose_result.v1`, read by `prepare_atlas_pose_inputs` |

Uploaded assets are confirmed `READY` before the submission. A slow asset is
re-checked on the next pass, never uploaded again.

The result keeps the provider's terminal operation with its signed `url` fields
removed; asset IDs are the record. It adds an `operation_receipt`, which holds:

- the binding and its digest;
- the exact request;
- `input_usage`: every submitted frame, with its frame ID, image digest,
  uploaded asset ID, content type, byte count, and the output files it
  produced and their SHA-256 digests;
- `billing: unreported`, because Atlas operations carry no usage or cost
  fields.

The reader reports `actual_provider_input_count` and
`provider_mutation_performed: true` only for an admitted run, meaning that the
retained WebApp admission and the adapter's completed state both name the
receipt's request. The receipt must also still bind the same frames and output
bytes. Otherwise it raises `website_atlas_pose_receipt_invalid` or
`website_atlas_retained_output_changed`.

Before completion, the reader returns the pending record with the producer's
progress (`atlas_pose_status`). Once frames are uploaded it reports
`provider_mutation_performed: true`. Once a task POST is sent it reports
`actual_provider_input_count` as unknown (`null`), so "nothing sent" is never
claimed after the fact. A result written just before an interrupted pass could
record completion is treated the same way.

Every result keeps `claim_ceiling: development_only`,
`metric_measurement_proven: false`, `physical_evidence: false` and
`scale_status: unscaled_provider_world_units`. The reader's blockers do not
change. A completed operation still reports
`website_atlas_final_scene_assets_required` and
`website_atlas_native_geometry_binding_required`.

## Failures and recovery

Every recovery is code; nothing is resolved by editing state.

| State or error | Meaning | What happens next |
|---|---|---|
| `submitting` (possibly bought) | A send had no definitive answer: a transport failure, 408, 409, 429, 5xx, a refused redirect, or an unusable 2xx. Also any refusal of a later resend, which cannot prove the earlier send bought nothing. | Within an hour of the first uncertain send, the next pass resends the identical bytes under the same `Idempotency-Key`, which is the documented recovery, up to three times. Past that window or limit the key's lifetime is undocumented, so the next pass reconciles from the operation list instead (below). Until reconciliation decides, such an attempt can never be archived. |
| `rejected`, retryable | A 402 (`CREDITS_UNAVAILABLE` or `SPEND_LIMIT_EXCEEDED`), when no earlier send was uncertain | The same bytes are resent after at least five minutes, up to three times (`website_atlas_pose_resend_not_due` before that). An uncertain answer to one of those resends starts its own one-hour window. |
| `rejected`, not retryable | Any other 4xx, when no earlier send was uncertain | `website_atlas_pose_attempt_rejected` |
| Reconciling | An uncertain send past the resend window or limit | `GET /operations` lists every operation the key's project accepted from 15 minutes before the first send to 15 minutes after the last; each one's `:trace` request is compared with ours. Either answer needs a complete listing with no unreadable trace and none carrying a frame this attempt uploaded without our request, since such an operation could be another purchase whose outputs would escape cleanup. One whose task and every field we sent match is **adopted** and collected (the oldest, if a lapsed key let a resend buy a duplicate; the rest are recorded and cleaned up with the job). **Nothing bought** is further concluded only at least 15 minutes after the last send, by a key that can still read a frame of the request (the same project). The attempt then reopens for the identical resend under the same key, which can buy at most that one operation; after two reopenings it closes as `rejected` and can be archived. Anything less raises `AtlasReconciliationPending` and the next pass reads again. Past the 30-day retention (less a day) it fails (`reconciliation_window_lapsed`). |
| Operation not found | A 404 or 410 while polling, within 30 days of the first uncertain send. Eventual consistency, another project's key or another host can all answer that for an operation still running and billed. | `website_atlas_pose_operation_not_found`; nothing changes |
| `failed` | The operation finished with an `error`, or it is still missing after the documented 30-day retention (`website_atlas_pose_operation_gone`) | `website_atlas_pose_attempt_failed`. No result is written. |
| `website_atlas_pose_api_base_changed` | `WORLDLABS_ATLAS_API_BASE_URL` no longer matches the root the operation was created on | Refused before polling |
| Archivable | An attempt that provably buys nothing more: `failed`, `rejected` (before any uncertain send), `uploading`, or a reservation with no submission record | `archive_closed_atlas_pose_attempt` moves it, and any stale result, to `atlas_pose_attempts/NNN/`. The next attempt is a new binding with its own reservation. |
| `uploading` | Interrupted between uploads or `READY` checks | The next pass resumes under the retained admission. The WebApp answers `already_reserved`, and only the exact retained admission turns that back into a grant. A re-reviewed price or other changed binding is refused before any second reservation, and the old attempt can then be archived. |
| `processing` | The operation is still running after `max_polls` | The next pass polls again without a new reservation or submission |
| `website_atlas_pose_admission_expired` | The sponsorship would expire within two minutes | Checked before the uploads and again before the POST |
| `website_atlas_pose_frames_invalid`, `_grid_not_requested`, `_operation_identity_mismatch`, `website_atlas_identifier_invalid`, `website_atlas_depth_not_exr`, `website_atlas_transfer_size_mismatch`, `website_atlas_output_size_unreported` | The returned bundle does not match the request | No result is written. References are checked before the first download. |

The reservation is never settled. Nothing reports what an Atlas operation
actually cost, so the full ceiling stays charged against the sponsorship.

## Not done, because it needs Atlas access, pricing or a decision

See the hand-over list in the PR. In short:

- **Access:** account access, the API root and a scoped key.
- **Price:** a reviewed price.
- **Retention:** a provider-retention decision.
- **WebApp consent:** the WebApp needs to accept `world_labs_atlas`, with each
  site's consent to Atlas's terms. That consent covers unedited frames.
- **First real response:** quaternion order, extrinsics direction, z-depth or
  ray depth, the meaning of `maskAsset`, the per-request frame limit, and
  whether output assets report `sizeBytes`. Only a real response can settle
  these.
- **Settlement:** a usage-based settlement.
- **Entry point:** an operator entry point. It must go through the allocator
  entry, which the verifier enforces.
- **The comparison:** none. The owner removed MapAnything on 2026-10-22, so
  the planned Atlas-vs-MapAnything comparison will not run; website scenes use
  Atlas only ([website-atlas-scene-source.md](website-atlas-scene-source.md)).
