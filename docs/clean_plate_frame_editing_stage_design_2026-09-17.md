# Clean-Plate Frame-Editing Stage — Design (extract → Atlas)

- **Status:** PROPOSED — pending owner confirmation before the consistency machinery is built.
- **Date:** 2026-09-17 (dated snapshot; see `docs/DOCTRINE_PRECEDENCE.md`).
- **Program:** `arm-decision-proof-v1`. **Backlog item (proposed):** `ADP-009B`
  (synthetic removal → inpainting → replacement), as the **own-capture,
  pre-reconstruction analog** of the public-splat edit path. **Field consumer:**
  `ADP-021` fresh-capture (registered clean-background + object-present).
- **Claim ceiling:** `development_only`. This stage conditions a reconstruction
  *input*; it never produces metric, collision, hidden-surface, or physical
  authority.

## 1. Problem and first principles

Atlas/WorldLabs reconstructs by fusing many frames of a **static** world. Two
kinds of content corrupt the fixed stage:

- **People / anything that moved** → multi-view reconstruction assumes the scene
  didn't move, so movers become ghosts/floaters (plus a privacy problem).
- **Manipulable task objects** (totes, bins, parts) → we do **not** want these
  baked into the immovable stage. They are rebuilt separately as individual,
  physics-ready 3D assets (image→3D via **`gpt-6-astra`**) from the **original**
  frames and composed back at their real pose.

Removing them in 2D **before** reconstruction beats carving them out of a
finished 3DGS: the existing repo removal family (`public_scene_gaussian_excision_*`,
`gaussian_object_partition`, `source_collider_subtree_removal`,
`articulated_excision_join`) all operate **post-reconstruction** on a finished
splat/USD — the expensive, ghost-prone path we are avoiding. That 3D-edit path
exists because the public reference scene (InteriorGS `840313`) ships only a
finished splat, not frames. **For Blueprint's own captures we own the frames**,
so we fix them upstream and the 3D-edit workaround is unnecessary.

## 2. Why this is admissible under the north star (not "unrelated new work")

The `scene_edit_contract` already governs object removal + background completion
+ SimReady replacement. This stage moves that seam earlier and honors the same
rules verbatim:

- **`preferred_background_rule`** — *use observed clean background or source-view
  coverage before generation.* Primary fill is **real observed pixels reprojected
  from other frames**; generation is a labeled fallback only. This is also the
  operator "film the empty space" option in the field (no manifest flag; inferred).
- **`hidden_surface_rule` / `generative_editing_rule`** — any generated fill is
  **`visual_candidate_only`**, never metric/collision/hidden-surface authority.
- **Measured blocker** (required by non-goal
  *"new reconstruction research without a measured blocker"*): a with/without
  reconstruction comparison on one real capture (§7), gating the default flip.
- **Keep raw capture truth**: originals are never mutated; the Astra rebuild is
  provenance-bound (`sha256`) to them.

## 3. Where it slots in

Reconstruction is behind a provider abstraction; **"Atlas" = the WorldLabs
Marble preview provider** (`world_labs` / `world_labs_marble`,
`provider_preview.py::run_preview_provider`), which consumes a **video**.

**Primary seam — orchestrator WorldLabs/Atlas path**
(`site_package_orchestrator.py::run_qualification_pipeline`):

```
… run_privacy_postprocess (:4714)          # people already removed (VIP video-inpaint) + verified
   _prepare_worldlabs_input_video (:4759)   # privacy-safe walkthrough → pipeline/worldlabs_input/worldlabs_input.mp4
   ── NEW: run_clean_plate_stage ──         # remove movable objects, view-consistently → clean_plate video
   write_blueprint_canonical_site_package (:5187)
   run_preview_provider(world_labs, …) (:5230)   # ← Atlas reconstruction
```

The stage consumes the privacy-safe WorldLabs input, and **when enabled and
non-empty** rewrites `metadata_payload["worldlabs_input_video_uri"]` to the
clean-plate video before the provider submit. When disabled / blocked / empty it
is a **safe no-op**: the privacy-safe video flows through unchanged, exactly as
today.

**Ordering vs. privacy (deliberate):** people-removal is *already done* upstream
by `privacy_processing.py::run_privacy_postprocess` (SAM3 detect → VIP
video-inpaint → SAM3 verify → DeepPrivacy2 fallback → `world_model_video_uri`).
This stage **does not re-implement person removal**; it *defers to and verifies*
it (require privacy `status ∈ {no_people_detected, person_removed,
face_anonymized_fallback}`; hard-stop on `failed_closed` or any residual person
detection) and spends its own new machinery only on **movable task-objects**.
Person handling here is a fail-safe cross-check — "align with, not bypass; fail
safe."

**Secondary seam (optional, later):** the discrete-frame local-splat path —
a `_PROFILE_STAGES` entry (`capture_reconstruction_routing.py`) after
`compile_frozen_frame_dataset` and before `train_gaussian_reconstruction`,
editing `candidate_dataset/**/*.png`. Same core; different frame source. Out of
scope for the first build unless requested.

## 4. The analysis → removal contract

A **separate** agentic Gemini video pass (the web app's `capture-coverage`
deliberately never describes people/targets — *"that is a separate review's
question"* — so this fills a real, non-overlapping gap). It uses Gemini's
**agentic video** processing (`processing="agentic"` on a Gemini 3.x Flash
model; see the
[agentic-video announcement](https://blog.google/innovation-and-ai/models-and-research/gemini-models/introducing-agentic-video-in-gemini/)),
so the model searches/seeks the walkthrough and **time-localizes distinct
objects across a multi-minute pass** rather than judging a fixed 16-frame
sample — the right shape for enumerating removal targets. Model and processing
mode are env-configurable (`BLUEPRINT_GEMINI_CLEAN_PLATE_MODEL`,
`BLUEPRINT_GEMINI_CLEAN_PLATE_PROCESSING`). It follows the gated paid-Gemini
precedent `wam_generated_video_success_label_gemini.py` (versioned prompt +
`PROMPT_TEMPLATE_SHA256`, provider-error mapping) with a **fail-closed gate**
`BLUEPRINT_ALLOW_GEMINI_CLEAN_PLATE_ANALYSIS` (collect blockers → skip the paid
call → `status="blocked"`). New per-provider module, not a new abstraction. This
directive — agentic video for the analysis — is scoped to *this stage's* video
analysis; migrating other repo Gemini passes (e.g. coverage) to agentic video is
a separate, non-blocking follow-up, not part of this change.

Output = **removal plan** `clean_plate_removal_plan.v1` (a fork of the existing
`public_scene_removal_selection.v1`, dropping the post-reconstruction USD/collider
fields `source_collider_prim_path` / `collider_deletion_id`). Per target:

| field | meaning |
| --- | --- |
| `target_id` | stable identity/label across frames |
| `semantic_label` | e.g. "cardboard tote", "person" |
| `target_class` | `person` \| `movable_object` \| `fixed_clutter` |
| `disposition` | `remove` \| `keep` (+ `reason`) |
| `rebuild_intent` | `rebuild_and_compose` \| `none` |
| `spatial_evidence` | per-frame boxes/regions + timestamps (Gemini), lifted to per-frame masks; `mask_track_ref` once SAM 3.1 tracks exist |
| `confidence` | model confidence |

**Defaults:** always remove **people** (privacy + noise) — but *deferred to the
privacy pipeline and verified here*; remove **movable task-objects** when
`rebuild_and_compose`; **keep fixed clutter** genuinely part of the environment.
**Clean-vs-cluttered is inferred** from the plan (no movable targets ⇒ near
no-op), never from a manifest flag (the shared contract carries none, and we must
not add one).

**Masks:** reuse the **SAM 3.1 source-track lane**
(`scene_placement/sam31_source_track_provider.py`, `facebook/sam3.1`, 16-slot
multiplex, `sparse_probability_rle.v1`) to lift regions+timestamps into per-frame
masks tracked across frames — today it already produces exactly this, only
consumed post-reconstruction. Materialize exact per-frame binary masks via the
`public_scene_calibrated_object_mask_set.v1` bridge, retargeted to capture frames.

## 5. The view-consistent removal approach (the crux — highest risk/cost)

**Do not** inpaint each frame independently (e.g. per-frame GPT Image 2.5) —
independent fills disagree across views and Atlas fuses the disagreement into
ghosts/blur exactly where we cleaned. Instead, a two-tier fill:

1. **Observed-background recovery first (factual, view-consistent by
   construction).** The camera moves, so the background behind a removed object
   in one frame is usually *directly filmed* from another viewpoint. For each
   masked region, warp in the real observed pixels from covering frames
   (optical-flow / geometry-consistent reprojection). Because every frame's fill
   is the *same* observed pixels reprojected, the fills agree across views — and
   they are factual, satisfying `preferred_background_rule`.
2. **Video-consistent generative fallback (labeled candidate).** For regions
   never observed in any frame (persistent occlusion), use a temporally-consistent
   **video** inpainter — the **VIP-family infrastructure the privacy pipeline
   already uses for people**, retargeted to object masks — not independent image
   fills. Output is tagged `visual_candidate_only`.

This is the RePaintGS / VEIGAR / GScream philosophy (reference-guided,
geometry-consistent removal) applied in **2D/video, pre-reconstruction**. Every
filled pixel is tagged observed-recovered vs generated-candidate; outside-mask
pixels are byte-exact untouched (reuse the
`public_scene_calibrated_exact_segment_repair_preflight.v1` "outside the mask
restored byte-for-byte" contract). Any released-code inpainter goes through the
existing rights gate `public_scene_released_code_inpainting_admission.v1`.

## 6. Artifact layout

Originals are **never** mutated (`raw/frames/`, raw walkthrough remain source of
truth; `raw_retained` stays true). New artifacts live under
`scenes/{sceneId}/captures/{captureId}/pipeline/clean_plate/`:

| artifact | schema | purpose |
| --- | --- | --- |
| `clean_plate.mp4` | — | cleaned video for WorldLabs/Atlas (primary seam) |
| `frames/{frame_id}.png` | — | cleaned frames (discrete-frame seam, if used) |
| `removal_plan.json` | `clean_plate_removal_plan.v1` | §4 analysis output |
| `removal_manifest.json` | `clean_plate_removal_manifest.v1` | what was removed, from which frames, `mask_track_ref`, per-target observed-vs-generated fill provenance, `observed_bounds_world_m`, and a **compose-back link slot** (`replacement_asset_id` + `pose_world`) filled later by the Astra rebuild |
| `clean_plate_stage_manifest.json` | `clean_plate_stage_manifest.v1` | receipt: `status`, `mode` (`noop`\|`objects_removed`\|`blocked`\|`failed_closed`), consumed `privacy_processing_manifest` digest, before/after pointers, `program_id`/`adp_item`/`day_gate`/`claim_ceiling` + boundary booleans `False` |
| `current.json` | — | immutable "current pointer" (repo stage convention) |

The `removal_manifest` is the bridge to the existing rebuild/compose-back chain:
`object_index.v2` / `object_geometry_manifest` (discovery) → `task_object_astra_authoring`
(rebuild from `observed_source` originals) → `replacement_asset_frame_registration.v1`
(pose) → `task_object_astra_native_adoption` (place at `pose_world`).

## 7. Validation plan

1. **Measured before/after on ONE real capture** (the required measured blocker):
   run Atlas reconstruction **with** and **without** the clean-plate stage;
   compare (a) reconstruction quality in cleaned regions (ghost/floater/blur
   reduction, held-out appearance), and (b) that removed objects can still be
   rebuilt from the **originals** (Astra authoring) and re-placed at real pose
   (compose-back). Only flip the default **after** the comparison shows a win.
2. **Hermetic fast-lane test** pinning the contract (per repo rule: every fix
   lands with a hermetic test + fail-closed gate): schema validity, the
   fail-closed gate (`BLUEPRINT_ALLOW_GEMINI_CLEAN_PLATE_ANALYSIS`), the privacy
   deferral/verify, the **keep-originals invariant**, and a deterministic
   validator rejecting any `claim_ceiling` elevation or `program_id`/`adp_item`
   mismatch (mirroring `public_scene_task_selection.py`).
3. **Flag default OFF**; the §7.1 comparison is the gate to change it.

## 8. Flag, safety, and fail-closed behavior

- `@dataclass(frozen=True) CleanPlatePolicy.from_env()` + `.to_dict()` embedded in
  every emitted record (idiom from `world_model_policy.py`); flag
  `BLUEPRINT_CLEAN_PLATE_ENABLED` (default `False`, opt-in-stage pattern like
  `BLUEPRINT_TASK_EVALUATION_AGENT_SUPERVISOR_ENABLED`).
- **Fail closed** → emit `blocked`/`failed_closed`, substitute **nothing**,
  reconstruction proceeds on the privacy-safe video (today's behavior) when: gate
  env unset, privacy `failed_closed` or residual person detected, paid analysis
  unavailable, masks empty/low-confidence, or any stage error. Never a silent
  degraded pass.
- **No shared-contract change:** `raw/manifest.json` is untouched; all new
  artifacts are pipeline-internal under `pipeline/clean_plate/`. An explicit
  "cleared" signal, if ever wanted, is a coordinated web-app manifest addition to
  request separately — not assumed here.
- If the fallback produces generated/edited media released beyond the pipeline,
  it must pass `generated_media_privacy_contract.v1` first.

## 9. Open decisions for the owner (confirm before building §5)

1. **Build now vs. sequence after ADP-009D.** Backlog says fresh-capture feature
   work "stays at zero" until after the Franka rehearsal. This stage is
   `development_only` and can be built now as an ADP-009B-aligned seam, or held as
   design-only. Owner's call.
2. **Backlog tag.** Proposed `ADP-009B` (+ `ADP-021` as field consumer). Confirm
   or retag.
3. **Seam.** Recommend the **WorldLabs/Atlas video seam** (primary) with a
   provider-agnostic core; discrete-frame seam deferred. Confirm.
4. **Phased build.** Recommend: build the **contract + scaffold first** (Gemini
   analysis → removal plan, SAM 3.1 masks, artifact layout, wiring, flag
   default-off, hermetic test) and **defer the view-consistent fill machinery
   (§5)** until the design is confirmed and the one-capture comparison is set up —
   that is where the risk and cost concentrate.
