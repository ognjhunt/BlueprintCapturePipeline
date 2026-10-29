# Beta L5: retained site-task simulation qualification

The beta recommendation needs a simulator score on the site's measured task.
The existing scored team path remains diagnostic unless the operator registry
pins an evidence-consuming qualification and the WebApp freezes that exact
receipt in its execution admission. No numeric result, boolean, caller-supplied
claim ceiling, successful robot-free probe, or appearance reconstruction alone
creates this qualification.

The current implementation supports a team's declared, verified robot model.
The default DROID provider asset has no retained model-byte proof at this
boundary and therefore stays diagnostic. This is a model-evidence gap, not a
change to which robot forms may apply. Every qualified result remains a bounded
simulation estimate: physical performance and ADP evidence eligibility are false.

## Local command

Use the existing `BLUEPRINT_TEAM_CANDIDATE_QUICK10_REGISTRY` operator registry.
Its ownership, permissions and registry digest are checked by the existing
loader. The command reads the exact evidence pinned on one capture's explicit
operator profile, rehashes the original capture files, recomputes the numerical
checks, and writes a new receipt without overwriting an existing file:

```bash
PYTHONPATH=src python -m blueprint_pipeline.site_task_simulation_qualification \
  --capture-root /absolute/retained/capture \
  --out /absolute/retained/qualification/receipt.json
```

Nothing is published or executed by this command. It does not create a review,
choose tolerances, change the registry, release a credential, or launch a job.
The test fixtures contain synthetic samples and cannot be used as operational
evidence.

Repeating the same command revalidates current evidence and preserves an existing
identical output. Different output bytes that represent a different receipt are
refused. A publication timeout can therefore be retried with the same output
path and idempotent receipt digest without deleting retained evidence.

## Operator authority and source files

Add `site_task_qualification` to the capture's existing explicit operator
profile. It contains these fields:

| Field | Meaning |
| --- | --- |
| `request_path`, `request_digest` | Absolute local request path and RFC 8785 digest of the complete request |
| `profile_path`, `profile_digest` | Absolute frozen fidelity-profile path and RFC 8785 digest |
| `review_path`, `review_digest` | Absolute authorized review path and RFC 8785 digest |
| `allowed_reviewers` | Existing authorized reviewer identities |
| `allowed_measurement_producers` | Authorized independent measurement producers |
| `allowed_simulator_producers` | Authorized simulator measurement producers |
| `canonical_uri_prefixes` | Company-controlled object-storage prefixes, each ending in `/` |
| `receipt_digest` | Exact final receipt digest; required for execution, added after local production |

The receipt's `authority_digest` excludes `receipt_digest` and the three local
locator fields (`request_path`, `profile_path`, `review_path`). Content hashes,
reviewer/producer authority and canonical-storage prefixes remain bound, so
pinning the output or relocating identical retained bytes does not change the
receipt's identity. Scene-derived profiles cannot choose this
authority. Reviewers and thresholds supplied by a customer request are not an
authority source.

The request schema is `blueprint.site_task_qualification_request.v1`. Its
`binding` contains `scene_request_id`, `site_id`, `task_id`, `capture_id`,
`source_capture_digest`, `testbed_digest_sha256`, `scene_revision_digest`,
`task_spec_digest`, `success_contract_digest`, `robot_model_digest`, and
`policy_interface_id`. The latter is the existing endpoint interface ID.

Its `source_artifacts` has exactly one entry for each role: `capture`, `testbed`,
`scene`, `task_spec`, `success_contract`, `robot_model`, `measurement`, and
`simulator_measurement`. Each entry has `role`, `path`, `uri`, `sha256`, and
`size_bytes`. Paths are relative to the request directory and cannot escape it;
URIs must fall under the operator's existing company-storage prefixes. SHA values
are `sha256:` followed by the digest of actual file bytes.

- `capture` is the validated `capture_intake_envelope.v1`. Retain every
  `original_files[].relative_path` alongside it, both locally and under its
  canonical object-storage directory. The qualifier verifies every original
  file's size and digest. The testbed must bind the exact intake ID, envelope
  digest, capture authority profile and source capture digest.
- `testbed` is the existing `maintained_site_task_testbed.v1` JSON. Its approved
  task ID and task distribution must include the requested task. Its digest is
  the existing testbed contract digest. Capture identity/digests retain their
  existing meaning; a capture digest is not assumed to be a manifest-file hash.
- `scene` is the actual `native_task_arena_scene_plan.v1.json` from the prepared
  packet. Its bytes are rechecked against the live staged scene at admission
  and closeout. `task_spec` is the scored scene task specification and keeps its
  existing Python canonical digest. `success_contract` uses the existing task
  contract validator and its RFC 8785 digest.
- `robot_model` is the exact declared model, also matched to the verified model
  staged for the team. It is never inferred from a robot name or picture.

The capture envelope's `scene_id` must equal the binding's `site_id`. The WebApp
also joins the complete receipt to its current capture session, testbed, site,
scene request and approved task before freezing an admission. These are exact
joins, not a permissive search for any matching digest.

## Measurements, frozen criteria and review

Both measurement documents use `blueprint.site_task_measurement_samples.v1` and
contain the exact request `binding`, `producer_id`, UTC `observed_at_iso`, and
`checks`. Each check has `id`, `dimension`, `unit`, and
`samples: [{sample_id, value}]`. Samples must be finite numbers. Independent and
simulator documents must provide the same unique sample IDs for every frozen
check, using the same dimension and units.

The profile schema is `blueprint.site_task_fidelity_profile.v1`, with `checks`.
Each frozen criterion has `id`, `dimension`, `unit`, `minimum_sample_count`,
`maximum_absolute_error`, and `maximum_rms_error`. At least one check is required
for each dimension: `scale`, `registration`, `collision`, `task_state`,
`observation`, and `task_dynamics`. Thresholds must come from the authorized,
task-specific fidelity profile; there are no production defaults. Every paired
sample error is recomputed. Both the maximum absolute error and RMS error must
meet the frozen limits, and all dimensions must pass.

The review schema is `blueprint.site_task_qualification_review.v1`. It contains
`request_digest`, `profile_digest`, `decision: accepted_for_bounded_simulation`,
`reviewed_by`, `reviewed_at_iso`, and `expires_at_iso`. The operator must have
pinned that exact review and authorized its reviewer. Measurement and simulator
producers must differ, and neither may be the reviewer. Measurements must
precede review; review must precede execution and remain valid through its
deadline. A receipt reports the reviewed evidence; it does not establish that
an arbitrary numeric measurement is a valid measurement method.

## Site measurements and fidelity-profile templates

For cohort 1 the site is the measurement producer (founder decision,
2026-10-19). A site records optional tape-measure readings on its job page.
WebApp keeps them with the job. For one request's exact binding, it exports
them as the `blueprint.site_task_measurement_samples.v1` source document:
`POST /api/admin/jobs/:requestId/site-measurements/source-document`, or the
admin job console. The measurement producer is
`site_job:<scene_request_id>`. A Pipeline service identity produces the
simulator measurements, and the founder reviews. Add each site's identity to
`allowed_measurement_producers` when the site is admitted.

`src/blueprint_pipeline/site_task_fidelity_profile_templates.py` holds one
template per scored family, `rigid_pick_place` and `articulated_open_close`.
Each check names the site readings it takes and how the simulator reads the
same quantity. Its thresholds are explicit, each with its basis. Reviewable
copies are in `docs/site_task_fidelity_profiles/`. The threshold-free catalog,
`docs/schemas/site_task_measurement_catalog.v1.json`, is the contract WebApp
mirrors byte for byte. Re-render all three with the module's `render` command
after any template edit.

The templates ship as `pending_review`, and a template is not a profile.
Before any profile is produced, a person records a review in the template's
`threshold_review` block: status `reviewed`, reviewer identity, UTC time, and
the digest that `checks-digest --family <family>` prints. The digest ties the
review to those exact checks, so any later change to a threshold, unit,
minimum sample count or reading makes the review stale until someone reviews
again. A review dated in the future is refused. Then:

```bash
PYTHONPATH=src python -m blueprint_pipeline.site_task_fidelity_profile_templates profile \
  --family rigid_pick_place --task-id <approved_task_id> \
  --out /absolute/retained/qualification/profile.json
```

The command refuses while thresholds are unreviewed. Its output carries the
family, task ID, template digest and threshold reviewer, and follows the
existing path. The operator pins its digest, the reviewer accepts it with the
request, and the qualifier consumes it unchanged.

Before simulator work, check that the site's readings can fill every check:

```bash
PYTHONPATH=src python -m blueprint_pipeline.site_task_fidelity_profile_templates coverage \
  --family rigid_pick_place --measurement /absolute/retained/qualification/measurement.json
```

It exits non-zero and names each missing reading, unexpected sample, unit
mismatch, or mix of hinged and sliding readings. A missing reading still fails
closed at qualification.

A reading used to set a scene's metric scale cannot also check that scale.
Name it as a calibration reading in WebApp's export, which then leaves it out
of every check. Website scenes currently carry estimated metres, so no site
reading calibrates a scene today.

The retained rigid-reading producer is
`blueprint_pipeline.site_task_simulator_measurements`. It calculates all six
dimensions from an exact native-engine readback and calibrated rendered RGB-D;
it never creates the independent site readings or accepts a collider extent as
an observation measurement. Articulated and other family extraction methods
remain unimplemented. The native runtime still needs to emit the exact
metrology snapshot and retained RGB-D described below before this producer can
be used for a real scene. No real measurement or reviewed qualification is
established by its hermetic tests.

### Retained rigid-reading preparation

The existing operator registry may supply `simulator_measurement_producer`,
with `recipe_path`, the recipe's cross-runtime canonical `recipe_digest`,
`evidence_root`, and `producer_id`. This field is operator-only; capture/job
payloads cannot supply it through derived scene profiles. Prepare readings
through the existing qualification entrypoint:

```bash
python -m blueprint_pipeline.site_task_simulation_qualification \
  --capture-root /absolute/capture \
  --prepare-simulator-measurements --out /absolute/retained/simulator-measurement.json
```

This command performs no model, GPU, network, qualification or publication
operation. It preserves an identical result on restart and refuses an existing
conflicting output. `--publish` cannot be combined with this preparation mode.

The pinned recipe schema is
`blueprint.site_task_simulator_measurement_recipe.v1`. It contains the exact
qualification `binding`, `family: rigid_pick_place`, `producer_id`, a
`snapshot: {path, sha256, size_bytes}` record relative to `evidence_root`, and
`readings`. Each reading binds a template `check_id` and `sample_id` to one
explicit extraction:

| Dimension | Operation | Retained source |
| --- | --- | --- |
| Scale | `point_distance` or `horizontal_distance` | Named authored world points in metres |
| Registration | `floor_height` | Named world point and unit-normal registered floor plane |
| Collision | `collision_ray_clearance` | Actual nearest-scene-collision query origin and hit point |
| Task state | `collision_extent` | Exact subject collision bounds and selected scoring-frame axis |
| Observation | `rgbd_length` | Two retained endpoint pixels, RGB, optical-axis depth, and pinhole intrinsics |
| Task dynamics | `rigid_body_mass` | Actual native-engine rigid-body mass attribute |

The original diagnostic snapshot schema is
`blueprint.site_task_runtime_metrology_snapshot.v1`. It binds the same full
qualification identity, `source_kind: native_engine_readback`, `engine_build`,
`runtime_digest`, UTC `observed_at_iso`, and its canonical `snapshot_digest`.
It retains digest/size/path records for the original scene and robot model,
plus `geometry_points_m`, `registered_floor_plane`, `collision_bounds_m`,
`collision_rays`, `rigid_body_masses_kg`, `calibration_sample_ids`, and named
`rendered_rgbd` records. An absent ray hit is a missing bounded measurement,
never infinite clearance.

Operator preparation and qualification recomputation require
`blueprint.site_task_runtime_metrology_snapshot.v2`. Alongside those retained
fields, v2 pins `robot_model_dependency_closure_digest` and
`robot_model_source_import_digest`. The existing operator-selected
`default_robot_model` import authenticates both pins. Its root model and every
dependency are reopened through the existing descriptor-safe verifier and
retained as exact source records in the measurement extraction. A changed
dependency, reviewed closure or import generation refuses even when the root
model file is unchanged. The emitted v2 extraction records the full closure
and import digest, with `runtime_open_proven: false`: retained-byte verification
does not establish which asset an engine opened. This preparation currently
uses the existing DROID default-model operator import; it does not add another
robot family or qualification profile.

Direct v1 extraction remains diagnostic and explicitly marks robot dependency
evidence unproven. It cannot fall back into operator qualification preparation.
Old numeric documents must not be relabelled v2 without the actual retained
closure and original snapshot producer binding. Runtime snapshot emission,
staging and actual engine-open evidence, independent paired site readings and
reviewed thresholds remain separate pending obligations.

Each render record retains its artifact, `model: rectified_pinhole`,
`depth_quantity: optical_axis_m`, `renderer_digest`, and `camera_pose_digest`.
Its lossless NPZ contains only `rgb` (uint8 H×W×3), `depth_m` (floating H×W),
and `intrinsics` (3×3). Depth is unprojected at the frozen endpoint pixels;
missing/invalid depth refuses. Array headers and decompressed byte counts are
bounded before allocation. A reading used for calibration cannot be reused in
any qualification check. No inferred scale, material or threshold fallback
exists.

The emitted existing measurement document also retains extraction identity,
recipe and implementation digests, every raw artifact digest, and separate
authored/collision/rendered/engine provenance. When the qualifier consumes
this generated document it reopens the operator-pinned recipe and original
sources and recomputes the readings, including at later admission/closeout.
A metrology snapshot's scene artifact must match the reviewed scene artifact's
exact SHA-256 and byte count; retaining the same revision label is insufficient.
A numerical document whose raw render has changed cannot retain its prior
qualification. Existing reviewed evidence without this producer's extraction
record retains its compatibility path.

### Blueprint default robot bytes

An operator may select the default DROID robot through `default_robot_model`:
`schema_version: blueprint.default_robot_model.v1`, the exact DROID
`embodiment_preset_id`, the absolute prepared `directory`, `model_sha256`, and
`dependency_closure_digest`. The directory must pass the existing
`deployment_embodiment_asset_import.verify_prepared_asset_directory` checks;
an asset name or downloaded root layer alone is insufficient.

`blueprint_robot_model_delivery.stage_default_robot_model` reuses the existing
closed-USD copier. It retains every dependency and adopts only byte-identical
staged content. The qualification authority must pin
`default_robot_model_closure_digest` to that same closure; qualification and
`require_current_qualification` recheck all staged model/dependency bytes with
anchored directory descriptors. Local DROID interface IDs are accepted by the
portable qualification validator alongside the existing admitted HTTPS
interfaces. Team model behavior is preserved.

This closes a local staging/qualifier seam. Provider bundle staging and the
runtime must still open these exact retained bytes and carry the same closure
through actual execution and closeout before a default-robot execution is
claimed qualified. No configured production model, runtime emission, genuine
paired site measurement, human threshold/review, or real execution is supplied
by these changes.

## Receipt, publication and execution

The full output schema is `blueprint.site_task_simulation_qualification.v1`.
`registration_receipt` contains scale, registration and collision checks;
`fidelity_review_receipt` contains task-state, observation and task-dynamics
checks. Both bind the exact request/profile, source measurement bytes,
simulator measurement bytes and full execution identity. Each numerical result
retains its sample count, observed maximum/RMS error and frozen thresholds.
All new JSON digests use RFC 8785, excluding only their own digest field.

The portable `qualification` projection has only `evidence_scope: simulation`,
`capture_id`, `scene_revision_digest`, `registration_receipt_digest`, and
`fidelity_review_receipt_digest`. The full receipt additionally retains original
source URIs/byte hashes/sizes, authority and review hashes, reviewer and dates.
It contains no local source path or credential. A source URI alone does not prove
remote accessibility; inspect authorized canonical storage and preserve local
bytes before treating a real evidence package as recoverable.

Explicit publication uses the existing protected token file and signed HMAC
transport, after the matching WebApp route has been deployed:

```bash
PYTHONPATH=src python -m blueprint_pipeline.site_task_simulation_qualification \
  --capture-root /absolute/retained/capture \
  --out /absolute/retained/qualification/published-receipt.json \
  --publish --webapp-url https://your-existing-blueprint-webapp \
  --token-file /absolute/existing/protected/pipeline-sync-token
```

The publisher requires the WebApp acknowledgement to return the same
`receiptDigest` and a boolean `created`. The WebApp retains the immutable full
JSON in its existing database, keyed by the receipt digest. Local JSON and the
source object-storage references remain the portable recovery route.

The WebApp freezes the full receipt and five-field projection in
`execution_admission.site_task_qualification_receipt` and
`execution_admission.site_task_qualification`. Pipeline admission rechecks the
operator pin, original bytes, exact current scene/task/model and expiry before
credential release. Closeout repeats the checks before sealing a qualified
result. The existing worker inputs and session results retain their diagnostic
claim ceiling; only the joined outer result can state `site_task_simulation`,
and only for the full scored matrix with the existing void and scoring rules.
Missing or stale evidence never upgrades a diagnostic result.

## Native rigid/DROID readback leaf

`site_task_native_metrology` adds an opt-in readback method on the existing
`NativeTaskArenaEnvironment`. It never constructs an environment, advances
physics, allocates a provider, or creates an evaluation stage. The selected
model option changes only the existing DROID spawn asset after the existing
operator import/closed-byte verifier runs. It preserves the DROID control,
joint-reset and camera configuration. The approved journey still has one
bounded evaluation path; no separate finalist tier is introduced.

The selectors bind the exact sealed scene-plan digest. Readbacks keep authored
world geometry, an actually loaded planar floor, loaded USD collision geometry,
native PhysX closest-hit results, native rigid reset pose, native mass/material
attributes and synchronized rendered RGB-D distinct. Loaded USD bounds are
explicitly not described as cooked PhysX AABBs. Missing native mechanisms,
ambiguous appearance features, changed render/reset generation or invalid
source units return typed `unproven` states. Task-request dimensions or mass
are never copied into observed values.

The range-to-optical-axis conversion follows the pinned Isaac Lab source:
[camera quantities](https://github.com/isaac-sim/IsaacLab/blob/ffff603eafc6b74264a5261cc0183d6a65390d78/source/isaaclab/isaaclab/sensors/camera/camera.py#L91)
and [orthogonalization](https://github.com/isaac-sim/IsaacLab/blob/ffff603eafc6b74264a5261cc0183d6a65390d78/source/isaaclab/isaaclab/utils/math.py#L1087).
The capture retains original depth values, actual intrinsics, converted optical
depth and a validity mask. Missing depth uses an explicit zero sentinel in the
derived array; a selected feature endpoint without measured depth refuses.
RGB-band feature endpoints are extracted from actual RGB pixels independently
of collider dimensions. This bounded extraction method is not a universal
feature detector or an accepted appearance-alignment threshold.

`retain_native_metrology` writes private, exclusive, fsynced raw capture/NPZ
artifacts using an anchored directory descriptor. It does not overwrite or
remove an interrupted attempt. These raw captures are not qualifier-ready
site measurements, an independent site-measurement source, or a replacement
for the existing review/receipt contract.

The model readback compares the actual native spawn path, live robot view,
observed loaded USD layer and reverified closed dependency bytes/import
generation. It deliberately keeps `runtime_open_proven=false`: those checks
do not prove the exact dependency asset streams consumed by the engine.
Native byte-open evidence, the authenticated source/capture/testbed join,
existing caller admission/closeout wiring, a permitted real native run,
independent paired site measurements and reviewed thresholds remain required.
Offline handle stubs establish adapter/refusal behavior only.
