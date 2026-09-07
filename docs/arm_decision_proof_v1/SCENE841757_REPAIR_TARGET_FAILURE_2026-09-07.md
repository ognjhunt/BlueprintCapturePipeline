# Scene 841757 repair-target failure

This repairs the ADP-009B construction precursor to the day-14 gate. Completion
requires a replayable object-free appearance result with independent visual
review; passing software tests alone does not qualify the scene.

The R11 run exposed two input failures. Seven of sixteen calibrated views had
empty SAM masks despite a visible book. Treating those views as preservation
targets left the book intact. In the other views, a tight repair mask and inner
feather could restore source book edges even when the raw generated candidate
removed them. The upstream SDK review assessed track identity and contamination;
it explicitly did not qualify per-view segmentation completeness. The appearance
run also used the explicit paused/ungraded review mode.

The driver now refuses empty repair support. The SAM input bridge derives a
separate candidate repair core from the existing calibrated object support and
SAM mask, with a 32-pixel repair margin. Raw SAM ownership and FlashSplat evidence
remain unchanged. Compositing retains generated pixels at full opacity over the
core and preserves source pixels outside repair support. The calibrated region
and margin are candidate repair allowances, not observed segmentation truth.

Required review mode now reviews the actual composited targets before training,
using the independent Agents SDK reviewer. Its rubric includes fragments, covers,
pages, outlines, shadows, material changes, seams, and protected-object damage.
A failed input review stops training. The review allowance is split between
pre-training and post-training review without increasing the existing cap.

Raw historical image edits can be selected using
`semantic_teacher_candidate_reuse.materialize_retained_selection`. The resulting
`semantic_teacher_retained_candidate_selection.v1` binds the original request,
result, image bytes, task, camera, and source-frame digest. The bundle accepts
`--retained-candidate-selection` (or the control-plane environment variable
`BLUEPRINT_SCENE_CONFIGURATION_RETAINED_CANDIDATE_SELECTION`). A source mismatch
fails before bundle admission. The runtime constructs a new request, retaining
both original and current mask identities. Reuse never becomes a new model call
or a new charge. The checkpoint retains historical receipts and counts reused
frames separately. Selective repair always requests fresh edits for failed views.

R11 candidates remain unqualified. The retained selection is subject to fresh
visual review under corrected support. This change does not establish complete
3D removal, configured-scene qualification, a successful policy evaluation, or
physical truth.

Focused verification covers repair-core opacity and source preservation,
corrupted and mismatched reuse evidence, new-call accounting, checkpoint lineage,
bundle transport, driver rejection before training, and selective repair. The
policy lifecycle rehearsal and provider import-closure tests protect the later
paid runtime boundary. No repository-wide suite is required for this bounded
experimental repair.

The first real host reuse replay refused before allocation because original
render PNG bytes differ from the RGB PNG re-encoding sent to the editor. Bundle
admission now verifies the retained render bytes and reproduces the existing
RGB staging before comparing the original request hash. Changed pixels or
unbound source bytes still refuse. A service-user replay admitted all six
selected R11 candidates against the retained render inputs without a model call.

R13 reached the allocated worker, then refused before image generation or
training because the provider envelope hydrator did not rebase the newly added
repair-support and object-core paths. Those files were present in the sealed
bundle, but the component runs with the toolchain directory as its working
directory. Retained candidate records had the same missing handoff. Both
provider runners now verify and rebase all five added file roles. Tests exercise
the component working directory and corruption of each role.

This was a process defect: the previous CPU replay used host-absolute paths and
did not reproduce provider hydration followed by the component working directory.
The correction must be replayed from the exact R13 ZIP through the image-request
boundary in an isolated CPU process before another paid attempt. R13 failure and
teardown evidence remain preserved; no new image candidates were produced.

### Controls catalog reference after a budget-cap installation

ADP-009, day-14 construction rehearsal: the R14 no-spend provisioner refused
`public_scene_provision_immutable_conflict` after the controls phase cap was
reduced from USD 2 to USD 1.75. The canonical installer changed the content
catalog, while public-scene machinery retained its prior exact-byte reference.
No R14 intent or provider resource was created by the refusal.

Use `python -m blueprint_pipeline.task_evaluation_public_scene_machinery_refresh`
with explicit machinery/catalog paths, source commit, and expected machinery
digest. Preview is the default; `--apply` archives the exact previous machinery
bytes and atomically refreshes only the catalog reference and machinery seal.
It validates the new catalog and retained runtime/assets, refuses stale expected
identity or concurrent changes, and preserves file ownership/mode. Existing
intent and attempt snapshots are never rewritten. This operator command is a
configuration maintenance action, not an execution launcher or a scientific claim.

`tests/test_task_evaluation_public_scene_machinery_refresh.py` verifies exact
archive preservation, idempotence, unchanged unrelated fields, and refusal of
stale identity, corrupt catalog, changed runtime assets, and a different catalog
path. The completion artifact is the refresh receipt plus resumed canonical
provisioning; successful appearance or policy evaluation still requires its own
execution evidence.

### R14: phase-appropriate target review and bounded recovery

ADP-009 / day-14 construction rehearsal. R14 passed the repaired provider handoff,
reused six edits and produced ten new ones. Independent review found the source
book absent in all 16 views, but rejected 15: mostly seams/texture differences,
plus an incorrect floor-like replacement in source-07. The GPU was destroyed and
provider-zero confirmed. No training or appearance qualification occurred.

The pre-training path had reused the final-appearance prompt (reject any visible
seam) and immediately aborted on any rejection. It now uses a distinct training
admission standard: reject object remnants, wrong materials/geometry, collateral
changes, orientation errors and major inconsistency; retain minor seams as
warnings in the rationale. A training acceptance has its own receipt type and
cannot seal final appearance. The final rendered-image criterion is unchanged.

One exact-mask corrective image-edit round can run before training, with the
reviewer's camera-specific feedback, preserving accepted sealed images exactly.
If a view remains rejected, a development-only selection may exclude its teacher:
at least eight and 75 percent of the original views must remain approved, with
distinct calibrated poses and two approved axes within 30 degrees of each omitted
view. This is an explicit training coverage heuristic, not a geometry or fidelity
proof. The full original final-review trajectory remains required.

The rejected teacher slot becomes a byte-exact original observation with its
outside-support anchor loss mask; the rejected generated pixels never enter
teacher staging. Keeping that masked observation slot preserves the released
training loader's camera indexing. Bundle and provider validators check the
partition, original bytes, masks, poses and review-bound selection. Final review
still includes every camera. Only one semantic correction round is allowed in the
whole stage; an exclusion cannot later be undone by merging the old teacher set.

Three reviewer reservations now fit inside the unchanged USD 6 external-services
cap: USD 4.80 semantic edits + USD 0.96 review + USD 0.20 content = USD 5.96. Each
image transport retains retry_count=0; the single corrective operation is bound
to a new feedback-bearing request and the remaining stage allowance.

The continuation also supports a binding-scoped `retained_prefix_only` preparation
mode. It creates the existing zero-cost, non-allocating preparation identity and
requires a verified prefix through segment_cutout before any preparation profile
or submission can be published. Missing/partial reuse cannot fall back to fresh
GPU source work. Other source bindings keep their paid-source behavior. This
avoids reserving USD 4.50 for already-completed source GPU stages while preserving
the original owner budget and all fresh admission checks.

Focused verification covers per-view repair/exclusion, insufficient or uncovered
views, rejected pixels absent from the real teacher staging path, anchor masks
excluding the original object, all final-review cameras retained, separate final
appearance authority, scoped no-spend source preparation, and cold-source refusal.
Provider import closure and the policy lifecycle rehearsal remain required before
paid continuation. The new live model decision and final 3D outcome remain unproven
until their execution receipts are produced.
