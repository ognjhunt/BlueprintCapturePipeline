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
