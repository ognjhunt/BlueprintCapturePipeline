# ArtiFixer source admission and completed-training recovery

ADP-009B / day-14 replayable appearance construction, prerequisite to the
ADP-009D two-candidate Franka rehearsal. This change does not complete either gate.

Scene 841757 attempt R21 completed 30,000 training steps and wrote its checkpoint,
then native export refused the unchanged source geometry. The original converted
publisher field and the retained pretraining field both contained 180 centers
near XY 10,000 m. An identity comparison already failed the existing center-to-
robust-diagonal limit: 827.0009 against 128. This was a missing pretraining
admission check, not evidence that optimization moved the geometry.

The GPU closed after failure. The returned archive excluded `artifixer_output`,
which contained the checkpoint. The training log and checkpoint size survived;
the learned weight bytes did not. There is no known recoverable checkpoint from
R21. A failure detectable by CPU replay consumed a paid attempt, and the output
archive also failed to preserve the completed optimization.

The candidate compiler now checks source geometry before teacher/model work.
Only an inherited center-outlier-only refusal can receive a bounded quarantine:
at most 0.1% of rows, with each eight-sigma bounding sphere outside every admitted
padded camera frustum. It preserves the original source, exact excluded rows,
original row indices, calibration, digests, and before/after quality reports.
The initialization copies remaining vertex rows byte-for-byte. This is explicitly
pretraining source conditioning for the development-only task, not learned-tensor
repair, measured pixel equality, or physical evidence. All existing quality and
geometry-freeze guards remain unchanged.

A network-disabled, device-disabled replay as the service user admitted 697,776
of 697,956 retained source rows. All 180 exclusions were outside all 16 cameras;
the minimum bounded-support separation was 5064.4176 m. Identity source-relative
export geometry passed. Replay admission receipt:
`sha256:002c43039a76e3d653aa6dfcbb46f839586c3a47387a125ff633a858426f396a`.
The accepted source/mask/teacher images remain immutable.

The runner now retains the completed checkpoint (including its training config),
reference initialization, request, and training log outside archive exclusions
before export. An export refusal retains its numeric geometry report separately.
Terminal results distinguish completed optimization from a completed task.
The checkpoint remains unqualified until export and independent review pass.

Focused verification covers visible/excessive quarantine refusal, exact row
preservation, the real export adapter's refusal/pass boundary, completed-training
failure archive recovery, candidate and dual-target contracts, runtime bundle
imports, and the policy lifecycle rehearsal. No GPU or model call is part of
these regression checks.

## R22 follow-up

R22 completed 30,000 steps and native export. Its independent post-training
review rejected all 16 views: the book was absent, but the repair was dark and
smeared and surrounding content changed. No configured revision or downstream
controls/policy result was produced. The GPU closed with provider-zero.

The new recovery path preserved the 438,231,861-byte checkpoint, verified at
`sha256:592ed2928f69d6409b2f6f4fcc58d70a17d895aeb9238b7dacd5dc7df8c4c177`.
A CPU re-export using the pinned released exporter reproduced both the original
PLY and USDZ hashes exactly. However, the archive still excluded the original
export files and the exact PNGs consumed by the reviewer. The follow-up retains
native exports immediately after export and normalized review PNGs before the
caller can reject them, for both fresh-training and checkpoint-reuse paths.
The original rejected PNG bytes have not been recovered; diagnostic renders of
the recovered field must not be described as those consumed review images.

## R22 missing background support

The cutout removed 2,791 splats. The exact retained FlashSplat receipt marks
2,518 of those as also contributing to protected background. The selection is
an any-view contribution union, not object ownership: removing a large shared
splat can remove the cabinet front or back panel along with book pixels.
The cutout contract requires subsequent complete deleted-layer repair support,
but the active preparation only supplied the book mask and a 32-pixel margin.
Training then froze positions, rotations, scales, addition and relocation.
This combination asked appearance-only optimization to fill missing geometry.

A source-01 diagnostic with the original retained geometry and black/white
backgrounds found 92.18% of repair-mask pixels fully uncovered at their original
opacity. A counterfactual with every opacity forced to its maximum still left
71.94% uncovered. These are diagnostics of one calibrated view, not the original
review PNGs or a qualification of the other views. Keeping all background-coupled
splats restored the cabinet but left most of the book visible in two inspected
views, so simply switching to the existing conservative classifier was insufficient.

The bounded correction preserves the immutable SAM/FlashSplat proposal and
produces a separately labeled initialization candidate. It removes only proposal
rows whose centers lie within the registered object box plus 5 cm and whose
largest activated scale is at most 8 cm. Other proposal rows remain background
candidates, not newly proven background ownership. Actual triangles of the
registered support mesh supply appearance-only surface samples within the object
XY box plus 10 cm. Their initial colors come from the accepted teacher images.
This does not alter the collision scene or establish physical geometry.

The real-input CPU replay removed 838 local candidates, preserved 1,953 remaining
proposal rows and retained every reused source vertex byte. It produced 699,729
frozen source rows and 224,250 generated surface rows at 1 mm spacing. Every new
point had at least 15 color views. Publisher support bounds and matched SAGE mesh
bounds differ by at most 3.877 mm; the receipt records both and enforces a 1 cm
appearance-admission ceiling. Sampling follows the actual triangles and refuses
incomplete mesh coverage instead of replacing the mesh with an AABB plane.

The new geometry mode freezes the entire declared initialization, including new
geometry and all opacity. Training may change only the generated rows' color/SH
features; full source SH starts active. Scoped gradient hooks leave the released
trainer and source files unchanged. Export independently requires exact original
color tensors and exact full density, position, rotation and scale tensors.
Initialization identity and the source/generated partition survive dual-target
packaging and checkpoint recovery. Completed checkpoints and failures remain
retained if the post-training appearance guard refuses the result.

The initial 3 mm prototype was inspected at all 16 calibrated views on local Metal;
the large void and visible book shape were absent, with remaining patch/texture
differences. The 1 mm initializer is a subsequent untrained candidate. Neither
prototype is an independently accepted appearance result, a native import, or a
completed controls/policy run. The next paid attempt must use the normal immutable
release, budget, watchdog and independent final-review gates.
