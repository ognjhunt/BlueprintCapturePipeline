# ArtiFixer appearance repair for the fixed-arm rehearsal

This policy unblocks ADP-009D's day-14/day-21 development-only construction and
Franka rehearsal. It grants no collision, physics, physical-trial or deployment
qualification. Native import, controls and the two frozen policies remain
separate downstream gates.

New production scene configurations require registered background support and
frozen source appearance. The immutable SAM3.1/FlashSplat proposal is retained;
its bounded foreground subset is removed, broad/background Gaussians are
preserved, and registered mesh-backed support Gaussians fill the exposed area.
Only the new Gaussians' appearance is trained. Exact geometry, density and
original-appearance checks run at export. Legacy appearance candidates are
refused before the production training runtime can start.

Final review uses `task_fit_object_free_appearance_v2`. Minor grain, brightness
and blending seams are recorded cosmetic warnings. Object remnants, missing
surfaces, large blank holes, damaged context, wrong material/geometry, incorrect
orientation and major cross-view inconsistency remain blocking. A patch boundary
alone is not a false obstacle. Review receipts bind the prompt digest, exact
images, per-view decisions and warnings. Historical rejected reviews remain
unchanged.

A blocking appearance failure can use one correction/retraining round. Select
the highest-priority failing views that fit the remaining stage budget, preserve
all unselected views and explicitly list deferred failures. Re-render and review
the complete set afterward. No extra retry or spend authority is inferred.

A terminal, provider-zero attempt may supply completed training to a successor.
Reuse requires exact initialization, source partition, camera calibration, masks,
source frames, teacher frames, configuration and training settings, plus validated
native export and retained output bytes. A real accepted review may also be
reused only when the complete multimodal input and rubric match. The successor
retains the original execution and an explicit derivative binding receipt;
reusing evidence never claims a new model call or new training run.

Training identity excludes refreshed consent timestamps, authorization references,
and SAM-plan transport references. Permission flags, operation names and all
scientific settings remain bound. Both full configuration hashes are retained in
provenance, and current rights and SAM admission still run before reuse.
