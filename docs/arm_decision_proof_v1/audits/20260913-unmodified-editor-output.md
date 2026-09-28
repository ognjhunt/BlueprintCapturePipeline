# Preserve editor output and generation-mask lineage

ADP-009B public-scene removal and inpainting prerequisite for the day-21 transition.
The completion artifact is an independently reviewed object-free scene; this fix alone does not qualify that artifact.

## Observed production failure

Attempt `source-b1abf8d7d17a565114b6f346`, release `4f42ab846ddcfd6b8e1f8cf88f8132e2cec7370a`, Vast instance `50924565` completed 30,000 ArtiFixer iterations and exported its checkpoint and 16 renders on 2026-09-13. Independent post-training review rejected all 16 for a vase-shaped patch/false edge. Human inspection also found a substantially missing neighboring white bottle in the trained representation.

Source-01 and source-08 were compared at three boundaries: untouched editor output, actual paired training target, and trained render. The untouched editor outputs remove the vase cleanly. The pipeline's exact-SAM compositing restores source edge/shadow pixels around the replacement and creates an outlined patch in the training targets. The neighboring white bottle remains in both sampled source/teacher image pairs but is damaged in the trained render. Its 3D preservation failure remains a separate unresolved issue.

The sealed pretraining capsule also confirms that ten retained edits came through earlier reuse receipts whose original generation mask differed from the current mask. Immediate-parent request equality hid this mismatch. The former loader overwrote `original_edit_mask_sha256` on each reuse hop, so even a matching legacy parent cannot establish the generating request's mask.

## Change

- Production uses SAM solely as editor object guidance and passes the returned image bytes unchanged into independent pretraining review and paired-target training.
- The same policy applies to selective repairs; the repair prompt lets the editor remove the object's boundary, halo and shadow while preserving neighboring objects.
- Source images, SAM masks and editor outputs remain separately digest-bound. Receipts report real outside-mask differences and explicitly avoid claiming exact source-pixel preservation when it did not occur. Gross change checks and independent pre/post-training reviews remain active.
- Historical exact-support compositing remains available for retained replay, rather than silently reinterpreting old receipts.
- Reuse carries a generation-mask binding unchanged across retries. Legacy reuse without that binding is skipped; original direct-generation receipts remain reusable when the actual mask and encoding match. Both automatic discovery and the runtime loader enforce this.
- Post-training review explicitly compares non-target silhouette, height, opacity and shape with the source anchor; frozen tensors are not evidence of visual preservation.

## Verification

Focused candidate-discovery/image-worker/lineage tests exercise original-generation reuse, legacy and changed-mask/encoding refusals, base/repair discovery, and repeated reuse. Focused locality/repair/driver tests prove complete editor bytes survive both passes, original mask bytes remain bound, outside changes are recorded truthfully, and the independent review path remains active. No provider call or GPU allocation is made by these tests.

This is a producer/consumer correction based on saved production evidence. It does not claim that the running old-release retry incorporates the change or that fixing image compositing repairs the separate 3D bottle loss.
