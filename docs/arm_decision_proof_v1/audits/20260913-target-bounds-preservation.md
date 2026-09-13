# Preserve neighboring geometry during ArtiFixer preparation

ADP-009B public-scene removal/inpainting prerequisite for the day-21 transition. The required completion artifact remains an independently reviewed, object-free scene with non-target content preserved.

The trained candidate from attempt `source-b1abf8d7d17a565114b6f346` kept the white bottle in its source and teacher images but substantially lost it in the 3D render. The background initialization selector enlarged the recorded vase bounds (about 8 cm wide) by 5 cm on every side, then deleted small source Gaussians in that enlarged volume. Geometry, opacity and original appearance were frozen afterward, so training could not restore those removed neighboring primitives.

The smallest correction is to use the recorded target bounds without a deletion margin. Keep the existing scale filter, immutable source partition, registered background surface, and independent post-training review. Source Gaussians outside the target bounds are restored exactly rather than being authorized for deletion by a repair band.

Saved-input replay on the actual 8,844-point segment candidate selects 78 points instead of 178, restoring 100 original source points. Under the recorded calibrated cameras, the old selection projects 57 centers into an inspected interior region of the white bottle in source-01 and 35 in source-08; the corrected selection projects zero into either region. These are center-projection diagnostics, not a claim of complete visual or 3D qualification. The next run must still verify target removal and neighboring-object preservation in its actual renders.

Verification: the background-initialization and production-driver suites pass (46 tests). The regression places neighboring splats 3 cm outside each face of an 8 cm target; the old 5 cm margin deletes them and the corrected selector preserves them. Existing tests retain exact source-row copying, registered-support constraints, and training/review contracts. No provider or GPU call is required for this replay or these tests.
