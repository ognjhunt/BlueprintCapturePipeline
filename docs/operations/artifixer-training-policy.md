# Production ArtiFixer training policy

This is the ADP-009B appearance repair needed for ADP-009D's day-14 two-policy
rehearsal and day-21 sealed results. It remains development-only evidence.

New autonomous scene configurations use `artifixer_training_policy` set to
`corrected_only_local_appearance`. Omitted/null values also resolve to this
default. The explicit `masked_original_anchors` option preserves the historical
training contract for deliberate comparisons; unknown values refuse before paid
image preparation. Neither a missing setting nor a failed new run falls back.

The corrected-only route uses only reviewer-admitted, whole-frame image-editor
outputs as training targets, with the recorded negative operator exclusions
preserved. It emits a separate training transform set with no original anchors
and no rejected cameras, verifies the released materializer's image hashes and
empty anchor list, and retains all original camera poses for final review.
Corrected targets enable direct RGB reconstruction (override multiplier 1.0)
and perceptual loss (weight 0.1). The pinned released trainer applies its
default `lambda_l1_override` factor of 0.8 within the reconstruction term; SSIM
is disabled. The historical zero reconstruction multiplier for teacher overrides
is forbidden in this mode; it relied on the original-image anchors.
Original images and SAM masks remain immutable provenance and review evidence.
There is no pixel clipping, compositing, or mask-based restoration of originals.

Color coefficients can change for generated support points and a bounded subset
of source points: the exact target box, plus surface-adjacent points in the
registered tabletop neighborhood extending 10 cm around it. Tabletop candidates'
three-sigma vertical extent, including rotation, must fit within 1 cm of the
registered top surface; source points with any scale exceeding 8 cm stay fixed.
This permits contact-shadow recoloring while excluding tall neighboring-object
points and distant room points. Geometry, opacity, and collision remain fixed.
This rule is a visual-candidate constraint, not a semantic ownership proof;
independent final review must still check the neighboring object and residuals.

The initialization receipt binds bounds, registered surface, rule version,
editable count, and source bytes. Runtime recomputes the region and checks exact
protected colors and all geometry/opacity after optimization. Receipts report
actual training frame indices and counts separately from source packet indices.
The partition participates in completed-training reuse identity, so an old
fully frozen checkpoint cannot silently satisfy the new policy. A subsequent
stage failure may reuse matching completed training, including partial accepted
teacher sets, through the existing checkpoint path.

Completion evidence is a live receipt proving the approved-only input set,
protected-source invariance, reviewed trained renders, and then policy and
online-result receipts. Unit tests or deployment alone do not prove visual
quality or end-to-end completion.
