# Exact SAM edit-mask audit — 2026-09-13

Scope: the owner's explicit choice of the SAM silhouette alone, with no added edit region. ADP-009D rehearsal dependency: avoid damaging retained objects while constructing removal teacher frames. No GPU/model calls, production changes, or deployment were performed by this audit.

## Findings and repairs

1. The earlier near-silhouette policy (#1919) still force-replaced calibrated pixels within 24 pixels of SAM, before the surface filter ran. A synthetic pale neighbour touching SAM was therefore included in the opaque core. The later SAM-only change (#1920) removes that expansion for nonempty SAM masks. This follow-up also removes the silent projected-box fallback: an empty or full-frame SAM mask must be corrected before requesting an edit. All 16 saved attempt-20 SAM masks are nonempty and usable under this check.
2. Raw edit reuse matched source RGB while deliberately allowing a different mask. That could retain old broad-mask generations after switching to SAM-only. Automatic discovery now requires the same edit mask, encoding, and base prompt. Repaired candidates are checked against their repair request's actual mask; discovery compares the base prompt so accepted bounded repairs remain reusable. The low-level retained-candidate loader excludes mismatched masks/encodings, and attachment removes those stale selections before the worker computes new-call costs. Identical-input reuse remains available.
3. Exact SAM core/support must remain full opacity inside, with source pixels preserved outside. The driver now supplies the source object mask as the core when a separate repair-core record is absent. Selective repair keeps the original exact mask and the core reference. For equal core and support, compositing bypasses the erosion/blur operation that would immediately be overwritten by the opaque core anyway.
4. The earlier feathering can jump sharply next to a core when its guard band is narrower than the feather radius. This was reproduced against the older guard-band policy. The requested SAM-only path has no guard band and bypasses this operation; this change does not redesign legacy feathering for unrelated wider-mask paths.

## Real retained-frame replay

Downloaded source images, SAM masks, calibrated projections, and saved raw outputs from attempt 20 into an isolated local audit directory. Verified source and mask files against the retained render-result digests, then executed the real repair-support builder and locality sealer on all 16 views. This is a diagnostic replay of old raw generations, not a fresh model run.

All 16 checks pass:

- emitted core and edit support equal the SAM silhouette pixel for pixel;
- zero additional editable pixels;
- every output pixel outside SAM equals the source;
- every output pixel inside SAM equals the saved raw generation;
- no feathering applied in the exact-core case.

Source-08: the original broad core covered 199,333 pixels; #1919's near-SAM core still covered 101,453; SAM itself covers 64,849.

## Visual limit — not a quality sign-off

The old raw outputs also change wood tone/texture and the retained bottle's shape. A clean-looking raw image does not prove preservation. When that old raw output is clipped to SAM, source-08 still has a visible vase-shaped patch and a shadow outside the silhouette. The compositor's exact outside-pixel contract preserves that shadow by construction.

The next attempt must obtain new edits for changed masks/prompts and review those results. Neither lowering the approved-view fraction nor passing these deterministic tests proves that the images look natural. This patch does not widen the SAM mask, introduce a blend band, or silently accept whole-frame generated pixels. Changing the mask sent to the editor and changing the returned-image compositing policy are separate decisions.

The local, full-resolution 16-view gallery is at `/private/tmp/blueprint-repair-mask-audit-20260913/audit_results/sam-only-final/index.html`; machine-readable counts are adjacent in `verification.json`. It pairs source, saved raw generation, and exact-SAM replay. The existing contact sheets show first-pass and repaired results; this gallery uses the saved first-pass raw results consistently.

## Verification

141 focused tests passed across support generation, neighbour preservation, candidate discovery, worker billing/reuse, the driver, selective repair, and diagnostic checkpoints. Tests additionally verify changed-mask and changed-prompt invalidation, unchanged-view reuse, missing/full-frame mask refusal before output, full core opacity, and exact outside preservation. The raw model is stubbed in tests; no paid stage was launched.
