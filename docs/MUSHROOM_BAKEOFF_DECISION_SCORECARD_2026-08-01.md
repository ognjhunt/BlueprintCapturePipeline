# MuSHRoom Koivu Teleport/Postshot Bakeoff — Frozen Decision Scorecard (2026-08-01)

Point-in-time snapshot. Frozen before any provider output exists for this
bakeoff; edits after a provider run require a new dated scorecard and must
preserve this one unchanged. Evaluation runs against the compiler-v3 proxy
(`mushroom_proxy_fea6da5dfeca8e6a`, frozen split
`sha256:75c0a00f8b70d05bcbdf406f3266026105348b8d02f4bf25de8965a6b9d3712b`,
pose-consistency gate: consistent, aggregate median 0.27px).

## Arms

| Arm | Input | Provider workflow |
|-----|-------|-------------------|
| T1 | 265 candidate images (zip `sha256:a476442c…`) | Teleport managed ModelV3, SH3, max splats, 3200px edge (live-reverified) |
| P1 | Point-seeded COLMAP text (`colmap_dataset_9de1972eae8fe5ef`, 91,990 points) | Postshot Splat3, imported poses+points |
| P2 | Same as P1 | Postshot MCMC, imported poses+points |
| P0 (conditional) | 265 candidate images only | Postshot self-tracked, attribution arm |

No provider receives hidden filenames, hidden pixels, hidden cameras, or
hidden metrics. Author-designated 29 held-out and 121 independent-short views
stay evaluator-only.

## Evaluator

`heldout_appearance_evaluation_v2` (windowed SSIM Wang-2004 11×11 σ1.5;
pinned LPIPS `lpips_alex_v0.1`, checkpoint
`sha256:df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0`,
backbone `sha256:7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02`),
rendering through `sealed_camera_render` (reference Spark exact-camera path,
geometrically verified) after `provider_splat_import` +
`align_provider_reconstruction` (candidate cameras only). Before any hidden
view is rendered, each accepted arm must pass a candidate-view render sanity
loop (render ≥8 candidate views, PSNR vs candidate pixels ≥ 15 dB mean) so a
convention or alignment defect fails before sealed evaluation.

## Frozen qualification floors (single set, applied per trajectory; both
trajectories must pass; trajectories are never averaged together)

- mean PSNR ≥ 16.5 dB
- mean windowed SSIM ≥ 0.55
- mean global SSIM ≥ 0.50
- mean absolute error ≤ 0.12
- mean LPIPS ≤ 0.55

Alignment gates (per arm, frozen): candidate-camera similarity residual RMS
≤ 0.05 target units, max ≤ 0.20, no reflection preference, ≥ 8 pairs by
explicit filename mapping.

## Comparative decision rule

Applies only to arms that pass all floors and gates:

1. Primary: mean LPIPS per trajectory (lower wins).
2. Secondary: mean PSNR, then mean windowed SSIM per trajectory.
3. A backend is *selected as default* only if it wins LPIPS by ≥ 0.03 on both
   trajectories, or wins PSNR by ≥ 1.0 dB on both trajectories with no LPIPS
   regression > 0.01 on either.
4. Otherwise emit `profile_conditional_default` (with deterministic routing
   conditions grounded in which arm won which trajectory/regime) or
   `inconclusive_more_evidence_required`.
5. If no arm passes floors: `no_backend_qualified`.

Operational scorecard (recorded, tie-break only, never overrides metrics):
total cost, wall-clock, retries/failures, artifact size, reproducibility,
automation burden, credential/activation friction, deletion/teardown result,
audit completeness.

Emitted verdict is one of:
`teleport_selected_as_managed_default`, `postshot_selected_as_controlled_default`,
`profile_conditional_default`, `no_backend_qualified`,
`inconclusive_more_evidence_required` — with uncertainty, operational cost,
claim ceiling (appearance reconstruction on a processed public proxy only,
never Blueprint raw-capture proof), and the evidence that could change it.

## Blockers standing at freeze time

- Teleport API credentials not yet supplied (T1 blocked at upload).
- Windows GPU provider credentials/quota not yet supplied (P0/P1/P2 blocked
  at worker admission); Vast rejected for Windows; AWS G6/G5 Windows (or
  Azure NVadsA10) recommended behind the canonical paid-resource seam.
- Live terms/pricing/CLI reverification required for both providers at
  execution time.
