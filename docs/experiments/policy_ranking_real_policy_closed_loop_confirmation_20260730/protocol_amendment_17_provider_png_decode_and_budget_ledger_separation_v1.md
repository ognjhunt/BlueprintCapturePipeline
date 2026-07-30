# Protocol Amendment 17: provider PNG decode and budget-ledger separation

Status: prospectively frozen before the first-WAM retry

Date: 2026-07-30

## Observed failure

The immutable `d416f4ca1a6267668e211e190901155d6ddcbb7d` first-WAM attempt
completed all 50 Ctrl-World diffusion steps and retained 15 generated PNGs:
five frames for each of the three registered camera views. Pillow independently
verified every PNG as RGB `320x192`. The runtime then emitted
`libpng error: bad parameters to zlib`, returned
`ctrl_world_current_reference_runtime_exception:ValueError`, and produced no
MP4. Provider teardown and provider-zero completed. The attempt cost estimate
was USD `0.167637` for `347.574521` observed live seconds.

This is an engineering transport failure, not evidence that Ctrl-World passed
or failed the registered WAM reliability or causal thresholds. No generated
frame from this failed attempt may be used to select or alter a scientific
threshold, action, policy, task, seed, or camera contract.

The provider adapter also used the caller's production campaign-ledger path for
its session-cost summary. That replaced the open ledger bytes after admission.
The original reservation is preserved in `campaign_budget_reservation.json`;
the provider-zero settlement is reconstructed in a separately versioned ledger
using the exact observed duration rounded up to 348 seconds and the USD
`0.167637` estimate. Future attempts must keep the production campaign ledger
and provider session-cost ledger at distinct paths.

## Frozen generic correction

The retained generated PNGs will be decoded by Pillow, the same library that
writes them, and converted deterministically from RGB to contiguous BGR arrays
before the existing OpenCV MP4 writer. PNG byte hashes, required geometry,
frame count, view ordering, generated-only rules, and final MP4 validation stay
unchanged. A focused regression test must prove the media path succeeds even if
`cv2.imread` is unavailable.

The retry must use:

- the identical staged scientific request and action conditioning;
- `pi05_droid`, query 0, seed 0, and the same three camera histories;
- a new runtime bundle and immutable pushed repository SHA;
- a new single-use compute authorization;
- distinct production campaign and provider session-cost ledger paths;
- the existing one-GPU, spend, TTL, watchdog, teardown, and provider-zero gates.

No scientific or cost ceiling is changed. Judges remain forbidden.
