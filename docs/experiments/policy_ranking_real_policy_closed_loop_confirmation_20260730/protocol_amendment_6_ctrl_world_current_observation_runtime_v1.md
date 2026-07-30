# Protocol Amendment 6: Ctrl-World Current-Observation Runtime

Frozen prospectively: 2026-07-30T12:23:14-0500

## Defect identified before WAM output

The released Ctrl-World interaction code provides two distinct visual inputs to
each world-model call:

- six selected three-view history latents using indices `[0, 0, -12, -9, -6,
  -3]`; and
- the latest three-view observation as a separate current-image latent.

Blueprint's initial transition contract exposed the six selected history frames
but did not separately expose the latest current observation. This is invisible
at the repeated frame-zero initialization but becomes incorrect after generated
observations enter the loop. No Ctrl-World current-reference WAM output had been
generated when the defect was found.

## Prospective correction

Every `blueprint_ctrl_world_current_reference_request.v1` request must now bind:

- exactly three registered views in released order;
- exactly six hash-bound native 192x320 RGB history frames per view;
- exactly one separately hash-bound latest native 192x320 RGB frame per view;
- exactly one finite float64 11x7 Cartesian conditioning array;
- five predicted frames, eight executed policy rows, and 8/15 seconds;
- one frozen nonnegative seed;
- the exact current-reference Ctrl-World source, WAM checkpoint, SVD, CLIP, and
  DROID state-stat identities.

Duplicate history references reuse the same encoded latent. The current frame is
encoded separately unless its byte-identical path is already present in the
history cache. The runtime may load neither a policy nor recorded future video.
It may access neither policy identity nor physical outcome labels.

The provider result must contain exactly five hash-bound generated PNG frames
for each of the three registered views. It must explicitly record no future
physical RGB, no recorded action trace, no outcome labels, no WAM-to-WAM
chaining, one frozen WAM across all views, timing, seed, and model identities.

The released Dynamics definition is loaded from the exact frozen `train2.py`
bytes. If the local preprocessing environment lacks `decord`, Blueprint may
provide a temporary fail-closed stub for that file's unused training-data import;
any attempted call through the stub raises immediately. The Dynamics class,
checkpoint, and FK bytes remain unchanged, and the receipt records whether the
stub was present. The pinned provider environment includes `decord`; live
execution must revalidate the exact checkpoint-bound result there before causal
credit.

## Admission and claim boundary

This correction changes no policy, action adapter, causal threshold, reliability
threshold, judge prompt, ranking rule, label custody, or budget. It is frozen
before any successor WAM output and therefore cannot be tuned to a WAM result.

The first live WAM stage remains forbidden until the three-policy identity
canary is terminal, provider zero is proved, its conservative reservation is
reconciled, and at least one complete real native action is preserved. The first
admitted WAM mutation is one generation from one such policy action. Model
execution alone earns no closed-loop, causal, ranking, abstention, physical,
captured-site, economic, or thesis credit.
