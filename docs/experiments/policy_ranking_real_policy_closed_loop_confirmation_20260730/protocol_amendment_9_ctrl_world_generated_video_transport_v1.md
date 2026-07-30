# Protocol Amendment 9: Ctrl-World Generated-Video Transport

Frozen prospectively: 2026-07-30T12:49:32-0500

## Defect identified before WAM output

The direct current-reference runtime preserved five exact generated PNGs for
each registered view, but Blueprint's canonical paid WAM completion recovery
also requires at least one generated-only MP4. Without that transport artifact,
a scientifically valid runtime result would be reported as incomplete by the
provider control plane.

## Prospective correction

After all fifteen generated PNG paths and hashes are validated, the runtime
encodes:

- one five-frame MP4 for each registered view; and
- one five-frame horizontal three-view MP4 in the released view order.

The MP4 writer reads only the retained generated PNGs. It includes no recorded
or future physical pixels. All four MP4s retain hashes, sizes, dimensions, FPS,
and frame counts. The PNGs remain authoritative for exact per-frame analysis;
the MP4s are transport and evaluator media. Any PNG hash mismatch, writer
failure, frame-count mismatch, or geometry mismatch makes the runtime fail
closed.

## Claim boundary

Successful encoding proves transport completeness only. It does not prove WAM
causality, rollout reliability, episode coherence, evaluator validity, policy
ranking, physical success, transfer, economics, or the thesis. No model input,
inference setting, causal threshold, reliability threshold, or budget changes.
