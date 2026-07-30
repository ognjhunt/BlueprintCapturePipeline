# Protocol Amendment 7: Ctrl-World Reliability Measurement Encoding

Frozen prospectively: 2026-07-30T12:42:00-0500

## Defect identified before WAM output

Ctrl-World conditions on absolute Cartesian pose rows, while Blueprint's frozen
rollout-reliability gate measures action energy from incremental translation,
rotation-6D, and gripper commands. The initial measurement adapter copied the
absolute XYZ and orientation into that delta-action representation. A stationary
nonzero pose would therefore look active and could invalidate the registered
no-motion control.

The first runtime contract also returned three exact generated PNG sequences,
whereas the legacy gate accepted one encoded video. No current-reference WAM
output existed when these defects were identified.

## Prospective correction

The WAM-conditioning contract is unchanged. For reliability measurement only:

- each absolute Cartesian pose is converted to the incremental translation and
  relative rotation from the preceding generated-frame pose;
- the first row is an explicit zero-translation, identity-rotation transition;
- gripper position remains absolute so the existing gate measures its changes;
- the latest hash-bound current view is prepended to each view's five generated
  frames, measuring the current-to-first-generated discontinuity;
- all three registered views are scored independently from their exact PNG bytes;
- failure in any registered view causes reliability abstention; and
- session-level timing aggregation remains required after three eligible windows.

This corrects representation and transport only. The thresholds frozen in
`protocol_v1.json` are unchanged. No result-driven threshold adjustment is
admitted.

## Claim boundary

Passing this gate is necessary rollout-reliability evidence. It does not prove
causal action following, object persistence, episode coherence, policy ranking,
physical success, captured-site transfer, economics, or the thesis.
