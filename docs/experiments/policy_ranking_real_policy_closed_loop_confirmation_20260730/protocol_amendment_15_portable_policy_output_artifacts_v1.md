# Protocol Amendment 15: Portable Policy Output Artifacts

Frozen prospectively in the policy-canary lane: 2026-07-30T13:45:05-0500

Incorporated into the Ctrl-World successor: 2026-07-30T13:51:34-0500

## Observed engineering defect

Offline inspection before any successor policy output was observed found that
the current-reference policy canary recorded provider-local absolute paths for
its native action and policy-receipt artifacts. The output archive preserved the
bytes, but a downloaded manifest could not resolve those paths under the local
result root. That would make the evidence non-portable and could allow a
manifest to appear complete without proving that its hashes bind the archived
artifacts.

No provider output, WAM output, judge output, or physical outcome was inspected
to make this amendment. The previously frozen policy identities, request count,
native action shapes, scientific thresholds, stage ordering, budgets, and claim
boundaries are unchanged.

## Prospective artifact contract

Every completed current-reference policy canary or same-candidate re-query must
declare `artifact_path_mode: result_root_relative` in the terminal manifest,
each policy result, and each policy receipt. Receipt and native-action paths
must be simple result-root-relative member names. Absolute provider paths are
invalid.

The terminal archive validator must independently bind, for every requested
frozen policy:

- the exact policy identifier and one completed result;
- the receipt member and its file hash;
- the receipt's canonical manifest hash;
- the native NumPy action member and its file hash;
- the exact native action shape: 10x8 for `pi0_droid` and
  `pi0_fast_droid`, and 15x8 for `pi05_droid`;
- `float64` finite action values;
- false WAM access and false physical-outcome access.

Missing, unreadable, path-drifted, hash-drifted, malformed, non-finite, or
shape-drifted artifacts make the terminal output invalid. An invalid initial
archive earns no real-policy evidence and cannot advance to WAM execution. An
invalid re-query archive cannot advance the closed loop.

## Retry identity

The earlier `e703ab22` source archive and signed input bundle predate this
amendment and are retired without execution. The policy canary is rebound to
the clean, pushed `b6be7344` source identity and newly versioned source archive,
signed input bundle, provider URLs, run name, admission records, and budget
ledger. Prior evidence is preserved without overwrite.

The Ctrl-World paid WAM profile remains separately bound to the exact immutable
Ctrl-World successor source SHA required by Amendment 14. This amendment does
not combine the policy and WAM provider requests or relax either request's
identity gate.

## Claim boundary

This amendment repairs evidence portability and terminal validation only. It is
not learned-policy inference, WAM execution, a policy re-query, causal
qualification, ranking evidence, captured-site transfer, physical evidence, or
an economics result.
