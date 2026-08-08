# ADP-009D overnight results — 2026-08-08

Status: in progress

Scope: ADP-009D, public-scene day-28 Franka rehearsal gate.  Every result in
this document remains `development_only`; none is physical evidence or a
partner-policy verdict.

## Landed changes

- `c9103d311` — P0 action-delivery-aware episode evidence.  Each episode now
  retains reset/end arm joint positions, maximum per-joint motion, query and
  joint-limit-clamp counts, commanded-action magnitudes, observed command
  response, and an explicit harness finding when a deterministic object-state
  score cannot be attributed to a policy.  Uninterpretable cells are excluded
  from policy-outcome counts and ordering, and any such cell blocks a top-level
  completed runtime result.
- `dca08f185` — P1 bounded policy readiness attempts and retained failure
  diagnostics.  A single blocked vendor import/client constructor can no
  longer starve the worker's readiness deadline.  The server-log digest and a
  bounded tail are embedded in the receipt while the original log remains in
  the provider output directory.
- `c061fc1b9` — fail-closed episode media.  Every newly scored ADP-009D
  policy episode now retains the exact post-preprocessing external and wrist
  image bytes shown to the policy, a digest-bound lossless PNG manifest, a
  terminal observation, and a derived human-review MP4.  Missing media blocks
  the top-level runtime rather than allowing a run to look evaluation-ready.
- `5e01b5d38` — cycle-time instrumentation.  Each episode now separates policy
  inference, policy-input acquisition, observation preprocessing, simulator
  steps (explicitly including render when enabled), settle, scoring, and media
  persistence.  These are operational diagnostics, not task metrics.
- `b9a3b30ac` — correct DROID velocity-to-position control mapping.  The
  released π0.5-DROID checkpoint emits seven joint-velocity dimensions plus
  absolute gripper position at 15 Hz; the harness had incorrectly treated the
  first seven values as absolute joint positions.  The replacement reuses the
  repository's pinned DROID mapping, clips the released action as the public
  runtime does, and forms each Arena position target from the currently
  observed joints.  The raw velocity, clipped action, mapping revisions, and
  resulting target are retained per query.

For both commits the required pre-commit gates passed:

```text
PYTHONPATH="$PWD/src" .venv/bin/pytest tests/ -q -k "adp009d or droid or episode or nurec or aura"
.venv/bin/ruff check src/ tests/
```

P0: `876 passed, 1 skipped, 9050 deselected`; Ruff passed.
P1: `878 passed, 1 skipped, 9050 deselected`; Ruff passed.
Media and cycle-time commits: `882 passed, 1 skipped, 9050 deselected`; Ruff
passed before each commit.
Velocity-mapping commit: `882 passed, 1 skipped, 9050 deselected`; Ruff passed.

## Paid-run ledger

The handoff estimated approximately `$6` spent before this continuation.  A
conservative sum of all retained v1-v62 top-level adapter receipts under the
shared evidence directory is `$10.998299`; with v63 it is `$11.365784`.  That
directory includes a broader
history than the overnight window.  The `$25` overnight cap is enforced using
the more conservative total when deciding whether another launch is allowed.

| Run | Immutable code / variable changed | Held constant | Cost | Terminal result |
| --- | --- | --- | ---: | --- |
| `native_microcheck_v63_p0_action_evidence` | `c9103d311`; P0 evidence only | π0.5 DROID, three episodes, 320x180 policy render, v3 recentered/exposed NuRec, sealed can/SAGE/task manifest, provisioning timeout and run caps | `$0.367485` | Native runtime completed and queried π0.5 for 60 chunks per episode.  The arm moved, but 448/480, 462/480, and 467/480 rows were joint-limit clamped.  Primary OpenPI sources establish that these are joint velocities, while this run sent them as absolute positions.  Result: typed harness fault; `never_moved` x3 is still not a policy verdict.  Teardown completed and a live Vast API query returned `active: 0 []`. |

## Open questions

- P0 remains open, but v63 resolved the original ambiguity in the important
  direction: actions were not dropped; the robot moved by up to 2.44 rad.
  It simultaneously falsified the harness action mapping.  OpenPI's released
  DROID runtime identifies each response as ten or fifteen rows of seven joint
  velocities plus gripper position, clips rows to `[-1, 1]`, and executes at
  15 Hz.  Therefore v63's `never_moved` x3 is a harness result, not a policy
  result.  The corrected mapping is committed; one controlled rerun is needed.
- P1 code is landed, but GR00T has not yet been rerun with the bounded attempt
  receipt; no cause is inferred from the prior hang.
- P2 float32 Aura payload and P4 the two-candidate ranked run remain deferred
  behind P0 and P1.  P3 cycle-time measurement is landed and will be populated
  by the next run.  Warm reuse across nominally separate runs is not adopted:
  it conflicts with the required provider-zero proof after every teardown;
  the supported same-instance optimization is the existing comma-separated
  multi-candidate batch used for P4.

## Single next action

Run one π0.5-only v64 canary on `b9a3b30ac` or later, holding every non-code
input constant, and require low/diagnosable joint-limit clamping plus observed
arm response before interpreting its deterministic object outcome.
