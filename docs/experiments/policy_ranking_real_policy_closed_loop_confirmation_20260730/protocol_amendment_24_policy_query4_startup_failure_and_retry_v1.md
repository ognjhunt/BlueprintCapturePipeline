# Protocol Amendment 24: policy query 4 startup failure and retry

Status: frozen after the first policy-query-4 allocation and before retry

Date: 2026-08-01

## Failed provider allocation

The first policy-query-4 allocation consumed authorization
`policy-ranking-real-policy-closed-loop-20260730-openpi-policy-query-4` and
created Vast instance `46499086`. The instance never booted. After 14 minutes
56 seconds, the provider still reported `actual_status=loading`, while its
current and intended states were `stopped` and no SSH endpoint existed. No
policy request completed and no provider output archive was produced. This is a
provider-startup failure, not a policy, WAM, interaction, or scientific result.

The owned unbooted instance was terminated under the existing 480-second
startup-dud and teardown-on-failure contracts. Exact-instance absence and
global provider zero were proven before the owner-to-watchdog cancellation
receipt was written. The independent watchdog reached provider-terminal state,
the pending-teardown record closed, and the local paid-lane lease released.
The campaign ledger charged the ceiling-safe 943 seconds and USD `0.196458`.
Committed campaign GPU use is now 35,736 seconds and USD `10.412277`; the
evaluator/API ledger remains USD `8.418512`.

Preserved failure evidence includes:

- the final unfavorable live-inventory snapshot SHA-256
  `6c595c99be19b5f053537c1f5c5f768f7aff6c922a4beb6b61e5aa803f918023`;
- the exact startup-dud reap receipt SHA-256
  `1010c3ea8329823ef4ffb658e375d942d0d59a19940c043f3189409b26b19e88`;
- the exact-instance and global-zero receipt SHA-256
  `928149f636a6ffddc449830b91b7f8817247d2e707e1cd8840042f4a1523cfc2`;
- the terminal independent-watchdog record SHA-256
  `288d074295d860baa1537854108d7d518f0d1c54279dc1f0c90b503ecc2bd273`;
  and
- the settled production campaign ledger SHA-256
  `14459ce53c356abf7e01219607caeb5c1ad33e1e14b1261955db4ccf2180f1ce`.

## Generic monitor correction

The failure exposed a generic OpenPI monitor defect: after provider creation,
the monitor polled only for the signed output object. It did not inspect the
owned provider resource, so a stopped or unbooted instance could retain the GPU
until the four-hour hard TTL even though the shared startup guard had already
classified it as a dud.

Runtime commit `529bc39639021448c0743f36f0cde75a8f5d6098` fixes the shared monitor for
both Vast and RunPod. It now:

- inspects the exact owned provider resource during output polling;
- preserves the existing 480-second startup threshold;
- distinguishes a healthy RunPod runtime from a pre-runtime resource;
- tears down a terminal or startup-timed-out resource through the existing
  owner, watchdog, pending-teardown, global-zero, and budget-settlement path;
  and
- returns a typed failure with `continuing_spend=false` only after the control
  plane and settlement are terminal.

Focused verification passed: all 15 OpenPI runtime tests, all 23 independent
watchdog tests, and three current-reference allocator tests. Ruff and diff
integrity checks passed. No scientific threshold, observation, policy,
checkpoint, action, WAM, or judge contract changed.

## Frozen identical-observation retry

The retry uses the same WAM3 generated camera frames, commanded-prefix state,
task prompt, policy identity, checkpoint inventories, image identity, and
physical-outcome blinding as the failed attempt. Only the generic runtime source
changes. The exact source archive is:

- commit `529bc39639021448c0743f36f0cde75a8f5d6098`;
- URL
  `https://codeload.github.com/ognjhunt/BlueprintCapturePipeline/tar.gz/529bc39639021448c0743f36f0cde75a8f5d6098`;
- archive SHA-256
  `a4b2066425b5cab35f4a95cb926a14865efe4101663355bf36095d79a9038632`.

The versioned retry input bindings are:

- input archive SHA-256
  `77a36e0f7f4b9e664a86c2f12829e9078956633ba62e6068fa7b56b52cc25e68`;
- input-receipt file SHA-256
  `d224579cd87db2de1f253e27607e786b122be9cd610904013cfa733d7ee691bc`;
- input-receipt manifest identity
  `9bf97494c6385d56e58e783ce831ae59b6e7131088c4fb90d70bd3854821fc9d`;
  and
- independent extraction receipt SHA-256
  `4f9ab6e805a66237231102b6281596e650e0571d521f68d4f376dc8e9cd338b4`.

A post-close inventory refresh observed a newly allocated, unrelated native-
warehouse GPU. Blueprint will not touch it. The retry is forbidden until a new
mutation-free preflight proves global provider zero, a fresh output key is
absent, signed transport passes, and all unchanged budget, one-GPU, watchdog,
TTL, teardown, and provider-zero gates pass.

## Decision boundary

The failed allocation contributes cost, time, reliability, and failure-layer
evidence only. It does not complete policy query 4 or interaction four. A
successful identical-observation retry may complete that interaction. Gemini
and GPT-5.6 Luna remain forbidden until the complete 12-interaction episode and
causal-control matrix pass.
