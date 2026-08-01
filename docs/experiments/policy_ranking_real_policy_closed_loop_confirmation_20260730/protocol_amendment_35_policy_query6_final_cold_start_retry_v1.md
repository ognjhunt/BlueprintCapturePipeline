# Protocol Amendment 35: Policy query 6 final cold-start retry

Status: frozen after retry 2 and before any retry-3 provider mutation

Date: 2026-08-01

## Retry-2 result

Retry 2 proved the registered two-GPU reconciliation path live. With one
independently owned Warehouse GPU active, Blueprint acquired its exact paid
lane, consumed one authorization, and created only one OpenPI allocation. It
did not touch or terminate the Warehouse resource.

The owned RTX 6000 Ada instance `46515928` on machine `117871` remained in
provider `loading` state and produced no output before the frozen 480-second
startup cutoff. It was deleted after 407 charged GPU seconds. The unrelated
Warehouse instance closed independently, and global Vast inventory reached
zero. The first watchdog retained the lane because the Warehouse pending
teardown was still open at its initial terminal reconciliation. A bounded
terminal-recovery receipt then proved both instances absent, both pending
records closed, released the retained lane, and settled USD `0.084792`.

Preserved evidence includes:

- adapter SHA-256
  `70782e64ba8d44d0f31580b23f3577bb45e92eb7ed5952999154853a95f38420`;
- monitor SHA-256
  `fe544d23b54d6187820d282c559267c2c008582f28e6f65af5d9b4e47fa07467`;
- terminal-recovery receipt SHA-256
  `6a4a424fd6c85a7824b6bb8bdc7613007ed6fbb5490ee76aade8c52165a17b5e`;
  and
- settled ledger SHA-256
  `b6d034b7405656e30275c9404903c2d9a6c44308c995debca240f516495a2d7e`.

Cumulative conservative GPU accounting is 37,943 seconds and USD
`11.068649`. With unchanged evaluator/API spend of USD `8.418512`, combined
GPU plus evaluator/API spend is USD `19.487161`.

## Demonstrated startup-cutoff defect

Two distinct Vast machines reached the same pre-inference cutoff with the same
large OpenPI image and no output. The current-reference authorization already
declared a startup timeout, but the monitor used a fixed internal 480-second
default rather than the prospectively authorized value. Runtime SHA
`bfb6fd6cc7e5fbce5d3ec4843558695f6aca20a0` now validates that the registered
startup timeout is at least 60 seconds and remains at least 60 seconds below
the hard TTL, then passes it exactly to the monitor. It also adds safe
same-host dead-watchdog terminal release after strict provider-zero
reconciliation. The focused provider, admission, authorization, campaign,
lease, and watchdog suite passes 210 tests.

The source archive SHA-256 is
`b6fe7ab72ba6865dde2daf4cb83a2722e541aadf785500072bac6db5e084db13`;
its audit file SHA-256 is
`7e111ef0e4e11ce6e1307a49aa8671fc1971f4cedf7f6aab4d89a79afa8bd645`.

## Final retry boundary

Retry 3 retains the exact WAM5 camera frames, registered commanded state,
policy checkpoint, task text, query index, one-allocation limit, USD 3 cap,
and two-GPU ceiling. Failed machines `27268` and `117871` are excluded. The
startup timeout is prospectively increased to exactly 900 seconds; the hard
TTL remains 14,400 seconds.

The input archive SHA-256 is
`2b7ce05729d247047718874149e3ecb075c0117a7dd3ed8e07860931bd204f1a`;
its receipt file SHA-256 is
`e7bb9371d5135e12597d218f059e78b48239d72186138bf5a4bc00580034b436`,
and its independent extraction receipt SHA-256 is
`050ad0a8233ce7334629e3b109b1560cce9dcfb368bdbf24c81ad62b63a1e2cc`.

This is the final Vast cold-start retry for policy query 6. If a third distinct
machine reaches the 900-second cutoff without policy output, the current
provider/image pairing is technically unavailable for the successor episode;
it will not receive an indefinite series of retries. No evaluator or VLM call
is authorized, and success would close only interaction six.
