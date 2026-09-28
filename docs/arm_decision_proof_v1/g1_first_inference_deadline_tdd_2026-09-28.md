# G1 first-inference deadline: reviewed TDD plan

Owner-authorized G1 development extension, ADP-050 Day 28 and private evidence
ADP-060/080. Exact paid attempt `controller-fe6061cc-blueprint-run-20260927d3`
finished Diffusion with a scored task failure. Its pi0.5 server loaded all keys,
started its owned listener, acknowledged reset and received the actual task's
first observation. The client raised `TimeoutError` before an action response.
Both retained pi0.5 configs set `compile_model=true`, `compile_mode=max-autotune`;
pinned publisher code compiles `sample_actions` with torch.compile, whose first
execution occurs inside `predict_action_chunk`. The client currently uses a
30-second socket timeout for reset and inference alike. Slow initial compilation
is consistent with these observations; its exact duration is still unproven.

## Smallest repair

- Keep the publisher source, checkpoint bytes, precision and policy behavior.
- Give only pi0.5 checkpoints with compilation enabled a bounded 600-second
  first-inference allowance. Reset and subsequent queries keep 30 seconds.
  Keep the existing independent provider watchdog and 45-minute episode cap.
- Never retry a timed-out observation, invent an action or count missing/invalid
  responses as successful inference. A seed reset must not reset compilation
  allowance after an already successful first response.
- Record secret-free phase, timeout allowance, actual query duration and status
  in a private diagnostic and supervised episode receipt, including failure.

## Test-first cases

1. Actual HTTP transport receives first/steady/reset deadlines independently;
   second inference and reset-after-first do not regain first allowance.
2. Nonfinite, Boolean, negative and over-cap allowances fail before I/O.
3. Socket timeout yields a typed phase-specific error and timing receipt;
   query-success flag remains false. No automatic retry occurs.
4. Invalid action response does not count as completed inference.
5. Supervisor picks 600 seconds only for exact pi0.5 compile-enabled config,
   binds the timeout policy into the child lease and reaps the owned child.
6. Existing supervisor/client/runtime assembly, sealed provider import closure
   and lifecycle rehearsal remain passing before any new paid attempt.

Design review: checked the retained model config, first-infer log, owned-listener
lease, actual client transport and child teardown. The change covers a measured
transport limitation without claiming the model now runs or succeeds. Accepted
for test-first implementation. Completion requires an actual returned action,
scored episodes/media and paid closeout, not these hermetic checks alone.

Implementation checkpoint: 32 focused client/supervisor/runtime-assembly cases
pass, including actual HTTP opener deadlines, compiled pi0.5 versus Diffusion,
reset-after-first, invalid budgets, typed timeout without retry and invalid
action rejection. Changed-file Ruff passes. First/steady query timing is
retained even when inference fails; no secret endpoint response is included.
Shared provider import closure and lifecycle rehearsal are being run before
any new paid attempt. No running attempt has been modified or restarted.
