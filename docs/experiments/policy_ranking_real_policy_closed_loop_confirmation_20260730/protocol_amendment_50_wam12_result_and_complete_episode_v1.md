# Protocol Amendment 50: WAM12 result and complete episode

Status: frozen after WAM12 and before causal-matrix execution

Date: 2026-08-01

## WAM12 result

WAM12 executed from clean pushed experiment SHA
`098eae5ca127757cb5e21903eb26e31ff4015b9a`. A first live invocation stopped
before authorization consumption or provider mutation because three registered
launch environment gates were absent; its reservation settled at zero cost and
the failed directory is preserved. The versioned second invocation supplied
only those already registered gates. Allocation 16 was then consumed exactly
once.

The exact provider output archive SHA-256 is
`dba51f8567d6664082ecb9af2f06907cd84af9814308a55896f30a342e647d27`.
Its 22-member allowlist, three generated camera views, five frames per view,
request SHA-256
`0dc4bbe246a1f30f5f332651ad2feb0ad8fcb27d8d66f1dd6e12ca20863e3229`,
and runtime identity
`c391434c2c9353e6cf700ce8f89973c45bbbd8c777e02a3bdff340ba570a71b5`
passed. The extraction receipt file SHA-256 is
`a7dd3f3b89e973420404b5b993603344aec9c0175e396da52f79cbe12d9c76d5`.

Immediate reliability passed with no flags or abstention. Exterior view 2 had
motion mean `0.1257395625` and timing correlation `0.1573638267`; exterior view
1 had motion mean `0.0853885010` and timing correlation `0.1890598687`; the
wrist view had motion mean `2.1338791907` and timing correlation
`0.6871719833`. The frozen timing threshold remained `0.15`. The reliability
report file SHA-256 is
`a426de1a8fd100b00c073ca90bcb16daf131ada9a0490ea9c2f90f5dc0160d9d`.

WAM12 charged 173 GPU seconds and USD `0.076964`. Cumulative GPU usage is
42,533 seconds and USD `12.518662`; evaluator/API spend remains USD `8.418512`.
The independent watchdog reached provider terminal, exact instance `46536179`
was absent, fresh authenticated global Vast inventory returned zero, and the
campaign ledger has zero open reservations.

## Complete label-free episode

The registered 12-interaction episode is complete. The same frozen
`pi05_droid` checkpoint was repeatedly re-queried from three WAM-generated
camera observations plus registered commanded state. Each policy output was a
new native action generated at query time; recorded action replay was not used
as policy re-query. No future physical RGB, future recorded physical state,
physical outcome, or judge result entered the loop.

This completion proves execution of the registered feedback chain only. It
does not prove that Ctrl-World is action-causal, that the task succeeded, that
virtual policy ranking agrees with physical outcomes, or that abstention,
captured-site transfer, or economics pass.

## Next gate

The next gate is the already frozen six-condition, two-seed Ctrl-World causal
matrix. The existing seed-0 own-action generation may count only if its exact
request and control identity match the frozen matrix. Every other cell must be
generated independently through the same adapter, checkpoint, three-view
contract, watchdog, budget, and teardown gates. Gemini and GPT-5.6 Luna remain
forbidden until the complete matrix passes all frozen causal and reliability
thresholds.
