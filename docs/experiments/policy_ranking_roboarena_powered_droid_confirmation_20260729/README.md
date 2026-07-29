# Powered DROID causal confirmation

This prospective experiment asks whether native `Cosmos3-Nano` visibly and
temporally follows frozen DROID-compatible policy actions across 17 independent
RoboArena sessions. It is the admission gate before spending on complete
policy-ranking rollouts.

A pass means the recorded-action rollouts beat valid no-motion, shuffled,
reversed, shifted, and real policy-swapped controls under the frozen causal,
seed-noise, collapse, and session-level reliability rules. A pass does not by
itself prove policy ranking, captured-site transfer, physical performance, or
economics.

The sessions come from the same published RoboArena snapshot as prior work but
exclude Phase-A sessions and the one later-exposed diagnostic session. Even a
successful result is therefore a disjoint-session, same-snapshot open-loop WAM
qualification—not a new-snapshot or live closed-loop Phase-B confirmation.

## Terminal result

The complete powered matrix executed and the native-Cosmos arm failed its
causal and reliability gates. Blueprint abstained. The frozen-stack overall
verdict is `thesis_not_supported`; captured-site transfer and completed-ranking
economics remain unmeasured. See [final_report_v1.md](final_report_v1.md) and
[terminal_result_v1.json](terminal_result_v1.json).

## Result

The experiment executed completely but the native Cosmos arm failed the frozen
qualification gates. The structured canary passed, then the runtime preserved
612/612 scientific responses and videos across 17 sessions, 51 windows, six
conditions, and two seeds. Frozen post-run analysis found:

- causal-validity pass rate: `0.0` (session-clustered 95% interval `0.0–0.0`)
- reliable-session rate: `0.0` (95% interval `0.0–0.0`)
- windows passing every gate: `0/51`
- sessions accepted as reliable: `0/17`
- sessions on which Blueprint abstained: `17/17`
- windows flagged `static_under_command`: `46/51`

The abstention is correct for the registered native-Cosmos arm: the measured
correct-action effect was generally smaller than random-seed variation and did
not robustly separate from the temporal placebos. This result blocks native
Cosmos from Phase-B policy-ranking admission. It does not measure skeleton-only,
OSCAR purpose-built WAM, a registered skeleton-conditioned Cosmos hybrid,
captured-site transfer, or useful-ranking economics.

The Vast adapter reported a terminal `vast_heartbeat_container_missing` after
the long-running container exited, but the complete 200,594,982-byte output
archive had already uploaded and passed ZIP, runtime-result, response, video,
and hash validation. The scientific result is therefore valid; the outer
status was an operational false failure. A reusable fail-closed recovery path
now checks the durable output archive before classifying this terminal-container
race.

Machine-readable evidence:

- `allocation_2_execution_receipt_v1.json`
- `native_cosmos_causal_result_v1.json`
- `cost_ledger_v1.json`
