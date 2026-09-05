# Strict private Quick-10 controls and paired delivery

ADP-009D/day-28 uses the existing internal policy canary with an optional,
preregistered success-contract prerequisite:

```json
"controls": {
  "mode": "required_per_cell",
  "control_ids": ["zero_action_negative", "deterministic_scripted_positive"]
}
```

This is an evaluation admission prerequisite. It does not let a candidate grade
itself or change a deterministic episode score. Contracts without the field
retain the existing nonblocking diagnostic-controls behavior.

The production entrypoint runs the strict controls stage before provisioning
policy servers. Each of the ten resolved cells uses the same scene, reset seed,
robot base, task, cameras and destination as its two learned episodes. The
existing rigid phase generator and native IK solver must cover every target;
the existing control runner executes zero action and scripted positive, with
one attempt per phase. A missing camera/sensor, failed control, failed IK target
or incomplete retained output prevents policy loading.

After all twenty controls pass, ordinary policy server provisioning runs once.
The policy stage revalidates each cell's digest-bound control receipt, native
camera binding, runtime identity and retained files. It does not rerun controls.
The first paired policy cell must have native action/motion/progress evidence,
classifiable deterministic scores, destination readback and complete media.
Task success by a learned policy is not required.

Before cells 1-9, the worker streams a ZIP of the first cell's retained output
to a separate private signed witness key, then streams GET readback and verifies
its exact size and SHA-256. The native provider secret bootstrap supplies:

- `BLUEPRINT_POLICY_CANARY_PAIRED_PUT_URL_FILE`
- `BLUEPRINT_POLICY_CANARY_PAIRED_GET_URL_FILE`
- `BLUEPRINT_POLICY_CANARY_PAIRED_DELIVERY_AUTHORITY_FILE`

The authority binds the run, runtime inputs, session authority, implementation
commit, actual downloaded provider bundle, private URL object identity, expiry
and archive size bound. Witness metadata carries the authority binding digest.
The final-output URL cannot substitute for the witness key. URLs are never
written into result receipts. After successful readback, original episode files
and the witness manifest remain; the duplicate local ZIP is removed so it is
not nested inside the final archive.

Results retain twenty separate `controls` envelopes, their original control
receipts, a digest-bound native cell parent receipt, and control evidence files.
They also retain `controls_gate`, `strict_paired_gate`, `paired_delivery` and
`strict_gate_blockers`. A failed prerequisite seals partial evidence and the
unexecuted learned-episode count before ordinary archive delivery and provider
closeout. It is not a completed scientific run. Successful closeout remains
`completed_unqualified`, private and development-only; it makes no physical,
commercial, safety or deployment claim.
