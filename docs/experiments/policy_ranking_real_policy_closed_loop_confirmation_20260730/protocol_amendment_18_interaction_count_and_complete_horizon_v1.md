# Protocol Amendment 18: interaction count and complete horizon

Status: prospectively frozen before interaction three or any later WAM output

Date: 2026-08-01

## Counting correction

One complete Blueprint interaction is one full
`policy -> Ctrl-World -> same policy re-query` unit. Policy queries alone are
not interactions. The already frozen README calls the WAM generation from the
first re-query "interaction two," so the completed sequence through policy
query 2 contains:

- three real `pi05_droid` policy queries;
- two Ctrl-World WAM generations; and
- two complete policy-to-WAM-to-same-policy interactions.

The registered three-interaction gate therefore requires one more frozen unit:
the query-2 native action, WAM generation 2, its immediate reliability gate,
and the same `pi05_droid` query 3 from generated views and commanded state.
The external v1 episode receipt that counted policy queries as interactions is
preserved but superseded by a versioned correction. It earns no gate credit.

## Complete label-free horizon

The current official Ctrl-World reference freezes `interact_num = 12` in
`config_eval.py`, and `rollout_interact_pi_eval.py` executes exactly
`range(interact_num)`. The source files were frozen before successor execution:

- `config_eval.py` SHA-256
  `de2222b16cb2ceae751d4aa6f767be4fb7bafb6c5ff0b3603cc98f87dd7fe216`;
- `rollout_interact_pi_eval.py` SHA-256
  `130523033c1b1b79d3ae5d967ea6137c7ee33d5b3b47cb18e208b9a7aaab662a`.

Blueprint therefore freezes the maximum complete label-free horizon at 12
full policy-to-WAM-to-same-policy interactions. Because Blueprint requires a
same-policy response after every WAM prediction, a maximum-horizon trace has
12 WAM generations and 13 policy queries, indexed 0 through 12. This is a
prospective Blueprint current-reference horizon, not a claim of exact paper
reproduction.

The loop terminates earlier only on a registered reliability, collapse,
repeated-frame, static-under-command, action/state compatibility, or safety
abstention. A horizon that terminates by abstention is complete as a trace but
is not a qualified episode and cannot admit judges.

## Claim and execution boundary

The horizon was selected from frozen public source, not from observed action
values or generated media. No reliability or causal threshold changes. Judges
remain forbidden until a complete episode and the registered causal-control
matrix pass. Every later paid unit still requires its own immutable request,
exact pushed source binding, budget reservation, watchdog, teardown, and
provider-zero proof.
