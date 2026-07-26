# Cost and resource report

Snapshot: 2026-07-26 10:21 America/Chicago.

| Stage | Provider mutation | Paid GPU | Measured or admitted cost |
|---|---|---|---|
| Metadata research and frozen indexing | None | None | $0 marginal provider cost |
| 49-rollout OSCAR pilot materialization | Public download only | None | $0 provider cost; approximately 175 MB local materialization |
| Label-blind action/motion diagnostic | None | None | $0 provider cost; local CPU |
| Playroom 3DGS download and 20k-splat hybrid preview | None | None | $0 provider cost; local macOS CPU with Spark.js/Chromium/SwiftShader |
| NVIDIA spray-can asset dependency materialization | Public download only | None | $0 provider cost; 143,801,263 bytes across 11 non-thumbnail files |
| Pilot judge attempt 001 | One generated-only request; no accepted score | None | Provider usage was not preserved; conservatively charged the $0.015 per-request admission ceiling |
| Pilot judge attempt 002 | One generated-only request; no accepted score | None | Measured conservative API cost $0.021295; provider reported `incomplete:max_output_tokens` |
| Pilot judge attempt 003 | 44 accepted generated-only judgments; four incomplete responses; excluded from ranking | None | $1.4001325 persisted metered estimate plus $0.36 worst-case allowance for four requests in flight at interruption; $1.7601325 conservative accounting |
| Current v2 98-call pilot judge | Not started | None | Conservative pre-call upper bound $8.2373725; $9.00 per-run cap; at most two exact attempts per request; cumulative upper bound including prior attempts $10.0338 |
| Controlled USD simulation | Not run | None | Unmeasured |
| Captured-site policy/WAM rollout | Not run | None | Unmeasured; any paid lane requires a separate admission decision |
| Physical robot evaluation | Forbidden in this goal | Forbidden | $0 spent |

The current evidence does not yet establish the thesis's speed or cost advantage. It establishes only that the label-blind preparation and hybrid-site ingestion can be performed locally without a paid GPU, and that the proposed pilot evaluator has a documented upper bound. The interrupted attempt's elapsed time was not finalized; the process was observed beyond 446 seconds and that lower-bound observation is preserved rather than converted into a false exact duration. A final verdict must use measured run duration and usage, and must state whether any physical-exhaustive comparison is measured, sourced, or merely a range.
