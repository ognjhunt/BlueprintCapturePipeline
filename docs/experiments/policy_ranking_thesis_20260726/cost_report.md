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
| V2 98-call pilot judge | 98 accepted generated-only judgments; no failed requests or blockers | None | $3.0161225 conservative metered cost; 420.179 seconds recorded for the completion invocation; $9.00 per-run cap was not approached |
| V2 98-call calibration judge | 98 accepted generated-only judgments; no failed requests or blockers | None | $2.89694 conservative metered cost; 474.981 seconds wall time; $9.00 per-run cap was not approached |
| Attributed OSCAR generation for 49-rollout pilot | Authors generated the released media; not a Blueprint provider mutation | Single GH200 in paper | At the paper's 2.214 FPS, an 81-frame-by-49 serial equivalent is approximately 1,792.7 seconds (29.88 minutes), excluding load/I/O/preprocessing; this is attributed, not Blueprint-measured |
| Controlled USD simulation | Not run | None | Unmeasured |
| Captured-site policy/WAM rollout | Not run | None | Unmeasured; any paid lane requires a separate admission decision |
| Physical robot evaluation | Forbidden in this goal | Forbidden | $0 spent |

The current evidence does not yet establish the thesis's speed or cost advantage. The 2.27x pilot ratio is judge-only and is not an end-to-end claim. Adding the paper-attributed single-GH200 OSCAR generation estimate gives approximately 37.34 minutes for pilot WAM generation plus judging, versus a 16.97-minute physical action-only lower bound; that comparison omits overhead on both sides and cannot decide the thesis. The experiment establishes that label-blind preparation and hybrid-site ingestion can be performed locally without a paid GPU and records the completed v2 pilot and calibration judge measurements. The interrupted attempt's elapsed time was not finalized; the process was observed beyond 446 seconds and that lower-bound observation is preserved rather than converted into a false exact duration. A final verdict still needs Blueprint-operated WAM/policy/simulator runtime and a fair physical counterfactual, and must state whether any physical-exhaustive comparison is measured, sourced, or merely a range.
