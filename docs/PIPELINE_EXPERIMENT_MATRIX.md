# Next 3 Runs Experiment Matrix

Use this matrix to run controlled, high-signal iterations while a long run is in flight.

| Run | Parameter Changes | Hypothesis | Expected Impact | Stop Criteria |
| --- | --- | --- | --- | --- |
| `E1_baseline_clear` | `NUREC_RERUN_PROFILE=clear_over_faithful` | Baseline-only high-capacity settings should improve geometry sharpness without synthetic hallucination. | Better visual clarity and cleaner collision mesh boundaries; lower synthetic artifact risk. | Stop if `capture_quality_report.json` registered ratio drops below `0.75` or if mesh has obvious structural holes in key views. |
| `E2_refine_force` | `POST_STAGE4_REFINE=force`, `POST_STAGE4_REFINE_MODEL=worldforge+gsfix3d`, `POST_STAGE4_DISTILL_ITERS=2400` | Forced repair with higher distill budget should reduce void regions versus baseline. | Lower hole ratio in `gap_analysis_report.json` and improved continuity in sparse regions. | Stop if `refinement_quality_gate.json` fails or sharpness regresses more than gate thresholds. |
| `E3_speed_guarded` | `NUREC_QUALITY_PROFILE=balanced`, `MAX_FRAMES=240`, `EXTRACT_FPS=4`, keep default quality gates | Reduced frame count should cut runtime while preserving acceptable downstream quality. | Faster end-to-end turnaround with small quality regression. | Stop if `swap_quality_report.json` fails any required gate or runtime improvement is less than 20%. |

## Execution Notes
- Run one experiment at a time and keep all non-listed params fixed.
- Save each run's `run_summary.json` and `log_summary.json` under a stable archive folder for side-by-side comparison.
- Do not promote a faster profile unless output checklist (see `docs/OUTPUT_VALIDATION_CHECKLIST.md`) still passes.
