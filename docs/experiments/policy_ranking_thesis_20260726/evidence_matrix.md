# Evidence matrix

| Required claim | Evidence now | What it proves | What remains |
|---|---|---|---|
| Frozen benchmark identity | RoboArena revision `7931db81...`; 3,883 sessions and 10,783 policy episodes | The independent real-policy answer key is attributable and frozen | Calibration and held-out labels remain sealed pending predictions |
| Candidate-policy overlap | 63 released sessions contain the same seven policies; 441 label-blind rows indexed | A complete cross-policy evaluation matrix is available | OSCAR paper says 65 sessions/455 rollouts; the 14-rollout discrepancy is unresolved |
| Frozen WAM/evaluator inputs | Protocol `fadd4e3b...`; evaluator `42d119e4...`; pilot inventory has 98 requests | Splits, gates, model, prompt, schema, frames, and cheap baseline were fixed before scoring | Provider calls require user approval |
| WAM action sensitivity diagnostic | 49/49 pilot action files analyzed without labels or physical-video pixels | Generated pixel motion has positive action-magnitude correlation in 47 nonconstant cases; median Pearson 0.488 | Pixel correlation is not 3-D action following; two cases are undefined and no pass threshold was retrofitted |
| Frozen benchmark calibration | Published OSCAR reference reports Spearman 0.750 and MMRV 0.571 | Independent prior work makes the experiment credible | Blueprint has not yet produced its own pilot/calibration/held-out scores |
| Controlled hybrid scene | NVIDIA warehouse control bundle `274f68e9...` | Same Franka/task/assets/evaluator contract can be specified in USD | Full scene not materialized; no simulator episode or policy result |
| Captured-site ingestion | Playroom bundle `c3cc38b4...`; 3DGS plus separately hashed local assets | A previously unseen real 3DGS visual source is retained without full-site USD rebuild | Metric scale and collision registration are not independently verified |
| Hybrid rendering | Local Spark/SwiftShader manifest `a14f72d2...`; 5/6 nonblank views and 9 proxies | Cheap CPU rendering composes captured 3DGS and task layers | One blank view, wrist occlusion, proxy-only arm, no physics |
| Captured-site ranking | None | Nothing yet | Needs at least four attributable runnable policies with the frozen evaluator and an action-conditioned rollout lane |
| Speed/cost advantage | Local stages used no paid GPU; pilot judge pre-call upper bound is $1.0912125 | The scoring pilot is tightly bounded | Actual wall time/cost and the fair exhaustive-physical counterfactual are not yet measured |

No row upgrades a generated-video score, simulator result, or prospective captured-site ranking into physical success.
