# Evidence matrix

| Required claim | Evidence now | What it proves | What remains |
|---|---|---|---|
| Frozen benchmark identity | RoboArena revision `7931db81...`; 3,883 sessions and 10,783 policy episodes | The independent real-policy answer key is attributable and frozen | Calibration and held-out labels remain sealed pending predictions |
| Candidate-policy overlap | 63 released sessions contain the same seven policies; 441 label-blind rows indexed | A complete cross-policy evaluation matrix is available | OSCAR paper says 65 sessions/455 rollouts; the 14-rollout discrepancy is unresolved |
| Frozen WAM/evaluator inputs | Protocol `fadd4e3b...`; evaluator `0c3633e3...`; pilot has 98 requests | Splits, gates, model, prompt, schema, frames, and cheap baseline are fixed; output-allowance changes after zero-score technical failures are ledgered | One 4,096-token validation must complete before pilot scoring |
| WAM action sensitivity diagnostic | 49/49 pilot action files analyzed without labels or physical-video pixels | Generated pixel motion has positive action-magnitude correlation in 47 nonconstant cases; median Pearson 0.488 | Pixel correlation is not 3-D action following; two cases are undefined and no pass threshold was retrofitted |
| Frozen benchmark calibration | Published OSCAR reference reports Spearman 0.750 and MMRV 0.571 | Independent prior work makes the experiment credible | Blueprint has not yet produced its own pilot/calibration/held-out scores |
| Controlled hybrid scene | NVIDIA warehouse control bundle `274f68e9...` | Same Franka/task/assets/evaluator contract can be specified in USD | Full scene not materialized; no simulator episode or policy result |
| Captured-site ingestion | Playroom bundle `993c5f4b...`; 3DGS plus separately hashed local assets | A previously unseen real 3DGS visual source is retained without full-site USD rebuild | Metric scale and collision registration are not independently verified |
| Hybrid rendering | Task-focused local Spark/SwiftShader manifest `780fa387...`; 6/6 nonblank views and 9 proxies | Cheap CPU rendering composes captured 3DGS and task layers | Wrist is fixed rather than link-mounted; proxy-only arm and no physics |
| Captured-site ranking | None | Nothing yet | Needs at least four attributable runnable policies with the frozen evaluator and an action-conditioned rollout lane |
| Four-policy availability | Four exact OpenPI PaliGemma DROID checkpoints, 43,400,758,584 bytes total, with pinned public object manifests | The prospective policy identities and 8-D DROID interface are attributable and match calibration | Weights, camera/state loop, and GPU inference have not been executed; checkpoint-specific terms remain unresolved |
| Speed/cost advantage | Local stages used no paid GPU; current pilot upper bound is $4.2232925 plus $0.036295 for two failed calls | The scoring pilot remains below the $5 hard cap | Actual wall time/cost and the fair exhaustive-physical counterfactual are not yet measured |

No row upgrades a generated-video score, simulator result, or prospective captured-site ranking into physical success.
