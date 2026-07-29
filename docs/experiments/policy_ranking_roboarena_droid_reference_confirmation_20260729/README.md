# Cosmos3 DROID reference and untouched-data confirmation

This namespace continues the terminal `inconclusive` RoboArena successor without
rewriting its evidence. NVIDIA published an exact Cosmos3 DROID
forward-dynamics cookbook after that run. The new source resolves the main
contract uncertainties prospectively:

- DROID forward dynamics uses 16 actions at 15 FPS;
- the action is raw 10-D `[position delta, backward-framewise rot6d delta,
  gripper]` with no external normalizer;
- the visual input is wrist view on top and left/right shoulder views below;
- the official vLLM-Omni request uses a 640x540 first frame, 30 steps,
  guidance 1.0, flow shift 10.0, and the asynchronous `/v1/videos` endpoint.

The first paid gate is exactly one structured recorded-action request from the
published DROID sample. Only a structurally valid, dynamic result admits its
paired valid no-motion request. Neither request can establish generalization.

If the paired reference gate passes, the next evidence tier is a prospectively
selected set of previously unseen public Cosmos3-DROID episodes for causal and
rollout-reliability qualification. Those episodes do not provide a seven-policy
RoboArena leaderboard, so they cannot by themselves establish policy-ranking
fidelity or admit thesis support. A genuinely new independently labeled
multi-policy snapshot remains required for confirmatory ranking.

No provider was called while creating this protocol. The local canary packet is
stored at
`external-evidence-store://policy-ranking-roboarena-droid-reference-confirmation-evidence-20260729/preflight_v1`.

## Governing artifacts

- `source_freeze_v1.json`
- `protocol_v1.json`
- `protocol_v2.json`
- `protocol_v3.json`
- `environment_and_source_manifest_v1.json`
- `goal_cost_authorization_amendment_v1.json`
- `compute_authorization_allocation_1.json`
- `compute_authorization_allocation_2.json`
- `allocation_1_infrastructure_failure_v1.json`
- external canary manifest digest:
  `3f29f83f6698543bd7ce13e23b632e355031e54e2a123f0d439868bac3906f04`
