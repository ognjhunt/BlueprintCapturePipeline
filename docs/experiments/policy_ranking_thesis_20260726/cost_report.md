# Cost and resource report

Snapshot: 2026-07-26 20:37 America/Chicago.

| Stage | Provider mutation | Paid GPU | Measured or admitted cost |
|---|---|---|---|
| Metadata research and frozen indexing | None | None | $0 marginal provider cost |
| 49-rollout OSCAR pilot materialization | Public download only | None | $0 provider cost; approximately 175 MB local materialization |
| Label-blind action/motion diagnostic | None | None | $0 provider cost; local CPU |
| Playroom 3DGS download and 20k-splat hybrid preview | None | None | $0 provider cost; local macOS CPU with Spark.js/Chromium/SwiftShader |
| InteriorGS full-splat hybrid preview | None | Local Apple Metal only | $0 provider cost; four 1920x1440 views from all 630,898 splats measured at 41.4 seconds; no Vast or cloud GPU |
| InteriorGS native-square camera revision | None | Local Apple Metal only | $0 provider cost; eight 1024x1024 candidates in 17.95 seconds and the selected two 1536x1536 views in 8.77 seconds from all 630,898 splats; exact 224x224 OpenPI DROID inputs materialized locally |
| Local articulated Franka task-feasibility oracle | None | None | $0 provider cost; local CPU MuJoCo 3.9.0; scripted control only, not a learned-policy result |
| Local DROID closed-loop controls | None | Local Apple graphics only | $0 provider cost; exact 224x224 external/wrist observations and 10x8 action chunks executed for a frozen 168-action scripted positive and 160-action zero negative control; the positive failed, so learned-policy GPU spend was stopped |
| Local DROID joint-position controls | None | Local Apple graphics only | $0 provider cost; source-attributable 15 Hz outer/1 kHz inner control with exact 224x224 views; 168-action positive lifted 0.10949 m and passed containment/stability, while the stationary negative was rejected |
| Dynamic InteriorGS hybrid controls | None | Local Apple graphics only | $0 provider cost; frozen 300,000-splat background plus live MuJoCo Panda/can/tray segmentation and live wrist render; stationary negative rejected and 168-step positive passed |
| NVIDIA Warehouse dynamic hybrid controls | Public download and local rendering only | Local Apple graphics only | $0 provider cost; selected 198,666,960-byte sorting-area workcell subset, 224x224 USD render, live MuJoCo Panda/can/tray segmentation, 168-step positive passed, and 160-step stationary negative rejected; no Isaac physics or learned policy |
| OpenPI GPU input/request preparation | Read-only Vast and RunPod APIs only | None | $0 provider cost; 104,716-byte two-scene private bundle `52140529...`. Vast is now the lane default: a 2026-07-26 read-only snapshot proved zero global billable inventory and found qualifying 45+ GB offers, including an A40 at about $0.28/hour; the frozen launch/budget ceiling is $0.75/hour. The earlier RunPod snapshot remains fallback evidence. Capacity is advisory until create; no reservation or mutation occurred |
| OpenPI AMD64 image validation | Local Colima/BuildKit only | None | $0 provider cost; all build stages passed under x86 emulation. The approximately 22 GB uncompressed local load was canceled after layer export took 648.2 seconds; this is build validation, not exact-main release or provider startup evidence |
| NVIDIA spray-can asset dependency materialization | Public download only | None | $0 provider cost; 143,801,263 bytes across 11 non-thumbnail files |
| Pilot judge attempt 001 | One generated-only request; no accepted score | None | Provider usage was not preserved; conservatively charged the $0.015 per-request admission ceiling |
| Pilot judge attempt 002 | One generated-only request; no accepted score | None | Measured conservative API cost $0.021295; provider reported `incomplete:max_output_tokens` |
| Pilot judge attempt 003 | 44 accepted generated-only judgments; four incomplete responses; excluded from ranking | None | $1.4001325 persisted metered estimate plus $0.36 worst-case allowance for four requests in flight at interruption; $1.7601325 conservative accounting |
| V2 98-call pilot judge | 98 accepted generated-only judgments; no failed requests or blockers | None | $3.0161225 conservative metered cost; 420.179 seconds recorded for the completion invocation; $9.00 per-run cap was not approached |
| V2 98-call calibration judge | 98 accepted generated-only judgments; no failed requests or blockers | None | $2.89694 conservative metered cost; 474.981 seconds wall time; $9.00 per-run cap was not approached |
| V2 held-out judge attempt 001 | 121/686 accepted before 565 provider rate-limit blockers; partial rows later overwritten by a no-key preflight | None | $3.6246275 conservative metered cost and 3,115.916 seconds wall time observed before overwrite; unusable for held-out ranking |
| V2 held-out judge attempt 002 | Exact frozen rerun at concurrency two; 0/686 accepted and 686 final provider rate-limit blockers | None | $0 metered usage cost; 2,199.172 seconds wall time; labels remained sealed and no held-out metric exists |
| Attributed OSCAR generation for 49-rollout pilot | Authors generated the released media; not a Blueprint provider mutation | Single GH200 in paper | At the paper's 2.214 FPS, an 81-frame-by-49 serial equivalent is approximately 1,792.7 seconds (29.88 minutes), excluding load/I/O/preprocessing; this is attributed, not Blueprint-measured |
| Controlled MuJoCo task simulation | Local controls only | None | Positive/negative controls measured locally for both InteriorGS and the NVIDIA Warehouse visual-domain bridge; no learned-policy episode |
| Exact-main OpenPI image build | One DigitalOcean CPU builder, torn down | None | 1,319.309 seconds; maximum compute spend $0.0610804; immutable digest `f8f4dc01...`; provider absence confirmed |
| Two-scene learned-policy rollout | One Vast instance, torn down | RTX 6000 Ada | 991 charged GPU-seconds; conservative settlement $0.206458; 24/24 episodes completed; exact/prefix/global absence confirmed; lease released |
| Physical robot evaluation | Forbidden in this goal | Forbidden | $0 spent |

The exact-main build plus learned-policy execution consumed about 38.5 serial
compute minutes and $0.267539. Conservative paid-provider accounting across the
recorded judge attempts, builder, and GPU campaign is approximately $11.60.
This proves that the offline experiment can be inexpensive, but not that it is
substantially faster or cheaper than exhaustive physical evaluation. The 2.27x
pilot ratio is judge-only and is not an end-to-end claim. Adding the
paper-attributed single-GH200 OSCAR generation estimate gives approximately
37.34 minutes for pilot WAM generation plus judging, versus a 16.97-minute
physical action-only lower bound; that comparison omits overhead on both sides.
Physical total monetary cost remains unmeasured. These economic gaps are
retained in the `inconclusive` verdict rather than filled with assumed robot
costs.
