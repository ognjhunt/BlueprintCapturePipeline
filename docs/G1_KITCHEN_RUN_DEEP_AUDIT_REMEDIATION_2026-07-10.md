# G1 kitchen deep-audit remediation status

Date: 2026-07-10  
Source audit: `G1_KITCHEN_RUN_DEEP_AUDIT_2026-07-10.md`

## Outcome

The repository-side contracts and fail-closed launch machinery from the audit
have been implemented. This does **not** close the live G1 kitchen lane. No new
immutable Isaac worker digest or fresh provider episode was produced in this
remediation, so the truthful lane status remains:

`local_contracts_advanced_live_end_to_end_task_success_not_proven`

The lane is simulation-only. Physical G1 readiness, deployment approval, and
real-world safety are outside this closure boundary.

## Finding reconciliation

| ID | Repository implementation | Evidence boundary after remediation |
|---|---|---|
| G1K-P0-01 | Added one immutable-identity `g1_kitchen_attempt_closure.v1`, independent proof rows, terminal normalization, strict row identity binding, API teardown/final-inventory requirements, and buyer-readout consumption. | A closure can pass synthetically; no fresh live closure has passed. |
| G1K-P0-02 | Wired the startup supervisor into paid parity by default. Asset, fast RTX, and review gates run on the promoted allocation; every archive is nonce/image bound; teardown and inventory fail closed. | Local integration and fault tests pass; no fresh provider promotion proves the current image. |
| G1K-P0-03 | Made the image self-identifying and added build/runtime evidence validation for package import, G1 USD, Isaac family/version, exact digest canaries, and teardown. | Build/push and exact-digest live canaries require registry credentials, an eligible linux/amd64 builder, provider authorization, and adequate disk. |
| G1K-P0-04 | Added immutable selection generations, append-only supersession, atomic attempt IDs, exact attempt-input hashes, active-pointer validation, and duplicate rejection. | New bundles enforce lineage; old evidence remains raw but ineligible. |
| G1K-P0-05 | Production geometry traverses boundable visual descendants and instance proxies, emits origin and visual provenance, and prevents nominal/link-only support geometry from proving live reach. Stance proposal and measured acceptance are separated. | A new Isaac run is required to prove repaired live G1 geometry. |
| G1K-P0-06 | The sealed path queries GR00T before action zero, requires the official GEAR-SONIC adapter, hashes the action/controller/model/FK evidence, rejects fixture/replay/shape-only substitutes, and carries state forward. | Official controller execution in the pinned worker is not yet live-proven. |
| G1K-P0-07 | Added a single-process persistent Isaac task executor and typed client. Every transition binds action, session, stage, prim, timestamps, values, units, runtime result, attestation, and registered criterion. | Microwave/dishwasher articulation success needs a live episode. |
| G1K-P0-08 | Added strict external scorer request/result validation and a calibrated HTTPS client. Forward/inverse results, timing, dimensions, units, uncertainty, calibration, state/motion hashes, termination, replay, and per-dimension errors fail closed. Historical local-CV labels are superseded. | No calibrated external scorer result exists for a new live episode. |
| G1K-P0-09 | Added every-frame ordered overview/robot-POV admission and an explicit external semantic-review adapter with frame hashes, occupancy, visibility, coherence, API provenance, and abstention/block behavior. | No fresh full-episode provider media has passed the adapter. |
| G1K-P1-01 | Catalog stock, reservation confidence, create outcome, allocation, and spend are separate. Create-without-ID is no allocation/no spend; RTX capability excludes A100/H100/H200 from the review lane. | Provider availability remains advisory until create. |
| G1K-P1-02 | Restart/storage contracts and spend reconciliation distinguish compute, container disk, persistent volume, and network volume; terminal cleanup requires absence proof. | No storage charge is flattened into compute proof. |
| G1K-P1-03 | The deterministic stance authority owns thresholds and measured validation; proposal generation receives bounded typed rejection evidence and cannot pass itself. | Live placement remains unproven until a new run. |
| G1K-P1-04 | Added bundle schema compatibility gates, immutable payload plus transport envelope, source commit/dirty-patch identity, exact artifact hashes, and worker-image evidence. Stale prepared bundles are marked ineligible rather than rewritten. | Dishwasher/microwave execution bundles cannot become eligible before current exact-image evidence exists. |
| G1K-P1-05 | Repaired CPU dependency selection, dependency-license policy, source-governance growth, systemd threshold encoding, and local WebApp HMAC compatibility. The real local sim-only forwarding gate passes. | Hosted CI on `main` cannot be current until these changes are published and rerun. |
| G1K-P2-01 | Added an append-only, lock-protected run index with relative hashed refs, selection/allocation/terminal events, and duplicate terminalization rejection. | Raw provider evidence is retained. |

## Prepared-bundle disposition

Failed or stale preparation attempts remain available as raw diagnostics but
are not execution-eligible. A replacement bundle requires a worker evidence
artifact for the exact current digest and must pass the bundle compatibility
gate. Local package preparation is not provider/runtime proof.

## Live closure prerequisites

A future paid attempt may close only when one exact source snapshot, image
digest, selection, task contract, bundle, allocation, action sequence, simulator
state, full ordered media set, scorer results, semantic review, teardown, and
zero inventory are bound into the same passing closure. Missing provider,
registry, scorer, or review evidence remains `blocked`; it is never inferred
from local tests or a completed leaf manifest.
