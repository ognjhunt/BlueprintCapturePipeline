# Buyer Evidence Trust Guide — Task Evaluation Runs

Audience: robot teams evaluating whether to buy, trust, and use a Blueprint Task
Evaluation Run, including an optional rights-cleared evidence export. This guide
describes what every export must contain, how to verify it, and exactly what it
does and does not prove. The historical standalone package name is deprecated.

## What you receive

Every qualifying evidence export from a Task Evaluation Run ships with:

| Artifact | Purpose |
|---|---|
| `buyer_package_readout.json` / `buyer_package_summary.md` | One fail-closed readout across all buyer-critical sections (schema `buyer_package_readout.v1`) |
| `dataset_card.json` | Counts, curation state, and the proof boundary |
| Site / Task / Scenario / Eval cards | The evaluated site, tasks, scenarios, and per-scenario eval evidence |
| `rights_packet.json` + proof boundaries | Rights, privacy, and provenance scope — authoritative over everything downstream |
| `data/attempts.jsonl` + `data/failure_labels.jsonl` | Every attempt and every failure label; failures are preserved, never filtered |
| `curation_report.json`, `semantic_dedup_report.json`, `sc3_action_normalization_report.json` | Quality gates the package had to pass to export |
| `calibration_report.json` | Calibration evidence; POV media additionally carries a camera metadata contract (intrinsics, extrinsics, calibration status) |
| `visual_augmentation_support_manifest.json` (when present) | Generated/model-derived media, explicitly labeled as support assets — never raw capture evidence |
| `package_index.json` + `checksums.json` | Full file inventory with SHA256 per file |
| `replay_review_instructions.md` | Step-by-step verify → review → replay protocol |

## The buyer readout fails closed

`buyer_package_readout.json` checks nine buyer-critical sections (cards,
rights/privacy/provenance, robot POV evidence, failure evidence, task success
criteria, calibration, media provenance, export integrity, replay/review
instructions). Any missing section produces `status: blocked_incomplete_package`
with named blockers — a package can be internally "export ready" and still be
blocked for buyer handoff (for example, when robot POV evidence is absent).

Pricing and entitlement wiring (`product_handoff`: SKU, entitlement id, buyer
review URL) is reported when present but is out of band and never gates
evidence review.

## Claim boundaries — what success means

Success claims sit on a strict ladder (`success_claim_contracts.py`), weakest
to strongest:

`no_claim → media_valid → review_task_success → simulator_task_success →
policy_task_success → contact_state_change_proven → physical_deployment_ready`

The readout echoes the highest truthful claim from the package's success-claim
ledger and can never invent a higher one. Hosted status projections label every
success rate with its evaluation substrate (`real_robot_outcome`,
`simulator_execution`, or `unproven_pipeline_output`).

Non-negotiable boundaries:

- Purchasing, reviewing, or receiving a package is **not** deployment approval.
- Simulator results are **not** real-world outcomes.
- Generated or model-derived media is **not** physical proof and is never raw
  capture evidence.
- A reviewer judging generated pixels as "success" is a judgment of pixels,
  not simulator state, contact physics, or real-world outcome.
- `physical_deployment_ready` requires real-robot execution evidence plus a
  named deployment approval; no simulation evidence can substitute.

## How to verify a package in 10 minutes

1. Open `buyer_package_summary.md` — confirm status and highest truthful claim.
2. Recompute SHA256 for every file in `checksums.json`.
3. Read `dataset_card.json` and confirm counts match `data/*.jsonl` line counts.
4. Spot-check failure labels against POV clips — failures must be present.
5. Read the rights packet: confirm your intended use is inside the allowed
   scope before training or redistribution.
6. Follow `replay_review_instructions.md` to replay attempts against the
   scenario matrix.
