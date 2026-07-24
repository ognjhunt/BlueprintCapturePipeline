# Customer output contracts and generated-media privacy

This document covers the third group of changes, following
`EVALUATOR_ATTRIBUTION_AND_PUBLIC_ANCHOR.md` and
`EXECUTION_COST_AND_ARCHITECTURE_GATES.md`. Where those concerned what Blueprint
can honestly claim internally, this concerns what a customer actually receives —
and one privacy defect that sat on the axis Blueprint's own doctrine calls
authoritative.

## 1. Generated media does not inherit its source's redaction

**This is the item to read first.** `privacy_processing` redacts capture
walkthrough video and emits a verification report. That control protects the
*captured* pixels. It cannot protect pixels a generative model invents
afterwards.

A world model conditioned on a site frame produces new pixels that can contain a
face, a badge, a name on a whiteboard, or a proprietary fixture — because the
conditioning frame still carried it, because inpainting over a redacted region
reconstructed something recognisable, or because the model simply synthesised a
plausible person into a workplace scene. The generated frame then flowed onward
carrying the source clip's rights metadata, into hosted sessions and
customer-visible artifacts.

That is privacy laundering through generation: the redaction evidence belonged to
a different pixel array, and inheriting it asserts a property nobody checked.

`blueprint_pipeline.generated_media_privacy` enforces three principles:

1. **Redaction does not survive generation.** A generated artifact's redaction
   status starts *unverified* regardless of how clean its source was, and every
   emitted artifact row states `redaction_status_inherited_from_source: false`.
2. **Conditioning provenance is part of the control.** Generating from
   unredacted pixels contaminates the output at the source, so conditioning
   assets must themselves be redaction-verified — `raw_capture` and friends are
   refused outright.
3. **Consent is checked at generation and again at release**, and artifacts
   carry `takedown_keys` (scene, capture, generation, artifact digests) so a
   later revocation reaches the derivative through the existing
   `consent_takedown` enumeration.

Release scopes are `blocked` / `internal_review_only` / `customer_visible`.
Customer-visible additionally requires a redaction pass over the **generated**
pixels, and that pass must cover those exact bytes — a verification naming a
different artifact digest proves nothing about this one. Absent consent is not
permission: `unknown` fails closed exactly as `revoked` does.

The hosted runtime now consults this at the serving boundary
(`native_runtime_backend.media_response`). Enforcement there is **staged**, and
the distinction matters:

- A chunk that **carries a contract** is held to it unconditionally. Anything
  short of `customer_visible` is withheld behind a labelled placeholder
  (`X-Blueprint-Render-Source: withheld_privacy_contract`) with the denial
  reason attached.
- A chunk with **no contract** is served and labelled
  `X-Blueprint-Generated-Media-Privacy: unverified`. No producer attaches a
  contract to a runtime chunk yet, so denying every uncontracted chunk would
  withhold the whole existing hosted-media path while protecting nothing that is
  not already unprotected. Setting
  `BLUEPRINT_ENFORCE_GENERATED_MEDIA_PRIVACY` withholds it outright.

State the consequence plainly: **until a producer attaches contracts, the
default posture labels rather than blocks**, so this defect is contained and
visible, not closed. The label is an honest report of what was checked; it is
not a clearance. Closing it requires the generation path to emit a contract per
chunk, at which point strict mode becomes the default.

## 2. Anchor return kits

Calibration joins a real-world anchor to a prediction on four exact keys, and a
row that does not reproduce all four is rejected as `unmatched_actual_row`. The
failure that created was expensive and late: an operator ran physical trials for
days and discovered only at ingest that the rows could not join.

`blueprint_pipeline.anchor_return_kit` issues, per prediction per planned trial,
a row that already carries all four keys — leaving only the outcome fields to
fill in — and validates a returned file *before* ingest. A mistyped `task_id` is
now caught while the robot is still available, alongside duplicate trials,
incomplete rows, and trials that were never returned.

## 3. Frozen held-out splits in every training package

`grep` for `holdout|held_out|registered_split` across
`post_training_data_package.py` previously returned nothing. Blueprint sold
captured clips as training data with no indication of which to keep out of
training, so a buyer's evaluation set could overlap the clips they had just been
sold — invisibly, from inside the package.

Every package now ships `holdout_split.json` and `holdout_package_check.json`:

- the held-out cut is carved from the **same capture**, so it is in-distribution
  rather than a different site with different lighting and layout;
- assignment is deterministic from `sha256(split_id || clip_id)` ranked within
  each stratum, so it is reproducible, independent of clip ordering, and cannot
  be quietly reshuffled until a favourable split appears;
- partitions are provably disjoint, and a training payload containing a held-out
  clip **fails closed**; and
- clips outside the frozen split are reported as unknown rather than assumed
  safe.

A capture too small to yield a meaningful held-out cut is blocked rather than
shipped with a token one.

## 4. One integrity contract across all customer roots

The Post-Training Data Package had real integrity machinery — a package index,
per-member checksums, and a self-excluded root signature. The other three
customer-facing manifest builders (`build_rights_provenance_review`,
`build_site_package_manifest`, `build_proof_pack_manifest`) accepted
`artifact_uris`: a bare mapping of names to locations.

A URI is a location, not a commitment. It says where bytes were, not which bytes
they were, so anything swapped, truncated, or regenerated between manifest time
and download time was undetectable — and two customers handed "the same" package
had no way to establish they received the same thing.

`blueprint_pipeline.signed_delivery_bundle` lifts the PTDP pattern into a form
any root can use. All three builders now emit a `delivery_integrity` block. A
member without a digest is a **blocker**, not a warning, so the gap is visible in
the manifest rather than invisible to the recipient. The root digest covers the
member *set*, so adding or removing a member changes it. Ed25519 signing is
optional; a bundle without a signature is still digest-verifiable and says so.

## 5. The OEM handoff reports its own completeness

`_oem_handoff_summary` returned six keys and degraded to a one-line prose string
when the skill returned nothing, against a `SKILL.md` naming ten required inputs.
A handoff missing most of its evidence was indistinguishable from a complete one.

It now enumerates the required inputs, reports `present_inputs` /
`missing_inputs` / `required_input_coverage`, and carries `status: incomplete`
with explicit blockers. Completeness means the inputs are present — not that
platform fit has been established.

The hosted runtime's placeholder card is also now labelled
(`X-Blueprint-Content-Kind: placeholder`, `X-Blueprint-Is-Site-Observation:
false`) so no client can present a status card as a rendered observation of the
site.

## 6. Per-site difficulty profiles

A policy scoring 80% at one site and 55% at another has not necessarily
regressed — the second site may have tighter clearances, more reflective floors,
or harder objects. Without a published description of that difference, every
cross-site comparison silently attributes site variance to the policy.

`blueprint_pipeline.site_difficulty_profile` derives six axes — spatial
constraint, geometric complexity, visual conditions, object difficulty, dynamic
environment, task horizon — entirely from measurements the pipeline already
computes and gates on. Scores are banded (`low` → `very_high`) because the
underlying measurements do not support distinguishing 0.61 from 0.64.

Two deliberate refusals: an axis with no inputs is reported `measured: false`
rather than scored as easy, and `compare_sites` publishes the **spread** rather
than a correction factor. Difficulty is a covariate for interpretation, never a
normaliser — dividing a success rate by a difficulty number would invent
precision nobody measured.

## Commands

```bash
python -m blueprint_pipeline.anchor_return_kit issue --input p.json --output kit.json --csv kit.csv
python -m blueprint_pipeline.anchor_return_kit validate --kit kit.json --returned filled.csv --output report.json
python -m blueprint_pipeline.post_training_holdout_split build --input clips.json --output split.json
python -m blueprint_pipeline.post_training_holdout_split check --split split.json --input pkg.json --output check.json
python -m blueprint_pipeline.signed_delivery_bundle seal --input members.json --output bundle.json [--sign]
python -m blueprint_pipeline.signed_delivery_bundle verify --input bundle.json --output verify.json
python -m blueprint_pipeline.site_difficulty_profile --input site.json --output profile.json
python -m blueprint_pipeline.generated_media_privacy --input gen.json --output contract.json
```

No command contacts a provider, allocates paid resources, or upgrades a claim.
