# Reconstruction Enhancement Audit — 2026-08-02 (licensing revision)

Status: licensing update only. Every candidate remains deterministically
rejected pending qualification. This revision extends
`reconstruction_enhancement_audit_2026-07-30.md`; that document's shared proof
boundary is unchanged and restated stronger below.

## What changed since 2026-07-30

1. **NVIDIA Fixer V2 (Cosmos) is now a registered candidate** (`method_id:
   fixer`). Source code is Apache-2.0 at
   `nv-tlabs/Fixer@b39dfcaf4eeec90dc943b057ff368c16252c6c6e`; the released
   weights are governed by the NVIDIA Open Model License Agreement and the
   model card states the model is "ready for commercial/non-commercial use"
   (Fixer V2, released November 2025, based on Cosmos-Predict-0.6B).
   Registration is not qualification: execution stays rejected until the
   dependency license receipt, checkpoint digest, Cosmos base-model digest,
   worker image smoke, and frozen real held-out baseline comparison are all
   digest-bound — the same ladder Harmonizer is held to.
2. **Harmonizer weights license confirmed commercial.** The model card and the
   NVIDIA Open Model License permit commercial use, so the recorded
   `model_license` now reads `NVIDIA Open Model License Agreement;
   commercial_use_permitted`. Nothing else moved: the rejection basis was
   never the weights license — it is the unreviewed source/dependency
   licensing and the unpinned checkpoint/base-model/worker digests, and those
   blockers are unchanged. Checkpoints: `diffusion_harmonizer.pkl` (temporal)
   and `harmonizer_nontemporal.pt`, base model Cosmos Predict2 0.6B.
3. **Difix3D+ and ArtiFixer remain rejected, with the legal bases stated
   precisely and not conflated.** Difix3D+ is limited by its NVIDIA license to
   non-commercial research or evaluation. ArtiFixer's weights are governed by
   a separate NVIDIA License restricting them to research and development
   only. Both fail the default commercial lane; neither is a "non-commercial"
   copy of the other's terms.

## Presentation ceiling (tightened)

The 2026-07-30 shared proof boundary holds, and this revision pins two
additional fail-closed booleans into every
`reconstruction_enhancement_method_audit.v1` record:

- `evaluation_evidence_use_permitted: false` and
  `policy_input_use_permitted: false` — enhancement output can never be
  evaluation evidence, ranking input, target binding, or a policy
  observation, regardless of license status.
- `offline_reconstruction_modification_permitted: false` — the models'
  offline/distillation modes (e.g. Harmonizer `offline_distillation`,
  ArtiFixer/Difix3D+ `iterative_3d_distillation`) modify the reconstruction
  itself and are outside this lane entirely.
- `presentation_enhancement_after_inputs_sealed_only: true` — only final
  image/video enhancement is in scope, and only after evaluation and policy
  inputs for the run have already been sealed.

## Qualification path (unchanged mechanics)

A new dated audit revision plus, per candidate: exact source, checkpoint,
base-model, container, and dependency digests; a hermetic worker smoke; and a
frozen real held-out comparison against the preserved unenhanced baseline.
License receipts alone flip nothing.

Primary sources: [Fixer code](https://github.com/nv-tlabs/Fixer),
[Fixer model card](https://huggingface.co/nvidia/Fixer),
[Harmonizer model card](https://huggingface.co/nvidia/Harmonizer),
[official NVIDIA NuRec documentation](https://docs.nvidia.com/nurec/fixer/model.html),
[ArtiFixer model card](https://huggingface.co/nvidia/ArtiFixer),
[Difix3D+ license](https://github.com/nv-tlabs/Difix3D/blob/main/LICENSE.txt).
