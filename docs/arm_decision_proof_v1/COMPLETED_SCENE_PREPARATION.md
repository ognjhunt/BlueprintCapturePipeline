# Completed-scene preparation

ADP-009D / day-28: turn a signed, persistent owner intent into a replayable
development-only construction input without reconstructing an uploaded asset.

This is preparation, not a simulator result or a launch authorization. The
installed service stops at `construction_prepared`; activation remains off.

## Input and implementation

- An already uploaded standard 3DGS plus a mesh in the same declared frame, or
  a standalone mesh. USD, triangle PLY, and untextured/vertex-coloured GLB are
  admitted by the current mesh adapter. External dependencies, articulations,
  and textured GLB are explicitly refused rather than silently simplified.
- Exact subject/support object names or IDs, a typed pick-and-place task, and
  an admitted destination selected by catalog ID or an unambiguous registered
  label. Destination poses use the normalized metre/Z-up runtime frame.
- Owner consent, source/companion ownership and digest checks remain mandatory.
  Descriptions without the necessary pose/success fields produce `needs_input`;
  completing the Website task-entry UX is separate from this backend path.

`task_evaluation_completed_scene_source` inspects all retained objects and keeps
the source identities. Mesh normalization preserves per-object geometry and
material bindings while converting declared units/up-axis. Nonidentity splat
frames use the pinned PlayCanvas converter with independent position, scale,
and covariance-rotation readback. Neither conversion invokes a trainer.

The real publication service reopens server-retained owner consent and source
receipts. Original bytes (and any complete frame-normalized splat) are kept in
the reserved `s3://blueprint/task-evaluation/host-only-owner-sources/` namespace.
That namespace resolves only to the local source store; it has no S3 fallback.
Only the admitted preparation metadata and derived runtime geometry are
published. No publisher admission receipt is synthesized for an owner upload.

The completed-scene compiler emits the existing six-capability construction
recipe with installed adapters and actual stage-configuration references.
Mesh appearance uses exact subtree excision. Replacement authoring preserves
the supplied mesh and applies bounded simulation physics priors, followed by
the existing independent static/native qualification stages. Splat appearance
uses the existing bounded-source ArtiFixer path; the public Scene 841757 SAM3.1
path is unchanged. The legacy ArtiFixer wire key `publisher_instance_id` carries
an explicitly owner-derived compatibility identifier on this route, alongside
the original mesh object ID and `source_origin`.

Preparation outputs never establish physical measurements, factual recovery of
unobserved surfaces, renderer fidelity, native import, grasp success, policy
performance, or physical validity. Each relevant downstream gate still has to
execute and produce its own evidence.

## Installation

The global bootstrap binds existing, independently checked destination assets;
it contains no per-scene task, credentials, or hand-maintained release SHA.
For an already admitted destination:

```bash
python -m blueprint_pipeline.task_evaluation_scene_preparation_installation \
  --destination-simready /absolute/path/passive_destination_simready_result.v1.json \
  --destination-alias "document tray" \
  --destination-alias "blue document tray"
```

The canonical `scripts/deploy_control_plane_commit.py` consumes the bootstrap
at `/etc/blueprint/task-evaluation-scene-preparation-bootstrap.json`. It installs
the managed machinery/configuration/environment files while automation is
quiesced. Its receipt records `scene_preparation_installation`. An absent
bootstrap is reported as `not_configured`, never as a working service.

The progression service uses its own preparation queue and the real preparation
worker, avoiding the legacy scene-specific worker wrapper. It derives its
release binding from deployment/runtime receipts, publishes/readbacks inputs,
and queues construction. The signed owner intent is sufficient for this local,
preparation-only handoff; a second Website round trip is unnecessary.

Free preparation identities are stored under `preparation-attempts`, separate
from paid reservations. They explicitly grant no allocation authority. The
service neither creates activation intents nor spends a paid-attempt slot.

Read the service result at
`/var/lib/blueprint/pipeline-control-plane/scene-preparation-service/latest.json`.
Check its source commit against the live intake identity, its actual per-intent
phase, and the service's `ConditionResult`/execution timestamp. An enabled timer
alone is not proof of execution.

## Verification and current checkpoint

The focused rehearsals use real producers, publication validation, preparation
consumers, provider-bundle validation/hydration, and mesh construction/static
qualification. External object-store transport is simulated. The splat
orchestration rehearsal simulates rasterization and makes no fidelity claim;
the separate nonidentity-frame check invokes the real pinned converter.

```bash
PYTHONPATH=src python -m pytest -m 'not gpu' -q \
  tests/test_completed_scene_consumer_rehearsal.py \
  tests/test_completed_scene_normalization.py \
  tests/test_scene_preparation_installation.py \
  tests/test_task_evaluation_completed_scene_progression.py \
  tests/test_task_evaluation_scene_configuration_submission_publication.py
```

This document accompanies the implementation checkpoint. Final compatibility
checks, protected-main merge, and live installation/readback are still required
before item #1 is reported complete. The original paid run remains paused.
