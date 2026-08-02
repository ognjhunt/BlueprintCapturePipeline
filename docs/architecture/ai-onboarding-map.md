# AI Onboarding Map

This map is for new engineers and AI agents (Claude, Codex, or other) entering
`BlueprintCapturePipeline`. Read it after the root `PLATFORM_CONTEXT.md`,
`WORLD_MODEL_STRATEGY_CONTEXT.md`, `VISION.md`, and `AGENTS.md`; when documents
disagree, apply [`../DOCTRINE_PRECEDENCE.md`](../DOCTRINE_PRECEDENCE.md).

The repo turns raw capture into a maintained Site-Task Testbed, then routes one
customer-facing Task Evaluation Run at claim level to qualified evidence. Raw
capture remains authoritative; generated, derived, provider, and hosted outputs
remain projections or support evidence.

## Route At A Glance

| Concern | Main files | Primary outputs | Truth label |
| --- | --- | --- | --- |
| Raw capture materialization | `src/blueprint_pipeline/materialization.py`, `scripts/stage_capture_bundle.py` | `capture_descriptor.json`, `qa_report.json`, `frames/index.jsonl` | Capture-grounded descriptor of raw evidence |
| Capture->package orchestration and trust support | `src/blueprint_pipeline/site_package_orchestrator.py` (formerly `qualification.py`) | `pipeline/qualification_summary.json`, `buyer_trust_score.json`, `rights_and_compliance_summary.json`, `world_model_fit_summary.json` | Core orchestration spine; the qualification/trust *outputs* it writes are support artifacts, not the product center |
| Canonical site package | `src/blueprint_pipeline/canonical_site_package.py` | `pipeline/site_package/canonical_site_package.json` | Derived package contract grounded in raw capture |
| Decision/evidence control plane | `decision_evidence_contracts.py`, `decision_evidence_router.py`, `task_site_measurement_routing.py`, `measurement_site_evidence_contracts.py`, `measurement_site_evidence_bridge.py`, `measurement_sensor_stream_pairing.py`, `measurement_observation_qualification.py`, `measurement_adapter_runtime.py`, `measurement_adapter_execution.py`, `measurement_pinocchio_coal_kinematic_adapter.py`, `measurement_geometry_kinematic_development_suite.py`, `measurement_mujoco_adapter.py`, `measurement_newton_rigid_adapter.py`, `measurement_drake_rigid_adapter.py`, `measurement_sapien_rigid_adapter.py`, `measurement_geometry_contact_development_suite.py`, `measurement_geometry_contact_cross_engine_development_suite.py`, `measurement_geometry_contact_drake_development_suite.py`, `measurement_geometry_contact_sapien_development_suite.py`, `measurement_mujoco_articulation_adapter.py`, `measurement_geometry_contact_articulation_development_suite.py`, `measurement_mujoco_insertion_adapter.py`, `measurement_geometry_contact_insertion_development_suite.py`, `measurement_mujoco_flex_cloth_adapter.py`, `measurement_deformation_cloth_development_suite.py`, `measurement_mujoco_granular_adapter.py`, `measurement_deformation_granular_development_suite.py`, `measurement_chrono_granular_adapter.py`, `measurement_deformation_granular_chrono_development_suite.py`, `measurement_direct_tactile_adapter.py`, `measurement_tactile_development_suite.py`, `measurement_world_model_action_fidelity_adapter.py`, `measurement_world_model_action_fidelity_suite.py`, `measurement_opencv_observation_adapter.py`, `measurement_observation_development_suite.py`, `measurement_pyelastica_cable_adapter.py`, `measurement_deformation_cable_development_suite.py`, `measurement_dlo_lab_cable_adapter.py`, `measurement_qualification_benchmarks.py`, `measurement_research_monitor.py`, `scripts/bootstrap_measurement_drake_development.py`, `scripts/bootstrap_measurement_chrono_development.py`, `decision_evidence_execution.py`, `physical_outcome_learning.py` | maintained testbed, request, task/site measurement requirements, site evidence profile, adapter research artifacts and development receipts, scoped qualification, plan, normalized results, Decision Envelope, physical-outcome join | Claim-level qualified single/composite measurement route or explicit abstention; development execution remains non-authorizing |
| Agentic Task Evaluation Supervisor | `task_evaluation_supervisor/`, `run_e2e.py`, `decision_evidence_cli.py`, `docs/architecture/task-evaluation-supervisor.md`, `docs/TASK_EVALUATION_SUPERVISOR_RUNBOOK.md` | Required capture-build ingress, SDK agent proposals, invocation manifests, append-only event ledger, terminal supervisor report | OpenAI Agents SDK control plane around—not inside—the deterministic proof kernel; every canonical capture build enters this stage |
| Provider adapter inputs | `src/blueprint_pipeline/canonical_site_package.py`, `src/blueprint_pipeline/provider_preview.py` | `pipeline/site_package/provider_adapter_inputs/world_labs_marble.json`, `worldlabs_request_manifest.json` | Provider-specific projection from canonical package |
| Privacy-safe media | `src/blueprint_pipeline/privacy_processing.py`, `docs/PRIVACY_RUNNER_SERVICES.md` | `privacy/final_walkthrough.*`, `pipeline/privacy_processing_manifest.json`, `pipeline/worldlabs_input_manifest.json` | Derived privacy-cleared media |
| Geometry and runtime | `src/blueprint_pipeline/geometry_stage.py`, `src/blueprint_pipeline/evaluation_prep_stage.py`, `src/blueprint_pipeline/native_runtime_backend.py` | `pipeline/geometry/*`, `pipeline/evaluation_prep/*`, hosted runtime records | Derived runtime support, live only when proof labels say so |
| Retrieval memory | `src/blueprint_pipeline/retrieval_index_stage.py` | `world_model_export/*`, `sites/<site_id>/reference_memory/*` | Derived site memory, privacy-safe by default |
| WebApp sync | `src/blueprint_pipeline/webapp_sync.py`, `src/blueprint_pipeline/alpha_readiness.py` | `pipeline/webapp_sync_result.json`, `pipeline/alpha_readiness_summary.json` | Projection to buyer/control-plane surfaces |
| Legacy paid-provider gate | `scripts/run_paid_marketplace_launch_gate.py`, `docs/PAID_MARKETPLACE_BETA_LAUNCH_GATE.md` | `output/paid_marketplace_launch_gate.md`, `.json` | Compatibility evidence snapshot; not a marketplace product |

## Capture Materialization Path

Materialization starts from a staged raw bundle under:

```text
scenes/<scene_id>/captures/<capture_id>/raw/
```

The durable trigger is:

```text
scenes/<scene_id>/captures/<capture_id>/raw/capture_upload_complete.json
```

The local materializer is `src/blueprint_pipeline/materialization.py`. It reads
`raw/manifest.json`, `raw/intake_packet.json`, `raw/capture_context.json`, raw
video candidates, ARKit/ARCore/companion-phone sidecars, route anchors, checkpoint
events, and optional task hypotheses. It emits:

```text
scenes/<scene_id>/captures/<capture_id>/capture_descriptor.json
scenes/<scene_id>/captures/<capture_id>/qa_report.json
scenes/<scene_id>/captures/<capture_id>/frames/index.jsonl
```

The descriptor is the bridge between raw capture truth and downstream package
work. It should describe what exists, not make downstream proof claims.

Common entrypoints:

```bash
python3 scripts/stage_capture_bundle.py --source-bundle /path/to/raw --storage-root /data/blueprint-storage --bucket local-blueprint --copy --run-qualification
python -m blueprint_pipeline.capture_orchestrator
```

## Capture->Package Orchestration And Trust Outputs

`src/blueprint_pipeline/site_package_orchestrator.py` (formerly
`qualification.py`; the old dotted path remains a deprecated import alias) is
the largest orchestration module and the core capture->package spine. It runs
intake/scoping/completeness checks, object indexing, optional Gemini review,
privacy post-processing, World Labs input preparation, package assembly,
preview routing, WebApp sync, and alpha-readiness summary writes. The
qualification/readiness *outputs* it writes are optional support artifacts per
platform doctrine; the module itself is not optional.

Important outputs under `pipeline/` include:

```text
qualification_summary.json
capture_quality_summary.json
rights_and_compliance_summary.json
buyer_trust_score.json
world_model_fit_summary.json
capturer_payout_recommendation.json
recapture_requirements.json
provider_preview_status.json
provenance_summary.json
gemini_capture_fidelity_review.json
rights_provenance_review.json
```

These are important support artifacts for trust, review, pricing, readiness, and
WebApp state. They are not the company product center and must not override raw
capture truth or canonical package truth.

## Canonical Site Package Path

The canonical package layer lives in:

```text
src/blueprint_pipeline/canonical_site_package.py
```

It writes:

```text
pipeline/site_package/canonical_site_package.json
pipeline/site_package/provider_adapter_inputs/world_labs_marble.json
```

The canonical package should be the first place an agent looks for package-level
truth about conditioning inputs, semantic task context, site identity/topology,
device modality, rights/privacy/provenance, provider readiness, and truth labels.

It is provider-agnostic. Adapter files are derived from it and may be swapped as
World Labs, internal viewers, Cosmos-like paths, or future providers change.

## Provider Adapter Path

Provider generation is optional and replaceable. The current primary adapter path
is World Labs Marble:

```text
src/blueprint_pipeline/provider_preview.py
```

The World Labs adapter consumes the canonical provider adapter input, not raw
capture data directly, when the canonical package is available. It writes:

```text
pipeline/worldlabs_request_manifest.json
pipeline/worldlabs_operation_manifest.json
pipeline/worldlabs_world_manifest.json
pipeline/provider_run_manifest.json
pipeline/preview_manifest.json
```

Provider output is a generated projection for review or hosted preview. It is not
authoritative raw-site evidence, not a live launch proof by itself, and not a
reason to claim provider readiness when credentials, SDKs, privacy input, or
operation manifests are missing.

Do not run live provider jobs unless the user explicitly approves it.

## Privacy Runner Path

Privacy service contracts are documented in:

```text
docs/PRIVACY_RUNNER_SERVICES.md
```

The production preview path requires privacy-safe walkthrough media before World
Labs submission:

```text
BlueprintCapture upload -> storage trigger -> materialize -> site-package orchestration -> privacy/final_walkthrough.mov -> World Labs generate/poll -> WebApp sync -> catalog launch
```

GPU-backed services are expected to sit behind URL/token contracts:

```text
PRIVACY_SAM3_URL
PRIVACY_VIP_URL
PRIVACY_DEPTH_ANYTHING_URL
PRIVACY_DEEPPRIVACY2_URL
PRIVACY_RUNNER_TOKEN
```

Raw-video World Labs bypass is allowed only for temporary internal demos when
`BLUEPRINT_ALLOW_RAW_WORLDLABS_BYPASS=true`. The bypass must remain labeled
non-production and unredacted.

## Geometry And Runtime Path

Geometry contract truth is documented in:

```text
docs/GEOMETRY_LANE_CONTRACT.md
```

The implementation is:

```text
src/blueprint_pipeline/geometry_stage.py
scripts/run_geometry_lane.py
```

It writes:

```text
pipeline/geometry/geometry_manifest.json
pipeline/geometry/geometry_summary.json
pipeline/geometry/geometry_run_status.json
pipeline/geometry/logs/provider_request.json
pipeline/geometry/logs/provider_result.json
pipeline/geometry/camera/*
pipeline/geometry/depth/*
pipeline/geometry/confidence/*
```

For non-ARKit captures, current `local_sfm` output is synthetic diagnostic
geometry only. It must remain blocked from reference-media indexing. Only a live
`video_to_world` provider result with `fallback_used=false`,
`provider_native_result=true`, `site_frame_available=true`,
`scale_resolved=true`, and `geometry_live_ready=true` can satisfy
site-faithful/SWM-style geometry proof. Fallback geometry is useful for local
contract shape debugging only.

Runtime-facing package work is mostly in:

```text
src/blueprint_pipeline/evaluation_prep_stage.py
src/blueprint_pipeline/native_runtime_backend.py
src/blueprint_pipeline/native_runtime_service.py
```

These produce or serve hosted-session and site-world runtime artifacts. They
should keep generated/chunked/runtime state separate from raw capture and package
truth.

## WebApp Sync Path

Pipeline-to-WebApp projection is implemented in:

```text
src/blueprint_pipeline/webapp_sync.py
src/blueprint_pipeline/alpha_readiness.py
```

The key output is:

```text
pipeline/webapp_sync_result.json
```

By default, sync requires real upstream links before projecting hosted-review or
buyer-access state:

```text
site_submission_id
request_id
buyer_request_id
capture_job_id
```

Placeholder sync is an explicit internal fallback, not launch evidence.

## Paid Marketplace Gate Path

The paid marketplace automated gate is:

```text
scripts/run_paid_marketplace_launch_gate.py
docs/PAID_MARKETPLACE_BETA_LAUNCH_GATE.md
```

It writes evidence snapshots under:

```text
output/paid_marketplace_launch_gate.md
output/paid_marketplace_launch_gate.json
```

Those files are useful evidence snapshots from one run. They are not authority by
themselves and must be refreshed before being used for current launch claims.
