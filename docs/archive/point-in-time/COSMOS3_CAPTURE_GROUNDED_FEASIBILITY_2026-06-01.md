# Cosmos 3 Capture-Grounded Feasibility

> Archived feasibility record. Cosmos is a replaceable support candidate, not product doctrine.

Date: 2026-06-01

Status: repo-local feasibility and preflight implementation. No Cosmos 3 model
download, GPU run, NVIDIA service call, or provider job was performed.

## Research Basis

Primary sources reviewed:

- NVIDIA Cosmos 3 research page: https://research.nvidia.com/labs/cosmos-lab/cosmos3/
- NVIDIA Cosmos 3 technical report: https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf
- NVIDIA Cosmos repository: https://github.com/NVIDIA/cosmos
- OpenAI Codex goals guide: https://developers.openai.com/codex/use-cases/follow-goals

The pasted research direction is mostly right with one important correction:
Cosmos 3 can be a candidate reasoner, generator, and evaluation layer around
Blueprint captures, but it should not be described as turning one walkthrough
into a persistent exact simulator of a house or facility.

The technical report describes Cosmos 3 as jointly modeling language, image,
video, audio, and action, and as supporting modes that include multimodal
reasoning, text-to-image, image-to-video, video-to-video future prediction,
video transfer, forward dynamics, inverse dynamics, and policy-style
video-action generation. It also makes the scale and hardware posture explicit:
At the date of this original assessment, Nano and Super were the released
variants in the paper, with Nano built around an
8B dense transformer and 16B total parameters, and Super around a 32B dense
transformer and 64B total parameters. Edge is described as a later release. The
report includes single-GPU serving measurements for Nano on H100 NVL and B200,
so "single GPU" should mean an adequate NVIDIA workstation/datacenter GPU, not
this local Mac.

Update as of 2026-07-21: NVIDIA released the 4B Cosmos 3 Edge model on
2026-07-20. It remains an unqualified experimental candidate for Blueprint and
does not inherit the Nano-specific SC3-Eval recipe or results. See
[`NVIDIA_SIGGRAPH_2026_STACK_IMPACT_2026-07-21.md`](NVIDIA_SIGGRAPH_2026_STACK_IMPACT_2026-07-21.md).

## Blueprint Fit

Cosmos 3 is a fit for Blueprint only when it is bounded by the existing evidence
ledger:

- `BlueprintCapture` owns raw walkthrough media, timestamps, device metadata,
  poses, intrinsics, depth when present, site identity, and rights/provenance.
- `BlueprintCapturePipeline` owns privacy-safe media, geometry, site-reference
  memory, Cosmos Predict 2.5 export/benchmark artifacts, and downstream package
  assembly.
- `Blueprint-WebApp` owns buyer, hosted-access, licensing, and public claim
  posture.

Cosmos 3 can reduce full sim-ready digital twin pressure for:

- visual site review
- site-conditioned future-frame prediction
- synthetic perception/navigation video
- capture quality and consistency evaluation

Cosmos 3 does not remove the need for:

- capture provenance
- privacy-safe conditioning media
- rights and derived-generation permissions
- pose, intrinsics, depth, and geometry summaries
- stable site-reference memory
- held-out revisits or second-pass validation
- action or teleoperation logs for action-policy claims
- measured geometry or sim-ready assets for collision, contact, manipulation,
  door/drawer interaction, and safety-critical robot evaluation

Generated Cosmos clips are derived support artifacts. They are not ground truth.

## Landed Local Checker

The reusable preflight lives in:

- `src/blueprint_pipeline/synthesis/cosmos3_readiness.py`
- `scripts/check_cosmos3_readiness.py`
- `tests/test_cosmos3_readiness.py`

The checker reads only local artifacts:

- `capture_descriptor.json`
- `raw/manifest.json` and raw sidecars
- `raw/capture_upload_complete.json`
- raw and privacy-safe walkthrough media paths
- `pipeline/geometry/geometry_summary.json`
- `sites/{site_id}/reference_memory/site_reference_manifest.json`
- `sites/{site_id}/reference_memory/site_reference_index.jsonl`
- `pipeline/cosmos_training_export/manifest.json`
- existing Cosmos Predict 2.5 benchmark manifests when present

It writes:

- `pipeline/cosmos3_readiness/cosmos3_capture_grounded_readiness.json`
- `pipeline/cosmos3_readiness/cosmos3_capture_grounded_readiness.md`

Run it with:

```bash
python3 scripts/check_cosmos3_readiness.py --capture-root /path/to/bucket/scenes/<scene_id>/captures/<capture_id>
```

The report is deliberately a local preflight. It sets:

- `provider_jobs_called=false`
- `model_download_required=false`
- `claim_policy=capture_grounded_local_preflight_only`

## Claim Boundary

Allowed internal conclusion:

Cosmos 3 is a credible candidate model family for a Blueprint reasoner,
generator, and evaluator layer if the capture has raw evidence, privacy/rights
lineage, pose/intrinsics/depth or reference-indexable geometry, site-reference
memory, Cosmos export substrate, and held-out validation evidence.

Blocked public claims from this preflight alone:

- Blueprint runs Cosmos 3 live.
- Cosmos 3 replaces digital twins for all robot evaluation.
- A single walkthrough creates an exact persistent simulator.
- Generated Cosmos output is ground truth.
- Local degraded geometry or fallback geometry proves live provider-native
  world-model readiness.

## What Can Be Done Today

Done in this pass:

- Added deterministic preflight code.
- Added a local CLI wrapper.
- Added tests covering stack mapping, fallback geometry rejection, held-out
  revisit requirements, and report writing.
- Added this repo-local feasibility memo.

Next local-only steps:

1. Run the checker on one staged real or fixture capture.
2. If it reports `missing_geometry_summary`, run the local geometry lane with
   `provider=local_sfm`.
3. If it reports missing site-reference artifacts, run the retrieval index lane
   after stable site identity and reference-indexable geometry exist.
4. If it reports missing Cosmos export, run the existing Cosmos Predict 2.5
   export lane before any model download.
5. Keep action-policy and collision/contact simulation claims blocked until
   action logs and stronger geometry/sim evidence exist.
