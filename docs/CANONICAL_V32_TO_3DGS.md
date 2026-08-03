# Canonical Blueprint V3.2 / ARKitScenes proxy to 3DGS

The service that turns the capture into a 3D Gaussian Splat is the
**reconstruction worker**, not the iOS app and not Niantic NSDK:

- **Postshot Splat3** is the precommitted primary trainer.
- **Nerfstudio Splatfacto 1.1.5 / gsplat 1.4.0** is the open, reproducible
  comparison trainer.
- Both receive the same byte-identical candidate-only COLMAP dataset.
- NSDK is an optional supplemental recorder. Its archive is never substituted
  for the canonical Blueprint Raw Contract V3.2 bundle and it is disabled by
  default.

## Data path

```text
BlueprintCapture V3.2 bundle OR explicitly admitted ARKitScenes proxy
  -> strict decoded-PTS / retention / ARKit / intrinsics validation
  -> exact digest-bound task/site frame-selection admission (no default)
  -> frozen training / validation / evaluator-hidden split
  -> candidate RGB + raw ARKit camera bindings
  -> confidence-filtered captured LiDAR depth back-projection
  -> depth-seeded COLMAP text dataset
  -> Postshot Splat3 primary worker       -> standard 3DGS PLY + PSHT + log
  -> Splatfacto comparison worker         -> standard 3DGS PLY + config + log
  -> digest-bound campaign receipt
  -> independent hidden-view evaluation (separate gate)
```

No step estimates missing raw timestamps, silently changes the raw ARKit poses,
or exposes evaluator-hidden pixels to either trainer. Captured LiDAR depth is an
initialization surface; it does not itself prove independent metric accuracy,
complete geometry, collision suitability, Isaac compatibility, or physical
task success.

The ARKitScenes path is a public-dataset proxy lane, not Blueprint Raw V3.2.
Its source admission records `raw_contract_3_2_proven=false`, binds the exact
retained compilation and official-loader world frame, and preserves the same
candidate/hidden separation. It cannot upgrade proxy data into raw-capture
truth.

## 1. Prepare the immutable training plan

Run this on the trusted Pipeline host after the canonical bundle has been
materialized:

```bash
python -m blueprint_pipeline.canonical_3dgs_cli prepare \
  --capture-root /captures/<capture-id> \
  --output-root /derived/<capture-id> \
  --intake-id <intake-id> \
  --capture-digest sha256:<64-hex> \
  --task-site-selection-profile /control/<capture-id>-frame-selection.json \
  --source-commit-sha <40-hex>
```

The command performs no provider allocation and no paid work. It writes the
preparation result, the COLMAP export result, a content-bound
`trainer_input/colmap_dataset_*/` directory, and
`canonical_3dgs_execution_plan.json`. The plan freezes Postshot as `primary`
and Splatfacto as `comparison`, hashes every input artifact, records the frozen
split and capture digest, and leaves `quality_winner` unset.

The selection profile must be `task_site_frame_selection_profile.v1`, bind the
exact `downstream_candidate_manifest.json` digest and current rights/revocation
evidence, and name explicit task/site parameters. The raw V3.2 path has no
`--maximum-frames` fallback: absent or invalid selection authority abstains with
`task_site_evidence_profile_with_frame_selection_parameters`. The preparation
records the candidate-manifest, selection-profile, and admission digests plus
the exact admitted decoded ordinals.

For an already compiled ARKitScenes proxy, name both retained roots explicitly;
the command never searches for a "newest" artifact:

```bash
python -m blueprint_pipeline.canonical_3dgs_cli prepare \
  --source-profile public_dataset_arkitscenes_proxy \
  --proxy-root /retained/40958756/compiled/arkitscenes_proxy_<digest> \
  --source-artifact-root /retained/40958756 \
  --output-root /derived/40958756 \
  --source-commit-sha <40-hex>
```

Package the exact plan and candidate-only dataset once for cross-platform
transport:

```bash
python -m blueprint_pipeline.canonical_3dgs_cli transport package \
  --plan /derived/<capture-id>/canonical_3dgs_execution_plan.json \
  --dataset-root /derived/<capture-id>/trainer_input/colmap_dataset_<digest> \
  --worker-wheel /derived/<capture-id>/worker-wheel/<exact-worker.whl> \
  --bundle /derived/<capture-id>/canonical_3dgs_transport.zip \
  --receipt /derived/<capture-id>/canonical_3dgs_transport_receipt.json
```

The ZIP is deterministic and contains neither evaluator-hidden pixels nor
credentials. Its receipt cannot authorize upload, provider allocation, or
paid work. Upload authorization and paid-resource admission remain separate,
explicit control-plane inputs.

The plan also digests every Python source file in the installed
`blueprint_pipeline` package. After the implementation is committed, build a
pure-Python wheel from the exact `source_commit_sha` with
`scripts/build_canonical_3dgs_worker_wheel.sh <40-hex-commit> <output-dir>` and
stage that wheel on both workers. The builder uses `git archive` in a fresh
temporary tree and refuses a wheel whose Python-source digest differs from the
commit, preventing stale files in a reused setuptools `build/` directory.
Each worker recomputes the installed-package digest before training, records it
in its receipt, and finalization rejects a different package even if its
marketing version is unchanged.

## 2. Run Postshot on the admitted Windows worker

The worker must already be provisioned, licensed, watchdog-protected, and
authorized by the existing paid-resource control plane. This command does not
create or pay for that worker.

After allocation, bootstrap the exact wrapper wheel and measure the installed
Postshot CLI before granting execution authority:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_canonical_postshot_worker.ps1 `
  -WorkerWheel C:\work\blueprint_capture_pipeline-2.0.0-py3-none-any.whl `
  -PostshotVersion <observed-Postshot-version>
$env:Path = "C:\work\blueprint-canonical-3dgs-venv\Scripts;$env:Path"
```

The setup requires Python 3.10-3.12 on the admitted image, installs only the
pinned NumPy/Pillow runtime needed by these worker entry points, installs the
exact Blueprint wheel with `--no-deps`, smoke-checks both commands, and emits
the actual Postshot executable digest without starting training.

Before execution, obtain an `execute_ready`
`reconstruction_gpu_canary_admission.v1` through
`python -m blueprint_pipeline.paid_resource_allocator gpu-canary`. It must use
operation `trainer_canary` and bind the plan digest, transport bundle, dataset,
split, commit, image, authority, budget, TTL, watchdog, provider-zero preflight,
and `retry_cap=0`. A provider-specific launcher is forbidden. Then bind that
allocator receipt plus the measured trainer digest into the arm admission:

```bash
python -m blueprint_pipeline.canonical_3dgs_cli admit-worker \
  --transport-receipt /derived/<capture-id>/canonical_3dgs_transport_receipt.json \
  --arm postshot-primary \
  --worker-platform windows \
  --paid-allocator-admission /control/postshot-allocator-admission.json \
  --worker-image-digest <image-name>@sha256:<64-hex> \
  --trainer-runtime-digest sha256:<Postshot-CLI-exe-digest> \
  --trainer-runtime-version <observed-Postshot-version> \
  --authority-id <explicit-authority-id> \
  --max-spend-usd <ceiling> \
  --hard-ttl-seconds <ttl> \
  --provider-upload-authorized \
  --paid-compute-authorized \
  --watchdog-armed \
  --provider-zero-before-allocation \
  --timestamp <UTC-ISO8601> \
  --output /derived/<capture-id>/postshot-worker-admission.json
```

Omitting any authorization or watchdog flag produces a blocked record. An
arbitrary allocation hash is not accepted. The admission is arm-specific,
expires at its issued timestamp plus its TTL, has `retry_cap=0`, and requires
provider-zero proof again after execution. The current Vast-first allocator is
Linux-oriented; until a Windows `trainer_canary` adapter is qualified, the
Postshot arm remains explicitly blocked rather than bypassing the allocator.

Set `POSTSHOT_LOGIN_EMAIL` and `POSTSHOT_LOGIN_PASSWORD` in the worker's secret
environment. They are passed as Postshot global flags, redacted from the
receipt, and never written into the plan. The worker rehashes the Postshot
executable against the admission immediately before training. Then run:

```powershell
$env:BLUEPRINT_WORKER_IMAGE_DIGEST = "<image-name>@sha256:<64-hex>"
```

```powershell
python -m blueprint_pipeline.canonical_3dgs_cli transport extract `
  --bundle C:\work\canonical_3dgs_transport.zip `
  --receipt C:\work\canonical_3dgs_transport_receipt.json `
  --output-root C:\work\materialized

python -m blueprint_pipeline.canonical_3dgs_cli run-arm `
  --arm postshot-primary `
  --plan C:\work\materialized\<bundle-digest>\campaign\canonical_3dgs_execution_plan.json `
  --dataset-root C:\work\materialized\<bundle-digest>\campaign\dataset `
  --output-root C:\work\results\postshot-primary `
  --receipt C:\work\results\postshot-primary\worker_receipt.json `
  --transport-receipt C:\work\canonical_3dgs_transport_receipt.json `
  --admission C:\work\postshot-worker-admission.json
```

The concrete command imports the COLMAP cameras and points into full-resolution
Postshot Splat3 with `--no-recenter-points`, and requires a `.psht` project, a
standard 3DGS `.ply`, and a training log. The worker also snapshots the exact
transport receipt and arm admission beside its self-digested worker receipt;
finalization refuses a receipt without those byte-bound control records.

## 3. Run Splatfacto on the admitted Linux CUDA worker

Install the pinned environment with `scripts/setup_splatfacto_venv.sh g1` and
run the worker entry point from that exact environment (the setup script
installs Blueprint itself with `--no-deps`, so it does not perturb the pinned
Nerfstudio/gsplat resolver):

```bash
python -m blueprint_pipeline.canonical_3dgs_cli admit-worker \
  --transport-receipt /derived/<capture-id>/canonical_3dgs_transport_receipt.json \
  --arm splatfacto-comparison \
  --worker-platform linux \
  --paid-allocator-admission /control/splatfacto-allocator-admission.json \
  --worker-image-digest <image-name>@sha256:<64-hex> \
  --trainer-runtime-digest sha256:913d5afd190a9bed736f6a978d472b58654f650d3bc173a07d8a5375d95703c6 \
  --trainer-runtime-version nerfstudio-1.1.5+gsplat-1.4.0 \
  --authority-id <explicit-authority-id> \
  --max-spend-usd <ceiling> \
  --hard-ttl-seconds <ttl> \
  --provider-upload-authorized \
  --paid-compute-authorized \
  --watchdog-armed \
  --provider-zero-before-allocation \
  --timestamp <UTC-ISO8601> \
  --output /derived/<capture-id>/splatfacto-worker-admission.json
```

Then execute:

```bash
export BLUEPRINT_WORKER_IMAGE_DIGEST='<image-name>@sha256:<64-hex>'

.venvs/splatfacto-g1/bin/python -m blueprint_pipeline.canonical_3dgs_cli transport extract \
  --bundle /work/canonical_3dgs_transport.zip \
  --receipt /work/canonical_3dgs_transport_receipt.json \
  --output-root /work/materialized

.venvs/splatfacto-g1/bin/python -m blueprint_pipeline.canonical_3dgs_cli run-arm \
  --arm splatfacto-comparison \
  --plan /work/materialized/<bundle-digest>/campaign/canonical_3dgs_execution_plan.json \
  --dataset-root /work/materialized/<bundle-digest>/campaign/dataset \
  --output-root /work/results/splatfacto-comparison \
  --receipt /work/results/splatfacto-comparison/worker_receipt.json \
  --transport-receipt /work/canonical_3dgs_transport_receipt.json \
  --admission /work/splatfacto-worker-admission.json
```

The worker refuses package drift from Nerfstudio 1.1.5 / gsplat 1.4.0, freezes
seed 42 and 30,000 iterations, selects the COLMAP parser with no image
downscaling, disables automatic recentering/reorientation/rescaling, uses every
candidate-visible image, and applies the quality-oriented alpha-culling
threshold. It explicitly freezes the 1.1.5 `stop_split_at=15000` refinement
stop; the newer `continue_cull_post_densification` option is not passed because
it does not exist in the pinned 1.1.5 source. This keeps its exported PLY in the
same canonical ARKit world frame as the Postshot `--no-recenter-points` arm.
It then finds exactly one produced `config.yml`, executes
`ns-export gaussian-splat`, and requires exactly one exported PLY. A zero
process exit without those artifacts is a failure.

The Splatfacto admission must use trainer version
`nerfstudio-1.1.5+gsplat-1.4.0` and runtime digest
`sha256:913d5afd190a9bed736f6a978d472b58654f650d3bc173a07d8a5375d95703c6`.
The worker derives that digest from the installed package versions and refuses
any mismatch before `ns-train` starts.

The qualified Vast adapter is `canonical_splatfacto_vast_v1`. Its request must
name that exact adapter and use an immutable Nerfstudio image such as
`dromni/nerfstudio@sha256:adcca86d1804a7db71dbe64648a5173cd3c8da850e20cfc1151e7149a60db6a6`.
It verifies and installs only the wheel embedded in the candidate-only
transport, returns `canonical_3dgs_vast_output_bundle.v1`, and has the
controller independently decode the standard 3DGS PLY. A generic Vast adapter
or bare adapter-qualified boolean cannot authorize this specialized worker.

The worker calculates the remaining admission TTL immediately before the
trainer subprocess and applies it as a local timeout. The independent watchdog
remains mandatory because a local process timeout is not provider teardown.

## 4. Finalize the two receipts

After copying each complete arm directory beneath the same results root:

```bash
python -m blueprint_pipeline.canonical_3dgs_cli finalize \
  --plan /work/canonical_3dgs_execution_plan.json \
  --dataset-root /work/dataset \
  --results-root /work/results
```

Finalization rehashes all original training bytes and every returned artifact,
checks the self-digested worker receipt, revalidates the snapshotted transport
and arm-specific admission, and writes normalized arm results plus
`canonical_3dgs_campaign_result.json`. A manually assembled result directory
without those production controls is rejected. Each standard PLY is parsed
into a digest-bound appearance-fidelity candidate binding with its splat count,
SH degree, full and robust bounds, coordinate-basis digest, and an explicit
`global_decimation_applied=false` declaration.

`candidates_ready_for_independent_evaluation` means both appearance candidates
exist and share the exact input dataset. It does **not** select a quality winner.
The best-quality decision requires both candidates to render the same frozen
hidden cameras through a qualified native 3DGS renderer and then pass
`appearance_fidelity_qualification.v1`. That gate requires SSIM, PSNR, and
LPIPS plus site/task-specific thresholds. The campaign deliberately supplies
no default thresholds and forbids selection before those measurements. No
trainer may see the real held-out pixels or grade itself.

The campaign preserves that provider-zero verification remains required after
both external executions and reports it as not yet verified. Teardown and
provider-zero are produced by the paid-resource control plane after result
collection; they are resource-safety evidence, never reconstruction-quality
evidence.

## 5. Run the evaluator-owned exact-camera comparison

Preparation also writes
`evaluator_input/canonical_3dgs_hidden_evaluator_input.json` plus its bound
reference frames. This directory is never included in the trainer transport.
Freeze a site/task-specific threshold JSON containing exactly:

```json
{
  "minimum_mean_psnr_db": 0.0,
  "minimum_mean_global_ssim": 0.0,
  "minimum_mean_windowed_ssim": 0.0,
  "maximum_mean_absolute_error": 0.0,
  "maximum_mean_lpips": 0.0
}
```

The zeroes above show the required fields only; they are not usable defaults.
The decision owner must supply justified values before any render begins.
Freeze the LPIPS runtime separately:

```json
{
  "model_id": "lpips_alex_v0.1",
  "checkpoint_digest": "sha256:df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0",
  "backbone_digest": "sha256:7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02"
}
```

Then run on the independent evaluator host with the repository's pinned Spark
native-3DGS exact-camera renderer and the `evaluation` Python extra installed:

```bash
python -m blueprint_pipeline.canonical_3dgs_cli evaluate \
  --campaign /work/results/canonical_3dgs_campaign_result.json \
  --results-root /work/results \
  --evaluator-input /derived/<capture-id>/evaluator_input/canonical_3dgs_hidden_evaluator_input.json \
  --evaluator-root /derived/<capture-id>/evaluator_input \
  --thresholds /work/site-task-appearance-thresholds.json \
  --lpips-model /work/lpips-model.json \
  --output-root /work/quality
```

The command renders both PLYs at the same frozen hidden cameras, computes
windowed/global SSIM, PSNR, MAE, and pinned LPIPS, writes one
`appearance_fidelity_qualification.v1` per arm, and selects only among arms
that pass the frozen thresholds. Its deterministic tie order is LPIPS, PSNR,
then windowed SSIM. If neither arm qualifies it abstains; it never upgrades
appearance evidence into metric, collision, Isaac, or physical-task proof.

## 6. Register one evaluated appearance candidate

After independent held-out evaluation selects a qualifying arm, independently
measure appearance-to-site correspondences and freeze the similarity transform
and residual thresholds. Then produce the registered-appearance candidate:

```bash
python -m blueprint_pipeline.canonical_3dgs_cli register \
  --source-admission /derived/<capture-id>/canonical_3dgs_source_admission.json \
  --campaign /work/results/canonical_3dgs_campaign_result.json \
  --results-root /work/results \
  --quality-comparison /work/quality/canonical_3dgs_quality_comparison.json \
  --registration-measurement /work/registration/measurement.json \
  --output /work/registration/canonical_registered_appearance.json
```

The producer independently decodes the selected standard 3DGS PLY, verifies
its campaign lineage, and reports RMSE, p95, and maximum registration residuals
in meters. It is `candidate_only` until both held-out appearance and
registration gates qualify. Even then, its ceiling is registered appearance;
it is not a `registered_site_reconstruction.v1` until the post-capture evidence
spine joins it to independently qualified dynamics geometry. Metric geometry,
collision, Isaac compatibility, and physical success remain false.

## No-authority handoff

When paid execution has not been authorized, compile the immutable request
instead of launching anything:

```bash
python -m blueprint_pipeline.canonical_3dgs_cli request-execution \
  --plan /derived/<capture-id>/canonical_3dgs_execution_plan.json \
  --transport-receipt /derived/<capture-id>/canonical_3dgs_transport_receipt.json \
  --worker-wheel-digest sha256:<64-hex> \
  --worker-wheel-filename blueprint_capture_pipeline-2.0.0-py3-none-any.whl \
  --timestamp <UTC-ISO8601> \
  --output /derived/<capture-id>/canonical_3dgs_execution_request.json
```

The request binds the exact retained bytes, authorizes neither upload nor
spend, sets no winner, and names the missing per-arm authority, budget, TTL,
image/runtime identities, watchdog, provider-zero preflight, credentials, and
allocator adapter/admission evidence.

## End-to-end test boundary

The hermetic fixture starts from a complete V3.2 bundle, validates real
depth/confidence PNGs, exports a depth-seeded COLMAP dataset, executes both
worker contracts with deterministic fake binaries, and finalizes their
receipts. Real visual quality still requires an authorized GPU run on a
representative capture; repository tests cannot honestly substitute for that
observation.
