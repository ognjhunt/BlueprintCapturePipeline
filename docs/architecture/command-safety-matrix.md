# Command Safety Matrix

This matrix classifies common commands by local risk. When in doubt, prefer
read-only checks and tests that use fixtures. Do not run live provider jobs,
GPU deployments, Stripe, WebApp mutation, or external sync without explicit user
approval.

## Safe Local Checks

| Command | Use | Risk |
| --- | --- | --- |
| `git status --short --branch` | Worktree/branch orientation | Read-only |
| `git diff --stat` | Diff orientation | Read-only |
| `rg <pattern>` | Fast code/doc search | Read-only |
| `pytest tests/test_site_world_packaging.py tests/test_alpha_readiness.py tests/test_webapp_sync.py -q` | Focused package/readiness/sync tests | Local fixture tests |
| `pytest tests/test_geometry_stage.py tests/test_retrieval_index_geometry_source.py -q` | Geometry and retrieval fail-closed checks | Local fixture tests |
| `pytest tests/test_launch_bundle.py tests/test_qualification_alpha.py -q` | Provider preview and qualification contract checks | Local fixture tests |
| `ruff check <touched-files>` | Lint touched Python files | Read-only analysis |
| `blueprint-capture-pipeline --lane current --descriptor-gcs-uri <gs://.../capture_descriptor.json>` | Run the active capture package flow | Expands to qualification, evaluation prep, and simulation automation; World Labs call only occurs if preview output is requested and input is ready; no simulator/GPU proof upgrades |
| `blueprint-build-robot-eval-dataset --capture-root <path>` | Build local real-site robot eval dataset artifacts | Local artifact writer; no providers, simulators, model downloads, sends, payments, or deploys |
| `blueprint-build-scene-asset-preflight --capture-root <path>` | Inspect local PLY/USD scene assets for CPU preflight | Local artifact writer; no provider calls, downloads, simulator execution, or proof upgrades |
| `blueprint-build-episode-specs --capture-root <path>` | Compile `episode_spec.v1` setup manifests | Local artifact writer; agents are advisory only and cannot set proof booleans |
| `blueprint-run-cpu-simulator-preflight --capture-root <path>` | Generate CPU MuJoCo/PyBullet setup manifests | Local artifact writer by default; optional smoke requires `BLUEPRINT_ALLOW_CPU_SIMULATOR_PREFLIGHT=true` plus `--allow-cpu-simulator-preflight` and remains local CPU preflight only |
| `blueprint-run-simulation-automation --capture-root <path>` | Build fail-closed simulation automation manifests | Local artifact writer; no providers, asset downloads, simulator execution, GPU training, sends, payments, or deploys |
| `blueprint-run-site-eval-director --capture-root <path>` | Build deterministic site-eval director manifests | Local artifact writer; optional SDK flags only write advisory request/blocked manifests |
| `blueprint-run-robot-eval-job --capture-root <path> --job-request <json> --job-id <id> --provisioner fixture_local --simulator fixture` | Run fixture-backed robot-eval job orchestration | Local artifact writer under `pipeline/robot_eval_jobs/<job_id>`; fixture proof only, no live provider, real simulator, GPU training, sends, payments, or deploys |

Use `PYTHONDONTWRITEBYTECODE=1` for verification commands when the goal is to
avoid `__pycache__` churn.

## Legacy Setup And Runtime Env Checks

| Command | Use | Risk |
| --- | --- | --- |
| `python3 scripts/setup_environment.py --check` | Reports legacy Python/ML/runtime gaps | Read-only-ish legacy environment probe; not current pipeline readiness proof |
| `python3 -m pip install -e .[dev]` | Installs repo in editable mode | Mutates local Python environment |
| `uv sync --extra dev` | Syncs development dependencies | Mutates local virtual environment |
| `./scripts/install_ml_stack.sh` | Installs legacy ML/runtime stack | Heavy local mutation, may download large packages |

`setup_environment.py --check` is safe to run for legacy blocker discovery. Treat
missing ML stack, ffmpeg, CUDA, Android SDK, model packages, or provider keys as
blockers when the requested proof depends on them.

## Paid Marketplace Gate

| Command | Use | Risk |
| --- | --- | --- |
| `PYTHONDONTWRITEBYTECODE=1 python3 scripts/run_paid_marketplace_launch_gate.py` | Cross-repo automated paid beta gate | Writes `output/paid_marketplace_launch_gate.md` and `.json` |

This gate is contract-first. It separates automated proof from operator,
toolchain, manual, and live evidence. It does not prove live Stripe payments,
live capturer payouts, real-device capture flows, or authenticated buyer access.

After running it, check:

```bash
git status --short --branch
git diff --stat
git diff -- output/paid_marketplace_launch_gate.md output/paid_marketplace_launch_gate.json
```

## External Alpha Gate And Cross-Repo Side Effects

| Command | Use | Risk |
| --- | --- | --- |
| `PYTHONDONTWRITEBYTECODE=1 python3 scripts/run_external_alpha_launch_gate.py` | Cross-repo alpha launch gate | May inspect adjacent repos and write gate artifacts |
| `python -m blueprint_pipeline.run_e2e` | Local e2e pipeline entrypoint | Can run multiple stages and write staged artifacts |
| `python -m blueprint_pipeline.capture_orchestrator` | Lane orchestrator | Can run package/runtime lanes and write artifacts |
| `PYTHONPATH=src python -m blueprint_pipeline.simready_assets --capture-root <path>` | Legacy local simready asset review lane | Writes `pipeline/simready/*`; does not run Isaac Sim, MuJoCo, PyBullet, providers, or model downloads |
| `PYTHONPATH=src python -m blueprint_pipeline.marble_sim_assets --capture-root <path>` | Legacy local Marble sim-asset handoff lane | Writes `pipeline/marble_sim_assets/*`; does not call World Labs, download remote assets, run Isaac Sim, MuJoCo, or PyBullet |
| `blueprint-build-robot-eval-dataset --capture-root <path>` | Local robot eval dataset contract lane | Writes `pipeline/robot_eval_dataset/*`; does not prove robot readiness, simulator execution, or actual outcomes |
| `blueprint-build-scene-asset-preflight --capture-root <path>` | Local CPU scene asset inspection | Writes inventory, dependency audit, scene preflight, collider/proxy, frame, and scorecard manifests; labels PLY/USD/GLB/GLTF/OBJ/URDF/MJCF evidence without collision/contact proof |
| `blueprint-build-episode-specs --capture-root <path>` | Local episode-spec compiler | Writes `task_anchor_proposal_manifest.json`, `episode_spec.v1.json`, `episode_specs.json`, `episode_spec_manifest.json`, and advisory agent proposals; specs remain review-required unless accepted anchors, scale, profile, and collision proof exist |
| `blueprint-run-cpu-simulator-preflight --capture-root <path>` | Local CPU MuJoCo/PyBullet setup lane | Writes spawn validation, CPU preflight, pre-GPU readiness, CPU fixture files, and `cpu_simulator_preflight_manifest.json`; optional CPU smoke is gated and never proves robot readiness, policy success, safety, contact, or owner-system simulator execution |
| `blueprint-run-simulation-automation --capture-root <path>` | Local simulation automation orchestration lane | Writes `pipeline/simulation_automation/*`, including `gpu_handoff_packet.json`, proof schema, checklist, and owner-GPU blocked manifest; simulator/training execution remains blocked unless explicit env and CLI gates are provided |
| `blueprint-run-site-eval-director --capture-root <path>` | Local site-eval director lane | Writes scenario/task/matrix/fixture-attempt/label/calibration/breakage/review/proof manifests under `pipeline/simulation_automation/*`; fixture attempts are local-only, and real engines/training stay blocked without explicit env and CLI gates |
| `blueprint-run-robot-eval-job --capture-root <path> --job-request <json> --job-id <id> ...` | Headless robot-eval job orchestration lane | Writes request/validation/plan/provisioning/simulator/training/evaluation/proof/job manifests under `pipeline/robot_eval_jobs/<job_id>/*`; fixture paths are local-only, and real provisioning/simulator/training/agent paths stay blocked without explicit env and CLI gates |
| `blueprint-run-robot-eval-job --capture-root <path> --job-request-inbox <dir> ...` | WebApp robot-eval job request inbox lane | Copies each `robot_eval_job_request.v1` to `pipeline/robot_eval_job_requests/<job_id>/job_request.json`, runs the same fail-closed job orchestrator, and writes `pipeline/robot_eval_job_requests/inbox_run_manifest.json`; fixture paths remain local-only |
| `blueprint-validate-provider-preview-packet --capture-root <path> --mode production --require-webapp-sync` | Validate privacy-safe World Labs input lineage before provider submission | Local artifact validator; writes `pipeline/provider_preview_qa_manifest.json`; exits nonzero when production raw bypass, missing privacy proof, checksum mismatch, missing/placeholder WebApp upstream ids, or adapter/canonical mismatch is present; no provider calls |
| `blueprint-build-production-handoff-readiness --capture-root <path> --mode production` | Summarize whether the repo-local World Labs/materialization/CPU/GPU handoff is ready except owner GPU | Local artifact validator; writes `pipeline/production_handoff_readiness_manifest.json`; production mode requires WebApp upstream-link truth; no provider calls, downloads, simulators, training, sends, payments, or deploys |

Run these only when broad gate refresh is requested or when docs/code changes
touch launch contracts enough to justify it. Always inspect worktree and output
drift afterward.

## GPU And Privacy Runner Commands

| Command | Use | Risk |
| --- | --- | --- |
| `python3 scripts/run_geometry_lane.py --capture-root <path> --provider video_to_world --model video_to_world-default` | Runs geometry lane against a capture | Calls configured runner when env is present; writes `pipeline/geometry/*` |
| `python3 scripts/start_native_runtime_vast.sh` | Starts native runtime on Vast-style host | Live/runtime risk |

Do not run GPU/privacy runner deployment or live inference without approval.
Fallback geometry is not live proof.

## Provider And API Commands

| Command | Use | Risk |
| --- | --- | --- |
| `blueprint-agent-review --capture-root <path> --provider openai` | Optional LLM-backed review | Calls external provider when configured |
| `blueprint-run-e2e --capture-root <path> --provider openai` | Optional e2e wrapper with provider review | Calls external provider when configured |
| `BLUEPRINT_PREVIEW_PROVIDER=world_labs ...` | World Labs preview path | Can submit live provider job when key/env are present |
| `blueprint-materialize-worldlabs-assets --capture-root <path>` | Download already-generated World Labs/Marble output assets into local checksum-backed files | Downloads remote CDN assets but does not start a new generation, run simulators, or prove robot readiness |
| `blueprint-run-site-eval-director --agents-sdk-site-eval --codex-sdk-code-maintainer ...` | Optional site-eval/Codex advisory request manifest lane | Writes request or blocked manifests; does not execute SDK agents or Codex MCP by itself |
| `blueprint-run-robot-eval-job --agent-mode agents-sdk ...` | Optional robot-eval job Agents SDK advisory request manifest lane | Writes advisory request or blocked manifests; does not execute a live agent unless `OPENAI_API_KEY`, `BLUEPRINT_ALLOW_AGENTS_SDK_JOB_ORCHESTRATION=true`, and `--agent-mode agents-sdk` are intentionally provided |
| `BLUEPRINT_ALLOW_GPU_PROVISIONING=true blueprint-run-robot-eval-job --allow-gpu-provisioning --provisioner <vast|runpod|gcp|local_process|docker_local> ...` | Explicit non-fixture provisioning request lane | Can prepare gated provisioning request/result manifests; fixture local remains the only default local proof path |
| `BLUEPRINT_ALLOW_SIMULATOR_EXECUTION=true blueprint-run-robot-eval-job --allow-simulator-execution --allow-simulator <framework> --simulator <framework> --simulator-command <framework>=<command> ...` | Explicit job-level simulator command lane | Can run a local MuJoCo, PyBullet, Newton, or Isaac Sim command and record stdout/stderr/exit code; public/readiness proof remains false without owner-system evidence |
| `BLUEPRINT_ALLOW_COSMOS_TRAINING=true blueprint-run-robot-eval-job --allow-training --training-command <command> ...` | Explicit job-level training command lane | Can run a gated training command; training proof requires completed command plus checkpoint/run manifest evidence |
| `BLUEPRINT_ALLOW_SIMULATOR_EXECUTION=true blueprint-run-simulation-automation --allow-simulator-execution --allow-simulator <framework> --simulator-command <framework>=<command> ...` | Explicit simulator execution request/result lane | Can run Isaac Sim, MuJoCo, PyBullet, or Newton command; capture stdout/stderr/exit code and inspect blocked/result manifests |
| `BLUEPRINT_ALLOW_COSMOS_TRAINING=true blueprint-run-simulation-automation --allow-training ...` | Explicit Cosmos training orchestration lane | Can call the Cosmos LoRA training runner and GPU training command when configured |

Do not run live provider jobs without explicit approval. A present provider key
does not mean the user approved spend or external API mutation.
Do not run simulator or training approvals unless the user explicitly requests
that proof-producing run and the dependency/cost/GPU boundary is understood.

## Deploy Scripts And Live-Risk Commands

| Command | Use | Risk |
| --- | --- | --- |
| `deploy/scripts/deploy.sh` | Deploys service infrastructure | Live deployment risk |
| `terraform apply` under `deploy/terraform/` | Creates/updates cloud resources | Live cloud mutation |
| `gcloud run deploy ...` | Deploys Cloud Run services | Live deployment risk |
| `git push origin main` | Publishes repo state | Remote mutation |

Do not run deploy, Terraform, cloud, Stripe, WebApp mutation, or push commands
unless the user explicitly requests that action.
