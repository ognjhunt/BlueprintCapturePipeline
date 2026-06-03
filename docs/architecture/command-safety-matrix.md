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
| `blueprint-build-simready-assets --capture-root <path>` | Build local simulator-review artifacts | Local artifact writer; no simulator/provider execution |
| `blueprint-build-marble-sim-assets --capture-root <path>` | Build local Marble simulator-review handoff artifacts | Local artifact writer; no World Labs call, asset download, or simulator execution |
| `blueprint-build-robot-eval-dataset --capture-root <path>` | Build local real-site robot eval dataset artifacts | Local artifact writer; no providers, simulators, model downloads, sends, payments, or deploys |

Use `PYTHONDONTWRITEBYTECODE=1` for verification commands when the goal is to
avoid `__pycache__` churn.

## Setup And Runtime Env Checks

| Command | Use | Risk |
| --- | --- | --- |
| `python3 scripts/setup_environment.py --check` | Reports local Python/ML/runtime gaps | Read-only-ish local environment probe |
| `python3 -m pip install -e .[dev]` | Installs repo in editable mode | Mutates local Python environment |
| `uv sync --extra dev` | Syncs development dependencies | Mutates local virtual environment |
| `./scripts/install_ml_stack.sh` | Installs ML/runtime stack | Heavy local mutation, may download large packages |

`setup_environment.py --check` is safe to run for blocker discovery. Treat
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
| `blueprint-build-simready-assets --capture-root <path>` | Local simready asset review lane | Writes `pipeline/simready/*`; does not run Isaac Sim, MuJoCo, PyBullet, providers, or model downloads |
| `blueprint-build-marble-sim-assets --capture-root <path>` | Local Marble sim-asset handoff lane | Writes `pipeline/marble_sim_assets/*`; does not call World Labs, download remote assets, run Isaac Sim, MuJoCo, or PyBullet |
| `blueprint-build-robot-eval-dataset --capture-root <path>` | Local robot eval dataset contract lane | Writes `pipeline/robot_eval_dataset/*`; does not prove robot readiness, simulator execution, or actual outcomes |

Run these only when broad gate refresh is requested or when docs/code changes
touch launch contracts enough to justify it. Always inspect worktree and output
drift afterward.

## GPU And Privacy Runner Commands

| Command | Use | Risk |
| --- | --- | --- |
| `python3 scripts/run_geometry_lane.py --capture-root <path> --provider video_to_world --model video_to_world-default` | Runs geometry lane against a capture | Calls configured runner when env is present; writes `pipeline/geometry/*` |
| `blueprint-video-to-world-runner` | Starts runner service entrypoint | Service runtime, may use GPU/model paths |
| `python3 scripts/run_site_world_runtime_local.py` | Starts/validates local runtime | Local runtime side effects |
| `python3 scripts/start_native_runtime_vast.sh` | Starts native runtime on Vast-style host | Live/runtime risk |

Do not run GPU/privacy runner deployment or live inference without approval.
Fallback geometry is not live proof.

## Provider And API Commands

| Command | Use | Risk |
| --- | --- | --- |
| `blueprint-agent-review --capture-root <path> --provider openai` | Optional LLM-backed review | Calls external provider when configured |
| `blueprint-run-e2e --capture-root <path> --provider openai` | Optional e2e wrapper with provider review | Calls external provider when configured |
| `BLUEPRINT_PREVIEW_PROVIDER=world_labs ...` | World Labs preview path | Can submit live provider job when key/env are present |

Do not run live provider jobs without explicit approval. A present provider key
does not mean the user approved spend or external API mutation.

## Deploy Scripts And Live-Risk Commands

| Command | Use | Risk |
| --- | --- | --- |
| `deploy/scripts/deploy.sh` | Deploys service infrastructure | Live deployment risk |
| `terraform apply` under `deploy/terraform/` | Creates/updates cloud resources | Live cloud mutation |
| `gcloud run deploy ...` | Deploys Cloud Run services | Live deployment risk |
| `git push origin main` | Publishes repo state | Remote mutation |

Do not run deploy, Terraform, cloud, Stripe, WebApp mutation, or push commands
unless the user explicitly requests that action.
