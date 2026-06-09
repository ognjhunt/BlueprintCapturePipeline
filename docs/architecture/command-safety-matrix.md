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
| `blueprint-build-palatial-physready --capture-root <path>` | Build Palatial PhysReady twin request/materialization manifests | Local artifact writer by default; selects task-critical object twins such as microwaves/totes; no provider calls, uploads, downloads, simulator execution, or proof upgrades unless explicit live/download gates are passed |
| `blueprint-build-episode-specs --capture-root <path>` | Compile `episode_spec.v1` setup manifests | Local artifact writer; proposal agents may write review inputs but cannot set proof booleans |
| `blueprint-run-cpu-simulator-preflight --capture-root <path>` | Generate CPU MuJoCo/PyBullet setup manifests | Local artifact writer by default; optional smoke requires `BLUEPRINT_ALLOW_CPU_SIMULATOR_PREFLIGHT=true` plus `--allow-cpu-simulator-preflight` and remains local CPU preflight only |
| `blueprint-run-simulation-automation --capture-root <path>` | Build fail-closed simulation automation manifests | Local artifact writer; no providers, asset downloads, simulator execution, GPU training, sends, payments, or deploys |
| `blueprint-run-site-eval-director --capture-root <path>` | Build deterministic site-eval director manifests | Local artifact writer; optional SDK flags can become live operators only with SDK, credential, CLI, and env gates |
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
| `python -m blueprint_pipeline.run_e2e --capture-root <path> --provider openai` | Local e2e pipeline entrypoint | Requires a complete staged capture under `scenes/<scene>/captures/<capture>`; can run multiple stages and write staged artifacts |
| `python -m blueprint_pipeline.capture_orchestrator` | Lane orchestrator | Can run package/runtime lanes and write artifacts |
| `PYTHONPATH=src python -m blueprint_pipeline.simready_assets --capture-root <path>` | Legacy local simready asset review lane | Writes `pipeline/simready/*`; does not run Isaac Sim, MuJoCo, PyBullet, providers, or model downloads |
| `PYTHONPATH=src python -m blueprint_pipeline.marble_sim_assets --capture-root <path>` | Legacy local Marble sim-asset handoff lane | Writes `pipeline/marble_sim_assets/*`; does not call World Labs, download remote assets, run Isaac Sim, MuJoCo, or PyBullet |
| `blueprint-build-robot-eval-dataset --capture-root <path>` | Local robot eval dataset contract lane | Writes `pipeline/robot_eval_dataset/*`; does not prove robot readiness, simulator execution, or actual outcomes |
| `blueprint-build-scene-asset-preflight --capture-root <path>` | Local CPU scene asset inspection | Writes inventory, dependency audit, scene preflight, collider/proxy, frame, and scorecard manifests; labels PLY/USD/GLB/GLTF/OBJ/URDF/MJCF evidence without collision/contact proof |
| `blueprint-build-palatial-physready --capture-root <path>` | Optional Palatial PhysReady twin request lane | Writes `pipeline/palatial_physready/*`; disabled-by-default provider lane that creates request/cost/lineage manifests and can materialize local provider-response exports without proving simulator execution, contact, safety, or robot readiness |
| `blueprint-build-episode-specs --capture-root <path>` | Local episode-spec compiler | Writes `task_anchor_proposal_manifest.json`, `episode_spec.v1.json`, `episode_specs.json`, `episode_spec_manifest.json`, and advisory agent proposals; specs remain review-required unless accepted anchors, scale, profile, and collision proof exist |
| `blueprint-run-cpu-simulator-preflight --capture-root <path>` | Local CPU MuJoCo/PyBullet setup lane | Writes spawn validation, CPU preflight, pre-GPU readiness, CPU fixture files, and `cpu_simulator_preflight_manifest.json`; optional CPU smoke is gated and never proves robot readiness, policy success, safety, contact, or owner-system simulator execution |
| `blueprint-run-simulation-automation --capture-root <path>` | Local simulation automation orchestration lane | Writes `pipeline/simulation_automation/*`, including `gpu_handoff_packet.json`, proof schema, checklist, and owner-GPU blocked manifest; simulator/training execution remains blocked unless explicit env and CLI gates are provided |
| `blueprint-run-site-eval-director --capture-root <path>` | Local site-eval director lane | Writes scenario/task/matrix/fixture-attempt/label/calibration/breakage/review/proof manifests under `pipeline/simulation_automation/*`; fixture attempts are local-only, and real engines/training stay blocked without explicit env and CLI gates |
| `blueprint-run-robot-eval-job --capture-root <path> --job-request <json> --job-id <id> ...` | Headless robot-eval job orchestration lane | Writes request/validation/plan/provisioning/simulator/training/evaluation/proof/job manifests under `pipeline/robot_eval_jobs/<job_id>/*`; fixture paths are local-only, and real provisioning/simulator/training/agent paths stay blocked without explicit env and CLI gates |
| `blueprint-run-robot-eval-job --capture-root <path> --job-request-inbox <dir> ...` | WebApp robot-eval job request inbox lane | Copies each `robot_eval_job_request.v1` to `pipeline/robot_eval_job_requests/<job_id>/job_request.json`, runs the same fail-closed job orchestrator, and writes `pipeline/robot_eval_job_requests/inbox_run_manifest.json`; fixture paths remain local-only |
| `blueprint-ingest-arena-results --capture-root <path> --arena-results-dir <dir> --scenario-count 500` | Local Arena result ingest and Post-Training Data Package support lane | Reads existing local Isaac Lab-Arena rollout artifacts and writes schedule/shard/retry/cost/resume, normalized trace, labels, clips, review, report, delivery, package, archive, and operator ledgers; does not run Arena, upload storage, call vision models, or execute live SDK operators without separate gates |
| `blueprint-audit-arena-package --capture-root <path> --package-dir <dir> --expected-scenario-count 500` | Local Arena package artifact and proof-boundary audit | Checks required package artifacts, counts, checksums/archive surfaces, delivery/operator ledgers, and forbidden proof booleans; writes `arena_package_proof_boundary_audit.json` and does not call providers or run simulators |
| `blueprint-smoke-arena-package-local --output-dir <dir>` | One-command local Arena package smoke | Creates synthetic local capture/results fixtures, runs the real ingest CLI path for a 500-scenario schedule, exercises review-required vision labels, local delivery, fake local operators, and the package audit; writes `arena_fixture_smoke_manifest.json`; does not prove WebApp upstream truth, owner-system Arena execution, robot policy execution, contact, safety, or readiness |
| `blueprint-audit-live-pipeline-setup --capture-root <path> --package-dir <dir> --arena-results-dir <dir> --digitalocean-droplet-name paperclip-prod-01 --digitalocean-droplet-ip 206.81.11.69` | Live setup/gate preflight | Loads local env files without printing secret values; checks gates, command hooks, owner-supplied Arena result directories, SDK modules, WebApp upstream IDs, package audit status, and optional control-plane host; does not run provider jobs, mutate DigitalOcean, upload storage, or claim GPU/Arena proof |
| `blueprint-audit-live-robot-eval-closure --capture-root <path> --job-dir <dir>` | Live robot-eval closure audit | Reads job artifacts and owner-supplied closure evidence, verifies capture/task/scenario/POV/eval/policy/simulator/deployment/delivery/safety gates, writes `live_eval_closure_manifest.json`, and only reports readiness when all required live proof gates are accepted; it does not call providers, run simulators, upload storage, or execute SDK operators |
| `blueprint-intake-live-pipeline-inputs --manifest-path <control-plane-manifest> --webapp-job-request <json> --arena-results-dir <dir> --policy-package <json> --deployment-outcomes <json> --live-closure-evidence <json> [--stage-webapp-request] [--stage-arena-results] [--stage-policy-package] [--stage-deployment-outcomes] [--stage-live-closure-evidence]` | Live external-input intake validator | Validates candidate WebApp request/envelope IDs against the configured capture root, inspects owner Arena result directories for JSON ingest artifacts, validates job-specific policy packages, deployment outcomes, and closure evidence, optionally copies a validated WebApp request into the configured inbox, optionally stages Arena pointers, and optionally copies policy/deployment/closure evidence under `pipeline/robot_eval_inputs/<job_id>/`; task/scenario outcome records can stage as real-world validation inputs, exact `scenario_eval_run_id` or `scenario_variation_instance_id` keys are required before predicted-vs-actual calibration is ready, and proof remains blocked on `real_world_deployment_outcome_owner_evidence` until owner evidence exists; does not process the job, run Arena, execute policies, mutate env files, upload storage, or upgrade proof claims |
| `blueprint-live-pipeline-intake-service --host 127.0.0.1 --port 8765` | Authenticated WebApp-to-Pipeline intake service | Requires `BLUEPRINT_LIVE_PIPELINE_INTAKE_TOKEN`; accepts direct WebApp job requests, queue envelopes, robot-team policy packages, deployment outcome JSON, and `live_robot_eval_closure_evidence.v1` bodies, stages validated requests/evidence through the same intake validator, and can optionally run a configured trigger command; does not run Arena, execute policies, call providers, publish claims, or set proof booleans |
| `blueprint-run-live-pipeline-control-plane --capture-root <path> --job-request-inbox <dir> --arena-results-dir <dir>` | Timer-safe live pipeline control-plane pass | Writes the control-plane manifest plus `live_pipeline_external_input_packet.json` and `.md`, drains queued WebApp job requests when configured, records exact missing WebApp IDs, owner Arena evidence, command hooks, and SDK/operator gates, and accepts WebApp inbox truth only when the queued request matches the configured capture root; exits 0 when blocked so timers do not restart-loop; does not run live simulator/provider/upload/operator paths unless separately gated |
| `blueprint-audit-live-pipeline-proof-boundary --manifest-path <control-plane-manifest>` | Live control-plane proof-boundary audit | Reads the control-plane manifest, external input packet, setup manifest, and staged-input pointers; writes `live_pipeline_proof_boundary_audit.json`; fails on internal artifact inconsistency, malformed staged pointers, secret leakage, or forbidden proof upgrades, but exits zero for a healthy waiting state with external blockers unless `--require-live-ready` is set |
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
| `BLUEPRINT_ALLOW_LIVE_AGENTS_SDK_OPERATORS=true blueprint-run-simulation-automation --agent-mode agents-sdk --allow-live-agent-operator ...` | Optional live Agents SDK simulation operator | Can execute a gated agent to inspect manifests/logs, choose deterministic reruns, summarize blockers, route review, and log proof effect; cannot set proof booleans directly |
| `BLUEPRINT_ALLOW_LIVE_CODEX_SDK_OPERATORS=true blueprint-run-simulation-automation --agent-mode codex-sdk --allow-live-agent-operator ...` | Optional live Codex SDK simulation code-maintainer operator | Can execute Codex SDK to diagnose failures, patch code, run focused tests, and summarize diffs; cannot set proof booleans directly |
| `BLUEPRINT_PREVIEW_PROVIDER=world_labs ...` | World Labs preview path | Can submit live provider job when key/env are present |
| `blueprint-materialize-worldlabs-assets --capture-root <path>` | Download already-generated World Labs/Marble output assets into local checksum-backed files | Downloads remote CDN assets but does not start a new generation, run simulators, or prove robot readiness |
| `BLUEPRINT_ENABLE_PALATIAL_PHYSREADY=true PALATIAL_API_KEY=<secret> blueprint-build-palatial-physready --capture-root <path> --allow-live-palatial` | Optional live Palatial PhysReady generation lane | Can upload selected object reference images/prompts and submit Palatial generation requests; still writes model-derived support artifacts only and cannot prove simulator execution, contact, safety, or robot readiness |
| `blueprint-build-palatial-physready --capture-root <path> --provider-response <json> --download-exports` | Optional Palatial export materialization lane | Can download already-generated Palatial export URLs into checksum-backed local files; does not start a new generation unless the live Palatial gate is also passed and does not prove simulator/contact behavior |
| `BLUEPRINT_ALLOW_LIVE_AGENTS_SDK_OPERATORS=true blueprint-run-site-eval-director --agents-sdk-site-eval --allow-live-agents-sdk-operator ...` | Optional live Agents SDK site-eval operator | Can execute a gated agent to inspect manifests/logs, choose deterministic reruns, route review, summarize blockers, and log proof effect; cannot set proof booleans directly |
| `BLUEPRINT_ALLOW_LIVE_CODEX_SDK_OPERATORS=true blueprint-run-site-eval-director --codex-sdk-code-maintainer --allow-live-codex-sdk-operator ...` | Optional live Codex SDK code-maintainer operator | Can execute Codex SDK to diagnose failures, patch code, run tests, and summarize diffs; cannot set proof booleans directly |
| `BLUEPRINT_ALLOW_LIVE_AGENTS_SDK_OPERATORS=true blueprint-run-robot-eval-job --agent-mode agents-sdk --allow-live-agent-operator ...` | Optional live robot-eval job Agents SDK operator | Can execute a gated agent to choose deterministic commands/reruns, request gated provisioning, summarize blockers, route review, and log proof effect; cannot override rights/privacy or proof gates |
| `BLUEPRINT_ALLOW_FAKE_LIVE_OPERATORS=true blueprint-ingest-arena-results --operator-mode fake ...` | Local fake Arena operator lane | Runs local fake Agents/Codex operator decisions without spend or network; proves only operator ledger and routing code paths |
| `BLUEPRINT_ALLOW_LIVE_AGENTS_SDK_OPERATORS=true BLUEPRINT_ALLOW_LIVE_CODEX_SDK_OPERATORS=true blueprint-ingest-arena-results --operator-mode agents-sdk --allow-live-agents-sdk --allow-live-codex-sdk ...` | Live Arena Agents SDK + Codex SDK operator lane | Can call live SDK operators when dependencies/auth exist; may inspect manifests, choose deterministic commands, route review, and diagnose code-maintenance issues, but cannot set proof booleans true without deterministic accepted artifacts |
| `BLUEPRINT_ALLOW_LIVE_CODEX_SDK_OPERATORS=true BLUEPRINT_ALLOW_CODEX_CLI_HOST_OAUTH=true ... --allow-live-codex-sdk...` | Optional Codex CLI host-OAuth transport | Can use an installed authenticated `codex` CLI instead of the Python Codex SDK where wired; auth remains in the host/user Codex profile, and the operator still cannot mutate proof booleans directly |
| `blueprint-run-live-pipeline-control-plane` | Timer-safe live control-plane pass | Loads env files, audits setup, and optionally consumes a robot-eval job inbox; exits 0 when blocked and does not prove simulator/GPU/robot readiness |
| `BLUEPRINT_ALLOW_GPU_PROVISIONING=true blueprint-run-robot-eval-job --allow-gpu-provisioning --provisioner <vast|runpod|gcp|local_process|docker_local> ...` | Explicit non-fixture provisioning request lane | Can prepare gated provisioning request/result manifests; fixture local remains the only default local proof path |
| `BLUEPRINT_ALLOW_ROLLOUT_VISION_LABELING=true blueprint-ingest-arena-results --allow-rollout-vision-labeling --vision-labeling-command <command> ...` | Explicit rollout vision labeling lane | Can run a configured local/HTTP wrapper command; fallback labels remain review-required and label output cannot prove contact/safety/robot readiness by itself |
| `BLUEPRINT_ALLOW_ROLLOUT_VISION_LABELING=true blueprint-label-rollout-vision-openai --output-dir <arena-package-dir>` | OpenAI rollout vision-label command hook | Calls OpenAI Responses with extracted keyframes and writes `rollout_vision_labels.command.json`; labels are forced review-required and cannot prove contact/safety/robot readiness |
| `BLUEPRINT_ALLOW_PACKAGE_DELIVERY_UPLOAD=true blueprint-ingest-arena-results --allow-delivery-upload --delivery-command <command> ...` | Explicit package delivery upload lane | Can run a configured upload/signed-access command and log signed-access status; local bundle is default and entitlement still requires owner-system review |
| `BLUEPRINT_ALLOW_PACKAGE_DELIVERY_UPLOAD=true blueprint-deliver-arena-package-local --output-dir <arena-package-dir>` | Local filesystem delivery hook | Copies `delivery_bundle/` to `BLUEPRINT_LOCAL_DELIVERY_ROOT` and writes `delivery_upload.command.json`; does not create signed URLs, upload cloud storage, verify entitlement, or upgrade proof |
| `BLUEPRINT_ALLOW_SIMULATOR_EXECUTION=true blueprint-run-robot-eval-job --allow-simulator-execution --allow-simulator <framework> --simulator <framework> --simulator-command <framework>=<command> ...` | Explicit job-level simulator command lane | Can run a local MuJoCo, PyBullet, Newton, Isaac Sim, or Isaac Lab-Arena command and record stdout/stderr/exit code; public/readiness proof remains false without owner-system evidence |
| `BLUEPRINT_ALLOW_COSMOS_TRAINING=true blueprint-run-robot-eval-job --allow-training --training-command <command> ...` | Explicit job-level training command lane | Can run a gated training command; training proof requires completed command plus checkpoint/run manifest evidence |
| `BLUEPRINT_ALLOW_SIMULATOR_EXECUTION=true blueprint-run-simulation-automation --allow-simulator-execution --allow-simulator <framework> --simulator-command <framework>=<command> ...` | Explicit simulator execution request/result lane | Can run Isaac Sim, Isaac Lab-Arena, MuJoCo, PyBullet, or Newton command; capture stdout/stderr/exit code and inspect blocked/result manifests |
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
