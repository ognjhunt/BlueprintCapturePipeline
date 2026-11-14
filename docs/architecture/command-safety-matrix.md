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
| `pytest tests/test_site_world_packaging.py tests/test_webapp_sync.py -q` | Focused package/readiness/sync tests | Local fixture tests |
| `pytest tests/test_geometry_stage.py tests/test_retrieval_index_geometry_source.py -q` | Geometry and retrieval fail-closed checks | Local fixture tests |
| `pytest tests/test_launch_bundle.py tests/test_qualification_alpha.py -q` | Provider preview and qualification contract checks | Local fixture tests |
| `ruff check <touched-files>` | Lint touched Python files | Read-only analysis |
| `python -m blueprint_pipeline.canonical_3dgs_cli prepare --capture-root <v3.2-bundle> --output-root <derived> ...` | Validate a canonical V3.2 bundle and materialize one depth-seeded candidate-only COLMAP dataset plus the two-arm plan | Local deterministic artifact writer; no provider, paid compute, trainer execution, hidden-view access, or quality/metric/collision/Isaac claim |
| `python -m blueprint_pipeline.canonical_3dgs_cli run-arm --arm <postshot-primary|splatfacto-comparison> ...` | Execute one precommitted trainer arm on an already admitted platform worker | Executes local trainer binaries and may consume an existing license/GPU; does not allocate infrastructure or infer paid authority; rehashes the plan inputs first |
| `python -m blueprint_pipeline.canonical_3dgs_cli finalize --plan <json> --dataset-root <dir> --results-root <dir>` | Normalize the two platform-worker receipts into one campaign | Local artifact verifier/writer; rehashes all inputs/outputs and does not select a quality winner |
| `blueprint-capture-pipeline --lane current --descriptor-gcs-uri <gs://.../capture_descriptor.json>` | Run the active capture package flow | Expands to qualification, evaluation prep, and simulation automation; World Labs call only occurs if preview output is requested and input is ready; no simulator/GPU proof upgrades |

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
| `python -m blueprint_pipeline.run_e2e --capture-root <path> --provider local` | Local e2e pipeline entrypoint | Requires a complete staged capture under `scenes/<scene>/captures/<capture>`; `--provider local` is deterministic and makes no LLM/provider calls, while `openai`/`claude` are optional configured review providers; runs evaluation prep/WebApp sync handoff by default to match the autonomous listener path; `--skip-evaluation-prep` is a developer-only narrow-run escape hatch |
| `python -m blueprint_pipeline.capture_orchestrator` | Lane orchestrator | Can run package/runtime lanes and write artifacts |
| `blueprint-ingest-arena-results --capture-root <path> --arena-results-dir <dir> --scenario-count 500` | Local Arena result ingest and optional Task Evaluation evidence-export lane | Reads existing local Isaac Lab-Arena rollout artifacts and writes schedule/shard/retry/cost/resume, normalized trace, labels, clips, review, report, delivery, package, archive, and operator ledgers; the legacy export is emitted only after explicit request and rights/provenance gates, and the command does not run Arena, upload storage, call vision models, or execute live SDK operators without separate gates |
| `blueprint-audit-arena-package --capture-root <path> --package-dir <dir> --expected-scenario-count 500` | Local Arena package artifact and proof-boundary audit | Checks required package artifacts, counts, checksums/archive surfaces, delivery/operator ledgers, and forbidden proof booleans; writes `arena_package_proof_boundary_audit.json` and does not call providers or run simulators |
| `blueprint-smoke-arena-package-local --output-dir <dir>` | One-command local Arena package smoke | Creates synthetic local capture/results fixtures, runs the real ingest CLI path for a 500-scenario schedule, exercises review-required vision labels, local delivery, fake local operators, and the package audit; writes `arena_fixture_smoke_manifest.json`; does not prove WebApp upstream truth, owner-system Arena execution, robot policy execution, contact, safety, or readiness |
| `blueprint-audit-live-pipeline-setup --capture-root <path> --job-request-inbox <dir> --package-dir <dir> --arena-results-dir <dir> --digitalocean-droplet-name paperclip-prod-01 --digitalocean-droplet-ip 206.81.11.69` | Live setup/gate preflight | Loads local env files without printing secret values; checks gates, command hooks, owner-supplied Arena result directories, SDK modules, single-capture WebApp upstream IDs or per-request inbox readiness, package audit status, and optional control-plane host; does not run provider jobs, mutate DigitalOcean, upload storage, or claim GPU/Arena proof |
| `blueprint-intake-live-pipeline-inputs --manifest-path <control-plane-manifest> --webapp-job-request <json> --arena-results-dir <dir> --policy-package <json> --live-closure-evidence <json> [--real-robot-pov <json>] [--deployment-outcomes <json>] [--stage-webapp-request] [--stage-arena-results] [--stage-policy-package] [--stage-real-robot-pov] [--stage-deployment-outcomes] [--stage-live-closure-evidence]` | Live external-input intake validator | Validates candidate WebApp request/envelope IDs and `site_package.capture_root` for either the configured single-capture root or explicit per-request capture-root mode, inspects owner Arena result directories for JSON ingest artifacts, validates job-specific policy packages, optional real robot POV/action diagnostics, optional deployment-outcome diagnostics, and closure evidence, optionally copies a validated WebApp request into the configured inbox, optionally stages Arena pointers, optionally copies real POV evidence to `pipeline/robot_eval_inputs/real_robot_pov_manifest.json`, and optionally copies policy/deployment/closure evidence under `pipeline/robot_eval_inputs/<job_id>/`; task/scenario outcome records can stage as optional calibration diagnostics, exact `scenario_eval_run_id` or `scenario_variation_instance_id` keys are needed only before a calibration score is claimed, and real POV evidence is needed only before claiming real POV proof; does not process the job, run Arena, execute policies, mutate env files, upload storage, require real-world evidence for sim-only work, or upgrade proof claims |
| `blueprint-live-pipeline-intake-service --host 127.0.0.1 --port 8765` | Authenticated WebApp-to-Pipeline intake service | Requires configured HMAC client secrets; accepts short-lived allowlisted capture download grants, direct WebApp job requests, queue envelopes, robot-team policy packages, real robot POV manifests, deployment outcome JSON, and `live_robot_eval_closure_evidence.v1` bodies. Capture transfer streams supported video or source-bound external PLY imports into quarantine, requires exact size/media checks plus a configured clean malware scan, computes SHA-256, content-addresses input without persisting the grant, and returns a separately validated deterministic Capture QA publication. An external PLY remains a derived appearance-only import, never raw, metric, collision, or physics authority. The service may stage validated requests/evidence and optionally run a configured trigger command; it does not run Arena, execute policies, call paid providers, establish reconstruction or task success, publish physical/deployment/safety claims, or set proof booleans. |
| `blueprint-run-live-pipeline-control-plane --job-request-inbox <dir> --arena-results-dir <dir>` | Timer-safe live pipeline control-plane pass | Writes the control-plane manifest plus `live_pipeline_external_input_packet.json` and `.md`, drains queued WebApp job requests when configured, resolves `site_package.capture_root` per request, records exact missing WebApp IDs, owner Arena evidence, command hooks, SDK/operator gates, and generated real-world follow-up draft queues, and keeps queued requests blocked when their required WebApp IDs or capture-root pointers are missing; the raw command exits 0 when blocked so timers do not restart-loop, but the production systemd unit runs `blueprint-control-plane-postchecks.sh` so blocked manifests page an operator or fail the unit when the required webhook is missing; does not auto-run follow-up reruns or run live simulator/provider/upload/operator paths unless separately gated |
| `blueprint-run-runpod-provider-adapter --provider-launch-request <job-dir>/gpu_provider_launch_request.json [--mode dry-run]` | RunPod provider adapter dry-run | Writes `runpod_provider_adapter_result.json` with RunPod serverless `/run` and GraphQL on-demand Pod request shapes plus the cost-control split between request policy, endpoint settings, and on-demand Pod watchdog requirements; performs no API call by default, stores no API key, and does not prove allocation, simulator execution, generated-world rank fidelity, or public proof |
| `python -m blueprint_pipeline.paid_resource_allocator gpu-canary ...` | Canonical bounded paid GPU canary | Validates the selected lane's immutable SHA/image/input bindings, provider preflight, resource count, spend, TTL, retries, and independently armed watchdog before granting the internal adapter call. Reconstruction worker, pose, trainer, and Isaac canaries are Vast-first and require scoped plus global provider-zero; Isaac uses a separate exact-package runtime bundle and one-GPU request. Output retrieval, teardown, and provider-zero are independent gates. A completed canary does not prove reconstruction quality, simulator task success, safety, physical success, deployment, or public proof. |
| `blueprint-build-oscar-wam-provider-bundle --wam-rollout-input-manifest <json> --oscar-input-dir <dir>` / `blueprint-build-oscar-wam-gpu-image --image-ref <versioned-image>` | OSCAR WAM provider bundle/image preparation | Prepares the OSCAR runtime bundle or reusable GPU image contract using the official `oscar-public` source pinned to `4dea2f657e221b0ff24c895fcc8ab4d46d5a9adb`, `zywu2115/OSCAR-2B` pinned to `c9781ffa7dd8556d862d7d9f338a2ea008a58ca6`, and the checked provider image digest when launching RunPod/Vast; writes provenance and visual-input gates but does not call providers, allocate GPUs, execute OSCAR, prove visual usefulness, task success, safety, physical readiness, deployment approval, or generated-world rank fidelity |
| `BLUEPRINT_ALLOW_DEEPINFRA_API_CALLS=true DEEPINFRA_API_KEY=<secret> blueprint-run-oscar-wam-provider-command-adapter --provider deepinfra --allow-paid-provider-launch` | Metered DeepInfra model API call (not resource allocation) | Calls a hosted model API and may incur per-request API spend, but it does not create a GPU/VM resource; it writes redacted execution/cost artifacts and does not prove task success, safety, generated-world rank fidelity, physical readiness, deployment approval, or raw capture truth |
| `blueprint-run-persistent-wam-short-visual-sanity --policy-observation <json> --provider <runpod|vast> --transition-count 2` | Local short WAM visual sanity gate; paid launch disabled | Runs source visual QA and may consume already-returned artifacts, but legacy RunPod/Vast creation is grant-gated with no public issuer; it performs no paid resource allocation until an explicit canonical allocator route is added and does not prove task success, safety, generated-world rank fidelity, or raw capture truth |
| `blueprint-build-wam-perception-harness-gpu-image --image-ref <versioned-image>` | Build reusable WAM perception harness image context | Writes a local Dockerfile plus build, push, healthcheck, and model-mount helper scripts for the WAM-derived perception harness provider stack; it bakes code/dependency/model-cache intent into the context but does not run Docker, push images, store raw registry/HF/object-store secrets, allocate GPUs, execute providers, prove perception accuracy, sensor depth, contact, safety, generated-world rank fidelity, or generated-world rank fidelity |
| `BLUEPRINT_ALLOW_SYNTHETIC_FALLBACK_WAM_LAUNCH_EXPERIMENT=true <persistent WAM bundle/run command>` | Synthetic fallback WAM experiment override | Allows synthetic fallback initial observations or synthetic 2D WAM seeds to enter live/review-quality WAM provider bundle experiments; artifacts must still label `capture_truth=false`, `geometry_truth=false`, and `visually_useful_rollout` separately from provider/runtime success; does not prove raw capture truth, geometry truth, task success, safety, generated-world rank fidelity, or generated-world rank fidelity |
| `blueprint-audit-live-pipeline-proof-boundary --manifest-path <control-plane-manifest>` | Live control-plane proof-boundary audit | Reads the control-plane manifest, external input packet, setup manifest, and staged-input pointers; writes `live_pipeline_proof_boundary_audit.json`; fails on internal artifact inconsistency, malformed staged pointers, secret leakage, or forbidden proof upgrades, but exits zero for a healthy waiting state with external blockers unless `--require-live-ready` is set |
| `blueprint-validate-provider-preview-packet --capture-root <path> --mode production --require-webapp-sync` | Validate privacy-safe World Labs input lineage before provider submission | Local artifact validator; writes `pipeline/provider_preview_qa_manifest.json`; exits nonzero when production raw bypass, missing privacy proof, checksum mismatch, missing/placeholder WebApp upstream ids, or adapter/canonical mismatch is present; no provider calls |

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
| `blueprint-agent-review --capture-root <path> --provider local` | Deterministic no-LLM agent review | Writes local contract review, blocker, memo, and recapture artifacts without external provider calls; `--provider openai` / `--provider claude` remain optional configured review-provider paths |
| `blueprint-run-e2e --capture-root <path> --provider local` | Optional e2e wrapper with no-LLM review | Runs the local deterministic review path plus default evaluation prep/WebApp sync handoff without external review-provider calls; `openai`/`claude` may call configured provider overrides |
| `BLUEPRINT_PREVIEW_PROVIDER=world_labs ...` | World Labs preview path | Can submit live provider job when key/env are present |
| `python scripts/fetch_g1_sonic_assets.py --output-dir <absolute-dir> [--verify-only]` | Stage the pinned default GEAR-SONIC encoder and decoder for G1 development rehearsal | Without `--verify-only`, downloads third-party model bytes from the pinned Hugging Face revision into local storage and requires external-model acquisition authorization; `--verify-only` reads local bytes only. Neither mode grants rights review, starts inference, allocates a GPU, or proves an episode. |
| `blueprint-materialize-worldlabs-assets --capture-root <path>` | Download already-generated World Labs/Marble output assets into local checksum-backed files | Downloads remote CDN assets but does not start a new generation, run simulators, or prove generated-world rank fidelity |
| `BLUEPRINT_ALLOW_FAKE_LIVE_OPERATORS=true blueprint-ingest-arena-results --operator-mode fake ...` | Local fake Arena operator lane | Runs local fake Agents/Codex operator decisions without spend or network; proves only operator ledger and routing code paths |
| `BLUEPRINT_ALLOW_LIVE_AGENTS_SDK_OPERATORS=true BLUEPRINT_ALLOW_LIVE_CODEX_SDK_OPERATORS=true blueprint-ingest-arena-results --operator-mode agents-sdk --allow-live-agents-sdk --allow-live-codex-sdk ...` | Live Arena Agents SDK + Codex SDK operator lane | Can call live SDK operators when dependencies/auth exist; may inspect manifests, choose deterministic commands, route review, and diagnose code-maintenance issues, but cannot set proof booleans true without deterministic accepted artifacts |
| `BLUEPRINT_ALLOW_LIVE_CODEX_SDK_OPERATORS=true BLUEPRINT_ALLOW_CODEX_CLI_HOST_OAUTH=true ... --allow-live-codex-sdk...` | Optional Codex CLI host-OAuth transport | Can use an installed authenticated `codex` CLI instead of the Python Codex SDK where wired; auth remains in the host/user Codex profile, and the operator still cannot mutate proof booleans directly |
| `blueprint-run-live-pipeline-control-plane` | Timer-safe live control-plane pass | Loads env files, audits setup, and optionally consumes a robot-eval job inbox; the raw command can return 0 with a blocked manifest for timer safety, while the deployed systemd postcheck pages or fails blocked passes; does not prove simulator/GPU/generated-world rank fidelity |
| `BLUEPRINT_ALLOW_ROLLOUT_VISION_LABELING=true blueprint-ingest-arena-results --allow-rollout-vision-labeling --vision-labeling-command <command> ...` | Explicit rollout vision labeling lane | Can run a configured local/HTTP wrapper command; fallback labels remain review-required and label output cannot prove contact/safety/generated-world rank fidelity by itself |
| `BLUEPRINT_ALLOW_ROLLOUT_VISION_LABELING=true blueprint-label-rollout-vision-openai --output-dir <arena-package-dir>` | OpenAI rollout vision-label command hook | Calls OpenAI Responses with extracted keyframes and writes `rollout_vision_labels.command.json`; labels are forced review-required and cannot prove contact/safety/generated-world rank fidelity |
| `BLUEPRINT_ALLOW_PACKAGE_DELIVERY_UPLOAD=true blueprint-ingest-arena-results --allow-delivery-upload --delivery-command <command> ...` | Explicit package delivery upload lane | Can run a configured upload/signed-access command and log signed-access status; local bundle is default and entitlement still requires owner-system review |
| `BLUEPRINT_ALLOW_PACKAGE_DELIVERY_UPLOAD=true blueprint-deliver-arena-package-local --output-dir <arena-package-dir>` | Package delivery hook | Copies `delivery_bundle/` to `BLUEPRINT_LOCAL_DELIVERY_ROOT` when set; with `BLUEPRINT_PACKAGE_DELIVERY_GCS_PREFIX=gs://...`, uploads the bundle/archive to GCS and writes `artifact_uri` plus a WebApp entitlement patch payload; optional `BLUEPRINT_PACKAGE_DELIVERY_SIGNED_URLS=true` mints short-lived URLs when credentials allow. It does not verify buyer entitlement, authorize access, approve deployment, or prove physical readiness |

Do not run live provider jobs without explicit approval. A present provider key
does not mean the user approved spend or external API mutation.
Do not run simulator or training approvals unless the user explicitly requests
that proof-producing run and the dependency/cost/GPU boundary is understood.

## Deploy Scripts And Live-Risk Commands

| Command | Use | Risk |
| --- | --- | --- |
| `deploy/scripts/deploy.sh` | Deploys service infrastructure | Live deployment risk |
| `python scripts/verify_deploy_release_provenance.py ...` | Reads canonical GitHub Actions run/job/artifact metadata and verifies exact full-lane collection evidence | Read-only GitHub/network access; writes local provenance evidence and does not deploy |
| `python scripts/build_cpu_full_lane_evidence.py ...` | Binds planned/executed pytest node IDs to exact JUnit outcomes and source digests | Local artifact reads/writes only; fails on every skip, failure, error, duplicate, or testcase substitution |
| `python scripts/build_gpu_provider_canary_evidence.py preflight\|convert ...` | Validates approved exact-image, bounded spend/time inputs and converts an already executed startup canary | Local artifact reads/writes only; conversion does not launch a provider and cannot close EVID-03 |
| `python scripts/verify_critical_capability_lanes.py --scope <scope> ...` | Revalidates scope-required CPU, container, GPU, Pub/Sub, and native LeRobot evidence against exact SHA, freshness, and retained sources | Local artifact reads only; cannot execute a provider, emulator, container, or native loader and cannot upgrade any claim beyond its source artifacts |
| `BLUEPRINT_ALLOW_VAST_API_CALLS=true BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH=true blueprint-run-unitree-groot-n17-sonic-vast-image-canary ...` | Runs the exact-image Unitree GR00T/SONIC Vast startup canary used by the explicit critical-lane dispatch | Live provider allocation and spend; must pass approved digest plus <=1 USD and <=30 minute preflight, and requires explicit approval |
| `python scripts/run_deployment_service_canaries.py ...` | Reads live Secret Manager payloads in process memory and calls IAM-protected Cloud Run health/auth canaries | Live cloud read/invocation; no model execution by contract, but requires explicit deployment approval |
| `python scripts/benchmark_ptdp_scalability.py --clip-count 100000 ...` | Runs synthetic bounded ANN and streaming-JSONL scalability workload | Local CPU/RAM/disk load only; no provider calls and no real-media quality proof |
| `terraform apply` under `deploy/terraform/` | Creates/updates cloud resources | Live cloud mutation |
| `gcloud run deploy ...` | Deploys Cloud Run services | Live deployment risk |
| `git push origin main` | Publishes repo state | Remote mutation |

Do not run deploy, Terraform, cloud, Stripe, WebApp mutation, or push commands
unless the user explicitly requests that action.
