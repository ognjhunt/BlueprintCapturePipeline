# Goal: Fully Automated Arena Eval + Post-Training Data Package Pipeline

Repo: $HOME/workspace/BlueprintCapturePipeline

Read first:
- AGENTS.md
- PLATFORM_CONTEXT.md
- WORLD_MODEL_STRATEGY_CONTEXT.md
- README.md
- pyproject.toml
- docs/SIMULATION_AUTOMATION_LANE.md
- docs/PIPELINE_CURRENT_PROCESS_AUDIT_2026-06-06.md
- src/blueprint_pipeline/simulation_automation.py
- src/blueprint_pipeline/site_eval_director.py
- src/blueprint_pipeline/robot_eval_job_orchestrator.py
- src/blueprint_pipeline/post_training_data_package.py
- src/blueprint_pipeline/capture_batch_registry.py

Current repo truth:
- isaac_lab_arena backend already exists alongside isaac_sim.
- pipeline/simulation_automation/arena_environment_packet.json is emitted.
- Arena packet translates Site/Task/Scenario/Eval cards plus episode_spec.v1.
- GPU handoff/job artifacts surface the Arena package and command template.
- Real simulator execution is intentionally blocked unless env + CLI + owner proof gates exist.
- Live-operator pass implemented: Agents SDK and Codex SDK adapters are live
  gated operators where wired, not advisory-only writers, while deterministic
  artifacts remain authoritative.

Non-negotiables:
- Preserve unrelated dirty worktree changes.
- Do not fake simulator, robot, safety, policy, contact, or delivery proof.
- Fixture/local tests may prove code paths only.
- Real GPU/provider/simulator/model/storage actions must require explicit env + CLI gates and must log proof artifacts.
- Live Agents SDK/Codex SDK operators are allowed and desired, but cannot override rights/privacy/proof booleans without accepted evidence artifacts.
- Deterministic manifests, checksums, schemas, and tests own truth. Agents coordinate, inspect, retry, summarize, and repair code under gates.

Implement these missing systems:

1. Arena result ingest
- Add src/blueprint_pipeline/arena_result_ingest.py and CLI.
- Parse Isaac Lab-Arena rollout outputs, shard manifests, stdout/stderr, metrics, videos, per-episode logs, artifact manifests.
- Emit canonical normalized_attempt_trace.json, failure_labels.json, metrics, artifact checksums, and ingest ledger.
- Integrate robot_eval_job_orchestrator so real Arena results feed evaluation_result instead of just copied fixture artifacts.

2. 500-scenario scheduler
- Add deterministic sharding/parallelization scheduler with num_envs, shard size, timeout, retry budget, GPU cost budget, resume state, and per-scenario rerun policy.
- Emit arena_eval_schedule.json, shard manifests, retry queue, cost ledger, and resume manifest.

3. Robot policy adapters
- Support policy API endpoint, Docker container, recorded action traces, high-level skill traces, teleop demos, and sim controller plugin.
- Emit policy_adapter_manifest.json with launch proof, blocked reasons, redacted secrets, interface contract, and command templates.

4. Clip extraction
- Add rollout clip extraction from rollout video/log timestamps.
- Emit clips_manifest.json, clip files or blocked/degraded status, keyframes/contact sheets if feasible.
- Use existing deps where possible; optional ffmpeg/moviepy/opencv paths must fail closed when missing.

5. Vision labeling
- Add rollout vision labeler using optional SAM3 or similar command/HTTP hook.
- Gate with env + CLI, e.g. BLUEPRINT_ALLOW_ROLLOUT_VISION_LABELING plus command/url config.
- Emit masks/object-state/contact/occlusion/threshold-miss/failure-evidence labels.
- Fallback deterministic labels must be review_required, not proof.

6. Human review resolver
- Add review resolution workflow that consumes review_required labels and writes accepted/rejected/edited labels.
- Emit review_resolution_ledger.json and accepted_failure_labels.json.
- Human acceptance or explicit owner proof is required before upgrading review-derived claims.

7. Dataset packager
- Expand post_training_data_package.py from manifest-only into a real package builder.
- Always produce JSONL/manifest/checksum package.
- Add optional RLDS, LeRobot, HDF5, Parquet, and video bundle exports when deps are installed; otherwise emit degraded/blocked format entries.
- Emit dataset card, license manifest, export manifest, checksums, package index, and archive.

8. Customer handoff report
- Add report generator with buyer-facing eval summary, scenario pass/fail metrics, failure review, known limits, proof boundary, data package inventory, and export instructions.
- Emit Markdown and JSON reports.

9. Storage/delivery automation
- Add local delivery bundle by default.
- Add gated upload/signed URL path only under env + CLI gates.
- Emit entitlement check, retention policy, egress estimate, delivery manifest, and signed-access manifest when actually run.

10. Autonomous rerun loop
- Detect failed, flaky, ambiguous, missing-artifact, timeout, and review-required scenarios.
- Retry only eligible shards/scenarios.
- Stop at retry/cost/time budget.
- Preserve rerun reason and lineage for every rerun.

11. Live Agents SDK + Codex SDK operators
- Replace advisory-only SDK status where appropriate with gated live operator mode.
- Agents SDK role: eval director and pipeline operator. It may inspect manifests/logs, choose next deterministic command, trigger allowed reruns, route review, summarize blockers, and maintain progress ledgers.
- Codex SDK role: code maintainer. It may diagnose failures, patch code, run tests, and produce diffs when pipeline failures require code changes.
- Add env + CLI gates for live operators and spend-bearing/external actions.
- Log every agent decision, tool call summary, command chosen, refusal, blocker, and proof effect.
- Agents may never directly set proof booleans to true without deterministic accepted artifacts.

Update docs:
- README.md
- docs/SIMULATION_AUTOMATION_LANE.md
- docs/architecture/command-safety-matrix.md
- docs/PIPELINE_CURRENT_PROCESS_AUDIT_2026-06-06.md or a new audit doc
- Add schemas/runbooks where helpful.

Tests and verification:
- Add focused unit tests for each new module.
- Add integration tests for robot_eval_job_orchestrator and site_eval_director with fixture Arena outputs.
- Assert default blocked behavior for real simulator/provider/vision/storage/agent execution.
- Assert gated fake/local live-operator mode works without real spend.
- Assert data package contains normalized trace, labels, metrics, clips manifest, dataset card, license manifest, checksums, and archive manifest.
- Run focused tests first:
  - pytest tests/test_simulation_automation.py tests/test_site_eval_director.py tests/test_robot_eval_job_orchestrator.py tests/test_production_handoff_readiness.py
  - plus all new test files
- Run full pytest if feasible.
- Run CLI help/smoke commands for every new CLI.
- Run python -m compileall src tests.
- Run ruff check if configured.
- Run git diff --check.
- Finish with a final diff/proof-boundary audit.

Done means:
- The repo can take an Arena-style fixture run representing a 500-scenario batch, ingest results, label failures, cut/manifest clips, build a data package, generate a customer report, prepare delivery artifacts, schedule targeted reruns, and log agent/operator decisions.
- Real external simulator/provider/model/storage execution remains blocked unless explicit owner gates and proof artifacts exist.
- Agents SDK and Codex SDK are live gated operators, not merely manifest writers.
