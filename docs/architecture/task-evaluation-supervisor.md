# Task Evaluation Supervisor

Status: Phases 0 through 4 have implementation-level contracts and focused
tests. Every canonical `run_e2e` capture build enters the supervisor, including
interruption-safe continuation. Live agent inference, provider recovery, paid
compute, candidate execution, and physical actions remain fail closed unless a
separately bounded authority envelope is supplied.

## Product boundary

The Task Evaluation Supervisor is a major product control plane around the
existing Decision/Evidence proof kernel. OpenAI Agents SDK is the required
agent harness for the durable supervisor and all six specialist agents:

```text
customer question
  -> OpenAI Agents SDK supervisor interpretation, planning, observation, and replanning
  -> deterministic validation at every authority/evidence transition
  -> deterministic Decision/Evidence aggregation
  -> Decision Envelope: decision, partial decision, or abstention
```

Agents SDK controls agent turns, typed outputs, future tool calls, tracing, and
future specialist coordination. Blueprint controls the run state, allowed-tool
surface, authority, artifact ledger, and proof kernel. Agents control search,
sequencing, explanation, and recovery proposals. The
deterministic proof kernel controls schema acceptance, hashes, rights, budgets,
provider admission, frozen splits, hidden labels, success predicates, evaluator
thresholds, claim ceilings, proof transitions, and the final decision.

Supervisor artifacts are never accepted evidence. A completed provider call or
agent explanation cannot set proof booleans or upgrade a claim.

## Current-state consolidation map

| Existing component | Actual current role | Supervisor relationship |
| --- | --- | --- |
| `decision_evidence_contracts.py` | Versioned testbed, request, method, qualification, plan, result, and decision contracts | Remains the proof-kernel contract authority |
| `decision_evidence_router.py` | Deterministically chooses the cheapest qualified sufficient evidence and compiles leaf Evaluation Runs | Called by the evaluation-method-router capability; the agent does not replace its selection logic |
| `decision_evidence_execution.py` | Explicit adapter registry, normalized evidence, and deterministic Decision Envelope aggregation | Remains execution and verdict authority |
| `evaluation_run_contract.py` | Stable provider-neutral leaf-run contract | Remains the compilation target |
| `agent_runtime/orchestrator.py` | Capture-readiness review with deterministic builders and optional provider overrides | Candidate input/adapter source for the capture-testbed capability; not the durable run supervisor |
| `agent_operator_runtime.py` | Gated one-shot OpenAI/Codex operator wrappers | Shared live-Agents-SDK admission gate only; not the supervisor harness |
| `site_eval_director.py` | Deterministic site/simulation planning plus optional advisory operators | Future registered non-spend tool surface after contract qualification |
| `stance_configuration_agent.py` | Deterministic bounded recovery/search | Candidate deterministic recovery tool, despite its historical “agent” name |
| `adaptive_task_stance_configurator.py` | Bounded candidate proposer with deterministic acceptance | Candidate non-spend proposal tool |
| `policy_autoresearch.py` and `autoresearch/` | Development-time candidate mutation under frozen evaluation | Kept outside hidden evaluation and proof authority |
| `robot_eval_job_orchestrator.py` | Robot-evaluation job lifecycle, provider gates, and optional operator choice | Future pre-authorized runtime tool surface; never bypassed |
| `run_e2e.py` | Required post-capture supervisor stage plus legacy optional agent review | Every completed capture build enters the new supervisor lifecycle; the old review remains a compatibility path |

The migration preserves the existing Decision/Evidence commands and contracts.
The supervisor has an explicit `supervise` operation and a separate artifact
family, and `run_e2e` now enters that same lifecycle after every completed
capture pipeline stage. There is no flag that omits the supervisor from the
normal capture-build path.

## Implemented interfaces

The package `blueprint_pipeline.task_evaluation_supervisor` provides:

- OpenAI Agents SDK as the required harness and core project dependency;
- one typed OpenAI Agents SDK manager that chooses the next eligible
  specialist, observes the exact digest-bound result, and replans after every
  completed specialist turn. Each result carries replay-checked structured
  tool observations, so the manager observes actual registered-tool outcomes
  rather than relying on specialist prose;
- one SDK `Agent` definition and strict structured-output contract for each of
  the six specialist capabilities;
- a typed injectable SDK Runner seam so hermetic tests never make live calls;
- bounded capture-build ingress that reads only known manifests and exposes an
  allowlisted projection, never arbitrary files or raw media;
- all six autonomy modes as validated enum values;
- an authority envelope that always denies proof, rights, budget, hidden-label,
  and physical-action authority;
- typed tool descriptors and a capability-gated registry;
- typed action proposals, capability results, invocation manifests, events,
  run state, and terminal reports;
- an independent deterministic supervisor evaluator with hidden expected
  properties, claim/clarification/recapture/routing/failure/abstention/audit
  metrics, critical-boundary checks, and an explicit no-self-grading flag;
- a hash-chained append-only JSONL event ledger that rejects partial records,
  sequence changes, run changes, and chain changes;
- idempotent completed-run reuse and interruption-safe continuation that invokes
  only capabilities not already committed to the ledger;
- a persistent inference reservation audit that records worst-case cost before
  each live SDK provider call, records completion separately, restores the
  cumulative ceiling after restart, and refuses to repeat a call whose prior
  billing/result state is ambiguous;
- normalized deterministic kernel-input snapshots plus a replay verifier that
  revalidates input, result, invocation, and event digests without a model call;
- one durable supervisor state machine per run;
- six OpenAI Agents SDK shadow capabilities:
  - claim and task interpreter;
  - capture and testbed supervisor;
  - evaluation method router;
  - runtime failure recovery;
  - scenario and adversarial-test proposer;
  - post-run diagnostician.
- deterministic customer decision reports bound to the validated request,
  testbed, evidence plan, normalized evidence, Decision Envelope, agent
  proposals, action artifacts, spend, runtime, uncertainty, claim ceilings,
  next experiments, and prohibited claims;
- clarification and authorization request/receipt contracts in which an agent
  may request input or authority but cannot answer or approve its own request;
- pre-evaluation scenario proposal artifacts plus a separate operator-only
  freeze contract that records evaluator, success-predicate, and hidden-label
  manifest digests without exposing hidden labels;
- a 12-case synthetic evaluation corpus with four development and eight
  held-out cases, case-specific inputs, hidden leakage canaries, 20 recorded
  human-guided baseline metrics spanning reasoning quality, spend, recovery,
  authority, leakage, scenarios, audit, and replay, plus aggregate comparison
  before autonomy promotion. The trigger-aware hermetic fixture scores
  `0.987500` versus the recorded `0.978313` baseline on eight held-out cases
  (`+0.009187`), with zero critical boundary violations; this remains below
  the frozen `+0.05` promotion threshold;
- a pre-authorized recovery controller with provider/action allowlists,
  immutable commit and input bindings, cumulative spend, expiry/TTL, retry
  ceilings, non-retryable scientific failures, watchdogs, mandatory teardown,
  and preserved failure evidence;
- a frozen agentic candidate PolicyAdapter plus a neutral three-stack suite for
  direct policy, decomposed planner+policy, and verify/recover supervisor
  candidates under the same frozen scenarios, evaluator, and predicates.

Historical deterministic implementations remain only as frozen baselines and
fixture oracles for independent agent evaluation. They are not a production
fallback and are not selected by the supervisor product path.

## Capture-build-first start

A completed capture build is sufficient to start the supervisor. It is not
sufficient to fabricate a customer decision. The deterministic ingress reads
only registered manifest paths, records their hashes and schema names, and
builds an allowlisted metadata projection. It does not read raw media.

When no Decision/Evidence Request or maintained Site-Task Testbed exists, the
manager first triggers claim interpretation, then capture/testbed inspection,
and stops with a typed clarification or blocker. Scenario, routing, recovery,
and post-run diagnosis are not invoked merely to satisfy a fixed call count.
They become eligible only when their required validated inputs exist. All six
specialists remain registered for every run. The proof kernel accepts no claim
until the missing task, robot, operating condition, success, rights, and
evidence contracts are deterministically valid.

The canonical lifecycle is:

```text
capture materialization -> capture pipeline -> required Task Evaluation
Supervisor -> optional legacy review/support stages -> deterministic evidence
and decision stages
```

There is no `--skip-task-evaluation-supervisor` control.

The absence of live inference authority never causes a fallback to a different
harness. The required supervisor records a typed blocked result until an
inference budget and the shared provider gate are present. The completed capture
remains usable; Blueprint simply has not yet performed the agent reasoning or
made a Task Evaluation decision.

## Shadow artifacts

An explicit shadow run writes:

```text
task_evaluation_supervisor_run.json
authority_envelope.json
tool_registry_manifest.json
proof_boundary.json
supervisor_events.jsonl
supervisor_state.json
terminal_supervisor_report.json
kernel_inputs_manifest.json
kernel_inputs/*.json
manager/decisions/step-*.json
manager/invocations/step-*.json
manager/refusals/step-*.json
capabilities/<capability>.json
invocations/<capability>.json
```

Every capability output is marked `authoritative=false`,
`proof_booleans_mutable=false`, and `proof_effect=none`. Every proposal is
classified against the tool registry and authority envelope. Shadow proposals
are recorded but never executed.

`replay_supervisor_run(...)` verifies the hash chain and all recorded contracts.
When a deterministic Decision Envelope is present, replay reproduces its digest,
outcome, and claim ceiling. Agent prose is intentionally not regenerated, so a
changed future model cannot change the replayed proof result.

Run the current vertical slice with:

```bash
blueprint-route-task-evaluation supervise \
  --capture-build /path/to/completed-capture \
  --mode shadow \
  --output-dir out/supervisor
```

`--request`, `--testbed`, evidence profiles, qualifications, plans, results, and
decisions can be added as the run accumulates them. The harness is always
OpenAI Agents SDK; there is no alternate production agent harness selector.
The standard `run_e2e` capture path invokes this lifecycle automatically and
records its status in the stage ledger and run summary.

Live inference is fail closed and requires `--allow-live-agent-sdk`, a positive
`--agent-inference-budget-usd`, and
`BLUEPRINT_ALLOW_LIVE_AGENTS_SDK_OPERATORS=true`. Before every SDK call the
harness reserves a conservative worst-case input/output cost against the one
run budget and writes that reservation before entering the provider. A separate
completion artifact closes it. Failed or interrupted calls keep their
reservation; after restart, Blueprint restores the cumulative ceiling and
refuses an identical call when the earlier billing/result state is unknown.
Tests inject the SDK Runner boundary and make no API call. No live call, paid
compute, or provider action was authorized or run while implementing this
slice.

Optional `--plan`, `--result`, and `--decision` inputs let the failure-recovery
and post-run capabilities inspect already validated artifacts. They do not
change those artifacts.

## Enabled and disabled behavior

| Mode | Current behavior |
| --- | --- |
| `disabled` | Explicit fail-closed administrative/test mode; writes lifecycle/proof-boundary evidence and invokes no capability. It is not an alternate product harness or a normal capture-build path. |
| `shadow` | Runs the SDK manager and only specialists whose deterministic prerequisites are present; records proposals and executes no proposed action |
| `advise` | Recognized but blocked until operator-approval receipts exist |
| `execute_non_spend` | Runs SDK agents with capability-scoped, registered read-only inspection tools. Tool observations are digest-bound, zero-cost, replay-validated, and have `proof_effect=none`; broader non-spend actions remain gated. |
| `execute_preauthorized` | Operational only when an operator-issued, digest-bound receipt and a scoped recovery controller are injected; otherwise blocked. The current receipt is not a cryptographic signature. The controller enforces provider/action allowlists, SHA/input bindings, spend, TTL, retries, watchdog, and teardown. |
| `candidate_policy` | The mode remains fail closed in the generic supervisor runner; candidate execution is compiled through the separate frozen neutral PolicyAdapter suite so candidate code never inherits supervisor/evaluator authority. |

Unknown modes fail before a run is created.

## Implemented phase gates and remaining evidence

Phase 2 includes capability-scoped read-only tools for claim-contract, testbed,
evidence-plan, normalized-result, and Decision Envelope inspection. It also
includes the first reversible non-spend action: the evaluation-method agent can
ask Blueprint to deterministically route the bound request and testbed, persist
the resulting Evidence Plan, and materialize validated leaf Evaluation Run
specs under that supervisor run. Generated paths are fixed by Blueprint, every
artifact is digest-bound and replay-validated, provider execution is not
started, and proof state is unchanged. The capture/testbed agent can also write
a targeted recapture proposal bound to the current capture or testbed. It may
not initiate capture, request a full-site recapture through this action, infer
rights, mutate raw capture, or create proof. Phase 2 also includes deterministic
customer report generation, clarification and authorization receipts,
scenario-proposal materialization with an operator-only freezing boundary, and
tests proving that identical accepted evidence yields the same kernel decision
even when agent prose changes. The recorded corpus comparison now runs each
held-out input separately and correctly refuses autonomy promotion for the
fixture runner. It proves the evaluation and promotion boundary, not that a live
production model has beaten the human-guided baseline.

Phase 3 routes recovery only through an injected pre-authorized controller. The
controller is provider-neutral and no live provider was called while building
it. A concrete `VastWAMRecoveryAdapter` now wraps the existing authorized Vast
WAM runner; it is not a second launcher. It binds the exact bundle, commit, and
input digests, forwards the receipt-bounded spend and runtime ceilings, requires
the independent watchdog, disables retention, and accepts provider-zero only
from versioned teardown evidence plus terminal watchdog confirmation. It refuses
to launch when less than Vast's one-minute minimum authority window remains.
The controller uses its own clock, binds the provider and action allowlists
inside the operator-issued receipt, rejects ambiguous adapter exceptions from
automatic retry, and accepts teardown only when `provider_zero=true` is
explicitly proven. Before calling an adapter it also requires the opaque
in-process grant issued by Blueprint's shared paid-resource admission
chokepoint; a provider-supplied Boolean cannot replace that grant. Operationally,
Vast is the preferred first canary because
its Blueprint qualification-session lane has stronger prior evidence; RunPod
remains a replaceable fallback that must independently pass the same admission
and provider-zero gates. Live provider recovery must still enter through the
canonical paid allocator seam before production enablement.

The canonical allocator's older generic `strict-policy-smoke` parser still
defaults to RunPod because that specific launcher is RunPod-only. That legacy
default is not inherited by the Task Evaluation Supervisor. The newer policy-
ranking path already defaults to Vast, and a supervisor recovery action can use
only a provider named in its trusted operator receipt and installed adapter.

Phase 4 admits Pigey-like or other agentic robot stacks only as frozen candidate
`PolicyAdapter` implementations. Candidate code, model/provider, prompt, tools,
memory/skills, budgets, retries, scenario pack, evaluator, and success predicates
are digest-bound before hidden evaluation. The runtime's execution-relevant
configuration has its own digest, so changing a model, external checkout,
scenario binding, tool interface, step/timeout envelope, endpoint, or cost rate
without refreezing the candidate is rejected before any output directory or
provider action is created. Candidate agents cannot see hidden labels or grade
themselves. The implementation compiles neutral Evaluation Run
specs and provides an explicitly gated neutral execution harness: candidate
runtimes receive only public frozen specs, while an independent evaluator alone
receives the digest-bound hidden manifest. Candidate traces and evaluator
outputs are independently digest-checked, and candidate-supplied scores or
unregistered result fields are refused. A paid candidate runtime must also use
authoritative cost accounting from its trusted Blueprint wrapper;
candidate-reported token usage cannot satisfy that gate. If a paid runtime
loses its result after execution begins, the suite stops, preserves a typed
failure, marks reported cost non-final, and requires reconciliation before
later candidates can run. A concrete
external-checkout Pigey adapter binds an exact clean upstream commit and
entrypoint digest, invokes the
public CLI without a shell, passes only allowlisted environment variables,
normalizes `trial.json` into Blueprint's candidate trace, and explicitly drops
Pigey's own success value. Pigey source is not vendored, and its current adapter
marks trial-reported usage as non-authoritative. A trusted spend-metering
gateway, separate third-party license review, and paid-execution authorization
are all required before a live run. Hermetic fixture execution proves this
boundary; no live Pigey/provider run or physical validation is claimed.

No phase may infer physical validation, deployment approval, safety
certification, or policy-ranking support from simulation or generated media.
