# Task Evaluation Supervisor operator runbook

## Product path

A completed capture build always enters the Task Evaluation Supervisor stage.
The capture is enough to start reasoning, but it is not enough to invent a task,
robot, success condition, rights grant, or customer decision.

The production agent harness is OpenAI Agents SDK. There is no alternate agent
harness or deterministic agent fallback. The deterministic proof kernel is a
separate authority boundary beneath the SDK agents.

```text
capture build
  -> six specialist agent capabilities
  -> registered proposals and observations
  -> deterministic contracts and proof kernel
  -> decision, partial decision, abstention, or typed blocker
```

## What starts each capability

1. The claim interpreter runs when a customer question or capture-only ingress
   arrives. It proposes claims and clarification questions.
2. The capture/testbed supervisor runs after interpretation and whenever a new
   capture or recapture receipt is attached. It proposes only the smallest
   useful recapture.
3. The evaluation router runs when validated claims and a maintained testbed are
   available. It compiles deterministic leaf runs and chooses only qualified
   evidence methods.
4. The recovery capability runs when a structured runtime or evidence failure is
   present. It may propose recovery in shadow/advise modes and may execute only
   through a pre-authorized controller.
5. The scenario proposer runs before hidden evaluation. Its proposals remain
   unfrozen until an independent operator receipt freezes scenario IDs,
   evaluator digest, success-predicate digest, and hidden-label manifest digest.
6. The post-run diagnostician runs only after a deterministic decision artifact
   exists. It explains that artifact and cannot change it.

## Autonomy modes

- `disabled`: write boundary evidence and invoke no agent.
- `shadow`: invoke agents and record proposals; execute nothing.
- `advise`: require an operator decision before any proposal can advance.
- `execute_non_spend`: permit registered, reversible, non-spending actions.
- `execute_preauthorized`: permit only receipt-bound recovery actions through an
  injected controller and registered provider adapter.
- `candidate_policy`: fail closed in the supervisor runner. Agentic robot stacks
  enter through the separate frozen PolicyAdapter evaluation suite.

Unknown modes and missing validators, receipts, adapters, budgets, or audit
records fail closed.

## Starting from a capture build

The normal end-to-end command invokes the supervisor automatically:

```bash
python -m blueprint_pipeline.run_e2e \
  --capture-root /path/to/staged-capture \
  --provider openai
```

The standalone control-plane command is:

```bash
blueprint-route-task-evaluation supervise \
  --capture-build /path/to/completed-capture \
  --mode shadow \
  --output-dir /path/to/supervisor-output
```

Live SDK inference additionally requires the explicit CLI admission, a positive
inference budget, and `BLUEPRINT_ALLOW_LIVE_AGENTS_SDK_OPERATORS=true`. Missing
live authority produces a typed blocker, not a fabricated local answer.

## Non-spend execution

`execute_non_spend` can inspect registered artifacts, deterministically compile
an Evidence Plan and leaf Evaluation Run specs, materialize clarification and
authorization requests, write targeted recapture proposals, and materialize
pre-evaluation scenario proposals. It cannot start capture, run a provider,
spend money, expose hidden labels, or mutate proof state.

Clarification responses and authorization grants are separate receipts created
at trusted customer/operator boundaries. Agent output never satisfies its own
request. Every accepted response is revalidated by deterministic contracts.

## Pre-authorized recovery and provider choice

The recovery controller requires all of the following before invoking an
adapter:

- an operator-issued receipt whose digest binds provider IDs, action IDs,
  immutable input digests, cost, TTL, retries, issue time, and expiry;
- an exact immutable commit SHA and exact input-digest set;
- a registered adapter for the named provider;
- an opaque grant from Blueprint's shared paid-resource admission chokepoint;
- controller-clock validity, remaining TTL, retry capacity, and spend capacity;
- a non-scientific, retryable failure class;
- a watchdog and mandatory teardown;
- explicit `provider_zero=true` closure.

Adapter exceptions are ambiguous and are not retried automatically. Failed
evidence is retained. A teardown status without provider-zero proof is failure.

Vast is the preferred first authorized canary because Blueprint has stronger
evidence for its maintained qualification-session path. RunPod is a fallback,
not the supervisor default. The older generic allocator defaults to RunPod only
because its `strict-policy-smoke` implementation is RunPod-specific. Any paid
mutation must still enter through:

```bash
python -m blueprint_pipeline.paid_resource_allocator gpu-canary ...
```

Do not invoke provider adapters as launchers. Do not run a paid canary without a
separate explicit spend authorization.

## Agentic candidate evaluation

Pigey-like planners, VLA/TAMP compositions, and verify/recover stacks are frozen
as `blueprint_agentic_candidate_policy@1` candidates. Before hidden evaluation,
freeze the code, model/provider/version, prompts, tools, memory/skills snapshot,
budgets, retries, scenario pack, evaluator, and success predicates.

The reviewed Pigey commit currently has no repository-level software license.
Do not execute it for Blueprint merely because the repository is public. The
runtime requires an independent `pigey_license_attestation.v1` bound to the exact
commit and authorizing the intended commercial code execution. Until the rights
holder grants permission or publishes acceptable terms, the live Pigey lane is
blocked. See `docs/third-party/pigey-license-review.md`.

The neutral suite contains one direct policy, one decomposed planner+policy, and
one verify/recover supervisor. Every candidate receives the same scenario IDs,
evaluator, predicates, and claim ceiling. Candidates receive no hidden labels,
evaluator authority, proof authority, or self-grading path.

The explicitly gated neutral execution harness gives each registered candidate
runtime only its frozen public Evaluation Run spec. It verifies the emitted
trace artifact, then passes that trace and the separately digest-bound hidden
manifest to the independent evaluator. Candidate runtime outputs have a strict
allowlist and cannot contain their own verdict or score. The resulting harness
artifact is still simulation/evaluation evidence only; it is not physical or
deployment proof.

### Pigey/OpenAI cost reconciliation

Pigey's `trial.json` token totals are diagnostic only. A paid Pigey runtime must
use a dedicated, pre-provisioned OpenAI project/API-key scope and bind the exact
project ID, API-key ID, and an operator-controlled exclusive-scope attestation
digest into both the frozen runtime configuration and its independent cost
authority. `OPENAI_PROJECT` must match that frozen project.

Before candidate execution, `OpenAIProjectCandidateCostAuthority` queries the
official organization Costs endpoint with both identifiers and requires a zero
baseline for the attribution window. It reads an admin key from a permission-
restricted file and never records the key. OpenAI cost data may lag, so the
first settlement normally remains `reconciliation_required`. After the frozen
reporting delay, use the read-only reconciliation command against the preserved
candidate execution directory and the same bound authority:

```bash
blueprint-route-task-evaluation reconcile-candidate-costs \
  --execution-dir /path/to/candidate-execution \
  --openai-project-id proj_example \
  --openai-api-key-id key_example \
  --openai-admin-key-file /permission-restricted/openai-admin-key \
  --scope-attestation /path/to/openai-cost-scope-attestation.json
```

The
reconciliation is content-addressed, does not rerun the candidate or evaluator,
and remains partial if provider evidence is missing, malformed, out of scope, or
over the reservation.

The organization Costs result is provider-reported spend evidence. Do not call
it invoice reconciliation, and do not launch Pigey directly merely because the
read-only cost adapter exists. Paid execution still requires an operator receipt
and must call `paid_resource_allocator.admit_pigey_candidate_runtime` in the same
process that will execute the neutral suite. The allocator derives the
worst-case Pigey runtime from its frozen scenario count and per-scenario timeout,
binds the exact suite, authorization, source SHA, cost scope, license, spend,
retry, watchdog, and teardown fields, persists
`openai_api_candidate_allocation_admission.v1`, and injects an opaque grant into
the runtime. `execute=False` is a readiness check only: it returns no grant.
Copying or editing the JSON artifact cannot authorize execution. An OpenAI
candidate presented with the older generic unbound grant is refused.

## Replay and customer report

Preserve the complete supervisor output directory. Replay verifies the
append-only event hash chain, accepted contracts, generated-artifact digests,
tool observations, and deterministic Decision Envelope. Replay does not ask a
current model to regenerate prose.

The deterministic customer report must identify the original question,
validated interpretation, claims, evidence, attempted/failed/skipped methods,
agent recommendations, deterministic validations, spend/runtime, outcome,
uncertainty, evidence ceiling, next useful experiment, and prohibited claims.

No supervisor artifact proves physical success, deployment readiness, safety
certification, or policy-ranking support by itself.

## Independent supervisor evaluation

The committed 12-case corpus is a public synthetic protocol fixture. Do not call
it secret held-out evidence or use its `+0.009187` fixture delta to promote an
autonomy mode. A production comparison requires a separately held
`task_evaluation_supervisor_eval_corpus.v2` with a canonical digest and an
operator-controlled freeze timestamp.

Validate the corpus without printing its hidden case properties:

```bash
blueprint-route-task-evaluation validate-supervisor-corpus \
  --corpus /sealed/evaluation_corpus.v2.json \
  --output /evaluation/corpus_validation.json
```

Before any held-out run, freeze the exact manager identity, all six specialist
identities, SDK version, instructions, tool registry, model, provider, and
inference ceiling. The spec must not contain hidden labels or extra fields:

```bash
blueprint-route-task-evaluation freeze-supervisor-evaluation \
  --corpus /sealed/evaluation_corpus.v2.json \
  --spec /evaluation/agent_configuration_spec.json \
  --output /evaluation/frozen_agent_configuration.json
```

After every case has a terminal recorded run, score the complete matrix. Supply
one `--run case_id=/path/to/run` argument for every sealed held-out case:

```bash
blueprint-route-task-evaluation evaluate-recorded-supervisor \
  --corpus /sealed/evaluation_corpus.v2.json \
  --configuration /evaluation/frozen_agent_configuration.json \
  --run case-a=/runs/case-a \
  --run case-b=/runs/case-b \
  --output-dir /evaluation/result
```

The evaluator first replays every ledger and artifact, verifies that the corpus
predates the configuration and the configuration predates every run, and checks
the recorded manager/specialist identities against the freeze. Every case must
reference a distinct run directory and replay-verified run identity. The
evaluator records matching before/after tree digests for each source run, invokes
no model, and writes outside the source run directories. Before creating its
output directory, it stream-scans every recorded run for that case's hidden
canaries and every other hidden canary in the sealed corpus. Missing cases,
reused runs, post-hoc configuration changes,
hidden-label leakage, replay failure, self-grading, critical proof violations,
or insufficient baseline improvement fail closed.
