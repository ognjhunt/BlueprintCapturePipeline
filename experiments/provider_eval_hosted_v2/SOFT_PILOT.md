# Authorized one-case hosted pilot

The user explicitly approved one real case across all four search modes with a
**$1 incremental soft target**, understanding hosted costs are not hard-capped.
Approval references are recorded in `soft_pilot.py`: assistant proposal
`Sentinel_51d1baf8e5ec8191ab1a53906ca66bf0`, user
`Sentinel_6c0c10271a64819181ead8587cfb6cb5: yes`.

Only execution task `01a0f3b8-6abe-775b-bfea-5102185b80ce` may run the pilot. The
implementation/review tasks make no paid calls. The protocol selects frozen
**BP-EVAL-01 (Chef Robotics)** and exactly Parallel Fast/Advanced and Perplexity
Fast/standard. No other case, model, project, saved agent or credential is enabled.

## Exact setup and commands

Fetch the reviewed remote head into the execution owner's isolated checkout.
Use its existing `.venv`; install the public, tested package pair:

```bash
.venv/bin/python -m pip --isolated install --index-url https://pypi.org/simple 'openai==3.22.1' 'openai-agents==0.22.3'
PYTHONPATH=src .venv/bin/python -m pytest experiments/provider_eval_hosted_v2 tests/test_agent_execution_sessions.py -q
.venv/bin/ruff check experiments/provider_eval_hosted_v2
```

The initial Agents SDK 0.19.1 requires OpenAI below 3, so upgrading only OpenAI is
not a compatible pair. These versions were installed from official PyPI and
tested together. The repository's production metadata still pins Agents SDK
below 0.20: this is a local experiment override and pip reports that constraint
conflict. No production requirements, deployments or researcher files are changed;
do not apply this experimental override to a production worker.

Run the read-only access check using existing bindings:

```bash
PYTHONPATH=src .venv/bin/python -m experiments.provider_eval_hosted_v2.soft_pilot --aggregate-root /workspace/provider-eval-private-live-20260930 --execution-owner-task-id 01a0f3b8-6abe-775b-bfea-5102185b80ce --check-access
```

This uses the actual SDK `client.beta.agents.sessions.list(limit=1)`, GET
`https://api.openai.com/v1/agents/sessions`, with Default project and `agents=v1`.
It does not create a session or prove inference/write permission. Required hosted
application-key scopes are `api.agents.read`, `api.agents.write` and
`api.responses.write`; Models is not called by this pilot. Existing binding
values never appear in output. An access error returns only safe HTTP status.
No new key or permission grant is requested; the authorized first session will
establish whether the existing binding supports hosted inference.

Prepare and inspect the explicit approval receipt, without HTTP:

```bash
PYTHONPATH=src .venv/bin/python -m experiments.provider_eval_hosted_v2.soft_pilot --aggregate-root /workspace/provider-eval-private-live-20260930 --execution-owner-task-id 01a0f3b8-6abe-775b-bfea-5102185b80ce --prepare-soft-pilot
```

The output supplies `soft_receipt_sha256`. Supply that exact value to the following
command in place of `ACTUAL_SHA_FROM_PREPARE`:

```bash
PYTHONPATH=src .venv/bin/python -m experiments.provider_eval_hosted_v2.soft_pilot --aggregate-root /workspace/provider-eval-private-live-20260930 --execution-owner-task-id 01a0f3b8-6abe-775b-bfea-5102185b80ce --soft-receipt-sha256 ACTUAL_SHA_FROM_PREPARE --execute
```

The same command resumes the same deterministic tasks and completed outputs; it
does not create a replacement session. Once stopped, further paid work is blocked.
Local carry-cost reporting remains available with `--status` replacing `--execute`.
It updates conservative upward holds but makes no HTTP call.

## Admission, monitoring and stop behavior

The receipt binds the exact public case/input hashes, four arms, owner/catalog,
Sol/Default, canonical journal root, original scope/access hashes, existing
journal prefix/exposure, clean reviewed source commit/code, SDK versions, $1 soft
target, $0.80 stop threshold, five-minute arm deadline and 48 function-call limit.
Its expiry is one hour. Missing/empty prior journals or loss of the known
$3.934105 prior hold checkpoint are refused. A changed receipt cannot absorb
hosted spend into a new baseline or authorize another four sessions.

The main $10 journal is preserved. Before any session, the pilot retains
**$0.635480** for four model targets ($0.22), four 60-minute container carry
allowances ($0.36), and isolated output-review/count allowances ($0.05548).
Maximum remaining native searches/extras add $0.0735, giving **$0.708980** projected
incremental exposure, or **$4.643085** with the supplied prior reserve. These are
reserves, not an actual bill. The earlier $0.468980 planning target assumed one
container period; the executable monitor holds three periods because cleanup
has not been authorized.

Each fresh session GET is observed before function dispatch or a tool-result
reply can resume model work. Unknown fresh usage stops the pilot; it is never
zero. Known cumulative usage is estimated using maximum long-context cache-write
and output rates, with high-water amounts and upward reservations. Cached-token
discounts and earlier allowances are not released. An observed overrun is still
recorded/reported if the nominal $10 ledger cannot admit an additional reserve.
No tool result or next arm is dispatched after a soft stop, unresolved call,
creation uncertainty or cancellation/deadline condition.

Only one root session is active at a time. The application polls at five-second
intervals. On stop it persists cancellation intent and makes at most three
bounded reconciliation polls. HTTP request timeouts and ongoing provider work
mean the application deadline and soft target are not hard time/spend caps.
Cancellation uncertainty remains explicit; no absence or lost stream is treated
as proof of nonacceptance. No automatic model/search retries are introduced.

Creation and observation proofs are anchored in the existing durable AgentJournal.
Changing/deleting a start sidecar cannot hide a bound task or reset its container
age. Completed/cancelled sessions continue contributing carry estimates from
their original start times, including across CLI restarts. No timeout is assumed
to guarantee sandbox expiry.

## Outputs and cleanup

Artifacts are under the aggregate root's
`protocols/hosted_agent_research_v2/soft_pilot/`: approval is one level above,
`agent_journal/` retains task/session/turn state, per-task directories retain
creation/usage proofs, and `reports/` retain final answers/status/cost snapshots.
The common evidence files remain in the separate case/arm directories.
Every progress/final report includes session/environment IDs and cleanup state.

**No session is permanently deleted by this pilot.** Completed or cancelled
containers may keep accruing costs; cancellation is not cleanup. Reports mark
them `retained_pending_exact_session_deletion_approval` and expose elapsed carry
estimates. After the pilot, the parent must promptly preserve outputs/IDs, report
the ongoing charge risk and obtain action-time approval for the exact sessions
before deletion. Until then, `--status` can refresh local carry accounting without
launching more work; there is no hidden background schedule or keepalive loop.

The controller is not its own ground-truth reviewer. Independent output review
uses the parent-isolated oracle and the reserved review allowance; this CLI does
not send the oracle to a controller or execute that later review. No winner is
claimed before source/unknown/citation grading and billing reconciliation.

## Stopped first-session read-only reconciliation

The first accepted session returned `network: {access: "disabled",
allowed_domains: []}`. The old exact-dictionary validator rejected this harmless
empty API default after session creation. Creation includes the initial input,
so the hosted agent can start before the create response is validated locally.
Binding occurred after validation, leaving `creation_unresolved` locally while
the usage observation retained the known session ID. This is a local validator
failure, not proof that the create was unaccepted or safe to retry.

The repair accepts only explicit `disabled`, with `allowed_domains` omitted or
an empty list. Enabled/restricted/missing access, nonempty domains, malformed
defaults and unknown policy keys remain refused. No production runtime is edited.

For the already stopped cohort, the execution owner may run this GET-only command
from the patched reviewed checkout, substituting the **original** approval SHA:

```bash
PYTHONPATH=src .venv/bin/python -m experiments.provider_eval_hosted_v2.soft_pilot --aggregate-root /workspace/provider-eval-private-live-20260930 --execution-owner-task-id 01a0f3b8-6abe-775b-bfea-5102185b80ce --soft-receipt-sha256 ORIGINAL_APPROVAL_SHA --reconcile-session sess_0efe74876b1df856006abdb27f210881979445fa9da044cf87
```

This narrow route accepts only the original `a3b583a93446d444a20b67279f71bf416802d28c`
code identity, still verifies the exact original receipt hash/root/journal prefix,
and requires an existing stop plus the retained creation/usage proof for the exact
session. It does not migrate the paid scope or change the approval. POST, DELETE,
other sessions and provider routes are blocked by the transport wrapper. Original
failure state, raw session, full turns/items and reconciliation reports are retained
under `soft_pilot/readonly_reconciliation/`, with hashes anchored in AgentJournal.

Only validated terminal root-turn evidence settles cancellation. Remote `idle`
alone leaves cancellation unresolved. Completed-after-stop output remains evidence,
never a successful benchmark answer. Repeated reads do not create another session,
repeat a search or release prior holds. Conservative usage/carry accounting may
increase; the supplied checkpoint reproduces $4.765920 reserved and $0.905315
projected, above the $0.80 stop threshold. The remaining three arms stay stopped.
No permanent deletion is performed; exact-session cleanup approval remains needed.

Primary references: [official SDK registry](https://pypi.org/project/openai/3.22.1/),
[compatible Agents SDK](https://pypi.org/project/openai-agents/0.22.3/),
[hosted lifetime and cleanup](https://developers.openai.com/api/docs/guides/agents-api/environments/openai-hosted),
[best-effort usage limitations](https://developers.openai.com/api/docs/guides/agents-api/observability).
