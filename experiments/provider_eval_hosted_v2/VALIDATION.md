# Offline validation

Focused command: `PYTHONPATH=src .venv/bin/python -m pytest experiments/provider_eval_hosted_v2 tests/test_agent_execution_sessions.py -q`.
Result: **106 passed**, comprising 30 hosted cohort instances, 40 soft-pilot cases,
and 36 existing durable-runtime tests. Ruff and `git diff --check` passed. Independent GPT-6.1 Sol
review reran these checks and reported no remaining must-fix findings.

All network responses, keys and admissions used in tests are synthetic; module-
relative key mocks prevent reading actual bindings under either pytest import
path. No provider/model/hosted-session call was made and no live journal copied,
reset, released, or modified by these checks.

Verified boundaries include:

- Actual wire-shaped managed session/function-tool flow with full initial files,
  source text beyond 20,000/25,000 characters, arbitrary-offset inspection, an
  agent-directed follow-up, durable tool-result replay after disconnect, and
  final turn validation. Initial files remain unchanged across restarts.
- Exact Sol/Default project/file binding, disabled hosted networking both in the
  request and returned environment, disabled delegation/hidden native search,
  and refusal of permanent session deletion.
- Ordinary `Scan&Sand` names under all four native search contracts, warning
  retention, equal search opportunities and separate provider-arm evidence.
- Read-only import from original completed v1 request/envelope hashes without
  double-counting reservation or pooling other arms' sources.
- Source identity and canonical manifest validation; changed cached manifests,
  input intents, raw sidecars and result source IDs are refused. A coherently
  altered raw/result/source set cannot override the independently anchored
  completed paid response. Replay needs no key lookup or new HTTP call.
- Canonical aggregate paths: retargeting a directory alias cannot switch to a
  fresh ledger or dispatch the same completed attempt again.
- A fake timeout after reservation remains unsettled in the actual durable
  runtime, with one mock dispatch and its full reservation retained. It cannot
  be reported as a refusal before dispatch or automatically retried.
- Public DNS pinning, no authorization/cookies or redirects on source retrieval,
  refusal of nonpublic destinations, whole accepted text body retention, and
  explicit unsupported-content gaps.
- Shared $10 exposure accounting, exact supplied $3.934105 checkpoint math,
  non-hard-cap target labeling, and a default zero-call paid admission guard.
- Executable one-case/four-arm soft pilot with actual managed wire-shaped runtime;
  deterministic task replay, immutable approval/root-prefix binding, no duplicate
  sessions, fresh unknown/high usage stopping before tools/model replies, bounded
  five-minute cancellation, and overrun reporting beyond the nominal ledger cap.
- Durable creation/usage anchors prevent missing or changed sidecars from hiding
  known sessions or resetting retained-container carry costs across restarts.
- Actual OpenAI SDK 3.22.1 read-only GET wire, Beta and Default project headers,
  using a mock HTTP transport; compatible Agents SDK 0.22.3 installed from PyPI.
- Canonical paid-model grant refusal stops session creation before HTTP; the same
  chokepoint applies to model resumption, separately from native search grants.
- Explicit disabled networking accepts only the absent/empty allowed-domain list
  default; unknown policy keys, malformed/nonempty domains and enabled/restricted
  access remain refused. A wire-shaped create response binds successfully.
- GET-only reconciliation of the original used/stopped receipt preserves failure,
  raw evidence, old journal prefix and holds; validates exact task/session/creation
  proof before binding; refuses mutations/other sessions and never admits the old
  receipt for patched paid execution. The retained 49,484/261 token checkpoint
  reproduces $4.765920/$0.905315. Idle without a terminal root turn, any earlier
  active root, or outstanding required action keeps cancellation unresolved.
- Offline exact-approved-deletion receipt adoption validates the raw hash,
  approval/intent/DELETE acknowledgement/session-and-environment absence, retained
  task/network/cancelled turn/creation and original ledger/reserves. No key or
  HTTP route is used. The historical creation-unresolved state is preserved;
  deletion anchors stop local container-age growth without releasing holds or
  claiming final teardown/billing. Tampered/incomplete evidence is refused.

Remaining limitations are documented in README: hosted inference/write access
is not verified by read-only saved-agent access; managed hidden-loop usage lacks
a documented hard cap; final billing/count/extras remain unreconciled. The user
has explicitly approved the one-case $1 soft pilot with its separate receipt;
the general hard-cap CLI remains blocked. Independent output grading is not run;
PDF/redirect/binary inspection and direct artifact-content downloading are absent.
The SDK pair is an experiment override of the unchanged project metadata's older
Agents SDK constraint, not a production dependency update. The new cohort is not
a completed provider comparison or a quality score; no paid calls ran here.
