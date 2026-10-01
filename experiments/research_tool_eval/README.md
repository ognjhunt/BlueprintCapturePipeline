# Blueprint public research tool comparison

Offline setup for the user-requested 20-case comparison. No provider evaluation
has run. The CLI cannot make network/model calls, and both injected boundaries
must be offline. No live agent, credentials, account, subscription, production
service or schedule was changed. This is one small stdlib harness, two grading
utilities and frozen data, not another agent platform.

The isolated branch is `experiment/blueprint-search-eval-20260930`, based on
`1f771e61` (pipeline PR #2485). It does not modify runner2486 or snapshot2487.
The existing Sol researcher and its Default credential/launcher are absent from
this replacement VM. No connection or runtime access is assumed.

This user-authorized, reversible experiment is outside the active ADP engineering
backlog. It does not claim to unblock an ADP gate or widen the pipeline program.
The applicable local overview/evidence/capability skills informed boundaries;
public vendor claims remain discovery/pre-screen evidence, not qualification.

## Run offline

From repository root, Python 3.10+ with no additional packages:

```sh
python experiments/research_tool_eval/harness.py validate --require-caches
python -m unittest discover -s experiments/research_tool_eval -p 'test_*.py' -v
python experiments/research_tool_eval/harness.py estimate --deepen
python experiments/research_tool_eval/harness.py mock --deepen --out /tmp/blueprint20-new-run
python experiments/research_tool_eval/review.py prepare /tmp/blueprint20-new-run/results.json /tmp/blueprint20-review
```

`validate` without `--require-caches` also works in a fresh checkout. It freezes
the portable expected facts and reports missing full-read caches explicitly.
The local ignored `source_cache/` and `docs_cache/` contain the direct-read
snapshots for parent inspection. Changed cache bytes fail validation and harness
initialization. Full-read caches must pass `--require-caches` before paid-run
source signoff; a missing snapshot requires independent rechecking/refreezing,
not silently treating a manifest as a retrieved page. Full third-party pages
are intentionally not redistributed in this PR.

The journal persists exact request bodies, results, IDs, states, usage/costs,
latency, attempts and polls. `results.json` adds the case, mode/API version,
controller/snapshot, prompt hash, evidence sources and pending grading fields.
Model/API calls are mocks. `$0 actual` is distinct from the simulated tariff.
Mock answers deliberately contain no expected facts and cannot generate a
provider-quality ranking.

## Frozen cases and independent evidence

`cases.json` is the only eligible external input. Selection used the existing
Notion Knowledge/Team Directory, then 23 primary public pages were separately
retrieved/read on 2026-09-30 **before any provider/model evaluation**. Rubrics are
hand-authored independently of provider results. `rubric.json`, source dates,
short quote anchors, full-read hashes and document hashes are frozen by
`freeze.json`. Publication/load/source-check dates remain distinct.

These are real company/public task contexts. Concrete site descriptions are
explicitly illustrative fixtures, not real prospects, site measurements or CRM
records. There are no private contacts, personal records, CRM exports or private
Notion document bodies in the provider inputs. Only public company/task data
and generic requested fields may be disclosed.

| Case | Team | Test |
| --- | --- | --- |
| 01 | Chef Robotics | Discover tray-portioning team and bounded production evidence |
| 02 | Chef Robotics | C-001748 module identifier and utility requirements |
| 03 | Weave | California availability versus unknown Austin eligibility |
| 04* | Weave | Sheet/FAQ conflict and specialist teleoperation |
| 05 | Weave | Isaac 0 operation versus Isaac 1 delivery roadmap |
| 06 | Agility | Named GXO/SPANX tote-transfer workflow |
| 07* | Agility | Fourth-generation workcell restrictions versus 2027 roadmap |
| 08* | Laundry Robotics | Robin folder feeding, dealer procurement and US unknowns |
| 09 | Dexterity | FedEx Hagerstown production expansion in July 2026 |
| 10 | Ambi Robotics | AmbiStack task versus historical AmbiSort/OSM evidence |
| 11 | GrayMatter | High-mix sanding cell and measurable acceptance gates |
| 12 | RightHand | RightPick One/4 versus stale RightPick 3 assumptions |
| 13* | Pickle Robot | Randa body/outer date conflict and deployment-specific limits |
| 14 | Bear Robotics | Indoor hospitality transport and outdoor/network gates |
| 15 | Avidbots | Historical Neo airport story versus current product/support claims |
| 16* | Universal Robots | UR12e-1300 ratings/controller versus historical UR10e |
| 17 | Universal Robots | Engineered WST/PCC CNC fixtures/grippers/changeovers |
| 18 | Intrinsic | September 22 Core/OMTS open release versus Flowstate scope |
| 19 | NVIDIA | GR00T N1.7 G1 tutorial versus representative success/Franka support |
| 20* | World Labs | Pending AMD transaction and generated unseen geometry |

`*` is the six-case hard subset, chosen before evaluation. It cannot change
after observing provider output without declaring a new exploratory protocol.
Source facts are a dated benchmark, not a claim that the live web is frozen.
Newer valid evidence must be adjudicated as a separately recorded source update;
do not silently move the answer key to favor a system.

## Exact experiment choices

| Arm | Request | Price assumption | Cells |
| --- | --- | --- | ---: |
| Parallel Search Fast | `POST https://api.parallel.ai/v1/search`, `mode=fast` | $0.001/request, ≤10 results | 20 |
| Parallel Search Advanced | Same endpoint, `mode=advanced` | $0.005/request, ≤10 results | 20 |
| Perplexity Search Fast | `POST https://api.perplexity.ai/search`, `search_type=fast` | $0.001/successful request | 20 |
| Perplexity Search standard | Same endpoint, `search_type=web` | $0.005/successful request | 20 |
| Optional Parallel Task Core | `POST /v1/tasks/runs`, `processor=core`, JSON output schema | $0.025/completed run | 6 |
| Optional Parallel Task Pro | Same endpoint, `processor=pro` | $0.10/completed run | 6 |

Endpoints, request schemas and tariff assumptions were checked against current
[Parallel Search](https://docs.parallel.ai/api-reference/search/search),
[Task](https://docs.parallel.ai/api-reference/tasks/create-task-run),
[Parallel pricing](https://docs.parallel.ai/getting-started/pricing),
[Perplexity Search](https://docs.perplexity.ai/api-reference/search-post) and
[Perplexity pricing](https://docs.perplexity.ai/docs/getting-started/pricing).
Provider backend versions are managed by providers and cannot be claimed to be
immutable model snapshots; run time, API path, mode and doc snapshot hashes are
recorded instead. No deprecated Parallel Search beta path is used.

Raw retrieval uses the same two frozen queries per case in **one** request per
arm, ten results, a common 16,000-character evidence envelope, one Sol answer,
no follow-up search/fetch, the same prompt and 8,000 input / 1,600 total output
token limits including reasoning. Perplexity requests 4,000 content tokens,
800/page; Parallel uses the documented character cap. These are the closest
API equivalents, not a guarantee that native tokenization/excerpt selection is
identical. Sol sees normalized passages with no mode/cost metadata. Arm order
rotates evenly across cases; failed/empty/truncated cells stay in denominators.

Sol remains `gpt-6.1-sol` through the **existing Agents API researcher**. Parent
integration must pin the actual model/snapshot, service tier and API contract;
enforce tokenizer admission, total output/reasoning tokens, timeout and resource
cleanup; and return authoritative usage. The harness only defines that boundary
and tests it with a mock. It does not replace the researcher with Responses API
or launch another hosted sandbox.

Core/Pro are separately labeled `delegated_research_system` results with their
own provider research and a Sol synthesis. Their internal tool budgets are not
controlled by the raw-search budget. Compare them on the predefined six-case
subset only; do not pool them with raw results, compare six versus twenty as an
overall winner, or call raw-search quality a deep-research result.

Perplexity Agent API is **not included** in this paid-run proposal. Adding it
requires another preregistered system arm with a pinned model, background
`POST /v1/agent`, explicit step/output/reasoning/tool bounds and all model plus
search/fetch costs. Its standard search tool price is $0.0025/invocation,
Fast $0.001 and fetch $0.0005, separately from inference. Do not reuse raw Search
pricing for it. Current [migration guidance](https://docs.perplexity.ai/docs/agent-api/migrate-from-sonar/overview)
states legacy Sonar support ended September 27, 2026 and old async is unsupported.

## Evidence grading

The requested fields and expected fact IDs are fixed per case. Independent
reviewers receive `blinded_review.json`; the parent keeps its mode mapping away
from them. `review.py apply` requires all fact/claim/field labels, including
failed cells, and rejects changed denominators, mapping and benchmark hashes.

Grade each expected fact as supported (1), partial (0.5), omitted/wrong (0).
Coverage is credited facts / frozen facts. Mark every factual output claim for
semantic support, correct citation, supporting quote and source freshness.
An exact quote match only screens passage support; it cannot establish semantic
entailment, public availability, independence or present operation. Reviewers
must inspect the source and version/date context. Explicitly distinguish
vendor, customer, research, roadmap and measured evidence.

An accepted field is useful, correctly versioned, evidence-backed and within its
claim ceiling; a correct `null` with an explicit unknown may be accepted for an
unverifiable field. Unsupported safety, geography, deployment, partnership,
availability or physical-truth assertions are critical errors. Review omissions,
unknowns, conflicts and stale updates explicitly. No provider grades itself.

Report per-arm field acceptance/coverage, unsupported claims, citation/quote
support, critical errors, freshness, latency median/p95, input/output/reasoning
tokens, provider/model costs and unknowns. Compare raw arms **within each case**;
report paired wins/losses/ties and failure counts with all 20 cases. With this
small purposive sample, results guide Blueprint selection, not broad market
quality claims. Core/Pro reports have a six-case denominator and separate label.
Calculate cost per useful accepted field only when the cost basis is complete;
otherwise show known cost plus unresolved components, not an invented total.

## Dispatch, recovery and stop behavior

SQLite WAL/FULL transactions reserve cost before dispatch, then atomically
claim work. The claim rechecks current spending and cancellation. Identity
binds frozen inputs, prompt, model/snapshot, tariffs and harness code hash.
Reservations remain until authoritative billing reconciliation; an error is
not an automatic refund. Known actual cost above the reservation stops future
dispatch, including already-reserved work. Unknown cost is retained as unknown.

Synchronous POST timeouts, lost responses, oversize results and crashes after
intent are uncertain work. They are not resubmitted. Retry only with a parent
transport's proof of rejection/nonacceptance, at most two attempts. No invented
provider idempotency header is used. Known Task IDs resume via bounded GET, not
another POST. Polling is at least 30 seconds apart, at most 24 observations,
with a 12-minute observation deadline; long work is reported back between polls.

Cancellation stops new dispatches durably and permits already-accepted Task
GET reconciliation. Parallel Task cancellation is not documented in the current
indexed Task API, so this adapter does **not** invent a cancel endpoint or claim
remote work stopped. Timeout/cancellation may leave chargeable provider work;
the parent must resolve provider IDs and billing. A disconnected observer or
cancelled coroutine likewise does not prove the existing Sol turn stopped.

## Cost and access

See [COST_ACCESS.md](COST_ACCESS.md) for the exact one-time estimate, proposed
budget, secure routes and parent checklist. Parent review of the frozen cases,
source facts and harness precedes any paidrun. The illustrative approval JSON
does not grant authority; live transport remains hard-disabled until separately
reviewed canonical paid admission and secure controller integration exist.
