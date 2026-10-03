# Lead evidence verification

Owner-requested scope, 2026-10-03: reusable Sites-First Research verification.
This supports the ADP-010 partner-discovery boundary ahead of the day-14 protocol
freeze; it does not satisfy partner admission, task-owner truth or rights.
Observed gap: QA's blanket `source_support_verified` and accepted-key list could
promote a plausible site with no recorded human-workflow assessment. The smallest
reversible repair adds an auditable source-assessment gate to the existing QA and
review transition. No provider integration, control, credential, spend or sending
activation is introduced.

`tools/daily_research/verification.py` is pure, provider-neutral code. The
qualification skill describes agent judgments; deterministic code enforces the
bound transition. It does not fetch pages or assess source meaning by keywords.
The authenticated reviewer/QA artifact owns the source and scope judgment.

| Dimension | Required evidence for promotion |
| --- | --- |
| operator | Named operator, source-owned identity and affiliation |
| physical_site | Actual selected operating site/location; headquarters or service area is insufficient |
| site_task | Relevant task explicitly linked to this site |
| human_workflow | Documented people performing the relevant physical workflow; general business capability is insufficient |
| plausible_fit | Source-backed fact or bounded inference, with material limitations |
| counterevidence | Actual bounded automation/contradiction search and resolution; no claim of exhaustive absence |

Each factual claim requires current retrieved primary/operator support and a
reason explaining the exact operator/site/task relationship. Sources retain URL,
publisher, publication/event date (null when unknown), actual retrieval time and
method, supporting excerpt, classification, freshness and rationale. A recent
retrieval does not by itself refresh historical operations. Sources are untrusted
data and never grant tool or commercial authority.

Claim states are `verified_fact`, `inference`, `unresolved`, `contradicted`,
`stale`, `unreachable`. Overall outcomes are `verified`, `unresolved`, `rejected`.
Missing, stale, inaccessible and ambiguous evidence stays unresolved, with
actionable reasons. Rejection requires supported contradiction. Every outcome
retains the raw assessment and its digest. Extra inert metadata is retained.
No extra research quota, provider start or perpetual retry is introduced.

QA puts a raw `blueprint.lead-verification.v1` assessment in each check's
`lead_verification`. It binds the exact supplied `candidate_digest`,
`assessed_at`, `valid_until`, `claims`, `sources` and `counterevidence`. Missing
assessments are normal unresolved outcomes, not formatting corrections. The
agent's `source_support_verified` boolean cannot pass the new gate by itself.

`qa_decision` evaluates the complete packet and stores `review.lead_verification`
with all results and metrics. It filters promotion to verified nonduplicates.
`Runner.review` recomputes the assessment and derived-result binding before
accepting each selected candidate, including its current validity. Raw discovery,
original artifacts, rejected/unresolved assessments and historical reviews remain
available. An old completed publication is not silently upgraded to verification.
WebApp recomputes this same evidence gate before downstream outreach eligibility;
its existing sending, contact, consent, approval, suppression and budget gates
remain separate.

The fixed-destination publisher rechecks the protected result, exact candidate
and raw assessment digests and current validity before a new write. GET-only
reconciliation validates at the retained review time so an expired assessment
does not prevent recovery of an already completed write or cause a duplicate
effect. Historical readback does not refresh evidence or grant another write.

Uniqueness is normalized operator + physical location + task. Site-label aliases
cannot create another qualified candidate; different locations or tasks remain
distinct. Conflicting duplicate assessments remain unresolved before promotion.
Existing semantic QA/CRM duplicate checks remain necessary for aliases beyond
normalization.

`cohort(candidates, assessments, now, actual_cost_usd=None, duplicate_checks=None)` evaluates every
candidate, including conditional entries. It reports verified unique site/task
candidates, unresolved/rejected counts, duplicates and assessed coverage. Missing
assessments are in the denominator. `compare(runs, now)` never automatically
declares an accuracy winner: specificity and six selected examples do not justify
whole-report accuracy claims. Cost comparison additionally requires known actual
costs covering the same discovery-and-verification scope. Estimates and partial
provider charges cannot fill missing invoices.

New packets retain all raw rows in `candidates` plus `duplicates`, ordered by
`discovery_index` and marked with `verification_cohort_version`. QA must assess
all of them. Semantic alias chains require bound original keys and nonblank
equivalence reasons at every link. Missing originals, cycles and conflicting
assessments remain unresolved; aliases never create extra verified yield.

Verification digests use sorted ASCII JSON-style keys and values, with finite
numbers encoded as unquoted `n:` plus big-endian IEEE754 bytes. This preserves
ordinary inert float metadata across Python and JavaScript without guessing JSON
float lexemes. Large IDs belong in strings. Existing raw packet/artifact hashes
are unchanged. The synthetic cross-repo fixture is
`tests/fixtures/daily_research/lead-verification.json`; it proves software contract
parity only, never a real lead's facts.

Buying intent, consent/rights, commercial qualification, robot compatibility and
deployment readiness remain explicitly separate gates. A verified public human
task is a research lead; it is not an interested buyer, a cleared capture, a robot
match or a ready deployment.

Release: package `verification.py` through the existing immutable standalone
archive and revise the two hash-bound instruction files as session overrides.
Keep the original saved-template inventory check; no template write is required.
Submit reviewed Pipeline and WebApp PRs to the existing sole release owner, with
focused and hosted CI evidence. Do not independently deploy or start paid
verification to prove this change.
