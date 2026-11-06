# Daily research → contact → CRM → draft operations audit (2026-10-04)

Public, sanitized record. Private run evidence (row blobs, QA artifacts, CRM rows, owner
directions, contact research) stays in company storage; this file cites only code, counts,
hashes and company-storage locations. Supports ADP-010 partner discovery (partner-phase
day-7 gate). Observed 2026-10-04 13:33Z onward (08:33 America/Chicago).

Status vocabulary: implemented · locally tested · independently reviewed · CI passed ·
merged · deployed · live readback verified · business workflow proven.

## 1. Answer

Blueprint does not yet reliably turn broad research into evidence-backed prospects,
contacts and reviewable drafts. On October 4 the process itself was sound (one admitted run,
fenced claims, no duplicate charges, nothing sent, cleanup completed), but four contract
defects stopped every candidate before the CRM, and a fifth would stop contacts:

1. The lead-verification gate rejected every assessment on a field-name mismatch and an
   honest unknown expiry, with one generic reason and no claim evaluated.
2. The output contract forced 22 structured findings into 20 strings and dropped one
   discovery; nothing downstream reads findings or unresolved candidates.
3. Reviewed research about an existing CRM row was refused as a duplicate, with no
   identity-preserving refresh.
4. The Exa tool advertised a cap range the host does not accept, and no producer exists for
   the all-in allocation that paid expansion requires.
5. The static contact-page reader treated any page with a script or stylesheet as unverifiable,
   so it would have rejected nearly every real operator page.

This batch repairs all five (the owner chose element-level visibility for 5) and ships them as
one coupled release, built for the first time from Pipeline `main`.

## 2. Architecture and ownership (as observed)

| Component | Identity / location | Trigger | Inputs → outputs | Owner |
|---|---|---|---|---|
| Daily research clock + child | Render worker `blueprint-webapp-worker`; vendored Pipeline archive `vendor/daily-research` → `dist/daily-research/release` | 07:00 America/Chicago (Python `render.py` scheduler); minute idle ticks | control doc, CRM snapshot, knowledge → run row, files, review packet, work item | Batch release owner (this release); research code on Pipeline `main` after #2549 |
| Research agent | Saved OpenAI agent (pinned instruction SHA in control), hosted environment template | one root turn per date; same-session repair/QA/publication turns | Perplexity search + static source reader + history tools; optional guarded Exa | — |
| Canonical state | Firestore `blueprintDailyResearch/sites-first` (control, runs, files/blobs, workItems, contactResearchRuns) and `blueprintCommunications/default` (directions, reviewedResearch, refreshRequests, contactResearchRequests, jobs, gmailDraftBindings, sendReceipts) | — | — | Canonical |
| Review/export | Google Sheets CRM `Prospects` (append-only publisher; manual imports) | publication | accepted candidates → 19-column rows | Projection only |
| Human report | Notion knowledge page children | publication | brief → report page | Projection only |
| Communications | Render worker communications loop (60 s) | completed work items | intake → contact gap / job → draft → Gmail DRAFT copy | Draft-only; send disabled |
| Owner directions | Company GCS `operations/recovery/2026-10-02/...` | — | $10/day (America/Chicago; $5 research / $5 communications; expires 2026-10-10T00:00Z); Gmail draft copy into the founder's verified mailbox binding, draft-only | Owner |

Source parity (verified): each vendored archive is byte-identical (53–54 files) to its pinned
Pipeline commit; PR-head trees equal their integration-branch merge trees; the live
October 4 run executed WebApp `e8bd5273` with Pipeline `b375c8fd`. The control document pins
`source_commit`; the worker refuses to create work while enabled if the installed manifest
differs, so a deploy and its re-pin are coupled.

## 3. Run-to-outcome trace (October 4, primary bytes)

| Stage | Count |
|---|---|
| Research searches | 20 (all succeeded) |
| Research source reads | 25 (21 succeeded, 4 failed: 3 unavailable, 1 HTTP failure) |
| Company-history lookups | 9 |
| Exa expansion starts | 1 attempted, skipped before any provider call (`expansion_remaining_all_in_allocation_unverified`; requested cap $0.50) |
| QA source reads | 4 |
| Formal candidates | 4 (0 duplicates) |
| Structured findings | 22 written (13 discovery-type, 17 with a next-evidence question) → 20 strings after repair; 1 distinct-site discovery dropped |
| Repair turns | 2 (summary-list type/count, coverage enum, next-action length) |
| Lead verification | 0/4 assessed (generic binding failure) → 4 unresolved |
| Accepted / CRM rows | 0 / 0 (Notion report published; zero-row Sheets result) |
| Communications | 1 orphan `research_owner_refresh` request; 0 contact tasks, 0 jobs, 0 drafts, 0 Gmail copies, 0 sends |
| Usage | 5.40M input tokens (5.20M cached), 25.3K output; search estimate $0.02; billing `unknown_pending_billing_reconciliation` |
| Cleanup | session deleted; 90 objects archived; `cleanup_required=false` |

Offline replay of the four retained assessments: original gate 0/4 assessed; the repaired v2
gate 4/4 assessed, 0 verified, 4 unresolved, each with located claim-level reasons (human
workflow at the exact site unconfirmed in all four). Format repair alone does not create an
eligible prospect.

## 4. Defect register

(see the companion table in the PR description and private receipts; summarized)

| ID | Sev | Status | Defect | Repair |
|---|---|---|---|---|
| D1 | P1 | fixed in release | Verification contract drift; catch-all reason; no repair route | Versioned v2 diagnostics; lossless alias; honest null expiry; exact QA contract; placeholder correction |
| D2 | P1 | fixed in release | Discovery loss at the output contract | `discovery_inventory` paged retention; byte ceilings; packet-budget feedback |
| D3 | P1 | fixed (production pinned to a `main` commit) | Production ran an unmerged integration branch; Pipeline `main` lacked the live research code; packaging a stale head rolls back live commits | Lineage enforced at packaging; integration branch merged to `main` (#2549); the package is built from `main` |
| D4 | P1 | fixed in release | Contact page visibility rule rejected real pages | Owner decision: element-level visibility on browser-equivalent parsing; every proof records `static_text_css_not_rendered` |
| D5 | P1 | fixed in release | Duplicate refusal for existing CRM rows | `refresh` binding (BP id + row digest); operator admission pending |
| D6 | P1 | open | Deploy SIGTERM cancels in-flight work; late QA artifacts dropped; some blocked states retry forever | Hardening backlog |
| D7 | P2 | owner decision | No producer for the all-in allocation (Exa, FindAll unreachable) | Needs owner policy on bounded estimates vs verified usage |
| D8 | P2 | fixed in release | Exa advertised cap ≠ enforced; conflated refusals; codes masked | Advertised = enforced; distinct codes; no authenticated context for a below-minimum start |
| D9 | P2 | fixed earlier (#2580) | Generic tool-error masking | — |
| D10 | P1 | open | Reply learning gets no input in draft-only mode | Read-only sent-draft observer (scope check required) |
| D11 | P2 | open | Research authority reference unverified and unexpiring | Bind to the content-addressed owner direction |
| D12 | P2 | open | Learning binding frozen to the first 11 CRM rows | Rebind on CRM growth |
| D13 | P2 | logged | Per-tick contact-research reconciliation reads grow with history | Indexed queries |
| D14 | P2 | open | Billing reconciler over-triggered by systemd `Wants=`/PathChanged | Unit decoupling |
| D16 | P2 | open | `research_owner_refresh` requests have no consumer | Consume or stop writing |
| D17 | P2 | fixed | Parallel cohort qualification/CRM-import evidence lived only in a local agent workspace | Copied verbatim to company storage with a sha256 manifest; readback verified |
| D18 | P2 | fixed in release | v3 outputs could exceed the review packet after validation (no repair route) | Packet-budget repair feedback |

Disproved: duplicate paid creates (every create holds a durable one-use claim before POST);
next-day refusal from October 4 cleanup; AWS Cost Explorer polling (no caller).

New during the batch (caught before release, not production defects):

| ID | Sev | Status | Defect | Repair |
|---|---|---|---|---|
| D19 | P2 | open (mitigated) | An in-flight Exa row is compared against the current advertised tool schema, so a deploy that changes the schema could strand it | Mitigated: no active rows at deploy and the deploy is outside the run window. Structural fix: compare against the row's frozen create-payload tools |
| D20 | P1 | fixed before release | The first element-level contact parser (a hand-written tolerant tree builder) released text browsers do not render: stray closing tags, nested or unclosed containers, unterminated quotes and comments. Independent review reproduced approved proofs end to end | Replaced with browser-equivalent (WHATWG) tree construction and bounded static hiding rules. Four further review rounds closed declarative shadow DOM, off-screen offsets, worker-crash inputs and wrong-recipient cases (truncated or reordered addresses, inline-laid-out blocks). The parser is frozen at the owner's rule |
| D21 | P1 | fixed before release | Deferring assessment-only QA defects after exhausted corrections also deferred copied example placeholders, so template text could be verified and published to the CRM; near-copies (case, spacing, punctuation) passed even exact-string detection | Placeholders always block QA, compared in normalized form (#2585, #2586) |
| D22 | P2 | fixed before release | A whole-inventory quarantine receipt was uncapped and attached after the packet ceiling check | Bounded receipt; exclusions join the packet before the check (#2585) |
| D23 | P3 | open | A packet-ceiling refusal leaves one orphan content-addressed inventory page and repeats on every tick | Write inventory pages after the ceiling check; stop retrying a deterministic refusal |
| D24 | P2 | open (design) | The contact proof re-verifies from raw HTML, so Blueprint maintains a static visibility parser | Agent rendered-retrieval evidence with a thin deterministic gate (follow-up design) |

## 5. Repairs and release evidence

| Change | Repo / PR | Review | CI | Merged | Deployed / live |
|---|---|---|---|---|---|
| Versioned lead verification, discovery inventory, packet guard, Exa cap contract, review fixes | Pipeline #2584 → integration branch | independent review (14 findings) → fixes → re-review | Impacted, research lifecycle 3.11/3.12/3.14 | `70d6f537` (tree = reviewed head `dd8b00e8`) | via WebApp package |
| Integration branch → `main` (closes the unmerged-production lineage) | Pipeline #2549 | conflict audit: main's only research change (#2567) is contained; research tree unchanged | full suite 4/4 shards on the head | `b0df68f0` | package source |
| Placeholder and receipt fixes found by re-review | Pipeline #2585, #2586 | re-review verified no deferral path remains | Impacted, research lifecycle 3.11/3.12/3.14 | `aae6c5ad`, `0fea06f1` | in package |
| Contact recovery (#846), versioned TS verification, element-level visibility, package | WebApp #848 | four review rounds (blockers closed each round); parser frozen at the owner's rule | build, check, test, e2e, rules-emulator | `199c77db` | web + worker live at `199c77db` (18:44–18:47Z), `/version.json` verified, health/ready 200 |
| Control re-pin | Firestore control `source_commit` | precondition-fenced single-field write | — | — | `42ca18ec` → `0fea06f1` at 18:48:29Z; readback verified; every other field unchanged |
| Evidence portability (D17) | company storage `operations/research/2026-10-03/parallel-findall-cohort/` | 9/9 files sha256-verified on readback | — | — | done |

## 6. Test matrix (summary)

| Area | Tests | Result |
|---|---|---|
| Lead verification v1 stability | stored v1 digests recomputed on real rows; shared 12-case Python/TS fixture | pass |
| v2 diagnostics | located reasons, alias conflicts, null expiry, basic UTC offsets, pinned result version | pass |
| QA correction loop | assessment-only defects after exhausted corrections do not block the day; feedback digested once and bounded | pass |
| Discovery inventory | paging, digests, forged manifests refuse, whole-field quarantine, packet-window regression | pass |
| Exa cap | advertised range = enforced; below-minimum start skipped before transport; distinct refusal codes | pass |
| Contact visibility | 23 browser-divergence probes, 9 end-to-end, 37 hiding styles + 10 controls, 14 embedded-rule cases, ambiguity guard, 16 adversarial 384 KiB timing cases, parser pin | pass (74 fail on the first implementation) |
| WebApp full suite | 6,140 tests in 675 files | pass |

## 7. Owner decisions and what remains

Decided 2026-10-04:
- **Paid expansion (D7):** Exa and FindAll may spend up to $10 combined per run. The amount is one owner-set value, changeable up or down by one command without a deploy, with a code ceiling and an audit trail. Exa lands first; FindAll needs its own blockers cleared (claim pinning, deadline cancel, grant issuer, key passthrough).
- **Reply learning (D10):** may read founder-sent drafts. The existing founder-mailbox binding already holds read access; a default-off, read-only observer is in implementation.
- **BP-000015/16:** refresh admissions are prepared from re-checked evidence; the owner submits them through the authenticated admin route.

Still open:
- FindAll activation also needs owner confirmation of the provider key on the worker and approval of what is disclosed to the provider.
- Engineering backlog: deploy handover (D6), frozen tool comparison (D19), learning rebinding (D12), billing trigger (D14), refresh-request consumer (D16), ceiling-refusal retry (D23), and the agent-evidence contact proof that would retire the static parser (D24).

## 8. Conclusion

**Worked (evidence-backed):**
- The October 4 run's process controls held: one admitted run, fenced claims, no duplicate paid calls, nothing sent, cleanup complete.
- The zero-prospect outcome was traced to contract defects in primary bytes. Under the repaired gate the four retained assessments are evaluated with located reasons; none verify, because human workflow at the exact site is unconfirmed. That is an honest result, not a format failure.
- Independent review caught, before release:
  - template text reaching the CRM;
  - a packet-ceiling bypass;
  - hidden-text and wrong-recipient contact cases;
  - worker-crash inputs.
- The release is live from Pipeline `main`, and the control is re-pinned with readback.

**Not yet proven:** no eligible prospect, contact or draft has come from the new code yet. The first live run is October 5 at 07:00 America/Chicago.

**Unverified:** billing for October 4 (`unknown_pending_billing_reconciliation`).

**Costs:**
- October 4 run: search estimate $0.02; model usage 5.40M input tokens (5.20M cached) and 25.3K output.
- This batch made no paid provider calls and no new spending commitments.
