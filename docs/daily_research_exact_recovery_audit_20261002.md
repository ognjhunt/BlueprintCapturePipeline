# Daily research recovery audit, October 2, 2026

Scope: owner-requested daily research recovery and research quality (handoff
items 1 and 7). No schedule activation, new research/review inference, sends,
access changes, credential movement or external publication is authorized by
this audit. The existing worker and company stores remain the architecture.

## Verified source and evidence

Pipeline main was `73d3be8d6838f67f96ef2555e65eff63886f3cbb` at inspection.
PRs 2518, 2535 and 2536 were merged. PR 2529 remained open and introduced a
second 06:45 timer and terminal writer; its bridge/input portions are integrated
here with the existing WebApp native hooks as the sole aggregate/terminal owner.
No Python 06:45 timer or separate terminal-record writer is added.

The retained canary is `baseline-20261002-attempt-0001`, dated `2026-10-01`.
Its canonical namespace is:
`blueprintDailyResearch/sites-first/canaries/baseline-20261002-attempt-0001`.
Authenticated read-only Firestore reads verified the complete raw bytes,
chunk integrity, byte counts and createTime for these immutable blobs:

| Stage | SHA256 | Raw bytes | Authenticated createTime (UTC) |
| --- | --- | ---: | --- |
| Before cancellation | `810a291fa405c58d1c891c9cd7771a98ed4ae0873b0953796a7246e42ecbe9c9` | 257543 | 2026-10-02 09:39:23.342442 |
| Cancellation intent | `a2302b4408e575d21049cb09a6366d8ad867bf7f21a1f8402f66e60cc141e876` | 257542 | 2026-10-02 09:39:28.549726 |
| Cancellation reply | `6a0e9e8054df2c90240fbaa9d10a0bcf47e6c28c7e550ebcab291735061f00b6` | 257652 | 2026-10-02 09:39:29.639687 |
| Current retained row | `59327fce14de04a18679932162a4342ddd3513b6e123d43dbc85d2693e957b9b` | 257642 | 2026-10-02 09:39:39.204609 |

The first two records **omit** `cancel_reply_received` and
`cancel_idempotency_key`. The latter two record a true reply and the exact
`blueprint-researcher:2026-10-01:qa:retry-phase:cancel` key. The current row has
`agent_qa_terminal_guard_failed`; the preceding reply has
`canary_total_observation_deadline`. These authenticated record timestamps
bound cancellation ordering; they are not exact request timestamps.

The GCS bundle remains in the existing private bucket:
`gs://blueprint-8c1ca.appspot.com/research-backups/retained-sessions/sess_06ea8f997fa27202006abf0b37b9f4819aacfaa2cb1414eb14/2026-10-02/qa-diagnostics/d43dd996521940543e5324dbc0173eb9118cc24fc4081834da391519168998a5.json.gz`.
Generation `1790935283944135`, gzip 2010808 bytes, SHA256 equals the filename;
uncompressed 5015811 bytes, SHA256
`ae56a4ee7c45157c7940403599ef9eeb41209588d959a0c143ffca22cc005842`.
All 80 outer entries and 64 embedded-original entries were hash/length checked.
The four Firestore ordering blobs are separately canonical and must be retrieved
from their blob/chunk paths, not assumed to be entries in this bundle.

Original research artifact: 29190 bytes,
`011c09c6e5900e910c65c71a7852a8a8aefb4b394a98abd91e8c4c9498085a36`.
Completed QA artifact: 9013 bytes,
`e58c22f954dc9e70c6a8982b7bc6189473bc4a19a01737606724df97f90338b3`.
The QA summary is 7301 characters; its candidate reason is 1103 characters.
The decision accepts one **unqualified** site discovery. It proves neither
buying interest, a manual bottleneck, robot fit, funding nor a pilot owner.

## Defects and changes

The previous collector was reproduced against all four exact blobs and refused
with `terminal_qa_collection_ordering_changed`, before any provider call.
It required an explicit false reply field in the cancellation intent. The
collector now permits an omitted or explicit false pending-reply field,
preserves its original presence and bytes, and still rejects null, true and
non-boolean pending values. The exact blob hashes, creation times, source
identity, QA artifact, packet, evidence, cancellation reason, reply and operation
key checks remain in force. An intent must omit its not-yet-recorded key.

The merged consumer already accepts the complete long summary/reason and the
publisher splits Notion paragraphs without truncation. Additional descriptive
QA metadata no longer rejects an otherwise bound review: original bytes are
retained, while required dispositions and their types stay mandatory. Unknown
metadata cannot change approval or acceptance.

New normal Render creates require frozen scoped learning/history before the
provider action. The private bridge calls
`learning_context(day, allow_create=true)` only for new creates. Legacy
observation/retries reuse their durable input, with no replacement learning
input. Read-only preflight uses `allow_create=false`. Missing required history
refuses before paid model creation. The exact `content_json` is frozen in the
row/create intent, hash-bound in metadata, and mounted as
`/workspace/inputs/blueprint-research-learning.json`; the prompt instructs the
agent to read it before searching. The admitted complete CRM's public identity
projection is likewise frozen and mounted before research; private contact
fields and credentials are excluded. Thus downstream dedupe no longer has
exclusive access to the admission identities.

Research instructions now carry contact relevance and public contact provenance,
actual interest/owner/budget uncertainty, task-specific briefs, counterevidence,
early-stage and component-team scope, distinct prospect counts and evidence-based
coverage/stopping. They impose no prospect minimum or maximum quota. Team
findings remain separate from site prospects and deployment qualification.

The existing WebApp native learning hooks own daily aggregation and terminal
records. The research worker forwards sanitized terminal date/state to
`learningHooks.afterRun`; that owner reads the durable manifest/source hash.
The private bridge exposes no second daily scheduler or terminal writer.

## Reproducible private replay and remaining acceptance

Retrieve the exact four Firestore blobs and the pinned GCS bundle using existing
authorized access into a mode-0700 directory; preserve mode-0600 original files.
Use `<sha256>.json`, `qa-bundle.json.gz`, and `read-receipt.json` containing the
authenticated `firestore` receipt list (sha256, bytes, created_at). Then run:

```bash
BLUEPRINT_RESEARCH_RETAINED_REPLAY_DIR=/absolute/private/evidence-dir \
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -o addopts= \
  tests/test_daily_research_qa_retry.py -k exact_retained
```

The actual private replay passed through the collector's retained ordering,
provider-inventory/evidence binding and lossless QA decision. It retained the
previous cancellation state, the 7301-character explanation and the one
unqualified accepted key. It deliberately stops before live publication;
retained QA CRM identities are replayed, not described as a fresh live CRM read.
Hermetic absence/null/false drift tests cover the same guards without private
evidence or live services.

Validation used Python 3.12 with the package's pinned `openai==3.22.1` and
Ruff 0.16.9. The complete 574-test research lane passed 568 tests, including
the exact private replay; one obsolete quota-wording assertion was corrected
and its focused replay passed. The remaining five failures (four native
systemd calendar cases and the GNU `timeout` process-watchdog case) reproduced
unchanged on an isolated exact-main `73d3be8` checkout on this macOS host.
They remain required Linux CI checks, without skips or weakened assertions.
The actual pinned SDK wire/deadline test passed. The store/worker Node tests
passed all 25 cases after adding shutdown drain coverage. Changed Python files
passed Ruff and all edits passed `git diff --check`.

The existing Notion Team Directory was read: 47 distinct `BP-TEAM` detail
sections, no duplicate detail IDs, and the declared 28 work/pilot research,
14 readiness/integration-gated and five reference scopes. It was last edited
September 30 at 22:00:44.179 UTC. The separate Capability register still has
four `BP-CAP` starters, last edited September 30 at 15:00:38.600 UTC. Current
reads do not refresh their September 29/30 public-source check dates or establish
partnership willingness. Directory scope already includes early-stage and
component teams; the audited defect was the thinner runtime instructions/input,
not a need to replace these pages or call all 47 qualified deployment partners.

The historical 20-case provider comparison still has a portability gap. Its
original thread identifies three executor-local compressed answer bundles and
the 264396-byte readable report (SHA256
`0621b3eb0c2092e7494c945e42aecf22991af88ee1fab3ed53ce203f503bbb4d`).
Those original bytes were not retrieved from this host or proven accessible in
company-owned storage. Transcript metadata and Library pointers are recovery
references, not verified replacement originals. No paid comparison was rerun.

Remaining live acceptance requires a freshly authorized CRM read, both Sheets
and Notion acknowledgments with real readbacks, exact installed package/file
hashes and normal-path context verification. No successful live publication,
new inference, deployed fix or daily schedule operation is claimed here. The
stopped schedule must stay stopped until the owner directs resumption. Local
private copies are conveniences; canonical recovery remains Firestore/GCS.
