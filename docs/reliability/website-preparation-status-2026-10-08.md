# Current website preparation status

ADP-010/day14; observed blocker: a source-selected preparation refusal commits a retryable local worker failure before the prepared-scene outbox exists. The customer scene-intake poll cannot represent that earlier failure. This slice adds a source-bound current read and a durable wake-up on the existing service and scheduled listener.

`POST /api/live-pipeline/website-preparation-status` uses the existing client HMAC/timestamp/nonce and requires `blueprint-webapp`. Its signed selectors are the request, scene, capture, marker generation, immutable producer delivery key, bare payload SHA-256, and current task-context digest. Only the existing authenticated server capture-root mapping may select a root. Missing/unconfigured/mismatched mapping, missing provenance/context/ledger, unstable revisions and unavailable current authority return a fixed unavailable response. The route body does not write capture, customer or provider state; existing nonce/admission bookkeeping still writes its own stores.

The response is an as-of current ledger snapshot. Retained original birth/membership is historical source proof; fresh owner observation and task context are also required. Original 64-bit generation strings stay strings. The ledger is read before and after verification. Raw paths, provider errors and owner details are excluded. `handed_off` requires the source handoff and required output commit and means only that source-stage handoff; it cannot establish native execution or an assessment result.

The worker persists a selector-only delivery record after committing its ledger. The existing scheduled drain scans committed ledgers, which also closes a crash between the ledger and delivery commits. A private durable cursor gives the bounded scan fair coverage. One delivery per production drain tick bounds transport delay; failed deliveries retain pending state. Network runs outside the ledger lock. Delivery retry never reopens provider processing. The WebApp callback must fetch the latest signed status instead of trusting the callback or a maximum previously received attempt. Consent and genuine customer results retain precedence in that consumer.

Replay with the repository's existing development Python environment:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python -m pytest -q -p no:cacheprovider tests/test_website_preparation_status.py tests/test_pubsub_handoff_listener.py
ruff check src/blueprint_pipeline/website_preparation_status.py src/blueprint_pipeline/website_preparation_status_http.py tests/test_website_preparation_status.py
```

Tests use owned synthetic fixtures, the actual generation-birth/lease/commit functions and signed FastAPI route. Source downloads and owner/context/callback transports are isolated fakes. One joined failure test runs the actual source-selected outer worker and initial reconstruction-pending handoff, with the qualification runner translating that refusal into an exception. No provider, real database, private video, email, native authoring or final assessment is executed. An optional private local trace file can be selected with `BLUEPRINT_RELIABILITY_STATUS_TRACE`; it is not a CI/public artifact. No new journey or frozen catalog coverage is claimed.

Operational acceptance still requires an existing exact server mapping, the reviewed WebApp consumer, protected release and deployed behavior verification. Current configuration was not inspected or widened by this slice. An unavailable mapping remains an explicit dependency.
