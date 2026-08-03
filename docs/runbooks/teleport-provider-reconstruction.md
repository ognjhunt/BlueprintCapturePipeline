# Teleport Provider Reconstruction

## Scope and claim boundary

The Teleport adapter is a fail-closed external appearance-reconstruction lane
for rights-cleared public candidate RGB images. It is not a generic customer
capture route. Provider output remains a derived candidate with claim ceiling
`appearance_reconstruction_candidate`. Provider `READY`, a downloaded PLY, a
successful import, camera alignment, or a held-out appearance pass does not
prove metric scale, collision validity, Isaac compatibility, task success,
physical truth, or deployment readiness.

The operator-facing mutation route is only:

```bash
python -m blueprint_pipeline.paid_resource_allocator provider-reconstruction \
  --provider teleport \
  --upload-packet /absolute/path/teleport_t1_upload_packet.v1.json \
  --execution-request /absolute/path/reconstruction_provider_execution_request.v1.json \
  --candidate-observations /absolute/path/candidate_observations.json \
  --sealed-evaluation-request /absolute/path/teleport_sealed_evaluation_request.v1.json \
  --output-dir /absolute/path/teleport-run
```

Without `--execute`, this writes a zero-mutation preflight and paid-lane
admission artifact. It still fails closed on a stale/invalid authority, dirty or
unpublished source commit, packet/hash mismatch, terms-review mismatch, or
invalid candidate observations.

## Official interface reviewed on 2026-08-03

The bound technical review is
[`docs/evidence/teleport_provider_terms_review_2026-08-03.json`](../evidence/teleport_provider_terms_review_2026-08-03.json).
It records the official authentication, multipart upload, ModelV3 parameters,
v2 PLY/COLMAP metadata, deletion, pricing, Terms, and DPA sources. Re-review and
emit a new immutable artifact if any live contract changes; never edit a frozen
run in place.

The implemented lifecycle is:

1. OAuth client-credentials token acquisition and expiry-aware refresh.
2. Reconcile or create one digest-derived capture identity.
3. Request each presigned part URL, stream the exact byte range, and record one
   validated ETag for every contiguous part.
4. Submit upload completion once, reconciling an ambiguous response from list
   state instead of blindly repeating it.
5. Poll with a request-bound retry cap and TTL, failing on provider failure or
   timeout.
6. Request v2 `content_profile=ply&coord_system=colmap` metadata; download and
   hash the provider-native metadata, camera JSON, and full PLY before import.
7. Import without rewriting provider-native bytes, estimate a similarity
   transform only from candidate camera correspondences, and enforce frozen RMS
   and maximum-residual thresholds.
8. Open evaluator-owned hidden inputs only after provider output is downloaded,
   imported, and aligned; render and score through the sealed held-out evaluator.
9. Attempt `DELETE /api/v1/captures/{eid}` on success and every failure after a
   capture identity exists, then perform two name-bound absence checks.

The HTTP transport never writes OAuth tokens or credential values to artifacts,
strips authorization on cross-origin redirects, rejects non-HTTPS and explicit
non-public hosts, bounds JSON and PLY downloads, and streams multipart uploads.

## Immutable public-data packet

Build the packet locally; this command performs no network access:

```bash
python scripts/build_teleport_t1_upload_packet.py \
  --proxy-root /absolute/path/to/frozen/mushroom_proxy \
  --output-dir /absolute/path/to/immutable/teleport_t1_packet
```

The builder validates the frozen proxy and candidate manifest, rehashes every
candidate image, refuses any hidden/held-out path, produces a deterministic
stored ZIP, binds the current terms-review digest, records that authorization
and spend are absent, and refuses to overwrite differing bytes.

## Live interlocks

`--execute` is insufficient by itself. All of the following must independently
pass:

- a clean, pushed immutable source commit (or an explicit pushed experimental
  diagnostic branch under the canary rule);
- a valid typed provider admission and human-issued authorization receipt bound
  to the same input digests, provider, actions, retry cap, TTL, and spend cap;
- a fresh human legal acceptance of the bound terms review;
- an exact provider quote no greater than the authorization cap;
- `TELEPORT_PUBLIC_DATA_UPLOAD_AUTHORIZED=true`;
- `TELEPORT_PUBLIC_DATA_SPEND_CAP_USD` exactly equal to the typed maximum;
- file-backed credentials at mode `0600` or stricter, normally
  `~/.blueprint-secrets/teleport_client_id` and
  `~/.blueprint-secrets/teleport_client_secret`;
- a sealed evaluation request with a frozen split, evaluator identity,
  thresholds, and evaluator-owned hidden files.

Inline credential environment variables are not accepted. No automatic live
retry is permitted after a terminal run; inspect typed receipts and issue new
authority for a new run.

## Receipts and limitations

The run emits progress, preflight, shared paid admission, provider execution,
provider-native file digests, import, alignment, sealed held-out evaluation,
deletion, cost, and aggregate run receipts. Stdout contains only a success bit.

Teleport deletion returns `204 No Content`; it is not a durable deletion
receipt. Two absent list checks increase operational confidence but do not prove
provider-zero, backup deletion, subprocessor deletion, deletion of training
derivatives, or final billing. The API exposes no final billing receipt, so the
cost receipt records the preapproved quote/cap and requires later statement
reconciliation.

## Customer and confidential data

Customer/confidential uploads remain blocked. The exact missing contractual and
technical evidence is recorded in
[`docs/evidence/teleport_future_customer_use_constraints_2026-08-03.json`](../evidence/teleport_future_customer_use_constraints_2026-08-03.json):
training-use carve-outs, executed confidentiality/DPA controls, retention and
durable deletion evidence, transferable/exportable and derivative model rights,
and qualified enterprise/on-prem ingestion of known poses, intrinsics, depth,
LiDAR, and gravity.
