# Buyer Delivery CDN Design (SCALE2-04) — 2026-07-20

Status: **validated design + feature-flagged partial implementation.** The
Cloud CDN signed-URL generator ships in
`arena_package_delivery_local.py` behind `BLUEPRINT_DELIVERY_CDN_ENABLED`
(default off, direct-GCS signed URLs remain the production path), but the
CDN infrastructure itself (backend bucket, load balancer, signing key) is
NOT provisioned in this change — that requires an owner decision and a
`terraform apply`. This is deliberately not claimed as "done".

## Problem

Every buyer delivery download today is a direct GCS signed URL
(`_generate_signed_url`, GCS egress ≈ $0.12/GiB) at ~1 GiB/capture. At 10k
deliveries/month that is ~10 TiB ≈ **$1,200/month** of egress billed at the
most expensive rate, with no caching for re-downloads of the same package.

## Options considered

**Cloud CDN in front of the delivery bucket (chosen design).** GCP-native
(no new primary provider), backend-bucket integration, egress at CDN rates
(~$0.04–0.08/GiB depending on region/volume tiers) plus cache-hit egress
that skips GCS entirely on re-downloads. Requires: enabling Cloud CDN on a
backend bucket behind an external HTTPS load balancer (~$18/month fixed for
the forwarding rule), a delivery hostname, and a **CDN signing key** —
because **GCS signed URLs and Cloud CDN signed URLs are different
mechanisms**:

- GCS signed URL: RSA (service-account key) V4 signature, verified by GCS,
  bypasses the CDN.
- Cloud CDN signed URL: `?Expires=<epoch>&KeyName=<key>&Signature=<hmac>`
  where the signature is **HMAC-SHA1 over the full URL** with a symmetric
  key registered on the backend bucket, base64url-encoded. Verified at the
  CDN edge; the CDN then fetches from GCS with its own service identity.

Signed *cookies* were rejected: deliveries are single-artifact downloads
handed to buyers as links (webapp mints per-request), so per-URL signing is
simpler and doesn't require the buyer's client to hold cookie state.

**Do nothing** stays acceptable below ~1k deliveries/month (egress <
$120/month) — which is why the flag defaults off.

## Entitlement gating (unchanged trust boundary)

Today the WebApp checks buyer entitlement before minting/handing out a
signed URL. That control point is **preserved exactly**: with CDN enabled,
the same entitlement check gates the minting of a *CDN* signed URL instead
of a *GCS* signed URL. The CDN never makes authorization decisions — it only
verifies the short-TTL signature produced after the entitlement check.
TTL stays the existing
`BLUEPRINT_PACKAGE_DELIVERY_SIGNED_URL_TTL_SECONDS` (default 15 min).

## Cache behavior + invalidation on republish

- Delivery objects are content-addressed in practice (prefix includes the
  output dir name; manifests carry sha256 per object). Set
  `Cache-Control: public, max-age=86400, immutable` on delivery uploads
  when CDN mode is on.
- Republish writes a **new** prefix (new output dir name), so stale-cache
  windows only exist if an object is overwritten in place — which delivery
  does not do. Belt-and-braces: a republish runbook step
  `gcloud compute url-maps invalidate-cdn-cache <map> --path "/<prefix>/*"`
  is documented for the exceptional in-place fix.
- Signature params (`Expires`) vary per mint but Cloud CDN strips signed-URL
  params from the cache key, so re-downloads hit cache across buyers of the
  same artifact only when the artifact is shared; per-buyer packages simply
  see no cross-buyer reuse (no harm).

## Cost model (at 10k deliveries/month, ~1 GiB each)

| Path | Egress | Fixed | Total/month |
| --- | --- | --- | --- |
| Direct GCS (today) | 10 TiB × $0.12/GiB ≈ $1,229 | — | ~$1,229 |
| Cloud CDN (cache-miss-heavy, worst case) | 10 TiB × ~$0.08/GiB ≈ $819 + cache-fill ~$0.01–0.04/GiB | ~$18 LB | ~$900–1,000 |
| Cloud CDN (warm, re-download-heavy) | approaching ~$0.04/GiB tiers | ~$18 LB | ~$430–600 |

Breakeven vs the $18/month fixed cost sits around ~40 GiB/month of egress —
far below any meaningful delivery volume, so the moment volume justifies
thinking about egress at all, the CDN pays for itself. Numbers use list
prices as of the round-1 audit; the owner should re-check current GCP
pricing before provisioning.

## Rollout plan

1. Owner provisions (Terraform, separate reviewed change): backend bucket on
   the delivery bucket, external HTTPS LB + hostname, Cloud CDN enabled,
   `signed-url-key` added to the backend bucket, key material stored at
   `~/.blueprint-secrets/cdn_url_signing_key` (base64url, never in source).
2. Set `BLUEPRINT_DELIVERY_CDN_ENABLED=1`,
   `BLUEPRINT_DELIVERY_CDN_BASE_URL=https://delivery.<domain>`,
   `BLUEPRINT_DELIVERY_CDN_KEY_NAME=<key-name>` on the delivery path.
3. Delivery manifests then carry CDN signed URLs; any signing failure falls
   back to the direct GCS signed URL (never a delivery outage).
4. Validate a real download end-to-end; watch cache-hit ratio; update the
   cost table above with observed numbers.

## What ships in this change

- `_generate_cdn_signed_url` implementing the Cloud CDN HMAC-SHA1 scheme,
  with unit tests pinning a reference vector.
- Flagged integration in the delivery upload path: CDN URL preferred,
  direct GCS signed URL fallback, per-object provenance recording which
  scheme produced the URL.
- This design doc + cost note in BETA_CAPACITY_COST_STORAGE_MODEL.
