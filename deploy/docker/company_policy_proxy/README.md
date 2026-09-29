# Blueprint-owned sandbox proxy image

ADP-011/day 7 admission and ADP-050/day 28 execution. Build from the repository
root with this Dockerfile. The image contains the fixed Unix proxy and its
shared trusted command dispatcher. It contains no customer policy, model,
scene, capture, scoring code or credentials.

Supply an approved OCI digest as `blueprint_proxy_image` in the trusted sandbox
configuration. Build and publish with Blueprint's credentials on a dedicated
builder. The worker bootstrap must be able to pull this Blueprint-owned image
before claiming a customer credential lease. When using a private registry,
remove that bootstrap credential and its Docker auth entry after the exact
proxy pull and before claiming the customer lease. The customer image pull uses
the separate single-use, image/admission/plan-bound broker credential. Neither
credential is mounted into either runtime container.

The 2026-09-28 live proof used a separately built private proxy with the same
proxy/entrypoint bytes and pinned base/dependencies:
`us-central1-docker.pkg.dev/blueprint-8c1ca/pipeline-jobs/company-policy-proxy-proof@sha256:3cca1ee8ab0ad16d9f6c095eb5c41449320243fa49e80127a94f50c9181febf7`.
The published proof artifact was built from a minimal generated context; this
repository-root Dockerfile does not promise a byte-identical OCI rebuild.
Bootstrap credentials were removed before the production customer lease claim.
The canonical executor verified the private policy digest, nine denied network
paths, synthetic conformance, credential erasure and terminal cleanup. The
operator-owned development boot key grants no production scene/task authority.
